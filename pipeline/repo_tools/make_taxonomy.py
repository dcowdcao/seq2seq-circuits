#!/usr/bin/env python3
"""Generate models/taxonomy.json: the machine-parseable coverage map of every circuit's
classification — the standing input for brainstorm gap-probes.

Merges these sources per circuit class in models/circuits.py:
  CODE      condition/collapse facts from the class itself (+ a CPU pair-mask simulation
            for ContrastCircuits: 64 sampled pairs -> per-row TRUE-output diff sets)
  ARTIFACT  circuits_data/<model>/<circuit>/smallest_circuit.json (+ the eval cache
            .smallest_circuit_evals/evals/ for the winner's kept-MLP count, and for d_*
            when the winner json lacks them)
  CENSUS    edge_signs/census.json per-model negative-edge mass
  ROUTER    task_router/<model>/diagnostic.json gating/outvoting/superposition verdict
            (folded into SuppressionContrastCircuit records)

C3 suppression circuits register on-demand (suppression_circuits_for) for every TaskMux model in
models.txt; they get a "suppression" section + coverage cells (release/collateral/router verdict).
Coverage also exposes model-axis cells (attn_only vs mlp/moe) so N1/N2/N3 variant twins are
visible. Run this BEFORE brainstorming new circuits (regenerate it at the start of each round).

Variant models (circuits_data/<base>__<variant>/) get the same records keyed
"<base>__<variant>/<circuit>" with tier="<variant>"; class facts come from the BASE
circuit class. A "probes" section summarizes probes_data/<model>/<variable>/
(site_scan.json + validation.json from probes/train_probes.py); each record carries the
phase the variable is decodable at (input vs generation positions), and the gen_* usage
cells are the variables that only materialise while the answer is being emitted.

Usage:
    python make_taxonomy.py                          # -> models/taxonomy.json
    python make_taxonomy.py --out models/taxonomy.json
"""
import argparse, itertools, json, re, sys
from collections import Counter
from pathlib import Path

MODELS_DIR = Path(__file__).parents[1] / "models"
sys.path.insert(0, str(MODELS_DIR))
sys.path.insert(0, str(Path(__file__).parent))

import gates
import model_registry
from algorithms import decode_algorithm_id, base_algorithm_id, TaskMux
from circuits import CIRCUITS, ContrastCircuit, build, suppression_circuits_for, task_circuits_for

CENSUS_PATH = Path(model_registry.PIPELINE_ROOT) / "edge_signs" / "census.json"
CIRCUITS_DATA = Path(model_registry.PIPELINE_ROOT) / "circuits_data"
PROBES_DATA = Path(model_registry.PIPELINE_ROOT) / "probes_data"
TASK_ROUTER = Path(model_registry.PIPELINE_ROOT) / "task_router"
RUNLIST = MODELS_DIR / "models.txt"
MASK_PAIRS = 64
# WHOLE-SEQUENCE exact-match cut points on d_affected (find_circuits FRINGE_METRIC_TAG)
NECESSARY_CUT, WEAK_CUT = 0.10, 0.02
PRE_V2_METRIC = "token_accuracy"  # unit of rows written before the whole-sequence fringe metrics

# --- Empty-cell complement (the brainstorm's G4/G5 gates read this) ---------------------
#: Levels recording ABSENCE of data, not a design choice: a cell empty only because nothing
#: was built is not an unexplored design cell, so these never enter the targetable space.
DEGENERATE_LEVELS = {"not_run", "null"}
#: Families whose emptiness is a BUILD backlog (a declared mux whose task circuits aren't built)
#: rather than unexplored design space — reported, but not creditable as novelty.
BUILD_GAP_FAMILIES = {"breadth_cells"}
#: Axis DEPENDENCIES. The product over per-axis levels over-counts when one axis is only populated
#: on some branch of another. domain_record returns early for no_faithful/not_run, so compactness
#: is set ONLY on the winner branch: a real compactness beside any other verdict is structurally
#: unreachable, not unexplored. contrast_record does NOT share this (it fills locality/necessity
#: from the eval cache on both branches — 8 no_faithful rows carry real values), so it has no rule.
REACHABLE = {
    "domain_cells": lambda parts: parts[1] == "winner",
    # A suppression winner must clear DEFAULT_RELEASE_FLOOR (0.5) and release_bucket() calls
    # >= 0.5 "strong", so a winner can ONLY ever bucket strong: winner|none|* and winner|weak|*
    # are unfillable by construction, not unexplored.
    "suppression_cells": lambda parts: parts[0] != "winner" or parts[1] == "strong",
}
#: Verdict levels meaning the circuit FAILED. Such a cell is a real coverage gap and stays in
#: `empty`, but it must not earn novelty credit: crediting a fill of `...|no_faithful` would
#: credit a proposal for predicting that its own circuit does not work.
FAILURE_LEVELS = {"no_faithful"}
#: Families whose FIRST axis is an open vocabulary — condition_family() falls back to the bare
#: condition name and breadth keys off the model id, so a level absent from the observed vocabulary
#: is a NEW cell, empty by construction. Such a cell cannot appear in the enumerated `empty` list
#: (that list is a product of OBSERVED levels), so the authoritative G4 test is non-membership in
#: `coverage[family]` plus reachability — see the `g4_predicate` note emitted below.
OPEN_VOCAB_FAMILIES = {"domain_cells", "breadth_cells"}

# Condition-name regex -> factory family (names are underscore-parameterized, not "fam(args)").
_FAMILIES = [(r"exclude_\d+$", "exclude_digit"), (r"sum_le_\d+$", "sum_at_most"),
             (r"pin_\d+_", "pin"), (r"at_most_\d+_of_\d+$", "at_most_k_of"),
             (r"exactly_one_in_", "exactly_one_in"), (r"cmp_\d+_gt_\d+$", "compare_positions"),
             (r"monotone_", "monotone"), (r"domain_", "domain"),
             (r"task\d+_of\d+$", "task_pin")]


def _json(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def condition_family(name: str) -> str:
    """conditions.py factory family with parameters stripped; parameter-free names stand as-is."""
    base = name.split("(")[0]
    return next((fam for pat, fam in _FAMILIES if re.match(pat, base)), base)


def verdict_of(sc: dict | None) -> str:
    if sc is None:
        return "not_run"
    return "no_faithful" if sc.get("no_faithful_circuit") else "winner"


def base_faithful_of(sc: dict | None) -> bool | None:
    """Was the row's BASE circuit faithful? New outputs record base_faithful (absent =
    faithful winner); OLD base_not_faithful markers imply False; None = never run."""
    if sc is None:
        return None
    if sc.get("reason") == "base_not_faithful":
        return False
    return sc.get("base_faithful", True)


def compactness(node_frac: float) -> str:
    return "tiny" if node_frac <= 0.25 else "medium" if node_frac <= 0.75 else "near_full"


def census_neg_mass(census: dict, model: str) -> float | None:
    """Model-level negative-edge mass: max max_neg_frac across the signed methods."""
    if model not in census:
        return None
    fracs = [m["max_neg_frac"] for m in census[model]["methods"].values() if m["signed"]]
    return max(fracs) if fracs else None


def best_eval(circuit_dir: Path) -> dict | None:
    """Cached differential candidate eval with the highest s_union (non-differential
    caches from earlier schema eras are skipped)."""
    evals = [e for p in circuit_dir.glob(".smallest_circuit_evals/evals/*.json")
             for e in [json.loads(p.read_text())] if "s_union" in e]
    return max(evals, key=lambda e: e["s_union"]) if evals else None


def mask_shape(cls: type) -> str:
    """Classify the per-row TRUE-output diff index sets of MASK_PAIRS sampled pairs."""
    inst, parent = build(cls), decode_algorithm_id(cls.algorithm)
    clean, corrupt = inst.sample_pairs(MASK_PAIRS, 0)
    rows = [(parent.eval(clean[i]) != parent.eval(corrupt[i])).nonzero().flatten().tolist()
            for i in range(clean.shape[0])]
    last = parent.output_length() - 1
    contig = lambda ix: ix[-1] - ix[0] + 1 == len(ix)
    if all(len(ix) == 1 for ix in rows):
        return "singleton"
    if all(contig(ix) and ix[-1] == last for ix in rows):
        return "suffix"
    if 2 * sum(1 for ix in rows if contig(ix) and ix[-1] < last) >= len(rows):
        return "window"
    if all(len(ix) <= 3 for ix in rows) and any(not contig(ix) for ix in rows):
        return "sparse"
    return "mixed"


def domain_record(cls: type, m: str, tier: str, census: dict) -> dict:
    c = cls.circuit_id
    cdir = Path(model_registry.data_dir(m, c))
    sc = _json(cdir / "smallest_circuit.json")
    spec = model_registry.get_spec(m)
    rec = {"model": m, "circuit": c, "tier": tier,
           "condition": str(cls._cond.name) if cls._cond is not None else "custom",
           "collapsed_eval": "eval" in cls.__dict__,
           "verdict": verdict_of(sc),
           "accuracy": None, "best_available_accuracy": None, "winning_candidate": None,
           "n_heads": None, "n_edges": None, "h_frac": None, "node_frac": None, "e_frac": None,
           "compactness": None, "selection_score": None,
           # the full model's own rate vs the fully-severed baseline: a real circuit vs a
           # constant-output isolation. relative_recovery may be the "zero_contrast" MARKER.
           "empty_sequence_accuracy": None, "relative_recovery": None,
           "census_neg_mass": census_neg_mass(census, m)}
    if sc is None:
        return rec
    rec["winning_candidate"] = sc["circuit_name"]
    if "perf_empty" in sc:      # absent on pre-v2 rows (both verdicts record it now)
        rec["empty_sequence_accuracy"] = sc["perf_empty"]["sequence_accuracy"]
    if rec["verdict"] == "no_faithful":
        # slim reason-markers (zero_contrast_dataset, positional_only_candidates) score nothing
        rec["best_available_accuracy"] = sc.get("best_available_accuracy")
        rec["relative_recovery"] = sc.get("best_available_relative_recovery")
        return rec
    rec["accuracy"], rec["selection_score"] = gates.candidate_accuracy(sc, "domain"), sc["selection_score"]
    rec["n_heads"], rec["n_edges"] = sc["n_heads"], sc["n_edges"]
    rec["relative_recovery"] = sc.get("relative_recovery_sequence_accuracy")
    rec["h_frac"] = round(sc["n_heads"] / (spec.n_layers * spec.n_heads), 4)  # attention-only, both sides
    # Edge budget = geometry (model_registry.live_edges), never a finder artifact: EAP-IG's
    # graph.real_edges counts phantom m{l} slots an attn-only model does not have.
    full_edges = sc["n_edges_full"] if "n_edges_full" in sc else model_registry.live_edges(spec)
    rec["e_frac"] = round(sc["n_edges"] / full_edges, 4)
    # node_frac is the selection score's node fraction: kept heads+MLPs over heads+MLPs full.
    ev = _json(cdir / ".smallest_circuit_evals" / "evals" / f"{sc['circuit_name']}.json")
    full_nodes = spec.n_layers * spec.n_heads + (0 if spec.attn_only else spec.n_layers)
    if ev is not None:
        rec["node_frac"] = round((sc["n_heads"] + ev.get("n_mlps", 0)) / full_nodes, 4)  # attn-only caches omit n_mlps
        rec["compactness"] = compactness(rec["node_frac"])
    return rec


def contrast_record(cls: type, m: str, tier: str) -> dict:
    c = cls.circuit_id
    cdir = Path(model_registry.data_dir(m, c))
    sc = _json(cdir / "smallest_circuit.json")
    rec = {"model": m, "circuit": c, "tier": tier, "base_circuit": cls.base_circuit,
           "local_predicate": cls.local_predicate, "edit_pattern": None,
           "mask_shape": mask_shape(cls), "verdict": verdict_of(sc),
           "base_faithful": base_faithful_of(sc),
           "s_union": None, "d_affected": None, "d_unaffected": None, "d_control": None,
           "necessity_class": None, "metric_unit": None, "locality_measured": None,
           "n_fringe_heads": None, "n_fringe_edges": None, "binding_gate": None}
    if sc is None:
        return rec
    nec = sc.get("necessity") or {}                       # suppression jsons carry no necessity
    if rec["verdict"] == "winner":
        rec["s_union"] = sc.get("sufficiency", {}).get("s_union")
        rec.update(d_affected=nec["d_affected"], d_unaffected=nec["d_unaffected"],
                   d_control=nec["d_control"], metric_unit=sc.get("metric_unit"),
                   n_fringe_heads=sc["n_heads"], n_fringe_edges=sc["n_edges"])
    else:
        rec["s_union"] = sc.get("best_available_s_union")
        best = best_eval(cdir)   # old no_faithful jsons omit d_*; the eval cache has them
        for k in ("d_affected", "d_unaffected", "d_control"):
            rec[k] = nec.get(k, best[k] if best else None)
        if best:
            rec["n_fringe_heads"], rec["n_fringe_edges"] = best["n_fringe_heads"], best["n_fringe_edges"]
            rec["s_union"] = rec["s_union"] if rec["s_union"] is not None else best["s_union"]
            rec["metric_unit"] = best.get("metric_unit")   # the cache the d_* came from
        floor = sc.get("min_accuracy_threshold")   # absent on old marker schemas
        if rec["s_union"] is not None and floor is not None and rec["s_union"] < floor:
            rec["binding_gate"] = "s_union"
    d = rec["d_affected"]
    if d is not None:
        rec["metric_unit"] = rec["metric_unit"] or PRE_V2_METRIC   # unrecorded unit == token deltas
        rec["necessity_class"] = ("necessary" if d >= NECESSARY_CUT else
                                  "weak" if d >= WEAK_CUT else "redundant")
    if rec["d_unaffected"] is not None:
        rec["locality_measured"] = "local" if rec["d_unaffected"] <= 0.05 else "global"
    return rec


def arch_family(tier: str) -> str:
    """N2 architecture family from the tier tag (base/small/large/wide/deep/xl = attn-only)."""
    if tier.startswith("mlp"): return "mlp"
    if tier.startswith("moe"): return "moe"
    return "attn_only"


def release_bucket(rec: dict) -> str:
    """Strength of the release the row was SCORED on: the recorded release_scored, else the
    stronger recorded reference (pre-v2 rows lack the field, and every winner passed on TONIC —
    bucketing the cued rate alone buckets the number they all failed on)."""
    rates = [r for r in (rec["release_rate"], rec["release_rate_tonic"]) if r is not None]
    scored = rec["release_scored"] if rec["release_scored"] is not None else max(rates, default=None)
    if scored is None: return "null"
    return "strong" if scored >= 0.5 else "weak" if scored >= 0.1 else "none"


def router_verdict(sc: dict | None, model: str, cued: int, wrong: int) -> str:
    """gating/outvoting/superposition for the (cued,wrong) pair. Prefers the verdict
    RECORDED on the row (new outputs' router_verdict; old router_gating markers'
    verdict), then falls back to task_router/<model>/diagnostic.json; 'not_run' when
    unmaterialized (transitional: both schemas co-exist on disk)."""
    if sc:
        v = sc.get("router_verdict") or (sc.get("verdict")
                                         if sc.get("reason") == "router_gating" else None)
        if v:
            return v
    diag = _json(TASK_ROUTER / model / "diagnostic.json")
    if diag is None:
        return "not_run"
    for pr in diag.get("pairs", []):
        if pr.get("cued") == cued and pr.get("wrong") == wrong:
            return pr.get("verdict", "not_run")
    return "not_run"


def suppression_record(cls: type, m: str, tier: str) -> dict:
    """C3 cue-flip circuit: the fringe suppressor scored by release/collateral (not d_*). BOTH
    sever references are recorded per verdict — release_rate is the cued-mean rate, and
    release_rate_tonic the wrong-cue (tonic) one that a cue-constant suppressor only shows up in."""
    c = cls.circuit_id
    cdir = Path(model_registry.data_dir(m, c))
    sc = _json(cdir / "smallest_circuit.json")
    rec = {"model": m, "circuit": c, "tier": tier, "base_circuit": cls.base_circuit,
           "cued_task": cls.cued_task, "wrong_task": cls.wrong_task, "verdict": verdict_of(sc),
           "router_verdict": router_verdict(sc, m, cls.cued_task, cls.wrong_task),
           "base_faithful": base_faithful_of(sc),
           "release_rate": None, "release_rate_tonic": None,
           "release_reference": None, "release_scored": None,
           "collateral": None, "collateral_tonic": None, "metric_unit": None,
           "wrong_wins_full": None, "wrong_wins_severed": None,
           "n_fringe_heads": None, "n_fringe_edges": None, "n_fringe_mlps": None}
    if sc is None:
        return rec
    if rec["verdict"] == "winner":
        rel = sc["release"]
        rec.update(release_rate=rel["release_rate"], release_rate_tonic=rel["release_rate_tonic"],
                   release_reference=rel.get("release_reference"),
                   release_scored=rel.get("release_scored"),
                   wrong_wins_full=rel["wrong_wins_full"], wrong_wins_severed=rel["wrong_wins_severed"],
                   collateral=sc["collateral"], collateral_tonic=sc["collateral_tonic"],
                   n_fringe_heads=sc["n_heads"], n_fringe_edges=sc["n_edges"],
                   n_fringe_mlps=sc.get("n_fringe_mlps"))
    # no_faithful rows carry the best-by-release fringe's size and both references; the slim
    # reason-markers (zero_contrast_dataset, positional_only_candidates) carry no release at all.
    elif "best_available_release" in sc:
        rec.update(release_rate=sc["best_available_release"],
                   release_rate_tonic=sc["best_available_release_tonic"],
                   release_reference=sc.get("best_available_release_reference"),
                   release_scored=sc.get("best_available_release_scored"),
                   collateral=sc.get("collateral"), collateral_tonic=sc.get("collateral_tonic"),
                   n_fringe_heads=sc.get("n_heads"), n_fringe_edges=sc.get("n_edges"),
                   n_fringe_mlps=sc.get("n_fringe_mlps"))
    # Unit of the release rates: the row's own tag, else the fringe eval caches' (no_faithful
    # suppression payloads omit it), else the pre-whole-sequence unit.
    cached = (json.loads(p.read_text()).get("metric_unit")
              for p in sorted(cdir.glob(".smallest_circuit_evals/evals/*.json")))
    rec["metric_unit"] = sc.get("metric_unit") or next((u for u in cached if u), None) or PRE_V2_METRIC
    return rec


def causal_gate_summary(val: dict | None) -> dict | None:
    """Flip/collateral/permutation summary of validation.json (None when never run)."""
    if val is None:
        return None
    cg, perm = val["causal_gate"], val["permutation_control"]["accuracy"]
    if "flip_rate" not in cg:                       # 425 skipped + 1 legacy 'note' scan
        return {"skipped": cg.get("skipped", cg.get("note", "no flip_rate recorded")),
                "permutation_accuracy": perm, "passed": False}
    #: steered_rate says only that steering was ATTEMPTED and is measured with the probe against
    #: itself (r=0.206 with flip_rate) -- flip_rate is whether the MODEL's output moved.
    passed = (cg["flip_rate"] is not None and cg["collateral_rate"] is not None
              and cg["flip_rate"] >= gates.FLIP_GATE
              and cg["collateral_rate"] <= gates.COLLATERAL_GATE)
    out = {"flip_rate": cg["flip_rate"], "collateral_rate": cg["collateral_rate"],
           "permutation_accuracy": perm, "passed": passed}
    for k in ("suffix_consistent_rate", "steered_rate", "probe_reads_source_rate", "mean_alpha"):
        if k in cg:
            out[k] = cg[k]
    return out


def probe_circuit_dirs(model: str, variable: str) -> list[str]:
    """"<circuit>/<candidate>" finder dirs (eap_probe_<var>_top*, acdc_probe_<var>_thresh_*)."""
    return sorted(f"{p.parent.parent.name}/{p.name}"
                  for p in CIRCUITS_DATA.glob(f"{model}/*/circuits/*_probe_{variable}_*")
                  if p.is_dir())


def probe_record(pdir: Path) -> dict:
    scan = json.loads((pdir / "site_scan.json").read_text())
    m, v = scan["model"], scan["variable"]
    acc = {int(l): a for l, a in scan["per_layer_accuracy"].items()}
    best = min(acc, key=lambda l: (-acc[l], l))
    return {"model": m, "variable": v, "n_classes": scan["n_classes"],
            "verdict": scan["verdict"],
            # generation-position reading; None on pre-rescan scans, which never measured it
            "gen_verdict": scan.get("gen_verdict"),
            "first_gen_decodable_layer": scan.get("first_gen_decodable_layer"),
            "decodable_phase": gates.decodable_phase(scan),
            "first_decodable_layer": scan["first_decodable_layer"],
            "best_layer": best, "best_accuracy": acc[best],
            "baseline_accuracy": scan["baseline_accuracy"],
            "per_layer_accuracy": scan["per_layer_accuracy"],
            "causal_gate": causal_gate_summary(_json(pdir / "validation.json")),
            "probe_circuits": probe_circuit_dirs(m, v)}


def probe_usage(rec: dict) -> str:
    """Usage cell, prefixed by the phase the variable is decodable at: gen_* rows are decodable
    only while the answer is being emitted, which is not the input-position reading."""
    if rec["decodable_phase"] is None:
        return {"absent": "not_decodable",
                "trivial_from_input": "trivial_from_input"}[rec["verdict"]]
    p = "decodable" if rec["decodable_phase"] == "input" else "gen_decodable"
    cg = rec["causal_gate"]
    if cg is None or "passed" not in cg:
        return f"{p}_untested"
    return f"{p}_and_used" if cg["passed"] else f"{p}_not_used"


def probe_kind(variable: str) -> str:
    if variable.startswith("bridge"): return "bridge"
    if variable.startswith("task"): return "task_id"
    return "hidden_state"


def coverage(domain: dict, contrast: dict, suppression: dict, probes: dict) -> dict:
    s = lambda v: "null" if v is None else str(v)
    base_only = lambda recs: [r for r in recs.values() if r["tier"] == "base"]
    cc = Counter(f"{s(r['locality_measured'])}|{s(r['necessity_class'])}|{r['verdict']}"
                 for r in base_only(contrast))
    dc = Counter(f"{condition_family(r['condition'])}|{r['verdict']}|{s(r['compactness'])}"
                 for r in base_only(domain))
    ms = Counter(r["mask_shape"] for r in base_only(contrast))
    pc = Counter(f"{probe_kind(r['variable'])}|{probe_usage(r)}|"
                 f"{'probe_circuit' if r['probe_circuits'] else 'no_probe_circuit'}"
                 for r in probes.values())
    # C3 suppression: verdict | scored-release strength | router gating/outvoting verdict
    sup = Counter(f"{r['verdict']}|{release_bucket(r)}|{r['router_verdict']}"
                  for r in base_only(suppression))
    # N1/N2/N3 model axes — NOT base_only (seeing the twins IS the point)
    all_recs = list(domain.values()) + list(contrast.values()) + list(suppression.values())
    mac = Counter(f"{arch_family(r['tier'])}|{r['verdict']}" for r in all_recs)
    # N3 breadth: per-mux task circuits
    bre = Counter(f"{r['model']}|{r['verdict']}" for r in domain.values()
                  if condition_family(r["condition"]) == "task_pin")
    return {"contrast_cells": dict(sorted(cc.items())),
            "domain_cells": dict(sorted(dc.items())),
            "mask_shapes": dict(sorted(ms.items())),
            "probe_cells": dict(sorted(pc.items())),
            "suppression_cells": dict(sorted(sup.items())),
            "model_axis_cells": dict(sorted(mac.items())),
            "breadth_cells": dict(sorted(bre.items()))}


def _axis_levels(cells: dict) -> list:
    """Per-axis targetable level vocabulary of a set of '|'-joined cell keys."""
    parts = [k.split("|") for k in cells]
    widths = {len(p) for p in parts}
    if len(widths) != 1:
        raise ValueError(f"ragged cell key widths {sorted(widths)}: {sorted(cells)[:3]}")
    return [sorted({p[i] for p in parts} - DEGENERATE_LEVELS) for i in range(widths.pop())]


def empty_cells(cov: dict) -> dict:
    """Per family, the targetable cell space MINUS what is occupied, so a gate can LOOK UP
    whether a cell is empty instead of re-deriving it. The space is each axis's observed level
    vocabulary with DEGENERATE_LEVELS dropped, so it only ever grows as classifiers gain levels.

    ``saturated`` marks a family whose whole space is already occupied — it CANNOT carry an
    emptiness gate until its classifier emits new levels (``mask_shape`` returns exactly five
    shapes and ``arch_family`` three, so both saturate structurally). ``creditable`` is false
    for BUILD_GAP_FAMILIES, whose empties are unbuilt rows rather than unexplored design.

    The enumerated ``empty`` list covers only the OBSERVED level vocabulary, so for an
    ``open_vocab`` family it is a floor, not the whole story: a brand-new condition family is a new
    cell that no enumeration can list. ``g4_predicate`` states the authoritative test."""
    out = {}
    for fam, cells in cov.items():
        keep = REACHABLE.get(fam, lambda _p: True)
        space = ["|".join(t) for t in itertools.product(*_axis_levels(cells)) if keep(t)]
        empty = sorted(set(space) - set(cells))
        open_vocab = fam in OPEN_VOCAB_FAMILIES
        scores = fam not in BUILD_GAP_FAMILIES
        # Novelty credit counts only cells a SUCCEEDING circuit can fill: a failure-verdict cell is
        # a real gap (kept in `empty`) but crediting it would credit predicting your own circuit fails.
        creditable_empty = sorted(c for c in empty
                                  if scores and not (FAILURE_LEVELS & set(c.split("|"))))
        out[fam] = {"targetable": len(space), "n_empty": len(empty),
                    "n_creditable_empty": len(creditable_empty),
                    "creditable_empty": creditable_empty,
                    "saturated": not creditable_empty and not open_vocab,
                    "creditable": scores, "open_vocab": open_vocab,
                    "g4_predicate": ("cell not in coverage[family] AND levels non-degenerate AND "
                                     "reachable; a NEW axis-0 level qualifies and will NOT be in "
                                     "`empty`" if open_vocab else "cell in `empty`"),
                    "empty": empty}
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Generate the circuit taxonomy (models/taxonomy.json)")
    p.add_argument("--out", default=str(MODELS_DIR / "taxonomy.json"), help="Output taxonomy json")
    args = p.parse_args()

    census = (_json(CENSUS_PATH) or {}).get("models", {})
    # C3: register every TaskMux model's cue-flip suppression circuits so they enter CIRCUITS
    for line in RUNLIST.read_text().splitlines():
        mid = line.split("#", 1)[0].strip()
        if mid and isinstance(decode_algorithm_id(base_algorithm_id(mid)), TaskMux):
            task_circuits_for(mid)
            suppression_circuits_for(mid)

    targets = [(cls.algorithm, cls, "base") for cls in CIRCUITS]
    by_id = {(cls.algorithm, cls.circuit_id): cls for cls in CIRCUITS}
    for vdir in sorted(CIRCUITS_DATA.glob("*__*")):  # variants reuse the BASE circuit class
        base, variant = vdir.name.split("__", 1)
        targets += [(vdir.name, by_id[(base, cdir.name)], variant)
                    for cdir in sorted(vdir.iterdir())
                    if cdir.is_dir() and (base, cdir.name) in by_id]

    domain, contrast, suppression = {}, {}, {}
    for model, cls, tier in sorted(targets, key=lambda t: (t[0], t[1].circuit_id)):
        key = f"{model}/{cls.circuit_id}"
        if getattr(cls, "suppression", False):
            suppression[key] = suppression_record(cls, model, tier)
        elif issubclass(cls, ContrastCircuit):
            contrast[key] = contrast_record(cls, model, tier)
        else:
            domain[key] = domain_record(cls, model, tier, census)

    probes = {f"{p.parent.parent.name}/{p.parent.name}": probe_record(p.parent)
              for p in sorted(PROBES_DATA.glob("*/*/site_scan.json"))}

    cov = coverage(domain, contrast, suppression, probes)
    tax = {"generated_by": "repo_tools/make_taxonomy.py",
           "domain": domain, "contrast": contrast, "suppression": suppression, "probes": probes,
           "coverage": cov, "empty_cells": empty_cells(cov)}
    Path(args.out).write_text(json.dumps(tax, indent=1) + "\n")

    s = lambda v: "-" if v is None else v
    print(f"{'model/circuit':<46}{'verdict':<13}{'mask_shape':<12}{'necessity':<11}{'locality':<9}s_union")
    for k, r in contrast.items():
        print(f"{k:<46}{r['verdict']:<13}{r['mask_shape']:<12}{s(r['necessity_class']):<11}"
              f"{s(r['locality_measured']):<9}{s(r['s_union'])}")
    variants = [r for r in list(domain.values()) + list(contrast.values()) if r["tier"] != "base"]
    print(f"\n{'variant model/circuit':<46}{'tier':<7}{'verdict':<13}{'acc':<9}{'heads':<7}edges")
    for r in sorted(variants, key=lambda r: (r["model"], r["circuit"])):
        acc = r.get("accuracy") if r["verdict"] == "winner" else r.get("best_available_accuracy")
        print(f"{r['model'] + '/' + r['circuit']:<46}{r['tier']:<7}{r['verdict']:<13}"
              f"{s(acc):<9}{s(r.get('n_heads', r.get('n_fringe_heads'))):<7}"
              f"{s(r.get('n_edges', r.get('n_fringe_edges')))}")
    print(f"\n{'model/variable':<42}{'n_cls':<7}{'verdict':<20}{'gen_verdict':<29}"
          f"{'first_dec':<11}{'best_L':<8}{'best_acc':<10}{'baseline':<10}{'flip':<7}"
          f"{'collat':<8}{'usage':<25}probe_circuits")
    for k, r in probes.items():
        cg = r["causal_gate"] or {}
        print(f"{k:<42}{r['n_classes']:<7}{r['verdict']:<20}{s(r['gen_verdict']):<29}"
              f"{s(r['first_decodable_layer']):<11}"
              f"{r['best_layer']:<8}{r['best_accuracy']:<10}{r['baseline_accuracy']:<10}"
              f"{s(cg.get('flip_rate')):<7}{s(cg.get('collateral_rate')):<8}"
              f"{probe_usage(r):<25}{len(r['probe_circuits'])}")
    for name, cells in tax["coverage"].items():
        print(f"\n{name}:")
        for cell, n in cells.items():
            print(f"  {cell}: {n}")
    print(f"\nempty cells (targetable space minus occupied — the brainstorm G4/G5 input):")
    for name, e in tax["empty_cells"].items():
        flags = ("  SATURATED (cannot gate)" if e["saturated"] else
                 "" if e["creditable"] else "  build-gap (not creditable as novelty)")
        print(f"  {name:<20}{e['n_creditable_empty']:>5} creditable / {e['n_empty']:>5} empty "
              f"/ {e['targetable']:>5} reachable{flags}")
    print(f"\nWrote {len(domain)} domain + {len(contrast)} contrast + "
          f"{len(suppression)} suppression + {len(probes)} probe records -> {args.out}")


if __name__ == "__main__":
    main()
