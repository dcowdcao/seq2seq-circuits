#!/usr/bin/env python3
"""Acceptance gates: what a circuit or probe row must satisfy to be reported.

One rule set: repo_tools/make_taxonomy.py re-derives the circuit accuracy and the probe
reading from these thresholds, and any summary/index script should apply the same checks
so a failing row is never reported. Every gate is keyed to the metric the row itself reports, because
the three circuit kinds do not share one: C0 reports an absolute accuracy on its own
edge set, C1 reports s_union over a fringe, C3 reports a release rate against the
reference it passed on. Gating a row on another kind's metric is the bug this module exists
to prevent.
"""
from __future__ import annotations

import json, sys
from dataclasses import dataclass
from pathlib import Path

PIPE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PIPE / "models"))

import model_registry

CKPT_SEQ_ACC_GATE = 0.95      # run_model_pipeline.SEQ_ACC_GATE: a sub-gate ckpt is not evidence
#: the documented probe causal gate, on flip_rate (NOT steered_rate); mirrored in
#: probes/validate_probe.py, which produces the rate and cannot import this module.
FLIP_GATE = 0.80
COLLATERAL_GATE = 0.05        # steering must not disturb earlier positions
D_CONTROL_MAX = 0.05          # severing a C1 fringe must not damage the base circuit's own pool
SUPPRESSION_EPSILON = 0.05    # smallest_faithful_circuit --epsilon default

#: smallest_faithful_circuit.build_payload names the measured accuracy per mode: the plain key
#: is the reported circuit's own, the fringe key the SOURCE CANDIDATE's (a different edge set).
PLAIN_ACC_KEY = "whole_sequence_absolute_accuracy"
FRINGE_ACC_KEY = "source_candidate_absolute_accuracy"


@dataclass(frozen=True)
class Defect:
    code: str
    detail: str


def full_node_budget(model: str) -> int:
    """Prunable node budget, the producer's definition (smallest_faithful_circuit ~L1913)."""
    spec = model_registry.get_spec(model)
    return spec.n_layers * spec.n_heads + (0 if spec.attn_only else spec.n_layers)


def full_edge_budget(model: str) -> int:
    """Full LIVE-edge budget from the model's geometry, the selection denominator -- never a
    finder's summary (EAP-IG counts phantom m{l} edges an attn-only model does not have)."""
    return model_registry.live_edges(model_registry.get_spec(model))


def candidate_accuracy(row: dict, kind: str) -> float:
    """The accuracy of the edge set it was MEASURED on. On a domain row that IS the reported
    circuit; on a C1/C3 row it is the SOURCE CANDIDATE's, so it may only ever be reported
    under a name that says so -- the fringe beside it has no standalone accuracy."""
    plain, fringe = row.get(PLAIN_ACC_KEY), row.get(FRINGE_ACC_KEY)
    if kind == "domain":
        if fringe is not None:
            raise ValueError(f"{row['circuit_name']}: domain row carries {FRINGE_ACC_KEY}")
        return row[PLAIN_ACC_KEY]
    if (plain is None) == (fringe is None):
        raise ValueError(f"{row['circuit_name']}: {kind} row must carry exactly one of "
                         f"{FRINGE_ACC_KEY}/{PLAIN_ACC_KEY}, has "
                         f"{sorted(k for k in (PLAIN_ACC_KEY, FRINGE_ACC_KEY) if k in row)}")
    return fringe if fringe is not None else plain   # pre-fix rows used the plain key's NAME


def winner_eval(circuit_dir: Path, circuit_name: str) -> dict | None:
    """The winning candidate's cached eval, the only place the row's MLP nodes survive."""
    p = circuit_dir / ".smallest_circuit_evals" / "evals" / f"{circuit_name}.json"
    if p.exists():
        return json.loads(p.read_text())
    for q in sorted((circuit_dir / ".smallest_circuit_evals" / "evals").glob("*.json")):
        e = json.loads(q.read_text())
        if e.get("name") == circuit_name:
            return e
    return None


def passed_reference(row: dict) -> str | None:
    """Which sever reference a C3 winner was accepted on: cued, tonic, or both."""
    rel, floor = row["release"], row["min_accuracy_threshold"]
    cued = rel["release_rate"] >= floor and row["collateral"] <= SUPPRESSION_EPSILON
    tonic = rel["release_rate_tonic"] >= floor and row["collateral_tonic"] <= SUPPRESSION_EPSILON
    return "both" if cued and tonic else "cued" if cued else "tonic" if tonic else None


def release_of_record(row: dict) -> float:
    """The release rate of the reference the row actually passed on -- the only one that ranks."""
    rel = row["release"]
    return rel["release_rate"] if passed_reference(row) in ("cued", "both") else rel["release_rate_tonic"]


def suppression_score(row: dict, ev: dict, node_budget: int, edge_budget: int) -> float:
    """_suppression_score with the release rate of the PASSED reference substituted in."""
    nodes = ev["n_fringe_heads"] + ev.get("n_fringe_mlps", 0)
    compact = 1.0 - (nodes / node_budget) * (ev["n_fringe_edges"] / edge_budget)
    return min(1.0, max(0.0, release_of_record(row))) * compact


def n_mlps(row_kind: str, ev: dict) -> int:
    """Kept MLP/MoE nodes of the row's edge set -- absent from smallest_circuit.json entirely,
    which is why rows on mlp/moe models can report n_heads 0. Both keys are omitted on
    attention-only models by design (smallest_faithful_circuit ~L1081), hence the 0 default."""
    return ev.get("n_fringe_mlps", 0) if row_kind != "domain" else ev.get("n_mlps", 0)


def circuit_defects(row: dict, kind: str, ev: dict | None,
                    ckpt_seq_acc: float | None) -> list[Defect]:
    """Every gate a circuit row fails, keyed to the metric that row reports."""
    out = []
    if ckpt_seq_acc is None:
        out.append(Defect("ckpt_accuracy_unrecorded", "owning checkpoint records no eval accuracy"))
    elif ckpt_seq_acc < CKPT_SEQ_ACC_GATE:
        out.append(Defect("sub_gate_checkpoint",
                          f"owner seq-acc {ckpt_seq_acc:.4f} < {CKPT_SEQ_ACC_GATE}"))
    if ev is None:
        return out + [Defect("winner_eval_missing", f"no cached eval for {row['circuit_name']}")]
    if row["n_heads"] == 0 and n_mlps(kind, ev) == 0:
        out.append(Defect("no_nodes", "circuit has neither attention heads nor MLP nodes"))
    by_kind = {"domain": _domain_defects, "contrast": _contrast_defects,
               "suppression": _suppression_defects}
    return out + by_kind[kind](row, ev)


def _domain_defects(row: dict, ev: dict) -> list[Defect]:
    out, acc, floor = [], candidate_accuracy(row, "domain"), row["min_accuracy_threshold"]
    if acc <= floor:
        out.append(Defect("accuracy_below_floor", f"{acc:.4f} <= own floor {floor}"))
    if not ev["perf_circuit"]["sequence_accuracy"] > ev["perf_empty"]["sequence_accuracy"]:
        out.append(Defect("no_recovery_over_empty",
                          f"R_sequence_accuracy {ev['R_sequence_accuracy']:.4f}: circuit "
                          f"{ev['perf_circuit']['sequence_accuracy']:.4f} does not beat the "
                          f"empty-ablation baseline {ev['perf_empty']['sequence_accuracy']:.4f}"))
    return out


def _contrast_defects(row: dict, ev: dict) -> list[Defect]:
    out, s_union, floor = [], row["sufficiency"]["s_union"], row["min_accuracy_threshold"]
    if s_union < floor:
        out.append(Defect("s_union_below_floor", f"{s_union:.4f} < own floor {floor}"))
    d_control = row["necessity"]["d_control"]
    if d_control > D_CONTROL_MAX:
        out.append(Defect("d_control_exceeded",
                          f"severing the fringe costs the base circuit {d_control:.4f} "
                          f"> {D_CONTROL_MAX}"))
    return out


def _suppression_defects(row: dict, ev: dict) -> list[Defect]:
    if passed_reference(row) is not None:
        return []
    return [Defect("no_passed_reference",
                   f"neither reference clears release {row['min_accuracy_threshold']} "
                   f"at collateral <= {SUPPRESSION_EPSILON}")]


def manifest_row_defects(row: dict) -> list[Defect]:
    """The summary-row consistency check: no accuracy may sit beside an edge count it was
    not measured on. `candidate_accuracy` is the SOURCE CANDIDATE's, while a C1/C3 row's
    n_heads/n_edges are its FRINGE, so an unqualified `accuracy` is only ever admissible
    when the row's edge count IS the measured one -- otherwise it must be named
    candidate_sequence_accuracy and carry candidate_n_edges."""
    out = []
    if row["kind"] == "probe":
        return out
    measured = row.get("candidate_n_edges")
    if measured is None:
        return [Defect("accuracy_edge_set_unstated",
                       "row reports no candidate_n_edges, so no accuracy in it can be attributed")]
    if row.get("accuracy") is not None and row["n_edges"] != measured:
        out.append(Defect("accuracy_edge_set_mismatch",
                          f"unqualified accuracy {row['accuracy']} sits beside {row['n_edges']} "
                          f"fringe edges but was measured on the candidate's {measured}"))
    if row["n_heads"] == 0 and not row.get("n_mlps"):
        out.append(Defect("zero_heads_unexplained",
                          "n_heads 0 and no MLP/MoE node count recorded"))
    return out


def decodable_phase(scan: dict) -> str | None:
    """Which position set the variable is linearly decodable at -- "input", "generation", or
    None. A variable that only materialises while decoding is real evidence, but it is NOT the
    input-position reading, so the phase travels with every row that reports it. Pre-rescan
    scans carry no gen_verdict at all: their generation phase was never measured."""
    if scan["verdict"] == "decodable":
        return "input"
    return "generation" if scan.get("gen_verdict") == "decodable" else None


def probe_defects(scan: dict, val: dict) -> list[Defect]:
    """Verdict at EITHER position set + the documented causal gate. The gate is on flip_rate,
    never steered_rate: steered_rate is the fraction of pairs where steering was ATTEMPTED."""
    out, cg = [], val["causal_gate"]
    if decodable_phase(scan) is None:
        out.append(Defect("verdict_not_decodable",
                          f"input {scan['verdict']}, generation "
                          f"{scan.get('gen_verdict', 'not_scanned')}"))
    if cg.get("flip_rate") is None:   # key absent = variable skipped; None = no row steered
        return out + [Defect("causal_gate_skipped",
                             cg.get("skipped") or cg.get("note") or "no flip_rate recorded")]
    if cg["flip_rate"] < FLIP_GATE:
        out.append(Defect("flip_rate_below_gate", f"{cg['flip_rate']:.4f} < {FLIP_GATE}"))
    if cg["collateral_rate"] > COLLATERAL_GATE:
        out.append(Defect("collateral_above_gate", f"{cg['collateral_rate']:.4f} > {COLLATERAL_GATE}"))
    if cg["base_model_seq_acc"] < CKPT_SEQ_ACC_GATE:
        out.append(Defect("base_model_sub_gate",
                          f"{cg['base_model_seq_acc']:.4f} < {CKPT_SEQ_ACC_GATE}"))
    return out


def checkpoint_accuracies(ckpt_dir: Path, cache_path: Path | None = None) -> dict[str, float | None]:
    """model -> recorded eval seq-acc (None when the checkpoint carries no metrics).
    mtime+size cached: unpickling every ckpt costs minutes and callers may run repeatedly."""
    import torch
    cache = json.loads(cache_path.read_text()) if cache_path and cache_path.exists() else {}
    out, fresh = {}, {}
    for p in sorted(ckpt_dir.glob("*.pt")):
        key = f"{p.stat().st_mtime_ns}:{p.stat().st_size}"
        hit = cache.get(p.stem)
        if hit is None or hit["key"] != key:
            d = torch.load(p, map_location="cpu", weights_only=False)
            hit = {"key": key, "acc": ((d.get("metrics") or {}).get("eval") or {}).get("accuracy")}
        fresh[p.stem] = hit
        out[p.stem] = hit["acc"]
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(fresh))
    return out
