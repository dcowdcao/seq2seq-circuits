#!/usr/bin/env python3
"""Build the compact, diffable zoo indexes from the pipeline's (gitignored) artifacts.

Reads
  * ``<circuits-data>/<model>/<circuit>/smallest_circuit.json`` -- the selection verdict of
    ``pipeline/find_circuits/smallest_faithful_circuit.py`` (a winner or a
    ``no_faithful_circuit`` record),
  * ``<probes-data>/<model>/<variable>/{site_scan,validation}.json`` -- written by
    ``pipeline/probes/train_probes.py`` and ``pipeline/probes/validate_probe.py``,
  * ``<checkpoints>/*.pt`` -- the trained models (``config``, ``metrics``, ``epoch``),

and writes three CSVs next to this script (or to ``--out``):

  circuits_index.csv  one row per model x circuit (DomainCircuit / ContrastCircuit /
                      SuppressionCircuit), no edge lists
  probes_index.csv    one row per model x probed variable (ProbeCircuit / BridgeCircuit)
  models_index.csv    one row per checkpoint

Only the 8 zoo architectures (``ARCHES``) are indexed unless ``--include-other-variants`` is
given. No paths are written. Rows are sorted and floats rounded, so a re-run on the same artifacts
is byte-identical. Booleans are 1/0; an empty cell means "not recorded / not applicable".

    python results/zoo/build_index.py \
        --circuits-data pipeline/circuits_data --probes-data pipeline/probes_data \
        --checkpoints pipeline/models/trained_models_pt
    python results/zoo/zoo_stats.py          # regenerates results/zoo/stats.md

Columns (all three files start with model, base, arch; arch = model-name suffix, none = large)

circuits_index.csv
  circuit, class         circuit id; DomainCircuit (plain selection), ContrastCircuit
                         (differential), SuppressionCircuit (suppression)
  faithful               a winner was selected; not_faithful_reason = below_floor (evaluated,
                         nothing cleared the floor) or the reason recorded in the artifact
  run, floor, n_test     selection floor and eval size recorded in the artifact; run = main
                         (n_test 5000) / early (n_test 20000)
  winner, finder,        winning candidate (eap_top<n> / acdc_thresh_<t> /
  finder_param           edge_pruning_sparsity_<s>); for no-faithful suppression rows the
                         best-by-release candidate
  n_heads, n_edges       kept heads / edges as recorded (Contrast/Suppression: the fringe =
                         candidate minus base circuit)
  n_mlp_nodes,           kept MLP (or MoE block) nodes and their layers ("0;1;3"), from the
  mlp_layers             edge endpoints; 0 / empty for attention-only models
  n_edges_live           n_edges minus the EAP graph's phantom MLP edges of attention-only models
  n_heads_full,          full node / edge budget from the checkpoint geometry: heads, MLP nodes,
  n_mlp_full,            live edges (data_gen/canon.live_edge_budget) and the EAP-IG graph count
  n_edges_full,          that includes phantom MLP edges (the denominator of the recorded
  n_edges_full_graph     selection_score)
  abs_seq_acc            whole-sequence accuracy under edge-level mean ablation (Domain: the
                         circuit; Contrast/Suppression: the whole source candidate); for
                         no-faithful rows the best available value
  s_union, d_affected,   ContrastCircuit sufficiency (candidate + base) and severing deltas;
  d_unaffected,          s_union is the best available value on no-faithful rows
  d_control,
  local_predicate
  base_circuit,          the circuit the fringe is cut against and whether it was faithful
  base_faithful          (recorded fallback flag, else that circuit's own verdict)
  release_rate(_tonic),  SuppressionCircuit release of the wrong task when the fringe is severed
  collateral(_tonic),    against cued-row (tonic: wrong-cue-row) means; release_reference = the
  release_reference      reference(s) that pass release >= 0.5 with collateral <= 0.05
  router_verdict         recorded routing diagnosis of the multiplexed model
  selection_score        as recorded
  geometry_source        checkpoint, or arch_default when the model has no checkpoint
  has_checkpoint         the model has a checkpoint in --checkpoints

probes_index.csv
  variable, class        variable id; BridgeCircuit when it is a bridge<k>_* intermediate
  n_classes, n, n_layers, n_positions, n_trimmed_positions   probe set-up
  chance_uniform, majority_rate, baseline_acc   1/n_classes, majority-class rate, layer-0
                         embedding (resid_pre) probe accuracy
  acc_L0..acc_L7         position-mean held-out accuracy of the per-position probes on each
                         layer's resid_post (input positions only)
  best_layer, best_acc   highest-accuracy layer (ties -> lowest) and its accuracy
  verdict, first_decodable_layer   train_probes.verdict
  eval_layer ... base_model_seq_acc   validate_probe.py: evaluation layer, permutation-control
                         accuracy, robustness agreement, causal_gate = measured / skipped /
                         no_steered_rows, its rates, gate_passed = flip >= 0.80 and
                         collateral <= 0.05

models_index.csv
  tier, escalated        trainer tier matching the checkpoint geometry; escalated = larger
                         than the suffix's tier
  n_layers ... d_vocab   checkpoint config (attn_only defaults to 1 when absent)
  n_params, param_dtype  floating-point parameter count and stored dtype(s)
  eval_seq_acc           recorded eval whole-sequence accuracy
  metrics_layout         steps+eval (epoch = optimizer steps), epoch+train+eval (epoch index,
                         train metrics stored) or flat+recipe (recipe written into metrics)
  epoch ... hparams_regime   training fields as recorded (empty when absent)
"""
import argparse
import csv
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PIPELINE = os.path.join(os.path.dirname(os.path.dirname(HERE)), "pipeline")

#: model-name suffix -> architecture label ("" = the unsuffixed default, Large).
ARCHES = ("small", "large", "wide", "deep", "xl", "mlp", "moe4", "moe8")
#: geometry of each trainer tier (mirrors interp_tools/model_loading.TIER_PRESETS); used to
#: name a checkpoint's tier and as the fallback geometry of a model with no checkpoint.
TIERS = {
    "small": dict(n_layers=2, n_heads=4, d_model=64, attn_only=True, d_mlp=None, num_experts=None),
    "large": dict(n_layers=4, n_heads=8, d_model=128, attn_only=True, d_mlp=None, num_experts=None),
    "wide": dict(n_layers=4, n_heads=16, d_model=256, attn_only=True, d_mlp=None, num_experts=None),
    "deep": dict(n_layers=8, n_heads=8, d_model=128, attn_only=True, d_mlp=None, num_experts=None),
    "xlarge": dict(n_layers=8, n_heads=16, d_model=256, attn_only=True, d_mlp=None, num_experts=None),
    "mlp": dict(n_layers=4, n_heads=8, d_model=128, attn_only=False, d_mlp=512, num_experts=None),
    "moe4": dict(n_layers=4, n_heads=8, d_model=128, attn_only=False, d_mlp=128, num_experts=4),
    "moe8": dict(n_layers=4, n_heads=8, d_model=128, attn_only=False, d_mlp=128, num_experts=8),
    "mlp_xl": dict(n_layers=8, n_heads=16, d_model=256, attn_only=False, d_mlp=1024, num_experts=None),
    "moe4_xl": dict(n_layers=8, n_heads=16, d_model=256, attn_only=False, d_mlp=256, num_experts=4),
    "moe8_xl": dict(n_layers=8, n_heads=16, d_model=256, attn_only=False, d_mlp=256, num_experts=8),
}
ARCH_TIER = {a: ("xlarge" if a == "xl" else a) for a in ARCHES}
CLASS_OF_MODE = {"plain": "DomainCircuit", "differential": "ContrastCircuit",
                 "suppression": "SuppressionCircuit"}
RELEASE_FLOOR, COLLATERAL_EPS = 0.5, 0.05      # suppression gates (smallest_faithful_circuit.py)
FLIP_GATE, COLLATERAL_GATE = 0.80, 0.05        # probe causal gate (validate_probe.py)
MAX_LAYERS = 8                                 # probes_index carries acc_L0..acc_L7


def split_model(model):
    """'foo_id__moe4' -> ('foo_id', 'moe4'); 'foo_id' -> ('foo_id', 'large')."""
    base, _, suffix = model.partition("__")
    return base, suffix or "large"


def live_edges(n_layers, n_heads, attn_only):
    """Live attribution-graph edge count (data_gen/canon.live_edge_budget)."""
    L, H = n_layers, n_heads
    if attn_only:
        return 3 * H * (L + H * L * (L - 1) // 2) + 1 + L * H
    writers = L + (H + 1) * L * (L - 1) // 2
    return 3 * H * writers + (writers + L * H) + (1 + L * H + L)


def tier_of(geom):
    """Name of the TIERS entry matching a geometry, or 'custom'."""
    keys = ("n_layers", "n_heads", "d_model", "attn_only", "d_mlp", "num_experts")
    for name, t in TIERS.items():
        if all(geom.get(k) == t[k] for k in keys):
            return name
    return "custom"


def fmt(x):
    if x is None:
        return ""
    if isinstance(x, bool):
        return "1" if x else "0"
    if isinstance(x, float):
        return repr(round(x, 6))
    return str(x)


def write_csv(path, cols, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(cols)
        for r in rows:
            w.writerow([fmt(r.get(c)) for c in cols])
    print(f"wrote {os.path.relpath(path)}  ({len(rows)} rows)")


def subdirs(path):
    return sorted(d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d)))


# --------------------------------------------------------------------------- models
MODEL_COLS = ["model", "base", "arch", "tier", "escalated", "n_layers", "n_heads", "d_model",
              "d_head", "d_mlp", "num_experts", "experts_per_token", "attn_only", "n_ctx",
              "d_vocab", "n_params", "param_dtype", "eval_seq_acc", "metrics_layout", "epoch",
              "train_acc", "train_loss", "lr", "batch_size", "steps", "max_steps",
              "early_stopped", "seed", "hparams_regime"]


def checkpoint_row(path):
    import torch
    ck = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    model = os.path.basename(path)[:-3]
    base, arch = split_model(model)
    cfg, m = ck.get("config", {}), ck.get("metrics", {}) or {}
    geom = dict(n_layers=cfg["n_layers"], n_heads=cfg["n_heads"], d_model=cfg["d_model"],
                attn_only=bool(cfg.get("attn_only", True)),   # model_loading's default
                d_mlp=cfg.get("d_mlp"), num_experts=cfg.get("num_experts"))
    tier = tier_of(geom)
    sd = ck.get("model_state_dict", {})
    floats = [v for v in sd.values() if hasattr(v, "is_floating_point") and v.is_floating_point()]
    hp = m.get("hparams") or {}
    if "eval" in m:                       # nested {epoch, [train], eval} metrics
        layout = "epoch+train+eval" if "train" in m else "steps+eval"
        acc = m["eval"].get("sequence_accuracy", m["eval"].get("accuracy"))
    else:                                 # flat metrics with the recipe written in
        layout = "flat+recipe"
        acc = m.get("test_sequence_accuracy")
    return {
        "model": model, "base": base, "arch": arch, "tier": tier,
        "escalated": arch in ARCH_TIER and tier != ARCH_TIER[arch],
        **geom, "d_head": cfg.get("d_head"), "experts_per_token": cfg.get("experts_per_token"),
        "n_ctx": cfg.get("n_ctx"), "d_vocab": cfg.get("d_vocab"),
        "n_params": sum(v.numel() for v in floats),
        "param_dtype": ";".join(sorted({str(v.dtype).replace("torch.", "") for v in floats})),
        "eval_seq_acc": acc, "metrics_layout": layout, "epoch": ck.get("epoch"),
        "train_acc": (m.get("train") or {}).get("accuracy"),
        "train_loss": (m.get("train") or {}).get("loss"),
        "lr": hp.get("lr", m.get("lr")), "batch_size": hp.get("batch", m.get("batch_size")),
        "steps": hp.get("steps", m.get("steps")), "max_steps": m.get("max_steps"),
        "early_stopped": m.get("early_stopped"), "seed": hp.get("seed", m.get("seed")),
        "hparams_regime": hp.get("regime"),
    }


def build_models(ckpt_dir):
    if not os.path.isdir(ckpt_dir):
        print(f"no checkpoint dir {ckpt_dir}: models_index is empty")
        return []
    names = sorted(f for f in os.listdir(ckpt_dir) if f.endswith(".pt"))
    return [checkpoint_row(os.path.join(ckpt_dir, f)) for f in names]


# --------------------------------------------------------------------------- circuits
CIRCUIT_COLS = ["model", "base", "arch", "circuit", "class", "faithful", "not_faithful_reason",
                "run", "floor", "n_test", "winner", "finder", "finder_param",
                "n_heads", "n_heads_full", "n_mlp_nodes", "n_mlp_full", "mlp_layers",
                "n_edges", "n_edges_live", "n_edges_full", "n_edges_full_graph",
                "abs_seq_acc", "s_union", "d_affected", "d_unaffected", "d_control",
                "local_predicate", "base_circuit", "base_faithful", "release_rate",
                "release_rate_tonic", "collateral", "collateral_tonic", "release_reference",
                "router_verdict", "selection_score", "geometry_source", "has_checkpoint"]
_MLP = re.compile(r"m(\d+)$")


def finder_of(name):
    """'eap_top871' -> ('eap', '871'); 'acdc_thresh_0.1' -> ('acdc', '0.1')."""
    for prefix, finder in (("eap_top", "eap"), ("acdc_thresh_", "acdc"),
                           ("edge_pruning_sparsity_", "edge_pruning")):
        if name and name.startswith(prefix):
            return finder, name[len(prefix):]
    return (name or None), None


def declared_class(base, circuit):
    """Class from the circuit registry (pipeline/models/circuits.py), for the reason-only
    markers that record no selection mode (e.g. positional_only_candidates)."""
    sys.path.insert(0, os.path.join(PIPELINE, "models"))
    import circuits as C                                   # noqa: E402 (needs torch)
    cls = next(c for c in C.CIRCUITS if c.algorithm == base and c.circuit_id == circuit)
    return ("SuppressionCircuit" if issubclass(cls, C.SuppressionContrastCircuit) else
            "ContrastCircuit" if issubclass(cls, C.ContrastCircuit) else "DomainCircuit")


def circuit_row(model, circuit, d, geom, geometry_source, has_ckpt):
    base, arch = split_model(model)
    mode = d.get("mode", "plain")
    klass = (CLASS_OF_MODE[mode] if "mode" in d or "selection_strategy" in d
             else declared_class(base, circuit))
    nf = bool(d.get("no_faithful_circuit"))
    L, H, attn_only = geom["n_layers"], geom["n_heads"], geom["attn_only"]
    n_test = d.get("n_test")
    row = {"model": model, "base": base, "arch": arch, "circuit": circuit,
           "class": klass, "faithful": not nf,
           "not_faithful_reason": (d.get("reason") or "below_floor") if nf else None,
           "run": {5000: "main", 20000: "early"}.get(n_test), "floor": d.get("min_accuracy_threshold"),
           "n_test": n_test, "n_heads_full": L * H, "n_mlp_full": 0 if attn_only else L,
           "n_edges_full": live_edges(L, H, attn_only),
           "n_edges_full_graph": live_edges(L, H, False),    # EAP-IG graph incl. phantom m{l}
           "base_circuit": d.get("base_circuit"), "base_faithful": d.get("base_faithful"),
           "router_verdict": d.get("router_verdict"), "selection_score": d.get("selection_score"),
           "geometry_source": geometry_source, "has_checkpoint": has_ckpt}
    row["winner"] = d.get("circuit_name") or d.get("best_available_candidate")
    row["finder"], row["finder_param"] = finder_of(row["winner"])
    if "edges" in d:
        edges = d["edges"]
        mlps = sorted({int(mm.group(1)) for e in edges for x in (e["parent"], e["child"])
                       if (mm := _MLP.match(x))})
        phantom = sum(1 for e in edges if _MLP.match(e["parent"]) or _MLP.match(e["child"]))
        row.update(n_heads=d["n_heads"], n_edges=d["n_edges"],
                   n_mlp_nodes=0 if attn_only else len(mlps),
                   mlp_layers="" if attn_only else ";".join(map(str, mlps)),
                   n_edges_live=d["n_edges"] - (phantom if attn_only else 0))
    row["abs_seq_acc"] = d.get("whole_sequence_absolute_accuracy", d.get("best_available_accuracy"))
    if mode == "differential":
        nec = d.get("necessity", {})
        row.update(s_union=d.get("sufficiency", {}).get("s_union", d.get("best_available_s_union")),
                   d_affected=nec.get("d_affected"), d_unaffected=nec.get("d_unaffected"),
                   d_control=nec.get("d_control"), local_predicate=nec.get("local_predicate"))
    if mode == "suppression":
        rel = d.get("release", {})
        row.update(release_rate=rel.get("release_rate", d.get("best_available_release")),
                   release_rate_tonic=rel.get("release_rate_tonic",
                                              d.get("best_available_release_tonic")),
                   collateral=d.get("collateral"), collateral_tonic=d.get("collateral_tonic"))
        if not nf:
            cued = (row["release_rate"] >= RELEASE_FLOOR and row["collateral"] <= COLLATERAL_EPS)
            tonic = (row["release_rate_tonic"] >= RELEASE_FLOOR
                     and row["collateral_tonic"] <= COLLATERAL_EPS)
            row["release_reference"] = {(True, True): "both", (True, False): "cued",
                                        (False, True): "tonic"}.get((cued, tonic), "neither")
    return row


def build_circuits(circ_dir, geoms):
    rows = []
    for model in subdirs(circ_dir):
        base, arch = split_model(model)
        if model in geoms:
            geom, src = geoms[model], "checkpoint"
        else:
            geom, src = TIERS.get(ARCH_TIER.get(arch, "large"), TIERS["large"]), "arch_default"
        for circuit in subdirs(os.path.join(circ_dir, model)):
            p = os.path.join(circ_dir, model, circuit, "smallest_circuit.json")
            if os.path.isfile(p):
                with open(p) as f:
                    rows.append(circuit_row(model, circuit, json.load(f), geom, src,
                                            model in geoms))
    # base_faithful: the base circuit's own verdict when the record does not carry one
    verdict = {(r["model"], r["circuit"]): r["faithful"] for r in rows}
    for r in rows:
        if r["base_circuit"] and r["base_faithful"] is None:
            r["base_faithful"] = verdict.get((r["model"], r["base_circuit"]))
    return rows


# --------------------------------------------------------------------------- probes
PROBE_COLS = (["model", "base", "arch", "variable", "class", "n_classes", "n", "n_layers",
               "n_positions", "n_trimmed_positions", "chance_uniform", "majority_rate",
               "baseline_acc"] + [f"acc_L{l}" for l in range(MAX_LAYERS)]
              + ["best_layer", "best_acc", "verdict", "first_decodable_layer", "eval_layer",
                 "eval_layer_is_first_decodable", "permutation_acc", "robustness_agree",
                 "causal_gate", "probe_reads_source_rate", "steered_rate", "mean_alpha",
                 "flip_rate", "suffix_consistent_rate", "collateral_rate", "gate_passed",
                 "base_model_seq_acc", "has_checkpoint"])


def probe_row(model, variable, scan, val, has_ckpt):
    base, arch = split_model(model)
    acc = {int(l): a for l, a in scan["per_layer_accuracy"].items()}
    if max(acc) >= MAX_LAYERS:
        raise SystemExit(f"{model}/{variable}: {max(acc) + 1} layers > MAX_LAYERS={MAX_LAYERS}")
    best = min(acc, key=lambda l: (-acc[l], l))          # argmax, ties -> lowest layer
    row = {"model": model, "base": base, "arch": arch, "variable": variable,
           "class": "BridgeCircuit" if re.match(r"bridge\d+_", variable) else "ProbeCircuit",
           "n_classes": scan["n_classes"], "n": scan["n"], "n_layers": len(acc),
           "n_positions": len(scan["positions"]),
           "n_trimmed_positions": len(scan.get("trimmed_positions", [])),
           "chance_uniform": 1 / scan["n_classes"], "majority_rate": scan["majority_class_rate"],
           "baseline_acc": scan["baseline_accuracy"],
           **{f"acc_L{l}": a for l, a in acc.items()},
           "best_layer": best, "best_acc": acc[best], "verdict": scan["verdict"],
           "first_decodable_layer": scan["first_decodable_layer"], "has_checkpoint": has_ckpt}
    if val:
        cg = val.get("causal_gate", {})
        status = ("skipped" if "skipped" in cg else
                  "measured" if cg.get("flip_rate") is not None else "no_steered_rows")
        row.update(eval_layer=val.get("eval_layer"),
                   eval_layer_is_first_decodable=val.get("eval_layer_is_first_decodable"),
                   permutation_acc=val.get("permutation_control", {}).get("accuracy"),
                   robustness_agree=val.get("robustness", {}).get("agree"),
                   causal_gate=status,
                   **{k: cg.get(k) for k in ("probe_reads_source_rate", "steered_rate",
                                             "mean_alpha", "flip_rate", "suffix_consistent_rate",
                                             "collateral_rate", "base_model_seq_acc")})
        if status == "measured":
            row["gate_passed"] = (cg["flip_rate"] >= FLIP_GATE
                                  and cg["collateral_rate"] <= COLLATERAL_GATE)
    return row


def build_probes(probe_dir, have_ckpt):
    rows = []
    for model in subdirs(probe_dir):
        for variable in subdirs(os.path.join(probe_dir, model)):
            d = os.path.join(probe_dir, model, variable)
            if not os.path.isfile(os.path.join(d, "site_scan.json")):
                continue
            with open(os.path.join(d, "site_scan.json")) as f:
                scan = json.load(f)
            val = None
            if os.path.isfile(os.path.join(d, "validation.json")):
                with open(os.path.join(d, "validation.json")) as f:
                    val = json.load(f)
            rows.append(probe_row(model, variable, scan, val, model in have_ckpt))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--circuits-data", default=os.path.join(PIPELINE, "circuits_data"))
    ap.add_argument("--probes-data", default=os.path.join(PIPELINE, "probes_data"))
    ap.add_argument("--checkpoints", default=os.path.join(PIPELINE, "models", "trained_models_pt"))
    ap.add_argument("--out", default=HERE)
    ap.add_argument("--overwrite", action="store_true",
                    help="replace index CSVs already present in --out (the committed snapshot)")
    ap.add_argument("--include-other-variants", action="store_true",
                    help="also index models whose suffix is not one of the 8 ARCHES "
                         "(e.g. retired experimental variants); off by default")
    a = ap.parse_args()
    missing = [d for d in (a.circuits_data, a.probes_data, a.checkpoints) if not os.path.isdir(d)]
    if missing:   # fail before overwriting any committed index
        ap.error(f"input directory not found: {', '.join(missing)} -- rerun the zoo first "
                 "(docs/reproducing.md, tier D) or pass --circuits-data/--probes-data/--checkpoints")
    existing = [f for f in ("models_index.csv", "circuits_index.csv", "probes_index.csv")
                if os.path.isfile(os.path.join(a.out, f))]
    if existing and not a.overwrite:   # the committed CSVs are the zoo snapshot
        ap.error(f"{', '.join(existing)} already in {a.out}: pass --overwrite to replace them, "
                 "or --out DIR to write elsewhere")

    def keep(rows):
        return rows if a.include_other_variants else [r for r in rows if r["arch"] in ARCHES]

    models = keep(build_models(a.checkpoints))
    geoms = {r["model"]: {k: r[k] for k in TIERS["large"]} for r in models}
    write_csv(os.path.join(a.out, "models_index.csv"), MODEL_COLS, models)
    write_csv(os.path.join(a.out, "circuits_index.csv"), CIRCUIT_COLS,
              keep(build_circuits(a.circuits_data, geoms)))
    write_csv(os.path.join(a.out, "probes_index.csv"), PROBE_COLS,
              keep(build_probes(a.probes_data, set(geoms))))


if __name__ == "__main__":
    main()
