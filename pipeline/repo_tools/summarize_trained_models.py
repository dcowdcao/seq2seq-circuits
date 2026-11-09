#!/usr/bin/env python3
"""Audit trained_models_pt/ for training uniformity and emit a per-model summary JSON.

Classifies every checkpoint into eras (this-run vs legacy, via --run-manifest name list),
identifies the trainer (vmap-stacked promotes carry no optimizer state; the per-model
torchrun trainer does), and verifies each this-run checkpoint's config matches the
canonical tier for its variant suffix. Writes models/trained_models_summary.json.

  python repo_tools/summarize_trained_models.py --run-manifest <names.txt> [--scope <ids.txt>]
"""
import argparse, json, os, sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PIPE = os.path.dirname(HERE)
CKPT_DIR = os.path.join(PIPE, "models", "trained_models_pt")
OUT = os.path.join(PIPE, "models", "trained_models_summary.json")

sys.path.insert(0, os.path.join(os.path.dirname(PIPE), "interp_tools"))
from model_loading import TIER_PRESETS  # single source of truth for tier geometry
sys.path.insert(0, PIPE)
from run_model_pipeline import UPGRADE_LADDER

GEOM = ("n_layers", "n_heads", "d_model")
SUFFIX_TO_TIER = {"": "large", "xl": "xlarge"}


def _tier_of(suffix: str, cfg: dict) -> tuple:
    """-> (canonical|escalated|None, tier_name). Accepts the tier's geometry or its ladder upgrade's."""
    tier = SUFFIX_TO_TIER.get(suffix, suffix)
    if tier not in TIER_PRESETS:
        return None, tier
    if all(cfg.get(k) == TIER_PRESETS[tier][k] for k in GEOM):
        return "canonical", tier
    up = UPGRADE_LADDER.get(tier)
    if up and all(cfg.get(k) == TIER_PRESETS[up][k] for k in GEOM):
        return "escalated", up
    return None, tier


def classify(path: str, run_names: set, scope: set) -> dict:
    name = os.path.basename(path)[:-3]
    base, _, suffix = name.partition("__")
    d = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(d, dict) or "config" not in d:
        return {"name": name, "era": "legacy", "note": "non-standard payload"}
    cfg = dict(d["config"])
    ev = (d.get("metrics") or {}).get("eval") or {}
    trainer = "vmap-stacked" if "optimizer_state_dict" not in d else "per-model"
    fit, tier = _tier_of(suffix, cfg) if name in run_names else (None, suffix or "large")
    return {
        "name": name, "base": base, "variant": suffix or "large",
        "era": "this-run" if name in run_names else "legacy",
        "in_scope": base in scope,
        "trainer": trainer,
        "config": {k: cfg.get(k) for k in ("n_layers", "n_heads", "d_model", "d_vocab", "n_ctx", "attn_only", "d_mlp")},
        "eval_seq_acc": ev.get("accuracy"),
        "epoch_or_steps": (d.get("metrics") or {}).get("epoch", d.get("epoch")),
        "tier_fit": fit, "resolved_tier": tier,
        "file_mb": round(os.path.getsize(path) / 2**20, 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-manifest", required=True, help="file of <name>.pt lines = this run's ckpts")
    ap.add_argument("--scope", help="file of base ids in the run scope")
    a = ap.parse_args()
    run_names = {l.strip().removesuffix(".pt") for l in open(a.run_manifest) if l.strip()}
    scope = set(open(a.scope).read().split()) if a.scope else set()

    rows = [classify(os.path.join(CKPT_DIR, f), run_names, scope)
            for f in sorted(os.listdir(CKPT_DIR)) if f.endswith(".pt")]
    this_run = [r for r in rows if r["era"] == "this-run"]
    bad = [r for r in this_run if r.get("tier_fit") is None]
    summary = {
        "checkpoint_dir": "models/trained_models_pt",
        "totals": {"all": len(rows), "this_run": len(this_run),
                   "legacy": len(rows) - len(this_run),
                   "in_scope": sum(1 for r in this_run if r["in_scope"]),
                   "escalated": sum(1 for r in this_run if r["tier_fit"] == "escalated")},
        "uniformity": {"this_run_tier_mismatches": [r["name"] for r in bad],
                       "trainers": sorted({r["trainer"] for r in this_run})},
        "models": rows,
    }
    json.dump(summary, open(OUT, "w"), indent=1)
    print(f"wrote {OUT}: {len(rows)} ckpts ({len(this_run)} this-run, "
          f"{summary['totals']['escalated']} escalated, {len(bad)} tier mismatches)")
    for r in bad[:10]:
        print(f"  MISMATCH {r['name']}: {r['config']}")


if __name__ == "__main__":
    main()
