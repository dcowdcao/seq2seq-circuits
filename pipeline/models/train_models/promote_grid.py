#!/usr/bin/env python3
"""Promote size-grid keepers into the real checkpoint tree.

The size grid (`train_sharded.py --sizes ...`, run locally or with `--modal`) writes
`trained_models_pt_grid/<name>.pt` + `architecture_grid.json` WITHOUT promoting.
This copies every keeper into `trained_models_pt/<name>.pt` so `run_model_pipeline.py`
gate-skips it, and prints the promoted variant names (the build list).

The keeper decision is made TWICE and both must agree: the grid's own `keeper` flag AND
`promote.check` re-reading the checkpoint's recorded seq-acc. A grid cell that claims
keeper but whose checkpoint does not record a passing accuracy is a hard failure, not a
promotion. See docs/reference/stacked_training.md.

  python promote_grid.py                       # promote all keepers in the default grid
  python promote_grid.py --models pmaxv9_id    # only these base models' keepers
"""
import argparse, json, sys
from pathlib import Path

_THIS = Path(__file__).resolve().parent
if str(_THIS) not in sys.path:
    sys.path.insert(0, str(_THIS))
import promote as PROMOTE                                            # noqa: E402

GRID = _THIS.parent / "trained_models_pt_grid"
REAL = _THIS.parent / "trained_models_pt"
#: friendly size -> __<suffix> ("" = large = no suffix). Mirrors train_sharded.SIZES.
SIZES = {"small": "small", "large": "", "wide": "wide", "deep": "deep", "xlarge": "xl"}


def variant_name(base: str, size: str) -> str:
    """The checkpoint/model name for (base, friendly-size); large has no suffix."""
    return base if not SIZES[size] else f"{base}__{SIZES[size]}"


def promote_keepers(grid_dir: Path, real_dir: Path, models: list | None,
                    gate: float = PROMOTE.GATE) -> list:
    """Copy every keeper checkpoint into real_dir via the ONE gate; return promoted names."""
    grid = json.load(open(grid_dir / "architecture_grid.json"))
    real_dir.mkdir(parents=True, exist_ok=True)
    promoted = []
    for base in sorted(grid):
        if models and base not in models:
            continue
        for size, cell in grid[base].items():
            name = variant_name(base, size)
            if not cell["keeper"]:
                print(f"  [skip] {name:40} acc={cell['acc']:.4f} (< target — not a keeper)")
                continue
            src = grid_dir / f"{name}.pt"
            if not src.exists():
                raise SystemExit(f"✗ keeper {name} has no checkpoint at {src}")
            if not PROMOTE.promote(src, real_dir / f"{name}.pt", gate):
                raise SystemExit(f"✗ grid says {name} is a keeper (acc={cell['acc']:.4f}) but its "
                                 f"checkpoint records no passing seq-acc — grid/checkpoint disagree")
            promoted.append(name)
    return promoted


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grid", default=str(GRID), help="grid dir (holds architecture_grid.json)")
    ap.add_argument("--out", default=str(REAL), help="real checkpoint tree to promote into")
    ap.add_argument("--models", nargs="*", help="restrict to these base model ids (default: all in the grid)")
    ap.add_argument("--gate", type=float, default=PROMOTE.GATE, help="seq-acc gate (default %(default)s)")
    a = ap.parse_args()
    grid_dir = Path(a.grid)
    if not (grid_dir / "architecture_grid.json").exists():
        raise SystemExit(f"✗ no architecture_grid.json in {grid_dir} — run the size grid first")
    promoted = promote_keepers(grid_dir, Path(a.out), a.models, a.gate)
    print(f"\n{len(promoted)} keeper(s) promoted:", " ".join(promoted) if promoted else "(none)")


if __name__ == "__main__":
    main()
