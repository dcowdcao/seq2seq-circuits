#!/usr/bin/env python3
"""The ONE promotion gate: a checkpoint enters models/trained_models_pt/ only through here.

There used to be four unconditional write paths, so 49 checkpoints below the 0.95 seq-acc
gate (40 of them below 0.50, 3 at exactly 0.0) shipped and 209 circuit rows were built on
them. promote_grid.py filtered keepers; run_training_job.sh copied best.pt after printing
RESULT: FAILURE; train_stacked.save_split wrote every spec regardless of --target;
run_model_pipeline's stacked pre-pass wrote straight into the real tree. All of them now
call `promote`.

A checkpoint with NO recorded accuracy is REFUSED, not promoted: an unmeasured checkpoint
cannot be filtered later, which is how the 49 got in.

    python promote.py <src.pt> <dst.pt> [--gate 0.95]     # exit 0 promoted, 1 rejected
    python promote.py --check <ckpt.pt> [--gate 0.95]     # verdict only, nothing copied
"""
from __future__ import annotations

import argparse
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

#: the pipeline's train gate (run_model_pipeline.SEQ_ACC_GATE).
GATE = 0.95
#: metric keys that hold a STRICT whole-sequence accuracy, most-authoritative first.
#: train.py's evaluate() and train_stacked's eval_acc both compute all-token exact match,
#: so their `eval.accuracy` IS a sequence accuracy despite the older name.
SEQ_ACC_KEYS = ("test_sequence_accuracy", "eval.sequence_accuracy", "val_sequence_accuracy",
                "sequence_accuracy", "eval.accuracy")


@dataclass(frozen=True)
class Verdict:
    acc: float | None
    source: str
    gate: float

    @property
    def passed(self) -> bool:
        return self.acc is not None and self.acc >= self.gate

    def line(self, name: str = "") -> str:
        shown = "NO RECORDED METRIC" if self.acc is None else f"seq-acc={self.acc:.5f}"
        return (f"{'PASS' if self.passed else 'REJECT'} {name} {shown} "
                f"(gate >= {self.gate}) from {self.source}")


def flat_metrics(metrics: dict) -> dict:
    """Flatten nested checkpoint metrics to dotted scalar keys."""
    out = {}

    def walk(d: dict, pfx: str = "") -> None:
        for k, v in d.items():
            if isinstance(v, dict):
                walk(v, pfx + k + ".")
            elif isinstance(v, (int, float)) and not isinstance(v, bool):
                out[pfx + k] = float(v)

    walk(metrics)
    return out


def recorded_seq_acc(state: dict) -> tuple:
    """(accuracy, source-key) recorded in a checkpoint, or (None, 'none') when absent."""
    metrics = state["metrics"] if isinstance(state, dict) and "metrics" in state else None
    if not isinstance(metrics, dict):
        return None, "none"
    flat = flat_metrics(metrics)
    for k in SEQ_ACC_KEYS:
        if k in flat:
            return flat[k], f"metrics.{k}"
    return None, "none"


def check(ckpt: str | Path, gate: float = GATE) -> Verdict:
    """Read a checkpoint's recorded sequence accuracy and judge it against the gate."""
    state = torch.load(ckpt, map_location="cpu", weights_only=False, mmap=True)
    acc, src = recorded_seq_acc(state)
    return Verdict(acc, src, gate)


def promote(src: str | Path, dst: str | Path, gate: float = GATE, quiet: bool = False) -> bool:
    """Copy src -> dst IFF its recorded seq-acc reaches the gate. Returns whether it did."""
    v = check(src, gate)
    name = Path(dst).stem
    if not quiet:
        print(f"  [{'promoted' if v.passed else 'blocked '}] {v.line(name)}", flush=True)
    if not v.passed:
        return False
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, dst)
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("src", help="checkpoint to gate (and, without --check, to promote)")
    ap.add_argument("dst", nargs="?", help="destination path (omit with --check)")
    ap.add_argument("--gate", type=float, default=GATE)
    ap.add_argument("--check", action="store_true", help="print the verdict; copy nothing")
    a = ap.parse_args()
    if a.check or a.dst is None:
        v = check(a.src, a.gate)
        print(v.line(Path(a.src).stem))
        sys.exit(0 if v.passed else 1)
    sys.exit(0 if promote(a.src, a.dst, a.gate) else 1)


if __name__ == "__main__":
    main()
