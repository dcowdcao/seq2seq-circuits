#!/usr/bin/env python3
"""build_models.py — thin batch driver for the DSL circuit-discovery pipeline.

Every model is a DSL ``algorithm_id`` whose task, output length, and architecture
are recovered from the NAME alone (models/algorithms.py). This driver just:

  1. reads the RUN LIST — ``models/models.txt`` (one algorithm_id per line;
     blank lines and ``#`` comments skipped), or ``--model`` / ``--only``;
  2. validates each name is a runnable DSL algorithm_id;
  3. delegates the WHOLE list to one ``run_model_pipeline.py`` invocation, which
     runs every model/circuit locally by default, or concurrently on Modal GPUs
     with --modal (one invocation per distinct threshold when
     --per-model-threshold is used).

Usage (from the repo root; the venv is ./.venv):
  ./.venv/bin/python pipeline/build_models.py                  # all of models/models.txt, locally
  ./.venv/bin/python pipeline/build_models.py --modal          # same, offloaded to Modal
  ./.venv/bin/python pipeline/build_models.py --only rev_id    # one entry
  ./.venv/bin/python pipeline/build_models.py --model take5_id --threshold 90
  ./.venv/bin/python pipeline/build_models.py --smoke          # tiny throwaway dev loop
  ./.venv/bin/python pipeline/build_models.py --list-models    # print the run-list
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

PIPELINE = os.path.dirname(os.path.abspath(__file__))      # .../pipeline
REPO = os.path.dirname(PIPELINE)                            # .../circuits
MODELS_DIR = os.path.join(PIPELINE, "models")
RUN_PIPELINE = os.path.join(PIPELINE, "run_model_pipeline.py")
MODELS_TXT = os.path.join(MODELS_DIR, "models.txt")

if MODELS_DIR not in sys.path:
    sys.path.insert(0, MODELS_DIR)


def _banner(title: str) -> None:
    print(f"\n── {title} " + "─" * max(2, 72 - len(title)), flush=True)


def _normalize_threshold(raw: str) -> float:
    """Parse a whole-sequence accuracy threshold. Accepts '80', '80%', '0.8'.
    Returns a fraction in (0, 1]."""
    s = str(raw).strip().rstrip("%").strip()
    x = float(s)
    if x > 1.0:
        x = x / 100.0
    if not (0.0 < x <= 1.0):
        raise ValueError(f"threshold must be in (0, 100]%, got {raw!r}")
    return x


def _load_names(path: str) -> list[str]:
    """Model names from the flat run-list: one DSL algorithm_id per line; blank lines
    and ``#`` comments skipped, inline ``# ...`` stripped."""
    return [ln.split("#", 1)[0].strip() for ln in open(path)
            if ln.split("#", 1)[0].strip()]


def _runnable(name: str, model_registry) -> bool:
    """True iff ``name`` resolves to a runnable spec (a valid DSL algorithm_id)."""
    try:
        model_registry.get_spec(name)
        return True
    except KeyError:
        return False


def run_group(names: list[str], target, a) -> int:
    """Delegate a group of models (one shared threshold) to ONE run_model_pipeline
    invocation — with --modal it parallelizes across models/circuits; return its exit code."""
    argv = [sys.executable, RUN_PIPELINE, "--model", *names,
            "--stages", *a.stages.split(), "--device", a.device]
    if a.smoke:
        argv.append("--smoke")
    if a.modal:
        argv.append("--modal")
    if target is not None:
        argv += ["--smallest-target", str(target)]
    if a.dry_run:
        argv.append("--dry-run")
    return subprocess.run(argv, cwd=REPO).returncode


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        prog="build_models.py",
        description="Thin driver: run the DSL circuit-discovery pipeline over a run-list.")
    ap.add_argument("--model", action="append", metavar="NAME",
                    help="run this DSL algorithm_id (repeatable); overrides the run-list")
    ap.add_argument("--only", default=None, metavar="N1,N2",
                    help="comma-separated subset of the run-list to run")
    ap.add_argument("--threshold", default=None, metavar="X",
                    help="smallest-circuit whole-sequence accuracy floor for ALL "
                         "models ('80', '80%%', or '0.8')")
    ap.add_argument("--per-model-threshold", action="append", metavar="NAME=X",
                    help="per-model threshold override (repeatable)")
    ap.add_argument("--stages", default="all",
                    help="pipeline stages to run per model (default: all). The "
                         "3-phase manual-circuit workflow over the run-list: "
                         "(1) '--stages train', then (2) declare each sub-circuit in "
                         "models/human_circuit_finding.json -> a DomainCircuit subclass in "
                         "models/circuits.py "
                         "+ '--stages data', then (3) '--stages \"patching edges eap "
                         "smallest audit\"' (EAP-IG is the only default finder; name "
                         "acdc/edge_pruning explicitly to add them). Imported "
                         "checkpoints: skip phase 1.")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--modal", action="store_true",
                    help="opt-in: offload GPU stages to Modal (default: run locally)")
    ap.add_argument("--local", action="store_true",
                    help="no-op alias kept for backward compatibility (local is the default)")
    ap.add_argument("--smoke", action="store_true",
                    help="forward --smoke to run_model_pipeline (tiny throwaway run)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print each model's plan without executing")
    ap.add_argument("--list-models", action="store_true",
                    help="print the run-list (models/models.txt) and exit")
    return ap.parse_args(argv)


def main(argv=None) -> None:
    a = parse_args(argv)
    import model_registry                                   # noqa: E402 (defer torch)

    if a.list_models:
        _banner("DSL model run-list (models/models.txt)")
        for nm in _load_names(MODELS_TXT):
            print(f"  {nm}")
        return

    names = list(a.model) if a.model else _load_names(MODELS_TXT)
    if a.only:
        wanted = {s.strip() for s in a.only.split(",") if s.strip()}
        names = [n for n in names if n in wanted]
    if not names:
        raise SystemExit("no models to run — populate "
                         "models/models.txt or pass --model/--only.")

    bad = [n for n in names if not _runnable(n, model_registry)]
    if bad:
        raise SystemExit(f"✗ not runnable: {bad}. Each name must be a valid DSL "
                         f"algorithm_id (see --list-models).")

    default_t = _normalize_threshold(a.threshold) if a.threshold else None
    thresholds = {n: default_t for n in names}
    for spec in (a.per_model_threshold or []):
        nm, _, x = spec.partition("=")
        thresholds[nm.strip()] = _normalize_threshold(x)

    # one invocation per distinct threshold; each parallelizes internally (on Modal with --modal)
    groups: dict = {}
    for nm in names:
        groups.setdefault(thresholds[nm], []).append(nm)

    failures = []
    for tgt, group in groups.items():
        _banner(f"{len(group)} model(s)" + (f"  (floor acc>{tgt})" if tgt is not None else "")
                + f": {' '.join(group)}")
        rc = run_group(group, tgt, a)
        if rc:
            failures.append((tuple(group), rc))

    _banner("build_models COMPLETE" + ("  [DRY RUN: nothing executed]" if a.dry_run else ""))
    if failures:
        raise SystemExit(f"✗ {len(failures)} group(s) had failures: {failures} "
                         "(see the per-job summary above)")


if __name__ == "__main__":
    main()
