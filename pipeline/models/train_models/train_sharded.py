#!/usr/bin/env python3
"""Model-sharding launcher for the size-grid trainer. Splits models across
GPUs and runs one `train_stacked.py` per shard — vmap-stacked, ZERO cross-GPU comms,
then gather. See docs/reference/stacked_training.md.

Buckets by config FIRST (you can only vmap-stack identical configs), micro-benches N*
PER BUCKET (bigger sizes fit fewer/GPU), shards each bucket, and rounds shards over GPUs.
`--sizes` trains every model at every size (the __<size> variant grammar) and writes an
architecture_grid.json (model x size -> seq-acc) for later architecture selection.

Runs LOCALLY by default (--local is a no-op alias); --modal forwards the run to the
optional Modal multi-GPU container (modal_pipeline_app.run_sharded).

  python train_sharded.py                                  # models.txt at their own size
  python train_sharded.py --sizes small large wide deep xlarge   # full size grid
  python train_sharded.py --models rcnt3_id --sizes all --gpus 4
Extra training flags (--target/--seed/--seeds) forward to train_stacked; the RECIPE itself
comes from hparams.py, keyed by architecture, and is not a flag.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

_THIS = Path(__file__).resolve().parent
if str(_THIS) not in sys.path:
    sys.path.insert(0, str(_THIS))
import hparams                                                      # noqa: E402

DEFAULT_OUT = str(_THIS.parent / "trained_models_pt_grid")
REAL_OUT = str(_THIS.parent / "trained_models_pt")
PROMOTE_GRID = str(_THIS / "promote_grid.py")
_SUMMARY = re.compile(r"^\s+(PASS|fail)\s+(\S+)\s+([0-9.]+)")
#: friendly size name -> __<variant> suffix ("" = large tier = no suffix). Mirrors VARIANT_TIER.
SIZES = {"small": "small", "large": "", "wide": "wide", "deep": "deep", "xlarge": "xl"}
ALL_SIZES = ["small", "large", "wide", "deep", "xlarge"]


def _gpu_count() -> int:
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout
    except FileNotFoundError:                    # no NVIDIA driver: CPU-only machine
        return 0
    return sum(1 for ln in out.splitlines() if ln.startswith("GPU "))


def chunk(xs: list, size: int) -> list:
    return [xs[i:i + size] for i in range(0, len(xs), size)]


def expand_grid(bases: list, sizes: list) -> list:
    """base model x size -> __<size> variant names (large = no suffix)."""
    return [b if not SIZES[s] else f"{b}__{SIZES[s]}" for b in bases for s in sizes]


def size_of(name: str) -> tuple:
    """(base, friendly-size) from a variant name."""
    base, _, suf = name.partition("__")
    inv = {v: k for k, v in SIZES.items()}
    return base, inv.get(suf, suf or "large")


def _pack_for(spec: str, rep_name: str, batch: int) -> int:
    """N* for a bucket: explicit int, or micro-benched for this config+batch+GPU."""
    if spec != "auto":
        return int(spec)
    import torch  # only for auto-tuning; keeps the launcher torch-free otherwise
    if not torch.cuda.is_available():
        return 8
    import train_stacked as TS
    return TS.measure_pack_size(TS.Spec(rep_name).config, batch, "cuda")


def build_shards(names: list, pack_spec: str, batch: int) -> list:
    """Bucket by config, micro-bench N* per bucket, chunk each bucket into shards."""
    import train_stacked as TS
    buckets = {}
    for n in names:
        buckets.setdefault(TS.Spec(n).key(), []).append(n)
    shards = []
    for group in buckets.values():
        if not TS.Spec(group[0]).config.get("attn_only", True):   # vmap can't batch MLP/MoE forwards
            print(f"  [skip] {len(group)} non-attn-only model(s) {group} — vmap unsupported; "
                  f"train these via run_model_pipeline (per-model trainer)", flush=True)
            continue
        pack = _pack_for(pack_spec, group[0], batch)
        _, sz = size_of(group[0])
        print(f"  bucket {sz:7} ({len(group)} models)  N*={pack}/GPU", flush=True)
        shards += chunk(group, pack)
    return shards


def launch(gpu: int, models: list, out: str, passthrough: list) -> subprocess.Popen:
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "PYTHONUNBUFFERED": "1"}
    cmd = [sys.executable, str(_THIS / "train_stacked.py"),
           "--models", *models, "--out", out, *passthrough]
    return subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def write_grid(results: list, path: str) -> dict:
    """model x size -> {acc, keeper}, for later architecture selection."""
    grid = {}
    for name, acc, passed in results:
        base, size = size_of(name)
        grid.setdefault(base, {})[size] = {"acc": round(acc, 4), "keeper": passed}
    json.dump(grid, open(path, "w"), indent=2)
    return grid


def _run_on_modal(argv: list) -> None:
    """Opt-in --modal backend: forward this run to the Modal multi-GPU container (mirrors
    run_model_pipeline). Needs `modal deploy modal_pipeline_app.py`."""
    try:
        import modal
    except ImportError:
        raise SystemExit("✗ --modal needs `pip install modal` (+ `modal deploy "
                         "modal_pipeline_app.py`); drop --modal to run on this machine.")
    import io
    import tarfile as _tf
    argv = [x for x in argv if x != "--modal"]          # the container runs it locally
    print(f"dispatching to Modal: {' '.join(argv)}", flush=True)
    r = modal.Function.from_name("circuits-pipeline", "run_sharded").remote(" ".join(argv))
    if r["rc"] != 0:
        raise SystemExit(f"✗ Modal training failed (rc {r['rc']})")
    data = b"".join(modal.Volume.from_name("circuits-artifacts").read_file("grid.tar"))
    with _tf.open(fileobj=io.BytesIO(data)) as tf:
        tf.extractall(str(_THIS.parents[2]), filter="data")     # -> circuits/pipeline/...
    print(f"  [ok] ({r['elapsed_s']}s) grid synced -> models/trained_models_pt_grid/", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modal", action="store_true", help="opt-in: run on the Modal multi-GPU container")
    ap.add_argument("--local", action="store_true", help="no-op alias (local is the default)")
    ap.add_argument("--models", nargs="*", help="base model names (default: models/models.txt)")
    ap.add_argument("--sizes", nargs="*", help=f"train every model at these sizes; 'all' = {ALL_SIZES}")
    ap.add_argument("--models-per-gpu", default="auto",
                    help="int, or 'auto' = micro-bench N* PER bucket (default)")
    ap.add_argument("--gpus", type=int, default=0,
                    help="0 = derive from #shards: min(available GPUs, #shards)")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--promote", action="store_true",
                    help=f"after the grid, run promote_grid.py (the ONE gate) into {REAL_OUT}")
    a, passthrough = ap.parse_known_args()
    if a.modal:                                  # DEFAULT = local; --modal offloads
        _run_on_modal(sys.argv[1:])
        return

    # Always write the grid dir; --promote runs promote_grid afterwards, so no path
    # writes into the real tree without passing the keeper gate.
    out = a.out
    bases = a.models or [ln.split("#", 1)[0].strip() for ln in open(_THIS.parent / "models.txt")
                         if ln.split("#", 1)[0].strip()]
    sizes = (ALL_SIZES if a.sizes == ["all"] else a.sizes) if a.sizes else None
    names = expand_grid(bases, sizes) if sizes else bases
    os.makedirs(out, exist_ok=True)
    print(f"{len(bases)} base model(s)"
          + (f" x {len(sizes)} sizes = {len(names)} jobs" if sizes else "") + f"  out={out}", flush=True)
    shards = build_shards(names, a.models_per_gpu, hparams.BATCH)
    gpus = a.gpus or max(1, min(_gpu_count(), len(shards)))   # CPU-only: one worker
    print(f"{len(shards)} shard(s) across {gpus} GPU(s)", flush=True)

    pending, running, logs, t0 = list(enumerate(shards)), {}, {}, time.time()
    while pending or running:
        while pending and len(running) < gpus:
            sid, shard = pending.pop(0)
            running[sid] = (launch(sid % gpus, shard, out, passthrough), sid % gpus)
            print(f"  [shard {sid}] GPU {sid % gpus} <- {len(shard)} models", flush=True)
        done = [sid for sid, (p, _) in running.items() if p.poll() is not None]
        for sid in done:
            p, gpu = running.pop(sid)
            logs[sid] = p.stdout.read()
            print(f"  [shard {sid}] GPU {gpu} done (rc {p.returncode})", flush=True)
            if p.returncode:  # surface the failing shard's log tail (a stack-poisoning model hides there)
                print("\n".join(f"    | {l}" for l in logs[sid].splitlines()[-15:]), flush=True)
        if not done:
            time.sleep(2)

    results = [(m.group(2), float(m.group(3)), m.group(1) == "PASS")
               for sid in sorted(logs) for m in map(_SUMMARY.match, logs[sid].splitlines()) if m]
    npass = sum(p for _, _, p in results)
    print(f"\n=== SUMMARY ({time.time()-t0:.0f}s wall, {gpus} GPU(s)) ===")
    # ALWAYS write the grid json: it is promote_grid's input, so a no---sizes run must be
    # promotable through the same gate as a size-grid run.
    grid = write_grid(results, os.path.join(out, "architecture_grid.json"))
    if sizes:
        hdr = "".join(f"{s[:5]:>7}" for s in sizes)
        print(f"  {'model':20}{hdr}   (seq-acc; * = keeper >= target)")
        for base in sorted(grid):
            row = "".join((f"{grid[base][s]['acc']:.3f}" + ("*" if grid[base][s]["keeper"] else " "))[:7].rjust(7)
                          if s in grid[base] else f"{'-':>7}" for s in sizes)
            print(f"  {base:20}{row}")
        print(f"architecture grid -> {os.path.join(out, 'architecture_grid.json')}")
    else:
        for name, acc, passed in sorted(results, key=lambda r: -r[1]):
            print(f"  {'PASS' if passed else 'fail'}  {name:22} {acc:.4f}")
    print(f"{npass}/{len(results)} keepers; {len(results)-npass} need the escalation fallback")
    if a.promote:
        print(f"\npromoting keepers via {os.path.basename(PROMOTE_GRID)} (the ONE gate)", flush=True)
        subprocess.run([sys.executable, PROMOTE_GRID, "--grid", out, "--out", REAL_OUT]
                       + (["--models", *bases] if a.models else []), check=True)


if __name__ == "__main__":
    main()
