"""modal_pipeline_app.py — OPTIONAL Modal execution backend: one H100 per job.

The pipeline runs locally by default. With ``run_model_pipeline.py --modal``,
every GPU stage (train + all analysis + smallest) runs here; only spawn/poll/sync
and the audit stay on the local machine. A container runs its analysis stages as a
parallel wave (PIPELINE_GPU_BUDGET is split across the wave's children); artifacts
tar back through the shared Volume. The whole pipeline/ tree plus the
shared repo-root modules ride the image, so RE-DEPLOY AFTER ANY CODE CHANGE:

  modal deploy pipeline/modal_pipeline_app.py

AUTH: run ``modal setup`` once (writes ~/.modal.toml); never put tokens here.

RUN — normally just use the orchestrator with --modal:
  ./.venv/bin/python pipeline/run_model_pipeline.py --model <name> --stages all --modal
or batch everything remote directly:
  modal run pipeline/modal_pipeline_app.py --only rcnt3_id

torch + transformer_lens are installed from stable PyPI wheels (reproducible
image builds).
"""

from __future__ import annotations

import io
import os
import tarfile
from pathlib import Path

import modal

_APP_DIR = Path(__file__).resolve().parent          # .../pipeline
_REPO = _APP_DIR.parent                             # .../circuits
REMOTE_REPO = "/workspace/circuits"

app = modal.App("circuits-pipeline")

_IGNORE_PIPELINE = [
    "**/__pycache__/**", "**/.smoke/**", "**/circuits_data/**", "**/circuits_figures/**",
    "**/gen_data/*/data/**", "**/gen_data/**/inputs.json",
    # trained_models_pt/*.pt are SHIPPED (not ignored): a model whose checkpoint is
    # already on Modal SKIPS --remote's forced `train` (it passes the seq-acc gate)
    # instead of retraining and shadowing the original.
    "**/*.png", "**/*.zip",
    "**/*.gv", "docs/**",
    "**/trained_models_pt_grid/**", "**/trained_models_pt_stacked/**",  # grid ckpts are (re)created remotely
]

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("graphviz", "libgraphviz-dev", "pkg-config")  # pygraphviz build deps (acdc/eap circuit drawing)
    .pip_install(
        "torch",                       # default PyPI wheel is CUDA-enabled (H100 ok)
        "transformer_lens==3.3.0",
        "einops>=0.8", "jaxtyping>=0.3", "numpy>=2", "pandas>=2", "tqdm",
        "matplotlib",                  # edge_pruning circuit.png render (else logs a skip)
        "pygraphviz",                  # vendored ACDC/EAP-IG circuit drawing (acdc + eap stages)
    )
    .env({
        "SEGREV3_DEVICE": "cuda",
        "SEGREV3_GPU_MEM_FRACTION": "0.9",   # serial stages (train) own the H100
        "PIPELINE_GPU_BUDGET": "0.9",        # parallel waves split this across their children
        "SEGREV3_MIN_FREE_GIB": "2",         # relax the host-RAM headroom floor in containers
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",  # avoid fragmentation across 5 procs
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
    })
    .add_local_dir(str(_APP_DIR), f"{REMOTE_REPO}/pipeline", ignore=_IGNORE_PIPELINE)
    .add_local_dir(str(_REPO / "interp_tools"), f"{REMOTE_REPO}/interp_tools",
                   ignore=["**/__pycache__/**"])
    .add_local_dir(str(_REPO / "data_gen"), f"{REMOTE_REPO}/data_gen",
                   ignore=["**/__pycache__/**", "datasets/**"])
    .add_local_dir(str(_REPO / "metrics"), f"{REMOTE_REPO}/metrics",
                   ignore=["**/__pycache__/**"])
    .add_local_dir(str(_REPO / "github_repos"), f"{REMOTE_REPO}/github_repos",  # vendored ACDC + EAP-IG
                   ignore=["**/__pycache__/**", "**/.git/**", "**/*.ipynb", "**/*.zip"])
)

artifacts = modal.Volume.from_name("circuits-artifacts", create_if_missing=True)
OUT = "/outputs"

# per-model artifacts to ship back (repo-relative); missing candidates are skipped.
_ARTIFACT_RELS = (
    "pipeline/models/trained_models_pt/{name}.pt",
    "pipeline/models/trained_models_pt/{name}_training_log.json",
    "pipeline/circuits_data/{name}",
    "pipeline/circuits_figures/{name}",
    # analyses' input pools ship back too: the image's torch may be a different
    # build than the local one, so local regeneration would NOT be bit-identical
    "pipeline/models/gen_data/{name}",
)


@app.function(image=image, gpu="H100", volumes={OUT: artifacts},
              timeout=4 * 60 * 60, memory=32768, cpu=8.0)
def run_stages(name: str, stages: str, circuit: str = "", extra_args: str = "",
               restore_keys: str = "") -> dict:
    """Run ``run_model_pipeline.py --stages <stages>`` for one model on the H100,
    then tar its artifacts to the Volume for sync-back.

    PER-CIRCUIT FAN-OUT (``circuit`` set): the orchestrator splits a multi-circuit
    model's work into one train job (``circuit=""``) and one FULL-ANALYSIS job per
    circuit (data + patching∥edges∥acdc∥eap∥edge_pruning + smallest), each on its
    own H100, restoring the checkpoint from ``{name}.tar`` on the shared volume and
    shipping back only that circuit's subtree under ``{name}__{circuit}.tar``."""
    import subprocess
    import sys
    import time

    # Restore the model's checkpoint from a prior train job's tar so the train gate
    # can skip a retrain (per-circuit jobs REQUIRE it unless the image carries the
    # ckpt; whole-model jobs use it to make respawns/resumes near-free).
    artifacts.reload()                # see other containers' commit()s
    src = f"{OUT}/{name}.tar"
    image_ckpt = (f"{REMOTE_REPO}/pipeline/models/"
                  f"trained_models_pt/{name}.pt")
    if os.path.exists(src):
        with tarfile.open(src) as tf:
            tf.extractall(REMOTE_REPO, filter="data")   # restores trained_models_pt/{name}.pt
    elif circuit and not os.path.exists(image_ckpt):
        return {"name": name, "circuit": circuit, "rc": 90, "elapsed_s": 0.0,
                "err": f"no checkpoint: neither {src} on the volume nor the image ckpt"}
    # dependency tars (e.g. a contrast job's base-circuit smallest artifacts)
    for key in filter(None, restore_keys.split(",")):
        dep = f"{OUT}/{key}.tar"
        if not os.path.exists(dep):
            return {"name": name, "circuit": circuit, "rc": 91, "elapsed_s": 0.0,
                    "err": f"missing dependency tar {dep}"}
        with tarfile.open(dep) as tf:
            tf.extractall(REMOTE_REPO, filter="data")

    cmd = [sys.executable,
           f"{REMOTE_REPO}/pipeline/run_model_pipeline.py",
           "--model", name, "--stages", *stages.split(),
           "--device", "auto", "--local"]
    if circuit:
        cmd += ["--only-circuit", circuit]
    if extra_args:
        cmd += extra_args.split()
    import collections
    t0 = time.time()
    tail = collections.deque(maxlen=40)               # ring-buffer for a fail-loud err tail
    proc = subprocess.Popen(cmd, cwd=REMOTE_REPO, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in proc.stdout:
        sys.stdout.write(line)                        # live -> Modal logs
        tail.append(line)
    proc.wait()
    elapsed = round(time.time() - t0, 1)

    # tar back: whole-model artifacts for a train/single job; ONLY this circuit's
    # subtree for a per-circuit job (the shared ckpt isn't re-shipped).
    if circuit:
        key = f"{name}__{circuit}"
        rels = (f"pipeline/circuits_data/{name}/{circuit}",
                f"pipeline/circuits_figures/{name}/{circuit}",
                f"pipeline/models/gen_data/{name}/{circuit}")
    else:
        key = name
        rels = tuple(rel.format(name=name) for rel in _ARTIFACT_RELS)
    tar_path = f"{OUT}/{key}.tar"
    with tarfile.open(tar_path, "w") as tf:
        for rel in rels:
            p = os.path.join(REMOTE_REPO, rel)
            if os.path.exists(p):
                tf.add(p, arcname=rel)
    artifacts.commit()
    result = {"name": name, "circuit": circuit, "rc": proc.returncode, "elapsed_s": elapsed}
    if proc.returncode != 0:
        result["err"] = "".join(tail)[-1500:]
    return result


GRID_GPUS = 8   # multi-GPU container for stacked/grid training; the launcher shards over min(this, #shards)


@app.function(image=image, gpu=f"H100:{GRID_GPUS}", volumes={OUT: artifacts},
              timeout=8 * 60 * 60, memory=65536, cpu=16.0)
def run_sharded(args: str) -> dict:
    """Run train_sharded.py (model-sharding + size grid, per-size N*) on ONE multi-GPU
    container, then tar the grid checkpoints + architecture_grid.json to the volume.
    ``args`` is the train_sharded arg line (e.g. '--sizes all --batch 32')."""
    import subprocess
    import sys
    import time

    grid = f"{REMOTE_REPO}/pipeline/models/trained_models_pt_grid"
    cmd = [sys.executable, f"{REMOTE_REPO}/pipeline/models/train_models/train_sharded.py",
           "--local", *args.split(), "--out", grid]
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=REMOTE_REPO)
    with tarfile.open(f"{OUT}/grid.tar", "w") as tf:
        if os.path.exists(grid):
            tf.add(grid, arcname="pipeline/models/trained_models_pt_grid")
    artifacts.commit()
    return {"rc": proc.returncode, "elapsed_s": round(time.time() - t0, 1)}


@app.local_entrypoint()
def sharded(args: str = "--sizes small large wide deep xlarge") -> None:
    print(f"running stacked/grid training on {GRID_GPUS}xH100: [{args}] …")
    r = run_sharded.remote(args)
    if r["rc"] != 0:
        raise SystemExit(f"✗ remote training failed (rc {r['rc']})")
    data = b"".join(artifacts.read_file("grid.tar"))
    with tarfile.open(fileobj=io.BytesIO(data)) as tf:
        tf.extractall(_REPO, filter="data")
    print(f"  [ok] ({r['elapsed_s']}s) -> models/trained_models_pt_grid/ (+ architecture_grid.json)")


@app.local_entrypoint()
def main(models_file: str = "",
         stages: str = "train data patching edges eap smallest",
         only: str = "") -> None:
    models_file = models_file or str(_APP_DIR / "models" / "models.txt")
    names = [ln.split("#", 1)[0].strip() for ln in open(models_file)
             if ln.split("#", 1)[0].strip()]
    if only:
        wanted = {s.strip() for s in only.split(",") if s.strip()}
        names = [n for n in names if n in wanted]
    if not names:
        raise SystemExit("no model names selected (check models/models.txt / --only)")

    print(f"running stages '{stages}' for {len(names)} model(s) on H100: {names}")
    # return_exceptions: one crashed container must not abort sync-back for the rest.
    results = list(run_stages.map(names, kwargs={"stages": stages},
                                  return_exceptions=True))

    by_name = {r["name"]: r for r in results if isinstance(r, dict)}
    errors = [r for r in results if not isinstance(r, dict)]
    for e in errors:
        print(f"  [CONTAINER ERROR] {e}")
    failed = []
    for name in names:
        r = by_name.get(name)
        if r is None or r["rc"] != 0:
            failed.append(name)
            print(f"  [FAIL rc={r['rc'] if r else '?'}] {name}  — artifacts NOT synced")
            continue
        # extract only after a clean rc: a failed remote run must not overwrite
        # good local artifacts with garbage.
        data = b"".join(artifacts.read_file(f"{name}.tar"))
        with tarfile.open(fileobj=io.BytesIO(data)) as tf:
            tf.extractall(_REPO, filter="data")   # restores pipeline/... locally
        print(f"  [ok ] {name}  ({r['elapsed_s']}s)  artifacts -> local repo")
    print("\nFinish locally (fast stages):")
    print("  ./.venv/bin/python pipeline/run_model_pipeline.py "
          "--model <name> --stages smallest audit")
    if failed:
        raise SystemExit(f"✗ {len(failed)} model(s) failed remotely: {failed}")
