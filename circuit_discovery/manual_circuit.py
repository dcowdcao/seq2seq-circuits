#!/usr/bin/env python
"""
manual_circuit.py
=================================================================================
Run the manual circuit-analysis pipeline for segrev3 -- activation patching,
attribution, direct logit attribution, path patching (vis_patching_dla) plus
QK/OV, copy scores, head composition (vis_edge) -- in one of two modes:

  * DEFAULT (no --circuit): the end-to-end FULL-MODEL pipeline: generate data ->
    manual activation & weight analysis -> validate the full model per-head.
    NO circuit is synthesized: the analyses and the validation cover all
    4x8 = 32 heads.
  * CIRCUIT-SCOPED (--circuit / --circuit-json): run the SAME analyses scoped to
    a configurable input circuit -- e.g. one discovered by ACDC or EAP-IG -- for
    further analysis of the small surviving subgraph (see CIRCUIT-SCOPED MODE
    below).

The input MODEL is configurable too: ``--model <ckpt.pt>`` points every stage at
a different checkpoint (exported to child stages via the ``SEGREV3_CKPT`` env
var; default: the canonical segrev3 checkpoint).

STAGES, default full-model mode (each stage's output feeds the next):

  0. DATA       generate clean/corrupted contrast datasets as .json
                (data_gen.make_clean_corrupted) into the central store
                datasets/manual/<scheme>.json
  1. ANALYSIS   1a. activation analysis -- patching + direct logit attribution
                    (visualizations/manual_circuit_analysis/vis_patching_dla.py)
                1b. weight analysis -- QK/OV circuits, copy scores, composition,
                    head-edge maps (.../vis_edge.py)
                -> figs + figs_data/data/{<dataset>/patching_dla_results.json,
                   edge_results.json}
  2. VALIDATION FULL-MODEL per-head ablation (validation/validation.py, no args):
                minimality = logit-diff drop of ablating EACH of the 32 heads
                individually; faithfulness/completeness are trivial (circuit ==
                full model) and kept for format parity.
                -> validation/figs_data/manual/{data/rough_validation.json,
                   figs/rough_{minimality,summary}.png}

(The old SYNTHESIS stage -- interp_tools/circuit_summary.py selecting a rough
head subset -- is no longer part of the pipeline. The script remains runnable
standalone, and any existing visualizations/rough_circuit.json keeps working
as a --circuit input.)

FINAL OUTPUTS (full-model mode):
  datasets/manual/<scheme>.json + visualizations/manual_circuit_analysis/figs_data/...
  validation/figs_data/{data,figs}/manual/rough_{validation.json,minimality.png,summary.png}

Run:
  python circuit_discovery/manual_circuit.py                  # full pipeline
  python circuit_discovery/manual_circuit.py -n 256           # bigger datasets
  python circuit_discovery/manual_circuit.py --skip-data      # reuse existing datasets/manual/

CIRCUIT-SCOPED MODE (--circuit / --circuit-json): re-run the manual analyses
(activation patching, attribution, DLA, path patching; QK/OV, copy scores, head
composition; PLUS the faithfulness / completeness / minimality VALIDATION)
SCOPED to an already-discovered circuit, so everything focuses on the small
surviving subgraph instead of the full 4x8 model. The circuit input is
configurable two ways:

  * ``--circuit`` -- a discovered-circuit selector (canon.parse_circuit):
    ``all`` | a method (``manual``/``acdc``/``eap`` = every circuit of that
    method found under visualizations/) | a circuit folder name
    (``rough`` | ``acdc_thresh_<tau>`` | ``eap_top<n>``; the method is inferred).
  * ``--circuit-json`` -- a path to ANY circuit graph file (an ACDC
    ``summary.json``, an EAP graph json, or a manual ``rough_circuit.json``;
    the dialect is sniffed by interp_tools/circuit_loader.load_circuit_from_file),
    so a circuit produced by ACDC / EAP-IG -- or built by hand -- can be analysed
    directly, wherever the file lives.

The contrast datasets come from the central store (datasets/<method>/...,
indexed by the tau / top-n parsed from the circuit name; generated on demand).
VIZ runs per (circuit, scheme, --scope full_circuit|ablated); VALIDATION is a
property of the head-set, so it runs ONCE per circuit (skip with --no-validate;
mode via --validate-mode). Jobs are independent and parallelize across worker
processes (--jobs; default auto). Outputs:

  figures -> visualizations/manual_circuit_analysis/figs_data/figs/circuits/<method>/<name>/<scope>/<scheme>/
  data    -> visualizations/manual_circuit_analysis/figs_data/data/circuits/<method>/<name>/<scope>/<scheme>/
  validation -> validation/figs_data/{data,figs}/<method>/<name>/  (like the discovery outputs)

  python circuit_discovery/manual_circuit.py --circuit acdc_thresh_0.055
  python circuit_discovery/manual_circuit.py --circuit eap --scheme all     # every eap circuit x 3 schemes
  python circuit_discovery/manual_circuit.py --circuit eap_top200 --scope ablated --no-validate
  python circuit_discovery/manual_circuit.py --circuit-json visualizations/eap/eap_top200/eap_top200.json

SINGLE-SEQUENCE MODE (--single): analyse ONE custom sequence instead of the
batched datasets -- patching/DLA on a single clean/corrupted pair whose clean
side has EXACTLY --n-threes (k) 3s and whose corrupted side follows --scheme
(replacing exactly --n-replace positions; k and n are independent knobs), plus
the edge realized-attention on a single input with the same k. Literal digit
rows (--clean / --corrupted / --edge-input) override generation. The VALIDATION
stage is SKIPPED: it ablates heads over batched contrasts, which a single pair
cannot support. --single composes with --circuit / --circuit-json: the same
single pair / edge input is analysed scoped to the selected circuit(s).

  python circuit_discovery/manual_circuit.py --single --n-threes 3 --scheme replace_threes_to_nonthrees
  python circuit_discovery/manual_circuit.py --single --scheme replace_non_threes_to_threes \
      --n-threes 2 --n-replace 2 --seed 1
  python circuit_discovery/manual_circuit.py --single --scheme replace_threes_to_nonthrees \
      --clean 5 3 1 2 3 9 8 3 0 6 --corrupted 5 7 1 2 3 9 8 3 0 6
  python circuit_discovery/manual_circuit.py --single --n-threes 3 \
      --scheme replace_threes_to_nonthrees --circuit eap_top200      # scoped single pair

NOTE on imports: every stage imports the foundation (``model_loading`` etc.) from
``interp_tools``. We put ``interp_tools`` and the other package dirs on sys.path here
AND on the child processes' PYTHONPATH, so each stage resolves regardless of the cwd
it is launched from. (This is defensive: the individual modules already set up their
own paths correctly.)
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

_THIS = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(_THIS)
VIZ = os.path.join(REPO, "visualizations")
MCA = os.path.join(VIZ, "manual_circuit_analysis")
# Manual-analysis datasets live in the central store (datasets/manual/<scheme>.json),
# which is what vis_patching_dla / vis_edge read.
TEST_DATA = os.path.join(REPO, "datasets", "manual")
# Circuit-scoped outputs live in a circuits/ subtree of the manual-analysis
# figs_data (figs/circuits/<method>/<name>/<scope>/<scheme>/, same for data/).
SCOPED_OUT = os.path.join(MCA, "figs_data")

# Package dirs that must be importable in-process and by every child stage.
_LIBS = [
    "interp_tools",                                   # model_loading, circuit_summary, ...
    "data_gen", "metrics", "validation",
    os.path.join("visualizations", "manual_circuit_analysis"),
]
for _p in _LIBS:
    _abs = os.path.join(REPO, _p)
    if _abs not in sys.path:
        sys.path.insert(0, _abs)

import canon  # noqa: E402  (pure constants + shared CLI fragments; no torch)

# Canonical datasets (stored in datasets/manual/ as <scheme>.json). The NAMES are
# the scheme names: circuit_summary synthesizes the rough circuit from the "random"
# contrast (its CANON_DATASET), and vis_edge's weight analysis is dataset-independent.
# The two corruption schemes are also generated so the per-dataset activation/
# attention figures cover all head roles.
DATASETS = [
    # file/label stem, generation kwargs. The replace_* files carry a
    # ``batched_`` prefix (matching dataset_store._scheme_stem) to distinguish
    # them from the single-sequence datasets/single/single_<scheme>_... files.
    ("random",                                dict(scheme="random")),
    ("batched_replace_threes_to_nonthrees",   dict(scheme="replace_threes_to_nonthrees", n=1)),
    ("batched_replace_non_threes_to_threes",  dict(scheme="replace_non_threes_to_threes", n=1)),
]


def _child_env() -> dict:
    """os.environ with our package dirs prepended to PYTHONPATH for subprocesses."""
    env = dict(os.environ)
    extra = os.pathsep.join(os.path.join(REPO, p) for p in _LIBS)
    env["PYTHONPATH"] = extra + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return env


def _rel(p: str) -> str:
    return os.path.relpath(p, REPO) if p.startswith(REPO) else p


def _banner(title: str) -> None:
    print(f"\n{'=' * 74}\n▶ {title}\n{'=' * 74}", flush=True)


def _run(label: str, argv: list) -> None:
    """Run one pipeline stage as a subprocess; abort the pipeline if it fails."""
    _banner(label)
    print("  $ " + " ".join(_rel(a) for a in argv), flush=True)
    t0 = time.time()
    rc = subprocess.run(argv, cwd=REPO, env=_child_env()).returncode
    dt = time.time() - t0
    if rc != 0:
        raise SystemExit(f"✗ STAGE FAILED: {label}  (exit {rc}, {dt:.0f}s)")
    print(f"✓ {label}  ({dt:.0f}s)", flush=True)


def stage_data(n_examples: int, seed: int) -> None:
    """Stage 0 -- generate the contrast datasets as .json (in-process)."""
    from data_gen import make_clean_corrupted  # noqa: E402  (path set up above)

    _banner(f"0. DATA GENERATION  ->  {_rel(TEST_DATA)}/")
    os.makedirs(TEST_DATA, exist_ok=True)
    for i, (name, kw) in enumerate(DATASETS):
        path = os.path.join(TEST_DATA, f"{name}.json")
        make_clean_corrupted(n_examples, seed=seed + i, save_path=path, **kw)
        extra = f", n_replace={kw['n']}" if "n" in kw else ""
        print(f"  ✓ {name}.json  (scheme={kw['scheme']}{extra}, examples={n_examples})", flush=True)


def stage_single_data(args) -> tuple:
    """Stage 0 (single mode) -- generate the ONE custom clean/corrupted pair and
    the ONE edge input into datasets/single/ via the central dataset store
    (in-process, like stage_data). Returns (pair_path, edge_path)."""
    import dataset_store as store  # noqa: E402  (path set up above)

    _banner(f"0. SINGLE-SEQUENCE DATA  ->  {_rel(store.SINGLE_DIR)}/")
    try:
        pair_path = store.generate_single_pair(
            scheme=args.scheme, n_threes=args.n_threes, replace_n=args.n_replace,
            seed=args.seed, clean_inputs=args.clean, corrupted_inputs=args.corrupted)
        # The edge input shares the pair's 3-count k unless given literally.
        k = store._derive_n_threes(args.n_threes, args.clean)
        edge_path = store.generate_single_edge(
            n_threes=None if args.edge_input is not None else k,
            seed=args.seed, inputs=args.edge_input)
    except ValueError as e:
        raise SystemExit(f"✗ single-sequence data: {e}")
    for p in (pair_path, edge_path):
        print(f"  ✓ {_rel(p)}", flush=True)
    return pair_path, edge_path


def run_single(args) -> None:
    """Single-sequence pipeline: ONE custom pair through patching/DLA + ONE custom
    input through the edge analysis (synthesis + validation are skipped -- they
    aggregate batched contrasts, which a single pair cannot support)."""
    py = sys.executable
    t_all = time.time()
    print("=" * 74)
    print("  MANUAL CIRCUIT-DISCOVERY PIPELINE — segrev3  [SINGLE-SEQUENCE MODE]")
    print("  single pair → activation analysis;  single input → weight analysis")
    print("=" * 74)

    pair_path, edge_path = stage_single_data(args)
    pair_label = os.path.splitext(os.path.basename(pair_path))[0]
    edge_label = os.path.splitext(os.path.basename(edge_path))[0]

    _run(f"1a. ACTIVATION analysis — patching + DLA  (single pair: {pair_label})",
         [py, os.path.join(MCA, "vis_patching_dla.py"), "--data", pair_path])
    _run(f"1b. WEIGHT analysis — QK/OV, copy scores, composition, head-edges  "
         f"(single input: {edge_label})",
         [py, os.path.join(MCA, "vis_edge.py"), "--data", edge_path])

    _banner("2. VALIDATION — skipped (single-sequence mode)")
    print("  (validation ablates heads over BATCHED contrasts -- a single pair\n"
          "   cannot support it. To scope this pair to a discovered circuit,\n"
          "   re-run with --single --circuit <name>.)", flush=True)

    # ---- expected outputs check ----
    expected_outputs = [
        os.path.join(MCA, "figs_data", "figs", "vis_patching_dla", pair_label),
        os.path.join(MCA, "figs_data", "data", pair_label, "patching_dla_results.json"),
        os.path.join(MCA, "figs_data", "figs", "edge", edge_label, "specific_heads"),
        os.path.join(MCA, "figs_data", "data", "edge_results.json"),
    ]
    missing = [d for d in expected_outputs if not os.path.exists(d)]
    _banner(f"SINGLE-SEQUENCE PIPELINE COMPLETE — {time.time() - t_all:.0f}s total")
    print("Expected outputs:")
    for d in expected_outputs:
        print(f"  [{'MISSING' if d in missing else 'ok'}] {_rel(d)}")
    if missing:
        raise SystemExit(f"✗ {len(missing)} expected output(s) missing — "
                         "check the stage logs above.")
    print("\n✓ all expected outputs present.")


# --------------------------------------------------------------------------- #
# CIRCUIT-SCOPED MODE (--circuit / --circuit-json)                            #
# --------------------------------------------------------------------------- #
# Run the manual analyses scoped to a configurable input circuit (e.g. one
# discovered by ACDC / EAP-IG). Pure glue over the verified building blocks:
#   interp_tools/circuit_loader  -- the discovered subgraph, uniform across methods
#   data_gen/dataset_store       -- the clean/corrupted contrast datasets
#   vis_patching_dla.make_figs   -- activation/attribution/DLA/path figures (scoped)
#   vis_edge.main                -- QK/OV, copy scores, composition (scoped)
#   validation.main              -- faithfulness / completeness / minimality
# The heavy modules (torch et al.) are imported lazily inside the run functions
# so the thin full-pipeline orchestrator path stays import-light.

def discover_names(method: str) -> list:
    """Enumerate the discovered circuit folder names for ``method``."""
    import glob
    if method == "acdc":
        dirs = sorted(glob.glob(os.path.join(VIZ, "acdc", "acdc_thresh_*")))
        return [os.path.basename(d) for d in dirs if os.path.isdir(d)]
    if method == "eap":
        dirs = sorted(glob.glob(os.path.join(VIZ, "eap", "eap_top*")))
        return [os.path.basename(d) for d in dirs if os.path.isdir(d)]
    if method == "manual":
        return ["rough"]
    raise ValueError(f"unknown method {method!r}")


def dataset_source(name: str) -> tuple:
    """(store app, index kwargs) for the contrast datasets behind circuit ``name``.

    The store is indexed by the tau / top-n parsed from the circuit name
    (``acdc_thresh_<tau>`` / ``eap_top<n>``); any other name -- e.g. an arbitrary
    --circuit-json stem -- falls back to the manual store (the files only differ
    by their provenance folder, so this is a safe default)."""
    import re
    m = re.search(r"acdc_thresh_([0-9.]+)", name)
    if m:
        return "acdc", dict(tau=float(m.group(1)))
    m = re.search(r"eap_top(\d+)", name)
    if m:
        return "eap", dict(topn=int(m.group(1)))
    return "manual", dict()


def _load_scoped_circuit(method: str, name: str, circuit_json):
    """Load the scoped circuit: from a graph file when ``circuit_json`` is given
    (dialect sniffed), else the discovered circuit ``(method, name)``."""
    import circuit_loader as CL
    if circuit_json:
        return CL.load_circuit_from_file(circuit_json)
    return CL.load_circuit(method, name)


def run_scoped_viz(method, name, scheme, scope, model, circuit) -> bool:
    """Run the scoped patching/DLA + edge analyses for one (circuit, scheme, scope).

    Returns True on success; on any failure prints a warning and returns False
    so the caller can continue with the next job.
    """
    import traceback
    import dataset_store as DS
    import vis_patching_dla as VP
    import vis_edge as VE
    try:
        app, idx = dataset_source(name)
        pairs, path = DS.load(app, scheme=scheme, **idx)

        fig_dir = os.path.join(SCOPED_OUT, "figs", "circuits", method, name, scope, scheme)
        data_dir = os.path.join(SCOPED_OUT, "data", "circuits", method, name, scope, scheme)
        head_dir = os.path.join(fig_dir, "specific_heads")
        for d in (fig_dir, data_dir, head_dir):
            os.makedirs(d, exist_ok=True)

        # 1. activation patching / attribution / DLA / path patching (scoped).
        VP.make_figs(pairs, label=scheme, model=model, data_path=path,
                     circuit=circuit, scope=scope,
                     fig_dir=fig_dir, data_dir=data_dir)
        # 2. QK/OV, copy scores, composition, per-head edges (scoped).
        VE.main(data_paths=[path], circuit=circuit, scope=scope,
                fig_dir=fig_dir, head_dir=head_dir, data_dir=data_dir, model=model)

        print(f"✓ viz  {method}/{name} [{scheme},{scope}] -> {_rel(fig_dir)}", flush=True)
        return True
    except Exception as e:
        print(f"⚠ FAILED viz {method}/{name} [{scheme},{scope}]: {e}", flush=True)
        traceback.print_exc()
        return False


def run_scoped_viz_single(method, name, scope, model, circuit, args) -> bool:
    """Single-sequence variant of ``run_scoped_viz``: the patching/DLA figures
    come from the ONE custom pair and the edge realized-attention from the ONE
    custom edge input (datasets/single/, the same files -- or the same
    parameters/literals -- as the full-model --single run), instead of the
    batched per-scheme store. The leaf output folder is the pair's LABEL (e.g.
    ``single_replace_threes_to_nonthrees_k3_n1``) so single-mode outputs never
    clobber the batched per-scheme ones.
    """
    import traceback
    import dataset_store as DS
    import vis_patching_dla as VP
    import vis_edge as VE
    try:
        pairs, pair_path = DS.load_single_pair(
            scheme=args.scheme, n_threes=args.n_threes, replace_n=args.n_replace,
            seed=args.seed, clean_inputs=args.clean, corrupted_inputs=args.corrupted)
        # The edge input shares the pair's 3-count k unless given literally.
        k = DS._derive_n_threes(args.n_threes, args.clean)
        _, edge_path = DS.load_single_edge(
            n_threes=None if args.edge_input is not None else k,
            seed=args.seed, inputs=args.edge_input)
        label = os.path.splitext(os.path.basename(pair_path))[0]

        fig_dir = os.path.join(SCOPED_OUT, "figs", "circuits", method, name, scope, label)
        data_dir = os.path.join(SCOPED_OUT, "data", "circuits", method, name, scope, label)
        head_dir = os.path.join(fig_dir, "specific_heads")
        for d in (fig_dir, data_dir, head_dir):
            os.makedirs(d, exist_ok=True)

        # 1. activation patching / attribution / DLA / path patching (scoped).
        VP.make_figs(pairs, label=label, model=model, data_path=pair_path,
                     circuit=circuit, scope=scope,
                     fig_dir=fig_dir, data_dir=data_dir)
        # 2. QK/OV, copy scores, composition, per-head edges (scoped) -- realized
        #    attention on the SINGLE edge input.
        VE.main(data_paths=[edge_path], circuit=circuit, scope=scope,
                fig_dir=fig_dir, head_dir=head_dir, data_dir=data_dir, model=model)

        print(f"✓ viz  {method}/{name} [single:{label},{scope}] -> {_rel(fig_dir)}",
              flush=True)
        return True
    except Exception as e:
        print(f"⚠ FAILED viz {method}/{name} [single,{scope}]: {e}", flush=True)
        traceback.print_exc()
        return False


def run_scoped_validation(name, circuit, mode, model=None, mean_cache=None) -> bool:
    """Validate the scoped head-set (faithfulness / completeness / minimality).

    Reuses ``validation.main``, which groups its outputs under
    ``validation/figs_data/{data,figs}/<acdc|eap|manual>/<name>/`` via the circuit
    name -- exactly how the discovery pipelines file their own validation.
    Validation is a property of the head-set, so this runs ONCE per circuit (not
    per scheme). A shared ``model`` / ``mean_cache`` avoids reloading per circuit.
    """
    import traceback
    import circuit_loader as CL
    import validation as VAL
    try:
        VAL.main(circuit_labels=CL.head_labels(circuit), name=name, mode=mode,
                 model=model, mean_cache=mean_cache)
        print(f"✓ validate  {name} (mode={mode}) -> validation/figs_data/.../{name}/", flush=True)
        return True
    except Exception as e:
        print(f"⚠ FAILED validate {name}: {e}", flush=True)
        traceback.print_exc()
        return False


# --------------------------------------------------------------------------- #
# Scoped-mode parallel workers                                                #
# --------------------------------------------------------------------------- #
# Each scoped-viz job (circuit x scheme) and each per-circuit validation job is
# INDEPENDENT (writes its own folder) and dominated by CPU-bound matplotlib
# rendering, so they parallelize near-linearly across the box's cores. We run
# them in a process pool whose workers load the model ONCE (initializer) and
# reuse it for every task.
_WORKER: dict = {}


def _init_worker(threads: int, val_mode: str) -> None:
    import torch
    import validation as VAL
    from model_loading import load_segrev3
    try:
        torch.set_num_threads(max(1, threads))
    except Exception:
        pass
    _WORKER["model"] = load_segrev3()
    # Validation's mean-ablation cache depends only on (model, mode) -> compute once.
    _WORKER["mean_cache"] = VAL._mean_z(_WORKER["model"]) if val_mode == "mean" else None


def _viz_worker(job):
    method, name, scheme, scope, circuit_json = job
    try:
        circuit = _load_scoped_circuit(method, name, circuit_json)
    except Exception as e:
        print(f"⚠ skip viz {method}/{name}: {e}", flush=True)
        return ("viz", False)
    return ("viz", run_scoped_viz(method, name, scheme, scope, _WORKER["model"], circuit))


def _val_worker(job):
    method, name, mode, circuit_json = job
    try:
        circuit = _load_scoped_circuit(method, name, circuit_json)
    except Exception as e:
        print(f"⚠ skip validate {method}/{name}: {e}", flush=True)
        return ("val", False)
    return ("val", run_scoped_validation(name, circuit, mode, model=_WORKER["model"],
                                         mean_cache=_WORKER["mean_cache"]))


def _run_scoped_jobs_parallel(viz_jobs, val_jobs, jobs, val_mode):
    """Run the viz + validation jobs across ``jobs`` worker processes; return
    (n_viz_ok, n_viz, n_val_ok, n_val)."""
    import concurrent.futures
    import multiprocessing as mp
    threads = max(1, (os.cpu_count() or 2) // max(1, jobs))
    # Prevent BLAS/OpenMP oversubscription: matplotlib/NumPy (OpenBLAS) and torch
    # would otherwise each spin up ~cpu_count threads in EVERY worker. Cap them so
    # jobs*threads ~ cpu_count. Set in the PARENT so spawned children inherit it at
    # numpy/torch import (torch's own setup honours OMP_NUM_THREADS).
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[v] = str(threads)
    ctx = mp.get_context("spawn")          # required for CUDA in subprocesses
    # Pre-flight: each worker loads the model on the GPU (capped per-process in
    # load_segrev3); refuse to fan out when host memory is already short.
    from model_loading import assert_memory_headroom
    assert_memory_headroom(context=f"manual_circuit scoped pool ({jobs} workers)")
    n_viz_ok = n_val_ok = 0
    with concurrent.futures.ProcessPoolExecutor(
            max_workers=jobs, mp_context=ctx,
            initializer=_init_worker, initargs=(threads, val_mode)) as ex:
        futs = [ex.submit(_val_worker, j) for j in val_jobs]
        futs += [ex.submit(_viz_worker, j) for j in viz_jobs]
        for fut in concurrent.futures.as_completed(futs):
            kind, ok = fut.result()
            if kind == "viz":
                n_viz_ok += int(ok)
            else:
                n_val_ok += int(ok)
    return n_viz_ok, len(viz_jobs), n_val_ok, len(val_jobs)


def resolve_scoped_targets(args) -> list:
    """Resolve --circuit / --circuit-json into [(method, name, circuit_json), ...]."""
    if args.circuit_json:
        path = os.path.abspath(args.circuit_json)
        circuit = _load_scoped_circuit(None, None, path)   # raises if unreadable
        return [(circuit.app, circuit.name, path)]
    method, name = canon.parse_circuit(args.circuit)
    methods = list(canon.METHODS) if method == "all" else [method]
    targets = []
    for m in methods:
        names = discover_names(m) if name == "all" else [name]
        targets.extend((m, n, None) for n in names)
    return targets


def run_circuit_scoped(args) -> None:
    """Driver for the circuit-scoped mode (batched and --single)."""
    from model_loading import load_segrev3

    try:
        targets = resolve_scoped_targets(args)
    except (ValueError, FileNotFoundError) as e:
        raise SystemExit(f"✗ {e}")
    if not targets:
        raise SystemExit(f"✗ no discovered circuits found for --circuit {args.circuit!r}.")
    tnames = [f"{m}/{n}" for (m, n, _) in targets]

    if args.single:
        print("=" * 74)
        print("  CIRCUIT-SCOPED MANUAL ANALYSIS — segrev3  [SINGLE-SEQUENCE]")
        print(f"  circuits={tnames}  scheme={args.scheme}  scope={args.scope}  "
              f"n_threes={args.n_threes}  n_replace={args.n_replace}")
        if args.validate:
            print("  validation: SKIPPED (batched-only stage; nothing to validate "
                  "on a single pair)")
        print("=" * 74, flush=True)

        print("Loading model ...", flush=True)
        model = load_segrev3()
        n_ok = n_run = 0
        for method, name, cj in targets:
            try:
                circuit = _load_scoped_circuit(method, name, cj)
            except Exception as e:
                print(f"⚠ skip {method}/{name}: {e}", flush=True)
                continue
            n_run += 1
            n_ok += run_scoped_viz_single(method, name, args.scope, model, circuit, args)
        print("=" * 74)
        print(f"DONE — viz {n_ok}/{n_run} single-sequence jobs.")
        print(f"  figures  -> {_rel(SCOPED_OUT)}/", flush=True)
        if n_ok == 0:
            raise SystemExit("✗ no jobs succeeded — see warnings above.")
        return

    schemes = list(canon.SCHEMES) if args.scheme == "all" else [args.scheme]

    # Build the independent job lists: one viz job per (circuit, scheme); one
    # validation job per circuit (head-set level, independent of scheme/scope).
    viz_jobs = [(m, n, scheme, args.scope, cj)
                for (m, n, cj) in targets for scheme in schemes]
    val_jobs = [(m, n, args.validate_mode, cj) for (m, n, cj) in targets] \
        if args.validate else []
    n_jobs_total = len(viz_jobs) + len(val_jobs)

    if args.jobs is not None:
        jobs = max(1, min(args.jobs, n_jobs_total))
    else:
        jobs = max(1, min(n_jobs_total, (os.cpu_count() or 2) // 2, 12))

    print("=" * 74)
    print("  CIRCUIT-SCOPED MANUAL ANALYSIS — segrev3")
    print(f"  circuits={tnames}  schemes={schemes}  scope={args.scope}  "
          f"validate={args.validate} (mode={args.validate_mode})")
    print(f"  jobs={jobs}  ({len(viz_jobs)} viz + {len(val_jobs)} validation)")
    print("=" * 74, flush=True)

    if jobs > 1:
        n_viz_ok, n_viz, n_val_ok, n_val = _run_scoped_jobs_parallel(
            viz_jobs, val_jobs, jobs, args.validate_mode)
    else:
        # Sequential path: still load the model + caches ONCE and reuse.
        import validation as VAL
        from tqdm import tqdm
        print("Loading model ...", flush=True)
        model = load_segrev3()
        mean_cache = VAL._mean_z(model) if args.validate_mode == "mean" else None
        n_viz_ok = n_viz = n_val_ok = n_val = 0
        for (method, name, mode, cj) in val_jobs:
            try:
                circuit = _load_scoped_circuit(method, name, cj)
            except Exception as e:
                print(f"⚠ skip validate {method}/{name}: {e}", flush=True)
                continue
            n_val += 1
            n_val_ok += run_scoped_validation(name, circuit, mode, model=model,
                                              mean_cache=mean_cache)
        for (method, name, scheme, scope, cj) in tqdm(
                viz_jobs, desc="circuit-scoped analysis", unit="job"):
            try:
                circuit = _load_scoped_circuit(method, name, cj)
            except Exception as e:
                print(f"⚠ skip {method}/{name}: {e}", flush=True)
                continue
            n_viz += 1
            n_viz_ok += run_scoped_viz(method, name, scheme, scope, model, circuit)

    print("=" * 74)
    print(f"DONE — viz {n_viz_ok}/{n_viz} jobs, validation {n_val_ok}/{n_val} circuits.")
    print(f"  figures  -> {_rel(SCOPED_OUT)}/")
    if args.validate:
        print(f"  validation -> {_rel(os.path.join(REPO, 'validation', 'figs_data'))}/"
              "{data,figs}/<method>/<name>/", flush=True)
    if n_viz_ok == 0 and n_val_ok == 0:
        raise SystemExit("✗ no jobs succeeded — see warnings above.")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run the manual circuit-analysis pipeline for segrev3: the "
                    "end-to-end full-model pipeline by default, or -- with "
                    "--circuit / --circuit-json -- the same analyses SCOPED to a "
                    "configurable input circuit (e.g. one discovered by ACDC or "
                    "EAP-IG).")
    ap.add_argument("-n", "--examples", dest="n_examples", type=int,
                    default=canon.CANON_MANUAL_EXAMPLES,
                    help="examples per generated dataset (default %(default)s)")
    ap.add_argument("--seed", type=int, default=0, help="base RNG seed (default 0)")
    ap.add_argument("--skip-data", action="store_true",
                    help="reuse existing test_data/*.json instead of regenerating")
    ap.add_argument("--model", default=None, metavar="CKPT",
                    help="model checkpoint path for every stage (default: the "
                         "canonical segrev3 checkpoint; exported to child stages "
                         "via SEGREV3_CKPT)")
    ap.add_argument("--scheme", default="random",
                    choices=list(canon.SCHEMES) + ["all"],
                    help="contrast scheme: how the corrupted side is built from "
                         "the clean side; 'all' = the 3 schemes (circuit-scoped "
                         "batched mode only). Default: random")

    c = ap.add_argument_group(
        "circuit-scoped mode (run the analyses scoped to an input circuit)")
    c.add_argument("--circuit", default=None,
                   help="discovered-circuit selector: 'all' | a method "
                        "(manual|acdc|eap = every circuit of that method) | a "
                        "circuit folder name (rough | acdc_thresh_<tau> | "
                        "eap_top<n>; the method is inferred)")
    c.add_argument("--circuit-json", default=None, dest="circuit_json", metavar="PATH",
                   help="path to a circuit graph file to scope to (an ACDC "
                        "summary.json, an EAP graph json, or a manual "
                        "rough_circuit.json; the dialect is sniffed) -- an "
                        "alternative to --circuit for graphs anywhere on disk")
    c.add_argument("--scope", default="full_circuit", choices=["full_circuit", "ablated"],
                   help="full_circuit (mask figures) or ablated (mean-ablate "
                        "complement). Default: full_circuit")
    c.add_argument("--no-validate", dest="validate", action="store_false",
                   help="skip the faithfulness/completeness/minimality validation")
    c.add_argument("--validate-mode", default="mean", choices=["mean", "zero", "resample"],
                   help="ablation mode for validation (default: mean)")
    c.add_argument("--jobs", type=int, default=None,
                   help="number of (circuit x scheme) viz + validation jobs to run "
                        "CONCURRENTLY across processes (default: auto, "
                        "min(#jobs, max(1, cpu//2), 12)). Use 1 to force sequential.")

    # The single-sequence flag set is shared (canon.py) with main.py so the two
    # CLIs cannot drift. In single mode the batched validation is skipped.
    canon.add_single_args(ap, switch=True)
    args = ap.parse_args()

    if args.circuit and args.circuit_json:
        ap.error("--circuit and --circuit-json are mutually exclusive")
    if args.model:
        if not os.path.exists(args.model):
            ap.error(f"--model checkpoint not found: {args.model}")
        os.environ["SEGREV3_CKPT"] = os.path.abspath(args.model)
    scoped = args.circuit is not None or args.circuit_json is not None

    if args.single:
        canon.check_single_args(args, ap.error)
        if scoped:
            run_circuit_scoped(args)
        else:
            run_single(args)
        return

    if scoped:
        run_circuit_scoped(args)
        return

    py = sys.executable
    t_all = time.time()
    print("=" * 74)
    print("  MANUAL CIRCUIT-ANALYSIS PIPELINE — segrev3")
    print("  data → activation+weight analysis → full-model per-head validation")
    print("=" * 74)

    # 0. DATA
    if args.skip_data:
        _banner("0. DATA GENERATION  (skipped — reusing existing test_data/*.json)")
        have = os.path.isdir(TEST_DATA) and any(f.endswith(".json") for f in os.listdir(TEST_DATA))
        if not have:
            raise SystemExit(f"✗ --skip-data but no datasets found in {_rel(TEST_DATA)}/")
    else:
        stage_data(args.n_examples, args.seed)

    # 1. ANALYSIS — activation (patching + DLA) then weight (QK/OV/composition/edges)
    _run("1a. ACTIVATION analysis — patching + DLA",
         [py, os.path.join(MCA, "vis_patching_dla.py")])
    _run("1b. WEIGHT analysis — QK/OV, copy scores, composition, head-edges",
         [py, os.path.join(MCA, "vis_edge.py")])

    # 2. VALIDATION — FULL-MODEL head ablation (no circuit is synthesized: the
    # no-args validation covers all 32 heads, so rough_minimality.png reports
    # the logit-diff drop of ablating each head individually).
    _run("2. VALIDATION — full-model per-head ablation (all 32 heads)",
         [py, os.path.join(REPO, "validation", "validation.py")])

    # ---- expected outputs check ----
    expected_outputs = [
        os.path.join(REPO, "validation", "figs_data", "data", "manual", "rough_validation.json"),
        os.path.join(REPO, "validation", "figs_data", "figs", "manual", "rough_minimality.png"),
        os.path.join(REPO, "validation", "figs_data", "figs", "manual", "rough_summary.png"),
    ]
    missing = [d for d in expected_outputs if not os.path.exists(d)]
    _banner(f"PIPELINE COMPLETE — {time.time() - t_all:.0f}s total")
    print("Expected outputs:")
    for d in expected_outputs:
        print(f"  [{'MISSING' if d in missing else 'ok'}] {_rel(d)}")
    print(f"  [dir] {_rel(os.path.join(MCA, 'figs_data'))}/  (per-dataset figs + results JSON)")
    if missing:
        raise SystemExit(f"✗ {len(missing)} expected output(s) missing — check the stage logs above.")
    print("\n✓ all expected outputs present.")


if __name__ == "__main__":
    main()
