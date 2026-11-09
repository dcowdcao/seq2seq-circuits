#!/usr/bin/env python
"""One-command orchestrator for pipeline: per-model, per-circuit stage runner.

Runs the pipeline as SUBPROCESSES of the per-stage scripts (per-stage timing, env
propagation, BLAS thread caps, memory-headroom checks between stages), with a wave
schedule that never exceeds GPU_PROC_CAP (4) concurrent GPU children:

    data -> train(skip-if-ckpt-passes-gate)
         -> 2a: patching ∥ edges ∥ eap (∥ acdc when named; up to 4 GPU slots)
         -> 2b: edge_pruning (when named, ALONE) -> 2c: smallest (ALONE) -> audit

STAGES (--stages, one or more; "all" = the DEFAULT pipeline — EAP-IG is the only
default finder (canon.DEFAULT_FINDERS); acdc/edge_pruning run only when named)
    data      models/gen_data/gen_circuit_data.py (one inputs.json pool per circuit)
    train     models/train_models/run_training_job.sh (arch-aware escalation ladder;
              per-arch recipe from train_models/hparams.py; promotion gated by
              train_models/promote.py; SKIPPED when trained_models_pt/<name>.pt passes
              the seq-acc gate; --smoke trains a THROWAWAY tiny ckpt under .smoke/)
    patching  interp_tools/patching.py  --analysis all      } WAVE 2a
    edges     interp_tools/edges.py     --analysis all      } (parallel,
    acdc      find_circuits/acdc.py     --taus <canon>      }  up to 4
    eap       find_circuits/eap.py --topn-regime <canon>    }  GPU slots)
              (a REGIME, not rungs: eap scales the ladder to the model's live edge budget)
    edge_pruning  find_circuits/edge_pruning.py --sparsities <canon>  WAVE 2b
              (ALONE; L0 edge-mask training, the dominant per-model cost)
    smallest  find_circuits/smallest_faithful_circuit.py    } WAVE 2c (ALONE)
              EVERY row adds --search (it contributes a high-n eap candidate; the mode
              still decides selection); suppression rows ALWAYS run and get
              --router-verdict from task_router/<model>/diagnostic.json's (cued,wrong)
              pair — recorded on the output, never a gate (the ROUTER step runs after
              the task circuits' smallest)
    audit     in-process per-circuit audit: candidate dirs (ENUMERATED, since frac-regime
              rung names are per-model) + smallest_circuit + trees + figures mirror,
              PLUS the model's checkpoint gate and each circuit's empty-circuit baseline;
              nonzero exit on any gap

Every analysis stage runs ONCE PER CIRCUIT (gen_circuit_data.circuits_for(name));
data + train run once per model. SEGREV3_DEVICE/SEGREV3_CKPT are exported to every
child (names historical — they are the global rails). --dry-run prints the exact
commands and executes nothing (no torch import — instant).

EXECUTION MODEL (default = LOCAL): every stage runs on this machine (--device
picks cpu/cuda; --gpus N fans the job DAG across N local GPUs). --local is accepted
as a no-op alias. Optional --modal backend: every GPU stage runs on Modal GPU
containers — one container per single-circuit model (train → data → parallel
analysis → smallest); multi-circuit models train once (R1) then fan one container
per circuit (R2). Failed containers get one automatic respawn; the batch never
aborts on one job. Local work under --modal: spawn/poll/sync + audit (--smoke
forces local). Deploy once: modal deploy pipeline/modal_pipeline_app.py

EXAMPLES
    run_model_pipeline.py --model rcnt3_id --stages all --dry-run
    run_model_pipeline.py --model rcnt3_id --stages all --smoke
    run_model_pipeline.py --model areset13_id reccount_id rcnt3_id \
        --stages smallest audit --smallest-target 0.85
    run_model_pipeline.py --model all --stages audit
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time

# --------------------------------------------------------------------------- #
# Layout (this file lives at <repo>/pipeline/run_model_pipeline.py).      #
# NO torch / model_registry import at module level — --dry-run must be instant;   #
# model_registry (which pulls torch transitively) is imported lazily via _glue(). #
# --------------------------------------------------------------------------- #
PIPELINE = os.path.dirname(os.path.abspath(__file__))   # .../pipeline
REPO = os.path.dirname(PIPELINE)                        # .../circuits
MODELS_DIR = os.path.join(PIPELINE, "models")
TRAINED_MODELS_PT = os.path.join(MODELS_DIR, "trained_models_pt")
SMOKE_DIR = os.path.join(PIPELINE, ".smoke")            # throwaway smoke ckpts

# canon.py and hparams.py are PURE STDLIB at import time — safe at module level (keeps
# --dry-run torch-free; hparams touches model_loading only inside for_tier()).
sys.path.insert(0, os.path.join(REPO, "data_gen"))
sys.path.insert(0, os.path.join(PIPELINE, "models", "train_models"))
import canon  # noqa: E402
import hparams  # noqa: E402

# Canonical stage order (== the wave order). "all" expands to DEFAULT_STAGES.
STAGE_ORDER = ("data", "train", "probes", "patching", "edges", "acdc", "eap",
               "edge_pruning", "smallest", "audit")
# EAP-IG is the default finder; the others (+ probes, patching, edges) run only when named.
NON_DEFAULT_FINDERS = tuple(f for f in ("acdc", "eap", "edge_pruning")
                            if f not in canon.DEFAULT_FINDERS)
DEFAULT_STAGES = tuple(s for s in STAGE_ORDER
                       if s not in NON_DEFAULT_FINDERS and s not in ("probes", "patching", "edges"))

#: train gate: an existing checkpoint must reach test seq-acc >= this to skip retrain.
SEQ_ACC_GATE = 0.95
#: smallest_faithful_circuit floor: the circuit's ABSOLUTE per-sequence accuracy
#: under edge-level mean ablation (keep in sync with
#: smallest_faithful_circuit.DEFAULT_TARGET). Override --smallest-target.
SMALLEST_TARGET = "0.85"
SMALLEST_JOBS = 4               # smallest-search eval children
#: per-stage dataset sizes.
PATCHING_N = 128
EDGES_N = 64
ACDC_EXAMPLES = canon.CANON_ACDC_EXAMPLES        # 128

# Package dirs every child must be able to import (incl. the pipeline's models/
# dir so children resolve `import model_registry` regardless of their bootstrap).
_LIBS = [
    os.path.join(REPO, "interp_tools"),
    os.path.join(REPO, "data_gen"),
    os.path.join(REPO, "metrics"),
    MODELS_DIR,
]

_GLUE = None  # lazy model_registry module handle


# --------------------------------------------------------------------------- #
# Conventional paths (printable WITHOUT importing model_registry/torch, so        #
# --dry-run is instant). At EXECUTION time `_verify_conventions` asserts they  #
# match model_registry.get_spec() exactly — hard failure on any drift.            #
# --------------------------------------------------------------------------- #
def conv_ckpt(name: str) -> str:
    # Mirrors model_registry._resolve_ckpt EXACTLY (trained_models_pt/<name>.pt) —
    # _verify_conventions hard-fails on drift. Pure os.path (no torch / model_registry
    # import) so --dry-run stays instant.
    return os.path.join(TRAINED_MODELS_PT, f"{name}.pt")


def list_models() -> list:
    """The models/models.txt run-list, TORCH-FREE (mirrors model_registry.all_models
    without importing it, so --dry-run stays instant). Any valid DSL algorithm_id is
    runnable even if absent here — the run-list is documentation, not a gate."""
    names = []
    try:
        with open(os.path.join(MODELS_DIR, "models.txt")) as f:
            for line in f:
                nm = line.split("#", 1)[0].strip()
                if nm and nm not in names:
                    names.append(nm)
    except OSError:
        pass
    return names


def _glue():
    """Lazy import of model_registry (pulls torch transitively — execution/audit only)."""
    global _GLUE
    if _GLUE is None:
        if MODELS_DIR not in sys.path:
            sys.path.insert(0, MODELS_DIR)
        import model_registry  # noqa: E402
        _GLUE = model_registry
    return _GLUE


def _verify_conventions(name: str) -> None:
    """Execution-time guard: the torch-free conventional paths used for planning must equal
    the model_registry catalog (the single source of truth), and the model's task must emit
    in-vocab outputs over a sampled sweep — an out-of-vocab output token is an instant
    IndexError at training (how altrun_id shipped emitting token 10 into d_vocab=10)."""
    glue = _glue()
    spec = glue.get_spec(name)      # also validates the name (KeyError on a bad DSL id)
    if os.path.abspath(conv_ckpt(name)) != os.path.abspath(spec.checkpoint):
        raise SystemExit(f"✗ path drift: run_model_pipeline ckpt {conv_ckpt(name)} != "
                         f"model_registry {spec.checkpoint}")
    glue.validate_spec(spec)


def _headroom(context: str, verbose: bool = False) -> None:
    """model_registry.assert_memory_headroom() between stages. The safety
    check ALWAYS runs; the terse headroom line prints only under --verbose."""
    avail = _glue().assert_memory_headroom(context=context)
    if verbose:
        print(f"    [headroom] {avail:.1f} GiB free", flush=True)


# --------------------------------------------------------------------------- #
# Subprocess plumbing (adapted from circuit_discovery/main.py).                #
# --------------------------------------------------------------------------- #
def _child_env(extra: dict | None = None) -> dict:
    env = dict(os.environ)
    libs = os.pathsep.join(_LIBS)
    env["PYTHONPATH"] = libs + (os.pathsep + env["PYTHONPATH"]
                                if env.get("PYTHONPATH") else "")
    if extra:
        env.update({k: str(v) for k, v in extra.items()})
    return env


def _rel(p) -> str:
    return os.path.relpath(p, REPO) if isinstance(p, str) and p.startswith(REPO) else str(p)


def _short(label: str) -> str:
    """Compact stage tag for the one-line progress markers: the stage verb +
    the model, e.g. 'edges  segrev3' from 'EDGES [segrev3] — …'. Falls back to
    the raw label when it doesn't match the 'VERB [model] — …' shape."""
    head = label.split("—", 1)[0].strip()        # drop the long description
    verb, _, rest = head.partition("[")
    verb = verb.strip().lower()
    model = rest.split("]", 1)[0].strip() if "]" in rest else ""
    return f"{verb}  {model}".rstrip() if verb else label


def _fmt_cmd(argv: list, env_extra: dict | None) -> str:
    """The full child command line (env + argv), used by --dry-run and --verbose."""
    envs = " ".join(f"{k}={_rel(v)}" for k, v in (env_extra or {}).items())
    cmd = " ".join(str(a) if str(a) == sys.executable else _rel(a) for a in argv)
    return f"{(envs + ' ') if envs else ''}{cmd}"


class Runner:
    """Runs (or, in dry-run, just prints) stages and tallies (label, secs, rc).

    Prints a single compact line per stage with its ordinal position k/N —
    `▶ [k/N] <stage>` on start, `  ✓ <stage> (Ns)` / `  ✗ <stage> (exit r, Ns)`
    on finish — instead of the old full-width banners. The full child argv is
    shown only under --verbose (or --dry-run, whose point is to show the plan)."""

    def __init__(self, dry_run: bool = False, verbose: bool = False, total: int = 0):
        self.dry_run = dry_run
        self.verbose = verbose
        self.total = total          # number of stage steps this invocation
        self.idx = 0                # current 1-based ordinal
        self.py = sys.executable
        self.ran: list = []
        self.start_t0 = None        # wall-clock start (set in main); for the heartbeat
        self.cur_label = None       # label of the step currently running
        self.cur_t0 = None          # when the current step started

    def tally(self, label: str, dt: float, rc: int) -> None:
        self.ran.append((label, dt, rc))

    def begin_step(self, tag: str) -> None:
        """Open ONE pipeline step: bumps the ordinal and prints `▶ [k/N] tag`.
        Called once per build_steps() entry so k/N counts actual stage steps."""
        self.idx += 1
        self.cur_label = tag
        self.cur_t0 = time.time()
        pos = f"[{self.idx}/{self.total}]" if self.total else f"[{self.idx}]"
        print(f"\n▶ {pos} {tag}", flush=True)

    def _done(self, tag: str, dt: float, rc: int) -> None:
        if rc == 0:
            print(f"  ✓ {tag}  ({dt:.0f}s)", flush=True)
        else:
            print(f"  ✗ {tag}  (exit {rc}, {dt:.0f}s)", flush=True)

    def run(self, label: str, argv: list, env_extra: dict | None = None,
            tally: bool = True) -> int:
        """Run one child job within the current step (the step header is printed
        by begin_step). Prints the argv only under --dry-run/--verbose. A failure
        is recorded (✗) and NEVER aborts the run — remaining steps/models continue
        and the final summary exits nonzero (by design: no result-dependent
        skips, no mid-run aborts).

        ``tally=False`` runs the job but keeps it out of ``self.ran`` (and thus
        out of the final ``N/M stages succeeded`` summary) — used for the
        skip-check PROBE, whose nonzero rc means "retrain needed", a normal
        outcome rather than a stage failure."""
        tag = _short(label)
        if self.dry_run or self.verbose:
            print(f"    $ {_fmt_cmd(argv, env_extra)}", flush=True)
        if self.dry_run:
            if tally:
                self.tally(label, 0.0, 0)
            return 0
        t0 = time.time()
        rc = subprocess.run(argv, cwd=REPO, env=_child_env(env_extra)).returncode
        dt = time.time() - t0
        if tally:
            self.tally(label, dt, rc)
        self._done(tag, dt, rc)
        if rc != 0 and tally:
            print(f"    ✗ STAGE FAILED (continuing): {label}  (exit {rc}, {dt:.0f}s)",
                  flush=True)
        return rc

    def run_parallel(self, group_label: str, specs: list) -> None:
        """Run (label, argv, env_extra) jobs CONCURRENTLY (waves are sized by stage
        dependencies only — no process-count cap). When PIPELINE_GPU_BUDGET is set
        (the Modal container sets 0.9) the budget is split evenly across the wave via
        per-child SEGREV3_GPU_MEM_FRACTION; locally each child keeps the standard
        per-process fraction rail. Each child's BLAS/OMP threads are capped to cpu_count//njobs
        (the main.py run_parallel:148–176 pattern); output is captured per job
        and printed in submission order once each finishes. The step header is
        printed by begin_step before this call."""
        import concurrent.futures
        n = len(specs)
        threads = max(1, (os.cpu_count() or 2) // max(1, n))
        blas = {v: str(threads) for v in
                ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")}
        budget = float(os.environ.get("PIPELINE_GPU_BUDGET", "0"))
        if budget > 0 and n > 1:
            # Per-process CAPS, not reservations: floor at half the budget so a
            # memory-hungry stage (patching peaks ~10 GiB) isn't starved — the
            # wave's REAL concurrent usage fits the A100; the cap only stops one
            # child from taking the whole GPU.
            blas["SEGREV3_GPU_MEM_FRACTION"] = f"{max(budget / n, budget / 2):.4f}"
        if self.dry_run or self.verbose:
            for label, argv, env_extra in specs:
                print(f"    {_short(label)}:  $ "
                      f"{_fmt_cmd(argv, {**blas, **(env_extra or {})})}", flush=True)
        if self.dry_run:
            for label, _, _ in specs:
                self.tally(label, 0.0, 0)
            return

        def _one(spec):
            label, argv, env_extra = spec
            t0 = time.time()
            p = subprocess.run(argv, cwd=REPO, env=_child_env({**blas, **(env_extra or {})}),
                               capture_output=True, text=True)
            return label, time.time() - t0, p.returncode, p.stdout, p.stderr

        with concurrent.futures.ThreadPoolExecutor(max_workers=n) as ex:
            results = list(ex.map(_one, specs))     # preserves submission order

        for label, dt, rc, out, err in results:
            tag = _short(label)
            print(f"  ── {tag} output ──", flush=True)
            if out:
                print(out, end="" if out.endswith("\n") else "\n", flush=True)
            if err:
                print(err, end="" if err.endswith("\n") else "\n", flush=True)
            self._done(tag, dt, rc)
            self.tally(label, dt, rc)
            if rc != 0:                          # recorded; the run continues
                print(f"    ✗ STAGE FAILED (continuing): {label}  (exit {rc})", flush=True)


# --------------------------------------------------------------------------- #
# Checkpoint resolution (SEGREV3_CKPT) — resolved at LAUNCH time, not plan     #
# time, so `--stages all` sees the checkpoint the train stage just produced.   #
# --------------------------------------------------------------------------- #
_CKPT_NOTES_SHOWN: set = set()


def resolve_ckpt(name: str, a) -> str:
    ckpt = conv_ckpt(name)
    if not os.path.exists(ckpt) and ("warn", name) not in _CKPT_NOTES_SHOWN:
        # No abort: the child fails loudly, is recorded ✗, and the run continues.
        _CKPT_NOTES_SHOWN.add(("warn", name))
        print(f"  NOTE: checkpoint missing: {_rel(ckpt)} — dependent stages for "
              f"{name!r} will fail and be recorded (train first)", flush=True)
    return ckpt


# --------------------------------------------------------------------------- #
# Stage command builders. Each job is (label, argv, needs_ckpt).               #
# --------------------------------------------------------------------------- #
def jobs_data(name: str, a) -> list:
    # ONE input pool per circuit, generated by the generic gen_circuit_data.
    return [
        (f"DATA [{name}] — per-circuit input pools ({a.only_circuit or 'all circuits'})",
         [sys.executable, os.path.join(MODELS_DIR, "gen_data", "gen_circuit_data.py"),
          "--model", name, "--circuit", (a.only_circuit or "all"),
          "--seed", str(canon.DEFAULT_SEED)], False),
    ]


def jobs_probes(name: str, a) -> list:
    # C2: fit linear probes for the model's variables (site-scan) + validate; per model.
    py, seed = sys.executable, str(canon.DEFAULT_SEED)
    return [
        (f"PROBES-TRAIN [{name}] — site-scan + linear probes",
         [py, os.path.join(PIPELINE, "probes", "train_probes.py"),
          "--model", name, "--site-scan", "--seed", seed], False),
        (f"PROBES-VALIDATE [{name}] — selectivity + causal gates",
         [py, os.path.join(PIPELINE, "probes", "validate_probe.py"),
          "--model", name, "--seed", seed], False),
    ]


#: escalation ladder: initial tier -> upgrade target when acc lands in [0.80,0.95).
#: Arch variants climb WITHIN their arch (xlarge geometry, arch flag kept); size
#: variants step up one; xlarge has nothing bigger (pinned).
UPGRADE_LADDER = {"large": "xlarge", "small": "large", "wide": "xlarge",
                  "deep": "xlarge", "xlarge": None, "mlp": "mlp_xl",
                  "moe4": "moe4_xl", "moe8": "moe8_xl"}


def _stacked_pretrain(models: list, a) -> None:
    """--train-backend stacked: batch-vmap-train the whole set on one GPU, then promote
    through promote_grid (train_sharded --promote runs it). The normal per-model `train`
    stage then gate-skips the >=SEQ_ACC_GATE keepers and retrains sub-gate stragglers via
    the sh escalation — so the straggler fallback is the existing gate, no extra logic."""
    sharded = os.path.join(MODELS_DIR, "train_models", "train_sharded.py")
    print(f"⇉ stacked pre-pass: vmap-training {len(models)} model(s) via train_sharded", flush=True)
    subprocess.run([sys.executable, sharded, "--local", "--promote",
                    "--target", str(SEQ_ACC_GATE), "--models", *models], check=False)


def job_train(name: str, a) -> tuple:
    """Train a DSL model via the TransformerLens trainer. run_training_job.sh trains
    the tier and, if acc lands in [0.80,0.95), escalates to UPGRADE_LADDER's arch-aware
    target, promoting the best ckpt to trained_models_pt/<name>.pt ONLY if it passes the
    gate. The recipe per tier comes from train_models/hparams.py, never from here."""
    py = sys.executable
    train_py = os.path.join(MODELS_DIR, "train_models", "train.py")
    if a.smoke:                                          # quick throwaway: tiny preset, 20 steps
        return (f"TRAIN [{name}] — SMOKE (THROWAWAY ckpt)",
                [py, train_py, "--algorithm", name, "--config", "tiny", "--smoke",
                 "--output_dir", SMOKE_DIR, "--allow_repeats", "--random_sampling"], False)
    sh = os.path.join(MODELS_DIR, "train_models", "run_training_job.sh")
    tier = hparams.tier_of(name)
    upgrade = UPGRADE_LADDER[tier]
    esc = ["--upgrade-config", upgrade] if upgrade else ["--no-upgrade"]
    seeds = ["--seeds", str(a.seeds)] if a.seeds > 1 else []
    return (f"TRAIN [{name}] — TL trainer (run_training_job.sh: "
            f"{tier + '->' + upgrade if upgrade else tier + ' pinned'}, {SEQ_ACC_GATE} gate, "
            f"lr {hparams.for_tier(tier).lr:g}) -> trained_models_pt/{name}.pt",
            ["bash", sh, "--config", tier, *esc, *seeds, name], False)


def job_analysis(name: str, stage: str, circuit: str, a) -> tuple:
    """patching | edges | acdc | eap | edge_pruning | smallest — all need
    SEGREV3_CKPT. Each analyzes ONE circuit of ``name`` (a slug from circuits.py);
    artifacts land under circuits_data/<name>/<circuit>/..."""
    py = sys.executable
    tail = ["--smoke"] if a.smoke else []
    circ = ["--circuit", circuit]
    sel = ["--circuit", circuit]
    clab = f" «{circuit}»"
    c4 = (["--zero-ablation"] if a.zero_ablation else []) + (["--use-pos-embed"] if a.use_pos_embed else [])
    if stage == "patching":
        argv = [py, os.path.join(PIPELINE, "interp_tools", "patching.py"),
                "--model", name, "--analysis", "all"] + sel
        argv += [] if a.smoke else ["-n", str(PATCHING_N)]
        return (f"PATCHING [{name}]{clab} — act/att/path/dla + heatmaps", argv + tail, True)
    if stage == "edges":
        argv = [py, os.path.join(PIPELINE, "interp_tools", "edges.py"),
                "--model", name, "--analysis", "all"] + sel
        argv += [] if a.smoke else ["-n", str(EDGES_N)]
        return (f"EDGES [{name}]{clab} — composition/QK_OV/pos_pos/pos_token + heatmaps",
                argv + tail, True)
    if stage == "acdc":
        taus = canon.CANON_TAUS[:1] if a.smoke else canon.CANON_TAUS
        return (f"ACDC [{name}]{clab} — recursive edge pruning (taus {taus})",
                [py, os.path.join(PIPELINE, "find_circuits", "acdc.py"),
                 "--model", name, "--taus", *map(str, taus),
                 "--examples", str(ACDC_EXAMPLES)] + circ + c4 + tail, True)
    if stage == "eap":
        # A REGIME, not rungs: eap resolves the ladder from the model's LIVE edge budget
        # (model_registry.topns). The old fixed ladder is what capped every C1/C3 row at
        # rung 800 — 62% of the graph at `large`, 3.6% at `xl`, and at `small` BOTH top200
        # and top800 aliased the whole unpruned graph (247% and 988% of 81 live edges).
        # --smoke shrinks eap's eval set, not the rung count: an explicit rung above the
        # live budget is now a hard error, so it cannot be sliced from here.
        return (f"EAP-IG [{name}]{clab} — edge attribution "
                f"({canon.CANON_TOPN_REGIME}-regime scale-relative ladder)",
                [py, os.path.join(PIPELINE, "find_circuits", "eap.py"),
                 "--model", name, "--topn-regime", canon.CANON_TOPN_REGIME]
                + circ + c4 + tail, True)
    if stage == "edge_pruning":
        sps = (canon.CANON_EDGE_PRUNING_SPARSITIES[:1] if a.smoke
               else canon.CANON_EDGE_PRUNING_SPARSITIES)
        return (f"EDGE_PRUNING [{name}]{clab} — L0 edge-mask training (sparsities {sps})",
                [py, os.path.join(PIPELINE, "find_circuits", "edge_pruning.py"),
                 "--model", name, "--sparsities", *map(str, sps)] + circ + c4 + tail, True)
    if stage == "smallest":
        # reads the candidates under circuits_data/<m>/<circuit>/circuits/ and spawns
        # its own <=SMALLEST_JOBS children. The target defaults to SMALLEST_TARGET
        # but can be overridden (--smallest-target) — the build_models driver wires
        # the user's whole-sequence accuracy threshold X here.
        target = str(a.smallest_target)
        base = contrast_base(name, circuit)   # ContrastCircuit -> differential vs base
        if base and contrast_is_suppression(name, circuit):
            diff = ["--base-circuit", base, "--suppression"]   # cue-flip C3: fringe suppressor
        elif base:
            diff = (["--base-circuit", base]
                    + (["--global-predicate"] if contrast_is_global(name, circuit) else []))
        else:
            diff = []
        # --search in EVERY mode. It used to be plain-only, which left differential and
        # suppression rows with only the fixed candidate ladder: the winning rung was
        # exactly the top ladder step at all 8 arches for all 1503 C1 and 400 C3 scopes,
        # 0 above, and C1's per-arch faithful rate tracks that rung's graph coverage with
        # rank correlation +1.000 over the attn-only tiers. It ADDS a high-n candidate to
        # the pool and nothing more: eap's search optimises a whole-sequence floor, which
        # is NOT the C1/C3 eligibility metric (n_fringe_edges > 0 and s_union >= target).
        # Breaking the cap is the point; selection stays the mode's own rule.
        return (f"SMALLEST [{name}]{clab} — smallest faithful circuit "
                f"(target {target}, jobs {SMALLEST_JOBS}, +search, runs ALONE)",
                [py, os.path.join(PIPELINE, "find_circuits", "smallest_faithful_circuit.py"),
                 "--model", name, "--target", target, "--jobs", str(SMALLEST_JOBS), "--search"]
                + circ + diff + tail, True)
    raise ValueError(stage)


def job_router(name: str) -> tuple:
    """Per-model TaskMux gating/outvoting diagnostic -> task_router/<name>/diagnostic.json."""
    return (f"ROUTER [{name}] — TaskMux gating/outvoting diagnostic "
            f"-> task_router/{name}/diagnostic.json",
            [sys.executable, os.path.join(PIPELINE, "task_router", "router_diagnostic.py"),
             "--model", name], True)


# --------------------------------------------------------------------------- #
# train skip-check: skip the retrain iff the ckpt exists AND passes the gate   #
# (recorded test seq-acc >= SEQ_ACC_GATE; falls back to a live eval). Runs as ONE #
# subprocess (check_ckpt_gate.py) so torch/GPU never enter the orchestrator.   #
# --------------------------------------------------------------------------- #
CHECK_GATE_PY = os.path.join(MODELS_DIR, "train_models", "check_ckpt_gate.py")


def run_train(name: str, a, r: Runner) -> None:
    label, argv, _ = job_train(name, a)
    # the shell trainer shells out to hparams.py/promote.py — hand it OUR interpreter
    train_env = {"PYTHON": sys.executable}
    check_argv = [sys.executable, CHECK_GATE_PY,
                  REPO, name, conv_ckpt(name), str(a.train_gate)]
    check_label = (f"TRAIN [{name}] — ckpt gate check "
                   f"(skip retrain iff seq-acc >= {a.train_gate})")
    if a.smoke:                                   # smoke always trains (throwaway out)
        r.run(label, argv, env_extra=train_env)
        return
    if a.dry_run:
        r.run(check_label, check_argv)
        print(f"    (retrain SKIPPED iff ckpt {_rel(conv_ckpt(name))} exists "
              f"and the check exits 0)", flush=True)
        r.run(label, argv, env_extra=train_env)
        return
    if os.path.exists(conv_ckpt(name)):
        rc = r.run(check_label, check_argv, tally=False)
        if rc == 0:
            print(f"    → skip retrain: ckpt passes the seq-acc gate", flush=True)
            r.tally(f"TRAIN [{name}] — SKIPPED (ckpt passes)", 0.0, 0)
            return
        print("    → ckpt fails the gate — retraining", flush=True)
    else:
        print(f"    (no ckpt at {_rel(conv_ckpt(name))} — training)", flush=True)
    r.run(label, argv, env_extra=train_env)


def _positional_suffix(a) -> str:
    """C4 candidate-dir suffix (mirrors eap.py:1001); '' in normal output-metric mode.
    Non-empty ⇒ finder-only positional analysis: candidates aren't output-metric
    comparable, so the smallest stage is dropped and the audit checks the suffixed dirs."""
    return ("_zero" if a.zero_ablation else "") + ("_pos" if a.use_pos_embed else "")


# --------------------------------------------------------------------------- #
# Tree audit (per-circuit existence check)                                     #
# --------------------------------------------------------------------------- #
#: candidate-dir name prefix per finder. Rung VALUES are per-model under the scale-relative
#: ladder (canon.CANON_TOPN_FRACS x the live edge budget), so the audit enumerates the dirs
#: that exist and checks the family is represented — it cannot predict rung names.
FINDER_PREFIX = {"acdc": "acdc_thresh_", "eap": "eap_top",
                 "edge_pruning": "edge_pruning_sparsity_"}


def ckpt_verdict(name: str, gate: float) -> "promote.Verdict":
    """The checkpoint's recorded seq-acc verdict (train_models/promote.py — the ONE gate)."""
    import promote                              # train_models is on sys.path (hparams import)
    return promote.check(conv_ckpt(name), gate)


def empty_circuit_baseline(name: str, circuit: str, smallest: dict | None) -> tuple:
    """(baseline, source) for the EMPTY circuit on this circuit's scope.

    Prefers the smallest stage's MEASURED ``perf_empty.sequence_accuracy`` (every edge
    severed). Where that has not been produced yet, computes the exact upper bound instead:
    a fully-ablated model is input-independent, so it can do no better than the best
    constant prediction — the modal output row's frequency in the input pool. Either way a
    baseline at/above the target means the isolation is degenerate (zero clean/corrupt
    contrast), and every candidate 'passes' for free."""
    if smallest and "perf_empty" in smallest:
        return smallest["perf_empty"]["sequence_accuracy"], "measured (perf_empty)"
    glue = _glue()
    import collections

    import torch
    pool = json.load(open(glue.circuit_dataset_path(name, circuit)))
    out = glue.get_spec(name).task_fn(torch.tensor(pool["inputs"], dtype=torch.long))
    rows = collections.Counter(tuple(r) for r in out.tolist())
    return rows.most_common(1)[0][1] / len(out), "constant-predictor bound"


def run_audit(name: str, a, enumerate_only: bool = False) -> int:
    """Per-circuit existence + soundness audit. For each hand-declared circuit it verifies
    the artifacts the pipeline produces under ``circuits_data/<name>/<circuit>/`` — the
    candidate dirs (one per default finder family, each with circuit.json + circuit.pt),
    smallest_circuit.json (+ .pt unless no_faithful), the circuits/ subtree
    (heads_only/ + edges/ only when their non-default stage was requested or already
    exists), and the figures mirror. ContrastCircuits additionally need the BASE
    circuit's smallest json+pt and a differential-mode smallest json.

    Two SOUNDNESS checks sit alongside the existence ones, because their absence is why
    209 circuit rows were built on 49 sub-gate checkpoints and why degenerate isolations
    shipped: the model's checkpoint must pass the train gate, and the circuit's
    empty-circuit baseline must be BELOW the smallest target. Returns the gap count."""
    glue = _glue()
    suffix = _positional_suffix(a)                  # C4 positional: candidate dirs gain _zero/_pos
    circuits = circuits_for_run(name, a)
    if enumerate_only:
        print(f"  DSL per-circuit audit — {len(circuits)} circuit(s) {circuits}; each expects "
              f">=1 candidate dir per finder {list(canon.DEFAULT_FINDERS)} + smallest_circuit.json "
              f"+ heads_only/ + edges/ + a figures mirror; plus the model's ckpt gate "
              f"(>= {a.train_gate}) and a per-circuit empty-circuit baseline "
              f"(< {a.smallest_target}).")
        return 0
    issues, checks = [], 0
    checks += 1
    v = ckpt_verdict(name, a.train_gate)            # (2): rows must not rest on a sub-gate ckpt
    if not v.passed:
        issues.append(("<model>", conv_ckpt(name), f"CKPT FAILS GATE: {v.line(name)}"))
    for circuit in circuits:
        dd, fd = glue.data_dir(name, circuit), glue.fig_dir(name, circuit)
        cdirs = (sorted(os.listdir(os.path.join(dd, "circuits")))
                 if os.path.isdir(os.path.join(dd, "circuits")) else [])
        for finder in canon.DEFAULT_FINDERS:
            checks += 1
            pfx = FINDER_PREFIX[finder]
            fam = [c for c in cdirs if c.startswith(pfx) and c.endswith(suffix)]
            if not fam:
                issues.append((circuit, os.path.join(dd, "circuits", pfx + "*" + suffix),
                               f"NO {finder} candidate dir"))
            for c in fam:
                for fn in ("circuit.json", "circuit.pt"):   # uniform across methods
                    checks += 1
                    cf = os.path.join(dd, "circuits", c, fn)
                    if not os.path.exists(cf):
                        issues.append((circuit, cf, "MISSING"))
        # Output-metric artifacts (smallest + contrast base) — SKIPPED in C4 positional mode,
        # whose candidates aren't output-metric comparable and produce no smallest_circuit.
        if not suffix:
            # smallest_circuit.json always required; .pt only when a faithful circuit was
            # selected (a no_faithful_circuit marker legitimately has no .pt).
            checks += 1
            sj_path = os.path.join(dd, "smallest_circuit.json")
            sj = None
            if not os.path.exists(sj_path):
                issues.append((circuit, sj_path, "MISSING"))
            else:
                try:
                    sj = json.load(open(sj_path))
                except (OSError, ValueError):
                    pass
            no_faithful = bool(sj.get("no_faithful_circuit")) if sj else False

            checks += 1                             # degenerate-isolation tripwire
            pool_path = glue.circuit_dataset_path(name, circuit)
            if not (sj and "perf_empty" in sj) and not os.path.exists(pool_path):
                issues.append((circuit, pool_path, "MISSING input pool (data stage)"))
            else:
                eb, src = empty_circuit_baseline(name, circuit, sj)
                if eb >= float(a.smallest_target):
                    issues.append((circuit, sj_path if "measured" in src else pool_path,
                                   f"DEGENERATE ISOLATION: empty-circuit baseline {eb:.4f} "
                                   f"[{src}] >= target {a.smallest_target} (no clean/corrupt "
                                   f"contrast — every candidate passes for free)"))
            if not no_faithful:
                checks += 1
                sp_path = os.path.join(dd, "smallest_circuit.pt")
                if not os.path.exists(sp_path):
                    issues.append((circuit, sp_path, "MISSING"))
            # ContrastCircuit: differential smallest consumed the base's artifacts.
            base = contrast_base(name, circuit)
            if base:
                checks += 1
                bj_path = os.path.join(glue.data_dir(name, base), "smallest_circuit.json")
                bj = None
                if not os.path.exists(bj_path):
                    issues.append((circuit, bj_path, "MISSING base artifact"))
                else:
                    try:
                        bj = json.load(open(bj_path))
                    except (OSError, ValueError):
                        pass
                if not (bool(bj.get("no_faithful_circuit")) if bj else False):   # unfaithful base has no .pt
                    checks += 1
                    bp_pt = os.path.join(glue.data_dir(name, base), "smallest_circuit.pt")
                    if not os.path.exists(bp_pt):
                        issues.append((circuit, bp_pt, "MISSING base artifact"))
                checks += 1
                want_mode = "suppression" if contrast_is_suppression(name, circuit) else "differential"
                if not (no_faithful or (sj or {}).get("mode") == want_mode):
                    issues.append((circuit, sj_path, f"not {want_mode} mode"))
        checks += 1
        if not os.path.isdir(os.path.join(dd, "circuits")):
            issues.append((circuit, os.path.join(dd, "circuits"), "MISSING dir"))
        # heads_only/ + edges/ come from the non-default patching/edges stages
        # (dropped from DEFAULT_STAGES 2026-07-13): required only when their stage
        # was requested this run; pre-existing trees still get checked.
        for d, st in (("heads_only", "patching"), ("edges", "edges")):
            if st in a.stages or os.path.isdir(os.path.join(dd, d)):
                checks += 1
                if not os.path.isdir(os.path.join(dd, d)):
                    issues.append((circuit, os.path.join(dd, d), "MISSING dir"))
        checks += 1
        if not os.path.isdir(fd):
            issues.append((circuit, fd, "MISSING figures dir"))
    gaps = len(issues)
    if issues:
        print(f"\n  ISSUES ({gaps}):")
        for circuit, path, reason in issues[:40]:
            print(f"    [{circuit}] {_rel(path)}: {reason}")
    print(f"\n  AUDIT {'PASS' if gaps == 0 else 'FAIL'}: {checks - gaps}/{checks} present "
          f"across {len(circuits)} circuit(s) {circuits} -> {gaps} gap(s)", flush=True)
    return gaps


# --------------------------------------------------------------------------- #
# Plan building + execution (the wave schedule, per model)                     #
# --------------------------------------------------------------------------- #
def _circuit_objs(name: str) -> list:
    """Built DomainCircuit instances for ``name`` (models/circuits.py via gen_circuit_data),
    stable-sorted so ContrastCircuits come LAST — their differential smallest stage
    reads the BASE circuit's smallest_circuit.json, which must land first."""
    gd = os.path.join(MODELS_DIR, "gen_data")
    for p in (MODELS_DIR, gd):
        if p not in sys.path:
            sys.path.insert(0, p)
    import gen_circuit_data as GCD             # lazy (pulls algorithms/torch)
    import circuits as dsl_circuits            # noqa: E402
    return sorted(GCD.circuits_for(name),
                  key=lambda c: isinstance(c, dsl_circuits.ContrastCircuit))


def model_circuits(name: str) -> list:
    """Circuits to run for ``name``: the HAND-DECLARED DomainCircuit subclasses from
    models/circuits.py — no whole, no auto-detection; non-contrast circuits first."""
    return [c.circuit_id for c in _circuit_objs(name)]


def contrast_is_global(name: str, circuit: str) -> bool:
    """True when the ContrastCircuit declares local_predicate=False (globally-consumed
    predicate -> smallest gets --global-predicate; d_unaffected reported, not gated)."""
    c = next(c for c in _circuit_objs(name) if c.circuit_id == circuit)
    return not c.local_predicate            # a ContrastCircuit dataclass field; callers gate on that


def contrast_base(name: str, circuit: str) -> str | None:
    """The declared base slug when ``circuit`` is a ContrastCircuit, else None."""
    c = next(c for c in _circuit_objs(name) if c.circuit_id == circuit)
    import circuits as dsl_circuits            # on sys.path after _circuit_objs
    if not isinstance(c, dsl_circuits.ContrastCircuit):
        return None
    if c.base_circuit is None:
        raise ValueError(f"{name}/{circuit}: ContrastCircuit without base_circuit")
    return c.base_circuit


def contrast_is_suppression(name: str, circuit: str) -> bool:
    """True when ``circuit`` is a SuppressionContrastCircuit (cue-flip C3): smallest gets
    --suppression (select the fringe suppressor by release_rate, not s_union)."""
    c = next(c for c in _circuit_objs(name) if c.circuit_id == circuit)
    import circuits as dsl_circuits            # on sys.path after _circuit_objs
    return isinstance(c, dsl_circuits.SuppressionContrastCircuit)


def mux_task_circuits(name: str) -> list:
    """The task{i}[_*] circuit slugs — the router diagnostic's per-task inputs."""
    return [c for c in model_circuits(name) if re.fullmatch(r"task\d+(_.*)?", c)]


def suppression_circuits(name: str) -> list:
    """The model's SuppressionContrastCircuit slugs (C3 rows)."""
    return [c for c in model_circuits(name) if contrast_is_suppression(name, c)]


def router_diagnostic(name: str) -> str:
    return os.path.join(PIPELINE, "task_router", name, "diagnostic.json")


def c3_universe() -> list:
    """Every model the C3 fan-out COULD cover: the run-list UNION the artifacts on disk.

    models.txt now carries all eight tiers (1936 rows = 8 x 242), so `--model all` reaches
    small/wide/xl too; the union still matters because a checkpoint or router diagnostic can
    exist for a model the run-list does not (or no longer) name."""
    ck = {f[:-3] for f in os.listdir(TRAINED_MODELS_PT) if f.endswith(".pt")} \
        if os.path.isdir(TRAINED_MODELS_PT) else set()
    rt = os.path.join(PIPELINE, "task_router")
    diag = {d for d in os.listdir(rt) if os.path.exists(router_diagnostic(d))} \
        if os.path.isdir(rt) else set()
    # C3 lives only on TaskMux models, whose ids are the mux<N>.<member>… grammar.
    return sorted(n for n in set(list_models()) | ck | diag if n.startswith("mux"))


def c3_worklist(models: list) -> list:
    """(model, circuit, has_ckpt, has_router) for every declared C3 row of ``models``."""
    rows = []
    for m in models:
        circs = suppression_circuits(m)
        if not circs:
            continue
        rows += [(m, c, os.path.exists(conv_ckpt(m)), os.path.exists(router_diagnostic(m)))
                 for c in circs]
    return rows


def print_c3_worklist(models: list) -> None:
    """Print the C3 fan-out work-list (models x circuits) + the blocked prerequisites."""
    rows = c3_worklist(models)
    ready = [(m, c) for m, c, ck, dg in rows if ck and dg]
    blocked = [(m, c, ck, dg) for m, c, ck, dg in rows if not (ck and dg)]
    per_arch = collections.Counter(m.partition("__")[2] or "large" for m, _ in ready)
    print(f"C3 suppression work-list — {len(ready)} row(s) runnable across "
          f"{len({m for m, _ in ready})} model(s)")
    for a in hparams.ARCHES:
        print(f"  {a:6} {per_arch[a]:5} row(s)")
    for m in sorted({m for m, _ in ready}):
        cs = [c for mm, c in ready if mm == m]
        print(f"  {m:60} {len(cs):3}  {' '.join(cs)}")
    print(f"\n  re-run command (fan-out covers every listed arch):\n"
          f"    run_model_pipeline.py --stages smallest audit --model "
          f"{' '.join(sorted({m for m, _ in ready}))}")
    if blocked:
        print(f"\n  {len(blocked)} row(s) BLOCKED on a missing prerequisite:")
        for m, c, ck, dg in blocked:
            why = ", ".join(w for w, ok in (("no checkpoint", ck), ("no router diagnostic", dg))
                            if not ok)
            print(f"    {m:60} {c:22} {why}")


def restore_keys_for(name: str, circuit: str | None) -> str:
    """Volume tars a per-circuit Modal container restores: the contrast base's smallest
    artifacts; a suppression row also needs EVERY task circuit's (its in-container
    router diagnostic reads them all)."""
    base = contrast_base(name, circuit) if circuit else None
    if base is None:
        return ""
    keys = [f"{name}__{base}"]
    if contrast_is_suppression(name, circuit):
        keys += [f"{name}__{t}" for t in mux_task_circuits(name) if t != base]
    return ",".join(keys)


def suppression_verdict(name: str, circuit: str) -> str:
    """The ROUTER diagnostic's (cued,wrong) verdict for a suppression row — echoed
    into the row's output via --router-verdict; recorded metadata, NEVER a gate.
    Fails loud on a missing diagnostic (stage-ordering bug)."""
    diag = os.path.join(PIPELINE, "task_router", name, "diagnostic.json")
    if not os.path.exists(diag):
        raise FileNotFoundError(f"suppression row {name}/{circuit}: {diag} missing — "
                                "the ROUTER diagnostic step must run first")
    c = next(c for c in _circuit_objs(name) if c.circuit_id == circuit)
    hits = [p for p in json.load(open(diag))["pairs"]
            if p["cued"] == c.cued_task and p["wrong"] == c.wrong_task]
    if len(hits) != 1:
        raise ValueError(f"suppression row {name}/{circuit}: {len(hits)} "
                         f"({c.cued_task},{c.wrong_task}) pairs in {diag} (want 1)")
    return hits[0]["verdict"]


def circuits_for_run(name: str, a) -> list:
    """``model_circuits(name)`` restricted to ``a.only_circuit`` when that flag is set
    (the --modal per-circuit edge_pruning fan-out scopes each Modal A100 container to
    ONE circuit via ``--only-circuit``). Unset → all circuits (the normal local path).
    Raises if the requested slug isn't a circuit of the model."""
    circuits = model_circuits(name)
    oc = a.only_circuit
    if oc is None:
        return circuits
    sel = [c for c in circuits if c == oc]
    if not sel:
        raise SystemExit(f"✗ --only-circuit {oc!r} is not a circuit of {name} "
                         f"(have {circuits})")
    return sel


def build_steps(name: str, stages: list, a) -> list:
    """Ordered steps; each is (kind, group_label, jobs) with kind in
    serial | parallel | train | audit. The analysis waves (2a patching∥edges∥acdc∥eap∥
    edge_pruning, 2b smallest — a real dependency, it reads the finders' candidates)
    run ONCE PER CIRCUIT; data + train
    run once per model.

    With --serial-waves the two parallel waves are emitted as SEPARATE serial
    single-GPU steps so a higher-level driver can run several MODELS at once while
    keeping each model to a single GPU process."""
    serial_waves = a.serial_waves
    steps = []
    if "data" in stages:
        steps.append(("serial", f"WAVE data [{name}] (GPU-free)", jobs_data(name, a)))
    if "train" in stages:
        steps.append(("train", f"WAVE train [{name}] (GPU slot 1)", None))
    if "probes" in stages:
        steps.append(("serial", f"WAVE probes [{name}] (GPU slot 1)", jobs_probes(name, a)))

    analysis = {"patching", "edges", "acdc", "eap", "edge_pruning", "smallest"}
    if analysis & set(stages):
        with_router = "smallest" in stages and not a.smoke   # smoke stays quarantined, no ROUTER
        router_added = False
        for circuit in circuits_for_run(name, a):
            clab = f" «{circuit}»"
            supp = with_router and contrast_is_suppression(name, circuit)
            if supp and not router_added:   # once, after every task circuit's smallest
                steps.append(("serial",
                              f"ROUTER [{name}] — TaskMux gating/outvoting diagnostic "
                              "(verdict recorded on the suppression rows)",
                              [job_router(name)]))
                router_added = True

            def add_wave(stage_group, wave, circuit=circuit, clab=clab):
                present = [s for s in stage_group if s in stages]
                if not present:
                    return
                if serial_waves or len(present) == 1:
                    for s in present:           # each its own serial single-GPU step
                        steps.append(("serial",
                                      f"WAVE {wave} [{name}]{clab} — {s} (1 GPU slot)",
                                      [job_analysis(name, s, circuit, a)]))
                else:
                    steps.append(("parallel",
                                  f"WAVE {wave} [{name}]{clab} — " + " ∥ ".join(present)
                                  + f" ({len(present)} GPU slots)",
                                  [job_analysis(name, s, circuit, a) for s in present]))

            add_wave(("patching", "edges", "acdc", "eap", "edge_pruning"), "2a")
            if "smallest" in stages:
                label_2b = (f"WAVE 2b [{name}]{clab} — smallest faithful circuit "
                            "(after the finders; spawns its own eval children)")
                sjob = [job_analysis(name, "smallest", circuit, a)]
                steps.append(("gated_smallest", label_2b + " [+router-verdict]",
                              (circuit, sjob)) if supp else ("serial", label_2b, sjob))
    if "audit" in stages:
        steps.append(("audit", f"AUDIT [{name}]", None))
    return steps


def _model_of(label: str) -> str:
    """The model name in a 'PREFIX [model] …' label, or '' if none."""
    return label.split("[", 1)[1].split("]", 1)[0].strip() if "[" in label else ""


def _step_tag(kind: str, label: str, jobs) -> str:
    """A compact `▶ [k/N] <tag>` header tag for one build_steps() step.
    Parallel groups read `2a: patching ∥ edges`; serial reads `<verb>  <model>`
    from the first job; train/audit read `train|audit  <model>`."""
    if kind == "parallel" and jobs:
        wave = ""
        head = label.split("[", 1)[0].strip()        # e.g. 'WAVE 2a'
        if head.upper().startswith("WAVE "):
            wave = head[5:].strip() + ": "
        return wave + " ∥ ".join(_short(j[0]).split()[0] for j in jobs)
    if kind == "serial" and jobs:
        return _short(jobs[0][0])
    if kind == "gated_smallest":                      # jobs = (circuit, [job])
        return _short(jobs[1][0][0]) + " [+router-verdict]"
    return f"{kind}  {_model_of(label)}".rstrip()     # train / audit


# --------------------------------------------------------------------------- #
# Modal execution (opt-in --modal): every GPU stage runs on A100 containers —   #
# one container per single-circuit model; multi-circuit models train once (R1)  #
# then fan one container per circuit (R2). Local work: spawn/poll/sync + audit. #
# Requires `modal deploy pipeline/modal_pipeline_app.py` once.       #
# --------------------------------------------------------------------------- #
REMOTE_OFFLOADABLE = ("train", "patching", "edges", "acdc", "eap", "edge_pruning",
                      "smallest")
MODAL_APP_NAME = "circuits-pipeline"
MODAL_FN_NAME = "run_stages"
MODAL_VOL_NAME = "circuits-artifacts"
REMOTE_POLL_SECS = 10           # how often to poll spawned remote jobs for completion
HEARTBEAT_SECS = int(os.environ.get("PIPELINE_HEARTBEAT_SECS", "30"))  # overall-progress cadence


ANALYSIS_STAGES = ("patching", "edges", "acdc", "eap", "edge_pruning", "smallest")


def remote_container_stages(offload: list) -> list:
    """The stages a single-path Modal container runs: the offloaded ones plus their
    in-container prerequisites — any analysis needs its dataset (`data`) and the
    trained checkpoint (`train`, gate-skipped when the ckpt already passes), and
    `smallest` needs the finders' candidates, which never ride the image."""
    s = set(offload)
    if "smallest" in s:
        s |= set(canon.DEFAULT_FINDERS)
    if s & set(ANALYSIS_STAGES):
        s |= {"data", "train"}
    return sorted(s, key=STAGE_ORDER.index)


def _modal_fn():
    """Resolve the deployed Modal A100 function, with a friendly error."""
    try:
        import modal
    except ImportError:
        raise SystemExit("✗ --modal needs the `modal` package: `pip install modal`, "
                         "then `modal setup` (once) and `modal deploy "
                         "pipeline/modal_pipeline_app.py`.")
    try:
        return modal.Function.from_name(MODAL_APP_NAME, MODAL_FN_NAME)
    except modal.exception.Error as e:                      # not deployed / not authed
        raise SystemExit(
            f"✗ Modal app '{MODAL_APP_NAME}' not reachable ({e}). Run `modal setup` "
            "then `modal deploy pipeline/modal_pipeline_app.py`.")


def _remote_extra_args(a, container_stages: list) -> str:
    """Flags threaded into the in-container run_model_pipeline invocation."""
    parts = [f"--smallest-target {a.smallest_target}"] if "smallest" in container_stages else []
    if {"acdc", "eap", "edge_pruning"} & set(container_stages):
        parts += (["--zero-ablation"] if a.zero_ablation else []) + (["--use-pos-embed"] if a.use_pos_embed else [])
    if "train" in container_stages and a.train_gate != SEQ_ACC_GATE:
        parts += [f"--train-gate {a.train_gate}"]
    return " ".join(parts)


def spawn_remote(name: str, container_stages: list, a, fn=None, circuit=None,
                 restore_keys: str = "") -> object | None:
    """Kick off the Modal A100 job for ONE model (optionally scoped to ONE ``circuit``)
    WITHOUT blocking, returning a handle to await later. Spawning every job up front
    lets the containers run CONCURRENTLY on separate A100s — Modal autoscales one
    container per ``.spawn()``. ``circuit`` set ⇒ a per-circuit R2 job: the container
    restores the model's checkpoint from the volume tar (image-baked ckpt as fallback),
    runs ``--only-circuit <circuit>``, and ships back only that circuit's subtree.
    Returns None in dry-run (nothing is launched)."""
    clab = "" if circuit is None else f" «{circuit}»"
    oc = [] if circuit is None else ["--only-circuit", circuit]
    extra = _remote_extra_args(a, container_stages)
    if name.partition("__")[2]:   # variant tiers (wide/xl) OOM concurrent 2a children on one A100
        extra = (extra + " --serial-waves").strip()
    if a.dry_run:
        print(f"    $ (Modal A100) run_model_pipeline.py --model {name} --stages "
              f"{' '.join(container_stages)} {' '.join(oc)} {extra}".replace("  ", " ").rstrip()
              + (f"  [restore: {restore_keys}]" if restore_keys else ""), flush=True)
        return None
    fn = fn or _modal_fn()
    call = fn.spawn(name, " ".join(container_stages), circuit or "", extra, restore_keys)
    cid = getattr(call, "object_id", "?")
    print(f"    ⇡ launched REMOTE {name}{clab} ({'+'.join(container_stages)}) on Modal "
          f"A100 [{cid}]", flush=True)
    return call


def sync_remote(name: str, summary, container_stages: list, a, r: "Runner",
                circuit=None) -> int:
    """Record the REMOTE step (ONE pipeline step) for a job whose result `summary`
    is already in hand, and extract its synced artifacts into the local repo.
    `summary is None` is the dry-run case. ``circuit`` set ⇒ a per-circuit job whose
    tar lives under the per-circuit key ``{name}__{circuit}.tar`` (concurrent circuit
    containers never clobber each other). Raises on remote failure."""
    clab = "" if circuit is None else f" «{circuit}»"
    key = name if circuit is None else f"{name}__{circuit}"
    label = f"REMOTE {name}{clab} — {'+'.join(container_stages)} on Modal A100"
    r.begin_step(label)
    if a.dry_run:
        r.tally(label, 0.0, 0)
        return 0
    import io
    import tarfile
    import modal
    vol = modal.Volume.from_name(MODAL_VOL_NAME)
    t0 = time.time()
    rc = int(summary.get("rc", 1))
    if rc == 0:
        # A local read_file() is server-direct (sees the container's commit()), so no
        # reload is needed — Modal >=1.5 forbids reload() outside a container anyway.
        data = b"".join(vol.read_file(f"{key}.tar"))
        with tarfile.open(fileobj=io.BytesIO(data)) as tf:
            tf.extractall(REPO, filter="data")             # restores pipeline/...
    dt = time.time() - t0
    r.tally(label, dt, rc)
    r._done(_short(label), dt, rc)
    if rc != 0:
        print(f"    ✗ REMOTE failed for {name}{clab} (rc {rc}"
              + (f"; {summary.get('err')}" if summary.get("err") else "")
              + ") — see Modal logs.", flush=True)
        return rc
    print(f"    remote ran {summary.get('elapsed_s', '?')}s; artifacts synced locally.",
          flush=True)
    return rc


def poll_completed(remaining: dict) -> tuple | None:
    """Non-blocking: return ``(key, summary)`` for ONE finished call in ``remaining``
    (Modal ``get(timeout=0)`` polls instantly), or ``None`` if none are ready yet.
    A crashed/timed-out container surfaces as a FAILED summary (rc 99), never as an
    exception — one bad container must not abort the batch."""
    import modal.exception as mexc
    for key, call in remaining.items():
        try:
            summary = call.get(timeout=0)
        except (mexc.TimeoutError, TimeoutError):         # not ready yet
            continue
        except Exception as e:                            # noqa: BLE001 — see below
            # The ONLY blanket catch in this file, and deliberate: call.get() re-raises
            # whatever the CONTAINER raised, so the type is arbitrary user code. One bad
            # container must not abort a 100-job batch; the rc-99 summary is the loud part.
            summary = {"rc": 99, "err": f"{type(e).__name__}: {e}"}
        return key, summary
    return None


# --------------------------------------------------------------------------- #
# Execution backends: one job = (name, circuit, phase). The fan-out loop below  #
# is backend-agnostic — spawn a job, poll for one finished, sync its result.    #
#   ModalBackend    one A100 container per job; artifacts via the shared Volume #
#   LocalGpuBackend one local subprocess per job pinned to a free GPU; N-wide;  #
#                   artifacts are already on local disk so sync is a no-op.     #
# --------------------------------------------------------------------------- #
_GPU_JOB_LOGS = os.path.join(tempfile.gettempdir(), "circuits_gpu_jobs")


class _ModalBackend:
    label = "Modal A100"

    def __init__(self, a) -> None:
        self.fn = None if a.dry_run else _modal_fn()

    def spawn(self, key, stages, a, circuit=None, restore_keys=""):
        return spawn_remote(key[0], stages, a, self.fn, circuit=circuit,
                            restore_keys=restore_keys)

    def poll(self, remaining):
        return poll_completed(remaining)

    def sync(self, name, summary, stages, a, r, circuit=None) -> int:
        return sync_remote(name, summary, stages, a, r, circuit=circuit)


class _LocalGpuBackend:
    """--local --gpus N: run each job as `run_model_pipeline … --local --only-circuit`
    pinned to a free GPU (CUDA_VISIBLE_DEVICES), N jobs at a time. Job logs stream to
    _GPU_JOB_LOGS/<key>.log so no PIPE can deadlock; the tail is the fail-loud err."""
    label = "local GPU"

    def __init__(self, a) -> None:
        self.free = list(range(a.gpus))
        self.queued = []                 # [(key, stages, circuit, extra)]
        self.running = {}                # key -> {proc, gpu, t0, log, path}

    def spawn(self, key, stages, a, circuit=None, restore_keys=""):
        stages = list(stages)
        if a.dry_run:
            oc = f" --only-circuit {circuit}" if circuit else ""
            print(f"    $ (local GPU) run_model_pipeline.py --model {key[0]} "
                  f"--stages {' '.join(stages)}{oc} --local".rstrip(), flush=True)
            return None
        self.queued.append((key, stages, circuit, _remote_extra_args(a, stages)))
        self._dispatch()
        return key

    def _dispatch(self) -> None:
        os.makedirs(_GPU_JOB_LOGS, exist_ok=True)
        while self.free and self.queued:
            key, stages, circuit, extra = self.queued.pop(0)
            gpu = min(self.free)                       # lowest free GPU (deterministic packing)
            self.free.remove(gpu)
            cmd = [sys.executable, os.path.join(PIPELINE, "run_model_pipeline.py"),
                   "--model", key[0], "--stages", *stages, "--device", "auto", "--local"]
            if circuit:
                cmd += ["--only-circuit", circuit]
            if key[0].partition("__")[2]:              # variant tiers OOM concurrent 2a children
                extra = (extra + " --serial-waves").strip()
            if extra:
                cmd += extra.split()
            path = os.path.join(_GPU_JOB_LOGS,
                                f"{key[0]}{('__' + circuit) if circuit else ''}.{key[2]}.log")
            log = open(path, "w")
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
            proc = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=log,
                                    stderr=subprocess.STDOUT, text=True)
            self.running[key] = {"proc": proc, "gpu": gpu, "t0": time.time(),
                                 "log": log, "path": path}
            clab = "" if circuit is None else f" «{circuit}»"
            print(f"    ⇡ launched LOCAL {key[0]}{clab} ({'+'.join(stages)}) on GPU {gpu} "
                  f"[{path}]", flush=True)

    def poll(self, remaining):
        self._dispatch()                               # fill any GPU freed last round
        for key, st in list(self.running.items()):
            rc = st["proc"].poll()
            if rc is None:
                continue
            st["log"].close()
            self.running.pop(key)
            self.free.append(st["gpu"])
            summary = {"rc": rc, "elapsed_s": round(time.time() - st["t0"], 1)}
            if rc != 0:
                with open(st["path"]) as f:
                    summary["err"] = "".join(f.readlines()[-40:])[-1500:]
            return key, summary
        return None

    def sync(self, name, summary, stages, a, r, circuit=None) -> int:
        clab = "" if circuit is None else f" «{circuit}»"
        label = f"LOCAL {name}{clab} — {'+'.join(stages)} on GPU"
        r.begin_step(label)
        rc = int(summary.get("rc", 1))
        dt = float(summary.get("elapsed_s", 0.0))
        r.tally(label, dt, rc)
        r._done(_short(label), dt, rc)
        if rc != 0:
            print(f"    ✗ LOCAL failed for {name}{clab} (rc {rc}"
                  + (f"; {summary.get('err')}" if summary.get("err") else "") + ")", flush=True)
        else:
            print(f"    local job ran {dt}s on GPU; artifacts already local.", flush=True)
        return rc


def _make_backend(a):
    return _LocalGpuBackend(a) if (a.local and a.gpus > 1) else _ModalBackend(a)


def heartbeat_loop(r: "Runner", stop: "threading.Event") -> None:
    """Emit an overall-progress line every ``HEARTBEAT_SECS`` until ``stop`` is set.
    Rendered with tqdm's own meter (so it's a real tqdm bar) but newline-terminated
    and tagged ``⏳ PROGRESS`` so it's readable in a backgrounded log and easy to
    grep/relay. Crucially this ALSO ticks during the remote train/edge_pruning phase
    — when the per-stage subprocess tqdm bars live in Modal's container logs and the
    local log would otherwise be silent for minutes."""
    from tqdm import tqdm
    while not stop.wait(HEARTBEAT_SECS):
        now = time.time()
        elapsed = now - r.start_t0
        cur_dt = (now - r.cur_t0) if r.cur_t0 else 0.0
        meter = tqdm.format_meter(r.idx, r.total, elapsed, ncols=60, prefix="steps")
        print(f"⏳ PROGRESS {meter} · now: {r.cur_label or '—'} ({cur_dt:.0f}s)", flush=True)


def execute(name: str, stages: list, a, r: Runner) -> int:
    """Run all selected stages for one model. Returns audit gap count (0 if no audit)."""
    gaps = 0
    steps = build_steps(name, stages, a)
    only_audit = stages == ["audit"]
    for kind, label, jobs in steps:
        r.begin_step(_step_tag(kind, label, jobs))
        if not a.dry_run:
            _headroom(f"before {label}", verbose=a.verbose)
        if kind == "audit":
            if a.dry_run:
                if only_audit:
                    t0 = time.time()
                    run_audit(name, a, enumerate_only=True)   # imports model_registry
                    r.tally(label + " (enumeration)", time.time() - t0, 0)
                else:
                    print("    in-process per-circuit existence audit "
                          "(nonzero exit on any gap).", flush=True)
                    print("    (full enumeration: `--stages audit --dry-run`)", flush=True)
                    r.tally(label + " (plan only)", 0.0, 0)
            else:
                t0 = time.time()
                g = run_audit(name, a)
                r.tally(label, time.time() - t0, 1 if g else 0)
                gaps += g
            continue
        if kind == "train":
            run_train(name, a, r)
            continue
        if kind == "gated_smallest":     # suppression row: verdict recorded, never gates
            circuit, sjobs = jobs
            if a.dry_run:
                print("    (router verdict from diagnostic.json's (cued,wrong) pair is "
                      "echoed via --router-verdict; the row always runs)", flush=True)
            verdict = None if a.dry_run else suppression_verdict(name, circuit)
            echo = [] if verdict is None else ["--router-verdict", verdict]
            for jlabel, argv, needs_ckpt in sjobs:
                env = {"SEGREV3_CKPT": resolve_ckpt(name, a)} if needs_ckpt else None
                r.run(jlabel, argv + echo, env_extra=env)
            continue
        if kind == "parallel":
            specs = []
            for jlabel, argv, needs_ckpt in jobs:
                env = {"SEGREV3_CKPT": resolve_ckpt(name, a)} if needs_ckpt else None
                specs.append((jlabel, argv, env))
            r.run_parallel(label, specs)
            continue
        # serial
        for jlabel, argv, needs_ckpt in jobs:
            env = {"SEGREV3_CKPT": resolve_ckpt(name, a)} if needs_ckpt else None
            r.run(jlabel, argv, env_extra=env)
    return gaps


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #
def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        prog="run_model_pipeline.py",
        description="One-command orchestrator for pipeline (subprocess "
                    "stages, wave/GPU-slot schedule, per-circuit tree audit).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="examples:\n"
               "  run_model_pipeline.py --model rcnt3_id --stages all --dry-run\n"
               "  run_model_pipeline.py --model rcnt3_id --stages audit --dry-run\n"
               "  run_model_pipeline.py --model rcnt3_id --stages all --smoke\n"
               "  run_model_pipeline.py --model all --stages audit\n")
    ap.add_argument("--model", nargs="+", metavar="MODEL",
                    help="one or more model names from models/models.txt "
                         "(space-separated), or 'all' for every catalogued model")
    ap.add_argument("--stages", nargs="+", metavar="STAGE",
                    choices=list(STAGE_ORDER) + ["all"],
                    help="stage word(s): " + " | ".join(STAGE_ORDER) + " | all")
    ap.add_argument("--print-hparams", action="store_true",
                    help="print the resolved per-architecture HP table (train_models/hparams.py) "
                         "and exit — the recipe BOTH training paths read")
    ap.add_argument("--print-c3-worklist", action="store_true",
                    help="print the C3 suppression fan-out work-list (models x circuits with a "
                         "checkpoint AND a router diagnostic) and exit; runs nothing")
    ap.add_argument("--smoke", action="store_true",
                    help="forward --smoke to every child that supports it; train "
                         "goes to a THROWAWAY ckpt under .smoke/; acdc "
                         "shrinks to 1 tau")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the exact commands per stage (with env) without "
                         "executing anything")
    ap.add_argument("--verbose", action="store_true",
                    help="verbose console output: echo each child's full command "
                         "line (env + argv) and the per-stage memory-headroom line. "
                         "Off by default — the orchestrator prints one compact "
                         "▶ [k/N] line per stage. (--dry-run always shows commands.)")
    ap.add_argument("--serial-waves", action="store_true",
                    help="emit the parallel analysis wave (2a patching∥edges∥"
                         "acdc∥eap) as SERIAL single-GPU-process steps (run one "
                         "after another) instead of parallel groups, so each model "
                         "uses only 1 GPU process. Lets a higher-level driver run "
                         "several MODELS concurrently. smallest (2c) is already "
                         "serial/alone. Default: off (parallel, up to GPU_PROC_CAP procs/wave).")
    ap.add_argument("--device", default="auto",
                    help="compute device for every child via SEGREV3_DEVICE "
                         "(default: %(default)s)")
    ap.add_argument("--modal", action="store_true",
                    help="opt-in: run all GPU stages on Modal containers — one per "
                         "single-circuit model, train + one-per-circuit for multi-circuit "
                         "models (needs `pip install modal` and a deployed "
                         "modal_pipeline_app.py; ignored under --smoke). Default: local.")
    ap.add_argument("--local", action="store_true",
                    help="no-op alias kept for backward compatibility (local is the default)")
    ap.add_argument("--only-circuit", default=None, metavar="SLUG",
                    help="restrict EVERY per-circuit stage to this single circuit "
                         "slug. Used by the per-circuit Modal fan-out so each A100 "
                         "container handles one circuit; rarely needed by hand.")
    ap.add_argument("--smallest-target", default=SMALLEST_TARGET,
                    help="ABSOLUTE whole-sequence accuracy floor for the smallest "
                         "stage (default: %(default)s; the driver wires the user's "
                         "accuracy threshold X here)")
    ap.add_argument("--zero-ablation", action="store_true",
                    help="C4 positional (finder-only): corrupt with zeros instead of the patch "
                         "(_zero candidate suffix; drops smallest, positional-aware audit)")
    ap.add_argument("--use-pos-embed", action="store_true",
                    help="C4 positional (finder-only): split the positional embedding as its own "
                         "source (_pos candidate suffix; drops smallest, positional-aware audit)")
    ap.add_argument("--train-backend", choices=("sh", "stacked"), default="sh",
                    help="sh = per-model run_training_job.sh; stacked = vmap-batch train_sharded "
                         "for the whole model set, sub-gate stragglers fall back to sh")
    ap.add_argument("--train-gate", type=float, default=SEQ_ACC_GATE,
                    help="seq-acc gate to skip retrain (default %(default)s); lower it to reuse "
                         "existing sub-gate checkpoints as-is instead of retraining them")
    ap.add_argument("--gpus", type=int, default=1,
                    help="fan the per-circuit job DAG across N local GPUs, one "
                         "job per GPU (1 = serial; >1 = parallel scheduler, e.g. a multi-GPU VM)")
    ap.add_argument("--seeds", type=int, default=1,
                    help="train N seeds per model (the reproducibility axis; the model zoo was "
                         "trained at n=1, and circuit winners can change on refit). "
                         "Seed 0 is the params-of-record; extras land in <name>.seed<k>.pt")
    a = ap.parse_args(argv)
    if a.print_hparams or a.print_c3_worklist:
        return a
    if not (a.model and a.stages):
        ap.error("--model and --stages are required (or use --print-hparams / --print-c3-worklist)")

    stages = []
    for s in a.stages:
        for x in (list(DEFAULT_STAGES) if s == "all" else [s]):
            if x not in stages:
                stages.append(x)
    a.stages = sorted(stages, key=STAGE_ORDER.index)
    if _positional_suffix(a) and "smallest" in a.stages:      # C4 positional analysis is finder-only
        a.stages = [s for s in a.stages if s != "smallest"]
        print("  [C4 positional] finder-only: dropped the smallest stage (positional "
              "candidates are not output-metric comparable)", flush=True)
    a.local = a.smoke or not a.modal         # LOCAL is the default; smoke is always local
    return a


def _ensure_venv() -> None:
    """Re-exec under ./.venv when launched with a bare system python (the
    circuit_discovery/main.py:389 pattern — in_venv check is the loop guard)."""
    venv_py = os.path.join(REPO, ".venv", "bin", "python")
    in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    if not in_venv and os.path.exists(venv_py):
        print(f"(re-exec under {os.path.relpath(venv_py, os.getcwd())})", flush=True)
        os.execv(venv_py, [venv_py, *sys.argv])


def main(argv=None) -> None:
    _ensure_venv()
    a = parse_args(argv)

    known = list_models()
    if a.print_hparams:
        hparams.print_table()
        return
    if a.print_c3_worklist:
        print_c3_worklist(c3_universe() if (not a.model or "all" in a.model) else a.model)
        return
    # --model takes one or more names (nargs="+"); 'all' (anywhere in the list)
    # expands to every catalogued model. Otherwise keep the given order, de-duped.
    requested = a.model if isinstance(a.model, list) else [a.model]
    if "all" in requested:
        models = known
    else:
        models = list(dict.fromkeys(requested))     # de-dupe, preserve order
        for m in models:
            if m not in known:
                # not in the catalog, but may be a valid DSL algorithm_id run
                # directly (e.g. a conditional model like segrev3_id). get_spec
                # validates it at execution time (_verify_conventions); --dry-run
                # is advisory.
                print(f"  (note: {m!r} is not in the catalog — treating it as a "
                      f"direct DSL algorithm_id; validated at execution)", flush=True)

    # ONE device for every child stage (resolve_device reads SEGREV3_DEVICE) —
    # the main.py:419 propagation pattern.
    os.environ["SEGREV3_DEVICE"] = a.device

    # DEFAULT: every stage runs locally; --modal offloads heavy stages to Modal. --gpus N>1
    # fans the SAME job DAG across N local GPUs instead (e.g. a multi-GPU VM).
    if a.gpus > 1 and not a.local:
        raise SystemExit("✗ --gpus N>1 fans jobs across LOCAL GPUs; drop --modal (the "
                         "multi-GPU path). Modal autoscales one GPU per job on its own.")
    remote_mode = not a.local
    local_parallel = a.local and a.gpus > 1
    offload = ([s for s in a.stages if s in REMOTE_OFFLOADABLE]
               if (remote_mode or local_parallel) else [])
    container_stages = remote_container_stages(offload) if offload else []
    local_stages = [s for s in a.stages if s not in offload]
    if offload:                    # data runs in the job (ships back on Modal); audit stays local
        local_stages = [s for s in local_stages if s != "data"]

    # Fan-out plan: a single-circuit model runs EVERYTHING in one container (container
    # overhead is first-order at these job sizes); a multi-circuit model trains once
    # (R1 = train only) then fans one container per circuit (R2 = data + analysis +
    # smallest), each restoring R1's checkpoint from the volume (image ckpt fallback).
    R1_STAGES = ["train"]
    R2_STAGES = [x for x in remote_container_stages(offload) if x != "train"]
    plan = {}                          # model -> ("single", [None]) | ("fanout", [circuit,...])
    if offload:
        analysis_offloaded = bool(set(offload) & set(ANALYSIS_STAGES))
        for m in models:
            circs = model_circuits(m)
            plan[m] = (("fanout", circs) if (analysis_offloaded and len(circs) > 1)
                       else ("single", [None]))

    # k/N denominator = local stage steps + remote steps (1 per single model;
    # 1 train + 1 per circuit for a fanned-out model — no R1 when train isn't requested).
    total_steps = sum(len(build_steps(m, local_stages, a)) for m in models)
    if offload:
        r1 = 1 if "train" in offload else 0
        total_steps += sum(1 if plan[m][0] == "single" else r1 + len(plan[m][1])
                           for m in models)

    mode = []
    if offload:
        mode.append("remote:" + "+".join(container_stages))
    elif remote_mode:
        mode.append("remote (no offloadable stage requested)")
    else:
        mode.append("LOCAL")
    if a.dry_run:
        mode.append("DRY RUN")
    if a.smoke:
        mode.append("smoke")
    if a.serial_waves:
        mode.append("serial-waves")
    if a.verbose:
        mode.append("verbose")
    print("pipeline · run_model_pipeline")
    print(f"  models {models}  ·  stages {' → '.join(a.stages)}")
    print(f"  device {a.device}  ·  {total_steps} stage step(s)"
          + (f"  ·  {', '.join(mode)}" if mode else ""), flush=True)

    if a.smoke and not a.dry_run:
        os.makedirs(SMOKE_DIR, exist_ok=True)
    if not a.dry_run:
        for m in models:
            _verify_conventions(m)      # conventional paths == model_registry catalog

    if a.train_backend == "stacked" and "train" in a.stages:
        if remote_mode:
            raise SystemExit("✗ --train-backend stacked is local-only (it packs models on ONE "
                             "GPU); for Modal multi-GPU run train_sharded.py directly.")
        if not a.dry_run and not a.smoke:
            _stacked_pretrain(models, a)   # keepers promoted; per-model train stage handles stragglers

    r = Runner(dry_run=a.dry_run, verbose=a.verbose, total=total_steps)
    t0 = time.time()
    r.start_t0 = t0
    gaps = 0
    failures = []                    # remote (name, circuit, rc) after the one retry
    # heartbeat: a frequent ⏳ PROGRESS line so a backgrounded run is NEVER silent —
    # it ticks even through the remote phase (whose bars live in Modal's logs). Daemon
    # thread: dies with the process; stopped cleanly before the final summary.
    _hb_stop = threading.Event()
    if not a.dry_run and HEARTBEAT_SECS > 0:
        threading.Thread(target=heartbeat_loop, args=(r, _hb_stop), daemon=True).start()
    # --modal: launch the offloaded GPU work on Modal A100s, parallel on TWO axes,
    # then run each model's LOCAL stages as soon as ITS remote work finishes — in
    # COMPLETION order, not submission order.
    #   • ACROSS MODELS: every model's first remote job is spawned up front (Modal
    #     autoscales one A100 per spawn), so an early finisher starts its local
    #     analysis immediately, overlapping slower models' still-running remotes.
    #   • ACROSS CIRCUITS: for a fanned-out model, training runs ONCE (phase R1 =
    #     data+train); the instant it lands, each circuit's edge_pruning fans out to
    #     its OWN A100 (phase R2), all restoring R1's single checkpoint from the
    #     volume — so per-circuit EP runs concurrently with no retrain or drift.
    # The local stages still serialize on the one local GPU (the GPU_PROC_CAP rule): a
    # model's locals run once ALL its remote jobs (the single job, or train + every
    # circuit's EP) have synced.
    if not offload:
        for m in models:
            gaps += execute(m, local_stages, a, r)
    elif a.dry_run:
        backend = _make_backend(a)
        for m in models:                                 # advisory: print the offload plan
            kind, circs = plan[m]
            if kind == "single":
                backend.spawn((m, None, "full"), container_stages, a)
            else:
                if "train" in offload:
                    backend.spawn((m, None, "train"), R1_STAGES, a)
                for c in circs:
                    backend.spawn((m, c, "circuit"), R2_STAGES, a, circuit=c,
                                  restore_keys=restore_keys_for(m, c))
            gaps += execute(m, local_stages, a, r)
    else:
        backend = _make_backend(a)
        n_fan = sum(1 for k, _ in plan.values() if k == "fanout")
        print(f"\n⇡ launching {backend.label} work — {len(models)} model(s)"
              + (f", per-circuit edge_pruning fan-out on {n_fan}" if n_fan else "")
              + " …", flush=True)
        # job-key = (name, circuit, phase); phase ∈ {full, train, circuit}
        remaining, meta = {}, {}
        pending_circ = {}                # model -> set of circuits whose R2 isn't synced
        pending_base = {}                # model -> non-contrast circuits not yet synced
        deferred = {}                    # model -> contrast circuits awaiting their bases

        def _spawn(key, stages_ran, circuit=None, retried=False):
            remaining[key] = backend.spawn(key, stages_ran, a, circuit=circuit,
                                           restore_keys=restore_keys_for(key[0], circuit))
            meta[key] = (stages_ran, circuit, retried)

        def _fan_out(m):
            """Wave A now; contrast circuits wait for every base to sync (two-phase)."""
            waveA = [c for c in plan[m][1] if contrast_base(m, c) is None]
            deferred[m] = [c for c in plan[m][1] if contrast_base(m, c) is not None]
            pending_base[m] = set(waveA)
            for c in waveA or deferred.pop(m, []):     # no bases at all -> volume tars from a prior run
                _spawn((m, c, "circuit"), R2_STAGES, circuit=c)

        for m in models:
            kind, circs = plan[m]
            if kind == "single":
                _spawn((m, None, "full"), container_stages)
            elif "train" in offload:
                pending_circ[m] = set(circs)
                _spawn((m, None, "train"), R1_STAGES)
            else:                        # analysis-only: fan out now (ckpt from image/volume)
                pending_circ[m] = set(circs)
                _fan_out(m)
        while remaining:
            picked = backend.poll(remaining)
            if picked is None:                           # none ready yet — wait
                time.sleep(REMOTE_POLL_SECS)
                continue
            key, summary = picked
            remaining.pop(key)
            stages_ran, circuit, retried = meta.pop(key)
            name, _, phase = key
            rc = backend.sync(name, summary, stages_ran, a, r, circuit=circuit)
            if rc != 0:
                if not retried:                          # ONE respawn retry per job
                    print(f"    ↻ retrying {name}"
                          + (f" «{circuit}»" if circuit else "") + " once", flush=True)
                    _spawn(key, stages_ran, circuit=circuit, retried=True)
                else:
                    failures.append((name, circuit, rc))
                continue
            if phase == "full":
                gaps += execute(name, local_stages, a, r)
            elif phase == "train":                       # ckpt in volume+local — fan out R2
                _fan_out(name)
            else:                                        # phase == "circuit"
                pending_circ[name].discard(circuit)
                pending_base.get(name, set()).discard(circuit)
                if not pending_base.get(name) and deferred.get(name):
                    for c in deferred.pop(name):         # bases synced -> contrast wave
                        _spawn((name, c, "circuit"), R2_STAGES, circuit=c)
                if not pending_circ[name]:               # all circuits synced
                    gaps += execute(name, local_stages, a, r)

    _hb_stop.set()                                  # stop the heartbeat before the summary
    ok = sum(1 for _, _, rc in r.ran if rc == 0)
    print(f"\n{'─' * 56}")
    print(f"PIPELINE COMPLETE — {time.time() - t0:.0f}s"
          + ("  [DRY RUN: nothing executed]" if a.dry_run else ""))
    print(f"{ok}/{len(r.ran)} stages succeeded:")
    for label, dt, rc in r.ran:
        print(f"  {'✓' if rc == 0 else '✗'} {_short(label):<24} {dt:>4.0f}s")
    if failures:
        print(f"✗ {len(failures)} remote job(s) failed after retry:")
        for name, circuit, rc in failures:
            print(f"    {name}" + (f" «{circuit}»" if circuit else "") + f"  (rc {rc})")
    stage_fails = len(r.ran) - ok
    if gaps or failures or stage_fails:
        raise SystemExit(f"✗ {stage_fails} stage failure(s), {len(failures)} remote "
                         f"failure(s), {gaps} audit gap(s).")


if __name__ == "__main__":
    main()
