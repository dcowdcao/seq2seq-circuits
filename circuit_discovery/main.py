#!/usr/bin/env python
"""
main.py
=================================================================================
ONE entry point that runs the WHOLE segrev3 circuit-discovery pipeline. You name
the STAGES to run as positional words; they execute in canonical order and the
always-on tails (per-circuit validation -> circuit-scoped figures -> figures.zip)
follow automatically after discovery.

STAGE WORDS
    all        data + manual + acdc + eap + figs  (the full pipeline)
    data       generate the central datasets/ store (data_gen/generate_datasets.py)
    manual     manual discovery pipeline (manual_circuit.py)
    acdc       ACDC recursive edge pruning (acdc.py)
    eap        EAP-IG edge attribution (eap.py)
    figs       circuit-scoped figures for --circuit (manual_circuit.py --circuit)
    validate   validation for --circuit (validation/validation.py)
    single     SINGLE-SEQUENCE mode: manual analyses + scoped figures on ONE
               custom clean/corrupted pair (runs alone; ACDC/EAP discovery
               excluded; the scoped figures use --circuit, default rough)
    zip        bundle every .png into figures.zip (collect_figures.py)

CIRCUIT SELECTION (figs / validate): one ``--circuit`` selector --
``all`` | a method (``manual``/``acdc``/``eap``) | a circuit folder name
(``rough`` | ``acdc_thresh_<tau>`` | ``eap_top<n>``; the method is inferred).

Canonical sweep parameters live in data_gen/canon.py (single source of truth,
shared with the standalone runners -- a bare runner and an orchestrator run can
no longer disagree).

--------------------------------------------------------------------------------
EXAMPLES (run from the repo root; venv: ``source .venv/bin/activate``)

  python circuit_discovery/main.py all                       # everything, canonical
  python circuit_discovery/main.py data acdc eap --parallel  # data, then both edge methods concurrently
  python circuit_discovery/main.py manual --examples 256     # one method (+ its tails)
  python circuit_discovery/main.py figs --scope all --scheme all          # re-scope EVERY circuit
  python circuit_discovery/main.py figs --circuit eap_top200 --scope ablated
  python circuit_discovery/main.py validate --circuit all --mode resample
  python circuit_discovery/main.py single --n-threes 3 --scheme replace_threes_to_nonthrees
  python circuit_discovery/main.py all --dry-run             # print the plan only

Run with NO stage to print the common recipes.

This is a thin, deterministic ORCHESTRATOR: every stage is the existing,
self-contained script, launched as a subprocess (the established pattern from
manual_circuit.py) with the package dirs on PYTHONPATH so imports resolve
regardless of cwd. It adds no analysis of its own.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

_THIS = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(_THIS)

# canon.py is pure stdlib (constants + shared CLI fragments) -- importing it does
# NOT pull torch into this thin orchestrator.
sys.path.insert(0, os.path.join(REPO, "data_gen"))
import canon  # noqa: E402

# Canonical discovery order (manual first so its rough circuit exists before the
# edge methods; also the order the docs use).
METHOD_ORDER = ["manual", "acdc", "eap"]

#: positional stage words, in execution order
STAGE_WORDS = ("all", "data", "manual", "acdc", "eap", "figs", "validate",
               "single", "zip")

RECIPES = """\
common recipes:
  main.py all                                          # full pipeline: data -> 3 methods -> validate -> figs -> zip
  main.py data acdc eap --parallel                     # regenerate data, both edge methods concurrently
  main.py manual --examples 256                        # one discovery method (+ its tails)
  main.py figs --scope all --scheme all                # re-scope EVERY discovered circuit
  main.py figs --circuit eap_top200 --scope ablated    # scoped figures for ONE circuit
  main.py validate --circuit all --mode resample       # revalidate every circuit
  main.py single --n-threes 3 --scheme replace_threes_to_nonthrees   # ONE custom pair
  main.py zip                                          # rebundle figures.zip

see --help for every option, docs/segrev3_runbook.md for the full reference.
"""

# Package dirs that every child stage must be able to import (mirrors
# manual_circuit.py).
_LIBS = [
    "interp_tools", "data_gen", "metrics", "validation",
    os.path.join("visualizations", "manual_circuit_analysis"),
]


# --------------------------------------------------------------------------- #
# Subprocess plumbing                                                          #
# --------------------------------------------------------------------------- #
def _child_env() -> dict:
    env = dict(os.environ)
    extra = os.pathsep.join(os.path.join(REPO, p) for p in _LIBS)
    env["PYTHONPATH"] = extra + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return env


def _rel(p: str) -> str:
    return os.path.relpath(p, REPO) if isinstance(p, str) and p.startswith(REPO) else p


def _banner(title: str) -> None:
    print(f"\n{'=' * 78}\n▶ {title}\n{'=' * 78}", flush=True)


class Runner:
    """Runs (or, in dry-run, just prints) the pipeline stages and tallies them."""

    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self.py = sys.executable
        self.ran: list = []          # (label, seconds, returncode)

    def run(self, label: str, argv: list, allow_fail: bool = False) -> int:
        _banner(label)
        printable = [self.py if a == self.py else _rel(a) for a in argv]
        print("  $ " + " ".join(str(a) for a in printable), flush=True)
        if self.dry_run:
            self.ran.append((label, 0.0, 0))
            return 0
        t0 = time.time()
        rc = subprocess.run(argv, cwd=REPO, env=_child_env()).returncode
        dt = time.time() - t0
        self.ran.append((label, dt, rc))
        if rc != 0:
            msg = f"✗ STAGE FAILED: {label}  (exit {rc}, {dt:.0f}s)"
            if allow_fail:
                print(msg + "  — continuing.", flush=True)
            else:
                raise SystemExit(msg)
        else:
            print(f"✓ {label}  ({dt:.0f}s)", flush=True)
        return rc

    def py_script(self, label, script_relpath, *args, allow_fail=False):
        return self.run(label, [self.py, os.path.join(REPO, script_relpath), *map(str, args)],
                        allow_fail=allow_fail)

    def run_parallel(self, specs: list) -> None:
        """Run several ``(label, argv, allow_fail)`` jobs CONCURRENTLY. Each child's
        BLAS/OMP thread count is capped to ``cpu_count // njobs`` so the jobs share
        the cores instead of oversubscribing them. Output is captured per job and
        printed (in submission order) once that job finishes, keeping logs readable;
        stages are tallied exactly like the sequential path."""
        import concurrent.futures
        n = len(specs)
        threads = max(1, (os.cpu_count() or 2) // max(1, n))
        env = _child_env()
        for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                  "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
            env[v] = str(threads)
        _banner(f"PARALLEL — {n} jobs concurrently, {threads} CPU threads each")
        for label, argv, _ in specs:
            print(f"  ⟫ {label}\n    $ "
                  + " ".join(self.py if a == self.py else str(_rel(a)) for a in argv), flush=True)
        if self.dry_run:
            self.ran.extend((label, 0.0, 0) for label, _, _ in specs)
            return

        def _one(spec):
            label, argv, allow_fail = spec
            t0 = time.time()
            p = subprocess.run(argv, cwd=REPO, env=env, capture_output=True, text=True)
            return label, time.time() - t0, p.returncode, p.stdout, p.stderr, allow_fail

        with concurrent.futures.ThreadPoolExecutor(max_workers=n) as ex:
            results = list(ex.map(_one, specs))   # preserves submission order

        hard_fail = None
        for label, dt, rc, out, err, allow_fail in results:
            _banner(f"[finished] {label}  ({dt:.0f}s, exit {rc})")
            if out:
                print(out, end="" if out.endswith("\n") else "\n", flush=True)
            if err:
                print(err, end="" if err.endswith("\n") else "\n", flush=True)
            self.ran.append((label, dt, rc))
            if rc != 0:
                if allow_fail:
                    print(f"✗ {label} exited {rc} — continuing.", flush=True)
                elif hard_fail is None:
                    hard_fail = (label, rc)
        if hard_fail:
            raise SystemExit(f"✗ STAGE FAILED: {hard_fail[0]}  (exit {hard_fail[1]})")


# --------------------------------------------------------------------------- #
# Stages                                                                       #
# --------------------------------------------------------------------------- #
def stage_data_gen(r: Runner, apps: list, n, seed: int) -> None:
    args = ["--apps", *apps, "--seed", str(seed)]
    if n is not None:
        args += ["-n", str(n)]
    r.py_script("DATA GENERATION — central datasets/ store",
                os.path.join("data_gen", "generate_datasets.py"), *args)


def discover_spec(a, method: str, data_gen_ran: bool):
    """Return ``(label, argv, allow_fail)`` for ``method``'s discovery command WITHOUT
    running it, so the caller can run methods sequentially or concurrently (--parallel)."""
    py = sys.executable
    sp = lambda rel: os.path.join(REPO, "circuit_discovery", rel)
    if method == "manual":
        if a.single:
            return ("DISCOVERY: manual — SINGLE-SEQUENCE patching/DLA + edge analysis",
                    [py, sp("manual_circuit.py"), *canon.single_argv(a)], False)
        examples = a.examples if a.examples is not None else canon.CANON_MANUAL_EXAMPLES
        args = ["-n", str(examples), "--seed", str(a.seed)]
        # If we just (re)generated the central store, reuse datasets/manual/ rather
        # than letting manual_circuit.py regenerate its own (keeps ONE source of truth).
        if data_gen_ran:
            args.append("--skip-data")
        return ("DISCOVERY: manual — data → analysis → full-model validation",
                [py, sp("manual_circuit.py"), *args], False)
    if method == "acdc":
        return ("DISCOVERY: ACDC — recursive edge pruning (KL over positions 9..18)",
                [py, sp("acdc.py"), "--thresholds", *map(str, a.thresholds),
                 "-n", str(a.acdc_examples), "--seed", str(a.seed),
                 "--device", a.device], False)
    if method == "eap":
        return ("DISCOVERY: EAP-IG — integrated-gradient edge attribution",
                [py, sp("eap.py"), "--ig-steps", str(a.ig_steps), "--export-n", str(a.export_n),
                 "--topns", *map(str, a.topns), "--seed", str(a.seed)], False)
    raise ValueError(f"unknown method {method!r}")


def stage_validate(r: Runner, method: str, mode: str, name: str = "all") -> None:
    if method == "manual" and name == "all":
        # The manual pipeline already ran the full-model per-head validation
        # (inside manual_circuit.py). Nothing extra to do.
        _banner("VALIDATION: manual — done inside manual_circuit.py (full model, all 32 heads)")
        print("  (per-head ablation drops already written under "
              "validation/figs_data/.../manual/)", flush=True)
        return
    r.py_script(f"VALIDATION: {method}/{name} — faithfulness / completeness / minimality",
                os.path.join("validation", "validation.py"),
                "--method", method, "--name", name, "--mode", mode,
                allow_fail=True)


def stage_scoped_figs(r: Runner, circuit: str, scope: str, scheme: str,
                      extra: list = ()) -> None:
    # manual_circuit.py --circuit also runs per-circuit validation; main.py already
    # runs validation as its own stage (so --no-validate / --no-figures stay
    # independent), so we pass --no-validate here to avoid validating twice.
    # manual_circuit's --scope is full_circuit|ablated only -> expand "all" ourselves.
    # ``circuit`` is the canon.parse_circuit selector (all | method | folder name);
    # ``extra`` appends extra CLI args (the single-sequence flags in single mode,
    # which already carry --scheme -- don't pass it twice).
    scopes = ["full_circuit", "ablated"] if scope == "all" else [scope]
    scheme_args = [] if "--scheme" in extra else ["--scheme", scheme]
    for sc in scopes:
        r.py_script(f"SCOPED FIGURES: {circuit} — patching / DLA / QK·OV (scope={sc}, scheme={scheme})",
                    os.path.join("circuit_discovery", "manual_circuit.py"),
                    "--circuit", circuit, "--scope", sc, *scheme_args,
                    "--no-validate", *extra, allow_fail=True)


def stage_collect(r: Runner) -> None:
    r.py_script("COLLECT FIGURES — bundle every .png into figures.zip",
                os.path.join("visualizations", "collect_figures.py"), allow_fail=True)


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #
def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        prog="main.py",
        description="Run the segrev3 circuit-discovery pipeline. Name the stages to "
                    "run as positional words (executed in canonical order); the "
                    "always-on tails (validate -> figs -> zip) follow discovery "
                    "automatically.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=RECIPES)

    ap.add_argument("stages", nargs="*", metavar="STAGE",
                    help="stage word(s): all | data | manual | acdc | eap | figs | "
                         "validate | single | zip. 'all' = data + the 3 methods + "
                         "figs. With no stage, the common recipes are printed.")

    g = ap.add_argument_group("global options")
    g.add_argument("-n", "--examples", type=int, default=None,
                   help="examples per generated dataset (data + manual stages; "
                        "default: per-app canonical sizes).")
    g.add_argument("--seed", type=int, default=canon.DEFAULT_SEED)
    g.add_argument("--device", default="auto",
                   help="compute device for every stage: 'auto' (default; cuda when "
                        "available, else cpu), 'cuda', or 'cpu'. Exported to all child "
                        "stages via SEGREV3_DEVICE so the whole pipeline shares one device.")
    g.add_argument("--model", default=None, metavar="CKPT",
                   help="model checkpoint path for every stage (default: the canonical "
                        "segrev3 checkpoint). Exported to all child stages via "
                        "SEGREV3_CKPT so the whole pipeline loads one model.")
    g.add_argument("--parallel", action="store_true",
                   help="run the selected DISCOVERY methods CONCURRENTLY (e.g. ACDC + "
                        "EAP at the same time), capping each process to cpu_count//njobs "
                        "CPU threads. Validation/figure stages stay sequential.")
    g.add_argument("--dry-run", action="store_true",
                   help="print the stage plan without executing anything.")

    t = ap.add_argument_group("always-on tails after discovery (opt out)")
    t.add_argument("--no-validate", action="store_true",
                   help="skip the automatic per-circuit validation.")
    t.add_argument("--no-figures", "--no-viz", dest="no_figures", action="store_true",
                   help="skip the automatic circuit-scoped figures.")
    t.add_argument("--no-collect", action="store_true",
                   help="skip the final figures.zip bundling.")

    f = ap.add_argument_group("figs / validate stage options")
    f.add_argument("--circuit", default="all",
                   help="which circuit(s) to scope/validate: 'all' (default), a method "
                        "(manual|acdc|eap = all of its circuits), or a circuit folder "
                        "name (rough | acdc_thresh_<tau> | eap_top<n>; method inferred).")
    f.add_argument("--scope", default="full_circuit",
                   choices=["full_circuit", "ablated", "all"],
                   help="scoped-figure scope (default: full_circuit; 'all' = both).")
    f.add_argument("--scheme", default="random", choices=list(canon.SCHEMES) + ["all"],
                   help="contrast scheme for scoped figures / the single pair "
                        "(default: random).")
    f.add_argument("--mode", default="mean", choices=["mean", "zero", "resample"],
                   help="validation ablation reference (default: mean).")

    ac = ap.add_argument_group("acdc stage options")
    ac.add_argument("--thresholds", type=float, nargs="+", default=canon.CANON_TAUS,
                    help="ACDC tau list (default: %(default)s).")
    ac.add_argument("--acdc-examples", type=int, default=canon.SEGREV3_ACDC_EXAMPLES,
                    dest="acdc_examples",
                    help="ACDC contrast examples (default: %(default)s; kept small — "
                         "ACDC runtime scales with it).")

    ep = ap.add_argument_group("eap stage options")
    ep.add_argument("--topns", type=int, nargs="+", default=canon.SEGREV3_TOPNS,
                    help="EAP top-n list (default: %(default)s).")
    ep.add_argument("--export-n", type=int, default=canon.CANON_EXPORT_N,
                    help="EAP headline export top-n (default: %(default)s).")
    ep.add_argument("--ig-steps", type=int, default=canon.CANON_IG_STEPS,
                    help="EAP integrated-gradient steps (default: %(default)s).")

    # The single stage's parameters (shared flag set -- canon.py). --scheme and
    # --seed are the global flags above.
    canon.add_single_args(ap, title="single stage options (see also --scheme / --seed)")

    a = ap.parse_args(argv)

    # ---- stage-word validation / expansion ----
    stages = []
    for s in a.stages:
        if s not in STAGE_WORDS:
            ap.error(f"unknown stage {s!r}; choose from: {' | '.join(STAGE_WORDS)}")
        if s not in stages:
            stages.append(s)
    if "all" in stages:
        if "single" in stages:
            ap.error("'single' cannot be combined with 'all' (ACDC/EAP are excluded "
                     "from single-sequence mode).")
        stages = ["data", "manual", "acdc", "eap", "figs"] \
            + [s for s in stages if s in ("validate", "zip")]
    a.stages = stages

    a.single = "single" in stages
    if a.single:
        if len(stages) > 1:
            ap.error("'single' runs on its own (it already includes its scoped "
                     "figures and the final zip).")
        canon.check_single_args(a, ap.error)

    a.data_gen = "data" in stages
    a.methods = [m for m in METHOD_ORDER if m in stages]
    a.figs_stage = "figs" in stages
    a.validate_stage = "validate" in stages
    a.zip_stage = "zip" in stages

    try:
        a.circuit_mn = canon.parse_circuit(a.circuit)
    except ValueError as e:
        ap.error(str(e))
    return a


def _ensure_venv() -> None:
    """Re-exec under the repo's ./.venv when launched with a bare system python
    (the venv holds torch/tqdm/matplotlib; the system python does not), so
    ``python3 circuit_discovery/main.py ...`` works without activating anything.
    An explicitly activated environment -- any venv -- is respected as-is."""
    # NB: do NOT compare interpreter paths -- .venv/bin/python is a SYMLINK to
    # the base interpreter, so realpaths collide; the in_venv check alone is the
    # loop guard (after re-exec, sys.prefix != base_prefix).
    venv_py = os.path.join(REPO, ".venv", "bin", "python")
    in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    if not in_venv and os.path.exists(venv_py):
        print(f"(re-exec under {os.path.relpath(venv_py, os.getcwd())} — "
              "the system python lacks the pipeline deps)", flush=True)
        os.execv(venv_py, [venv_py, *sys.argv])


def main(argv=None) -> None:
    _ensure_venv()
    a = parse_args(argv)

    if not a.stages:
        print(RECIPES, end="")
        return

    r = Runner(dry_run=a.dry_run)

    # Pin the compute device for EVERY child stage in one place. resolve_device()
    # (in interp_tools/model_loading.py) reads SEGREV3_DEVICE, so exporting it here
    # makes the whole pipeline -- including stages with no --device flag (manual,
    # eap, validation, figures) -- agree on one device. "auto" => cuda when present.
    os.environ["SEGREV3_DEVICE"] = a.device
    # Same for the model checkpoint: resolve_checkpoint() reads SEGREV3_CKPT.
    if a.model:
        if not os.path.exists(a.model):
            raise SystemExit(f"✗ --model checkpoint not found: {a.model}")
        os.environ["SEGREV3_CKPT"] = os.path.abspath(a.model)

    print("=" * 78)
    print("  segrev3 CIRCUIT-DISCOVERY PIPELINE — main.py"
          + ("  [SINGLE-SEQUENCE MODE]" if a.single else ""))
    print(f"  stages: {' → '.join(a.stages)}")
    print(f"  validate={not a.no_validate}  figures={not a.no_figures}  "
          f"collect={not a.no_collect}  mode={a.mode}  scope={a.scope}  scheme={a.scheme}"
          f"  circuit={a.circuit}  parallel={a.parallel}  device={a.device}"
          + ("  [DRY RUN]" if a.dry_run else ""))
    if a.single:
        print(f"  single: n_threes={a.n_threes}  n_replace={a.n_replace}  "
              f"clean={'literal' if a.clean else 'generated'}  "
              f"corrupted={'literal' if a.corrupted else a.scheme}  "
              f"edge={'literal' if a.edge_input else 'generated'}")
    print("=" * 78, flush=True)

    t0 = time.time()

    # 1. DATA GENERATION (central store). Default to the apps we'll actually use.
    if a.data_gen:
        apps = a.methods if a.methods else ["all"]
        stage_data_gen(r, apps, n=a.examples, seed=a.seed)

    # 2. DISCOVERY (optionally CONCURRENT via --parallel) + per-method validation.
    if a.single:
        label, cmd, allow_fail = discover_spec(a, "manual", data_gen_ran=False)
        r.run(label, cmd, allow_fail=allow_fail)
        if not a.no_validate:
            _banner("VALIDATION — skipped (single-sequence mode)")
            print("  (validation aggregates batched contrasts of its own; a single "
                  "pair cannot support it.)", flush=True)
    elif a.methods:
        specs = [discover_spec(a, m, data_gen_ran=a.data_gen) for m in a.methods]
        if a.parallel and len(specs) > 1:
            r.run_parallel(specs)
        else:
            for label, cmd, allow_fail in specs:
                r.run(label, cmd, allow_fail=allow_fail)
        # Validation depends only on each method's own discovery -> safe to run now.
        # (Skipped when an explicit `validate` stage was named -- that one wins.
        #  --circuit narrows the tail to ONE circuit instead of every ran method's.)
        if not a.no_validate and not a.validate_stage:
            method, name = a.circuit_mn
            if method == "all":
                for m in a.methods:
                    stage_validate(r, m, mode=a.mode)
            else:
                stage_validate(r, method, mode=a.mode, name=name)

    # 3. EXPLICIT VALIDATION stage (--circuit selects what to validate).
    if a.validate_stage:
        method, name = a.circuit_mn
        r.py_script(f"VALIDATION: {a.circuit} — faithfulness / completeness / minimality",
                    os.path.join("validation", "validation.py"),
                    "--method", method, "--name", name, "--mode", a.mode,
                    allow_fail=True)

    # 4. SCOPED VISUALIZATION: the explicit `figs` stage (--circuit) or, after
    #    discovery, the automatic per-method tail. Single mode forwards its flags
    #    so the figures use the SAME single pair / edge input as the manual stage.
    if a.single:
        if not a.no_figures:
            # default --circuit all -> the manual rough circuit (any concrete
            # circuit selector scopes the single pair to that circuit instead).
            single_circuit = "rough" if a.circuit == "all" else a.circuit
            stage_scoped_figs(r, single_circuit, scope=a.scope, scheme=a.scheme,
                              extra=canon.single_argv(a))
    elif a.figs_stage:
        stage_scoped_figs(r, a.circuit, scope=a.scope, scheme=a.scheme)
    elif a.methods and not a.no_figures:
        # --circuit narrows the automatic tail to ONE circuit; default = every
        # circuit of each method that ran.
        method, name = a.circuit_mn
        if method == "all":
            for m in a.methods:
                stage_scoped_figs(r, m, scope=a.scope, scheme=a.scheme)
        else:
            stage_scoped_figs(r, a.circuit, scope=a.scope, scheme=a.scheme)

    # 5. COLLECT every figure into figures.zip (explicit `zip` stage, or the
    #    always-on tail after any other work unless --no-collect).
    did_work = a.data_gen or a.single or bool(a.methods) or a.figs_stage or a.validate_stage
    if a.zip_stage or (did_work and not a.no_collect):
        stage_collect(r)

    # ---- summary ----
    _banner(f"PIPELINE COMPLETE — {time.time() - t0:.0f}s total"
            + ("  [DRY RUN: nothing executed]" if a.dry_run else ""))
    ok = sum(1 for _, _, rc in r.ran if rc == 0)
    for label, dt, rc in r.ran:
        flag = "ok " if rc == 0 else "FAIL"
        print(f"  [{flag}] {label}  ({dt:.0f}s)")
    print(f"\n{ok}/{len(r.ran)} stages succeeded.")
    if not a.dry_run and (a.zip_stage or (did_work and not a.no_collect)):
        print(f"  figures bundle: {_rel(os.path.join(REPO, 'figures.zip'))}")


if __name__ == "__main__":
    main()
