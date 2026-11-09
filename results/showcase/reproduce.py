#!/usr/bin/env python
"""Reproduce the showcase circuits on CPU.

    python results/showcase/reproduce.py --verify   # re-evaluate every stored circuit.json
                                                    # (seed 0) and heldout_seed1.json (seed 1)
                                                    # and check every number (floats to 1e-6,
                                                    # the rest exactly)
    python results/showcase/reproduce.py --full     # re-run discovery (EAP-IG candidates,
                                                    # probes, router diagnostic) and selection
                                                    # from the checkpoints, then regenerate
                                                    # figures, cards, index and COMMANDS.md

--full first deletes the generated (gitignored) artifacts of the five examples under
pipeline/circuits_data, pipeline/probes_data and pipeline/task_router, so the candidate pool
is exactly what the listed commands produce; it then rewrites circuit.json, candidates.json
(and base_circuit.json where a class has a base circuit), re-evaluates the selected circuit
on held-out rows (--seed 1: fresh evaluation and fresh mean-reference rows, everything else
identical; heldout_seed1.json, because --select picks the winner on the seed-0 rows it
reports), and records each command's wall time in <example>/runtimes.json, which COMMANDS.md
is rendered from.

Selection (pipeline/find_circuits/evaluate_circuit.py, --threads 4 everywhere so float
reductions are reproducible) implements the class definitions:
  DomainCircuit       --select over the EAP-IG candidates (all output slots, whole-sequence
                      accuracy under edge-level mean ablation; floor 0.90).
  ContrastCircuit     --select directly under interchange patching (--ablation auto); edges
                      the edit never changes contribute nothing there, so no union with the
                      base. The fringe against the base DomainCircuit (mexhat_full, selected
                      the same way and stored as base_circuit.json) and the mean-ablation a are
                      reported alongside.
  SuppressionCircuit  alternating cue-flip pairs, mean ablation, each row scored against its
                      own reference output. Every candidate is evaluated as-is AND unioned with
                      the cued task's circuit (task1, base_circuit.json; --union-base), because
                      the circuit must route and run both tasks; the best passing selection
                      score over both modes wins and circuit.json records the mode.
  Probe/Bridge        discovery (eap.py --probe) and evaluation both read each probed
                      position's own probe (--probe-readout position), at the layer whose
                      per-position probes are most accurate (eap.py's site choice, passed to
                      the evaluator as --probe-layer from the candidates' meta). Discovery runs on two
                      pair pools per variable; for graviclump_mean, whose layer-0 probe only 43
                      edges can reach, each pool also gets the explicit rungs
                      17 20 23 26 30 34 38 43 (showcase_spec.Example.probe_topns), because the
                      4-rung frac ladder over a 43-edge budget is too coarse.
When no candidate passes the floor, the most faithful candidate is shipped with passes=false.

Options: --only <example dir> ... restricts either mode; --discovery-only stops --full after
discovery; --skip-discovery redoes only selection, the held-out file and the generated pages;
--heldout-only redoes only the held-out file and the pages.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import showcase_spec as S  # noqa: E402

sys.path.insert(0, os.path.join(S.PIPELINE, "models"))      # algorithms (the DSL), torch-only

PY = sys.executable
RUN = "pipeline/run_model_pipeline.py"
EAP = "pipeline/find_circuits/eap.py"
EVAL = "pipeline/find_circuits/evaluate_circuit.py"
SMALLEST = "pipeline/find_circuits/smallest_faithful_circuit.py"
SEED = 0
THREADS = "2"          # BLAS threads per process (the smallest stage splits them over its children)
SCRATCH = os.path.join(S.PIPELINE, "circuits_data", "_showcase")   # gitignored
EVAL_THREADS = "4"     # the evaluator's fixed torch thread count (selection and --verify)
PROBE_READOUT = "position"   # probe readout for discovery AND evaluation (see probe_eap)
HELDOUT_SEED = 1       # the held-out re-evaluation: fresh rows and fresh mean-reference rows


def pipe(model: str, circuit: str, *stages: str, target: str | None = None) -> list:
    argv = [RUN, "--local", "--device", "cpu", "--model", model, "--only-circuit", circuit,
            "--stages", *stages]
    return argv + (["--smallest-target", target] if target else [])


def probe_eap(model: str, circuit: str, variable: str, topns=()) -> list:
    """Probe-metric EAP-IG over the canonical frac ladder (under --probe, eap.py scales it by
    the edges that can reach the probe, i.e. the graph cut at the probe layer), or over
    explicit ``topns`` rungs (see showcase_spec.Example.probe_topns). The KL metric reads
    each probed position's own probe (--probe-readout position): the probes whose accuracy
    picks the probe layer, and the readout the evaluator scores with."""
    ladder = ["--topns", *map(str, topns)] if topns else ["--topn-regime", "frac"]
    return [EAP, "--model", model, *ladder, "--circuit", circuit, "--probe", variable,
            "--probe-readout", PROBE_READOUT, "--device", "cpu"]


def probe_steps(key: str, pools: tuple) -> list:
    """Probe-metric EAP-IG on each pair pool: the canonical ladder, then the example's
    explicit rungs if it declares any. Every candidate of every pool enters selection."""
    ex = S.BY_KEY[key]
    steps = []
    for c in pools:
        steps.append((f"probe EAP-IG on {c} pairs", probe_eap(ex.model, c, ex.circuit)))
        if ex.probe_topns:
            steps.append((f"probe EAP-IG on {c} pairs, explicit rungs "
                          f"{' '.join(map(str, ex.probe_topns))}",
                          probe_eap(ex.model, c, ex.circuit, ex.probe_topns)))
    return steps


#: Discovery, per example: (step name, argv relative to the repo root). Order matters: a contrast's differential smallest stage reads its base
#: circuit's selection, and the suppression row's router diagnostic reads both task circuits'.
#: --smallest-target is the class floor (0.90 for domain-type circuits, 0.80 for the paired
#: classes); the smallest stage's --search adds one EAP-IG candidate tuned to it.
MUX = "mux2.pmstep-id.sr12-id__moe8"
DISCOVERY = {
    "01_domain_schmitt_id__moe8": [
        ("schmitt_noband", pipe("schmitt_id__moe8", "schmitt_noband",
                                "data", "train", "eap", "smallest", "audit", target="0.9")),
    ],
    "02_contrast_mexhat_id__moe8": [
        ("mexhat_fill: data + EAP-IG", pipe("mexhat_id__moe8", "mexhat_fill",
                                            "data", "train", "eap")),
        ("mexhat_full (base)", pipe("mexhat_id__moe8", "mexhat_full",
                                    "data", "eap", "smallest", "audit", target="0.9")),
        ("mexhat_fill: smallest", pipe("mexhat_id__moe8", "mexhat_fill", "smallest", "audit",
                                       target="0.8")),
    ],
    "03_suppression_mux2.pmstep-id.sr12-id__moe8": [
        # task1 (the 0/1 latch): most random pairs' outputs agree everywhere and are dropped, so
        # eap.py reads a longer prefix of the pool to get its 200 + 256 attribution pairs. The
        # smallest stage scores the candidates without its extra --search candidate: that is how
        # the shipped task1 pool was generated (before eap.py could extend a short pool, the
        # search failed on it); adding --search now would add one more candidate.
        ("task1 (base): data + EAP-IG", pipe(MUX, "task1", "data", "train", "eap")),
        ("task1 (base): smallest", [SMALLEST, "--model", MUX, "--circuit", "task1",
                                    "--target", "0.9", "--jobs", "4"]),
        ("task0", pipe(MUX, "task0", "data", "eap", "smallest", "audit", target="0.9")),
        ("suppress_t1_over_t0 (+ router diagnostic)",
         pipe(MUX, "suppress_t1_over_t0", "data", "eap", "smallest", "audit", target="0.8")),
    ],
    "04_probe_graviclump_id__moe4": [
        ("probes (site scan, training, validation)",
         pipe("graviclump_id__moe4", "graviclump_full", "data", "train", "probes")),
        ("pair pool graviclump_edit", pipe("graviclump_id__moe4", "graviclump_edit", "data")),
    ] + probe_steps("04_probe_graviclump_id__moe4", ("graviclump_full", "graviclump_edit")),
    "05_bridge_psum_novel_id__moe8": [
        ("probes (site scan, training, validation)",
         pipe("psum_novel_id__moe8", "psum_novel_whole", "data", "train", "probes")),
        ("pair pool psum_novel_fresh", pipe("psum_novel_id__moe8", "psum_novel_fresh", "data")),
    ] + probe_steps("05_bridge_psum_novel_id__moe8", ("psum_novel_whole", "psum_novel_fresh")),
}

#: generated artifacts --full deletes first (all gitignored), per example
GENERATED = {
    "01_domain_schmitt_id__moe8": ["circuits_data/schmitt_id__moe8/schmitt_noband"],
    "02_contrast_mexhat_id__moe8": ["circuits_data/mexhat_id__moe8/mexhat_fill",
                                    "circuits_data/mexhat_id__moe8/mexhat_full"],
    "03_suppression_mux2.pmstep-id.sr12-id__moe8": [
        f"circuits_data/{MUX}/task0", f"circuits_data/{MUX}/task1",
        f"circuits_data/{MUX}/suppress_t1_over_t0", f"task_router/{MUX}"],
    "04_probe_graviclump_id__moe4": [
        "circuits_data/graviclump_id__moe4/graviclump_full",
        "circuits_data/graviclump_id__moe4/graviclump_edit",
        "probes_data/graviclump_id__moe4/graviclump_mean"],
    "05_bridge_psum_novel_id__moe8": [
        "circuits_data/psum_novel_id__moe8/psum_novel_whole",
        "circuits_data/psum_novel_id__moe8/psum_novel_fresh",
        "probes_data/psum_novel_id__moe8/bridge1_novel_id"],
}

#: the circuit whose selected edges are the base of the class's fringe (Contrast: the
#: declared DomainCircuit base; Suppression: the cued task's own circuit). Selected with the
#: same evaluator and stored as base_circuit.json, so the fringe is reproducible from the repo.
BASE = {"02_contrast_mexhat_id__moe8": "mexhat_full",
        "03_suppression_mux2.pmstep-id.sr12-id__moe8": "task1"}


def probes_root(ex) -> str:
    """Where an example's probe artifacts are committed (probe classes only)."""
    return os.path.join(ex.dir, "probes")


def env() -> dict:
    e = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", SEGREV3_DEVICE="cpu",
             OMP_NUM_THREADS=THREADS, MKL_NUM_THREADS=THREADS, OPENBLAS_NUM_THREADS=THREADS)
    e["PYTHONPATH"] = os.pathsep.join(S.PY_PATHS + [e.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    return e


def run(argv: list, log: str | None = None) -> float:
    """Run ``python <argv>`` from the repo root; return wall seconds; raise on failure."""
    print("$ python " + " ".join(argv), flush=True)
    t0 = time.time()
    out = open(log, "w") if log else None
    rc = subprocess.run([PY, *argv], cwd=S.REPO, env=env(), stdout=out,
                        stderr=subprocess.STDOUT if out else None).returncode
    if out:
        out.close()
    if rc:
        raise SystemExit(f"command failed (exit {rc}): python {' '.join(argv)}"
                         + (f"  [log: {log}]" if log else ""))
    return time.time() - t0


def runtimes_path(ex) -> str:
    return os.path.join(ex.dir, "runtimes.json")


def record_runtimes(ex, phase: str, steps: list) -> None:
    """<example>/runtimes.json: measured wall seconds per command (rendered in COMMANDS.md)."""
    os.makedirs(ex.dir, exist_ok=True)
    p = runtimes_path(ex)
    rt = json.load(open(p)) if os.path.exists(p) else {}
    rt[phase] = steps
    with open(p, "w") as f:
        json.dump(rt, f, indent=1)
        f.write("\n")


def discover(ex) -> None:
    for d in GENERATED[ex.key]:
        shutil.rmtree(os.path.join(S.PIPELINE, d), ignore_errors=True)
    os.makedirs(SCRATCH, exist_ok=True)
    steps = []
    for i, (name, argv) in enumerate(DISCOVERY[ex.key]):
        secs = run(argv, log=os.path.join(SCRATCH, f"{ex.key}.{i}.log"))
        steps.append({"step": name, "command": "python " + " ".join(argv),
                      "wall_seconds": round(secs)})
        print(f"  done in {secs:.0f}s", flush=True)
    record_runtimes(ex, "discovery", steps)


# ---------------------------------------------------------------------------- #
# selection: evaluator --select over the discovered candidates -> circuit.json  #
# ---------------------------------------------------------------------------- #
def eval_argv(ex, circuit: str, out: str, candidate: str | None = None,
              base: str | None = None, probes: str | None = None, extra=(),
              seed: int = SEED) -> list:
    argv = [EVAL, "--model", ex.model, "--circuit", circuit]
    argv += ["--candidate", candidate] if candidate else ["--select"]
    argv += ["--base-candidate", base] if base else []
    argv += ["--probes-root", probes] if probes else []
    return argv + [*extra, "--device", "cpu", "--seed", str(seed), "--threads", EVAL_THREADS,
                   "--out", out]


def probe_flags(ex) -> list:
    """Probe classes are evaluated at the layer and with the readout eap.py --probe used,
    both read from the candidates' recorded probe meta."""
    if ex.cls not in ("ProbeCircuit", "BridgeCircuit"):
        return []
    metas = glob.glob(os.path.join(S.PIPELINE, "circuits_data", ex.model, "*", "circuits",
                                   f"eap_probe_{ex.circuit}_top*", "summary.json"))
    sites = {(m["layer"], m["readout"]) for p in metas
             for m in [json.load(open(p))["meta"]["probe"]]}
    if len(sites) != 1:
        raise SystemExit(f"{ex.key}: probe candidates disagree on (or lack) the probe layer / "
                         f"readout: {sites}")
    layer, readout = sites.pop()
    return ["--probe-layer", str(layer), "--probe-readout", readout]


def stored_probe_flags(rec: dict) -> list:
    p = rec.get("probe")
    return ["--probe-layer", str(p["layer"]), "--probe-readout", p["readout"]] if p else []


def ship_probe(ex, probe: dict) -> str:
    """Copy the probe the evaluator used (site scan + the chosen layer's probe) next to the
    circuit, so --verify does not depend on retraining probes; return its root (repo-rel)."""
    src = os.path.join(S.PIPELINE, "probes_data", ex.model, probe["variable"])
    dst = os.path.join(probes_root(ex), ex.model, probe["variable"])
    shutil.rmtree(probes_root(ex), ignore_errors=True)
    os.makedirs(dst)
    for fn in ("site_scan.json", f"site_L{probe['layer']}.pt"):
        shutil.copyfile(os.path.join(src, fn), os.path.join(dst, fn))
    return S.rel(probes_root(ex))


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def architecture(ex) -> dict:
    import torch
    cfg = torch.load(ex.checkpoint, map_location="cpu", weights_only=False)["config"]
    keys = ("n_layers", "n_heads", "d_model", "d_head", "d_vocab", "n_ctx", "attn_only",
            "d_mlp", "act_fn", "num_experts", "experts_per_token")
    return {k: cfg.get(k) for k in keys}


def dump(obj, path: str) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=1)
        f.write("\n")


#: evaluator fields copied into circuit.json, and compared by --verify
EVAL_FIELDS = ("class", "repo_class", "ablation", "floor", "seed", "a", "a_clean", "a_corrupt",
               "passes", "n_nodes", "n_nodes_full", "n_edges", "n_edges_full",
               "selection_score", "nodes", "edges", "fringe", "secondary", "probe",
               "reference", "node_definition", "edge_definition", "n_rows", "n_pairs",
               "pairs_kept", "pairs_dropped", "model_seq_acc")


def evaluate(ex, circuit: str, tag: str, steps: list, **kw) -> dict:
    """Run the evaluator once (``--select`` unless ``candidate=``) and return its payload."""
    out = os.path.join(SCRATCH, f"{ex.key}.{tag}.json")
    argv = eval_argv(ex, circuit, S.rel(out), **kw)
    secs = run(argv)
    steps.append({"step": f"evaluate: {tag}", "command": "python " + " ".join(argv),
                  "wall_seconds": round(secs)})
    return json.load(open(out))


def choose(ex, circuit: str, runs: dict, steps: list, **kw) -> tuple:
    """(mode, payload) of the selected circuit. ``runs`` maps a selection mode to its --select
    payload. The winner maximizes the selection score among candidates with a > floor over
    all modes; when nothing passes, the most faithful candidate (highest a, then score) is
    evaluated on its own and shipped with passes = false."""
    won = [(m, p) for m, p in runs.items() if p.get("winner")]
    if won:
        return max(won, key=lambda mp: mp[1]["selection_score"])
    mode, row = max(((m, r) for m, p in runs.items() for r in p["candidates"]),
                    key=lambda mr: (mr[1]["a"], mr[1]["selection_score"]))
    print(f"  {ex.key}/{circuit}: NO candidate passes the floor; shipping the most faithful "
          f"one ({row['candidate']}, a={row['a']:.4f}, mode {mode}) with passes=false")
    flags = list(kw.pop("extra", ())) + (["--union-base"] if mode != "candidate" else [])
    p = evaluate(ex, circuit, f"{circuit}.best", steps, candidate=row["candidate"],
                 extra=flags, **kw)
    return mode, dict(p, winner=row["candidate"])


def record(ex, circuit: str, mode: str, payload: dict, fn: str, base_rel: str | None,
           probes: str | None) -> dict:
    """circuit.json / base_circuit.json: the selected circuit's evaluator fields plus
    provenance (repo-relative paths, no times)."""
    import algorithms  # noqa: E402
    rec = {"model": ex.model, "circuit": circuit}
    rec.update({k: payload[k] for k in EVAL_FIELDS if k in payload})
    me = S.rel(os.path.join(ex.dir, fn))
    rec.update({
        "candidate": payload["winner"],
        "selection_mode": mode,
        "checkpoint": S.rel(ex.checkpoint),
        "checkpoint_sha256": sha256(ex.checkpoint),
        "architecture": architecture(ex),
        "algorithm": str(algorithms.decode_algorithm_id(
            algorithms.base_algorithm_id(ex.model))),
        "selection_command": payload["command"],
        "command": "python " + " ".join(eval_argv(
            ex, circuit, "<out.json>", candidate=me, base=base_rel, probes=probes,
            extra=stored_probe_flags(payload))),
    })
    return rec


def heldout_argv(ex, out: str) -> list:
    """The evaluator on the shipped circuit.json with --seed HELDOUT_SEED: fresh evaluation
    rows (pairs) and fresh mean-reference rows; everything else as in circuit.json's command."""
    rec = json.load(open(os.path.join(ex.dir, "circuit.json")))
    base = (S.rel(os.path.join(ex.dir, "base_circuit.json"))
            if os.path.exists(os.path.join(ex.dir, "base_circuit.json")) else None)
    probes = S.rel(probes_root(ex)) if rec.get("probe") else None
    return eval_argv(ex, rec["circuit"], out, candidate=S.rel(os.path.join(ex.dir, "circuit.json")),
                     base=base, probes=probes, extra=stored_probe_flags(rec), seed=HELDOUT_SEED)


def heldout(ex) -> None:
    """<example>/heldout_seed1.json: the shipped circuit re-evaluated on held-out rows (the
    selection picked the winner on the seed-0 rows it reports)."""
    os.makedirs(SCRATCH, exist_ok=True)
    out = os.path.join(SCRATCH, f"{ex.key}.heldout.json")
    argv = heldout_argv(ex, S.rel(out))
    secs = run(argv)
    payload = json.load(open(out))
    rec = {"model": ex.model, "circuit": payload["circuit"]}
    rec.update({k: payload[k] for k in EVAL_FIELDS if k in payload})
    rec["command"] = "python " + " ".join(heldout_argv(ex, "<out.json>"))
    dump(rec, os.path.join(ex.dir, "heldout_seed1.json"))
    record_runtimes(ex, "heldout", [{"step": f"held-out re-evaluation (seed {HELDOUT_SEED})",
                                     "command": "python " + " ".join(argv),
                                     "wall_seconds": round(secs)}])


def select(ex) -> None:
    os.makedirs(SCRATCH, exist_ok=True)
    steps, base_rel = [], None
    if ex.key in BASE:                                  # the base circuit first
        b = BASE[ex.key]
        mode, payload = choose(ex, b, {"candidate": evaluate(ex, b, f"{b}.select", steps)}, steps)
        dump(record(ex, b, mode, payload, "base_circuit.json", None, None),
             os.path.join(ex.dir, "base_circuit.json"))
        base_rel = S.rel(os.path.join(ex.dir, "base_circuit.json"))
    extra = probe_flags(ex)
    runs = {"candidate": evaluate(ex, ex.circuit, f"{ex.circuit}.select", steps,
                                  base=base_rel, extra=extra)}
    if ex.cls == "SuppressionCircuit":                  # also: candidate + the cued task's circuit
        runs[f"candidate + {BASE[ex.key]}"] = evaluate(
            ex, ex.circuit, f"{ex.circuit}.select-union", steps, base=base_rel,
            extra=extra + ["--union-base"])
    mode, payload = choose(ex, ex.circuit, runs, steps, base=base_rel, extra=extra)
    first = next(iter(runs.values()))
    table = {"model": ex.model, "circuit": ex.circuit, "class": first["class"],
             "floor": first["floor"], "selected": payload["winner"], "selected_mode": mode,
             "selection_rule": "maximize (1 - n_nodes/n_nodes_full) * (1 - n_edges/n_edges_full) "
                               "among candidates with a > floor",
             "n_nodes_full": first["n_nodes_full"], "n_edges_full": first["n_edges_full"],
             "modes": [{"mode": m, "command": p["command"], "winner": p.get("winner"),
                        "candidates": p["candidates"]} for m, p in runs.items()]}
    dump(table, os.path.join(ex.dir, "candidates.json"))
    probes = ship_probe(ex, payload["probe"]) if payload.get("probe") else None
    dump(record(ex, ex.circuit, mode, payload, "circuit.json", base_rel, probes),
         os.path.join(ex.dir, "circuit.json"))
    record_runtimes(ex, "selection", steps)


#: --verify tolerance for float fields (accuracies, scores): float reductions can differ in
#: the last bits across machines and BLAS thread counts. Everything else must match exactly.
FLOAT_TOL = 1e-6


def compare(stored, fresh, path: str, bad: list) -> float:
    """Append mismatches to ``bad``: structure, strings, ints and bools exactly, floats within
    FLOAT_TOL. Return the largest float deviation seen."""
    if isinstance(stored, float) and isinstance(fresh, (int, float)) and \
            not isinstance(fresh, bool):
        dev = abs(stored - fresh)
        if dev > FLOAT_TOL:
            bad.append(f"{path}: stored {stored!r} != fresh {fresh!r} (|diff| {dev:.3g})")
        return dev
    if isinstance(stored, dict) and isinstance(fresh, dict):
        if stored.keys() != fresh.keys():
            bad.append(f"{path}: keys differ: {sorted(stored.keys() ^ fresh.keys())}")
        return max([compare(stored[k], fresh[k], f"{path}.{k}", bad)
                    for k in stored.keys() & fresh.keys()], default=0.0)
    if isinstance(stored, list) and isinstance(fresh, list) and len(stored) == len(fresh):
        return max([compare(a, b, f"{path}[{i}]", bad)
                    for i, (a, b) in enumerate(zip(stored, fresh))], default=0.0)
    if stored != fresh or type(stored) is not type(fresh):
        bad.append(f"{path}: stored {stored!r} != fresh {fresh!r}")
    return 0.0


def verify(ex) -> tuple:
    """Re-evaluate the stored circuit.json (and base_circuit.json) with seed 0 and the
    held-out heldout_seed1.json with seed 1; return (mismatches, largest float deviation)."""
    os.makedirs(SCRATCH, exist_ok=True)
    bad, worst = [], 0.0
    files = (["base_circuit.json"] if ex.key in BASE else []) + ["circuit.json",
                                                                 "heldout_seed1.json"]
    for fn in files:
        stored = json.load(open(os.path.join(ex.dir, fn)))
        out = os.path.join(SCRATCH, f"{ex.key}.verify.{fn}")
        base = (S.rel(os.path.join(ex.dir, "base_circuit.json"))
                if fn != "base_circuit.json" and ex.key in BASE else None)
        probes = S.rel(probes_root(ex)) if stored.get("probe") else None
        t0 = time.time()
        if fn == "heldout_seed1.json":
            run(heldout_argv(ex, S.rel(out)))
        else:
            run(eval_argv(ex, stored["circuit"], S.rel(out),
                          candidate=S.rel(os.path.join(ex.dir, fn)), base=base, probes=probes,
                          extra=stored_probe_flags(stored)))
        fresh = json.load(open(out))
        dev = compare({k: stored[k] for k in EVAL_FIELDS if k in stored},
                      {k: fresh.get(k) for k in EVAL_FIELDS if k in stored}, f"{ex.key}/{fn}", bad)
        worst = max(worst, dev)
        print(f"  {ex.key}/{fn}: a={fresh['a']:.4f} nodes={fresh['n_nodes']}/"
              f"{fresh['n_nodes_full']} edges={fresh['n_edges']}/{fresh['n_edges_full']} "
              f"passes={fresh['passes']}  max float deviation {dev:.3g}  "
              f"({time.time() - t0:.0f}s)", flush=True)
    return bad, worst


def render() -> None:
    for script in ("make_figures.py", "make_cards.py", "make_index.py", "make_commands.py"):
        run([S.rel(os.path.join(HERE, script))])


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--verify", action="store_true")
    mode.add_argument("--full", action="store_true")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--discovery-only", action="store_true")
    ap.add_argument("--skip-discovery", action="store_true")
    ap.add_argument("--heldout-only", action="store_true",
                    help="with --full: only (re)write heldout_seed1.json, then the pages")
    a = ap.parse_args(argv)
    exs = [e for e in S.EXAMPLES if a.only is None or e.key in a.only]
    if a.verify:
        results = [verify(ex) for ex in exs]
        bad = [m for b, _ in results for m in b]
        worst = max(w for _, w in results)
        print("\n".join(bad) if bad else
              f"VERIFY OK: {len(exs)} example(s); structure, counts and pass flags identical, "
              f"max float deviation {worst:.3g} (tolerance {FLOAT_TOL:g})")
        raise SystemExit(1 if bad else 0)
    for ex in exs:
        if not (a.skip_discovery or a.heldout_only):
            discover(ex)
        if not a.discovery_only:
            os.makedirs(ex.dir, exist_ok=True)
            if not a.heldout_only:
                select(ex)
            heldout(ex)
    if not a.discovery_only:
        render()


if __name__ == "__main__":
    main()
