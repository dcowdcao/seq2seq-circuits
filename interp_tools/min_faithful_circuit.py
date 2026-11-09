"""
min_faithful_circuit.py
=================================================================================
Find the SMALLEST head-level circuit that reaches a target faithfulness
(default: recovered token-accuracy >= 0.80), by sweeping BOTH discovery
methods over a wide parameter grid IN PARALLEL and then validating that the
grid actually brackets the answer:

  * ACDC    over tau    (larger tau  -> more pruning -> smaller circuit)
  * EAP-IG  over top-n  (smaller n   -> fewer edges  -> smaller circuit)

Pipeline (one "round"):
  1. DISCOVERY (parallel). Missing taus run via ``circuit_discovery/acdc.py
     --thresholds ...`` (its own spawn-based process pool over taus); missing
     top-ns run via ``circuit_discovery/eap.py --topns ...`` (ONE attribution
     pass shared by every top-n). The two subprocesses run concurrently.
     Params whose artifacts already exist under visualizations/ are reused
     (pass --force to recompute).
  2. EVALUATION. Every discovered circuit is reduced to its head set
     (interp_tools/circuit_loader) and scored on ONE SHARED random dataset
     (data_gen.segrev3_batch(eval_n, seed)) -- unlike validation.py, which
     scores each circuit on its own (tiny, N=20 for acdc) store file -- with
     the standard mean-ablation faithfulness:
         recovered_acc = (acc(circuit_only) - acc(empty)) / (acc(full) - acc(empty))
     All keep-sets are evaluated in one batched forward
     (validation/n_threes_faithfulness._eval_keep_many_single).
  3. RANGE VALIDATION. The grid must BRACKET the target on the small-circuit
     side, otherwise the true minimum could lie outside it:
       * acdc: the LARGEST tau must FAIL the bar (or prune to 0 heads), and at
         least one tau must PASS. Otherwise extend the grid (x2 up / /2 down).
       * eap : the SMALLEST top-n must FAIL the bar, and at least one top-n
         must PASS. Otherwise extend (//2 down / x2 up).
  4. REFINEMENT. While the bracket holds, insert midpoints between the
     smallest PASSING parameter and its FAILING neighbor (geometric mean for
     tau, integer midpoint for top-n) to tighten the minimal circuit, for
     --refine-rounds extra rounds.

Selection: among ALL evaluated circuits (both methods) with
recovered_acc >= target and >0 heads, pick the fewest heads; ties broken by
higher recovered_acc, then name. The winner is finally re-validated through
the standard validation/validation.py pipeline (its own artifacts + numbers).

Outputs:
  visualizations/min_faithful_circuit_search.json          -- full table + winner
  validation/figs_data/figs/min_faithful_circuit_search.png -- heads vs faithfulness
  the winner's circuit graph at its native artifact path (acdc/<name>/circuit.png |
  eap/<name>/<name>.png) -- only if the discovery run wrote none (no pygraphviz):
  then a matplotlib reimplementation of the method's native style
  circuit folders for parameters beyond the shipped sweep (gitignored); the
  combined acdc/eap sweep summaries are restored to their pre-search content

Run:  python interp_tools/min_faithful_circuit.py
      (--target 0.8 --eval-n 512 --force --acdc-jobs 4 ... ; see --help)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS)
for _sub in ("interp_tools", "data_gen", "metrics", "validation"):
    _p = os.path.join(_ROOT, _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch                                                    # noqa: E402

import circuit_loader as CL                                     # noqa: E402
from model_loading import assert_memory_headroom, load_segrev3, repo_relpath  # noqa: E402
from data_gen import segrev3_batch                              # noqa: E402
import validation as V                                          # noqa: E402
import n_threes_faithfulness as NTF                             # noqa: E402
from canon import SEGREV3_ACDC_EXAMPLES                         # noqa: E402

PY = sys.executable
VIZ = os.path.join(_ROOT, "visualizations")

# Wide initial grids. They deliberately overshoot on BOTH sides of the expected
# 80% boundary (known anchors: eap_top50/9 heads and acdc tau=0.1/18 heads fail
# on random data; acdc tau=0.055/25 heads passes); step 3 of the pipeline
# verifies the bracket and extends these automatically if they are too narrow.
DEFAULT_TAUS = [0.01, 0.02, 0.04, 0.055, 0.08, 0.1, 0.15, 0.2, 0.3, 0.5]
DEFAULT_TOPNS = [5, 10, 15, 20, 30, 40, 50, 75, 100, 150, 200, 400, 800]
TOPN_CAP = 3200          # an EAP top-n beyond every real edge == full model


def _tau_str(tau) -> str:
    return "%g" % float(tau)


def _round_tau(tau) -> float:
    """2 significant digits -> tidy folder names for refined taus."""
    return float("%.2g" % float(tau))


def acdc_name(tau) -> str:
    return f"acdc_thresh_{_tau_str(tau)}"


def eap_name(n) -> str:
    return f"eap_top{int(n)}"


def _json_ok(path) -> bool:
    """True iff ``path`` exists AND parses as JSON -- a killed discovery run can
    leave a truncated artifact behind, which must count as not-done."""
    try:
        with open(path) as f:
            json.load(f)
        return True
    except (OSError, json.JSONDecodeError):
        return False


def acdc_done(tau) -> bool:
    return _json_ok(os.path.join(VIZ, "acdc", acdc_name(tau), "summary.json"))


def eap_done(n) -> bool:
    nm = eap_name(n)
    return _json_ok(os.path.join(VIZ, "eap", nm, f"{nm}.json"))


# --------------------------------------------------------------------------- #
# 1. Discovery (ACDC and EAP-IG as two concurrent subprocesses)               #
# --------------------------------------------------------------------------- #


def _child_env():
    env = os.environ.copy()
    libs = [os.path.join(_ROOT, d) for d in
            ("interp_tools", "data_gen", "metrics", "validation", "visualizations")]
    extra = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(libs + ([extra] if extra else []))
    return env


def _run_cmd(label, cmd):
    t0 = time.time()
    r = subprocess.run(cmd, cwd=_ROOT, env=_child_env(),
                       capture_output=True, text=True)
    dt = time.time() - t0
    if r.returncode != 0:
        tail = "\n".join((r.stdout + "\n" + r.stderr).strip().splitlines()[-30:])
        raise RuntimeError(f"{label} failed (exit {r.returncode}, {dt:.0f}s):\n{tail}")
    print(f"    {label}: done ({dt:.0f}s)")


def run_discoveries(taus, topns, force=False, acdc_jobs=4, session_done=None):
    """Run the two discovery methods IN PARALLEL for every parameter that does
    not already have a (valid) artifact on disk.

    ``force`` recomputes parameters even if their artifacts exist -- but only
    once per session: ``session_done`` (a set of artifact names, updated here)
    records what this session already computed, so params proposed in LATER
    rounds are also force-refreshed exactly once. After the subprocesses
    return, every requested artifact is re-checked on disk -- a discovery
    runner that swallows a per-parameter failure cannot slip through.

    ACDC parallelizes across taus internally (spawn ProcessPoolExecutor, safe
    per the per-tau output folders); EAP-IG computes its attribution once and
    materializes every requested top-n from the same scores. The two
    subprocesses share the GPU -- the model is tiny and launch-latency bound,
    so concurrent processes interleave well.

    NB: both runners read their datasets from the central store files as-is
    (acdc: N=20, eap: N=512); acdc.py gets ``-n`` so a tau whose store file is
    missing gets a same-size (20-example) set.
    """
    sd = set() if session_done is None else session_done
    need_taus = sorted({float(t) for t in taus
                        if (force and acdc_name(t) not in sd) or not acdc_done(t)})
    need_topns = sorted({int(n) for n in topns
                         if (force and eap_name(n) not in sd) or not eap_done(n)})
    jobs = []
    if need_taus:
        jobs.append((f"acdc  taus={[_tau_str(t) for t in need_taus]}",
                     [PY, os.path.join(_ROOT, "circuit_discovery", "acdc.py"),
                      "--thresholds", *[_tau_str(t) for t in need_taus],
                      "-n", str(SEGREV3_ACDC_EXAMPLES), "--device", "auto",
                      "--jobs", str(min(acdc_jobs, len(need_taus)))]))
    if need_topns:
        jobs.append((f"eap   topns={need_topns}",
                     [PY, os.path.join(_ROOT, "circuit_discovery", "eap.py"),
                      "--topns", *map(str, need_topns)]))
    if not jobs:
        print("    (all discovery artifacts already on disk)")
        return
    # Pre-flight: the jobs below fan out CUDA subprocesses (acdc additionally
    # spawns its own worker pool); refuse to launch into a starved box. Each
    # child process is also capped per-process inside load_segrev3.
    assert_memory_headroom(context=f"min_faithful_circuit launching "
                                   f"{len(jobs)} discovery subprocess(es)")
    with ThreadPoolExecutor(max_workers=len(jobs)) as ex:
        futs = [ex.submit(_run_cmd, label, cmd) for label, cmd in jobs]
        for f in futs:
            f.result()                       # surface the first failure
    missing = ([f"acdc tau={_tau_str(t)}" for t in need_taus if not acdc_done(t)]
               + [f"eap topn={n}" for n in need_topns if not eap_done(n)])
    if missing:
        raise RuntimeError("discovery reported success but artifacts are "
                           f"missing/invalid for: {missing}")
    sd.update(acdc_name(t) for t in need_taus)
    sd.update(eap_name(n) for n in need_topns)


# --------------------------------------------------------------------------- #
# 2. Shared-dataset faithfulness evaluation                                   #
# --------------------------------------------------------------------------- #


def evaluate_circuits(model, taus, topns, eval_n, eval_seed, mean_cache):
    """recovered_acc for every (method, param) circuit, all in one batched
    forward on ONE shared random dataset. Returns (rows, baselines)."""
    circuits = []
    for t in sorted({float(t) for t in taus}):
        heads = CL.head_tuples(CL.load_circuit("acdc", acdc_name(t)))
        circuits.append(("acdc", float(_tau_str(t)), acdc_name(t), heads))
    for n in sorted({int(n) for n in topns}):
        heads = CL.head_tuples(CL.load_circuit("eap", eap_name(n)))
        circuits.append(("eap", int(n), eap_name(n), heads))

    dev = next(model.parameters()).device
    toks, _, outs = segrev3_batch(eval_n, seed=eval_seed)
    toks, outs = toks.to(dev), outs.to(dev)
    all_heads = [(l, h) for l in range(model.cfg.n_layers)
                 for h in range(model.cfg.n_heads)]
    keep_sets = [all_heads, []] + [c[3] for c in circuits]
    accs = NTF._eval_keep_many_single(model, keep_sets, toks, outs, mean_cache)
    full_acc, empty_acc = accs[0], accs[1]
    denom = full_acc - empty_acc

    rows = []
    for (method, param, name, heads), acc in zip(circuits, accs[2:]):
        rec = (acc - empty_acc) / denom if abs(denom) > 1e-9 else float("nan")
        rows.append({"method": method, "param": param, "name": name,
                     "n_heads": len(heads),
                     "heads": [f"L{l}H{h}" for (l, h) in heads],
                     "circuit_acc": acc, "recovered_acc": rec})
    baselines = {"full_acc": full_acc, "empty_acc": empty_acc,
                 "eval_n": eval_n, "eval_seed": eval_seed,
                 "eval_data": f"data_gen.segrev3_batch({eval_n}, seed={eval_seed})"
                              " -- SHARED across all circuits",
                 "ablation": "mean (validation._mean_z, global random-data mean z)"}
    return rows, baselines


def _passes(row, target) -> bool:
    return row["n_heads"] > 0 and row["recovered_acc"] >= target


# --------------------------------------------------------------------------- #
# 3+4. Range validation (bracket the target) and boundary refinement          #
# --------------------------------------------------------------------------- #


def range_checks(rows, target):
    """Verify each method's grid BRACKETS the target faithfulness on the
    small-circuit side; propose grid extensions where it does not.

    Bracketing means: moving toward smaller circuits (larger tau / smaller
    top-n) the grid reaches a parameter that FAILS the bar, and at least one
    parameter PASSES. Without the failing end, an even smaller faithful
    circuit could exist beyond the grid; without the passing end, the bar was
    never reached at all and the grid must extend toward larger circuits.
    """
    props, notes = {"taus": [], "topns": []}, []
    acdc = sorted((r for r in rows if r["method"] == "acdc"), key=lambda r: r["param"])
    eap = sorted((r for r in rows if r["method"] == "eap"), key=lambda r: r["param"])

    if acdc:
        if not any(_passes(r, target) for r in acdc):
            props["taus"].append(_round_tau(acdc[0]["param"] / 2))
            notes.append("acdc: no tau passes the bar -> extend grid DOWN "
                         f"(add tau={props['taus'][-1]})")
        top = acdc[-1]
        if _passes(top, target):
            props["taus"].append(_round_tau(top["param"] * 2))
            notes.append("acdc: LARGEST tau still passes -> a smaller faithful "
                         f"circuit may lie beyond the grid; add tau={props['taus'][-1]}")
    if eap:
        if not any(_passes(r, target) for r in eap):
            props["topns"].append(min(eap[-1]["param"] * 2, TOPN_CAP))
            notes.append("eap: no top-n passes the bar -> extend grid UP "
                         f"(add topn={props['topns'][-1]})")
        bottom = eap[0]
        if _passes(bottom, target) and bottom["param"] > 1:
            props["topns"].append(max(1, bottom["param"] // 2))
            notes.append("eap: SMALLEST top-n still passes -> a smaller faithful "
                         f"circuit may lie below the grid; add topn={props['topns'][-1]}")
    ok = not (props["taus"] or props["topns"])
    return ok, props, notes


def refine_proposals(rows, target):
    """Midpoints between the smallest PASSING parameter (toward small circuits)
    and its FAILING neighbor, per method, to tighten the minimal circuit."""
    props = {"taus": [], "topns": []}
    acdc = sorted((r for r in rows if r["method"] == "acdc"), key=lambda r: r["param"])
    passing = [r["param"] for r in acdc if _passes(r, target)]
    if passing:
        t_pass = max(passing)                          # largest tau that passes
        bigger = [r["param"] for r in acdc if r["param"] > t_pass]
        if bigger:                                     # its failing neighbor
            mid = _round_tau(math.sqrt(t_pass * min(bigger)))
            if all(acdc_name(mid) != r["name"] for r in acdc):
                props["taus"].append(mid)
    eap = sorted((r for r in rows if r["method"] == "eap"), key=lambda r: r["param"])
    passing = [r["param"] for r in eap if _passes(r, target)]
    if passing:
        n_pass = min(passing)                          # smallest topn that passes
        smaller = [r["param"] for r in eap if r["param"] < n_pass]
        if smaller:
            mid = (n_pass + max(smaller)) // 2
            if max(smaller) < mid < n_pass:
                props["topns"].append(mid)
    return props


def pick_winner(rows, target):
    elig = [r for r in rows if _passes(r, target)]
    if not elig:
        return None
    return sorted(elig, key=lambda r: (r["n_heads"], -r["recovered_acc"], r["name"]))[0]


# --------------------------------------------------------------------------- #
# Winner circuit rendering (native ACDC / EAP styles, matplotlib backend)     #
# --------------------------------------------------------------------------- #
# The native renderers (acdc.acdc_graphics.show, eap Graph.to_image) both
# hard-require pygraphviz (and the system graphviz `dot` binary); where those are
# missing, freshly discovered folders carry no circuit PNG. These helpers reproduce each method's style with a
# graphviz-"dot"-like layered layout in matplotlib and write the PNG to the
# method's NATIVE artifact path (visualizations/acdc/<name>/circuit.png /
# visualizations/eap/<name>/<name>.png), never clobbering a real native render.

# EAP edge colors (github_repos/EAP-IG/src/eap/visualization.py)
_EAP_EDGE_COLORS = {"q": "#FF00FF", "k": "#00FF00", "v": "#0000FF"}


def _pastel_colors(names, seed=0):
    """Per-node random pastel fills -- both native styles draw nodes with
    random colormap colors (EAP: Pastel2; ACDC graphics: random pastels)."""
    import numpy as np
    from matplotlib import colormaps
    rng = np.random.RandomState(seed)
    cmap = colormaps["Pastel2"]
    return {nm: cmap(rng.randint(0, 256) / 255.0) for nm in names}


def _dot_ranks(names, edges_raw):
    """graphviz-'dot' rank assignment: LONGEST PATH from the sources (this is
    what gives the native renders their organic, dependency-driven rows --
    e.g. an a3 head whose parents are all in a0 sits high up, not in an
    'a3 row'). Kahn topological sweep; the graphs are DAGs."""
    from collections import deque
    children = {n: [] for n in names}
    indeg = {n: 0 for n in names}
    for s, r in edges_raw:
        if s == r:        # self-loop (e.g. ACDC's collapsed q_in->q): no rank info
            continue
        children[s].append(r)
        indeg[r] += 1
    rank = {n: 0 for n in names}
    q = deque(n for n in names if indeg[n] == 0)
    while q:
        n = q.popleft()
        for c in children[n]:
            rank[c] = max(rank[c], rank[n] + 1)
            indeg[c] -= 1
            if indeg[c] == 0:
                q.append(c)
    return rank


def _barycenter_order(by_rank, edges_raw):
    """dot's crossing-reduction step (median/barycenter heuristic): order each
    rank by the mean position of the neighbors in the adjacent rank, sweeping
    down (by parents) then up (by children). Returns {name: index_in_rank}."""
    parents, children = {}, {}
    for s, r in edges_raw:
        parents.setdefault(r, []).append(s)
        children.setdefault(s, []).append(r)
    order = {rk: sorted(ns) for rk, ns in by_rank.items()}
    x = {n: i for ns in order.values() for i, n in enumerate(ns)}
    for _ in range(3):
        for rk in sorted(order)[1:]:                       # downward: parents
            order[rk].sort(key=lambda n: (sum(x[p] for p in parents.get(n, []))
                                          / len(parents[n])) if parents.get(n) else x[n])
            x.update({n: i for i, n in enumerate(order[rk])})
        for rk in sorted(order, reverse=True)[1:]:         # upward: children
            order[rk].sort(key=lambda n: (sum(x[c] for c in children.get(n, []))
                                          / len(children[n])) if children.get(n) else x[n])
            x.update({n: i for i, n in enumerate(order[rk])})
    return x


def _layered_circuit_png(nodes, edges, edges_raw, path, transparent=False):
    """graphviz-'dot'-style DAG render: longest-path ranks, barycenter
    ordering, compact wide-and-short geometry, rounded boxes, curved directed
    splines WITH arrowheads (graphviz default).

    nodes: {name: fill_color}; edges: [(sender, receiver, color, width)];
    edges_raw: [(sender, receiver)]; transparent: ACDC's bgcolor=transparent.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    ranks = _dot_ranks(set(nodes), edges_raw)
    by_rank = {}
    for nm in nodes:
        by_rank.setdefault(ranks[nm], []).append(nm)
    n_ranks = max(by_rank) + 1
    width = max(len(v) for v in by_rank.values())
    xidx = _barycenter_order(by_rank, edges_raw)

    pos = {}
    for rk, names in by_rank.items():
        names.sort(key=lambda n: xidx[n])
        step = width / (len(names) + 1)
        for i, nm in enumerate(names):
            pos[nm] = ((i + 1) * step, n_ranks - 1 - rk)

    # Native proportions: wide and short (nodesep-dominant, tight ranksep).
    fig, ax = plt.subplots(figsize=(max(8, 0.55 * width), max(1.8, 0.62 * n_ranks)))
    box_h = 0.34
    for s, r, color, w in edges:                       # edges under the boxes
        (x0, y0), (x1, y1) = pos[s], pos[r]
        # graphviz 'normal' arrowhead: small solid triangle at the receiver
        ax.annotate("", xy=(x1, y1 + box_h / 2), xytext=(x0, y0 - box_h / 2),
                    arrowprops=dict(arrowstyle="-|>,head_width=0.25,head_length=0.45",
                                    mutation_scale=8, color=color, lw=w,
                                    alpha=0.9, shrinkA=0, shrinkB=0,
                                    connectionstyle="arc3,rad=0.12"))
    for nm, (x, y) in pos.items():
        bw = min(0.9 * width / (len(by_rank[ranks[nm]]) + 1),
                 0.10 + 0.052 * len(nm))
        # graphviz style="rounded": corner radius is a large fraction of box height
        ax.add_patch(FancyBboxPatch((x - bw / 2, y - box_h / 2), bw, box_h,
                                    boxstyle="round,pad=0.015,rounding_size=0.11",
                                    facecolor=nodes[nm], edgecolor="black",
                                    lw=0.7, zorder=3))
        ax.text(x, y, nm, ha="center", va="center", fontsize=5.5,
                family="sans-serif", zorder=4)
    ax.set_xlim(-0.2, width + 0.2)
    ax.set_ylim(-0.55, n_ranks - 0.45)
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(path, dpi=150, bbox_inches="tight",
                transparent=transparent,
                facecolor="none" if transparent else "white")
    plt.close(fig)


def _plot_acdc_circuit(name, path):
    """ACDC native style (cf. the vendored acdc_thresh_0.02/circuit.gv): plain
    head nodes <a0.7> plus per-channel nodes <a0.7_q/k/v> (labels keep the
    angle brackets, as in the gv), random pastel fills, every edge drawn in its
    SENDER's fill color at fixed penwidth 0.6, TRANSPARENT background."""
    with open(os.path.join(VIZ, "acdc", name, "summary.json")) as f:
        summ = json.load(f)

    def node(label):
        """ACDC label -> gv node, as acdc_graphics maps them: the qkv INPUT
        ('a0.7_qin') becomes the channel node <a0.7_q>; the qkv OUTPUT
        ('a0.7_q', hook_q) belongs to the head and becomes <a0.7> -- so
        DIRECT_COMPUTATION edges render as <a0.7_q> -> <a0.7>."""
        if re.search(r"_[qkv]in$", label):
            return f"<{label[:-2]}>"
        return f"<{re.sub(r'_[qkv]$', '', label)}>"

    edges_raw = sorted({(node(e["sender_label"]), node(e["receiver_label"]))
                        for e in summ["surviving_edges"]})
    names = {n for e in edges_raw for n in e}
    fills = _pastel_colors(sorted(names))
    edges = [(s, r, fills[s], 0.6) for s, r in edges_raw]
    _layered_circuit_png(fills, edges, edges_raw, path, transparent=True)


def _plot_eap_circuit(name, path):
    """EAP native style (cf. Graph.to_image): in-graph nodes as pastel rounded
    boxes; edge color by receiver channel (q magenta / k green / v blue, plain
    edges black or red when the score is negative); penwidth = |score| min-max
    normalized over ALL edge scores into [0.6, 5.0]."""
    with open(os.path.join(VIZ, "eap", name, f"{name}.json")) as f:
        g = json.load(f)

    names = sorted(nm for nm, nd in g["nodes"].items() if nd.get("in_graph"))
    fills = _pastel_colors(names)
    all_abs = [abs(e["score"]) for e in g["edges"].values()
               if e.get("score") is not None]
    lo, hi = min(all_abs), max(all_abs)

    edges, edges_raw = [], []
    for estr, e in g["edges"].items():
        if not e.get("in_graph"):
            continue
        sender, child = estr.split("->")
        m = re.match(r"(.+?)<([qkv])>$", child)
        receiver, qkv = (m.group(1), m.group(2)) if m else (child, None)
        color = (_EAP_EDGE_COLORS[qkv] if qkv else
                 ("#FF0000" if e["score"] < 0 else "#000000"))
        norm = (abs(e["score"]) - lo) / (hi - lo) if hi != lo else abs(e["score"])
        edges.append((sender, receiver, color, max(0.6, norm * 5.0)))
        edges_raw.append((sender, receiver))

    _layered_circuit_png(fills, edges, edges_raw, path)


def plot_winner_circuit(method, name):
    """Render the winning circuit in ITS method's native style, at the native
    artifact path. Skips (and returns the existing path) if a real native
    render is already there."""
    if method == "acdc":
        path = os.path.join(VIZ, "acdc", name, "circuit.png")
    else:
        path = os.path.join(VIZ, "eap", name, f"{name}.png")
    if not os.path.exists(path):
        (_plot_acdc_circuit if method == "acdc" else _plot_eap_circuit)(name, path)
    return path


# --------------------------------------------------------------------------- #
# Reporting                                                                   #
# --------------------------------------------------------------------------- #


def _print_table(rows, target):
    print(f"    {'circuit':<22} {'param':>8} {'heads':>5} {'circ_acc':>9} "
          f"{'recovered':>9}  >= {target:.2f}?")
    for r in sorted(rows, key=lambda r: (r["method"], r["param"])):
        mark = "PASS" if _passes(r, target) else "fail"
        print(f"    {r['name']:<22} {r['param']:>8g} {r['n_heads']:>5} "
              f"{r['circuit_acc']:>9.3f} {r['recovered_acc']:>9.3f}  {mark}")


def plot(rows, target, winner, baselines, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for method, color in (("acdc", "#2980b9"), ("eap", "#e67e22")):
        rs = [r for r in rows if r["method"] == method]
        ax.scatter([r["n_heads"] for r in rs], [r["recovered_acc"] for r in rs],
                   s=26, color=color, label={"acdc": "ACDC (tau)",
                                             "eap": "EAP-IG (top-n)"}[method])
        for r in rs:
            ax.annotate(f"{r['param']:g}", (r["n_heads"], r["recovered_acc"]),
                        fontsize=6, xytext=(3, 3), textcoords="offset points")
    ax.axhline(target, color="crimson", ls="--", lw=1.0,
               label=f"target = {target:.2f}")
    if winner:
        ax.scatter([winner["n_heads"]], [winner["recovered_acc"]], s=160,
                   facecolors="none", edgecolors="k", lw=1.6, label=f"winner: "
                   f"{winner['name']} ({winner['n_heads']} heads)")
    ax.set_xlabel("heads in circuit")
    ax.set_ylabel("recovered token-acc (shared eval set)")
    ax.set_title("Smallest faithful circuit search: ACDC tau-sweep vs EAP-IG "
                 f"top-n sweep\n(mean ablation, {baselines['eval_n']} shared "
                 "random examples; labels = parameter values)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Driver                                                                      #
# --------------------------------------------------------------------------- #


def search(target=0.80, taus=None, topns=None, eval_n=512, eval_seed=0,
           force=False, acdc_jobs=4, max_extend_rounds=4, refine_rounds=2):
    taus = set(DEFAULT_TAUS if taus is None else map(float, taus))
    topns = set(DEFAULT_TOPNS if topns is None else map(int, topns))

    model = load_segrev3()
    mean_cache = V._mean_z(model)

    # Separate budgets: grid EXTENSIONS (bracket not yet established) and
    # boundary REFINEMENTS (bracket holds, tightening the minimum). A budget is
    # only spent on proposals that are actually discovered+evaluated -- every
    # loop iteration ends in an evaluation of the full current grid, so the
    # final `rows` always covers exactly `taus`/`topns`.
    extends_left, refines_left = max_extend_rounds, refine_rounds
    session_done = set()
    # The combined sweep summaries describe the circuits present before the search;
    # the search's extra circuits only get their own folders (and this report).
    sweep_summaries = [os.path.join(VIZ, "acdc", "acdc_all_thresholds_summary.json"),
                       os.path.join(VIZ, "eap", "eap_all_topn_summary.json")]
    saved = {p: open(p).read() for p in sweep_summaries if os.path.exists(p)}
    rounds_log, rows = [], []
    rnd = 0
    while True:
        rnd += 1
        print(f"== round {rnd}: discovery over {len(taus)} taus + {len(topns)} "
              "top-ns (parallel; existing artifacts reused) ==")
        run_discoveries(sorted(taus), sorted(topns), force=force,
                        acdc_jobs=acdc_jobs, session_done=session_done)
        rows, baselines = evaluate_circuits(model, taus, topns, eval_n,
                                            eval_seed, mean_cache)
        _print_table(rows, target)

        ok, ext_props, notes = range_checks(rows, target)
        entry = {"round": rnd, "taus": sorted(taus), "topns": sorted(topns),
                 "bracket_ok": ok, "notes": list(notes)}
        props = {"taus": [], "topns": []}
        if not ok:
            for n in notes:
                print(f"    RANGE CHECK: {n}")
            if extends_left > 0:
                extends_left -= 1
                props = ext_props
            else:
                entry["notes"].append("extension budget exhausted; proposals "
                                      "dropped (bracket NOT established)")
                print("    RANGE CHECK: extension budget exhausted -- stopping "
                      "with an unconfirmed bracket.")
        elif refines_left > 0:
            rprops = refine_proposals(rows, target)
            if rprops["taus"] or rprops["topns"]:
                refines_left -= 1
                props = rprops
                entry["notes"].append(f"refining boundary: +taus={props['taus']}"
                                      f" +topns={props['topns']}")
                print(f"    REFINE: {entry['notes'][-1]}")

        new_t = [t for t in props["taus"] if acdc_name(t) not in
                 {acdc_name(x) for x in taus}]
        new_n = [n for n in props["topns"] if n not in topns]
        rounds_log.append(entry)
        if not new_t and not new_n:
            if ok:
                print("    range checks PASSED: both grids bracket the target.")
            break
        taus.update(new_t)
        topns.update(new_n)
    for p, text in saved.items():
        with open(p, "w") as f:
            f.write(text)

    winner = pick_winner(rows, target)
    final_ok, _, final_notes = range_checks(rows, target)

    # Re-validate the winner through the standard repo pipeline (writes the
    # usual per-circuit artifacts; numbers there use the circuit's own store
    # dataset, so they complement -- not replace -- the shared-set numbers).
    winner_standard, winner_png = None, None
    if winner is not None:
        try:
            res = V.validate_circuit(winner["method"], winner["name"],
                                     mode="mean", model=model,
                                     mean_cache=mean_cache)
            winner_standard = {"recovered_acc": res["faithfulness"]["recovered_acc"],
                               "recovered_logit_diff":
                                   res["faithfulness"]["recovered_logit_diff"],
                               "data": res.get("data")}
        except Exception as e:                                  # non-fatal
            winner_standard = {"error": str(e)}
        try:
            # the winner's circuit graph, in its method's native style
            winner_png = repo_relpath(plot_winner_circuit(winner["method"], winner["name"]))
        except Exception as e:                                  # non-fatal
            winner_png = f"render failed: {e}"

    report = {
        "target_recovered_acc": target,
        "selection": "fewest heads among circuits with recovered_acc >= target; "
                     "ties -> higher recovered_acc",
        "baselines": baselines,
        "range_validation": {"bracket_ok": final_ok, "notes": final_notes,
                             "criteria": "largest tau FAILS & some tau PASSES; "
                                         "smallest top-n FAILS & some top-n PASSES"},
        "rounds": rounds_log,
        "results": sorted(rows, key=lambda r: (r["n_heads"], -r["recovered_acc"])),
        "winner": winner,
        "winner_standard_validation": winner_standard,
        "winner_circuit_png": winner_png,
    }
    out_json = os.path.join(VIZ, "min_faithful_circuit_search.json")
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)
    out_png = os.path.join(V.FIG_DIR, "min_faithful_circuit_search.png")
    plot(rows, target, winner, baselines, out_png)

    print("\n=== smallest faithful circuit ===")
    if winner is None:
        print(f"  NO circuit reached recovered_acc >= {target} -- see {out_json}")
    else:
        print(f"  {winner['name']}  ({winner['method']}, param={winner['param']:g}): "
              f"{winner['n_heads']} heads, recovered_acc={winner['recovered_acc']:.3f}")
        print(f"  heads: {winner['heads']}")
        print(f"  circuit graph ({winner['method']} style): {winner_png}")
        if not final_ok:
            print("  WARNING: range checks not satisfied "
                  f"({'; '.join(final_notes)}) -- treat as a lower bound only.")
    print(f"  wrote {os.path.relpath(out_json, _ROOT)} and "
          f"{os.path.relpath(out_png, _ROOT)}")
    return report


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[2])
    ap.add_argument("--target", type=float, default=0.80,
                    help="faithfulness bar: recovered token-acc (default 0.80)")
    ap.add_argument("--taus", type=float, nargs="+", default=None,
                    help=f"initial ACDC tau grid (default {DEFAULT_TAUS})")
    ap.add_argument("--topns", type=int, nargs="+", default=None,
                    help=f"initial EAP-IG top-n grid (default {DEFAULT_TOPNS})")
    ap.add_argument("--eval-n", type=int, default=512,
                    help="shared evaluation dataset size (default 512)")
    ap.add_argument("--eval-seed", type=int, default=0)
    ap.add_argument("--acdc-jobs", type=int, default=4,
                    help="parallel ACDC workers across taus (default 4)")
    ap.add_argument("--force", action="store_true",
                    help="re-run discovery even where artifacts exist")
    ap.add_argument("--max-extend-rounds", type=int, default=4)
    ap.add_argument("--refine-rounds", type=int, default=2)
    return ap.parse_args()


if __name__ == "__main__":
    a = parse_args()
    search(target=a.target, taus=a.taus, topns=a.topns, eval_n=a.eval_n,
           eval_seed=a.eval_seed, force=a.force, acdc_jobs=a.acdc_jobs,
           max_extend_rounds=a.max_extend_rounds,
           refine_rounds=a.refine_rounds)
