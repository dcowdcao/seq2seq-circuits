#!/usr/bin/env python3
"""Gating/outvoting diagnostic for a TaskMux (multi-task, N3-breadth rung 1) model.

For every ordered task pair (A, B) on A-cued inputs (token 0 = 10+A):
  fringe   B's smallest-circuit edges minus A's (B's distinctive machinery); its
           activation share = mean |head z| of fringe heads vs all heads, plus the
           same fringe's level on B-cued inputs (ON reference) -> fringe_on_ratio.
  margin   logit(B(payload)) - logit(A(payload)) at output positions where the two
           members disagree (DSL-exact counterfactuals), at the final logits and as
           a per-layer logit-lens trajectory (mid-stack B evidence).
  verdict  gating (fringe OFF) / outvoting (fringe ON, wrong answer suppressed) /
           superposition (wrong answer live) — see docs/reference/variation_matrix.md C3xN3.

Per-task circuits are read from circuits_data/<model>/<circuit>/smallest_circuit.json;
by default task i's circuit dir is `task{i}` or `task{i}_*`, overridable with
--task-circuits (comma list, '-' = none). A task WITHOUT a faithful winner falls back
to its best-available candidate's edges (smallest_faithful_circuit rule; recorded as
task_faithful=false) so the pair still gets a real verdict; ONLY a task with zero
candidates degrades the pair to its logit-lens half (which needs only the model).
--dry-run validates structure (decode, samplers, dispatch, circuit discovery,
fringe math) without loading a model.

Usage:
  python task_router/router_diagnostic.py --model mux3.rev-id.dblodd-id.psum-id [--dry-run]
Output: task_router/<model>/diagnostic.json + a printed table.
"""
from __future__ import annotations

import argparse, json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))            # task_router/
PIPE = os.path.dirname(HERE)                                 # pipeline/
sys.path.insert(0, os.path.join(PIPE, "models"))
sys.path.insert(0, os.path.join(PIPE, "find_circuits"))

import torch

import algorithms
import conditions
import model_registry
from smallest_faithful_circuit import best_available_candidate   # ONE fallback rule

HEAD_RE = re.compile(r"^a(\d+)\.h(\d+)$")
GATING_ON_RATIO = 0.5      # fringe below half its own-task level on A-cued inputs = OFF


class DiagnosticError(Exception):
    pass


def die(msg: str) -> None:
    raise DiagnosticError(msg)


def mux_expr(model_name: str) -> "algorithms.TaskMux":
    expr = algorithms.decode_algorithm_id(algorithms.base_algorithm_id(model_name))
    if not isinstance(expr, algorithms.TaskMux):
        die(f"{model_name!r} is not a TaskMux id (decoded to {expr})")
    return expr


def load_edges(path: str) -> set[tuple] | None:
    d = json.load(open(path))
    if d.get("no_faithful_circuit") or "edges" not in d:
        return None                          # no faithful smallest circuit -> no edge set
    return {(e["parent"], e["child"], e.get("qkv")) for e in d["edges"]}


def heads_in(edges: set[tuple]) -> set[tuple[int, int]]:
    return {(int(m.group(1)), int(m.group(2)))
        for p, c, _ in edges
        for m in (HEAD_RE.match(p), HEAD_RE.match(c)) if m}


def resolve_task_circuits(model_name: str, n_tasks: int,
                          override: str | None) -> list[tuple[str, set[tuple], bool] | None]:
    """Per task: (circuit_name, edge set, faithful) or None (ZERO candidates ->
    logit-lens only). Default = dir `task{i}`/`task{i}_*` under circuits_data/<model>/;
    a task without a faithful winner uses its best-available candidate's edges
    (faithful=False). --task-circuits names must exist (explicit ask fails loud)."""
    root = model_registry.data_dir(model_name)

    def entry(name: str) -> tuple[str, set[tuple], bool] | None:
        path = os.path.join(root, name, "smallest_circuit.json")
        ed = load_edges(path) if os.path.exists(path) else None
        if ed is not None:
            return (name, ed, True)
        try:   # unfaithful winner -> best-available fallback; no candidates -> logit-lens only
            cand, _ = best_available_candidate(os.path.join(root, name))
        except FileNotFoundError:
            return None
        return (name, {(e["parent"], e["child"], e["qkv"]) for e in cand.edges}, False)

    if override is not None:
        names = [c.strip() for c in override.split(",")]
        if len(names) != n_tasks:
            die(f"--task-circuits has {len(names)} entries for {n_tasks} tasks")
        out = []
        for name in names:
            if name == "-":
                out.append(None); continue
            if not os.path.exists(os.path.join(root, name, "smallest_circuit.json")):
                die(f"no smallest_circuit.json for requested circuit {name!r} "
                    f"({os.path.join(root, name, 'smallest_circuit.json')})")
            out.append(entry(name))
        return out
    have = sorted(os.listdir(root)) if os.path.isdir(root) else []
    out = []
    for i in range(n_tasks):
        hits = [d for d in have if d == f"task{i}" or d.startswith(f"task{i}_")]
        if len(hits) > 1:
            die(f"task {i} matches several circuit dirs {hits}; disambiguate with --task-circuits")
        out.append(entry(hits[0]) if hits else None)
    return out


def run_cued(model, mux, task: int, n: int, seed: int) -> dict:
    """One teacher-forced forward on task-pinned inputs: tokens = [cue, payload, correct out].
    Returns correct outputs, final logits, per-layer logit-lens logits, per-head mean |z|."""
    device = next(model.parameters()).device
    x = conditions.task_pin(task, len(mux.members)).sample(n, seed)
    y = algorithms.batch_eval(mux, x)
    tokens = torch.cat([x, y], dim=1).to(device)
    names = lambda nm: nm.endswith("attn.hook_z") or nm.endswith("hook_resid_post")
    with torch.no_grad():
        logits, cache = model.run_with_cache(tokens, names_filter=names)
        L = model.cfg.n_layers
        lens = [model.unembed(model.ln_final(cache[f"blocks.{l}.hook_resid_post"])).cpu()
                for l in range(L)]
        act = torch.stack([cache[f"blocks.{l}.attn.hook_z"].norm(dim=-1).mean(dim=(0, 1))
                           for l in range(L)])                 # [L, H]
    return {"x": x, "y": y, "logits": logits.cpu(), "lens": lens, "act": act.cpu()}


def disagree_margin(logits: torch.Tensor, yA: torch.Tensor, yB: torch.Tensor,
                    input_len: int) -> tuple[float | None, float | None, int]:
    """(mean logit[B]-logit[A] over disagreeing output slots, wrong-top1 rate, n slots)."""
    diff = yB != yA
    if not bool(diff.any()):
        return None, None, 0
    pos = torch.arange(yA.shape[1]) + input_len - 1
    lg = logits[:, pos, :]
    margin = (lg.gather(-1, yB.unsqueeze(-1)) - lg.gather(-1, yA.unsqueeze(-1))).squeeze(-1)
    top1 = lg.argmax(-1) == yB
    return margin[diff].mean().item(), top1[diff].float().mean().item(), int(diff.sum())


def fringe_metrics(actA: torch.Tensor, actB: torch.Tensor,
                   fringe_heads: set[tuple[int, int]]) -> dict:
    idx = torch.tensor(sorted(fringe_heads), dtype=torch.long)
    fA = actA[idx[:, 0], idx[:, 1]]
    fB = actB[idx[:, 0], idx[:, 1]]
    return {"fringe_activity_share": (fA.sum() / actA.sum()).item(),
            "fringe_activity_ratio": (fA.mean() / actA.mean()).item(),
            "fringe_on_ratio": (fA.mean() / fB.mean()).item()}


def verdict(fringe: dict | None, margin: float | None, top1: float | None) -> str:
    if margin is None:
        return "identical-outputs"
    wrong_live = margin >= 0 or top1 > 0.5
    if fringe is None:
        return "wrong-answer-live (no circuits)" if wrong_live else "logit-lens-only (no circuits)"
    if wrong_live:
        return "superposition"
    return "gating" if fringe["fringe_on_ratio"] < GATING_ON_RATIO else "outvoting"


def pair_report(runs: list[dict], circuits: list, mux, a: int, b: int) -> dict:
    rA = runs[a]
    yB = algorithms.batch_eval(mux.members[b], rA["x"][:, 1:])
    margin, top1, n_slots = disagree_margin(rA["logits"], rA["y"], yB, mux.input_length)
    by_layer = [disagree_margin(l, rA["y"], yB, mux.input_length)[0] for l in rA["lens"]]
    rep = {"cued": a, "wrong": b, "wrong_answer_margin": margin,
           "wrong_top1_rate": top1, "n_disagree_slots": n_slots,
           "margin_by_layer": by_layer, "fringe_n_edges": None, "fringe_heads": None}
    fr = None
    if circuits[a] and circuits[b]:
        fringe_edges = circuits[b][1] - circuits[a][1]
        fringe_heads = heads_in(fringe_edges)
        rep["fringe_n_edges"] = len(fringe_edges)
        rep["fringe_heads"] = sorted(f"L{l}H{h}" for l, h in fringe_heads)
        if not fringe_heads:
            rep["verdict"] = "empty-fringe (B heads subset of A)"
            return rep
        fr = fringe_metrics(rA["act"], runs[b]["act"], fringe_heads)
        rep.update(fr)
    rep["verdict"] = verdict(fr, margin, top1)
    return rep


def print_table(pairs: list[dict]) -> None:
    fmt = lambda v, w: (f"{v:>{w}.3f}" if isinstance(v, float) else f"{str(v):>{w}}")
    print(f"\n{'pair':>7} {'f_edges':>7} {'f_share':>7} {'f_ratio':>7} {'on_ratio':>8} "
          f"{'margin':>8} {'top1':>6}  verdict")
    for p in pairs:
        print(f"{p['cued']:>3}->{p['wrong']:<3} {fmt(p['fringe_n_edges'], 7)} "
              f"{fmt(p.get('fringe_activity_share'), 7)} {fmt(p.get('fringe_activity_ratio'), 7)} "
              f"{fmt(p.get('fringe_on_ratio'), 8)} {fmt(p['wrong_answer_margin'], 8)} "
              f"{fmt(p['wrong_top1_rate'], 6)}  {p['verdict']}")


def dry_run(model_name: str, mux, circuits: list) -> None:
    n = len(mux.members)
    for i, m in enumerate(mux.members):
        assert m.output_length() == 10, f"member {i} not length-preserving"
    assert algorithms.generate_algorithm_id(mux) == algorithms.base_algorithm_id(model_name)
    xs = conditions.mux_sampler(n)(16, 0)
    assert torch.equal(xs, conditions.mux_sampler(n)(16, 0)), "mux_sampler not seed-deterministic"
    for i in range(n):
        c = conditions.task_pin(i, n)
        s = c.sample(8, 0)
        assert bool((s[:, 0] == 10 + i).all()) and all(c.holds(row) for row in s)
        assert bool((s[:, 1:] < 10).all()) and bool((s[:, 1:] >= 0).all())
        for row in s[:2]:
            assert torch.equal(mux.eval(row), mux.members[i].eval(row[1:])), "mux dispatch broken"
    found = [f"{c[0]}{'' if c[2] else ' (best-available)'}" if c else None for c in circuits]
    print(f"DRY-RUN OK: {model_name} decodes to {n}-task mux, vocab_size {mux.vocab_size()}, "
          f"members length-preserving, samplers deterministic, dispatch verified")
    print(f"  per-task circuits: {found}")
    for a in range(n):
        for b in range(n):
            if a != b and circuits[a] and circuits[b]:
                fringe = circuits[b][1] - circuits[a][1]
                print(f"  fringe {a}->{b}: {len(fringe)} edges, {len(heads_in(fringe))} heads")
    print(f"  planned: {n * (n - 1)} ordered pairs "
          f"({sum(1 for c in circuits if c)}/{n} with circuit edges; rest logit-lens only)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="TaskMux algorithm_id (mux<N>.<m1>.<m2>...)")
    ap.add_argument("--task-circuits", default=None,
                    help="comma list of circuit dir names, one per task ('-' = none); "
                         "default: auto-discover task{i}[_*] dirs")
    ap.add_argument("--n", type=int, default=256, help="inputs per cued task")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="validate structure without loading a model; writes nothing")
    args = ap.parse_args()

    mux = mux_expr(args.model)
    n_tasks = len(mux.members)
    member_ids = [algorithms.generate_algorithm_id(m) for m in mux.members]
    circuits = resolve_task_circuits(args.model, n_tasks, args.task_circuits)
    if args.dry_run:
        dry_run(args.model, mux, circuits)
        return

    model = model_registry.load_model(args.model, device=args.device)
    runs = [run_cued(model, mux, i, args.n, args.seed) for i in range(n_tasks)]
    pairs = [pair_report(runs, circuits, mux, a, b)
             for a in range(n_tasks) for b in range(n_tasks) if a != b]

    out = {"model": args.model, "n_tasks": n_tasks, "member_ids": member_ids,
           "n_inputs": args.n, "seed": args.seed,
           "task_circuits": [c[0] if c else None for c in circuits],
           "task_faithful": [c[2] if c else None for c in circuits],
           "gating_on_ratio_threshold": GATING_ON_RATIO,
           "per_task_head_activity": {str(i): r["act"].tolist() for i, r in enumerate(runs)},
           "pairs": pairs}
    out_path = os.path.join(HERE, args.model, "diagnostic.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    json.dump(out, open(out_path + ".tmp", "w"), indent=2)
    os.replace(out_path + ".tmp", out_path)   # atomic: concurrent per-circuit runs can't interleave
    print_table(pairs)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    try:
        main()
    except DiagnosticError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)
