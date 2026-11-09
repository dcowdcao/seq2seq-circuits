#!/usr/bin/env python3
"""Faithfulness ``a`` and size of ONE circuit, using the definition of its circuit class.

Classes (inferred from the registered circuit or probe variable named by ``--circuit``):

  DomainCircuit       edge-level per-position mean ablation (means over ``--n-rows`` rows of the
                      circuit's input domain, one clean unablated forward pass). a = whole-sequence
                      accuracy against the DSL output on ``--n-rows`` fresh domain rows, teacher-forced
                      with the true output. Floor 0.90.
  ContrastCircuit     interchange patching (``--ablation auto``) on the circuit's clean/corrupt pairs.
                      Both members share one teacher-forced prefix (the DSL output of the CLEAN input);
                      labels are the full model's argmax outputs on each member. Pairs with identical
                      label rows, or whose only label difference is the edited digit passed straight
                      through to its own output slot, are dropped. Forward: baseline = the corrupt
                      member's unablated activations, the clean member is patched in through the
                      declared edges and must reproduce every clean label; reverse: the mirror.
                      a = a_clean (forward) x a_corrupt (reverse). Empty circuit -> 0 and full circuit
                      -> 1 by construction. The mean-ablation number (alternating pairs, DSL labels) is
                      reported under ``secondary``. Floor 0.80.
  SuppressionCircuit  TaskMux cue flip: clean = cued task's cue + payload, corrupt = wrong task's cue
                      + the same payload (payloads where the two tasks disagree). Mean ablation with
                      means over the mixed rows; labels = each row's own DSL output;
                      a = a_clean x a_corrupt. Floor 0.80.
  ProbeCircuit /      the linear probe the probes stage trained (``train_probes.py``) at layer l, the
  BridgeCircuit       layer with the highest mean probe accuracy in its ``site_scan.json``. The graph is
                      truncated at layer l: the terminal reader is the residual at the probe site, not
                      the logits. Probe accuracy over the variable's probed input positions is
                      normalized so 0 = per-position majority class (the best constant predictor on the
                      same rows) and 1 = the full model's probe, clamped at 0. a_clean on fresh uniform
                      inputs; a_corrupt on, for each clean input, the single-digit edit (all positions x
                      digits) that maximizes the full model's mean true-class probe margin.
                      a = a_clean x a_corrupt. Floor 0.80.

Edge-level ablation: every reader of a circuit node (head q/k/v input, MLP input, terminal) reads
``baseline_site + sum_declared(live_s - baseline_s)``; circuit nodes are recomputed in order from
their declared input edges. The baseline is the per-position mean (mean ablation; the pipeline's
``smallest_faithful_circuit._edge_site_means`` / ``_edge_forward``) or the partner member's
activations (interchange; ``_patched_forward`` below, which also serves the truncated probe graph).

Size: n_nodes counts attention heads plus one node per MLP/MoE block (all experts of a layer are one
node), the pipeline's node count; n_edges counts (writer, reader) edges with q/k/v inputs separate,
over the live edge universe ``canon.live_edge_budget``. ``selection_score`` =
(1 - n_nodes/n_nodes_full)(1 - n_edges/n_edges_full); ``--select`` scores every candidate under the
circuit's candidates dir and picks the maximum score among candidates with a > floor.

Output: ``circuits_data/<model>/<circuit>/evaluation.json`` (probe classes:
``probes_data/<model>/<variable>/evaluation.json``). Deterministic given ``--seed``.

Usage:
  python pipeline/find_circuits/evaluate_circuit.py --model <model_id> --circuit <circuit or variable>
      [--candidate <circuit dir/json> | --select] [--device cpu] [--seed 0]
      [--n-rows 10000] [--n-pairs 5000] [--ablation auto|mean|interchange] [--out <json>]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field

_THIS = os.path.dirname(os.path.abspath(__file__))            # .../pipeline/find_circuits
PIPELINE_ROOT = os.path.dirname(_THIS)
REPO_ROOT = os.path.dirname(PIPELINE_ROOT)
for _p in (_THIS, os.path.join(PIPELINE_ROOT, "probes")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch                                                   # noqa: E402

import smallest_faithful_circuit as SFC                        # noqa: E402  (bootstraps sys.path)
import model_registry as MG                                    # noqa: E402
import gen_circuit_data                                        # noqa: E402
import circuits as dsl_circuits                                # noqa: E402
import conditions                                              # noqa: E402
from canon import live_edge_budget                             # noqa: E402
import train_probes as TP                                      # noqa: E402
import variables as VARS                                       # noqa: E402

FLOORS = {"DomainCircuit": 0.90, "ContrastCircuit": 0.80, "SuppressionCircuit": 0.80,
          "ProbeCircuit": 0.80, "BridgeCircuit": 0.80}
BATCH = 250
DEVICE = "cpu"                     # set from --device in run()
NODE_DEFINITION = ("attention heads a{l}.h{h} plus one node per MLP/MoE block m{l} (all experts of "
                   "a layer form one node); the input embedding and the terminal (logits, or the "
                   "probe site for probe classes) are edge endpoints, not nodes")
REFERENCE_NOTE = ("model = the unablated model under the same metric (the full circuit up to float "
                  "noise; 1 by construction for interchange, by definition for probe classes); "
                  "empty = no edges")
EDGE_DEFINITION = ("(writer, reader) pairs; writers = input embedding, heads, MLP/MoE blocks; "
                   "readers = each head's q/k/v input (separately), each MLP input, the terminal")


# --------------------------------------------------------------------------- #
# Circuits: edge sets in the pipeline's node dialect                          #
# --------------------------------------------------------------------------- #
def _sender_name(sid) -> str:
    if sid == "input":
        return "input"
    return f"m{sid[1]}" if sid[0] == "m" else f"a{sid[0]}.h{sid[1]}"


def _sender_key(sid) -> tuple:
    """Fixed forward order of senders (sets of mixed tuples/strings iterate in hash order)."""
    if sid == "input":
        return (-1, 0, 0)
    return (sid[1], 1, 0) if sid[0] == "m" else (sid[0], 0, sid[1])


def _node_layer(name: str) -> tuple:
    """(layer, rank, head) sort key of a node: input first, heads before the MLP of their
    layer, logits last."""
    if name == "input":
        return (-1, 0, 0)
    if name == "logits":
        return (10 ** 6, 0, 0)
    if name[0] == "m":
        return (int(name[1:]), 1, 0)
    l, h = name[1:].split(".h")
    return (int(l), 0, int(h))


def graph_edges(n_layers: int, n_heads: int, attn_only: bool) -> list:
    """Every live edge of a ``n_layers``-layer graph (the pipeline's edge universe)."""
    out = []
    for site, senders in SFC.full_declared_edges(n_layers, n_heads, attn_only).items():
        if site == "final":
            child, qkv = "logits", None
        elif site[0] == "m":
            child, qkv = f"m{site[1]}", None
        else:
            child, qkv = f"a{site[0]}.h{site[1]}", site[2]
        out += [{"parent": _sender_name(s), "child": child, "qkv": qkv} for s in senders]
    assert len(out) == live_edge_budget(n_layers, n_heads, attn_only)
    return sorted(out, key=_edge_key)


def _nodes_of(edges: list) -> set:
    """Head / MLP nodes touched by ``edges`` (input and terminal are endpoints, not nodes)."""
    return ({e["parent"] for e in edges} | {e["child"] for e in edges}) - {"input", "logits"}


def _edge_key(e) -> tuple:
    return (_node_layer(e["child"]), e["qkv"] or "", _node_layer(e["parent"]))


@dataclass
class Circuit:
    name: str
    edges: list                                   # [{parent, child, qkv}], pruned, causal
    source: str = ""
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        uniq = {(e["parent"], e["child"], e.get("qkv")) for e in self.edges}
        self.edges = sorted(({"parent": p, "child": c, "qkv": q} for p, c, q in uniq),
                            key=_edge_key)
        nodes = sorted(_nodes_of(self.edges), key=_node_layer)
        heads, mlps = SFC._load_fc_eap().heads_mlps_from_nodes(nodes)
        self.nodes, self.heads, self.mlps = nodes, set(heads), set(mlps)
        decl = SFC._edges_to_declared(self.edges)
        self.declared = {k: sorted(v, key=_sender_key) for k, v in decl.items()}


def _within(e: dict, top: int) -> bool:
    """Both endpoints inside the graph truncated at layer ``top`` (logits = the terminal)."""
    return (_node_layer(e["parent"])[0] <= top
            and (e["child"] == "logits" or _node_layer(e["child"])[0] <= top))


def _check_causal(edges: list, top: int) -> None:
    """Every edge's writer precedes its reader (a head feeds its own layer's MLP, not its
    own layer's heads), inside the graph truncated at layer ``top``."""
    for e in edges:
        p, c = _node_layer(e["parent"]), _node_layer(e["child"])
        if not (_within(e, top) and (e["child"] == "logits" or p[:2] < c[:2])):
            raise ValueError(f"edge {e['parent']} -> {e['child']} is not a causal edge of "
                             f"the graph truncated at layer {top}")


def _parse_edges(raw) -> list:
    """Edges from any supported JSON: a finder dialect (circuit_loader-sniffable), a
    smallest_circuit.json / evaluation.json style ``edges`` list (dicts with parent/child/qkv, or
    [sender, receiver] pairs with receivers like ``a1.h3<q>``), or a bare edge list."""
    dialect = SFC._sniff_dialect(raw)
    if dialect is not None:
        return SFC._full_edges(raw, dialect)
    items = raw.get("edges") if isinstance(raw, dict) else raw
    if not isinstance(items, list):
        raise ValueError("no edge list found in candidate JSON")
    out = []
    for e in items:
        if isinstance(e, dict):
            p, c, q = SFC._node_name(e["parent"]), SFC._node_name(e["child"]), e.get("qkv")
        else:
            p, c, q = SFC._parse_edge_str(f"{e[0]}->{e[1]}")
        if not (p and c and (p == "input" or p[0] in "am") and (c == "logits" or c[0] in "am")):
            raise ValueError(f"unrecognized edge {e!r}")
        out.append({"parent": p, "child": c, "qkv": q, "score": 0.0})
    return out


def _pruned(name: str, edges: list, source: str, attn_only: bool, meta=None) -> Circuit:
    """Dangling-node pruning (+ phantom MLP strip on attn-only models) exactly as the
    pipeline's selection stage does it (``smallest_faithful_circuit._prune_candidate``)."""
    heads, mlps = SFC._load_fc_eap().heads_mlps_from_nodes(sorted(_nodes_of(edges)))
    cand = SFC._prune_candidate(SFC.Candidate(name, source, "evaluation", heads,
                                              [dict(e, score=0.0) for e in edges], mlps),
                                attn_only)
    return Circuit(name, cand.edges, source, meta or {})


def load_candidate(path: str, attn_only: bool, prune: bool = True) -> Circuit:
    """A candidate dir (finder output) or circuit JSON; pruned unless ``prune=False``."""
    if os.path.isdir(path):
        cand = SFC._sniff_candidate_dir(path, os.path.basename(path.rstrip("/")), attn_only)
        if cand is None:
            raise ValueError(f"no parseable circuit JSON in {path}")
        return Circuit(cand.name, cand.edges, path)
    with open(path) as f:
        raw = json.load(f)
    name = os.path.splitext(os.path.basename(path))[0]
    meta = {k: raw[k] for k in ("mode", "base_circuit") if isinstance(raw, dict) and k in raw}
    edges = _parse_edges(raw)
    return _pruned(name, edges, path, attn_only, meta) if prune else Circuit(name, edges, path, meta)


def _edge_set(c: Circuit) -> set:
    return {(e["parent"], e["child"], e["qkv"]) for e in c.edges}


def _from_set(name: str, keys: set, source: str = "") -> Circuit:
    return Circuit(name, [{"parent": p, "child": c, "qkv": q} for p, c, q in keys], source)


# --------------------------------------------------------------------------- #
# Forward passes                                                               #
# --------------------------------------------------------------------------- #
def _cache_names(top: int, attn_only: bool) -> list:
    names = [f"blocks.{l}.hook_resid_pre" for l in range(top + 1)]
    names += [f"blocks.{l}.attn.hook_result" for l in range(top + 1)]
    names += [f"blocks.{top}.hook_resid_post"]
    if not attn_only:
        names += [f"blocks.{l}.hook_mlp_in" for l in range(top + 1)]
        names += [f"blocks.{l}.hook_mlp_out" for l in range(top + 1)]
    return names


def _row_base(cache, top: int, attn_only: bool) -> dict:
    """Per-row baselines (one partner forward) for ``_patched_forward``."""
    rng = range(top + 1)
    base = {"input": cache["blocks.0.hook_resid_pre"],
            "rpre": {l: cache[f"blocks.{l}.hook_resid_pre"] for l in rng},
            "head": {l: cache[f"blocks.{l}.attn.hook_result"] for l in rng},
            "term": cache[f"blocks.{top}.hook_resid_post"]}
    if not attn_only:
        base["rmid"] = {l: cache[f"blocks.{l}.hook_mlp_in"] for l in rng}
        base["mlp"] = {l: cache[f"blocks.{l}.hook_mlp_out"] for l in rng}
    return base


def _mean_base(means: dict, top: int, n_layers: int) -> dict:
    """``_edge_site_means`` output as broadcastable [1, S, ...] baselines for ``_patched_forward``;
    the terminal at layer ``top`` < L-1 is resid_post[top] == resid_pre[top+1]."""
    term = means["m_rpost"] if top == n_layers - 1 else means["m_rpre"][top + 1]
    base = {"input": means["mean_input"][None], "term": term[None],
            "rpre": {l: v[None] for l, v in means["m_rpre"].items()},
            "head": {l: v[None] for l, v in means["mean_head"].items()}}
    if "mean_mlp" in means:
        base["rmid"] = {l: v[None] for l, v in means["m_rmid"].items()}
        base["mlp"] = {l: v[None] for l, v in means["mean_mlp"].items()}
    return base


def _patched_forward(model, tokens, c: Circuit, base: dict, top: int | None = None):
    """Edge-level forward against a PER-ROW or broadcast baseline (a local generalization of
    ``smallest_faithful_circuit._edge_forward``, whose baselines are per-position means only).
    ``top=None``: full graph, returns logits. ``top=l``: graph truncated at layer l, returns the
    reconstructed resid_post[l] (the probe terminal)."""
    L, H = model.cfg.n_layers, model.cfg.n_heads
    attn_only = bool(model.cfg.attn_only)
    T = L - 1 if top is None else top
    live = {"head": {}, "mlp": {}}

    def delta(sid):
        if sid == "input":
            return live["input"] - base["input"]
        if sid[0] == "m":
            return live["mlp"][sid[1]] - base["mlp"][sid[1]]
        return live["head"][sid[0]][:, :, sid[1]] - base["head"][sid[0]][:, :, sid[1]]

    def recon(site_base, decl, like):
        r = site_base.expand_as(like).clone()
        for sid in decl:
            r = r + delta(sid)
        return r

    def cap(store, key):
        def f(value, hook):
            store[key] = value
            return value
        return f

    def ow_qkv(l, chan):
        def f(value, hook):
            for h in range(H):
                if (l, h) in c.heads:
                    value[:, :, h] = recon(base["rpre"][l], c.declared.get((l, h, chan), ()),
                                           value[:, :, h])
            return value
        return f

    def ow_mlp(l):
        def f(value, hook):
            return recon(base["rmid"][l], c.declared.get(("m", l), ()), value) \
                if l in c.mlps else value
        return f

    hooks = [("blocks.0.hook_resid_pre", cap(live, "input"))]
    for l in range(T + 1):
        hooks += [(f"blocks.{l}.hook_{ch}_input", ow_qkv(l, ch)) for ch in "qkv"]
        hooks.append((f"blocks.{l}.attn.hook_result", cap(live["head"], l)))
        if not attn_only:
            hooks.append((f"blocks.{l}.hook_mlp_in", ow_mlp(l)))
            hooks.append((f"blocks.{l}.hook_mlp_out", cap(live["mlp"], l)))
    hooks.append((f"blocks.{T}.hook_resid_post",
                  lambda v, hook: recon(base["term"], c.declared.get("final", ()), v)))
    _hook_mode(model, True, True)
    if top is None:
        return model.run_with_hooks(tokens, return_type="logits", fwd_hooks=hooks)
    return model.run_with_hooks(tokens, stop_at_layer=T + 1, fwd_hooks=hooks)


def _hook_mode(model, split_qkv: bool, head_results: bool) -> None:
    """Per-head q/k/v inputs and per-head outputs cost ~9x a plain forward, so they are switched
    on only for the forwards that read them (edge-level forwards; head-output caches)."""
    model.cfg.use_split_qkv_input, model.cfg.use_attn_result = split_qkv, head_results


def _site_means(model, toks) -> dict:
    _hook_mode(model, False, True)
    return SFC._edge_site_means(model, toks, BATCH)


def _plain(model, *args, **kw):
    _hook_mode(model, False, False)
    return model(*args, **kw)


def _mean_forward(model, tokens, c: Circuit, means: dict):
    """The pipeline's edge-level per-position mean-ablated forward (logits)."""
    L, H = model.cfg.n_layers, model.cfg.n_heads
    _hook_mode(model, True, True)
    return SFC._edge_forward(model, tokens, c.declared, means, c.heads, c.mlps, L, H,
                             bool(model.cfg.attn_only))


def _batched(fn, *tensors) -> torch.Tensor:
    """``fn`` over row batches of ``tensors`` (moved to the model device); result on CPU."""
    with torch.no_grad():
        return torch.cat([fn(*(t[i:i + BATCH].to(DEVICE) for t in tensors)).cpu()
                          for i in range(0, tensors[0].shape[0], BATCH)])


# --------------------------------------------------------------------------- #
# Class evaluators: each returns (rows aligned with ``circuits``, info dict)   #
# --------------------------------------------------------------------------- #
@dataclass
class Ctx:
    model: object
    spec: object
    kind: str                     # circuit class name
    seed: int
    n_rows: int
    n_pairs: int
    inst: object = None           # built DomainCircuit (non-probe classes)
    var: object = None            # ProbeVariable instance (probe classes)
    probe: dict = None            # probe artifact (probe classes)
    readout: str = "position"

    def stream(self, tag: str) -> int:
        return conditions.stream_key(self.seed, f"evaluate_circuit:{tag}")

    @property
    def out_idx(self) -> list:
        idx = getattr(self.inst, "scored_output_indices", None)
        return list(range(self.spec.n_ctx - self.spec.input_len)) if idx is None else list(idx)

    @property
    def positions(self) -> torch.Tensor:
        return torch.tensor(self.spec.answer_positions)[self.out_idx]

    def teacher(self, x, y=None) -> torch.Tensor:
        return torch.cat([x, self.spec.task_fn(x) if y is None else y], dim=1)


def _frac(ok: torch.Tensor) -> float:
    return int(ok.sum()) / max(1, ok.numel())


def _seq_ok(logits, ctx: Ctx, labels) -> torch.Tensor:
    return (logits[:, ctx.positions.to(logits.device)].argmax(-1) == labels).all(dim=1)


def _argmax_labels(ctx: Ctx, toks) -> torch.Tensor:
    return _batched(lambda t: _plain(ctx.model, t)[:, ctx.positions.to(t.device)].argmax(-1), toks)


def eval_domain(ctx: Ctx, circuits: list) -> tuple:
    x_ref = ctx.inst.sample(ctx.n_rows, ctx.stream("domain:means"))
    x = ctx.inst.sample(ctx.n_rows, ctx.stream("domain:eval"))
    means = _site_means(ctx.model, ctx.teacher(x_ref))
    toks, labels = ctx.teacher(x), ctx.spec.task_fn(x)[:, ctx.out_idx]
    model_acc = _batched(lambda t, y: _seq_ok(_plain(ctx.model, t), ctx, y), toks, labels)
    rows = [{"a": _frac(_batched(lambda t, y: _seq_ok(_mean_forward(ctx.model, t, c, means), ctx,
                                                      y), toks, labels))} for c in circuits]
    acc = _frac(model_acc)
    return rows, {"ablation": "mean", "n_rows": ctx.n_rows, "model_seq_acc": acc,
                  "model_reference": {"a": acc, "a_clean": acc, "a_corrupt": None}}


def _dedup_pairs(ctx: Ctx, n: int, tag: str) -> tuple:
    clean, corrupt = ctx.inst.sample_pairs(n, ctx.stream(tag))
    seen, keep = set(), []
    for i, key in enumerate(map(tuple, torch.cat([clean, corrupt], 1).tolist())):
        if key not in seen:
            seen.add(key)
            keep.append(i)
    return clean[keep], corrupt[keep], clean.shape[0] - len(keep)


def _paired_mean(ctx: Ctx, circuits: list, clean, corrupt) -> tuple:
    """Alternating pairs under mean ablation: means over the mixed clean+corrupt rows of a
    separate draw; each row teacher-forced and labelled with its own DSL output."""
    c_ref, k_ref, _ = _dedup_pairs(ctx, max(1, ctx.n_rows // 2), "paired:means")
    means = _site_means(ctx.model, torch.cat([ctx.teacher(c_ref), ctx.teacher(k_ref)]))

    def score(fwd) -> dict:
        acc = [_frac(_batched(lambda t, y: _seq_ok(fwd(t), ctx, y), ctx.teacher(x),
                              ctx.spec.task_fn(x)[:, ctx.out_idx])) for x in (clean, corrupt)]
        return {"a": acc[0] * acc[1], "a_clean": acc[0], "a_corrupt": acc[1]}

    rows = [score(lambda t, c=c: _mean_forward(ctx.model, t, c, means)) for c in circuits]
    return rows, score(lambda t: _plain(ctx.model, t))          # (circuits, unablated model)


def _pass_through(ctx: Ctx, clean, corrupt, lab_c, lab_k) -> torch.Tensor:
    """Pairs whose only label difference is the single edited digit copied to its own output
    slot (output slot k <-> input index k + input_len - 10, i.e. after a TaskMux cue)."""
    out = torch.zeros(clean.shape[0], dtype=torch.bool)
    offset, slots = ctx.spec.input_len - VARS.INPUT_LEN, ctx.out_idx
    for r in range(clean.shape[0]):
        edit = (clean[r] != corrupt[r]).nonzero().flatten()
        diff = (lab_c[r] != lab_k[r]).nonzero().flatten()
        if edit.numel() != 1 or diff.numel() != 1 or int(edit) - offset not in slots:
            continue
        i, j = int(edit), slots.index(int(edit) - offset)
        out[r] = (int(diff) == j and int(lab_c[r, j]) == int(clean[r, i])
                  and int(lab_k[r, j]) == int(corrupt[r, i]))
    return out


def eval_contrast(ctx: Ctx, circuits: list, primary: str) -> tuple:
    clean, corrupt, n_dup = _dedup_pairs(ctx, ctx.n_pairs, "contrast:eval")
    y = ctx.spec.task_fn(clean)                     # shared teacher-forced prefix
    tc, tk = ctx.teacher(clean, y), ctx.teacher(corrupt, y)
    lab_c, lab_k = _argmax_labels(ctx, tc), _argmax_labels(ctx, tk)
    identical = (lab_c == lab_k).all(dim=1)
    passthru = _pass_through(ctx, clean, corrupt, lab_c, lab_k) & ~identical
    keep = ~identical & ~passthru
    tc, tk, lab_c, lab_k = tc[keep], tk[keep], lab_c[keep], lab_k[keep]
    L, attn_only = ctx.model.cfg.n_layers, bool(ctx.model.cfg.attn_only)
    names = _cache_names(L - 1, attn_only)
    fwd = [[] for _ in circuits]
    rev = [[] for _ in circuits]
    with torch.no_grad():
        for i in range(0, tc.shape[0], BATCH):
            bc_, bk_, yc, yk = (t[i:i + BATCH].to(DEVICE) for t in (tc, tk, lab_c, lab_k))
            _hook_mode(ctx.model, False, True)
            base_c = _row_base(ctx.model.run_with_cache(bc_, names_filter=names)[1], L - 1, attn_only)
            base_k = _row_base(ctx.model.run_with_cache(bk_, names_filter=names)[1], L - 1, attn_only)
            for j, c in enumerate(circuits):
                fwd[j].append(_seq_ok(_patched_forward(ctx.model, bc_, c, base_k), ctx, yc).cpu())
                rev[j].append(_seq_ok(_patched_forward(ctx.model, bk_, c, base_c), ctx, yk).cpu())
    inter = []
    for f, r in zip(fwd, rev):
        a_c = _frac(torch.cat(f)) if f else 0.0
        a_k = _frac(torch.cat(r)) if r else 0.0
        inter.append({"a": a_c * a_k, "a_clean": a_c, "a_corrupt": a_k})
    mean, mean_model = _paired_mean(ctx, circuits, clean, corrupt)
    model_ref = {"a": 1.0, "a_clean": 1.0, "a_corrupt": 1.0}   # labels ARE the model's outputs
    info = {"n_pairs": ctx.n_pairs, "pairs_kept": int(keep.sum()),
            "model_reference": model_ref if primary == "interchange" else mean_model,
            "pairs_dropped": {"duplicate": n_dup, "identical_labels": int(identical.sum()),
                              "pass_through": int(passthru.sum())}}
    first, second = (inter, mean) if primary == "interchange" else (mean, inter)
    other = "mean" if primary == "interchange" else "interchange"
    for row, sec in zip(first, second):
        row["secondary"] = {"ablation": other, f"{other}_ablation_a": sec["a"],
                            "a_clean": sec["a_clean"], "a_corrupt": sec["a_corrupt"],
                            "model_a": (mean_model if other == "mean" else model_ref)["a"]}
    return first, dict(info, ablation=primary, n_rows=ctx.n_rows)


def eval_suppression(ctx: Ctx, circuits: list) -> tuple:
    clean, corrupt, n_dup = _dedup_pairs(ctx, ctx.n_pairs, "suppression:eval")
    rows, model_ref = _paired_mean(ctx, circuits, clean, corrupt)
    return rows, {"ablation": "mean", "n_rows": ctx.n_rows, "n_pairs": ctx.n_pairs,
                  "pairs_kept": int(clean.shape[0]), "pairs_dropped": {"duplicate": n_dup},
                  "model_reference": model_ref}


def _probe_logits(ctx: Ctx, resid) -> torch.Tensor:
    x = TP.ln(resid[:, ctx.probe["positions"]])
    if ctx.readout == "pooled":
        return x @ ctx.probe["W"].T + ctx.probe["b"]
    return torch.einsum("bpd,pcd->bpc", x, ctx.probe["W_pos"]) + ctx.probe["b_pos"]


def _probe_labels(ctx: Ctx, x) -> torch.Tensor:
    declared = ctx.var.valid_positions().tolist()
    return ctx.var.labels(x)[:, [declared.index(p) for p in ctx.probe["positions"].tolist()]]


def _full_resid(ctx: Ctx, x) -> torch.Tensor:
    return _batched(lambda t: _plain(ctx.model, t, stop_at_layer=ctx.probe["layer"] + 1), x)


def _majority(labels) -> float:
    """Accuracy of the best constant predictor per probed position (position mean)."""
    return sum(labels[:, j].bincount().max().item() for j in range(labels.shape[1])) / labels.numel()


def _strongest_edit(ctx: Ctx, x) -> torch.Tensor:
    """Per row: the single-digit edit maximizing the full model's mean true-class probe margin."""
    best, best_m = x.clone(), torch.full((x.shape[0],), -float("inf"))
    for i in range(x.shape[1]):
        for d in range(10):
            xe = x.clone()
            xe[:, i] = d
            lg, lab = _probe_logits(ctx, _full_resid(ctx, xe)), _probe_labels(ctx, xe)
            true = lg.gather(-1, lab[..., None])[..., 0]
            other = lg.scatter(-1, lab[..., None], -float("inf")).max(-1).values
            m = (true - other).mean(dim=1).masked_fill(x[:, i] == d, -float("inf"))
            better = m > best_m
            best[better], best_m[better] = xe[better], m[better]
    return best


def eval_probe(ctx: Ctx, circuits: list) -> tuple:
    layer = int(ctx.probe["layer"])
    x_ref = TP.sample_inputs(ctx.n_rows, ctx.stream("probe:means"))
    x = TP.sample_inputs(ctx.n_pairs, ctx.stream("probe:eval"))
    means = _site_means(ctx.model, x_ref)                      # input positions only (causal)
    base = _mean_base(means, layer, ctx.model.cfg.n_layers)
    sides = {"clean": x, "corrupt": _strongest_edit(ctx, x)}
    rows = [{} for _ in circuits]
    info = {"ablation": "mean", "n_rows": ctx.n_rows, "n_pairs": ctx.n_pairs, "probe": {},
            "model_reference": {"a": 1.0, "a_clean": 1.0, "a_corrupt": 1.0}}   # normalization
    for side, xs in sides.items():
        lab = _probe_labels(ctx, xs)
        acc = lambda resid: _frac(_probe_logits(ctx, resid).argmax(-1) == lab)
        maj, full = _majority(lab), acc(_full_resid(ctx, xs))
        info["probe"][f"majority_acc_{side}"], info["probe"][f"full_acc_{side}"] = maj, full
        for row, c in zip(rows, circuits):
            raw = acc(_batched(lambda t: _patched_forward(ctx.model, t, c, base, layer), xs))
            row[f"probe_acc_{side}"] = raw
            row[f"a_{side}"] = max(0.0, (raw - maj) / (full - maj)) if full > maj else 0.0
    for row in rows:
        row["a"] = row["a_clean"] * row["a_corrupt"]
    info["probe"].update(majority_acc=info["probe"]["majority_acc_clean"],
                         full_acc=info["probe"]["full_acc_clean"])
    return rows, info


# --------------------------------------------------------------------------- #
# Resolution, size accounting, output                                          #
# --------------------------------------------------------------------------- #
def resolve(model_name: str, circuit_id: str):
    """(circuit class, built circuit | None, probe variable class | None)."""
    for inst in gen_circuit_data.circuits_for(model_name):
        if inst.circuit_id == circuit_id:
            if isinstance(inst, dsl_circuits.SuppressionContrastCircuit):
                return "SuppressionCircuit", inst, None
            if isinstance(inst, dsl_circuits.ContrastCircuit):
                return "ContrastCircuit", inst, None
            return "DomainCircuit", inst, None
    try:
        var = TP.resolve_variable(model_name, circuit_id)
    except KeyError:
        m = re.fullmatch(r"(bridge\d+)_.+", circuit_id)
        if not m:
            raise
        var = TP.resolve_variable(model_name, m.group(1))
    return ("BridgeCircuit" if hasattr(var, "_split_k") else "ProbeCircuit"), None, var


def load_probe(model_name: str, variable_id: str, layer: int | None) -> dict:
    """The probes stage's artifact at the best layer of its site scan (highest mean per-position
    accuracy, ties -> lower layer) unless ``layer`` is given."""
    vdir = TP.probe_dir(model_name, variable_id)
    if layer is None:
        with open(os.path.join(vdir, "site_scan.json")) as f:
            acc = {int(k): v for k, v in json.load(f)["per_layer_accuracy"].items()}
        layer = min(acc, key=lambda l: (-acc[l], l))
    art = torch.load(os.path.join(vdir, f"site_L{layer}.pt"), map_location="cpu",
                     weights_only=False)
    assert art["hook"] == f"blocks.{layer}.hook_resid_post" and art["normalize"] == "ln"
    art = dict(art)
    art["positions"] = torch.as_tensor(art["positions"], dtype=torch.long)
    return art


def _rel(path: str) -> str:
    p = os.path.abspath(path)
    return os.path.relpath(p, REPO_ROOT) if p.startswith(REPO_ROOT + os.sep) else \
        f"<external>/{os.path.basename(p.rstrip('/'))}"


def _command(argv: list) -> str:
    args = [_rel(a) if os.path.isabs(a) else a for a in argv]
    return " ".join(["python", "pipeline/find_circuits/evaluate_circuit.py"] + args)


def _size(c: Circuit, full: Circuit) -> dict:
    n_nodes, n_edges = len(c.nodes), len(c.edges)
    return {"n_nodes": n_nodes, "n_edges": n_edges,
            "selection_score": (1 - n_nodes / len(full.nodes)) * (1 - n_edges / len(full.edges))}


def _describe(c: Circuit) -> dict:
    return {"nodes": c.nodes,
            "edges": [[e["parent"], e["child"] + (f"<{e['qkv']}>" if e["qkv"] else "")]
                      for e in c.edges]}


def _base_edges(args, kind: str, inst, attn_only: bool):
    """(base circuit name, Circuit | None) for the fringe: --base-candidate, else the base's
    evaluation.json / smallest_circuit.json under circuits_data."""
    base = getattr(inst, "base_circuit", None) if kind in ("ContrastCircuit",
                                                           "SuppressionCircuit") else None
    if base is None:
        return None, None
    paths = [args.base_candidate] if args.base_candidate else [
        os.path.join(MG.data_dir(args.model, base), f) for f in ("evaluation.json",
                                                                 "smallest_circuit.json")]
    for p in paths:
        if p and os.path.exists(p):
            try:
                return base, load_candidate(p, attn_only)
            except (ValueError, KeyError):
                continue
    return base, None


def _fringe(c: Circuit, base_name, base: Circuit | None) -> dict:
    if base_name is None:
        return {}
    if base is None:
        return {"base_circuit": base_name, "available": False}
    f = _from_set("fringe", _edge_set(c) - _edge_set(base))
    nodes = sorted(_nodes_of(f.edges), key=_node_layer)
    return {"base_circuit": base_name, "available": True, "base_source": _rel(base.source),
            "n_edges": len(f.edges), "n_nodes": len(nodes), "nodes": nodes,
            "edges": _describe(f)["edges"]}


def candidate_paths(args, kind: str, var_id: str | None) -> list:
    if kind in ("ProbeCircuit", "BridgeCircuit"):
        pats = [os.path.join(MG.data_dir(args.model), "*", "circuits", f"*_probe_{var_id}_*"),
                os.path.join(TP.probe_dir(args.model, var_id), "circuits", "*")]
        return sorted(p for pat in pats for p in glob.glob(pat) if os.path.isdir(p))
    d = os.path.join(MG.data_dir(args.model, args.circuit), "circuits")
    return [os.path.join(d, s) for s in sorted(os.listdir(d)) if os.path.isdir(os.path.join(d, s))
            and not SFC._is_variant_candidate_dir(s)] if os.path.isdir(d) else []


def run(args) -> tuple:
    global DEVICE
    t0 = time.time()
    DEVICE = args.device
    torch.set_num_threads(args.threads)      # fixed thread count: reproducible float reductions
    torch.manual_seed(args.seed)
    if args.probes_root:
        TP.PROBES_ROOT = args.probes_root
    spec = MG.get_spec(args.model)
    model = (MG.load_segrev3(checkpoint_path=args.checkpoint, device=args.device)
             if args.checkpoint else MG.load_model(args.model, device=args.device))
    attn_only = bool(model.cfg.attn_only)
    if not attn_only:
        model.cfg.use_hook_mlp_in = True
    L, H = model.cfg.n_layers, model.cfg.n_heads

    kind, inst, var_cls = resolve(args.model, args.circuit)
    ablation = {"ContrastCircuit": "interchange"}.get(kind, "mean") \
        if args.ablation == "auto" else args.ablation
    if ablation == "interchange" and kind != "ContrastCircuit":
        raise SystemExit(f"--ablation interchange is defined for ContrastCircuits only ({kind})")
    ctx = Ctx(model, spec, kind, args.seed, args.n_rows, args.n_pairs, inst=inst,
              readout=args.probe_readout)
    top = L - 1
    if var_cls is not None:
        ctx.var = var_cls()
        ctx.probe = load_probe(args.model, var_cls.variable_id, args.probe_layer)
        top = int(ctx.probe["layer"])
    full_graph = graph_edges(top + 1, H, attn_only)
    full = Circuit("full", full_graph, "full")
    empty = Circuit("empty", [], "empty")

    def prep(c: Circuit) -> Circuit:
        if top < L - 1:                       # probe classes: drop edges above the probe layer
            keep = [e for e in c.edges if _within(e, top)]
            if len(keep) < len(c.edges):
                c.meta["n_edges_dropped_above_probe_layer"] = len(c.edges) - len(keep)
                c = Circuit(c.name, keep, c.source, c.meta)
        _check_causal(c.edges, top)
        return c

    if args.select or args.candidate is None:
        paths = candidate_paths(args, kind, getattr(var_cls, "variable_id", None))
        if not paths:
            raise SystemExit(f"no candidates found for {args.model}/{args.circuit}; "
                             "pass --candidate <dir|json> (or 'full' / 'empty')")
        cands = [load_candidate(p, attn_only) for p in paths]
    elif args.candidate in ("full", "empty"):
        cands = [{"full": full, "empty": empty}[args.candidate]]
    else:
        cands = [load_candidate(args.candidate, attn_only)]
    base_name, base = _base_edges(args, kind, inst, attn_only)
    if args.union_base:                       # union BEFORE pruning: a fringe alone dangles
        if base is None:
            raise SystemExit(f"--union-base: no base circuit edges found for {base_name!r}")
        raw = [load_candidate(c.source, attn_only, prune=False) if os.path.exists(c.source)
               else c for c in cands]
        cands = [_pruned(c.name, _from_set(c.name, _edge_set(c) | _edge_set(base)).edges,
                         c.source, attn_only, c.meta) for c in raw]
    cands = [prep(c) for c in cands]

    circuits = cands + ([] if any(c is empty for c in cands) else [empty])
    if kind == "DomainCircuit":
        rows, info = eval_domain(ctx, circuits)
    elif kind == "ContrastCircuit":
        rows, info = eval_contrast(ctx, circuits, ablation)
    elif kind == "SuppressionCircuit":
        rows, info = eval_suppression(ctx, circuits)
    else:
        rows, info = eval_probe(ctx, circuits)
    floor = FLOORS[kind]
    for c, row in zip(circuits, rows):
        row.update(_size(c, full), passes=row["a"] > floor)
        row.setdefault("a_clean", row["a"] if kind == "DomainCircuit" else None)
        row.setdefault("a_corrupt", None)
    row_of = {id(c): r for c, r in zip(circuits, rows)}

    def detail(c: Circuit, row: dict) -> dict:
        out = {"candidate": c.name if c.source in ("full", "empty") else _rel(c.source),
               **{k: v for k, v in row.items() if k != "secondary"}, **_describe(c)}
        if "secondary" in row:
            out["secondary"] = row["secondary"]
        if c.meta:
            out["candidate_meta"] = c.meta
        if base_name is not None:
            out["fringe"] = _fringe(c, base_name, base)
        return out

    payload = {
        "model": args.model, "circuit": args.circuit, "class": kind,
        "repo_class": type(inst).__name__ if inst is not None else var_cls.__name__,
        "ablation": info["ablation"], "floor": floor, "seed": args.seed,
        "n_nodes_full": len(full.nodes), "n_edges_full": len(full.edges),
        "node_definition": NODE_DEFINITION, "edge_definition": EDGE_DEFINITION,
        "reference": {"model": info.pop("model_reference"),
                      "empty": {k: row_of[id(empty)][k] for k in ("a", "a_clean", "a_corrupt")},
                      "note": REFERENCE_NOTE},
        "command": _command(args.argv),
        **{k: v for k, v in info.items() if k != "ablation"},
    }
    if var_cls is not None:
        payload["probe"].update(layer=top, readout=args.probe_readout,
                                positions=ctx.probe["positions"].tolist(),
                                n_classes=int(ctx.probe["n_classes"]),
                                variable=var_cls.variable_id)
    if args.select or args.candidate is None:
        table = [{"candidate": _rel(c.source), **{k: r[k] for k in (
            "a", "a_clean", "a_corrupt", "passes", "n_nodes", "n_edges", "selection_score")}}
            for c, r in zip(cands, rows)]
        elig = [i for i, r in enumerate(rows[:len(cands)]) if r["passes"]]
        win = min(elig, key=lambda i: (-rows[i]["selection_score"], rows[i]["n_edges"],
                                       rows[i]["n_nodes"], -rows[i]["a"], cands[i].name),
                  default=None)
        payload.update(candidates=table, winner=None if win is None else _rel(cands[win].source))
        if win is not None:
            payload.update(detail(cands[win], rows[win]))
        else:
            payload["passes"] = False
    else:
        payload.update(detail(cands[0], rows[0]))
    out = args.out or (os.path.join(TP.probe_dir(args.model, var_cls.variable_id), "evaluation.json")
                       if var_cls is not None else
                       os.path.join(MG.data_dir(args.model, args.circuit), "evaluation.json"))
    print(f"[evaluate_circuit] {args.model}/{args.circuit} ({kind}, {info['ablation']}): "
          f"{len(cands)} candidate(s), {time.time() - t0:.1f}s")
    for c, r in zip(circuits, rows):
        print(f"    {c.name:<28} a={r['a']:.4f} (clean={r['a_clean']}, corrupt={r['a_corrupt']}) "
              f"nodes={r['n_nodes']} edges={r['n_edges']} score={r['selection_score']:.4f} "
              f"passes={r['passes']}")
    return payload, out


def dump_json(payload: dict, path: str) -> None:
    """Pipeline float rounding + sorted keys (``smallest_faithful_circuit._round_floats``), one
    edge per line."""
    text = json.dumps(SFC._round_floats(payload), indent=2, sort_keys=True)
    text = re.sub(r'\[\s+("[^"]*"),\s+("[^"]*")\s+\]', r"[\1, \2]", text)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        f.write(text + "\n")


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True, help="model id (a DSL algorithm_id + variant)")
    ap.add_argument("--circuit", required=True,
                    help="registered circuit id, or a probe variable id (bridge<k> accepted)")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--candidate", default=None,
                   help="candidate circuit dir or JSON (finder output, smallest_circuit.json, "
                        "evaluation.json), or 'full' / 'empty'")
    g.add_argument("--select", action="store_true",
                   help="score every candidate under the circuit's candidates dir and pick the "
                        "winner (the default when --candidate is not given)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-rows", type=int, default=10000,
                    help="rows for the mean-ablation reference (and the DomainCircuit eval set)")
    ap.add_argument("--n-pairs", type=int, default=5000,
                    help="clean/corrupt pairs for the paired classes")
    ap.add_argument("--ablation", choices=("auto", "mean", "interchange"), default="auto",
                    help="auto = interchange for ContrastCircuits, mean otherwise")
    ap.add_argument("--out", default=None, help="output JSON (default: evaluation.json in the "
                                                "circuit's circuits_data / probes_data dir)")
    ap.add_argument("--threads", type=int, default=4,
                    help="torch CPU threads (fixed for reproducible float reductions)")
    ap.add_argument("--checkpoint", default=None,
                    help="checkpoint override (default: models/trained_models_pt/<model>.pt)")
    ap.add_argument("--probes-root", default=None,
                    help="probe artifacts root (default: pipeline/probes_data)")
    ap.add_argument("--probe-layer", type=int, default=None,
                    help="probe layer override (default: best layer of the site scan)")
    ap.add_argument("--probe-readout", choices=("position", "pooled"), default="position",
                    help="per-position probes (default) or the pooled probe")
    ap.add_argument("--base-candidate", default=None,
                    help="edges of the base circuit for the fringe (default: the base's "
                         "evaluation.json / smallest_circuit.json)")
    ap.add_argument("--union-base", action="store_true",
                    help="evaluate candidate edges UNION the base circuit's edges (for stored "
                         "fringe-only circuits)")
    a = ap.parse_args(argv)
    a.argv = list(sys.argv[1:] if argv is None else argv)
    return a


def main(argv=None) -> int:
    payload, out = run(parse_args(argv))
    dump_json(payload, out)
    print(f"    wrote {_rel(out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
