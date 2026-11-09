"""
eap.py
=================================================================================
EAP / EAP-IG edge attribution for the ``segrev3`` task.

Runs the EAP-IG library (``EAP-IG/src/eap``) on the segrev3 attention-only model
to discover an edge-scored circuit, selects top-n circuits, measures their
faithfulness against the full-model baseline, and exports the graph as JSON / PT
/ PNG plus a faithfulness summary to ``visualizations/eap/``.

--------------------------------------------------------------------------------
WHY THIS NEEDS TWO MONKEYPATCHES (the EAP-IG repo is READ-ONLY)
--------------------------------------------------------------------------------
1. NO TOKENIZER.  segrev3 has ``tokenizer_name=None``; the model's vocab IS the
   set of token ids (digits 0-9).  EAP-IG's ``utils.tokenize_plus`` calls
   ``model.to_tokens`` + ``get_attention_mask(model.tokenizer, ...)`` which both
   require a real tokenizer.  We instead feed *pre-tokenized* integer tensors
   ``[B, 20]`` and monkeypatch ``tokenize_plus`` (in every eap module that
   imported it) to a pass-through that returns ``(tokens, ones_mask,
   full_lengths, n_pos=20)``.  Our "dataloader" is just a list of
   ``(clean_tokens[B,20], corrupted_tokens[B,20], labels[B,2,10])`` tuples.

2. HARD-CODED CUDA.  Several ``get_scores_*`` helpers create their score
   accumulator with ``torch.zeros(..., device='cuda')`` (e.g. attribute.py:116).
   We wrap the pipeline in a context manager (``force_zeros_device``) that
   rewrites that literal ``'cuda'`` to whatever device the model actually lives
   on -- cuda on a GPU box (fast), cpu on a CPU-only box -- so the accumulators
   always match the model.  ``Graph`` tensors and ``make_hooks_and_matrices``
   already follow ``model.cfg.device``, so no other change is needed.

Everything else (Graph.from_model, attribute, apply_topn, evaluate_graph,
evaluate_baseline, to_json/to_pt/to_image) is the stock EAP-IG code.  The model's
attention-only architecture means the 4 dummy MLP nodes (m0..m3) receive zero
score (their hook points exist as no-op ``HookPoint``s and never fire) and are
pruned away — harmless.

METRIC.  ``segrev3_logit_diff`` is the differentiable mean (correct - incorrect)
logit over ANSWER POSITIONS 9..18, matching ``metrics.metrics.logit_diff`` for
the ``random`` scheme (correct = clean output digit, incorrect = corrupted
output digit).  ``loss=True`` (negated) is used for attribution; ``loss=False``
for evaluation.  We sanity-check it against the task's own ``logit_diff``.

Run (with the requirements installed):  python circuit_discovery/eap.py
"""
from __future__ import annotations

import os
import sys
import json
import argparse
import contextlib
import dataclasses
from functools import partial

import torch
from tqdm import tqdm

# --------------------------------------------------------------------------- #
# Paths                                                                        #
# --------------------------------------------------------------------------- #
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(_THIS_DIR)
sys.path.insert(0, os.path.join(REPO_ROOT, "interp_tools"))
sys.path.insert(0, os.path.join(REPO_ROOT, "data_gen"))
sys.path.insert(0, os.path.join(REPO_ROOT, "metrics"))
sys.path.insert(0, os.path.join(REPO_ROOT, "github_repos", "EAP-IG", "src"))

OUT_DIR = os.path.join(REPO_ROOT, "visualizations", "eap")
os.makedirs(OUT_DIR, exist_ok=True)
# One json (in OUT_DIR) summarising EVERY top-n run thus far (one per eap_top<n>/ folder).
COMBINED_SUMMARY = os.path.join(OUT_DIR, "eap_all_topn_summary.json")


def _topn_folder(n: int) -> str:
    """Per-top-n output folder: visualizations/eap/eap_top<n>/."""
    return os.path.join(OUT_DIR, f"eap_top{n}")


def _load_json(p):
    with open(p) as f:
        return json.load(f)

from model_loading import (assert_memory_headroom, load_segrev3,  # noqa: E402
                           ANSWER_POSITIONS, repo_relpath)
from data_gen import make_clean_corrupted  # noqa: E402
import dataset_store as store  # noqa: E402
from metrics import logit_diff as task_logit_diff  # noqa: E402

# eap modules (import the *modules* so we can monkeypatch their namespaces)
import eap.utils as eap_utils          # noqa: E402
import eap.attribute as eap_attribute  # noqa: E402
import eap.evaluate as eap_evaluate    # noqa: E402
from eap.graph import Graph            # noqa: E402
from eap.attribute import attribute    # noqa: E402
from eap.evaluate import evaluate_graph, evaluate_baseline  # noqa: E402

ANSWER_POS = torch.tensor(ANSWER_POSITIONS, dtype=torch.long)  # [9..18], len 10

# Manual circuit (ground truth for the overlap comparison).
MANUAL_MOVERS = ["a3.h1", "a3.h3", "a3.h6", "a2.h7"]
MANUAL_DETECTORS = ["a0.h7", "a0.h2", "a0.h1", "a0.h4", "a0.h6"]
MANUAL_HEADS = set(MANUAL_MOVERS + MANUAL_DETECTORS)


# --------------------------------------------------------------------------- #
# Monkeypatch 1: tokenize_plus pass-through for pre-tokenized integer tensors  #
# --------------------------------------------------------------------------- #
def tokenize_plus_passthrough(model, inputs, max_length=None):
    """Replacement for eap.utils.tokenize_plus that accepts an already-tokenized
    LongTensor ``[B, S]`` (segrev3 has no tokenizer).  Returns the tokens, an
    all-ones attention mask, per-example lengths (== S, no padding), and n_pos."""
    tokens = inputs if torch.is_tensor(inputs) else torch.as_tensor(inputs)
    tokens = tokens.to(device=model.cfg.device, dtype=torch.long)
    if tokens.ndim == 1:
        tokens = tokens.unsqueeze(0)
    B, S = tokens.shape
    attention_mask = torch.ones((B, S), dtype=torch.long, device=tokens.device)
    input_lengths = torch.full((B,), S, dtype=torch.long, device=tokens.device)
    return tokens, attention_mask, input_lengths, S


def _install_tokenize_patch():
    eap_utils.tokenize_plus = tokenize_plus_passthrough
    eap_attribute.tokenize_plus = tokenize_plus_passthrough
    eap_evaluate.tokenize_plus = tokenize_plus_passthrough


# --------------------------------------------------------------------------- #
# Monkeypatch 2: redirect EAP-IG's hard-coded device='cuda' to OUR device      #
# --------------------------------------------------------------------------- #
# Several EAP-IG ``get_scores_*`` helpers allocate their score accumulator with a
# literal ``torch.zeros(..., device='cuda')``. We rewrite that hard-coded 'cuda'
# to whatever device the model actually lives on, so the accumulators match the
# model: on a GPU box that keeps them on cuda (fast), and on a CPU-only box it
# transparently falls back to cpu. Other EAP tensors already follow
# ``model.cfg.device``.
@contextlib.contextmanager
def force_zeros_device(target):
    target = str(target)
    orig_zeros = torch.zeros

    def patched_zeros(*args, **kwargs):
        dev = kwargs.get("device", None)
        if dev is not None and str(dev).startswith("cuda"):
            kwargs["device"] = target
        return orig_zeros(*args, **kwargs)

    torch.zeros = patched_zeros
    try:
        yield
    finally:
        torch.zeros = orig_zeros


# --------------------------------------------------------------------------- #
# Metric (EAP signature) — differentiable logit-diff over positions 9..18     #
# --------------------------------------------------------------------------- #
def segrev3_logit_diff(logits, clean_logits, input_lengths, labels,
                       mean: bool = True, loss: bool = False):
    """metric(logits, clean_logits, input_lengths, labels) for EAP-IG.

    logits  : [B, 20, 10]
    labels  : [B, 2, 10]  (labels[:,0]=correct/clean digit, labels[:,1]=incorrect)
    Returns mean (correct-incorrect) logit over the 10 answer positions.
    loss=True negates it (lower == better) for attribution / gradient ascent.
    mean=True returns a scalar; mean=False returns per-example [B] (for eval).
    """
    pos = ANSWER_POS.to(logits.device)
    sel = logits[:, pos, :]                                   # [B, 10, 10]
    correct = labels[..., 0, :].to(logits.device)            # [B, 10]
    incorrect = labels[..., 1, :].to(logits.device)          # [B, 10]
    cor = sel.gather(-1, correct.unsqueeze(-1)).squeeze(-1)   # [B, 10]
    inc = sel.gather(-1, incorrect.unsqueeze(-1)).squeeze(-1)
    d = (cor - inc).mean(dim=-1)                             # [B]
    if loss:
        d = -d
    return d.mean() if mean else d


# --------------------------------------------------------------------------- #
# Data: build a list-of-batches "dataloader" from a Segrev3Pairs (random)       #
# --------------------------------------------------------------------------- #
def build_dataloader(n_examples: int, batch_size: int, seed: int, scheme="random"):
    pairs = make_clean_corrupted(n_examples, scheme=scheme, seed=seed)
    return build_dataloader_from_pairs(pairs, batch_size), pairs


def build_dataloader_from_pairs(pairs, batch_size: int):
    """Build the EAP list-of-batches dataloader from a ``Segrev3Pairs`` slice.

    Identical label construction to ``build_dataloader``:
    labels[:, 0] = correct (clean answer), labels[:, 1] = incorrect (corrupt answer)."""
    n_examples = pairs.clean_tokens.shape[0]
    labels = torch.stack([pairs.correct, pairs.incorrect], dim=1).long()  # [B,2,10]
    batches = []
    for i in range(0, n_examples, batch_size):
        batches.append((
            pairs.clean_tokens[i:i + batch_size],
            pairs.corrupted_tokens[i:i + batch_size],
            labels[i:i + batch_size],
        ))
    return batches


def slice_pairs(pairs, start: int, stop: int):
    """Return a ``Segrev3Pairs`` with all tensor fields sliced ``[start:stop]``
    (keeps ``.scheme`` and non-tensor fields)."""
    sliced = {}
    for f in dataclasses.fields(pairs):
        v = getattr(pairs, f.name)
        sliced[f.name] = v[start:stop] if torch.is_tensor(v) else v
    return dataclasses.replace(pairs, **sliced)


# --------------------------------------------------------------------------- #
# Circuit inspection / manual-circuit comparison                              #
# --------------------------------------------------------------------------- #
def head_of(node_name: str):
    """'a3.h1' -> 'a3.h1' ; 'input'/'m0'/'logits' -> None."""
    return node_name if node_name.startswith("a") and ".h" in node_name else None


def top_edges(graph: Graph, k: int):
    """Return list of (edge_name, score) for the top-k edges by |score|."""
    scored = [(e.name, float(e.score)) for e in graph.edges.values()]
    scored.sort(key=lambda x: abs(x[1]), reverse=True)
    return scored[:k]


def node_importance(graph: Graph):
    """Aggregate |edge score| per attention head, split by role.

    out_total : sum |score| over edges where the head is the PARENT (it writes)
    to_logits : |score| of the head's direct edge to logits (copy/mover signal)
    in_total  : sum |score| over edges where the head is the CHILD (it reads)
    """
    out_total, to_logits, in_total = {}, {}, {}
    for e in graph.edges.values():
        s = abs(float(e.score))
        ph, ch = head_of(e.parent.name), head_of(e.child.name)
        if ph is not None:
            out_total[ph] = out_total.get(ph, 0.0) + s
            if e.child.name == "logits":
                to_logits[ph] = to_logits.get(ph, 0.0) + s
        if ch is not None:
            in_total[ch] = in_total.get(ch, 0.0) + s
    return out_total, to_logits, in_total


# --------------------------------------------------------------------------- #
# Combined "thus far" summary (merges every eap_top<n>/ run)                   #
# --------------------------------------------------------------------------- #
def build_eap_combined_summary(this_run_per_topn: dict, run_meta: dict) -> dict:
    """Merge this run's per-top-n findings with previously-recorded ones and write
    ``eap_all_topn_summary.json`` covering EVERY top-n run thus far (one
    ``eap_top<n>/`` folder each; entries whose folder was removed are dropped)."""
    prior = {}
    if os.path.exists(COMBINED_SUMMARY):
        try:
            prior = _load_json(COMBINED_SUMMARY).get("per_topn", {})
        except Exception:
            prior = {}
    merged = dict(prior)
    merged.update(this_run_per_topn)                       # current run overwrites
    per_topn = {k: v for k, v in merged.items() if os.path.isdir(_topn_folder(int(k)))}
    ns = sorted(int(k) for k in per_topn)
    summary = {
        "description": "Combined summary of ALL top-n EAP runs thus far "
                       "(one eap_top<n>/ folder each).",
        **run_meta,
        "manual_movers": MANUAL_MOVERS,
        "manual_detectors": MANUAL_DETECTORS,
        "topns_thus_far": ns,
        "per_topn": {str(n): per_topn[str(n)] for n in ns},
    }
    with open(COMBINED_SUMMARY, "w") as f:
        json.dump(summary, f, indent=2)
    return summary


# --------------------------------------------------------------------------- #
# Main pipeline                                                                #
# --------------------------------------------------------------------------- #
def run(method="EAP-IG-inputs", ig_steps=30, attr_n=200, attr_bs=50,
        eval_n=256, eval_bs=64, topns=(50, 100, 200, 400, 800), seed=0,
        export_n=200, scheme="random"):
    _install_tokenize_patch()

    print("=" * 78)
    print(f"EAP-IG circuit discovery on segrev3 | method={method} ig_steps={ig_steps}")
    print("=" * 78)

    assert_memory_headroom(context="eap.py before attribution")
    model = load_segrev3()
    _dev = next(model.parameters()).device   # match EAP score accumulators to the model
    # CRITICAL flags for EAP-IG hook support (set on this local instance).
    model.cfg.use_attn_result = True
    model.cfg.use_split_qkv_input = True
    model.cfg.use_hook_mlp_in = True

    # ---- data ----
    # Load ONE dataset from the central store and slice it for both attribution
    # and eval (so the whole run is traceable to a single file). The store's eap
    # default is a 512-example dataset, leaving margin to slice attr + eval.
    all_pairs, data_path = store.load("eap", scheme, topn=export_n)
    attr_pairs = slice_pairs(all_pairs, 0, attr_n)
    eval_pairs = slice_pairs(all_pairs, attr_n, attr_n + eval_n)
    attr_dl = build_dataloader_from_pairs(attr_pairs, attr_bs)
    eval_dl = build_dataloader_from_pairs(eval_pairs, eval_bs)
    print(f"loaded data from {os.path.relpath(data_path, REPO_ROOT)}")
    print(f"attribution: {attr_n} ex / bs {attr_bs} ({len(attr_dl)} batches); "
          f"eval: {eval_n} ex / bs {eval_bs} ({len(eval_dl)} batches)")

    # ---- metric sanity check vs task's own logit_diff ----
    with torch.no_grad():
        cl = model(attr_pairs.clean_tokens)
    mine = segrev3_logit_diff(
        cl, None, None,
        torch.stack([attr_pairs.correct, attr_pairs.incorrect], dim=1).long(),
        mean=True, loss=False,
    ).item()
    theirs = task_logit_diff(cl, attr_pairs).item()
    print(f"metric self-check: mine={mine:+.4f}  task.logit_diff={theirs:+.4f}  "
          f"(|diff|={abs(mine - theirs):.2e})")
    assert abs(mine - theirs) < 1e-3, "metric disagrees with metrics.metrics.logit_diff"

    # ---- graph ----
    graph = Graph.from_model(model)
    print(f"graph: n_forward={graph.n_forward} n_backward={graph.n_backward} "
          f"real_edges={int(graph.real_edge_mask.sum())}")

    # ---- attribution ----
    metric_attr = partial(segrev3_logit_diff, loss=True, mean=True)
    used_method = method
    with force_zeros_device(_dev):
        try:
            attribute(model, graph, attr_dl, metric_attr, method=method,
                      ig_steps=ig_steps, intervention="patching", quiet=True)
        except Exception as exc:  # pragma: no cover - defensive fallback
            print(f"[WARN] method={method} failed ({type(exc).__name__}: {exc}); "
                  f"falling back to plain 'EAP'.")
            used_method = "EAP"
            graph = Graph.from_model(model)
            attribute(model, graph, attr_dl, metric_attr, method="EAP",
                      intervention="patching", quiet=True)

    nz = int((graph.scores.abs() > 0).sum())
    print(f"attribution done (method={used_method}); nonzero edge scores: {nz} / "
          f"{graph.scores.numel()}; max|score|={graph.scores.abs().max():.4f}")

    # ---- baselines (full model on clean; and corrupted lower bound) ----
    metric_eval = partial(segrev3_logit_diff, loss=False, mean=False)
    with force_zeros_device(_dev):
        clean_baseline = evaluate_baseline(model, eval_dl, metric_eval,
                                           quiet=True).mean().item()
        corrupt_baseline = evaluate_baseline(model, eval_dl, metric_eval,
                                             run_corrupted=True, quiet=True).mean().item()
    print(f"baseline logit-diff: clean (full model)={clean_baseline:+.4f}  "
          f"corrupted (lower bound)={corrupt_baseline:+.4f}")

    # ---- per-top-n: evaluate faithfulness + export ONE folder each ----
    # Every top-n (the sweep n's AND the headline export_n) gets its own folder
    # eap_top<n>/ with eap_top<n>.{json,pt,png}.
    denom = (clean_baseline - corrupt_baseline) or 1.0
    all_ns = sorted(set(topns) | {export_n})
    # Data provenance: ONE dataset feeds the whole run (attribution is computed
    # once and shared across all top-n), so every per_topn entry records the
    # same source file.
    data_block = {
        "store_label": store.dataset_label("eap", scheme, topn=export_n),
        "path": repo_relpath(data_path),
        "scheme": scheme,
        "attr_examples": attr_n,
        "eval_examples": eval_n,
    }
    per_topn = {}
    faithfulness = []
    for n in tqdm(all_ns, desc="EAP-IG top-n sweep", unit="topn"):
        with force_zeros_device(_dev):
            graph.apply_topn(n, absolute=True, level="edge", reset=True, prune=True)
            circ = evaluate_graph(model, graph, eval_dl, metric_eval,
                                  intervention="patching", quiet=True).mean().item()
        n_edges = graph.count_included_edges()
        n_nodes = graph.count_included_nodes()
        ratio = (circ - corrupt_baseline) / denom
        heads_n = sorted({nm for nm in graph.nodes
                          if head_of(nm) is not None and graph.nodes[nm].in_graph})
        movers_n = sorted(set(heads_n) & set(MANUAL_MOVERS))
        detectors_n = sorted(set(heads_n) & set(MANUAL_DETECTORS))

        folder = _topn_folder(n)
        os.makedirs(folder, exist_ok=True)
        n_json = os.path.join(folder, f"eap_top{n}.json")
        n_pt = os.path.join(folder, f"eap_top{n}.pt")
        n_png = os.path.join(folder, f"eap_top{n}.png")
        graph.to_json(n_json)
        graph.to_pt(n_pt)
        n_png_ok = True
        try:
            graph.to_image(n_png, colorscheme="Pastel2", seed=0)
        except Exception as exc:  # pragma: no cover
            n_png_ok = False
            print(f"[WARN] to_image(top{n}) failed ({type(exc).__name__}: {exc})")

        per_topn[str(n)] = {
            "n": n, "folder": os.path.basename(folder),
            "edges_after_prune": int(n_edges), "nodes": int(n_nodes),
            "circuit_logit_diff": circ, "faithfulness_ratio": ratio,
            "circuit_heads": heads_n,
            "movers_found": movers_n, "n_movers_found": len(movers_n),
            "detectors_found": detectors_n, "n_detectors_found": len(detectors_n),
            "data": data_block,
            "files": {"json": repo_relpath(n_json), "pt": repo_relpath(n_pt),
                      "png": repo_relpath(n_png) if n_png_ok else None},
        }
        faithfulness.append({
            "n_requested": n, "edges_after_prune": int(n_edges),
            "nodes": int(n_nodes), "circuit_logit_diff": circ,
            "faithfulness_ratio": ratio,
        })
        print(f"  top-{n:>4}: edges={int(n_edges):>3} nodes={int(n_nodes):>2} "
              f"circuit_ld={circ:+.4f}  faithfulness={ratio:6.1%}  "
              f"movers={len(movers_n)}/{len(MANUAL_MOVERS)} det={len(detectors_n)}/{len(MANUAL_DETECTORS)}")

    # ---- inspect top edges + node importance for export_n circuit ----
    with force_zeros_device(_dev):
        graph.apply_topn(export_n, absolute=True, level="edge", reset=True, prune=True)

    top30 = top_edges(graph, 30)
    out_total, to_logits, in_total = node_importance(graph)
    movers_ranked = sorted(to_logits.items(), key=lambda x: x[1], reverse=True)
    readers_ranked = sorted(in_total.items(), key=lambda x: x[1], reverse=True)
    writers_ranked = sorted(out_total.items(), key=lambda x: x[1], reverse=True)

    print("\nTop 30 edges by |score| (full graph):")
    for name, sc in top30:
        print(f"    {name:<26} {sc:+.4f}")

    print("\nHeads writing to logits (|score| of head->logits), top 8:")
    for h, v in movers_ranked[:8]:
        tag = " <- MANUAL MOVER" if h in MANUAL_MOVERS else ""
        print(f"    {h:<7} {v:.4f}{tag}")
    print("Heads by total READ importance (incoming |score|), top 8:")
    for h, v in readers_ranked[:8]:
        tag = " <- MANUAL DETECTOR" if h in MANUAL_DETECTORS else ""
        print(f"    {h:<7} {v:.4f}{tag}")

    # overlap with the manual circuit (heads present anywhere in the export_n circuit)
    circuit_heads = {n for n in graph.nodes
                     if head_of(n) is not None and graph.nodes[n].in_graph}
    movers_found = sorted(circuit_heads & set(MANUAL_MOVERS))
    detectors_found = sorted(circuit_heads & set(MANUAL_DETECTORS))
    print(f"\nManual movers in top-{export_n} circuit:    "
          f"{movers_found}  ({len(movers_found)}/{len(MANUAL_MOVERS)})")
    print(f"Manual detectors in top-{export_n} circuit: "
          f"{detectors_found}  ({len(detectors_found)}/{len(MANUAL_DETECTORS)})")
    print(f"All circuit heads (top-{export_n}): {sorted(circuit_heads)}")

    # ---- combined "thus far" summary (one json in OUT_DIR; per-n files already
    # written to their eap_top<n>/ folders above) ----
    tag = used_method.replace("-", "_")
    run_meta = {
        "method": used_method,
        "tag": tag,
        "ig_steps": ig_steps,
        "attr_examples": attr_n, "eval_examples": eval_n,
        "scheme": scheme,
        "data": data_block,
        "metric": "mean(correct-incorrect) logit over answer positions 9..18",
        "graph": {"n_forward": graph.n_forward, "n_backward": graph.n_backward,
                  "real_edges": int(graph.real_edge_mask.sum())},
        "clean_baseline_logit_diff": clean_baseline,
        "corrupted_baseline_logit_diff": corrupt_baseline,
        "faithfulness_curve": faithfulness,
        "export_n": export_n,
        "circuit_heads_top_export_n": sorted(circuit_heads),
        "manual_movers_found": movers_found,
        "manual_detectors_found": detectors_found,
        "top30_edges": [{"edge": n, "score": s} for n, s in top30],
        "heads_to_logits": dict(movers_ranked),
        "heads_read_importance": dict(readers_ranked),
        "heads_write_importance": dict(writers_ranked),
    }
    summary = build_eap_combined_summary(per_topn, run_meta)
    print(f"\nWrote combined summary -> {COMBINED_SUMMARY}")
    print("  top-n folders: " + ", ".join(
        os.path.basename(_topn_folder(k)) for k in sorted(summary["per_topn"], key=int)))
    print("DONE.")
    return summary


def parse_args():
    from canon import SEGREV3_TOPNS, CANON_EXPORT_N, CANON_IG_STEPS, SCHEMES

    p = argparse.ArgumentParser(description="EAP-IG circuit discovery for segrev3")
    # The science knobs only. The internals (method=EAP-IG-inputs, attr_n=200,
    # attr_bs=50, eval_n=256, eval_bs=64) are FIXED in run()'s defaults: the
    # committed results use exactly those, and attr_bs in particular changes the
    # SCALE of the attribution scores (EAP-IG normalizes by n_batches * ig_steps),
    # so they are deliberately not CLI-exposed -- edit run()'s signature if you
    # truly need to change them.
    p.add_argument("--ig-steps", type=int, default=CANON_IG_STEPS,
                   help="integrated-gradient interpolation steps (default: %(default)s)")
    p.add_argument("--topns", type=int, nargs="+", default=SEGREV3_TOPNS,
                   help="top-n circuits to export (default: %(default)s)")
    p.add_argument("--export-n", type=int, default=CANON_EXPORT_N,
                   help="headline circuit size (default: %(default)s)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--scheme", default="random", choices=list(SCHEMES),
                   help="data-generation scheme to load from the central store")
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    run(ig_steps=a.ig_steps, topns=tuple(a.topns),
        seed=a.seed, export_n=a.export_n, scheme=a.scheme)
