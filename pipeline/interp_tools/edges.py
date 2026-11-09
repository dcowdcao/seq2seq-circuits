#!/usr/bin/env python
"""Per-circuit EDGE analyses + heatmaps for pipeline.

Compute is delegated to the repo-root ``interp_tools/edge_tools.py``; this CLI
writes, for ``--model <m> --circuit <c>``, the four subtrees under
``circuits_data/<m>/<c>/edges/`` with 1:1 PNG mirrors under ``circuits_figures/``:
  composition   weight-based Q/K/V composition -> composition/{Q,K,V}_comp.json
  qkov          per-head QK/OV token circuits  -> QK_OV_matrix/head_L#H#/{QK,OV}.json
  pos_pos       realized mean attention        -> pos_pos/clean_corrupt_dataset/<c>/head_L#H#.json
  pos_token     attention mass by source digit -> pos_token/clean_corrupt_dataset/<c>/head_L#H#.json
Data-dependent analyses read the circuit's single stored pool via
``model_registry.load_circuit_pairs`` (CLEAN side, one shared forward sweep);
``--smoke`` re-roots all outputs under ``.smoke/run2/``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time

import torch
import torch.nn.functional as F
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# sys.path bootstrap: model_registry itself puts the shared repo module dirs on sys.path.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_ROOT = os.path.dirname(_THIS_DIR)
_MODELS_DIR = os.path.join(PIPELINE_ROOT, "models")
if _MODELS_DIR not in sys.path:
    sys.path.insert(0, _MODELS_DIR)

import model_registry as MG                                # noqa: E402
import edge_tools as ET                                   # noqa: E402  (repo interp_tools)

GENERATED_BY = "pipeline/interp_tools/edges.py"
ANALYSES = ("composition", "qkov", "pos_pos", "pos_token")
MAX_FORWARD_BATCH = 32          # GPU politeness: never run a bigger forward batch
SMOKE_ROOT = os.path.join(PIPELINE_ROOT, ".smoke", "run2")


def edges_roots(name: str, circuit: str, smoke: bool) -> tuple[str, str]:
    """(data_root, fig_root) for ``<...>/<name>/<circuit>/edges`` (smoke-
    quarantined under .smoke/run2/ if asked)."""
    droot = os.path.join(MG.data_dir(name, circuit), "edges")
    froot = os.path.join(MG.fig_dir(name, circuit), "edges")
    if smoke:
        droot = os.path.join(SMOKE_ROOT, os.path.relpath(droot, PIPELINE_ROOT))
        froot = os.path.join(SMOKE_ROOT, os.path.relpath(froot, PIPELINE_ROOT))
    return droot, froot


# JSON envelope + serialization helpers.
def make_meta(spec, checkpoint: str, method: str, *, pair=None, scheme=None,
              n_examples: int = 0, seed: int = 0, source_dataset_path=None) -> dict:
    return {
        "model": spec.name,
        "pair": pair,
        "scheme": scheme,
        "n_examples": int(n_examples),
        "seed": int(seed),
        "method": method,
        "checkpoint": MG.repo_relpath(checkpoint),
        "generated_by": GENERATED_BY,
        "source_dataset_path": MG.repo_relpath(source_dataset_path),
    }


def write_json(path: str, obj: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True, allow_nan=False)


def _tolist(t: torch.Tensor):
    return t.detach().cpu().tolist()


def _nan_to_none(rows):
    """Nested-list NaN -> None (JSON null); keeps the artifact strict-JSON."""
    return [[None if (isinstance(v, float) and math.isnan(v)) else v for v in row]
            for row in rows]


# Plot helpers (NaN cells light grey 0.85, title carries model [+ dataset]).
def _savefig(fig, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def comp_heatmap(mat, valid, title: str, path: str, baseline: float,
                 n_heads: int) -> None:
    """Triangular sender->receiver composition heatmap. Composition is CAUSAL
    (sender layer < receiver layer), so last-layer sender rows and first-layer
    receiver cols are cropped; remaining invalid cells grey."""
    m = mat.detach().cpu().numpy().copy().astype(float)
    m[~valid.detach().cpu().numpy()] = np.nan
    n = m.shape[0]
    row_hi, col_lo = n - n_heads, n_heads
    if row_hi <= 0 or n - col_lo <= 0:                    # degenerate (single layer)
        row_hi, col_lo = n, 0
    m = m[:row_hi, col_lo:]
    row_labels = [f"L{i // n_heads}H{i % n_heads}" for i in range(row_hi)]
    col_labels = [f"L{(col_lo + j) // n_heads}H{(col_lo + j) % n_heads}"
                  for j in range(n - col_lo)]

    fig, ax = plt.subplots(figsize=(8, 7))
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad("0.85")                                  # invalid cells -> light grey
    finite_max = np.nanmax(m) if np.isfinite(m).any() else 1.0
    im = ax.imshow(m, cmap=cmap, vmin=0, vmax=float(finite_max) or 1.0,
                   aspect="auto")
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=90, fontsize=5)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=5)
    ax.set_xlabel("receiver head (later layer)")
    ax.set_ylabel("sender head (earlier layer)")
    for k in range(0, row_hi + 1, n_heads):
        ax.axhline(k - 0.5, color="w", lw=0.4)
    for k in range(0, (n - col_lo) + 1, n_heads):
        ax.axvline(k - 0.5, color="w", lw=0.4)
    ax.set_title(title + f"  (random baseline={baseline:.3f})", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                 label="||W_OV^A W_x^B||_F / (||W_OV^A||_F ||W_x^B||_F)")
    fig.tight_layout()
    _savefig(fig, path)


def token_heatmap(mat, title: str, path: str, xlabel: str, ylabel: str,
                  cmap: str, diverging: bool = False) -> None:
    """[d, d] token x token heatmap with integer ticks."""
    m = mat.detach().cpu().numpy() if isinstance(mat, torch.Tensor) else np.asarray(mat)
    d = m.shape[1]
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    cm = plt.get_cmap(cmap).copy()
    cm.set_bad("0.85")
    if diverging:
        finite_max = np.nanmax(np.abs(m)) if np.isfinite(m).any() else 1.0
        vmax = float(finite_max) or 1.0
        im = ax.imshow(m, cmap=cm, vmin=-vmax, vmax=vmax, aspect="auto")
    else:
        im = ax.imshow(m, cmap=cm, aspect="auto")
    ax.set_xticks(range(d)); ax.set_xticklabels(range(d), fontsize=7)
    ax.set_yticks(range(m.shape[0])); ax.set_yticklabels(range(m.shape[0]), fontsize=7)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    _savefig(fig, path)


def pos_pos_heatmap(mean_patt, title: str, path: str) -> None:
    """[n_ctx, n_ctx] dest x src realized-attention heatmap; the causal upper
    triangle (future sources) is NaN'd -> light grey (display only; the JSON
    keeps the raw rows, which sum to 1)."""
    m = mean_patt.detach().cpu().numpy().copy().astype(float)
    n = m.shape[0]
    m[np.triu_indices(n, k=1)] = np.nan
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("0.85")
    finite_max = np.nanmax(m) if np.isfinite(m).any() else 1.0
    im = ax.imshow(m, cmap=cmap, vmin=0, vmax=float(finite_max) or 1.0,
                   aspect="auto")
    ax.set_xticks(range(n)); ax.set_xticklabels(range(n), fontsize=6)
    ax.set_yticks(range(n)); ax.set_yticklabels(range(n), fontsize=6)
    ax.set_xlabel("source position"); ax.set_ylabel("destination position")
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="mean attention")
    fig.tight_layout()
    _savefig(fig, path)


def pos_token_heatmap(mass_by_digit, title: str, path: str) -> None:
    """[n_ctx, d_vocab] dest-position x source-DIGIT attention-mass heatmap."""
    m = (mass_by_digit.detach().cpu().numpy()
         if isinstance(mass_by_digit, torch.Tensor) else np.asarray(mass_by_digit))
    n, d = m.shape
    fig, ax = plt.subplots(figsize=(4.8, 5.0))
    im = ax.imshow(m, cmap="viridis", vmin=0, vmax=float(m.max()) or 1.0,
                   aspect="auto")
    ax.set_xticks(range(d)); ax.set_xticklabels(range(d), fontsize=7)
    ax.set_yticks(range(n)); ax.set_yticklabels(range(n), fontsize=6)
    ax.set_xlabel("source digit"); ax.set_ylabel("destination position")
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="mean attention mass")
    fig.tight_layout()
    _savefig(fig, path)


# Analysis 1: composition (weight-based).
def run_composition(model, spec, checkpoint: str, droot: str, froot: str) -> int:
    """Q/K/V composition over ALL heads (the full (L·H)x(L·H) grid — a property
    of the whole model, not of --heads). Direct call into edge_tools."""
    comp = ET.qkv_composition(model)
    head_labels = [f"L{l}H{h}" for (l, h) in comp.heads]
    n_written = 0
    for ch, mat in (("Q", comp.q_comp), ("K", comp.k_comp), ("V", comp.v_comp)):
        payload = {
            "meta": make_meta(
                spec, checkpoint,
                method=f"qkv_composition[{ch}] — normalized-Frobenius overlap "
                       f"||W_OV^sender W_{ch}^receiver||_F / "
                       f"(||W_OV^sender||_F ||W_{ch}^receiver||_F); weight-based, "
                       "data-independent (edge_tools.qkv_composition)"),
            "heads": head_labels,
            "matrix": _tolist(mat),
            "baseline": float(comp.baseline),
            "valid_mask": [[bool(v) for v in row] for row in comp.valid_mask],
            "axes": "row = sender head index (heads list, row-major l*n_heads+h), "
                    "col = receiver head index; valid_mask True iff "
                    "sender layer < receiver layer",
        }
        write_json(os.path.join(droot, "composition", f"{ch}_comp.json"), payload)
        comp_heatmap(
            mat, comp.valid_mask,
            f"{spec.name}: {ch}-composition (sender -> receiver)\n"
            "weight-based — independent of test data",
            os.path.join(froot, "composition", f"{ch}_comp.png"),
            comp.baseline, n_heads=model.cfg.n_heads)
        n_written += 1
    tqdm.write(f"  [composition] wrote {n_written} JSON+PNG pairs "
               f"(baseline={comp.baseline:.4f}, valid edges={int(comp.valid_mask.sum())})")
    return n_written


# Analysis 2: QK / OV matrices per head (weight-based).
def run_qkov(model, spec, checkpoint: str, droot: str, froot: str,
             sel_heads) -> int:
    """Per-head QK + OV token circuits (edge_tools.qk_circuit / ov_circuit,
    called directly). Data-independent: computed once per model."""
    n_written = 0
    for (l, h) in tqdm(sel_heads, desc=f"edges:qkov[{spec.name}]", unit="head",
                       disable=not sys.stderr.isatty()):
        hd = MG.head_dirname(l, h)
        ddir = os.path.join(droot, "QK_OV_matrix", hd)
        fdir = os.path.join(froot, "QK_OV_matrix", hd)

        qk = ET.qk_circuit(model, l, h, include_positional=True)
        qk_payload = {
            "meta": make_meta(
                spec, checkpoint,
                method="qk_circuit — content QK token matrix "
                       "W_E W_Q W_K^T W_E^T / sqrt(d_head) (+ row-softmax + "
                       "causal positional QK over n_ctx positions); weight-based, "
                       "data-independent (edge_tools.qk_circuit)"),
            "layer": l,
            "head": h,
            "token_matrix": _tolist(qk.token_matrix),
            "token_softmax": _tolist(qk.token_softmax),
            "positional_matrix_causal": (
                _nan_to_none(_tolist(qk.positional_matrix_causal))
                if qk.positional_matrix_causal is not None else None),
            "scale": float(qk.scale),
            "axes": "token_matrix/token_softmax: row = query token, col = key "
                    "token; positional_matrix_causal: row = query position, col "
                    "= key position (future key>query cells = null)",
        }
        write_json(os.path.join(ddir, "QK.json"), qk_payload)
        token_heatmap(
            qk.token_matrix,
            f"{spec.name} L{l}H{h} content QK circuit (query -> key)\n"
            "weight-based — independent of test data",
            os.path.join(fdir, "QK.png"),
            xlabel="key token", ylabel="query token",
            cmap="RdBu_r", diverging=True)

        ov = ET.ov_circuit(model, l, h)
        ov_payload = {
            "meta": make_meta(
                spec, checkpoint,
                method="ov_circuit — OV token matrix W_E W_V W_O W_U + eigenvalue "
                       "copying score (Elhage et al. 2021); weight-based, "
                       "data-independent (edge_tools.ov_circuit)"),
            "layer": l,
            "head": h,
            "token_matrix": _tolist(ov.token_matrix),
            "copying_score": float(ov.copying_score),
            "frac_positive_eig": float(ov.frac_positive_eig),
            "axes": "row = attended (source) token, col = output-logit token",
        }
        write_json(os.path.join(ddir, "OV.json"), ov_payload)
        token_heatmap(
            ov.token_matrix,
            f"{spec.name} L{l}H{h} OV circuit (attended -> output)  "
            f"copy={ov.copying_score:+.3f}\n"
            "weight-based — independent of test data",
            os.path.join(fdir, "OV.png"),
            xlabel="output token", ylabel="attended token",
            cmap="RdBu_r", diverging=True)
        n_written += 2
    tqdm.write(f"  [qkov] wrote {n_written} JSON+PNG pairs across {len(sel_heads)} head dirs")
    return n_written


# Analyses 3+4: realized attention (data-dependent; clean side of the dataset).
@torch.no_grad()
def clean_side_attention(model, tokens: torch.Tensor,
                         d_vocab: int) -> tuple[torch.Tensor, torch.Tensor]:
    """ONE chunked forward sweep over ``tokens`` shared by every head and by
    both data-dependent analyses.

    Returns (mean_patt [L,H,S,S] dest x src, mass_by_digit [L,H,S,d_vocab]).
    """
    device = next(model.parameters()).device
    n_layers = model.cfg.n_layers
    B, S = tokens.shape
    patt_sum = torch.zeros(n_layers, model.cfg.n_heads, S, S, device=device)
    mass_sum = torch.zeros(n_layers, model.cfg.n_heads, S, d_vocab, device=device)
    for chunk in tokens.split(MAX_FORWARD_BATCH):
        chunk = chunk.to(device)
        _, cache = model.run_with_cache(
            chunk, names_filter=lambda nm: nm.endswith("attn.hook_pattern"))
        onehot = F.one_hot(chunk, num_classes=d_vocab).float()      # [b,S,d_vocab]
        for layer in range(n_layers):
            patt = cache[f"blocks.{layer}.attn.hook_pattern"]       # [b,H,S,S]
            patt_sum[layer] += patt.sum(dim=0)
            mass_sum[layer] += torch.einsum("bhds,bsv->hdv", patt, onehot)
    return patt_sum / B, mass_sum / B


def _pos_pos_stats(mean_patt: torch.Tensor, spec) -> dict:
    """Per-head positional stats."""
    stats = {
        "argmax_src_per_dest": [int(x) for x in mean_patt.argmax(dim=1)],
        "mean_max_attention": float(mean_patt.max(dim=1).values.mean()),
    }
    # Reversal mirror (only meaningful when the sequence is [input | output] of
    # equal halves): dest = input_len-1+k attends to reflected src = input_len-1-k.
    n = mean_patt.shape[0]
    in_len = spec.input_len
    if 2 * in_len == spec.n_ctx:
        refl = [float(mean_patt[in_len - 1 + k, in_len - 1 - k])
                for k in range(in_len) if in_len - 1 + k < n]
        stats["mean_attention_to_reflected_position"] = (
            float(np.mean(refl)) if refl else None)
    return stats


def run_pos_analyses(model, spec, checkpoint: str, droot: str, froot: str,
                     args, circuit: str, sel_heads,
                     do_pos_pos: bool, do_pos_token: bool) -> int:
    """pos_pos and/or pos_token for the circuit's single clean/corrupt dataset,
    sharing one chunked forward sweep on the CLEAN side; artifacts land under
    ``clean_corrupt_dataset/<circuit>/``."""
    pairs, src_path = MG.load_circuit_pairs(spec.name, circuit, n=args.n, consumer="attribution")
    src_path = os.path.relpath(src_path, MG.REPO_ROOT)
    scheme = pairs.scheme
    seed = int(pairs.extra["seed"])      # the seed that GENERATED the stored data
    tokens = pairs.clean_tokens                                     # CLEAN side
    n_examples = int(tokens.shape[0])

    mean_patt, mass_by_digit = clean_side_attention(model, tokens, spec.d_vocab)

    n_written = 0
    which = "+".join(a for a, on in (("pos_pos", do_pos_pos),
                                     ("pos_token", do_pos_token)) if on)
    for (l, h) in tqdm(sel_heads, desc=f"edges:{which}[{circuit}]", unit="head",
                       disable=not sys.stderr.isatty()):
        hd = MG.head_dirname(l, h)
        if do_pos_pos:
            payload = {
                "meta": make_meta(
                    spec, checkpoint,
                    method="realized mean attention (dest position x src "
                           "position), CLEAN side of the pair "
                           "(vis_edge._realized_attention)",
                    pair=circuit, scheme=scheme, n_examples=n_examples,
                    seed=seed, source_dataset_path=src_path),
                "layer": l,
                "head": h,
                "attention_pattern_dest_src": _tolist(mean_patt[l, h]),
                "stats": _pos_pos_stats(mean_patt[l, h], spec),
                "axes": "row = destination position, col = source position; "
                        "rows sum to 1 over the causal (src<=dest) prefix; "
                        "future cells are exactly 0",
            }
            write_json(os.path.join(droot, "pos_pos", "clean_corrupt_dataset",
                                    circuit, f"{hd}.json"), payload)
            pos_pos_heatmap(
                mean_patt[l, h],
                f"{spec.name} [{circuit}] L{l}H{h} realized mean attention "
                f"(clean side)\nn={n_examples}, seed={seed}; future src "
                "greyed (causal)",
                os.path.join(froot, "pos_pos", "clean_corrupt_dataset",
                             circuit, f"{hd}.png"))
            n_written += 1
        if do_pos_token:
            payload = {
                "meta": make_meta(
                    spec, checkpoint,
                    method="realized attention mass by SOURCE DIGIT per dest "
                           "position, CLEAN side of the pair "
                           "(vis_edge._realized_attention content split)",
                    pair=circuit, scheme=scheme, n_examples=n_examples,
                    seed=seed, source_dataset_path=src_path),
                "layer": l,
                "head": h,
                "attention_mass_by_source_digit_per_dest": _tolist(mass_by_digit[l, h]),
                "stats": {},
                "axes": "row = destination position, col = source DIGIT "
                        f"0..{spec.d_vocab - 1} (boundary column = None); "
                        "rows sum to 1",
            }
            write_json(os.path.join(droot, "pos_token", "clean_corrupt_dataset",
                                    circuit, f"{hd}.json"), payload)
            pos_token_heatmap(
                mass_by_digit[l, h],
                f"{spec.name} [{circuit}] L{l}H{h} attention by source digit "
                f"(clean side)\nn={n_examples}, seed={seed}",
                os.path.join(froot, "pos_token", "clean_corrupt_dataset",
                             circuit, f"{hd}.png"))
            n_written += 1
    tqdm.write(f"  [{which}] circuit {circuit}: wrote {n_written} JSON+PNG pairs "
               f"({len(sel_heads)} heads, n={n_examples}, clean side)")
    return n_written


# CLI.
def _parse_heads(arg: str, spec) -> list[tuple[int, int]]:
    if arg == "all":
        return MG.heads(spec)
    out = []
    for tok in arg.split(","):
        m = re.fullmatch(r"L(\d+)H(\d+)", tok.strip())
        if not m:
            raise SystemExit(f"--heads: bad head {tok!r} (expected L#H#)")
        l, h = int(m.group(1)), int(m.group(2))
        if not (0 <= l < spec.n_layers and 0 <= h < spec.n_heads):
            raise SystemExit(f"--heads: {tok!r} out of range for {spec.name} "
                             f"({spec.n_layers}L x {spec.n_heads}H)")
        out.append((l, h))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Per-circuit edge analyses (composition / QK+OV / pos_pos "
                    "/ pos_token) + heatmaps for pipeline.")
    ap.add_argument("--model", required=True,
                    help="model name (a DSL algorithm_id)")
    ap.add_argument("--circuit", required=True,
                    help="circuit slug: selects circuits_data/<model>/<circuit>/edges "
                         "and the circuit's single gen_data inputs.json")
    ap.add_argument("--analysis", default="all",
                    choices=list(ANALYSES) + ["all"])
    ap.add_argument("--heads", default="all",
                    help='"all" or comma list "L0H7,L3H1" (qkov/pos_pos/'
                         'pos_token; composition always uses the full grid)')
    ap.add_argument("-n", type=int, default=MG.APP_SIZES["edges"],
                    help="pairs to load from the circuit pool (default 64; "
                         "capped by the pool size)")
    ap.add_argument("--smoke", action="store_true",
                    help="quarantine outputs under .smoke/run2/")
    args = ap.parse_args(argv)

    name, circuit = args.model, args.circuit
    do = set(ANALYSES) if args.analysis == "all" else {args.analysis}

    t0 = time.time()
    total = 0
    spec = MG.get_spec(name)
    droot, froot = edges_roots(name, circuit, smoke=args.smoke)
    sel_heads = _parse_heads(args.heads, spec)

    tqdm.write(f"[edges] model={name} circuit={circuit} analyses={sorted(do)} "
               f"heads={len(sel_heads)} smoke={args.smoke}\n"
               f"[edges]   data -> {droot}\n[edges]   figs -> {froot}")

    MG.wait_for_headroom(context=f"edges.py load_model({name!r})")
    model = MG.load_model(name)
    cfg = model.cfg
    assert (cfg.n_layers, cfg.n_heads, cfg.d_vocab, cfg.n_ctx) == \
           (spec.n_layers, spec.n_heads, spec.d_vocab, spec.n_ctx), \
        f"loaded model dims disagree with the ModelSpec for {name!r}"

    try:
        if "composition" in do:
            total += run_composition(model, spec, spec.checkpoint, droot, froot)
        if "qkov" in do:
            total += run_qkov(model, spec, spec.checkpoint, droot, froot, sel_heads)
        if "pos_pos" in do or "pos_token" in do:
            total += run_pos_analyses(
                model, spec, spec.checkpoint, droot, froot, args, circuit,
                sel_heads, do_pos_pos="pos_pos" in do,
                do_pos_token="pos_token" in do)
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"[edges] DONE — {total} JSON+PNG artifact pairs in "
          f"{time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
