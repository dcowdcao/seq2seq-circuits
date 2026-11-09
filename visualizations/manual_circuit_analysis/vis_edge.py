"""
vis_edge.py
=================================================================================
Phase-3 "edge" (weight-based, head->head) circuit VISUALIZATION driver for segrev3.

This is the *rendering only* counterpart to the pure-compute library
``interp_tools/edge_tools.py``. Every numeric quantity shown here is computed by
that library; this script only turns the results into figures + a small JSON/MD
report. The "edge" name refers to the weight graph's edges between heads (and the
QK/OV end-to-end circuits and the OV eigenvalue copying score that live on them).

WHAT IS DRAWN
  1. copy_scores.png       -- OV eigenvalue copying score for all 32 heads (bar).
  2. composition_{q,k,v}.png -- 32x32 sender x receiver Q/K/V composition heatmaps.
  3. per specific head, in specific_heads/:
       L{l}H{h}_QK.png      -- the head's CONTENT QK circuit [10x10]
                               (query token y, key token x). CONTENT ONLY.
       L{l}H{h}_OV.png      -- the head's OV circuit [10x10] (attended y, output x),
                               title carries the eigenvalue copying score.
       L{l}H{h}_attn.png    -- the head's REALIZED mean attention pattern (dest y,
                               src x) on a SINGLE dataset with a given number of 3s.
                               POSITIONAL view: shows the reversal mirror, but blurs
                               the 3s (random positions across the batch).
       L{l}H{h}_attn_content.png -- the CONTENT heatmap of the same pass: same
                               format as *_attn.png but x = SOURCE DIGIT instead
                               of source position (cell = mass on all sources
                               holding that digit; framed column 3 = boundary
                               attention, tracking 3s vs non-3s per destination).

DATA = SINGLE DATASETS BY NUMBER-OF-3s (no patching here)
  The edge analysis does NOT patch, so it needs no clean/corrupted pair -- just
  inputs. It loads the SINGLE (non-paired) datasets ``datasets/edge/n_threes_<k>.json``
  (each input has exactly k 3s; see data_gen.make_single_dataset). The meaningful
  axis is the NUMBER OF 3s (boundaries), since that sets the segment structure --
  e.g. L0H7's realized reflection-mass is ~0.9 at n_threes=0 (pure mirror) and
  collapses once boundaries appear.

  QK / OV / copying-score / composition are ALL weight-based -- read straight off
  the weights, INPUT-INDEPENDENT; the dataset is used for exactly one thing: to
  render each requested head's REALIZED mean attention pattern (``*_attn.png``),
  the empirical counterpart of the static (content) QK circuit.

DATASET PROVENANCE CONVENTION (followed exactly throughout this script)
  Every generated PNG and data file states truthfully which dataset it relates to,
  distinguishing weight-based (no data) from data-dependent (attention).

  * A dataset's **label** = the json filename stem
    (``os.path.splitext(os.path.basename(path))[0]`` -> e.g. "n_threes_3").

  * WEIGHT-BASED outputs (``copy_scores.png``, ``composition_{q,k,v}.png`` and the
    per-head ``L{l}H{h}_QK.png`` / ``L{l}H{h}_OV.png``) are IDENTICAL regardless of
    dataset, so they are written ONCE, FLAT in ``figs_data/figs/edge/`` (and its
    ``specific_heads/``). Each carries a title line ``weight-based -- independent of
    test_data``. They are NEVER duplicated per dataset and NEVER labelled with a
    dataset name (that would be misleading).

  * DATA-DEPENDENT outputs (the realized attention pattern ``*_attn.png``) are
    produced PER DATASET into a per-dataset subfolder and titled with the dataset:
    ``figs_data/figs/edge/<label>/specific_heads/L{l}H{h}_attn.png`` whose title
    includes ``test_data: <label>``.

MULTI-DATASET SUPPORT
  ``main(data_paths="all", heads=None)`` accepts:
    * a single path string,
    * a list of path strings, or
    * the literal "all" (-> ``sorted(glob.glob(TEST_DATA_DIR/*.json))``).
  Default = "all". ``plot_head(model, l, h, pairs=..., label=...)`` still works for
  any head; only the ``*_attn.png`` plots and the realized-attention stats depend on
  the dataset.

  ``edge_results.json`` keeps the weight-based sections written ONCE (with
  ``meta.weight_based_note``) and adds a top-level ``attn_by_dataset`` keyed by
  label covering every dataset run, plus ``meta.test_data_datasets``. ``edge_report.md``
  keeps the weight-based findings and notes which datasets the realized-attention
  plots/stats were computed for.

CAVEAT (inherited from edge_tools): these weight products drop the per-position
LayerNormPre scale, so they are the standard Elhage approximations. Confirm high
scores causally with path patching -- composition magnitude alone does not prove
an edge is used.

Usage:
    python vis_edge.py                        # all datasets in datasets/edge/
    python vis_edge.py --data /abs/a.json     # one dataset (e.g. a single-sequence
                                              # datasets/single/edge_k<k>.json)
    python vis_edge.py --data a.json b.json   # a subset
"""

from __future__ import annotations

import contextlib
import glob
import json
import math
import os
import sys

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------- #
# sys.path setup                                                              #
# --------------------------------------------------------------------------- #
_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_THIS))   # repo root
for p in ("interp_tools", "data_gen", "metrics"):
    sys.path.insert(0, os.path.join(_ROOT, p))

import edge_tools as E                                   # noqa: E402
import circuit_loader as CL                               # noqa: E402
import n_threes as N3                                     # noqa: E402
from model_loading import load_segrev3, BOUNDARY_DIGIT, repo_relpath   # noqa: E402
from data_gen import load_single, load_tokens_any, make_single_dataset    # noqa: E402  (single or paired datasets)

# --------------------------------------------------------------------------- #
# Output locations                                                            #
# --------------------------------------------------------------------------- #
FIG_DIR = os.path.join(_THIS, "figs_data", "figs", "edge")
HEAD_DIR = os.path.join(FIG_DIR, "specific_heads")   # FLAT weight-based per-head dir
DATA_DIR = os.path.join(_THIS, "figs_data", "data")
# The "edge" analysis does NO patching, so it uses SINGLE (non-paired) datasets
# indexed by the NUMBER OF 3s (boundaries): datasets/edge/n_threes_<k>.json.
TEST_DATA_DIR = os.path.join(_ROOT, "datasets", "edge")


def _label_of(path):
    """Dataset label = the json filename stem (provenance convention)."""
    return os.path.splitext(os.path.basename(path))[0]


def _resolve_data_paths(data_paths):
    """Normalize the ``data_paths`` argument into a concrete list of json paths.

    Accepts the literal "all" (-> every *.json in TEST_DATA_DIR, sorted), a single
    path string, or a list of path strings.
    """
    if data_paths == "all":
        return sorted(glob.glob(os.path.join(TEST_DATA_DIR, "*.json")))
    if isinstance(data_paths, str):
        return [data_paths]
    return list(data_paths)


def _attn_head_dir(label):
    """Per-dataset data-dependent per-head dir for the realized ``*_attn.png``."""
    return os.path.join(FIG_DIR, label, "specific_heads")

# The known reflection/mover heads and the two L0 hubs (defaults for per-head plots).
MOVERS = [(3, 1), (3, 3), (3, 6), (2, 7)]
DETECTOR_HUB = (0, 7)          # L0H7: realized-reflection / boundary-attention hub
Q_HUB = (0, 4)                 # L0H4: Q-composition hub into the movers; OV 3->3 self-boost
DEFAULT_HEADS = MOVERS + [DETECTOR_HUB, Q_HUB]

# Reversal-task geometry (segrev3): SEQ_LEN=20, input pos 0..9, output pos 9..18.
# The "mirror" maps output dest 9+k -> reflected input src 9-k, for k=0..9.
IN_LEN = 10


# --------------------------------------------------------------------------- #
# Generic rendering helpers (adapted from run_phase3_analysis._bar/_comp_heatmap)
# --------------------------------------------------------------------------- #
def _bar(values, labels, title, path, highlight=None, ylabel="score", n3_caption=None):
    fig, ax = plt.subplots(figsize=(13, 3.6))
    colors = ["#c0392b" if (highlight and labels[i] in highlight) else "#3477b8"
              for i in range(len(values))]
    ax.bar(range(len(values)), values, color=colors)
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_title(title, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=9)
    fig.tight_layout()
    N3.annotate(fig, n3_caption)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _comp_heatmap(mat, valid, title, path, baseline, n_heads=8, n3_caption=None,
                  circuit_mask=None, circuit_color="black"):
    """Triangular sender->receiver composition heatmap. Composition is CAUSAL: a
    sender head only composes into a receiver in a STRICTLY LATER layer. So the
    last-layer senders (rows) and first-layer receivers (cols) never participate
    and are cropped out, and the remaining invalid (same/earlier-layer) cells are
    greyed -- leaving the meaningful upper-block-triangular region only (no wasted
    half-plot).

    When ``circuit_mask`` (boolean [n,n] sender x receiver, True = keep) is given,
    the structurally-VALID cells outside the mask are painted ``circuit_color``
    (opaque overlay) and the colour scale comes from the kept cells only -- used
    to scope the view to a circuit's heads without changing the kept values.
    Invalid cells stay grey (a different category: structurally impossible)."""
    m = mat.cpu().numpy().copy().astype(float)
    m[~valid.cpu().numpy()] = np.nan
    n = m.shape[0]
    # crop the structurally-empty band: drop last-layer senders (last n_heads rows)
    # and first-layer receivers (first n_heads cols).
    row_hi, col_lo = n - n_heads, n_heads
    m = m[:row_hi, col_lo:]
    cmask = (np.asarray(circuit_mask, dtype=bool)[:row_hi, col_lo:]
             if circuit_mask is not None else None)
    row_labels = [f"L{i // n_heads}H{i % n_heads}" for i in range(row_hi)]
    col_labels = [f"L{(col_lo + j) // n_heads}H{(col_lo + j) % n_heads}" for j in range(n - col_lo)]

    fig, ax = plt.subplots(figsize=(8, 7))
    cmap = plt.get_cmap("magma").copy(); cmap.set_bad("0.85")   # invalid cells -> light grey
    vmax = float(np.nanmax(m[cmask] if cmask is not None else m))
    im = ax.imshow(m, cmap=cmap, vmin=0, vmax=vmax, aspect="auto")
    if cmask is not None:
        from matplotlib.colors import to_rgba
        overlay = np.zeros(m.shape + (4,))                       # transparent
        overlay[(~cmask) & ~np.isnan(m)] = to_rgba(circuit_color)
        ax.imshow(overlay, aspect="auto")
    ax.set_xticks(range(len(col_labels))); ax.set_xticklabels(col_labels, rotation=90, fontsize=5)
    ax.set_yticks(range(len(row_labels))); ax.set_yticklabels(row_labels, fontsize=5)
    ax.set_xlabel("receiver head (later layer)"); ax.set_ylabel("sender head (earlier layer)")
    for k in range(0, row_hi + 1, n_heads):
        ax.axhline(k - 0.5, color="w", lw=0.4)
    for k in range(0, (n - col_lo) + 1, n_heads):
        ax.axvline(k - 0.5, color="w", lw=0.4)
    ax.set_title(title + f"  (random baseline={baseline:.3f})", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    N3.annotate(fig, n3_caption)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _token_heatmap(mat, title, path, xlabel, ylabel, cmap, diverging=False,
                   n3_caption=None):
    """[d, d] token x token heatmap with integer tick labels 0..d-1."""
    m = mat.cpu().numpy() if isinstance(mat, torch.Tensor) else np.asarray(mat)
    d = m.shape[0]
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    if diverging:
        vmax = float(np.nanmax(np.abs(m))) or 1.0
        im = ax.imshow(m, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")
    else:
        im = ax.imshow(m, cmap=cmap, aspect="auto")
    ax.set_xticks(range(d)); ax.set_xticklabels(range(d), fontsize=7)
    ax.set_yticks(range(m.shape[0])); ax.set_yticklabels(range(m.shape[0]), fontsize=7)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    N3.annotate(fig, n3_caption)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 1. Copy-score bar chart (weight-based)                                      #
# --------------------------------------------------------------------------- #
def plot_copy_scores(model, heads=None, save=None):
    """Bar chart of the OV eigenvalue copying score for every head. Returns
    (labels, scores)."""
    if heads is None:
        heads = [(l, h) for l in range(model.cfg.n_layers)
                 for h in range(model.cfg.n_heads)]
    labels = [f"L{l}H{h}" for (l, h) in heads]
    scores = [E.ov_circuit(model, l, h).copying_score for (l, h) in heads]
    if save is None:
        save = os.path.join(FIG_DIR, "copy_scores.png")
    highlight = {f"L{l}H{h}" for (l, h) in MOVERS}
    _bar(scores, labels,
         "OV eigenvalue copying score per head (red = known movers)\n"
         "weight-based -- independent of test_data",
         save, highlight=highlight, ylabel="copying score [-1,1]",
         n3_caption=N3.INPUT_INDEPENDENT)
    return labels, scores


# --------------------------------------------------------------------------- #
# 2. Composition heatmaps (weight-based)                                      #
# --------------------------------------------------------------------------- #
def plot_composition(model, save_dir=None):
    """32x32 Q/K/V composition heatmaps. Returns the CompositionResult."""
    if save_dir is None:
        save_dir = FIG_DIR
    comp = E.qkv_composition(model)
    for ch, mat in (("q", comp.q_comp), ("k", comp.k_comp), ("v", comp.v_comp)):
        _comp_heatmap(mat, comp.valid_mask,
                      f"{ch.upper()}-composition (sender -> receiver)\n"
                      f"weight-based -- independent of test_data",
                      os.path.join(save_dir, f"composition_{ch}.png"), comp.baseline,
                      n3_caption=N3.INPUT_INDEPENDENT)
    return comp


# --------------------------------------------------------------------------- #
# 3. Per-head plots                                                           #
# --------------------------------------------------------------------------- #
def _realized_attention(model, data, layer, head):
    """Mean attention pattern of (layer, head) on the dataset's tokens (``data`` is
    a Segrev3Single, so ``data.tokens`` = [input | segrev3(input)]).
    Returns ([dest, src] tensor, stats dict).

    Besides the POSITIONAL view (mean [dest, src] pattern -- the reversal mirror
    is positional), also computes the CONTENT split: per destination, how much
    attention mass lands on sources holding the boundary digit 3 vs non-3
    sources, against the uniform-attention baseline (the fraction of 3s in each
    destination's causal prefix). The positional mean blurs the 3s' locations
    (they sit at random positions across the batch); the content split is what
    shows boundary-seeking."""
    _, cache = model.run_with_cache(data.tokens)
    patt = cache[f"blocks.{layer}.attn.hook_pattern"][:, head]   # [batch, dest, src]
    mean_patt = patt.mean(dim=0)                                  # [dest, src]
    n = mean_patt.shape[0]

    # Per-output-destination max attention (which src each dest most attends to).
    argmax_src = [int(x) for x in mean_patt.argmax(dim=1)]
    max_attn = [float(x) for x in mean_patt.max(dim=1).values]

    # Reversal mirror: output dest 9+k should attend to reflected input src 9-k.
    # Average the attention mass landing on the reflected source over output dests.
    refl_mass = []
    for k in range(IN_LEN):
        dest = IN_LEN - 1 + k          # 9..18
        src = IN_LEN - 1 - k           # 9..0
        if dest < n and src < n:
            refl_mass.append(float(mean_patt[dest, src]))

    # CONTENT split: collapse the source axis from POSITION to DIGIT -- for each
    # destination, the attention mass landing on all sources holding digit d
    # (batch-meaned) -> [dest, 10]. Column BOUNDARY_DIGIT tracks 3-attention;
    # rows sum to 1 over the causal prefix, so non-3 mass = 1 - column 3.
    tokens = data.tokens.to(patt.device)
    onehot = torch.nn.functional.one_hot(tokens, num_classes=10).float()    # [B,S,10]
    mass_by_digit = torch.einsum("bds,bsv->bdv", patt, onehot).mean(dim=0)  # [dest,10]
    mass3 = mass_by_digit[:, BOUNDARY_DIGIT]                      # [dest]
    is3 = (tokens == BOUNDARY_DIGIT).float()                      # [B, S]
    # uniform-attention baseline: fraction of 3s in each dest's causal prefix
    denom = torch.arange(1, tokens.shape[1] + 1, device=patt.device).float()
    base3 = (is3.cumsum(dim=1) / denom).mean(dim=0)               # [dest]
    ans = slice(IN_LEN - 1, 2 * IN_LEN - 1)                       # answer dests 9..18

    stats = {
        "argmax_src_per_dest": argmax_src,
        "mean_max_attention": float(np.mean(max_attn)),
        "mean_attention_to_reflected_position": float(np.mean(refl_mass)) if refl_mass else None,
        # content split (source-digit axis; column 3 = boundary attention)
        "attention_mass_by_source_digit_per_dest": [[float(v) for v in row]
                                                    for row in mass_by_digit],
        "attention_mass_to_threes_per_dest": [float(x) for x in mass3],
        "uniform_baseline_to_threes_per_dest": [float(x) for x in base3],
        "mean_mass_to_threes_at_answer_dests": float(mass3[ans].mean()),
        "mean_mass_to_non_threes_at_answer_dests": float(1.0 - mass3[ans].mean()),
        "uniform_baseline_to_threes_at_answer_dests": float(base3[ans].mean()),
    }
    return mean_patt, stats


def _attn_content_plot(stats, layer, head, label, path, n3_caption):
    """Heatmap of realized attention by SOURCE CONTENT -- the same format as the
    positional ``*_attn.png`` heatmap (dest position on y, viridis mass scale)
    but with the x-axis collapsed from source POSITION to source DIGIT:
    cell (dest, d) = mean attention mass from that destination onto all sources
    holding digit d. The framed column is the boundary digit 3 -- read it
    top-to-bottom to track how much each destination attends to 3s vs non-3s."""
    from matplotlib.patches import Rectangle
    m = np.asarray(stats["attention_mass_by_source_digit_per_dest"])   # [S, 10]
    n, d = m.shape
    fig, ax = plt.subplots(figsize=(4.8, 5.0))
    im = ax.imshow(m, cmap="viridis", vmin=0, vmax=float(m.max()) or 1.0, aspect="auto")
    ax.set_xticks(range(d)); ax.set_xticklabels(range(d), fontsize=7)
    ax.set_yticks(range(n)); ax.set_yticklabels(range(n), fontsize=6)
    ax.set_xlabel("source digit"); ax.set_ylabel("destination position")
    ax.add_patch(Rectangle((BOUNDARY_DIGIT - 0.5, -0.5), 1.0, float(n), fill=False,
                           edgecolor="#c0392b", lw=1.6))
    ax.get_xticklabels()[BOUNDARY_DIGIT].set_color("#c0392b")
    ax.set_title(f"L{layer}H{head} realized mean attention by source digit\n"
                 f"test_data: {label}", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    N3.annotate(fig, n3_caption)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_head(model, layer, head, pairs=None, save_dir=None,
              label=None, attn_save_dir=None):
    """Render the QK (content), OV, and -- if ``pairs`` is given -- the realized
    mean attention pattern for a SPECIFIC (layer, head). Returns a dict of stats.

    WEIGHT-BASED (input-independent): the QK content circuit, OV circuit and copying
    score are read off the weights and DO NOT depend on ``pairs``. They are written
    FLAT into ``save_dir`` (default ``HEAD_DIR``) and titled ``weight-based --
    independent of test_data``.

    DATA-DEPENDENT: ``pairs`` only drives the realized-attention (``*_attn.png``) plot
    and its stats. When ``pairs`` is given, the attn plot goes into ``attn_save_dir``
    (per-dataset; default ``HEAD_DIR`` for ad-hoc use) and is titled ``test_data:
    <label>`` (``label`` defaults to "data").
    """
    if save_dir is None:
        save_dir = HEAD_DIR
    os.makedirs(save_dir, exist_ok=True)

    # --- QK content circuit (CONTENT ONLY; no positional panel) -- WEIGHT-BASED ---
    qk = E.qk_circuit(model, layer, head, include_positional=False)
    qk_path = os.path.join(save_dir, f"L{layer}H{head}_QK.png")
    _token_heatmap(
        qk.token_matrix,
        f"L{layer}H{head} content QK circuit (query token -> key token)\n"
        f"weight-based -- independent of test_data",
        qk_path, xlabel="key token", ylabel="query token",
        cmap="RdBu_r", diverging=True, n3_caption=N3.INPUT_INDEPENDENT,
    )

    # --- OV circuit -- WEIGHT-BASED ---
    ov = E.ov_circuit(model, layer, head)
    ov_path = os.path.join(save_dir, f"L{layer}H{head}_OV.png")
    _token_heatmap(
        ov.token_matrix,
        f"L{layer}H{head} OV circuit (attended -> output)  copy={ov.copying_score:+.3f}\n"
        f"weight-based -- independent of test_data",
        ov_path, xlabel="output token", ylabel="attended token",
        cmap="RdBu_r", diverging=True, n3_caption=N3.INPUT_INDEPENDENT,
    )

    qk_summary = qk.summary(top_k=5)
    ov_summary = ov.summary(top_k=5)
    out = {
        "layer": layer, "head": head,
        "copying_score": ov.copying_score,
        "qk_top_query_to_key": qk_summary["top_query_to_key"],
        "ov_top_attended_to_output": ov_summary["top_attended_to_output"],
    }

    # --- realized attention (DATA-DEPENDENT) ---
    if pairs is not None:
        if label is None:
            label = "data"
        if attn_save_dir is None:
            attn_save_dir = HEAD_DIR
        os.makedirs(attn_save_dir, exist_ok=True)
        mean_patt, stats = _realized_attention(model, pairs, layer, head)
        attn_path = os.path.join(attn_save_dir, f"L{layer}H{head}_attn.png")
        m = mean_patt.cpu().numpy()
        n = m.shape[0]
        fig, ax = plt.subplots(figsize=(5.6, 5.0))
        im = ax.imshow(m, cmap="viridis", vmin=0, vmax=float(m.max()), aspect="auto")
        ax.set_xticks(range(n)); ax.set_xticklabels(range(n), fontsize=6)
        ax.set_yticks(range(n)); ax.set_yticklabels(range(n), fontsize=6)
        ax.set_xlabel("source position"); ax.set_ylabel("destination position")
        ax.set_title(f"L{layer}H{head} realized mean attention\n"
                     f"test_data: {label}", fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        # The pattern is a mean over this dataset's inputs -> mark their 3-count.
        N3.annotate(fig, N3.caption(getattr(pairs, "inputs", None)))
        fig.savefig(attn_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        # CONTENT companion: mass on 3-sources vs non-3 sources per destination.
        _attn_content_plot(stats, layer, head, label,
                           os.path.join(attn_save_dir, f"L{layer}H{head}_attn_content.png"),
                           N3.caption(getattr(pairs, "inputs", None)))
        out["realized_attention"] = stats

    return out


# --------------------------------------------------------------------------- #
# Driver                                                                       #
# --------------------------------------------------------------------------- #
WEIGHT_BASED_NOTE = ("QK/OV/copy/composition are weight-based (input-independent); "
                     "test_data only drives the *_attn realized-attention plots/stats "
                     "(see attn_by_dataset).")


def _cleanup_stale_flat_attn():
    """Remove the old FLAT data-dependent ``*_attn.png`` from the previous version.
    They are superseded by the per-dataset ``edge/<label>/specific_heads/`` copies.
    The flat ``_QK.png``/``_OV.png`` (weight-based) STAY."""
    for p in glob.glob(os.path.join(HEAD_DIR, "*_attn.png")):
        os.remove(p)


def main(data_paths="all", heads=None, circuit=None, scope="full_circuit",
         fig_dir=None, head_dir=None, data_dir=None, model=None):
    """Generate all edge-circuit figures + the JSON/MD report.

    WEIGHT-BASED figures (copy_scores, composition_{q,k,v}, per-head QK/OV) are
    written ONCE, FLAT, and are independent of test_data. DATA-DEPENDENT realized
    attention (``*_attn.png``) is produced PER DATASET into ``edge/<label>/``.

    ``data_paths`` accepts a single path string, a list of path strings, or the
    literal "all" (-> every test_data/*.json, sorted). Default = "all".

    CIRCUIT SCOPING (optional): when ``circuit`` (a circuit_loader.Circuit) is given,
    the per-head plot set defaults to the circuit's heads, the copy-score bar and the
    per-head QK/OV/attn plots are restricted to circuit heads, and a circuit-wiring
    composition figure ``composition_circuit.png`` is emitted (with per-edge Q/K/V
    composition scores). ``scope="ablated"`` additionally mean-ablates every head NOT
    in the circuit while computing the realized-attention pass (circuit-only model);
    ``scope="full_circuit"`` (default) leaves the full model intact. The weight-based
    plots are input-independent so ``scope`` does not affect them.

    OUTPUT DIRS: ``fig_dir``/``head_dir``/``data_dir`` override the module defaults
    ``FIG_DIR``/``HEAD_DIR``/``DATA_DIR`` when given (threaded through to the plot
    helpers and the JSON/MD writes).
    """
    # Resolve output dirs (local; do not rely on module globals once overridden).
    fig_dir = FIG_DIR if fig_dir is None else fig_dir
    head_dir = HEAD_DIR if head_dir is None else head_dir
    data_dir = DATA_DIR if data_dir is None else data_dir

    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(head_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    # Only clean up the canonical flat dir (the legacy stale-attn cleanup).
    if head_dir == HEAD_DIR:
        _cleanup_stale_flat_attn()

    if model is None:
        model = load_segrev3()

    circuit_heads = CL.head_tuples(circuit) if circuit is not None else None
    if heads is None:
        heads = list(circuit_heads) if circuit is not None else list(DEFAULT_HEADS)

    paths = _resolve_data_paths(data_paths)
    if not paths:
        raise ValueError(f"no datasets resolved from data_paths={data_paths!r}")
    dataset_labels = [_label_of(p) for p in paths]

    all_heads = [(l, h) for l in range(model.cfg.n_layers)
                 for h in range(model.cfg.n_heads)]

    # ===================================================================== #
    # WEIGHT-BASED (computed ONCE, written FLAT, independent of test_data)   #
    # ===================================================================== #
    # 1. copy scores -- over circuit heads only when a circuit is given.
    copy_heads = circuit_heads if circuit is not None else all_heads
    labels, scores = plot_copy_scores(
        model, heads=copy_heads, save=os.path.join(fig_dir, "copy_scores.png"))
    copy_scores = {labels[i]: float(scores[i]) for i in range(len(labels))}
    top_copy = sorted(copy_scores.items(), key=lambda kv: -kv[1])[:8]

    # 2. composition -- always the full 32x32; plus a circuit-wiring figure when scoped.
    comp = plot_composition(model, save_dir=fig_dir)
    comp_summary = comp.summary(top_k=8)
    if circuit is not None:
        # Per-edge composition score on the circuit's (non-self-loop) edges.
        # comp.{q,k,v}_comp are [32,32] sender x receiver tensors indexed by l*8+h.
        def _sidx(lh):
            return lh[0] * model.cfg.n_heads + lh[1]
        ev = {}
        for e in circuit.edges:
            s = CL.parse_head_label(e["sender"])
            r = CL.parse_head_label(e["receiver"])
            if s is None or r is None or s == r:        # skip non-heads + self-loops
                continue
            si, ri = _sidx(s), _sidx(r)
            ch = e.get("channel")
            chan_map = {"q": comp.q_comp, "k": comp.k_comp, "v": comp.v_comp}
            if ch in chan_map:
                val = float(chan_map[ch][si][ri])
            else:                                        # channel None -> max over q/k/v
                val = max(float(comp.q_comp[si][ri]),
                          float(comp.k_comp[si][ri]),
                          float(comp.v_comp[si][ri]))
            ev[(e["sender"], e["receiver"])] = val
        CL.draw_circuit_metric(
            circuit, {}, "Q/K/V composition on circuit edges",
            os.path.join(fig_dir, "composition_circuit.png"),
            edge_values=ev, diverging=False, subtitle=N3.INPUT_INDEPENDENT)

    # 3. per-head WEIGHT-BASED QK/OV (flat) -- attn done per-dataset below.
    #    First pass: no pairs -> only the flat weight-based QK/OV + weight summaries.
    head_weight_results = {}
    for (l, h) in heads:
        head_weight_results[f"L{l}H{h}"] = plot_head(
            model, l, h, pairs=None, save_dir=head_dir)

    # ===================================================================== #
    # DATA-DEPENDENT (realized attention) -- one pass per dataset            #
    # ===================================================================== #
    # When circuit-scoped with scope="ablated", run the realized-attention pass
    # inside a complement-ablation so the patterns reflect the circuit-only model.
    ablate = (circuit is not None and scope == "ablated")
    attn_by_dataset = {}
    with (CL.ablate_complement(model, circuit_heads)
          if ablate else contextlib.nullcontext()):
        for path, label in zip(paths, dataset_labels):
            # accept either a single (n_threes) OR a paired (clean/corrupted)
            # dataset path -- the circuit-scoped analysis (manual_circuit.py
            # --circuit) passes the scheme's paired file.
            data = load_tokens_any(path)
            n_threes = getattr(data, "n_threes", None)
            attn_dir = (os.path.join(fig_dir, label, "specific_heads")
                        if fig_dir != FIG_DIR else _attn_head_dir(label))
            os.makedirs(attn_dir, exist_ok=True)
            per_head = {}
            for (l, h) in heads:
                r = plot_head(model, l, h, pairs=data, save_dir=head_dir,
                              label=label, attn_save_dir=attn_dir)
                per_head[f"L{l}H{h}"] = r.get("realized_attention", {})
            attn_by_dataset[label] = {
                "data_path": repo_relpath(os.path.abspath(path)),
                "n_threes": n_threes,
                "specific_heads_attn": per_head,
            }

    # ---- results JSON ----
    results = {
        "meta": {
            "weight_based_note": WEIGHT_BASED_NOTE,
            "test_data_datasets": dataset_labels,
        },
        # ---- WEIGHT-BASED sections (written ONCE; input-independent) ----
        "copy_scores": copy_scores,
        "copy_scores_top8": [{"head": k, "copying_score": v} for k, v in top_copy],
        "composition_baseline": comp.baseline,
        "composition_top8": {
            "Q": comp_summary["top_q_composition"],
            "K": comp_summary["top_k_composition"],
            "V": comp_summary["top_v_composition"],
        },
        "specific_heads": head_weight_results,   # weight-based QK/OV summaries only
        # ---- DATA-DEPENDENT section (per dataset) ----
        "attn_by_dataset": attn_by_dataset,
    }
    if circuit is not None:
        results["meta"]["circuit"] = {
            "name": circuit.name,
            "app": circuit.app,
            "scope": scope,
            "heads": CL.head_labels(circuit),
            "n_heads": len(circuit_heads),
        }
    json_path = os.path.join(data_dir, "edge_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    # ---- report MD ----
    lines = []
    lines.append("# Edge-circuit (Phase 3, weight-based) report\n")
    lines.append(f"**test_data (attn):** {', '.join(dataset_labels)}\n")
    lines.append("- **QK / OV / copying-score / composition are WEIGHT-BASED "
                 "(independent of test_data); written once, flat.**")
    lines.append("- The realized-attention plots/stats (`*_attn.png`) ARE "
                 "data-dependent and were computed per dataset for: "
                 f"{', '.join(dataset_labels)} "
                 "(see `figs/edge/<label>/specific_heads/` and `attn_by_dataset`).\n")

    lines.append("## Top OV copy-score heads  (weight-based)")
    for k, v in top_copy:
        lines.append(f"- {k}: {v:+.3f}")
    lines.append("")

    lines.append(f"## Top composition edges  (weight-based; random baseline = {comp.baseline:.3f})")
    for ch in ("Q", "K", "V"):
        edges = results["composition_top8"][ch][:3]
        s = ", ".join(f"{e['sender']}->{e['receiver']} ({e['score']:.3f})" for e in edges)
        lines.append(f"- {ch}: {s}")
    lines.append("")

    lines.append("## Requested specific heads  (QK/OV weight-based)")
    for label_h, r in head_weight_results.items():
        qk0 = r["qk_top_query_to_key"][0]
        ov0 = r["ov_top_attended_to_output"][0]
        lines.append(
            f"- {label_h}: copy={r['copying_score']:+.3f}; "
            f"QK top q{qk0['query_token']}->k{qk0['key_token']} ({qk0['score']:.2f}); "
            f"OV top attend {ov0['attended_token']}->out {ov0['boosts_output_token']} ({ov0['score']:.2f})"
        )
    lines.append("")

    lines.append("## Realized attention: reflected position + 3-content split  (DATA-DEPENDENT, per dataset)")
    for label in dataset_labels:
        lines.append(f"### test_data: {label}")
        per_head = attn_by_dataset[label]["specific_heads_attn"]
        for label_h in head_weight_results:
            attn = per_head.get(label_h, {})
            refl = attn.get("mean_attention_to_reflected_position")
            refl_s = f"{refl:.3f}" if refl is not None else "n/a"
            m3 = attn.get("mean_mass_to_threes_at_answer_dests")
            b3 = attn.get("uniform_baseline_to_threes_at_answer_dests")
            m3_s = (f"{m3:.3f} (uniform baseline {b3:.3f})"
                    if m3 is not None and b3 is not None else "n/a")
            mm = attn.get("mean_max_attention")
            mm_s = f"{mm:.3f}" if mm is not None else "n/a"
            lines.append(f"- {label_h}: mean-mass-to-reflected-pos={refl_s}, "
                         f"mass-on-3-sources@answers={m3_s}, "
                         f"mean-max-attention={mm_s}")
        lines.append("")

    lines.append("## Generated figures")
    lines.append("Weight-based (flat, independent of test_data):")
    lines.append("- figs/edge/copy_scores.png")
    lines.append("- figs/edge/composition_q.png, composition_k.png, composition_v.png")
    for label_h in head_weight_results:
        lines.append(f"- figs/edge/specific_heads/{label_h}_QK.png, {label_h}_OV.png")
    lines.append("")
    lines.append("Data-dependent realized attention (per dataset; *_attn = positional "
                 "view, *_attn_content = 3-vs-non-3 source split):")
    for label in dataset_labels:
        heads_s = ", ".join(f"{lh}_attn{{,_content}}.png" for lh in head_weight_results)
        lines.append(f"- figs/edge/{label}/specific_heads/: {heads_s}")
    lines.append("")

    md_path = os.path.join(data_dir, "edge_report.md")
    with open(md_path, "w") as f:
        f.write("\n".join(lines))

    # ---- console summary ----
    print("=== vis_edge summary ===")
    print("test_data (attn) datasets:", dataset_labels)
    print("Top copy-score heads:", [(k, round(v, 2)) for k, v in top_copy[:6]])
    print("Composition baseline:", round(comp.baseline, 4))
    print("Top V-composition edges:",
          [(e["sender"] + "->" + e["receiver"], round(e["score"], 3))
           for e in comp_summary["top_v_composition"][:5]])
    print("Top Q-composition edges:",
          [(e["sender"] + "->" + e["receiver"], round(e["score"], 3))
           for e in comp_summary["top_q_composition"][:5]])
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    print(f"figures in {fig_dir}")
    return results


if __name__ == "__main__":
    # minimalist --data parsing, mirroring vis_patching_dla.py: one or more dataset
    # paths after --data (default "all" -> every datasets/edge/*.json).
    _data = "all"
    if "--data" in sys.argv:
        i = sys.argv.index("--data") + 1
        _data = sys.argv[i:]            # one or more paths after --data
        if len(_data) == 1:
            _data = _data[0]
    main(data_paths=_data)
