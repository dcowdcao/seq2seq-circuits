"""
vis_patching_dla.py
=================================================================================
Phase-2 (activation patching + DLA) VISUALIZATION driver for the segrev3 model.

This script is PURELY a visualization / reporting driver: every numeric result
comes from the pure-compute library ``interp_tools/patching_dla.py`` (do not put
analysis logic here). For EACH dataset it runs the four Phase-2 techniques on a
contrast ``pairs`` object and renders them as heatmaps plus a structured JSON +
markdown report:

    1. activation_patching  (component="z", direction="noising")  -> 4x8 heatmap
    2. attribution_patching (component="z", direction="denoising", EAP-IG(8)) -> 4x8
    3. direct_logit_attribution (mode="correct_vs_rest")          -> 4x8 heatmap
       NB: DLA does NO patching, so it runs on SINGLE sequences -- the dataset's
       clean side only (correct digit vs mean of the other 9, positions 9..18);
       the corrupted partner is never touched.
    4. path_patching_qkv_sweep (direction="noising")              -> 32x32 heatmap

DATASET PROVENANCE & MULTI-DATASET SUPPORT
--------------------------------------------------------------------------------
ALL FOUR plots are DATA-DEPENDENT (driven by the loaded clean/corrupted pairs),
so every output is organized by, and labelled with, the dataset it came from.

  * A dataset's **label** is the json filename stem, e.g.
    ``os.path.splitext(os.path.basename(path))[0]`` -> "random_example".
  * ``main(data_paths="all")`` accepts:
      - a single path string,
      - a list of path strings,
      - the literal string "all" (-> sorted glob of ``test_data/*.json``).
    Default is "all". The argument is normalized to a list of paths and each
    dataset is processed in turn.
  * Per-dataset outputs are written into label-named SUBFOLDERS:
      figures -> figs_data/figs/vis_patching_dla/<label>/{activation,attribution,dla,path}_heatmap.png
      data    -> figs_data/data/<label>/patching_dla_results.json
                 figs_data/data/<label>/patching_dla_report.md
  * Provenance is recorded everywhere:
      - every PNG carries a ``test_data: <label>`` line in its title (suptitle),
      - every JSON has ``meta["test_data"] = "<label>"`` and
        ``meta["data_path"] = <abs path>``,
      - every markdown report's first line after the H1 is
        ``**test_data:** <label>``.

The stale FLAT outputs written by the previous (single-dataset) version are
removed at startup (``_cleanup_flat_outputs``) so the tree contains only the
per-dataset subfolders.

RUNTIME
--------------------------------------------------------------------------------
``path_patching_qkv_sweep`` is the slow analysis. To keep multi-dataset runs
tractable the pairs used for the path sweep are SUBSAMPLED to at most
``PATH_SWEEP_MAX`` (48) examples; the count actually used is recorded in
``meta["path_sweep_batch"]``. The other three analyses use the full dataset.

CUSTOM DATA
--------------------------------------------------------------------------------
To test the techniques on your own contrast, generate a dataset with data_gen
and drop the JSON into ``datasets/manual/`` (or point ``--data`` at any path):

    from data_gen import make_clean_corrupted
    make_clean_corrupted(64, scheme="replace_non_threes_to_threes", seed=1,
                         save_path=".../datasets/manual/my_example.json")

The expected JSON format is a data_gen dataset with keys:
    scheme, n, seed, boundary_digit, input_len, answer_positions,
    clean_inputs, corrupted_inputs
(tokens/answers are reconstructed deterministically by data_gen).

Usage:
    python vis_patching_dla.py                       # all datasets in test_data/
    python vis_patching_dla.py --data /abs/a.json    # one dataset
    python vis_patching_dla.py --data a.json b.json   # a subset
"""

from __future__ import annotations

import dataclasses
import glob
import json
import os
import sys

import numpy as np
import torch

# --------------------------------------------------------------------------- #
# sys.path setup (repo convention)                                            #
# --------------------------------------------------------------------------- #
_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_THIS))   # repo root
for p in ("interp_tools", "data_gen", "metrics"):
    sys.path.insert(0, os.path.join(_ROOT, p))

import patching_dla as P                                            # noqa: E402
import n_threes as N3                                               # noqa: E402
from model_loading import load_segrev3, repo_relpath                # noqa: E402
from data_gen import load_pairs, make_clean_corrupted               # noqa: E402

# circuit_loader is only needed for the circuit-scoped figures; import it lazily
# so the default (circuit=None) pipeline does not depend on it being present.
try:
    import circuit_loader as CL                                     # noqa: E402
except Exception:                                                   # pragma: no cover
    CL = None

# --------------------------------------------------------------------------- #
# Output locations                                                            #
# --------------------------------------------------------------------------- #
FIG_DIR = os.path.join(_THIS, "figs_data", "figs", "vis_patching_dla")
DATA_DIR = os.path.join(_THIS, "figs_data", "data")
# Manual-analysis datasets live in the central store: datasets/manual/<scheme>.json.
TEST_DATA_DIR = os.path.join(_ROOT, "datasets", "manual")
DEFAULT_DATA = os.path.join(TEST_DATA_DIR, "random.json")

# Slow path sweep: subsample to at most this many examples per dataset.
PATH_SWEEP_MAX = 48


def _label_of(path: str) -> str:
    """Dataset label = json filename stem (e.g. 'random_example')."""
    return os.path.splitext(os.path.basename(path))[0]


def _resolve_data_paths(data_paths):
    """Normalize ``data_paths`` (single str path, list of str paths, or the
    literal 'all') into a list of dataset paths."""
    if data_paths is None or data_paths == "all":
        return sorted(glob.glob(os.path.join(TEST_DATA_DIR, "*.json")))
    if isinstance(data_paths, str):
        return [data_paths]
    return list(data_paths)


def _cleanup_flat_outputs():
    """Idempotently remove the stale FLAT outputs from the previous (single
    dataset) version, which are superseded by the per-dataset subfolders. Only
    removes the four flat PNGs directly under FIG_DIR and the two flat
    patching_dla_* files directly under DATA_DIR -- leaves everything else."""
    for name in ("activation", "attribution", "dla", "path"):
        f = os.path.join(FIG_DIR, f"{name}_heatmap.png")
        if os.path.isfile(f):
            os.remove(f)
    for name in ("patching_dla_results.json", "patching_dla_report.md"):
        f = os.path.join(DATA_DIR, name)
        if os.path.isfile(f):
            os.remove(f)


# --------------------------------------------------------------------------- #
# Plot helpers (matplotlib, headless) -- adapted from run_phase2_analysis.py  #
# --------------------------------------------------------------------------- #


def _heatmap_4x8(grid: np.ndarray, title: str, label: str, path: str,
                 diverging: bool = True, fmt: str = "{:+.2f}", mask: np.ndarray = None,
                 n3_caption: str = None, cbar_label: str = None,
                 bad_color: str = "0.85"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    nL, nH = grid.shape
    # When ``mask`` is given (boolean [nL,nH], True = keep), cells outside the mask
    # are rendered as NaN so the cmap paints them ``bad_color`` (default light
    # grey) -- used to scope a 4x8 grid to a circuit's heads without changing the
    # colour scale of the kept cells.
    if mask is not None:
        grid = grid.copy()
        grid[~mask] = np.nan
    vmax = float(np.nanmax(np.abs(grid))) or 1.0
    fig, ax = plt.subplots(figsize=(nH * 0.95 + 1.5, nL * 0.95 + 1.4))
    base = "RdBu_r" if diverging else "magma"
    cmap = matplotlib.cm.get_cmap(base).copy()
    cmap.set_bad(bad_color)
    im = ax.imshow(grid, cmap=cmap, vmin=-vmax if diverging else 0, vmax=vmax, aspect="auto")
    ax.set_xticks(range(nH)); ax.set_xticklabels([f"H{h}" for h in range(nH)])
    ax.set_yticks(range(nL)); ax.set_yticklabels([f"L{l}" for l in range(nL)])
    for l in range(nL):
        for h in range(nH):
            if np.isnan(grid[l, h]):
                continue
            ax.text(h, l, fmt.format(grid[l, h]), ha="center", va="center",
                    fontsize=7, color="black")
    ax.set_title(f"{title}\ntest_data: {label}", fontsize=10)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if cbar_label:
        cbar.set_label(cbar_label, fontsize=7)
    fig.tight_layout(); N3.annotate(fig, n3_caption)
    fig.savefig(path, dpi=140, bbox_inches="tight"); plt.close(fig)


def _heatmap_32x32(mat: np.ndarray, title: str, label: str, path: str,
                   n3_caption: str = None, cbar_label: str = None,
                   mask: np.ndarray = None, bad_color: str = "0.85"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = mat.shape[0]
    # When ``mask`` is given (boolean [n,n], True = keep), cells outside the mask
    # are rendered as NaN so the cmap paints them ``bad_color``, and the colour
    # scale comes from the kept cells only (cf. _heatmap_4x8).
    if mask is not None:
        mat = mat.copy()
        mat[~mask] = np.nan
    vmax = float(np.nanmax(np.abs(mat))) or 1.0
    fig, ax = plt.subplots(figsize=(9, 8.4))
    cmap = matplotlib.cm.get_cmap("RdBu_r").copy()
    cmap.set_bad(bad_color)
    im = ax.imshow(mat, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")
    labels = [f"L{i//8}H{i%8}" for i in range(n)]
    ax.set_xticks(range(n)); ax.set_xticklabels(labels, rotation=90, fontsize=5)
    ax.set_yticks(range(n)); ax.set_yticklabels(labels, fontsize=5)
    ax.set_xlabel("receiver head"); ax.set_ylabel("sender head")
    for k in range(0, n + 1, 8):
        ax.axhline(k - 0.5, color="k", lw=0.5); ax.axvline(k - 0.5, color="k", lw=0.5)
    ax.set_title(f"{title}\ntest_data: {label}", fontsize=10)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if cbar_label:
        cbar.set_label(cbar_label, fontsize=7)
    fig.tight_layout(); N3.annotate(fig, n3_caption)
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)


# --------------------------------------------------------------------------- #
# Reductions / helpers                                                        #
# --------------------------------------------------------------------------- #


def _top_heads(grid: torch.Tensor, k: int = 10):
    """Top-k heads of a [nL, nH] grid by |value|, labelled L{l}H{h}."""
    nH = grid.shape[1]
    vals, idx = torch.topk(grid.abs().flatten(), min(k, grid.numel()))
    return [{"head": f"L{int(i)//nH}H{int(i)%nH}", "value": float(grid.flatten()[int(i)])}
            for i in idx]


def _combine_qkv(mats: dict) -> np.ndarray:
    """Combine the q/k/v [32,32] sender x receiver matrices into one 32x32 by
    keeping, for each cell, the SIGNED value of the channel with the largest |value|
    (strongest causal sender->receiver effect across Q/K/V)."""
    stacked = np.stack([mats["q"], mats["k"], mats["v"]], axis=0)  # [3, 32, 32]
    chan = np.abs(stacked).argmax(axis=0)                          # [32, 32]
    combined = np.take_along_axis(stacked, chan[None], axis=0)[0]  # [32, 32] signed
    return combined


def _top_edges(mat: np.ndarray, nH: int = 8, k: int = 15):
    flat = np.abs(mat).flatten()
    order = flat.argsort()[::-1][:k]
    edges = []
    for idx in order:
        s, r = divmod(int(idx), mat.shape[1])
        if mat[s, r] == 0:
            continue
        edges.append({"sender": f"L{s//nH}H{s%nH}", "receiver": f"L{r//nH}H{r%nH}",
                      "effect": float(mat[s, r])})
    return edges


def _subsample_pairs(pairs, n: int):
    """Return a Segrev3Pairs whose tensor fields are sliced to the first ``n``
    rows (keeping ``scheme`` and any non-tensor fields). Used to make the slow
    path sweep tractable."""
    sliced = {}
    for f in dataclasses.fields(pairs):
        v = getattr(pairs, f.name)
        if isinstance(v, torch.Tensor) and v.dim() >= 1:
            sliced[f.name] = v[:n]
    return dataclasses.replace(pairs, **sliced)


# --------------------------------------------------------------------------- #
# Core driver                                                                 #
# --------------------------------------------------------------------------- #


def _circuit_mask(circuit, nL: int, nH: int) -> np.ndarray:
    """Boolean [nL, nH] grid, True for heads that belong to ``circuit``."""
    mask = np.zeros((nL, nH), dtype=bool)
    for (l, h) in CL.head_tuples(circuit):
        mask[l, h] = True
    return mask


def _grid_to_head_values(grid: np.ndarray) -> dict:
    """{(l, h): value} for every cell of a [nL, nH] grid (for draw_circuit_metric)."""
    nL, nH = grid.shape
    return {(l, h): float(grid[l, h]) for l in range(nL) for h in range(nH)}


def _filter_top_by_circuit(top: list, circuit) -> list:
    """Restrict a _top_heads list to heads that belong to ``circuit``."""
    keep = set(CL.head_labels(circuit))
    return [d for d in top if d["head"] in keep]


def make_figs(pairs, label: str, model=None, data_path: str = "<in-memory>",
              circuit=None, scope: str = "full_circuit", fig_dir=None, data_dir=None):
    """Run the four Phase-2 techniques on `pairs` for dataset ``label``, writing
    4 figures + results JSON + markdown report into the per-dataset subfolders.
    Returns the results dict.

    When ``circuit`` (a ``circuit_loader.Circuit``) is given the figures/JSON are
    additionally SCOPED to that circuit's heads+edges:
      * ``scope="full_circuit"`` (default): analyses run on the full model; the 4x8
        heatmaps are masked to the circuit's heads, the circuit graph is painted
        with each per-head metric (``*_circuit.png``), and the combined path matrix
        is projected onto the circuit's edges (``path_circuit.png``).
      * ``scope="ablated"``: the four analyses run with the circuit's COMPLEMENT
        mean-ablated (``circuit_loader.ablate_complement``); figures are the same.
    With ``circuit=None`` the behaviour is identical to the un-scoped driver.

    ``fig_dir`` / ``data_dir`` override the default per-label subfolders (the driver
    may pre-compute grouped paths); if None the defaults under FIG_DIR/DATA_DIR are
    used."""
    if fig_dir is None:
        fig_dir = os.path.join(FIG_DIR, label)
    if data_dir is None:
        data_dir = os.path.join(DATA_DIR, label)
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    if circuit is not None and CL is None:
        raise RuntimeError(
            "circuit-scoped make_figs requested but circuit_loader could not be imported")
    if model is None:
        model = load_segrev3()
    nL, nH = model.cfg.n_layers, model.cfg.n_heads
    batch = int(pairs.clean_tokens.shape[0])

    # Optionally mean-ablate the circuit's complement for the four analyses below.
    import contextlib
    if circuit is not None and scope == "ablated":
        ablate_cm = CL.ablate_complement(model, CL.head_tuples(circuit))
    else:
        ablate_cm = contextlib.nullcontext()

    with ablate_cm:
        # 1. activation patching (noising, z per-head) -> reduce [4,8,20] -> [4,8]
        # SIGNED: take the effect at the position of max |effect| (position is
        # picked by magnitude so strong negative effects are not hidden, but the
        # displayed value keeps its sign).
        print("[1/4] activation_patching (z, noising) ...")
        act = P.activation_patching(model, pairs, component="z", direction="noising")
        act_grid = act.effect.gather(
            2, act.effect.abs().argmax(dim=2, keepdim=True)).squeeze(2)  # [4, 8] signed

        # 2. attribution patching (EAP-IG(8), z, denoising) -> [4,8], signed likewise
        print("[2/4] attribution_patching (z, denoising, EAP-IG 8) ...")
        attr = P.attribution_patching(model, pairs, component="z", direction="denoising",
                                      n_ig_steps=8)
        attr_grid = attr.effect.gather(
            2, attr.effect.abs().argmax(dim=2, keepdim=True)).squeeze(2)  # [4, 8] signed

        # 3. direct logit attribution -> [4,8]. DLA does NO patching, so it runs
        #    on SINGLE sequences (the clean tokens only -- the corrupted side is
        #    never touched): correct digit vs the MEAN of the other 9, over all
        #    answer positions 9..18.
        print("[3/4] direct_logit_attribution (correct_vs_rest; clean sequences only) ...")
        dla = P.direct_logit_attribution(model, pairs, mode="correct_vs_rest")
        dla_grid = dla.head_attribution                                 # [4, 8]

        # 4. path patching qkv sweep (slow) -> three [32,32] -> one combined 32x32
        #    Subsample pairs to PATH_SWEEP_MAX to keep the slow sweep tractable.
        sweep_pairs = _subsample_pairs(pairs, PATH_SWEEP_MAX)
        path_sweep_batch = int(sweep_pairs.clean_tokens.shape[0])
        print(f"[4/4] path_patching_qkv_sweep (noising) -- slow, "
              f"batch={path_sweep_batch} ...")
        path_ok = True
        try:
            sweep = P.path_patching_qkv_sweep(model, sweep_pairs, direction="noising")
        except Exception as exc:
            # ablate_complement may compose awkwardly with the path sweep's own
            # hooking; under ablated scope we tolerate that and skip the path figure.
            if circuit is not None and scope == "ablated":
                print(f"  [path sweep] skipped under ablated scope: {exc}")
                path_ok = False
            else:
                raise
        if path_ok:
            mats = {ch: sweep.matrices[ch].cpu().numpy() for ch in ("q", "k", "v")}
            combined = _combine_qkv(mats)
            to_logits = sweep.to_logits.cpu().numpy()
        else:
            mats = {ch: np.zeros((nL * nH, nL * nH)) for ch in ("q", "k", "v")}
            combined = np.zeros((nL * nH, nL * nH))
            to_logits = np.zeros(nL * nH)

    # ---- figures ----
    # Mark the number of 3s (segrev3 boundary digit) in the clean inputs that drove
    # these analyses. Each figure aggregates over the whole batch, so the honest
    # label is the batch's 3-count distribution (see interp_tools/n_threes.py).
    n3 = N3.caption(pairs.clean_inputs, prefix="clean n_3s")
    n3_path = N3.caption(_subsample_pairs(pairs, PATH_SWEEP_MAX).clean_inputs,
                         prefix="clean n_3s") if path_ok else None
    scoped = circuit is not None
    mask = _circuit_mask(circuit, nL, nH) if scoped else None
    _heatmap_4x8(act_grid.cpu().numpy(),
                 "Activation patching (z, noising; signed effect at max-|effect| position)",
                 label, os.path.join(fig_dir, "activation_heatmap.png"), fmt="{:+.2f}",
                 mask=mask, n3_caption=n3,
                 cbar_label="signed E at the position of max |E|,  E = (LD patched - LD clean) / (LD corrupted - LD clean)\n"
                            "LD = logit diff;  E=1: corrupting this head's z alone destroys clean behavior")
    _heatmap_4x8(attr_grid.cpu().numpy(),
                 "Attribution patching EAP-IG(8) (z, denoising; signed effect at max-|effect| position)",
                 label, os.path.join(fig_dir, "attribution_heatmap.png"), fmt="{:+.3f}",
                 mask=mask, n3_caption=n3,
                 cbar_label="signed E at the position of max |E|,  E = EAP-IG(8) gradient estimate of\n"
                            "(LD patched - LD corrupted) / (LD clean - LD corrupted);  LD = logit diff")
    _heatmap_4x8(dla_grid.cpu().numpy(),
                 "Direct logit attribution (correct vs mean-of-rest; positions 9..18)\n"
                 "test_data: SINGLE sequences (clean side only -- no patching, no corrupted partner)",
                 label, os.path.join(fig_dir, "dla_heatmap.png"), fmt="{:+.2f}",
                 mask=mask, n3_caption=n3,
                 cbar_label="SIGNED (no abs): head's direct write onto the\n"
                            "(correct - mean-of-rest) logit direction, in logits")
    if path_ok:
        _heatmap_32x32(combined,
                       "Path patching: combined Q/K/V (max|.| signed) sender -> receiver (noising)",
                       label, os.path.join(fig_dir, "path_heatmap.png"), n3_caption=n3_path,
                       cbar_label="signed E of strongest-|.| channel among Q/K/V,  E = (LD patched - LD clean) /\n"
                                  "(LD corrupted - LD clean);  patch = corrupted sender z into receiver's Q/K/V input")

    # ---- circuit-scoped figures: the circuit graph painted with each metric ----
    if scoped:
        CL.draw_circuit_metric(
            circuit, _grid_to_head_values(act_grid.cpu().numpy()),
            f"Activation patching (z, noising) on circuit {circuit.name}\ntest_data: {label}",
            os.path.join(fig_dir, "activation_circuit.png"), diverging=True, subtitle=n3)
        CL.draw_circuit_metric(
            circuit, _grid_to_head_values(attr_grid.cpu().numpy()),
            f"Attribution patching EAP-IG(8) on circuit {circuit.name}\ntest_data: {label}",
            os.path.join(fig_dir, "attribution_circuit.png"), diverging=True,
            value_fmt="{:+.3f}", subtitle=n3)
        CL.draw_circuit_metric(
            circuit, _grid_to_head_values(dla_grid.cpu().numpy()),
            f"Direct logit attribution (correct vs mean-of-rest) on circuit {circuit.name}\n"
            f"test_data: {label} (SINGLE sequences -- clean side only)",
            os.path.join(fig_dir, "dla_circuit.png"), diverging=True, subtitle=n3)
        if path_ok:
            # project the combined path matrix onto this circuit's edges (sidx=l*8+h)
            edge_values = {}
            for e in circuit.edges:
                s, r = e["sender"], e["receiver"]
                if s == r:                       # skip self-loops (e.g. L0H7->L0H7)
                    continue
                slh, rlh = CL.parse_head_label(s), CL.parse_head_label(r)
                if slh is None or rlh is None:
                    continue
                sidx, ridx = slh[0] * nH + slh[1], rlh[0] * nH + rlh[1]
                edge_values[(s, r)] = float(combined[sidx, ridx])
            CL.draw_circuit_metric(
                circuit, {},
                f"Path patching (combined Q/K/V) on circuit {circuit.name}\ntest_data: {label}",
                os.path.join(fig_dir, "path_circuit.png"), edge_values=edge_values,
                diverging=True, value_fmt="{:+.3f}", subtitle=n3_path)

    # ---- results json ----
    activation_top = _top_heads(act_grid)
    attribution_top = _top_heads(attr_grid)
    dla_top = _top_heads(dla_grid)
    if scoped:
        activation_top = _filter_top_by_circuit(activation_top, circuit)
        attribution_top = _filter_top_by_circuit(attribution_top, circuit)
        dla_top = _filter_top_by_circuit(dla_top, circuit)
    results = {
        "meta": {
            "test_data": label,
            "data_path": repo_relpath(os.path.abspath(data_path))
                         if os.path.isfile(str(data_path)) else data_path,
            "scheme": pairs.scheme,
            "batch": batch,
            # DLA does no patching: it ran on the SINGLE clean sequences only
            # (correct vs mean-of-rest), never reading the corrupted side.
            "dla_mode": "correct_vs_rest",
            "dla_test_data": "single sequences (clean side only)",
            "path_sweep_batch": path_sweep_batch,
            "n_layers": nL,
            "n_heads": nH,
            "clean_logit_diff": act.clean_metric,
            "corrupted_logit_diff": act.corrupted_metric,
        },
        "activation_grid": act_grid.tolist(),
        "attribution_grid": attr_grid.tolist(),
        "dla_grid": dla_grid.tolist(),
        "dla_component_attribution": {k: float(v) for k, v in dla.component_attribution.items()},
        "activation_top": activation_top,
        "attribution_top": attribution_top,
        "dla_top": dla_top,
        "path_matrices": {ch: mats[ch].tolist() for ch in ("q", "k", "v")},
        "path_combined": combined.tolist(),
        "path_to_logits": to_logits.tolist(),
        "path_top_edges": _top_edges(combined, nH=nH, k=15),
    }
    if scoped:
        results["meta"]["circuit"] = {
            "name": circuit.name,
            "app": circuit.app,
            "scope": scope,
            "heads": CL.head_labels(circuit),
            "n_heads": len(circuit.heads),
        }
    results_name = "circuit_patching_dla_results.json" if scoped else "patching_dla_results.json"
    with open(os.path.join(data_dir, results_name), "w") as f:
        json.dump(results, f, indent=2)

    # ---- markdown report ----
    _write_report(results, os.path.join(data_dir, "patching_dla_report.md"))

    # ---- console summary ----
    print(f"\n=== vis_patching_dla summary [test_data: {label}] ===")
    print("DLA top (direct logit writers):",
          [(d["head"], round(d["value"], 2)) for d in results["dla_top"][:6]])
    print("Activation top (necessity):",
          [(d["head"], round(d["value"], 3)) for d in results["activation_top"][:6]])
    print("Attribution top (EAP-IG):",
          [(d["head"], round(d["value"], 3)) for d in results["attribution_top"][:6]])
    print("Path top edges (combined):",
          [(e["sender"] + "->" + e["receiver"], round(e["effect"], 3))
           for e in results["path_top_edges"][:6]])
    print(f"\nwrote figures to {fig_dir}")
    print(f"wrote data to {data_dir}")
    return results


def _write_report(results: dict, path: str):
    m = results["meta"]
    def fmt_heads(lst, n=8):
        return ", ".join(f"`{d['head']}` ({d['value']:+.3f})" for d in lst[:n])
    def fmt_edges(lst, n=8):
        return ", ".join(f"`{e['sender']}->{e['receiver']}` ({e['effect']:+.3f})" for e in lst[:n])
    # top DLA writers by signed value (writers that push toward correct answer)
    dla_signed = sorted(results["dla_top"], key=lambda d: -d["value"])
    lines = [
        "# Phase-2 patching + DLA report",
        "",
        f"**test_data:** {m['test_data']}",
        "",
        f"- **Data**: `{m['data_path']}` (scheme=`{m['scheme']}`, batch={m['batch']}, "
        f"path_sweep_batch={m['path_sweep_batch']}, "
        f"{m['n_layers']} layers x {m['n_heads']} heads)",
        f"- **Clean logit-diff**: {m['clean_logit_diff']:+.3f}  |  "
        f"**Corrupted logit-diff**: {m['corrupted_logit_diff']:+.3f}",
        "",
        "## Top direct logit writers (DLA, single-position contrast)",
        f"By |attribution|: {fmt_heads(results['dla_top'])}",
        "",
        f"Most positive (write toward correct answer): "
        f"{fmt_heads(dla_signed)}",
        "",
        "## Top heads by activation-patching importance (z, noising = necessity)",
        fmt_heads(results["activation_top"]),
        "",
        "## Top heads by attribution-patching importance (EAP-IG(8), denoising)",
        fmt_heads(results["attribution_top"]),
        "",
        "## Top causal path edges (combined Q/K/V, noising)",
        fmt_edges(results["path_top_edges"]),
        "",
        "## Figures",
        "- `activation_heatmap.png` -- 4x8 activation-patching (z, noising) head necessity, signed effect at the max-|effect| position.",
        "- `attribution_heatmap.png` -- 4x8 EAP-IG(8) attribution-patching (z, denoising) head importance, signed likewise.",
        "- `dla_heatmap.png` -- 4x8 direct-logit-attribution: each head's direct write onto the (correct-incorrect) logit direction.",
        "- `path_heatmap.png` -- 32x32 sender->receiver path effect, combining Q/K/V by the strongest-|.| channel (signed, RdBu_r).",
        "",
    ]
    with open(path, "w") as f:
        f.write("\n".join(lines))


def main(data_paths="all"):
    """Drive all four analyses for one or more datasets. ``data_paths`` may be a
    single path string, a list of path strings, or the literal "all" (default ->
    all ``test_data/*.json``). For each dataset, outputs go into label-named
    subfolders under the figs/ and data/ trees, with provenance recorded in every
    figure title, JSON meta and markdown report."""
    torch.manual_seed(0)
    _cleanup_flat_outputs()
    paths = _resolve_data_paths(data_paths)
    if not paths:
        raise ValueError(f"no datasets resolved from data_paths={data_paths!r}")
    model = load_segrev3()
    out = {}
    for path in paths:
        label = _label_of(path)
        print(f"\n########## dataset: {label}  ({path}) ##########")
        pairs = load_pairs(path)
        out[label] = make_figs(pairs, label, model=model, data_path=path)
    print(f"\nProcessed {len(paths)} dataset(s): {[_label_of(p) for p in paths]}")
    return out


if __name__ == "__main__":
    _data = "all"
    if "--data" in sys.argv:
        i = sys.argv.index("--data") + 1
        _data = sys.argv[i:]            # one or more paths after --data
        if len(_data) == 1:
            _data = _data[0]
    main(data_paths=_data)
