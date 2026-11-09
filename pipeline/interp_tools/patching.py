"""Per-circuit activation/attribution/path patching + DLA for pipeline.

Compute is delegated to the repo-root ``interp_tools/patching_dla.py`` backend;
this CLI writes, for ``--model <m> --circuit <c>``:
  circuits_data/<m>/<c>/heads_only/dla.json                                (+ .png)
  circuits_data/<m>/<c>/heads_only/clean_corrupt_dataset/<c>/{act,att}_patching.json
  circuits_data/<m>/<c>/edges/path_patching/clean_corrupt_dataset/<c>/path_patching.json
with 1:1 PNG mirrors under circuits_figures/<m>/<c>/. Data is the circuit's single
stored pool via ``model_registry.load_circuit_pairs`` (shared with edges.py and the
finders); ``--smoke`` re-roots all outputs under ``.smoke/run1/``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

# sys.path bootstrap: model_registry itself puts the shared repo module dirs on sys.path.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_ROOT = os.path.dirname(_THIS_DIR)
_MODELS_DIR = os.path.join(PIPELINE_ROOT, "models")
if _MODELS_DIR not in sys.path:
    sys.path.insert(0, _MODELS_DIR)

import model_registry as MG                                # noqa: E402
import patching_dla as P                                  # noqa: E402  (repo interp_tools)
import n_threes as N3                                     # noqa: E402  (repo interp_tools)

GENERATED_BY = "pipeline/interp_tools/patching.py"
SMOKE_ROOT = os.path.join(PIPELINE_ROOT, ".smoke", "run1")

DIRECTIONS = ("noising", "denoising")
PATH_DIRECTION = "noising"          # the source vis driver's convention
DLA_TOL = 1e-2                      # documented exactness tolerance (patching_dla.py)
IG_STEPS = 30                       # EAP-IG steps (frozen by repo rule)
ANALYSES = ("act", "att", "path", "dla")


def _smoke_rebase(path: str) -> str:
    """Re-root a PIPELINE_ROOT-relative artifact path under .smoke/run1/."""
    return os.path.join(SMOKE_ROOT, os.path.relpath(path, PIPELINE_ROOT))


def artifact_paths(name: str, kind: str, circuit: str,
                   smoke: bool = False) -> tuple[str, str]:
    """(json_path, png_path) for one artifact kind, built EXCLUSIVELY from the
    model_registry path builders; the ``clean_corrupt_dataset/<circuit>`` level
    carries the circuit name (its single dataset)."""
    droot, froot = MG.data_dir(name, circuit), MG.fig_dir(name, circuit)
    if kind == "dla":
        dj = os.path.join(droot, "heads_only", "dla.json")
        fp = os.path.join(froot, "heads_only", "dla.png")
    elif kind in ("act", "att"):
        dj = os.path.join(droot, "heads_only", "clean_corrupt_dataset",
                          circuit, f"{kind}_patching.json")
        fp = os.path.join(froot, "heads_only", "clean_corrupt_dataset",
                          circuit, f"{kind}_patching.png")
    elif kind == "path":
        dj = os.path.join(droot, "edges", "path_patching",
                          "clean_corrupt_dataset", circuit, "path_patching.json")
        fp = os.path.join(froot, "edges", "path_patching",
                          "clean_corrupt_dataset", circuit, "path_patching.png")
    else:
        raise ValueError(f"unknown artifact kind {kind!r}")
    if smoke:
        dj, fp = _smoke_rebase(dj), _smoke_rebase(fp)
    return dj, fp


def _sibling_png(png_path: str, suffix: str) -> str:
    """foo.png -> foo_<suffix>.png (per-direction / per-channel figure mirror)."""
    base, ext = os.path.splitext(png_path)
    return f"{base}_{suffix}{ext}"


def make_meta(*, model: str, pair: str, scheme: str, n_examples: int, seed: int,
              method: str, checkpoint: str, source_dataset_path: str) -> dict:
    """The shared meta envelope — EXACTLY these nine keys."""
    return {
        "model": model,
        "pair": pair,
        "scheme": scheme,
        "n_examples": int(n_examples),
        "seed": int(seed),
        "method": method,
        "checkpoint": MG.repo_relpath(checkpoint),
        "generated_by": GENERATED_BY,
        "source_dataset_path": MG.repo_relpath(source_dataset_path),
    }


def _write_json(path: str, obj: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
    tqdm.write(f"  wrote {path}")


def signed_max_reduce(effect: torch.Tensor) -> torch.Tensor:
    """[nL, nH, P] -> [nL, nH]: the SIGNED effect at the max-|effect| position."""
    return effect.gather(2, effect.abs().argmax(dim=2, keepdim=True)).squeeze(2)


def combine_qkv(mats: dict) -> np.ndarray:
    """Per cell, the SIGNED value of the Q/K/V channel with the largest |value|."""
    stacked = np.stack([mats["q"], mats["k"], mats["v"]], axis=0)
    chan = np.abs(stacked).argmax(axis=0)
    return np.take_along_axis(stacked, chan[None], axis=0)[0]


# Heatmap conventions: RdBu_r symmetric-about-0, NaN cells light grey, H0../L0..
# axes, value annotations on per-head grids, DPI 140 per-head / 150 head x head.


def _draw_head_grid(ax, grid: np.ndarray, fmt: str):
    """Render one [nL, nH] per-head grid onto ``ax``; returns the image handle."""
    nL, nH = grid.shape
    finite_max = np.nanmax(np.abs(grid)) if np.isfinite(grid).any() else 1.0
    vmax = float(finite_max) or 1.0
    cmap = matplotlib.colormaps["RdBu_r"].copy()   # modern API (cm.get_cmap removed in mpl>=3.9)
    cmap.set_bad("0.85")
    im = ax.imshow(grid, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(nH)); ax.set_xticklabels([f"H{h}" for h in range(nH)])
    ax.set_yticks(range(nL)); ax.set_yticklabels([f"L{l}" for l in range(nL)])
    for l in range(nL):
        for h in range(nH):
            if np.isnan(grid[l, h]):
                continue
            ax.text(h, l, fmt.format(grid[l, h]), ha="center", va="center",
                    fontsize=7, color="black")
    return im


def heatmap_heads(grid: np.ndarray, title: str, path: str,
                  fmt: str = "{:+.2f}", cbar_label: Optional[str] = None,
                  n3_caption: Optional[str] = None) -> None:
    """Single per-head [nL, nH] heatmap (DPI 140)."""
    nL, nH = grid.shape
    fig, ax = plt.subplots(figsize=(nH * 0.95 + 1.5, nL * 0.95 + 1.4))
    im = _draw_head_grid(ax, grid, fmt)
    ax.set_title(title, fontsize=10)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if cbar_label:
        cbar.set_label(cbar_label, fontsize=7)
    fig.tight_layout(); N3.annotate(fig, n3_caption)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight"); plt.close(fig)
    tqdm.write(f"  wrote {path}")


def heatmap_heads_panels(grids, panel_titles, suptitle: str, path: str,
                         fmt: str = "{:+.2f}", cbar_label: Optional[str] = None,
                         n3_caption: Optional[str] = None) -> None:
    """k side-by-side per-head [nL, nH] heatmaps (one per direction), each with
    its own symmetric-about-0 scale + colorbar. DPI 140."""
    k = len(grids)
    nL, nH = grids[0].shape
    fig, axes = plt.subplots(1, k, figsize=(k * (nH * 0.95 + 1.8), nL * 0.95 + 1.9))
    axes = np.atleast_1d(axes)
    for ax, g, t in zip(axes, grids, panel_titles):
        im = _draw_head_grid(ax, g, fmt)
        ax.set_title(t, fontsize=9)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        if cbar_label:
            cbar.set_label(cbar_label, fontsize=6)
    fig.suptitle(suptitle, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93)); N3.annotate(fig, n3_caption)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight"); plt.close(fig)
    tqdm.write(f"  wrote {path}")


def heatmap_headhead(mat: np.ndarray, title: str, path: str, n_heads: int,
                     cbar_label: Optional[str] = None,
                     n3_caption: Optional[str] = None) -> None:
    """(L*H) x (L*H) sender x receiver heatmap (RdBu_r symmetric-about-0,
    layer block separators every n_heads, DPI 150)."""
    n = mat.shape[0]
    finite_max = np.nanmax(np.abs(mat)) if np.isfinite(mat).any() else 1.0
    vmax = float(finite_max) or 1.0
    fig, ax = plt.subplots(figsize=(9, 8.4))
    cmap = matplotlib.colormaps["RdBu_r"].copy()   # modern API (cm.get_cmap removed in mpl>=3.9)
    cmap.set_bad("0.85")
    im = ax.imshow(mat, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")
    labels = [f"L{i // n_heads}H{i % n_heads}" for i in range(n)]
    ax.set_xticks(range(n)); ax.set_xticklabels(labels, rotation=90, fontsize=5)
    ax.set_yticks(range(n)); ax.set_yticklabels(labels, fontsize=5)
    ax.set_xlabel("receiver head"); ax.set_ylabel("sender head")
    for kk in range(0, n + 1, n_heads):
        ax.axhline(kk - 0.5, color="k", lw=0.5); ax.axvline(kk - 0.5, color="k", lw=0.5)
    ax.set_title(title, fontsize=10)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if cbar_label:
        cbar.set_label(cbar_label, fontsize=7)
    fig.tight_layout(); N3.annotate(fig, n3_caption)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    tqdm.write(f"  wrote {path}")


# Per-analysis runners (compute via patching_dla; write JSON + figure mirrors).


def _title(base: str, name: str, circuit: str) -> str:
    """Every figure title carries model + dataset (the circuit)."""
    return f"{base}\nmodel: {name} | pair: {circuit}"


_ACT_CBAR = {
    "noising": ("signed E at the position of max |E|,  "
                "E = (LD patched - LD clean) / (LD corrupted - LD clean)\n"
                "LD = logit diff;  E=1: corrupting this head's z alone destroys clean behavior"),
    "denoising": ("signed E at the position of max |E|,  "
                  "E = (LD patched - LD corrupted) / (LD clean - LD corrupted)\n"
                  "LD = logit diff;  E=1: restoring this head's z alone restores clean behavior"),
}


def run_act(model, name, circuit, pairs, src_path, n_used, seed_used,
            ckpt, smoke) -> str:
    """activation_patching on component 'z' (per-head), BOTH directions in one
    JSON + per-direction heatmaps + a 2-panel 1:1 mirror PNG."""
    results = {d: P.activation_patching(model, pairs, component="z", direction=d)
               for d in DIRECTIONS}
    grids = {d: signed_max_reduce(results[d].effect) for d in DIRECTIONS}
    json_path, fig_path = artifact_paths(name, "act", circuit, smoke)
    any_res = results["noising"]
    payload = {
        "meta": make_meta(model=name, pair=circuit, scheme=pairs.scheme,
                          n_examples=n_used, seed=seed_used,
                          method="activation_patching", checkpoint=ckpt,
                          source_dataset_path=src_path),
        "component": "z",
        "reduction": "signed effect at the max-|effect| position (per head)",
        "clean_logit_diff": any_res.clean_metric,
        "corrupted_logit_diff": any_res.corrupted_metric,
    }
    for d in DIRECTIONS:
        payload[d] = grids[d].tolist()
        payload[f"{d}_detail"] = {
            "effect_layer_head_position": results[d].effect.tolist(),
            "positions": list(results[d].coords["position"]),
            "summary": results[d].summary(top_k=12),
        }
    _write_json(json_path, payload)

    n3 = N3.caption(pairs.clean_inputs, prefix="clean n_3s")
    for d in DIRECTIONS:
        heatmap_heads(
            grids[d].numpy(),
            _title(f"Activation patching (z, {d}; signed effect at max-|effect| position)",
                   name, circuit),
            _sibling_png(fig_path, d), fmt="{:+.2f}",
            cbar_label=_ACT_CBAR[d], n3_caption=n3)
    heatmap_heads_panels(
        [grids[d].numpy() for d in DIRECTIONS], list(DIRECTIONS),
        _title("Activation patching (z; signed effect at max-|effect| position)",
               name, circuit),
        fig_path, fmt="{:+.2f}",
        cbar_label="signed E at max-|E| position (per-direction normalization; "
                   "see per-direction figures)",
        n3_caption=n3)
    top = results["noising"].summary(top_k=1)["top_sites"][0]
    tqdm.write(f"  [act:{circuit}] clean LD={any_res.clean_metric:+.4f} "
               f"corrupted LD={any_res.corrupted_metric:+.4f} | noising top site: {top}")
    return json_path


def run_att(model, name, circuit, pairs, src_path, n_used, seed_used,
            ckpt, smoke) -> str:
    """attribution_patching (EAP-IG, IG_STEPS steps) on component 'z', BOTH
    directions (each costs only IG_STEPS forward+backward passes)."""
    results = {d: P.attribution_patching(model, pairs, component="z", direction=d,
                                         n_ig_steps=IG_STEPS)
               for d in DIRECTIONS}
    grids = {d: signed_max_reduce(results[d].effect) for d in DIRECTIONS}
    json_path, fig_path = artifact_paths(name, "att", circuit, smoke)
    any_res = results["noising"]
    payload = {
        "meta": make_meta(model=name, pair=circuit, scheme=pairs.scheme,
                          n_examples=n_used, seed=seed_used,
                          method="attribution_patching", checkpoint=ckpt,
                          source_dataset_path=src_path),
        "component": "z",
        "n_ig_steps": IG_STEPS,
        "reduction": "signed effect at the max-|effect| position (per head)",
        "clean_logit_diff": any_res.clean_metric,
        "corrupted_logit_diff": any_res.corrupted_metric,
    }
    for d in DIRECTIONS:
        payload[d] = grids[d].tolist()
        payload[f"{d}_detail"] = {
            "effect_layer_head_position": results[d].effect.tolist(),
            "positions": list(results[d].coords["position"]),
            "summary": results[d].summary(top_k=12),
        }
    _write_json(json_path, payload)

    n3 = N3.caption(pairs.clean_inputs, prefix="clean n_3s")
    cbar = (f"signed E at the position of max |E|,  E = EAP-IG({IG_STEPS}) gradient "
            "estimate\nof the activation-patching effect, normalized by the "
            "clean-corrupted LD gap")
    for d in DIRECTIONS:
        heatmap_heads(
            grids[d].numpy(),
            _title(f"Attribution patching EAP-IG({IG_STEPS}) (z, {d}; "
                   "signed effect at max-|effect| position)", name, circuit),
            _sibling_png(fig_path, d), fmt="{:+.3f}", cbar_label=cbar, n3_caption=n3)
    heatmap_heads_panels(
        [grids[d].numpy() for d in DIRECTIONS], list(DIRECTIONS),
        _title(f"Attribution patching EAP-IG({IG_STEPS}) (z; signed effect at "
               "max-|effect| position)", name, circuit),
        fig_path, fmt="{:+.3f}", cbar_label=cbar, n3_caption=n3)
    top = results["denoising"].summary(top_k=1)["top_sites"][0]
    tqdm.write(f"  [att:{circuit}] IG steps={IG_STEPS} | denoising top site: {top}")
    return json_path


def run_path(model, name, circuit, pairs, src_path, n_used, seed_used,
             ckpt, smoke) -> str:
    """path_patching_qkv_sweep -> q/k/v (L*H)x(L*H) matrices + to_logits vector
    + combined (max-|.| per cell); heatmaps for q/k/v/combined + to_logits."""
    nL, nH = model.cfg.n_layers, model.cfg.n_heads
    sweep = P.path_patching_qkv_sweep(model, pairs, direction=PATH_DIRECTION)
    mats = {ch: sweep.matrices[ch].cpu().numpy() for ch in ("q", "k", "v")}
    combined = combine_qkv(mats)
    to_logits = sweep.to_logits.cpu().numpy()
    json_path, fig_path = artifact_paths(name, "path", circuit, smoke)
    payload = {
        "meta": make_meta(model=name, pair=circuit, scheme=pairs.scheme,
                          n_examples=n_used, seed=seed_used,
                          method="path_patching_qkv_sweep", checkpoint=ckpt,
                          source_dataset_path=src_path),
        "direction": PATH_DIRECTION,
        "clean_logit_diff": sweep.clean_metric,
        "corrupted_logit_diff": sweep.corrupted_metric,
        "head_labels": [f"L{i // nH}H{i % nH}" for i in range(nL * nH)],
        "q": mats["q"].tolist(),
        "k": mats["k"].tolist(),
        "v": mats["v"].tolist(),
        "to_logits": to_logits.tolist(),
        "combined": combined.tolist(),
        "combined_rule": "per cell, the SIGNED value of the Q/K/V channel with max |value|",
        "summary": sweep.summary(top_k=12),
    }
    _write_json(json_path, payload)

    n3 = N3.caption(pairs.clean_inputs, prefix="clean n_3s")
    ch_cbar = ("signed E,  E = (LD patched - LD clean) / (LD corrupted - LD clean);\n"
               "patch = corrupted sender z into the receiver's {} input (direct path only)")
    for ch in ("q", "k", "v"):
        heatmap_headhead(
            mats[ch],
            _title(f"Path patching: sender -> receiver.{ch.upper()} ({PATH_DIRECTION})",
                   name, circuit),
            _sibling_png(fig_path, ch), n_heads=nH,
            cbar_label=ch_cbar.format(ch.upper()), n3_caption=n3)
    heatmap_headhead(
        combined,
        _title(f"Path patching: combined Q/K/V (max|.| signed) sender -> receiver "
               f"({PATH_DIRECTION})", name, circuit),
        fig_path, n_heads=nH,
        cbar_label=("signed E of strongest-|.| channel among Q/K/V,  "
                    "E = (LD patched - LD clean) /\n(LD corrupted - LD clean);  "
                    "patch = corrupted sender z into receiver's Q/K/V input"),
        n3_caption=n3)
    heatmap_heads(
        to_logits.reshape(nL, nH),
        _title(f"Path patching: head -> logits direct effect ({PATH_DIRECTION})",
               name, circuit),
        _sibling_png(fig_path, "to_logits"), fmt="{:+.2f}",
        cbar_label=("signed E,  E = (LD patched - LD clean) / (LD corrupted - LD clean);\n"
                    "patch = corrupted sender z, all other heads frozen; read logits"),
        n3_caption=n3)
    top_v = (sweep.summary(top_k=1)["top_edges"].get("v") or [None])[0]
    tqdm.write(f"  [path:{circuit}] clean LD={sweep.clean_metric:+.4f} "
               f"corrupted LD={sweep.corrupted_metric:+.4f} | top V edge: {top_v}")
    return json_path


def run_dla(model, name, circuit, pairs, src_path, n_used, seed_used,
            ckpt, smoke) -> str:
    """direct_logit_attribution(mode='single') on the circuit dataset; asserts
    the documented additive-exactness property (tol DLA_TOL)."""
    dla = P.direct_logit_attribution(model, pairs, mode="single")
    total = (float(dla.head_attribution.sum())
             + float(sum(dla.component_attribution.values())))
    abs_err = abs(total - dla.logit_diff)
    if abs_err >= DLA_TOL:
        raise AssertionError(
            f"DLA exactness violated: sum(DLA)={total:+.6f} vs decomposed "
            f"logit_diff={dla.logit_diff:+.6f} (|err|={abs_err:.2e} >= {DLA_TOL})")
    json_path, fig_path = artifact_paths(name, "dla", circuit, smoke)
    payload = {
        "meta": make_meta(model=name, pair=circuit, scheme=pairs.scheme,
                          n_examples=n_used, seed=seed_used,
                          method="direct_logit_attribution", checkpoint=ckpt,
                          source_dataset_path=src_path),
        "mode": dla.mode,
        "logit_diff": dla.logit_diff,
        "head_attribution": dla.head_attribution.tolist(),
        "component_attribution": {k: float(v)
                                  for k, v in dla.component_attribution.items()},
        "head_labels": list(dla.head_labels),
        "exactness": {"dla_total": total, "decomposed_logit_diff": dla.logit_diff,
                      "abs_error": abs_err, "tolerance": DLA_TOL},
        "summary": dla.summary(top_k=12),
    }
    _write_json(json_path, payload)

    heatmap_heads(
        dla.head_attribution.numpy(),
        _title("Direct logit attribution (mode=single: per-example single "
               "contrast position)", name, circuit),
        fig_path, fmt="{:+.2f}",
        cbar_label=("SIGNED (no abs): head's direct write onto the (correct - incorrect)\n"
                    "logit direction at the single contrast position, in logits"),
        n3_caption=N3.caption(pairs.clean_inputs, prefix="clean n_3s"))
    tops = [(h["layer"], h["head"]) for h in dla.summary(top_k=3)["top_heads"]]
    tqdm.write(f"  [dla:{circuit}] sum(DLA)={total:+.4f} logit_diff={dla.logit_diff:+.4f} "
               f"|err|={abs_err:.2e} (<{DLA_TOL}) | top heads: {tops}")
    return json_path


# Dataset access + driver.


def circuit_pairs(name: str, circuit: str, n: int) -> tuple[MG.Segrev3Pairs, str, int, int]:
    """(pairs, source_dataset_path, n_used, dataset_seed) from the circuit's
    single stored pool; the recorded seed is the one that GENERATED the data."""
    pairs, path = MG.load_circuit_pairs(name, circuit, n=n, consumer="attribution")
    return pairs, path, int(pairs.clean_tokens.shape[0]), int(pairs.extra["seed"])


def run_model(name: str, args) -> list[str]:
    """All requested analyses for one model+circuit, sharing ONE model load and
    ONE dataset load."""
    spec = MG.get_spec(name)
    analyses = ANALYSES if args.analysis == "all" else (args.analysis,)
    torch.manual_seed(args.seed)
    MG.wait_for_headroom(context=f"patching.py load_model({name})")
    model = MG.load_model(name)
    written = []
    try:
        pairs, src, n_used, seed_used = circuit_pairs(name, args.circuit, args.n)
        tqdm.write(f"[{name}] circuit {args.circuit}: n={n_used} (source={src})")
        runners = {"act": run_act, "att": run_att, "path": run_path, "dla": run_dla}
        for a in analyses:
            written.append(runners[a](model, name, args.circuit, pairs, src,
                                      n_used, seed_used, spec.checkpoint, args.smoke))
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return written


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Per-circuit activation/attribution/path patching + DLA "
                    "(JSON + heatmaps) for pipeline.")
    ap.add_argument("--model", required=True,
                    help="model name (a DSL algorithm_id)")
    ap.add_argument("--circuit", required=True,
                    help="circuit slug: selects circuits_data/<model>/<circuit>/ "
                         "and the circuit's single gen_data inputs.json")
    ap.add_argument("--analysis", default="all",
                    choices=list(ANALYSES) + ["all"],
                    help="which analysis to run (default: all)")
    ap.add_argument("-n", type=int, default=MG.APP_SIZES["manual"],
                    help=f"pairs to load from the circuit pool (default {MG.APP_SIZES['manual']}; "
                         "capped by the pool size)")
    ap.add_argument("--seed", type=int, default=0,
                    help="torch.manual_seed")
    ap.add_argument("--smoke", action="store_true",
                    help="quarantine ALL outputs under .smoke/run1/")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    t0 = time.time()
    written = run_model(args.model, args)
    print(f"\n[patching] DONE: {len(written)} JSON artifacts "
          f"({time.time() - t0:.1f}s) "
          f"{'[SMOKE -> ' + SMOKE_ROOT + ']' if args.smoke else '[production]'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
