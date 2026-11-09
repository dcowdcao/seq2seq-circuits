"""
patching_dla.py
=================================================================================
Activation-based mechanistic-interpretability tools for the `segrev3` model
(4-layer, 8-head, attention-only, Pre-LayerNorm with a PARAMETER-FREE
LayerNormPre; d_model=128, d_head=32, d_vocab=10, n_ctx=20), loaded via
TransformerLens. This module REPLACES the old ``basic_interp_circuit_tools.py``.

FOUR tools (activation screens; docs/segrev3_case_study.md §4a), each a pure-computation function (NO visualization
here -- plotting lives in a separate script):

    1. activation_patching          - exact causal patching (noising / denoising)
    2. attribution_patching         - 1st-order gradient approx (+ optional EAP-IG)
    3. path_patching / *_sweep /    - sender -> receiver.{q,k,v} direct-path effect,
       path_patching_qkv_sweep        incl. the full 32x32 Q/K/V sweep + head->logits
    4. direct_logit_attribution     - exact additive logit-diff decomposition (DLA)

(The old ``attention_attribution`` tool was removed -- it was never consumed by
any driver or visualization.)

CORRUPTION-TYPE PARAMETERIZATION
--------------------------------------------------------------------------------
Every tool is parameterized by a ``Segrev3Pairs`` (from ``data_gen/data_gen.py``),
which bundles the clean/corrupted tokens, the answer positions (9..18), the
correct/incorrect answer tokens AND the corruption ``scheme`` that produced them.
Build one with ``make_clean_corrupted(batch, scheme=...)`` and reuse it across
tools so they all study the SAME contrast. The three schemes (see data_gen) each
probe a different part of the circuit:

    * "random"             -- corrupted = an independent random batch (global
                              contrast; the ACDC / EAP / validation default).
    * "replace_threes_to_nonthrees"     -- n boundary 3s -> random non-3 (targets the
                              3-detector / boundary heads).
    * "replace_non_threes_to_threes" -- n non-3s -> 3 (targets the reversal / copy / mover
                              heads).

The default metric for every tool is ``metrics.logit_diff(logits, pairs)`` -- the
mean (correct - incorrect) logit over the pair's answer positions. Pass a custom
``metric=`` callable (logits -> scalar) to override.

LAYERNORM NOTES (segrev3 uses LayerNormPre before every block and the unembed)
--------------------------------------------------------------------------------
  * activation_patching / path_patching RUN the real forward pass -> EXACT.
  * direct_logit_attribution is additive only with the standard FROZEN final-LN
    scale (``apply_ln=True`` in the resid decomposition); it then sums exactly to
    the logit difference being decomposed.
  * The weight-based QK/OV circuits (edge_tools.py) are the usual
    Elhage APPROXIMATIONS modulo the per-position LN scale -- confirm them
    causally with path_patching here.
See ``interp_tools/model_loading.py`` for the full architecture ground truth.

References:
  * Activation patching: Vig 2020; Meng (ROME) 2022; Geiger 2021; Heimersheim &
    Nanda 2024; Zhang & Nanda 2024.
  * Attribution patching: Nanda 2023; Syed (EAP) 2023; Hanna (EAP-IG) 2024;
    Kramar (AtP*) 2024.
  * Path patching: Wang (IOI) 2022; Goldowsky-Dill et al. 2023.
  * DLA: Elhage et al. 2021 ("A Mathematical Framework for Transformer Circuits").
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import torch
from torch import Tensor
from tqdm import tqdm

from transformer_lens import HookedTransformer

# --------------------------------------------------------------------------- #
# Imports from the shared pipeline (model loader, data, metrics).             #
# patching_dla lives in interp_tools/, so add it + data_gen/ + metrics/ to the #
# path BEFORE importing them (they resolve model_loading transitively).        #
# --------------------------------------------------------------------------- #
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS_DIR)
for _p in (_THIS_DIR, os.path.join(_ROOT, "data_gen"), os.path.join(_ROOT, "metrics")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from model_loading import (  # noqa: E402
    load_segrev3, DEFAULT_CKPT, ANSWER_POSITIONS,
)
from data_gen import (  # noqa: E402
    Segrev3Pairs, make_clean_corrupted, single_position_contrast,
)
from metrics import logit_diff  # noqa: E402

MetricFn = Callable[[Tensor], Tensor]

# Cap on replicated sequences per batched forward (groups x examples). The segrev3
# model is tiny, so this is generous; it only
# bounds peak memory when a sweep replicates the batch over hundreds of sites.
_MAX_GROUP_ROWS = 1 << 16


def make_logit_diff_metric(pairs: Segrev3Pairs) -> MetricFn:
    """The default metric for a pair set: logits -> mean (correct - incorrect)
    logit over the pair's answer positions (``metrics.logit_diff``)."""
    return lambda logits: logit_diff(logits, pairs)


def _resolve_metric(pairs: Segrev3Pairs, metric: Optional[MetricFn]) -> MetricFn:
    return metric if metric is not None else make_logit_diff_metric(pairs)


# --------------------------------------------------------------------------- #
# Result containers (JSON-serializable .summary(); no plotting)               #
# --------------------------------------------------------------------------- #


@dataclass
class PatchingResult:
    method: str
    component: str
    direction: str                 # "noising" | "denoising"
    scheme: str                    # which corruption scheme produced the pairs
    effect: Tensor                 # normalized effect grid
    dim_labels: Sequence[str]      # names of the axes of `effect`
    coords: dict                   # axis name -> list of coordinate values
    clean_metric: float
    corrupted_metric: float
    extra: dict = field(default_factory=dict)

    def summary(self, top_k: int = 12) -> dict:
        flat = self.effect.abs().flatten()
        k = min(top_k, flat.numel())
        vals, idx = torch.topk(flat, k)
        shape = self.effect.shape
        top = []
        for rank in range(k):
            multi = _unravel(int(idx[rank]), shape)
            site = {self.dim_labels[a]: self.coords[self.dim_labels[a]][multi[a]]
                    for a in range(len(shape))}
            site["effect"] = float(self.effect[multi])
            top.append(site)
        return {
            "method": self.method,
            "component": self.component,
            "direction": self.direction,
            "scheme": self.scheme,
            "clean_metric": self.clean_metric,
            "corrupted_metric": self.corrupted_metric,
            "effect_shape": list(shape),
            "dim_labels": list(self.dim_labels),
            "top_sites": top,
            **self.extra,
        }


@dataclass
class DLAResult:
    logit_diff: float                      # the metric being decomposed
    head_attribution: Tensor               # [n_layers, n_heads]
    component_attribution: dict            # {label: float} incl. embeds/biases
    per_position: Optional[Tensor]         # [n_layers, n_heads, seq] or None
    head_labels: Sequence[str]
    scheme: str
    mode: str                              # "single" | "multi" | "correct_vs_rest"

    def summary(self, top_k: int = 12) -> dict:
        hm = self.head_attribution
        k = min(top_k, hm.numel())
        vals, idx = torch.topk(hm.abs().flatten(), k)
        n_heads = hm.shape[1]
        top_heads = []
        for rank in range(k):
            l, h = int(idx[rank]) // n_heads, int(idx[rank]) % n_heads
            top_heads.append({"layer": l, "head": h, "attribution": float(hm[l, h])})
        return {
            "logit_diff": self.logit_diff,
            "scheme": self.scheme,
            "mode": self.mode,
            "top_heads": top_heads,
            "component_attribution": {k_: float(v) for k_, v in self.component_attribution.items()},
        }


@dataclass
class PathQKVSweepResult:
    """The full sender->receiver path-effect structure: a 32x32 matrix per Q/K/V
    channel (sender head x receiver head) plus a length-32 head->logits vector."""
    matrices: dict                 # {"q": [n,n], "k": [n,n], "v": [n,n]} (sender x receiver)
    to_logits: Tensor              # [n] head -> logits direct effect
    n_layers: int
    n_heads: int
    direction: str
    scheme: str
    clean_metric: float
    corrupted_metric: float

    def _label(self, i: int) -> str:
        return f"L{i // self.n_heads}H{i % self.n_heads}"

    def summary(self, top_k: int = 12) -> dict:
        out = {"method": "path_patching_qkv_sweep", "direction": self.direction,
               "scheme": self.scheme, "clean_metric": self.clean_metric,
               "corrupted_metric": self.corrupted_metric, "top_edges": {}}
        for ch, m in self.matrices.items():
            flat = m.abs().flatten()
            k = min(top_k, flat.numel())
            _, idx = torch.topk(flat, k)
            ncol = m.shape[1]
            out["top_edges"][ch] = [
                {"sender": self._label(int(i) // ncol),
                 "receiver": self._label(int(i) % ncol),
                 "effect": float(m.flatten()[int(i)])}
                for i in idx if float(m.flatten()[int(i)]) != 0.0
            ]
        tl = self.to_logits
        k = min(top_k, tl.numel())
        _, idx = torch.topk(tl.abs().flatten(), k)
        out["top_to_logits"] = [
            {"head": self._label(int(i)), "effect": float(tl[int(i)])} for i in idx
        ]
        return out


def _unravel(flat_idx: int, shape) -> tuple:
    out = []
    for s in reversed(shape):
        out.append(flat_idx % s)
        flat_idx //= s
    return tuple(reversed(out))


# --------------------------------------------------------------------------- #
# Internal hook utilities                                                     #
# --------------------------------------------------------------------------- #

_COMPONENT_HOOK = {
    "resid_pre": lambda l: f"blocks.{l}.hook_resid_pre",
    "resid_post": lambda l: f"blocks.{l}.hook_resid_post",
    "attn_out": lambda l: f"blocks.{l}.hook_attn_out",
    "z": lambda l: f"blocks.{l}.attn.hook_z",   # per-head; sliced on head axis
}


def _patch_site_hook(src_value: Tensor, pos: Optional[int], head: Optional[int]):
    """Return a fwd hook that overwrites a site with `src_value` (broadcast)."""
    def hook(act: Tensor, hook) -> Tensor:
        if head is None and pos is None:
            act[...] = src_value
        elif head is None:
            act[:, pos] = src_value[:, pos]
        elif pos is None:
            act[:, :, head] = src_value[:, :, head]
        else:
            act[:, pos, head] = src_value[:, pos, head]
        return act
    return hook


# --------------------------------------------------------------------------- #
# 1. Activation patching                                                      #
# --------------------------------------------------------------------------- #


def activation_patching(
    model: HookedTransformer,
    pairs: Segrev3Pairs,
    component: str = "resid_pre",
    direction: str = "denoising",
    positions: Optional[Sequence[int]] = None,
    metric: Optional[MetricFn] = None,
) -> PatchingResult:
    """Exact causal (activation) patching over a (layer x [head] x position) grid,
    for the corruption carried by `pairs`.

    For each site we splice an activation from the *source* run into the
    *destination* run and measure the normalized change in the metric.

      * direction="denoising": destination = corrupted run, source = clean cache.
        Tests SUFFICIENCY ("is restoring this component enough?"). E=1 means the
        clean value fully restores clean behavior (ROME-style causal tracing).
      * direction="noising": destination = clean run, source = corrupted cache.
        Tests NECESSITY ("does corrupting this destroy clean behavior?"). E=1 means
        the corrupted value fully destroys clean behavior (resample ablation).

    `component` in {"resid_pre","resid_post","attn_out","z"}; "z" is per-head.
    The two directions are NOT mirror images (denoising under-detects distributed
    AND-circuits; noising under-detects redundant OR/backup-head circuits) -- run
    both. Returns a `PatchingResult` whose `.effect` is the normalized grid.
    """
    if direction not in ("noising", "denoising"):
        raise ValueError("direction must be 'noising' or 'denoising'")
    if component not in _COMPONENT_HOOK:
        raise ValueError(f"component must be one of {list(_COMPONENT_HOOK)}")

    device = next(model.parameters()).device.type
    clean_tokens = pairs.clean_tokens.to(device)
    corrupted_tokens = pairs.corrupted_tokens.to(device)
    seq = clean_tokens.shape[1]
    met = _resolve_metric(pairs, metric)

    hook_name_fn = _COMPONENT_HOOK[component]
    per_head = component == "z"
    names = [hook_name_fn(l) for l in range(model.cfg.n_layers)]

    with torch.no_grad():
        clean_logits, clean_cache = model.run_with_cache(clean_tokens, names_filter=names)
        corr_logits, corr_cache = model.run_with_cache(corrupted_tokens, names_filter=names)

    m_clean = float(met(clean_logits))
    m_corr = float(met(corr_logits))

    if direction == "denoising":
        dest_tokens, src_cache = corrupted_tokens, clean_cache
        baseline, denom = m_corr, (m_clean - m_corr)
    else:
        dest_tokens, src_cache = clean_tokens, corr_cache
        baseline, denom = m_clean, (m_corr - m_clean)
    denom = denom if abs(denom) > 1e-8 else 1e-8

    pos_list = list(range(seq)) if positions is None else list(positions)
    n_layers, n_heads = model.cfg.n_layers, model.cfg.n_heads
    if per_head:
        effect = torch.zeros(n_layers, n_heads, len(pos_list))
        dim_labels = ("layer", "head", "position")
        coords = {"layer": list(range(n_layers)), "head": list(range(n_heads)),
                  "position": pos_list}
    else:
        effect = torch.zeros(n_layers, len(pos_list))
        dim_labels = ("layer", "position")
        coords = {"layer": list(range(n_layers)), "position": pos_list}

    # Every (layer, [head], position) site is an INDEPENDENT single-position patch,
    # so instead of one forward per site we replicate the batch once per site
    # (group-major) and patch each group's site in a single masked `where` per
    # layer -- collapsing O(layers*heads*positions) forwards into one chunked pass.
    B = dest_tokens.shape[0]
    sites = [(l, h, p, pi)
             for l in range(n_layers)
             for h in (range(n_heads) if per_head else [None])
             for pi, p in enumerate(pos_list)]
    G = len(sites)
    per_chunk = max(1, _MAX_GROUP_ROWS // max(1, B))
    _pbar = tqdm(total=G, desc=f"activation_patching[{component}]", leave=False)
    for c0 in range(0, G, per_chunk):
        chunk = sites[c0:c0 + per_chunk]
        gc = len(chunk)
        if per_head:
            masks = {l: torch.zeros(gc, seq, n_heads, dtype=torch.bool, device=device)
                     for l in range(n_layers)}
            for gi, (l, h, p, pi) in enumerate(chunk):
                masks[l][gi, p, h] = True
        else:
            masks = {l: torch.zeros(gc, seq, dtype=torch.bool, device=device)
                     for l in range(n_layers)}
            for gi, (l, h, p, pi) in enumerate(chunk):
                masks[l][gi, p] = True

        toks_rep = dest_tokens.repeat(gc, 1)
        hooks = []
        for l in range(n_layers):
            name = hook_name_fn(l)
            src = src_cache[name]
            m_l = masks[l]
            if per_head:
                def hook(act, hook, src=src, m_l=m_l, gc=gc):
                    dH = act.shape[-1]
                    av = act.view(gc, B, seq, n_heads, dH)
                    av = torch.where(m_l.view(gc, 1, seq, n_heads, 1),
                                     src.view(1, B, seq, n_heads, dH), av)
                    return av.reshape(gc * B, seq, n_heads, dH)
            else:
                def hook(act, hook, src=src, m_l=m_l, gc=gc):
                    dM = act.shape[-1]
                    av = act.view(gc, B, seq, dM)
                    av = torch.where(m_l.view(gc, 1, seq, 1),
                                     src.view(1, B, seq, dM), av)
                    return av.reshape(gc * B, seq, dM)
            hooks.append((name, hook))

        with torch.no_grad():
            patched = model.run_with_hooks(toks_rep, fwd_hooks=hooks)
            patched = patched.view(gc, B, seq, -1)
            # Stack the per-group metric and sync ONCE per chunk (avoids gc
            # CPU<->GPU syncs from per-site float()).
            vals = torch.stack([met(patched[gi]) for gi in range(gc)])
        es = ((vals - baseline) / denom).tolist()
        for gi, (l, h, p, pi) in enumerate(chunk):
            if per_head:
                effect[l, h, pi] = es[gi]
            else:
                effect[l, pi] = es[gi]
        _pbar.update(gc)
    _pbar.close()

    return PatchingResult(
        method="activation_patching", component=component, direction=direction,
        scheme=pairs.scheme, effect=effect, dim_labels=dim_labels, coords=coords,
        clean_metric=m_clean, corrupted_metric=m_corr,
    )


# --------------------------------------------------------------------------- #
# 2. Attribution patching (+ optional integrated gradients = EAP-IG)          #
# --------------------------------------------------------------------------- #


def attribution_patching(
    model: HookedTransformer,
    pairs: Segrev3Pairs,
    component: str = "z",
    direction: str = "denoising",
    n_ig_steps: int = 1,
    metric: Optional[MetricFn] = None,
) -> PatchingResult:
    """First-order gradient approximation of activation patching (all sites in ONE
    forward + backward pass per IG step), for the corruption carried by `pairs`.

    Defined over the ADDITIVE residual writers -- `component` in {"z" (per head),
    "attn_out" (per layer)} -- for which the decomposition is well-posed (each is
    an independent leaf added to the residual stream). The cumulative residual
    points resid_pre/resid_post are NOT supported (use activation_patching).

    Denoising estimate (Nanda 2023): destination = corrupted run,
        effect(site) ~= ( a_clean - a_corrupt ) . d(metric)/d(a) |_corrupt
    Noising swaps the roles. n_ig_steps > 1 switches to EAP-IG (Hanna et al.
    2024): the gradient is averaged over m points interpolated along the
    corrupt->clean path, curing softmax/saturation zero-gradient false negatives.

    Returns the same `PatchingResult` shape as activation_patching, normalized by
    the same clean/corrupt gap (so sign/scale are comparable). With EAP-IG the
    site attributions additively decompose the metric delta of jointly patching
    all sites (completeness, exact in the step limit).
    """
    if direction not in ("noising", "denoising"):
        raise ValueError("direction must be 'noising' or 'denoising'")
    if component not in ("z", "attn_out"):
        raise ValueError(
            "attribution_patching is defined over the additive writers "
            "'z' or 'attn_out'; for 'resid_pre'/'resid_post' use activation_patching."
        )

    device = next(model.parameters()).device.type
    clean_tokens = pairs.clean_tokens.to(device)
    corrupted_tokens = pairs.corrupted_tokens.to(device)
    batch, seq = clean_tokens.shape
    met = _resolve_metric(pairs, metric)

    per_head = component == "z"
    names = [_COMPONENT_HOOK[component](l) for l in range(model.cfg.n_layers)]

    with torch.no_grad():
        clean_logits, clean_cache = model.run_with_cache(clean_tokens, names_filter=names)
        corr_logits, corr_cache = model.run_with_cache(corrupted_tokens, names_filter=names)
    m_clean = float(met(clean_logits))
    m_corr = float(met(corr_logits))

    if direction == "denoising":
        dest_tokens = corrupted_tokens
        src_cache, dest_cache = clean_cache, corr_cache       # patch vector = clean - corrupt
    else:
        dest_tokens = clean_tokens
        src_cache, dest_cache = corr_cache, clean_cache        # patch vector = corrupt - clean

    grad_accum = {n: torch.zeros_like(dest_cache[n]) for n in names}
    steps = max(1, n_ig_steps)
    alphas = [0.0] if steps == 1 else [k / steps for k in range(1, steps + 1)]

    for alpha in tqdm(alphas, desc="attribution_patching (IG steps)", leave=False):
        stored = {}

        def make_save(name):
            def save_hook(act: Tensor, hook) -> Tensor:
                leaf = dest_cache[name] + alpha * (src_cache[name] - dest_cache[name])
                leaf = leaf.detach().requires_grad_(True)
                stored[name] = leaf
                return leaf
            return save_hook

        model.reset_hooks()
        logits = model.run_with_hooks(
            dest_tokens, fwd_hooks=[(n, make_save(n)) for n in names])
        m = met(logits)
        model.zero_grad(set_to_none=True)
        m.backward()
        for n in names:
            grad_accum[n] += stored[n].grad.detach()
    model.reset_hooks()
    for n in names:
        grad_accum[n] /= len(alphas)

    pos_list = list(range(seq))
    n_layers, n_heads = model.cfg.n_layers, model.cfg.n_heads
    if per_head:
        effect = torch.zeros(n_layers, n_heads, seq)
        dim_labels = ("layer", "head", "position")
        coords = {"layer": list(range(n_layers)), "head": list(range(n_heads)),
                  "position": pos_list}
    else:
        effect = torch.zeros(n_layers, seq)
        dim_labels = ("layer", "position")
        coords = {"layer": list(range(n_layers)), "position": pos_list}

    denom = (m_clean - m_corr) if direction == "denoising" else (m_corr - m_clean)
    denom = denom if abs(denom) > 1e-8 else 1e-8

    for l in range(n_layers):
        name = _COMPONENT_HOOK[component](l)
        patch_vec = (src_cache[name] - dest_cache[name])       # patch vector
        attr = (patch_vec * grad_accum[name]).sum(dim=-1)      # reduce d_head/d_model
        attr = attr.sum(dim=0) / denom                         # sum over batch, normalize
        if per_head:
            effect[l] = attr.transpose(0, 1)                   # [head, pos]
        else:
            effect[l] = attr                                   # [pos]

    return PatchingResult(
        method="attribution_patching", component=component, direction=direction,
        scheme=pairs.scheme, effect=effect, dim_labels=dim_labels, coords=coords,
        clean_metric=m_clean, corrupted_metric=m_corr,
        extra={"n_ig_steps": steps, "approximation": True},
    )


# --------------------------------------------------------------------------- #
# 3. Path patching (single edge, sweep into one receiver, full Q/K/V sweep)    #
# --------------------------------------------------------------------------- #


def path_patching(
    model: HookedTransformer,
    pairs: Segrev3Pairs,
    sender: tuple,                       # (layer, head)
    receiver: tuple,                     # (layer, head) or "logits"
    receiver_input: str = "v",           # "q" | "k" | "v"  (ignored for logits)
    direction: str = "noising",
    metric: Optional[MetricFn] = None,
) -> PatchingResult:
    """Measure the DIRECT-PATH effect of a sender head on a receiver head's Q/K/V
    input (or on the logits), holding all other paths fixed, for the corruption
    carried by `pairs`.

    Method (Goldowsky-Dill et al. 2023; Wang et al. 2022), default
    direction="noising":
      Pass 1 (on clean tokens): set the sender head's z to its CORRUPTED value,
        FREEZE every other head's z to its clean value (the sender's perturbation
        reaches the receiver only via the direct residual path), and record the
        receiver head's hook_{q|k|v}.
      Pass 2 (on clean tokens): patch ONLY the receiver head's hook_{q|k|v} to the
        recorded value; measure the metric.
    For receiver="logits" we patch the sender (freezing other heads) and read the
    logits -- the sender's direct effect on the output.

    direction="denoising" swaps clean<->corrupted (sender set to clean on a
    corrupted, frozen run; tests sufficiency of the path).
    """
    if receiver != "logits" and receiver_input not in ("q", "k", "v"):
        raise ValueError("receiver_input must be 'q', 'k', or 'v'")
    if direction not in ("noising", "denoising"):
        raise ValueError("direction must be 'noising' or 'denoising'")

    device = next(model.parameters()).device.type
    clean_tokens = pairs.clean_tokens.to(device)
    corrupted_tokens = pairs.corrupted_tokens.to(device)
    met = _resolve_metric(pairs, metric)

    n_layers, n_heads = model.cfg.n_layers, model.cfg.n_heads
    z_names = [f"blocks.{l}.attn.hook_z" for l in range(n_layers)]

    with torch.no_grad():
        clean_logits, clean_cache = model.run_with_cache(clean_tokens)
        corr_logits, corr_cache = model.run_with_cache(corrupted_tokens)
    m_clean = float(met(clean_logits))
    m_corr = float(met(corr_logits))

    if direction == "noising":
        base_cache, sender_src_cache = clean_cache, corr_cache
        base_tokens, base_metric, denom = clean_tokens, m_clean, (m_corr - m_clean)
    else:
        base_cache, sender_src_cache = corr_cache, clean_cache
        base_tokens, base_metric, denom = corrupted_tokens, m_corr, (m_clean - m_corr)
    denom = denom if abs(denom) > 1e-8 else 1e-8

    sl, sh = sender

    def freeze_z(name, l):
        def hook(act: Tensor, hook) -> Tensor:
            act[...] = base_cache[name]              # freeze every head to base
            if l == sl:
                act[:, :, sh] = sender_src_cache[name][:, :, sh]  # except the sender
            return act
        return hook

    fwd_hooks = [(z_names[l], freeze_z(z_names[l], l)) for l in range(n_layers)]

    if receiver == "logits":
        with torch.no_grad():
            patched_logits = model.run_with_hooks(base_tokens, fwd_hooks=fwd_hooks)
        m_pt = float(met(patched_logits))
        effect = torch.tensor((m_pt - base_metric) / denom).reshape(1)
        return PatchingResult(
            method="path_patching", component=f"sender L{sl}H{sh} -> logits",
            direction=direction, scheme=pairs.scheme, effect=effect,
            dim_labels=("edge",), coords={"edge": [f"L{sl}H{sh}->logits"]},
            clean_metric=m_clean, corrupted_metric=m_corr,
            extra={"path_effect": float(effect[0])},
        )

    rl, rh = receiver
    recv_hook_name = f"blocks.{rl}.attn.hook_{receiver_input}"
    captured = {}

    def capture(act: Tensor, hook) -> Tensor:
        captured["recv"] = act[:, :, rh].detach().clone()
        return act

    with torch.no_grad():
        model.run_with_hooks(base_tokens, fwd_hooks=fwd_hooks + [(recv_hook_name, capture)])

    def patch_recv(act: Tensor, hook) -> Tensor:
        act[:, :, rh] = captured["recv"]
        return act

    with torch.no_grad():
        patched_logits = model.run_with_hooks(base_tokens, fwd_hooks=[(recv_hook_name, patch_recv)])
    m_pt = float(met(patched_logits))
    effect = torch.tensor((m_pt - base_metric) / denom).reshape(1)

    return PatchingResult(
        method="path_patching",
        component=f"sender L{sl}H{sh} -> receiver L{rl}H{rh}.{receiver_input}",
        direction=direction, scheme=pairs.scheme, effect=effect, dim_labels=("edge",),
        coords={"edge": [f"L{sl}H{sh}->L{rl}H{rh}.{receiver_input}"]},
        clean_metric=m_clean, corrupted_metric=m_corr,
        extra={"path_effect": float(effect[0])},
    )


def _sweep_grid_batched(model, pairs, receiver, receiver_input, direction, met,
                        clean_cache, corr_cache, m_clean, m_corr, sender_layers=None):
    """Batched core of ``path_patching_sweep``: for a fixed receiver.{q|k|v}, compute
    the direct-path effect of EVERY sender head in ONE freeze+capture pass and ONE
    patch pass (was 2 forwards PER sender). Returns the [n_layers, n_heads] grid.

    The clean/corrupted caches and their metrics are passed in so a full Q/K/V sweep
    computes them once instead of re-running the model for every (channel, receiver,
    sender) triple."""
    n_layers, n_heads = model.cfg.n_layers, model.cfg.n_heads
    rl, rh = receiver
    device = next(model.parameters()).device.type
    base_tok = pairs.clean_tokens.to(device) if direction == "noising" else pairs.corrupted_tokens.to(device)
    if direction == "noising":
        base_cache, sender_src = clean_cache, corr_cache
        base_metric, denom = m_clean, (m_corr - m_clean)
    else:
        base_cache, sender_src = corr_cache, clean_cache
        base_metric, denom = m_corr, (m_clean - m_corr)
    denom = denom if abs(denom) > 1e-8 else 1e-8

    layers = range(rl) if sender_layers is None else [l for l in sender_layers if l < rl]
    senders = [(sl, sh) for sl in layers for sh in range(n_heads)]
    grid = torch.zeros(n_layers, n_heads)
    nS = len(senders)
    if nS == 0:
        return grid

    B, seq = base_tok.shape
    z_names = [f"blocks.{l}.attn.hook_z" for l in range(n_layers)]
    recv_name = f"blocks.{rl}.attn.hook_{receiver_input}"
    # per-layer sender mask [nS, n_heads]
    smask = {l: torch.zeros(nS, n_heads, dtype=torch.bool, device=base_tok.device)
             for l in range(n_layers)}
    for g, (sl, sh) in enumerate(senders):
        smask[sl][g, sh] = True
    toks_rep = base_tok.repeat(nS, 1)

    # ---- pass 1: freeze every head to base (sender head -> source), capture recv ----
    captured = {}

    def make_freeze(base_v, src_v, m_l):
        def fh(act, hook):
            dH = act.shape[-1]
            av = torch.where(m_l.view(nS, 1, 1, n_heads, 1),
                             src_v.view(1, B, seq, n_heads, dH),
                             base_v.view(1, B, seq, n_heads, dH))
            return av.reshape(nS * B, seq, n_heads, dH)
        return fh

    freeze_hooks = [(z_names[l], make_freeze(base_cache[z_names[l]],
                                             sender_src[z_names[l]], smask[l]))
                    for l in range(n_layers)]

    def cap(act, hook):
        captured["v"] = act.view(nS, B, seq, n_heads, act.shape[-1])[:, :, :, rh].clone()
        return act

    with torch.no_grad():
        model.run_with_hooks(toks_rep, fwd_hooks=freeze_hooks + [(recv_name, cap)])

    # ---- pass 2: patch ONLY the receiver channel to the captured value ----
    cap_v = captured["v"]

    def patch(act, hook):
        dH = act.shape[-1]
        av = act.view(nS, B, seq, n_heads, dH)
        av[:, :, :, rh] = cap_v
        return av.reshape(nS * B, seq, n_heads, dH)

    with torch.no_grad():
        logits = model.run_with_hooks(toks_rep, fwd_hooks=[(recv_name, patch)])
        logits = logits.view(nS, B, seq, -1)
        vals = torch.stack([met(logits[g]) for g in range(nS)])
    effs = ((vals - base_metric) / denom).tolist()
    for g, (sl, sh) in enumerate(senders):
        grid[sl, sh] = effs[g]
    return grid


def _to_logits_batched(model, pairs, direction, met,
                       clean_cache, corr_cache, m_clean, m_corr):
    """Batched ``path_patching(sender -> logits)`` over EVERY sender head: one freeze
    pass with each group's sender perturbed, read the logits per group. Returns the
    length-(n_layers*n_heads) head->logits direct-effect vector."""
    n_layers, n_heads = model.cfg.n_layers, model.cfg.n_heads
    device = next(model.parameters()).device.type
    base_tok = pairs.clean_tokens.to(device) if direction == "noising" else pairs.corrupted_tokens.to(device)
    if direction == "noising":
        base_cache, sender_src = clean_cache, corr_cache
        base_metric, denom = m_clean, (m_corr - m_clean)
    else:
        base_cache, sender_src = corr_cache, clean_cache
        base_metric, denom = m_corr, (m_clean - m_corr)
    denom = denom if abs(denom) > 1e-8 else 1e-8

    senders = [(sl, sh) for sl in range(n_layers) for sh in range(n_heads)]
    nS = len(senders)
    B, seq = base_tok.shape
    z_names = [f"blocks.{l}.attn.hook_z" for l in range(n_layers)]
    smask = {l: torch.zeros(nS, n_heads, dtype=torch.bool, device=base_tok.device)
             for l in range(n_layers)}
    for g, (sl, sh) in enumerate(senders):
        smask[sl][g, sh] = True
    toks_rep = base_tok.repeat(nS, 1)

    def make_freeze(base_v, src_v, m_l):
        def fh(act, hook):
            dH = act.shape[-1]
            av = torch.where(m_l.view(nS, 1, 1, n_heads, 1),
                             src_v.view(1, B, seq, n_heads, dH),
                             base_v.view(1, B, seq, n_heads, dH))
            return av.reshape(nS * B, seq, n_heads, dH)
        return fh

    freeze_hooks = [(z_names[l], make_freeze(base_cache[z_names[l]],
                                             sender_src[z_names[l]], smask[l]))
                    for l in range(n_layers)]
    with torch.no_grad():
        logits = model.run_with_hooks(toks_rep, fwd_hooks=freeze_hooks)
        logits = logits.view(nS, B, seq, -1)
        vals = torch.stack([met(logits[g]) for g in range(nS)])
    effs = ((vals - base_metric) / denom).tolist()
    to_logits = torch.zeros(nS)
    for g in range(nS):
        to_logits[g] = effs[g]
    return to_logits


def path_patching_sweep(
    model: HookedTransformer,
    pairs: Segrev3Pairs,
    receiver: tuple,
    receiver_input: str = "v",
    direction: str = "noising",
    sender_layers: Optional[Sequence[int]] = None,
    metric: Optional[MetricFn] = None,
) -> PatchingResult:
    """Sweep ALL sender heads (in layers strictly before the receiver) into a fixed
    receiver.{q|k|v}. Returns an [n_layers, n_heads] path-effect grid (zeros where
    layer >= receiver layer)."""
    n_layers, n_heads = model.cfg.n_layers, model.cfg.n_heads
    met = _resolve_metric(pairs, metric)
    device = next(model.parameters()).device.type
    with torch.no_grad():
        clean_logits, clean_cache = model.run_with_cache(pairs.clean_tokens.to(device))
        corr_logits, corr_cache = model.run_with_cache(pairs.corrupted_tokens.to(device))
    m_clean, m_corr = float(met(clean_logits)), float(met(corr_logits))
    grid = _sweep_grid_batched(model, pairs, receiver, receiver_input, direction, met,
                               clean_cache, corr_cache, m_clean, m_corr,
                               sender_layers=sender_layers)
    return PatchingResult(
        method="path_patching_sweep",
        component=f"-> receiver L{receiver[0]}H{receiver[1]}.{receiver_input}",
        direction=direction, scheme=pairs.scheme, effect=grid, dim_labels=("layer", "head"),
        coords={"layer": list(range(n_layers)), "head": list(range(n_heads))},
        clean_metric=m_clean, corrupted_metric=m_corr,
    )


def path_patching_qkv_sweep(
    model: HookedTransformer,
    pairs: Segrev3Pairs,
    direction: str = "noising",
    metric: Optional[MetricFn] = None,
    channels: Sequence[str] = ("q", "k", "v"),
) -> PathQKVSweepResult:
    """The full path-patching structure: a 32x32 (sender x receiver) path-effect
    matrix for each Q/K/V channel, plus a length-32 head->logits vector. This is
    the edge-level wiring the manual pipeline compares against ACDC/EAP.

    NB: this runs path_patching for every (channel, receiver, sender) triple and
    every (sender -> logits), so it is the slow tool -- O(channels x heads^2)
    interventional passes. Use a modest batch (e.g. 64-128)."""
    met = _resolve_metric(pairs, metric)
    n_layers, n_heads = model.cfg.n_layers, model.cfg.n_heads
    n = n_layers * n_heads
    mats = {ch: torch.zeros(n, n) for ch in channels}
    device = next(model.parameters()).device.type

    # Compute the clean/corrupted caches + metrics ONCE and reuse them for every
    # (channel, receiver, sender) triple and for the head->logits pass -- the old
    # code re-ran the model twice inside every one of ~O(channels*heads^2) calls.
    with torch.no_grad():
        clean_logits, clean_cache = model.run_with_cache(pairs.clean_tokens.to(device))
        corr_logits, corr_cache = model.run_with_cache(pairs.corrupted_tokens.to(device))
    m_clean, m_corr = float(met(clean_logits)), float(met(corr_logits))

    # one bar over all (channel x receiver) edges + the (sender -> logits) pass
    _total = len(channels) * (n_layers - 1) * n_heads + 1
    _pbar = tqdm(total=_total, desc="path_patching_qkv_sweep", leave=False)
    for ch in channels:
        for rl in range(1, n_layers):
            for rh in range(n_heads):
                grid = _sweep_grid_batched(model, pairs, receiver=(rl, rh),
                                           receiver_input=ch, direction=direction, met=met,
                                           clean_cache=clean_cache, corr_cache=corr_cache,
                                           m_clean=m_clean, m_corr=m_corr)
                ri = rl * n_heads + rh
                for sl in range(rl):
                    for sh in range(n_heads):
                        mats[ch][sl * n_heads + sh, ri] = grid[sl, sh]
                _pbar.update(1)

    # head -> logits: batch ALL senders into one freeze pass (was one forward each).
    to_logits = _to_logits_batched(model, pairs, direction, met,
                                   clean_cache, corr_cache, m_clean, m_corr)
    _pbar.update(1)
    _pbar.close()

    return PathQKVSweepResult(
        matrices=mats, to_logits=to_logits, n_layers=n_layers, n_heads=n_heads,
        direction=direction, scheme=pairs.scheme,
        clean_metric=m_clean, corrupted_metric=m_corr,
    )


# --------------------------------------------------------------------------- #
# 4. Direct Logit Attribution                                                 #
# --------------------------------------------------------------------------- #


def direct_logit_attribution(
    model: HookedTransformer,
    pairs: Segrev3Pairs,
    mode: str = "single",
    per_position: bool = False,
) -> DLAResult:
    """Additive decomposition of the logit difference into per-component direct
    contributions, exact under the standard frozen final-LN scale, for the
    corruption carried by `pairs`.

    segrev3 has a final LayerNormPre, so logits = ln_final(resid_final) @ W_U + b_U.
    apply_ln=True freezes that scale (uses the cached ln_final scale as a constant),
    making the map linear; the decomposition over the additive residual writers
    (token & positional embeddings, every head's output, attention biases) then
    sums EXACTLY to the logit difference. For component c at answer position p,
        DLA(c) = < r_c , W_U[:,correct] - W_U[:,incorrect] >.

    `mode` (the answer-position / contrast convention; see data_gen):
      * "single" -- per example use the ONE answer position where clean != corrupted
        output (single_position_contrast). Isolates the direct writers for this
        corruption. (default)
      * "multi"  -- decompose the logit-diff averaged over ALL answer positions
        9..18 (the full metrics.logit_diff metric).
      * "correct_vs_rest" -- NO corrupted partner (DLA does no patching, so it
        runs on SINGLE sequences): at every answer position decompose the margin
        of the correct digit over the MEAN of the other 9 digits, direction
        W_U[:,correct] - mean_{d != correct} W_U[:,d]. Reads only the clean
        tokens + true outputs of `pairs`; the corrupted side is never touched.
    `per_position` (single mode only) also returns a [n_layers,n_heads,seq] grid.
    DLA captures only the DIRECT path; a head that matters via composition shows
    small DLA but large patching effect.
    """
    if mode not in ("single", "multi", "correct_vs_rest"):
        raise ValueError("mode must be 'single', 'multi' or 'correct_vs_rest'")
    if per_position and mode != "single":
        raise ValueError("per_position is only supported in mode='single'")

    device = next(model.parameters()).device.type
    tokens = pairs.clean_tokens.to(device)
    batch, seq = tokens.shape

    if mode == "single":
        pos, cor, inc = single_position_contrast(pairs)           # [B] each
        pos = pos.to(device)[:, None]; cor = cor.to(device)[:, None]; inc = inc.to(device)[:, None]
    elif mode == "correct_vs_rest":
        pos = pairs.answer_positions.to(device)                   # [B, P]
        cor = pairs.correct.to(device)
        inc = None                                                # no corrupted side
    else:
        pos = pairs.answer_positions.to(device)                   # [B, P]
        cor = pairs.correct.to(device); inc = pairs.incorrect.to(device)
    P = pos.shape[1]

    with torch.no_grad():
        logits, cache = model.run_with_cache(tokens)

    b_idx = torch.arange(batch, device=device)[:, None]           # [B,1]
    # reference logit difference being decomposed (mean over batch and positions)
    V = model.cfg.d_vocab
    sel_logits = logits[b_idx, pos]                               # [B,P,V]
    cor_logit = torch.gather(sel_logits, 2, cor.unsqueeze(-1)).squeeze(-1)   # [B,P]
    if inc is None:
        # margin of the correct digit over the mean of the other V-1 digits
        rest_logit = (sel_logits.sum(dim=2) - cor_logit) / (V - 1)           # [B,P]
        logit_diff_value = float((cor_logit - rest_logit).mean())
    else:
        inc_logit = torch.gather(sel_logits, 2, inc.unsqueeze(-1)).squeeze(-1)
        logit_diff_value = float((cor_logit - inc_logit).mean())

    # per-(example,position) unembed directions
    W_U_T = model.W_U.T                                           # [d_vocab, d_model]
    if inc is None:
        # W_U[:,correct] - mean_{d != correct} W_U[:,d]
        rest_dir = (W_U_T.sum(dim=0) - W_U_T[cor]) / (V - 1)     # [B,P,d_model]
        dirs = (W_U_T[cor] - rest_dir).detach()
    else:
        # W_U[:,correct] - W_U[:,incorrect]
        dirs = (W_U_T[cor] - W_U_T[inc]).detach()                # [B,P,d_model]

    # Additive decomposition with the frozen final-LN scale (apply_ln=True).
    decomp, labels = cache.get_full_resid_decomposition(
        layer=model.cfg.n_layers, return_labels=True,
        expand_neurons=False, apply_ln=True, pos_slice=None,
    )  # [C, B, S, d_model]
    decomp = decomp.detach()

    decomp_sel = decomp[:, b_idx, pos]                            # [C, B, P, d_model]
    contrib = torch.einsum("cbpd,bpd->cbp", decomp_sel, dirs).mean(dim=(1, 2))  # [C]

    n_layers, n_heads = model.cfg.n_layers, model.cfg.n_heads
    head_attr = torch.zeros(n_layers, n_heads)
    component_attr = {}
    for ci, lab in enumerate(labels):
        val = float(contrib[ci])
        if lab.startswith("L") and "H" in lab:
            try:
                l = int(lab[1:lab.index("H")])
                h = int(lab[lab.index("H") + 1:])
                head_attr[l, h] = val
                continue
            except ValueError:
                pass
        component_attr[lab] = val

    # the logit DIFFERENCE also contains the unembed-bias difference
    if inc is None:
        ub = (model.b_U[cor] - (model.b_U.sum() - model.b_U[cor]) / (V - 1)).float().mean()
    else:
        ub = (model.b_U[cor] - model.b_U[inc]).float().mean()
    component_attr["unembed_bias"] = float(ub.detach())

    per_pos = None
    if per_position:                                              # single mode only
        dir_single = dirs[:, 0]                                  # [B, d_model]
        contrib_all = torch.einsum("cbsd,bd->cbs", decomp, dir_single).mean(dim=1)  # [C, S]
        per_pos = torch.zeros(n_layers, n_heads, seq)
        for ci, lab in enumerate(labels):
            if lab.startswith("L") and "H" in lab:
                try:
                    l = int(lab[1:lab.index("H")])
                    h = int(lab[lab.index("H") + 1:])
                    per_pos[l, h] = contrib_all[ci]
                except ValueError:
                    pass

    return DLAResult(
        logit_diff=logit_diff_value, head_attribution=head_attr,
        component_attribution=component_attr, per_position=per_pos,
        head_labels=[f"L{l}H{h}" for l in range(n_layers) for h in range(n_heads)],
        scheme=pairs.scheme, mode=mode,
    )


# --------------------------------------------------------------------------- #
# Smoke test                                                                  #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    torch.manual_seed(0)
    model = load_segrev3()

    print("== building clean/corrupted pairs for all 3 schemes ==")
    schemes = ("random", "replace_threes_to_nonthrees", "replace_non_threes_to_threes")
    pairs_by_scheme = {s: make_clean_corrupted(16, scheme=s, seed=1, n=1) for s in schemes}
    for s, p in pairs_by_scheme.items():
        met = make_logit_diff_metric(p)
        with torch.no_grad():
            mc = float(met(model(p.clean_tokens))); mk = float(met(model(p.corrupted_tokens)))
        print(f"  [{s:18s}] clean logit-diff={mc:+.3f}  corrupted={mk:+.3f}")

    pr = pairs_by_scheme["replace_non_threes_to_threes"]

    print("\n== 1. activation_patching (denoising, resid_pre) ==")
    r = activation_patching(model, pr, component="resid_pre", direction="denoising")
    print("  scheme:", r.scheme, "| top:", r.summary(top_k=2)["top_sites"][0])
    print("== 1. activation_patching (noising, z per-head) ==")
    r = activation_patching(model, pr, component="z", direction="noising")
    print("  top:", r.summary(top_k=2)["top_sites"][0])

    print("\n== 2. attribution_patching (plain) vs EAP-IG(5) ==")
    ra = attribution_patching(model, pr, component="z")
    rb = attribution_patching(model, pr, component="z", n_ig_steps=5)
    print("  plain top:", ra.summary(top_k=1)["top_sites"][0])
    print("  eapig top:", rb.summary(top_k=1)["top_sites"][0])

    print("\n== 3. path_patching (L0H0 -> L2H0.v) + sweep + qkv-sweep ==")
    rp = path_patching(model, pr, sender=(0, 0), receiver=(2, 0), receiver_input="v")
    print("  single edge:", rp.extra["path_effect"])
    rs = path_patching_sweep(model, pr, receiver=(2, 0), receiver_input="v")
    print("  sweep top:", rs.summary(top_k=1)["top_sites"][0])
    # tiny qkv sweep (use a small batch to keep the smoke test fast)
    small = make_clean_corrupted(8, scheme="random", seed=3)
    rq = path_patching_qkv_sweep(model, small)
    print("  qkv sweep matrices:", {k: tuple(v.shape) for k, v in rq.matrices.items()},
          "| to_logits:", tuple(rq.to_logits.shape))
    print("  qkv top V-edge:", (rq.summary(top_k=1)["top_edges"]["v"] or ["(none)"])[0])

    print("\n== 4. direct_logit_attribution (exactness check, both modes) ==")
    for mode in ("single", "multi"):
        d = direct_logit_attribution(model, pr, mode=mode)
        total = float(d.head_attribution.sum() + sum(d.component_attribution.values()))
        ok = abs(total - d.logit_diff) < 1e-2
        print(f"  [{mode}] sum(DLA)={total:+.4f}  logit_diff={d.logit_diff:+.4f}  match={ok}")
        print(f"         top heads: {[(h['layer'], h['head']) for h in d.summary(top_k=3)['top_heads']]}")
    d = direct_logit_attribution(model, pr, mode="single", per_position=True)
    print("  per_position grid:", tuple(d.per_position.shape))

    print("\nALL SMOKE TESTS RAN.")
