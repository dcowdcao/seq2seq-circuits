"""
edge_tools.py
=================================================================================
Weight-based (input-independent) circuit-analysis tools for `segrev3`
(4-layer, 8-head, attention-only, Pre-LayerNorm/LayerNormPre; d_model=128, d_head=32,
d_vocab=10, n_ctx=20), loaded via TransformerLens.

Six tools (weight-based analysis; docs/segrev3_case_study.md §5.6). The first three are the original Elhage et al.
end-to-end circuits; the last three expose the remaining edges of the weight
graph so the full 2x2 below is first-class (not buried inside qk/ov_circuit).

    1. qk_circuit       - the QK circuit of one head among the input tokens:
                          which (query token) attends to which (key token),
                          read straight off the weights. (+ optional positional
                          QK over the n_ctx positions -- the "mirror" plot for a
                          reversal task.) Now ALSO returns its intermediate
                          input->head read maps (q_map = W_E W_Q, k_map = W_E W_K).
    2. ov_circuit       - the OV circuit of one head among the input tokens:
                          if the head attends to input token i, how it shifts the
                          logit of each output token j; plus the eigenvalue
                          "copying score". Now ALSO returns its two factors:
                          input->head[V] (v_map = W_E W_V) and head->logit
                          (logit_map = W_O W_U), with token_matrix = v_map @ logit_map.
    3. qkv_composition  - Q/K/V composition scores between every ordered pair of
                          heads in a circuit subgraph (sender layer < receiver
                          layer), with a random-weight baseline.
    4. direct_path      - the input->logit DIRECT PATH (bigram / skip circuit)
                          W_E W_U among the input tokens, carrying the same
                          eigenvalue copying score as ov_circuit (+ optional
                          positional W_pos W_U). The only edge that touches no head.
    5. input_to_head    - the input->head read maps W_E W_{Q/K/V} of one head
                          ([d_vocab, d_head] each): which input tokens project most
                          strongly into the head's Q/K/V subspace (+ optional
                          positional / bias). The embedding-as-sender analog of
                          qkv_composition.
    6. head_to_logit    - the head->logit write maps of one head: z_map = W_O W_U
                          ([d_head, d_vocab]) and resid_map = W_OV W_U
                          ([d_model, d_vocab]); which output tokens the head is
                          wired to push. The logits-as-receiver analog of
                          qkv_composition. (ov_circuit.token_matrix == W_E @ resid_map.)

A unifying core (residual_write_matrix / residual_read_matrix / edge_circuit)
implements EVERY cell uniformly as (sender write) @ (receiver read):

                  receiver: head Q/K/V                 receiver: logits
                  read = W_{Q/K/V}  [d_model, d_head]   read = W_U  [d_model, d_vocab]
  sender: input   input->head:  W_E W_{Q/K/V}          input->logit:  W_E W_U  (DIRECT PATH)
  write = W_E     [d_vocab, d_head]                     [d_vocab, d_vocab]
  sender: head A  head->head:   W_OV^A W_{Q/K/V}        head->logit:   W_OV^A W_U
  write = W_OV^A  [d_model, d_head]                     [d_model, d_vocab]
  (the positional embedding W_pos is also a valid sender: pos->head, pos->logit)

What these compute: segrev3 has no MLP, so the residual stream is an additive
channel of head/embedding writes; the full QK circuit is W_E W_Q W_K^T W_E^T, the
full OV circuit is W_E W_V W_O W_U, the direct path is W_E W_U, and Q/K/V
composition is the normalized Frobenius overlap of the sender's OV write-matrix
with the receiver's read matrix.

CAVEAT -- these are APPROXIMATIONS, not exact. segrev3 has a parameter-free
Pre-LayerNorm (LayerNormPre) before every Q/K/V read and before the unembed,
which RMS-normalizes each position's residual by an input-dependent per-position
scale. The weight products above drop that scale, so they are the usual Elhage
weight-circuit approximations (the standard, very useful first-order picture --
exact only up to the per-position LN normalization). Treat magnitudes as
relative, not absolute logit/attention contributions.

These analyses are INPUT-INDEPENDENT (computed from weights alone). They are
the static "wiring diagram"; confirm every high score causally with
`path_patching` from patching_dla.py (composition magnitude does
not prove the path is used, and the random baseline is ~1/sqrt(d_model), not 0).

References: Elhage et al. 2021, "A Mathematical Framework for Transformer
Circuits" (QK/OV circuits, eigenvalue copying score, K/Q/V-composition).
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass, field
from typing import Optional, Sequence

import torch
from torch import Tensor

from transformer_lens import HookedTransformer

# The canonical loader lives in model_loading.py (single source of truth).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_loading import load_segrev3  # noqa: E402

Head = tuple  # (layer, head)


# --------------------------------------------------------------------------- #
# Result containers                                                           #
# --------------------------------------------------------------------------- #


@dataclass
class QKCircuitResult:
    layer: int
    head: int
    token_matrix: Tensor            # [d_vocab, d_vocab] (query token x key token)
    token_softmax: Tensor           # row-softmax of token_matrix
    q_map: Tensor                   # [d_vocab, d_head]  W_E W_Q  (input->head[Q])
    k_map: Tensor                   # [d_vocab, d_head]  W_E W_K  (input->head[K])
    positional_matrix: Optional[Tensor]   # [n_ctx, n_ctx] RAW (unmasked) or None
    positional_matrix_causal: Optional[Tensor]  # positional_matrix with future (key>query) = NaN
    pos_q_map: Optional[Tensor]     # [n_ctx, d_head] W_pos W_Q, or None
    pos_k_map: Optional[Tensor]     # [n_ctx, d_head] W_pos W_K, or None
    include_bias: bool
    scale: float

    def summary(self, top_k: int = 8) -> dict:
        m = self.token_matrix
        d = m.shape[0]
        out = {"layer": self.layer, "head": self.head, "kind": "QK",
               "scale": self.scale, "include_bias": self.include_bias}
        # strongest (query->key) token preferences
        vals, idx = torch.topk(m.flatten(), min(top_k, m.numel()))
        out["top_query_to_key"] = [
            {"query_token": int(i) // d, "key_token": int(i) % d, "score": float(m.flatten()[int(i)])}
            for i in idx
        ]
        # for each query token, its single most-attended key token (by content)
        out["argmax_key_per_query"] = [int(x) for x in m.argmax(dim=1)]
        if self.positional_matrix is not None:
            pm = self.positional_matrix
            n = pm.shape[0]
            # The model is CAUSAL: a query at position q can only attend to keys at
            # positions <= q. So all positional metrics are computed over the
            # reachable (lower-triangular) region only; the upper triangle (future
            # keys) is never used by the model and is excluded here.
            tril = torch.tril(torch.ones(n, n, dtype=torch.bool, device=pm.device))
            pm_causal = pm.masked_fill(~tril, float("-inf"))
            # The reversal "mirror" for this task: output position 9+k reads the
            # reflected input index 9-k, i.e. the dest+src = (n_ctx-2) anti-diagonal
            # over the output destinations -- and crucially src=9-k <= dest=9+k, so
            # it is fully inside the causal region (unlike the matrix's main
            # anti-diagonal i->n-1-i, ~half of which is future/masked).
            in_len = n // 2
            refl = torch.tensor([float(pm[in_len - 1 + k, in_len - 1 - k]) for k in range(in_len)])
            maindiag = pm.diagonal(0)
            out["positional"] = {
                "argmax_src_per_dest": [int(x) for x in pm_causal.argmax(dim=1)],  # over key<=query
                "mean_reflection_diagonal": float(refl.mean()),   # causal reversal mirror (dest 9+k -> src 9-k)
                "mean_main_diagonal": float(maindiag.mean()),
                "note": "causal: future (key>query) cells excluded; "
                        "use positional_matrix_causal for plotting",
            }
        return out


@dataclass
class OVCircuitResult:
    layer: int
    head: int
    token_matrix: Tensor            # [d_vocab, d_vocab] (attended token -> output-logit token)
    v_map: Tensor                   # [d_vocab, d_head]  W_E W_V  (input->head[V])
    logit_map: Tensor               # [d_head, d_vocab]  W_O W_U  (head->logit); token_matrix = v_map @ logit_map
    copying_score: float            # sum(Re lambda)/sum|lambda|  in [-1, 1]
    frac_positive_eig: float        # fraction of eigenvalues with positive real part
    eigenvalues: Tensor             # complex [d_vocab]

    def summary(self, top_k: int = 8) -> dict:
        m = self.token_matrix
        d = m.shape[0]
        vals, idx = torch.topk(m.flatten(), min(top_k, m.numel()))
        return {
            "layer": self.layer, "head": self.head, "kind": "OV",
            "copying_score": self.copying_score,
            "frac_positive_eig": self.frac_positive_eig,
            "top_attended_to_output": [
                {"attended_token": int(i) // d, "boosts_output_token": int(i) % d,
                 "score": float(m.flatten()[int(i)])} for i in idx
            ],
            "argmax_output_per_attended": [int(x) for x in m.argmax(dim=1)],
        }


@dataclass
class CompositionResult:
    heads: Sequence[Head]
    q_comp: Tensor                  # [n_heads, n_heads] sender x receiver
    k_comp: Tensor
    v_comp: Tensor
    baseline: float                 # random-weight normalized-Frobenius baseline
    valid_mask: Tensor              # [n_heads, n_heads] True where sender layer < receiver layer

    def summary(self, top_k: int = 8) -> dict:
        def top_edges(mat: Tensor, channel: str):
            masked = mat.clone()
            masked[~self.valid_mask] = float("-inf")
            k = min(top_k, int(self.valid_mask.sum()))
            if k <= 0:
                return []
            vals, idx = torch.topk(masked.flatten(), k)
            n = len(self.heads)
            out = []
            for i in idx:
                si, ri = int(i) // n, int(i) % n
                score = float(masked.flatten()[int(i)])
                out.append({
                    "channel": channel,
                    "sender": f"L{self.heads[si][0]}H{self.heads[si][1]}",
                    "receiver": f"L{self.heads[ri][0]}H{self.heads[ri][1]}",
                    "score": score,
                    "ratio_to_baseline": score / self.baseline if self.baseline else None,
                })
            return out
        return {
            "kind": "QKV_composition",
            "n_heads": len(self.heads),
            "baseline": self.baseline,
            "top_q_composition": top_edges(self.q_comp, "Q"),
            "top_k_composition": top_edges(self.k_comp, "K"),
            "top_v_composition": top_edges(self.v_comp, "V"),
        }


@dataclass
class DirectPathResult:
    token_matrix: Tensor            # [d_vocab, d_vocab] (input token -> output-logit token)
    copying_score: float            # sum(Re lambda)/sum|lambda|  in [-1, 1]
    frac_positive_eig: float        # fraction of eigenvalues with positive real part
    eigenvalues: Tensor             # complex [d_vocab]
    positional_matrix: Optional[Tensor]   # [n_ctx, d_vocab] (W_pos W_U) or None
    include_bias: bool

    def summary(self, top_k: int = 8) -> dict:
        m = self.token_matrix
        d = m.shape[1]
        vals, idx = torch.topk(m.flatten(), min(top_k, m.numel()))
        out = {
            "kind": "direct_path",
            "copying_score": self.copying_score,
            "frac_positive_eig": self.frac_positive_eig,
            "include_bias": self.include_bias,
            "top_input_to_output": [
                {"input_token": int(i) // d, "boosts_output_token": int(i) % d,
                 "score": float(m.flatten()[int(i)])} for i in idx
            ],
            "argmax_output_per_input": [int(x) for x in m.argmax(dim=1)],
        }
        if self.positional_matrix is not None:
            out["positional"] = {
                "argmax_output_per_position": [int(x) for x in self.positional_matrix.argmax(dim=1)],
            }
        return out


@dataclass
class InputToHeadResult:
    layer: int
    head: int
    q_map: Tensor                   # [d_vocab, d_head]  W_E W_Q
    k_map: Tensor                   # [d_vocab, d_head]  W_E W_K
    v_map: Tensor                   # [d_vocab, d_head]  W_E W_V
    pos_q_map: Optional[Tensor]     # [n_ctx, d_head] W_pos W_Q, or None
    pos_k_map: Optional[Tensor]
    pos_v_map: Optional[Tensor]
    include_bias: bool

    def summary(self, top_k: int = 8) -> dict:
        def channel(name: str, m: Tensor) -> dict:
            norms = m.norm(dim=1)               # per input token: ||token's write into subspace||
            vals, idx = torch.topk(norms, min(top_k, m.shape[0]))
            return {
                "channel": name,
                "argmax_token": int(norms.argmax()),
                "top_tokens_by_write_norm": [
                    {"token": int(i), "write_norm": float(norms[int(i)])} for i in idx
                ],
            }
        return {
            "kind": "input_to_head",
            "layer": self.layer, "head": self.head, "include_bias": self.include_bias,
            "Q": channel("Q", self.q_map),
            "K": channel("K", self.k_map),
            "V": channel("V", self.v_map),
        }


@dataclass
class HeadToLogitResult:
    layer: int
    head: int
    z_map: Tensor                   # [d_head, d_vocab]   W_O W_U      (head output space -> logits)
    resid_map: Tensor               # [d_model, d_vocab]  W_OV W_U     (source residual dir -> logits)

    def summary(self, top_k: int = 8) -> dict:
        col_norms = self.resid_map.norm(dim=0)   # per output token: max logit gain per unit residual
        vals, idx = torch.topk(col_norms, min(top_k, col_norms.shape[0]))
        return {
            "kind": "head_to_logit",
            "layer": self.layer, "head": self.head,
            # ord=2 spectral norm is SVD-based; run on CPU (tiny matrices) to avoid
            # the cuda SVD JIT kernel that nvrtc can't build for this GPU's arch.
            "spectral_norm_z": float(torch.linalg.matrix_norm(self.z_map.cpu(), ord=2)),
            "spectral_norm_resid": float(torch.linalg.matrix_norm(self.resid_map.cpu(), ord=2)),
            "top_promotable_output_tokens": [
                {"output_token": int(i), "gain_norm": float(col_norms[int(i)])} for i in idx
            ],
        }


# --------------------------------------------------------------------------- #
# 1. QK circuit                                                               #
# --------------------------------------------------------------------------- #


@torch.no_grad()
def qk_circuit(
    model: HookedTransformer,
    layer: int,
    head: int,
    include_bias: bool = False,
    include_positional: bool = True,
) -> QKCircuitResult:
    """The QK (attention-score) circuit of a single head among the input tokens.

    Content circuit (the main result, "among input tokens"):
        QK[i, j] = ( W_E[i] W_Q + b_Q ) . ( W_E[j] W_K + b_K )  /  sqrt(d_head)
    i.e. how strongly a query carrying token i attends to a key carrying token j,
    purely from the weights. With `include_bias=False` this is the canonical full
    QK circuit  W_E W_Q W_K^T W_E^T / sqrt(d_head)  (shape [d_vocab, d_vocab]).

    Because reversal is a POSITIONAL permutation, `include_positional=True` also
    returns the positional QK circuit
        QK_pos[a, b] = (W_pos[a] W_Q) . (W_pos[b] W_K) / sqrt(d_head)
    over the n_ctx positions. NOTE: the model is CAUSAL, so only the lower triangle
    (key <= query) is ever used; `positional_matrix` is the RAW (unmasked) product
    and `positional_matrix_causal` masks the future (upper) triangle to NaN for
    honest plotting. The reversal mirror for this task is the CAUSAL reflection
    diagonal (output dest 9+k attends to the reflected input src 9-k), which the
    summary reports as `mean_reflection_diagonal` (not the matrix's main
    anti-diagonal i->n-1-i, ~half of which is future/masked).

    The row-softmax `token_softmax` shows, per query token, the content-only
    attention distribution over key tokens.
    """
    W_E = model.W_E                         # [d_vocab, d_model]
    W_Q = model.W_Q[layer, head]            # [d_model, d_head]
    W_K = model.W_K[layer, head]            # [d_model, d_head]
    scale = math.sqrt(model.cfg.d_head)

    q = W_E @ W_Q                           # [d_vocab, d_head]
    k = W_E @ W_K
    if include_bias:
        q = q + model.b_Q[layer, head]
        k = k + model.b_K[layer, head]
    token_matrix = (q @ k.T) / scale        # [d_vocab, d_vocab]
    token_softmax = torch.softmax(token_matrix, dim=-1)

    positional_matrix = None
    qp = kp = None
    if include_positional and model.cfg.positional_embedding_type == "standard":
        W_pos = model.W_pos                 # [n_ctx, d_model]
        qp = W_pos @ W_Q
        kp = W_pos @ W_K
        if include_bias:
            qp = qp + model.b_Q[layer, head]
            kp = kp + model.b_K[layer, head]
        positional_matrix = (qp @ kp.T) / scale   # [n_ctx, n_ctx]

    # Causal version for honest plotting: the model can only attend to key<=query,
    # so set the future (upper-triangular) cells to NaN (rendered blank, not 0).
    positional_matrix_causal = None
    if positional_matrix is not None:
        n = positional_matrix.shape[0]
        future = torch.triu(torch.ones(n, n, dtype=torch.bool, device=positional_matrix.device),
                            diagonal=1)
        positional_matrix_causal = positional_matrix.masked_fill(future, float("nan"))

    return QKCircuitResult(
        layer=layer, head=head,
        token_matrix=token_matrix.detach(),
        token_softmax=token_softmax.detach(),
        q_map=q.detach(), k_map=k.detach(),
        positional_matrix=None if positional_matrix is None else positional_matrix.detach(),
        positional_matrix_causal=None if positional_matrix_causal is None else positional_matrix_causal.detach(),
        pos_q_map=None if qp is None else qp.detach(),
        pos_k_map=None if kp is None else kp.detach(),
        include_bias=include_bias, scale=scale,
    )


# --------------------------------------------------------------------------- #
# 2. OV circuit                                                               #
# --------------------------------------------------------------------------- #


def _copying_score(square_matrix: Tensor) -> tuple[float, float, Tensor]:
    """Eigenvalue copying score of a square token->token map (Elhage et al. 2021).

    Returns (copying_score, frac_positive_eig, eigenvalues):
        copying_score = sum(Re lambda) / sum |lambda|   in [-1, 1]
    near +1 => the map copies the input token to its own logit; near -1 =>
    suppresses it. frac_positive_eig is the fraction of eigenvalues with Re > 0.
    """
    # eigvals of a general (non-symmetric) complex map: run on CPU. The matrix is
    # tiny (d_vocab x d_vocab), so this is instant, and it sidesteps the cuda
    # complex-eig JIT kernel (nvrtc has no codegen for this GPU's arch).
    eig = torch.linalg.eigvals(square_matrix.cpu())   # complex
    abs_sum = float(eig.abs().sum())
    copying_score = float(eig.real.sum() / abs_sum) if abs_sum > 0 else 0.0
    frac_pos = float((eig.real > 0).float().mean())
    return copying_score, frac_pos, eig.detach()


@torch.no_grad()
def ov_circuit(
    model: HookedTransformer,
    layer: int,
    head: int,
) -> OVCircuitResult:
    """The OV (value-output) circuit of a single head among the input tokens.

    Full OV circuit (shape [d_vocab, d_vocab]):
        OV[i, j] = W_E[i] W_V W_O W_U[:, j]
    = if the head attends to input token i, the (approximate) additive
    contribution to the logit of output token j -- approximate because the final
    LayerNormPre rescales the residual per position before the unembed; this drops
    that input-dependent scale (the standard Elhage OV-circuit picture).

    Eigenvalue copying score (Elhage et al. 2021): the OV token-to-token matrix
    is square (d_vocab x d_vocab); its eigenvalues summarize copy vs anti-copy
    behavior.
        copying_score = sum(Re lambda_i) / sum |lambda_i|   in [-1, 1]
    Near +1 => the head copies the attended token to its own logit (a classic
    "copying"/mover head); near -1 => it suppresses it. Also reports the fraction
    of eigenvalues with positive real part.
    """
    W_E = model.W_E                         # [d_vocab, d_model]
    W_V = model.W_V[layer, head]            # [d_model, d_head]
    W_O = model.W_O[layer, head]            # [d_head, d_model]
    W_U = model.W_U                         # [d_model, d_vocab]

    v_map = W_E @ W_V                        # [d_vocab, d_head]  input->head[V]
    logit_map = W_O @ W_U                    # [d_head, d_vocab]  head->logit
    ov = v_map @ logit_map                   # [d_vocab, d_vocab]  == W_E W_V W_O W_U
    copying_score, frac_pos, eig = _copying_score(ov)

    return OVCircuitResult(
        layer=layer, head=head, token_matrix=ov.detach(),
        v_map=v_map.detach(), logit_map=logit_map.detach(),
        copying_score=copying_score, frac_positive_eig=frac_pos,
        eigenvalues=eig,
    )


# --------------------------------------------------------------------------- #
# 3. Q/K/V composition between heads                                          #
# --------------------------------------------------------------------------- #


def _normalized_frobenius(A: Tensor, B: Tensor) -> float:
    """||A B||_F / (||A||_F ||B||_F)."""
    A, B = A.detach(), B.detach()
    num = torch.linalg.matrix_norm(A @ B, ord="fro")
    den = torch.linalg.matrix_norm(A, ord="fro") * torch.linalg.matrix_norm(B, ord="fro")
    return float(num / den) if float(den) > 0 else 0.0


def _composition_baseline(d_model: int, d_head: int, device: str,
                          n_samples: int = 100) -> float:
    """Expected normalized-Frobenius overlap of a random (d_model,d_model)
    OV-shaped matrix with a random (d_model,d_head) read matrix. ~ 1/sqrt(d_model).
    The composition score is only meaningful relative to this floor."""
    gen = torch.Generator().manual_seed(0)   # CPU draws: the same baseline on every device
    vals = []
    for _ in range(n_samples):
        A = torch.randn(d_model, d_model, generator=gen).to(device)
        B = torch.randn(d_model, d_head, generator=gen).to(device)
        vals.append(_normalized_frobenius(A, B))
    return sum(vals) / len(vals)


@torch.no_grad()
def qkv_composition(
    model: HookedTransformer,
    heads: Optional[Sequence[Head]] = None,
    baseline_samples: int = 100,
) -> CompositionResult:
    """Q/K/V composition scores between every ordered pair of heads in a circuit
    subgraph (sender must be in an earlier layer than receiver).

    Sender head A writes to the residual stream via its OV circuit
        W_OV^A = W_V^A W_O^A            (d_model x d_model).
    Receiver head B reads the residual stream via W_Q^B / W_K^B / W_V^B
        (d_model x d_head each). A token's value flows A -> B's query as
        x . (W_OV^A W_Q^B), so the composition strength is the normalized
        Frobenius overlap
        comp_Q(A->B) = || W_OV^A W_Q^B ||_F / ( ||W_OV^A||_F ||W_Q^B||_F )
    and analogously comp_K (with W_K^B) and comp_V (with W_V^B).

    `heads` is the subgraph (list of (layer, head)); default = all heads. Returns
    [n_heads, n_heads] sender-by-receiver matrices for Q/K/V, a `valid_mask`
    (True where sender layer < receiver layer), and a random-weight `baseline`
    (~1/sqrt(d_model)). Report scores relative to the baseline, and CONFIRM high
    scores causally with path_patching -- composition magnitude alone does not
    prove the edge is used.
    """
    device = next(model.parameters()).device.type
    if heads is None:
        heads = [(l, h) for l in range(model.cfg.n_layers)
                 for h in range(model.cfg.n_heads)]
    heads = list(heads)
    n = len(heads)

    # Precompute OV write-matrices for potential senders and read-matrices for receivers.
    W_OV = {hd: (model.W_V[hd[0], hd[1]] @ model.W_O[hd[0], hd[1]]) for hd in heads}

    q_comp = torch.zeros(n, n, device=device)
    k_comp = torch.zeros(n, n, device=device)
    v_comp = torch.zeros(n, n, device=device)
    valid = torch.zeros(n, n, dtype=torch.bool, device=device)

    for si, A in enumerate(heads):
        ov_a = W_OV[A]
        for ri, B in enumerate(heads):
            if A[0] >= B[0]:                       # sender must precede receiver
                continue
            valid[si, ri] = True
            q_comp[si, ri] = _normalized_frobenius(ov_a, model.W_Q[B[0], B[1]])
            k_comp[si, ri] = _normalized_frobenius(ov_a, model.W_K[B[0], B[1]])
            v_comp[si, ri] = _normalized_frobenius(ov_a, model.W_V[B[0], B[1]])

    baseline = _composition_baseline(model.cfg.d_model, model.cfg.d_head,
                                     device, n_samples=baseline_samples)

    return CompositionResult(
        heads=heads, q_comp=q_comp, k_comp=k_comp, v_comp=v_comp,
        baseline=baseline, valid_mask=valid,
    )


# --------------------------------------------------------------------------- #
# 4. Direct path (input -> logits)                                            #
# --------------------------------------------------------------------------- #


@torch.no_grad()
def direct_path(
    model: HookedTransformer,
    include_bias: bool = False,
    include_positional: bool = True,
) -> DirectPathResult:
    """The input->logit DIRECT PATH (bigram / skip circuit) among the input tokens.

        DP[i, j] = W_E[i] . W_U[:, j]  = (W_E W_U)[i, j]
    = the additive contribution to the logit of output token j made purely by
    input token i travelling the residual skip connection straight to the
    unembedding, touching NO attention head (approximate: the embedding term
    survives additively to the final residual, but the final LayerNormPre then
    rescales it by an input-dependent per-position factor that this product drops).
    This is the model's pure unigram baseline; the heads then correct it. It is the
    top-right cell of the 2x2 and the only edge with no head.

    Square (d_vocab x d_vocab), so it carries the same eigenvalue copying score as
    the OV circuit: near +1 => the direct path tends to repeat the current token.
    With include_positional, also returns W_pos W_U ([n_ctx, d_vocab]) -- the
    additive logit contribution of merely being at each position. include_bias adds
    the unembedding bias b_U (a per-output-token constant shift), if present.
    """
    W_E = model.W_E                         # [d_vocab, d_model]
    W_U = model.W_U                         # [d_model, d_vocab]
    b_U = getattr(model, "b_U", None)

    dp = W_E @ W_U                          # [d_vocab, d_vocab]
    if include_bias and b_U is not None:
        dp = dp + b_U
    copying_score, frac_pos, eig = _copying_score(dp)

    positional_matrix = None
    if include_positional and model.cfg.positional_embedding_type == "standard":
        positional_matrix = model.W_pos @ W_U   # [n_ctx, d_vocab]
        if include_bias and b_U is not None:
            positional_matrix = positional_matrix + b_U

    return DirectPathResult(
        token_matrix=dp.detach(),
        copying_score=copying_score, frac_positive_eig=frac_pos, eigenvalues=eig,
        positional_matrix=None if positional_matrix is None else positional_matrix.detach(),
        include_bias=include_bias,
    )


# --------------------------------------------------------------------------- #
# 5. Input -> head read maps                                                  #
# --------------------------------------------------------------------------- #


@torch.no_grad()
def input_to_head(
    model: HookedTransformer,
    layer: int,
    head: int,
    include_bias: bool = False,
    include_positional: bool = True,
) -> InputToHeadResult:
    """The input->head read maps W_E W_{Q/K/V} of a single head (top-left cell of
    the 2x2 -- embedding as sender, head Q/K/V as receiver).

    For channel C in {Q, K, V}:
        map_C[i] = W_E[i] W_C   (+ b_C if include_bias)   in R^{d_head}
    i.e. the vector input token i writes into the head's C-subspace. The row norm
    ||map_C[i]|| ranks which input tokens this head is most tuned to read on each
    channel. These are exactly the factors qk_circuit (Q, K) and ov_circuit (V)
    glue together; exposed here as first-class edges so the embedding-as-sender row
    has the same granularity as qkv_composition. With include_positional, also
    returns the positional read maps W_pos W_{Q/K/V} ([n_ctx, d_head]).
    """
    W_E = model.W_E
    W_Q = model.W_Q[layer, head]
    W_K = model.W_K[layer, head]
    W_V = model.W_V[layer, head]

    q_map = W_E @ W_Q
    k_map = W_E @ W_K
    v_map = W_E @ W_V
    if include_bias:
        q_map = q_map + model.b_Q[layer, head]
        k_map = k_map + model.b_K[layer, head]
        v_map = v_map + model.b_V[layer, head]

    pos_q = pos_k = pos_v = None
    if include_positional and model.cfg.positional_embedding_type == "standard":
        W_pos = model.W_pos
        pos_q = W_pos @ W_Q
        pos_k = W_pos @ W_K
        pos_v = W_pos @ W_V
        if include_bias:
            pos_q = pos_q + model.b_Q[layer, head]
            pos_k = pos_k + model.b_K[layer, head]
            pos_v = pos_v + model.b_V[layer, head]

    return InputToHeadResult(
        layer=layer, head=head,
        q_map=q_map.detach(), k_map=k_map.detach(), v_map=v_map.detach(),
        pos_q_map=None if pos_q is None else pos_q.detach(),
        pos_k_map=None if pos_k is None else pos_k.detach(),
        pos_v_map=None if pos_v is None else pos_v.detach(),
        include_bias=include_bias,
    )


# --------------------------------------------------------------------------- #
# 6. Head -> logit write maps                                                 #
# --------------------------------------------------------------------------- #


@torch.no_grad()
def head_to_logit(
    model: HookedTransformer,
    layer: int,
    head: int,
) -> HeadToLogitResult:
    """The head->logit write maps of a single head (bottom-right cell of the 2x2).

        z_map[:, j]     = (W_O W_U)[:, j]     in R^{d_head}   -- head output space -> logit j
        resid_map[:, j] = (W_V W_O W_U)[:, j] in R^{d_model}  -- a source-position residual
                                                                 direction -> logit j

    resid_map is the head->logit cell of the 2x2 (sender write-matrix = W_OV,
    receiver read-matrix = W_U). Composing it with the embedding write recovers the
    OV circuit EXACTLY: ov_circuit(...).token_matrix == W_E @ resid_map. Use this
    when the head reads a V-COMPOSED signal from an upstream head rather than the
    raw token embedding: resid_map says where that moved signal lands in logit
    space. Column norm ||resid_map[:, j]|| ranks the output tokens the head is wired
    to push (magnitude only -- confirm sign/causality with direct_logit_attribution
    / path_patching from patching_dla.py).
    """
    W_V = model.W_V[layer, head]
    W_O = model.W_O[layer, head]
    W_U = model.W_U

    z_map = W_O @ W_U                        # [d_head, d_vocab]
    resid_map = W_V @ z_map                  # [d_model, d_vocab]  = W_V W_O W_U
    return HeadToLogitResult(
        layer=layer, head=head,
        z_map=z_map.detach(), resid_map=resid_map.detach(),
    )


# --------------------------------------------------------------------------- #
# Unifying core: every weight-based edge = (sender write) @ (receiver read)    #
# --------------------------------------------------------------------------- #
#
# source   ::= "embed" | "pos" | (layer, head)
# receiver ::= "logits" | (layer, head, "q" | "k" | "v")


@torch.no_grad()
def residual_write_matrix(model: HookedTransformer, source) -> Tensor:
    """Matrix mapping a sender's native space into the residual stream.

        "embed"       -> W_E              [d_vocab, d_model]
        "pos"         -> W_pos            [n_ctx,  d_model]
        (layer, head) -> W_OV = W_V W_O   [d_model, d_model]
    """
    if source == "embed":
        return model.W_E
    if source == "pos":
        return model.W_pos
    layer, head = source
    return model.W_V[layer, head] @ model.W_O[layer, head]


@torch.no_grad()
def residual_read_matrix(model: HookedTransformer, receiver) -> Tensor:
    """Matrix reading the residual stream into a receiver's native space.

        "logits"                 -> W_U            [d_model, d_vocab]
        (layer, head, channel)   -> W_{Q/K/V}      [d_model, d_head]
    """
    if receiver == "logits":
        return model.W_U
    layer, head, channel = receiver
    W = {"q": model.W_Q, "k": model.W_K, "v": model.W_V}[channel.lower()]
    return W[layer, head]


@torch.no_grad()
def edge_circuit(model: HookedTransformer, source, receiver) -> Tensor:
    """The raw weight-based circuit of ANY edge: (sender write) @ (receiver read).

    Covers the whole 2x2 (plus positional sender) uniformly -- this is the object
    to build an ACDC/EAP-style edge graph on top of:

        ("embed", "logits")        -> W_E W_U      [d_vocab, d_vocab]  (direct path)
        ("embed", (l, h, "q"))     -> W_E W_Q      [d_vocab, d_head]   (input->head)
        ((l0,h0), (l1,h1,"v"))     -> W_OV W_V     [d_model, d_head]   (head->head, V)
        ((l, h), "logits")         -> W_OV W_U     [d_model, d_vocab]  (head->logit)

    NOTE: this is the UN-normalized product. qkv_composition reports the
    normalized-Frobenius version for the head->head cell. For head->head edges the
    product is only causally meaningful when source layer < receiver layer.
    """
    return residual_write_matrix(model, source) @ residual_read_matrix(model, receiver)


# --------------------------------------------------------------------------- #
# Smoke test                                                                  #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    model = load_segrev3()

    print("== qk_circuit (L1H3, content + positional) ==")
    qk = qk_circuit(model, layer=1, head=3, include_positional=True)
    s = qk.summary(top_k=3)
    print("token_matrix shape:", tuple(qk.token_matrix.shape),
          "| positional shape:", tuple(qk.positional_matrix.shape),
          "| causal NaN cells:", int(qk.positional_matrix_causal.isnan().sum()))
    print("top query->key:", s["top_query_to_key"][:2])
    print("positional (causal reflection vs main diag):",
          round(s["positional"]["mean_reflection_diagonal"], 3),
          round(s["positional"]["mean_main_diagonal"], 3))

    print("\n== ov_circuit (L3H3) ==")
    ov = ov_circuit(model, layer=3, head=3)
    print("OV shape:", tuple(ov.token_matrix.shape),
          "| copying_score:", round(ov.copying_score, 4),
          "| frac_positive_eig:", round(ov.frac_positive_eig, 3))
    print(ov.summary(top_k=2)["top_attended_to_output"])

    print("\n== qkv_composition (all 32 heads) ==")
    comp = qkv_composition(model)
    cs = comp.summary(top_k=3)
    print("baseline (~1/sqrt(d_model)=%.4f):" % (1 / math.sqrt(model.cfg.d_model)),
          round(comp.baseline, 4))
    print("q_comp shape:", tuple(comp.q_comp.shape),
          "| valid edges:", int(comp.valid_mask.sum()))
    for e in cs["top_q_composition"][:2]:
        print("  Q-comp:", e)
    for e in cs["top_v_composition"][:2]:
        print("  V-comp:", e)

    # sanity: composition restricted to a small subgraph
    print("\n== qkv_composition (subgraph [L0H0, L1H3, L3H3]) ==")
    comp2 = qkv_composition(model, heads=[(0, 0), (1, 3), (3, 3)])
    print("valid edges:", int(comp2.valid_mask.sum()),
          "| max V-comp:", round(float(comp2.v_comp.max()), 4))

    print("\n== direct_path (input -> logits) ==")
    dp = direct_path(model, include_positional=True)
    ds = dp.summary(top_k=3)
    print("DP shape:", tuple(dp.token_matrix.shape),
          "| copying_score:", round(dp.copying_score, 4),
          "| frac_positive_eig:", round(dp.frac_positive_eig, 3))
    print("top input->output:", ds["top_input_to_output"][:2])

    print("\n== input_to_head (L1H3) ==")
    i2h = input_to_head(model, layer=1, head=3)
    print("q/k/v map shapes:", tuple(i2h.q_map.shape), tuple(i2h.k_map.shape), tuple(i2h.v_map.shape))
    print("top Q-write tokens:", i2h.summary(top_k=2)["Q"]["top_tokens_by_write_norm"])

    print("\n== head_to_logit (L3H3) ==")
    h2l = head_to_logit(model, layer=3, head=3)
    print("z_map/resid_map shapes:", tuple(h2l.z_map.shape), tuple(h2l.resid_map.shape))
    print("top promotable outputs:", h2l.summary(top_k=2)["top_promotable_output_tokens"])

    # consistency: the new atoms reconstruct the end-to-end circuits exactly.
    print("\n== consistency checks ==")
    qk13 = qk_circuit(model, layer=1, head=3, include_positional=False)
    ov33 = ov_circuit(model, layer=3, head=3)
    i2h33 = input_to_head(model, layer=3, head=3)
    # qk_circuit factors == input->head[Q],[K]
    assert torch.allclose(qk13.q_map, input_to_head(model, 1, 3).q_map, atol=1e-5)
    # ov factors: token_matrix == v_map @ logit_map, and v_map == input->head[V]
    assert torch.allclose(ov33.token_matrix, ov33.v_map @ ov33.logit_map, atol=1e-4)
    assert torch.allclose(ov33.v_map, i2h33.v_map, atol=1e-5)
    # ov == W_E @ head_to_logit.resid_map  (the headlogit cell glued to the embedding)
    assert torch.allclose(ov33.token_matrix, model.W_E @ head_to_logit(model, 3, 3).resid_map, atol=1e-3)
    # edge_circuit reproduces direct_path, input->head, head->logit
    assert torch.allclose(edge_circuit(model, "embed", "logits"), direct_path(model, include_positional=False).token_matrix, atol=1e-5)
    assert torch.allclose(edge_circuit(model, "embed", (1, 3, "q")), i2h.q_map, atol=1e-5)
    assert torch.allclose(edge_circuit(model, (3, 3), "logits"), head_to_logit(model, 3, 3).resid_map, atol=1e-5)
    print("all atoms reconstruct qk/ov circuits and edge_circuit agrees. OK")

    print("\nALL SMOKE TESTS RAN.")
