"""
metrics.py
=================================================================================
The segrev3 task METRICS: scoring functions over the answer positions (9..18)
for the clean/corrupted pairs produced by ``data_gen/data_gen.py``. Separated
from data generation so analyses can depend on the scoring layer independently.

    logit_diff     - mean (correct - incorrect) logit over the metric's answer
                     positions; works for single-position ([B]) and
                     multi-position ([B,P]) specs.
    kl_divergence  - mean KL(clean || patched) over answer positions 9..18 (the
                     standard ACDC metric).
    task_accuracy  - token- and sequence-level accuracy of a model on (tokens,
                     outputs).

The answer-position convention and the ``Segrev3Pairs`` container come from the
data-generation module; see ``interp_tools/model_loading.py`` for the architecture
ground truth.
"""

from __future__ import annotations

import os
import sys

import torch
from torch import Tensor

# Make the canonical constants and the pair container importable regardless of cwd.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
sys.path.insert(0, os.path.join(_REPO_ROOT, "interp_tools"))
sys.path.insert(0, os.path.join(_REPO_ROOT, "data_gen"))
from model_loading import ANSWER_POSITIONS                # noqa: E402
from data_gen import Segrev3Pairs                         # noqa: E402


def _gather_positions(logits: Tensor, positions: Tensor) -> Tensor:
    """logits [B, S, V], positions [B] or [B, P] -> [B, V] or [B, P, V]."""
    b = torch.arange(logits.shape[0], device=logits.device)
    if positions.ndim == 1:
        return logits[b, positions]                         # [B, V]
    out = []
    for p in range(positions.shape[1]):
        out.append(logits[b, positions[:, p]])
    return torch.stack(out, dim=1)                          # [B, P, V]


def _masked_position_mean(x: Tensor, mask) -> Tensor:
    """x [B, P] -> [B]; bool mask [B, P] selects valid positions (None = all)."""
    if mask is None:
        return x.mean(dim=1)
    m = mask.to(device=x.device, dtype=x.dtype)
    return (x * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)


def logit_diff(logits: Tensor, pairs: Segrev3Pairs, per_example: bool = False) -> Tensor:
    """Mean (correct - incorrect) logit over the metric's answer positions.

    Works for both single-position ([B]) and multi-position ([B,P]) specs.
    """
    sel = _gather_positions(logits, pairs.answer_positions)
    if pairs.answer_positions.ndim == 1:
        b = torch.arange(logits.shape[0], device=logits.device)
        d = sel[b, pairs.correct] - sel[b, pairs.incorrect]   # [B]
    else:
        cor = torch.gather(sel, 2, pairs.correct.unsqueeze(-1)).squeeze(-1)   # [B,P]
        inc = torch.gather(sel, 2, pairs.incorrect.unsqueeze(-1)).squeeze(-1)
        d = _masked_position_mean(cor - inc, pairs.answer_mask)               # [B]
    return d if per_example else d.mean()


def kl_divergence(logits: Tensor, clean_logits: Tensor, pairs: Segrev3Pairs,
                  per_example: bool = False) -> Tensor:
    """Mean KL( clean || patched ) over the metric's answer positions (the standard
    ACDC metric). Uses the PAIR's own answer positions so variable-length and
    position-partition (parity) circuits score the right block; honors per-row
    [B, P] positions plus the optional ``answer_mask`` (padded rows count only
    their valid positions)."""
    pos = pairs.answer_positions
    if pos.ndim == 1:
        pos = pos.to(logits.device)
        p = clean_logits[:, pos, :].log_softmax(-1)         # [B, P, V]
        q = logits[:, pos, :].log_softmax(-1)
        kl = (p.exp() * (p - q)).sum(-1).mean(dim=1)        # [B]
        return kl if per_example else kl.mean()
    pos = pos.to(logits.device)
    p = _gather_positions(clean_logits, pos).log_softmax(-1)   # [B, P, V]
    q = _gather_positions(logits, pos).log_softmax(-1)
    kl = _masked_position_mean((p.exp() * (p - q)).sum(-1), pairs.answer_mask)  # [B]
    return kl if per_example else kl.mean()


def task_accuracy(model, tokens: Tensor, outputs: Tensor,
                  answer_positions=None) -> dict:
    """Token- and sequence-level accuracy of `model` on (tokens, outputs).

    ``answer_positions`` (a 1-D index sequence) selects the scored logit
    positions; it defaults to the fixed segrev3 block [9..18]. Variable-length DSL
    models pass their own ``spec.answer_positions`` (its length must equal
    ``outputs.shape[1]``)."""
    pos = torch.as_tensor(ANSWER_POSITIONS if answer_positions is None
                          else answer_positions, device=tokens.device)
    with torch.no_grad():
        pred = model(tokens)[:, pos, :].argmax(-1)
    tok = (pred == outputs).float().mean().item()
    seq = (pred == outputs).all(1).float().mean().item()
    return {"token_acc": tok, "sequence_acc": seq}


# --------------------------------------------------------------------------- #
# Smoke test (metrics on the real model + generated data)                     #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    from model_loading import load_segrev3                 # noqa: E402
    from data_gen import make_clean_corrupted, segrev3_batch  # noqa: E402

    model = load_segrev3()
    toks, inp, out = segrev3_batch(3000, seed=0)
    print("task accuracy:", task_accuracy(model, toks, out))

    for scheme in ("random", "replace_threes_to_nonthrees", "replace_non_threes_to_threes"):
        pairs = make_clean_corrupted(256, scheme=scheme, seed=2)
        with torch.no_grad():
            cl = model(pairs.clean_tokens)
            co = model(pairs.corrupted_tokens)
        ld_clean = logit_diff(cl, pairs).item()
        ld_corr = logit_diff(co, pairs).item()
        kl = kl_divergence(co, cl, pairs).item()
        print(f"[{scheme:13s}] clean logit-diff={ld_clean:+.3f}  "
              f"corrupted logit-diff={ld_corr:+.3f}  KL(clean||corr)={kl:.3f}")
    print("ALL SMOKE TESTS RAN.")
