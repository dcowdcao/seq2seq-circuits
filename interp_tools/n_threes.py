"""
n_threes.py
=================================================================================
Shared helper for MARKING THE NUMBER OF 3s (the segrev3 boundary digit) in the
input sequences that drive every dataset and figure.

The segrev3 task segments each input on the boundary digit ``3`` (see
``data_gen/data_gen.py``), so the number of 3s in an input sets its segment
structure and is the single most informative per-input summary. This module is
the one place that:

  * counts 3s per input row (``count_threes``),
  * formats a one-line caption describing a BATCH's 3-count distribution
    (``caption``) -- every analysis figure aggregates over a batch of inputs, so
    the honest label is a distribution, which collapses to an exact count when
    every input carries the same number of 3s (e.g. the ``edge`` datasets), and
  * stamps that caption onto a matplotlib figure as a small footer
    (``annotate``).

Weight-based figures (QK/OV, copying score, composition) and purely structural
circuit graphs have NO input sequence; they are marked with the
``INPUT_INDEPENDENT`` / ``STRUCTURAL`` sentinels instead.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_loading import BOUNDARY_DIGIT  # noqa: E402

# Sentinels for figures with no input sequence.
INPUT_INDEPENDENT = "n_3s: input-independent (weight-based)"
STRUCTURAL = "n_3s: input-independent (structural circuit)"


def _rows(inputs):
    """Normalize a tensor [B, L] / list-of-lists / single list into a list of
    integer rows, so counting works regardless of the caller's representation."""
    if inputs is None:
        return []
    # torch.Tensor (duck-typed: avoid importing torch here)
    if hasattr(inputs, "tolist"):
        inputs = inputs.tolist()
    if not inputs:
        return []
    # single 1-D sequence -> wrap as one row
    if not isinstance(inputs[0], (list, tuple)):
        return [list(inputs)]
    return [list(r) for r in inputs]


def count_threes(inputs):
    """Per-row count of the boundary digit (3) in ``inputs``.

    ``inputs`` may be a tensor [B, L], a list of rows, or a single 1-D sequence.
    Returns a list[int] (one count per row)."""
    return [int(sum(1 for d in row if d == BOUNDARY_DIGIT)) for row in _rows(inputs)]


def caption(inputs, prefix: str = "n_3s") -> str:
    """One-line caption describing the 3-count of a batch of input sequences.

    * empty batch         -> ``"<prefix>: n/a"``
    * all rows equal (k)  -> ``"<prefix> = k (all N inputs)"``
    * otherwise           -> ``"<prefix>: mean M, range lo-hi, N=N"``
    """
    counts = count_threes(inputs)
    n = len(counts)
    if n == 0:
        return f"{prefix}: n/a"
    lo, hi = min(counts), max(counts)
    if lo == hi:
        return f"{prefix} = {lo} (all {n} inputs)"
    mean = sum(counts) / n
    return f"{prefix}: mean {mean:.2f}, range {lo}-{hi}, N={n}"


def annotate(fig, text: str) -> None:
    """Stamp ``text`` as a small italic footer in the bottom-right of ``fig``.

    Call right before ``savefig`` (after any ``tight_layout``); the figures here
    save with ``bbox_inches="tight"`` so the footer is always captured."""
    if not text:
        return
    fig.text(0.995, 0.005, text, ha="right", va="bottom", fontsize=7,
             color="#444444", style="italic")
