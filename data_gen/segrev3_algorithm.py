"""
segrev3_algorithm.py
=================================================================================
The pure segrev3 ALGORITHM and nothing else -- the single source of truth for the
transform. Input-only and deterministic (no model, no randomness, no batching of
random data): given a sequence of digits it returns the segment-reversed
sequence. Data generation (``data_gen/data_gen.py``) and any other consumer
import the algorithm from here so there is exactly one implementation.

TASK ``segrev3`` == DSL ``SegmentReverse(boundary_digit=3)``:
  reverse each maximal run of non-3 digits; every 3 stays fixed in place.
  e.g.  [1, 7, 3, 4, 2, 5] -> [7, 1, 3, 5, 2, 4]
        leading/trailing runs (no bounding 3) are reversed too.

  segrev3_list   : the canonical map -- a sequence of 10 digits in, the
                   segment-reversed sequence of 10 digits out.
  segrev3_tensor : the same map applied row-wise to a [batch, L] LongTensor.
  source_index_map : for each output position i, the input index it copies from,
                   s(i) = a + b - 1 - i within the segment [a, b) (an INVOLUTION:
                   s(s(i)) = i); a boundary position copies itself.
"""

from __future__ import annotations

import os
import sys
from typing import Sequence

import torch
from torch import Tensor

# The boundary digit is owned by the architecture ground truth; import it so the
# default can never drift from the rest of the codebase.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_TOOLS_DIR = os.path.join(os.path.dirname(_THIS_DIR), "interp_tools")
sys.path.insert(0, _TOOLS_DIR)
from model_loading import BOUNDARY_DIGIT, INPUT_LEN  # noqa: E402


def segrev3_list(x: Sequence[int], boundary: int = BOUNDARY_DIGIT) -> list:
    """segrev3 on a python sequence (matches the official DSL SegmentReverse):
    a sequence of digits in, the segment-reversed sequence of digits out."""
    x = list(x)
    L = len(x)
    out = [0] * L
    seg_start = 0
    for i in range(L):
        if x[i] == boundary:
            seg = x[seg_start:i][::-1]
            for j, v in enumerate(seg):
                out[seg_start + j] = v
            out[i] = boundary
            seg_start = i + 1
    seg = x[seg_start:L][::-1]
    for j, v in enumerate(seg):
        out[seg_start + j] = v
    return out


def segrev3_tensor(inputs: Tensor, boundary: int = BOUNDARY_DIGIT) -> Tensor:
    """Vectorized-by-row segrev3 over a [batch, L] LongTensor -> [batch, L]."""
    rows = [segrev3_list(r.tolist(), boundary) for r in inputs]
    return torch.tensor(rows, dtype=torch.long, device=inputs.device)


def source_index_map(x: Sequence[int], boundary: int = BOUNDARY_DIGIT) -> list:
    """For each output position i, the input index it copies from (s(i)); for a
    boundary position i (x[i]==boundary) we return i (it copies itself)."""
    x = list(x)
    L = len(x)
    src = list(range(L))
    seg_start = 0
    for i in range(L + 1):
        if i == L or x[i] == boundary:
            a, b = seg_start, i                      # segment [a, b)
            for j in range(a, b):
                src[j] = a + b - 1 - j               # reflect
            seg_start = i + 1
    return src


# --------------------------------------------------------------------------- #
# Smoke test (the algorithm + the involution property of its source map)      #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    # algorithm matches the official examples
    assert segrev3_list([3, 8, 2, 5, 7, 1, 9, 5, 4, 6], boundary=5) == \
        [2, 8, 3, 5, 9, 1, 7, 5, 6, 4]
    assert segrev3_list([0, 3, 7, 2, 0, 8, 1, 0, 4, 6], boundary=0) == \
        [0, 2, 7, 3, 0, 1, 8, 0, 6, 4]
    # default-boundary (3) example
    assert segrev3_list([1, 7, 3, 4, 2, 5]) == [7, 1, 3, 5, 2, 4]
    # segrev3_tensor agrees with segrev3_list row-wise
    batch = torch.randint(0, 10, (50, INPUT_LEN))
    assert (segrev3_tensor(batch) ==
            torch.tensor([segrev3_list(r.tolist()) for r in batch])).all()
    # involution property of the source map
    for r in batch:
        s = source_index_map(r.tolist())
        rl = r.tolist()
        for i in range(INPUT_LEN):
            if rl[i] != BOUNDARY_DIGIT:
                assert s[s[i]] == i
    print("segrev3 algorithm + involution checks PASSED")
