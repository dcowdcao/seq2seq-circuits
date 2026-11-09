"""
data_gen.py
=================================================================================
The segrev3 task DATA GENERATION. Generates clean/corrupted digit sequences and
stores them as **.json files** (lists of digits + metadata -- the canonical
on-disk representation); a loader reconstructs the tensor ``Segrev3Pairs`` that
the analyses consume. The pure segrev3 transform lives in
``data_gen/segrev3_algorithm.py`` (imported here); the METRICS that score these
pairs live in ``metrics/metrics.py``. See ``interp_tools/model_loading.py`` for the
architecture ground truth.

THREE DATA-GENERATION TYPES (the contrast pairs that drive every analysis):

  1. "random"            -- random_inputs(): a random sample of INPUT_LEN digits
                            0-9. As a clean/corrupted pair, corrupted is an
                            INDEPENDENT random batch. The standard global
                            contrast (ACDC / EAP / validation default).
  2. "replace_threes_to_nonthrees"    -- replace_threes_to_nonthrees(inputs, n): corrupted = clean with n
                            of its boundary digits (3) replaced by a random non-3
                            digit. Changes the segment structure -> TARGETS the
                            heads that FIND THE 3s (the boundary / 3-detector
                            heads). `n` controls the corruption strength.
  3. "replace_non_threes_to_threes"-- replace_non_threes_to_threes(inputs, n): corrupted = clean with
                            n of its non-3 digits replaced by a 3. Injects new
                            boundaries -> TARGETS the heads that REVERSE / COPY
                            the non-3s & boundary digits (the reflection / mover
                            heads). `n` controls the corruption strength.

For the two corruption types the clean inputs are rejection-sampled to contain at
least `n` eligible positions, so EXACTLY n replacements happen per example and
every pair is a valid, non-degenerate corruption (no null pairs).

TASK ``segrev3`` == DSL ``SegmentReverse(boundary_digit=3)``: reverse each
maximal run of non-3 digits; every 3 stays fixed in place. The model is fed the
20-token sequence ``[ x_0..x_9 (input) | y_0..y_9 (output = segrev3(x)) ]`` and
predicts y_k at logit position 9+k, so ANSWER POSITIONS are 9..18. Teacher-forced
true outputs are the correct on-distribution input for circuit analysis, which is
what the generators below produce.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import torch
from torch import Tensor

# Make the canonical loader / constants importable regardless of cwd.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_TOOLS_DIR = os.path.join(os.path.dirname(_THIS_DIR), "interp_tools")
sys.path.insert(0, _TOOLS_DIR)
from model_loading import (  # noqa: E402
    INPUT_LEN, BOUNDARY_DIGIT, ANSWER_POSITIONS, FIRST_ANSWER_POS,
    resolve_device,
)

# The pure segrev3 transform is the single source of truth in its own module.
sys.path.insert(0, _THIS_DIR)
from segrev3_algorithm import (  # noqa: E402,F401
    segrev3_list, segrev3_tensor, source_index_map,
)

D_VOCAB = 10
_NON3 = [d for d in range(D_VOCAB) if d != BOUNDARY_DIGIT]
_NON3_T = torch.tensor(_NON3, dtype=torch.long)

#: the three data-generation types exposed by make_clean_corrupted()
#: (canonical definition lives in canon.py, the shared pure-constants module)
from canon import SCHEMES  # noqa: E402


def _gen(seed: Optional[int] = None) -> torch.Generator:
    g = torch.Generator(device="cpu")
    if seed is not None:
        g.manual_seed(seed)
    return g


# --------------------------------------------------------------------------- #
# Type 1: random sample of digits, and token assembly                          #
# --------------------------------------------------------------------------- #


def random_inputs(batch: int, seed: Optional[int] = None,
                  device: Optional[str] = None) -> Tensor:
    """TYPE 1 -- [batch, INPUT_LEN] uniform-random digits 0-9 (repeats allowed)."""
    return torch.randint(0, D_VOCAB, (batch, INPUT_LEN),
                         generator=_gen(seed)).to(resolve_device(device))


def make_tokens(inputs: Tensor) -> tuple[Tensor, Tensor]:
    """inputs [B, INPUT_LEN] -> (tokens [B, SEQ_LEN], outputs [B, OUTPUT_LEN])."""
    outputs = segrev3_tensor(inputs)
    tokens = torch.cat([inputs, outputs], dim=1)
    return tokens, outputs


def segrev3_batch(batch: int, seed: Optional[int] = None, device: Optional[str] = None
                  ) -> tuple[Tensor, Tensor, Tensor]:
    """Convenience: (tokens [B,20], inputs [B,10], outputs [B,10])."""
    inputs = random_inputs(batch, seed=seed, device=device)
    tokens, outputs = make_tokens(inputs)
    return tokens, inputs, outputs


def _random_inputs_with_min(batch: int, n: int, want_threes: bool,
                            gen: torch.Generator) -> Tensor:
    """[batch, INPUT_LEN] random digits, each row guaranteed to have >= n eligible
    positions (>= n threes if want_threes else >= n non-threes), via rejection
    sampling so the corruption can always apply exactly n replacements."""
    if not (0 <= n <= INPUT_LEN):
        raise ValueError(f"n={n} out of range [0, {INPUT_LEN}]")
    out = torch.empty(batch, INPUT_LEN, dtype=torch.long)
    filled = 0
    while filled < batch:
        cand = torch.randint(0, D_VOCAB, (max(batch, 256), INPUT_LEN), generator=gen)
        cnt = ((cand == BOUNDARY_DIGIT) if want_threes
               else (cand != BOUNDARY_DIGIT)).sum(dim=1)
        ok = cand[cnt >= n]
        take = min(ok.shape[0], batch - filled)
        out[filled:filled + take] = ok[:take]
        filled += take
    return out


# --------------------------------------------------------------------------- #
# Types 2 & 3: corruptions targeting specific head roles                       #
# --------------------------------------------------------------------------- #


def _corrupt(inputs: Tensor, n: int, want_threes: bool, gen: torch.Generator
             ) -> tuple[Tensor, Tensor]:
    """Replace n eligible positions per row. If want_threes: pick n of the 3s and
    set each to a random non-3 digit. Else: pick n of the non-3s and set each to
    3. If a row has fewer than n eligible positions, replaces all available.
    Returns (corrupted_inputs, actual_n_replaced[B])."""
    corr = inputs.clone()
    B = inputs.shape[0]
    actual = torch.zeros(B, dtype=torch.long)
    for b in range(B):
        row = corr[b]
        elig = ((row == BOUNDARY_DIGIT) if want_threes
                else (row != BOUNDARY_DIGIT)).nonzero(as_tuple=True)[0]
        k = min(n, int(elig.numel()))
        if k == 0:
            continue
        chosen = elig[torch.randperm(int(elig.numel()), generator=gen)[:k]]
        if want_threes:
            row[chosen] = _NON3_T[torch.randint(0, len(_NON3), (k,), generator=gen)]
        else:
            row[chosen] = BOUNDARY_DIGIT
        actual[b] = k
    return corr, actual


def replace_threes_to_nonthrees(inputs: Tensor, n: int, seed: Optional[int] = None) -> Tensor:
    """TYPE 2 -- copy of `inputs` with n of each row's boundary digits (3) replaced
    by a random non-3 digit (targets the 3-detector / boundary heads)."""
    corr, _ = _corrupt(inputs, n, want_threes=True, gen=_gen(seed))
    return corr


def replace_non_threes_to_threes(inputs: Tensor, n: int, seed: Optional[int] = None) -> Tensor:
    """TYPE 3 -- copy of `inputs` with n of each row's non-3 digits replaced by a 3
    (targets the reversal / copy / mover heads)."""
    corr, _ = _corrupt(inputs, n, want_threes=False, gen=_gen(seed))
    return corr


# --------------------------------------------------------------------------- #
# Clean / corrupted pairs (tensor view) + JSON storage                         #
# --------------------------------------------------------------------------- #


@dataclass
class Segrev3Pairs:
    """A clean/corrupted dataset plus everything needed to score a logit-diff or
    KL metric over the answer positions (see ``metrics/metrics.py``).

      clean_tokens / corrupted_tokens : [B, SEQ_LEN]
      answer_positions                : [B, P] logit positions to read (9..18)
      correct / incorrect             : [B, P] token ids (clean / corrupted output)
      scheme                          : which of SCHEMES built the corruption
      answer_mask                     : [B, P] bool, valid positions per row
                                        (None = all valid; rows may be padded)
    Use ``single_position_contrast`` to derive a [B] single-position contrast for
    tools that need one (e.g. direct logit attribution).
    """
    clean_tokens: Tensor
    corrupted_tokens: Tensor
    answer_positions: Tensor
    correct: Tensor
    incorrect: Tensor
    scheme: str
    clean_inputs: Tensor = field(default=None)
    corrupted_inputs: Tensor = field(default=None)
    target_out_pos: Optional[Tensor] = None
    answer_mask: Optional[Tensor] = None
    extra: dict = field(default_factory=dict)


def count_threes(inputs) -> list:
    """Per-example count of the boundary digit (3) in ``inputs`` ([B, INPUT_LEN]
    tensor or list of rows). The number of 3s sets each input's segment structure,
    so it is stored alongside the raw digits to make every dataset self-describing."""
    rows = inputs.tolist() if hasattr(inputs, "tolist") else inputs
    return [int(sum(1 for d in row if d == BOUNDARY_DIGIT)) for row in rows]


def build_dataset(scheme: str, clean_inputs: Tensor, corrupted_inputs: Tensor,
                  n: int, seed: Optional[int]) -> dict:
    """Assemble the JSON-serializable dataset dict (the on-disk representation).
    Only the raw input digits are stored; outputs/tokens/labels are derived
    deterministically by ``dataset_to_pairs`` (via the segrev3 algorithm). The
    per-example number of 3s (boundary digits) is recorded for both the clean and
    corrupted inputs so every dataset is self-describing."""
    return {
        "scheme": scheme,
        "n": int(n),
        "seed": seed,
        "boundary_digit": BOUNDARY_DIGIT,
        "input_len": INPUT_LEN,
        "answer_positions": list(ANSWER_POSITIONS),
        "clean_inputs": clean_inputs.tolist(),
        "corrupted_inputs": corrupted_inputs.tolist(),
        "clean_n_threes": count_threes(clean_inputs),
        "corrupted_n_threes": count_threes(corrupted_inputs),
    }


def save_dataset(dataset: dict, path: str) -> str:
    """Write a dataset dict to a .json file (creating parent dirs)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(dataset, f)
    return path


def load_dataset(path: str) -> dict:
    """Read a dataset dict back from a .json file."""
    with open(path) as f:
        return json.load(f)


def dataset_to_pairs(dataset: dict, device: Optional[str] = None) -> Segrev3Pairs:
    """Reconstruct the tensor ``Segrev3Pairs`` from a stored dataset dict."""
    device = resolve_device(device)
    clean_in = torch.tensor(dataset["clean_inputs"], dtype=torch.long)
    corr_in = torch.tensor(dataset["corrupted_inputs"], dtype=torch.long)
    clean_tokens, clean_out = make_tokens(clean_in)
    corr_tokens, corr_out = make_tokens(corr_in)
    B = clean_in.shape[0]
    pos = torch.tensor(dataset["answer_positions"]).unsqueeze(0).expand(B, -1).clone()
    return Segrev3Pairs(
        clean_tokens.to(device), corr_tokens.to(device), pos.to(device),
        clean_out.to(device), corr_out.to(device), dataset["scheme"],
        clean_inputs=clean_in.to(device), corrupted_inputs=corr_in.to(device),
        extra={"n": dataset.get("n"), "seed": dataset.get("seed")},
    )


def load_pairs(path: str, device: Optional[str] = None) -> Segrev3Pairs:
    """load_dataset + dataset_to_pairs."""
    return dataset_to_pairs(load_dataset(path), device=device)


def make_clean_corrupted(
    batch: int,
    scheme: str = "random",
    seed: Optional[int] = None,
    device: Optional[str] = None,
    n: int = 1,
    save_path: Optional[str] = None,
) -> Segrev3Pairs:
    """Build a clean/corrupted pair set under one of the three SCHEMES, returning a
    tensor ``Segrev3Pairs`` (and optionally persisting the dataset to ``save_path``
    as JSON). ``n`` is the per-example replacement count for the two corruption
    schemes (ignored by "random"). The multi-position metric spans positions
    9..18 with correct = clean output, incorrect = corrupted output.
    """
    gen = _gen(seed)
    if scheme == "random":
        clean = torch.randint(0, D_VOCAB, (batch, INPUT_LEN), generator=gen)
        corrupted = torch.randint(0, D_VOCAB, (batch, INPUT_LEN), generator=gen)
    elif scheme == "replace_threes_to_nonthrees":
        clean = _random_inputs_with_min(batch, n, want_threes=True, gen=gen)
        corrupted, _ = _corrupt(clean, n, want_threes=True, gen=gen)
    elif scheme == "replace_non_threes_to_threes":
        clean = _random_inputs_with_min(batch, n, want_threes=False, gen=gen)
        corrupted, _ = _corrupt(clean, n, want_threes=False, gen=gen)
    else:
        raise ValueError(f"unknown scheme {scheme!r}; use one of {SCHEMES}")

    dataset = build_dataset(scheme, clean, corrupted, n, seed)
    if save_path is not None:
        save_dataset(dataset, save_path)
    return dataset_to_pairs(dataset, device=device)


def single_position_contrast(pairs: Segrev3Pairs) -> tuple[Tensor, Tensor, Tensor]:
    """Derive a per-example single-position contrast (positions[B], correct[B],
    incorrect[B]) from a multi-position ``Segrev3Pairs`` -- choosing, per row, an
    answer position where the clean and corrupted answers differ (the corruption's
    effect). For tools that need a single-position metric, e.g. DLA. Falls back to
    the first answer position for a row with no difference."""
    ap, cor, inc = pairs.answer_positions, pairs.correct, pairs.incorrect  # [B,P]
    diff = cor != inc                                                      # [B,P]
    has = diff.any(dim=1)
    idx = torch.where(has, diff.float().argmax(dim=1),
                      torch.zeros(ap.shape[0], dtype=torch.long, device=ap.device))
    b = torch.arange(ap.shape[0], device=ap.device)
    return ap[b, idx], cor[b, idx], inc[b, idx]


# --------------------------------------------------------------------------- #
# SINGLE (non-paired) datasets parameterized by the NUMBER OF 3s               #
# --------------------------------------------------------------------------- #
# For analyses that do NOT patch (e.g. the weight-based "edge" visualizations and
# the realized-attention plots), there is no clean/corrupted contrast -- you just
# feed inputs. The meaningful axis there is how many boundary digits (3s) the input
# has, since that sets the segment structure. These build a single dataset whose
# every input has EXACTLY ``n_threes`` 3s.


def inputs_with_n_threes(batch: int, n_threes: int, seed: Optional[int] = None,
                         device: Optional[str] = None) -> Tensor:
    """[batch, INPUT_LEN] inputs in which EVERY row has EXACTLY ``n_threes`` 3s
    (at ``n_threes`` random positions) and the remaining positions are random
    NON-3 digits. ``n_threes`` ranges 0..INPUT_LEN (0 = no boundaries / one
    reversed run; INPUT_LEN = all 3s / identity)."""
    if not (0 <= n_threes <= INPUT_LEN):
        raise ValueError(f"n_threes={n_threes} out of range [0, {INPUT_LEN}]")
    gen = _gen(seed)
    out = torch.empty(batch, INPUT_LEN, dtype=torch.long)
    for b in range(batch):
        row = _NON3_T[torch.randint(0, len(_NON3), (INPUT_LEN,), generator=gen)]  # all non-3
        if n_threes > 0:
            pos = torch.randperm(INPUT_LEN, generator=gen)[:n_threes]
            row[pos] = BOUNDARY_DIGIT
        out[b] = row
    return out.to(resolve_device(device))


@dataclass
class Segrev3Single:
    """A single (non-paired) dataset: inputs + their segrev3 outputs as 20-token
    sequences. Carries the ``n_threes`` it was generated with. Use ``.tokens`` for
    a forward pass (e.g. realized attention)."""
    tokens: Tensor                 # [B, SEQ_LEN]  == [input(10) | segrev3(input)(10)]
    inputs: Tensor                 # [B, INPUT_LEN]
    outputs: Tensor                # [B, OUTPUT_LEN]
    n_threes: int
    seed: Optional[int] = None


def build_single_dataset(inputs: Tensor, n_threes: int, seed: Optional[int]) -> dict:
    """JSON-serializable dict for a single dataset (only the input digits are
    stored; tokens/outputs are derived deterministically by ``load_single``)."""
    return {
        "kind": "single",
        "n_threes": int(n_threes),
        "seed": seed,
        "boundary_digit": BOUNDARY_DIGIT,
        "input_len": INPUT_LEN,
        "answer_positions": list(ANSWER_POSITIONS),
        "inputs": inputs.tolist(),
        # per-example 3-count; every row is built with exactly ``n_threes`` 3s, so
        # this is a constant list -- stored for parity with the paired datasets and
        # as an on-disk check that the construction held.
        "inputs_n_threes": count_threes(inputs),
    }


def single_dataset_to_obj(dataset: dict, device: Optional[str] = None) -> Segrev3Single:
    device = resolve_device(device)
    inputs = torch.tensor(dataset["inputs"], dtype=torch.long)
    tokens, outputs = make_tokens(inputs)
    return Segrev3Single(tokens.to(device), inputs.to(device), outputs.to(device),
                         int(dataset["n_threes"]), dataset.get("seed"))


def make_single_dataset(batch: int, n_threes: Optional[int] = None,
                        seed: Optional[int] = None, device: Optional[str] = None,
                        save_path: Optional[str] = None, inputs=None
                        ) -> Segrev3Single:
    """Build a single dataset of ``batch`` inputs each with exactly ``n_threes`` 3s
    (and optionally persist it to ``save_path`` as JSON).

    LITERAL OVERRIDE: ``inputs`` (a digit row, list of rows, or tensor) replaces
    generation (``batch`` is then ignored); the rows must all share one 3-count,
    which becomes -- and must match any given -- ``n_threes``."""
    if inputs is not None:
        inputs = _as_input_rows(inputs)
        counts = count_threes(inputs)
        if len(set(counts)) != 1:
            raise ValueError(f"literal inputs must share one 3-count, got {counts}")
        if n_threes is not None and counts[0] != n_threes:
            raise ValueError(f"literal inputs have {counts[0]} threes "
                             f"but n_threes={n_threes}")
        n_threes = counts[0]
    else:
        if n_threes is None:
            raise ValueError("pass n_threes (exact 3-count) or literal inputs")
        inputs = inputs_with_n_threes(batch, n_threes, seed=seed)
    dataset = build_single_dataset(inputs, n_threes, seed)
    if save_path is not None:
        save_dataset(dataset, save_path)
    return single_dataset_to_obj(dataset, device=device)


def load_single(path: str, device: Optional[str] = None) -> Segrev3Single:
    """Load a single (n_threes) dataset json saved by ``make_single_dataset``."""
    return single_dataset_to_obj(load_dataset(path), device=device)


def load_tokens_any(path: str, device: Optional[str] = None) -> Segrev3Single:
    """Load EITHER dataset kind as a ``Segrev3Single`` token view.

    Single (n_threes / ``edge``) datasets carry ``inputs``; paired
    (clean/corrupted) datasets carry ``clean_inputs``/``corrupted_inputs``. Some
    consumers (e.g. ``vis_edge``'s realized-attention pass) only need a forward
    pass over representative tokens; for a paired dataset that is the CLEAN side.
    ``n_threes`` is -1 for paired datasets (not a fixed-3s dataset)."""
    device = resolve_device(device)
    d = load_dataset(path)
    if "inputs" in d:
        return single_dataset_to_obj(d, device=device)
    if "clean_inputs" in d:
        inputs = torch.tensor(d["clean_inputs"], dtype=torch.long)
        tokens, outputs = make_tokens(inputs)
        return Segrev3Single(tokens.to(device), inputs.to(device), outputs.to(device),
                             n_threes=-1, seed=d.get("seed"))
    raise KeyError(f"{path}: dataset has neither 'inputs' nor 'clean_inputs'")


# --------------------------------------------------------------------------- #
# EXACT-3-COUNT pairs + literal digit overrides (single-sequence mode)         #
# --------------------------------------------------------------------------- #
# The single-sequence pipeline (manual_circuit.py --single) analyses ONE custom
# clean/corrupted pair. Unlike ``make_clean_corrupted`` -- whose clean side is
# rejection-sampled to have AT LEAST n eligible positions -- the clean side here
# has EXACTLY ``n_threes`` (k) 3s, so the clean 3-count k and the replaced-subset
# size n are INDEPENDENT knobs (n <= k for replace_threes_to_nonthrees, n <= INPUT_LEN - k for
# replace_non_threes_to_threes; violations raise instead of silently replacing fewer).
# Either side can also be given LITERALLY as digit rows, overriding generation.


def _as_input_rows(inputs) -> Tensor:
    """Normalize literal input digits -- a single row, a list of rows, or a
    tensor -- into a [B, INPUT_LEN] long tensor, validating length and range."""
    if hasattr(inputs, "tolist"):
        inputs = inputs.tolist()
    rows = list(inputs)
    if rows and not isinstance(rows[0], (list, tuple)):
        rows = [rows]
    t = torch.tensor(rows, dtype=torch.long)
    if t.ndim != 2 or t.shape[1] != INPUT_LEN:
        raise ValueError(f"each input row must have exactly {INPUT_LEN} digits "
                         f"(got shape {tuple(t.shape)})")
    if ((t < 0) | (t >= D_VOCAB)).any():
        raise ValueError(f"input digits must be in 0..{D_VOCAB - 1}: {rows}")
    return t


def make_pair_with_n_threes(
    batch: int = 1,
    n_threes: Optional[int] = None,
    scheme: str = "random",
    n: int = 1,
    seed: Optional[int] = None,
    device: Optional[str] = None,
    save_path: Optional[str] = None,
    clean_inputs=None,
    corrupted_inputs=None,
) -> Segrev3Pairs:
    """Clean/corrupted pair(s) whose CLEAN side has EXACTLY ``n_threes`` 3s
    (via ``inputs_with_n_threes``), under one of the three SCHEMES:

      * "random"             -- corrupted is an independent uniform-random row.
      * "replace_threes_to_nonthrees"     -- exactly ``n`` of the clean 3s -> random non-3
                                (requires n <= n_threes).
      * "replace_non_threes_to_threes" -- exactly ``n`` of the clean non-3s -> 3
                                (requires n <= INPUT_LEN - n_threes).

    LITERAL OVERRIDES: ``clean_inputs`` (a digit row, list of rows, or tensor)
    replaces generation of the clean side (``batch`` is then ignored; a given
    ``n_threes`` must match its 3-count); ``corrupted_inputs`` (requires
    ``clean_inputs``) replaces the scheme corruption and is stored verbatim.

    Stored via the same JSON format as ``make_clean_corrupted`` (``save_path``),
    so ``load_pairs`` / every pairs consumer works unchanged.
    """
    if scheme not in SCHEMES:
        raise ValueError(f"unknown scheme {scheme!r}; use one of {SCHEMES}")
    if corrupted_inputs is not None and clean_inputs is None:
        raise ValueError("corrupted_inputs given without clean_inputs -- the pair "
                         "is anchored on the clean side; pass both (or neither)")
    if clean_inputs is not None:
        clean = _as_input_rows(clean_inputs)
        counts = count_threes(clean)
        if n_threes is not None and any(c != n_threes for c in counts):
            raise ValueError(f"literal clean_inputs have {counts} threes "
                             f"but n_threes={n_threes}")
    else:
        if n_threes is None:
            raise ValueError("pass n_threes (exact clean 3-count) or literal clean_inputs")
        # build on CPU (like make_clean_corrupted); dataset_to_pairs moves to device
        clean = inputs_with_n_threes(batch, n_threes, seed=seed, device="cpu")
        counts = [n_threes] * batch

    gen = _gen(seed)
    B = clean.shape[0]
    if corrupted_inputs is not None:
        corrupted = _as_input_rows(corrupted_inputs)
        if corrupted.shape != clean.shape:
            raise ValueError(f"corrupted_inputs shape {tuple(corrupted.shape)} != "
                             f"clean_inputs shape {tuple(clean.shape)}")
    elif scheme == "random":
        corrupted = torch.randint(0, D_VOCAB, (B, INPUT_LEN), generator=gen)
    elif scheme == "replace_threes_to_nonthrees":
        if n > min(counts):
            raise ValueError(f"replace_threes_to_nonthrees needs n <= clean 3-count "
                             f"(n={n}, 3-count={min(counts)})")
        corrupted, _ = _corrupt(clean, n, want_threes=True, gen=gen)
    else:  # replace_non_threes_to_threes
        if n > INPUT_LEN - max(counts):
            raise ValueError(f"replace_non_threes_to_threes needs n <= non-3 count "
                             f"(n={n}, non-3 count={INPUT_LEN - max(counts)})")
        corrupted, _ = _corrupt(clean, n, want_threes=False, gen=gen)

    dataset = build_dataset(scheme, clean, corrupted, n, seed)
    if save_path is not None:
        save_dataset(dataset, save_path)
    return dataset_to_pairs(dataset, device=device)


# --------------------------------------------------------------------------- #
# Smoke test (generation + JSON storage + load roundtrip)                      #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import tempfile
    OUT = tempfile.mkdtemp(prefix="segrev3_data_gen_")   # scratch: never the datasets/ store

    specs = [
        ("random", dict(seed=0)),
        ("replace_threes_to_nonthrees", dict(seed=1, n=1)),
        ("replace_non_threes_to_threes", dict(seed=2, n=1)),
    ]
    for scheme, kw in specs:
        path = os.path.join(OUT, f"{scheme}.json")
        pairs = make_clean_corrupted(64, scheme=scheme, save_path=path, **kw)
        # JSON round-trips to identical tensors
        rt = load_pairs(path)
        assert torch.equal(rt.clean_tokens, pairs.clean_tokens)
        assert torch.equal(rt.corrupted_tokens, pairs.corrupted_tokens)
        # shapes + metric spec
        assert pairs.clean_tokens.shape == pairs.corrupted_tokens.shape
        assert pairs.answer_positions.shape[1] == len(ANSWER_POSITIONS)
        # every example is genuinely corrupted (no null pairs)
        assert (pairs.clean_inputs != pairs.corrupted_inputs).any(dim=1).all()

    # the corruptions touch the right kind of position
    p2 = make_clean_corrupted(128, "replace_threes_to_nonthrees", seed=5, n=1)
    changed2 = p2.clean_inputs != p2.corrupted_inputs
    assert (p2.clean_inputs[changed2] == BOUNDARY_DIGIT).all()       # only 3s changed
    assert (p2.corrupted_inputs[changed2] != BOUNDARY_DIGIT).all()   # -> non-3

    p3 = make_clean_corrupted(128, "replace_non_threes_to_threes", seed=6, n=2)
    changed3 = p3.clean_inputs != p3.corrupted_inputs
    assert (p3.clean_inputs[changed3] != BOUNDARY_DIGIT).all()       # only non-3s changed
    assert (p3.corrupted_inputs[changed3] == BOUNDARY_DIGIT).all()   # -> 3
    assert int(changed3.sum(dim=1).max()) <= 2                       # at most n per row

    # single-position contrast is well-formed
    pos1, cor1, inc1 = single_position_contrast(p2)
    assert pos1.shape == cor1.shape == inc1.shape == (128,)

    # exact-3-count pairs (single-sequence mode): k and n are independent knobs
    p4 = make_pair_with_n_threes(1, n_threes=4, scheme="replace_threes_to_nonthrees", n=2, seed=7)
    assert count_threes(p4.clean_inputs) == [4]
    assert int((p4.clean_inputs != p4.corrupted_inputs).sum()) == 2
    # literal clean override: n_threes derived from the row, exactly n replaced
    lit = make_pair_with_n_threes(scheme="replace_non_threes_to_threes", n=1, seed=8,
                                  clean_inputs=[5, 3, 1, 2, 3, 9, 8, 3, 0, 6])
    assert count_threes(lit.clean_inputs) == [3]
    assert count_threes(lit.corrupted_inputs) == [4]
    # literal pair override is stored verbatim
    both = make_pair_with_n_threes(scheme="replace_threes_to_nonthrees", n=1,
                                   clean_inputs=[5, 3, 1, 2, 3, 9, 8, 3, 0, 6],
                                   corrupted_inputs=[5, 7, 1, 2, 3, 9, 8, 3, 0, 6])
    assert count_threes(both.corrupted_inputs) == [2]
    # constraint violations raise instead of silently replacing fewer
    for bad_kw in (dict(n_threes=0, scheme="replace_threes_to_nonthrees", n=1),
                   dict(n_threes=10, scheme="replace_non_threes_to_threes", n=1)):
        try:
            make_pair_with_n_threes(1, seed=9, **bad_kw)
            raise AssertionError(f"expected ValueError for {bad_kw}")
        except ValueError:
            pass
    # single dataset literal override
    s1 = make_single_dataset(1, inputs=[5, 3, 1, 2, 3, 9, 8, 3, 0, 6])
    assert s1.n_threes == 3 and s1.inputs.shape == (1, INPUT_LEN)

    print("generation + JSON storage + roundtrip checks PASSED")
    print(f"wrote datasets to {OUT}/ : " + ", ".join(s for s, _ in specs))
