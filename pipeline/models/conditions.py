#!/usr/bin/env python3
"""conditions.py — isolation conditions: each a (predicate, sampler) pair over the input domain.

A `Condition` bundles `holds(x)->bool` (the DomainCircuit.condition predicate) with `sample(batch,seed)`
(the matching uniform draw over the domain, used by DomainCircuit.sample and gen_circuit_data). The two
are kept together so a circuit's predicate and its sampler can never drift apart. The low-level
sampler factories below are the single home for the pipeline's samplers (gen_circuit_data imports
them from here)."""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from itertools import combinations, product
from typing import Callable, Optional

import torch

INPUT_LEN = 10

def stream_key(seed: int, tag: str) -> int:
    """Per-sampler seed from ``(seed, tag)``: sibling conditions of one model must draw INDEPENDENT
    streams, else all_even is literally 2*(full % 5) row-for-row. blake2b, not the salted builtin
    ``hash``, so a pool regenerated in another process is bit-identical."""
    return int.from_bytes(hashlib.blake2b(f"{int(seed)}|{tag}".encode(), digest_size=7).digest(), "big")


def _uniform_digits(batch: int, allowed, seed: int, tag: str,
                    length: int = INPUT_LEN) -> "torch.LongTensor":
    """``[batch, length]`` long tensor, each digit drawn uniformly from ``allowed``."""
    allowed = sorted({int(d) for d in allowed})
    if not allowed:
        raise ValueError("allowed digit set must be non-empty")
    if length < 1:
        raise ValueError(f"length must be >= 1 (got {length})")
    g = torch.Generator().manual_seed(stream_key(seed, tag))
    pal = torch.tensor(allowed, dtype=torch.long)
    return pal[torch.randint(0, pal.numel(), (batch, int(length)), generator=g)]


def domain_sampler(allowed, length: int = INPUT_LEN) -> Callable[[int, int], "torch.LongTensor"]:
    """Sampler drawing every digit uniformly from the digit subset ``allowed``."""
    allowed = sorted({int(d) for d in allowed})
    tag = f"domain:{allowed}:{length}"
    return lambda batch, seed: _uniform_digits(batch, allowed, seed, tag, length)


def full_domain_sampler(length: int = INPUT_LEN) -> Callable[[int, int], "torch.LongTensor"]:
    """Sampler over the full digit domain 0-9 (the whole-model circuit)."""
    return domain_sampler(range(10), length)


def bounded_sum_sampler(max_sum: int, length: int = INPUT_LEN,
                        lo: int = 0, hi: int = 9) -> Callable[[int, int], "torch.LongTensor"]:
    """Sampler drawing length-``length`` digit sequences (each digit in ``[lo, hi]``)
    whose TOTAL sum is ``<= max_sum``, UNIFORMLY over that set.

    Use this to isolate a threshold/accumulator sub-circuit: keeping the running sum
    below a reset threshold means the reset branch never fires, so the model reduces
    to a plain prefix-sum. Sampling is EXACT-UNIFORM via a stars-and-bars DP and is
    deliberately NOT rejection sampling — for sum<=12 over 10 digits the accept rate
    is ~0.017%, far too slow. ``f[k][b]`` counts length-k sequences (digits in
    ``[0, hi-lo]``) summing ``<= b``; inputs are drawn by inverse-CDF on those counts.
    """
    if not (0 <= lo <= hi):
        raise ValueError(f"require 0 <= lo <= hi (got lo={lo}, hi={hi})")
    span = hi - lo
    B = max_sum - lo * length              # budget in shifted digits d' = d - lo in [0, span]
    if B < 0:
        raise ValueError(f"no sequences: length={length} min-sum={lo * length} > max_sum={max_sum}")
    # f[k][b] = #{ x in [0,span]^k : sum(x) <= b }
    f = [[0] * (B + 1) for _ in range(length + 1)]
    f[0] = [1] * (B + 1)
    for k in range(1, length + 1):
        prev, cur = f[k - 1], f[k]
        for b in range(B + 1):
            s = 0
            for c in range(min(span, b) + 1):
                s += prev[b - c]
            cur[b] = s
    tag = f"sum_le:{max_sum}:{length}:{lo}:{hi}"

    def _sample(batch: int, seed: int) -> "torch.LongTensor":
        g = torch.Generator().manual_seed(stream_key(seed, tag))
        u = torch.rand((batch, length), generator=g).tolist()
        rows = []
        for r in range(batch):
            budget, ur, row = B, u[r], []
            for i in range(length):
                k_rest = length - i - 1                  # positions after this one
                target = ur[i] * f[length - i][budget]   # f[length-i][budget] = total completions
                acc, c, cmax = 0, 0, min(span, budget)
                while c < cmax:
                    acc += f[k_rest][budget - c]
                    if target < acc:
                        break
                    c += 1
                row.append(c + lo)
                budget -= c
            rows.append(row)
        return torch.tensor(rows, dtype=torch.long)

    return _sample


def bounded_count_sampler(digit: int, max_count: int, length: int = INPUT_LEN,
                          domain=range(10)) -> Callable[[int, int], "torch.LongTensor"]:
    """Sampler drawing length-``length`` sequences (each value in ``domain``) in which
    the target ``digit`` appears AT MOST ``max_count`` times, UNIFORMLY over that set.

    Use this to isolate an occurrence-count sub-circuit: capping the number of copies
    of ``digit`` means a multi-step counter/ladder keyed on that digit can never step
    past ``max_count``, so the model collapses to the corresponding short behaviour
    (e.g. max_count=1 → detect-the-one-digit + prefix-OR). "At most k occurrences of D"
    is a GLOBAL condition that couples positions, so it cannot be done with a
    per-position ``domain_sampler``; this draws EXACT-UNIFORM and is deliberately NOT
    rejection sampling. It first picks the occurrence count ``c`` in ``[0, max_count]``
    with weight ``C(length, c) * m**(length-c)`` (``m`` = #non-target values), then
    places the ``c`` copies of ``digit`` on a uniform random subset of positions and
    fills the rest uniformly from the non-target values.
    """
    digit = int(digit)
    dom = sorted({int(d) for d in domain})
    if digit not in dom:
        raise ValueError(f"digit {digit} not in domain {dom}")
    if max_count < 0:
        raise ValueError(f"max_count must be >= 0 (got {max_count})")
    others = [d for d in dom if d != digit]
    m = len(others)                                   # number of non-target values
    cap = min(int(max_count), length)
    # weight[c] = #{sequences with EXACTLY c copies of digit} = C(length, c) * m**(length-c)
    weights = [math.comb(length, c) * (m ** (length - c)) for c in range(cap + 1)]
    total = sum(weights)
    if total == 0:
        raise ValueError("no sequences satisfy the constraint")
    tag = f"count:{digit}<={cap}:{length}:{dom}"

    def _sample(batch: int, seed: int) -> "torch.LongTensor":
        g = torch.Generator().manual_seed(stream_key(seed, tag))
        u = torch.rand((batch,), generator=g).tolist()   # one draw per row → pick count c
        rows = []
        for r in range(batch):
            target, acc, c = u[r] * total, 0, cap
            for cc, w in enumerate(weights):
                acc += w
                if target < acc:
                    c = cc
                    break
            perm = torch.randperm(length, generator=g)    # which positions hold `digit`
            digit_pos = set(perm[:c].tolist())
            fill = (torch.randint(0, m, (length,), generator=g).tolist()
                    if m > 0 else [0] * length)
            row = [digit if i in digit_pos else others[fill[i]] for i in range(length)]
            rows.append(row)
        return torch.tensor(rows, dtype=torch.long)

    return _sample


def one_odd_sampler(length: int = INPUT_LEN) -> Callable[[int, int], "torch.LongTensor"]:
    """Sampler drawing length-``length`` sequences with EXACTLY ONE odd-valued digit,
    UNIFORMLY over that set.

    Use this to isolate the 'localize + transform the lone odd digit' sub-circuit: with
    a single odd digit the parity-classify gate selects the doubling branch at exactly
    one position and the identity/passthrough branch everywhere else. "Exactly one odd
    digit" is a GLOBAL parity-count condition that couples positions, so it cannot be
    done with a per-position ``domain_sampler``; this draws EXACT-UNIFORM and is
    deliberately NOT rejection sampling. Construction: fill the row with uniform even
    digits {0,2,4,6,8}, then overwrite one uniformly-chosen position with a uniform odd
    digit {1,3,5,7,9}. Each sequence with one odd digit at position p is produced with
    probability (1/length)*(1/5)^length, i.e. uniform over the length*5^length such
    sequences.
    """
    ODD = torch.tensor([1, 3, 5, 7, 9], dtype=torch.long)
    EVEN = torch.tensor([0, 2, 4, 6, 8], dtype=torch.long)
    tag = f"one_odd:{length}"

    def _sample(batch: int, seed: int) -> "torch.LongTensor":
        g = torch.Generator().manual_seed(stream_key(seed, tag))
        out = EVEN[torch.randint(0, EVEN.numel(), (batch, length), generator=g)]
        pos = torch.randint(0, length, (batch,), generator=g)
        odd_vals = ODD[torch.randint(0, ODD.numel(), (batch,), generator=g)]
        out[torch.arange(batch), pos] = odd_vals
        return out

    return _sample


def position_domain_sampler(constraints: dict,
                            length: int = INPUT_LEN) -> Callable[[int, int], "torch.LongTensor"]:
    """Uniform digits 0-9, but each position listed in ``constraints`` (a
    ``{position: allowed_digits}`` dict) is drawn only from its allowed set.

    Use to PIN an anchor/address position to a constant (collapsing a data-dependent
    op -- rotate/reflect/clamp/gather/Caesar-shift, or a length read from a position --
    to a FIXED operation), or to restrict a single position (e.g. the last digit < 9 so
    a ripple-carry add never overflows and stays a purely local increment)."""
    cons = {int(p): sorted({int(d) for d in dd}) for p, dd in constraints.items()}
    for p, allowed in cons.items():
        if not allowed:
            raise ValueError(f"position {p} has an empty allowed set")
        if not 0 <= p < length:
            raise ValueError(f"constrained position {p} outside a length-{length} row")
    tag = f"posdomain:{sorted(cons.items())}:{length}"

    def _sample(batch: int, seed: int) -> "torch.LongTensor":
        out = _uniform_digits(batch, range(10), seed, tag, length)
        g = torch.Generator().manual_seed(stream_key(seed, tag + ":pins"))
        for p, allowed in cons.items():
            pal = torch.tensor(allowed, dtype=torch.long)
            out[:, p] = pal[torch.randint(0, pal.numel(), (batch,), generator=g)]
        return out

    return _sample


def one_in_set_sampler(trigger, length: int = INPUT_LEN) -> Callable[[int, int], "torch.LongTensor"]:
    """Exactly ONE digit drawn from ``trigger``, every other position from the
    complement (0-9 minus ``trigger``), UNIFORMLY. Generalizes ``one_odd_sampler`` to
    any trigger set -- isolates a 'localize + transform/detect the single triggering
    digit' sub-circuit (e.g. exactly one digit != 1 for a running product; exactly one
    digit >= 5 for a threshold sample-and-hold latch)."""
    trig = sorted({int(d) for d in trigger})
    comp = [d for d in range(10) if d not in set(trig)]
    if not trig or not comp:
        raise ValueError("trigger and its complement must both be non-empty")
    TRIG = torch.tensor(trig, dtype=torch.long)
    COMP = torch.tensor(comp, dtype=torch.long)
    tag = f"one_in:{trig}:{length}"

    def _sample(batch: int, seed: int) -> "torch.LongTensor":
        g = torch.Generator().manual_seed(stream_key(seed, tag))
        out = COMP[torch.randint(0, COMP.numel(), (batch, length), generator=g)]
        pos = torch.randint(0, length, (batch,), generator=g)
        out[torch.arange(batch), pos] = TRIG[torch.randint(0, TRIG.numel(), (batch,), generator=g)]
        return out

    return _sample


def one_repeat_sampler(length: int = INPUT_LEN) -> Callable[[int, int], "torch.LongTensor"]:
    """Exactly ONE repeated value: a single value appears twice, every other value is
    distinct (needs length <= 10). Isolates a novelty / seen-before detector to its
    single repeat (one 0 in the 0/1 output marks the lone non-novel position)."""
    if length > 10:
        raise ValueError("one_repeat_sampler needs length <= 10")
    tag = f"one_repeat:{length}"

    def _sample(batch: int, seed: int) -> "torch.LongTensor":
        g = torch.Generator().manual_seed(stream_key(seed, tag))
        rows = []
        for _ in range(batch):
            row = torch.randperm(10, generator=g)[:length].tolist()    # distinct digits
            i, j = torch.randperm(length, generator=g)[:2].tolist()
            row[j] = row[i]                                            # duplicate i's value into j
            rows.append(row)
        return torch.tensor(rows, dtype=torch.long)

    return _sample


def one_fixedpoint_sampler(length: int = INPUT_LEN) -> Callable[[int, int], "torch.LongTensor"]:
    """Exactly ONE position with x[i]==i. Every position is filled with a digit != its
    index, then one chosen position p is set to p. Isolates a value==index detector to
    its single hit (one 1 in the 0/1 output)."""
    tag = f"one_fixedpoint:{length}"

    def _sample(batch: int, seed: int) -> "torch.LongTensor":
        g = torch.Generator().manual_seed(stream_key(seed, tag))
        rows = []
        for _ in range(batch):
            row = []
            for i in range(length):
                d = torch.randint(0, 9, (1,), generator=g).item()      # 0..8
                if d >= i:
                    d += 1                                             # skip i -> digit != i
                row.append(d)
            p = torch.randint(0, length, (1,), generator=g).item()
            row[p] = p                                                 # the single fixed point
            rows.append(row)
        return torch.tensor(rows, dtype=torch.long)

    return _sample


def monotone_sampler(increasing: bool = False,
                     length: int = INPUT_LEN) -> Callable[[int, int], "torch.LongTensor"]:
    """Monotone (sorted) inputs: non-decreasing if ``increasing`` else non-increasing, drawn
    EXACT-UNIFORM over that set. Forces a sliding-window extremum to the window's edge,
    collapsing the model to a pure temporal shift (a delay-line).

    Sorting iid digits is NOT uniform: it weights each sorted row by its multinomial
    coefficient, so over 10 digits the near-constant rows -- the ones where the delay-line
    collapse is most visible -- come out up to 10! = 3.6e6 times too rare. Instead sample the
    GAP vector: a non-decreasing row in [0,9] is the cumulative sum of its gaps, and the legal
    gap vectors are exactly ``{g in [0,9]^length : sum(g) <= 9}``, so ``bounded_sum_sampler``
    draws them uniformly by inverse-CDF and the bijection carries uniformity to the rows.
    """
    gaps = bounded_sum_sampler(9, length, 0, 9)
    tag = f"monotone:{'inc' if increasing else 'dec'}:{length}"

    def _sample(batch: int, seed: int) -> "torch.LongTensor":
        rows = gaps(batch, stream_key(seed, tag)).cumsum(dim=1)
        return rows if increasing else rows.flip(1)

    return _sample


def compare_positions_sampler(i: int, j: int,
                              length: int = INPUT_LEN) -> Callable[[int, int], "torch.LongTensor"]:
    """Uniform digits but with x[i] > x[j], drawn UNIFORMLY over the 45 legal ordered pairs.
    Forces a compare-and-swap comparator's branch so the conditional swap becomes an
    UNCONDITIONAL positional swap of i and j.

    Drawing x[j] ~ U{0..8} then x[i] ~ U{x[j]+1..9} is NOT uniform: it flattens x[j]'s
    marginal, which must be (9-k)/45, and so over-weights the tightest comparison x[j]=8 by
    5x. Draw the pair jointly instead.
    """
    i, j = int(i), int(j)
    if i == j or not (0 <= i < length and 0 <= j < length):
        raise ValueError(f"need distinct positions inside a length-{length} row (got i={i}, j={j})")
    pairs = torch.tensor([(a, b) for b in range(9) for a in range(b + 1, 10)], dtype=torch.long)
    tag = f"cmp:{i}>{j}:{length}"

    def _sample(batch: int, seed: int) -> "torch.LongTensor":
        out = _uniform_digits(batch, range(10), seed, tag, length)
        g = torch.Generator().manual_seed(stream_key(seed, tag + ":pair"))
        p = pairs[torch.randint(0, pairs.shape[0], (batch,), generator=g)]
        out[:, i], out[:, j] = p[:, 0], p[:, 1]
        return out

    return _sample


def mux_sampler(n_tasks: int, length: int = INPUT_LEN + 1) -> Callable[[int, int], "torch.LongTensor"]:
    """Sampler for a TaskMux model: ``[batch, length]`` rows ``[task_token, payload...]``,
    task token uniform in ``[10, 10+n_tasks)``, payload uniform digits 0-9. The payload
    is drawn before the task token, so for a fixed seed all tasks (and every
    ``task_pin`` variant) share identical payloads."""
    if n_tasks < 2:
        raise ValueError(f"n_tasks must be >= 2 (got {n_tasks})")

    def _sample(batch: int, seed: int) -> "torch.LongTensor":
        g = torch.Generator().manual_seed(int(seed))   # bare seed, NOT _key: task_pin siblings must share payloads
        out = torch.randint(0, 10, (batch, length), generator=g)
        out[:, 0] = 10 + torch.randint(0, int(n_tasks), (batch,), generator=g)
        return out

    return _sample

# --------------------------------------------------------------------------- #
# Condition: predicate + matching sampler (the reusable isolation-lever library) #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Condition:
    """An isolation condition: `holds(x)->bool` plus a uniform `sample(batch,seed)` over its domain."""
    name: str
    holds: Callable[["torch.Tensor"], bool]
    sample: Callable[[int, int], "torch.LongTensor"]


def domain(digits, name: Optional[str] = None) -> Condition:
    pal = torch.tensor(sorted({int(d) for d in digits}))
    nm = name or f"domain_{'_'.join(map(str, pal.tolist()))}"
    return Condition(nm, lambda x, _p=pal: bool(torch.isin(x, _p).all()), domain_sampler(pal.tolist()))

FULL = Condition("full", lambda x: True, full_domain_sampler())   # no input restriction (output-subset circuits)
ALL_EVEN = Condition("all_even", lambda x: bool((x % 2 == 0).all()), domain_sampler([0, 2, 4, 6, 8]))
ALL_ODD = Condition("all_odd", lambda x: bool((x % 2 == 1).all()), domain_sampler([1, 3, 5, 7, 9]))
BINARY = Condition("binary", lambda x: bool(((x == 0) | (x == 1)).all()), domain_sampler([0, 1]))

def exclude_digit(d: int) -> Condition:
    d = int(d)
    return domain([k for k in range(10) if k != d], name=f"exclude_{d}")

def pin(pos: int, vals) -> Condition:
    pos = int(pos); s = {int(v) for v in vals}
    return Condition(f"pin_{pos}_{sorted(s)}", lambda x, _p=pos, _s=s: int(x[_p]) in _s,
                     position_domain_sampler({pos: sorted(s)}))

def exactly_one_in(trigger) -> Condition:
    trig = torch.tensor(sorted({int(d) for d in trigger}))
    return Condition(f"exactly_one_in_{trig.tolist()}",
                     lambda x, _t=trig: int(torch.isin(x, _t).sum()) == 1, one_in_set_sampler(trig.tolist()))

ONE_ODD = Condition("one_odd", lambda x: int((x % 2 == 1).sum()) == 1, one_odd_sampler())

def at_most_k_of(d: int, k: int = 1) -> Condition:
    d, k = int(d), int(k)
    return Condition(f"at_most_{k}_of_{d}", lambda x, _d=d, _k=k: int((x == _d).sum()) <= _k,
                     bounded_count_sampler(d, k))

ONE_REPEAT = Condition("one_repeat", lambda x: (len(x) - len(set(x.tolist()))) == 1, one_repeat_sampler())
ONE_FIXEDPOINT = Condition("one_fixedpoint",
                           lambda x: int((x == torch.arange(len(x))).sum()) == 1, one_fixedpoint_sampler())

def sum_at_most(n: int) -> Condition:
    return Condition(f"sum_le_{n}", lambda x, _n=int(n): int(x.sum()) <= _n, bounded_sum_sampler(int(n)))

def monotone(increasing: bool = False) -> Condition:
    pred = (lambda x: bool((x[1:] >= x[:-1]).all())) if increasing else (lambda x: bool((x[1:] <= x[:-1]).all()))
    return Condition(f"monotone_{'inc' if increasing else 'dec'}", pred, monotone_sampler(increasing))

def compare_positions(i: int, j: int) -> Condition:
    i, j = int(i), int(j)
    return Condition(f"cmp_{i}_gt_{j}", lambda x, _i=i, _j=j: int(x[_i]) > int(x[_j]), compare_positions_sampler(i, j))

def task_pin(i: int, n_tasks: int) -> Condition:
    """Per-task C0 condition on a TaskMux model: token 0 pinned to task i's token (10+i)."""
    i, n_tasks = int(i), int(n_tasks)
    if not 0 <= i < n_tasks:
        raise ValueError(f"task index {i} out of range [0, {n_tasks})")
    base = mux_sampler(n_tasks)

    def _sample(batch: int, seed: int) -> "torch.LongTensor":
        out = base(batch, seed)
        out[:, 0] = 10 + i
        return out

    return Condition(f"task{i}_of{n_tasks}", lambda x, _t=10 + i: int(x[0]) == _t, _sample)


# ---------------------------------------------------------------------------- #
# W22/W23 integration (2026-08-03): conditions for the 17 brainstormed operators.
# Every sampler here is EXACT-UNIFORM over its domain and was proven so before merge
# (chi-square/TVD against the enumerated domain, or an analytic path-measure check).
# The ledger's monotone_sampler shipped at TVD 0.4723 by sorting iid digits;
# heap_ordered is the same trap and is DP-sampled, not heapified.
# ---------------------------------------------------------------------------- #


# --- cfreq_dsub ---
"""New models/conditions.py entries for the cfreq_id / dsub_id integration.

Paste the four sampler factories in with the other sampler factories (above the
``Condition`` dataclass) and the three constants + one factory with the other Conditions.
Every sampler here is EXACT-UNIFORM by construction -- none rejection-samples, so none
carries an accept rate. The evidence is a full-support chi-square of the sampler's own
draws against the brute-force-enumerated domain, run at a reduced length where the domain
is enumerable and at length 10 where it is small enough (two_values); each Condition's
``name`` is the taxonomy family key, so it must stay as written.

Needs nothing new imported: ``math``, ``torch``, ``INPUT_LEN``, ``stream_key`` and
``Condition`` are already module-level in conditions.py.
"""

def two_values_sampler(length: int = INPUT_LEN) -> Callable[[int, int], "torch.LongTensor"]:
    """Exactly TWO distinct digits present with UNEQUAL counts, drawn EXACT-UNIFORM.

    Isolates a cross-value ranking to ONE binary comparison: with a single majority
    value every "is your count beaten" test has the same answer, so the op collapses to
    majority detection plus a count broadcast. A row is a digit pair {a<b} times the mask
    of positions holding ``a``, and that map is a bijection, so drawing the pair (45) and
    the mask uniformly draws the row uniformly. The masks are enumerated, not rejected:
    all-``a``/all-``b`` are dropped (only one value present) and the even split is dropped
    (no majority) -- 45*770 = 34650 rows at length 10.
    """
    pairs = torch.tensor([(a, b) for a in range(10) for b in range(a + 1, 10)], dtype=torch.long)
    keep = [m for m in range(1, 2 ** length - 1) if 2 * bin(m).count("1") != length]
    masks = torch.tensor([[(m >> i) & 1 for i in range(length)] for m in keep], dtype=torch.bool)
    tag = f"two_values:{length}"

    def _sample(batch: int, seed: int) -> "torch.LongTensor":
        g = torch.Generator().manual_seed(stream_key(seed, tag))
        p = pairs[torch.randint(0, pairs.shape[0], (batch,), generator=g)]
        m = masks[torch.randint(0, masks.shape[0], (batch,), generator=g)]
        return torch.where(m, p[:, :1], p[:, 1:])

    return _sample

def count_profile_sampler(counts, length: int = INPUT_LEN) -> Callable[[int, int], "torch.LongTensor"]:
    """Digits whose occurrence counts are exactly the multiset ``counts``, EXACT-UNIFORM.

    Pins the whole histogram, so any statistic OF the counts is a constant and a
    count-comparison stage becomes a fixed lookup table on the digit's own count.
    A row is an ordered choice of ``len(counts)`` distinct digits (one per count class)
    times an arrangement of the resulting multiset; a uniform digit permutation gives the
    first and a uniform position permutation the second, each row arising equally often.
    """
    counts = [int(c) for c in counts]
    if sum(counts) != length:
        raise ValueError(f"counts {counts} sum to {sum(counts)}, not length {length}")
    if len(counts) > 10 or min(counts) < 1:
        raise ValueError(f"need 1..10 positive counts (got {counts})")
    klass = torch.repeat_interleave(torch.arange(len(counts)), torch.tensor(counts))
    tag = f"count_profile:{counts}:{length}"

    def _sample(batch: int, seed: int) -> "torch.LongTensor":
        g = torch.Generator().manual_seed(stream_key(seed, tag))
        digits = torch.argsort(torch.rand(batch, 10, generator=g), dim=1)[:, :len(counts)]
        order = torch.argsort(torch.rand(batch, length, generator=g), dim=1)
        return torch.empty(batch, length, dtype=torch.long).scatter_(1, order, digits[:, klass])

    return _sample

def repeats_adjacent_sampler(length: int = INPUT_LEN) -> Callable[[int, int], "torch.LongTensor"]:
    """Every value's occurrences are CONTIGUOUS -- a repeat's previous occurrence is at
    i-1 -- drawn EXACT-UNIFORM.

    Pins a data-dependent "where did this value last occur" retrieval to a FIXED 1-back
    read, collapsing a value-addressed recurrence to a plain second-order one. Such a row
    is a composition of ``length`` into k runs times an ordered choice of k distinct
    digits, so pick k with weight C(length-1, k-1) * P(10, k), then the cut set and the
    digits uniformly; every row is then hit with probability 1/total.
    """
    kmax = min(length, 10)
    weights = [math.comb(length - 1, k - 1) * math.perm(10, k) for k in range(1, kmax + 1)]
    total = sum(weights)
    tag = f"repeats_adjacent:{length}"

    def _sample(batch: int, seed: int) -> "torch.LongTensor":
        g = torch.Generator().manual_seed(stream_key(seed, tag))
        u = torch.rand((batch,), generator=g).tolist()
        rows = []
        for r in range(batch):
            target, acc, k = u[r] * total, 0, kmax
            for kk, w in enumerate(weights, start=1):
                acc += w
                if target < acc:
                    k = kk
                    break
            cuts = sorted(torch.randperm(length - 1, generator=g)[:k - 1].tolist())
            digits = torch.randperm(10, generator=g)[:k].tolist()
            bounds = [0, *(c + 1 for c in cuts), length]
            rows.append([digits[j] for j in range(k) for _ in range(bounds[j + 1] - bounds[j])])
        return torch.tensor(rows, dtype=torch.long)

    return _sample

def no_adjacent_repeat_sampler(length: int = INPUT_LEN) -> Callable[[int, int], "torch.LongTensor"]:
    """No two ADJACENT digits equal, EXACT-UNIFORM.

    The matched control for ``repeats_adjacent_sampler``: it forbids exactly the 1-back
    hop the other forces, so a value-addressed retrieval has to survive. Each digit is its
    predecessor plus a nonzero mod-10 step, which is uniform over the 9 digits that differ
    from it, so all 10*9**(length-1) rows are equally likely -- a rejection filter over
    uniform rows would reach the same set at a 0.9**(length-1) accept rate.
    """
    tag = f"no_adjacent_repeat:{length}"

    def _sample(batch: int, seed: int) -> "torch.LongTensor":
        g = torch.Generator().manual_seed(stream_key(seed, tag))
        start = torch.randint(0, 10, (batch, 1), generator=g)
        steps = torch.randint(1, 10, (batch, length - 1), generator=g).cumsum(1)
        return torch.cat([torch.zeros(batch, 1, dtype=torch.long), steps], 1).add(start) % 10

    return _sample

def _two_values(x: "torch.Tensor") -> bool:
    c = torch.unique(x, return_counts=True)[1]
    return c.numel() == 2 and int(c[0]) != int(c[1])

def _repeats_adjacent(x: "torch.Tensor") -> bool:
    runs = [v for i, v in enumerate(x.tolist()) if i == 0 or v != x[i - 1]]
    return len(runs) == len(set(runs))

TWO_VALUES = Condition("two_values", _two_values, two_values_sampler())

REPEATS_ADJACENT = Condition("repeats_adjacent", _repeats_adjacent, repeats_adjacent_sampler())

NO_ADJACENT_REPEAT = Condition("no_adjacent_repeat", lambda x: bool((x[1:] != x[:-1]).all()),
                               no_adjacent_repeat_sampler())

def count_profile(counts) -> Condition:
    """Occurrence counts pinned to the multiset ``counts`` (e.g. [4,3,2,1] -> 'counts_4321')."""
    want = tuple(sorted((int(c) for c in counts), reverse=True))
    return Condition("counts_" + "".join(map(str, counts)),
                     lambda x, _w=want: tuple(sorted(torch.unique(x, return_counts=True)[1].tolist(),
                                                     reverse=True)) == _w,
                     count_profile_sampler(counts))


# --- codonskip_fraglen ---
"""New models/conditions.py pieces for codonskip_id + fraglen_id (Wave 22, comp-bio).

Paste BLOCK 1 into the low-level sampler-factory section (beside bounded_count_sampler /
one_in_set_sampler) and BLOCK 2 into the Condition section (beside ONE_ODD / BINARY).
Nothing else in conditions.py changes; both blocks use only names the file already has.

`leading_gaps` is deliberately absent: see the drop note in BLOCK 2.
"""

def inframe_gaps_sampler(length: int = INPUT_LEN) -> Callable[[int, int], "torch.LongTensor"]:
    """Sampler drawing length-``length`` sequences that hold AT LEAST ONE gap (digit 0) and in
    which every maximal gap run has length divisible by 3, UNIFORMLY over that set.

    Use this to isolate a content-gated modular register to its undisturbed frame: three skipped
    positions advance a mod-3 counter by a full turn, so the frame never drifts off the plain
    positional stencil. The run-length rule couples positions, so no per-position
    ``domain_sampler`` expresses it, and rejection is hopeless (a uniform row lands in the set
    1.1% of the time). Counted by a DP over (position, gap-run length mod 3, gap seen) and drawn
    by inverse-CDF on those counts: the walk reconstructs each row by exactly one path, so its
    product of branch probabilities telescopes to 1/#domain.

    The at-least-one-gap clause is what keeps the circuit traceable. Without it the set is
    98.9% gap-free rows, whose output is the single constant row the dropped ``exclude_0``
    isolation was killed for (measured: modal output share 0.9871 of a 4096-row pool, versus
    0.1350 with the clause).
    """
    if length < 3:
        raise ValueError(f"need length >= 3 to hold one in-frame gap run (got {length})")
    # f[i][r][h] = #completions of positions i.. given gap-run residue r, h = a gap was seen
    f = [[[0, 0], [0, 0], [0, 0]] for _ in range(length + 1)]
    f[length][0][1] = 1
    for i in range(length - 1, -1, -1):
        for r in range(3):
            for h in range(2):
                closed = 9 * f[i + 1][0][h] if r == 0 else 0     # a base closes the run
                f[i][r][h] = closed + f[i + 1][(r + 1) % 3][1]   # or the run grows by one gap
    tag = f"inframe_gaps:{length}"

    def _sample(batch: int, seed: int) -> "torch.LongTensor":
        g = torch.Generator().manual_seed(stream_key(seed, tag))
        u = torch.rand((batch, length), generator=g).tolist()
        bases = torch.randint(1, 10, (batch, length), generator=g).tolist()
        rows = []
        for b in range(batch):
            r, h, row = 0, 0, []
            for i in range(length):
                gap_w = f[i + 1][(r + 1) % 3][1]
                base_w = 9 * f[i + 1][0][h] if r == 0 else 0
                if u[b][i] * (gap_w + base_w) < gap_w:
                    row.append(0); r, h = (r + 1) % 3, 1
                else:
                    row.append(bases[b][i]); r = 0
            rows.append(row)
        return torch.tensor(rows, dtype=torch.long)

    return _sample

def k_cuts_sampler(k: int, threshold: int = 5,
                   length: int = INPUT_LEN) -> Callable[[int, int], "torch.LongTensor"]:
    """Sampler drawing length-``length`` sequences with EXACTLY ``k`` digits ``>= threshold``
    among positions 1..length-1, UNIFORMLY over that set. Position 0 is unconstrained: it starts
    a segment whatever it holds.

    Use this to isolate a data-dependent segment measurement to a fixed number of segments (one
    cut = one locate plus two broadcasts; two cuts = two ordered locates plus a 3-way segment
    assignment). "Exactly k digits over threshold" is a GLOBAL count condition, so this
    CONSTRUCTS rather than rejects: choose the k cut positions uniformly among the
    C(length-1, k) subsets, fill them uniformly from the high digits and every other position
    uniformly from the low digits. The cut set is read back off the row, so the construction is
    a bijection onto the domain and every member lands with probability
    1 / (10 * C(length-1, k) * (10-threshold)**k * threshold**(length-1-k)).
    """
    k = int(k)
    if not 0 <= k <= length - 1:
        raise ValueError(f"need 0 <= k <= {length - 1} (got {k})")
    if not 1 <= threshold <= 9:
        raise ValueError(f"threshold must leave a non-empty high and low digit set (got {threshold})")
    hi = torch.arange(threshold, 10, dtype=torch.long)
    lo = torch.arange(0, threshold, dtype=torch.long)
    sites = torch.combinations(torch.arange(1, length, dtype=torch.long), r=k)
    tag = f"cuts:{k}:ge{threshold}:{length}"

    def _sample(batch: int, seed: int) -> "torch.LongTensor":
        g = torch.Generator().manual_seed(stream_key(seed, tag))
        out = lo[torch.randint(0, lo.numel(), (batch, length), generator=g)]
        out[:, 0] = torch.randint(0, 10, (batch,), generator=g)
        pick = sites[torch.randint(0, sites.shape[0], (batch,), generator=g)]
        rows = torch.arange(batch).unsqueeze(1).expand_as(pick)
        out[rows, pick] = hi[torch.randint(0, hi.numel(), pick.shape, generator=g)]
        return out

    return _sample

def _gaps_in_frame(x: "torch.Tensor", modulus: int = 3) -> bool:
    """At least one gap, and every maximal run of 0s a whole number of codons long."""
    n, i, seen = len(x), 0, False
    while i < n:
        if int(x[i]) != 0:
            i += 1
            continue
        j = i
        while j < n and int(x[j]) == 0:
            j += 1
        if (j - i) % modulus:
            return False
        seen, i = True, j
    return seen

GAPS_IN_FRAME = Condition("gaps_in_frame", _gaps_in_frame, inframe_gaps_sampler())

ONE_CUT = Condition("one_cut", lambda x: int((x[1:] >= 5).sum()) == 1, k_cuts_sampler(1))

TWO_CUTS = Condition("two_cuts", lambda x: int((x[1:] >= 5).sum()) == 2, k_cuts_sampler(2))


# --- divzeta_mobinv ---
"""New models/conditions.py factories for the W23 order-theory ops (divzeta_id, mobinv_id).

Paste the two samplers next to the other `*_sampler` factories and the two `Condition`s next to
`ALL_EVEN`/`BINARY`. `DIVISORS` belongs at module top with `INPUT_LEN`.

The slice's other four conditions need NO new code — they are the existing `domain()` factory with an
explicit name so the taxonomy sees a NEW condition_family instead of the occupied `domain` one:
    div_chain     = domain([1, 2, 4, 8], name="div_chain")      # divisibility is a total chain here
    div_antichain = domain([2, 3, 5, 7], name="div_antichain")  # pairwise incomparable
    div_b2        = domain([1, 2, 3, 6], name="div_b2")         # divisor lattice of 6 = Boolean B2
    gf2_flat05    = domain([0, 5], name="gf2_flat05")           # mod-10 arithmetic closes into GF(2)
"""

DIVISORS = [[d for d in range(1, i + 1) if i % d == 0] for i in range(INPUT_LEN)]

def constant_row_sampler(length: int = INPUT_LEN,
                         digits=range(10)) -> Callable[[int, int], "torch.LongTensor"]:
    """All-equal rows, the shared value uniform over ``digits`` — uniform over the |digits| rows."""
    pal = torch.tensor(sorted({int(d) for d in digits}), dtype=torch.long)
    tag = f"all_equal:{pal.tolist()}:{length}"

    def _sample(batch: int, seed: int) -> "torch.LongTensor":
        g = torch.Generator().manual_seed(stream_key(seed, tag))
        return pal[torch.randint(0, pal.numel(), (batch, 1), generator=g)].expand(batch, length).contiguous()

    return _sample

def zeta_image_sampler(length: int = INPUT_LEN) -> Callable[[int, int], "torch.LongTensor"]:
    """Rows in the image of the divisor-lattice zeta transform of a 0/1 vector:
    ``x[0] = g[0]``, ``x[i] = #{d | i : g[d] = 1}``. Isolates a Moebius-inversion op to the domain
    where it is exactly the INVERSE map (``mobinv(zeta(g)) == g``).

    zeta is injective, so drawing ``g`` uniform over ``{0,1}^length`` draws the row uniformly over the
    2**length images — exact-uniform with no rejection. Entries stay <= max_i d(i) = 4, so the image
    is in-alphabet and the op's mod-10 stage is provably dead here."""
    tag = f"zeta_image:{length}"

    def _sample(batch: int, seed: int) -> "torch.LongTensor":
        g = torch.randint(0, 2, (batch, length), dtype=torch.long,
                          generator=torch.Generator().manual_seed(stream_key(seed, tag)))
        out = g.clone()
        for i in range(1, length):
            out[:, i] = g[:, DIVISORS[i]].sum(1)
        return out

    return _sample

def _zeta_preimage(x) -> list:
    """Integer Moebius inversion of a row; it is a zeta image iff every entry lands in {0, 1}."""
    g = [int(x[0])]
    for i in range(1, len(x)):
        g.append(int(x[i]) - sum(g[d] for d in DIVISORS[i][:-1]))
    return g

ALL_EQUAL = Condition("all_equal", lambda x: bool((x == x[0]).all()), constant_row_sampler())

ZETA_IMAGE = Condition("zeta_image", lambda x: all(v in (0, 1) for v in _zeta_preimage(x)),
                       zeta_image_sampler())


# --- factnorm_nonadj4 ---
FACTORADIC_RADIX = [min(i + 2, 10) for i in range(INPUT_LEN)]

def below_factoradic_radix(positions, name: str) -> Condition:
    """Slot ``i`` already holds less than its factorial-base capacity ``min(i+2, 10)`` at every
    position in ``positions`` — the positions where FactoradicNormalize cannot generate a carry."""
    cons = {int(p): list(range(FACTORADIC_RADIX[int(p)])) for p in positions}
    return Condition(name, lambda x, _c=cons: all(int(x[p]) < FACTORADIC_RADIX[p] for p in _c),
                     position_domain_sampler(cons))

FACTORADIC_CANONICAL = below_factoradic_radix(range(INPUT_LEN), "factoradic_canonical")

CARRY_SOURCE_ONLY_FIRST = below_factoradic_radix(range(1, INPUT_LEN), "carry_source_only_first")

LO_EVEN_HI_ODD = Condition(
    "lo_even_hi_odd",
    lambda x: all(int(v) <= 2 if i % 2 == 0 else int(v) >= 7 for i, v in enumerate(x)),
    position_domain_sampler({p: (range(3) if p % 2 == 0 else range(7, 10)) for p in range(INPUT_LEN)}))


# --- hkconf_baksnep ---
"""New models/conditions.py factories for the W23 physics slice (hkconf_id, baksnep_id).

Paste the sampler block above the `Condition` section of models/conditions.py and the Condition
factories below `compare_positions`. Every sampler here is EXACT-UNIFORM over its domain: the three
hkconf domains are value-set couplings with base rates 0.005%-3.3% and the three baksnep domains are
extremum couplings, so rejection over 0-9 is either dead or wasteful. `_subsets` / `_onto` /
`support_sampler` are shared by all three hkconf conditions; `value_set_condition` derives `holds`
AND the sampler from ONE set-level rule so they cannot drift.
"""

def _subsets(values) -> list[list[int]]:
    """Every non-empty subset of ``values``, each sorted (10 digits -> 1023 sets)."""
    vals = sorted(values)
    return [[v for i, v in enumerate(vals) if m >> i & 1] for m in range(1, 1 << len(vals))]

def _onto(k: int, length: int) -> int:
    """# of length-``length`` rows that use EVERY one of k given values (surjections)."""
    return sum((-1) ** j * math.comb(k, j) * (k - j) ** length for j in range(k + 1))

def support_sampler(supports, tag: str, length: int = INPUT_LEN) -> Callable[[int, int], "torch.LongTensor"]:
    """Sampler over rows whose SET of distinct values is one of ``supports``, UNIFORMLY.

    Value-coupling domains (a bounded-confidence radius links values, not positions) have no
    per-position form and base rates down to 5e-5, so rejection over 0-9 is hopeless. Every row has
    exactly ONE support, so weighting support S by its surjection count and drawing a surjective row
    onto S is exact-uniform. Only the surjectivity draw rejects (accept = _onto(|S|)/|S|**length:
    0.95 at |S|=3, 0.78 at |S|=4, 0.27 at |S|=6).
    """
    sups = [sorted({int(v) for v in s}) for s in supports]
    if not sups:
        raise ValueError("empty support family")
    weights = torch.tensor([float(_onto(len(s), length)) for s in sups], dtype=torch.float64)

    def _sample(batch: int, seed: int) -> "torch.LongTensor":
        g = torch.Generator().manual_seed(stream_key(seed, tag))
        pick = torch.multinomial(weights, batch, replacement=True, generator=g)
        out = torch.empty((batch, length), dtype=torch.long)
        for k in pick.unique().tolist():
            pal = torch.tensor(sups[k], dtype=torch.long)
            want = int((pick == k).sum())
            kept, need = [], want
            while need > 0:
                blk = pal[torch.randint(0, pal.numel(), (max(4 * need, 64), length), generator=g)]
                good = blk[(blk.unsqueeze(2) == pal).any(1).all(1)]      # surjective onto pal
                kept.append(good); need -= good.shape[0]
            out[pick == k] = torch.cat(kept)[:want]
        return out

    return _sample

def first_extreme_sampler(smallest: bool = True,
                          length: int = INPUT_LEN) -> Callable[[int, int], "torch.LongTensor"]:
    """Rows whose FIRST digit is the row's min (``smallest``) or its max, EXACT-UNIFORM.

    Pinning the extremum to position 0 kills the argmin/argmax LOCALIZATION while leaving the
    reduction itself in place. x[0]=v admits ``(10-v)**(length-1)`` tails as a min and
    ``(v+1)**(length-1)`` as a max, so drawing v by those weights and filling the tail uniformly
    from the admissible range is uniform over the set; rejection would drop ~85% of rows and would
    also skew nothing but the runtime.
    """
    weights = torch.tensor([float((10 - v) ** (length - 1) if smallest else (v + 1) ** (length - 1))
                            for v in range(10)], dtype=torch.float64)
    tag = f"{'min' if smallest else 'max'}_first:{length}"

    def _sample(batch: int, seed: int) -> "torch.LongTensor":
        g = torch.Generator().manual_seed(stream_key(seed, tag))
        v = torch.multinomial(weights, batch, replacement=True, generator=g)
        span = (10 - v) if smallest else (v + 1)
        tail = (torch.rand((batch, length - 1), generator=g) * span[:, None]).long()
        return torch.cat([v[:, None], tail + v[:, None] if smallest else tail], dim=1)

    return _sample

def unique_extremes_sampler(length: int = INPUT_LEN) -> Callable[[int, int], "torch.LongTensor"]:
    """Rows with a UNIQUE min AND a UNIQUE max, EXACT-UNIFORM — the tie-break-free domain of any
    extremal-selection op (no leftmost-argmin branch to take).

    The two extreme positions are an ordered pair and the other ``length-2`` digits sit strictly
    inside, so (min a, max b) carries weight ``(b-a-1)**(length-2)`` (the position factor is
    constant and cancels) and b >= a+2 is forced. Rejection would keep ~35% but would still need
    this enumeration to prove uniformity.
    """
    pairs = [(a, b) for a in range(10) for b in range(a + 2, 10)]
    lo = torch.tensor([a for a, _ in pairs], dtype=torch.long)
    hi = torch.tensor([b for _, b in pairs], dtype=torch.long)
    weights = torch.tensor([float((b - a - 1) ** (length - 2)) for a, b in pairs], dtype=torch.float64)
    tag = f"no_extreme_tie:{length}"

    def _sample(batch: int, seed: int) -> "torch.LongTensor":
        g = torch.Generator().manual_seed(stream_key(seed, tag))
        pick = torch.multinomial(weights, batch, replacement=True, generator=g)
        a, b = lo[pick][:, None], hi[pick][:, None]
        rows = a + 1 + (torch.rand((batch, length), generator=g) * (b - a - 1)).long()
        pos = torch.rand((batch, length), generator=g).argsort(dim=1)      # 2 distinct positions
        return rows.scatter(1, pos[:, :1], a).scatter(1, pos[:, 1:2], b)

    return _sample

def value_set_condition(name: str, ok: Callable[[list], bool], length: int = INPUT_LEN) -> Condition:
    """Condition on the SET of distinct values a row uses: ``ok(sorted_values) -> bool``.
    ``holds`` and the sampler are both derived from ``ok``, so the pair cannot drift."""
    return Condition(name, lambda x: ok(sorted(set(x.tolist()))),
                     support_sampler([s for s in _subsets(range(10)) if ok(s)], name, length))

def range_at_most(k: int = 2) -> Condition:
    """Global range max-min <= k: every value sits in one confidence window, so a
    bounded-confidence mean has a single cluster and broadcasts the global mean."""
    k = int(k)
    return value_set_condition(f"range_le{k}", lambda s, _k=k: s[-1] - s[0] <= _k)

def separated_values(gap: int = 3) -> Condition:
    """Distinct values pairwise >= ``gap`` apart: with gap > the confidence radius nobody is in
    anybody else's window, so a bounded-confidence mean is the IDENTITY."""
    gap = int(gap)
    return value_set_condition(f"sep{gap}_values",
                               lambda s, _g=gap: all(b - a >= _g for a, b in zip(s, s[1:])))

def _two_clusters(vals: list, eps: int = 2) -> bool:
    """Sorted distinct ``vals`` form exactly TWO eps-linked groups, each of spread <= eps."""
    cuts = [i for i, (a, b) in enumerate(zip(vals, vals[1:])) if b - a > eps]
    return len(cuts) == 1 and vals[cuts[0]] - vals[0] <= eps and vals[-1] - vals[cuts[0] + 1] <= eps

BIMODAL_GAP = value_set_condition("bimodal_gap", _two_clusters)

MIN_FIRST = Condition("min_first", lambda x: int(x[0]) == int(x.min()), first_extreme_sampler(True))

MAX_FIRST = Condition("max_first", lambda x: int(x[0]) == int(x.max()), first_extreme_sampler(False))

NO_EXTREME_TIE = Condition("no_extreme_tie",
                           lambda x: int((x == x.min()).sum()) == 1 and int((x == x.max()).sum()) == 1,
                           unique_extremes_sampler())


# --- hsifte ---
"""NEW conditions for hsifte_id (Wave 22, tree/graph/DP) — paste into models/conditions.py.

`_tree_families` + `tree_order_sampler` + `_tree_holds` go with the other low-level sampler
factories (after `mux_sampler`); the `_*_legal` fns and the two Conditions go in the Condition
block, beside `monotone` / `compare_positions`. `from itertools import product` joins the module
imports; `Optional` is already imported.

SCOPE ASYMMETRY (deliberate, measured): `heap_ordered` constrains EVERY family of the tree — it is
the ordinary heap property, and it is what makes one condition collapse both the even-depth round
(hsifte) and the odd-depth round (hsifto) to the identity. `antiheap_leftmax` constrains only the
EVEN-depth families, the minimal precondition for hsifte's fixed permutation; extending it to the
whole tree would shrink the domain 658x (122,512,500 -> 186,329 rows) and constrain positions 5 and
6, which hsifte never reads.
"""

def _tree_families(length: int = INPUT_LEN, depth_parity: Optional[int] = None) -> list[tuple[int, list[int]]]:
    """(parent, children) per implicit-binary-tree node that has children, BFS order (parents first);
    ``depth_parity`` keeps only parents at even (0) or odd (1) depth."""
    fams = [(p, [c for c in (2 * p + 1, 2 * p + 2) if c < length]) for p in range(length)]
    return [(p, ch) for p, ch in fams
            if ch and (depth_parity is None or ((p + 1).bit_length() - 1) % 2 == depth_parity)]

def tree_order_sampler(legal: Callable[[int, tuple], bool], families: list[tuple[int, list[int]]],
                       length: int = INPUT_LEN,
                       tag: str = "tree_order") -> Callable[[int, int], "torch.LongTensor"]:
    """Sampler over the digit rows whose every listed implicit-binary-tree family (parent value
    ``a``, children values ``kids``) satisfies ``legal(a, kids)``, drawn EXACT-UNIFORM over that
    set. Positions in no listed family stay uniform 0-9.

    Use this for a tree-shaped partial order (heap order, its inversion) that a per-position
    sampler cannot express. Heapifying iid digits is NOT uniform -- it is the sorting trap that
    shipped ``monotone_sampler`` at TVD 0.47 -- and rejection costs ~326 draws/row at heap order.
    Instead count exactly: the listed families form a forest, subtrees are independent given their
    root values and the only cross-subtree constraint sits inside one family, so ``M[v][a]``
    (#labelings of ``v``'s component below ``v`` with ``x[v] == a``) factorizes as ``sum over legal
    kid-tuples of prod_c M[c][kid_c]``. Each forest root is then drawn by inverse-CDF on ``M[root]``
    and each family's kid-tuple by inverse-CDF on its own weights, which is uniform by construction.
    """
    tup = {p: torch.tensor(list(product(range(10), repeat=len(ch))), dtype=torch.long) for p, ch in families}
    M = [[1] * 10 for _ in range(length)]          # M[v][a] = #labelings below v with x[v] = a
    W = {}                                          # W[p][a, t] = weight of kid-tuple t under parent value a
    for p, ch in reversed(families):
        rows = [[(math.prod(M[c][int(t[i])] for i, c in enumerate(ch)) if legal(a, tuple(t.tolist())) else 0)
                 for t in tup[p]] for a in range(10)]
        W[p] = torch.tensor(rows, dtype=torch.float64)
        M[p] = [int(sum(r)) for r in rows]
    kids = {c for _, ch in families for c in ch}
    roots = [v for v in range(length) if v not in kids]
    root_w = torch.tensor([M[v] for v in roots], dtype=torch.float64)
    if not bool((root_w.sum(1) > 0).all()):
        raise ValueError(f"no length-{length} row satisfies the family rule")

    def _sample(batch: int, seed: int) -> "torch.LongTensor":
        g = torch.Generator().manual_seed(stream_key(seed, tag))
        out = torch.zeros((batch, length), dtype=torch.long)
        for i, v in enumerate(roots):
            out[:, v] = torch.multinomial(root_w[i].expand(batch, 10), 1, generator=g).squeeze(1)
        for p, ch in families:
            pick = torch.multinomial(W[p][out[:, p]], 1, generator=g).squeeze(1)
            out[:, ch] = tup[p][pick]
        return out

    return _sample

def _tree_holds(legal: Callable[[int, tuple], bool],
                families: list[tuple[int, list[int]]]) -> Callable[["torch.Tensor"], bool]:
    """Predicate sharing ``legal`` and ``families`` with ``tree_order_sampler`` so the two cannot drift."""
    return lambda x: all(legal(int(x[p]), tuple(int(x[c]) for c in ch)) for p, ch in families)

def _heap_legal(a: int, kids: tuple) -> bool:
    return all(k <= a for k in kids)

def _antiheap_legal(a: int, kids: tuple) -> bool:
    return all(k > a for k in kids) and (len(kids) < 2 or kids[0] >= kids[1])

_HEAP_FAMS = _tree_families()

_ANTIHEAP_FAMS = _tree_families(depth_parity=0)

HEAP_ORDERED = Condition("heap_ordered", _tree_holds(_heap_legal, _HEAP_FAMS),
                         tree_order_sampler(_heap_legal, _HEAP_FAMS, tag="heap_ordered"))

ANTIHEAP_LEFTMAX = Condition("antiheap_leftmax", _tree_holds(_antiheap_legal, _ANTIHEAP_FAMS),
                             tree_order_sampler(_antiheap_legal, _ANTIHEAP_FAMS, tag="antiheap_leftmax"))


# --- lprobe_dutch ---
"""New Condition factories for the Wave-23 DSA slice (lprobe_id, dutch_id).

Paste this block verbatim at the END of models/conditions.py: it uses only names already defined
there (INPUT_LEN, stream_key, domain_sampler, domain, Condition, Callable, torch), so appending is
the whole edit. Every sampler is EXACT-UNIFORM over its domain and draws on its own
``stream_key(seed, tag)`` stream.
"""

NONADJ_SETS = ([0, 2, 4, 6, 8], [1, 3, 5, 7, 9])

FLAG_CLASSES = ([0, 1, 2], [3, 4, 5, 6], [7, 8, 9])

def spaced_pairs_sampler(length: int = INPUT_LEN) -> Callable[[int, int], "torch.LongTensor"]:
    """Each of five pairwise NON-ADJACENT digits exactly twice, UNIFORMLY over that set.

    Isolates an allocator's collision path to one step: with every value doubled and no two values
    cyclically adjacent, a second copy of v always finds v+1 free, so the probe never walks twice.
    Uniform by construction rather than by rejection (accept rate ~2e-5): draw the value set
    uniformly from the two admissible 5-sets, then a uniform permutation of the fixed multiset --
    each of the 2 * 10!/2^5 rows comes out with probability 2^5/(2 * 10!).
    """
    if length != 2 * len(NONADJ_SETS[0]):
        raise ValueError(f"spaced_pairs needs length {2 * len(NONADJ_SETS[0])} (got {length})")
    pal = torch.tensor(NONADJ_SETS, dtype=torch.long).repeat_interleave(2, dim=1)   # [2, length]
    tag = f"spaced_pairs:{length}"

    def _sample(batch: int, seed: int) -> "torch.LongTensor":
        g = torch.Generator().manual_seed(stream_key(seed, tag))
        which = torch.randint(0, pal.shape[0], (batch,), generator=g)
        perm = torch.argsort(torch.rand(batch, length, generator=g), dim=1)
        return pal[which].gather(1, perm)

    return _sample

def flag_ordered_sampler(length: int = INPUT_LEN) -> Callable[[int, int], "torch.LongTensor"]:
    """Rows already grouped small (0-2) then medium (3-6) then large (7-9), UNIFORMLY over that set.

    Isolates a stable 3-way partition to its no-op case: the reordering provably never fires.
    "Already grouped" is a GLOBAL counting constraint (accept rate ~1e-4 under rejection), so draw
    the group-size composition (a, b, c) with weight 3**a * 4**b * 3**c -- exactly the number of
    rows it admits -- then fill each block iid-uniform from its class. Every ordered row belongs to
    exactly one composition, so the weighting makes the draw exact-uniform.
    """
    comps = [(a, b, length - a - b) for a in range(length + 1) for b in range(length + 1 - a)]
    weights = torch.tensor([float(len(FLAG_CLASSES[0]) ** a * len(FLAG_CLASSES[1]) ** b
                                  * len(FLAG_CLASSES[2]) ** c) for a, b, c in comps], dtype=torch.float64)
    pals = [torch.tensor(G, dtype=torch.long) for G in FLAG_CLASSES]
    tag = f"flag_ordered:{length}"

    def _sample(batch: int, seed: int) -> "torch.LongTensor":
        g = torch.Generator().manual_seed(stream_key(seed, tag))
        picks = torch.multinomial(weights, batch, replacement=True, generator=g).tolist()
        rows = []
        for k in picks:
            row = [pal[torch.randint(0, pal.numel(), (n,), generator=g)]
                   for n, pal in zip(comps[k], pals)]
            rows.append(torch.cat(row))
        return torch.stack(rows)

    return _sample

_SPACED_MULTISETS = [sorted(v for v in s for _ in (0, 1)) for s in NONADJ_SETS]

_FLAG_CLASS_OF = torch.tensor([0] * 3 + [1] * 4 + [2] * 3)

SPACED_PAIRS = Condition("spaced_pairs", lambda x: sorted(x.tolist()) in _SPACED_MULTISETS,
                         spaced_pairs_sampler())

NO_MIDS = domain([0, 1, 2, 7, 8, 9], name="no_mids")

FLAG_ORDERED = Condition("flag_ordered",
                         lambda x: bool((_FLAG_CLASS_OF[x][1:] >= _FLAG_CLASS_OF[x][:-1]).all()),
                         flag_ordered_sampler())


# --- slew2_rms3 ---
"""New models/conditions.py factories for the Wave-22 signal/numerical slice (slew2_id, rms3_id).

Paste the two sampler factories into the sampler section (after ``compare_positions_sampler``) and
the four Conditions into the Condition section (after ``compare_positions``). Nothing else in
conditions.py changes; ``ZEROFLANK`` reuses the existing ``position_domain_sampler``.

Families opened: step_le2, flatwin1_ge2, slew_sat, zeroflank.

The import block below is scaffolding so this file runs standalone; drop it on paste (conditions.py
already has every name it pulls in).
"""

def _chain_sampler(nxt: "torch.Tensor", tag: str,
                   length: int = INPUT_LEN) -> Callable[[int, int], "torch.LongTensor"]:
    """EXACT-UNIFORM sampler over the length-``length`` rows a digit-chain state machine accepts.

    ``nxt[s, d]`` is the state after emitting digit ``d`` from state ``s``, or -1 when ``d`` is
    illegal there; the LAST row of ``nxt`` is the pre-start state, so which digits may open a row
    is part of the same table. A chain constraint couples neighbouring positions, so a
    per-position ``domain_sampler`` cannot draw it and rejection collapses with length (the three
    domains below accept 1 uniform row in 1105, 102478 and 415). ``c[k][s]`` counts the ``k``-step
    continuations from state ``s``; emitting ``d`` from ``s`` with weight ``c[k-1][nxt[s, d]]``
    telescopes to probability ``1/|S|`` for EVERY accepted row -- exact-uniform, unlike a clamped
    random walk, which over-weights the boundary values it clamps onto.
    """
    ok = (nxt >= 0).to(torch.float64)
    counts = [torch.ones(nxt.shape[0], dtype=torch.float64)]
    for _ in range(length):
        counts.append((counts[-1][nxt.clamp(min=0)] * ok).sum(1))
    if counts[length][-1] == 0:
        raise ValueError(f"no sequences accepted by the chain ({tag})")

    def _sample(batch: int, seed: int) -> "torch.LongTensor":
        g = torch.Generator().manual_seed(stream_key(seed, tag))
        u = torch.rand((batch, length), generator=g, dtype=torch.float64)
        state = torch.full((batch,), nxt.shape[0] - 1, dtype=torch.long)
        out = torch.empty((batch, length), dtype=torch.long)
        for i in range(length):
            row = nxt[state]                                        # [batch, 10] successors
            cdf = (counts[length - 1 - i][row.clamp(min=0)] * ok[state]).cumsum(1)
            d = torch.searchsorted(cdf.contiguous(), u[:, i:i + 1] * cdf[:, -1:], right=True).squeeze(1)
            out[:, i] = d
            state = row.gather(1, d.unsqueeze(1)).squeeze(1)
        return out

    return _sample

def bounded_step_sampler(max_step: int, lo: int = 0, hi: int = 9,
                         length: int = INPUT_LEN) -> Callable[[int, int], "torch.LongTensor"]:
    """Sampler drawing rows whose digits stay in ``[lo, hi]`` and whose every consecutive step
    satisfies ``|x[i] - x[i-1]| <= max_step``, UNIFORMLY over that set (see ``_chain_sampler``).

    Use this to isolate a rate-limited tracker's TRACKING branch: an input that never moves
    faster than the limiter never engages it, so the op collapses to a copy."""
    if not (0 <= lo <= hi <= 9) or max_step < 0:
        raise ValueError(f"need 0 <= lo <= hi <= 9 and max_step >= 0 (got {lo}, {hi}, {max_step})")
    legal = [d for d in range(10) if lo <= d <= hi]
    nxt = torch.full((11, 10), -1, dtype=torch.long)
    for s in legal:
        for d in legal:
            if abs(d - s) <= max_step:
                nxt[s, d] = d
    nxt[10, legal] = torch.tensor(legal)                            # pre-start: any legal digit
    return _chain_sampler(nxt, f"step_le{max_step}:{lo}:{hi}:{length}", length)

def slew_saturated_sampler(rate: int = 2,
                           length: int = INPUT_LEN) -> Callable[[int, int], "torch.LongTensor"]:
    """Sampler drawing rows on which a rate-``rate`` slew limiter is ENGAGED at every step --
    ``|x[i] - y[i-1]| > rate`` for all ``i >= 1``, where ``y[0] = x[0]`` and
    ``y[i] = clamp(x[i], y[i-1] - rate, y[i-1] + rate)`` -- UNIFORMLY over that set.

    The limiter state ``y[i-1]`` is the chain state (``_chain_sampler``), so this is a path count
    over 10 states, not rejection: the domain accepts 1 uniform row in 415."""
    nxt = torch.full((11, 10), -1, dtype=torch.long)
    for s in range(10):
        for d in range(10):
            if abs(d - s) > rate:
                nxt[s, d] = s - rate if d < s else s + rate         # the clamp that engaged
    nxt[10, :] = torch.arange(10)                                   # pre-start: y[0] = x[0]
    return _chain_sampler(nxt, f"slew_sat{rate}:{length}", length)

def _slew_engaged(x: "torch.Tensor", rate: int = 2) -> bool:
    """Is a rate-``rate`` slew limiter engaged at EVERY step of ``x``?"""
    p = int(x[0])
    for v in x[1:].tolist():
        if abs(v - p) <= rate:
            return False
        p = min(max(v, p - rate), p + rate)
    return True

STEP_LE2 = Condition("step_le2", lambda x: bool(((x[1:] - x[:-1]).abs() <= 2).all()),
                     bounded_step_sampler(2))

FLATWIN1_GE2 = Condition("flatwin1_ge2",
                         lambda x: bool(((x[1:] - x[:-1]).abs() <= 1).all() and (x >= 2).all()),
                         bounded_step_sampler(1, lo=2))

SLEW_SAT = Condition("slew_sat", _slew_engaged, slew_saturated_sampler())

ZEROFLANK = Condition("zeroflank", lambda x: bool((x[1::2] == 0).all()),
                      position_domain_sampler({p: [0] for p in range(1, INPUT_LEN, 2)}))


# --- spn2_xpar ---
SPN2_SBOX = [5, 2, 1, 9, 6, 0, 4, 8, 7, 3]

def _spn2_round_keys(x: "torch.Tensor") -> tuple[int, int]:
    """The two keys spn2 derives from the row itself: t1 = Σx, t2 = Σ S[(x+t1)%10] (mod 10)."""
    t1 = int(x.sum()) % 10
    return t1, sum(SPN2_SBOX[(int(d) + t1) % 10] for d in x) % 10

def _spn2_counts(t1: int, length: int) -> list:
    """``cnt[k][s][t]`` = #{length-k suffixes with Σd ≡ s and Σ S[(d+t1)%10] ≡ t (mod 10)}."""
    f = [SPN2_SBOX[(d + t1) % 10] for d in range(10)]
    cnt = [[[0] * 10 for _ in range(10)] for _ in range(length + 1)]
    cnt[0][0][0] = 1
    for k in range(1, length + 1):
        prev, cur = cnt[k - 1], cnt[k]
        for d in range(10):
            for s in range(10):
                src, dst = prev[s], cur[(s + d) % 10]
                for t in range(10):
                    dst[(t + f[d]) % 10] += src[t]
    return cnt

def _inverse_cdf(u: float, weights: list) -> int:
    """Index drawn with probability proportional to ``weights`` from a uniform ``u`` in [0,1)."""
    target, acc = u * sum(weights), 0
    for i, w in enumerate(weights):
        acc += w
        if target < acc:
            return i
    return len(weights) - 1

def _spn2_sampler(t1_values, t2_zero: bool, tag: str,
                  length: int = INPUT_LEN) -> Callable[[int, int], "torch.LongTensor"]:
    """EXACT-uniform draw over ``{x : Σx ≡ t1 ∈ t1_values}`` intersected with ``Σ S[(x+t1)%10] ≡ 0``
    when ``t2_zero``.

    Both constraints are GLOBAL mod-10 reductions that couple all ten positions, and the tightest
    slice accepts ~1% of rows, so this counts instead of rejecting: ``_spn2_counts`` is a DP over
    the pair of mod-10 registers the cipher keys on, the row's t1 class is drawn proportional to
    its size, and the digits are filled left to right by inverse-CDF on the completion counts.
    Uniform by construction — every row with the same (t1, t2) class has the same probability
    because the DP counts completions, not paths.
    """
    keys = [int(a) for a in t1_values]
    cnt = {a: _spn2_counts(a, length) for a in keys}
    fmap = {a: [SPN2_SBOX[(d + a) % 10] for d in range(10)] for a in keys}
    marg = {a: [[sum(cnt[a][k][s]) for s in range(10)] for k in range(length + 1)] for a in keys}
    key_weights = [cnt[a][length][a][0] if t2_zero else marg[a][length][a] for a in keys]
    if not sum(key_weights):
        raise ValueError(f"no rows satisfy {tag}")

    def _sample(batch: int, seed: int) -> "torch.LongTensor":
        g = torch.Generator().manual_seed(stream_key(seed, tag))
        u = torch.rand((batch, length + 1), generator=g).tolist()
        rows = []
        for r in range(batch):
            a = keys[_inverse_cdf(u[r][length], key_weights)]
            c, f, rem_s, rem_t, row = cnt[a], fmap[a], a, 0, []
            for i in range(length):
                k = length - i - 1
                ws = [c[k][(rem_s - d) % 10][(rem_t - f[d]) % 10] if t2_zero
                      else marg[a][k][(rem_s - d) % 10] for d in range(10)]
                d = _inverse_cdf(u[r][i], ws)
                row.append(d)
                rem_s, rem_t = (rem_s - d) % 10, (rem_t - f[d]) % 10
            rows.append(row)
        return torch.tensor(rows, dtype=torch.long)

    return _sample

def spn2_transparent() -> Condition:
    """Both spn2 round keys vanish (Σx ≡ 0 and Σ S[x] ≡ 0): the involution undoes itself, spn2 ≡ COPY."""
    return Condition("spn2_transparent", lambda x: _spn2_round_keys(x) == (0, 0),
                     _spn2_sampler([0], True, "spn2_transparent"))

def spn2_key1zero() -> Condition:
    """Round-1 key vanishes (Σx ≡ 0): spn2 collapses to a single cipher round applied to S[x]."""
    return Condition("spn2_key1zero", lambda x: _spn2_round_keys(x)[0] == 0,
                     _spn2_sampler([0], False, "spn2_key1zero"))

def spn2_key2zero() -> Condition:
    """Round-2 key vanishes (Σ S[(x+Σx)%10] ≡ 0): spn2 collapses to the substitution-free
    y = (x + Σx) mod 10, because S is an involution."""
    return Condition("spn2_key2zero", lambda x: _spn2_round_keys(x)[1] == 0,
                     _spn2_sampler(range(10), True, "spn2_key2zero"))

def _xpar_encode(msg: "torch.Tensor") -> "torch.LongTensor":
    """``[B,4]`` message digits -> ``[B,10]`` codeword: digit 4 zeroes row 0's sum, row 1 negates row 0."""
    top = torch.cat([msg, (-msg.sum(1, keepdim=True)) % 10], dim=1)
    return torch.cat([top, (-top) % 10], dim=1)

def _xpar_residues(x: "torch.Tensor") -> tuple["torch.Tensor", "torch.Tensor"]:
    """(row residues [2], column residues [5]) of a length-10 row read as a 2x5 grid, mod 10."""
    grid = x.reshape(2, 5)
    return grid.sum(1) % 10, grid.sum(0) % 10

def _xpar_codeword_sampler(tag: str) -> Callable[[int, int], "torch.LongTensor"]:
    """EXACT: 4 uniform message digits -> the unique codeword completing them. The parity map is a
    BIJECTION onto the 10^4 codewords (the 7 checks have rank 6), so the draw is uniform."""
    def _sample(batch: int, seed: int) -> "torch.LongTensor":
        g = torch.Generator().manual_seed(stream_key(seed, tag))
        return _xpar_encode(torch.randint(0, 10, (batch, 4), generator=g))

    return _sample

def _xpar_error_sampler(positions, tag: str) -> Callable[[int, int], "torch.LongTensor"]:
    """EXACT: uniform codeword + a uniform NONZERO error at a uniform site in ``positions``. The
    code has minimum distance 4, so (codeword, site, magnitude) -> received word is injective and
    the uniform draw on the three factors is uniform on the received-word domain."""
    pal = torch.tensor(sorted({int(p) for p in positions}), dtype=torch.long)

    def _sample(batch: int, seed: int) -> "torch.LongTensor":
        g = torch.Generator().manual_seed(stream_key(seed, tag))
        x = _xpar_encode(torch.randint(0, 10, (batch, 4), generator=g))
        p = pal[torch.randint(0, pal.numel(), (batch,), generator=g)]
        e = torch.randint(1, 10, (batch,), generator=g)
        rows = torch.arange(batch)
        x[rows, p] = (x[rows, p] + e) % 10
        return x

    return _sample

def _xpar_holds_codeword(x: "torch.Tensor") -> bool:
    r, c = _xpar_residues(x)
    return not bool(r.any()) and not bool(c.any())

def _xpar_holds_onerror(x: "torch.Tensor") -> bool:
    r, c = _xpar_residues(x)
    return int((r != 0).sum()) == 1 and int((c != 0).sum()) == 1

def _xpar_holds_errrow0(x: "torch.Tensor") -> bool:
    r, c = _xpar_residues(x)
    return int(r[0]) != 0 and int(r[1]) == 0 and int((c != 0).sum()) == 1

def xpar_codeword() -> Condition:
    """Valid 2x5 product-code codewords (every row and column sum ≡ 0 mod 10): xpar ≡ COPY."""
    return Condition("xpar_codeword", _xpar_holds_codeword, _xpar_codeword_sampler("xpar_codeword"))

def xpar_onerror() -> Condition:
    """Codeword plus exactly one nonzero error — one broken row and one broken column, so the full
    locate-and-repair decoder runs and returns the codeword."""
    return Condition("xpar_onerror", _xpar_holds_onerror,
                     _xpar_error_sampler(range(10), "xpar_onerror"))

def xpar_errrow0() -> Condition:
    """Codeword plus one nonzero error confined to the TOP row: the row locator is constant, leaving
    only the 5-way column search."""
    return Condition("xpar_errrow0", _xpar_holds_errrow0,
                     _xpar_error_sampler(range(5), "xpar_errrow0"))
