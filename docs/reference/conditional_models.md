# Conditional operators in the DSL

> **Note:** sub-circuits are declared by hand, not auto-detected. This file is a reference for
> deciding which operators warrant a hand-defined DomainCircuit; to create one, author a
> `DomainCircuit` subclass in [`pipeline/models/circuits.py`](../../pipeline/models/circuits.py)
> (its `_cond` comes from [`pipeline/models/conditions.py`](../../pipeline/models/conditions.py))
> as described in [declaring_circuits.md](declaring_circuits.md).

Reference catalog of the **conditional** seq2seq operators in the DSL engine
[`pipeline/models/algorithms.py`](../../pipeline/models/algorithms.py), i.e. operators whose
algorithm has an explicit branch keyed on a recognizable feature of the input — the `segrev3`
pattern. Every worked example below is reproduced by the operator's `eval()`. Full operator list:
[operators.md](operators.md).

## What "conditional" means here

A model is **conditional** when its algorithm has an explicit branch keyed on a
recognizable feature of the input — so there is a class of inputs *lacking* that
feature on which the model collapses to a simpler, well-defined **base algorithm**.
You trace the base circuit by running only on those condition-free inputs.

Canonical example: `segrev3` = `SegmentReverse(boundary_digit=3, Id)`. With no `3`
in the input, the whole sequence is one segment → plain reverse.

In the DSL, a model NAME is an `algorithm_id`; the conditional behavior comes from the
**operator**. Any model whose dominant operator is one of the below is a conditional
model. They sort into three tiers by how clean the condition-free base is.

| Tier | Operators | Gate | Condition-free base |
|------|-----------|------|---------------------|
| 1 | `SegmentReverse` | presence of digit `d` | clean, non-trivial (`Reverse`) |
| 2 | `ConditionalJump`, `AccumulateReset`, `SwitchedScan` | value/threshold | single-regime walk / sum / recurrence |
| 3 | `RunningCountDigit`, `CompareShift`, `ZigzagCount`, `RecordCount`, `VanEckExtension` | input feature/pattern | constant output (degenerate) |

**Tracing usability.** Tier 1 and Tier 2 give a *meaningful* base circuit (a DomainCircuit)
by excluding the feature / large values. Tier 3 detectors give a constant base, so you isolate
the mechanism by *contrasting* feature-present vs feature-absent inputs, i.e. with a
ContrastCircuit ([circuit_types.md §4](circuit_types.md)).

---

## Tier 1 — Digit-gated, clean base

### `SegmentReverse(boundary_digit d, child)`

**Algorithm.** Take the child's output sequence `x`. Scan left→right; every position
where `x[i] == d` is a **boundary**. The boundary digits stay fixed in place; each
maximal run of non-`d` tokens *between* boundaries is reversed independently.

**Worked example** (`d=5`, input `[3,8,2,5,7,1,9,5,4,6]`):

```
segments:   [3,8,2] | 5 | [7,1,9] | 5 | [4,6]
reverse ea: [2,8,3] | 5 | [9,1,7] | 5 | [6,4]
output:     [2,8,3,5,9,1,7,5,6,4]
```

- **Trigger.** Presence of digit `d`. Each `d` partitions the sequence.
- **Condition-free base.** No `d` anywhere → one segment spanning the whole input →
  **plain `Reverse`**. This is exactly `segrev3` (`d=3`); the family spans `d = 0…9`.
- **What it must compute.** A boundary detector (`token == d`) + a local,
  variable-length reversal that respects those boundaries — position-relative copying
  whose source offset depends on the (input-dependent) segment geometry.

---

## Tier 2 — Value/threshold-gated regime switch

### `ConditionalJump(T, threshold, child)`

**Algorithm.** Treat values as signed jump sizes over a length-`L` ring. Start at
`i = u[0] % L`. For `t = 0…T-1`: emit `y[t] = u[i]`, then move
- **forward** `i = (i + u[i]) % L` if `u[i] > threshold`,
- **backward** `i = (i − u[i]) % L` otherwise.

**Worked example** (`u=[3,8,2,7,1,9,4,6,5,0]`, `T=8`, `threshold=5`):

```
i0=3, u[3]=7 >5 → y=7, i=(3+7)%10=0
      u[0]=3 ≤5 → y=3, i=(0-3)%10=7
      u[7]=6 >5 → y=6, i=(7+6)%10=3
      u[3]=7 >5 → y=7, i=0 ... cycles
output: [7,3,6,7,3,6,7,3]
```

- **Trigger.** Per-step: current value above/below `threshold` flips the jump *direction*.
- **Condition-free base.** Restrict input digits to all `≤ threshold` → every step jumps
  **backward** (a single unidirectional index walk); restrict to all `> threshold` →
  every step jumps **forward**. Either way it collapses to a `JumpIndex`-style
  one-direction pointer walk.
- **What it must compute.** A magnitude comparison (`> threshold`) feeding a signed
  pointer update — a content-addressed read whose stride and sign are data-dependent.

### `AccumulateReset(threshold, child)`

**Algorithm.** Running accumulator `acc = 0`. For each input `u[i]`: `acc += u[i]`;
**if `acc ≥ threshold`, fold it down with `acc = acc % threshold`**; then emit
`y[i] = acc % 10`. (The reset is a modulo by `threshold`, not a zeroing.)

**Worked example** (`threshold=15`, input `[9,9,2,7,…]`):

```
acc=9            <15 → y=9
acc=18  ≥15 →3        → y=3
acc=3+2=5             → y=5
acc=5+7=12           → y=2  (12%10)
```

- **Trigger.** The accumulator crossing `threshold`.
- **Condition-free base.** Use small-valued inputs so the cumulative sum never reaches
  `threshold` → the reset never fires → **plain `PrefixSumMod`** (running sum mod 10).
  (The trigger is on the hidden accumulator, so you remove it by bounding input
  magnitudes, not by banning a single token.)
- **What it must compute.** A monotone running sum plus a threshold-comparison-and-fold —
  sequential state with a conditional correction.

### `SwitchedScan(a1,b1,a2,b2,thresh, child)`

**Algorithm.** Hidden state `s = 0`. For each input `x[i]`, pick one of two linear
recurrences by the **current state**:
- `s = (a1·s + b1·x[i]) % 10` if `s < thresh` (low regime),
- `s = (a2·s + b2·x[i]) % 10` if `s ≥ thresh` (high regime),

then emit `y[i] = s`.

**Worked example** (`a1=3,b1=1,a2=7,b2=3,thresh=5`, input `[1,2,3,4,5,6,7,8,9,0]`):

```
s=0<5 → 3·0+1·1=1            y=1
s=1<5 → 3·1+1·2=5            y=5
s=5≥5 → (7·5+3·3)%10=4       y=4
s=4<5 → 3·4+1·4=16%10=6      y=6
... → output [1,5,4,6,7,7,0,8,3,9]
```

- **Trigger.** The running state being below/above `thresh` selects which recurrence runs.
- **Condition-free base.** Weakest of Tier 2 — the gate is on internal state, not the raw
  input. With `thresh` set outside the state's reachable range (e.g. `thresh=0` ⇒ always
  high regime), it degenerates to a **single fixed `HashScan`-style linear recurrence**.
- **What it must compute.** Two affine state updates plus a state-comparator that routes
  between them — piecewise-linear dynamics no single-recurrence head can reproduce.

---

## Tier 3 — Feature/pattern detectors (condition-free base is *constant*)

Genuinely conditional, but removing the input feature drives the output to a constant —
so the productive contrast is run **with vs without** the feature to isolate the detector,
rather than tracing a rich base.

### `RunningCountDigit(d, child)`

**Algorithm.** `out[i] = (# positions j ≤ i with x[j] == d) % 10` — a cumulative count
of digit `d`.

**Example** (`d=1`, input `[3,1,4,1,5,1,2,6,1,3]`): `out = [0,1,1,2,2,3,3,3,4,4]`.

- **Trigger.** Occurrences of digit `d`. **Base:** no `d` → **all zeros**. (Digit-gated
  exactly like segrev3, but the base is degenerate.)
- **What it must compute.** A token-`d` detector + a monotone counter mod 10.

### `CompareShift(k, child)`

**Algorithm.** `out[i] = 1` if `x[i] == x[i−k]` else `0`; for `i < k`, `out[i] = 0`
(boundary). A lag-`k` equality detector.

**Example** (`k=2`, input `[3,1,3,1,3,1,2,6,5,3]`): `out = [0,0,1,1,1,1,0,0,0,0]`.

- **Trigger.** Matches at distance `k`. **Base:** no lag-`k` repeats → **all zeros**.
- **What it must compute.** A fixed-offset look-back + equality test (pointer equality
  across a constant distance).

### `ZigzagCount(child)`

**Algorithm.** Cumulative count of strict local extrema. Seeing `x[i]` resolves whether
`x[i−1]` was a peak (`x[i−1] > x[i−2]` and `x[i−1] > x[i]`) or valley (both `<`).
`out[i] =` count over positions `{1…i−1}` mod 10; `out[0]=out[1]=0`.

**Example** (input `[1,5,2,8,3,9,4,7,2,6]`, fully alternating): `out = [0,0,1,2,3,4,5,6,7,8]`.

- **Trigger.** Local up/down reversals. **Base:** a monotone (sorted) input → no extrema
  → **all zeros**.
- **What it must compute.** A 3-tap comparator (compare a position to both neighbors)
  feeding a counter.

### `RecordCount(child)`

**Algorithm.** Cumulative count of left-to-right maxima ("records"). A record at `j` means
`x[j] >` everything before it. `out[i] = (# records in x[0:i+1]) % 10`; `x[0]` is always a
record so `out[0]=1`. (Distinct from `PrefixMax`, which emits the max *value*; this emits
how many times the max *changed*.)

**Example** (input `[3,1,4,1,5,9,2,6,5,3]`, records at positions 0,2,4,5):
`out = [1,1,2,2,3,4,4,4,4,4]`.

- **Trigger.** New running maxima. **Base:** a strictly descending input → only `x[0]` is
  a record → **all ones**.
- **What it must compute.** A running-max comparator that increments a counter only on
  strict improvement.

### `VanEckExtension(T, child)`

**Algorithm.** Treat the length-`L` input as the first terms of a sequence, then generate
`T` more by Van Eck's rule. Track `last_seen` (built from `seed[0..L-2]`), with
`prev_val = seed[L-1]`, `prev_pos = L-1`. For each `t`:
- `y[t] = (prev_pos − last_seen[prev_val]) % 10` if `prev_val` was seen before, else `0`;
- update `last_seen[prev_val] = prev_pos`; advance `prev_pos = L + t`; set
  `prev_val = y[t]` (self-referential — outputs feed back in).

**Example** (input `[5,7,4,8,3,9,2,0,1,6]`, `T=10`):

```
prev=6 unseen → y=0
prev=0 last at pos7 → 10-7=3 → y=3
prev=3 last at pos4 → 11-4=7 → y=7
... → output [0,3,7,1,5,4,3,5,3,2]
```

- **Trigger.** Whether a value has appeared before. **Base:** an all-distinct seed → the
  first generated term is `0`; behavior then becomes self-referential off the generated tail.
- **What it must compute.** Full-history "last occurrence" lookup + gap arithmetic, with
  feedback from its own outputs — the most memory-intensive of the group.

---

## Not conditional

Apply the same rule at every step (or are data-dependent walks/reorderings with no
input-feature toggle), so there is no condition-free input subset that simplifies them:

`Id, Const, Reverse, Rotate, Take, Drop, MapAdd, MapMul, PadTo, PrefixMax, PrefixSumMod,
XORScan, DiffShift, ModRemainder, HashScan, WindowHash, SelfConvolve, CellularAutomaton,
LFSR, ReductionTree, JumpIndex, IndexMix, TwoPointerMerge, CycleWalk, ChainedIndex,
PermuteByRank, ChunkSort, RunLengthEncode, InverseLookup, ScatterWrite, KMPFailure, BWT,
LZ78Complexity, TreeWalkDFS/BFS`.

Near-miss: **`ParityInterleave`** branches on **position** parity
(`i % 2`), not on an input value — so there is no condition-free *input* subset. Not
conditional in this sense.

---

## Practical notes

- All of these operators live in the DSL engine `pipeline/models/algorithms.py`. To build a
  model on one, add its `algorithm_id` to `pipeline/models/models.txt`, then train and analyse
  it with `python pipeline/run_model_pipeline.py --model <algorithm_id> --stages all`.
- **`SegmentReverse(d)`** is the direct generalization of the `segrev3` pattern (boundary
  digit `d = 0…9`); `segrev3_id` = `SegmentReverse(3, Id)` is the worked example, and its
  `reverse_only` sub-circuit is a `DomainCircuit` subclass in `pipeline/models/circuits.py`
  (isolated on no-`3` inputs via `conditions.exclude_digit(3)`). The full case study of the
  segrev3 model is [segrev3_case_study.md](../segrev3_case_study.md).
