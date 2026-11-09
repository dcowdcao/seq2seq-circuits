# Contrast-pair circuit catalog

Per-model pair designs for ContrastCircuits ([circuit_types.md §4](circuit_types.md)): clean/corrupt
pairs differing by one minimal edit of a causal variable; scored positions auto-derive as
`{i : parent.eval(clean)[i] != parent.eval(corrupt)[i]}`. Every mask below was checked against the
`eval()` implementations in [`pipeline/models/algorithms.py`](../../pipeline/models/algorithms.py).
Slugs are the registered `circuit_id`s in
[`pipeline/models/circuits.py`](../../pipeline/models/circuits.py), except the variants marked
*design only*. How to declare a new one: [declaring_circuits.md](declaring_circuits.md).

Ratings: CLEAN = edit toggles exactly the latent, mask exact. MEDIUM = one benign confound needing a
constraint or a paired control. CONFOUNDED = latent inseparable from a second change; run with eyes
open.

## Operator semantics that change the designs

1. `AccumulateReset(13)` does NOT reset to 0: `acc %= threshold` on crossing, output
   `(S_i mod 13) mod 10`. A δ=1 edit at the crossing provably flips every downstream output
   (residues m vs m−1 mod 13 never collide through %10) — suffix masks are gap-free.
2. `CompareSwap` swaps on strict `>`, so ties don't swap — enables tie-straddles.
3. `SRLatch` drivers are idempotent — a set-toggle needs prior state pinned to 0.
4. Records are strict `>` (`RecordCount`, `RecordIndicator`) — corrupt = tie is the
   minimal non-record.
5. `Deplete` floors with `max(floor, acc - v)`, so landing exactly on 0 and clipping to 0
   are output-identical forever — the clip bit is output-invisible (ProbeCircuit territory).
6. `ConditionalJump`: −5 ≡ +5 mod 10, so straddling 6↔5 lands the pointer at adjacent cells —
   the minimal trajectory perturbation. `x[0]` is double-duty (pointer init AND cell-0 content).
7. `DoubleOdd` output preimages pair up (2←{1,2}, 6←{3,6}, 0←{0,5}, 4←{7,4}, 8←{9,8};
   `DoubleOdd.eval`): those parity-toggles have identical outputs → EMPTY mask →
   unusable as contrast (verified). Same trap in `ParityRoute` (o−1, o+1 both → o).

## Per-model summary

| model | primary contrast slug | edit pattern | mask (verified) | rating |
|---|---|---|---|---|
| rcnt3/rcnt5 | rcnt_digit_toggle | digit-toggle d↔non-d at p | {p..9} exact | CLEAN |
| cntodd | cntodd_parity_toggle | parity-toggle δ=1 | {p..9} exact | CLEAN |
| walk12 | walk12_updown_flip (+up_toggle, *design only*) | class-flip 1↔2 (δ=1) | {p..9} exact | CLEAN |
| reccount | reccount_record_toggle | tie-straddle m+1↔m, suffix ≤ m | {p..9} exact | CLEAN |
| psumodd | psumodd_parity_toggle + value control | parity δ=1 / +2-odd control | {p..9} exact | MEDIUM/CLEAN pair |
| prod | prod_zero_toggle (0↔1, units domain) | annihilate vs no-op | {p..9} exact | CLEAN |
| reset3 | reset3_reset_toggle | digit-toggle (no value path at all) | {p..q−1} window | CLEAN |
| tresetmax3 | tresetmax3_reset_toggle | digit-toggle, c ≤ prefix-max, m ≥ 1 | {p..q−1} | MEDIUM |
| sr12 | sr12_reset_toggle / set_toggle (*design only*) | digit-toggle, prior state pinned | {p..q−1} | CLEAN |
| holdge5 | holdge5_threshold_straddle (5↔4) + 6↔5 control | straddle δ=1, held reg ≠ 5 | {p..q−1} | CLEAN |
| pmax | pmax_newmax_toggle | tie-straddle m+1↔m | {p..q−1} | CLEAN |
| dep9f0 | dep9f0_clip_toggle | straddle δ=2, suffix 0s to lengthen | {p..q−1}, short | MEDIUM |
| dblodd | dblodd_parity_toggle | parity δ=1, AVOID preimage pairs | {p} | CLEAN |
| pmstep | pmstep_parity_toggle | parity δ=1, avoid (o−1,o+1) | {p} | CLEAN |
| fixedpoint | fixedpoint_hit_toggle | x[p]=p ↔ p±1 | {p} | CLEAN (machinery smoke test) |
| novel | novel_repeat_toggle | seen-value ↔ globally-absent value | {p} | CLEAN |
| recordind | recordind_record_toggle (+record_steal, *design only*) | tie-straddle / two-site steal | {p} / {p,q} | CLEAN / MEDIUM |
| bmax1 | bmax1_local_straddle | tie-straddle δ=1, freeze x[p+1] | {p} | CLEAN |
| anc0 | anc0_anchor_perturb (+payload control) | anchor ±1 | {1..9} exact (0 invariant) | CLEAN |
| fclamp0 | fclamp0_anchor_perturb / clamp_straddle (*design only*) | anchor ±1 / straddle δ=2 | {0}∪{i: x[i]≤b} / {p} | MEDIUM / CLEAN |
| drot0 | drot0_amount_perturb | rotation r ↔ r+1 | near-full | MEDIUM |
| refl0 | refl0_pivot_perturb | pivot ±1 | near-full, fixed-point holes | MEDIUM |
| stride0 | stride0_stride_perturb | pinned k-pair (regimes differ by k) | near-full | CONFOUNDED |
| pref0 | pref0_k_perturb | k ↔ k+1, k ∈ [2..8] | ⊆ {0..k} | MEDIUM |
| psort0 | psort0_rank_move (payload, not anchor) | value-perturb changing rank | rank window | MEDIUM |
| cmpswap0p9 | cmpswap09_tie_straddle (+ mirror at 9) | straddle b+1↔b at pos 0 | **{9}** (effect at far end) | CLEAN |
| carryk1 | carryk1_carry_gate (depth j family) | straddle 9↔8 in a 9-run | {j−1, j} | CLEAN |
| argminmax | argminmax_max_swap / tiebreak_demote (*design only*) | position-swap / δ=1 demote | {p,q} sparse | CLEAN |
| areset13 | areset13_crossing_toggle / compensated | straddle δ=1 / two-site compensated | {k..9} / {k} | MEDIUM / CLEAN |
| segrev3 | segrev3_second_boundary_toggle | 3-toggle at q, anchor 3 kept at p<q | {p+1..9} | MEDIUM |
| cjump10t5 | cjump_branch_straddle (6↔5 at visited cell) | straddle on trajectory | {t}∪divergent suffix, per-pair | CONFOUNDED |

Notes on the non-obvious rows:

- **reccount / recordind**: without the suffix ≤ m constraint a later `x[q]=m+1` is a record in
  corrupt but not clean (second event). The unconstrained variant ("record steal", mask `{p,q}`)
  is a deliberate two-site probe of max-memory routing, not a bug.
- **cmpswap09_tie_straddle** is the catalog's most elegant design: clean `x[0]=b+1` (swap fires),
  corrupt `x[0]=b` (tie, no swap). Clean y[0]=x[9]=b equals corrupt y[0]=x[0]=b — the copy confound
  at the edited position cancels identically; the entire differential is the branch decision routed
  to position 9. The mirror (edit at 9, mask {0}) separately probes the other copy path.
- **carryk1_carry_gate** varies carry depth j: suffix `x[j+1..9]=9…9`, `x[j−1]≤8`; clean `x[j]=9`
  (propagates), corrupt `x[j]=8` (absorbs). Mask exactly {j−1, j}. A parametric family measuring
  how deep the attention-only ripple actually reaches (the `carryk1_local` DomainCircuit pins
  depth 0).
- **areset13 compensated variant**: corrupt additionally sets `x[k+1] += 1`, restoring the sum from
  k+1 on → mask exactly {k}. Honest relabeling: the latent becomes "reset at k vs at k+1" (timing
  shift); the found circuit is the compare-and-wrap readout at k with downstream held fixed.
  General recipe for any 1-D accumulator: restore the state one step later to shrink a suffix mask
  to the event site.
- **segrev3**: the kept boundary at p provably fences outputs 0..p (identical in both) — a free
  invariant region for the necessity check. The single-3 presence-toggle has a near-full mask
  (clean = two local reversals, corrupt = one global reversal) — use it as the *sufficiency-side*
  dataset for the union test, not for discovery.
- **cjump**: branch straddle is irreducibly confounded — the branch's only consequence IS the
  trajectory (straddle at the final step is consequence-free). Genuine emitted-value coincidences
  occur mid-trajectory (both walks pass the same cell); only per-pair auto-derivation gets the mask
  right. Interpret as "comparator + both walk branches".

## Five worked flagships

Every input/output row below is reproduced by the operator's `eval()` in
`pipeline/models/algorithms.py`.

### F1. segrev3_second_boundary_toggle (kept 3 at p=2, toggled 3 at q=6)

```
clean   in : [4, 1, 3, 7, 2, 8, 3, 5, 6, 9]     [4,1] | 3 | [7,2,8] | 3 | [5,6,9]
corrupt in : [4, 1, 3, 7, 2, 8, 5, 5, 6, 9]     [4,1] | 3 | [7,2,8,5,5,6,9]
clean   out: [1, 4, 3, 8, 2, 7, 3, 9, 6, 5]
corrupt out: [1, 4, 3, 9, 6, 5, 5, 8, 2, 7]
mask       : outputs {3..9}; outputs {0,1,2} identical (fenced by the kept boundary)
```
Expected: 3-detector head keyed on q + segment-membership routing into the reversal movers.
Union prediction: this ∪ `reverse_only` (7 heads / 18 edges in the zoo index) ≈ segrev3 — a
decomposition test that a single whole-model circuit cannot provide.

### F2. areset13 both variants (recall y[i] = (S_i mod 13) mod 10; prefix [5,3,4] → S=12 at k−1, k=3)

```
suffix variant:      clean x[3]=1 (S=13, acc→0)   corrupt x[3]=0 (S=12, acc=12)
clean   in : [5, 3, 4, 1, 2, 6, 3, 7, 1, 4]   clean   out: [5, 8, 2, 0, 2, 8, 1, 5, 6, 0]
corrupt in : [5, 3, 4, 0, 2, 6, 3, 7, 1, 4]   corrupt out: [5, 8, 2, 2, 1, 7, 0, 4, 5, 9]
mask       : {3..9}, provably gap-free

compensated variant: corrupt additionally x[4]: 2→3 (restores S from k+1)
corrupt in : [5, 3, 4, 0, 3, 6, 3, 7, 1, 4]   corrupt out: [5, 8, 2, 2, 2, 8, 1, 5, 6, 0]
mask       : {3} only
```
Pair the suffix variant with a both-stay-≤12 δ=1 control (pure accumulator); contrast-minus-control
isolates the ≥13 comparator + wrap. The compensated variant isolates the readout at k directly.

### F3. dblodd_parity_toggle (p=3: 7 odd → 4; corrupt 8 even → 8)

```
clean   in : [2, 7, 4, 7, 0, 6, 3, 8, 5, 1]   clean out : [2, 4, 4, 4, 0, 6, 6, 8, 0, 2]
corrupt in : [2, 7, 4, 8, 0, 6, 3, 8, 5, 1]   corrupt out: [2, 4, 4, 8, 0, 6, 6, 8, 0, 2]
mask       : {3}
```
Trap validated: preimage pair 3↔6 at the same slot → identical outputs, EMPTY mask. The pair
validator must assert non-empty masks per row. Expected circuit: the known parity heads attending
p + p's direct path; should be near-contained in the intersection of the `one_odd` and `all_odd`
DomainCircuits.

### F4. sr12_reset_toggle (set at 1, toggle reset at 4, next driver at 7)

```
clean   in : [0, 1, 4, 5, 2, 7, 8, 1, 3, 9]   clean out : [0, 1, 1, 1, 0, 0, 0, 1, 1, 1]
corrupt in : [0, 1, 4, 5, 4, 7, 8, 1, 3, 9]   corrupt out: [0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
mask       : {4, 5, 6}   (resync at the p=7 set)
```
Union sr12_setlatch ∪ reset-contrast ≈ SR latch — the crispest union-sufficiency test in this
catalog (binary output, exact window).

### F5. cjump_branch_straddle (cell 7 first visited at t=2; 6→5)

```
clean   in : [3, 8, 2, 7, 1, 9, 4, 6, 5, 0]   trace 3→0→7→3→0→7→3→0→7→3
corrupt in : [3, 8, 2, 7, 1, 9, 4, 5, 5, 0]   trace 3→0→7→2→0→7→2→0→7→2
clean   out: [7, 3, 6, 7, 3, 6, 7, 3, 6, 7]
corrupt out: [7, 3, 5, 2, 3, 5, 2, 3, 5, 2]
mask       : {2, 3, 5, 6, 8, 9}   (steps 4 and 7 COINCIDE — both pass cell 0, emit 3)
```
The coincidence holes are why per-pair auto-derivation is non-negotiable here.

## Cross-cutting design lessons

1. **Five edit patterns cover every model in this catalog**: digit-toggle (trigger in/out of a set;
   confound-free when output never copies token values), threshold/tie-straddle δ=1 (read the op's
   strict/non-strict convention first), value-perturb-within-class (the CONTROL pattern — pair
   every value-magnitude confound with its matched control and subtract circuits), position-swap
   (multiset-preserving, zero value confound), compensated two-site edit (shrink suffix → event
   site).
2. **The mask SHAPE is a semantic fingerprint — assert it.** Positionwise map → {p}; pure
   accumulator → suffix {p..9}; resynchronizing machine → window {p..q−1}; comparator/carry →
   sparse; global router → near-full. Per-row validation: mask non-empty (catches dblodd preimage
   pairs, sr12 idempotent edits, holdge5 reg=5), mask within the family's predicted shape, outputs
   equal off-mask. Every degenerate case in this catalog trips one of those three asserts.
3. **Pin vs vary p**: default vary p across the batch (denies the finder a position-special-cased
   answer); pin only when the op anchors it structurally (the position-0 anchor operators
   anc0/fclamp0/drot0/refl0/stride0/pref0/psort0 at 0, cmpswap at 0/9, carryk1 at the tail) or
   for per-position head-role maps.
4. **Budget**: 1 toggle + 1 matched control per model is the default (2 discovery runs). Third
   variants only for genuinely two-latent ops: areset13 (suffix + compensated), sr12 (set + reset),
   argminmax (swap + tiebreak), segrev3 (anchored + presence), walk12 (up + down).
5. **Ill-posed → ProbeCircuit handoffs**: output-invisible latents (dblodd/pmstep preimage pairs,
   dep9f0 clip-vs-exact-zero, fclamp0 clamp-tie x[p]=b) have empty masks by construction — they are
   ready-made datasets for ProbeCircuits ([probe_variable_catalog.md](probe_variable_catalog.md)),
   not ContrastCircuits. Near-full-mask designs
   (drot0/refl0/stride0, cjump pointer, segrev presence) are valid but approach whole-model; the
   informative object is their difference from the pinned DomainCircuit.
6. **Sampler engineering is trivial except twice**: areset13 needs "prefix sum lands in [a,b] at
   step k" (one-line generalization of `bounded_sum_sampler`'s DP); cjump needs simulate-and-check
   on the trajectory. Everything else is constructive.
7. **The pair improves the metric for free**: at every masked position (clean_answer,
   corrupt_answer) is a semantically meaningful token pair — logit-diff against the corrupt answer
   is a designed edit rather than an arbitrary in-domain neighbour. The resample foil's
   zero-contrast slots are no longer *scored* (plain pools carry an `answer_mask` too), but they
   are still uninformative attribution.
