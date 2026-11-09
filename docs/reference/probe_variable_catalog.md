# Probe-variable catalog

Per-model internal variables for ProbeCircuits ([circuit_types.md §6](circuit_types.md)):
DSL-computable latents to probe on the residual stream, with what a probe-metric circuit would show
that the output-metric circuit cannot. Semantics follow the `eval()` implementations in
[`pipeline/models/algorithms.py`](../../pipeline/models/algorithms.py); declared variables live in
[`pipeline/probes/variables.py`](../../pipeline/probes/variables.py).

Shared facts for the default `large` architecture: 4 layers × 8 heads, hidden size 128, LNPre,
attention-only; token stream `[x0..x9 | y0..y_{L-1}]`; verdict sites =
`blocks.0.hook_resid_pre` (baseline) + `blocks.{0..3}.hook_resid_post`.

**Two phases, two verdicts.** `train_probes.py` scans the input positions (→ `verdict`) and the
generation positions `input_len..n_ctx-1`, one fit per declared position × decode step (→
`gen_verdict`). The zoo index records which phase a variable was decodable in. The
"Output-exposed?" column below is about what the *output* prints; it does not say where the
variable is readable.

**The teacher-forcing confound:** if the forward is fed the DSL ground truth (`--decode teacher`),
then at output position 9+k the token y_{k−1} is in context and any output-exposed state is
trivially readable off it. The default is `--decode free` — the model's own argmax — and the
output-position layer-0 `resid_pre` baseline is the control for what the just-emitted token gives
away either way. Consequences: (a) INPUT positions 0-8 remain the clean site for "eager scan vs
lazy readout"; (b) every probe must beat the layer-0 baseline at the same site; (c) a `gen`
decodability above that baseline means "computed", not "read back".

## Group A — hidden auxiliary state behind an exposed scan (highest value)

| Model | Var | Label | Type | Output-exposed? | Probe adds |
|---|---|---|---|---|---|
| areset13 | `acc(i)` ∈ [0,12] | replay eval, record acc | 13c | partially — output = acc%10, so the "tens bit" 1[acc≥10] is NEVER emitted | the full 13-state register vs its mod-10 shadow |
| areset13 | `crossed(i)` | 1[acc_pre ≥ 13] | bin | never | the comparator/reset trigger — the machinery `areset13_accum` structurally cannot contain |
| segrev3 | `is3(i)` | 1[x_i==3] | bin | only at boundaries | the 3-detector, invisible to reverse_only |
| segrev3 | `seg_idx(i)` | #{j≤i: x_j==3} | 10c | never | the segment address — the reversal's routing key |
| segrev3 | `src(i)` | mirror source seg_start+seg_end−i | 10c | only its VALUE is emitted | address-vs-value split feeding the mover heads |
| recordind | `cur_max(i)` | max(x[0..i]) | 10c | never (output = 0/1 flag) | a full 10-class value computed but never emitted |
| reccount | `cur_max(i)`, `is_record(i)` | as above | 10c, bin | never / Δ-derivable | max-maintenance vs comparator vs counter |
| novel | `seen_d(i)`, d=0..9 | 1[d ∈ x[0..i−1]] | 10× bin | never | the 10-bit set memory; per-digit decodability map |
| walk12 | `n1(i)`, `n2(i)` | counts of 1s, 2s | 10c each | only (n1−n2)%10 | two counters or only the difference? the de-confounding testbed |
| reset3 | `last3(i)` | position of last 3 | 10c | never (output = distance) | pointer vs lazily-recomputed distance |
| tresetmax3 | `last3(i)` / `seg_idx(i)` | as reset3 | 10c | never | segment bookkeeping reuse vs direct attention gating |
| prod | `haszero(i)`; `prod2(i)`, `prod5(i)` | absorbing flag; CRT components | bin; bin,5c | not uniquely; jointly ≡ output | absorbing-state flag; does the model factor mod 10 via CRT? |
| dep9f0 | `floored(i)`; raw `cumsum(i)` | saturation flag; discarded sum | bin; reg | ≈ / discarded | is discarded information really destroyed post-saturation? |

## Group B — data-dependent addressing (pointer / program-counter variables)

| Model | Var | Type | Probe adds |
|---|---|---|---|
| cjump10t5 | `cursor(t)` = i_t (replay the walk) | 10c, sites 9+t only | a genuine program counter; iterative vs unrolled mechanism (see F3) |
| drot0 / refl0 / stride0 | `src(i)` = (i+r)%10 / (2r−i)%10 / (i·k)%10 | 10c | explicit source address vs attention parameterized directly by the anchor |
| pref0 / psort0 | `inprefix(i)`; `rank(i)` | bin; 10c | region-membership gate; the sort permutation |
| anc0 | `anchor(i)` = x_0 carried at i | 10c | pure broadcast localization by layer |
| fclamp0 | `ge(i)` = 1[x_i ≥ x_0] | bin | comparator bit vs select/route |
| cmpswap0p9 | `swap` = 1[x_0 > x_9] | bin, sites ≥ 9 | comparator alone (the output circuit conflates comparator + router) |
| argminmax | `maxval`, `minval` | 10c each | value computation feeding the exposed arg indicators |
| carryk1 | `carry(i)` = 1[x_{i+1..9} all 9] | bin, sites ≥ 9 ONLY (anti-causal in input) | suffix-AND aggregation; the valid-site set is itself a finding |

## Group C — mask/parity detectors (shared-core questions)

`odd(i)` probes on dblodd (calibration: known parity heads), cntodd, psumodd (+ `contrib(i)` =
x_i·odd(i) separating gate from payload), rcnt3 `is3(i)` (full signal on the unrestricted domain,
which the no-3 DomainCircuit cannot provide), holdge5 `qual(i)` + `lastqual(i)` pointer.
Cross-model question: is the parity core literally shared (edge-set IoU across the three probe
circuits)? A site scan showing the psumodd sum decodable only late and diffusely would explain a
large output-metric circuit for that model.

## Group D — skip (probe adds ~nothing)

fixedpoint (positionwise, output = variable), bmax1 (local), pmstep/sr12 (state fully exposed —
input-position depth maps only). rcnt5 and pmax are catalogued and skipped on value.

## Priorities and flagships

Priority: areset13, segrev3, cjump10t5, novel, recordind, then walk12 (methods control), reset3,
carryk1, drot0, psort0, argminmax, cmpswap0p9, …

- **F1 areset13** — 13c `acc` + bin `crossed` probes at all sites, de-confounded label sets
  (crossed ⟺ acc_pre≥13 by construction: match pairs on acc value, differ on reset-en-route).
  The answer can be phase-dependent: a register that is `absent` across the input positions but
  `decodable` during generation, localised to `9+p` (the residual that emits `y_p`), exists but
  is materialised at emission rather than maintained across the input scan — which is why the
  generation scan exists. Next: finders on `crossed` at the winning generation site →
  comparator+trigger circuit; sufficiency: probe-circuit(crossed) ∪ `areset13_accum` ≈ full task
  (a union test, as for ContrastCircuits).
- **F2 segrev3** — hypothesized decodability cascade is3@L0 → seg_idx@L1 → src@L2-3, matching the
  detector/mover split of the [segrev3 case study](../segrev3_case_study.md). Finders on seg_idx =
  the deciding machinery extracted with NO domain restriction; src-circuit minus seg_idx-circuit =
  pure address arithmetic.
- **F3 cjump** — `cursor(t)` decodability by layer distinguishes iterative (fixed shallow layer
  ∀t: reuses the teacher-forced token + one hop) from unrolled (decodable layer grows with t,
  saturates at 4, accuracy falls where decoding stops — should track the model's error curve).
  Probe circuit at large t* = the address chain; output circuit at t* adds the value-lookup hop —
  the edge-set delta IS the dereference.
- **F4 novel** — is the seen-set eagerly maintained (10 directions) or lazily matched
  (induction-style compare of x_i vs history)? Per-digit decodability map; same-heads/different
  value-channels ⇒ superposed set storage. Closest small-scale Othello analog; crispest causal
  check (patch one seen_d direction → only digit-d novelty outputs flip).
- **F5 recordind** — `cur_max` computed but never emitted: the pure hidden-variable case.
  Prediction: output circuit = max circuit + one comparator layer. Cross-model: reccount's cur_max
  circuit (same variable, different output head). De-confound: fit only on non-record positions.

## Pitfalls (all mandatory to handle)

1. **Decodable ≠ used** — run the causal check (`validate_probe.py`): directional patch along the
   probe subspace (r ← r − Πr + Πr̃) at the exact metric site; it passes when ≥80% of matched pairs
   flip exactly the V-dependent output positions (DSL-computable) and ≤5% of V-independent tokens
   change (`flip_rate ≥ 0.80`, `collateral_rate ≤ 0.05`). `steered_rate` (the probe moved) is
   recorded but is not evidence: steering along the probe's own direction and reading it back
   with the same probe measures the patch, not the model. A probe that fails is a reportable
   negative ("decodable, not used"), not a discovery target. Plus a null control: a
   random-rotation probe should attribute ≈ nothing.
2. **Correlated labels** — concrete instances: areset13 crossed⟺acc_pre≥13; walk12 n1−n2 ≡ output;
   psumodd contrib vs odd; recordind cur_max vs x_i at records. Levers (exact, DSL-labeled):
   stratified training slices (V2 held constant), matched contrast pairs for the metric (the
   ContrastCircuit pair machinery), and the cross-probe leakage matrix `validate_probe.py` reports.
3. **LayerNormPre** — probe the LN'd residual (or verify raw probes survive norm shuffling);
   directional patches must be checked post-LN (the probe's own post-patch reading must equal the
   target class before scoring behavior); never compare raw metric scales across layers.
4. **Position-specific probes carry the verdict; a finder reads whichever readout it is given.**
   The verdict and the causal check use the per-position `W_pos`, because a position-shared probe
   hides position-rotating codes. `eap.py --probe` reads the pooled `W`/`b` by default and each
   probed position's own `W_pos`/`b_pos` with `--probe-readout position`. The showcase uses
   `position`. Both numbers travel together in the artifact (`accuracy` vs `pooled_accuracy`), and
   a variable can be decodable per position but not pooled. Report the shared-vs-specific gap as a
   finding, and score any probe circuit with the readout it was optimized on
   (`evaluate_circuit.py --probe-readout`). For monotone
   variables, add a shuffled-prefix control against position-proxy shortcuts.
5. **valid_positions is first-class** — variables computed from future input (carryk1) have no
   valid input-position sites; accidental decodability at an "impossible" site is a label-function
   bug detector. This is enforced at registration: `variables.validate` resamples each declared
   position's suffix and raises if the label moves, so a class cannot register with a
   non-determinate position.
