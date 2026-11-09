# Circuit types

Reference for the circuit taxonomy as implemented in [`pipeline/`](../../pipeline/): what each
class is, how it is declared in code, which data it is found on, and how the `smallest` stage
selects its circuit. The introduction to the five classes is
[circuit_classes.md](../circuit_classes.md); the definitions of a circuit, of edge-level mean
ablation and of the faithfulness floors used for the showcase are in [circuits.md](../circuits.md).
Zoo-wide statistics: [results/zoo/stats.md](../../results/zoo/stats.md).

Worked-example numbers below are single rows of the zoo index
([`results/zoo/circuits_index.csv`](../../results/zoo/circuits_index.csv),
[`results/zoo/probes_index.csv`](../../results/zoo/probes_index.csv)); each row records the
selection floor and evaluation size it was produced under (`floor`, `n_test`).

## 1. The five classes

| class | code | activated by | clean / corrupt | measured at | reported object |
|---|---|---|---|---|---|
| **DomainCircuit** | `DomainCircuit` ([`models/circuits.py`](../../pipeline/models/circuits.py)) | conditioning on an input domain on which the model collapses to a simpler algorithm | both drawn from the conditioned domain (no designed edit) | output logits | the winning edge set |
| **ContrastCircuit** | `ContrastCircuit` | one deliberate edit that toggles a causal predicate | predicate present / predicate flipped at one position | output logits, only at slots where the two true outputs differ | candidate − base DomainCircuit (the *fringe*; zoo convention — the showcase reports the full candidate, fringe alongside; see [circuits.md §6.1](../circuits.md)) |
| **SuppressionCircuit** | `SuppressionContrastCircuit` (built on demand by `suppression_circuits_for`) | swapping which task a multiplexed model is cued for | cued task's cue + payload / another task's cue + same payload | output logits, where the two tasks disagree | candidate − the cued task's own circuit (the fringe; zoo convention — the showcase reports the full candidate, fringe alongside; see [circuits.md §6.1](../circuits.md)) |
| **ProbeCircuit** | `ProbeVariable` ([`probes/variables.py`](../../pipeline/probes/variables.py)) | an internal variable the output never prints | uniform inputs for fitting the probe; for circuit finding, clean/corrupt pair pools (resample pairs or single-digit edit pairs, see [circuit_classes.md](../circuit_classes.md#probecircuit)); for evaluation, the single-digit edit that makes the probe's class most decodable | a linear probe at one layer | the edge set feeding the probe |
| **BridgeCircuit** | `BridgeVariable`, made by `bridge_variable(composite_id, k)` | the intermediate value B(x) of a composed model A(B(x)) | as ProbeCircuit | a linear probe at one layer | as ProbeCircuit |

Earlier versions of the code and of the zoo index used short labels. The reference docs use them
only where the code still does (below):

| class | old label |
|---|---|
| DomainCircuit | C0 |
| ContrastCircuit | C1 |
| SuppressionCircuit (also called *routing*) | C3 |
| ProbeCircuit, BridgeCircuit | C2 (bridge = C2 sub-kind) |
| positional finder option (§9; not a class) | C4 |

Code still uses mode strings for the two fringe selection modes (`mode: differential`,
`mode: suppression`; plain rows carry no mode), a few flag names such as
`--print-c3-worklist`, and the old labels as the section keys of
[`pipeline/repo_tools/arch_coverage.py`](../../pipeline/repo_tools/arch_coverage.py)
(`c0`–`c4`, plus `bridge`).

Class hierarchy:

```
DomainCircuit                         pipeline/models/circuits.py
  └── ContrastCircuit
        └── SuppressionContrastCircuit   (registered per mux model, on demand)

ProbeVariable                         pipeline/probes/variables.py   (parallel hierarchy)
  └── BridgeVariable                   (factory-made by bridge_variable)
```

Two things that are **not** classes:

- **Mux** is a property of a *model* (§8). A mux model yields N DomainCircuits and N(N−1)
  SuppressionCircuits; there is no `MuxCircuit`.
- The **positional option** (§9) is an experimental finder setting, not a sixth class.

One showcase circuit per class lives in [`results/showcase/`](../../results/showcase/): Schmitt
trigger without its hysteresis band (`schmitt_id__moe8` / `schmitt_noband`), the dip detector
(`mexhat_id__moe8` / `mexhat_fill`), the parity-step/latch multiplexer
(`mux2.pmstep-id.sr12-id__moe8` / `suppress_t1_over_t0`), gravity's centre of mass
(`graviclump_id__moe4` / `graviclump_mean`) and the novelty bridge inside psum∘novel
(`psum_novel_id__moe8` / `bridge1_novel_id`).

## 2. Shared machinery

**Stages.** `run_model_pipeline.py --stages all` runs `data → train → eap → smallest → audit`
(`DEFAULT_STAGES`); `probes`, `patching`, `edges`, `acdc` and `edge_pruning` run only when named.
Each analysis stage runs once per declared circuit; `data` and `train` run once per model. The
pipeline emits **sub-circuits only**: a model with no declared circuit produces nothing.

**Data.** The `data` stage writes one input pool per circuit,
`pipeline/models/gen_data/<model>/<circuit>/inputs.json`
(`models/gen_data/gen_circuit_data.py`). `model_registry.load_circuit_pairs` splits the pool into
clean/corrupt pairs at load time and hands each consumer a disjoint row window
(`model_registry.POOL_WINDOWS`: attribution from row 0, the EAP-IG ladder search from row 5000,
final scoring from row 10000), so a winner is never scored on the rows it was picked on.

**Finders.** Each finder writes a ladder of candidate circuits under
`circuits_data/<model>/<circuit>/circuits/`:

| finder | default? | candidates | parameters |
|---|---|---|---|
| EAP-IG (`find_circuits/eap.py`) | yes | `eap_top{n}` | attribution pinned at `attr_bs=50`, `attr_n=200`, `ig_steps=30` (scores depend on the batch size) |
| ACDC (`find_circuits/acdc.py`) | when named | `acdc_thresh_{t}` | `canon.CANON_TAUS` = 0.2, 0.1, 0.055, 0.02 |
| Edge pruning (`find_circuits/edge_pruning.py`) | when named | `edge_pruning_sparsity_{s}` | `canon.CANON_EDGE_PRUNING_SPARSITIES` = 0.7, 0.9, 0.95, 0.98 |

**The EAP-IG ladder is a regime, not a list of rungs.** With `CANON_TOPN_REGIME = "frac"`
([`data_gen/canon.py`](../../data_gen/canon.py)) the rungs are `CANON_TOPN_FRACS` =
0.02, 0.05, 0.15, 0.40 of the model's live edge budget, so no rung can cover the whole graph:

| architecture | live edges | `eap_top{n}` rungs |
|---|---|---|
| small | 81 | 2, 4, 12, 32 |
| large | 1281 | 26, 64, 192, 512 |
| wide | 4865 | 97, 243, 730, 1946 |
| deep | 5633 | 113, 282, 845, 2253 |
| xl | 22017 | 440, 1101, 3303, 8807 |
| mlp / moe4 / moe8 | 1519 | 30, 76, 228, 608 |

The absolute ladder `CANON_TOPNS` = 50, 200, 800 survives as the `"abs"` regime, used only to
reproduce candidates produced before the fractional regime existed (the zoo index records which
regime a row came from). In addition, every `smallest` run adds one EAP-IG candidate from
`eap.py --target-seq-acc` (`--search`), which scans a geometric set of top-*n* values from one
attribution and keeps the most compact one above the target.

**Edge universe.** The full graph's size is pure geometry: `n_edges_full =
model_registry.live_edges(spec)` (= `canon.live_edge_budget`) and `n_nodes_full = n_layers ·
n_heads` (+ `n_layers` with an MLP/MoE block). The finders' own graph objects count different
universes at the same geometry (edge pruning, ACDC and EAP-IG report 1184 / 1377 / 1519 edges at
4L×8H attention-only); EAP-IG's graph also wires an MLP node per layer into attention-only models,
and those edges are excluded from the live budget. Formula and table: [circuits.md §1](../circuits.md).

**Faithfulness in the `smallest` stage.** Every candidate is scored by whole-sequence exact-match
accuracy under edge-level, per-position mean ablation (each reader's input is the per-position
mean plus Σ over declared incoming edges of (live − mean); undeclared edges carry the sender's
mean), on 5000 pairs (`SMALLEST_EVAL_N`) from the scoring window. What counts as eligible and
which candidate wins depends on the class (§3–§5). The **showcase** circuits are scored instead by
[`pipeline/find_circuits/evaluate_circuit.py`](../../pipeline/find_circuits/evaluate_circuit.py),
which implements the per-class faithfulness definitions and floors of [circuits.md](../circuits.md)
(DomainCircuit 0.90; the paired classes 0.80 with *a* = *a*<sub>clean</sub> × *a*<sub>corrupt</sub>;
interchange patching for ContrastCircuits; normalized probe accuracy for Probe/BridgeCircuits).
The two scoring paths are not interchangeable; [circuits.md §6](../circuits.md) lists what is
comparable with what.

**The training gate is separate.** The 0.95 sequence-accuracy gate (`SEQ_ACC_GATE`) decides
only whether a checkpoint is kept or retrained; it never enters circuit selection.

## 3. DomainCircuit

**Definition.** Restrict the input domain until the model's task collapses to a simpler
algorithm that can be written down exactly, then find the smallest circuit that still performs it
on that domain.

**Declaration.** A subclass of the model's DSL operator *and* the `DomainCircuit` mixin, with
`circuit_id`, `algorithm` and a `_cond` (a `conditions.Condition`: a predicate plus an
exact-uniform sampler over the domain it names). Overriding `eval` declares the collapsed
algorithm; leaving it alone declares a pure domain restriction. `circuits.validate` draws 2000
in-domain rows and raises unless `circuit.eval(x) == parent.eval(x)` on every one, so the
collapse claim is machine-checked. Defining the class registers it in `circuits.CIRCUITS`;
`repo_tools/make_circuits_txt.py` validates every registered circuit and writes the catalog
`pipeline/models/circuits.txt`. How to declare one by hand: [declaring_circuits.md](declaring_circuits.md).

```python
class ReverseOnlyCircuit(SegmentReverse, DomainCircuit):
    circuit_id = "reverse_only"
    algorithm = "segrev3_id"
    _cond = conditions.exclude_digit(3)          # no boundary digit -> one segment
    def eval(self, input_seq):
        return Reverse(Id(self.input_length)).eval(input_seq)
```

**Scope has two independent axes.** A DomainCircuit's scope is the pair (condition, target
algorithm), and the two can be narrowed separately:

| axis | question | code marker |
|---|---|---|
| input restriction | does the condition narrow the input distribution? | `_cond` / `condition()`; a condition accepting essentially every uniform row is *vacuous* |
| task restriction | is the target a simpler algorithm than the model's? | `eval` defined in the subclass |

| input | task | example |
|---|---|---|
| unrestricted | unrestricted | the `*_whole` / `*_full` family (e.g. `zc_whole`) |
| restricted | unrestricted | `one_odd`, `maxrun_onepair`, every mux `task{i}` |
| restricted | collapsed | `reverse_only`, `all_odd`, `bwt`, `sarank` |
| unrestricted | collapsed | impossible — the override would have to equal the parent everywhere |

An unrestricted-input scope is a whole-model circuit in all but name: the circuit must reproduce
the full algorithm on the full input distribution. Declaring a collapsed `eval` lets you *state
what the circuit computes* and gets it machine-checked.

**Counterfactual: none by design.** The pool holds in-domain rows only; the loader splits it in
half and treats the second half as the "corrupt" partner (`model_registry._split_inputs`). The
EAP-IG metric is the logit difference against the partner row's true answer, i.e. an arbitrary
in-domain neighbour. Positions where a clean row and its partner have the same true output carry
no contrast and are masked out (`answer_mask`, `model_registry.dataset_to_pairs`); a row that
agrees everywhere is dropped, and a scope where every row agrees at every scored position raises
`zero-contrast isolation` instead of reporting 0/0.

**Selection (plain mode, `smallest_faithful_circuit._passes`).** A candidate is eligible iff it
keeps at least one node and

1. its absolute whole-sequence accuracy is ≥ `--target` (`DEFAULT_TARGET = 0.85`),
2. its relative recovery R = (acc<sub>circuit</sub> − acc<sub>empty</sub>) / (acc<sub>full</sub> −
   acc<sub>empty</sub>) is a number and ≥ `--relative-floor` (`DEFAULT_RELATIVE_FLOOR = 0.85`), and
3. the empty circuit's accuracy is **below** `--target`, so a near-constant-output domain cannot
   credit a circuit for an output it does not compute.

The winner maximizes (1 − n<sub>nodes</sub>/n<sub>nodes_full</sub>)·(1 − n<sub>edges</sub>/n<sub>edges_full</sub>);
ties go to fewer edges, then fewer nodes, then higher accuracy, then name. These are the `smallest`
stage's selection thresholds, not the showcase floors (§2).

**Example — `dblodd_id / all_odd`.** `dblodd_id = DoubleOdd(Id(10))` doubles odd digits mod 10
and passes evens through. `_cond = conditions.ALL_ODD` restricts inputs to `{1,3,5,7,9}^10`, so the
parity gate is always on and the operator collapses to `MapMul(2)`:

```
condition        every digit odd
collapsed eval   (2*x) mod 10, position-wise        [1,3,5,7,9,...] -> [2,6,0,4,8,...]
winner           eap_top38      8 heads / 17 edges
accuracy         0.945           selection_score 0.739
```

Sibling on the same model: `one_odd` (exactly one odd digit, a localize-and-transform scope) won
at `eap_top50` with 11 heads / 31 edges, selection score 0.643.

| further examples | winner | heads / edges |
|---|---|---|
| `segrev3_id / reverse_only` — no digit 3, so the whole sequence reverses | `acdc_thresh_0.2` | 7 / 18 |
| `pmaxv9_id / pmaxv9_no9` — no 9s, so a plain running max | `eap_top150` | 20 / 138 |
| `maxrun_id / maxrun_onepair` | `eap_top534` | 32 / 534 |

Mux `task{i}` circuits are ordinary DomainCircuits whose condition pins the cue token
(`conditions.task_pin(i, n)`).

## 4. ContrastCircuit

**Definition.** Two nearly identical inputs differing by one deliberate edit that toggles a
predicate. Only the output slots where the two true outputs differ are scored. The circuit is the
machinery the toggle recruits on top of a base circuit already known from a domain restriction —
the detector, not the executor.

**Declaration.** A `ContrastCircuit` subclass adds `sample_pairs(batch, seed)`, names a sibling
`base_circuit`, and does **not** override `eval`: keeping the true operator on both sides makes
the per-row mask derivable. Datasets store the sentinel `answer_positions: "diff"`, and
`model_registry._diff_answer_slots` recomputes each row's differing slots at load time.
`circuits.validate_pairs` rejects any pair whose inputs are identical or whose true outputs do
not differ (the degenerate-contrast trap: `DoubleOdd` maps both 1 and 2 to 2), and checks an
optional `affected_outputs` claim. `local_predicate` declares whether the toggled predicate is
consumed only at the mask positions. The first line of `sample_pairs` threads
`conditions.stream_key(seed, circuit_id)` so sibling scopes of one model draw independent rows.

**Base circuit and fringe.** On a clean/corrupt pair, attribution returns the union of the
detector and the executor that produces the output at all; the executor is already known as the
base DomainCircuit, so subtracting it leaves the detector. Mechanically:

```python
base_keys = {(e["parent"], e["child"], e["qkv"]) for e in base_edges}
fringe = [e for e in edges if (e["parent"], e["child"], e["qkv"]) not in base_keys]
```

applied to every raw candidate *before* selection (`smallest_faithful_circuit._differential_metrics`).
Consequences:

- The base is read from `<base_circuit>/smallest_circuit.json`, so contrast circuits are built
  after their base (`run_model_pipeline._circuit_objs`). The base is a snapshot, fingerprinted by
  `base_edges_sig` on every fringe row, so a re-selected base invalidates the cached fringe.
- An unfaithful base does not block the row: it falls back to its best available candidate and
  the output records `base_faithful: false`.
- **The reported heads/edges are the fringe**, not the candidate (zoo selection stage). The
  accuracy beside them is the whole candidate's, recorded as `source_candidate_absolute_accuracy`,
  because a fringe has no standalone task accuracy.

**Selection (differential mode, `--base-circuit`, `_passes_differential`).** Eligible iff the
fringe is non-empty and `s_union`, the whole-sequence accuracy of fringe ∪ base, is ≥ `--target`.
The winner maximizes (1 + clamp(d<sub>affected</sub>, 0, 1)) · (1 − (n<sub>fringe nodes</sub>/n<sub>nodes_full</sub>)·(n<sub>fringe edges</sub>/n<sub>edges_full</sub>)).
Severing uses the dual edge forward (fringe edges carry the mean, everything else live). Every
metric is whole-sequence exact match over its position set:

| metric | meaning | role |
|---|---|---|
| `s_union` | accuracy of fringe ∪ base | gates eligibility |
| `d_affected` | accuracy drop at the mask positions when the fringe is severed | bonus in the score (at most 2×) |
| `d_unaffected` | collateral drop off the mask | recorded |
| `d_control` | drop on the base circuit's own pool when the fringe is severed | recorded |

Because eligibility keys on `s_union`, a selected fringe is *sufficient in union with its base*;
whether it is also *necessary* is what `d_affected` says. Any claim that a fringe "is the
detector" should quote its `d_affected`.

**Showcase evaluation differs.** The showcase ContrastCircuit is scored by interchange patching
(each row's baseline is its partner's activations, a shared teacher-forced prefix, labels from the
full model). Under that evaluation everything the edit does not change is identical in both runs,
so the fringe subtraction is not needed for *evaluation*; patched faithfulness is not comparable
with the differential mode above. See [circuits.md §4.1](../circuits.md). The showcase reports the
full candidate as the circuit, with the fringe against the base alongside (`evaluate_circuit.py`,
[circuits.md §6.1](../circuits.md)).

**Example with sufficiency but no necessity — `dblodd_id / dblodd_parity_toggle`.** Base
`all_odd`. Flip one digit's parity at a random position with the partner table
`[1,0,3,2,5,4,7,6,7,0]`, which avoids the 1↔2 and 8↔9 preimage collisions that would produce an
empty mask:

```
winner           eap_top50     fringe = 6 heads / 4 edges
s_union          0.977
d_affected       0.000       d_unaffected 0.000      d_control 0.000      local_predicate True
```

A compact fringe, but `d_affected = 0`: severing it does not degrade the masked outputs.

**Example with necessity — `indecho_id / indecho_matchbreak`.**

```
base   indecho_alldistinct      winner eap_top600     fringe = 32 heads / 559 edges
s_union 0.996    d_affected 0.500    d_unaffected 0.001    d_control 0.012    local True
```

Necessity and compactness trade off: this fringe is necessary for half the masked outputs but
uses every head.

**Example of a non-local predicate — `segrev3_id / three_detector`.** Clean = any row containing
at least one 3; corrupt = one uniformly chosen 3 replaced by a uniform non-3, i.e. one segment
boundary deleted (`local_predicate = False`: segment membership feeds every output position).

```
clean   [4,9,3,0,3,9,7,3,7,3] -> [9,4,3,0,3,7,9,3,7,3]
corrupt [4,9,3,0,6,9,7,3,7,3] -> [9,4,3,7,9,6,0,3,7,3]     mask = {3,4,5,6}
```

Its zoo row records no faithful fringe at the floor it was selected under (`floor` 0.9,
`s_union` 0.891), although the base `reverse_only` won at 7 heads / 18 edges. The anchored variant
`segrev3_second_boundary_toggle` pins boundaries at `p` and `q > p`, toggles only `q`, and
rejects a pair unless the mask is exactly the fenced suffix `{p+1..9}`.

Per-model pair designs: [contrast_pair_catalog.md](contrast_pair_catalog.md).

## 5. SuppressionCircuit

**Definition.** In a multiplexed model, the edges retained not to compute the cued task's answer
but to hold a *specific* competing task's answer down. Structurally it is a ContrastCircuit (a
fringe over a base); the difference is the comparison: a ContrastCircuit wants the correct answer
to **drop** when the fringe is severed, a SuppressionCircuit wants a named wrong answer to **rise**.
That fringe is what the zoo selection stage reports. The showcase SuppressionCircuit is the full
candidate, which has to compute both tasks and route between them on the cue
([circuits.md §4.3](../circuits.md)); its fringe beyond the cued task's circuit is reported
alongside.

**Pairs.** `clean = [10+cued] + payload`, `corrupt = [10+wrong] + same payload`; rejection keeps
only payloads where the two member tasks disagree somewhere, and those slots are the mask
(`SuppressionContrastCircuit.sample_pairs` / `affected_outputs`). Pairs are ordered, so an N-task
mux has N(N−1) of them (`suppression_circuits_for`), each named `suppress_t{cued}_over_t{wrong}`
with `base_circuit = task{cued}`. Circuit ids carry no mux identity, so the key is
`(base_algorithm, circuit_id)`.

**Mux-only, enforced in code at four points.** (1) `suppression_circuit()` /
`suppression_circuits_for()` raise `CircuitError` unless the parent decodes to a `TaskMux`;
(2) the class body assumes a mux (`condition()` tests `input_seq[0] == 10 + cued_task`;
`sample_pairs()` and `affected_outputs()` read `self.members`); (3) a suppression row needs every
task circuit's artifacts first; (4) the router verdict needs `task_router/<model>/diagnostic.json`.
The reason: a release rate needs a named wrong answer, which a mux supplies for free (the other
member's output on the same payload). A single-algorithm model has no second algorithm, so the
forbidden output would have to be written by hand. The nearest single-algorithm case,
`pmaxv9_id` (running max where 9 is disqualified from being the record), declares its veto
machinery as a plain ContrastCircuit, `pmaxv9_disqualify` (clean `x[p]=8`, a real new record;
corrupt `x[p]=9`, disqualified).

**Selection (suppression mode, `--suppression --base-circuit task{i}`, `_passes_suppression`).**
Rates are whole-sequence: a release counts when *all* of a row's affected slots flip.

```
release_rate = wrong_task_wins(fringe severed) − wrong_task_wins(full model)
eligible iff  n_fringe_edges > 0
              AND ( (release_rate       ≥ 0.5 AND collateral       ≤ 0.05)
                 OR (release_rate_tonic ≥ 0.5 AND collateral_tonic ≤ 0.05) )
score        = clamp(release of the reference that passed, 0, 1)
               · (1 − (n_fringe_nodes/n_nodes_full)·(n_fringe_edges/n_edges_full))
```

(`DEFAULT_RELEASE_FLOOR = 0.5`, `DEFAULT_EPSILON = 0.05`; `collateral` is measured on the positions
where the two tasks agree.) `_release_reference` names the qualifying reference — the larger
release when both pass — and the output records it as `release_reference`.

The **tonic** reference is a second sever pass with means computed over *wrong-cue* rows. A
suppressor that is constant across cues equals its own cued-row mean, so mean-ablating it against
cued-row means changes nothing and `release_rate ≡ 0`; the tonic pass closes that blind spot. In
the zoo snapshot, all 125 faithful SuppressionCircuits pass only under the tonic reference
([results/zoo/stats.md](../../results/zoo/stats.md), §6).

**Not negative-sign edges.** Selection never consults attribution sign: on an A-vs-B margin, sign
cannot separate a suppressor (writes −B) from an A-promoter ([design.md](design.md)).
`edge_signs/edge_sign_census.py` is a separate, descriptive census of edge signs.

**Routing diagnostic** (`task_router/router_diagnostic.py`). For every ordered pair (A, B) on
A-cued inputs: `fringe` = B's circuit edges − A's; `fringe_on_ratio` = the fringe heads' mean |z|
on A-cued inputs ÷ on B-cued inputs; `margin` = logit(B(x)) − logit(A(x)) at the disagreeing slots,
also as a per-layer logit-lens trajectory.

| verdict | criterion |
|---|---|
| superposition | wrong answer live: `margin ≥ 0` or wrong-answer top-1 rate > 0.5 |
| gating | fringe off: `fringe_on_ratio < 0.5` |
| outvoting | fringe on, wrong answer suppressed |
| empty-fringe | B's heads ⊆ A's |
| logit-lens-only | no circuits available for the pair |

The verdict is recorded on the suppression row (`router_verdict`), never used as a gate.

**Example — `mux5.psum-id.cmpswap0p9-id.jump10-id.pmstep-id.segrev3-id / suppress_t1_over_t3`.**

```
base task1     winner eap_top800     fringe = 31 heads / 604 edges
                      cued reference      tonic reference
release_rate              0.0025             0.9828
collateral                0.2028             0.0000
wrong_task_wins (full)    0.000
router_verdict  outvoting        release_reference tonic
```

Severing against cued-row means changes nothing; severing against wrong-cue means flips the model
almost entirely onto the wrong task. A fringe this large is close to the whole model, which limits
how much "this is the suppressor" the release number can support.

## 6. ProbeCircuit

**Definition.** A quantity the algorithm must compute but never emits. A linear readout of the
residual stream either recovers it at a low layer (the model keeps a *register*, an "eager"
reading) or does not (the value is recomputed at readout, a "lazy" reading). `eager`/`lazy` are
readings of the verdict plus `first_decodable_layer`; they do not appear in the code.

**Declaration.** A `ProbeVariable` subclass sets `variable_id`, `algorithm`, `n_classes` and
implements `labels(inputs)` (a pure-Python DSL replay, never the model), `valid_positions()`
(input positions where the variable is defined), `check_row(x, lab)` (raises unless the labels are
jointly consistent with the true output) and optionally `counterfactual(x, p, new_class)` (the true
output had the variable been `new_class` at `p`; needed for the causal check below).
`variables.validate` draws 2000 uniform rows and raises on out-of-range labels, a constant label at
a declared position, a position whose label moves when its suffix is resampled (no causal residual
stream could hold it), or any failing `check_row`.

**Instrument** ([`probes/train_probes.py`](../../pipeline/probes/train_probes.py)). Multinomial
logistic regression (LBFGS, 25% row holdout, weight decay 1e-3) on the LayerNorm'd residual, fit
**per (layer, position)**: a probe pooled across positions hides position-rotating codes, so the
per-position probes carry the verdict. Verdict sites are `blocks.0.hook_resid_pre` (the baseline:
"is it already in the embedding?") and `blocks.{l}.hook_resid_post`; `--sites components` (default)
adds `hook_attn_out` / `hook_mlp_out` per layer for localisation only. Inputs are 4096 uniform
random rows (`--n`).

**Two position sets, two verdicts.** Input positions (causally trimmed declared positions) carry
`verdict`. Output positions `input_len..n_ctx−1` carry `gen_verdict`, one fit per (declared
position × decode step), each declared position taking its best decode step. The forward
free-runs on the model's own argmax by default (`--decode free`), and the output-position layer-0
baseline controls for what the just-emitted token gives away (its trivial case is reported as
`trivial_from_emitted_output`).

**Verdict rule** (`train_probes.verdict`, position-mean accuracies, `TRIVIAL_ACC = 0.9`,
`ERR_REDUCTION = 0.5`):

```
floor = 1 − 0.5·(1 − majority)
trivial_from_input   iff  baseline ≥ 0.9 and baseline ≥ floor
decodable at layer l iff  1 − acc_l ≤ 0.5·(1 − baseline)  and  acc_l ≥ floor     (first such l)
absent               otherwise
```

Both errors must halve: the probe error relative to the embedding baseline, and the majority-class
error. A fixed additive margin (e.g. "baseline + 0.1") would be unsatisfiable at high baselines.

**Validation** ([`probes/validate_probe.py`](../../pipeline/probes/validate_probe.py), report
only): a permutation control (refit on shuffled labels; should sit at chance), a leakage matrix
(each probe's predictions mapped onto sibling variables' labels), robustness over three seed /
weight-decay refits, and a **causal check**: patch the residual along
`W_pos[p][target] − W_pos[p][source]`, scaled by α ∈ (0.5, 1, 2, 4, 8) × the row's residual norm
until the probe reads the target class post-LN, then compare the model's output with the DSL
counterfactual. It reports `steered_rate` (the probe moved — near-tautological, since the probe
reads its own steering direction) and `flip_rate` / `collateral_rate` (the model's output moved
at the toggled slot / elsewhere); the reported pass criterion is `flip_rate ≥ 0.80` and
`collateral_rate ≤ 0.05`. Variables without `counterfactual()` skip the causal check.

**Finding a ProbeCircuit.** `eap.py`, `acdc.py` and `edge_pruning.py` take `--probe <variable>`
(with `--probe-layer`, `--probes-root`): the graph's `logits` node is retargeted to the probe
site, every edge whose writer sits above the probe layer is cut (it cannot affect the probe), and
the metric becomes KL(clean probe log-probs ‖ current probe log-probs) averaged over the probed
positions. The probe layer must be below the last layer. Candidates are written as
`eap_probe_<variable>_top<n>`. The `smallest` stage skips probe-metric candidates (they are not
comparable on the output metric); ProbeCircuits are scored by `evaluate_circuit.py` with
normalized probe accuracy — see [circuits.md §4.2](../circuits.md).

Two details matter when reading a probe circuit:

- The verdict and the causal check use the per-position probes (`W_pos`). The finder reads the
  pooled probe (`W`, `b`) by default, or each probed position's own probe with
  `eap.py --probe-readout position`, which is what the showcase uses. Both accuracies are stored
  (`accuracy`, `pooled_accuracy`). Score a probe circuit with the readout it was optimized on.
- `eap.load_probe_artifact(layer=None)` picks the highest-accuracy site, which need not be
  `first_decodable_layer`; pin `--probe-layer`.

**Worked example — `maxrun_id / maxrun_cur`.** `maxrun_id = Maxrun(Id(10))` emits the longest run
so far (capped at 9). The latent is the **current** run length, which the output never exposes:

```
x    = [4, 4, 7, 7, 7, 2, 9, 9, 9, 9]
y    = [1, 2, 2, 2, 3, 3, 3, 3, 3, 4]      emitted: longest run so far
cur  = [-, -, 1, 2, 3, 1, 1, 2, 3, -]      probed: current run, positions 2..8
```

```
majority_class_rate  0.9008     cur = 1 whenever a digit differs from its predecessor
baseline (L0 pre)    0.9009     the embedding knows x[p] and p, not x[p-1]
per-layer mean acc   L0 0.9957   L1 0.9968   L2 0.9961   L3 0.9962
floor                0.9504     = 1 - 0.5*(1 - 0.9008)
trivial?             baseline 0.9009 >= 0.9 but < 0.9504  -> no
decodable?           1 - 0.9957 = 0.0043 <= 0.0496 and 0.9957 >= 0.9504  -> yes, first_decodable_layer 0

validation at layer 0
permutation control      0.9001          (chance = majority rate)
probe_reads_source_rate  0.9961
steered_rate             0.9883          mean_alpha 0.883
flip_rate                0.7589          collateral_rate 0.0356
suffix_consistent_rate   0.3043
```

The current-run-length register is linearly readable from layer 0 and causally load-bearing at
the toggled slot most of the time, but `flip_rate` is just below the 0.80 criterion and the
downstream suffix follows the steered value only about a third of the time. An additive
"baseline + 0.1" rule would have demanded 1.0009 — unsatisfiable — which is why the verdict
halves errors instead.

Per-model variable designs: [probe_variable_catalog.md](probe_variable_catalog.md).

## 7. BridgeCircuit

`bridge_variable(composite_id, k)` strips the **k outermost** operators of a composite
`A(B(x))` and probes the remainder B(x) as a `ProbeVariable`; it is registered on creation and
idempotent. Decodable means the model materializes the intermediate value (composition as
stages); absent means the operators have fused into one map.

Guards in `variables.py`:

- **Load-bearing slots only.** `_bridge_positions` keeps intermediate slots that are causally
  determinate, non-constant, and reach the composite output (`_graft_matters`); a split with no
  such slot raises.
- **Identified depth.** `_split_is_identified` rejects a split depth whose intermediate another
  depth reproduces on those slots.
- **Graft check.** `check_row` grafts the *claimed* labels back into the outer chain and requires
  the composite to reproduce the true output, and requires the output to depend on the slot.
- **Measured alphabet.** `n_classes` is measured from 2048 sampled label rows, not assumed to be 10.
- `split_k` that strips every operator (leaving `Id`) raises: the "intermediate" would be the input.

```
dblodd_walk10_recordind_id = DoubleOdd(SignedWalk(1, 0, RecordIndicator(Id(10))))
  k=1 -> SignedWalk(1, 0, RecordIndicator(Id(10)))     bridge1_walk10_recordind_id
  k=2 -> RecordIndicator(Id(10))                       bridge2_recordind_id
```

Both are registered at import (bottom of `variables.py`); `train_probes.py --variables bridge{k}`
creates others on demand. BridgeCircuits are found and scored exactly like ProbeCircuits (§6).
Which bridges stay decodable at which architecture is in the zoo index
([`results/zoo/probes_index.csv`](../../results/zoo/probes_index.csv), `class` = BridgeCircuit,
column `verdict`).

## 8. Mux models

A `TaskMux` model is one transformer trained on N DSL algorithms: input token 0 is the cue
`10 + i`, followed by the 10-digit payload, and the model must run member task i on the payload
(input length 11, vocabulary 10 + N; id format `mux{N}.{m1}.{m2}...`). It yields:

- N DomainCircuits `task{i}` with the cue pinned (`task_circuits_for`, `conditions.task_pin`),
  the base executors, and
- N(N−1) SuppressionCircuits `suppress_t{a}_over_t{b}` (§5).

## 9. Positional finder option

`--use-pos-embed --zero-ablation` is an experimental finder option that attributes positional
rather than token-value routing (the input source moves from `hook_embed` to `hook_pos_embed`, and
edges are ablated to zero). Its candidates (`*_zero`, `*_pos`, e.g. `eap_top{n}_zero_pos`) are not
comparable on the output metric and are excluded from selection
(`smallest_faithful_circuit._is_variant_candidate_dir`). It is not a circuit class.

## 10. Architectures

`TIER_PRESETS` in [`interp_tools/model_loading.py`](../../interp_tools/model_loading.py); the
model-name suffix selects the variant (`__small`, `__wide`, …; no suffix = `large`). All
attention-only unless noted:

| variant | layers | heads | hidden size | d_head | feed-forward |
|---|---|---|---|---|---|
| `small` | 2 | 4 | 64 | 16 | — |
| **`large`** (default) | **4** | **8** | **128** | **32** | — |
| `wide` | 4 | 16 | 256 | 32 | — |
| `deep` | 8 | 8 | 128 | 32 | — |
| `xl` (`xlarge`) | 8 | 16 | 256 | 64 | — |
| `mlp` | 4 | 8 | 128 | 32 | `d_mlp = 512`, GELU |
| `moe4` | 4 | 8 | 128 | 32 | `d_mlp = 128`, 4 experts, top-1 |
| `moe8` | 4 | 8 | 128 | 32 | `d_mlp = 128`, 8 experts, top-1 |

Escalation-only twins at xl geometry keep the architecture flag (`mlp_xl` with `d_mlp = 1024`,
`moe4_xl`, `moe8_xl` with `d_mlp = 256`) for variants that cannot reach the training gate at their
base size. An MoE block enters the circuit graph as one node per layer (`m{l}`), not one node per
expert. Training recipes: [models.md](../models.md).

## 11. Open design questions

| item | state |
|---|---|
| ContrastCircuit necessity | selection keys on `s_union`; `d_affected` is a bonus, never a gate, so a selected fringe need not be necessary |
| SuppressionCircuit controls | no zero-ablation control is built; large fringes approach the whole model |
| single-algorithm suppression | the concept transfers (e.g. the un-vetoed running max for `pmaxv9`), the implementation is mux-only |
| ProbeCircuit readout | verdicts use per-position probes; finders read the pooled probe unless given `--probe-readout position` |
| ProbeCircuit causal check | skipped for variables without `counterfactual()` (all bridges) |
| MoE granularity | MoE is one `m{l}` node per layer; per-expert nodes are not built |
| feed-forward keep-set | a plain `smallest_circuit.json` lists `heads` but has no MLP field; kept `m{l}` nodes are recoverable only from the edge list |
| EAP-IG `--search` predicate | `seq > target`, not the three-way `_passes`; unifying them is undecided |
