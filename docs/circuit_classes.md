# The five circuit classes

A circuit class is a recipe for asking one kind of question of a trained model: *which subgraph
does this?* The classes differ in where the signal comes from (a restricted input domain, or a
pair of inputs differing by one edit), where it is measured (output logits, or a linear probe
mid-stack), and what is reported (the candidate circuit, or the candidate minus a base circuit).
Definitions of the graph, the ablation and faithfulness are in [circuits.md](circuits.md).

| class | code (repo class) | file | asks |
|---|---|---|---|
| [DomainCircuit](#domaincircuit) | `DomainCircuit` | [`pipeline/models/circuits.py`](../pipeline/models/circuits.py) | what does the model compute on a restricted input domain? |
| [ContrastCircuit](#contrastcircuit) | `ContrastCircuit` | [`pipeline/models/circuits.py`](../pipeline/models/circuits.py) | what detects one predicate? |
| [SuppressionCircuit](#suppressioncircuit) | `SuppressionContrastCircuit` | [`pipeline/models/circuits.py`](../pipeline/models/circuits.py) | in a multi-task model, what keeps the non-cued task's answer out? |
| [ProbeCircuit](#probecircuit) | `ProbeVariable` | [`pipeline/probes/variables.py`](../pipeline/probes/variables.py) | what computes an internal variable the output never prints? |
| [BridgeCircuit](#bridgecircuit) | `BridgeVariable`, built by `bridge_variable` | [`pipeline/probes/variables.py`](../pipeline/probes/variables.py) | in a composition A(B(x)), what computes the intermediate B(x)? |

The output-side classes form one hierarchy (`DomainCircuit` → `ContrastCircuit` →
`SuppressionContrastCircuit`); the probe-side classes form a parallel one (`ProbeVariable` →
`BridgeVariable`). How a new circuit is declared in code is in
[reference/declaring_circuits.md](reference/declaring_circuits.md); detailed internals of every
class are in [reference/circuit_types.md](reference/circuit_types.md).

Candidates are found with EAP-IG (or slight variations of it, [circuit_finding.md](circuit_finding.md)):
every showcase circuit and 3,329 of the 3,341 faithful zoo circuits are EAP-IG candidates; the other
12, DomainCircuits from the early zoo run at Large, came from ACDC (6) and Edge Pruning (6)
([`results/zoo/circuits_index.csv`](../results/zoo/circuits_index.csv), columns `run`, `finder`).
The EAP-IG attribution metric is

$$L = -\frac{1}{P}\sum_{p}\bigl(\text{logit}_p[\text{correct}] - \text{logit}_p[\text{incorrect}]\bigr)$$

over the class's scored positions *p*, except for the probe classes, which use a KL divergence on
the probe readout. Each section below states what "correct", "incorrect" and the scored positions
are.

---

## DomainCircuit

**Definition.** Conditioning on an input domain collapses the model's algorithm to a simpler one;
that simpler algorithm is the circuit.

**Data.** There is no contrastive corrupt: clean and corrupt are both drawn from the conditioned
domain. A circuit draws one pool of inputs uniformly from its domain (`DomainCircuit.sample`, via
the `Condition`'s exact sampler in [`pipeline/models/conditions.py`](../pipeline/models/conditions.py));
[`pipeline/models/gen_data/gen_circuit_data.py`](../pipeline/models/gen_data/gen_circuit_data.py)
writes it to `gen_data/<model>/<circuit>/inputs.json`, and `model_registry.load_circuit_pairs`
splits it into clean / corrupt halves at load time.

**Discovery.** EAP-IG with correct = the clean sequence's output, incorrect = the corrupt
sequence's output, scored at all output slots. (Slots where the two outputs coincide contribute
exactly zero; the loader masks them, `model_registry.dataset_to_pairs`.)

**Faithfulness.** Whole-sequence accuracy on the conditioned domain under edge-level mean
ablation; floor **0.90**.

**Code.** A DomainCircuit subclasses its model's DSL operator together with the `DomainCircuit`
mixin and sets

- `circuit_id` (the slug), `algorithm` (the model's DSL id),
- `_cond` — a `conditions.Condition` bundling the predicate `holds(x)` with an exact uniform
  sampler over the domain (e.g. `conditions.exclude_digit(3)`, `conditions.domain([...])`,
  `conditions.task_pin(i, n)`), or overrides `condition()` / `sample()`,
- optionally `eval()` — the collapsed sub-algorithm, and `scored_output_indices` to score a subset
  of output positions.

Defining the class registers it in `circuits.CIRCUITS`. `circuits.validate(cls)` draws inputs from
the domain and fails if `eval()` disagrees with the full operator anywhere on it.

**Example — segrev3.** `segrev3_id` = `SegmentReverse(3, Id(10))` reverses each run of non-3
digits; 3s stay fixed as boundaries (`4135792386 → 1432975368`). Conditioning on "no digit 3"
makes the whole input one segment, so the model collapses to plain reversal:

```python
class ReverseOnlyCircuit(SegmentReverse, DomainCircuit):
    circuit_id = 'reverse_only'
    algorithm = 'segrev3_id'
    _cond = conditions.exclude_digit(3)
    def eval(self, input_seq):
        return Reverse(Id(self.input_length)).eval(input_seq)
```

**Showcase.** Schmitt trigger, `schmitt_id__moe8` / `schmitt_noband`. The Schmitt trigger is a
switch with a high and a low threshold plus memory: it outputs 0 after a digit ≤ 3, 9 after a digit
≥ 6, and holds its previous state on 4 and 5 (initial state 9). On inputs with no 4s or 5s the
memory is never used, so the circuit is just the threshold detector
(`_cond = conditions.domain([0, 1, 2, 3, 6, 7, 8, 9])`). See
[`results/showcase/`](../results/showcase/).

**Multiplexed models.** For a mux model the per-task DomainCircuits are generated by
`circuits.task_circuit(mux_id, i)` (`circuit_id = task{i}`, cue token pinned to `10+i`), so a mux
model with *N* tasks has *N* DomainCircuits without any hand-written class.

---

## ContrastCircuit

**Definition.** Two nearly identical inputs differ by one deliberate edit that toggles a
predicate; that predicate defines the circuit — the detector, not the executor.

**Data.** Clean / corrupt = predicate present / absent at one position. The class's
`sample_pairs(batch, seed)` builds row-aligned minimal-edit pairs; `gen_circuit_data.py` stores them
with `scheme = "contrast"` and `answer_positions = "diff"`, and at load
`model_registry._diff_answer_slots` derives each row's scored slots (the output positions where the
two true outputs differ). `circuits.validate_pairs(cls)` checks that every clean row is in-domain,
the inputs differ, the true outputs differ, and an optional `affected_outputs()` claim matches.

**Discovery.** EAP-IG with correct = the clean output and incorrect = the corrupt output at each
differing position, scored only at output slots where the two outputs disagree.

**Reported circuit.** On a clean/corrupt pair, attribution returns the predicate's detector together
with the executor that produces any output at all. The executor is already known from the base
DomainCircuit (`base_circuit`), and candidate edges minus base edges, the **fringe**, isolate the
detector.

- **Showcase** ([`evaluate_circuit.py`](../pipeline/find_circuits/evaluate_circuit.py)): the
  reported circuit is the selected candidate itself, scored by interchange patching, which already
  ignores everything the edit does not touch. The fringe against the base is reported alongside.
- **Zoo-wide runs** (`smallest_faithful_circuit.py --base-circuit`,
  [circuits.md §6.2](circuits.md#62-smallest_faithful_circuitpy--the-zoo-wide-runs)): the
  reported circuit is the fringe, and it is scored by mean ablation of candidate ∪ base.

**Faithfulness.** $a = a_\text{clean} \times a_\text{corrupt}$ on alternating clean / corrupt
pairs, evaluated by **interchange patching** (each row's baseline is its partner's activations,
both rows share the clean teacher-forced prefix, labels are the full model's outputs); floor
**0.80**. Under interchange patching the parts of the model the edit never changes drop out by
themselves, so the subtraction above is not needed for evaluation. Details and rationale:
[circuits.md §4.1](circuits.md#41-interchange-patching-for-contrastcircuits).

**Code.** `class ContrastCircuit(DomainCircuit)` adds `base_circuit`, `sample_pairs()` (whose first
line must reseed from `conditions.stream_key(seed, self.circuit_id)` so sibling circuits draw
independent rows), the optional `affected_outputs()` claim, and `local_predicate` (False when the
toggled predicate feeds every output position).

**Example — segrev3.** `ThreeDetectorCircuit` (`three_detector`, `base_circuit = 'reverse_only'`):
clean inputs contain at least one 3; the corrupt flips one 3 to a non-3, which deletes a segment
boundary. Subtracting the reverse-only DomainCircuit leaves the 3-boundary-detection circuit.
The example illustrates the declaration only: in the zoo-wide runs no candidate of `three_detector`
cleared the floor, on `segrev3_id` or on `segrev3_id__wide` (`below_floor` in
[`results/zoo/circuits_index.csv`](../results/zoo/circuits_index.csv)).

**Showcase.** Dip detector, `mexhat_id__moe8` / `mexhat_fill`. `mexhat` sets an interior digit to 0
when it sits in a dip (`x[i-1] + x[i+1] − 2·x[i] ≥ 4`) and copies it otherwise
(`3141592653 → 3040590653`). `MexhatFillCircuit` draws clean rows from the full domain and corrupts
one digit so that the true output changes; its base is `mexhat_full`.

---

## SuppressionCircuit

(Also called a *routing* circuit; the code name is `SuppressionContrastCircuit`.)

**Definition.** Only for **multiplexed (mux) models**: one transformer trained on several
algorithms, the active one selected by a cue token at position 0 (not compositional models). On a
row cued for task A, the output disagrees with task B's answer either because the model amplifies
A or because it suppresses B. The question is what keeps B's answer out. The **fringe** is the
candidate's edges beyond task A's own DomainCircuit (`base_circuit = task{A}`). Under the
showcase's own-task labels it holds the routing together with task B's execution. The showcase reports the full candidate, with the fringe alongside;
the zoo-wide runs report the fringe.

**Caveat (zoo).** All 125 faithful zoo SuppressionCircuits pass only under the wrong-cue (*tonic*)
sever reference ([`results/zoo/stats.md`](../results/zoo/stats.md) §6). The zoo severs the fringe
by replacing its edges with per-position means, taken either over cued rows or over wrong-cue rows
(the tonic reference). A suppressor that is constant across cues (a *tonic* one) equals its own
cued-row mean, so severing it with cued-row means changes nothing: it is invisible to the first
reference, and only the wrong-cue means expose it
([circuits.md §6.2](circuits.md#62-smallest_faithful_circuitpy--the-zoo-wide-runs)).

**Data.** Clean / corrupt = task A's cue + payload / task B's cue + the same payload
(`SuppressionContrastCircuit.sample_pairs` keeps only payloads on which the two tasks disagree).
Scored where the two tasks' outputs disagree (`affected_outputs`).

**Discovery.** EAP-IG with correct = the cued task's answer and incorrect = the other task's answer
at each position where the two tasks differ.

**Reported circuit.** In the showcase, the selected candidate itself. Each row is scored against its
own task's output (below), so the candidate must contain task A's execution, task B's execution and
the routing between them. The fringe beyond task A's DomainCircuit is reported alongside. A model
with *N* tasks has *N(N−1)* SuppressionCircuits, one per ordered pair.

**Faithfulness.** $a = a_\text{clean} \times a_\text{corrupt}$ on alternating clean / corrupt
pairs under mean ablation. Each member is labelled with its **own** DSL output: task A's answer for
the A-cued row and task B's answer for the B-cued row. Floor **0.80** (the paired rule;
[circuits.md §4.3](circuits.md#43-suppression-pairs)). The zoo-wide runs selected the fringe by a
different criterion, namely whether severing it *releases* B's answer (see
[circuits.md §6.2](circuits.md#62-smallest_faithful_circuitpy--the-zoo-wide-runs)).

**Code.** `class SuppressionContrastCircuit(ContrastCircuit)` with `cued_task`, `wrong_task`;
instances are manufactured, not hand-written: `circuits.suppression_circuit(mux_id, cued, wrong)`
(`circuit_id = suppress_t{cued}_over_t{wrong}`) and `circuits.suppression_circuits_for(mux_id)`
(all ordered pairs). Both raise unless the model decodes to a `TaskMux`.
[`pipeline/task_router/router_diagnostic.py`](../pipeline/task_router/router_diagnostic.py) records,
per ordered pair, whether the wrong task's distinctive machinery is off on A-cued inputs (gating),
on but out-voted (outvoting), or live (superposition).

**Showcase.** `mux2.pmstep-id.sr12-id__moe8`: a two-task model choosing between `pmstep` (task 0,
cue token 10: add 1 to each even digit, subtract 1 from each odd digit) and `sr12` (task 1, cue token
11: a latch that turns on at 1 and off at 2). The reported circuit is `suppress_t1_over_t0` — on
rows cued for the latch, what keeps the parity-step answer out — with base `task1`; see
[`results/showcase/`](../results/showcase/).

---

## ProbeCircuit

**Definition.** An internal variable that the output never prints. The circuit is the subgraph
that computes it.

**Is there a variable to find?** Linear probes are trained on the residual stream at the input
positions, at every layer; the layer *l* with the highest mean probe accuracy is chosen, and the
variable counts as encoded — a circuit exists — if mean probe accuracy beats chance (random
guessing of the variable's class). The stricter criterion this release uses before searching for a
ProbeCircuit is the three-way verdict that the probe stage
([`pipeline/probes/train_probes.py`](../pipeline/probes/train_probes.py)) records per variable:
`decodable` means some layer's probe error is at most half the layer-0 embedding baseline's error
and at most half the majority-class error; `trivial_from_input` means the embedding alone already
decodes the variable; otherwise `absent` (zoo-wide counts:
[`results/zoo/stats.md`](../results/zoo/stats.md) §7). Both showcase variables are `decodable`
(`verdict` in their `site_scan.json`); `eap.py --probe` itself does not check the verdict.
[`pipeline/probes/validate_probe.py`](../pipeline/probes/validate_probe.py) runs a permutation
control, a leakage matrix against sibling variables, robustness refits, and a causal steering gate.

**Data and discovery.** EAP-IG with metric = KL(clean probe log-probs ‖ current probe log-probs)
over the probe's classes, averaged over the probed input positions. The clean/corrupt pairs are
taken from the input pool of an ordinary circuit of the host model (`--circuit`). The showcase runs
discovery on two pools per variable and puts every candidate from both into one selection: plain
resample pairs over the full domain (the host's full-domain DomainCircuit, `graviclump_full`,
`psum_novel_whole`) and single-digit edit pairs (a ContrastCircuit's pool: `graviclump_edit`
changes one digit so that the output changes; `psum_novel_fresh` replaces a repeated digit with one
not yet in the row). These discovery pairs are not the corrupt member used for evaluation (below).
The model is cut off at the probe layer *l*: components above *l* cannot affect the probe, so there
are no logit readers, and the probe site
`blocks.{l}.hook_resid_post` takes the place of the logits (`retarget_graph_to_probe`,
`probe_cutoff_indices`, `make_probe_metric` in
[`pipeline/find_circuits/eap.py`](../pipeline/find_circuits/eap.py), run as
`eap.py --probe <variable>`).

**Faithfulness.** Normalized probe accuracy (0 = majority class, 1 = full model, clamped at 0),
$a = \hat a_\text{clean} \times \hat a_\text{corrupt}$ with the corrupt member the single-digit
edit that makes the probe's correct class most strongly decodable; floor **0.80**. The circuit is
scored at the layer it was discovered at, with the probe readout that discovery attributed against
([circuits.md §4.2](circuits.md#42-normalized-probe-accuracy-probecircuit-bridgecircuit)).

**Code.** A `ProbeVariable` subclass sets `variable_id`, `algorithm`, `n_classes` and implements

- `labels(inputs) → [B, P]` — the variable, replayed in pure Python from the DSL, never from the model;
- `valid_positions()` — **input** positions where the variable is defined (output positions are
  excluded: at output position 9+k the teacher-forced answer is already in context);
- `check_row(x, lab)` — raises unless the labels are jointly consistent with the operator's output
  (an independent check of the claimed variable);
- optionally `counterfactual(x, p, c)` — the true output had the variable at position *p* been *c*
  (used by the causal gate).

`variables.validate(cls)` rejects declarations whose labels are constant at a declared position or
depend on future tokens (`causal_positions`).

**Example — maxrun.** `maxrun_id` emits the running length of the longest run of identical adjacent
digits (capped at 9). The variable `maxrun_cur` is the length of the *current* run, which the output
never shows (declared at input positions 2–8, 10 classes). A matching discovery pool is the
ContrastCircuit `maxrun_extend`, whose corrupt member copies a digit to lengthen a run.

**Showcase.** Gravity, `graviclump_id__moe4` / `graviclump_mean`. `graviclump` pulls every output
digit one step toward the input's center of mass (the rounded mean of the row:
`3141592653 → 4242483544`); the circuit computes the input's average. The variable is the rounded
mean, declared at input position 9 only (it needs every digit), 10 classes.

---

## BridgeCircuit

**Definition.** For a compositional model A(B(x)), the intermediate value B(x) — computed, never
printed. Found exactly like a ProbeCircuit.

**Code.** `bridge_variable(composite_id, split_k)` strips the *k* outermost operators of the
composite and returns a registered `BridgeVariable` (a `ProbeVariable` subclass) whose labels are
the inner expression's output, with `variable_id = bridge{k}_<inner id>`. Construction refuses a
split that carries nothing (no prefix-determined, non-constant intermediate slot the composite's
output depends on) or that another split depth reproduces (`_split_is_identified`); `n_classes` is
measured from sampled labels. `check_row` grafts the claimed intermediate back into the composite
and requires it to reproduce the full output.

**Showcase.** `psum_novel_id__moe8`, the variable from `bridge_variable('psum_novel_id', 1)`.
`psum_novel_id` = `PrefixSumMod(NoveltyIndicator(Id(10)))`: the running sum (mod 10) of a
"novelty" indicator. Stripping the outer prefix sum leaves `NoveltyIndicator` — "is this digit new
among the digits seen so far" — so the bridge variable is `bridge1_novel_id` (2 classes, input
positions 1–9; position 0 is always novel and therefore excluded).

---

## Not a class: mux

"Mux" is a property of a model, not a circuit class: one transformer, *N* DSL tasks, cue token
`10+i` at position 0 (model ids `mux<N>.<task0>.<task1>...`, see [models.md](models.md#the-dsl)). A
mux model yields *N* DomainCircuits (`task{i}`) and *N(N−1)* SuppressionCircuits.

## Where the classes came from

The classes were added one at a time, each to answer a question the previous ones could not —
domain restriction first, then contrast predicates on top of a base circuit, suppression on
multiplexed models, probes for internal variables, and bridges for compositions. That history is in
[brainstorming.md](brainstorming.md#5-how-the-five-circuit-classes-emerged).
