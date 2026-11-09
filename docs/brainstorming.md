# How the algorithms and circuit classes were brainstormed

The zoo was not designed in one pass. It grew from a single case study, through an explicit map of
what could be varied, into rounds of proposals that were scored against a coverage matrix,
verified, and integrated into the DSL. This page describes that process; the full record is in the
appendices:

- [brainstorming/brainstormed_models.md](brainstorming/brainstormed_models.md) — the master table
  of brainstormed algorithms and the circuits each one yields, with scores and status;
- [brainstorming/operator_glossary.md](brainstorming/operator_glossary.md) — what every
  brainstormed operator computes, in plain language, grouped by mechanism;
- [reference/variation_axes.md](reference/variation_axes.md),
  [reference/variation_matrix.md](reference/variation_matrix.md),
  [reference/conditional_models.md](reference/conditional_models.md),
  [reference/contrast_pair_catalog.md](reference/contrast_pair_catalog.md),
  [reference/probe_variable_catalog.md](reference/probe_variable_catalog.md) — the design
  catalogs behind each step below.

## 1. Starting point: segrev3

The repository began as a case study of one model: a 4-layer × 8-head attention-only transformer
trained on `segrev3` (reverse each run of non-3 digits, keep the 3s fixed), reverse-engineered with
three discovery methods that had to agree, plus a validation suite
([segrev3_case_study.md](segrev3_case_study.md)). Two lessons from it
shaped everything after:

- **Conditioning isolates a sub-algorithm.** On inputs without a 3, segrev3 *is* reversal, so a
  circuit found on that domain is the reversal machinery alone. This became the first circuit class.
- **Many operators have this structure.** An operator with an explicit branch keyed on a
  recognizable input feature has a class of inputs on which it collapses to a simpler, well-defined
  base algorithm. Cataloguing the DSL's conditional operators by how clean that base is (digit-gated
  with a clean base, value/threshold-gated regime switches, feature detectors whose base is
  constant) gave the first list of models worth building
  ([reference/conditional_models.md](reference/conditional_models.md)).

## 2. From one model to a zoo of operators

To ask the same question of many algorithms, the task generator became a DSL
([models.md](models.md#the-dsl)): unary operators that compose, each with a pure reference
implementation, so that a model is fully specified by an id and every analysis (data, answer
positions, ground-truth outputs, counterfactuals) is derived from that id. Operators were then
added to cover many *kinds* of computation rather than many variants of one: per-digit maps,
running accumulators, order statistics, local-window arithmetic, permutations, sorting networks,
cellular automata and relaxation dynamics, signal processing, number theory, group products,
recurrences, state machines, bit-level operations, ciphers and codes, scheduling and regulators,
compute-then-veto operators, streaming data structures, compositions, and multiplexed models — the
twenty families of the [operator glossary](brainstorming/operator_glossary.md).

## 3. Axes of variation

A circuit-discovery experiment has independent knobs. They were written down, each anchored in
published work, in [reference/variation_axes.md](reference/variation_axes.md):

| axis | what varies |
|---|---|
| discovery dataset | how clean / corrupt inputs are built — resampling within a domain vs. targeted minimal edits |
| architecture | attention-only vs. MLP vs. mixture-of-experts feed-forward blocks |
| operator structure | **breadth** — one model, *N* tasks selected by a cue token; **depth** — composed algorithms A(B(x)) |
| size and geometry | layers, heads, width (Small / Large / Wide / Deep / XL) |
| discovery target | what the circuit explains: the output logits, an internal variable read by a probe, or the routing signal |
| comparison | what the circuit is measured against: another architecture, a sub-task model, a sibling circuit |
| behavioral regime | which inputs are explained: correct behavior, errors, out-of-distribution lengths |

The first four are properties of the model; the last three are properties of the question.

## 4. The variation matrix and its rubric

Crossing circuit axes with model axes gives a matrix
([reference/variation_matrix.md](reference/variation_matrix.md)). Rows are the candidate circuit
families (domain restriction, contrast pairs, probe-metric circuits, suppression, positional);
columns are model variations (size, +MLP, +MoE, number of operators). Every cell was rated before
anything was built:

| rating | meaning |
|---|---|
| ●●● | load-bearing — the pairing *is* the experiment |
| ●● | strong — a sharp, testable prediction |
| ● | useful but secondary |
| — | not needed |

The matrix records predictions, and the reference page lists the main ones with their status
against the zoo snapshot; most cannot be decided from it yet (the absolute edge cap, the
training-recipe confound, the missing router hooks). The positional row never became a circuit
class — it survives only as two exploratory finder flags.

Once circuits existed, the matrix was complemented by a **coverage map** generated from the code and
the artifacts: [`pipeline/repo_tools/make_taxonomy.py`](../pipeline/repo_tools/make_taxonomy.py)
classifies every declared circuit and writes `pipeline/models/taxonomy.json`. Its cell families are

| family | cell = |
|---|---|
| `domain_cells` | condition family × verdict (faithful winner or not) × compactness (tiny / medium / near-full) |
| `contrast_cells` | locality × necessity × verdict |
| `mask_shapes` | the shape of a contrast pair's output-difference mask (singleton, window, suffix, …) |
| `probe_cells` | probe kind × how the variable is used × whether a probe circuit exists |
| `suppression_cells` | verdict × release strength × router verdict (gating / outvoting / superposition) |
| `model_axis_cells` | architecture family × verdict |
| `breadth_cells` | per multiplexed model, the verdicts of its task circuits |

and `empty_cells` lists, per family, the product of the observed levels of each axis minus the
occupied cells, after removing unreachable combinations, failure levels (a proposal cannot earn
credit for predicting that its own circuit fails) and build backlog. The map was regenerated at the
start of each proposal round, so "which cells are still empty" was a lookup, not a judgment.

## 5. How the five circuit classes emerged

Each class was added because the previous ones structurally could not answer a question.

1. **Domain restriction → DomainCircuit.** The segrev3 isolation, generalized: a condition (a
   predicate plus an exact sampler) restricts the input domain and the circuit is the model's
   behavior there. Its limits motivated everything else: it finds the *executor* of a domain but
   not the *detector* that decides the domain; it degenerates when the restricted output is
   constant; and it only sees variables that reach the output.
2. **Contrast predicates on top of a base circuit → ContrastCircuit.** A minimal edit that toggles
   one predicate while holding the rest of the input fixed recruits the detector; subtracting the
   already-known base DomainCircuit leaves it. Designs were catalogued per model with
   machine-verified masks and rated clean / medium / confounded
   ([reference/contrast_pair_catalog.md](reference/contrast_pair_catalog.md)) — for example, an
   edit whose two outputs coincide (an empty mask) cannot be used, which rules out several
   parity toggles.
3. **Suppression on multiplexed models → SuppressionCircuit.** The breadth axis (one model, a cue
   token selecting among *N* tasks) raised a new question: on a row cued for task A, is task B
   switched off, or computed and then cancelled? The DSL supplies the wrong task's exact answer on
   every input, so "suppression" can be defined operationally: the circuit whose removal lets B's
   answer out. It was first predicted to show up as negative-sign edges; the implemented criterion
   is the release of B's answer, because on an A-vs-B margin a sign cannot separate suppressing B
   from promoting A.
4. **Probes for internal variables → ProbeCircuit.** Contrast pairs can only see what reaches the
   output (a non-empty mask). State carried through time that the output never exposes — the
   13-valued register behind a mod-10 output, a current run length, a seen-set — needs a different
   instrument: a linear probe as the discovery target. Candidate variables were catalogued per model
   ([reference/probe_variable_catalog.md](reference/probe_variable_catalog.md)), and the
   teacher-forcing confound (at output position 9+k the answer *y*₍k−1₎ is already in context) fixed
   probes to input positions.
5. **Bridges for compositions → BridgeCircuit.** With output metrics, a composed model A(B(x))
   returns the union of both stages, since every stage affects the output. Probing the intermediate
   B(x) — the bridge — asks whether the model computes B, writes it down and then computes A, or fuses
   the two. `bridge_variable(composite, k)` generates the variable from the model id itself.

Multiplexing itself was deliberately **not** made a class: it is a property of a model, which then
yields *N* DomainCircuits and *N(N−1)* SuppressionCircuits.

## 6. Proposal rounds

New operators and circuits were proposed in rounds (the "batches" of the
[master table](brainstorming/brainstormed_models.md)). Every proposal became one row: the operator in
one plain-English sentence, a worked example computed from its reference code, the circuit classes
it is expected to yield (primary first), the specific circuit rows it is designed to support, and a
status. Most rows were never built; their predictions are hypotheses.

**What each round looked for** changed as the zoo filled up:

- early rounds targeted absent operator families, empty taxonomy cells and leverage across
  architecture variants, then specific gaps (edit patterns, composition depth, routing structure,
  suppression, probe circuits);
- later rounds sourced operators from **named algorithms and scientific computations** — cellular
  automata, sandpiles and relaxation dynamics, queueing, signal processing, ciphers and
  error-correcting codes, number theory, group theory, tree and graph dynamic programs — with a
  mild standing preference for operators whose semantics fit one plain sentence;
- a **novelty-first** phase ranked proposals by mechanism distance from the whole existing catalog
  (bit-level operations, comparison topologies, bounded machines, new algebras);
- the final **empty-cell-first** phase required every survivor to fill currently empty, creditable
  cells of the coverage map (§4), sourced across several different fields.

**The rubric.** Proposals were ranked on four criteria:

| criterion | measures |
|---|---|
| novelty | how far the mechanism is from every existing operator |
| yield | how many of its circuit rows survive verification (§7) |
| emptiness | how many currently empty cells of the coverage map those rows fill |
| leverage | how much the operator gains from the architecture variants |

In the empty-cell-first phase emptiness was weighted most heavily and became a hard gate as well: a
proposal filling no creditable empty cell was dropped, every domain row had to target an unfilled
domain cell, and each round had to draw on several source fields with only a few survivors per
field. Rejections were binding across rounds: a rejected or duplicated mechanism was not proposed
again.

## 7. Verification

A proposal had to survive three kinds of check before it could be integrated:

1. **A reference implementation.** Every operator is a pure-Python reference function on a row of
   ten digits; every example in the table is computed from it, and duplicates of existing operators
   are caught by comparing outputs.
2. **Self-tests of every claimed circuit row.** A claimed domain collapse must hold on every sampled
   in-domain row (the table reports these as, e.g., 512/512); a claimed contrast must produce a
   non-empty output-difference mask of the claimed shape on every sampled pair, with a provably
   empty edit as an invariance control where one exists; a claimed probe variable must be
   prefix-determined, non-constant and consistent with the operator's output.
3. **Learnability.** A hard trainability gate: mechanisms expected to be unlearnable at the zoo's
   sizes (for example unbounded pointer walks or dynamic programs without bounded look-back) were
   recast into a learnable form or dropped. After integration, the pipeline promotes a checkpoint
   only once it reaches the 0.95 sequence-accuracy gate ([models.md](models.md#training)); the zoo
   snapshot nevertheless contains 8 checkpoints below the gate, with 61 circuit rows (8 faithful),
   included in every table of [stats.md](../results/zoo/stats.md) (`models_index.csv`,
   `circuits_index.csv`).

## 8. Integration into the DSL

A surviving proposal becomes code in three places:

- the operator: a `_ref_<name>` reference function and a `DSLExpr` class with its id token in
  [`pipeline/models/algorithms.py`](../pipeline/models/algorithms.py);
- its circuits: `DomainCircuit` / `ContrastCircuit` subclasses with their conditions in
  [`pipeline/models/circuits.py`](../pipeline/models/circuits.py) and
  [`pipeline/models/conditions.py`](../pipeline/models/conditions.py), and `ProbeVariable`s or
  `bridge_variable` calls in [`pipeline/probes/variables.py`](../pipeline/probes/variables.py)
  (how to declare them: [reference/declaring_circuits.md](reference/declaring_circuits.md));
- its models: entries in [`pipeline/models/models.txt`](../pipeline/models/models.txt) for every
  architecture (`pipeline/repo_tools/make_models_txt.py`).

The declarations are then checked by code: `circuits.validate` (the collapsed `eval` equals the
full operator on the whole domain), `circuits.validate_pairs` (every contrast pair is in-domain with
a non-empty mask), `variables.validate` (probe labels are well-formed, non-constant and causal),
and `pipeline/repo_tools/make_circuits_txt.py`, which validates every circuit against its parent
operator and regenerates the circuit catalog `pipeline/models/circuits.txt`. From there a model goes
through the ordinary pipeline: train, generate data, find and select circuits
([reproducing.md](reproducing.md)).
