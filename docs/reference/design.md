# Design notes

Rationale for the zoo pipeline (`pipeline/`) that does not fit in one-line source comments. This
describes the current design. Definitions of the five circuit classes are in
[circuit_types.md](circuit_types.md); the faithfulness definitions and floors used for the
showcase circuits are in [../circuits.md](../circuits.md) (implemented by
[pipeline/find_circuits/evaluate_circuit.py](../../pipeline/find_circuits/evaluate_circuit.py)). Zoo-wide statistics:
[results/zoo/stats.md](../../results/zoo/stats.md).

## Execution model

Local execution is the default: `pipeline/run_model_pipeline.py` runs every stage on the current
machine (`--device` picks cpu/cuda, `--gpus N` spreads the job graph over N local GPUs; `--local`
is accepted as a no-op alias). An optional `--modal` backend runs the GPU stages on Modal
containers instead: a single-circuit model runs in one container (train → data → finders →
smallest), and a multi-circuit model trains once and then uses one container per circuit.
Deploy the app once with `modal deploy pipeline/modal_pipeline_app.py`; `--smoke` always runs
locally.

Within one machine, the finder wave (EAP-IG, plus ACDC / edge pruning when named) runs as
concurrent GPU processes that split the per-process memory budget; serial stages (edge pruning,
smallest) run alone. The orchestrator checks memory headroom between stages
(`model_registry.assert_memory_headroom`).

## Directory naming

- `circuits_data/<model>/<circuit>/`: per-circuit discovery artifacts (candidate circuits, eval
  caches, `smallest_circuit.json`).
- `circuits_figures/<model>/<circuit>/`: the matching PNG tree. Built independently through
  `model_registry.fig_dir()` (not by string-replacing the data path), so the two trees can move
  independently.
- `models/gen_data/<model>/<circuit>/inputs.json`: one input pool per circuit.
- `probes_data/<model>/<variable>/`: probe artifacts (`site_L{l}.pt`, scans, validations).
- `task_router/<model>/diagnostic.json`: the mux routing diagnostic.

`model_registry.data_dir()` / `fig_dir()` are the only path builders for the circuit trees; no
other module string-formats these roots. All of these directories are generated and gitignored.

## Input pools

One `gen_data/<model>/<circuit>/inputs.json` pool per circuit, read by every finder and analysis
through `model_registry.load_circuit_pairs`, which splits it into clean/corrupt at load time.

**Sibling scopes of one model draw independent rows.** Every sampler seeds itself from
`conditions.stream_key(seed, tag)` (blake2b, so a pool regenerated in another process is
bit-identical) rather than from the bare seed. With one shared RNG, a model's scopes would be
row-for-row functions of each other, and a ContrastCircuit's "clean" pool could be its base
circuit's own pool, which makes the `d_control` control degenerate. The one deliberate exception
is `conditions.mux_sampler`: every `task_pin` sibling of a TaskMux model draws the same payloads at
a fixed seed, so the per-task circuits get content-matched datasets.

**Plain pools carry an `answer_mask`.** A resample pair carries no evidence at a position where the
clean and corrupt labels agree (the logit difference there is identically zero under any
ablation), so those positions are masked out and rows that agree everywhere are dropped
(`model_registry.dataset_to_pairs`). A pool where every row agrees everywhere raises: the scope is
unscorable, not faithful. This, together with the pool windows below, is the pairs regime
`canon.CANON_PAIRS_REGIME = "pairs-v2"` (`data_gen/canon.py`), stamped into every result so
results from different regimes are never averaged together.

**Attribution, selection and scoring read disjoint windows.** `model_registry.POOL_WINDOWS` gives
each consumer a start offset (attribution 0, selection 5000, scoring 10000), and selection and
scoring each read `SMALLEST_EVAL_N` = 5000 pairs from there, so a 15000-pair pool covers all three.
A winner picked for clearing a floor on the same rows it is then scored on would be selected on its
own noise. `load_circuit_pairs` fails loudly when the pool is too short for the requested window,
and the resolved window travels in `pairs.extra` so every consumer records what it read.

## Floors and gates: do not conflate them

| Threshold | Value | Defined in | Governs |
|---|---|---|---|
| train gate `SEQ_ACC_GATE` | 0.95 (`>=`) | `run_model_pipeline.py`, `models/train_models/promote.py` (`GATE`), `repo_tools/gates.py` (`CKPT_SEQ_ACC_GATE`) | Training only. A checkpoint must reach this test sequence accuracy to skip retraining; below it the trainer reruns and escalates within the architecture family. Never a circuit metric. |
| absolute selection threshold `DEFAULT_TARGET` | 0.85 (`>=`) | `find_circuits/smallest_faithful_circuit.py` | The `smallest` stage's selection: absolute whole-sequence exact-match accuracy under edge-level mean ablation. `run_model_pipeline.SMALLEST_TARGET` mirrors it; `--smallest-target` overrides. |
| relative selection threshold `DEFAULT_RELATIVE_FLOOR` | 0.85 (`>=`) | `find_circuits/smallest_faithful_circuit.py` | The `smallest` stage's selection, conjoined with the absolute threshold. |
| acceptance gates | see [Acceptance gates](#acceptance-gates) | `repo_tools/gates.py` | Which rows the taxonomy/summary scripts report. Never feeds back into selection. |
| showcase faithfulness floors | see [../circuits.md](../circuits.md) | `find_circuits/evaluate_circuit.py` | The per-class faithfulness `a` reported for the showcase circuits. |

The 0.95 train gate is defined in several places (`promote.GATE`,
`run_model_pipeline.SEQ_ACC_GATE`, `run_training_job.sh`, `gates.CKPT_SEQ_ACC_GATE`,
`repo_tools/arch_coverage.py`); keep them in sync when changing it.

Two further 0.95s are unrelated: `edge_pruning_sparsity_0.95` is a target sparsity, and the
`faithfulness_target` field in `models/human_circuit_finding.json` is not read by any code.

## Smallest-faithful selection

Eligibility conjoins three conditions on a non-empty candidate (`smallest_faithful_circuit._passes`):

1. its absolute whole-sequence exact-match accuracy under edge-level per-position mean ablation
   (`perf_circuit.sequence_accuracy`) is `>= --target`,
2. its relative recovery `R = (circuit − empty)/(full − empty)` is a number and
   `>= --relative-floor`,
3. the empty baseline (every edge severed) is below `--target`.

The absolute threshold alone measures the checkpoint as much as the circuit: on a weak checkpoint
a circuit inherits credit it did not earn, and on a near-constant-output scope the fully
mean-ablated model already clears the threshold, so the "circuit" would be credited for an output
it does not compute. On a healthy checkpoint (full ≈ 1, empty ≈ 0) the relative condition is
implied by the absolute one, so it binds exactly where the absolute threshold cannot see. When
full == empty the ratio is 0/0 and the field carries the `zero_contrast` marker instead of a
number (`model_registry.recovery_ratio` raises rather than inventing a denominator), which fails
the gate: the scope is unscorable.

Among eligible candidates the winner maximizes the selection score

    (1 - n_nodes_circuit / n_nodes_full) * (1 - n_edges_circuit / n_edges_full)

a product of complements: the circuit must be small in both dimensions, so a head-heavy candidate
cannot win on edge count alone. (It is not `1 − (h/H)(e/E)`.)

Both denominators are pure geometry, never an artifact of which finders ran:
`n_nodes_full = n_layers*n_heads + (0 if attn_only else n_layers)` and
`n_edges_full = model_registry.live_edges(spec)`, computed by `canon.live_edge_budget`. The three
finders report three different edge counts at identical geometry (see
[Edge universes](#edge-universes)), so reading the denominator off a finder's summary would make
the score comparable only within one finder.

Ties break by fewer edges, then fewer nodes, then higher absolute accuracy, then name. Accuracy
comes last because every all-heads candidate scores exactly 0, and an accuracy-first key would
resolve those ties to the larger candidate. Accuracy gates eligibility only; it does not enter the
score.

`SELECTION_REGIME` names this rule set and is stamped on every output (`selection_regime`). Rows
from different regimes are not comparable, and changing it re-selects every row.

## Edge universes

"How many edges does this model have" has three answers, and the tooling reports all three.

**The reported EAP-IG number includes dead MLP slots.** The vendored `Graph.from_model` allocates
one `m{l}` MLP node per layer unconditionally (it never reads `cfg.attn_only`), so on the five
attention-only architectures `graph.real_edges` counts edges that can never carry signal.
`smallest_faithful_circuit._edges_to_declared` drops them, and `canon.live_edge_budget` is the
number selection divides by:

| arch | EAP-IG `real_edges` | live edges | dead `m{l}` edges |
|---|---|---|---|
| small | 110 | 81 | 29 |
| large | 1519 | 1281 | 238 |
| wide | 5327 | 4865 | 462 |
| deep | 6637 | 5633 | 1004 |
| xl | 23981 | 22017 | 1964 |
| mlp / moe4 / moe8 | 1519 | 1519 | 0 |

(Live edges are `canon.live_edge_budget(n_layers, n_heads, attn_only)`; the EAP-IG counts are the
ones listed in its docstring.)

**The three finders also disagree at identical geometry.** At the `large` geometry (4 layers × 8
heads, attention-only):

| finder | field | value | counts |
|---|---|---|---|
| edge pruning | `n_edges_total` | 1184 | maskable edges only: no MLP slots on attn-only, no always-on embedding writer |
| ACDC | `n_prunable_edges_initial` | 1377 | the 1281 live edges plus `3·L·H = 96` Q/K/V-input → projection bookkeeping edges |
| EAP-IG | `real_edges` | 1519 | every matrix slot, dead `m{l}` included |

These are three definitions of "edge", not a bug in any one finder. A candidate's "fraction of the
full model" is only meaningful next to the denominator that produced it, which is why selection
uses none of the three.

## The candidate ladder is scale-relative

EAP-IG's rungs are fractions of the model's live edge budget, not fixed counts:
`CANON_TOPN_FRACS = [0.02, 0.05, 0.15, 0.40]` under `CANON_TOPN_REGIME = "frac"`
(`data_gen/canon.py`), resolved per model by `canon.topn_ladder`:

| arch | live edges | rungs |
|---|---|---|
| small | 81 | 2, 4, 12, 32 |
| large | 1281 | 26, 64, 192, 512 |
| wide | 4865 | 97, 243, 730, 1946 |
| deep | 5633 | 113, 282, 845, 2253 |
| xl | 22017 | 440, 1101, 3303, 8807 |
| mlp / moe4 / moe8 | 1519 | 30, 76, 228, 608 |

An explicit rung above the live budget is a hard error. The fixed ladder `[50, 200, 800]`
(`CANON_TOPNS`, the `"abs"` regime) is kept only to reproduce candidates built with it. A fixed
ladder means something different at every scale: at `small`, rungs 200 and 800 exceed the whole
graph (81 live edges), while at `xl` the top rung is under 4% of it. Contrast and Suppression
candidates built under the `"abs"` regime are therefore capped at 800 edges.

`--search` is passed in every mode (the `smallest` job in `run_model_pipeline.py`). It runs
EAP-IG's one-attribution `--target-seq-acc` scan (`find_circuits/eap.py`, `_search_smallest_topn`),
which sweeps a geometric rung set and takes the argmax of the same size score over the passing
rungs. Faithfulness is not monotone in top-n, so a bisection could return a non-minimal rung or
miss a passing one. The scan reads the selection pool window, disjoint from both the attribution
rows and the rows the `smallest` stage finally scores on, and contributes one extra candidate,
usually below the ladder ceiling. Its pass predicate is a strict `seq > target`, which is simpler
than selection's three-way `_passes`; the `smallest` stage re-scores the candidate with the full
rule.

## Differential mode (ContrastCircuits, `--base-circuit`)

A `ContrastCircuit` (minimal-edit clean/corrupt pairs; for example segrev3's `three_detector`, base
`reverse_only`) is selected differentially, not by standalone accuracy. The tested detector is the
**fringe**: the candidate's edges minus the base winner's edges. Raw top-n candidates contain the
executor outright (the movers rank highest in the attribution), so severing a raw candidate
destroys behaviour everywhere and fails `d_control` by construction.

Eligibility is exactly two conditions (`smallest_faithful_circuit._passes_differential`): a
non-empty fringe, and `s_union >= target` (whole-sequence accuracy of candidate ∪ base over all
positions). The winner maximizes

    (1 + clamp(d_affected, 0, 1)) · (1 − (fringe_nodes/nodes_full)(fringe_edges/edges_full))

Size dominates; necessity is a bounded bonus of at most 2×. The leading `1 +` is load-bearing:
without it every `d_affected == 0` candidate would score exactly 0 and could never win.

| metric | meaning |
|---|---|
| `d_affected` | drop at the mask positions when the fringe is severed |
| `d_unaffected` | collateral change off the mask |
| `d_control` | effect of severing the fringe on the base circuit's own pool |

**These metrics never gate selection.** They are always computed and recorded. Classification
lives in `models/taxonomy.json`, written by `repo_tools/make_taxonomy.py`, which labels
`necessity_class` = necessary (`d_affected >= 0.10`) / weak (`>= 0.02`) / redundant, and
`locality_measured` = local (`d_unaffected <= 0.05`) / global. The acceptance gates
(`repo_tools/gates.py`, `D_CONTROL_MAX = 0.05`) additionally require that severing a fringe does not
damage the base circuit's own pool. `--global-predicate` / `local_predicate = False` (for example
segrev3's `three_detector`, whose boundary information is consumed at every position) only records
the flag; no eligibility function branches on it.

Every one of these metrics is whole-sequence exact match (`_seq_acc`) over its position set, tagged
`FRINGE_METRIC_TAG = "whole_sequence_exact_match"` so token-accuracy caches are never reused. That
unit matches the plain gate and the thresholds `taxonomy.json` applies. `d_control` reads the base
circuit's pool through the same scoring window as everything else.

The winner's `heads`/`edges` and its `smallest_circuit.pt` are the fringe (`circuit_name` keeps the
source candidate). Differential keys: `mode`, `base_circuit`, `base_edges_sig`, `sufficiency`,
`necessity`. Nothing eligible → `no_faithful_circuit` with `best_available_s_union`.

## Suppression mode (SuppressionCircuits, `--suppression`)

A SuppressionCircuit (code class `SuppressionContrastCircuit` in `models/circuits.py`, built by
`suppression_circuit(mux_id, cued, wrong)` / `suppression_circuits_for`) is a ContrastCircuit
whose pair is a **cue flip**: clean = `[10+cued, payload]`, corrupt = `[10+wrong, payload]`, with a
shared payload. `eval` stays the true TaskMux operator, so the per-pair mask is the set of
positions where `cued(payload) ≠ wrong(payload)`: the slots the wrong task could intrude on.

The finder is unchanged (standard contrast EAP-IG on the cue-flip pairs); the logic lives in the
selection mode. `--base-circuit` is the cued task's executor (`task{cued}`), and the suppressor is
its fringe. The metric is **release**, not attribution sign: on an A-vs-B margin, sign cannot
separate a suppressor (writes −B) from an A-promoter. Release is measured by severing:
`release_rate = wrong_task_wins(fringe severed) − wrong_task_wins(full)` over the affected
positions, where wrong-task-wins means the argmax equals `pairs.incorrect` = B(x) (the cued answer
is `pairs.correct` = A(x)). Ablating a real suppressor releases the wrong answer while leaving the
agreeing positions intact.

**Tonic reference.** Severing against cued-input means is a mathematical no-op on a cue-constant
("tonic") suppressor, which equals its own mean, so a cued-reference release of 0 cannot
distinguish "no suppressor" from "tonic suppressor". A second sever pass against wrong-cue means
gives `release_rate_tonic` and `collateral_tonic`. Eligibility is a two-branch OR
(`smallest_faithful_circuit._passes_suppression`): non-empty fringe AND either
(`release_rate >= --release-floor` with `collateral <= --epsilon`) or
(`release_rate_tonic >= --release-floor` with `collateral_tonic <= --epsilon`); collateral binds per
reference. Defaults: release floor 0.5, epsilon 0.05 (`DEFAULT_RELEASE_FLOOR`, `DEFAULT_EPSILON`).

The winner maximizes `clamp(release, 0, 1) · (1 − (fringe_nodes/nodes_full)(fringe_edges/edges_full))`
where `release` is the reference that passed (the larger when both do), chosen by
`_release_reference` / `_scored_release` and recorded on the row as `release_reference` /
`release_scored`, so a reader can always tell which sever pass the number came from.

As in differential mode, every rate is whole-sequence exact match: a release counts when all of a
row's affected slots flip to the wrong task. `min_accuracy_threshold` records the release floor, and
`--target` plays no role. The reported json is the fringe with `mode: "suppression"`,
`base_circuit`, a `release` block and `collateral`; nothing eligible → `no_faithful_circuit` with
`best_available_release`. A model-free self-check lives in `find_circuits/suppression_selfcheck.py`.

**The router verdict is recorded, never a gate.** `run_model_pipeline.build_steps` inserts a ROUTER
step (`task_router/router_diagnostic.py`) after the task circuits' `smallest` and before the first
suppression row. Each suppression row's `smallest` is a `gated_smallest` step that always runs and
receives the diagnostic's (cued, wrong) verdict via `--router-verdict`, which is echoed into the
payload. A missing `diagnostic.json` raises (it indicates a stage-ordering bug).

**An unfaithful base does not block a row.** Both fringe modes fall back to the base's
best-available candidate (highest cached mean-ablation whole-sequence accuracy; ties → fewer edges →
name; `best_available_candidate`, shared with `router_diagnostic.py`), evaluate the row fully, and
record `base_faithful: false`, `base_accuracy` and `base_candidate`. Only a base with zero candidates
raises. `router_diagnostic.py` applies the same fallback per task (`task_faithful` in
`diagnostic.json`); a pair degrades to logit-lens-only only when a task has zero candidates. The
diagnostic write is atomic (`os.replace`), so concurrent per-circuit children cannot interleave it.

## smallest_circuit.json

The payload builders (`build_payload` / `build_no_faithful_payload` in
`smallest_faithful_circuit.py`) are the schema; derive the key list from them rather than from a
copy. Three things about it are load-bearing.

**The accuracy key names the edge set it was measured on.** Plain rows report
`whole_sequence_absolute_accuracy`, the reported circuit's own. Contrast and Suppression rows report
`source_candidate_absolute_accuracy`, because the reported `heads`/`edges` are the fringe while the
accuracy belongs to the whole source candidate. A fringe is a detector or a suppressor, never a
sufficient circuit, so it has no standalone task accuracy; its evidence is `s_union` (Contrast) or
release + collateral (Suppression). `gates.candidate_accuracy` refuses a row carrying both keys or
neither.

**Every row carries what it takes to distrust it:** `perf_full` and `perf_empty` (without them no
reader can tell a real circuit from a constant-output scope or a weak checkpoint),
`relative_recovery_sequence_accuracy`, `n_heads_full` / `n_edges_full`, `min_accuracy_threshold` /
`min_relative_recovery`, and `pairs_regime` / `pairs_window` / `n_requested`. The no-faithful case
carries `circuit_name: null`, `no_faithful_circuit: true`, the same provenance, and
`best_available_accuracy`.

**Skip-if-done matches provenance, not just the threshold** (`_output_done`): `selection_regime` +
`candidate_set_sig` + `n_requested` + `pairs_regime` + `pairs_window`, plus a sibling `.pt` for a
winner. `candidate_set_sig` is an md5 of the candidate-directory name set, i.e. the selection's
input universe: without it, a later run that merely adds a candidate would leave the old winner in
place even though a smaller one now exists. The match is on `n_requested`, not `n_test`, because
the `answer_mask` makes the kept-row count smaller than the requested one on any pool with
agreeing rows.

## Sub-circuits only

The pipeline produces only the sub-circuits declared in `models/circuits.py` (requested through
`models/human_circuit_finding.json`; see [declaring_circuits.md](declaring_circuits.md)). There is
no implicit whole-model circuit, and a model with no declared sub-circuit produces nothing.

## Checkpoint format

The whole stack, training included, runs on a TransformerLens `HookedTransformer`. A checkpoint
`<model>.pt` is a dict `{config, model_state_dict, algorithm_id, metrics, epoch}` with
TransformerLens parameter names. To load one outside the pipeline, build a
`HookedTransformerConfig` from `ck["config"]` with `normalization_type="LNPre"` (the state dict
carries no LayerNorm parameters; loading without LNPre silently breaks the model) and call
`load_state_dict(strict=False)`. Inside the pipeline, always use `model_registry.load_model`.

## Multi-task (TaskMux) models

One model, N tasks, token 0 selects the task (the breadth axis in
[variation_matrix.md](variation_matrix.md)). Input = `[task_token, x_0..x_9]` (length 11); task i
uses token `10+i`, above the digit vocabulary, so `d_vocab = 10+N`. `DSLExpr.vocab_size()` (base
default 10, `TaskMux` returns 10+N) is the contract the trainer and loader read. Every member must
be length-preserving (10 → 10) so the output length is uniform; the constructor rejects anything
else.

**algorithm_id grammar.** `decode_algorithm_id` splits ids on `_`, so a mux id is one
underscore-free token: `mux<N>.<m1>.<m2>...<mN>`, where `<mi>` is member i's algorithm_id with `_`
replaced by `-` (e.g. `mux3.rev-id.dblodd-id.psum-id` = TaskMux([rev_id, dblodd_id, psum_id])).
`.` separates members, `-` stands in for `_` inside a member, and encoding rejects member ids that
already contain either character (negative-`Rotate` ids like `rot-1_id` and nested muxes are
therefore not encodable as members; nested muxes are impossible anyway, since their input length is
11). The leading `<N>` is redundant with the member count and is asserted at decode time as a
corruption check.

**Answer positions.** Dataset answer positions must derive from the decoded expression's
`input_length` (11 for a mux), never from `conditions.INPUT_LEN` (10). Getting this wrong wraps the
last answer index to `out[9]` via negative indexing in `dataset_to_pairs`, scoring an untrained
input logit and never scoring the last output. The loader raises on any negative output index, so a
stale dataset fails loudly.

**Gating/outvoting diagnostic** (`task_router/router_diagnostic.py`): for each ordered task pair
(A, B) on A-cued inputs, it reports B's distinctive fringe (B's smallest-circuit edges minus A's)
activation share relative to its own-task level, plus the logit(B(payload)) − logit(A(payload))
margin at the output positions where the two tasks disagree (final logits and a per-layer logit
lens). Verdicts: gating (fringe off), outvoting (fringe on, wrong answer suppressed), superposition
(wrong answer live). `--dry-run` validates structure without a model. Output:
`task_router/<model>/diagnostic.json`.

## Probes

Probes are linear by design (an MLP probe would answer "is the information there", not "does the
model read it off this way") and are fit per (site, position) on causally determined positions:
`variables.causal_positions` resamples the suffix after each declared position, and a position
whose label moves under suffix resampling cannot live in a causal residual stream. A pooled
single-`W` fit hides position-rotating codes, so the verdict uses the per-position fits.
`site_L{l}.pt` keeps the pooled `W`/`b` (the probe-metric finders' default readout, see below)
alongside the per-position `W_pos`/`b_pos` that carry the verdict, which `eap.py --probe-readout
position` reads instead.

Train/test splits are on row index (`probes/train_probes.py`), never on flattened (row, position)
examples, so the pooled fit's holdout never shares a row with training. One `seed` drives both
`sample_inputs` and every layer/position split, so the same holdout rows serve while the
best-of-L layer is chosen.

### Two position sets, two verdicts

**Input** positions (the declared positions) → `verdict`. **Generation** positions,
`input_len .. n_ctx-1`, one fit per (declared position × decode step) → `gen_verdict`
(`train_probes.fit_generation`). Without the generation scan, a variable that only materialises
while the answer is decoded would be undetectable, and every `absent` verdict would be an inference
about a phase that was never measured.

Both use the same three-way rule (`train_probes.verdict`, on the position-mean of the per-position
accuracies; `TRIVIAL_ACC = 0.9`, `ERR_REDUCTION = 0.5`, all comparisons non-strict):

    floor = 1 - 0.5*(1 - majority)
    trivial_from_input  iff  baseline >= 0.9 and baseline >= floor        # tested first
    decodable           iff  some layer l has (1 - acc_l) <= 0.5*(1 - baseline) and acc_l >= floor
    absent              otherwise

`baseline` is the layer-0 `resid_pre` probe ("was this already in the embedding?") and `majority`
is the per-position majority-class rate. The floor halves the majority-class error rather than
adding a fixed margin, so it stays satisfiable at any majority rate (an additive rule such as
`acc >= baseline + 0.1` is unsatisfiable when the baseline is above 0.9). `trivial_from_input`
short-circuits, so a site satisfying both is reported trivial. In generation space the baseline is
the layer-0 embedding of the model's own emitted token, so the trivial case is named
`trivial_from_emitted_output`, and generation decodability above that baseline means "computed",
not "read back from the token just written". Each declared position takes its best decode step, an
optimistic statistic, so a generation `absent` is the strong result.

A variable can be absent while the input is read and decodable while the answer is emitted; the
two phases have different baselines, so `gates.decodable_phase` records which phase a decodable
reading came from.

The decode regime matters: `--decode free` (default) feeds the model its own argmax, which is what
inference does; `--decode teacher` feeds the DSL ground truth, which leaks the answer into the
residual stream. `decode_seq_acc` records the divergence either way.

### What a probe verdict does not tell you

- **`steered_rate` measures the probe, not the model.** It is the fraction of pairs where the probe
  read the source class before patching and some `alpha ∈ (0.5, 1, 2, 4, 8)` makes the same probe
  read the target class (`probes/validate_probe.py`), i.e. only that steering was attempted. The
  model-output criterion is the separate `flip_rate`, and the causal gate is on the output:
  `flip_rate >= 0.80` and `collateral_rate <= 0.05` (`FLIP_GATE`, `COLLATERAL_GATE` in
  `repo_tools/gates.py`, mirrored in `validate_probe.py`, whose `--gate` defaults to `output`).
- **`check_row` is an independent cross-check, not a proof.** Every variable class validates its
  labels against `self.expr.eval` (no `labels()` body calls it, and no `check_row` calls
  `self.labels()`), so none is tautological, but some checks constrain the labels more tightly than
  others.

### Probe-metric discovery (find_circuits/eap.py and acdc.py)

This is how ProbeCircuits and BridgeCircuits are found. `--probe <variable>` swaps the discovery
metric from the output logit difference / KL to a probe readout at `blocks.{l}.hook_resid_post`.
Artifacts follow the probes contract: `<probes_root>/<model>/<variable>/site_L{l}.pt` with
`W [C, hidden], b [C], layer, hook, positions [P], n_classes, accuracy, normalize` (hidden = the model width); the root
resolves `--probes-root` > `$PROBES_DATA_ROOT` > `<pipeline>/probes_data`. With no
`--probe-layer`, the site with the highest stored accuracy wins (pin `--probe-layer` when the first
decodable layer is wanted). `site_baseline.pt` (the layer-0 control) is never globbed.
`normalize: "ln"` is TransformerLens LayerNormPre (center, then RMS) applied to the selected
positions before `W, b`. With `eap.py --probe-readout position`, each probed position is read by its
own `W_pos`/`b_pos` instead of the shared `W, b`.

Capture-site subtlety (verified empirically): TransformerLens implements backward hooks through a
wrapper applied to the HookPoint's returned output, so a tensor captured by a forward hook at the
probe site bypasses the wrapper; backward hooks there never fire (the direct-into-site column
would stay silently zero) and later-added same-point forward hooks would be missed. The capture
therefore rides `blocks.{l+1}.hook_resid_pre` (bitwise the same value, downstream of the wrapper),
hence probe layer `< n_layers-1` in both finders.

EAP-IG mechanics (all instance patches on the built `Graph`; the vendored source is untouched): the
residual at the probe site is captured by a permanent forward hook (undetached, so
`metric_value.backward()` differentiates through the readout); the metric ignores logits entirely.
The clean-reference probe log-probs travel through the opaque `labels` tensor ([B, P, C] float),
replacing the [B, 2/4, P] logit-difference transport. `graph.nodes['logits'].in_hook` (and its
parent edges' `.hook`) are retargeted to the probe hook so direct-into-site edges are scored in
attribution and reconstructed in evaluation, and `graph.prev_index` is shadowed to clamp the logits
node to the causal cutoff `1+(l+1)(H+1)`. This keeps `make_hooks_and_matrices` (attribution) and
`make_input_construction_hooks` (evaluation) from mixing writers above the probe layer into the
site (`retarget_graph_to_probe`). `real_edge_mask` is masked above the cutoff, so top-n never
selects causally dead edges and the top-n clamp reports the reduced edge universe; after
attribution the same cutoff is applied to `graph.scores` and asserted. The attribution loss is
KL(clean probe ‖ current probe); faithfulness is probe-argmax agreement (circuit vs full vs
corrupted) through the usual recovery ratio. The vendored `evaluate_baseline` runs
corrupted-then-clean per batch, so a capture-reading metric would score the wrong forward; probe
mode computes baselines with its own two-forward loop.

ACDC mechanics: `things.validation_metric` is swapped wrapper-side for probe-KL read from a
permanent capture hook. `TLACDCExperiment.__init__` asserts zero forward hooks before its
topological sort, and TransformerLens's `reset_hooks()` spares permanent hooks by default, so the
capture installs itself on the first metric call (which happens inside `__init__`, after the
assert) and re-runs the one forward it missed. Two vendored behaviours required wrapper fixes:

1. `current_node_connected()` deletes all incoming edges of any node with no present outgoing edge
   (connectivity to the output node) and asserts the metric did not move. With a mid-layer probe
   every output-node edge is zero-effect and gets removed, so the cascade would delete the probe
   circuit itself and trip the assert. Probe mode instance-patches
   `exp.current_node_connected = lambda: True`; tau alone deletes the exactly-zero-effect downstream
   edges.
2. The ACDC graph has no receiver node at a mid-stream `resid_post`, so direct writer → probe edges
   cannot be tested by `exp.step()`. After the run, a wrapper sweep patches each layer-≤l writer's
   direct contribution out of the clean residual at the site (corrupted activations; zeros under
   `--zero-ablation`) and keeps `writer→logits` iff the probe-KL jump clears tau: the same
   intervention ACDC would apply if the node existed, measured against the clean model rather than
   the greedy partial state. These edges carry `direct_to_probe: true` in `surviving_edges` and are
   counted in `n_direct_probe_edges`.

Under ACDC the final layer is excluded twice over: besides the capture-site rule above, the final
`resid_post` is ACDC's intervened output node, so a same-point capture would read pre-intervention
values.

Candidate naming: `eap_probe_<variable>_top<n>` / `acdc_probe_<variable>_thresh_<tau>`, disjoint
from the canonical `eap_top*` / `acdc_thresh_*` names the audit depends on. The `smallest` stage
excludes them: `_is_variant_candidate_dir` skips any directory containing `_probe_` or ending in
`_zero` / `_pos`, so probe candidates are never re-scored on the output metric, and they are left out
of `candidate_set_sig` for the same reason.

**Positional attribution (experimental finder option, not a circuit class).**
`--zero-ablation` and `--use-pos-embed` thread two literals into `TLACDCExperiment` so attribution
scores which edges route positional rather than token-value information; `--use-pos-embed` senders
`hook_embed` / `hook_pos_embed` both map to the drawer's `input` node (the token/position split
stays visible in `surviving_edges.sender_label`). These candidates (suffixes `_zero` / `_pos`) are
not comparable on the output metric and are excluded from selection. `--probe --use-pos-embed` is
rejected: the direct sweep has no positional-embedding corruption reference, since the vendored
positional corruption is a seeded shuffle, not a resample.

## Training recipe

`models/train_models/hparams.py` is the one per-architecture recipe table, read by both training
paths (`train.py` and the stacked `train_stacked.py`), and `REGIME = "perarch-v1"` is stamped into
every checkpoint, so the recipe is a property of the architecture rather than of which script ran.
The checkpoints shipped with the repo do not all share one recipe; see the known limitations in
[../models.md](../models.md).

`hparams.BATCH` is the effective batch; `per_rank_batch(hp, world_size)` splits it and raises when
it does not divide, so `--nproc_per_node` changes throughput only, never the recipe. The learning
rate is scaled by width and depth from a measured `large` anchor; tiers listed in
`LR_PENDING_MEASUREMENT` use the rule's output rather than a measurement and are printed with a
trailing `*`.

`promote.py` is the one promotion gate (`GATE = 0.95`); a checkpoint with no recorded metric is
refused rather than promoted on trust.

In the stacked (vmap) trainer, a frozen (banked) model must take no update. Masking the loss
produces a zeros gradient, not `None`, so AdamW would keep stepping and decoupled weight decay
would shrink the "frozen" weights; banked rows are therefore rewritten from the bank after every
optimizer step. See [stacked_training.md](stacked_training.md).

## Acceptance gates

`repo_tools/gates.py` is the one gate set for which rows the summary scripts report; it is imported
by `repo_tools/make_taxonomy.py`, so a row cannot be admitted under one rule and summarized under
another. Nothing here feeds back into selection.

Each gate is keyed to the metric the row itself reports, because the classes do not share one: a
DomainCircuit row reports an absolute accuracy on its own edge set, a ContrastCircuit row reports
`s_union` over a fringe, and a SuppressionCircuit row reports a release rate against the reference it
passed on. Gating a row on another class's metric is the bug the module exists to prevent, which is
also why the Contrast/Suppression accuracy must be named `source_candidate_absolute_accuracy` and
why `manifest_row_defects` refuses an unqualified `accuracy` next to an edge count it was not
measured on.

The set: the owning checkpoint reaches `>= 0.95` and its accuracy is recorded; a non-empty node set
(counting MLP/MoE nodes, which `smallest_circuit.json` does not carry; they come from the winner's
cached eval); a DomainCircuit strictly above its own threshold and beating its empty baseline; a
ContrastCircuit with `s_union` above its threshold and `d_control <= 0.05`; a SuppressionCircuit
passing at least one sever reference at `collateral <= 0.05`; a probe decodable at input or
generation positions with `flip_rate >= 0.80` and `collateral_rate <= 0.05`. The DomainCircuit
comparison is strict here and non-strict in selection, so a row sitting exactly on 0.85 is
selectable but not reported. Suppression rows are scored on `release_of_record` (the reference they
passed on) rather than the stored `selection_score`.

`repo_tools/repick_smallest.py` re-picks a winner from cached evals only when those caches carry the
current pool regime and window and postdate the dead-edge strip; anything else is reported as
`stale_regime` and left alone. A re-picked row is written through the stage's own `build_payload`,
so it cannot acquire a divergent schema. It never demotes an existing winner to `no_faithful`.
