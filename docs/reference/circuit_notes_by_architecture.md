# Circuit notes by architecture

One short entry per architecture: what the architecture varies, which circuit classes it can host,
and what to watch for when reading its circuits. The per-class reasoning (why a class can be absent,
the candidate ladder, the training-recipe confound, how to regenerate the coverage tables) is in
[circuit_types_across_architectures.md](circuit_types_across_architectures.md); definitions are in
[circuit_types.md](circuit_types.md) and [../circuits.md](../circuits.md).

Every count, rate, median size and per-architecture comparison from the trained zoo lives in
[results/zoo/stats.md](../../results/zoo/stats.md). Statements below marked **observed** quote
those statistics: they come from the zoo snapshot under the zoo selection rule
([../circuits.md §6.2](../circuits.md)) and are not comparable with the showcase circuits.

## Reading a size claim: three different objects

| object | what is counted | caveat |
|---|---|---|
| DomainCircuit | the winning edge set itself | the only one of the three that is a standalone circuit |
| ContrastCircuit fringe | candidate edges minus the base DomainCircuit's edges | a small fringe sits on top of the base; it is not the whole mechanism |
| SuppressionCircuit fringe | candidate edges minus the cued task's own circuit | same caveat |

Fringe necessity is measured by mean-ablating ("severing") the fringe edges and re-running, as
whole-sequence exact match over the relevant positions (`metric_unit:
whole_sequence_exact_match` in [pipeline/find_circuits/smallest_faithful_circuit.py](../../pipeline/find_circuits/smallest_faithful_circuit.py)).
None of these metrics gates selection.

| metric | definition |
|---|---|
| `d_affected` | exact match over the diff-mask positions with the fringe intact, minus the same after severing it; ~0 means severing costs nothing |
| `d_unaffected` | the same drop at the off-mask positions (collateral damage) |
| `d_control` | the effect of severing the fringe on the base circuit's own domain pool |
| `necessity_class` | `necessary` iff `d_affected >= 0.10`, `weak` iff `>= 0.02`, else `redundant` (`make_taxonomy.py`) |

Head fractions count attention heads only. On the feed-forward architectures each layer's MLP or
MoE block is one `m{l}` node; a circuit's kept blocks appear only as `m{l}` endpoints in its edge
list.

Scope: unrestricted-input DomainCircuits (the `*_whole` / `*_full` scopes, whose condition accepts
essentially every input) must reproduce the full algorithm on the full input distribution, so they
are whole-model circuits in all but name. Hold the scope fixed when comparing architectures, and
exclude ContrastCircuits whose base is such a scope, since base ∪ fringe passing is then nearly
automatic.

## Small (2 layers × 4 heads, hidden size 64, 81 live edges)

- The absolute EAP ladder's 200- and 800-edge rungs exceed the whole graph, so a passing candidate
  almost always exists; a Small DomainCircuit is weak evidence of localization. The scale-relative
  ladder is 2 / 4 / 12 / 32 edges.
- ContrastCircuit fringes can be empty: when the base DomainCircuit already covers nearly the whole
  graph, nothing can fall outside it.
- SuppressionCircuits need a mux model to train at this size first; check checkpoint coverage
  before reading an absence (2 of the 19 mux run-list entries have a Small checkpoint; their 26
  SuppressionCircuit rows include none faithful; stats.md §1, §6).
- **Observed** in the zoo snapshot (zoo selection rule; stats.md §4): faithful DomainCircuits keep
  a median 8 of 8 heads (all heads in 65.1% of the 255) and a median 53.1% of live edges.

## Large (4 × 8, hidden size 128, 1281 live edges)

- The default architecture (no suffix) and the geometry of the segrev3 case study
  ([../segrev3_case_study.md](../segrev3_case_study.md)).
- All five classes can be built here (it hosts mux models, composites and probe variables like
  every other tier).
- **Observed** in the zoo snapshot (zoo selection rule; stats.md §4): faithful DomainCircuits keep
  a median 31 of 32 heads but a median 24.7% of live edges (n = 301). Pooled over architectures,
  99.1% of faithful DomainCircuits keep a smaller fraction of edges than of heads.

## Wide (4 × 16, hidden size 256, 4865 live edges)

- Same depth as Large with twice the heads and width: it isolates the effect of extra heads at
  fixed depth.
- In the absolute ladder the 800-edge cap is 16% of the graph (candidate-reach confound for
  ContrastCircuits and SuppressionCircuits).
- **Observed** in the zoo snapshot (zoo selection rule; stats.md §4–§5): faithful DomainCircuits
  keep a median 57 of 64 heads, and all heads in 28.1% of them (Large: 47.2%); 87.2% of faithful
  ContrastCircuit fringes have `d_affected < 0.02` (Large: 80.4%), i.e. are sufficient with the
  base but not shown to be necessary.

## Deep (8 × 8, hidden size 128, 5633 live edges)

- Same heads per layer and d_head as Large with twice the layers: it isolates depth. Compare with
  Wide, which has the same total head count.
- In the absolute ladder the 800-edge cap is 14% of the graph.
- **Observed** in the zoo snapshot (zoo selection rule; stats.md §4, §7): faithful DomainCircuits
  keep a median 63 of 64 heads, and all heads in 46.3% of them (Wide, with the same head count:
  57 of 64 and 28.1%). Of the 35 `decodable` probe variables, 94.3% are first decodable at
  layer 0.

## XL (8 × 16, hidden size 256, d_head 64, 22017 live edges)

- In the absolute ladder the 800-edge cap is 3.6% of the graph, the strongest candidate-reach
  confound in the grid.
- Training is optimization-limited: a learning rate that is too high for the width makes the
  embeddings grow and the run diverge. `hparams.py` now scales the learning rate with width and
  depth and checks the embedding norm at XL (`check_we_norm`). Every XL checkpoint in this
  snapshot clears the 0.95 gate (lowest eval sequence accuracy 0.954, stats.md §1); run-list
  coverage is in [results/zoo/stats.md](../../results/zoo/stats.md) (§1, *Run-list coverage*).
- BridgeCircuits require a composite model that trains at XL.

## MLP (4 × 8 + MLP, d_mlp 512, 1519 live edges)

- Same attention geometry as Large plus one GELU MLP per layer, one `m{l}` node each.
- Head fractions exclude the MLP blocks: a circuit with few heads can still contain every MLP.
- Recipe confound: the zoo's feed-forward checkpoints were trained with a different recipe from the
  most of the attention-only ones (675 of 725, stats.md §8; see [circuit_types_across_architectures.md](circuit_types_across_architectures.md#9-training-recipe-confound)).
- **Observed** in the zoo snapshot (zoo selection rule; stats.md §4, §7), with the recipe confound
  above: 98.8% of faithful DomainCircuits keep at least one MLP node and 81.0% keep all four, with
  a median 19 of 32 heads; 45.2% of the 93 probe variables are `decodable` (Large: 71.3% of
  115). The statistics do not break out whether threshold and comparison predicates concentrate
  into MLP nodes.

## MoE4 and MoE8 (4 × 8 + top-1 mixture of 4 or 8 experts, d_mlp 128, 1519 live edges)

- Each layer's mixture-of-experts block enters the graph as a single `m{l}` node; there are no
  per-expert nodes, so a circuit cannot select individual experts.
- The router diagnostic reads only attention-head activity (see
  [circuit_types_across_architectures.md](circuit_types_across_architectures.md#5-suppressioncircuit)).
  This release reports no expert-routing statistics;
  [`moe_routing_census.py`](../../pipeline/repo_tools/moe_routing_census.py) measures expert usage
  given the checkpoints, but its output is not included.
- Same recipe confound as MLP.
- **Observed** in the zoo snapshot (zoo selection rule; stats.md §3–§6), with the recipe
  confound: MoE4 and MoE8 give similar numbers (faithful SuppressionCircuits 36/74 and 33/74, all
  via the wrong-cue, "tonic", sever reference; DomainCircuit median edge fraction 3.9% for both;
  ContrastCircuit fringes with `d_affected ≥ 0.10`: 9.7% and 9.5%). Most faithful ContrastCircuit
  fringes are not shown to be necessary (`d_affected < 0.02` for 86.2% and 82.9%).
