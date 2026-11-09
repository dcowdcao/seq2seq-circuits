# Circuit types across architectures

Which circuit classes can be built at which of the eight architectures, what has to be true of a
model before a class can exist on it, and what an absence does and does not tell you. This page
is the per-class reasoning; the per-architecture view is
[circuit_notes_by_architecture.md](circuit_notes_by_architecture.md). Class definitions and
selection rules are in [circuit_types.md](circuit_types.md) and [../circuits.md](../circuits.md).

All counts, rates and per-architecture comparisons from the trained zoo are in
[results/zoo/stats.md](../../results/zoo/stats.md). Nothing on this page is a measurement.

## 1. The eight architectures

Geometry is `TIER_PRESETS` in [interp_tools/model_loading.py](../../interp_tools/model_loading.py);
a model name's `__<variant>` suffix selects the tier (no suffix = `large`). "Live edges" is the size
of the attribution graph an ablation can actually touch, computed from the geometry by
`canon.live_edge_budget` in [data_gen/canon.py](../../data_gen/canon.py) (it excludes the `m{l}`
slots that EAP-IG's graph instantiates even for attention-only models).

| variant | suffix | layers | heads | hidden size | d_head | feed-forward | live edges |
|---|---|---|---|---|---|---|---|
| Small | `__small` | 2 | 4 | 64 | 16 | – | 81 |
| Large | (none) | 4 | 8 | 128 | 32 | – | 1281 |
| Wide | `__wide` | 4 | 16 | 256 | 32 | – | 4865 |
| Deep | `__deep` | 8 | 8 | 128 | 32 | – | 5633 |
| XL | `__xl` (tier key `xlarge`) | 8 | 16 | 256 | 64 | – | 22017 |
| MLP | `__mlp` | 4 | 8 | 128 | 32 | d_mlp = 512, GELU | 1519 |
| MoE4 | `__moe4` | 4 | 8 | 128 | 32 | d_mlp = 128, 4 experts, top-1 | 1519 |
| MoE8 | `__moe8` | 4 | 8 | 128 | 32 | d_mlp = 128, 8 experts, top-1 | 1519 |

`TIER_PRESETS` also holds `mlp_xl` / `moe4_xl` / `moe8_xl`: escalation targets at XL geometry that
keep their feed-forward flag (`UPGRADE_LADDER` in
[pipeline/run_model_pipeline.py](../../pipeline/run_model_pipeline.py)). They are not separate
architectures.

## 2. Why a class can be missing at an architecture

Every absence in a "circuits found per architecture" table resolves to one of four causes. Only the
last one says anything about the architecture.

| cause | what it means | how to check |
|---|---|---|
| **Declaration** | The class needs a model property the base algorithm does not have: a `TaskMux` for a SuppressionCircuit, a composite `A(B(x))` for a BridgeCircuit, a registered `ProbeVariable` for a ProbeCircuit. | Is the base algorithm in the class's universe? |
| **Training** | The base algorithm has no checkpoint at that architecture that passes the 0.95 training gate (`SEQ_ACC_GATE`; never trained, or trained and not promoted). | `pipeline/models/trained_models_pt/<base>__<variant>.pt` |
| **Stage** | The stage that produces the class did not run. `probes` (and the positional finder option) run only when named; the default stages are `data train eap smallest audit` (`DEFAULT_STAGES`). | Did the stage run for that model? |
| **Findability** | Built and searched on a gate-passing checkpoint; no candidate passed selection. | verdict `no_faithful` |

The first three are bookkeeping. They produce absences that look exactly like the fourth in a
summary table, so an absence is evidence about an architecture only when it is a findability
failure on a checkpoint that learned its task.

## 3. DomainCircuit

**Can exist wherever** the base algorithm declares a `DomainCircuit` subclass and has a
gate-passing checkpoint. A `DomainCircuit` is a mixin on the model's DSL operator, so every trained
architecture of a circuit-declaring model gets the circuit; there is no architecture-specific
declaration.

**What an architecture changes:** compactness more than existence. Things to keep in mind when
comparing sizes across architectures:

- Head fractions (`n_heads`, `h_frac`) count attention heads only. On the feed-forward
  architectures the MLP or MoE block of each layer enters the graph as one `m{l}` node, and a
  circuit's kept blocks show up only as `m{l}` endpoints in its edge list (and as `n_mlps` in the
  evaluation cache). A "few heads" feed-forward circuit can still contain every feed-forward block,
  so head fractions are not comparable across the attention-only / feed-forward boundary.
- The selection denominator is the live edge count above, so edge fractions are comparable across
  finders but the absolute graph grows by a factor of ~270 from Small to XL.
- At Small the absolute EAP ladder (`[50, 200, 800]`) has rungs larger than the whole 81-edge graph,
  so some candidate almost always passes; a DomainCircuit at Small is weak evidence of
  localization. The scale-relative ladder (section 7) removes the aliasing.

## 4. ContrastCircuit

**Can exist wherever** a sibling DomainCircuit (the base) exists on the same model, because the
reported circuit is the fringe beyond that base (see [circuit_types.md](circuit_types.md)).

**What an architecture changes:**

- **Candidate reach.** Contrast candidates come from the EAP top-n ladder. In the absolute regime
  the largest candidate is `eap_top800`, which is the whole graph at Small, 62% of it at Large, 53%
  at the 4L×8H feed-forward variants, 16% at Wide, 14% at Deep and 3.6% at XL (800 divided by the
  live edge counts in section 1). Any size gradient in ContrastCircuit findability measured with
  that ladder is confounded with candidate reach. Section 7 describes the regime that removes the
  confound.
- **Empty fringes at small sizes.** When the base DomainCircuit already covers nearly the whole
  graph, no candidate can fall outside it and the fringe is empty. This is expected at Small.
- **Necessity.** Selection only requires base ∪ fringe to pass; `d_affected`, `d_unaffected` and
  `d_control` are recorded, not gated. Architectures that spread a predicate over many parallel
  heads can yield fringes that are sufficient with the base but not demonstrably necessary. Quote
  `d_affected` with any per-architecture claim about detectors.

## 5. SuppressionCircuit

**Can exist only on a trained mux model.** `suppression_circuits_for` refuses any parent that does
not decode to a `TaskMux` (cue token `10 + i` selects member task `i`), and a SuppressionCircuit
also needs every member task's DomainCircuit (`task{i}`) and the router diagnostic
(`pipeline/task_router/router_diagnostic.py`) to have run first. A mux model with N tasks yields
N(N−1) ordered SuppressionCircuits.

**What an architecture changes:**

- Mux models must first train past the gate at that architecture. Where a mux model was not
  trained, or did not pass the gate, the missing SuppressionCircuits are a training absence, not
  a findability result; check checkpoint coverage before reading a zero.
- The same candidate-reach caveat as ContrastCircuits applies (section 4).
- On MoE models a router-level gate is the natural hypothesis for suppression, but the router
  diagnostic hooks only `attn.hook_z` and `hook_resid_post` and drops `m{l}` nodes: its "gating"
  verdict means attention-head suppression. This release reports no expert-routing statistics;
  [`pipeline/repo_tools/moe_routing_census.py`](../../pipeline/repo_tools/moe_routing_census.py)
  measures expert usage given the checkpoints, but its output is not included.

## 6. ProbeCircuit and BridgeCircuit

**A ProbeCircuit can exist** for any registered `ProbeVariable` of the base algorithm, on any
architecture where the `probes` stage was run and a checkpoint exists. **A BridgeCircuit can exist
only on a composite** `A(B(x))`: `bridge_variable(composite_id, k)` strips the k outermost operators
and probes the remainder, so an architecture at which no composite trains has no BridgeCircuits at
all (an empty cell, not a negative result).

**What an architecture changes:**

- **Where a variable is written.** A linear probe can only find a value that is written into the
  residual stream. A feed-forward block can recompute a value where it is consumed instead of
  carrying it, so "not linearly decodable" on an MLP/MoE model does not imply the model does not
  compute the variable; compare against the attention-only twin.
- **Probe sites.** The verdict reported in the zoo statistics is the input-position verdict
  (`verdict`). The prober in [pipeline/probes/train_probes.py](../../pipeline/probes/train_probes.py)
  also fits the generation positions (`gen_verdict`), but that verdict is not part of the reported
  statistics, so read an `absent` there as "not linearly present while the input is read".
- **Depth.** Deeper models add probe sites; they do not by themselves create later-appearing
  variables. Most declared variables are prefix-determined streaming quantities that one attention
  hop can produce.

## 7. The EAP candidate ladder: absolute vs scale-relative

The EAP finder emits a ladder of top-n candidates. Two regimes exist in
[data_gen/canon.py](../../data_gen/canon.py):

- `"abs"`: the fixed rungs `CANON_TOPNS = [50, 200, 800]`.
- `"frac"` (the default, `CANON_TOPN_REGIME`): `CANON_TOPN_FRACS = [0.02, 0.05, 0.15, 0.40]` of the
  model's live edge budget, resolved per model by `canon.topn_ladder`. A rung above the live budget
  is an error.

| variant | live edges | frac ladder | abs top rung (800) as a share of live edges |
|---|---|---|---|
| Small | 81 | 2 / 4 / 12 / 32 | > 1 (whole graph) |
| Large | 1281 | 26 / 64 / 192 / 512 | 0.62 |
| MLP / MoE4 / MoE8 | 1519 | 30 / 76 / 228 / 608 | 0.53 |
| Wide | 4865 | 97 / 243 / 730 / 1946 | 0.16 |
| Deep | 5633 | 113 / 282 / 845 / 2253 | 0.14 |
| XL | 22017 | 440 / 1101 / 3303 / 8807 | 0.036 |

In the frac regime the top rung is 40% of the graph at every architecture, so candidate reach can
no longer vary with size. The `smallest` stage additionally runs EAP's `--target-seq-acc` search for
every row (DomainCircuit, ContrastCircuit and SuppressionCircuit alike), which adds the
score-maximizing candidate that clears a strict accuracy floor. The zoo's ContrastCircuit and
SuppressionCircuit candidates were built with the absolute ladder and top out at 800 edges;
interpret cross-architecture ContrastCircuit and SuppressionCircuit comparisons in
[results/zoo/stats.md](../../results/zoo/stats.md) with that cap in mind.

## 8. Positional option

The positional finder option (`--use-pos-embed --zero-ablation`) attributes positional rather than
token-value routing. It is an experimental finder option, not one of the five circuit classes: its
candidates (`eap_top*_zero_pos`) are not comparable on the output metric and are excluded from
selection, so it has no faithfulness number at any architecture.

## 9. Training-recipe confound

Every MLP / MoE4 / MoE8 checkpoint in the zoo was trained by the per-model trainer
(`run_training_job.sh`), and most attention-only checkpoints by the stacked size-grid trainer
([stacked_training.md](stacked_training.md)); [results/zoo/stats.md](../../results/zoo/stats.md)
§8 counts the checkpoints of each kind per architecture. The module docstring of
[pipeline/models/train_models/hparams.py](../../pipeline/models/train_models/hparams.py) records
that the two paths used different learning rates, batch sizes, step budgets, dtypes and gate eval
sizes. What the repository records of those recipes:

- the size grid ran the anchor recipe, lr 1e-3 at batch 64 (`ANCHOR_LR` and the docstring), on a
  20,000-step budget (`STEPS`; stats.md §8 gives 20,000 as the median recorded step count of the
  stacked checkpoints);
- the MLP / MoE models saw about 23× more data (the comment on `SWEEP_SPACE["steps"]`), and the
  per-model trainer gated on 10⁶ eval rows against the stacked trainer's 4,096 (the comments on
  `EVAL_N` in `hparams.py` and on the eval set in `train.py`);
- fp32 and bf16 both occur in the old runs (the comment on `SWEEP_SPACE["dtype"]`), with no record
  of which path used which; every stored parameter tensor is float32 (stats.md §8);
- the only checkpoints that record a recipe are some Large ones: lr 0.004, batch 4096, at most
  30,000 steps (stats.md §8). No other checkpoint records a learning rate, batch size or
  precision.

Any attention-only vs feed-forward difference in the zoo is therefore confounded with the recipe.
Both trainers now read one per-architecture table (`hparams.for_tier`, stamped into every new
checkpoint as `metrics.hparams.regime`), so models retrained from here are comparable.

Related: XL training can diverge under a learning rate that is too high for its width. `hparams.py`
scales the learning rate with width and depth and installs a post-training tripwire on the
embedding norm at XL only (`check_we_norm`, `WE_NORM_MAX`).

## 10. Choosing an architecture for a question

| question | build at | why |
|---|---|---|
| does a domain restriction yield a sufficient subgraph? | any | DomainCircuits can be declared everywhere |
| is the circuit sparse in heads? | Wide, or a feed-forward twin (counting `m{l}` separately) | extra heads at fixed depth are the droppable resource |
| is a ContrastCircuit fringe a necessary detector? | any, with `d_affected` quoted | selection does not gate on necessity; extra heads leave room for parallel partial detectors (hypothesis) |
| suppression / routing between tasks | the 4L×8H family, where mux models train | mux training is the prerequisite |
| is an internal variable stored or recomputed? | an attention-only model, paired with its feed-forward twin | feed-forward blocks can recompute on demand |
| did a composition fuse? | a feed-forward architecture paired against its attention-only twin | the paired difference is the measurement |
| redundancy / parallel copies | Wide, Deep, XL | overcapacity is the phenomenon |

## 11. Regenerating the coverage tables

[pipeline/repo_tools/arch_coverage.py](../../pipeline/repo_tools/arch_coverage.py) prints the
presence, checkpoint-quality and per-class coverage tables to stdout (it writes no file):

```bash
python pipeline/repo_tools/make_taxonomy.py              # refresh pipeline/models/taxonomy.json first
python pipeline/repo_tools/arch_coverage.py              # every section
python pipeline/repo_tools/arch_coverage.py --section c1 # one section (repeatable)
```

`--section` takes `presence`, `quality`, or a per-class key named by the old class labels
([circuit_types.md §1](circuit_types.md#1-the-five-classes)): `c0` DomainCircuit, `c1`
ContrastCircuit, `c3` SuppressionCircuit, `c2` ProbeCircuit, `bridge` BridgeCircuit, `c4` the
positional finder option.

It reads `pipeline/models/taxonomy.json` (verdicts), the checkpoints in
`pipeline/models/trained_models_pt/` (realized geometry and the accuracy recorded at promotion),
`pipeline/models/trained_models_pt_grid/` (pre-promotion size-grid attempts),
`pipeline/models/models.txt` (the mux universe) and `pipeline/task_router/` (the SuppressionCircuit
prerequisite). The repository ships only the showcase checkpoints, so the full tables need the
complete checkpoint store; the zoo-wide numbers are in
[results/zoo/stats.md](../../results/zoo/stats.md).
