# Variation matrix: circuit classes × model variations

Which model-side variation each circuit class needs, and why. This is the experiment-design layer
over [circuit_types.md](circuit_types.md) (what each class is, as built) and
[variation_axes.md](variation_axes.md) (the literature behind each axis). It records the
**predictions** that motivated the eight architectures and the multi-operator models; measured
per-class, per-architecture results are in [results/zoo/stats.md](../../results/zoo/stats.md), and
the structural reasons a class can be absent are in
[circuit_types_across_architectures.md](circuit_types_across_architectures.md).

## The frame

Every experiment is a **(model, question)** pair, where the question is "which subgraph of this
model performs this function?". Circuit classes vary the question; model variations vary the
model. Directions that do not have this shape (circuit algebra, formation over training, multi-seed
universality, comparison against compiled ground-truth programs) are comparisons and
meta-evaluation rather than circuit finding, and are listed as open directions in
[variation_axes.md](variation_axes.md).

DomainCircuits are the default question for every model; the other classes extend the question set
and none replaces it.

## Model variations

**N1 — size.** Tiers are `TIER_PRESETS` in
[interp_tools/model_loading.py](../../interp_tools/model_loading.py); a model name's `__<variant>`
suffix selects the tier. The intent is *controlled* shapes, each varying one thing:

| tier | suffix | geometry | what it isolates |
|---|---|---|---|
| Small | `__small` | 2L×4H, hidden size 64 | the capacity floor |
| Large | (none) | 4L×8H, hidden size 128 | the default instrument |
| Wide | `__wide` | 4L×16H, hidden size 256 | width (extra heads) at fixed depth |
| Deep | `__deep` | 8L×8H, hidden size 128 | depth at fixed heads per layer |
| XL | `__xl` (tier key `xlarge`) | 8L×16H, hidden size 256 | both; the capacity ceiling |

Excess capacity along an unneeded axis is not neutral: spare width buys redundant parallel copies
and backup or suppression heads, which can make ContrastCircuits and ProbeCircuits harder to read.
Escalating a model that fails the training gate should therefore go along the axis the task needs
(wider for breadth, deeper for depth) rather than to the generic bigger tier; `UPGRADE_LADDER` in
[pipeline/run_model_pipeline.py](../../pipeline/run_model_pipeline.py) implements the current
escalation map.

**N2 — feed-forward block.** MLP (`__mlp`, one GELU MLP per layer) and MoE (`__moe4`, `__moe8`, a
top-1 mixture of experts per layer), at Large's attention geometry. Both are TransformerLens
configurations, so the same finders run on them; each layer's block enters the attribution graph
as a single `m{l}` node.

**N3 — number of operators.**

- **breadth**: one model, N tasks, a cue token selects the operator (`TaskMux`, `mux{N}.…`).
- **depth**: a composition `A(B(x))`, operators stacked, each feeding the next (a composite DSL id).
- **combined**: a breadth model whose tasks are themselves compositions.

## Circuit classes

- **DomainCircuit**: a condition restricts the input domain; the circuit is the model's behavior
  on that domain (segrev3's `reverse_only` = the no-3 domain).
- **ContrastCircuit**: a minimal clean/corrupt edit toggles one predicate; the circuit is the
  machinery for that predicate (designs in [contrast_pair_catalog.md](contrast_pair_catalog.md)).
- **ProbeCircuit / BridgeCircuit**: the target is an internal variable read by a linear probe
  instead of the output logits; a BridgeCircuit probes the intermediate value `B(x)` of a composite
  (catalog in [probe_variable_catalog.md](probe_variable_catalog.md)).
- **SuppressionCircuit**: in a mux model, the machinery that keeps a competing task's answer down.
  The original prediction was negative-sign edges; the pipeline instead selects on a release rate
  and never consults edge sign (see [circuit_types.md](circuit_types.md)).
- The positional finder option (`--use-pos-embed --zero-ablation`) is an experimental finder
  option, not a sixth class: it attributes positional rather than token-value routing, and its
  candidates are excluded from selection because they are not comparable on the output metric.

## The matrix

| | N1 size | N2 +MLP | N2 +MoE | N3 #operators |
|---|---|---|---|---|
| **DomainCircuit** | ● default at every tier | ●● the twin-comparison object | ● runs unchanged (MoE = one node) | ●● breadth vehicle (condition = cue token) |
| **ContrastCircuit** | — (expected to degrade with overcapacity) | ●● threshold-detector subset | ●●● flagship pairing | ●● new contrast types |
| **ProbeCircuit / BridgeCircuit** | ● follows the variables | ●● where state lives | ● router already observable | ●●● the enabling tool |
| **SuppressionCircuit** | ●●● predicted driver | — | ● exploratory | ●● gating vs outvoting per pair |

●●● load-bearing (the pairing is the experiment) · ●● strong (sharp testable prediction) ·
● useful but secondary · — not needed.

## Per-class reasoning

### DomainCircuit

Condition + collapsed `eval` in [pipeline/models/circuits.py](../../pipeline/models/circuits.py),
resample pairs within the domain, and the full discovery / smallest / audit stack. It carries weight
in three cells:

- **×MLP/MoE**: the comparison object for architecture twins — the same question on an
  attention-only model and its feed-forward twin (winner, `n_heads`, `n_mlps`, faithfulness).
- **×N3 breadth**: a per-task circuit in a mux model is a DomainCircuit whose condition pins the cue
  token (`conditions.task_pin`), so breadth models need no new circuit machinery for their task
  circuits.
- **×N1**: the default output at every tier.

Known limits, which are why the other classes exist: it finds the executor for a domain but cannot
isolate the detector or controller that decides the domain; constant-output isolations are
degenerate (no clean/corrupt contrast); and it only sees variables that reach the output.

### ContrastCircuit

**Size.** Prediction: overcapacity distributes the detector across backups and lowers differential
necessity (`d_affected`), so contrast questions are best asked at Large. Caveat for testing it:
with the absolute EAP ladder the largest candidate is `eap_top800`, a shrinking share of the graph
as size grows (the whole graph at Small, 3.6% at XL), so a size gradient measured with that ladder
is confounded with candidate reach. The scale-relative ladder
([circuit_types_across_architectures.md](circuit_types_across_architectures.md#7-the-eap-candidate-ladder-absolute-vs-scale-relative))
removes the confound.

**+MLP: the threshold subset.** Digit-identity detectors (segrev3's is-3) are QK-implementable, so
attention suffices. Threshold and comparison detectors (an accumulator crossing its reset
threshold, a ≥5 test, compare-swap ties) are where a nonlinearity is needed. Prediction
(MLP-as-decider): in the MLP twin, the ContrastCircuit for a threshold predicate concentrates into
MLP nodes. Caveat: the zoo's attention-only and feed-forward checkpoints were trained with
different recipes, so this comparison is recipe-confounded until the models are retrained under
the shared table in `hparams.py`
([circuit_types_across_architectures.md](circuit_types_across_architectures.md#9-training-recipe-confound)).

**+MoE: flagship — the router is an explicit decision site, and contrast pairs are the matched
instrument.** Experts are per-token MLPs and cannot move information across positions, so an
executor like reversal can never live in an expert; the router is a candidate home for the
*predicate*, not the executor. Routing is per token and per layer, so a sequence-level branch
(no-3 inputs → pure reversal) can only appear as "the boundary expert never fires on this input".
Because a pair flips the predicate while holding content fixed, it buys:

- **router-as-detector test**: does routing flip at exactly the affected positions? Random corrupts
  cannot answer this, since content changes everywhere.
- **free layer bound**: routing flips at layer ℓ ⇒ the predicate is computed upstream of ℓ.
- **built-in causal switch**: pin routing to the clean pattern while feeding the corrupt input; if
  the output follows the pinned routing, the branch genuinely lives in the router.
- the negative result (the router is content-agnostic load balancing and the detector stays in
  attention) also answers whether SGD uses an explicit branch primitive.

This test needs per-expert graph nodes, or expert routing read on clean/corrupt pairs. The pipeline
has neither: each MoE block is one `m{l}` node, and the only reader of the routing hooks,
[`pipeline/repo_tools/moe_routing_census.py`](../../pipeline/repo_tools/moe_routing_census.py),
runs on uniform random inputs (its output is not included in this release).

**#operators: two contrast types that exist nowhere else.** Breadth: toggle the cue token — a
one-position edit whose circuit is the router itself. Depth: toggle the inner operator's trigger
inside a composite — does the composite reuse the standalone model's detector?

### ProbeCircuit and BridgeCircuit

**The ContrastCircuit / ProbeCircuit boundary — event vs state.** A ContrastCircuit predicate is an
event at a moment (did the reset fire at position t?): its pair straddles the threshold and the
toggle is visible downstream (a non-empty output mask). A probe variable is state carried through
time (a 13-valued accumulator register): the output exposes at most its value mod 10, so part of
the register never reaches the output, no contrast pair with a non-empty mask can exist for it by
construction, and a probe metric is the only instrument. The two classes partition the mechanism:
ContrastCircuits cover what reaches the output, ProbeCircuits what does not; the catalog's
empty-mask pairs are ready-made probe datasets. Expected anatomy: register maintenance
(ProbeCircuit) → comparator (ContrastCircuit) → output.

**Size: follows the variables.** Depth becomes load-bearing only when the variable set is
multi-stage, i.e. for composite models.

**+MLP.** Prediction: nonlinear state (a mod-13 register, a seen-set) crystallizes in MLP hidden
units where an attention-only model encodes it diffusely. The competing expectation is that a
feed-forward block recomputes a value where it is consumed instead of writing it to the residual
stream, which would make it *less* linearly decodable; attention-only vs twin decodability decides
between them.

**+MoE: deliberately weak.** MoE makes the most probe-worthy variable (the branch decision)
readable from the router, and per-token experts do not carry cross-position state. Leftover
question: does routing partition by register state (near vs far from a threshold)?

**#operators: the enabling tool.** With output metrics the finders return the whole pipeline of
stages (every stage affects the output). The decomposition instruments are the bridge-value probe
(a BridgeCircuit at the seam is operator B's machinery) and, in breadth models, a task-identity
probe.

### SuppressionCircuit

**MLP/MoE: not needed.** Published suppression circuits are attention-head phenomena
(copy-suppression heads in GPT-2; suppression in a one-layer attention-only digit sorter). Negative
OV needs no MLP. Routing-as-suppression in MoE (an unselected expert is gated off) is a different
mechanism and exploratory only.

**Size: the one class where N1 is the experiment.** Prediction: suppression, backup and calibration
are overcapacity phenomena — a minimal model has no spare heads for them — so suppression should be
absent at Small and grow with tier, with Wide separating width overcapacity (the hypothesized
driver) from depth. Testing it requires mux models that train at every tier; outcomes are in
[results/zoo/stats.md](../../results/zoo/stats.md) and summarized under prediction 2 below.

**"Wrong task" in a multi-operator model.** On input x cued for task A, the wrong task's output is
the DSL-computable counterfactual B(x), evaluated only at positions where B(x) ≠ A(x)
(whole-sequence comparisons are vacuous). In a composite, the analogue is the truncated pipeline's
bridge value B(x) against the full A(B(x)). The suppressor's operational signature: ablating it
does not break the right answer, it *releases* the wrong one.

**The gating / outvoting fork** (SuppressionCircuit × number of operators). The model keeps B out of
A-cued outputs in one of two ways, and which one is an empirical result per model and ordered pair,
recorded on every suppression row as `router_verdict` by
[pipeline/task_router/router_diagnostic.py](../../pipeline/task_router/router_diagnostic.py) (never
a selection gate):

1. **Gating (B is off).** B's distinctive fringe (B's circuit minus A's; the raw union would make
   the test vacuous) is inactive on A-cued inputs (`fringe_on_ratio < 0.5`). No negative circuit
   exists; the suppressor is the router, and a cue-token ContrastCircuit finds it.
2. **Outvoting (B runs and is cancelled).** B's fringe is active, B(x) evidence appears mid-stack
   (logit lens), and something downstream cancels it. This is a true SuppressionCircuit: sever the
   fringe and measure whether the wrong answer is released.

The diagnostic also reports `superposition` (the wrong answer is live at the output) and
`empty-fringe` (B's heads are a subset of A's). Selection keys on the release rate, under either
the cued or the wrong-cue ("tonic") sever reference; see [circuit_types.md](circuit_types.md) and
[design.md](design.md).

## Multi-operator models: a ladder of decomposition problems

Multi-operator models pose circuit-discovery problems that a single-operator model cannot, in
increasing order of difficulty for the method:

| rung | model | question | what it stresses | natural geometry |
|---|---|---|---|---|
| 1 | breadth (cue token, N atomic operators) | the circuit of every task | coverage and separation of overlapping circuits | Wide |
| 2 | depth (`A(B(x))`) | the circuit of each stage | stage boundaries are invisible in the output | Deep |
| 3 | combined (tasks are compositions) | one named circuit under one operator | targeted retrieval amid distractors | XL |

- **Rung 1.** The targets are DomainCircuits (condition = cue token). The N circuits will not be
  disjoint: the interesting structure is the shared core vs each task's fringe, plus the router.
  Operator similarity is the knob.
- **Rung 2.** A single output-metric attribution run cannot separate stages, because every stage
  affects the output; probes at the seam (BridgeCircuits) and contrast toggles of the inner
  operator are needed. Precondition: verify the model learned a staged implementation at all
  (shortcut composition has no stage structure to find).
- **Rung 3.** Combines rungs 1 and 2 and mirrors realistic interpretability work, where the job is
  to find one circuit under a description rather than enumerate all of them.

Design points:

1. **Train the atomic operators as standalone tasks inside the combined model** (cue tokens for A,
   B and A(B)). Co-training constituents is what makes composed circuits form at all (see the data
   mix axis in [variation_axes.md](variation_axes.md)), and the standalone circuits become in-model
   references for checking a decomposition.
2. **Exact counterfactuals at every rung.** The DSL gives exact outputs for every operator on every
   input (faithfulness on the right domain, wrong-operator margins, bridge-value patching), so no
   hand-labelled edge lists are needed.
3. **Depth budget.** Roughly two layers per nontrivial stage (one deduction step per layer, at most
   L−1; ⌊log₂k⌋+2 layers for k dependent lookups): 4 layers hold 2 stages, 3–4 stages want the Deep
   tier, and 5 or more stages will hit trainability before capacity. Breadth capacity grows
   sublinearly (shared core plus thin fringes) and wants width, not depth. An MLP is needed if a
   constituent operator is arithmetic; MoE is breadth's experimental arm (with a dense model as
   control), never a requirement.

## Predictions to check

| # | prediction | status |
|---|---|---|
| 1 | ContrastCircuit detectors degrade with size (overcapacity spreads the detector, `d_affected` drops) | Not testable with the absolute ladder: its 800-edge cap covers a shrinking share of the graph as size grows. Needs a scale-relative (`frac`) rebuild; numbers in [stats.md](../../results/zoo/stats.md). |
| 2 | SuppressionCircuits are an overcapacity phenomenon: absent at Small, growing with tier | Observed in the zoo snapshot (zoo selection rule, [../circuits.md §6.2](../circuits.md)): no faithful SuppressionCircuit at Small (0 of 26 rows), Wide (0/26), Deep (0/6) or XL (0/26); faithful ones only at Large (30/94), MLP (26/74), MoE4 (36/74) and MoE8 (33/74), all via the wrong-cue (tonic) sever reference. Mux coverage is thin outside the 4L×8H family (2, 2, 1 and 2 of the 19 mux run-list entries have a checkpoint at Small, Wide, Deep and XL, against 12–13 at the other four), so the absence above Large does not test overcapacity. [stats.md](../../results/zoo/stats.md) §1, §3, §6. |
| 3 | ×MLP: threshold detectors concentrate into MLP nodes | Recipe-confounded in the zoo (different training recipes for attention-only and feed-forward checkpoints); needs retraining under the shared `hparams.py` table. |
| 4 | ×MoE: the router is the detector's home | Untested: the attribution graph has no per-expert nodes, and this release reports no expert-routing statistics; `moe_routing_census.py` measures expert usage given the checkpoints, but its output is not included. |
| 5 | Positional routing needs NoPE to become a discoverable circuit | An experimental NoPE tier was tried and dropped; only the positional finder option remains. |

The NoPE reasoning, kept as motivation: positional routing is shallow and enters through QK, so MLP
and MoE are irrelevant to it and size barely matters. Deleting the positional embedding forces
position to be *constructed* by a circuit, which turns "find the position-computing circuit" into a
genuine question, and avoids the degeneracy of fixed-length inputs (resampling the input never
changes the positional embedding, so a pos-embed contrast is identically zero). Without
retraining, the positional finder option re-targets the attribution input from the token embedding
to the positional embedding with zero ablation.

## Open experiments

- Edge-sign retention in the finders, to test the original negative-edge prediction for
  SuppressionCircuits directly.
- Per-expert graph nodes for MoE, and expert routing read on contrast pairs (prediction 4 and the
  router-as-detector test).
- A zero-ablation control for SuppressionCircuit release rates.
- Retraining the attention-only and feed-forward architectures under one recipe (prediction 3).
- A scale-relative-ladder rebuild of ContrastCircuits and SuppressionCircuits across sizes
  (prediction 1).
