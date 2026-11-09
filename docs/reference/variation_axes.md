# Axes of variation for circuit discovery

The organizing frame for the research program: a circuit-discovery experiment has independent
knob-families, and each knob below is backed by published evidence (preprint-only items flagged).
Architectures are transformer-only by scope. Companion docs:
[contrast_pair_catalog.md](contrast_pair_catalog.md),
[probe_variable_catalog.md](probe_variable_catalog.md), and
[variation_matrix.md](variation_matrix.md) (which circuit class needs which model variation).
Class definitions: [circuit_types.md](circuit_types.md); architectures:
[../models.md](../models.md).

## Where each axis lives in this repository

| axis | implemented as |
|---|---|
| 1 discovery dataset | ContrastCircuits (minimal-edit clean/corrupt pairs, [contrast_pair_catalog.md](contrast_pair_catalog.md)) next to DomainCircuits (resample pairs within a restricted domain) |
| 2 architecture | the eight architectures (Small, Large, Wide, Deep, XL, MLP, MoE4, MoE8) |
| 3a breadth | `TaskMux` models (cue token selects the task) and their SuppressionCircuits |
| 3b depth | composite models `A(B(x))` and their BridgeCircuits |
| 4 size and geometry | the Small / Large / Wide / Deep / XL size grid |
| 5 discovery target | ProbeCircuits and BridgeCircuits (probe-metric discovery, [probe_variable_catalog.md](probe_variable_catalog.md)) |
| 8 suppression | SuppressionCircuits, selected on a release rate rather than on edge sign |
| 6, 7, 9 | not implemented; open directions |

## Four axes, with evidence

### Axis 1 — Discovery dataset (clean/corrupt design)
Targeted minimal-edit pairs isolate the detector/controller machinery a resample corrupt can never
see. Designed per model in [contrast_pair_catalog.md](contrast_pair_catalog.md) (masks verified
against the DSL implementations). Validation anchors: corruption
choice determines the found circuit more than metric choice (Zhang & Nanda ICLR 2024); different
corruptions yield structurally different circuits for one mechanism (Bayat Makou et al.,
arXiv:2606.06267, preprint).

### Axis 2 — Architecture (transformer variants)
Beyond the MLP/MoE architectures already in the grid, further sub-dials with published
anchors:

| Variant | Anchor | What it unlocks | Cost |
|---|---|---|---|
| Bilinear MLP (swap GELU) | Pearce et al., ICLR 2025 (arXiv:2410.08417) | circuits from WEIGHTS alone — a second, dataset-free circuit estimate to cross-validate the finders | one-line activation change in the MLP architecture |
| Weight-sparse training | Gao et al. (OpenAI), arXiv:2511.13653 | model is trained INTO near-circuit form; neuron-level legibility with causal pruning validation | training-loop L0 constraint only; finders unchanged |
| Codebook features (VQ bottleneck) | Tamkin et al., ICML 2024; validated causally on a 100-state FSM task — isomorphic to our accumulator/reset tasks | discrete code→code transition circuits ("code #417 = state 12"); code-swap interventions | insert codebooks + finetune; extend node vocabulary |
| Transformer Programs (hard categorical attention) | Friedman, Wettig, Chen, NeurIPS 2023 oral; trained on reverse/sort/histogram/Dyck — our task family | model decompiles to a RASP-like PROGRAM = trained ground-truth circuit to score the finders against | standalone model class; decompiler replaces finders |
| LayerNorm-free | Heimersheim, arXiv:2409.13710; scaled arXiv:2507.02559 | exactly linear attribution (EAP-IG's main error term vanishes) | config flag; trainable from scratch at our scale |
| NoPE (an experimental tier was tried and dropped; it is not one of the eight architectures) | Kazemnejad et al., NeurIPS 2023; Haviv et al., Findings EMNLP 2022 | position becomes a DISCOVERABLE circuit (position-constructing heads) instead of an input node | delete pos embedding |
| Looped/weight-tied | Giannou et al. ICML 2023; Saunshi et al. ICLR 2025 (loops ≈ depth) | (head, iteration)-indexed circuits: does loop body = algorithm step? NO published circuit finding on weight-tied models — open niche | new model class; finders run on the unrolled graph |
| SIIT ground-truth twins | InterpBench, NeurIPS 2024 D&B (arXiv:2407.14494) | models with known-by-construction circuits at our scale → finder precision/recall | training-loss change only |

Skipped for cause: Backpack LMs (payoff is lexical polysemy — trivial at 10-digit vocab),
parallel attn+MLP blocks and encoder-bidirectional twins (no published interp consequence),
Mamba/RNNs (out of scope: not transformers).

### Axis 3 — Operator structure (splits into TWO axes)

**3a. Breadth (N tasks, task-id prefix).** The union hypothesis is PARTIALLY SUPPORTED with a
predicted structure:
- Direct precedent: Mondorf, Wold & Plank, "Circuit Compositions", ACL 2025 (arXiv:2410.01434) —
  one ~58M model on 10 string-edit ops; per-op circuits sparse and faithful (0.90–0.99);
  set-operations compose (union(repeat, reverse) scores 78% on echo = reverse∘repeat where each
  constituent scores 0%) but can also degrade constituents (swap∪reverse → 45% on reverse).
- Expect a large shared intersection + thin task-specific fringes: 78% head overlap between IOI
  and Colored Objects circuits (Merullo et al., ICLR 2024); same-circuit-for-same-process across
  independently trained models (Zhang et al., "The Same But Different", ICLR 2025); entanglement
  is a capacity-threshold effect hitting low-importance features first (Scherlis et al.,
  arXiv:2210.01892).
- Documented failure modes: reused nodes go POLYSEMANTIC under joint training — freezing them
  hurts (Quirke, Neo & Barez, arXiv:2402.02619); task superposition — multiple task circuits run
  at once under ambiguous cues (Xiong et al., ICML 2025) → must verify the wrong-task circuit is
  OFF, not outvoted, by corrupting the task token; circuits can lack SPECIFICITY — ablating task
  A's circuit hurts task B about as much (Li & Subramani, arXiv:2605.08348, preprint) — our
  from-scratch DSL setup is the clean testbed to adjudicate whether that is a pretraining artifact.
- The task token = a discoverable ROUTER: a small head set carries a portable function vector
  (Todd et al., ICLR 2024; Hendel et al., Findings EMNLP 2023); in from-scratch toy models the
  task signal may be weak/dispersed rather than localized (Yang et al., arXiv:2501.09240 — an
  auxiliary loss is the published fix). An attention-implemented router exists in a 3-layer model
  (D'Angelo et al., "Selective Induction Heads", ICLR 2025).
- Closest published analog of the task-id-prefix setup: Furuta et al., TMLR 2024
  (arXiv:2402.16726) — mixed modular-operation training where the operator token IS the task-id;
  distinct ops get distinct circuit signatures, similar ops "co-grok", dissimilar ops transfer
  poorly. Also He et al., NeurIPS 2024 oral (arXiv:2406.02550): one tiny model represents a
  parameterized task FAMILY and selects the member from context.
- Specificity-test nuance: ablate task-A's circuit MINUS the shared core when checking task-B
  damage — otherwise the intersection guarantees cross-task damage and the test is vacuous.
- Protocol corrections that matter: measure union faithfulness ONLY on output positions where
  tasks differ (Mondorf: copy scores 0.64 on swap full-sequence but 0.01 on differing positions);
  compare circuits by faithfulness transfer, not node overlap (Hanna et al., COLM 2024); ≥3 seeds
  (Chughtai et al., ICML 2023) with weight decay held constant.

**3b. Depth (composed A(B(x)))**. Expected layout if learned cleanly: B in earlier layers →
bridge representation → A in later layers, with A's layers shifted vs standalone-A (Wang et al.,
"Grokked Transformers are Implicit Reasoners", NeurIPS 2024: two-hop circuit with a clean layer
split; Biran et al., EMNLP 2024: bridge resolves early, hop-2 late; Brinkmann et al., Findings ACL
2024: one deduction step per layer, max depth L−1). Verified-subcircuit reuse inside a composite
is published proof-of-concept (Quirke et al., arXiv:2402.02619: 60–88% of inserted addition-circuit
nodes active in the add+sub model). Null hypothesis to rule out: shortcut/subgraph-matching
composition with no real B→A route (Dziri et al., NeurIPS 2023); check OOD generalization and the
causal share of the B→A route (Yang et al., ACL 2024).

**New sub-axis surfaced by the survey — the DATA MIX:** whether the composed circuit forms at all
depends on co-training A-alone + B-alone + A(B(x)) (He et al., arXiv:2601.09049, preprint: right
mix forms the circuit WITHOUT grokking; Ramesh et al., ICML 2024: composition generalizes only
with the right mixture/order exposure). The training mixture is itself a meaningful axis.

### Axis 4 — Size and geometry
- Usable sizing law: k dependent lookups need ⌊log₂k⌋+2 layers — proved AND empirically matched
  at the learnability threshold (Sanford, Hsu & Telgarsky, ICML 2024, arXiv:2402.09268). 4L ≈
  4-hop; 6L ≈ 16-hop.
- Running ops (max, prefix sum mod 10, resets, clamps, counters) are SOLVABLE-semigroup automata →
  constant-depth shortcuts exist and SGD finds them (Liu et al., ICLR 2023 oral, arXiv:2210.10749).
  Avoid permutation-composition (NC¹-hard) ops at 4L.
- Content-keyed lookback needs ≥2 layers — unconditional 1L impossibility for induction (Sanford,
  Hsu & Telgarsky, arXiv:2408.14332; Elhage et al. 2021; Bietti et al., NeurIPS 2023).
- Across sizes on a fixed task: functional roles stable, head identities churn, circuit size grows
  with model size r≈0.6–0.9 (Tigges et al., NeurIPS 2024) — compare at ROLE level, never head
  coordinates.
- Overcapacity warning: wide models learn PARALLEL REDUNDANT copies of the algorithm (Zhong et
  al., "The Clock and the Pizza", NeurIPS 2023 oral: one model held 6 circular embeddings) — the
  smallest-faithful search lands on one copy and completeness looks bad; count backup-style heads
  per tier. No controlled redundancy-vs-size study exists — an open, publishable gap.
- Practical frontier: every COMPLETE reverse-engineering used ≤6 layers / ≤~4M params; the most
  on-point precedent for our setup is the 4L attention-only docstring circuit (Heimersheim &
  Janiak, AlignmentForum 2023). Our 4L×8H sits exactly at the fully-understandable frontier.
- Attention-only caveat for arithmetic: mod-arithmetic empirically lands in MLPs (Nanda et al.
  ICLR 2023; Zhong et al. 2023); attention-only models must fake the nonlinearity → messier
  circuits expected for arithmetic ops. No impossibility theorem exists (softmax is a
  nonlinearity) — don't overclaim.

## The genuinely new axes (meaningful only)

### Axis 5 — Discovery TARGET (what the circuit explains)
Output logits (DomainCircuits, ContrastCircuits, SuppressionCircuits) vs an internal variable's
probe (ProbeCircuits and BridgeCircuits; e.g. a 13-state register, a program counter, a seen-set)
vs the TASK-ID routing signal itself (function-vector precedent). Multiplies against every other axis. Anchors: Todd et al. ICLR 2024; Hendel et al.
2023; sparse-feature-circuit classifier metric (Marks et al., ICLR 2025); no published
probe-logit-as-discovery-metric — our methods delta.

### Axis 6 — COMPARISON (what the circuit is measured against)
A circuit alone is a description; a comparison is a result. Validated comparisons: vs ground truth
(tracr / SIIT / decompiled Transformer Program), vs a sub-task model (reuse — Merullo, Quirke),
vs a checkpoint (formation — Nanda ICLR 2023; Tigges NeurIPS 2024), vs a seed (universality —
Chughtai ICML 2023), vs a size tier (Tigges), vs sibling circuits in the same model (algebra —
Mondorf's set-operations result is the direct precedent for union/intersection tests).

### Axis 7 — BEHAVIORAL REGIME (which inputs you explain)
Correct-behavior inputs (what every class explains now) vs error inputs (the failure circuit) vs
out-of-length inputs.
Validated decomposition: length-invariant symbolic core vs position-special-cased heads that break
OOD (Zhou et al., ICLR 2024 RASP-L; positional-vs-symbolic heads, arXiv:2605.31558; counting
strategies with different generalization profiles, Behrens et al., arXiv:2407.11542).

### Axis 8 — EDGE SIGN (suppression circuits)
Hunting negative-score edges surfaces copy-suppression/calibration circuits the positive-only
selection systematically discards — validated in a 1-layer ATTENTION-ONLY digit sorter (Urdshals &
Urdshals, arXiv:2501.18666), GPT-2 (McDougall et al. 2023), IOI negative movers (Wang et al.
2022). Cost: keep the sign + one 32-head signed ablation sweep.

What the pipeline does instead: a SuppressionCircuit is selected on a *release rate* — sever the
fringe, see whether the named wrong answer rises — and never reads edge sign, because on an A-vs-B
margin the sign cannot separate a suppressor from an A-promoter (see
[circuit_types.md](circuit_types.md) and [design.md](design.md)).
[pipeline/edge_signs/edge_sign_census.py](../../pipeline/edge_signs/edge_sign_census.py) is a
separate, descriptive census of edge signs, so the edge-sign axis itself remains open.

### Axis 9 — TRAINING DATA MIX (see 3b)
Atomic-vs-composed mixture ratios, curriculum order, and coverage determine which circuits form
and whether composition is real or shortcut. Anchors: He et al. 2026 preprint; Ramesh et al. ICML
2024; Varma et al. 2023 (dataset size gates grokking-competition).

## Instrument settings (real knobs, not research axes)
Node granularity (head / QK-OV / neuron / expert / code), ablation scheme (mean / resample /
zero), scored positions, vocab size, PE choice per se — reach for these when a specific question
needs them; they change the measurement, not the question.

## Recommended first experiments on the new axes
1. **The union experiment (3a; partly realized by the `TaskMux` models):** one 4L×8H model (6L if
   any chosen op needs it), 5 DSL ops spanning circuit families (accumulator, comparator, pointer, router, detector), token-0 =
   task-id, 3 seeds. Per-op circuits → union faithfulness on differing positions only →
   cross-task ablation specificity → task-token corruption (superposition check) → router probe +
   function-vector-style patch of token-0. Every protocol step has a published anchor; the
   combination (from-scratch, attention-only, DSL ground truth, explicit task-id) is unpublished.
2. **Composed A(B(x)) with the data-mix control (3b + 9):** co-train B-alone/A-alone/A(B); test
   B-circuit reuse by faithfulness transfer, layer-shift, and bridge back-patching.
3. **Cheap architecture twins (2):** bilinear MLP variant of the MLP architecture + one
   weight-sparse training run + LN-free flag — three config-level changes, each with a
   validated legibility payoff.
