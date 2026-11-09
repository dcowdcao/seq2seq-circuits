# segrev3 case study

A single-model deep dive that predates the model zoo in [`pipeline/`](../pipeline/README.md): one
attention-only transformer, one algorithm (`segrev3`), three circuit-discovery methods (manual
analysis, ACDC, EAP-IG) and one validation suite, all compared on the same model. The zoo pipeline
reuses its model loader (`interp_tools/model_loading.py`), its canonical constants
(`data_gen/canon.py`) and its EAP-IG settings.

Every number below is read from a file committed in this repository or printed by a command in
the [runbook](segrev3_runbook.md); the source file is named next to each table. The zoo's own
`segrev3_id` model and its `reverse_only` DomainCircuit are described in
[`circuit_classes.md`](circuit_classes.md); this page is about the standalone checkpoint
`segrev3/segrev3_id_checkpoint.pt`.

---

## 0. Summary

A 4-layer × 8-head attention-only transformer solves **`segrev3`**: reverse each maximal run of
non-`3` digits, leaving every `3` in place as a segment boundary. All three discovery methods and
the weight-based analysis point at the same three-stage mechanism:

> **Layer-0 detectors** (L0H1/H2/H4/H6/H7) find the `3` boundaries and the reflected source
> position → **L1/L2 relay heads** carry that information forward → **L2/L3 mover heads**
> (L3H1, L3H3, L3H6, L2H7, with further backups) write the digit from the reflected position to
> the logits.

The mechanism is **role-sparse but head-dense**:

- Keeping the top-18 heads by whole-head knockout importance (all others mean-ablated) recovers
  0.91 of the model's token accuracy on the 128-input manual set; 24 heads recover 0.98
  (`validation/figs_data/data/manual/faithfulness_curve.json`).
- The smallest circuit found by ACDC or EAP-IG that reaches the case study's target of 0.80
  recovered token accuracy is **EAP-IG top-87: 21 heads, 73 edges, recovered accuracy 0.839**
  (`visualizations/min_faithful_circuit_search.json`, produced by `interp_tools/min_faithful_circuit.py`).
- Solo-ablating any L0 detector costs 2.0–3.8 logits of margin; no head outside layer 0 costs more
  than 0.32 (`validation/figs_data/data/manual/rough_validation.json`). The movers are
  interchangeable backups, which is why edge-based circuits keep many heads.

**Loading matters.** The checkpoint stores no LayerNorm parameters, but the model was trained with
parameter-free pre-LayerNorm. It must be loaded with `normalization_type="LNPre"` (as
`interp_tools/model_loading.load_segrev3` does). Loading with `None` raises no error, yet the model
then mostly predicts its first input token and scores near chance.

---

## 1. Model and task

### 1.1 Model
Source of truth: `interp_tools/model_loading.py` and the checkpoint's stored `config`.

| Property | Value |
|---|---|
| Type | attention-only (no MLP), causal |
| Layers × heads | 4 × 8 (32 heads) |
| Hidden size / head dim | 128 / 32 |
| Vocabulary | 10 (digits 0–9, no BOS/SEP) |
| Context | 20 positions |
| Normalization | `LayerNormPre` (parameter-free: center and divide by RMS, eps 1e-5) before every block and before the unembedding |
| Positional embeddings | learned absolute |
| Checkpoint | `segrev3/segrev3_id_checkpoint.pt` (~2 MB) |
| Accuracy | stored eval accuracy 0.99964 (`metrics.eval.accuracy` in the checkpoint); the loader self-test (`python interp_tools/model_loading.py`, 2,000 random inputs) scores token and sequence accuracy 1.000 |

### 1.2 The task: `segrev3 = SegmentReverse(boundary=3)`
Scan left to right; at each `3`, write the accumulated run of non-`3` digits reversed, copy the `3`
unchanged, and start a new run; reverse the trailing run at the end.

```
[1,7,3,4,2,5]  ->  [7,1,3,5,2,4]      # runs {1,7} and {4,2,5} reversed; the 3 stays
[1,2,3,4,5,6]  ->  [2,1,3,6,5,4]
[3,3,3]        ->  [3,3,3]            # all boundaries: identity
```

`data_gen/segrev3_algorithm.py` is the single implementation (`segrev3_list`, `segrev3_tensor`,
`source_index_map`). Within a segment `[a, b)` output position `j` copies input `a + b − 1 − j`, a
reflection; boundary positions copy themselves.

### 1.3 Data layout and answer positions
The model sees 20 tokens, `[input(10) | segrev3(input)(10)]`, trained autoregressively. Output digit
`y_k` is predicted at position `9 + k`, so the **answer positions are 9..18**
(`ANSWER_POSITIONS` in `model_loading.py`). Each output depends on the input and on the earlier
outputs.

### 1.4 The LNPre gate (why it matters everywhere)
- Activation patching and path patching are exact (the real forward pass includes the
  normalization).
- Direct logit attribution is an exact decomposition only with the frozen final normalization
  scale (`apply_ln=True`); the smoke test asserts that the parts sum to the margin.
- Weight-based QK / OV / composition scores drop the per-position normalization scale, so they are
  approximations: read their magnitudes as relative and confirm causally.

---

## 2. Where things live

```
circuit_discovery/   main.py (orchestrator), acdc.py, eap.py, manual_circuit.py
interp_tools/        model_loading.py (canonical loader + ANSWER_POSITIONS), patching_dla.py,
                     edge_tools.py, circuit_loader.py, circuit_summary.py,
                     min_faithful_circuit.py, n_threes.py
data_gen/            segrev3_algorithm.py, canon.py (canonical sweep values), data_gen.py,
                     dataset_store.py, generate_datasets.py
metrics/metrics.py   logit difference, KL divergence, task accuracy
validation/          validation.py (faithfulness / completeness / minimality),
                     n_threes_faithfulness.py, figs_data/{data,figs}/<method>/<circuit>/
datasets/            the exact inputs of every committed result: manual/ acdc/ eap/ edge/ single/
visualizations/      acdc/acdc_thresh_<tau>/, eap/eap_top<n>/, manual_circuit_analysis/figs_data/,
                     rough_circuit.{json,md,png}, min_faithful_circuit_search.json,
                     render_circuit_png.py, collect_figures.py
segrev3/             the checkpoint
github_repos/        vendored ACDC and EAP-IG (see github_repos/VENDORED_SOURCES.md)
```

`interp_tools/model_loading.py`, `data_gen/` and `metrics/metrics.py` are the shared foundation
that every script imports. The vendored ACDC and EAP-IG code is wrapped from `circuit_discovery/`
rather than modified, apart from a one-line TransformerLens import fix in ACDC and a new
`acdc/segrev3/` task package.

### 2.1 The dataset store
`datasets/<app>/<param>/<scheme>.json` stores raw input digits only; tensors are rebuilt
deterministically on load. Every result JSON computed from stored data records the file it read (a
`data` block or a `data_path` field); the shared evaluation set of `min_faithful_circuit.py` is
generated in memory and recorded as its generator call.

| app | file | examples in the committed files | used by |
|---|---|---|---|
| `manual` | `<scheme>.json` | 128 | manual analyses, solo-knockout validation, faithfulness curve, greedy search |
| `acdc` | `acdc_thresh_<tau>/<scheme>.json` | 20 | ACDC and the validation of ACDC circuits |
| `eap` | `eap_top<n>/<scheme>.json` | 512 | EAP-IG (attribution 200 + evaluation 256 sliced from `eap_top200`) and the validation of each EAP circuit |
| `edge` | `n_threes_<k>.json`, k = 0..10 | 64 | realized-attention and by-number-of-3s analyses (unpaired inputs with exactly k threes) |
| `single` | `single_<scheme>_k<k>_n<n>.json`, `edge_k<k>.json` | 1 | single-sequence mode |

The replace-scheme files of the batched apps carry a `batched_` prefix. Every file uses seed 0,
except the manual sets, which follow `manual_circuit.py` and use seed 0 / 1 / 2 for `random` /
`replace_threes_to_nonthrees` / `replace_non_threes_to_threes`. `python circuit_discovery/main.py
data` rewrites the committed files byte for byte (ACDC at 20 examples, `SEGREV3_ACDC_EXAMPLES` in
`data_gen/canon.py`; the zoo pipeline's ACDC uses its own `CANON_ACDC_EXAMPLES` = 128). The ACDC
sets of the extra thresholds of the min-faithful search (§5.1) are written by `acdc.py` with the
same seed and size.

---

## 3. Contrast data and metrics

All causal methods run on clean/corrupted pairs built by `data_gen/data_gen.py` (`Segrev3Pairs`:
clean and corrupted 20-token sequences, answer positions, correct and incorrect token ids, scheme).

| Scheme | Corrupted input | Targets |
|---|---|---|
| `random` | an independent random input (global contrast) | everything; the default for ACDC, EAP-IG and validation |
| `replace_threes_to_nonthrees` (n) | n of the clean `3`s replaced by random non-`3`s (deletes boundaries) | the boundary detectors |
| `replace_non_threes_to_threes` (n) | n clean non-`3`s replaced by `3` (adds boundaries) | the reversal / mover machinery |

Replace-scheme clean inputs are rejection-sampled to contain at least n of the digit to replace, so
exactly n replacements happen. Every dataset stores per-example counts of `3`s and every figure
carries a footer describing its inputs' 3-count distribution (`interp_tools/n_threes.py`);
weight-based figures are marked input-independent.

Metrics (`metrics/metrics.py`), all over answer positions 9..18:
- `logit_diff`: mean (correct − incorrect) logit; the differentiable metric for EAP-IG and validation.
- `kl_divergence`: mean KL(clean ‖ patched); the ACDC metric.
- `task_accuracy`: token accuracy and sequence accuracy (all ten digits correct).

---

## 4. Methods

### 4a. Manual analysis (`manual_circuit.py` → `vis_patching_dla.py`, `vis_edge.py`)
One command runs data → activation analysis → weight analysis → full-model validation.

Activation tools (`interp_tools/patching_dla.py`, exact under LNPre):
- **Direct logit attribution** on clean sequences only, in `correct_vs_rest` mode: each head's write
  onto `W_U[:, correct] − mean_{d≠correct} W_U[:, d]`. Finds the writers.
- **Activation patching** of each head's output, clean ↔ corrupted (noising measures necessity,
  denoising sufficiency), normalized to [0, 1].
- **Attribution patching** with integrated gradients (a first-order estimate of the same quantity).
- **Path patching**: one sender → receiver (Q, K or V) edge at a time with every other head's output
  frozen, swept as a 32 × 32 matrix per channel plus a head → logits vector.

Weight tools (`interp_tools/edge_tools.py`, input-independent):
- **OV circuit** `W_E W_V W_O W_U` with the eigenvalue copying score `Σ Re(λ) / Σ |λ|` in [−1, 1]
  (near +1: the head copies the attended token).
- **QK circuit**, content (10 × 10) and positional (20 × 20, future keys masked).
- **Q/K/V composition**: normalized Frobenius overlap of a sender's OV write with a receiver's Q/K/V
  read, against the mean overlap of random matrices of the same shapes (0.0883 for 100 seeded draws,
  ≈ 1/√128); only sender layer < receiver layer.

The manual pipeline does not synthesize a circuit: it validates the full model, so minimality
reports the cost of ablating each of the 32 heads alone. `interp_tools/circuit_summary.py` builds a
heuristic "rough" head set on demand (§5.3).

Circuit-scoped mode (`manual_circuit.py --circuit <name>` or `--circuit-json <file>`, the
orchestrator's `figs` stage) re-runs the same analyses restricted to a discovered circuit.
Single-sequence mode (`--single`, the orchestrator's `single` stage) runs patching and attribution
on one hand-chosen clean/corrupted pair and realized attention on one input.

### 4b. ACDC: recursive edge pruning (`circuit_discovery/acdc.py`)
- Graph: every head is split into an output node and separate Q, K, V input nodes
  (`use_attn_result`, `use_split_qkv_input`). ACDC reports 1,377 prunable edges: the 1,281 live
  sender → receiver edges plus 96 internal input → Q/K/V edges (3 per head).
- Walk nodes in reverse topological order; remove each incoming edge whose removal increases
  KL(clean ‖ patched) by less than τ. Gradient-free and iterative.
- Corrupted reference: the `random` scheme. Committed runs use the 20-example ACDC sets.
- Outputs: `visualizations/acdc/acdc_thresh_<tau>/{summary.json, circuit.gv, circuit.pth, circuit.png}`
  and the combined `acdc_all_thresholds_summary.json`.

### 4c. EAP-IG: integrated-gradient edge attribution (`circuit_discovery/eap.py`)
- One-shot and gradient-based. The edge score is Σ_batch,pos ∂metric/∂(receiver input) ·
  (corrupted − clean)(sender output), with the `EAP-IG-inputs` variant interpolating the input
  embeddings over 30 steps.
- Graph: EAP-IG's `Graph.from_model` reports 37 forward / 101 backward nodes and 1,519 "real" edges.
  It wires an MLP node per layer even for an attention-only model, so 238 of those edges touch MLP
  nodes this model does not have; the **live edge count is 1,281**
  (`data_gen/canon.py: live_edge_budget(4, 8)`). Top-n rungs below are counts in EAP-IG's graph.
- Metric: mean logit difference over positions 9..18. Keep the top-n edges by |score| and prune
  dangling nodes.
- Fixed internals: attribution on 200 examples in batches of 50, evaluation on 256 in batches of 64.
  The batch size is deliberately not a CLI flag: EAP-IG normalizes by n_batches × ig_steps, so it
  changes the score scale.
- Outputs: `visualizations/eap/eap_top<n>/eap_top<n>.{json,pt,png}` and the combined
  `eap_all_topn_summary.json`.

### 4d. Validation (`validation/validation.py`)
Head-level: a circuit is reduced to the heads its edges touch; every other head's output is replaced
by a reference (mean by default; zero or resample optional), at answer positions 9..18.
- **Faithfulness**: recovered = (circuit − empty) / (full − empty), for token accuracy and for
  logit difference.
- **Completeness**: the complement alone should fail (`complement_acc` low), and no single outside
  head should change the circuit much when added (`max_incompleteness`).
- **Minimality**: each circuit head, removed alone, should cost something.
- `greedy_head_circuit` drops the least damaging head while recovered token accuracy stays above a
  threshold (default 0.95); `candidates=` restricts the pool.

Each circuit is validated on its own family's dataset (ACDC circuits on their 20-example set, EAP
circuits on their 512-example set, manual on the 128-example set), so absolute numbers are not
directly comparable across families. `interp_tools/min_faithful_circuit.py` re-scores every
discovered circuit on one shared set of 512 random inputs for that reason.

### 4e. The orchestrator (`circuit_discovery/main.py`)
Positional stage words (`all | data | manual | acdc | eap | figs | validate | single | zip`) run in
canonical order; per-circuit validation, circuit-scoped figures and the figure bundle follow
discovery automatically. Commands and flags: [segrev3_runbook.md](segrev3_runbook.md).

---

## 5. Results

Unless stated otherwise: mean ablation, `random` contrast scheme, recovered values normalized
between the empty circuit (0) and the full model (1). "Detectors" and "movers" refer to the
reference role sets fixed in `circuit_discovery/eap.py` (`MANUAL_DETECTORS` = L0H1, L0H2, L0H4,
L0H6, L0H7; `MANUAL_MOVERS` = L2H7, L3H1, L3H3, L3H6).

### 5.1 ACDC τ sweep
Sources: `visualizations/acdc/acdc_thresh_<tau>/summary.json` (edges, heads, KL) and
`validation/figs_data/data/acdc/acdc_thresh_<tau>/acdc_thresh_<tau>_validation.json` (recovered
values, 20 examples). Initial KL is 0 for every run.

| τ | edges | heads | final KL | recovered acc | recovered logit diff | detectors / movers kept |
|---|---|---|---|---|---|---|
| 0.2 | 44 | 13 | 9.08 | 0.425 | 0.395 | L0H4, L0H7 / L2H7, L3H6 |
| 0.1 | 76 | 18 | 8.15 | 0.369 | 0.388 | L0H2, L0H4, L0H7 / L2H7, L3H6 |
| 0.055 | 142 | 25 | 6.63 | 0.966 | 0.871 | all 5 / L2H7, L3H6 |
| 0.02 | 371 | 27 | 2.90 | 0.994 | 0.928 | all 5 / L2H7, L3H3, L3H6 |

Re-scored on the shared 512-input set (`min_faithful_circuit_search.json`), the same four circuits
give 0.379, 0.395, 0.982 and 0.999. Going from 13 to 18 heads buys almost nothing (on ACDC's own
20-example set the 13-head cut even scores higher); faithfulness only jumps once the circuit
contains all five L0 detectors. The denser τ grid of the search places the jump between 19 heads
(τ = 0.08, 0.601) and 24 heads (τ = 0.06–0.064, 0.955), with 23 heads (τ = 0.065, 0.066) at 0.709.
ACDC recovers the detectors first and the movers last: L3H1 is in no ACDC circuit of the search,
down to τ = 0.01.

### 5.2 EAP-IG top-n sweep
Sources: `visualizations/eap/eap_all_topn_summary.json` (`per_topn.<n>`: edges after pruning, heads,
edge-level recovery) and `validation/figs_data/data/eap/eap_top<n>/eap_top<n>_validation.json`
(head-level recovered accuracy, 512 examples). Clean logit difference +10.28, corrupted −10.34.

| top-n | edges kept | heads | edge-level logit-diff recovery | head-level recovered acc | detectors / movers |
|---|---|---|---|---|---|
| 50 | 20 | 9 | 0.081 | 0.329 | 2 / 0 |
| 75 | 52 | 15 | 0.142 | 0.533 | 4 / 1 |
| 87 | 73 | 21 | 0.182 | 0.839 | 5 / 3 |
| 100 | 87 | 23 | 0.208 | 0.847 | 5 / 3 |
| 200 | 199 | 31 | 0.402 | 1.000 | 5 / 4 |
| 400 | 400 | 32 | 0.686 | 1.000 | 5 / 4 |
| 800 | 800 | 32 | 0.820 | 1.000 | 5 / 4 |

The two faithfulness columns measure different things. *Edge-level recovery* is EAP-IG's own
`evaluate_graph`: every edge outside the circuit carries its corrupted-run activation, and the
circuit's logit difference is normalized as (circuit − corrupted) / (clean − corrupted).
*Head-level recovered accuracy* is `validation.py`: heads outside the circuit are mean-ablated, edges
between kept heads stay intact. The head set is complete long before the edge set: by top-200 every
head but L3H0 is in, and L3H0 is also the only outside head the completeness check flags (adding it
changes the logit difference by 0.07). At top-400 and top-800 the circuit touches all 32 heads, so
their head-level validations are identical.

EAP-IG's highest-scoring edges are `input → L0 detector (V)` (input → L0H7 V 0.082, input → L0H4 V
0.057, input → L0H2 V 0.039), and the heads with the largest direct-to-logit scores are L3H1, L3H6
and L3H3 (`top30_edges`, `heads_to_logits` in the same summary).

### 5.3 The heuristic manual circuit
`interp_tools/circuit_summary.py` selects movers as the top-6 heads by |DLA|, detectors as the top-6
by activation-patching necessity, and adds endpoints of strong Q/K/V composition edges, giving 21
heads (`visualizations/rough_circuit.json`: L0H1/2/3/4/6/7, L1H6, L2H0/1/2/3/5/6/7, L3H0–H6).
Validated with the runbook's rough-circuit snippet (128 examples): recovered accuracy 0.50,
recovered logit difference 0.48. The completeness check is the informative part: the outside heads
that would help most are the entire L1 family (adding L1H7 +1.18 logits, L1H4 +1.12, L1H1 +1.08,
L1H0 +1.01, L1H2 +0.97). The heuristic finds the writers (DLA) and the detectors (necessity) but
misses the L1 relays, which have small direct effects and small solo-ablation effects.

### 5.4 Minimality: the same story in every circuit
Logit-difference drop when one head is removed from the circuit (`minimality` in each validation
JSON; the last row is the full model, from `rough_validation.json`).

| circuit | five largest drops |
|---|---|
| ACDC τ = 0.02 | L0H7 3.47, L0H4 3.37, L0H2 2.29, L0H6 2.16, L0H1 1.97 |
| ACDC τ = 0.055 | L0H7 3.50, L0H4 2.96, L0H2 2.22, L0H6 2.08, L0H1 2.06 |
| EAP-IG top-87 | L0H4 3.22, L0H7 3.16, L0H1 2.72, L0H2 2.20, L0H6 1.75 |
| EAP-IG top-200 | L0H4 3.87, L0H7 3.50, L0H2 2.54, L0H6 2.15, L0H1 2.00 |
| EAP-IG top-400 / 800 | L0H4 3.90, L0H7 3.53, L0H2 2.57, L0H6 2.16, L0H1 2.01 |
| full model (32 heads) | L0H4 3.78, L0H7 3.57, L0H2 2.56, L0H6 2.09, L0H1 2.02 |

In the full model no head outside layer 0 costs more than 0.32 when removed alone, and most L2/L3
heads cost close to 0: the detectors are the bottleneck and the movers back each other up. A negative
drop means removing the head helps (L0H2 in the τ = 0.1 ACDC circuit: −0.39), a sign that a small
cut kept a head that is harmful without its partners.

### 5.5 Activation screens on the `random` scheme
Source: `visualizations/manual_circuit_analysis/figs_data/data/random/patching_dla_results.json`
(128 pairs; clean logit difference +10.21, corrupted −10.29).

- **Writers (DLA):** L3H1 +0.884, L3H6 +0.859, L3H3 +0.783, then L3H4, L3H5, L3H2, L2H7, L2H1.
- **Necessity (activation patching, noising):** the eight largest are all layer-0 heads (L0H2
  0.027, L0H4 0.027, L0H7 0.025, L0H1 0.022, L0H6 0.018, …).
- **Causal paths (path patching, combined Q/K/V):** the strongest edges all run from layer 0 into
  layer 1: L0H7 → L1H7 +0.019, L0H7 → L1H1 +0.016, L0H2 → L1H1 +0.016, L0H4 → L1H6 +0.015.

The same pipeline writes the two replace schemes next to it
(`.../figs_data/data/batched_replace_*/`, also committed).

### 5.6 Weight-based and attention evidence
Source: `visualizations/manual_circuit_analysis/figs_data/data/edge_results.json` unless noted.

- **OV copying scores separate the roles.** Layer-0 detectors copy (L0H1 +0.935, L0H2 +0.931, L0H0
  +0.889, L0H6 +0.886, L0H3 +0.816, L0H4 +0.727, L0H7 +0.706; L0H5 is the exception at −0.647).
  Movers have negative scores (L3H2 −0.680, L3H1 −0.378, L2H7 −0.347, L3H4 −0.333, L3H3 −0.310,
  L3H6 −0.253): they do not map the attended token to its own logit, consistent with writing a
  digit read through the layer-0 re-encoding rather than the raw token.
- **The `3 → 3` self-boost.** L0H4's OV circuit has one dominant entry, attended `3` → output `3`
  = 1.86 (the next largest entry is 0.12): the detector preserves the boundary digit in place.
- **Composition.** L0H4 is the dominant sender on all three channels: Q-composition into the movers
  (L0H4 → L3H0, L3H5, L2H7, each 0.208, 2.35–2.36× the 0.0883 baseline), K (L0H4 → L3H4 0.197) and V
  (L0H4 → L3H4 0.170, L0H4 → L1H6 0.158). L0H7's strongest V-composition targets are L1H7 and L1H1
  (0.129, 0.128), the same two edges that top the causal path-patching list (§5.5).
- **Realized attention depends on the boundaries.** On inputs with no `3`, L0H7 puts 0.898 of its
  attention from each output position on the reflected input position. With exactly one `3` this
  drops to 0.189, and L0H7 instead puts 0.73 of its answer-position attention on the `3` (0.10 under
  uniform attention). The reflection is computed dynamically from the boundary positions, not
  stored as a fixed positional pattern. (`attn_by_dataset.n_threes_<k>.specific_heads_attn.L0H7`
  in the same file, computed on the 64-input `datasets/edge/n_threes_<k>.json` sets; the figures of
  the single input `datasets/single/edge_k2.json` are in `figs_data/figs/edge/edge_k2/`.)

### 5.7 Faithfulness by number of 3s
Source: `validation/figs_data/data/eap/faithfulness_by_n_threes_eap_top87.json` (raw token accuracy
on the 64-input `datasets/edge/n_threes_<k>.json` sets; heads outside the circuit mean-ablated).

| 3s in input | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| EAP-IG top-87 (21 heads) | 0.98 | 0.81 | 0.75 | 0.75 | 0.76 | 0.76 | 0.81 | 0.85 | 0.88 | 0.95 | 1.00 |
| full model | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.99 | 0.98 | 0.97 | 0.96 | 0.95 | 1.00 |
| all heads ablated | 0.04 | 0.10 | 0.17 | 0.24 | 0.29 | 0.36 | 0.42 | 0.50 | 0.56 | 0.64 | 0.70 |

The 21-head circuit handles the two extremes (no boundary: plain reversal; all boundaries: identity)
almost perfectly and loses accuracy where segment structure is richest. The same JSON format for all
kept circuits is in `faithfulness_by_n_threes_{acdc,eap}.json`.

---

## 6. The mechanism and the caveat

```
input ─► L0 DETECTORS ───────► L1/L2 RELAYS ──────► L2/L3 MOVERS ─► logits
         find the 3s, attend    carry the boundary /   write the digit from
         to the reflected       reflected-position     the reflected position
         position (L0H7),       information (the       (L3H1/H3/H6, L2H7
         preserve 3s (L0H4)     L1 family)             and backups)
```

Evidence chain: layer-0 heads copy (OV ≈ +0.7 to +0.94), L0H4 preserves `3`s (`3 → 3` = 1.86) and
L0H7 attends to the reflected position on 3-free inputs (0.898) and to the boundaries otherwise; the
strongest causal paths run from L0H7 and L0H2 into layer 1; the top writers are L3 heads with
negative copying scores. ACDC recovers the detectors first; EAP-IG recovers detectors and movers by
top-200; the heuristic manual circuit recovers writers and detectors but under-counts the L1 relays.

**The caveat.** segrev3 is role-sparse but head-dense and redundant:
- Ordering heads by whole-head knockout importance (the order is stored in
  `faithfulness_curve.json`), the recovered token accuracy is 0.84 / 0.91 / 0.98 / 1.00 with
  16 / 18 / 24 / 30 heads (128-input manual set; 30 is the first k at exactly 1.00).
- No ACDC or EAP-IG circuit with fewer than 21 heads reaches 0.80 recovered accuracy on the shared
  evaluation (EAP-IG: 20 heads 0.719, 21 heads 0.839; ACDC: 19 heads 0.601, 23 heads 0.709, 24 heads
  0.955).
- Edge-based circuits projected to heads appear not to be the most compact head sets: the top-18
  heads by knockout importance recover 0.91 on the 128-input manual set, while the smallest
  discovered circuit above 0.80 needs 21 heads (0.84 on the shared 512-input set; the two
  evaluation sets differ, so compare loosely). Because the movers
  are interchangeable, a forced cut can land on a bad subset (ACDC's 13- and 18-head circuits), and
  the last mover (L3H1) is in no ACDC circuit of the search, down to τ = 0.01.

The useful report for a model like this separates the **mechanistic story** (a few named heads per
stage, each supported by two or more independent methods) from the **circuit as a subgraph** (a
role-complete head and edge set with its measured faithfulness).

---

## 7. Reproducing and caveats

All commands are in [segrev3_runbook.md](segrev3_runbook.md); its §1 lists the committed run step
by step. Notes:
- The committed `datasets/` files are the exact inputs of the committed results, and the `data`
  stage (`python circuit_discovery/main.py data`) rewrites them byte for byte.
- The committed artifacts were computed on CPU. In a fresh checkout on the same machine, re-running
  the fast steps (data, the validation of `acdc_thresh_0.02` and `eap_top87`, `circuit_summary.py`,
  the by-number-of-3s run for `eap_top87`, the faithfulness curves) reproduced every output file
  byte for byte, figures included, and ACDC at τ = 0.065 gave byte-identical results with 1, 4 and
  16 BLAS threads.
- On other hardware floating-point results can differ in the last digits, and ACDC, which keeps or
  prunes an edge by comparing a KL change with τ, can then flip borderline edges: an earlier GPU
  run of these commands gave τ = 0.064 a 23-head circuit (0.709 on the shared set) where the
  committed CPU run has 24 heads (0.955).
- `validation/figs_data/data/manual/rough_validation.json` holds the **full-model** per-head
  validation written by the manual pipeline. `python validation/validation.py --circuit rough`
  validates the 21-head heuristic circuit and writes to the same file name, so use the runbook's
  print-only snippet for the heuristic circuit.

---

## 8. Methodology lessons

A recipe distilled from this case study, with the failure modes that motivated each step.

**Heads first, then edges, as a loop.** Head-level screens are the right way to find candidate
heads, with two corrections:
1. **Rank edges globally, not per head.** A weak head's strongest edge can score below a strong
   head's fifth; per-head thresholds distort the circuit. EAP-IG scores every edge in one pass on one
   scale and ACDC prunes with one global τ. Per-head edge inspection (path patching) belongs after,
   as verification of the globally ranked survivors.
2. **A single pass misses two kinds of heads.** Redundant banks (the movers) read ≈ 0 in solo
   knockouts, so necessity screens miss them; composition-only relays (the L1 family) have small
   direct effects and small solo-ablation effects. Group ablations, the completeness check and the
   edge methods themselves catch both (the completeness check named the whole L1 family for the
   heuristic circuit, §5.3).

**Steps.**
1. *Foundations.* One canonical loader with an accuracy gate before any analysis (a mis-loaded model
   produces plausible-looking results everywhere downstream); metrics defined once with exact answer
   positions; contrast data under a global scheme plus targeted schemes that each corrupt one causal
   variable; check the exactness invariants of every tool (DLA sums to the margin; weight-based
   scores are approximate under LNPre).
2. *Candidate heads.* Activation patching in the noising direction (necessity), attribution patching
   in the denoising direction (sufficiency), DLA (writers) and weight-based role priors (copying
   scores, QK structure). Noising and denoising answer different questions (necessity misses
   redundant backups; sufficiency misses distributed AND-like computations), so run both directions;
   comparing exact patching with its gradient estimate in the same direction is a check of the
   approximation, not a new screen. Then solo knockouts of every head, group ablations by layer and
   role, and a completeness loop: add the most helpful outside head and repeat until the gain
   plateaus. Choose screen cut-offs for recall (a false positive costs one validation run, a false
   negative silently distorts the circuit).
3. *Edges, on the full graph.* Run EAP-IG and ACDC on the full computational graph, not on the
   candidate heads (conditioning on step 2 would inherit its blind spots). Sweep circuit size (top-n,
   τ) and read the whole faithfulness curve; expect non-monotonic or flat stretches when redundancy is
   high (§5.1). Verify kept edges causally with path patching and support them with composition
   scores.
4. *Validate and report.* Faithfulness, completeness and minimality under mean ablation, with zero or
   resample ablation as a robustness check; feed completeness failures back into step 2. When
   redundancy is high, build the head set directly (knockout ordering, `greedy_head_circuit`, or a
   search such as `interp_tools/min_faithful_circuit.py`) rather than projecting an edge circuit to
   heads, and report the mechanistic story separately from the circuit as a subgraph.

**Pitfalls met in this project.**
- Saturated softmax makes plain gradients vanish; use integrated gradients.
- EAP-IG scores scale with the attribution batch size; fix it and never compare across values.
- Necessity is not sufficiency; run both patching directions.
- Mean, zero and resample ablation give different numbers; name the reference and spot-check another.
- In single-pair patching, teacher forcing contaminates late answer positions; the earliest answer
  slots are the cleanest.
- Repeated digits confound attention-argmax readouts; use inputs with distinct digits.
- A head with ≈ 0 solo effect may belong to a redundant bank; group-ablate before discarding it.
- Weight-based magnitudes are approximate under LNPre; use them for ranking and confirm causally.
- A mis-loaded model fails silently; gate every session on task accuracy.
