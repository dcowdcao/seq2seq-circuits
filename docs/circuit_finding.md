# Circuit-finding methods

This page covers the tools used to find circuits: component-level interpretability techniques,
the three algorithmic edge finders (EAP-IG, ACDC, Edge Pruning), linear probes, and the selection
and evaluation stages. The definitions they all serve — the graph, the ablation, faithfulness per
class — are in [circuits.md](circuits.md); the per-class data and metrics are in
[circuit_classes.md](circuit_classes.md).

Two code bases share these tools:

| | where | scope |
|---|---|---|
| **segrev3 case study** | `circuit_discovery/`, `interp_tools/`, `data_gen/`, `metrics/`, `validation/`, `visualizations/` (repo root) | one model; three discovery methods and a validation suite, compared in depth — [segrev3_case_study.md](segrev3_case_study.md) |
| **multi-model pipeline** | `pipeline/` | every model of the zoo, per declared circuit, EAP-IG by default |

The pipeline's per-circuit interpretability CLIs delegate their computation to the root
`interp_tools/` modules, so both code bases run the same implementations.

## 1. Component-level techniques

| technique | what it measures | implementation | per-circuit CLI |
|---|---|---|---|
| QK / OV circuits | from the weights alone: which token attends to which (`qk_circuit`), and what a head writes to the logits when it attends to a token, with an eigenvalue copying score (`ov_circuit`) | [`interp_tools/edge_tools.py`](../interp_tools/edge_tools.py) | `pipeline/interp_tools/edges.py --analysis qkov` |
| Q/K/V composition | weight-based composition scores between every ordered head pair, against a random-weight baseline (`qkv_composition`); plus `direct_path`, `input_to_head`, `head_to_logit` | `interp_tools/edge_tools.py` | `edges.py --analysis composition` |
| realized attention | mean attention by (query, key) position and by source digit on the circuit's data | `pipeline/interp_tools/edges.py` | `edges.py --analysis pos_pos` / `pos_token` |
| direct logit attribution (DLA) | exact additive decomposition of the logit difference into per-component contributions | [`interp_tools/patching_dla.py`](../interp_tools/patching_dla.py) `direct_logit_attribution` | `pipeline/interp_tools/patching.py --analysis dla` |
| activation patching | causal effect of patching one component's activation between clean and corrupt runs (noising and denoising) | `patching_dla.py` `activation_patching` | `patching.py --analysis act` |
| attribution patching | first-order (gradient) approximation of activation patching | `patching_dla.py` `attribution_patching` | `patching.py --analysis att` |
| path patching | sender → receiver Q/K/V direct-path effects, including the full Q/K/V sweep and head → logits | `patching_dla.py` `path_patching`, `path_patching_qkv_sweep` | `patching.py --analysis path` |
| logit lens | per-layer readout of the logit margin between two tasks' answers in multiplexed models | [`pipeline/task_router/router_diagnostic.py`](../pipeline/task_router/router_diagnostic.py) | `router_diagnostic.py --model <mux id>` |

The per-circuit CLIs take `--model <id> --circuit <slug>`, read the circuit's own input pool, and
write JSON plus PNG heatmaps under `pipeline/circuits_data/<model>/<circuit>/` and
`pipeline/circuits_figures/<model>/<circuit>/`. In `run_model_pipeline.py` they are the
non-default `patching` and `edges` stages. All patching uses the parameter-free `LNPre`
normalization; DLA is exact under it (see the module docstring of `patching_dla.py`).

## 2. Algorithmic finders

All three finders write one folder per candidate circuit (`write_candidate_folder` in
`pipeline/find_circuits/eap.py`):
`pipeline/circuits_data/<model>/<circuit>/circuits/<candidate>/{summary.json, circuit.json,
circuit.pt, circuit.gv}` plus a PNG under `circuits_figures/`. The folder layout and `circuit.pt`
(the vendored EAP-IG `Graph` tensor format, written for ACDC and Edge Pruning by `save_circuit_pt`)
are shared; `circuit.json` is not: EAP-IG writes the vendored `Graph` JSON, ACDC and Edge Pruning
an edge list ([circuits.md §2](circuits.md#2-a-circuit)). Their canonical sweeps live in one
place, [`data_gen/canon.py`](../data_gen/canon.py), so a bare script run and an orchestrated run
agree.

### EAP-IG

**Edge Attribution Patching with Integrated Gradients** (Hanna, Pezzelle & Belinkov, 2024) is the
workhorse: every showcase circuit and 3,329 of the 3,341 faithful zoo circuits are EAP-IG candidates
or come from a slight variation of it; the other 12, DomainCircuits from the early zoo run at Large,
came from ACDC (6) and Edge Pruning (6)
([`results/zoo/circuits_index.csv`](../results/zoo/circuits_index.csv), columns `run`, `finder`).
The segrev3 case study also compares it with ACDC and manual analysis (§6).
[`pipeline/find_circuits/eap.py`](../pipeline/find_circuits/eap.py) wraps the vendored EAP-IG
([`github_repos/EAP-IG`](../github_repos/EAP-IG/), method `EAP-IG-inputs`, read-only with two runtime
patches: pre-tokenized inputs and device placement).

Inputs: the model, its graph, *N* clean/corrupt pairs, *K* integration steps, and a metric *L*
(per class, [circuit_classes.md](circuit_classes.md)). Output: a score matrix *S* over
|writers| × |readers|.

```
for each batch of B pairs:
    run corrupt  -> store every writer's activation at every position
    run clean    -> store every writer's activation
    D = corrupt − clean activations, per writer
    for k = 0 .. K−1:
        e_k = e_corrupt + (k/K)(e_clean − e_corrupt)      # interpolate the input embedding
        forward on the clean tokens from e_k              # every head / MLP recomputes
        L = metric;  backward -> g = dL/d(reader input) for every reader
        S[w, r] += Σ_{scored positions p} ⟨D_w[p], g_r[p]⟩
score = S / (N·K);   keep the top-n edges by |score|;   drop dangling components
```

Each edge is scored by how much it carries the clean-vs-corrupt difference in the output: the dot
product of the writer's activation difference with the gradient of the logit gap at the reader's
input. The metric sums over the batch and the score is divided once by the number of examples, so
scores are per-example means, comparable across batch sizes (`sum_reduced`).

Settings (pinned in `eap.py`): *K* = 30 (`CANON_IG_STEPS`), *N* = 200 attribution pairs in batches
of 50, faithfulness of each candidate checked on 256 further pairs. Top-*n* by absolute score is the
vendored `Graph.apply_topn(n, absolute=True, prune=True)`.

**The top-*n* ladder.** One attribution serves every rung. The default `frac` regime takes
`CANON_TOPN_FRACS = [0.02, 0.05, 0.15, 0.40]` of the model's live edge budget, so a rung covers the
same fraction of the graph at every architecture; the `abs` regime reproduces the fixed
`CANON_TOPNS = [50, 200, 800]` used by the zoo-wide runs (at Small, 800 exceeds the whole graph,
which is why it is no longer the default). `--target-seq-acc T` instead scans a geometric set of
rungs (ratio 1.5, then 1.05 inside the bracket) and emits only the rung maximizing
$(1-n_\text{heads}/N)(1-n_\text{edges}/E)$ among those with whole-sequence accuracy > *T*. The
pipeline's `smallest` stage runs this search for every DomainCircuit, ContrastCircuit and
SuppressionCircuit (`smallest_faithful_circuit.py --search`) and adds the emitted rung to the
candidates; selection still applies the class's own rule
([circuits.md §6.2](circuits.md#62-smallest_faithful_circuitpy--the-zoo-wide-runs)).

```bash
python pipeline/find_circuits/eap.py --model schmitt_id__moe8 --circuit schmitt_noband --device cpu
python pipeline/find_circuits/eap.py --model <id> --circuit <slug> --topns 30 76 228   # explicit rungs
python pipeline/find_circuits/eap.py --model <id> --circuit <host slug> --probe <variable> [--probe-layer l]
```

**Probe mode** (`--probe <variable>`, ProbeCircuits and BridgeCircuits): the metric becomes
KL(clean probe log-probs ‖ current) at the probe site `blocks.{l}.hook_resid_post`; the graph's
logits node is retargeted to that site and every writer above layer *l* is cut
(`retarget_graph_to_probe`, `probe_cutoff_indices`); the clean/corrupt pairs come from the circuit
named by `--circuit`. Candidates are written as `eap_probe_<variable>_top<n>`. Without
`--probe-layer`, the probe site with the highest stored accuracy is used.

Two optional flags change the attribution: `--zero-ablation` zero-ablates instead of patching in
the corrupted run, and `--use-pos-embed` scores positional-embedding edges instead of the token
input. They are exploratory and their candidates are excluded from selection.

### ACDC

**Automatic Circuit DisCovery** (Conmy et al., 2023) prunes the graph greedily from the output
backwards, removing every edge whose removal changes the metric by less than a threshold τ.
[`pipeline/find_circuits/acdc.py`](../pipeline/find_circuits/acdc.py) wraps the vendored ACDC
([`github_repos/Automatic-Circuit-Discovery`](../github_repos/Automatic-Circuit-Discovery/)) with metric KL(clean ‖ patched)
over the answer positions, one candidate `acdc_thresh_<τ>` per τ in `CANON_TAUS = [0.2, 0.1, 0.055,
0.02]`, on 128 pairs (`CANON_ACDC_EXAMPLES`). It accepts the same `--probe` option. Run it with the
non-default `acdc` stage.

### Edge Pruning

**Edge Pruning** (Bhaskar et al., 2024) learns a hard-concrete (L0) mask over edges by gradient
descent. [`pipeline/find_circuits/edge_pruning.py`](../pipeline/find_circuits/edge_pruning.py)
ports it onto the frozen TransformerLens models (the L0 machinery, `pipeline/find_circuits/l0.py`,
is vendored verbatim from the upstream Edge Pruning repository): a masked edge's writer is swapped to its corrupted-input
state, the loss is KL to the full model over the answer positions plus a sparsity constraint, and
edges with gate ≥ 0.5 are kept. One candidate `edge_pruning_sparsity_<s>` per target sparsity in
`CANON_EDGE_PRUNING_SPARSITIES = [0.7, 0.9, 0.95, 0.98]`. Run it with the non-default
`edge_pruning` stage; it is the most expensive finder.

### Edge signs

[`pipeline/edge_signs/edge_sign_census.py`](../pipeline/edge_signs/edge_sign_census.py) tabulates
how much of each candidate's attribution mass is negative (a possible signature of suppression).
It is descriptive only; SuppressionCircuits are selected by their release behavior, not by edge
sign.

## 3. Linear probes

Probes answer "is there an internal variable to find a ProbeCircuit or BridgeCircuit for?".

1. **Declare** the variable as a `ProbeVariable` (labels replayed from the DSL; input positions
   only) or build a bridge with `bridge_variable(composite, k)` — see
   [circuit_classes.md](circuit_classes.md#probecircuit).
2. **Site scan** — [`pipeline/probes/train_probes.py`](../pipeline/probes/train_probes.py)
   `--model <id> --site-scan`: one cached forward on 4,096 uniform random inputs; a multinomial
   logistic-regression probe on the LayerNorm'd residual stream per (layer, position) at
   `blocks.0.hook_resid_pre` (the embedding-only baseline) and every `blocks.{l}.hook_resid_post`,
   25 % held out by row. Optional component sites (`hook_attn_out`, `hook_mlp_out`, per-head) only
   localize. Probes are strictly linear.
3. **Layer choice** — the layer with the highest mean probe accuracy over the variable's positions
   (the default site for `eap.py --probe`); the variable counts as encoded if that accuracy beats
   chance. The scan also records a three-way verdict per variable (`trivial_from_input`,
   `decodable`, `absent`).
4. **Causal check** — [`pipeline/probes/validate_probe.py`](../pipeline/probes/validate_probe.py):
   a permutation control (refit on shuffled labels should sit at chance), a leakage matrix against
   sibling variables, robustness over three refits, and a causal gate that steers the residual along
   the probe's class direction and checks whether the model's **output** follows the DSL
   counterfactual (`flip_rate ≥ 0.80`, `collateral_rate ≤ 0.05`; skipped when the variable defines
   no `counterfactual()`).

Artifacts: `pipeline/probes_data/<model>/<variable>/site_L{l}.pt`, `site_scan.json`,
`validation.json`. In `run_model_pipeline.py` this is the non-default `probes` stage.

## 4. Selection and evaluation

| stage | script | what it does |
|---|---|---|
| selection (zoo-wide runs) | [`pipeline/find_circuits/smallest_faithful_circuit.py`](../pipeline/find_circuits/smallest_faithful_circuit.py) | re-scores every candidate of a circuit under edge-level mean ablation on a held-out window and picks the winner (plain / differential / suppression mode) → `smallest_circuit.json` |
| evaluation (showcase) | [`pipeline/find_circuits/evaluate_circuit.py`](../pipeline/find_circuits/evaluate_circuit.py) | scores a candidate (`--candidate PATH`) or selects among a circuit's candidates (`--select`) with the per-class faithfulness of [circuits.md §4](circuits.md#4-faithfulness-a-by-circuit-class) → `evaluation.json` |

Both are specified precisely in [circuits.md §6](circuits.md#6-the-two-scoring-paths-in-this-repository).

## 5. The pipeline, end to end

```bash
python pipeline/run_model_pipeline.py --model <id> --stages all          # data train eap smallest audit
python pipeline/run_model_pipeline.py --model <id> --stages data eap smallest --only-circuit <slug> --device cpu
python pipeline/run_model_pipeline.py --model <id> --stages all --dry-run   # print the exact commands
```

Stages, in order: `data` (per-circuit input pools) → `train` (skipped when the checkpoint passes
the gate) → `probes` → `patching` ∥ `edges` ∥ `acdc` ∥ `eap` → `edge_pruning` → `smallest` →
`audit` (checks every expected artifact exists). `all` = `data train eap smallest audit`; the
others run only when named. Every analysis stage runs once per declared circuit; `--only-circuit`
restricts it to one. Execution is local by default (`--modal` selects the optional Modal backend).
[`pipeline/build_models.py`](../pipeline/build_models.py) batches the run-list.

## 6. The segrev3 case study

The study that started the repository applies three discovery methods to one 4-layer × 8-head
attention-only model trained on `segrev3`, validates every circuit they produce with one suite, and
checks that they converge:

1. **manual interpretability** — DLA, activation / attribution / path patching, weight-based QK/OV
   and composition (`circuit_discovery/manual_circuit.py`, `interp_tools/circuit_summary.py`);
2. **ACDC** threshold sweep (`circuit_discovery/acdc.py`);
3. **EAP-IG** top-*n* sweep (`circuit_discovery/eap.py`).

The **validation** suite scores every circuit for faithfulness, completeness and minimality under
mean / zero / resample ablation (`validation/validation.py`).

In the committed results, ranking heads by knockout effect, 18 heads recover 0.91 of the model's
accuracy and 24 recover 0.98 (`validation/figs_data/data/manual/faithfulness_curve.json`); the
smallest ACDC / EAP-IG circuit at the case study's 0.80 target is EAP-IG top-87 — 21 heads, 73
edges, recovered accuracy 0.839 (`visualizations/min_faithful_circuit_search.json`,
`visualizations/eap/eap_all_topn_summary.json`). `circuit_discovery/main.py` sequences all of the
methods. Methods, file layout and validated results:
[segrev3_case_study.md](segrev3_case_study.md); every command and parameter:
[segrev3_runbook.md](segrev3_runbook.md).

## References

- A. Conmy, A. Mavor-Parker, A. Lynch, S. Heimersheim, A. Garriga-Alonso. *Towards Automated
  Circuit Discovery for Mechanistic Interpretability.* NeurIPS 2023.
- M. Hanna, S. Pezzelle, Y. Belinkov. *Have Faith in Faithfulness: Going Beyond Circuit Overlap
  When Finding Model Mechanisms.* COLM 2024.
- A. Bhaskar, A. Wettig, D. Friedman, D. Chen. *Finding Transformer Circuits with Edge Pruning.*
  NeurIPS 2024.
- N. Elhage et al. *A Mathematical Framework for Transformer Circuits.* Transformer Circuits
  Thread, 2021 (QK/OV circuits, composition scores).
