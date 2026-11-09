# What a circuit is, and how it is scored

This page defines the objects the rest of the documentation uses: the computational graph, a
circuit, the ablation that evaluates a circuit, faithfulness for each of the five circuit classes,
and the size criterion used to choose among faithful circuits. How each class builds its data is in
[circuit_classes.md](circuit_classes.md); the methods that propose candidate circuits are in
[circuit_finding.md](circuit_finding.md).

## 1. The computational graph

Every model is a TransformerLens `HookedTransformer` with parameter-free pre-LayerNorm (`LNPre`)
and serial attention → MLP blocks (see [models.md](models.md)). Circuits live on the standard
edge-level graph used by EAP-IG and ACDC:

| role | nodes | TransformerLens hook |
|---|---|---|
| **writers** (add to the residual stream) | the input (token + positional embedding, one node), every attention head's output, every MLP / MoE block's output | `blocks.0.hook_resid_pre`, `blocks.{l}.attn.hook_result`, `blocks.{l}.hook_mlp_out` |
| **readers** (read from the residual stream) | each head's Q, K and V input, separately; each MLP / MoE block's input; the final logits | `blocks.{l}.hook_{q,k,v}_input`, `blocks.{l}.hook_mlp_in`, `blocks.{L-1}.hook_resid_post` |

An **edge** is a writer → reader pair in which the writer comes earlier in the forward pass: a
head in layer *l* reads the input and every writer in layers < *l*; the MLP of layer *l* also reads
layer *l*'s heads; the logits read every writer. An MoE block is one node per layer (`m{l}`), not
one node per expert.

The size of the full graph is pure geometry (`canon.live_edge_budget` in
[`data_gen/canon.py`](../data_gen/canon.py), wrapped by `model_registry.live_edges`). For an
attention-only model with *L* layers and *H* heads,

$$E = 3H\left(L + H\,\tfrac{L(L-1)}{2}\right) + 1 + LH .$$

Nodes counted for circuit size are heads plus MLP/MoE blocks, *N = LH* (+ *L* with MLP/MoE); the
input and the logits are not counted. For the eight architectures:

| architecture | N (nodes) | E (live edges) |
|---|---|---|
| small | 8 | 81 |
| large | 32 | 1281 |
| wide | 64 | 4865 |
| deep | 64 | 5633 |
| xl | 128 | 22017 |
| mlp / moe4 / moe8 | 36 | 1519 |

Regenerate with:

```bash
python -c "import sys; sys.path.insert(0,'data_gen'); import canon
for a,(L,H,ao) in {'small':(2,4,1),'large':(4,8,1),'wide':(4,16,1),'deep':(8,8,1),'xl':(8,16,1),'mlp/moe':(4,8,0)}.items():
    print(a, L*H+(0 if ao else L), canon.live_edge_budget(L,H,bool(ao)))"
```

The vendored EAP-IG `Graph` also wires an MLP node per layer in attention-only models; those
edges can never carry signal, score exactly 0, and are excluded from *E*
(`phantom_mlp_indices` in [`pipeline/find_circuits/eap.py`](../pipeline/find_circuits/eap.py)).

## 2. A circuit

A **circuit** is a subgraph (nodes + edges) of this graph that is responsible for the model's
behavior on some input domain: a set of edges plus the node set it implies. Heads are named
`a{l}.h{h}`, MLP/MoE blocks `m{l}`, the endpoints `input` and `logits`, and an edge into a head
names which of its inputs it feeds (`q`, `k` or `v`). On disk a circuit takes one of three forms:

| written by | format |
|---|---|
| EAP-IG candidates (the source of every showcase circuit and of most zoo winners) | `circuit.json` in the vendored EAP-IG `Graph` format, `{"cfg", "nodes", "edges": {"input->a0.h0<q>": {"score", "in_graph"}, ...}}`, every edge of the graph with an `in_graph` flag |
| ACDC and Edge Pruning candidates (`circuit.json`); `smallest_circuit.json` | an edge list `"edges": [{"parent", "child", "qkv", "score"}, ...]` beside the kept `heads` (the two finders also write `nodes` and `mlps`) |
| the showcase (`results/showcase/*/circuit.json`) | `nodes` plus `edges` as `[writer, reader]` pairs, a head reader carrying its input (`["input", "a0.h1<k>"]`) |

Every finder's candidate folder also holds `circuit.pt` in the EAP-IG `Graph` tensor format
(`write_candidate_folder` / `save_circuit_pt` in
[`pipeline/find_circuits/eap.py`](../pipeline/find_circuits/eap.py)).
`evaluate_circuit.py --candidate` reads any of them: a finder's candidate folder, a
`smallest_circuit.json`, an `evaluation.json` or a showcase `circuit.json`.

**There is no ground-truth circuit.** The models are trained, not compiled, so nothing fixes "the"
circuit a model uses. No circuit-finding method in the literature is known to be the best, and
smaller faithful circuits than any method finds may exist. A circuit in this repository is
therefore a *claim* with two measured properties — its faithfulness *a* and its size (*n* nodes,
*e* edges) — and the circuits reported are the smallest faithful ones the finders produced, not
provably minimal ones.

## 3. Edge-level, per-position mean ablation

A circuit is evaluated by running the model with every edge outside the circuit severed. Severed
edges carry the sender's **per-position mean** (on-distribution, not zero). For a reader *r* at
sequence position *p* with declared incoming edges from the sender set *D(r)*:

$$\tilde x_r[p] \;=\; \bar x_r[p] \;+\; \sum_{s \in D(r)} \bigl(a_s[p] - \bar a_s[p]\bigr)
\;=\; \sum_{s \in D(r)} a_s[p] \;+\; \sum_{s \notin D(r)} \bar a_s[p],$$

where $\bar x_r[p]$ is the reader's mean input at position *p* and $\bar a_s[p]$ the sender's mean
output, both taken from one clean, unablated forward pass over a fixed reference set; the second
form holds because the residual stream is the sum of the writer outputs. **Nodes in the circuit are
recomputed** from their reconstructed inputs, so $a_s$ for a circuit node is its output in the
*ablated* forward pass; nodes outside the circuit contribute only their means.

Two identities pin the ablation down:

- declaring **every** edge reproduces the full model (in `smallest_faithful_circuit.py`,
  `assert_full_declared` checks this in each candidate evaluation, max |Δlogit| < 1e-2,
  `FULL_DECLARED_TOL`; `evaluate_circuit.py` instead records the unablated model's score under the
  same metric as `reference.model` in its output);
- declaring **no** edge feeds every reader its pure per-position mean (the *empty circuit*).

Implementation: `_edge_forward` / `_edge_site_means` in
[`pipeline/find_circuits/smallest_faithful_circuit.py`](../pipeline/find_circuits/smallest_faithful_circuit.py).
The reference set for the means is a fixed set of 10,000 clean rows in
[`pipeline/find_circuits/evaluate_circuit.py`](../pipeline/find_circuits/evaluate_circuit.py), and
the first 2,048 clean rows of the scoring window (`MEAN_N_PROD`) in `smallest_faithful_circuit.py`.

## 4. Faithfulness *a*, by circuit class

What counts as "accurate" depends on the class. Every accuracy is measured with the ablation above
(or, for ContrastCircuits, interchange patching — §4.1) and compared with a floor.

| class | faithfulness *a* | floor |
|---|---|---|
| **DomainCircuit** | whole-sequence accuracy (every output digit correct) on the conditioned input domain | **0.90** |
| **ContrastCircuit** | alternating clean / corrupt pairs, $a = a_\text{clean} \times a_\text{corrupt}$, evaluated by **interchange patching** (§4.1) | **0.80** |
| **SuppressionCircuit** | alternating clean / corrupt pairs, $a = a_\text{clean} \times a_\text{corrupt}$, mean ablation; each row is labelled with its *own* task's output (§4.3) | **0.80** |
| **ProbeCircuit** | normalized probe accuracy (§4.2), $a = \hat a_\text{clean} \times \hat a_\text{corrupt}$ | **0.80** |
| **BridgeCircuit** | as ProbeCircuit | **0.80** |

The paired classes use 0.80 rather than 0.90 because their *a* is a product of two accuracies.

### 4.1 Interchange patching for ContrastCircuits

ContrastCircuits are evaluated by interchange patching rather than mean ablation:

- the two members of a pair share **one teacher-forced prefix** — the clean input's true output —
  so the only difference between the two runs is the edited input digit;
- each row's ablation baseline is its **partner's** unablated activations (instead of a dataset
  mean), and labels are the **full model's own outputs**;
- **forward pass**: baseline = the corrupt run; patch the clean row in through the declared edges;
  the result must reproduce the full model's clean outputs. **Reverse pass**: the mirror image.
  $a = \text{forward accuracy} \times \text{reverse accuracy}$ (still $a_\text{clean} \times
  a_\text{corrupt}$);
- pairs whose two label rows do not differ, and pairs whose only difference is the edited digit
  passed straight through to its own output slot (a pure pass-through), are dropped.

Why: under interchange patching every activation that the edit does not change is identical in
the two runs, so it is the same whether or not its edges are declared. The circuit is therefore
exactly the path that carries the edit's effect, and the parts of the model the edit never touches
drop out automatically — the "candidate − DomainCircuit" subtraction used to isolate a contrast
fringe (see [circuit_classes.md](circuit_classes.md#contrastcircuit)) is not needed for evaluation.
By construction the empty circuit scores 0 (it outputs the partner's answers, which differ on every
retained pair) and the full model scores exactly 1 (labels are its own outputs). Patched
faithfulness is **not comparable** with mean-ablation faithfulness.

### 4.2 Normalized probe accuracy (ProbeCircuit, BridgeCircuit)

A probe circuit is scored by the accuracy of the variable's linear probe read from the ablated
model, normalized so that 0 is the majority class (the best constant predictor) and 1 is the full,
unablated model, and clamped at 0:

$$\hat a = \max\!\left(0,\; \frac{\text{acc}_\text{circuit} - \text{acc}_\text{majority}}
{\text{acc}_\text{full} - \text{acc}_\text{majority}}\right), \qquad
a = \hat a_\text{clean} \times \hat a_\text{corrupt},$$

where the corrupt member of each pair is the single-digit edit that makes the probe's correct
class most strongly decodable. The model is cut at the probe layer *l*: components above *l*
cannot affect the probe, so they are outside the graph, and there are no logit readers. The
normalization is clamped at 0 only, so a circuit that decodes the variable slightly *better* than
the full model scores $\hat a$ a little above 1.

Two choices have to match how the circuit was found:

- **the layer.** A circuit is scored at the layer it was discovered at (`--probe-layer`). The
  default is the layer with the highest mean probe accuracy in the variable's site scan.
- **the readout.** The variable can be read by one probe per input position (`--probe-readout
  position`, the evaluator's default) or by one probe pooled over the probed positions
  (`pooled`). Probe-metric EAP-IG takes the same option (`eap.py --probe <variable>
  --probe-readout position|pooled`; its default is `pooled`). The showcase uses the per-position
  probes for both discovery and evaluation, because those probes also decide the variable's
  verdict and best layer. For a variable declared at a single position the two readouts are
  identical. For a multi-position variable they can differ a lot: for the showcase bridge
  variable the per-position probes score 0.998 at layer 2 and the pooled probe 0.876
  (`site_scan.json`).

### 4.3 Suppression pairs

The clean member is the cued task's cue followed by a payload. The corrupt member is the other
task's cue followed by the **same** payload, restricted to payloads on which the two tasks
disagree. Both members are scored by mean ablation (means over the mixed rows) against their
**own** DSL output. On these pairs the clean label is task A's answer and the corrupt label is task
B's answer, so a faithful circuit has to compute both tasks *and* route between them on the cue.
An executor for task A alone fails the corrupt half.

## 5. Choosing among faithful circuits

Among candidates with *a* strictly above the class floor, the reported circuit maximizes

$$\left(1 - \frac{n_\text{nodes}}{N}\right)\left(1 - \frac{n_\text{edges}}{E}\right),$$

with *N* and *E* the node and edge counts of the graph the circuit lives on. That is the full graph
(§1), except for probe classes, where it is the graph truncated at the probe layer. For example, a
4-layer MoE model probed at layer 0 has 9 nodes (8 heads + `m0`) and 43 edges. The product of complements requires a
circuit to be small in **both** nodes and edges; a candidate keeping every node scores 0 however
few edges it has.

## 6. The two scoring paths in this repository

Two scripts score circuits. They implement the definitions above in two different ways, and their
numbers are not interchangeable.

### 6.1 `evaluate_circuit.py` — the showcase and the definitions above

[`pipeline/find_circuits/evaluate_circuit.py`](../pipeline/find_circuits/evaluate_circuit.py)
implements §3–§5 exactly as stated, per class (mean ablation for Domain / Suppression; interchange
patching for Contrast; normalized probe accuracy for Probe / Bridge):

```bash
python pipeline/find_circuits/evaluate_circuit.py --model M --circuit C \
    [--candidate PATH | --select] [--device cpu] [--seed 0] [--ablation auto|mean|interchange]
```

`--circuit` takes a registered circuit id or a probe variable id (`bridge<k>_…` included); the
class, and therefore the definition of *a* and the floor, is inferred from it. Means come from
`--n-rows` (10,000) rows and the paired classes use `--n-pairs` (5,000) clean/corrupt pairs.
Probe classes read the probe trained by the probes stage, at `--probe-layer` (default: the best
layer of its site scan) with `--probe-readout` (§4.2). For ContrastCircuits and
SuppressionCircuits the reported circuit is the candidate itself. Interchange patching (contrast)
and own-task labels (suppression) make subtracting a base circuit unnecessary. The fringe against
the base circuit (`--base-candidate`) is still reported alongside, and `--union-base` scores
candidate ∪ base, which is how a stored fringe-only circuit is evaluated. CPU float reductions
depend on the thread count, so `--threads` (default 4) is fixed for reproducibility. It writes `evaluation.json` next to the circuit's artifacts
(`pipeline/circuits_data/<model>/<circuit>/`, or `pipeline/probes_data/<model>/<variable>/` for
the probe classes). The five showcase circuits in [`results/showcase/`](../results/showcase/) are
selected and scored this way.

### 6.2 `smallest_faithful_circuit.py` — the zoo-wide runs

The zoo-wide discovery runs (hundreds of models; index in [`results/zoo/`](../results/zoo/)) used
the pipeline's own selection stage,
[`pipeline/find_circuits/smallest_faithful_circuit.py`](../pipeline/find_circuits/smallest_faithful_circuit.py),
which scores every candidate under the edge-level mean ablation of §3 on a held-out window of the
circuit's input pool. It has three modes, chosen by the circuit's class
(`run_model_pipeline.py` passes `--base-circuit` for ContrastCircuits and adds `--suppression` for
SuppressionCircuits):

The accuracy floor is `--target` (`run_model_pipeline.py --smallest-target`; suppression mode uses
its own release floor, below); the zoo index records the floor each row was selected at.

**Plain mode (DomainCircuits).** A candidate is eligible iff it keeps at least one node and

1. its absolute exact-match accuracy is ≥ `--target` (`DEFAULT_TARGET = 0.85`),
2. its relative recovery $R = (\text{acc}_\text{circuit} - \text{acc}_\text{empty}) /
   (\text{acc}_\text{full} - \text{acc}_\text{empty})$ is a number and ≥ `--relative-floor`
   (`DEFAULT_RELATIVE_FLOOR = 0.85`; $R$ is undefined when full = empty, which fails), and
3. the empty circuit's accuracy is **below** `--target` (a near-constant-output domain cannot
   credit a circuit for an output it does not compute).

The winner maximizes $(1 - n_\text{nodes}/N)(1 - n_\text{edges}/E)$; ties go to fewer edges, then
fewer nodes, then higher accuracy (`_passes`, `pick_winner`). "Exact match" here is over each row's
scored slots: the input pool is split into clean/resample pairs, and slots where a clean row's true
output equals its resample partner's carry no contrast and are masked out
(`model_registry.dataset_to_pairs`). Scoring uses 5,000 pairs from the pool window
[10000:15000) (`SMALLEST_EVAL_N`, `POOL_WINDOWS["scoring"]`), disjoint from the rows the finders
attributed on.

**Differential mode (ContrastCircuits, `--base-circuit`).** The reported circuit is the
**fringe** = candidate edges − the base DomainCircuit's selected edges. Eligible iff the fringe is
non-empty and the union candidate ∪ base reaches `--target` whole-sequence accuracy (`s_union`).
The winner maximizes
$(1 + \text{clamp}(d_\text{affected}, 0, 1)) \cdot (1 - \tfrac{n_\text{fringe nodes}}{N}\cdot
\tfrac{n_\text{fringe edges}}{E})$, where $d_\text{affected}$ is the drop in whole-sequence
accuracy on the affected output positions when the fringe alone is severed. $d_\text{affected}$,
$d_\text{unaffected}$ and $d_\text{control}$ (severing the fringe on the base circuit's own pool)
are recorded but never gate.

**Suppression mode (SuppressionCircuits, `--suppression --base-circuit task{i}`).** The reported
circuit is the fringe = candidate edges − the cued task's own circuit. Severing the fringe should
**release** the wrong task: `release_rate` = (rows on which every affected slot shows the wrong
task's answer with the fringe severed) − (the same with the full model). Eligible iff the fringe is
non-empty and `release_rate ≥ --release-floor` (0.5) with `collateral ≤ --epsilon` (0.05) on the
positions where the two tasks agree, under **either** of two sever references: per-position means
over cued rows, or over wrong-cue rows (`release_rate_tonic`; a suppressor that is constant across
cues equals its own cued-row mean and is invisible to the first reference). The winner maximizes
$\text{clamp}(\text{release}, 0, 1) \cdot (1 - \tfrac{n_\text{fringe nodes}}{N}\cdot
\tfrac{n_\text{fringe edges}}{E})$, reading the release of whichever reference passed.

In the current code every row, in all three modes, adds one extra EAP-IG candidate from
`eap.py --target-seq-acc` (`run_model_pipeline.py` passes `--search` to every `smallest` run), which
scans a geometric set of top-*n* values from one attribution and keeps the one maximizing
$(1 - n_\text{heads}/N)(1 - n_\text{edges}/E)$ among those with whole-sequence accuracy strictly
above the target, on the pool window [5000:…) (`POOL_WINDOWS["selection"]`). The search only adds a
candidate, and the winner is still chosen by the mode's own rule. By default the stage also runs one
round of midpoint refinement (`--refine-rounds`, default 1). That round discovers and scores
candidates at parameters between adjacent rungs.

Each `smallest_circuit.json` records the `selection_regime` and `selection_strategy` it was selected
under. Some zoo rows were produced by earlier versions of this stage and of the finders (for
example, the absolute EAP top-*n* ladder that capped contrast and suppression candidates at 800
edges); [`results/zoo/stats.md`](../results/zoo/stats.md) documents what the index records.

### 6.3 What is and is not comparable

| comparison | comparable? |
|---|---|
| two showcase circuits of the same class | yes — same evaluator, same definition |
| a showcase ContrastCircuit vs. a zoo contrast row | no — interchange patching vs. differential mode on a fringe |
| a showcase DomainCircuit vs. a zoo domain row | no — floor 0.90 on whole-sequence accuracy vs. the 0.85 three-way test on masked slots (0.90 in the early run) |
| zoo rows with different `selection_regime` | no |

## 7. Floors versus the training gate

The faithfulness floors above govern **circuits**. The 0.95 sequence-accuracy gate in
`promote.py` / `run_model_pipeline.SEQ_ACC_GATE` governs **checkpoints** (whether a trained model is
kept or retrained, see [models.md](models.md#training)) and never enters circuit selection.
