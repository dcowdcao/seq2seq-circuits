# The seq2seq models: task format, DSL, architectures, training

Every model in this repository is a small attention-based transformer trained from scratch to
implement **one algorithm** on digit sequences. The algorithm is named by an id in a small DSL, and
the id alone determines the task, the output length, the context length and the answer positions.

## Task format

| | |
|---|---|
| input | 10 digits, 0–9 (mux models: a cue token, then 10 digits) |
| output | *L* digits, *L* = `output_length()` of the algorithm (10 for almost every model) |
| model sees | `[x | y]` — the input followed by the output, `n_ctx = len(x) + L` tokens; no BOS / separator |
| training | teacher forcing: the logit at position 9+k predicts *y_k* (cross-entropy on all output positions) |
| answer positions | 9 … 9+L−1 (mux models: 10 … 10+L−1) — `ModelSpec.answer_positions` |
| vocabulary | 10 digit tokens (mux models with *N* tasks: 10 + *N*, the cue tokens are 10 … 9+N) |
| evaluation | whole-sequence exact match: every output digit correct |

`pipeline/models/model_registry.py` builds a `ModelSpec` from the model name (`get_spec`): task
function, lengths and answer positions come from the DSL id, architecture from the checkpoint's
stored `config` (or the variant's preset before training). Models are loaded only through
`interp_tools/model_loading.py:load_segrev3`, which builds the TransformerLens config with
`normalization_type="LNPre"` (parameter-free pre-LayerNorm). Loading these checkpoints with
`normalization_type=None` silently breaks them — see
[segrev3_case_study.md §1.4](segrev3_case_study.md#14-the-lnpre-gate-why-it-matters-everywhere).

## The DSL

[`pipeline/models/algorithms.py`](../pipeline/models/algorithms.py) defines the DSL engine.

- Every operator is a `DSLExpr` with `eval(x)` (a pure reference implementation on a tensor),
  `output_length()` (static, without evaluating — this is what fixes `n_ctx`), `vocab_size()` and
  `sample_row(u)` (maps a uniform digit draw to a valid input row; identity except for mux models).
- Operators are **unary** and compose by nesting: `Op(child)`, with `Id(10)` (the raw input) as the
  leaf. Parameters live in the operator, e.g. `SegmentReverse(3, ·)`, `SRLatch(1, 2, ·)`.
- An **id** spells the expression outermost-first, one token per operator, separated by `_` and
  ending in `id`; parameters are part of the token. `decode_algorithm_id` parses an id,
  `base_algorithm_id` strips the architecture suffix, `task_fn_for` / `output_length_for` give the
  batched task function and output length.
- **Architecture suffix**: `<id>__<variant>` with variant ∈ {`small`, `wide`, `deep`, `xl`, `mlp`,
  `moe4`, `moe8`}; no suffix = `large`. The suffix changes only the architecture, never the task.
- **Mux models** (`TaskMux`): one underscore-free token `mux<N>.<m0>.<m1>...`, where each member
  id has `_` replaced by `-`. The input is `[cue, payload×10]` with cue token `10+i` selecting
  member *i*; all members map 10 digits to 10 digits.

Examples (input `3141592653`; outputs computed with `decode_algorithm_id(id).eval`):

| id | expression | output |
|---|---|---|
| `rev_id` | `Reverse(Id(10))` | `3562951413` |
| `segrev3_id` | `SegmentReverse(3, Id(10))` — reverse each run of non-3 digits | `3562951413` (here the 3s sit at both ends) |
| `novel_id` | `NoveltyIndicator(Id(10))` — 1 if the digit has not appeared before | `1110111100` |
| `psum_novel_id` | `PrefixSumMod(NoveltyIndicator(Id(10)))` — composition | `1233456777` |
| `schmitt_id` | `Schmitt(Id(10))` — two-threshold switch with memory | `0000090990` |
| `sr12_id` | `SRLatch(1, 2, Id(10))` — on at 1, off at 2 | `0111110000` |
| `argminmax_id` | `ArgMinMax(Id(10))` — the one 20-digit output in the run-list | `00000100000100000000` |
| `mux2.pmstep-id.sr12-id` | `TaskMux([ParityRoute(Id(10)), SRLatch(1, 2, Id(10))])` | cue 10 → `pmstep`, cue 11 → `sr12` |

Operators added through the brainstorming process ([brainstorming.md](brainstorming.md)) are
written as a pure-Python reference function `_ref_<name>(x)` wrapped by a `DSLExpr` class; the older
core operators are written directly against tensors. The full operator table is
[reference/operators.md](reference/operators.md); a plain-language description of every
brainstormed operator, grouped by mechanism, is
[brainstorming/operator_glossary.md](brainstorming/operator_glossary.md).

**The run-list** [`pipeline/models/models.txt`](../pipeline/models/models.txt) enumerates one entry
per (base algorithm, architecture). It documents the zoo; it is not a gate — any valid DSL id can be
trained and analysed. `pipeline/models/circuits.txt` is the generated catalog of declared circuits
(`pipeline/repo_tools/make_circuits_txt.py`). Counts are in
[`results/zoo/stats.md`](../results/zoo/stats.md).

### Domains the algorithms span

The algorithms were chosen to cover many kinds of computation (the glossary groups them into
twenty mechanism families). A few representatives, all in the run-list:

| domain | examples |
|---|---|
| physics-like dynamics | `graviclump_id` (one step toward the row's average — gravity / center-of-mass drift), `spile4_id` (sandpile toppling), `cyca_id` (cyclic cellular automaton) |
| electrical engineering / signals | `schmitt_id` (Schmitt trigger), `sr12_id` (set/reset latch), `ewma2_id` (leaky running total) |
| computer science | `segrev3_id` (segment reversal), `heapmax_id` (binary-heap parent max), `runmex_id` (smallest digit not yet seen), `vig3_id` (Vigenère-style repeating-key shift) |
| mathematics | `psum_id` (prefix sum mod 10), `areset13_id` (accumulator with a mod-13 reset), `euclstep_id` (Euclid remainder step), `s3prod_id` (running product in the symmetric group S₃) |
| compositions | `psum_novel_id`, `dblodd_walk10_recordind_id` (three operators stacked) |
| multiplexed | `mux2.pmstep-id.sr12-id`, `mux5.psum-id.cmpswap0p9-id.jump10-id.pmstep-id.segrev3-id` |

## Architectures

All variants are TransformerLens `HookedTransformer`s: causal, learned absolute positional
embeddings, parameter-free pre-LayerNorm (`LNPre`) before every block and the unembedding.
Geometry comes from `TIER_PRESETS` in
[`interp_tools/model_loading.py`](../interp_tools/model_loading.py); the name suffix → tier map is
`VARIANT_TIER` in [`pipeline/models/train_models/hparams.py`](../pipeline/models/train_models/hparams.py).

| variant | suffix | tier | layers | heads | hidden size | d_head | feed-forward |
|---|---|---|---|---|---|---|---|
| Small | `__small` | `small` | 2 | 4 | 64 | 16 | – (attention-only) |
| Large | *(none)* | `large` | 4 | 8 | 128 | 32 | – |
| Wide | `__wide` | `wide` | 4 | 16 | 256 | 32 | – |
| Deep | `__deep` | `deep` | 8 | 8 | 128 | 32 | – |
| XL | `__xl` | `xlarge` | 8 | 16 | 256 | 64 | – |
| MLP | `__mlp` | `mlp` | 4 | 8 | 128 | 32 | d_mlp = 512, GELU |
| MoE4 | `__moe4` | `moe4` | 4 | 8 | 128 | 32 | d_mlp = 128, 4 experts, top-1 |
| MoE8 | `__moe8` | `moe8` | 4 | 8 | 128 | 32 | d_mlp = 128, 8 experts, top-1 |

**Escalation twins.** `TIER_PRESETS` also defines `mlp_xl`, `moe4_xl` and `moe8_xl` (XL geometry,
8 layers × 16 heads, hidden size 256, keeping the feed-forward type; d_mlp 1024 / 256 / 256). They
are not separate architectures of the zoo; they are where training escalates when a model cannot
reach the gate at its own size (below).

## Training

`python pipeline/run_model_pipeline.py --model <id> --stages train` runs
[`run_training_job.sh`](../pipeline/models/train_models/run_training_job.sh) →
[`train.py`](../pipeline/models/train_models/train.py) (via `torchrun`; a single process without
DDP when run directly). The recipe is not a command-line flag: both training paths read it from
one per-architecture table, `hparams.for_tier` (`REGIME = "perarch-v1"`). Print it with
`python pipeline/models/train_models/hparams.py --table` or
`python pipeline/run_model_pipeline.py --print-hparams`.

| | `perarch-v1` recipe (current code) |
|---|---|
| data | uniform random 10-digit rows (through `sample_row`), labelled on the fly by the DSL task function; a 32,768-row training pool (`POOL`), a 65,536-row evaluation set (`EVAL_N`, fixed seed 42) |
| loss | cross-entropy on every output position, teacher-forced `[x | y]` |
| optimizer | `torch.optim.AdamW` (PyTorch defaults besides the learning rate) |
| learning rate | `1e-3 · sqrt(128/hidden size) · sqrt(4/layers)` (`lr_for`): small 2e-3, large / mlp / moe 1e-3, wide and deep 7.07e-4, xl 5e-4; wide's value comes from the rule alone, with no sweep run for it (`--table` marks it `*`) |
| schedule | 200 steps of linear warmup, then constant; gradient clipping at 1.0 |
| batch / steps | effective batch 64 (split across DDP ranks), 20,000 steps |
| precision | fp32 |
| early stop / checkpoint | evaluated once per pass over the pool; the best-evaluation checkpoint is kept; training stops as soon as evaluation sequence accuracy ≥ 0.95 |
| divergence tripwire | at XL only, training fails if ‖W_E‖ exceeds 4.76 (`check_we_norm`) |

**Gate and escalation.** A checkpoint is promoted to `pipeline/models/trained_models_pt/<id>.pt`
only through [`promote.py`](../pipeline/models/train_models/promote.py), and only if its recorded
whole-sequence accuracy is ≥ 0.95; a checkpoint with no recorded accuracy is refused. If the best
accuracy lands in [0.80, 0.95), `run_training_job.sh` retries one tier up along
`UPGRADE_LADDER` in [`run_model_pipeline.py`](../pipeline/run_model_pipeline.py): `small → large`,
`large / wide / deep → xlarge`, `mlp → mlp_xl`, `moe4 → moe4_xl`, `moe8 → moe8_xl`, `xlarge` pinned.
On a re-run, training is skipped when the existing checkpoint passes the gate
([`check_ckpt_gate.py`](../pipeline/models/train_models/check_ckpt_gate.py)). The published
snapshot nevertheless contains 8 checkpoints below 0.95. Earlier versions of the training scripts
wrote checkpoints without this gate. The 8 are listed in [stats.md](../results/zoo/stats.md) §1, and
they and their rows are included in every zoo table.

**Checkpoint format.** A `torch.save` dict with `model_state_dict` (TransformerLens names),
`config` (geometry, `d_vocab`, `n_ctx`, `attn_only`, `d_mlp`, `act_fn`, `eps`, and the MoE keys),
`algorithm_id`, `epoch`, `metrics` (the epoch's train / eval numbers; checkpoints trained under
`perarch-v1` also carry `seed` and the full `hparams` recipe) and, from `train.py`, the optimizer
and scaler states.

**Batched training.** [`train_stacked.py`](../pipeline/models/train_models/train_stacked.py) trains
many same-configuration attention-only models in one vectorized (`vmap`) run and
[`train_sharded.py`](../pipeline/models/train_models/train_sharded.py) shards a model set across
GPUs (`--train-backend stacked`; MLP/MoE models are not batched this way) — see
[reference/stacked_training.md](reference/stacked_training.md).

**Determinism.** Model initialization and data order are seeded (`hp.seed`), but the training pool
itself is drawn from an unseeded generator (`SequenceDataset(seed=None)`), so retraining reproduces
the recipe and the gate, not bit-identical weights. Circuits are therefore always reported against
a specific checkpoint; the showcase records store its sha256. The checkpoints themselves are not
included in the repository.

### Provenance of the checkpoints

The recipe table above is the current code. Every checkpoint in this snapshot, including the five
showcase checkpoints, was trained **before** it existed: none carries an `hparams` block
(`hparams_regime` is empty for all of them in
[`results/zoo/models_index.csv`](../results/zoo/models_index.csv)). Before the table, the recipe
depended on which training path ran; the module docstring of `hparams.py` records that the two
paths used different learning rates, batch sizes, step budgets, dtypes and gate eval sizes. The
checkpoints record three metadata layouts ([`results/zoo/stats.md`](../results/zoo/stats.md) §8):

- `steps+eval` — the vectorized size-grid trainer, attention-only, 675 checkpoints: the step count
  (median 20,000) and eval accuracy only. Its budget is the one the table keeps (20,000 steps,
  batch 64, a 32,768-row pool; `STEPS`, `BATCH`, `POOL` in `hparams.py`) at the anchor learning
  rate 1e-3, and its gate was evaluated on 4,096 rows.
- `epoch+train+eval` — the per-model trainer, used for every MLP/MoE checkpoint and every
  multiplexed one (578 checkpoints): an epoch index and train / eval metrics. Its gate was
  evaluated on 10⁶ rows, and its MLP/MoE models saw ≈23× more data than the size grid (the
  `SWEEP_SPACE` and `EVAL_N` comments in `hparams.py`; `train.py`).
- `flat+recipe` — 23 Large checkpoints with the recipe written in: lr 0.004, batch 4096, max
  steps 30,000.

The corpus contains both fp32 and bf16 training (`SWEEP_SPACE` comment); every checkpoint stores
float32 parameters, and none records its training precision. Beyond the 23 `flat+recipe`
checkpoints, no checkpoint records a learning rate or batch size, so what is known about the older
recipes is the `hparams.py` and `train.py` comments cited above.

## Known limitations

- **Shared hyperparameters.** Most attention-only checkpoints (the size grid above) were trained
  with one hyperparameter configuration shared across their architectures. Every XL checkpoint in
  the snapshot clears the gate, but only 95 of the 242 XL run-list entries have a checkpoint in the
  snapshot, the lowest coverage of any architecture ([stats.md](../results/zoo/stats.md) §1,
  "Run-list coverage").
- **Recipe confound between architecture families.** Every MLP/MoE checkpoint came from the
  per-model trainer and 675 of the 725 attention-only ones from the size grid
  ([stats.md](../results/zoo/stats.md) §8); the two paths differed in learning rate, batch size,
  step budget, dtype and gate eval size, and the MLP/MoE models saw ≈23× more data (above).
  Attention-only vs. feed-forward comparisons in the zoo are therefore confounded with the recipe
  until the zoo is retrained under `perarch-v1`.
- **Coverage.** The zoo-wide runs select DomainCircuits, ContrastCircuits and SuppressionCircuits;
  for the probe classes they record probe decodability only ([stats.md](../results/zoo/stats.md)
  §7), and the showcase has one circuit of each. SuppressionCircuit rows exist at all eight
  architectures, but faithful ones only at Large, MLP, MoE4 and MoE8: none of the 26 rows at each
  of Small, Wide and XL, or of the 6 at Deep, is faithful (stats.md §3), and the snapshot has only
  1–2 mux checkpoints at each of those four architectures against 12–13 at the others (stats.md §6).
- **Probes read input positions only**, never output positions.
- **Candidate cap.** In the zoo-wide runs, Contrast and Suppression candidates were capped at 800
  edges, the top of the absolute EAP-IG top-*n* ladder then in use (`CANON_TOPNS = [50, 200, 800]`,
  now the `abs` regime; the default is the scale-relative `frac` ladder, see
  [circuit_finding.md](circuit_finding.md#eap-ig)).
- **Fixed input length.** Every model reads exactly 10 digits (`INPUT_LEN = 10`).
- **One seed per model.** Each model was trained once; circuits are not compared across seeds.
