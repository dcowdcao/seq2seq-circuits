# Stacked training (`train_stacked.py`)

Train every same-architecture toy model at once on one GPU, then split the result into individual
drop-in checkpoints. Code: [pipeline/models/train_models/train_stacked.py](../../pipeline/models/train_models/train_stacked.py)
(one GPU) and [train_sharded.py](../../pipeline/models/train_models/train_sharded.py) (several GPUs,
and the size grid). Architectures and the training recipe table are described in
[../models.md](../models.md).

## Why

Each toy model is tiny (Large is 4 layers × 8 heads, hidden size 128), so a single model badly
under-uses a modern GPU. The per-model trainer (`train.py`, driven by `run_training_job.sh`) trains
models one at a time. Stacked training packs N models onto a leading "model" dimension and trains
them in one `torch.func.vmap` forward/backward. Each model sees only its own task's data and shares
no weights, so it is N independent trainings in parallel lanes sharing one kernel launch; splitting
is slicing lane `i` off every stacked weight.

## Mechanism

`torch.func.stack_module_state(models)` → a param dict of `[N, …]` tensors →
`vmap(functional_call)` runs the forward across the model axis → one summed loss (the sum over
models of each model's mean cross-entropy) gives every model exactly its own gradient → slice
`params[k][i]` back into `<name>.pt`. Because the summed loss decouples across lanes, a split
checkpoint is equivalent to having trained that model alone with the same recipe and seed.

## When it wins

The speedup exists only at small per-model batch. Once a single model's batch is large enough to
saturate the GPU, vmap only adds bookkeeping and stacking can be slower than sequential training.
At small batch one model under-uses the GPU and vmap fills it, so N stacked models cost little more
than one. This is why the shared recipe uses a small batch (`hparams.BATCH = 64`).

The batch is a property of the architecture, not of this script: both training paths resolve their
recipe from one per-architecture table,
[pipeline/models/train_models/hparams.py](../../pipeline/models/train_models/hparams.py)
(`hparams.for_tier` / `hparams.for_model`), and record it in every checkpoint
(`metrics.hparams.regime`). The checkpoints in the zoo were trained before this table existed, with
two different recipes for the attention-only and the feed-forward architectures; see
[circuit_types_across_architectures.md](circuit_types_across_architectures.md#9-training-recipe-confound).

Trade-off against training each model on its own GPU: stacking cuts total GPU time on one GPU at
the cost of longer wall-clock than a fully parallel fan-out, so it wins when GPU count is the
constraint.

## Contract

- **Buckets.** Models are bucketed by identical config (`Spec.key()`). Different tiers, output
  lengths or vocabularies (mux models) cannot share a stack, so models fall into one bucket per
  (tier, output length, vocabulary); off-shape models (e.g. a longer `n_ctx`) end up as
  singletons.
- **Reuse.** The data path (`utils.SequenceDataset`), model build (`model_loading.tier_config` /
  `segrev3_config`), loss, and the teacher-forced sequence accuracy that gates the pipeline
  (`metrics.task_accuracy`) are the per-model trainer's.
- **Target.** `--target` defaults to the pipeline's training gate (`promote.GATE`, 0.95), so a
  passing checkpoint is one the pipeline will not retrain. Each model is **banked** at the step it
  passes: its weights are snapshotted and it stops training. Masking its loss alone does not freeze
  it, because AdamW applies decoupled weight decay to a zero-gradient parameter; a banked model's
  rows are therefore rewritten from the bank after every optimizer step.
- **Only models that reach the target are written.** Promotion into the real checkpoint tree goes
  through `promote_grid.py` → `promote.py` (the single gate), never a direct write.
- **Checkpoints** are standard: a fresh model loads the slice and its `state_dict()` is saved in the
  usual format, with the strict sequence accuracy, the seed and the resolved recipe under
  `metrics`, so the gate can check a checkpoint without re-evaluating it.
- **Safe by default.** Output goes to `pipeline/models/trained_models_pt_stacked/` (`--out`),
  never the real tree.
- **Fallback.** Models that do not reach the target within the recipe's step budget
  (`hparams.STEPS`) are left to the per-model trainer, whose escalation ladder (`UPGRADE_LADDER`
  in [pipeline/run_model_pipeline.py](../../pipeline/run_model_pipeline.py)) retries them at a
  larger geometry.
- **Seeds.** `--seeds N` trains N initializations per model; extra seeds land in
  `<name>.seed<k>.pt` so the registry's `<name>.pt` is untouched.

From the orchestrator, `run_model_pipeline.py --train-backend stacked` runs a stacked pre-pass
(via `train_sharded.py --promote`) before the normal per-model `train` stage; it is local-only.

```bash
python pipeline/models/train_models/train_stacked.py --models rcnt3_id dblodd_id --target 0.95
python pipeline/run_model_pipeline.py --model rcnt3_id dblodd_id --stages train --train-backend stacked
```

## Multiple GPUs: model sharding

`train_sharded.py` is the multi-GPU path. It splits the model list into packs of `N*` models, runs
one `train_stacked.py` per GPU pinned via `CUDA_VISIBLE_DEVICES`, with no cross-GPU communication
(the models are independent), and gathers the checkpoints. `N*` is the vmap pack size at the
throughput knee for the given model, batch and GPU; `--models-per-gpu auto` (the default)
micro-benchmarks it per bucket (on a machine without CUDA it falls back to 8). It uses
`min(available GPUs, number of packs)` GPUs unless `--gpus` is given. Data is an on-the-fly stream,
so there is nothing to shard on the data side; each model just needs steps.

Model sharding is the right axis here: DDP/FSDP exist for one model too large for a GPU, with
communication every step, whereas these models are tiny and independent. Buckets that vmap cannot
batch (MLP and MoE forwards) are skipped and go to the per-model trainer.

GPU count is a throughput knob, never a recipe change: `hparams.BATCH` is the effective batch and
`hparams.per_rank_batch(hp, world_size)` splits it across ranks, raising if it does not divide.

### Alternative considered: local SGD across spare GPUs

When more GPUs are free than there are packs, the spares could data-parallelize one pack: R
replicas on their own data, all-reducing every K steps. Averaging both the weights and Adam's
moment estimates is the reconciliation that works (averaging weights only leaves stale momentum;
zeroing the moments is unstable), but the gain is sublinear in the number of replicas, so model
sharding dominates whenever distinct models remain. It is not implemented.

## The size grid (`train_sharded.py --sizes`)

Train every model at every size, then choose architectures per model afterwards.
`--sizes small large wide deep xlarge` (or `--sizes all`) expands each base model into the
`__<size>` variant names, buckets by config, benchmarks `N*` per bucket (larger sizes fit fewer
models per GPU), shards and trains. Outputs:

- per-(model, size) checkpoints `pipeline/models/trained_models_pt_grid/<model>__<size>.pt`
- `architecture_grid.json`: `{model: {size: {acc, keeper}}}`, the input to selection.

Flow:

1. `python pipeline/models/train_models/train_sharded.py --sizes all` — runs locally by default;
   `--modal` optionally forwards the run to a Modal multi-GPU container.
2. Read `architecture_grid.json` and pick the size(s) per model (for example the smallest that
   clears the training gate).
3. Promote the chosen checkpoints into `pipeline/models/trained_models_pt/` with
   `promote_grid.py` (or `train_sharded.py --promote`); both the grid's `keeper` flag and
   `promote.check` must agree.
4. `python pipeline/run_model_pipeline.py --model <model>__<size> --stages all` runs circuit
   finding on the chosen architecture.

Nothing in the analysis pipeline changes: the grid only produces standard checkpoints under
standard variant names.

## Known limits

- **Pool generation is the setup bottleneck.** `SequenceDataset` labels rows one at a time in
  Python; parallelizing (DataLoader workers) or vectorizing the DSL evaluation would help if it
  dominates.
- **Evaluation memory is bounded by construction.** `EVAL_ROWS_PER_PASS = 16_384` fixes the row
  budget per forward and the per-model chunk derives from the live bucket size, so evaluating a
  large bucket at `hparams.EVAL_N = 65_536` rows never needs one giant forward.
- Small-batch convergence needs more steps than large-batch training; a straggler that plateaus
  below the target runs to the end of the step budget (banked peers ride along unchanged) and then
  takes the fallback.
- MLP and MoE buckets are skipped (vmap cannot batch their forwards) and mux buckets are untested
  (mux models differ in vocabulary).
