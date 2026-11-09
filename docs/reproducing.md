# Reproducing the results

Reproduction is tiered by cost. Tiers A, B and E run on a CPU; tier C (retraining) is meant for a
GPU; tier D regenerates the zoo statistics from a committed index, and only rebuilding that index
requires rerunning the zoo.

The trained checkpoints of the five showcase models are not included in the repository. Tiers A
and B load them from `pipeline/models/trained_models_pt/<id>.pt`, so train them first with tier C.
The segrev3 checkpoint (tier E) is included.

| tier | what | needs |
|---|---|---|
| [A](#a-verify-the-five-showcase-circuits) | re-evaluate the five stored showcase circuits on their checkpoints | CPU |
| [B](#b-rerun-discovery-for-a-showcase-example) | rediscover a showcase circuit from a checkpoint | CPU |
| [C](#c-retrain-a-model-from-scratch) | retrain any model | GPU recommended |
| [D](#d-zoo-statistics) | zoo-wide statistics from the committed index; rebuild the index after rerunning the zoo | CPU (rerunning the zoo: many GPU jobs) |
| [E](#e-the-segrev3-case-study) | the segrev3 case study | CPU |

Tier A takes about 10–16 minutes on a 16-core CPU, depending on load. Per-stage run times for B
are listed in [`results/showcase/COMMANDS.md`](../results/showcase/COMMANDS.md); all five examples
together take about 4 hours of CPU.

## Setup

```bash
# system dependency of pygraphviz (circuit drawings)
sudo apt-get install graphviz libgraphviz-dev pkg-config   # Debian/Ubuntu;  macOS: brew install graphviz

# from the repository root
python3 -m venv .venv
source .venv/bin/activate
pip install torch==2.12.0 --index-url https://download.pytorch.org/whl/cpu   # CPU-only machines
pip install -r requirements.txt                   # pinned versions
```

The release was developed and tested with Python 3.12 on a CPU-only machine. Create the
environment as `.venv` in the repository root: `circuit_discovery/main.py` and
`pipeline/run_model_pipeline.py` re-execute themselves under `./.venv` when launched with a bare
system Python. No API keys or network access are needed after installation; every model and
dataset is local.

Every script takes a device. Pass `--device cpu` where it is offered (the pipeline forwards it to
every stage through the `SEGREV3_DEVICE` environment variable); `auto` picks CUDA when available.
Scripts are seeded (`--seed`, default 0).

## A. Verify the five showcase circuits

```bash
python results/showcase/reproduce.py --verify
```

This loads each showcase checkpoint
(`pipeline/models/trained_models_pt/{schmitt_id__moe8, mexhat_id__moe8, graviclump_id__moe4,
psum_novel_id__moe8, mux2.pmstep-id.sr12-id__moe8}.pt`) and each committed `circuit.json`, recomputes
faithfulness and size with `evaluate_circuit.py`, and checks every recorded number in
[`results/showcase/`](../results/showcase/). `--only <example dir> …` restricts it to some
examples. The stored numbers hold only for the weights they were computed on, whose sha256 each
`circuit.json` records (`checkpoint_sha256`); a retrained model has different weights, so on
retrained checkpoints use tier B instead. The evaluator's own self-test, which needs the showcase
checkpoints too (original or retrained), runs as a plain script and takes about a minute. It checks the
edge and node universes, that the edge-level forward matches the pipeline's, the full-circuit and
empty-circuit identities for each class, the probe normalization, and determinism:

```bash
python pipeline/find_circuits/test_evaluate_circuit.py
```

## B. Rerun discovery for a showcase example

```bash
python results/showcase/reproduce.py --full
```

deletes the examples' generated (gitignored) artifacts, reruns discovery from the checkpoints
(data, EAP-IG candidates, probes and the routing diagnostic where the class needs them), selects
each circuit with `evaluate_circuit.py --select`, and regenerates the figures, the pages and
`COMMANDS.md` with measured wall times. It rewrites the committed example files, so on retrained
checkpoints they then describe the retrained models, whose circuits and numbers differ from the
stored ones (`git checkout results/showcase` restores those). The exact per-example commands are in
[`results/showcase/COMMANDS.md`](../results/showcase/COMMANDS.md); for the DomainCircuit example
they are

```bash
# input pool, checkpoint gate check (training is skipped), EAP-IG candidates, zoo-style selection
python pipeline/run_model_pipeline.py --device cpu --model schmitt_id__moe8 \
    --only-circuit schmitt_noband --stages data train eap smallest audit --smallest-target 0.9
# per-class faithfulness and selection among the candidates
python pipeline/find_circuits/evaluate_circuit.py --model schmitt_id__moe8 --circuit schmitt_noband \
    --select --device cpu --seed 0
```

Candidates land in `pipeline/circuits_data/schmitt_id__moe8/schmitt_noband/circuits/eap_top<n>/`,
drawings in `pipeline/circuits_figures/…`, and `evaluate_circuit.py` writes `evaluation.json` in
the circuit's `circuits_data` directory. To score one specific candidate instead of selecting, pass
`--candidate PATH` (or `full` / `empty`); `--ablation auto|mean|interchange` overrides the class's
default ablation. Probe and bridge examples additionally run the `probes` stage and
`eap.py --probe <variable>`. Print any pipeline invocation without running it with `--dry-run`.

## C. Retrain a model from scratch

```bash
python pipeline/run_model_pipeline.py --model <id> --stages train
python pipeline/run_model_pipeline.py --print-hparams          # the recipe that will be used
```

`<id>` is any DSL id with an optional architecture suffix (for example `segrev3_id__wide`); it does
not have to appear in `pipeline/models/models.txt`. The stage runs `run_training_job.sh`, which
launches `train.py` under `torchrun` with one process per visible GPU, escalates one architecture
tier if the model lands in [0.80, 0.95), and promotes the checkpoint to
`pipeline/models/trained_models_pt/<id>.pt` only if its sequence accuracy is ≥ 0.95
([models.md](models.md#training)). If a checkpoint already exists and passes the gate, the stage
skips training. The five showcase models are `schmitt_id__moe8`, `mexhat_id__moe8`,
`mux2.pmstep-id.sr12-id__moe8`, `psum_novel_id__moe8` (tier `moe8`) and `graviclump_id__moe4`
(tier `moe4`).

`run_training_job.sh` always launches `torchrun`, whose distributed backend needs a GPU. Without a
GPU, call the trainer directly — it runs as a single process when not launched by `torchrun` — and
promote the result through the same gate. The trainer runs on a CPU, but slowly at the full
recipe (20,000 steps); a GPU is recommended:

```bash
python pipeline/models/train_models/train.py --algorithm <id> --config <tier> \
    --allow_repeats --random_sampling --output_dir /tmp/ckpts
python pipeline/models/train_models/promote.py /tmp/ckpts/checkpoint_<id>_best.pt \
    pipeline/models/trained_models_pt/<id>.pt
```

where `<tier>` is the variant's tier (`large` for no suffix, `xlarge` for `__xl`, otherwise the
suffix itself). The training pool is drawn from an unseeded generator, so a retrained model matches
the recipe and the gate, not the committed weights bit for bit; circuits found on a retrained model
are circuits of that model. The full chain for one model is

```bash
python pipeline/run_model_pipeline.py --model <id> --stages all     # data train eap smallest audit
```

## D. Zoo statistics

The zoo-wide training and discovery runs took hundreds of GPU jobs, so the repository ships the code,
the commands and a compact index of their results rather than every artifact:

```bash
python results/zoo/zoo_stats.py           # regenerate the statistics from the committed index
```

The index (`results/zoo/circuits_index.csv`, `probes_index.csv`, `models_index.csv`) and what each
column means are described in [`results/zoo/stats.md`](../results/zoo/stats.md). To rebuild it from
scratch, rerun the zoo and then rebuild the index from the regenerated artifacts:

```bash
python pipeline/build_models.py --dry-run         # the run-list and the exact pipeline invocation
python pipeline/build_models.py                   # every entry of pipeline/models/models.txt, locally
python pipeline/build_models.py --modal           # the same, on the optional Modal backend
python pipeline/run_model_pipeline.py --model <id> --stages probes   # probe scans (non-default stage)
python results/zoo/build_index.py --overwrite     # re-index pipeline/circuits_data, probes_data, checkpoints
```

`build_index.py` replaces the committed CSVs only with `--overwrite`, so run it that way only
after rerunning the zoo. Without the flag it refuses to touch existing CSVs, and it also stops
before writing if an input directory is missing. To compare a partial rerun with the committed
index, write elsewhere with `--out DIR`. A `reproduce.py --full` run creates `pipeline/circuits_data`
and `pipeline/probes_data` for the five showcase models only, so indexing them over the committed
snapshot would erase it.

A rerun uses the current code, which differs from the code that produced some committed zoo rows
(training recipe, EAP-IG ladder, selection holdout — see [models.md](models.md#provenance-of-the-checkpoints)
and [circuits.md §6.2](circuits.md#62-smallest_faithful_circuitpy--the-zoo-wide-runs)); compare
rebuilt and committed indexes with that in mind.

## E. The segrev3 case study

```bash
python interp_tools/model_loading.py              # load check: asserts token accuracy > 0.99
python circuit_discovery/main.py all --dry-run    # print the stage plan of the whole study
```

`model_loading.py` is the regression test for the `LNPre` loader; if it fails, every downstream
number is invalid. To reproduce the committed case-study numbers, follow
[segrev3_runbook.md §1](segrev3_runbook.md#1-reproduce-the-committed-numbers), which maps every
result to its command, for example

```bash
python circuit_discovery/main.py data                                           # inputs (datasets/)
python circuit_discovery/acdc.py --thresholds 0.2 0.1 0.055 0.02 -n 20 --device cpu   # ACDC sweep
python circuit_discovery/eap.py --topns 50 75 87 100 200 400 800                # EAP-IG sweep
python validation/validation.py --circuit eap                                   # validation
python interp_tools/min_faithful_circuit.py --refine-rounds 5                   # smallest circuit at the 0.80 target
```

The committed `datasets/` files are the exact inputs of the committed results; every runner loads
them as-is, and the `data` stage (`main.py data`, also part of `main.py all`) rewrites them byte for
byte. The committed artifacts were computed on CPU with the runbook's commands, in the runbook's
order. The methods and the validated results are in
[segrev3_case_study.md](segrev3_case_study.md).

## Troubleshooting

- **`pygraphviz` fails to build** — the system Graphviz headers are missing (see Setup).
- **A model scores at chance** — it was loaded without `normalization_type="LNPre"`. Load models only
  through `interp_tools/model_loading.py` (`load_segrev3`) or `model_registry.load_model`.
- **"checkpoint for model … not found"** — the model has no trained checkpoint; run tier C first.
- **"no circuit … for model …"** — only declared circuits can be analysed; see
  [reference/declaring_circuits.md](reference/declaring_circuits.md).
- **On a GPU, a stage refuses to start for lack of memory** — the memory-headroom check
  (`SEGREV3_MIN_FREE_GIB`, default 24 GiB of free system memory; ≤ 0 disables) applies to CUDA runs
  only.
