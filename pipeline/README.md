# pipeline: the model zoo and its circuits

Code that trains small transformers on symbolic seq2seq digit tasks and finds circuits in them.
Every model name is a DSL `algorithm_id` (for example `schmitt_id__moe8`), resolved by
`models/algorithms.py`: the name fixes the task, the output length, the context length, the answer
positions and the architecture. Circuits are declared by hand as Python classes and then found
automatically with EAP-IG (ACDC and Edge Pruning are optional finders).

Concepts and results live in [`../docs/`](../docs/README.md): what a circuit is and how
faithfulness is measured ([circuits.md](../docs/circuits.md)), the five circuit classes
([circuit_classes.md](../docs/circuit_classes.md)), architectures and training
([models.md](../docs/models.md)), the finders ([circuit_finding.md](../docs/circuit_finding.md)) and
how to reproduce the showcase ([reproducing.md](../docs/reproducing.md)). Implementation-level
reference: [`../docs/reference/`](../docs/reference/).

## Entry points

Run from the repository root with the requirements installed (`pip install -r requirements.txt`).
Everything runs locally by default; `--device cpu|cuda|auto` picks the device.

```bash
# one model, the default stages (data train eap smallest audit)
python pipeline/run_model_pipeline.py --model schmitt_id__moe8 --stages all
python pipeline/run_model_pipeline.py --model schmitt_id__moe8 --stages all --dry-run   # print commands only
python pipeline/run_model_pipeline.py --model schmitt_id__moe8 --stages smallest audit --only-circuit schmitt_noband

# a batch over models/models.txt
python pipeline/build_models.py --list-models
python pipeline/build_models.py --only schmitt_id__moe8,mexhat_id__moe8 --dry-run
```

`--modal` is an optional backend that runs the GPU stages in Modal containers
(`modal deploy pipeline/modal_pipeline_app.py` once; remote runs execute the deployed image).
`--smoke` trains a throwaway tiny checkpoint under `pipeline/.smoke/` to exercise the stages.

## Stages (`run_model_pipeline.py --stages …`)

| Stage | Script | Output |
|---|---|---|
| `data` | `models/gen_data/gen_circuit_data.py` | one input pool per declared circuit: `models/gen_data/<model>/<circuit>/inputs.json` |
| `train` | `models/train_models/run_training_job.sh` → `train.py` (recipe in `hparams.py`, promotion in `promote.py`) | `models/trained_models_pt/<model>.pt`; skipped when an existing checkpoint passes the 0.95 sequence-accuracy gate |
| `eap` | `find_circuits/eap.py` | EAP-IG candidates `circuits_data/<model>/<circuit>/circuits/eap_top<n>/` |
| `acdc`, `edge_pruning` | `find_circuits/acdc.py`, `find_circuits/edge_pruning.py` | more candidates (only when named) |
| `smallest` | `find_circuits/smallest_faithful_circuit.py` | the selected circuit: `circuits_data/<model>/<circuit>/smallest_circuit.json` (+ `.pt`, PNG under `circuits_figures/`) |
| `audit` | in-process | checks that every expected artifact exists |
| `probes` | `probes/train_probes.py`, `probes/validate_probe.py` | linear probes for ProbeCircuit / BridgeCircuit variables: `probes_data/<model>/<variable>/` (only when named) |
| `patching`, `edges` | `interp_tools/patching.py`, `interp_tools/edges.py` | per-circuit patching / DLA and weight-composition analyses (only when named) |

`all` expands to `data train eap smallest audit`. Every per-circuit stage runs once for each circuit
declared for the model. Selection rules of the `smallest` stage:
[`../docs/reference/circuit_types.md`](../docs/reference/circuit_types.md); the faithfulness
definitions and floors used for the showcase circuits: [`../docs/circuits.md`](../docs/circuits.md).

Other entry points: `task_router/router_diagnostic.py` (routing diagnostic for multi-task mux
models, needed by SuppressionCircuits), `edge_signs/edge_sign_census.py` (sign census of circuit
edges), `repo_tools/` (catalog and index generators, e.g. `make_circuits_txt.py`,
`make_taxonomy.py`, `arch_coverage.py`).

## Directory map

```
pipeline/
  run_model_pipeline.py     per-model, per-circuit stage orchestrator (local by default, --modal optional)
  build_models.py           batch driver over models/models.txt
  modal_pipeline_app.py     optional Modal backend
  models/
    algorithms.py           the DSL: operators, id encode/decode, task_fn
    circuits.py             circuit declarations (DomainCircuit, ContrastCircuit, SuppressionContrastCircuit)
    conditions.py           input-domain conditions (predicate + exact-uniform sampler)
    model_registry.py       model specs, loading, data-pool windows, every artifact path
    models.txt, circuits.txt  run list and generated circuit catalog
    human_circuit_finding.json  hand-written circuit requests
    taxonomy.json           generated per-circuit index
    trained_models_pt/      checkpoints (<model>.pt)
    train_models/           training recipe, trainers (train.py, train_stacked.py, train_sharded.py), promotion gate
    gen_data/               per-circuit input pools
  find_circuits/            eap.py, acdc.py, edge_pruning.py, smallest_faithful_circuit.py, evaluate_circuit.py, suppression_selfcheck.py
  probes/                   variables.py (ProbeVariable / bridge_variable), train_probes.py, validate_probe.py
  task_router/              router_diagnostic.py
  interp_tools/             patching.py, edges.py (shared compute lives in ../interp_tools/)
  edge_signs/               edge_sign_census.py
  repo_tools/               catalog / index generators
  circuits_data/, circuits_figures/, probes_data/   generated artifacts (not committed)
```

Shared modules at the repository root (`../interp_tools/`, `../data_gen/` with the canonical
constants in `canon.py`, `../metrics/`) are put on `sys.path` by `models/model_registry.py`.

## Further reading

- [`../docs/reference/declaring_circuits.md`](../docs/reference/declaring_circuits.md): how to declare a new circuit.
- [`../docs/reference/design.md`](../docs/reference/design.md): why the pipeline is built this way.
- [`../docs/reference/file_map.md`](../docs/reference/file_map.md): annotated inventory of this directory.
- [`../docs/reference/operators.md`](../docs/reference/operators.md): the DSL operators.
- [`../results/zoo/stats.md`](../results/zoo/stats.md): zoo-wide statistics.
