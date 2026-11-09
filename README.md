# Circuits in toy seq2seq transformers

Small attention-based transformers, each trained from scratch to implement one algorithm on
10-digit sequences, and the circuits found inside them. Every algorithm is an expression in a small
DSL — physics-like dynamics, electrical-engineering primitives, computer-science and string
algorithms, arithmetic and group products, compositions and multi-task models — trained at up to
eight architectures (attention-only, MLP and mixture-of-experts). For each model, sub-circuits are
declared by hand as one of five circuit classes, candidates are found with EAP-IG (Edge Attribution
Patching with Integrated Gradients; Hanna, Pezzelle & Belinkov, 2024, *Have Faith in
Faithfulness*), and the smallest faithful candidate is reported with its components, size and
faithfulness. Every showcase circuit and 3,329 of the 3,341 faithful zoo circuits are EAP-IG
candidates; the other 12, DomainCircuits from the early zoo run, came from ACDC (Automatic Circuit
DisCovery; Conmy et al., 2023) and Edge Pruning (Bhaskar et al., 2024)
([`results/zoo/circuits_index.csv`](results/zoo/circuits_index.csv); references in
[docs/circuit_finding.md](docs/circuit_finding.md#references)). The repository also contains the
original single-model case study (`segrev3`) in which manual interpretability, ACDC and EAP-IG were
compared on one model and every circuit was validated. There is no ground-truth circuit: a
reported circuit is the smallest faithful one the methods found, not a provably minimal one.

## The five circuit classes

| class | isolates | clean / corrupt | faithfulness *a* | floor | showcase |
|---|---|---|---|---|---|
| **DomainCircuit** | the simpler algorithm the model collapses to on a restricted input domain | both drawn from the conditioned domain | whole-sequence accuracy on the domain | 0.90 | Schmitt trigger without its memory — `schmitt_id__moe8` / `schmitt_noband` |
| **ContrastCircuit** | the detector for one predicate | one deliberate edit toggles the predicate | a_clean × a_corrupt under interchange patching | 0.80 | dip detector — `mexhat_id__moe8` / `mexhat_fill` |
| **SuppressionCircuit** | in a multi-task (mux) model, what keeps the non-cued task's answer out | cue A + payload / cue B + the same payload | a_clean × a_corrupt | 0.80 | latch vs. parity step — `mux2.pmstep-id.sr12-id__moe8` |
| **ProbeCircuit** | an internal variable the output never prints | for evaluation, corrupt = the single-digit edit that makes the probe's class most decodable (discovery uses separate pair pools) | normalized probe accuracy, clean × corrupt | 0.80 | gravity: the input's average — `graviclump_id__moe4` / `graviclump_mean` |
| **BridgeCircuit** | the intermediate value B(x) inside a composition A(B(x)) | as ProbeCircuit | as ProbeCircuit | 0.80 | "is this digit new?" inside psum∘novel — `psum_novel_id__moe8` / `bridge_variable('psum_novel_id', 1)` |

Definitions: [docs/circuits.md](docs/circuits.md) and [docs/circuit_classes.md](docs/circuit_classes.md).
Each showcase example — `circuit.json`, figure and README — is in
[`results/showcase/`](results/showcase/); numbers are in [docs/results.md](docs/results.md).

## The pipeline

```
 algorithm DSL id                 pipeline/models/algorithms.py       e.g. psum_novel_id = PrefixSumMod(NoveltyIndicator(Id))
        │  (+ architecture suffix __small … __moe8)
        ▼
 train seq2seq model              pipeline/models/train_models/       [x | y] teacher forcing, gate: seq-acc ≥ 0.95
        │
        ▼
 declare circuits                 pipeline/models/circuits.py         DomainCircuit / ContrastCircuit / SuppressionContrastCircuit
        │                         pipeline/probes/variables.py        ProbeVariable / bridge_variable
        ▼
 generate data                    pipeline/models/gen_data/           clean / corrupt pools from the circuit's condition or edit
        │
        ▼
 EAP-IG candidates                pipeline/find_circuits/eap.py       top-n edges by |attribution|, a ladder of sizes
        │
        ▼
 faithfulness + selection         pipeline/find_circuits/             evaluate_circuit.py (per-class a, floors)
        │                                                             smallest_faithful_circuit.py (zoo-wide selection)
        ▼
 circuit JSON                     nodes, edges, size, faithfulness
```

One command runs a model through it: `python pipeline/run_model_pipeline.py --model <id> --stages all`
(local by default; `--modal` selects an optional cloud backend).

## Quick start

```bash
# setup: system Graphviz for pygraphviz (macOS: brew install graphviz), then the pinned packages
sudo apt-get install graphviz libgraphviz-dev pkg-config
python3 -m venv .venv && source .venv/bin/activate
pip install torch==2.12.0 --index-url https://download.pytorch.org/whl/cpu   # CPU-only machines
pip install -r requirements.txt

# the trained checkpoints are not included: train the showcase models first
# (docs/reproducing.md, tier C), then rerun discovery and selection on them (CPU, ~4 h)
python results/showcase/reproduce.py --full

# segrev3 case study: model-load check, then the stage plan of the full study
python interp_tools/model_loading.py
python circuit_discovery/main.py all --dry-run
```

Run times, per-example commands and the full reproduction ladder (retraining, zoo statistics):
[docs/reproducing.md](docs/reproducing.md) and
[`results/showcase/COMMANDS.md`](results/showcase/COMMANDS.md).

## Repository layout

```
README.md  LICENSE  requirements.txt
docs/                      documentation — start at docs/README.md
results/
  showcase/                one example per circuit class + reproduce.py
  zoo/                     compact index of the zoo-wide runs + stats scripts
pipeline/                  multi-model pipeline
  run_model_pipeline.py      per-model, per-circuit stage orchestrator
  build_models.py            batch driver over models/models.txt
  models/
    algorithms.py            the DSL: operators, id encoding, task functions
    circuits.py              circuit classes and every declared circuit
    conditions.py            input-domain conditions (predicate + exact sampler)
    model_registry.py        model specs, loading, datasets, artifact paths
    models.txt  circuits.txt the run-list and the generated circuit catalog
    train_models/            trainer, per-architecture recipe (hparams.py), promotion gate
    gen_data/                per-circuit input-pool generator
    trained_models_pt/       checkpoints, written by training (not included)
  find_circuits/             eap.py, acdc.py, edge_pruning.py, smallest_faithful_circuit.py, evaluate_circuit.py
  probes/                    variables.py (ProbeVariable, bridge_variable), train_probes.py, validate_probe.py
  interp_tools/              per-circuit patching / DLA / QK-OV CLIs
  task_router/               routing diagnostic for mux models
  edge_signs/  repo_tools/   edge-sign census; catalog, taxonomy and coverage tools
interp_tools/              shared library: model_loading.py (the loader: LNPre = parameter-free pre-LayerNorm,
                           TransformerLens normalization_type="LNPre"), patching_dla.py, edge_tools.py, …
circuit_discovery/         segrev3 case study: main.py orchestrator, manual_circuit.py, acdc.py, eap.py
data_gen/                  canon.py (canonical sweeps), segrev3 data generation
metrics/  validation/  visualizations/  datasets/   segrev3 metrics, validation, figures, datasets
segrev3/segrev3_id_checkpoint.pt                    the case-study model
github_repos/              vendored ACDC and EAP-IG (pinned commits, VENDORED_SOURCES.md)
```

## Documentation

| | |
|---|---|
| [docs/circuits.md](docs/circuits.md) | circuits, ablation, faithfulness, floors, selection |
| [docs/circuit_classes.md](docs/circuit_classes.md) | the five classes in detail |
| [docs/models.md](docs/models.md) | task format, DSL, architectures, training, limitations |
| [docs/circuit_finding.md](docs/circuit_finding.md) | interpretability techniques, EAP-IG, ACDC, Edge Pruning, probes |
| [docs/brainstorming.md](docs/brainstorming.md) | how the algorithms and circuit classes were brainstormed |
| [docs/segrev3_case_study.md](docs/segrev3_case_study.md) | the original case study: methods and results |
| [docs/reproducing.md](docs/reproducing.md) | reproducing everything |
| [docs/results.md](docs/results.md) | key results |

## License

[MIT](LICENSE). The vendored repositories under `github_repos/` keep their own licenses;
`pipeline/find_circuits/l0.py` is vendored from Edge Pruning (MIT).
