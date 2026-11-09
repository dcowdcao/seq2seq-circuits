# Documentation

Start with the repository [README](../README.md). The pages below are ordered by the questions
they answer.

## Core

| page | answers |
|---|---|
| [circuits.md](circuits.md) | What is a circuit here? The graph, edge-level per-position mean ablation, interchange patching, faithfulness and floors per class, the selection criterion, and how the two scoring scripts differ |
| [circuit_classes.md](circuit_classes.md) | The five circuit classes — DomainCircuit, ContrastCircuit, SuppressionCircuit, ProbeCircuit, BridgeCircuit: data, discovery metric, faithfulness, code, examples |
| [models.md](models.md) | The seq2seq setup, the algorithm DSL, the eight architectures, the training recipe and gates, checkpoint provenance, known limitations |
| [circuit_finding.md](circuit_finding.md) | Component-level techniques, EAP-IG / ACDC / Edge Pruning, probes, selection and evaluation, the pipeline end to end |
| [brainstorming.md](brainstorming.md) | How the algorithms and the circuit classes were brainstormed, scored, verified and integrated |
| [reproducing.md](reproducing.md) | Setup and tiered reproduction: verify the showcase, rediscover, retrain, zoo statistics, segrev3 |
| [results.md](results.md) | Key results |

## The segrev3 case study

| page | |
|---|---|
| [segrev3_case_study.md](segrev3_case_study.md) | Methodology (manual interpretability, ACDC, EAP-IG, validation) and validated results for the original single-model study |
| [segrev3_runbook.md](segrev3_runbook.md) | Every command and parameter of the case study |

## Results

| location | |
|---|---|
| [`results/showcase/`](../results/showcase/) | One example per circuit class: `circuit.json`, figure, README; `reproduce.py --verify` / `--full` |
| [`results/zoo/`](../results/zoo/) | Compact index of the zoo-wide runs (`circuits_index.csv`, `probes_index.csv`, `models_index.csv`), `zoo_stats.py`, `build_index.py`, [`stats.md`](../results/zoo/stats.md) |

## Reference

| page | |
|---|---|
| [reference/operators.md](reference/operators.md) | DSL operator table |
| [reference/circuit_types.md](reference/circuit_types.md) | Detailed internals of every circuit class |
| [reference/declaring_circuits.md](reference/declaring_circuits.md) | How a sub-circuit is declared as a `DomainCircuit` (or `ContrastCircuit`) subclass |
| [reference/contrast_pair_catalog.md](reference/contrast_pair_catalog.md) | Per-model contrast-pair designs with verified masks |
| [reference/probe_variable_catalog.md](reference/probe_variable_catalog.md) | Per-model internal variables for probes |
| [reference/conditional_models.md](reference/conditional_models.md) | Operators with a branch that a domain restriction can isolate |
| [reference/variation_axes.md](reference/variation_axes.md) | Axes of variation for circuit discovery, with literature anchors |
| [reference/variation_matrix.md](reference/variation_matrix.md) | Circuit axes × model axes: predictions and observations |
| [reference/circuit_notes_by_architecture.md](reference/circuit_notes_by_architecture.md) | Per-architecture notes on the circuits found |
| [reference/circuit_types_across_architectures.md](reference/circuit_types_across_architectures.md) | Which circuit classes exist at which architecture |
| [reference/stacked_training.md](reference/stacked_training.md) | Vectorized training of many models at once |
| [reference/design.md](reference/design.md) | Design notes for the multi-model pipeline |
| [reference/file_map.md](reference/file_map.md) | What every file in the pipeline does |

## Brainstorming appendices

| page | |
|---|---|
| [brainstorming/brainstormed_models.md](brainstorming/brainstormed_models.md) | Master table of brainstormed algorithms and their circuits |
| [brainstorming/operator_glossary.md](brainstorming/operator_glossary.md) | Plain-language description of every brainstormed operator |
