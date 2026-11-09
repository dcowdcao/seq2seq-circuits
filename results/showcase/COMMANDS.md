# Commands and CPU runtimes

Every command below was run from the repository root by
`python results/showcase/reproduce.py --full`, in this order, on CPU only, with
`PYTHONPATH` set to the pipeline's import directories, `SEGREV3_DEVICE=cpu` and
`OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=2` (the smallest stage splits
that thread budget across its evaluation children). Wall times are as measured on that run (a shared
16-core machine, so expect variation); they are read from each example's `runtimes.json`.

Before discovery, `--full` deletes the examples' generated artifacts (all gitignored) under
`pipeline/circuits_data/`, `pipeline/probes_data/` and `pipeline/task_router/`, so the
candidate pool is exactly what these commands produce. `train` is a gate check: it finds the
checkpoint, confirms its recorded held-out sequence accuracy clears the gate, and
skips training.

Selection runs `pipeline/find_circuits/evaluate_circuit.py --select`, which scores every
EAP-IG candidate of the circuit with the class's faithfulness definition and keeps the
candidate above the class floor that maximizes (1 − nodes/full)·(1 − edges/full).
Each selected circuit is then re-evaluated with `--seed 1` (fresh evaluation rows and fresh
mean-reference rows, everything else identical) into `heldout_seed1.json`, because the
selection picks the winner on the seed-0 rows it reports. `reproduce.py --verify` re-runs
the evaluator on each stored `circuit.json`, `base_circuit.json` and `heldout_seed1.json`
(the `command` field inside each) and checks that nodes, edges, counts and pass flags match
exactly and every float field within 1e-06.

## DomainCircuit: `schmitt_id__moe8` / `schmitt_noband`

| step | wall time |
|---|---|
| discovery: schmitt_noband | 46 min 21 s |
| selection: evaluate: schmitt_noband.select | 4 min 29 s |
| heldout: held-out re-evaluation (seed 1) | 1 min 20 s |
| **total** | **52 min 10 s** |

```bash
python pipeline/run_model_pipeline.py --local --device cpu --model schmitt_id__moe8 --only-circuit schmitt_noband --stages data train eap smallest audit --smallest-target 0.9
python pipeline/find_circuits/evaluate_circuit.py --model schmitt_id__moe8 --circuit schmitt_noband --select --device cpu --seed 0 --threads 4 --out pipeline/circuits_data/_showcase/01_domain_schmitt_id__moe8.schmitt_noband.select.json
python pipeline/find_circuits/evaluate_circuit.py --model schmitt_id__moe8 --circuit schmitt_noband --candidate results/showcase/01_domain_schmitt_id__moe8/circuit.json --device cpu --seed 1 --threads 4 --out pipeline/circuits_data/_showcase/01_domain_schmitt_id__moe8.heldout.json
```

## ContrastCircuit: `mexhat_id__moe8` / `mexhat_fill`

| step | wall time |
|---|---|
| discovery: mexhat_fill: data + EAP-IG | 3 min 14 s |
| discovery: mexhat_full (base) | 31 min 47 s |
| discovery: mexhat_fill: smallest | 35 min 50 s |
| selection: evaluate: mexhat_full.select | 3 min 34 s |
| selection: evaluate: mexhat_fill.select | 5 min 06 s |
| heldout: held-out re-evaluation (seed 1) | 1 min 55 s |
| **total** | **81 min 26 s** |

```bash
python pipeline/run_model_pipeline.py --local --device cpu --model mexhat_id__moe8 --only-circuit mexhat_fill --stages data train eap
python pipeline/run_model_pipeline.py --local --device cpu --model mexhat_id__moe8 --only-circuit mexhat_full --stages data eap smallest audit --smallest-target 0.9
python pipeline/run_model_pipeline.py --local --device cpu --model mexhat_id__moe8 --only-circuit mexhat_fill --stages smallest audit --smallest-target 0.8
python pipeline/find_circuits/evaluate_circuit.py --model mexhat_id__moe8 --circuit mexhat_full --select --device cpu --seed 0 --threads 4 --out pipeline/circuits_data/_showcase/02_contrast_mexhat_id__moe8.mexhat_full.select.json
python pipeline/find_circuits/evaluate_circuit.py --model mexhat_id__moe8 --circuit mexhat_fill --select --base-candidate results/showcase/02_contrast_mexhat_id__moe8/base_circuit.json --device cpu --seed 0 --threads 4 --out pipeline/circuits_data/_showcase/02_contrast_mexhat_id__moe8.mexhat_fill.select.json
python pipeline/find_circuits/evaluate_circuit.py --model mexhat_id__moe8 --circuit mexhat_fill --candidate results/showcase/02_contrast_mexhat_id__moe8/circuit.json --base-candidate results/showcase/02_contrast_mexhat_id__moe8/base_circuit.json --device cpu --seed 1 --threads 4 --out pipeline/circuits_data/_showcase/02_contrast_mexhat_id__moe8.heldout.json
```

## SuppressionCircuit: `mux2.pmstep-id.sr12-id__moe8` / `suppress_t1_over_t0`

| step | wall time |
|---|---|
| discovery: task1 (base): data + EAP-IG | 2 min 53 s |
| discovery: task1 (base): smallest | 4 min 42 s |
| discovery: task0 | 36 min 57 s |
| discovery: suppress_t1_over_t0 (+ router diagnostic) | 33 min 14 s |
| selection: evaluate: task1.select | 2 min 57 s |
| selection: evaluate: suppress_t1_over_t0.select | 3 min 14 s |
| selection: evaluate: suppress_t1_over_t0.select-union | 3 min 17 s |
| heldout: held-out re-evaluation (seed 1) | 1 min 32 s |
| **total** | **88 min 46 s** |

```bash
python pipeline/run_model_pipeline.py --local --device cpu --model mux2.pmstep-id.sr12-id__moe8 --only-circuit task1 --stages data train eap
python pipeline/find_circuits/smallest_faithful_circuit.py --model mux2.pmstep-id.sr12-id__moe8 --circuit task1 --target 0.9 --jobs 4
python pipeline/run_model_pipeline.py --local --device cpu --model mux2.pmstep-id.sr12-id__moe8 --only-circuit task0 --stages data eap smallest audit --smallest-target 0.9
python pipeline/run_model_pipeline.py --local --device cpu --model mux2.pmstep-id.sr12-id__moe8 --only-circuit suppress_t1_over_t0 --stages data eap smallest audit --smallest-target 0.8
python pipeline/find_circuits/evaluate_circuit.py --model mux2.pmstep-id.sr12-id__moe8 --circuit task1 --select --device cpu --seed 0 --threads 4 --out pipeline/circuits_data/_showcase/03_suppression_mux2.pmstep-id.sr12-id__moe8.task1.select.json
python pipeline/find_circuits/evaluate_circuit.py --model mux2.pmstep-id.sr12-id__moe8 --circuit suppress_t1_over_t0 --select --base-candidate results/showcase/03_suppression_mux2.pmstep-id.sr12-id__moe8/base_circuit.json --device cpu --seed 0 --threads 4 --out pipeline/circuits_data/_showcase/03_suppression_mux2.pmstep-id.sr12-id__moe8.suppress_t1_over_t0.select.json
python pipeline/find_circuits/evaluate_circuit.py --model mux2.pmstep-id.sr12-id__moe8 --circuit suppress_t1_over_t0 --select --base-candidate results/showcase/03_suppression_mux2.pmstep-id.sr12-id__moe8/base_circuit.json --union-base --device cpu --seed 0 --threads 4 --out pipeline/circuits_data/_showcase/03_suppression_mux2.pmstep-id.sr12-id__moe8.suppress_t1_over_t0.select-union.json
python pipeline/find_circuits/evaluate_circuit.py --model mux2.pmstep-id.sr12-id__moe8 --circuit suppress_t1_over_t0 --candidate results/showcase/03_suppression_mux2.pmstep-id.sr12-id__moe8/circuit.json --base-candidate results/showcase/03_suppression_mux2.pmstep-id.sr12-id__moe8/base_circuit.json --device cpu --seed 1 --threads 4 --out pipeline/circuits_data/_showcase/03_suppression_mux2.pmstep-id.sr12-id__moe8.heldout.json
```

## ProbeCircuit: `graviclump_id__moe4` / `graviclump_mean`

| step | wall time |
|---|---|
| discovery: probes (site scan, training, validation) | 1 min 32 s |
| discovery: pair pool graviclump_edit | 14 s |
| discovery: probe EAP-IG on graviclump_full pairs | 1 min 11 s |
| discovery: probe EAP-IG on graviclump_full pairs, explicit rungs 17 20 23 26 30 34 38 43 | 1 min 20 s |
| discovery: probe EAP-IG on graviclump_edit pairs | 1 min 11 s |
| discovery: probe EAP-IG on graviclump_edit pairs, explicit rungs 17 20 23 26 30 34 38 43 | 1 min 19 s |
| selection: evaluate: graviclump_mean.select | 2 min 24 s |
| heldout: held-out re-evaluation (seed 1) | 40 s |
| **total** | **9 min 51 s** |

Besides the canonical frac ladder, probe-metric EAP-IG also runs the explicit rungs 17, 20, 23, 26, 30, 34, 38, 43 on each pair pool: only 43 edges can reach this probe, and the 4-rung frac ladder over 43 edges is too coarse.

```bash
python pipeline/run_model_pipeline.py --local --device cpu --model graviclump_id__moe4 --only-circuit graviclump_full --stages data train probes
python pipeline/run_model_pipeline.py --local --device cpu --model graviclump_id__moe4 --only-circuit graviclump_edit --stages data
python pipeline/find_circuits/eap.py --model graviclump_id__moe4 --topn-regime frac --circuit graviclump_full --probe graviclump_mean --probe-readout position --device cpu
python pipeline/find_circuits/eap.py --model graviclump_id__moe4 --topns 17 20 23 26 30 34 38 43 --circuit graviclump_full --probe graviclump_mean --probe-readout position --device cpu
python pipeline/find_circuits/eap.py --model graviclump_id__moe4 --topn-regime frac --circuit graviclump_edit --probe graviclump_mean --probe-readout position --device cpu
python pipeline/find_circuits/eap.py --model graviclump_id__moe4 --topns 17 20 23 26 30 34 38 43 --circuit graviclump_edit --probe graviclump_mean --probe-readout position --device cpu
python pipeline/find_circuits/evaluate_circuit.py --model graviclump_id__moe4 --circuit graviclump_mean --select --probe-layer 0 --probe-readout position --device cpu --seed 0 --threads 4 --out pipeline/circuits_data/_showcase/04_probe_graviclump_id__moe4.graviclump_mean.select.json
python pipeline/find_circuits/evaluate_circuit.py --model graviclump_id__moe4 --circuit graviclump_mean --candidate results/showcase/04_probe_graviclump_id__moe4/circuit.json --probes-root results/showcase/04_probe_graviclump_id__moe4/probes --probe-layer 0 --probe-readout position --device cpu --seed 1 --threads 4 --out pipeline/circuits_data/_showcase/04_probe_graviclump_id__moe4.heldout.json
```

## BridgeCircuit: `psum_novel_id__moe8` / `bridge1_novel_id`

| step | wall time |
|---|---|
| discovery: probes (site scan, training, validation) | 1 min 47 s |
| discovery: pair pool psum_novel_fresh | 14 s |
| discovery: probe EAP-IG on psum_novel_whole pairs | 3 min 21 s |
| discovery: probe EAP-IG on psum_novel_fresh pairs | 4 min 17 s |
| selection: evaluate: bridge1_novel_id.select | 2 min 44 s |
| heldout: held-out re-evaluation (seed 1) | 1 min 34 s |
| **total** | **13 min 57 s** |

```bash
python pipeline/run_model_pipeline.py --local --device cpu --model psum_novel_id__moe8 --only-circuit psum_novel_whole --stages data train probes
python pipeline/run_model_pipeline.py --local --device cpu --model psum_novel_id__moe8 --only-circuit psum_novel_fresh --stages data
python pipeline/find_circuits/eap.py --model psum_novel_id__moe8 --topn-regime frac --circuit psum_novel_whole --probe bridge1_novel_id --probe-readout position --device cpu
python pipeline/find_circuits/eap.py --model psum_novel_id__moe8 --topn-regime frac --circuit psum_novel_fresh --probe bridge1_novel_id --probe-readout position --device cpu
python pipeline/find_circuits/evaluate_circuit.py --model psum_novel_id__moe8 --circuit bridge1_novel_id --select --probe-layer 2 --probe-readout position --device cpu --seed 0 --threads 4 --out pipeline/circuits_data/_showcase/05_bridge_psum_novel_id__moe8.bridge1_novel_id.select.json
python pipeline/find_circuits/evaluate_circuit.py --model psum_novel_id__moe8 --circuit bridge1_novel_id --candidate results/showcase/05_bridge_psum_novel_id__moe8/circuit.json --probes-root results/showcase/05_bridge_psum_novel_id__moe8/probes --probe-layer 2 --probe-readout position --device cpu --seed 1 --threads 4 --out pipeline/circuits_data/_showcase/05_bridge_psum_novel_id__moe8.heldout.json
```

Sum over the five examples: 246 min 10 s of wall time.

## Verification

`python results/showcase/reproduce.py --verify` runs, for each example, the `command` stored in its `circuit.json`, `base_circuit.json` and `heldout_seed1.json`, i.e.:

```bash
python pipeline/find_circuits/evaluate_circuit.py --model schmitt_id__moe8 --circuit schmitt_noband --candidate results/showcase/01_domain_schmitt_id__moe8/circuit.json --device cpu --seed 0 --threads 4 --out <out.json>
python pipeline/find_circuits/evaluate_circuit.py --model schmitt_id__moe8 --circuit schmitt_noband --candidate results/showcase/01_domain_schmitt_id__moe8/circuit.json --device cpu --seed 1 --threads 4 --out <out.json>
python pipeline/find_circuits/evaluate_circuit.py --model mexhat_id__moe8 --circuit mexhat_full --candidate results/showcase/02_contrast_mexhat_id__moe8/base_circuit.json --device cpu --seed 0 --threads 4 --out <out.json>
python pipeline/find_circuits/evaluate_circuit.py --model mexhat_id__moe8 --circuit mexhat_fill --candidate results/showcase/02_contrast_mexhat_id__moe8/circuit.json --base-candidate results/showcase/02_contrast_mexhat_id__moe8/base_circuit.json --device cpu --seed 0 --threads 4 --out <out.json>
python pipeline/find_circuits/evaluate_circuit.py --model mexhat_id__moe8 --circuit mexhat_fill --candidate results/showcase/02_contrast_mexhat_id__moe8/circuit.json --base-candidate results/showcase/02_contrast_mexhat_id__moe8/base_circuit.json --device cpu --seed 1 --threads 4 --out <out.json>
python pipeline/find_circuits/evaluate_circuit.py --model mux2.pmstep-id.sr12-id__moe8 --circuit task1 --candidate results/showcase/03_suppression_mux2.pmstep-id.sr12-id__moe8/base_circuit.json --device cpu --seed 0 --threads 4 --out <out.json>
python pipeline/find_circuits/evaluate_circuit.py --model mux2.pmstep-id.sr12-id__moe8 --circuit suppress_t1_over_t0 --candidate results/showcase/03_suppression_mux2.pmstep-id.sr12-id__moe8/circuit.json --base-candidate results/showcase/03_suppression_mux2.pmstep-id.sr12-id__moe8/base_circuit.json --device cpu --seed 0 --threads 4 --out <out.json>
python pipeline/find_circuits/evaluate_circuit.py --model mux2.pmstep-id.sr12-id__moe8 --circuit suppress_t1_over_t0 --candidate results/showcase/03_suppression_mux2.pmstep-id.sr12-id__moe8/circuit.json --base-candidate results/showcase/03_suppression_mux2.pmstep-id.sr12-id__moe8/base_circuit.json --device cpu --seed 1 --threads 4 --out <out.json>
python pipeline/find_circuits/evaluate_circuit.py --model graviclump_id__moe4 --circuit graviclump_mean --candidate results/showcase/04_probe_graviclump_id__moe4/circuit.json --probes-root results/showcase/04_probe_graviclump_id__moe4/probes --probe-layer 0 --probe-readout position --device cpu --seed 0 --threads 4 --out <out.json>
python pipeline/find_circuits/evaluate_circuit.py --model graviclump_id__moe4 --circuit graviclump_mean --candidate results/showcase/04_probe_graviclump_id__moe4/circuit.json --probes-root results/showcase/04_probe_graviclump_id__moe4/probes --probe-layer 0 --probe-readout position --device cpu --seed 1 --threads 4 --out <out.json>
python pipeline/find_circuits/evaluate_circuit.py --model psum_novel_id__moe8 --circuit bridge1_novel_id --candidate results/showcase/05_bridge_psum_novel_id__moe8/circuit.json --probes-root results/showcase/05_bridge_psum_novel_id__moe8/probes --probe-layer 2 --probe-readout position --device cpu --seed 0 --threads 4 --out <out.json>
python pipeline/find_circuits/evaluate_circuit.py --model psum_novel_id__moe8 --circuit bridge1_novel_id --candidate results/showcase/05_bridge_psum_novel_id__moe8/circuit.json --probes-root results/showcase/05_bridge_psum_novel_id__moe8/probes --probe-layer 2 --probe-readout position --device cpu --seed 1 --threads 4 --out <out.json>
```
