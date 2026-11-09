# segrev3 runbook

Every runnable command of the [segrev3 case study](segrev3_case_study.md), with its flags and the
command that regenerates each committed number. All commands run from the repository root; every
script sets up its own `sys.path`. Flags below were checked against each script's `argparse`.

---

## 0. Setup

```bash
sudo apt-get install graphviz libgraphviz-dev pkg-config   # pygraphviz; macOS: brew install graphviz
python -m venv .venv
.venv/bin/pip install torch==2.12.0 --index-url https://download.pytorch.org/whl/cpu   # CPU-only machines
.venv/bin/pip install -r requirements.txt
source .venv/bin/activate            # then plain `python` below; `deactivate` when done
export OMP_NUM_THREADS=4             # BLAS threads per process (the committed run's setting)
```

Set `OMP_NUM_THREADS` before running anything below. Unset, every process uses all cores, and
`acdc.py` runs up to 4 such workers at once. The results are identical, but on a 16-core machine
the ACDC sweep runs several times slower.

- `circuit_discovery/main.py` re-executes itself under `./.venv` when launched with a system Python
  that is not inside any virtual environment; the standalone scripts need the environment
  activated (or call them as `.venv/bin/python <script>`).
- **Device.** Every stage resolves its device through `interp_tools/model_loading.resolve_device`:
  `SEGREV3_DEVICE=auto|cuda|cpu` (exported for all stages by `main.py --device …`). Everything
  runs on CPU; a GPU only makes ACDC and the figure stages faster.
- **Checkpoint.** `SEGREV3_CKPT` (exported by `main.py --model …`) overrides the default
  `segrev3/segrev3_id_checkpoint.pt`.
- **Circuit PNGs.** ACDC renders `circuit.png` through pygraphviz; when pygraphviz is not importable
  the PNG is skipped and `summary.json`, `circuit.gv` and `circuit.pth` are still written. EAP-IG
  graphs can be rendered without pygraphviz from the saved `.pt`:
  `python visualizations/render_circuit_png.py visualizations/eap/eap_top87/eap_top87.pt`
  (needs the graphviz `dot` binary).
- **Canonical parameters** (sweeps, dataset sizes, schemes) live in `data_gen/canon.py`, imported by
  the orchestrator and every runner.
- **Corruption schemes**, named for which positions get replaced: `random` (independent random
  input), `replace_threes_to_nonthrees` (n clean `3`s → non-`3`s: deletes boundaries),
  `replace_non_threes_to_threes` (n clean non-`3`s → `3`s: adds boundaries).

---

## 1. Reproduce the committed numbers

Every committed segrev3 artifact was produced on CPU by the commands below, run in this order from
the repository root (a step reads what earlier steps wrote). The committed `datasets/` files are the
exact inputs of the results: every runner loads an existing store file as-is, and step 1 rewrites
them byte for byte.

| step | case-study result | command | output |
|---|---|---|---|
| — | model accuracy (§1.1) | `python interp_tools/model_loading.py` | printed token / sequence accuracy |
| 1 | inputs (§2.1) | `python circuit_discovery/main.py data` | `datasets/{manual,acdc,eap,edge}/` |
| 2 | realized attention on one input | `python visualizations/manual_circuit_analysis/vis_edge.py --data datasets/single/edge_k2.json` | `visualizations/manual_circuit_analysis/figs_data/figs/edge/edge_k2/` |
| 3 | activation screens (§5.5), weight-based and attention evidence (§5.6), full-model minimality (§0, §5.4) | `python circuit_discovery/manual_circuit.py --skip-data` | `visualizations/manual_circuit_analysis/figs_data/data/<scheme>/patching_dla_results.json`, `.../figs_data/data/edge_results.json`, `validation/figs_data/data/manual/rough_validation.json` |
| 4 | heuristic manual circuit (§5.3) | `python interp_tools/circuit_summary.py`, then the snippet in §1.1 | `visualizations/rough_circuit.{json,md,png}` |
| 5 | ACDC sweep (§5.1) | `python circuit_discovery/acdc.py --thresholds 0.2 0.1 0.055 0.02 -n 20 --device cpu` | `visualizations/acdc/acdc_thresh_<tau>/summary.json`, `visualizations/acdc/acdc_all_thresholds_summary.json` |
| 6 | EAP-IG sweep (§5.2) | `python circuit_discovery/eap.py --topns 50 75 87 100 200 400 800` | `visualizations/eap/eap_top<n>/`, `visualizations/eap/eap_all_topn_summary.json` (`per_topn`) |
| 7 | validation of both sweeps (§5.1, §5.2, §5.4) | `python validation/validation.py --circuit acdc`, then `python validation/validation.py --circuit eap` | `validation/figs_data/data/<method>/<circuit>/<circuit>_validation.json` |
| 8 | faithfulness by number of 3s (§5.7) | `python validation/n_threes_faithfulness.py`, then `python validation/n_threes_faithfulness.py --circuits eap_top87 --tag top87` | `validation/figs_data/data/{acdc,eap}/faithfulness_by_n_threes_*.json` |
| 9 | faithfulness curves (§0, §6) | the snippet in §1.2 | `validation/figs_data/data/manual/faithfulness_curve.json`, `validation/figs_data/data/{acdc,eap}/<method>_faithfulness_curve.json` |
| 10 | smallest circuit at the 0.80 target (§0, §5.1, §6) | `python interp_tools/min_faithful_circuit.py --refine-rounds 5` | `visualizations/min_faithful_circuit_search.json`, `validation/figs_data/figs/min_faithful_circuit_search.png` |

`python circuit_discovery/main.py all` runs steps 1 (without the `edge` sets, which it loads as
committed), 3, 5, 6 and 7, followed by circuit-scoped figures (gitignored) and `figures.zip`.

Order constraints:
- `vis_edge.py` rewrites `edge_results.json` and `edge_report.md` as a whole on every run. Step 3
  runs it on the eleven `datasets/edge/n_threes_<k>.json` sets, so the committed files hold those;
  step 2 contributes only the `edge_k2/` figures.
- The ACDC combined summary compares each circuit with `visualizations/rough_circuit.json` (step 4).
- Steps 7 and 9 and the no-flag run of step 8 cover every circuit folder on disk. Step 10 adds
  folders for the extra parameters of its search, so run 7–9 first (or delete those folders).

Wall times of the committed run (16-core CPU shared with other jobs, 4 BLAS threads per process):
step 1: 6 s · 2: 17 s · 3: 12 min · 4: 8 s · 5: 13 min · 6: 8 min · 7: 10 s + 63 s · 8: 19 s ·
9: 71 s · 10: 21 min — about 58 min in sequence. Steps 2–4, 5 and 6 are independent of each other
apart from the `rough_circuit.json` read at the end of step 5, and were run concurrently.

Notes:
- `min_faithful_circuit.py` sweeps ACDC τ ∈ {0.01 … 0.5} and EAP-IG top-n ∈ {5 … 800}, reuses
  discovery artifacts already on disk, launches `acdc.py -n 20` / `eap.py` for missing ones, then
  refines the bracket around the target; the committed search used `--refine-rounds 5`. Only the
  circuits of steps 5 and 6 are committed. The search writes the others to their own folders
  (gitignored), lists every circuit it evaluated in its JSON, and restores the two combined sweep
  summaries to their pre-search content. Their ACDC input sets are committed under `datasets/acdc/`.
  Flags: `--target` (0.80), `--taus`, `--topns`, `--eval-n` (512), `--eval-seed` (0),
  `--acdc-jobs` (4), `--force`, `--max-extend-rounds` (4), `--refine-rounds` (2).

### 1.1 Validate the heuristic manual circuit (print only)
`python validation/validation.py --circuit rough` would write to `rough_validation.json`, which by
default holds the full-model validation of the manual pipeline. To print the heuristic circuit's
numbers without overwriting anything:

```bash
python - <<'EOF'
import sys; sys.path.insert(0, "validation")
import validation as V, circuit_loader as CL
from model_loading import load_segrev3
c = CL.load_circuit("manual", "rough")                  # visualizations/rough_circuit.json
res = V.validate(c, model=load_segrev3(device="cpu"), mode="mean")
print({k: round(v, 3) for k, v in res["faithfulness"].items() if isinstance(v, float)})
gaps = sorted(res["completeness"]["per_outside_head"].items(),
              key=lambda kv: -kv[1]["logit_diff_change"])
print([(h, round(g["logit_diff_change"], 2)) for h, g in gaps[:5]])
EOF
```

### 1.2 Faithfulness curves
`validation.plot_faithfulness_curve()` keeps the top-k heads of a whole-head knockout ordering
(every other head mean-ablated) and scores them on the 128-input `datasets/manual/random.json`. It
reads the ordering from the `ranking` field of the JSON it rewrites
(`validation/figs_data/data/manual/faithfulness_curve.json`; the knockout run that produced the
ordering is not shipped). `method_faithfulness_curve` validates every ACDC / EAP-IG circuit folder
on disk and plots the result against τ / top-n. Step 9 of the table:

```bash
python - <<'EOF'
import sys; sys.path.insert(0, "validation")
import validation as V
cur = V.plot_faithfulness_curve()       # manual/faithfulness_curve.{json,png}
print({k: round(cur["recovered_acc"][k - 1], 3) for k in (16, 18, 24, 30)})
V.method_faithfulness_curve("acdc")     # acdc/acdc_faithfulness_curve.{json,png}
V.method_faithfulness_curve("eap")      # eap/eap_faithfulness_curve.{json,png}
EOF
```

---

## 2. The orchestrator: `circuit_discovery/main.py`

Name the stages as positional words; they execute in canonical order, and after discovery the
per-circuit validation → circuit-scoped figures → `figures.zip` tails run automatically. With no
stage it prints the common recipes; `--dry-run` prints the plan without executing.

```
python circuit_discovery/main.py STAGE [STAGE ...] [options]
```

| Stage | What it runs |
|---|---|
| `all` | `data` + `manual` + `acdc` + `eap` + `figs` |
| `data` | `data_gen/generate_datasets.py` for the selected methods' apps |
| `manual` | `circuit_discovery/manual_circuit.py` (data → analyses → full-model validation) |
| `acdc` | `circuit_discovery/acdc.py` |
| `eap` | `circuit_discovery/eap.py` |
| `figs` | circuit-scoped figures for `--circuit` (`manual_circuit.py --circuit`) |
| `validate` | `validation/validation.py` for `--circuit` |
| `single` | single-sequence mode (runs alone; see §3) |
| `zip` | `visualizations/collect_figures.py` |

| Group | Flag | Default | Meaning |
|---|---|---|---|
| global | `-n / --examples <int>` | per-app | examples per generated dataset (`data` and `manual`) |
| global | `--seed <int>` | 0 | base seed, forwarded everywhere |
| global | `--device auto\|cuda\|cpu` | auto | exported to every stage as `SEGREV3_DEVICE` |
| global | `--model <ckpt.pt>` | canonical checkpoint | exported to every stage as `SEGREV3_CKPT` |
| global | `--parallel` | off | run the selected discovery methods concurrently |
| global | `--dry-run` | off | print the plan only |
| tails | `--no-validate`, `--no-figures` (alias `--no-viz`), `--no-collect` | on | opt out of a tail |
| figs / validate | `--circuit <sel>` | all | `all` \| a method (`manual`, `acdc`, `eap`) \| a circuit folder name (`rough`, `acdc_thresh_<tau>`, `eap_top<n>`); also narrows the automatic tails |
| figs / validate | `--scope full_circuit\|ablated\|all` | full_circuit | scoped-figure scope |
| figs / validate | `--scheme <scheme>\|all` | random | contrast scheme for scoped figures / the single pair |
| figs / validate | `--mode mean\|zero\|resample` | mean | validation ablation reference |
| acdc | `--thresholds τ …` | 0.2 0.1 0.055 0.02 | ACDC τ sweep |
| acdc | `--acdc-examples <int>` | 20 | ACDC set size if the set must be generated |
| eap | `--topns n …` | 50 75 87 100 200 400 800 | EAP-IG top-n list |
| eap | `--export-n <int>` | 200 | EAP-IG headline size (also selects the dataset folder `eap_top<n>`) |
| eap | `--ig-steps <int>` | 30 | integrated-gradient steps |
| single | `--n-threes`, `--n-replace`, `--clean`, `--corrupted`, `--edge-input` | | see §3 |

Recipes:

```bash
python circuit_discovery/main.py                                   # print recipes
python circuit_discovery/main.py all --device cpu                 # steps 1, 3, 5, 6, 7 of §1 (+ scoped figures, zip)
python circuit_discovery/main.py all --dry-run                     # show the full plan
python circuit_discovery/main.py data acdc eap --parallel          # regenerate data, both edge methods concurrently
python circuit_discovery/main.py figs --scope all --scheme all     # re-scope every circuit
python circuit_discovery/main.py figs --circuit eap_top87 --scope ablated
python circuit_discovery/main.py validate --circuit all --mode resample
python circuit_discovery/main.py zip                               # rebuild figures.zip
```

---

## 3. Single-sequence mode (the `single` stage)

Runs the manual analyses and circuit-scoped figures on one clean/corrupted pair and one input for
the realized-attention view. ACDC and EAP-IG discovery are not available here and validation is
skipped (it needs batches). The scoped figures default to the heuristic `rough` circuit;
`--circuit <name>` scopes the same pair to another circuit. Datasets land in `datasets/single/`
and their file stem labels every output folder.

| Flag | Default | Meaning |
|---|---|---|
| `--n-threes <k>` | — | exact number of `3`s in the clean (and attention) input; required unless `--clean` is given |
| `--scheme <scheme>` | random | how the corrupted side is built (one scheme, not `all`) |
| `--n-replace <n>` | 1 | replaced positions; needs n ≤ k (`replace_threes_to_nonthrees`) or n ≤ 10 − k (`replace_non_threes_to_threes`) |
| `--clean D … (10 digits)` | generated | literal clean input |
| `--corrupted D …` | per scheme | literal corrupted input (requires `--clean`) |
| `--edge-input D …` | generated with k | literal input for the attention view |
| `--seed <int>` | 0 | seed for generated sides |

```bash
python circuit_discovery/main.py single --n-threes 3 --scheme replace_threes_to_nonthrees
python circuit_discovery/main.py single --n-threes 2 --scheme replace_non_threes_to_threes --n-replace 3
python circuit_discovery/main.py single --scheme replace_threes_to_nonthrees \
    --clean 5 3 1 2 3 9 8 3 0 6 --corrupted 5 7 1 2 3 9 8 3 0 6 --edge-input 5 3 1 2 3 9 8 3 0 6
```

The same flag set works on `manual_circuit.py --single`, unscoped or scoped to a circuit; with
identical parameters both entry points read and write the same
`datasets/single/single_<scheme>_k<k>[_n<n>].json` and `edge_k<k>.json`:

```bash
python circuit_discovery/manual_circuit.py --single --n-threes 3 --scheme replace_threes_to_nonthrees
python circuit_discovery/manual_circuit.py --single --n-threes 3 \
    --scheme replace_threes_to_nonthrees --circuit eap_top200 --scope ablated
```

---

## 4. Data generation: `data_gen/generate_datasets.py`

```
python data_gen/generate_datasets.py [--apps …] [--schemes …] [--taus …] [--topns …]
       [--n-threes …] [-n N] [--seed S] [--replace-n R]
```

| Flag | Default | Meaning |
|---|---|---|
| `--apps manual\|acdc\|eap\|edge\|all …` | all | which applications |
| `--schemes <scheme>\|all …` | all | corruption schemes (paired apps) |
| `--taus τ …` | 0.2 0.1 0.055 0.02 | ACDC per-τ folders |
| `--topns n …` | 50 75 87 100 200 400 800 | EAP per-top-n folders |
| `--n-threes k …` | 0..10 | edge datasets, one per count of `3`s |
| `-n / --examples <int>` | per app: manual 128, acdc 20, eap 512, edge 64 | examples per dataset |
| `--seed <int>` | 0 | seed (the manual sets use seed + i for the i-th scheme: random, replace_threes_to_nonthrees, replace_non_threes_to_threes) |
| `--replace-n <int>` | 1 | replacements for the replace schemes |

```bash
python data_gen/generate_datasets.py                                   # everything, as `main.py data`
python data_gen/generate_datasets.py --apps acdc --schemes random --taus 0.055 0.02
python data_gen/generate_datasets.py --apps manual eap --schemes all --topns 100 200 -n 256
python data_gen/generate_datasets.py --apps edge --n-threes 0 5 10
```

Layout: `datasets/manual/`, `datasets/acdc/acdc_thresh_<tau>/`, `datasets/eap/eap_top<n>/` (each with
`random.json` and `batched_replace_*.json`), `datasets/edge/n_threes_<k>.json`,
`datasets/single/single_<scheme>_k<k>[_n<n>].json` and `datasets/single/edge_k<k>.json`.

Smoke tests (no arguments):

```bash
python data_gen/data_gen.py            # generation, JSON round trip, exact-3-count pairs (writes to a temp dir)
python data_gen/dataset_store.py       # store paths and a small generate/load round trip
python data_gen/segrev3_algorithm.py   # the transform's own checks
```

---

## 5. Discovery runners

### 5.1 Manual pipeline: `circuit_discovery/manual_circuit.py`

Default mode: data → activation analysis (`vis_patching_dla.py`) → weight analysis (`vis_edge.py`)
→ full-model validation (`validation.py`, all 32 heads).

| Flag | Default | Meaning |
|---|---|---|
| `-n / --examples <int>` | 128 | examples per generated dataset |
| `--seed <int>` | 0 | base seed |
| `--skip-data` | off | reuse `datasets/manual/*.json` |
| `--model <ckpt.pt>` | canonical | checkpoint (exported as `SEGREV3_CKPT`) |
| `--scheme <scheme>\|all` | random | contrast scheme (`all` only in circuit-scoped mode) |
| `--circuit <sel>` | — | circuit-scoped mode: `all` \| a method \| a circuit folder name |
| `--circuit-json <path>` | — | circuit-scoped mode on a graph file (ACDC `summary.json`, EAP graph JSON or `rough_circuit.json`; format detected) |
| `--scope full_circuit\|ablated` | full_circuit | mask the figures, or mean-ablate the complement |
| `--no-validate` | validate | skip the per-circuit validation in scoped mode |
| `--validate-mode mean\|zero\|resample` | mean | validation ablation |
| `--jobs <int>` | auto | concurrent (circuit × scheme) jobs |
| `--single` + §3 flags | off | single-sequence mode |

```bash
python circuit_discovery/manual_circuit.py --skip-data
python circuit_discovery/manual_circuit.py --circuit acdc_thresh_0.055
python circuit_discovery/manual_circuit.py --circuit eap --scheme all --scope ablated
python circuit_discovery/manual_circuit.py --circuit-json visualizations/eap/eap_top200/eap_top200.json
```

Scoped outputs: `visualizations/manual_circuit_analysis/figs_data/{figs,data}/circuits/<method>/<name>/<scope>/<scheme>/`
(single mode replaces `<scheme>` with the pair's label); scoped validation lands in
`validation/figs_data/{data,figs}/<method>/<name>/`.

### 5.2 ACDC: `circuit_discovery/acdc.py`

| Flag | Default | Meaning |
|---|---|---|
| `--thresholds τ …` | 0.2 0.1 0.055 0.02 | one output folder per τ |
| `-n / --examples <int>` | 20 | set size used only when the store file must be generated (an existing file is used as-is; its size is what is printed and recorded) |
| `--device auto\|cuda\|cpu` | auto | compute device |
| `--seed <int>` | 0 | dataset seed |
| `--scheme <scheme>` | random | contrast scheme from the store |
| `--jobs <int>` | min(#τ, 4) | τ values run concurrently (1 = sequential) |

Outputs: `visualizations/acdc/acdc_thresh_<tau>/{summary.json, circuit.gv, circuit.pth, circuit.png}`
and `visualizations/acdc/acdc_all_thresholds_summary.json`.

### 5.3 EAP-IG: `circuit_discovery/eap.py`

| Flag | Default | Meaning |
|---|---|---|
| `--ig-steps <int>` | 30 | integrated-gradient steps |
| `--topns n …` | 50 75 87 100 200 400 800 | top-n circuits to export |
| `--export-n <int>` | 200 | headline size; also the dataset folder (`datasets/eap/eap_top<n>/`) |
| `--seed <int>` | 0 | dataset seed |
| `--scheme <scheme>` | random | contrast scheme |

The internals (`EAP-IG-inputs`, 200 attribution examples in batches of 50, 256 evaluation examples
in batches of 64) are fixed in `run()`: the attribution batch size changes the score scale, because
EAP-IG normalizes by n_batches × ig_steps. Outputs: `visualizations/eap/eap_top<n>/eap_top<n>.{json,pt,png}`
and `visualizations/eap/eap_all_topn_summary.json`. Its `per_topn` block merges earlier runs'
folders, but its top-level `faithfulness_curve` covers only the current run. ACDC's combined
summary behaves the same way: `config.last_run_thresholds` lists only the current run. A partial
rerun with one τ or one top-n therefore rewrites these fields of the two committed summaries, and
rerunning the full documented sweep (steps 5 and 6) restores them.

---

## 6. Validation: `validation/validation.py`

| Flag | Default | Meaning |
|---|---|---|
| `--circuit <sel>` | — | `all` \| a method \| a circuit folder name |
| `--method` / `--name` | — | older spelling of the same selection |
| `--mode mean\|zero\|resample` | mean | ablation reference |

With no arguments it validates the full model (all 32 heads) under the name `rough`, which is what
the manual pipeline calls. Each circuit is validated on its own family's dataset
(`datasets/acdc/acdc_thresh_<tau>/`, `datasets/eap/eap_top<n>/`, or `datasets/manual/`).

```bash
python validation/validation.py                                    # full model, per-head drops
python validation/validation.py --circuit all
python validation/validation.py --circuit acdc_thresh_0.02 --mode resample
```

Custom head sets and the greedy head-level search (Python only):

```bash
python - <<'EOF'
import sys, json; sys.path.insert(0, "validation")
import validation as V
ALL = [f"L{l}H{h}" for l in range(4) for h in range(8)]
V.main(circuit_labels=[x for x in ALL if not x.startswith("L1")], name="ablate_L1_bank")
print(json.dumps(V.greedy_head_circuit(mode="mean", thresh=0.95, seed=11)))
EOF
```

`V.main(...)` writes `validation/figs_data/{data,figs}/manual/<name>_*`. `greedy_head_circuit`
removes the least damaging head while recovered token accuracy stays ≥ `thresh` × full;
`candidates=[(l, h), …]` restricts the pool. Outputs: `validation/figs_data/{data,figs}/<method>/<name>/`.

`python validation/n_threes_faithfulness.py [--circuits NAME …] [--tag TAG]` scores ACDC / EAP-IG
circuits on `datasets/edge/n_threes_<k>.json` (token accuracy per count of `3`s); `--tag` suffixes
the output names so a filtered run does not overwrite the full one.

---

## 7. Standalone analysis scripts

`vis_patching_dla.py` and `vis_edge.py` take no flags except `--data <path> …` (everything after it
is a path).

```bash
# activation patching + attribution + DLA + path patching (paired datasets; default: datasets/manual/*.json)
python visualizations/manual_circuit_analysis/vis_patching_dla.py
python visualizations/manual_circuit_analysis/vis_patching_dla.py --data datasets/manual/batched_replace_threes_to_nonthrees.json
python visualizations/manual_circuit_analysis/vis_patching_dla.py --data datasets/single/single_replace_threes_to_nonthrees_k3_n1.json

# weight-based QK / OV / copying scores / composition + realized attention (default: datasets/edge/*.json)
python visualizations/manual_circuit_analysis/vis_edge.py
python visualizations/manual_circuit_analysis/vis_edge.py --data datasets/edge/n_threes_0.json

# heuristic rough circuit (reads the random-scheme patching results + edge_results.json)
python interp_tools/circuit_summary.py

# bundle every PNG under visualizations/ and validation/figs_data/
python visualizations/collect_figures.py                      # -> figures.zip
python visualizations/collect_figures.py --roots visualizations --out manual_only.zip
```

DLA runs on single (clean) sequences only, in `correct_vs_rest` mode (correct digit vs the mean of
the other nine, positions 9..18). `vis_edge.py` writes weight-based figures once (flat, in
`figs_data/figs/edge/`) and realized-attention figures per dataset (`figs_data/figs/edge/<label>/`);
`*_attn.png` is the positional view and `*_attn_content.png` the source-digit view (column 3 =
attention to boundaries).

---

## 8. Library smoke tests

```bash
python interp_tools/model_loading.py    # accuracy gate (> 0.99) for the LNPre load
python interp_tools/patching_dla.py     # DLA parts sum to the logit difference; patching sanity
python interp_tools/edge_tools.py       # QK / OV / composition invariants
python interp_tools/circuit_loader.py   # loads every discovered circuit
python metrics/metrics.py               # metrics on all three schemes
```

Programmatic use:

```python
import sys; sys.path += ["interp_tools", "data_gen", "metrics"]
from model_loading import load_segrev3, ANSWER_POSITIONS
from data_gen import segrev3_batch, make_clean_corrupted, load_pairs
from segrev3_algorithm import segrev3_list
from metrics import logit_diff
model = load_segrev3(device="cpu")
tokens, inputs, outputs = segrev3_batch(256, seed=0)
pairs = make_clean_corrupted(256, scheme="replace_threes_to_nonthrees", n=1, seed=0)
```

---

## 9. Known rough edges

- Two naming styles for single-input files: batched `datasets/edge/n_threes_<k>.json` vs
  single-mode `datasets/single/edge_k<k>.json`.
- Store paths do not encode the seed, and loaders reuse an existing file even if another seed is
  requested.
- `vis_patching_dla.py` / `vis_edge.py` use minimal `--data` parsing (no `--help`);
  `circuit_summary.py` has no flags (its top-k values are constants in `build_circuit`).
- The full-model validation and the heuristic-circuit validation share the output name `rough`
  (§1.1).
- `validation.plot_faithfulness_curve()` takes its head order from the committed curve JSON; the
  knockout run that produced that order is not shipped (§1.2).
