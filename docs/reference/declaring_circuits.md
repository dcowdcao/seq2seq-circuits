# Declaring a circuit by hand

Sub-circuits are declared by hand, not auto-detected. This page is the field guide for that step:
choose the slice of inputs on which a model collapses to a simpler algorithm, write it down as a
`DomainCircuit` (or `ContrastCircuit`) subclass, generate its data, check the isolation, and run
the analysis. Class definitions and selection rules: [circuit_types.md](circuit_types.md).

## Background

The pipeline produces **sub-circuits only**: there is no implicit whole-model circuit, and a
model with no declared circuit produces nothing. A sub-circuit isolates **one** aspect of the task
by **restricting the inputs** (or **scoping the scored output positions**) so that the model
collapses to a simpler, well-defined base algorithm on that slice.

Canonical example: `segrev3_id = SegmentReverse(3, Id)`. On inputs with **no 3s** the whole
sequence is one segment, so the model just **reverses**. Drawing the circuit's data only from
no-3 inputs isolates the **reversal** sub-circuit, `reverse_only`.

## The four ways to isolate

| method | when | example rule | becomes, in the subclass |
|---|---|---|---|
| `digits` | restrict or exclude **raw** digits | "exclude digit 3", "digits in {0,1}" | `_cond = conditions.exclude_digit(3)` or `conditions.domain([0, 1])` |
| `condition` | keep inputs meeting a condition | "every value ≤ 4" (raw); "total digit-sum ≤ 12" (running state) | a `conditions.*` predicate with its sampler, e.g. `conditions.sum_at_most(12)` (constructive `bounded_sum_sampler`) |
| `output_positions` | score only some output positions | "output indices 0–9" | `scored_output_indices = (...)` with `_cond = conditions.FULL` |
| `custom` | anything else | describe the exact input sampling | a custom `conditions.Condition(name, holds, sample)` |

A `Condition` bundles the predicate `holds(x)` with a matching uniform sampler `sample(batch,
seed)`, so the two cannot drift apart. The library in
[`pipeline/models/conditions.py`](../../pipeline/models/conditions.py) includes `domain`,
`exclude_digit`, `pin`, `exactly_one_in`, `at_most_k_of`, `sum_at_most`, `monotone`,
`compare_positions`, `task_pin`, the constants `FULL`, `ALL_EVEN`, `ALL_ODD`, `BINARY`,
`ONE_ODD`, `ONE_REPEAT`, `ONE_FIXEDPOINT`, and the sampler factories they use
(`domain_sampler`, `bounded_sum_sampler`, `bounded_count_sampler`, `one_in_set_sampler`,
`position_domain_sampler`, `monotone_sampler`, `compare_positions_sampler`, …).

## Step by step

1. **Optionally record the request** in
   [`pipeline/models/human_circuit_finding.json`](../../pipeline/models/human_circuit_finding.json),
   the structured intake file (one object per sub-circuit in `"requests"`; the schema is inline
   under `_request_schema` / `_methods`):

   ```jsonc
   {
     "model"      : "<DSL algorithm_id, e.g. segrev3_id>",
     "checkpoint" : "<\"trained\" | \"imported as models/<dir>/<name>.pt\">",
     "slug"       : "<short_lowercase circuit id, e.g. reverse_only>",
     "isolate"    : "<plain English, e.g. \"the reversal behaviour only\">",
     "isolate_by" : { "method": "<digits | condition | output_positions | custom>",
                      "detail": "<the exact rule>" }
   }
   ```

   The file is documentation: no stage reads it to build or select a circuit. The `smallest` stage
   only copies a request's `isolate` and `isolate_by.detail` text into the `input_sequence_data`
   description of `smallest_circuit.json`. Its `faithfulness_target` field is read by no code —
   the selection thresholds are those of the `smallest` stage
   ([circuit_types.md §3](circuit_types.md)).

2. **Author the subclass** in [`pipeline/models/circuits.py`](../../pipeline/models/circuits.py).
   Subclass the model's parent DSL operator together with `DomainCircuit`; set `circuit_id` (the
   slug), `algorithm` (the model id) and `_cond`; override `eval` when the circuit computes
   something simpler on its domain. Defining the class registers it in `circuits.CIRCUITS`.

   ```python
   class ReverseOnlyCircuit(SegmentReverse, DomainCircuit):
       circuit_id = "reverse_only"
       algorithm = "segrev3_id"
       _cond = conditions.exclude_digit(3)
       def eval(self, input_seq):
           return Reverse(Id(self.input_length)).eval(input_seq)
   ```

   For a ContrastCircuit, subclass `ContrastCircuit` instead, name the sibling `base_circuit`,
   implement `sample_pairs(batch, seed)` (its first line must be
   `seed = conditions.stream_key(seed, self.circuit_id)`), and leave `eval` as the true operator.
   Pair designs per model: [contrast_pair_catalog.md](contrast_pair_catalog.md).

3. **Validate the declaration.** `circuits.validate(cls)` draws 2000 in-domain rows and raises if
   the sampler leaves the domain or `eval` disagrees with the true operator anywhere;
   `circuits.validate_pairs(cls)` checks every contrast pair has differing inputs and a non-empty
   output mask. `python pipeline/repo_tools/make_circuits_txt.py` runs both over every registered
   circuit and rewrites the catalog `pipeline/models/circuits.txt`.

4. **Generate the data.**

   ```bash
   python pipeline/run_model_pipeline.py --model <model> --stages data
   ```

   writes one input pool per circuit to `pipeline/models/gen_data/<model>/<slug>/inputs.json`
   (`models/gen_data/gen_circuit_data.py`; no checkpoint needed). Clean/corrupt pairs are split
   from the pool at load time by `model_registry.load_circuit_pairs`.

5. **Check the isolation.** Confirm that the sampled inputs satisfy the restriction and, where
   checkable, that the operator on that domain computes the intended base algorithm (step 3 does
   the latter for any circuit with an `eval` override). If the branch you want to remove keys on a
   running state rather than on a raw digit, a raw-digit restriction will not pin it; use a
   `condition` or `custom` sampler that bounds the state itself (see example 2 below), or declare
   a ContrastCircuit instead.

6. **Run the analysis** for the model:

   ```bash
   python pipeline/run_model_pipeline.py --model <model> --stages all     # local by default
   python pipeline/run_model_pipeline.py --model <model> --stages all --modal   # optional
   ```

   `all` = `data train eap smallest audit`; `probes`, `patching`, `edges`, `acdc` and
   `edge_pruning` run only when named. `--only-circuit <slug>` restricts the per-circuit stages to
   one circuit. Outputs: `pipeline/circuits_data/<model>/<slug>/{circuits/, smallest_circuit.json}`
   and figures under `pipeline/circuits_figures/<model>/<slug>/`. The optional `--modal` backend
   runs the GPU stages on Modal containers from an image built by
   `pipeline/modal_pipeline_app.py`; the image is a snapshot of the code, so redeploy it after
   adding a circuit.

## Worked examples (all registered in `models/circuits.py`)

**1. Reversal, by excluding a digit** (`digits`)

```json
{ "model": "segrev3_id", "slug": "reverse_only",
  "isolate": "the plain whole-sequence reverse: with no copies of the boundary digit 3 present, the whole input is a single segment, so the model just reverses",
  "isolate_by": { "method": "digits", "detail": "exclude digit 3" } }
```

→ `_cond = conditions.exclude_digit(3)` (uniform over the digits `{0,1,2,4,…,9}`), `eval` =
`Reverse`.

**2. The accumulator alone, by bounding the running sum** (`condition`;
`areset13_id = AccumulateReset(13, Id)`)

```json
{ "model": "areset13_id", "slug": "areset13_accum",
  "isolate": "the running-sum accumulation (prefix-sum mod 10) with the threshold-13 reset never triggered, so the reset machinery is unexercised and ablatable",
  "isolate_by": { "method": "condition", "detail": "total digit-sum <= 12" } }
```

→ `_cond = conditions.sum_at_most(12)`, `eval` = `PrefixSumMod`. The reset is a running-state
branch, not a raw-digit one, so it needs a **constructive** sampler (`bounded_sum_sampler`, an
exact-uniform stars-and-bars draw over sequences with sum ≤ 12) rather than rejection sampling,
which would almost never accept a uniform row. The reset path is never exercised on this slice,
so its machinery is ablatable; the contrast circuits `areset13_crossing_toggle` /
`areset13_crossing_compensated` then expose it.

**3. One half of the output, by scoping output positions** (`output_positions`)

```json
{ "model": "argminmax_id", "slug": "argminmax_argmax",
  "isolate": "global-argmax indicator (first output half)",
  "isolate_by": { "method": "output_positions", "detail": "score output indices 0-9" } }
```

→ `_cond = conditions.FULL`, `scored_output_indices = (0, 1, …, 9)`.

**4. An anchored operator with the anchor pinned** (`custom`; recorded as `custom` in the intake
file, implemented with the library condition `pin`)

```json
{ "model": "anc0_id", "slug": "anc0_caesar",
  "isolate": "Caesar shift via a pinned anchor",
  "isolate_by": { "method": "custom", "detail": "pin x[0]=3" } }
```

→ `_cond = conditions.pin(0, [3])` (sampler `position_domain_sampler({0: [3]})`).
`anc0_id` computes `(x[i] − x[0]) mod 10`; pinning the anchor value collapses it to a Caesar
shift by −3.

## Tips for a clean isolation

- The restriction should make your target the **only** behaviour exercised: pick the slice on
  which the model reduces to a single well-defined base algorithm.
- **Input restriction works when the branch keys on the raw input** (like segrev3's boundary
  digit). If it keys on a transformed value or an accumulated state, state the condition precisely
  (`condition` or `custom`) and sample it constructively.
- Keep the domain non-degenerate (**≥ 2 distinct digits**) so attribution has signal.
- **A sampler must be exact-uniform over the domain its condition names**, not merely supported on
  it. The instructive failure: drawing monotone rows by *sorting* i.i.d. digits weights each row by
  its multinomial coefficient, making near-constant rows (the ones where the collapse is most
  visible) far too rare. `monotone_sampler` and `compare_positions_sampler` draw uniformly; if you
  add a sampler, state its support and check its marginals.
- **Sibling scopes must not share rows.** Every sampler seeds off
  `conditions.stream_key(seed, tag)`, so two scopes on one model draw independent pools (the mux
  `task_pin` siblings are the deliberate exception: they share a payload). Without it a contrast
  circuit's clean pool could be its base circuit's pool, which makes `d_control` degenerate.
- The slug becomes a folder name under `circuits_data/<model>/` and `circuits_figures/<model>/`:
  keep it short, lowercase and descriptive.

Which DSL operators have conditional structure worth isolating:
[conditional_models.md](conditional_models.md). Internal variables worth probing instead:
[probe_variable_catalog.md](probe_variable_catalog.md).
