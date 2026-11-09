#!/usr/bin/env python3
"""gen_circuit_data.py — per-circuit input-pool generator for the DSL pipeline.

Every circuit is a ``DomainCircuit`` subclass in ``models/circuits.py`` (its parent DSL operator plus a
``condition``/``eval``/``sample`` over the isolating domain, sampled via ``models/conditions.py``).
``circuits_for(name)`` returns a model's built circuits (variant names like ``segrev3_id__xl``
inherit the BASE algorithm's circuits); ``gen_one`` draws ONE input pool (``2*n`` rows) from a
circuit's sampler and writes it to ``gen_data/<model>/<circuit>/inputs.json``.

There is NO clean/corrupt split on disk: resample-patching pairs are split from the single pool at
load time (``model_registry.load_circuit_pairs``); outputs are recomputed downstream via ``task_fn``.
EXCEPT for ContrastCircuits (``scheme="contrast"``): those store row-aligned minimal-edit
clean/corrupt pairs with ``answer_positions="diff"`` (per-row masks derived at load, never stored).
The rest of the pipeline (find_circuits/*, interp_tools/*, run_model_pipeline) loops over
``circuits_for(name)`` and reads each circuit's dataset via ``load_circuit_pairs(name, circuit, n)``.

To add a circuit: define a ``DomainCircuit`` subclass in ``models/circuits.py``. A model with no
circuit there produces nothing.

Usage:
  python gen_circuit_data.py --model <algorithm_id> [--circuit all|<slug>] [-n 20000] [--seed 0]
"""
from __future__ import annotations

import argparse
import os
import sys

# sys.path bootstrap: this file lives at models/gen_data/gen_circuit_data.py.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))     # .../models/gen_data
_MODELS_DIR = os.path.dirname(_THIS_DIR)                   # .../models
for _p in (_MODELS_DIR, _THIS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch

import algorithms                       # noqa: E402  (DSL engine: models/algorithms.py)
import circuits as dsl_circuits         # noqa: E402  (canonical circuit registry: models/circuits.py)
from conditions import INPUT_LEN        # noqa: E402  (shared input length)

DEFAULT_N = 20000                        # PAIRS; the pool is 2*N inputs (split clean/corrupt at load).
                                         # smallest-faithful eval (SMALLEST_EVAL_N) needs the most;
                                         # acdc/eap/edge_pruning still slice their first-N (20/512/512)


def circuits_for(name: str) -> list["dsl_circuits.DomainCircuit"]:
    """The model's circuits as built ``DomainCircuit`` instances (from ``circuits.py``, the canonical
    source), matched on the BASE algorithm id so variant models inherit them."""
    base = algorithms.base_algorithm_id(name)
    if isinstance(algorithms.decode_algorithm_id(base), algorithms.TaskMux):
        dsl_circuits.task_circuits_for(base)          # the per-task base executors
        dsl_circuits.suppression_circuits_for(base)   # the C3 ordered-pair suppressors (base = task{cued})
    return [dsl_circuits.build(c) for c in dsl_circuits.CIRCUITS if c.algorithm == base]


def _answer_positions(in_len: int, out_len: int, subset) -> list:
    """Scored sequence positions: all output positions (in_len-1 .. in_len-1+out_len-1), or the
    selected 0-based OUTPUT indices mapped to sequence positions (in_len-1 + i)."""
    if subset is None:
        return list(range(in_len - 1, in_len - 1 + out_len))
    return [in_len - 1 + int(i) for i in subset]


def gen_one(name: str, circuit, n: int, seed: int) -> str:
    """Write one circuit's dataset and return the path; ``circuit`` is a built ``DomainCircuit``
    instance. Plain circuits store ONE ``2*n``-row pool (split clean/corrupt at load);
    ContrastCircuits store ``n`` row-aligned minimal-edit pairs with
    ``answer_positions="diff"`` (per-row masks recomputed at load, never stored)."""
    import model_registry as MR                      # lazy (heavy: torch + transformer_lens)
    from data_gen import save_dataset                # importable via model_registry's sys.path bootstrap above
    # True lengths come from the EXPR (TaskMux inputs are 11 tokens), not INPUT_LEN.
    expr = algorithms.decode_algorithm_id(algorithms.base_algorithm_id(name), input_length=INPUT_LEN)
    if isinstance(circuit, dsl_circuits.ContrastCircuit):
        clean, corrupt = circuit.sample_pairs(n, seed)
        if clean.shape[1] != expr.input_length or corrupt.shape[1] != expr.input_length:
            raise ValueError(f"{circuit.circuit_id}: sample_pairs() emitted widths "
                             f"{clean.shape[1]}/{corrupt.shape[1]}, model expects {expr.input_length}")
        for r in range(clean.shape[0]):
            if not circuit.condition(clean[r]):
                raise ValueError(f"{circuit.circuit_id}: clean row {r} violates condition()")
            if torch.equal(circuit.eval(clean[r]), circuit.eval(corrupt[r])):
                raise ValueError(f"{circuit.circuit_id}: pair {r} has identical TRUE outputs")
        ds = {
            "scheme": "contrast",
            "circuit": circuit.circuit_id,
            "model": name,
            "description": (type(circuit).__doc__ or "").strip(),
            "seed": seed,
            "n": n,
            "input_len": expr.input_length,
            "answer_positions": "diff",              # sentinel: per-row diff slots, derived at load
            "clean_inputs": clean.tolist(),
            "corrupted_inputs": corrupt.tolist(),
        }
    else:
        inputs = circuit.sample(2 * n, seed)
        if inputs.shape[1] != expr.input_length:
            raise ValueError(f"{circuit.circuit_id}: sample() emitted width {inputs.shape[1]}, "
                             f"model expects {expr.input_length}")
        if not all(circuit.condition(inputs[r]) for r in range(inputs.shape[0])):
            raise ValueError(f"{circuit.circuit_id}: sample() drew rows violating condition()")
        ds = {
            "scheme": "random",
            "circuit": circuit.circuit_id,
            "model": name,
            "description": (type(circuit).__doc__ or "").strip(),
            "seed": seed,
            "n": n,
            "input_len": expr.input_length,
            "answer_positions": _answer_positions(expr.input_length, expr.output_length(),
                                                  circuit.scored_output_indices),
            "inputs": inputs.tolist(),
        }
    path = MR.circuit_dataset_path(name, circuit.circuit_id)
    save_dataset(ds, path)                            # creates parent dirs
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="DSL algorithm_id")
    ap.add_argument("--circuit", default="all", help="'all' or a specific circuit slug")
    ap.add_argument("-n", type=int, default=DEFAULT_N)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    all_circuits = circuits_for(a.model)
    circuits = all_circuits if a.circuit == "all" else [c for c in all_circuits if c.circuit_id == a.circuit]
    if not circuits:
        raise SystemExit(f"no circuit {a.circuit!r} for model {a.model!r}; "
                         f"have {[c.circuit_id for c in all_circuits]}")
    for c in circuits:
        path = gen_one(a.model, c, a.n, a.seed)
        print(f"  [ok]   {a.model}/{c.circuit_id}: {a.n} pairs -> "
              f"{path.split('pipeline/')[-1]}")


if __name__ == "__main__":
    main()
