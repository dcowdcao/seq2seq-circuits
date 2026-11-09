#!/usr/bin/env python
"""Write results/showcase/COMMANDS.md from each example's runtimes.json (written by
``reproduce.py --full``): the exact commands, in order, with their measured wall time.

    python results/showcase/make_commands.py
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import showcase_spec as S  # noqa: E402

HEAD = """# Commands and CPU runtimes

Every command below was run from the repository root by
`python results/showcase/reproduce.py --full`, in this order, on CPU only, with
`PYTHONPATH` set to the pipeline's import directories, `SEGREV3_DEVICE=cpu` and
`OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS={threads}` (the smallest stage splits
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
exactly and every float field within {tol:g}.
"""


def fmt(secs: int) -> str:
    return f"{secs // 60} min {secs % 60:02d} s" if secs >= 60 else f"{secs} s"


def main() -> None:
    import reproduce as R
    out = [HEAD.format(threads=R.THREADS, tol=R.FLOAT_TOL)]
    total = 0
    for ex in S.EXAMPLES:
        p = os.path.join(ex.dir, "runtimes.json")
        if not os.path.exists(p):
            continue
        rt = json.load(open(p))
        out += [f"## {ex.cls}: `{ex.model}` / `{ex.circuit}`", "",
                "| step | wall time |", "|---|---|"]
        sub = 0
        for phase in ("discovery", "selection", "heldout"):
            for s in rt.get(phase, []):
                out.append(f"| {phase}: {s['step']} | {fmt(s['wall_seconds'])} |")
                sub += s["wall_seconds"]
        total += sub
        out += [f"| **total** | **{fmt(sub)}** |", ""]
        if ex.probe_topns:
            n = json.load(open(os.path.join(ex.dir, "circuit.json")))["n_edges_full"]
            out += [f"Besides the canonical frac ladder, probe-metric EAP-IG also runs the explicit "
                    f"rungs {', '.join(map(str, ex.probe_topns))} on each pair pool: only {n} edges "
                    f"can reach this probe, and the 4-rung frac ladder over {n} edges is too "
                    "coarse.", ""]
        out += ["```bash"]
        out += [s["command"] for phase in ("discovery", "selection", "heldout") for s in rt.get(phase, [])]
        out += ["```", ""]
    out += [f"Sum over the five examples: {fmt(total)} of wall time.", "",
            "## Verification", "",
            "`python results/showcase/reproduce.py --verify` runs, for each example, the "
            "`command` stored in its `circuit.json`, `base_circuit.json` and "
            "`heldout_seed1.json`, i.e.:", "",
            "```bash"]
    for ex in S.EXAMPLES:
        for fn in ("base_circuit.json", "circuit.json", "heldout_seed1.json"):
            p = os.path.join(ex.dir, fn)
            if os.path.exists(p):
                out.append(json.load(open(p))["command"])
    out += ["```", ""]
    with open(os.path.join(S.HERE, "COMMANDS.md"), "w") as f:
        f.write("\n".join(out))
    print("wrote COMMANDS.md")


if __name__ == "__main__":
    main()
