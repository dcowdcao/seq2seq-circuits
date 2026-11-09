#!/usr/bin/env python3
"""Generate the circuit catalog: one `<algorithm_id>\t<circuit_id>` line per registered circuit.

Circuits are specs, not trained. This lists every DomainCircuit subclass in circuits.py,
validating each matches its parent algorithm wherever its condition holds.

Usage:
    python make_circuits_txt.py                 # -> circuits.txt
    python make_circuits_txt.py --trials 5000   # more random inputs per validation
    python make_circuits_txt.py --out circuits.txt
"""
import argparse
import sys
from pathlib import Path

MODELS_DIR = Path(__file__).parents[1] / "models"
sys.path.insert(0, str(MODELS_DIR))

from circuits import CIRCUITS, CircuitError, ContrastCircuit, validate, validate_pairs


def die(msg: str) -> None:
    raise CircuitError(msg)


def main() -> None:
    p = argparse.ArgumentParser(description="Generate the circuit catalog (circuits.txt)")
    p.add_argument("--out", default=str(MODELS_DIR / "circuits.txt"),
                   help="Output circuit catalog")
    p.add_argument("--trials", type=int, default=2000,
                   help="Random inputs per circuit for the condition/eval check")
    args = p.parse_args()

    seen, lines = set(), []
    for c in CIRCUITS:
        if c.circuit_id in seen:
            die(f"Duplicate circuit_id: {c.circuit_id}")
        seen.add(c.circuit_id)
        validate(c, trials=args.trials)   # raises on any eval/condition mismatch
        if issubclass(c, ContrastCircuit):
            validate_pairs(c)             # pairs gate: in-domain clean, differing inputs + TRUE outputs
        lines.append(f"{c.algorithm}\t{c.circuit_id}")

    lines.sort()
    Path(args.out).write_text("\n".join(lines) + "\n")
    print(f"Wrote {len(lines)} circuits -> {args.out}")


if __name__ == "__main__":
    try:
        main()
    except CircuitError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)
