#!/usr/bin/env python3
"""
Generate the model list: one depth-2 (single-operator) DSL algorithm per line,
i.e. IDs of the form `<op>_id`. One trained transformer ("model") per line.

Every operator in the chosen pool is guaranteed to appear (unlike random
sampling via DSLProgramGenerator.generate_batch, which draws operators
uniformly and can miss some). Parameter-free operators (rev, pmax, psum,
xorscan, ...) have exactly one depth-2 program, so --samples-per-op > 1 yields
1 for them by design.

Usage:
    python make_models_txt.py                         # complex ops, 1 each
    python make_models_txt.py --ops all               # include basic transforms
    python make_models_txt.py --samples-per-op 3      # 3 param settings per op
    python make_models_txt.py --out models.txt
"""
import argparse
import sys
from pathlib import Path

MODELS_DIR = Path(__file__).parents[1] / "models"
sys.path.insert(0, str(MODELS_DIR))

from algorithms import (
    Id, BASIC_OPS, COMPLEX_OPS, ALL_OPS,
    DSLProgramGenerator, generate_algorithm_id, decode_algorithm_id,
)

INPUT_LEN = 10


def main() -> None:
    p = argparse.ArgumentParser(description="Generate the depth-2 (single-operator) DSL model list")
    p.add_argument("--ops", choices=["complex", "all", "basic"], default="complex",
                   help="Operator pool (default: complex — skips the trivial basic transforms)")
    p.add_argument("--samples-per-op", type=int, default=1,
                   help="Distinct parameterizations to attempt per operator (default: 1)")
    p.add_argument("--out", default=None,
                   help="Write the list here (default: print to stdout). models.txt is the "
                        "curated run-list — pass --out models.txt only to overwrite it deliberately.")
    p.add_argument("--seed", type=int, default=2026, help="RNG seed for parameter choices")
    args = p.parse_args()

    pool = {"complex": COMPLEX_OPS, "all": ALL_OPS, "basic": BASIC_OPS}[args.ops]
    gen = DSLProgramGenerator(input_length=INPUT_LEN, depth=2, seed=args.seed)

    seen, out = set(), []
    for op in pool:
        made = tries = 0
        while made < args.samples_per_op and tries < 200:
            tries += 1
            expr = gen._build_op(op, Id(INPUT_LEN))
            aid = generate_algorithm_id(expr)
            if aid in seen:
                continue
            # round-trip guard: ID must decode back to the same ID
            if generate_algorithm_id(decode_algorithm_id(aid)) != aid:
                continue
            seen.add(aid)
            out.append(aid)
            made += 1

    text = "\n".join(out) + "\n"
    if args.out:
        Path(args.out).write_text(text)
        print(f"Wrote {len(out)} depth-2 models ({len(pool)} operators, "
              f"{args.samples_per_op} sample(s)/op) -> {args.out}", file=sys.stderr)
    else:
        sys.stdout.write(text)


if __name__ == "__main__":
    main()
