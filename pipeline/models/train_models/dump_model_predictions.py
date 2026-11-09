#!/usr/bin/env python3
"""
Dump a trained model's predictions on random inputs to a CSV (columns: inputs, predictions).

Usage:
    python dump_model_predictions.py --checkpoint path/to/checkpoint.pt --num_samples 1000 --seed 42
    python dump_model_predictions.py --checkpoint path/to/checkpoint.pt --num_samples 1000 --allow_repeats
"""

import argparse
import csv
import random
import sys
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parents[3] / "interp_tools"))

from hparams import INPUT_LEN as INPUT_LENGTH
from model_loading import load_segrev3
from utils import factoradic_to_permutation, get_total_permutations


def load_model(checkpoint_path: str, device: torch.device) -> tuple:
    """Return (model, checkpoint dict) from a train.py checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    return load_segrev3(checkpoint_path=checkpoint_path, device=str(device)), checkpoint


def generate_random_permutation() -> tuple:
    return factoradic_to_permutation(random.randint(0, get_total_permutations(10) - 1), 10)


def generate_random_sequence() -> tuple:
    return tuple(random.choices(range(10), k=10))


@torch.no_grad()
def predict_batch(model, sequences: list, device, input_length: int, output_length: int) -> list:
    """Greedy predictions: int per row for output_length 1, digit string otherwise."""
    tokens = torch.tensor(sequences, dtype=torch.long, device=device)
    if output_length == 1:
        return model(tokens)[:, -1, :].argmax(dim=-1).cpu().tolist()
    for _ in range(output_length):
        next_token = model(tokens)[:, -1, :].argmax(dim=-1, keepdim=True)
        tokens = torch.cat([tokens, next_token], dim=1)
    return ["".join(map(str, row)) for row in tokens[:, input_length:].cpu().tolist()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Dump model predictions on random inputs to CSV")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--num_samples", type=int, required=True, help="Number of samples to generate")
    parser.add_argument("--allow_repeats", action="store_true", help="Random digit sequences instead of permutations")
    parser.add_argument("--output", type=str, default=None,
                        help="Output CSV (default: <checkpoint stem>_test_data.csv beside the checkpoint)")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for inference")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible inputs")

    args = parser.parse_args()

    ckpt = Path(args.checkpoint)
    output = Path(args.output) if args.output else ckpt.parent / f"{ckpt.stem}_test_data.csv"

    if args.seed is not None:
        random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model from {ckpt} (device: {device})...")
    model, checkpoint = load_model(args.checkpoint, device)

    n_ctx = checkpoint["config"]["n_ctx"]
    output_length = n_ctx - INPUT_LENGTH
    print(f"Model config: n_ctx={n_ctx}, input_length={INPUT_LENGTH}, output_length={output_length}")
    print(f"Predicting {args.num_samples} samples (allow_repeats={args.allow_repeats}, batch_size={args.batch_size})...")

    rows = []
    for start in tqdm(range(0, args.num_samples, args.batch_size), desc="Predicting batches"):
        n = min(args.batch_size, args.num_samples - start)
        sequences = [generate_random_sequence() if args.allow_repeats else generate_random_permutation()
                     for _ in range(n)]
        predictions = predict_batch(model, sequences, device, INPUT_LENGTH, output_length)
        rows.extend({"inputs": "".join(map(str, seq)), "predictions": str(pred)}
                    for seq, pred in zip(sequences, predictions))

    with open(output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["inputs", "predictions"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} predictions to {output}")


if __name__ == "__main__":
    main()
