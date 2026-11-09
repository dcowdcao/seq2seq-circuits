#!/usr/bin/env python3
"""
Evaluate a trained model against DSL ground truth on a dumped-predictions CSV.

Usage:
    python eval_test_data.py --checkpoint <path> --algorithm <id> [--test_data <csv>]
"""

import argparse
import sys
from pathlib import Path

# models/ for the engine; this dir for utils; repo-root interp_tools for the loader
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parents[3] / "interp_tools"))

import pandas as pd
import torch

from model_loading import load_segrev3
from utils import get_algorithm


def load_model(checkpoint_path: str, device: str = "cpu"):
    """Load an eval-mode HookedTransformer from a train.py checkpoint."""
    return load_segrev3(checkpoint_path=checkpoint_path, device=device)


def compute_ground_truth(sequences: list, algorithm: str) -> list:
    """Per-sequence ground truth: digit string for sequence outputs, int for scalars."""
    algorithm_fn = get_algorithm(algorithm)  # raises KeyError: never let the model score itself
    ground_truth = []
    for seq in sequences:
        gt = algorithm_fn(torch.tensor([int(d) for d in seq], dtype=torch.long))
        if isinstance(gt, torch.Tensor):
            ground_truth.append("".join(str(x) for x in gt.tolist()) if gt.ndim else int(gt.item()))
        else:
            ground_truth.append(int(gt))
    return ground_truth


def print_report(results: dict, seq_to_seq: bool) -> None:
    """Print accuracy, error distribution, and sample wrong predictions."""
    total = results["total"]
    print(f"Results:")
    print(f"  Correct: {results['correct']} / {total}")
    print(f"  Accuracy: {results['accuracy']:.4%}")
    print(f"  Mean {'Token Error' if seq_to_seq else 'Absolute Error'}: {results['mean_error']:.4f}")
    print(f"  Max {'Token Error' if seq_to_seq else 'Error'}: {results['max_error']}")
    print()

    if len(results["error_distribution"]) > 1:  # skip if all perfect
        print(f"Error Distribution:")
        for err, count in sorted(results["error_distribution"].items()):
            label = f"{err} token(s) wrong" if seq_to_seq else f"Off by {err}"
            print(f"  {label}: {count:5d} ({100 * count / total:5.2f}%)")
        print()

    errors = results["errors"]
    if errors:
        print(f"Wrong Predictions ({len(errors)}):")
        for err in errors[:10]:
            csv_note = f" [CSV: {err['csv_prediction']}]" if err["csv_prediction"] != str(err["predicted"]) else ""
            detail = f"{err['error']} token(s) wrong" if seq_to_seq else f"off by {err['error']}"
            print(f"  {err['sequence']}: GT={err['expected']}, model={err['predicted']} ({detail}){csv_note}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")


def evaluate_test_data(
    checkpoint_path: str,
    test_data_path: str,
    device: str = "cpu",
    batch_size: int = 512,
    algorithm: str = "sortedness",
    seq_to_seq: bool = False,
) -> dict:
    """Score the model against DSL ground truth on the CSV's inputs (batched inference)."""
    print(f"Loading model from: {checkpoint_path}")
    model = load_model(checkpoint_path, device)

    print(f"Loading test data from: {test_data_path}")
    df = pd.read_csv(test_data_path, dtype={"inputs": str, "predictions": str})
    sequences = df["inputs"].tolist()
    csv_predictions = df["predictions"].tolist()

    input_length = len(sequences[0])
    output_length = model.cfg.n_ctx - input_length
    print(f"Rows: {len(sequences)} | algorithm: {algorithm} | "
          f"mode: {'seq_to_seq' if seq_to_seq else 'single_score'} | "
          f"input_length: {input_length} | output_length: {output_length} | batch_size: {batch_size}")
    print()

    print(f"Computing ground truth using {algorithm} function...")
    ground_truth = compute_ground_truth(sequences, algorithm)
    if not seq_to_seq:
        csv_predictions = [int(p) for p in csv_predictions]
    csv_mismatches = sum(1 for p, gt in zip(csv_predictions, ground_truth) if p != gt)
    if csv_mismatches > 0:
        print(f"CSV predictions differ from ground truth on {csv_mismatches} rows")
    else:
        print(f"✓ CSV predictions match ground truth perfectly")
    print()

    print(f"Running inference on {len(sequences)} sequences...")
    all_predictions = []
    for i in range(0, len(sequences), batch_size):
        batch_tokens = torch.tensor(
            [[int(d) for d in seq] for seq in sequences[i:i + batch_size]],
            dtype=torch.long,
        ).to(device)
        with torch.no_grad():
            if seq_to_seq:
                tokens = batch_tokens
                for _ in range(output_length):
                    next_token = model(tokens)[:, -1, :].argmax(dim=-1, keepdim=True)
                    tokens = torch.cat([tokens, next_token], dim=1)
                all_predictions.extend(
                    "".join(str(x) for x in row) for row in tokens[:, input_length:].cpu().tolist()
                )
            else:
                all_predictions.extend(model(batch_tokens)[:, -1, :].argmax(dim=-1).cpu().tolist())

    correct = 0
    errors = []
    error_values = []
    for seq, csv_pred, expected, predicted in zip(sequences, csv_predictions, ground_truth, all_predictions):
        # Hamming distance for sequences, absolute error for scores
        error = (sum(1 for e, p in zip(expected, predicted) if e != p) if seq_to_seq
                 else abs(predicted - expected))
        error_values.append(error)
        if error == 0:
            correct += 1
        else:
            errors.append({
                "sequence": seq,
                "expected": expected,
                "predicted": predicted,
                "csv_prediction": str(csv_pred),
                "error": error,
            })

    total = len(sequences)
    error_distribution = {}
    for err in error_values:
        error_distribution[err] = error_distribution.get(err, 0) + 1

    results = {
        "accuracy": correct / total if total > 0 else 0,
        "correct": correct,
        "total": total,
        "mean_error": sum(error_values) / len(error_values) if error_values else 0,
        "max_error": max(error_values) if error_values else 0,
        "error_distribution": error_distribution,
        "errors": errors,
    }
    print_report(results, seq_to_seq)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model on dumped test data")
    parser.add_argument("--checkpoint", "-c", required=True, help="Path to model checkpoint")
    parser.add_argument("--test_data", "-t", default=None,
                        help="Path to predictions CSV (default: <checkpoint stem>_test_data.csv beside the checkpoint)")
    parser.add_argument("--device", "-d", default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run on (default: cuda if available, else cpu)")
    parser.add_argument("--batch_size", "-b", type=int, default=512, help="Batch size for inference")
    parser.add_argument("--algorithm", "-a", default="sortedness", help="Algorithm name")
    parser.add_argument("--seq_to_seq", action="store_true",
                        help="Evaluate as sequence-to-sequence (token-level errors) instead of single-score prediction")

    args = parser.parse_args()
    ckpt = Path(args.checkpoint)
    test_data = args.test_data or str(ckpt.parent / f"{ckpt.stem}_test_data.csv")

    evaluate_test_data(
        args.checkpoint,
        test_data,
        args.device,
        args.batch_size,
        args.algorithm,
        args.seq_to_seq,
    )
