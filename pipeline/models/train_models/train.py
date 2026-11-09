#!/usr/bin/env python3
"""
Training script for TransformerLens HookedTransformers on symbolic seq->seq tasks.

Supports on-the-fly permutation/sequence datasets, multi-GPU DDP and autoregressive
seq->seq loss. Output length and input length come from the DSL algorithm. Tier configs
(--config) come from interp_tools/model_loading.py (TIER_PRESETS), so the trainer builds
the exact architecture the analysis loader (load_segrev3) reconstructs.

The RECIPE (lr / batch / steps / warmup / grad-clip / dtype / eval size / seed) is NOT a
flag: it is resolved from hparams.for_tier(--config), the same table train_stacked.py
reads, so the recipe is a property of the architecture rather than of which script ran.
Its `batch` is the EFFECTIVE batch — DDP splits it across ranks, so --nproc_per_node
changes throughput and nothing else.

Usage:
    # Single GPU / CPU (debugging, smoke)
    python train.py --algorithm rcnt3_id --config tiny --smoke --output_dir ./checkpoints

    # Multi-GPU with DDP (production, via run_training_job.sh)
    torchrun --nproc_per_node=4 train.py --algorithm rcnt3_id --config large \
        --allow_repeats --random_sampling
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

# models/ for the engine; this dir for utils; repo-root interp_tools for the loader
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parents[3] / "interp_tools"))

from transformer_lens import HookedTransformer

import hparams
import promote
from model_loading import TIER_PRESETS, segrev3_config, tier_config
from utils import ALGORITHM_METADATA, PermutationDataset, SequenceDataset, get_algorithm

DATALOADER_WORKERS = 4          # per-GPU input pipeline; not part of the recipe


def setup_ddp() -> int:
    """Initialize DDP (nccl on GPU, gloo on a CPU-only machine) and return the local rank."""
    dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    local_rank = int(os.environ["LOCAL_RANK"])
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    return local_rank


def cleanup_ddp() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def log(msg: str, force: bool = False) -> None:
    """Log only on the main process (or if force=True)."""
    if force or is_main_process():
        print(msg, flush=True)


def train_epoch(
    model,
    train_loader,
    optimizer,
    scaler,
    device,
    epoch: int,
    use_amp: bool,
    hp,
    step: int,
    input_length: int = 10,
    output_length: int = 1,
) -> dict:
    """Train one epoch (stopping at hp.steps total); returns {'loss','accuracy','step'}."""
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    pbar = tqdm(
        train_loader,
        desc=f"Epoch {epoch}",
        disable=not is_main_process(),
        dynamic_ncols=True,
    )

    for batch in pbar:
        if step >= hp.steps:
            break
        batch = batch.to(device, non_blocking=True)
        output_tokens = batch[:, input_length:]

        step += 1
        set_lr(optimizer, hp, step)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            if output_length == 1:
                # Single output prediction at the last input position
                logits = model(batch[:, :input_length])
                loss = F.cross_entropy(logits[:, -1, :], output_tokens[:, 0])
            else:
                # Concatenated autoregressive: predict next token at each output position
                logits = model(batch[:, :-1])
                output_logits = logits[:, input_length - 1:, :]
                loss = F.cross_entropy(
                    output_logits.reshape(-1, logits.size(-1)),
                    output_tokens.reshape(-1),
                )

        scaler.scale(loss).backward()
        if hp.grad_clip:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), hp.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        batch_size = batch.size(0)
        total_loss += loss.item() * batch_size
        with torch.no_grad():
            if output_length == 1:
                preds = logits[:, -1, :].argmax(dim=-1)
                total_correct += (preds == output_tokens[:, 0]).sum().item()
            else:
                # Strict accuracy: ALL output tokens must match
                preds = output_logits.argmax(dim=-1)
                total_correct += (preds == output_tokens).all(dim=1).sum().item()
        total_samples += batch_size

        pbar.set_postfix(
            loss=f"{total_loss / total_samples:.4f}",
            acc=f"{total_correct / total_samples:.4f}",
        )

    avg_loss = total_loss / total_samples if total_samples > 0 else 0
    avg_acc = total_correct / total_samples if total_samples > 0 else 0
    return {"loss": avg_loss, "accuracy": avg_acc, "step": step}


def set_lr(optimizer, hp, step: int) -> float:
    """Linear warmup over hp.warmup steps, then hp.lr flat. Returns the applied LR."""
    lr = hp.lr * min(1.0, step / hp.warmup) if hp.warmup else hp.lr
    for g in optimizer.param_groups:
        g["lr"] = lr
    return lr


@torch.no_grad()
def evaluate(
    model, data_loader, device, use_amp: bool, is_ddp: bool,
    input_length: int = 10, output_length: int = 1,
) -> dict:
    """Sequence-exact accuracy over the full dataset; restores the model's mode."""
    was_training = model.training
    model.eval()

    total_correct = 0
    total_samples = 0
    last_correct = False

    pbar = tqdm(
        data_loader,
        desc="Evaluating",
        disable=not is_main_process(),
        dynamic_ncols=True,
        leave=False,
    )

    for batch in pbar:
        batch = batch.to(device, non_blocking=True)
        input_tokens = batch[:, :input_length]
        output_tokens = batch[:, input_length:]

        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            if output_length == 1:
                logits = model(input_tokens)
                correct = logits[:, -1, :].argmax(dim=-1) == output_tokens[:, 0]
            else:
                # Autoregressive generation (no teacher forcing), strict all-token match
                current_tokens = input_tokens.clone()
                for _ in range(output_length):
                    logits = model(current_tokens)
                    next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                    current_tokens = torch.cat([current_tokens, next_token], dim=1)
                correct = (current_tokens[:, input_length:] == output_tokens).all(dim=1)

        total_correct += correct.sum().item()
        total_samples += batch.size(0)
        last_correct = bool(correct[-1].item())

    if is_ddp:
        # DistributedSampler pads by wrapping to the start; each padded rank's LAST sample
        # is the duplicate (eval sampler is sequential), so drop it before aggregating.
        world_size = dist.get_world_size()
        pad = len(data_loader.sampler) * world_size - len(data_loader.dataset)
        if pad > 0 and dist.get_rank() >= world_size - pad:
            total_samples -= 1
            total_correct -= int(last_correct)
        counts = torch.tensor([total_correct, total_samples], device=device)
        dist.all_reduce(counts, op=dist.ReduceOp.SUM)
        total_correct, total_samples = counts[0].item(), counts[1].item()

    model.train(was_training)
    return {"accuracy": total_correct / total_samples if total_samples > 0 else 0}


def save_checkpoint(
    model, optimizer, scaler, epoch: int, metrics: dict, output_dir, is_ddp: bool,
    algorithm_id: str, config: dict, filename: str, hp,
) -> None:
    """Save a checkpoint ({config, model_state_dict, algorithm_id, metrics, ...}) on rank 0.
    ``config`` is the tier_config dict — the persisted schema segrev3_config reads.
    ``metrics`` carries the resolved recipe + seed, so a checkpoint always records the
    regime it was trained under and is reproducible from its own record."""
    if not is_main_process():
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_to_save = model.module if is_ddp else model
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model_to_save.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "metrics": {**metrics, "seed": hp.seed, "hparams": hp.as_dict()},
        "algorithm_id": algorithm_id,
        "config": dict(config),
    }
    checkpoint_path = output_dir / filename
    torch.save(checkpoint, checkpoint_path)
    log(f"Saved checkpoint: {checkpoint_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a seq2seq transformer (attention-only, MLP or MoE) on a DSL algorithm"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="small",
        choices=sorted(TIER_PRESETS),
        help="Model tier name (interp_tools/model_loading.py TIER_PRESETS)",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default="sortedness",
        help="Algorithm to train on (sortedness, inversion_count, or any DSL algorithm ID)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./checkpoints",
        help="Directory for checkpoints",
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="DataLoader workers per GPU"
    )
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed; recorded in the checkpoint (default: %(default)s)")
    parser.add_argument("--target", type=float, default=promote.GATE,
                        help="stop early at this eval seq-acc (default: %(default)s)")
    parser.add_argument("--smoke", action="store_true",
                        help="throwaway budget (hparams.smoke): 20 steps, tiny pools")
    parser.add_argument(
        "--allow_repeats",
        action="store_true",
        help="Allow repeated digits in sequences (uses SequenceDataset instead of PermutationDataset)",
    )
    parser.add_argument(
        "--random_sampling",
        action="store_true",
        help="Use random sampling instead of stratified sampling with --allow_repeats (more diverse)",
    )

    args = parser.parse_args()
    # THE recipe: resolved from the per-arch table, never from flags (hparams.py).
    hp = hparams.smoke() if args.smoke else hparams.for_tier(args.config, seed=args.seed)

    # Detect if running with torchrun (DDP)
    is_ddp = "LOCAL_RANK" in os.environ

    if is_ddp:
        local_rank = setup_ddp()
        device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
        world_size = dist.get_world_size()
    else:
        local_rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        world_size = 1

    torch.manual_seed(hp.seed + local_rank)  # Different seed per rank for shuffling

    use_amp = hp.dtype == "bf16" and torch.cuda.is_available()
    rank_batch = hparams.per_rank_batch(hp, world_size)   # hp.batch is the EFFECTIVE batch
    epochs = hparams.epochs_for(hp)

    # Variant twins (name__xl): task semantics from the base id, artifacts under the full name.
    full_algorithm = args.algorithm
    args.algorithm = full_algorithm.split("__", 1)[0]
    get_algorithm(args.algorithm)  # trigger DSL registration if needed

    # Output length + vocab come from the DSL algorithm; scalar tasks (sortedness, ...)
    # emit 1 token over the default digit vocab.
    algorithm_id = None
    output_length = 1
    d_vocab = 10
    if args.algorithm in ALGORITHM_METADATA:
        metadata = ALGORITHM_METADATA[args.algorithm]
        algorithm_id = full_algorithm if full_algorithm != args.algorithm else metadata["id"]
        output_length = metadata["output_length"]
        d_vocab = metadata["expr"].vocab_size()
        log(f"  Algorithm ID: {algorithm_id}")
        log(f"  Algorithm: {metadata['str']}")
    input_length = metadata["expr"].input_length if algorithm_id else hp.input_len

    log(f"Training config (regime {hp.regime}, tier {hp.tier}):")
    log(f"  Algorithm: {args.algorithm}")
    log(f"  lr {hp.lr:g}  batch {hp.batch} effective ({rank_batch}/GPU x {world_size})  "
        f"steps {hp.steps}  warmup {hp.warmup}  clip {hp.grad_clip}  dtype {hp.dtype}")
    log(f"  pool {hp.pool}  eval_n {hp.eval_n}  seed {hp.seed}  target {args.target}")
    log(f"  World size: {world_size}  epochs {epochs} (steps/epoch "
        f"{max(1, hp.pool // hp.batch)})")
    log(f"  Device: {device}")

    cfg = tier_config(args.config, input_length=input_length,
                      output_length=output_length, d_vocab=d_vocab)
    model = HookedTransformer(segrev3_config(cfg)).to(device)
    log(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    log(f"  Input length: {input_length}")
    log(f"  Output length: {output_length}")
    log(f"  d_vocab: {d_vocab}")
    log(f"  n_ctx: {cfg['n_ctx']}")

    if is_ddp:
        # MoE top-k routing can leave whole experts gradient-free in a step
        model = DDP(model, device_ids=[local_rank] if torch.cuda.is_available() else None,
                    find_unused_parameters=bool(cfg.get("num_experts")))

    if args.allow_repeats:
        sampling_mode = "random" if args.random_sampling else "stratified"
        # Pool/eval sizes come from the table, so the 0.95 gate carries the SAME sampling
        # noise at every architecture (it used to be 1e6 here vs 4096 in the vmap path).
        eval_dataset = SequenceDataset(
            algorithm=args.algorithm,
            n_digits=input_length,
            num_samples=hp.eval_n,
            random_sampling=args.random_sampling,
            seed=42,  # Fixed seed for eval reproducibility
        )
        train_dataset = SequenceDataset(
            algorithm=args.algorithm,
            n_digits=input_length,
            num_samples=hp.pool,
            random_sampling=args.random_sampling,
            seed=None,  # Unseeded: fresh draw per run
        )
        log(f"  Eval samples: {len(eval_dataset):,} (sequences with repeats, {sampling_mode} sampling)")
        log(f"  Train samples: {len(train_dataset):,} (sequences with repeats, {sampling_mode} sampling)")
    else:
        if input_length != 10:
            raise ValueError("permutation datasets are 10-digit only; use --allow_repeats --random_sampling for TaskMux models")
        eval_dataset = PermutationDataset(algorithm=args.algorithm, n_digits=input_length)
        train_dataset = PermutationDataset(algorithm=args.algorithm, n_digits=input_length)
        log(f"  Eval samples: {len(eval_dataset):,} (all permutations)")
        log(f"  Train samples: {len(train_dataset):,} (all permutations)")

    def make_loader(dataset, shuffle=True):
        sampler = DistributedSampler(dataset, shuffle=shuffle) if is_ddp else None
        return DataLoader(
            dataset,
            batch_size=rank_batch,
            sampler=sampler,
            shuffle=(sampler is None and shuffle),
            num_workers=DATALOADER_WORKERS,
            pin_memory=True,
            persistent_workers=DATALOADER_WORKERS > 0,
            prefetch_factor=2 if DATALOADER_WORKERS > 0 else None,
        )

    train_loader = make_loader(train_dataset, shuffle=True)
    eval_loader = make_loader(eval_dataset, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=hp.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    all_metrics = []
    training_start_time = time.time()
    best_eval_acc = -1.0  # -1 so epoch 0 always checkpoints, even at accuracy 0
    step = 0

    for epoch in range(epochs):
        if is_ddp:
            train_loader.sampler.set_epoch(epoch)

        train_metrics = train_epoch(
            model, train_loader, optimizer, scaler, device, epoch, use_amp, hp, step,
            input_length=input_length, output_length=output_length,
        )
        step = train_metrics["step"]

        eval_metrics = evaluate(
            model, eval_loader, device, use_amp, is_ddp,
            input_length=input_length, output_length=output_length,
        )
        # evaluate() is a STRICT all-token exact match — name it so downstream gates
        # (promote.SEQ_ACC_KEYS) never have to infer what "accuracy" meant.
        eval_acc = eval_metrics["sequence_accuracy"] = eval_metrics["accuracy"]

        is_best = eval_acc > best_eval_acc
        if is_best:
            best_eval_acc = eval_acc

        # 7 decimals to distinguish 1-3 wrong out of 3.6M
        num_wrong = int(round((1 - eval_acc) * len(eval_dataset)))
        log(
            f"Epoch {epoch} (step {step}/{hp.steps}): train_loss={train_metrics['loss']:.4f} "
            f"train_acc={train_metrics['accuracy']:.6f} "
            f"eval_acc={eval_acc:.7f} ({num_wrong} wrong) lr={optimizer.param_groups[0]['lr']:.1e}"
            + (" *BEST*" if is_best else "")
        )

        epoch_metrics = {
            "epoch": epoch,
            "step": step,
            "train": train_metrics,
            "eval": eval_metrics,
        }
        all_metrics.append(epoch_metrics)

        if is_best:
            best_filename = f"checkpoint_{algorithm_id}_best.pt" if algorithm_id else "checkpoint_best.pt"
            save_checkpoint(
                model, optimizer, scaler, epoch, epoch_metrics, args.output_dir, is_ddp,
                algorithm_id=algorithm_id, config=cfg, filename=best_filename, hp=hp,
            )

        if eval_acc >= args.target:
            log(f"Eval accuracy {eval_acc:.6f} >= {args.target}. Stopping early at epoch {epoch}.")
            break
        if step >= hp.steps:
            log(f"Step budget exhausted ({step}/{hp.steps}) at epoch {epoch}.")
            break

    # Post-training tripwire for the confirmed embedding divergence (hparams.check_we_norm).
    if is_main_process():
        w_e = (model.module if is_ddp else model).embed.W_E
        hparams.check_we_norm(float(w_e.detach().float().norm()), hp)

    if is_main_process():
        metrics_path = Path(args.output_dir) / "training_metrics.json"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, "w") as f:
            json.dump(all_metrics, f, indent=2)
        log(f"\nSaved training metrics to {metrics_path}")

    # Dump best-model predictions and validate them against the DSL ground truth
    # (10-digit models only: the digit-string CSV cannot encode task tokens >= 10;
    # TaskMux models are validated by the trainer's own exact-match eval above)
    if args.smoke:
        log("\nSmoke run: skipping the dump/validate subprocesses.")
    elif is_main_process() and algorithm_id and input_length != 10:
        log(f"\nSkipping dump/eval subprocesses: digit-string CSV is 10-digit only "
            f"(input_length={input_length}); trainer eval is authoritative here.")
    elif is_main_process() and algorithm_id:
        best_checkpoint = Path(args.output_dir) / f"checkpoint_{algorithm_id}_best.pt"
        assert best_checkpoint.exists(), f"no best checkpoint at {best_checkpoint}"
        test_data_path = Path(args.output_dir) / f"{algorithm_id}_test_data.csv"

        log("\nDumping model predictions...")
        cmd = [
            sys.executable,
            str(Path(__file__).parent / "dump_model_predictions.py"),
            "--checkpoint", str(best_checkpoint),
            "--num_samples", "10000",
            "--output", str(test_data_path),
            "--seed", str(hp.seed),
        ]
        if args.allow_repeats:
            cmd.append("--allow_repeats")
        subprocess.run(cmd, check=True)
        log(f"Dumped predictions: {test_data_path}")

        log("\nValidating predictions against ground truth algorithm...")
        validation_cmd = [
            sys.executable,
            str(Path(__file__).parent / "eval_test_data.py"),
            "--checkpoint", str(best_checkpoint),
            "--test_data", str(test_data_path),
            "--algorithm", args.algorithm,
            "--batch_size", "256",
        ]
        if output_length > 1:
            validation_cmd.append("--seq_to_seq")
        subprocess.run(validation_cmd, check=True)

    training_time = time.time() - training_start_time
    hours, remainder = divmod(training_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    log(f"\nTraining time: {int(hours)}h {int(minutes)}m {seconds:.1f}s")
    log(f"Best eval accuracy: {best_eval_acc:.7f}")

    if is_ddp:
        cleanup_ddp()

    log("Training complete!")


if __name__ == "__main__":
    main()
