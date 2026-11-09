#!/usr/bin/env python3
"""Stacked trainer: train every same-architecture toy model AT ONCE on one GPU
(vmap ensemble over a leading model dim), then split into individual drop-in
checkpoints. See docs/reference/stacked_training.md.

Buckets a model list by identical config, vmap-trains each bucket at SMALL batch
(the only regime where stacking beats sequential — see the POC), freezes each model
as it reaches the target, and writes byte-compatible trained_models_pt/<name>.pt.
Reuses the per-model trainer's data path (utils.SequenceDataset), model build
(model_loading.tier_config/segrev3_config), loss, and the teacher-forced seq-acc
that gates the pipeline (metrics.task_accuracy).

The RECIPE (lr / batch / steps / pool / warmup / grad-clip / dtype / eval size / seed) is
resolved from hparams.for_tier — the same table train.py reads — so a bucket's recipe is a
property of its architecture, not of this script.

  python train_stacked.py                          # models/models.txt -> SAFE out dir
  python train_stacked.py --models rcnt3_id dblodd_id --target 0.95
  python train_stacked.py --seeds 3                 # 3 inits per model (<name>.seed<k>.pt)
"""
import argparse
import dataclasses
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.func as tf
import torch.nn.functional as F

_THIS = Path(__file__).resolve().parent
for p in (str(_THIS.parents[2] / "interp_tools"), str(_THIS.parent), str(_THIS)):
    if p not in sys.path:
        sys.path.insert(0, p)

from model_loading import segrev3_config, tier_config           # noqa: E402
from transformer_lens import HookedTransformer                  # noqa: E402
from algorithms import decode_algorithm_id                      # noqa: E402
import hparams                                                  # noqa: E402
import promote as PROMOTE                                       # noqa: E402
import utils                                                    # noqa: E402

DEFAULT_OUT = str(_THIS.parent / "trained_models_pt_stacked")   # SAFE: never clobbers the real dir
REAL_OUT = str(_THIS.parent / "trained_models_pt")


class Spec:
    """Everything the stacker needs about one model: its DSL expr, config and recipe."""
    def __init__(self, name: str, seed: int = 0, override: dict | None = None):
        self.name = name
        self.base = name.partition("__")[0]      # task/data come from the base id; size is the variant
        self.hp = hparams.for_model(name, seed=seed)
        if override:                             # sweep arms only; see hparams.SWEEP_SPACE
            unknown = set(override) - {f.name for f in dataclasses.fields(self.hp)}
            if unknown:
                raise ValueError(f"unknown hparams override key(s): {sorted(unknown)}")
            self.hp = dataclasses.replace(self.hp, **override)
        self.expr = decode_algorithm_id(self.base, input_length=self.hp.input_len)
        self.input_length = self.expr.input_length
        self.output_length = self.expr.output_length()
        self.d_vocab = self.expr.vocab_size()
        self.config = tier_config(self.hp.tier, input_length=self.input_length,
                                  output_length=self.output_length, d_vocab=self.d_vocab)
        self.n_ctx = self.config["n_ctx"]

    def key(self) -> tuple:
        return tuple(sorted((k, tuple(v) if isinstance(v, list) else v) for k, v in self.config.items()))


def bucket(names: list, seeds: int = 1, base_seed: int = 0, override: dict | None = None) -> dict:
    """Group model names by identical config (the vmap-stacking invariant). With
    ``seeds`` > 1 every model enters its bucket once per seed — multi-seed is free here,
    since identical configs are exactly what vmap stacks."""
    specs = [Spec(n, seed=base_seed + k, override=override) for n in names for k in range(seeds)]
    groups = {}
    for s in specs:
        groups.setdefault(s.key(), []).append(s)
    return groups


def pool(spec: Spec, size: int, seed: int, device: str) -> torch.Tensor:
    """[size, n_ctx] rows [input | expr.eval(input)] via the per-model trainer's dataset."""
    ds = utils.SequenceDataset(algorithm=spec.base, num_samples=size, random_sampling=True, seed=seed)
    return torch.stack([ds[j] for j in range(size)]).to(device)


def make_stack(config: dict, n: int, seed: int, device: str) -> tuple:
    """N identical HookedTransformers stacked on a model dim; returns (fmodel, params, buffers, base)."""
    cfg = segrev3_config(config)
    models = []
    for i in range(n):
        torch.manual_seed(seed + i)
        models.append(HookedTransformer(cfg).to(device))
    params, buffers = tf.stack_module_state(models)
    base = models[0]                                            # functional_call template (values ignored)
    f = lambda p, b, x: tf.functional_call(base, (p, b), (x,))  # noqa: E731
    return tf.vmap(f, in_dims=(0, 0, 0)), params, buffers, base


def _split_logits(out: torch.Tensor, input_length: int):
    return out[:, :, input_length - 1:, :]                      # [N, B, OUT, V] teacher-forced positions


def loss_and_ce(fmodel, params, buffers, X, input_length, vocab) -> torch.Tensor:
    """Per-model MEAN cross-entropy -> [N] (each == that model's solo-training gradient)."""
    out = fmodel(params, buffers, X[:, :, :-1])
    olog = _split_logits(out, input_length)
    tgt = X[:, :, input_length:]
    ce = F.cross_entropy(olog.reshape(-1, vocab), tgt.reshape(-1), reduction="none")
    return ce.reshape(X.shape[0], -1).mean(dim=1)


#: (model x sequence) rows per eval forward — the eval peak scales with bucket x chunk, so
#: the CHUNK derives from the bucket and the peak is bounded whatever the bucket size. At xl
#: geometry a full pass over 84 stacked models x eval_n 65536 needs ~107GB for a SINGLE
#: residual tensor; this budget caps that tensor at ~0.3GB for any N.
EVAL_ROWS_PER_PASS = 16_384


@torch.no_grad()
def eval_acc(fmodel, params, buffers, Xe, input_length, live: torch.Tensor) -> torch.Tensor:
    """Teacher-forced strict all-token seq-acc -> [len(live)] (== metrics.task_accuracy) for
    the stack rows in ``live``; banked models are frozen, so their accuracy cannot move.
    Chunked over the eval axis; the running sum is exactly the unbatched mean."""
    chunk = max(1, EVAL_ROWS_PER_PASS // len(live))
    p, b = {k: v[live] for k, v in params.items()}, {k: v[live] for k, v in buffers.items()}
    hits = torch.zeros(len(live), device=Xe.device)
    for i in range(0, Xe.shape[1], chunk):
        xb = Xe[live, i:i + chunk]
        olog = _split_logits(fmodel(p, b, xb[:, :, :-1]), input_length)
        hits += (olog.argmax(-1) == xb[:, :, input_length:]).all(-1).float().sum(-1)
    return hits / Xe.shape[1]


def slice_state(base, params, i: int) -> dict:
    """Model i's state_dict, sliced out of the stack and detached to CPU."""
    base.load_state_dict({k: v[i] for k, v in params.items()}, strict=False)
    return {k: v.detach().cpu().clone() for k, v in base.state_dict().items()}


def ckpt_name(spec: Spec) -> str:
    """<name>.pt for the params-of-record (seed 0); <name>.seed<k>.pt for extra seeds,
    so a multi-seed run never shadows the registry's <name>.pt."""
    return f"{spec.name}.pt" if spec.hp.seed == 0 else f"{spec.name}.seed{spec.hp.seed}.pt"


def save_split(state: dict, spec: Spec, acc: float, step: int, out_dir: str) -> str:
    """Write model i's byte-compatible checkpoint, recording the STRICT seq-acc, the seed
    and the resolved recipe (so promote.py can gate it and a rerun can reproduce it)."""
    ckpt = {"config": dict(spec.config), "model_state_dict": state, "algorithm_id": spec.base,
            "epoch": step, "metrics": {"epoch": step, "step": step, "seed": spec.hp.seed,
                                       "hparams": spec.hp.as_dict(),
                                       "eval": {"accuracy": acc, "sequence_accuracy": acc}}}
    path = os.path.join(out_dir, ckpt_name(spec))
    torch.save(ckpt, path)
    return path


def clip_per_model(params: dict, clip: float) -> torch.Tensor:
    """Clip each STACKED model's gradient to `clip` INDEPENDENTLY (a global norm over the
    stack would let one diverging model shrink every peer's step). Returns per-model norms."""
    grads = [p.grad for p in params.values() if p.grad is not None]
    norms = sum(g.reshape(g.shape[0], -1).pow(2).sum(1) for g in grads).sqrt()
    scale = (clip / (norms + 1e-6)).clamp(max=1.0)
    for g in grads:
        g.mul_(scale.view(-1, *([1] * (g.dim() - 1))))
    return norms


def train_bucket(specs: list, args, device: str) -> list:
    """vmap-train a homogeneous bucket; BANK each model the moment it reaches the target and
    HOLD its weights there for the rest of the run (a masked loss alone does not freeze a
    model: AdamW's decoupled weight decay keeps shrinking a zero-gradient parameter); then
    save only the models that reached it. Returns [(name, acc, passed)]."""
    n, config, hp = len(specs), specs[0].config, specs[0].hp
    in_len, vocab = specs[0].input_length, specs[0].d_vocab
    amp = hp.dtype == "bf16"
    print(f"\n[bucket] {n} model(s) @ {config['n_layers']}L/{config['n_heads']}H "
          f"d{config['d_model']} n_ctx{config['n_ctx']} attn_only={config['attn_only']} "
          f"· tier {hp.tier} lr {hp.lr:g} batch {hp.batch} steps {hp.steps} "
          f"warmup {hp.warmup} clip {hp.grad_clip} {hp.dtype}")
    train_pools = [pool(s, hp.pool, s.hp.seed + 1 + i, device) for i, s in enumerate(specs)]
    Xe = torch.stack([pool(s, hp.eval_n, s.hp.seed + 10_000 + i, device)
                      for i, s in enumerate(specs)])

    fmodel, params, buffers, base = make_stack(config, n, hp.seed, device)
    opt = torch.optim.AdamW(list(params.values()), lr=hp.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    active = torch.ones(n, device=device)      # 0 once a model is banked — it never trains again
    g = torch.Generator(device=device).manual_seed(hp.seed)
    acc = torch.zeros(n)
    banked: dict = {}                          # model index -> (state_dict, acc, step) at the pass
    held_idx = torch.zeros(0, dtype=torch.long, device=device)
    held: dict = {}                            # banked models' rows, rewritten after every step
    t0 = time.time()
    for step in range(1, hp.steps + 1):
        idx = torch.randint(0, hp.pool, (n, hp.batch), generator=g, device=device)
        X = torch.stack([train_pools[i][idx[i]] for i in range(n)])
        for grp in opt.param_groups:           # linear warmup, shared with train.py's rule
            grp["lr"] = hp.lr * min(1.0, step / hp.warmup) if hp.warmup else hp.lr
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            per_model = loss_and_ce(fmodel, params, buffers, X, in_len, vocab)
            loss = (per_model * active).sum()  # banked (active=0) models contribute no gradient
        scaler.scale(loss).backward()
        if hp.grad_clip:
            scaler.unscale_(opt)
            clip_per_model(params, hp.grad_clip)
        scaler.step(opt)
        scaler.update()
        if held:                               # undo the optimizer for banked models: a zero
            with torch.no_grad():              # gradient still moves them under AdamW decay
                for k, p in params.items():
                    p[held_idx] = held[k]
        if step % hp.eval_every == 0 or step == hp.steps:
            live = (active > 0).nonzero(as_tuple=True)[0]
            acc[live.cpu()] = eval_acc(fmodel, params, buffers, Xe, in_len, live).cpu()
            for i in live.tolist():
                if float(acc[i]) >= args.target:
                    banked[i] = (slice_state(base, params, i), float(acc[i]), step)
                    active[i] = 0.0
            if len(banked) > len(held_idx):
                held_idx = torch.tensor(sorted(banked), device=device)
                held = {k: v[held_idx].detach().clone() for k, v in params.items()}
            print(f"  step {step:6}  banked {len(banked)}/{n}  min {acc.min():.4f}  "
                  f"mean {acc.mean():.4f}  ({time.time()-t0:.0f}s)", flush=True)
            if len(banked) == n:
                break
    dt = time.time() - t0
    os.makedirs(args.out, exist_ok=True)
    out = []
    for i, s in enumerate(specs):
        state, a, at = banked.get(i) or (None, float(acc[i]), step)
        if state is None:                      # never reached the target -> NOT a checkpoint
            print(f"  [blocked ] {s.name:40} seq-acc={a:.5f} < target {args.target} "
                  f"— no checkpoint written", flush=True)
            out.append((s.name, a, False))
            continue
        w_e_norm = float(state["embed.W_E"].float().norm())
        hparams.check_we_norm(w_e_norm, s.hp)
        save_split(state, s, a, at, args.out)
        print(f"  [saved   ] {s.name:40} seq-acc={a:.5f} @ step {at}  "
              f"||W_E||={w_e_norm:.3f}", flush=True)
        out.append((s.name, a, True))
    print(f"[bucket] done in {dt:.0f}s -> {args.out}")
    return out


def measure_pack_size(config: dict, batch: int, device: str, cap: int = 64) -> int:
    """Micro-bench the vmap pack size (models/GPU) that MAXIMIZES throughput for this
    exact config + batch + GPU — so N* (and the GPU count) derive from the model and
    hardware, never a constant. Returns argmax_N of models-per-second."""
    best_n, best_thru = 1, 0.0
    for n in (1, 2, 4, 8, 16, 24, 32, 48, 64):
        if n > cap:
            break
        try:
            fmodel, params, buffers, _ = make_stack(config, n, 0, device)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            break
        X = torch.randint(0, config["d_vocab"], (n, batch, config["n_ctx"] - 1), device=device)
        opt = torch.optim.AdamW(list(params.values()), lr=1e-3)
        def step():
            opt.zero_grad(set_to_none=True)
            fmodel(params, buffers, X).float().pow(2).mean().backward(); opt.step()
        for _ in range(8):
            step()
        torch.cuda.synchronize(); t0 = time.time()
        for _ in range(30):
            step()
        torch.cuda.synchronize()
        thru = n / ((time.time() - t0) / 30)
        if thru > best_thru:
            best_n, best_thru = n, thru
        del fmodel, params, buffers, opt, X
        torch.cuda.empty_cache()
    return best_n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--models", nargs="*", help="model names (default: models/models.txt)")
    ap.add_argument("--out", default=DEFAULT_OUT,
                    help="output dir; NEVER the real tree — promotion goes through promote_grid.py")
    ap.add_argument("--target", type=float, default=PROMOTE.GATE,
                    help="seq-acc a model must reach to be banked AND written (default %(default)s)")
    ap.add_argument("--seed", type=int, default=0, help="base seed (recipe seed 0 = params-of-record)")
    ap.add_argument("--hparams-override", default="",
                    help='JSON dict overriding hparams.for_model keys for a SWEEP arm, e.g. '
                         '\'{"lr": 2.5e-4, "warmup": 0, "grad_clip": 0.0}\' — sweeps only, '
                         'never a way to ship a recipe that is not in the table')
    ap.add_argument("--seeds", type=int, default=1,
                    help="train N inits per model (seeds base..base+N-1); extra seeds land in "
                         "<name>.seed<k>.pt so the registry's <name>.pt is untouched")
    a = ap.parse_args()
    if a.models is None:
        txt = _THIS.parent / "models.txt"
        a.models = [ln.split("#", 1)[0].strip() for ln in open(txt) if ln.split("#", 1)[0].strip()]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    override = json.loads(a.hparams_override) if a.hparams_override else None
    groups = bucket(a.models, seeds=a.seeds, base_seed=a.seed, override=override)
    print(f"device={device} torch={torch.__version__}  {len(a.models)} model(s) x {a.seeds} seed(s) "
          f"-> {len(groups)} bucket(s)  out={a.out}")
    results = []
    for specs in sorted(groups.values(), key=len, reverse=True):
        results += train_bucket(specs, a, device)

    print("\n=== SUMMARY (target seq-acc >= {:.3f}) ===".format(a.target))
    npass = sum(p for _, _, p in results)
    for name, acc, passed in sorted(results, key=lambda r: -r[1]):
        print(f"  {'PASS' if passed else 'fail'}  {name:22} {acc:.4f}")
    print(f"{npass}/{len(results)} models reached target (keepers); "
          f"{len(results)-npass} need the per-model escalation fallback")


if __name__ == "__main__":
    main()
