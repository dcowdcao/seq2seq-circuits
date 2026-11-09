#!/usr/bin/env python3
"""Per-architecture training hyperparameters — ONE table, read by BOTH training paths.

The recipe is a property of the ARCHITECTURE, not of which script runs: train_stacked (the
vmap size grid — small/large/wide/deep/xl) and run_training_job.sh (the per-model DDP path —
every mlp/moe4/moe8) both resolve their recipe from `for_tier` / `for_model` here and gate on
the same eval size. (Earlier the two paths used different learning rates, batch sizes, step
budgets, dtypes and gate eval sizes, which confounded every attn-only-vs-feedforward
comparison.)

LR rule (stated so it can be argued with): depth-and-width-scaled from a measured-healthy
anchor rather than a constant. lr = ANCHOR_LR * sqrt(ANCHOR_D/d_model) * sqrt(ANCHOR_L/n_layers).
The anchor is `large` (4L, d128, batch 64, lr 1e-3), where nearly every size-grid run reached
the gate. The 1/sqrt(d_model) half is µP-adjacent (Adam LR ~ 1/fan_in^0.5 for the
matrix-like parameters); the 1/sqrt(n_layers) half is the depth-µP residual-branch scaling.
It is a DEFAULT, not a measured optimum — see SWEEP_SPACE.

Why width matters: at xl, under the one anchor recipe every architecture used to share, a
large fraction of runs DIVERGED, and the divergence is diagnosable from the weights alone —
the dead runs' ||embed.W_E|| sits well above the live runs', with no overlap. Embeddings that
GREW is a learning rate too high for the width, not a capacity limit, so a width-scaled LR is
the fix for a width-dependent divergence.

The tripwire (`check_we_norm`) is therefore xl-SPECIFIC and deliberately not global: at the
other attn-only tiers the same statistic does NOT separate dead from live runs, so one global
constant would fail healthy runs and pass dead ones. Tiers without a calibrated bound say so
out loud instead of guessing.
"""
from __future__ import annotations

import argparse
import ast
import math
import os
from dataclasses import asdict, dataclass, replace

#: bump when ANY value below changes; recorded in every checkpoint as metrics.hparams.regime
#: so a checkpoint trained under one recipe is never silently compared against another.
REGIME = "perarch-v1"

#: measured-healthy anchor for the LR rule (see module docstring).
ANCHOR_LR, ANCHOR_D_MODEL, ANCHOR_N_LAYERS = 1e-3, 128, 4

#: shared budget: the vmap size grid's regime.
#: BATCH is the EFFECTIVE batch — the DDP path splits it across ranks (per_rank_batch), so
#: a 4-GPU host trains the same recipe as a 1-GPU one instead of 4x it.
STEPS, BATCH, POOL, EVAL_EVERY = 20_000, 64, 32_768, 1_000
#: gate eval size — ONE value for every arch, so the 0.95 gate means the same thing
#: everywhere (sigma ~ 0.00085 at p=0.95; the old 4096-vs-1e6 split was ~16x apart).
EVAL_N = 65_536
#: tier -> ||embed.W_E|| bound above which the run has diverged (see check_we_norm).
#: CALIBRATED PER TIER from raw size-grid runs, and only xl separates dead from live runs;
#: 4.76 is the midpoint of the gap between them. No other tier's dead runs are
#: distinguishable this way, so they get no bound rather than a borrowed one.
WE_NORM_MAX = {"xlarge": 4.76}
#: THE sequence-length knob for the training side. n_ctx = input_len + output_length, and
#: almost every checkpoint sits at n_ctx 20 because the number used to be hardcoded in five
#: places under train_models/. It is now one constant, carried on every HParams and recorded
#: in every checkpoint, so length is a config value rather than an assumption.
#: Sweeping it needs FOUR more surfaces to take it, all outside train_models/ and all owned
#: elsewhere: algorithms.DEFAULT_INPUT_LENGTH / decode_algorithm_id's default (samplers and
#: the id round-trip), algorithms TaskMux member decoding (pinned at 10 independently),
#: model_registry.ModelSpec.input_len / _INPUT_LEN (answer_positions, every analysis), and
#: model_loading.INPUT_LEN (tier_config's n_ctx). Until those thread it through, changing
#: this alone trains a model the analysis side cannot read.
INPUT_LEN = 10

#: model-name suffix -> trainer tier. THE mapping, imported by both training paths and the
#: orchestrator (it used to be copy-pasted in three places).
VARIANT_TIER = {"": "large", "xl": "xlarge", "wide": "wide", "small": "small",
                "deep": "deep", "mlp": "mlp", "moe4": "moe4", "moe8": "moe8"}
#: the 8 architectures, in report order.
ARCHES = ("small", "large", "wide", "deep", "xl", "mlp", "moe4", "moe8")

#: what a sweep must settle. These are NOT guesses to ship as final values — the LR rule
#: fixes the SHAPE of the curve across tiers; the sweep fixes its height, and whether the
#: two d256 tiers need more than the rule gives them.
SWEEP_SPACE = {
    "lr_multiplier": [0.25, 0.5, 1.0, 2.0],      # multiplies the rule's per-tier lr
    "warmup": [0, 200, 1000],                    # steps of linear warmup (neither path had any)
    "grad_clip": [0.0, 1.0],                     # 0 = off (neither path clipped)
    "steps": [20_000, 60_000],                   # mlp/moe previously saw ~23x more data
    "batch": [64, 256],                          # 64 is the vmap-throughput regime, not a measurement
    "dtype": ["fp32", "bf16"],                   # the two regimes already in the corpus
}
#: tiers where the shipped default is least trustworthy, so a sweep should start here.
#: xlarge is now MEASURED (see below); wide is not.
SWEEP_PRIORITY = ("wide",)
#: tiers whose `lr` is PROVISIONAL pending a measured sweep — printed as such so nobody
#: reads the rule's output as a measurement. The recipe sweep only ever covered xl, and wide is
#: the other d256 tier, so its lr is still whatever the rule says.
LR_PENDING_MEASUREMENT = ("wide",)
#: xlarge MEASURED by a recipe sweep on diverged xl models: halving the LR to the rule's 5e-4
#: revived most of them; warmup + gradient clipping alone revived fewer, and only models the
#: halved LR also revived, so stabilisation is a weaker version of the same mechanism, not an
#: independent fix. A lower LR did not beat the rule beyond noise, so the rule stands.
LR_MEASURED = {"xlarge": "recipe sweep: the rule's 5e-4 revives most anchor-lr divergences"}


@dataclass(frozen=True)
class HParams:
    """One architecture's complete training recipe. Recorded verbatim into every checkpoint."""
    tier: str
    n_layers: int
    n_heads: int
    d_model: int
    lr: float
    batch: int
    steps: int
    pool: int
    eval_every: int
    eval_n: int
    warmup: int
    grad_clip: float
    dtype: str
    seed: int
    we_norm_max: float | None      # None = no calibrated divergence bound at this tier
    input_len: int
    regime: str

    def as_dict(self) -> dict:
        return asdict(self)


MODEL_LOADING = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "interp_tools", "model_loading.py"))
_PRESETS: dict = {}


def _presets() -> dict:
    """TIER_PRESETS from interp_tools/model_loading.py — the single source of truth for
    geometry. Read with `ast` instead of imported: model_loading pulls torch, and the
    orchestrator imports this module at module level (its --dry-run must stay torch-free).
    Fails loud if TIER_PRESETS ever stops being a literal."""
    if not _PRESETS:
        tree = ast.parse(open(MODEL_LOADING).read())
        node = next(n.value for n in tree.body if isinstance(n, ast.Assign)
                    and any(getattr(t, "id", None) == "TIER_PRESETS" for t in n.targets))
        _PRESETS.update(ast.literal_eval(node))
    return _PRESETS


def lr_for(d_model: int, n_layers: int) -> float:
    """The LR rule: anchor scaled by 1/sqrt(width) and 1/sqrt(depth)."""
    scale = math.sqrt(ANCHOR_D_MODEL / d_model) * math.sqrt(ANCHOR_N_LAYERS / n_layers)
    return float(f"{ANCHOR_LR * scale:.3g}")


def for_tier(tier: str, seed: int = 0) -> HParams:
    """The recipe for a trainer tier (a TIER_PRESETS key). Fails loud on an unknown tier."""
    g = _presets()[tier]
    return HParams(tier=tier, n_layers=g["n_layers"], n_heads=g["n_heads"], d_model=g["d_model"],
                   lr=lr_for(g["d_model"], g["n_layers"]), batch=BATCH, steps=STEPS, pool=POOL,
                   eval_every=EVAL_EVERY, eval_n=EVAL_N, warmup=200, grad_clip=1.0, dtype="fp32",
                   seed=seed, we_norm_max=WE_NORM_MAX.get(tier), input_len=INPUT_LEN,
                   regime=REGIME)


def tier_of(name: str) -> str:
    """Trainer tier for a model name ('foo_id__wide' -> 'wide', 'foo_id' -> 'large')."""
    return VARIANT_TIER[name.partition("__")[2]]


def for_model(name: str, seed: int = 0) -> HParams:
    """The recipe for a model name — the entry point both training paths call, so the
    recipe is a property of the architecture and not of which script ran."""
    return for_tier(tier_of(name), seed=seed)


def smoke() -> HParams:
    """Throwaway budget for --smoke (a dev loop, never a promoted checkpoint)."""
    return replace(for_tier("tiny"), steps=20, pool=512, eval_every=10, eval_n=512, warmup=0)


def check_we_norm(w_e_norm: float, hp: HParams) -> None:
    """Post-training tripwire for the confirmed embedding divergence (see module docstring).
    A LARGE norm is the failure: dead xl runs sit in [5.13, 7.74], live ones in [1.94, 4.40].
    Raises — a run past the bound is an optimiser failure, not a weak model."""
    if hp.we_norm_max is None:
        print(f"    ||embed.W_E|| = {w_e_norm:.3f} (no calibrated bound at tier {hp.tier} — "
              f"the statistic separates dead from live runs only at xl)", flush=True)
        return
    if w_e_norm > hp.we_norm_max:
        raise SystemExit(
            f"✗ W_E TRIPWIRE: ||embed.W_E|| = {w_e_norm:.3f} > {hp.we_norm_max} at tier "
            f"{hp.tier} (lr={hp.lr:g}). The embeddings GREW — the divergence signature of an "
            f"lr too high for this width (76 of 176 xl grid runs, dead median 6.59 vs live "
            f"2.15), not a capacity limit. Lower this tier's lr in hparams.py or raise its "
            f"warmup; do NOT promote this checkpoint.")


def per_rank_batch(hp: HParams, world_size: int) -> int:
    """Per-rank batch: hp.batch is the EFFECTIVE batch, so DDP splits it across ranks and the
    recipe is a property of the architecture, not of how many GPUs the host happens to have."""
    if hp.batch % world_size:
        raise SystemExit(f"✗ effective batch {hp.batch} (tier {hp.tier}) is not divisible by "
                         f"world_size {world_size} — launch torchrun with a GPU count that "
                         f"divides it, or change BATCH in hparams.py.")
    return hp.batch // world_size


def epochs_for(hp: HParams) -> int:
    """Epochs that spend exactly `steps` optimizer steps over a `pool`-row dataset. World-size
    free: the ranks split the effective batch, so steps/epoch is the same at any GPU count."""
    return max(1, math.ceil(hp.steps / max(1, hp.pool // hp.batch)))


def table(seed: int = 0) -> list:
    """The resolved recipe for every architecture, in report order."""
    return [for_tier(VARIANT_TIER[s] if s in VARIANT_TIER else s, seed=seed)
            for s in ("small", "", "wide", "deep", "xl", "mlp", "moe4", "moe8")]


def print_table(seed: int = 0) -> None:
    """Print the resolved recipe for all 8 architectures + the sweep space."""
    cols = ("arch", "tier", "L", "H", "d_model", "lr", "batch", "steps", "pool",
            "eval_n", "warmup", "clip", "dtype", "seed", "|W_E|>", "in_len")
    w = [9] * len(cols)
    w[5] = 11                                   # lr column carries the provisional '*'
    print(f"resolved per-architecture HP table  ·  regime {REGIME}  ·  "
          f"lr = {ANCHOR_LR:g} * sqrt({ANCHOR_D_MODEL}/d_model) * sqrt({ANCHOR_N_LAYERS}/n_layers)")
    print("  " + "".join(f"{c:>{n}}" for c, n in zip(cols, w)))
    for a, hp in zip(ARCHES, table(seed)):
        lr = f"{hp.lr:.2e}" + ("*" if hp.tier in LR_PENDING_MEASUREMENT else "")
        row = (a, hp.tier, hp.n_layers, hp.n_heads, hp.d_model, lr, hp.batch,
               hp.steps, hp.pool, hp.eval_n, hp.warmup, hp.grad_clip, hp.dtype, hp.seed,
               hp.we_norm_max if hp.we_norm_max else "-", hp.input_len)
        print("  " + "".join(f"{str(v):>{n}}" for v, n in zip(row, w)))
    if LR_PENDING_MEASUREMENT:
        print(f"\n  * = lr PROVISIONAL, pending a measured sweep: {LR_PENDING_MEASUREMENT}")
    for tier, note in LR_MEASURED.items():
        print(f"\n  {tier} lr MEASURED — {note}")
    print(f"  |W_E|> = divergence tripwire; calibrated only at xl; '-' = no calibrated "
          f"bound, norm logged not gated")
    print(f"  sweep search space (values NOT settled by the rule): {SWEEP_SPACE}")
    print(f"  sweep first at: {SWEEP_PRIORITY} — d_model=256, same width as xl, whose anchor-lr "
          f"divergence the sweep traced to the LR")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tier", help="emit one tier's recipe (a TIER_PRESETS key)")
    ap.add_argument("--model", help="emit one model name's recipe (resolves the __<suffix> tier)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sh", action="store_true",
                    help="emit HP_<FIELD>=<value> lines for `eval \"$(hparams.py --tier T --sh)\"`")
    ap.add_argument("--table", action="store_true", help="print the resolved table for all 8 arches")
    a = ap.parse_args()
    if a.table or not (a.tier or a.model):
        print_table(a.seed)
        return
    hp = for_model(a.model, seed=a.seed) if a.model else for_tier(a.tier, seed=a.seed)
    if a.sh:
        for k, v in hp.as_dict().items():
            print(f"HP_{k.upper()}={v}")
        return
    for k, v in hp.as_dict().items():
        print(f"{k:12} {v}")


if __name__ == "__main__":
    main()
