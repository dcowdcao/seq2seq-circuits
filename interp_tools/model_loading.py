"""
model_loading.py
=================================================================================
Canonical loader for the `segrev3` checkpoint into a TransformerLens
``HookedTransformer``. This is the SINGLE source of truth for how the model is
built; every other module (interp tools, weight tools, ACDC / EAP-IG
integrations, validation) imports ``load_segrev3`` from here so they can never
disagree about the architecture.

--------------------------------------------------------------------------------
GROUND TRUTH (verified against the original training code and empirically
against the checkpoint):

  * attention-only (no MLP), 4 layers, 8 heads, d_model=128, d_head=32,
    d_vocab=10 (digits 0-9, NO BOS/SEP token), n_ctx=20.
  * **Pre-LayerNorm with a PARAMETER-FREE LayerNormPre** (center + divide by
    RMS, eps=1e-5) before every attention block AND a final ``ln_final`` before
    the unembed. => ``normalization_type="LNPre"``.
  * CAUSAL attention, scale = sqrt(d_head) = sqrt(32).
  * learned absolute positional embeddings (W_pos), and biases
    b_Q / b_K / b_V / b_O / b_U are all present.

WHY THIS MATTERS: the checkpoint's state-dict has NO layernorm parameters
(LayerNormPre has none), so on disk it is indistinguishable from a model with
``normalization_type=None``. But loading it with ``None`` SKIPS the
center+normalize step and the model breaks completely -- it emits a constant
(a copy of the token at position 0) at every position with confidence ~1.0 and
scores at chance on the real task. Loading with ``"LNPre"`` restores
token-accuracy 0.99995 / sequence-accuracy 0.99950 (the checkpoint reports
0.99964). An earlier version of this codebase loaded with ``None``; every
numeric result produced before the fix is invalid.

CONSEQUENCES of the LN being present (correcting the old "LayerNorm-free"
claims): the residual stream is RMS-normalized per position before each
Q/K/V read and before the unembed, so it is NOT a globally linear function of
the component outputs. Therefore:
  * Direct logit attribution is additive only with the standard *frozen final-LN
    scale* treatment (``apply_ln=True`` in the resid decomposition, which uses
    the cached ``ln_final`` scale as a constant). See
    ``patching_dla.direct_logit_attribution``.
  * The weight-based QK / OV / composition circuits (W_E W_Q W_K^T W_E^T,
    W_E W_V W_O W_U, ...) are the usual Elhage *approximations* modulo the
    per-position LN scale -- not exact. Confirm them causally with patching.
  * Activation patching and path patching remain EXACT (they run the real
    forward pass, LN included).

--------------------------------------------------------------------------------
TASK / DATA convention (see ``data_gen/data_gen.py`` and ``metrics/metrics.py``):
  * input = 10 uniform-random digits 0-9 (repeats allowed).
  * the model sees the 20-token sequence ``[input(10) | segrev3(input)(10)]``.
  * output digit ``y_k`` is predicted at logit position ``9 + k`` (next-token
    over the concatenation), so the ANSWER POSITIONS are 9..18.
"""

from __future__ import annotations

import os
from typing import Optional

import torch

from transformer_lens import HookedTransformer, HookedTransformerConfig


# --------------------------------------------------------------------------- #
# Hardware backend tuning (SINGLE source of truth, applied once at import).    #
# --------------------------------------------------------------------------- #
# This module is imported by EVERY stage of the pipeline, so configuring the
# torch backends here makes the whole pipeline agree on one set of flags. On GPUs
# with TF32/BF16 tensor cores torch leaves TF32 matmul OFF by default; enabling it (and the high matmul precision)
# speeds every matmul at no accuracy cost that matters for this analysis (and
# DLA/patching numerics are dominated by the LN-frozen decomposition, not by the
# last few mantissa bits). Idempotent and safe on CPU-only boxes.
def _configure_torch_backend() -> None:
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True          # fixed shapes -> autotune once
        # "high" lets matmul use TF32 tensor cores while keeping fp32 accumulation.
        # This changes results only at ~1e-4 (discrete outputs -- ACDC edges, DLA
        # exactness, top-n selections -- are unaffected) for a measured ~20% matmul
        # speed-up on a tensor-core GPU.
        torch.set_float32_matmul_precision("high")
        # Use all cores for CPU-side ops -- BUT respect an explicit OMP_NUM_THREADS
        # (e.g. main.py --parallel caps it so concurrent jobs share the cores).
        if not os.environ.get("OMP_NUM_THREADS"):
            n = os.cpu_count() or 1
            if torch.get_num_threads() < n:
                torch.set_num_threads(n)
    except Exception:
        # Never let backend tuning break a load on an unusual platform.
        pass


_configure_torch_backend()


# --------------------------------------------------------------------------- #
# Paths / constants                                                           #
# --------------------------------------------------------------------------- #

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(_THIS_DIR)
DEFAULT_CKPT = os.path.join(REPO_ROOT, "segrev3", "segrev3_id_checkpoint.pt")


def repo_relpath(path):
    """An ABSOLUTE ``path`` made relative to the repo root (None/"" and already-relative
    paths pass through) -- the form every persisted artifact records, so results are
    machine-independent."""
    return os.path.relpath(path, REPO_ROOT) if path and os.path.isabs(path) else path

# --------------------------------------------------------------------------- #
# GPU memory safety rails.                                                    #
# --------------------------------------------------------------------------- #
# On GPUs with shared or limited memory, a single runaway CUDA allocation (or
# too many concurrent workers) can exhaust the device or the host. Two rails,
# both env-tunable:
#
#   * apply_gpu_memory_cap(): per-process ceiling on torch CUDA allocations,
#     applied inside load_segrev3 (the single chokepoint every pipeline process
#     loads the model through). Exceeding the cap raises torch.OutOfMemoryError
#     in THAT process instead of consuming the machine. SEGREV3_GPU_MEM_FRACTION overrides the default; <= 0 disables.
#
#   * assert_memory_headroom(): loud pre-flight check before spawning GPU
#     workers -- refuses to launch into an already-starved machine. Only
#     enforced when CUDA is available (a no-op on CPU-only runs).
#     SEGREV3_MIN_FREE_GIB overrides the floor; <= 0 disables.

DEFAULT_GPU_MEM_FRACTION = 0.125     # per-process share of total GPU memory
DEFAULT_MIN_FREE_GIB = 24.0


def _env_float(var: str, default: float) -> float:
    try:
        return float(os.environ.get(var, "") or default)
    except ValueError:
        return default


def apply_gpu_memory_cap(device: Optional[str] = None) -> Optional[float]:
    """Cap THIS process's torch CUDA allocations at a fraction of total memory.
    Returns the fraction applied, or None when disabled / not on cuda."""
    if device is not None and not str(device).startswith("cuda"):
        return None
    if not torch.cuda.is_available():
        return None
    frac = _env_float("SEGREV3_GPU_MEM_FRACTION", DEFAULT_GPU_MEM_FRACTION)
    if frac <= 0:
        return None
    frac = min(frac, 1.0)
    torch.cuda.set_per_process_memory_fraction(frac)
    return frac


def assert_memory_headroom(min_free_gib: Optional[float] = None,
                           context: str = "") -> float:
    """Raise RuntimeError when system MemAvailable is below the floor (CUDA
    runs only; CPU-only runs never raise). Returns the available GiB otherwise
    (0.0 when /proc/meminfo is absent)."""
    if min_free_gib is None:
        min_free_gib = _env_float("SEGREV3_MIN_FREE_GIB", DEFAULT_MIN_FREE_GIB)
    try:
        with open("/proc/meminfo") as f:
            avail_kb = next(int(line.split()[1]) for line in f
                            if line.startswith("MemAvailable:"))
    except (OSError, StopIteration):
        return 0.0                      # non-Linux: nothing to check
    avail_gib = avail_kb / 2**20
    if min_free_gib > 0 and avail_gib < min_free_gib and torch.cuda.is_available():
        where = f" ({context})" if context else ""
        raise RuntimeError(
            f"memory headroom check failed{where}: only {avail_gib:.1f} GiB "
            f"available (< {min_free_gib:.0f} GiB floor). On GPUs that share "
            "memory with the host, launching now risks exhausting it. Free memory first, or lower/disable "
            "the floor via SEGREV3_MIN_FREE_GIB.")
    return avail_gib


# Task geometry (kept here so every module agrees).
INPUT_LEN = 10
OUTPUT_LEN = 10
SEQ_LEN = INPUT_LEN + OUTPUT_LEN          # 20 == n_ctx
BOUNDARY_DIGIT = 3
# logit position p predicts the token at p+1; output digit y_k sits at sequence
# position INPUT_LEN + k, so it is predicted at position INPUT_LEN - 1 + k.
FIRST_ANSWER_POS = INPUT_LEN - 1          # 9  (predicts y_0)
LAST_ANSWER_POS = SEQ_LEN - 2             # 18 (predicts y_9)
ANSWER_POSITIONS = list(range(FIRST_ANSWER_POS, LAST_ANSWER_POS + 1))  # [9..18]


def resolve_device(device: Optional[str] = None) -> str:
    """Resolve the compute device for the whole pipeline (SINGLE source of truth).

    Precedence: an explicit ``device`` arg  >  the ``SEGREV3_DEVICE`` env var
    (set once by the orchestrator so every child subprocess agrees)  >  auto
    (``cuda`` when a GPU is visible, else ``cpu``). The literal string
    ``"auto"`` (from a CLI default) is treated the same as ``None``.

    Both the model loader AND the data loaders call this, so the model and its
    input tensors always land on the same device without per-call-site wiring.
    """
    if device is not None and device != "auto":
        return device
    env = os.environ.get("SEGREV3_DEVICE")
    if env and env != "auto":
        return env
    return "cuda" if torch.cuda.is_available() else "cpu"


# Backwards-compatible private alias (older call sites import the underscored name).
_resolve_device = resolve_device


def resolve_checkpoint(checkpoint_path: Optional[str] = None) -> str:
    """Resolve the model checkpoint path (SINGLE source of truth, like
    ``resolve_device``). Precedence: an explicit ``checkpoint_path`` arg  >  the
    ``SEGREV3_CKPT`` env var (set once by an orchestrator so every child
    subprocess loads the same model)  >  the canonical ``DEFAULT_CKPT``."""
    if checkpoint_path:
        return checkpoint_path
    return os.environ.get("SEGREV3_CKPT") or DEFAULT_CKPT


def segrev3_config(ckpt_config: dict) -> HookedTransformerConfig:
    """Build the TransformerLens config from the checkpoint's stored ``config``
    dict. ``normalization_type="LNPre"`` is the load-bearing choice (see module
    docstring).

    Attention-only is the DEFAULT (segrev3 and every model whose config omits
    ``attn_only``), so existing checkpoints are byte-for-byte unchanged. A config
    with ``attn_only=False`` additionally reads ``d_mlp`` (default 4*d_model) and
    ``act_fn`` (default "gelu"); with ``num_experts``/``experts_per_token`` also
    set it builds a TL MoE transformer (per-expert width = ``d_mlp``)."""
    c = ckpt_config
    attn_only = bool(c.get("attn_only", True))
    kw = dict(
        n_layers=c["n_layers"],
        n_heads=c["n_heads"],
        d_model=c["d_model"],
        d_head=c["d_head"],
        d_vocab=c["d_vocab"],
        n_ctx=c["n_ctx"],
        attn_only=attn_only,
        normalization_type=c.get("normalization_type", "LNPre"),
        positional_embedding_type="standard",
        tokenizer_name=None,
        # the original model centers+normalizes with eps=1e-5
        eps=c.get("eps", 1e-5),
    )
    if attn_only:
        kw["act_fn"] = None
    else:
        kw["act_fn"] = c.get("act_fn") or "gelu"
        kw["d_mlp"] = int(c["d_mlp"]) if c.get("d_mlp") else 4 * int(c["d_model"])
        if c.get("num_experts"):
            # TL 3.3.0 MoE (components/mlps/moe.py): softmax-top-k routed gated experts
            kw["num_experts"] = int(c["num_experts"])
            kw["experts_per_token"] = int(c["experts_per_token"])
    return HookedTransformerConfig(**kw)


# --------------------------------------------------------------------------- #
# Trainer tier presets (--config names). tier_config() finalizes a preset into #
# the ckpt-config dict that save_checkpoint persists and segrev3_config reads, #
# so trainer, registry, and loader can never disagree about an architecture.   #
# --------------------------------------------------------------------------- #
#: Geometry (+ arch flags) per tier.
TIER_PRESETS = {
    "tiny":   {"n_layers": 2, "n_heads": 4,  "d_model": 64,  "d_head": 16},
    "small":  {"n_layers": 2, "n_heads": 4,  "d_model": 64,  "d_head": 16},
    "large":  {"n_layers": 4, "n_heads": 8,  "d_model": 128, "d_head": 32},
    "deep":   {"n_layers": 8, "n_heads": 8,  "d_model": 128, "d_head": 32},
    "wide":   {"n_layers": 4, "n_heads": 16, "d_model": 256, "d_head": 32},
    "xlarge": {"n_layers": 8, "n_heads": 16, "d_model": 256, "d_head": 64},
    "mlp":    {"n_layers": 4, "n_heads": 8,  "d_model": 128, "d_head": 32,
               "attn_only": False, "d_mlp": 512, "act_fn": "gelu"},
    "moe4":   {"n_layers": 4, "n_heads": 8,  "d_model": 128, "d_head": 32,
               "attn_only": False, "d_mlp": 128, "act_fn": "gelu",
               "num_experts": 4, "experts_per_token": 1},
    "moe8":   {"n_layers": 4, "n_heads": 8,  "d_model": 128, "d_head": 32,
               "attn_only": False, "d_mlp": 128, "act_fn": "gelu",
               "num_experts": 8, "experts_per_token": 1},
    # xlarge-geometry twins of the arch tiers — escalation targets that keep the
    # arch flag but scale to xlarge geometry (8L/16H/d256), so a variant that can't
    # reach the gate at its base size upgrades WITHIN its own architecture family.
    "mlp_xl":  {"n_layers": 8, "n_heads": 16, "d_model": 256, "d_head": 64,
                "attn_only": False, "d_mlp": 1024, "act_fn": "gelu"},
    "moe4_xl": {"n_layers": 8, "n_heads": 16, "d_model": 256, "d_head": 64,
                "attn_only": False, "d_mlp": 256, "act_fn": "gelu",
                "num_experts": 4, "experts_per_token": 1},
    "moe8_xl": {"n_layers": 8, "n_heads": 16, "d_model": 256, "d_head": 64,
                "attn_only": False, "d_mlp": 256, "act_fn": "gelu",
                "num_experts": 8, "experts_per_token": 1},
}


def tier_config(tier: str, input_length: int = INPUT_LEN, output_length: int = 1,
                d_vocab: int = 10) -> dict:
    """The complete ckpt-config dict for a trainer tier (the persisted schema)."""
    p = TIER_PRESETS[tier]
    cfg = {"n_layers": p["n_layers"], "n_heads": p["n_heads"],
           "d_model": p["d_model"], "d_head": p["d_head"],
           "d_vocab": d_vocab, "n_ctx": input_length + output_length,
           "attn_only": p.get("attn_only", True),
           "d_mlp": p.get("d_mlp"), "act_fn": p.get("act_fn"), "eps": 1e-5}
    cfg.update({k: p[k] for k in ("num_experts", "experts_per_token") if k in p})
    return cfg


def load_segrev3(
    checkpoint_path: Optional[str] = None,
    device: Optional[str] = None,
) -> HookedTransformer:
    """Load the segrev3 checkpoint into an eval-mode ``HookedTransformer``.

    ``checkpoint_path`` is resolved via ``resolve_checkpoint`` (explicit arg >
    ``SEGREV3_CKPT`` env var > the canonical default), so an orchestrator can
    point the WHOLE pipeline at a different checkpoint by exporting the env var.

    The checkpoint stores ``model_state_dict`` (already in TransformerLens
    parameter naming) plus a ``config`` dict. We build the config with
    ``normalization_type="LNPre"`` (see module docstring for why this is
    mandatory) and load the weights. ``strict=False`` only to tolerate TL's
    regenerated causal-mask / IGNORE buffers, which are not learned parameters;
    any OTHER missing/unexpected key raises.
    """
    checkpoint_path = resolve_checkpoint(checkpoint_path)
    device = _resolve_device(device)
    apply_gpu_memory_cap(device)        # GPU memory rail (see module top)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = segrev3_config(ckpt["config"])
    model = HookedTransformer(cfg)
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    bad_missing = [k for k in missing if "mask" not in k and "IGNORE" not in k]
    if bad_missing or list(unexpected):
        raise RuntimeError(
            "Unexpected state-dict mismatch loading segrev3: "
            f"missing={bad_missing}, unexpected={list(unexpected)}"
        )
    model.to(device)
    model.eval()
    return model


if __name__ == "__main__":
    # Self-check: confirm the model is loaded correctly and scores ~1.0 on the
    # real task (this is the regression test for the LNPre fix).
    import sys
    sys.path.insert(0, os.path.join(REPO_ROOT, "data_gen"))
    from data_gen import segrev3_batch  # type: ignore  # noqa: E402

    dev = resolve_device()
    print(f"self-check device: {dev}")
    m = load_segrev3(device=dev)
    toks, _x, y = segrev3_batch(2000, seed=0, device=dev)
    with torch.no_grad():
        logits = m(toks)
    pred = logits[:, ANSWER_POSITIONS, :].argmax(-1)
    tok_acc = (pred == y).float().mean().item()
    seq_acc = (pred == y).all(1).float().mean().item()
    print(f"token-acc={tok_acc:.5f}  sequence-acc={seq_acc:.5f}  "
          f"(expected ~0.99964; PASS if >0.99)")
    assert tok_acc > 0.99, "LNPre load is broken!"
    print("model_loading self-check PASSED.")
