"""Model catalog + glue for pipeline: specs, loading, artifact paths, datasets.

A model NAME is a DSL algorithm_id (resolved by models/algorithms.py). ``get_spec``
builds the ModelSpec from the name alone: task_fn, output length, n_ctx and answer
positions derive from the id; architecture is read from the trained checkpoint's
``config`` (the variant's tier preset before training). ``load_model`` delegates to the
repo-root ``interp_tools/model_loading.py`` loader (the load-bearing
``normalization_type="LNPre"`` fix and the GPU memory rails — never build a
HookedTransformerConfig by hand).

``data_dir``/``fig_dir`` are the ONLY builders for the ``circuits_data/`` and
``circuits_figures/`` trees; no other module may string-format those paths.
``load_circuit_pairs`` is the single read path for per-circuit datasets: one
``inputs.json`` pool per circuit (written by ``gen_data/gen_circuit_data.py``),
split into clean/corrupt halves at load, outputs recomputed via ``task_fn``. Each
consumer passes its role (``POOL_WINDOWS``) and gets a DISJOINT window of the pool, so a
winner is never scored on the rows it was selected on. ``canon.CANON_PAIRS_REGIME`` marks
which scoring regime built a ``Segrev3Pairs``.

Env contract (the SEGREV3_* names are historical; they are the global rails):
  SEGREV3_DEVICE            device for all children (resolve_device precedence)
  SEGREV3_CKPT              checkpoint override (set by the find_circuits runners)
  SEGREV3_GPU_MEM_FRACTION  per-process CUDA cap (default 0.125)
  SEGREV3_MIN_FREE_GIB      MemAvailable floor for spawns (default 24)
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional

# sys.path bootstrap: models/, gen_data/, and the shared repo-root module dirs.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))       # .../pipeline/models
PIPELINE_ROOT = os.path.dirname(_THIS_DIR)                   # .../pipeline
REPO_ROOT = os.path.dirname(PIPELINE_ROOT)                   # .../circuits
for _p in (_THIS_DIR,
           os.path.join(_THIS_DIR, "gen_data"),
           os.path.join(REPO_ROOT, "interp_tools"),
           os.path.join(REPO_ROOT, "data_gen"),
           os.path.join(REPO_ROOT, "metrics")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402

# Re-exported so every pipeline module gets the loader + memory rails from one
# import site (`from model_registry import assert_memory_headroom`, etc.).
from model_loading import (  # noqa: E402,F401
    TIER_PRESETS, load_segrev3, resolve_device, resolve_checkpoint,
    apply_gpu_memory_cap, assert_memory_headroom, repo_relpath,
)
from data_gen import Segrev3Pairs, load_dataset  # noqa: E402,F401
from canon import (  # noqa: E402
    CANON_ACDC_EXAMPLES, CANON_MANUAL_EXAMPLES, CANON_PAIRS_REGIME,
    CANON_TOPN_REGIME, TopnLadder, live_edge_budget, topn_ladder,
)
from dataset_store import DEFAULT_N as _STORE_DEFAULT_N      # noqa: E402

# The DSL engine (models/algorithms.py). Pure/torch only; imports nothing from
# this module (no cycle).
import algorithms  # noqa: E402

if TYPE_CHECKING:
    from transformer_lens import HookedTransformer


@dataclass(frozen=True, kw_only=True)
class ModelSpec:
    name: str                    # the DSL algorithm_id
    checkpoint: str              # ABS path -> trained_models_pt/<name>.pt
    n_layers: int
    n_heads: int
    d_model: int
    d_head: int
    d_vocab: int = 10
    n_ctx: int = 20              # input_len + DSL output length
    input_len: int = 10
    answer_positions: tuple = tuple(range(9, 19))
    task_fn: Callable = None     # inputs [B,10] -> outputs [B,L]
    gen_data_dir: str = ""       # models/gen_data/<name>
    normalization_type: str = "LNPre"
    attn_only: bool = True
    d_mlp: Optional[int] = None            # per-expert width when num_experts set
    act_fn: Optional[str] = None
    num_experts: Optional[int] = None
    experts_per_token: Optional[int] = None


_TRAINED_MODELS_PT = os.path.join(_THIS_DIR, "trained_models_pt")
_INPUT_LEN = 10
_DYN_CACHE: dict = {}

#: The canonical run-list (models/models.txt, one algorithm_id per line). It is
#: documentation + the known-model universe; specs are built from the NAME, not from
#: this file (any valid DSL algorithm_id is runnable even if not listed).
_MODELS_TXT = os.path.join(_THIS_DIR, "models.txt")

#: Pre-train arch preset per ``__<variant>`` model-name suffix ("" = no suffix).
#: Values are the trainer tiers (model_loading.TIER_PRESETS) so trainer/registry/
#: loader agree. This is the INITIAL tier; when 0.80 <= val seq-acc < 0.95 the
#: trainer escalates within the same arch family (run_model_pipeline.UPGRADE_LADDER).
VARIANT_PRESETS = {
    "": TIER_PRESETS["large"], "xl": TIER_PRESETS["xlarge"],
    "wide": TIER_PRESETS["wide"], "small": TIER_PRESETS["small"],
    "deep": TIER_PRESETS["deep"], "mlp": TIER_PRESETS["mlp"],
    "moe4": TIER_PRESETS["moe4"], "moe8": TIER_PRESETS["moe8"],
}


def _variant(name: str) -> str:
    """The ``__<variant>`` suffix of a model name ("" when none)."""
    return name.split("__", 1)[1] if "__" in name else ""


def _arch_from_checkpoint(ckpt_path: str) -> Optional[dict]:
    """The FULL stored ``config`` dict from a trained checkpoint (geometry plus
    attn_only/d_mlp/act_fn/eps and MoE keys when persisted). None when the
    checkpoint doesn't exist yet (pre-train — caller falls back to the variant
    preset); a malformed checkpoint fails loud."""
    if not os.path.exists(ckpt_path):
        return None
    return dict(torch.load(ckpt_path, map_location="cpu", weights_only=False)["config"])


def _resolve_ckpt(name: str) -> str:
    """``trained_models_pt/<name>.pt`` — the single checkpoint convention.
    ``run_model_pipeline.conv_ckpt`` mirrors this exactly (torch-free planning);
    ``_verify_conventions`` hard-fails on any drift, so keep the two in lock-step."""
    return os.path.join(_TRAINED_MODELS_PT, f"{name}.pt")


def _build_spec(name: str) -> ModelSpec:
    """Build the spec for ``name``. Task semantics (task_fn, lengths, answer positions)
    come from the BASE algorithm_id; a ``__<variant>`` suffix (e.g. segrev3_id__xl)
    only selects the checkpoint/data paths (full name) and the pre-train arch preset.
    Raises KeyError when the base name is not a valid DSL algorithm_id."""
    base, variant = algorithms.base_algorithm_id(name), _variant(name)
    if variant not in VARIANT_PRESETS:
        raise ValueError(f"unknown model variant {variant!r} in {name!r}; "
                         f"know {sorted(VARIANT_PRESETS)}")
    try:
        expr = algorithms.decode_algorithm_id(base, input_length=_INPUT_LEN)
    except Exception as e:
        raise KeyError(f"unknown model {name!r}: base {base!r} is not a valid DSL "
                       f"algorithm_id ({type(e).__name__}: {e}).")
    out_len = expr.output_length()
    # Lengths/vocab come from the EXPR, not module constants: TaskMux ids have
    # input_length 11 and vocab 10+N (DSLExpr.vocab_size, default 10).
    in_len = expr.input_length
    vocab = getattr(expr, "vocab_size", lambda: 10)()
    ckpt = _resolve_ckpt(name)
    arch = _arch_from_checkpoint(ckpt) or dict(VARIANT_PRESETS[variant])
    return ModelSpec(
        name=name,
        checkpoint=ckpt,
        n_layers=arch["n_layers"], n_heads=arch["n_heads"],
        d_model=arch["d_model"], d_head=arch["d_head"],
        d_vocab=arch.get("d_vocab", vocab),        # legacy ckpts omit non-geometry keys
        attn_only=arch.get("attn_only", True),
        d_mlp=arch.get("d_mlp"), act_fn=arch.get("act_fn"),
        num_experts=arch.get("num_experts"),
        experts_per_token=arch.get("experts_per_token"),
        n_ctx=in_len + out_len,
        input_len=in_len,
        answer_positions=tuple(range(in_len - 1, in_len - 1 + out_len)),
        task_fn=algorithms.task_fn_for(base, input_length=in_len),
        gen_data_dir=os.path.join(_THIS_DIR, "gen_data", name),
    )


def _load_run_list(path: str) -> list:
    """Names from the flat run-list (blank lines and ``#`` comments skipped,
    inline ``# ...`` stripped). Empty if missing."""
    try:
        return [ln.split("#", 1)[0].strip() for ln in open(path)
                if ln.split("#", 1)[0].strip()]
    except FileNotFoundError:
        return []


def all_models() -> list:
    """The models/models.txt run-list. Any valid DSL algorithm_id is runnable even
    if absent here — the run-list is documentation, not a gate."""
    return list(dict.fromkeys(_load_run_list(_MODELS_TXT)))


def get_spec(name: str) -> ModelSpec:
    """The ModelSpec for ``name``, built (and cached) from the DSL id."""
    if name not in _DYN_CACHE:
        _DYN_CACHE[name] = _build_spec(name)
    return _DYN_CACHE[name]


def validate_spec(spec: ModelSpec, n: int = 2048, seed: int = 0) -> None:
    """Assert ``spec.task_fn`` emits in-vocab outputs of the declared shape over a SAMPLED
    sweep of ``n`` valid inputs (uniform digit rows through the DSL expr's ``sample_row`` —
    the training distribution). ONE hand-picked row is not a vocab check: that is how Altrun
    shipped emitting token 10 into a d_vocab=10 model. Call once per model at pipeline entry."""
    expr = algorithms.decode_algorithm_id(algorithms.base_algorithm_id(spec.name),
                                          input_length=spec.input_len)
    u = torch.randint(0, 10, (n, spec.input_len),
                      generator=torch.Generator().manual_seed(seed))
    rows = torch.stack([expr.sample_row(u[i]) for i in range(n)])
    out_len = spec.n_ctx - spec.input_len
    out = spec.task_fn(rows)
    if out.shape != (n, out_len) or out.dtype != torch.int64:
        raise ValueError(f"{spec.name}: task_fn returned {tuple(out.shape)}/{out.dtype}, "
                         f"expected {(n, out_len)}/torch.int64")
    for what, t in (("INPUT (sample_row)", rows), ("OUTPUT (task_fn)", out)):
        bad = (t < 0) | (t >= spec.d_vocab)
        if bad.any():
            r = int(bad.any(1).nonzero()[0])
            raise ValueError(f"{spec.name}: out-of-vocab {what} tokens "
                             f"{sorted(set(t[r][bad[r]].tolist()))} (d_vocab={spec.d_vocab}) "
                             f"on row {rows[r].tolist()}; {int(bad.any(1).sum())}/{n} rows hit")


def load_model(name: str, device: Optional[str] = None) -> "HookedTransformer":
    """Load ``name`` into an eval-mode HookedTransformer via the LNPre-correct
    loader (``interp_tools/model_loading.py`` — TF32 tuning + GPU memory cap)."""
    spec = get_spec(name)
    if not os.path.exists(spec.checkpoint):
        raise FileNotFoundError(
            f"checkpoint for model {name!r} not found: {spec.checkpoint}\n"
            "Run `run_model_pipeline.py --stages train` first (or import a .pt "
            "named trained_models_pt/<algorithm_id>.pt).")
    return load_segrev3(checkpoint_path=spec.checkpoint, device=device)


# --------------------------------------------------------------------------- #
# Path builders — the SINGLE source of truth for the artifact trees.           #
# --------------------------------------------------------------------------- #


def data_dir(name: str, circuit: Optional[str] = None) -> str:
    """``pipeline/circuits_data/<name>[/<circuit>]`` (absolute).
    ``circuit=None`` returns the model root (used to iterate a model's circuits)."""
    base = os.path.join(PIPELINE_ROOT, "circuits_data", name)
    return base if circuit is None else os.path.join(base, circuit)


def fig_dir(name: str, circuit: Optional[str] = None) -> str:
    """``pipeline/circuits_figures/<name>[/<circuit>]`` (absolute);
    mirrors ``data_dir`` (built independently, so the trees can be renamed apart)."""
    base = os.path.join(PIPELINE_ROOT, "circuits_figures", name)
    return base if circuit is None else os.path.join(base, circuit)


def circuit_data_dir(name: str, circuit: str) -> str:
    """``models/gen_data/<name>/<circuit>`` — the per-circuit input pool dir."""
    return os.path.join(get_spec(name).gen_data_dir, circuit)


def circuit_dataset_path(name: str, circuit: str) -> str:
    """One circuit's input pool (``inputs.json``, written by gen_circuit_data.py;
    read by every finder and analysis via ``load_circuit_pairs``)."""
    return os.path.join(circuit_data_dir(name, circuit), "inputs.json")


def head_dirname(l: int, h: int) -> str:
    """Per-head directory name, e.g. ``head_L0H0``."""
    return f"head_L{l}H{h}"


def heads(spec: ModelSpec) -> list:
    """All (layer, head) index pairs for ``spec``, row-major."""
    return [(l, h) for l in range(spec.n_layers) for h in range(spec.n_heads)]


def live_edges(spec: ModelSpec) -> int:
    """``spec``'s live attribution-graph edge count — the ONE place to get it. Smaller than
    the ``graph.real_edges`` EAP-IG reports, which counts the phantom ``m{l}`` edges of an
    attn-only model (see ``canon.live_edge_budget``)."""
    return live_edge_budget(spec.n_layers, spec.n_heads, spec.attn_only)


def topns(spec: ModelSpec, regime: str = CANON_TOPN_REGIME) -> TopnLadder:
    """``spec``'s canonical EAP top-n ladder, resolved against its live edge budget."""
    return topn_ladder(live_edges(spec), regime)


# --------------------------------------------------------------------------- #
# Datasets — the single read path for per-circuit analysis data.               #
# --------------------------------------------------------------------------- #

#: canonical dataset size (in PAIRS) per consumer.
APP_SIZES = {
    "manual": CANON_MANUAL_EXAMPLES,           # 128 (patching)
    "edges": _STORE_DEFAULT_N["edge"],         # 64
    "acdc": CANON_ACDC_EXAMPLES,               # 128 (canon owns the ACDC budget)
    "eap": _STORE_DEFAULT_N["eap"],            # 512
    "edge_pruning": _STORE_DEFAULT_N["eap"],   # 512 (same contrast set kind as eap)
}

#: Pool window START (in PAIRS) per consumer role. Attribution, winner SELECTION and final
#: SCORING must read DISJOINT rows: a winner picked for clearing a floor on the very rows it
#: is then scored on is selected on its own noise (in the zoo runs, many plain winners sat within one
#: standard error of their floor when selection and scoring shared rows). "legacy" is the
#: pre-window regime every call site still gets until it passes ``consumer=``.
POOL_WINDOWS = {
    "attribution": 0,        # eap / edge_pruning / acdc / manual: attribution + their own eval
    "selection": 5000,       # eap.py --target-seq-acc ladder search (PICKS the winner)
    "scoring": 10000,        # smallest_faithful_circuit's reported faithfulness numbers
    "legacy": 0,
}


def _split_inputs(dataset: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Clean/corrupt input tensors from a stored dataset. Pool schema: one
    ``inputs`` pool split into two equal halves (the second half is the
    resample-patching reference). Pair schema (``scheme="contrast"`` + legacy
    pools): row-aligned ``clean_inputs``/``corrupted_inputs``."""
    if "inputs" in dataset:
        all_in = torch.tensor(dataset["inputs"], dtype=torch.long)
        rows = all_in.shape[0]
        if rows < 2 or rows % 2 != 0:  # guard BEFORE the halving, or the halves overlap silently
            raise ValueError(f"pool holds {rows} rows; the clean/corrupt split needs an "
                             f"even count >= 2")
        return all_in[:rows // 2], all_in[rows // 2:]
    clean_in = torch.tensor(dataset["clean_inputs"], dtype=torch.long)
    corr_in = torch.tensor(dataset["corrupted_inputs"], dtype=torch.long)
    if clean_in.shape[0] != corr_in.shape[0]:
        raise ValueError(f"clean/corrupt length mismatch ({clean_in.shape[0]} vs {corr_in.shape[0]})")
    return clean_in, corr_in


def _diff_answer_slots(input_len: int, clean_out: torch.Tensor,
                       corr_out: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-row scored slots for ``answer_positions="diff"`` datasets: the output indices
    where the TRUE outputs differ, mapped to logit positions (k + input_len - 1) and
    right-padded to the batch max by repeating each row's FIRST diff slot; returns
    (positions, correct, incorrect, mask) with mask marking the real slots."""
    diffs = [(clean_out[b] != corr_out[b]).nonzero().flatten() for b in range(clean_out.shape[0])]
    for b, d in enumerate(diffs):
        if d.numel() == 0:
            raise ValueError(f"contrast row {b}: clean and corrupt outputs identical")
    P = max(d.numel() for d in diffs)
    idx = torch.stack([torch.cat([d, d[:1].expand(P - d.numel())]) for d in diffs])
    mask = torch.stack([torch.arange(P) < d.numel() for d in diffs])
    return idx + input_len - 1, clean_out.gather(1, idx), corr_out.gather(1, idx), mask


def dataset_to_pairs(dataset: dict, task_fn: Callable,
                     device: Optional[str] = None) -> Segrev3Pairs:
    """Tensor ``Segrev3Pairs`` from a stored dataset dict: inputs from the pool
    (``_split_inputs``), OUTPUTS recomputed here via ``task_fn``. Contrast datasets
    (``answer_positions == "diff"``) get per-row positions/labels + ``answer_mask``.

    PLAIN datasets get an ``answer_mask`` too (CANON_PAIRS_REGIME v2): a resample pair
    carries no evidence at a position where the clean and corrupt labels agree — the
    logit diff there is identically zero for any model under any ablation — so those
    positions are masked out and rows that agree EVERYWHERE are dropped."""
    device = resolve_device(device)
    clean_in, corr_in = _split_inputs(dataset)
    clean_out = task_fn(clean_in)
    corr_out = task_fn(corr_in)
    clean_tokens = torch.cat([clean_in, clean_out], dim=1)
    corr_tokens = torch.cat([corr_in, corr_out], dim=1)
    B = clean_in.shape[0]
    if dataset["answer_positions"] == "diff":
        pos, clean_lbl, corr_lbl, mask = _diff_answer_slots(clean_in.shape[1], clean_out, corr_out)
        return Segrev3Pairs(
            clean_tokens.to(device), corr_tokens.to(device), pos.to(device),
            clean_lbl.to(device), corr_lbl.to(device), dataset["scheme"],
            clean_inputs=clean_in.to(device), corrupted_inputs=corr_in.to(device),
            answer_mask=mask.to(device),
            extra={"n": dataset.get("n"), "seed": dataset.get("seed"),
                   "regime": CANON_PAIRS_REGIME, "n_agreeing_rows_dropped": 0},
        )
    ans_pos = list(dataset["answer_positions"])
    # LABELS are the output AT the scored answer positions; they differ from the
    # full output when answer_positions is a strict subset (e.g. argminmax's
    # length-20 output scored over only its first 10).
    out_idx = torch.tensor([p - (clean_in.shape[1] - 1) for p in ans_pos], dtype=torch.long)
    if (out_idx < 0).any():   # stale pre-fix dataset (mux answer-position off-by-one): regenerate --stages data
        raise ValueError(f"answer_positions {ans_pos} precede the first output slot "
                         f"(input_len {clean_in.shape[1]}); regenerate this dataset")
    clean_lbl = clean_out[:, out_idx]
    corr_lbl = corr_out[:, out_idx]
    mask = clean_lbl != corr_lbl
    keep = mask.any(dim=1)
    kept = int(keep.sum())
    if kept == 0:
        raise ValueError(f"zero-contrast isolation: all {B} rows agree with their resample "
                         f"reference at every scored position {ans_pos}, so the logit diff "
                         f"is identically 0 and faithfulness is 0/0")
    pos = torch.tensor(ans_pos).unsqueeze(0).expand(kept, -1).clone()
    return Segrev3Pairs(
        clean_tokens[keep].to(device), corr_tokens[keep].to(device), pos.to(device),
        clean_lbl[keep].to(device), corr_lbl[keep].to(device), dataset["scheme"],
        clean_inputs=clean_in[keep].to(device), corrupted_inputs=corr_in[keep].to(device),
        answer_mask=mask[keep].to(device),
        extra={"n": dataset.get("n"), "seed": dataset.get("seed"),
               "regime": CANON_PAIRS_REGIME, "n_agreeing_rows_dropped": B - kept},
    )


def load_circuit_pairs(name: str, circuit: str, n: Optional[int] = None,
                       device: Optional[str] = None, consumer: str = "legacy",
                       ) -> tuple[Segrev3Pairs, str]:
    """``(Segrev3Pairs, path)`` for ONE circuit's dataset
    (``gen_data/<name>/<circuit>/inputs.json``). Every finder and analysis for
    ``circuit`` reads from here, so they share one dataset. ``n`` is the number of
    PAIRS (consumers pass their APP_SIZES); None takes the whole pool.

    ``consumer`` picks the POOL_WINDOWS row range so attribution, winner selection and
    final scoring never share rows; the resolved window lands in ``pairs.extra`` for the
    caller to record. Fails loud when the pool is too short for the window."""
    spec = get_spec(name)
    path = circuit_dataset_path(name, circuit)
    ds = load_dataset(path)
    if consumer not in POOL_WINDOWS:
        raise ValueError(f"unknown consumer {consumer!r}; know {sorted(POOL_WINDOWS)}")
    off = POOL_WINDOWS[consumer]
    pooled = "inputs" in ds
    avail = len(ds["inputs"]) // 2 if pooled else len(ds["clean_inputs"])
    if n is None and off > 0:
        raise ValueError(f"consumer {consumer!r} reads the pool from pair {off}, so it needs "
                         f"an explicit n (n=None means the whole pool)")
    lo, hi = off, avail if n is None else off + n
    if hi > avail:
        raise ValueError(f"{path}: consumer {consumer!r} needs pairs [{lo}:{hi}] but the pool "
                         f"holds {avail}; regenerate it with -n {hi}")
    if n is not None:
        ds = dict(ds)
        if pooled:                                      # n PAIRS = window of each half
            half = len(ds["inputs"]) // 2
            ds["inputs"] = ds["inputs"][lo:hi] + ds["inputs"][half + lo:half + hi]
        else:
            ds["clean_inputs"] = ds["clean_inputs"][lo:hi]
            ds["corrupted_inputs"] = ds["corrupted_inputs"][lo:hi]
    pairs = dataset_to_pairs(ds, spec.task_fn, device=device)
    pairs.extra.update(consumer=consumer, window=[lo, hi])
    return pairs, path


# --------------------------------------------------------------------------- #
# Shared rails/metrics used by every finder and analysis.                      #
# --------------------------------------------------------------------------- #


def wait_for_headroom(context: str, retries: int = 5, sleep_s: float = 60.0) -> float:
    """``assert_memory_headroom`` with the standard sleep-60-retry (max 5) behavior
    instead of failing on the first probe. The one retry rule for every stage."""
    for attempt in range(retries + 1):
        try:
            return assert_memory_headroom(context=context)
        except RuntimeError as e:
            if attempt == retries:
                raise
            print(f"[headroom] {e}", flush=True)
            print(f"[headroom] retry {attempt + 1}/{retries} in {sleep_s:.0f}s ...",
                  flush=True)
            time.sleep(sleep_s)


class ZeroContrastError(ValueError):
    """The full model and the fully-ablated model score the same: no contrast to recover."""


def recovery_ratio(x: float, full: float, empty: float) -> float:
    """Faithfulness R = (x - empty) / (full - empty). A zero denominator means the EMPTY
    circuit already equals the full model — a degenerate isolation whose R is 0/0 — so it
    raises instead of quietly reporting the finite ratio a ``or 1.0`` guard invents."""
    if full == empty:
        raise ZeroContrastError(f"faithfulness undefined: full == empty == {full!r}; the "
                                f"fully-ablated model already matches the full model, so "
                                f"this isolation has nothing to recover")
    return (x - empty) / (full - empty)


if __name__ == "__main__":
    names = all_models()
    print(f"all_models() = {names}")
    bad = []
    for nm in names:
        s, err = get_spec(nm), None
        try:                                 # collect every offender: one run, one list
            validate_spec(s)
        except ValueError as e:
            err = str(e)
            bad.append(err)
        print(f"  [{nm}] {s.n_layers}Lx{s.n_heads}H d_model={s.d_model} "
              f"n_ctx={s.n_ctx} ckpt={'yes' if os.path.exists(s.checkpoint) else 'NO'} "
              f"live_edges={live_edges(s)} topns={topns(s).topns}"
              f"{' SPEC-INVALID' if err else ''}")
    if bad:
        raise SystemExit(f"model_registry self-check FAILED ({len(bad)}/{len(names)} "
                         f"models):\n  " + "\n  ".join(bad))
    print("model_registry self-check PASSED.")
