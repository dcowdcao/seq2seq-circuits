"""EAP-IG circuit discovery + the shared find_circuits helper hub.

Runs vendored EAP-IG attribution for ONE model+circuit and writes one candidate
folder per top-n: circuits_data/<model>/<circuit>/circuits/eap_top<n>/
{summary.json, circuit.json, circuit.pt, circuit.gv} + the PNG under
circuits_figures/. Also hosts the FROZEN drawer (draw_circuit_png),
save_circuit_pt, prune_dangling and write_candidate_folder shared by acdc.py,
edge_pruning.py and smallest_faithful_circuit.py.

Attribution params attr_bs=50 / attr_n=200 / ig_steps=30 are PINNED for
throughput, not for correctness: the attribution metric SUMS over the batch and
EAP-IG divides the accumulated gradient by total_items, so edge scores are
per-example means and comparable across batch sizes. --smoke shrinks only the
eval set and the sweep list, and quarantines outputs under .smoke/run3/.

The top-n ladder comes from canon (model_registry.topns) and is SCALE-RELATIVE
by default: absolute rungs capped every architecture at the same 800 edges,
which is 988% of the live graph at `small` and 3.6% at `xl`.
model_registry.live_edges(spec) is the method-independent edge budget every
finder reports as n_edges_universe, so scores normalised by "total edges" are
comparable across acdc / eap / edge_pruning. The phantom m{l} slots EAP-IG's
Graph.from_model wires unconditionally (it never reads cfg.attn_only) score
exactly 0.0 and are masked out of real_edge_mask here, never in the vendored repo.

Two monkeypatches ride the read-only vendored EAP-IG: tokenize_plus becomes a
pass-through (no tokenizer; pre-tokenized int tensors) and hard-coded
device='cuda' allocations are redirected to the model's device.

--probe <variable> switches to probe-metric discovery (Direction 3): the
metric becomes probe-readout KL at the variable's site (artifact root:
--probes-root / $PROBES_DATA_ROOT / <pipeline>/probes_data), the built Graph
is instance-patched so its 'logits' node reads the probe site, a causal
cutoff zeroes every edge whose writer sits above the probe layer, and
candidates write under eap_probe_<variable>_top<n>/ (never colliding with
the canonical output-circuit folders). The metric reads the pooled probe (W, b) by
default; --probe-readout position reads each probed position's own probe (W_pos, b_pos)
instead. See docs/reference/design.md "Probe-metric discovery".

--zero-ablation / --use-pos-embed thread the C4 positional options (mirror
acdc.py): zero-ablation drops the corrupted patch (EAP-IG-activations attribution
with intervention='zero'); use-pos-embed repurposes the input source to the
positional embedding (hook_pos_embed) with the corrupted positions scrambled
(seed 49) so positional edges get scored. Candidate folders gain _zero / _pos
suffixes so canonical artifacts are never clobbered.

Must NOT be imported under the name 'eap' (collides with the vendored package);
load via importlib under a unique name (see acdc.py::_load_fc_eap).
"""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import glob
import json
import math
import os
import sys
from functools import partial
from typing import Callable
from typing import Iterator
from typing import Optional

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))        # .../find_circuits
PIPELINE_ROOT = os.path.dirname(_THIS_DIR)                    # .../pipeline
REPO_ROOT = os.path.dirname(PIPELINE_ROOT)                    # .../circuits
EAP_SRC = os.path.join(REPO_ROOT, "github_repos", "EAP-IG", "src")

if __name__ == "eap":  # pragma: no cover - import-hygiene guard
    raise ImportError(
        "pipeline/find_circuits/eap.py must not be imported as "
        "'eap' — that name belongs to the vendored EAP-IG package "
        "(github_repos/EAP-IG/src/eap). Load this file via importlib under a "
        "unique module name instead (see acdc.py::_load_fc_eap).")

# EAP_SRC + models dir go BEFORE the script dir so `import eap.utils` resolves
# to the vendored package, never to this file.
for _p in (os.path.join(PIPELINE_ROOT, "models"), EAP_SRC):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402
import pygraphviz as pgv  # noqa: E402
import torch  # noqa: E402
from tqdm import tqdm  # noqa: E402

import model_registry  # noqa: E402  (bootstraps interp_tools/data_gen/metrics onto sys.path)
from model_registry import (  # noqa: E402
    Segrev3Pairs, recovery_ratio, repo_relpath, resolve_device, wait_for_headroom,
)
from canon import (  # noqa: E402
    CANON_IG_STEPS, CANON_TOPN_REGIME, DEFAULT_SEED, TopnLadder, topn_ladder,
)

GENERATED_BY = "pipeline/find_circuits/eap.py"
SMOKE_ROOT = os.path.join(PIPELINE_ROOT, ".smoke", "run3")

# PINNED for throughput (not correctness — scores are per-example means, see the
# module docstring); not CLI-exposed, --smoke must never change them.
ATTR_BS = 50
ATTR_N = 200
IG_STEPS = CANON_IG_STEPS

EVAL_N = 256            # faithfulness eval examples (production)
EVAL_BS = 64            # eval batch size
SMOKE_EVAL_N = 64       # --smoke shrinks ONLY this

#: --target-seq-acc scan: coarse geometric sweep, then a fine sweep inside the
#: bracket below the first passing rung (faithfulness is NOT monotone in top-n,
#: so this is a scan + argmax, never a bisection).
SEARCH_COARSE_RATIO = 1.5
SEARCH_FINE_RATIO = 1.05


# --------------------------------------------------------------------------- #
# The phantom m{l} slots of the EAP-IG score matrix                            #
# --------------------------------------------------------------------------- #
def phantom_mlp_indices(n_layers: int, n_heads: int) -> tuple[list, list]:
    """``(forward rows, backward cols)`` of the ``m{l}`` slots in the EAP-IG score
    matrix. ``Graph.from_model`` wires an MLP node per layer without ever reading
    ``cfg.attn_only`` (github_repos/EAP-IG/src/eap/graph.py:602-693), and
    TransformerLens does create the hook_mlp_{in,out} HookPoints on an attn-only
    model — they just never fire, so every edge here scores exactly 0.0 while still
    inflating ``real_edge_mask`` by 8-26%. run_eap masks these out of the LIVE graph;
    the edge BUDGET comes from ``model_registry.live_edges``."""
    H = n_heads
    return ([1 + l * (H + 1) + H for l in range(n_layers)],
            [l * (3 * H + 1) + 3 * H for l in range(n_layers)])


# --------------------------------------------------------------------------- #
# Shared pipeline helpers (also imported by acdc.py / edge_pruning.py)         #
# --------------------------------------------------------------------------- #
def circuit_dirs(model_name: str, circuit_name: str, *, circuit: str,
                 smoke: bool = False) -> tuple[str, str]:
    """``(data_dir, fig_dir)`` for ONE candidate circuit, under
    ``<model>/<circuit>/circuits/<circuit_name>``; smoke quarantines both
    under ``.smoke/run3/``."""
    if smoke:
        d = os.path.join(SMOKE_ROOT, "circuits_data", model_name, circuit,
                         "circuits", circuit_name)
        f = os.path.join(SMOKE_ROOT, "figures", model_name, circuit,
                         "circuits", circuit_name)
        return d, f
    return (os.path.join(model_registry.data_dir(model_name, circuit),
                         "circuits", circuit_name),
            os.path.join(model_registry.fig_dir(model_name, circuit),
                         "circuits", circuit_name))


def set_checkpoint_env(model_name: str) -> str:
    """Export ``SEGREV3_CKPT`` so everything downstream — the in-process
    vendored stacks AND spawned workers — loads the same checkpoint. MUST run
    before the vendored repos build a model. Returns the checkpoint path."""
    ckpt = model_registry.get_spec(model_name).checkpoint
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"checkpoint for model {model_name!r} not found: {ckpt}")
    os.environ["SEGREV3_CKPT"] = ckpt
    return ckpt


def artifact_meta(model_name: str, *, method: str, n_examples: int, seed: int,
                  scheme=None, source_dataset_path=None,
                  generated_by: str = GENERATED_BY) -> dict:
    """The JSON meta envelope shared by every circuits_data artifact."""
    return {
        "model": model_name,
        "scheme": scheme,
        "n_examples": int(n_examples),
        "seed": int(seed),
        "method": method,
        "checkpoint": repo_relpath(os.environ["SEGREV3_CKPT"]),   # set_checkpoint_env runs first
        "generated_by": generated_by,
        "source_dataset_path": repo_relpath(source_dataset_path),
    }


def load_run_pairs(model_name: str, circuit: str, num_examples: int,
                   seed: int, smoke: bool, device: str) -> tuple:
    """``(pairs, provenance)`` — the single dataset source for the finders.
    Production reads the circuit's stored ``inputs.json`` in the ATTRIBUTION pool
    window (``model_registry.POOL_WINDOWS``), disjoint from the windows the
    ``--target-seq-acc`` selection and the smallest stage's scoring read;
    ``--smoke`` samples the same circuit's domain IN MEMORY (never racing the data
    stage's writes)."""
    if smoke:
        pairs = _make_circuit_smoke_pairs(model_name, circuit, num_examples,
                                          seed, device)
        return pairs, {"scheme": pairs.scheme, "path": None,
                       "num_examples": int(pairs.clean_tokens.shape[0]),
                       "in_memory": True, "circuit": circuit,
                       "consumer": "attribution"}
    pairs, path = model_registry.load_circuit_pairs(
        model_name, circuit, n=num_examples, device=device,
        consumer="attribution")
    return pairs, {"scheme": pairs.scheme, "path": path,
                   "num_examples": int(pairs.clean_tokens.shape[0]),
                   "circuit": circuit, "consumer": "attribution"}


def _make_circuit_smoke_pairs(model_name: str, circuit: str, num_examples: int,
                              seed: int, device: str) -> Segrev3Pairs:
    """In-memory pairs for ONE circuit (--smoke): draw a pool from the built
    ``DomainCircuit`` instance and feed the SAME dataset dict a stored inputs.json
    carries through ``model_registry.dataset_to_pairs``."""
    import gen_circuit_data as GCD           # on sys.path via model_registry bootstrap
    spec = model_registry.get_spec(model_name)
    circs = GCD.circuits_for(model_name)
    match = [c for c in circs if c.circuit_id == circuit]
    if not match:
        raise KeyError(f"model {model_name!r} has no circuit {circuit!r} "
                       f"(have {[c.circuit_id for c in circs]})")
    circ = match[0]
    inputs = circ.sample(2 * num_examples, seed)   # one pool; loader splits clean/corrupt
    out_len = spec.task_fn(inputs[:1]).shape[1]
    base_pos = list(range(spec.input_len - 1, spec.input_len - 1 + out_len))
    if circ.scored_output_indices is not None:
        base_pos = [base_pos[i] for i in circ.scored_output_indices]
    dataset = {
        "inputs": inputs.tolist(),
        "answer_positions": base_pos,
        "scheme": "random",
        "n": int(num_examples),
        "seed": int(seed),
    }
    pairs = model_registry.dataset_to_pairs(dataset, spec.task_fn, device=device)
    pairs.extra.update(consumer="attribution", window=[0, int(num_examples)])
    return pairs


# =========================================================================== #
# FROZEN DRAWER — the EXACT EAP-IG circuit style                               #
# Transcribed from EAP-IG graph.py::to_image + visualization.py; imported by acdc.py, edge_pruning.py and smallest_faithful_circuit.py.
# =========================================================================== #

#: edge colors (eap/visualization.py::EDGE_TYPE_COLORS)
EDGE_TYPE_COLORS = {"q": "#FF00FF", "k": "#00FF00", "v": "#0000FF", None: "#000000"}
MINIMUM_PENWIDTH = 0.6
MAXIMUM_PENWIDTH = 5.0
#: Uniform penwidth for gate-score methods (Edge Pruning): its L0 gate is ~0.8-1.0, so min/max-normalizing pins every edge at the 5.0 ceiling; a kept gate carries no strength signal, so render all such edges at one medium width (passed as uniform_penwidth to draw_circuit_png).
EDGE_PRUNING_PENWIDTH = 1.5


def _get_color(qkv, score: float) -> str:
    """eap/visualization.py::get_color, verbatim: q/k/v get their fixed hex;
    non-qkv edges are black when score >= 0, red (#FF0000) when negative."""
    if qkv is not None:
        return EDGE_TYPE_COLORS[qkv]
    elif score < 0:
        return "#FF0000"
    else:
        return "#000000"


_PASTEL2_LUT = None


def _pastel2_lut() -> np.ndarray:
    """The 256-entry RGB LUT cmapy builds for matplotlib colormaps —
    transcription of eap/visualization.py::cmap(..., rgb_order=True)."""
    global _PASTEL2_LUT
    if _PASTEL2_LUT is None:
        import matplotlib
        import matplotlib.cm
        c_map = matplotlib.colormaps.get_cmap("Pastel2")
        rgba = matplotlib.cm.ScalarMappable(cmap=c_map).to_rgba(
            np.arange(0, 1.0, 1.0 / 256.0), bytes=True)
        _PASTEL2_LUT = rgba[:, :3]
    return _PASTEL2_LUT


def _generate_random_color() -> str:
    """eap/visualization.py::generate_random_color("Pastel2"): one draw from
    the GLOBAL numpy RNG (to_image seeds it once per render) indexes the LUT."""
    rgb = _pastel2_lut()[int(np.random.randint(0, 256))]
    return "#{:02x}{:02x}{:02x}".format(int(rgb[0]), int(rgb[1]), int(rgb[2]))


def _penwidths(edges) -> list:
    """graph.py::to_image lines 847–853 verbatim: min/max-normalize |score|
    over ALL edge scores; penwidth = max(0.6, normalized * 5.0). (Including
    the upstream quirk that when max == min the raw |score| is used.)"""
    scores = [abs(float(s)) for (_p, _c, _q, s) in edges]
    if not scores:
        return []
    mx, mn = max(scores), min(scores)
    out = []
    for s in scores:
        normalized = (s - mn) / (mx - mn) if mx != mn else s
        out.append(max(MINIMUM_PENWIDTH, normalized * MAXIMUM_PENWIDTH))
    return out


def draw_circuit_png(nodes, edges, out_png, out_gv, seed=0,
                     uniform_penwidth=None) -> None:
    """FROZEN contract — the ONE circuit drawer for the whole pipeline.

    nodes : iterable of node names ("input", "a0.h3", "m1", "logits")
    edges : [(parent_name, child_name, qkv: "q"|"k"|"v"|None, score: float)]

    EXACT EAP-IG style (graph.py::to_image + visualization.py::get_color):
    dot-layout AGraph, white bg, rounded filled boxes with random Pastel2
    fills, edge colors q=#FF00FF k=#00FF00 v=#0000FF (non-qkv black/red by
    score sign), penwidth = max(0.6, minmax_norm(|score|)*5.0).
    ``uniform_penwidth`` (e.g. EDGE_PRUNING_PENWIDTH) renders ALL edges at a
    fixed width instead — for gate-score methods whose normalized scaling
    would pin every edge at the ceiling. Writes the .gv source, then the PNG.
    """
    edges = [(str(p), str(c), q, float(s)) for (p, c, q, s) in edges]
    # dedupe node names preserving order; append edge endpoints missing from `nodes`
    node_list = list(dict.fromkeys(str(n) for n in nodes))
    known = set(node_list)
    for p, c, _q, _s in edges:
        for nm in (p, c):
            if nm not in known:
                known.add(nm)
                node_list.append(nm)

    np.random.seed(seed)                       # to_image: np.random.seed(seed)
    colors = {nm: _generate_random_color() for nm in node_list}
    penwidths = ([float(uniform_penwidth)] * len(edges)
                 if uniform_penwidth is not None else _penwidths(edges))

    for path in (out_png, out_gv):
        d = os.path.dirname(os.path.abspath(path))
        if d:
            os.makedirs(d, exist_ok=True)

    # strict=False (deliberate): strict AGraphs silently COLLAPSE parallel edges — a head pair's q/k/v edges would render as one edge with only the last channel's color.
    g = pgv.AGraph(directed=True, strict=False, bgcolor="white",
                   overlap="false", splines="true", layout="dot")
    for nm in node_list:
        g.add_node(nm,
                   fillcolor=colors[nm],
                   color="black",
                   style="filled, rounded",
                   shape="box",
                   fontname="Helvetica")
    for (p, c, q, s), pw in zip(edges, penwidths):
        g.add_edge(p, c, penwidth=str(pw), color=_get_color(q, s))
    g.write(out_gv)                            # ALWAYS emit the .gv source
    g.draw(out_png, prog="dot")


# =========================================================================== #
# EAP-IG runner (adapted wholesale from circuit_discovery/eap.py)              #
# =========================================================================== #

def _import_vendored_eap() -> tuple:
    """Import the vendored EAP-IG modules (lazily, so merely importing this
    file for the drawer stays light, and so SEGREV3_CKPT can be exported
    first). Fails loudly if something non-package shadows the 'eap' name."""
    if "eap" in sys.modules and not hasattr(sys.modules["eap"], "__path__"):
        raise ImportError(
            "sys.modules['eap'] is occupied by a non-package module "
            f"({sys.modules['eap']!r}) — the vendored EAP-IG package cannot "
            "load. Never import find_circuits/eap.py under the name 'eap'.")
    import eap.utils as eap_utils            # noqa: F401
    import eap.attribute as eap_attribute    # noqa: F401
    import eap.evaluate as eap_evaluate      # noqa: F401
    return eap_utils, eap_attribute, eap_evaluate


# ---- Monkeypatch 1: tokenize_plus pass-through (pre-tokenized int tensors) --
def tokenize_plus_passthrough(model, inputs, max_length=None) -> tuple:
    """Replacement for eap.utils.tokenize_plus that accepts an already-tokenized
    LongTensor ``[B, S]`` (these models have no tokenizer). Returns the tokens,
    an all-ones attention mask, per-example lengths (== S, no padding), n_pos."""
    tokens = inputs if torch.is_tensor(inputs) else torch.as_tensor(inputs)
    tokens = tokens.to(device=model.cfg.device, dtype=torch.long)
    if tokens.ndim == 1:
        tokens = tokens.unsqueeze(0)
    B, S = tokens.shape
    attention_mask = torch.ones((B, S), dtype=torch.long, device=tokens.device)
    input_lengths = torch.full((B,), S, dtype=torch.long, device=tokens.device)
    return tokens, attention_mask, input_lengths, S


def _install_tokenize_patch(eap_utils, eap_attribute, eap_evaluate) -> None:
    eap_utils.tokenize_plus = tokenize_plus_passthrough
    eap_attribute.tokenize_plus = tokenize_plus_passthrough
    eap_evaluate.tokenize_plus = tokenize_plus_passthrough


# ---- Monkeypatch 2: redirect EAP-IG's hard-coded device='cuda' --------------
@contextlib.contextmanager
def force_zeros_device(target) -> Iterator[None]:
    """Rewrite literal ``torch.zeros(..., device='cuda')`` allocations (e.g.
    eap/attribute.py:116) to the model's actual device for the duration."""
    target = str(target)
    orig_zeros = torch.zeros

    def patched_zeros(*args, **kwargs):
        dev = kwargs.get("device", None)
        if dev is not None and str(dev).startswith("cuda"):
            kwargs["device"] = target
        return orig_zeros(*args, **kwargs)

    torch.zeros = patched_zeros
    try:
        yield
    finally:
        torch.zeros = orig_zeros


# ---- Metric (EAP signature): differentiable logit-diff over answer positions
def make_logit_diff_metric(answer_positions) -> Callable:
    """metric(logits, clean_logits, input_lengths, labels) for EAP-IG.

    logits : [B, n_ctx, d_vocab];  labels : [B, 2, P] or [B, 4, P]
    (labels[:,0]=correct digit, [:,1]=incorrect digit; per-pair datasets append
    [:,2]=per-row answer positions, [:,3]=valid-position mask — the vendored
    EAP-IG passes labels through opaquely, so this is the per-row transport).
    Returns mean (correct - incorrect) logit over the answer positions.
    loss=True negates it (for attribution); mean=False returns per-example [B].
    """
    pos_cpu = torch.tensor(list(answer_positions), dtype=torch.long)

    def metric(logits, clean_logits, input_lengths, labels,
               mean: bool = True, loss: bool = False):
        correct = labels[..., 0, :].to(logits.device)             # [B, P]
        incorrect = labels[..., 1, :].to(logits.device)           # [B, P]
        if labels.shape[-2] == 4:                                 # per-row positions + mask
            pos = labels[..., 2, :].to(logits.device)             # [B, P]
            sel = logits.gather(1, pos.unsqueeze(-1).expand(-1, -1, logits.shape[-1]))
            mask = labels[..., 3, :].to(logits.device)            # [B, P]
        else:
            sel = logits[:, pos_cpu.to(logits.device), :]         # [B, P, V]
            mask = None
        cor = sel.gather(-1, correct.unsqueeze(-1)).squeeze(-1)   # [B, P]
        inc = sel.gather(-1, incorrect.unsqueeze(-1)).squeeze(-1)
        d = cor - inc                                             # [B, P]
        if mask is None:
            d = d.mean(dim=-1)                                    # [B]
        else:
            m = mask.to(d.dtype)
            d = (d * m).sum(-1) / m.sum(-1).clamp(min=1)          # [B]
        if loss:
            d = -d
        return d.mean() if mean else d

    return metric


def sum_reduced(metric: Callable) -> Callable:
    """The ATTRIBUTION form of an EAP metric: loss, SUMMED over the batch.

    EAP-IG accumulates gradients across batches and then divides once by
    total_items, so a batch-MEAN metric divides twice and every edge score comes
    out at 1/batch_size of its per-example value (rankings survive — the factor is
    uniform — but scores stop being comparable across attr_bs). Summing here makes
    the final scores per-example means, and weights a short last batch correctly."""
    return lambda *a: metric(*a, mean=False, loss=True).sum()


def pack_labels(pairs) -> torch.Tensor:
    """[B,2,P] labels (correct, incorrect); [B,4,P] (+positions, +mask) for per-pair sets."""
    rows = [pairs.correct, pairs.incorrect]
    if pairs.answer_mask is not None:
        rows += [pairs.answer_positions, pairs.answer_mask]
    return torch.stack([r.long() for r in rows], dim=1)


# --------------------------------------------------------------------------- #
# Probe-metric mode (Direction 3) — helpers shared with acdc.py                #
# --------------------------------------------------------------------------- #
PROBES_ROOT_ENV = "PROBES_DATA_ROOT"


def probes_root_default() -> str:
    """Probe artifacts root: $PROBES_DATA_ROOT (fixtures) or <pipeline>/probes_data."""
    return os.environ.get(PROBES_ROOT_ENV) or os.path.join(PIPELINE_ROOT, "probes_data")


def load_probe_artifact(model_name: str, variable: str,
                        layer: Optional[int] = None,
                        probes_root: Optional[str] = None,
                        readout: str = "pooled") -> dict:
    """Load ``<probes_root>/<model>/<variable>/site_L{l}.pt`` (contract: W
    [n_classes, d_model], b [n_classes], layer, hook, positions [P],
    n_classes, accuracy, normalize "ln"|"none"; per-position W_pos [P, n_classes,
    d_model], b_pos [P, n_classes]). ``layer=None`` picks the site with the highest
    stored accuracy. ``readout`` picks the probe ``probe_logprobs`` reads: "pooled"
    (W, b; the default) or "position" (each position's own W_pos[p], b_pos[p] — the
    probes whose accuracy picks the site). Adds ``path``/``variable``/``readout`` and
    tensorizes ``positions``."""
    vdir = os.path.join(probes_root or probes_root_default(), model_name, variable)
    # site_L[0-9]*.pt: never match the sibling site_baseline.pt artifact
    paths = ([os.path.join(vdir, f"site_L{int(layer)}.pt")] if layer is not None
             else sorted(glob.glob(os.path.join(vdir, "site_L[0-9]*.pt"))))
    if not paths or not os.path.exists(paths[0]):
        raise FileNotFoundError(f"no probe artifact for {model_name}/{variable} "
                                f"(layer={layer}) under {vdir}")
    art, path = max(((torch.load(p, map_location="cpu"), p) for p in paths),
                    key=lambda ap: float(ap[0]["accuracy"]))
    art = dict(art)
    assert art["hook"] == f"blocks.{int(art['layer'])}.hook_resid_post", \
        f"probe artifact hook {art['hook']!r} breaks the site_L{{l}} contract"
    art["positions"] = torch.as_tensor(art["positions"], dtype=torch.long)
    if readout not in ("pooled", "position"):
        raise ValueError(f"unknown probe readout {readout!r}; expected 'pooled' or 'position'")
    if readout == "position" and not ("W_pos" in art and "b_pos" in art):
        raise ValueError(f"{path} has no per-position probes (W_pos/b_pos)")
    art.update(path=path, variable=variable, readout=readout)
    return art


def probe_meta(probe: dict) -> dict:
    """The summary.json meta block recording the probe artifact provenance.

    ``accuracy`` is the artifact's headline number (the position-mean accuracy of the
    PER-POSITION probes) and is what picks the site, but ``probe_logprobs`` reads the
    POOLED W/b readout — the finder contract (probes/train_probes.py:15-16). Those two
    differ a lot (anc0_id/anc0_anchor L0: 0.885 vs 0.706), so both are recorded here;
    ``readout`` names the one the discovery metric actually used (--probe-readout)."""
    return {"variable": probe["variable"], "layer": int(probe["layer"]),
            "hook": probe["hook"], "path": repo_relpath(probe["path"]),
            "accuracy": float(probe["accuracy"]),
            "pooled_accuracy": float(probe["pooled_accuracy"]),
            "readout": probe.get("readout", "pooled"),
            "n_classes": int(probe["n_classes"]),
            "normalize": probe["normalize"],
            "positions": probe["positions"].tolist()}


def probe_capture_hook(probe: dict) -> str:
    """Where the capture hook lives: blocks.{l+1}.hook_resid_pre — the SAME
    value as the probe site resid_post, but downstream of its TL BackwardHook
    wrapper. A tensor captured AT the site bypasses that wrapper, so bwd hooks
    there (EAP's direct-into-site gradients) would never fire, and same-point
    fwd hooks added later (EAP's eval input-construction) would be missed.
    Requires probe layer < n_layers - 1."""
    return f"blocks.{int(probe['layer']) + 1}.hook_resid_pre"


def check_probe_layer(probe: dict, n_layers: int) -> None:
    if int(probe["layer"]) >= n_layers - 1:
        raise ValueError(
            f"probe layer {int(probe['layer'])} needs to be < {n_layers - 1}: "
            "the capture rides blocks.{l+1}.hook_resid_pre (see "
            "probe_capture_hook), and a final-layer probe is just the output "
            "readout anyway")


def lnpre(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """TransformerLens LayerNormPre: center, then scale by RMS (no learned params)."""
    x = x - x.mean(-1, keepdim=True)
    return x / (x.pow(2).mean(-1, keepdim=True) + eps).sqrt()


def probe_logprobs(resid: torch.Tensor, probe: dict) -> torch.Tensor:
    """[B, P, n_classes] probe log-probs from a [B, S, d_model] residual capture, read
    with the pooled probe (W, b) or, for ``readout == "position"``, each position's own
    probe (W_pos[p], b_pos[p])."""
    x = resid[:, probe["positions"].to(resid.device), :]
    if probe["normalize"] == "ln":
        x = lnpre(x)
    if probe.get("readout", "pooled") == "position":
        W = probe["W_pos"].to(device=x.device, dtype=x.dtype)          # [P, C, d]
        b = probe["b_pos"].to(device=x.device, dtype=x.dtype)          # [P, C]
        return (torch.einsum("bpd,pcd->bpc", x, W) + b).log_softmax(-1)
    W = probe["W"].to(device=x.device, dtype=x.dtype)
    b = probe["b"].to(device=x.device, dtype=x.dtype)
    return (x @ W.T + b).log_softmax(-1)


def install_probe_capture(model, hook_name: str) -> dict:
    """PERMANENT forward hook storing each forward's residual at the probe site
    (undetached, so attribution backprops through the probe readout). Permanent
    hooks survive model.hooks() contexts and vendored reset_hooks() calls."""
    store = {"resid": None}

    def _capture(act, hook):
        store["resid"] = act

    model.add_perma_hook(hook_name, _capture)
    return store


def make_probe_metric(probe: dict, capture: dict) -> Callable:
    """EAP-signature metric reading the probe capture instead of the logits.

    ``labels`` transport the clean-reference probe log-probs [B, P, C] (the
    vendored EAP-IG passes labels through opaquely — same transport trick as
    the logit-diff labels). loss=True -> KL(clean_ref || current) per example
    (attribution); loss=False -> probe argmax agreement per example (eval)."""

    def metric(logits, clean_logits, input_lengths, labels,
               mean: bool = True, loss: bool = False):
        cur = probe_logprobs(capture["resid"], probe)              # [B, P, C]
        ref = labels.to(device=cur.device, dtype=cur.dtype)        # [B, P, C]
        if loss:
            d = (ref.exp() * (ref - cur)).sum(-1).mean(-1)         # [B] KL
        else:
            d = (cur.argmax(-1) == ref.argmax(-1)).float().mean(-1)  # [B]
        return d.mean() if mean else d

    return metric


def clean_probe_reference(model, capture: dict, clean_tokens: torch.Tensor,
                          probe: dict, batch_size: int) -> torch.Tensor:
    """[N, P, C] clean probe log-probs, batched exactly like the consumer's
    dataloader so the clean-baseline agreement is exactly 1.0."""
    outs = []
    with torch.no_grad():
        for i in range(0, clean_tokens.shape[0], batch_size):
            model(clean_tokens[i:i + batch_size])
            outs.append(probe_logprobs(capture["resid"], probe).float())
    return torch.cat(outs)


def probe_baselines(model, dataloader, metric_eval) -> tuple:
    """(clean, corrupted) mean probe agreement. The vendored evaluate_baseline
    runs the corrupted forward BEFORE the clean one, so a capture-reading
    metric would score the wrong residual — probe mode uses this loop."""
    cl, co = [], []
    with torch.no_grad():
        for clean, corrupted, label in dataloader:
            model(clean)
            cl.append(metric_eval(None, None, None, label))
            model(corrupted)
            co.append(metric_eval(None, None, None, label))
    return torch.cat(cl).mean().item(), torch.cat(co).mean().item()


def probe_cutoff_indices(cfg, layer: int) -> tuple:
    """(n_forward_kept, n_backward_kept): writers/readers at layers <= the
    probe layer, in EAP-IG's index scheme (see save_circuit_pt)."""
    H = int(cfg.n_heads)
    return 1 + (layer + 1) * (H + 1), (layer + 1) * (3 * H + 1)


def retarget_graph_to_probe(graph, probe: dict) -> tuple:
    """Instance-patch a vendored Graph (never the vendored source) so its
    'logits' node IS the probe site: in_hook moves to the probe hook (direct-
    into-site edges get scored, and reconstructed there during evaluation) and
    prev_index clamps to the causal cutoff so only writers at/below the probe
    layer feed that node in BOTH attribution and evaluation. Edges that cannot
    reach the probe leave real_edge_mask. Returns (fwd_cutoff, bwd_cutoff)."""
    fwd_cut, bwd_cut = probe_cutoff_indices(graph.cfg, int(probe["layer"]))
    node = graph.nodes["logits"]
    node.in_hook = probe["hook"]
    for e in node.parent_edges:                # Edge.hook was copied at build time
        e.hook = probe["hook"]
    orig = graph.prev_index
    graph.prev_index = lambda n: fwd_cut if n is node else orig(n)
    graph.real_edge_mask[fwd_cut:, :] = False
    graph.real_edge_mask[:, bwd_cut:-1] = False
    return fwd_cut, bwd_cut


# --------------------------------------------------------------------------- #
# C4 positional (--use-pos-embed) — helpers shared with acdc.py's flags        #
# --------------------------------------------------------------------------- #
POS_EMBED_HOOK = "hook_pos_embed"


def retarget_graph_to_positional(graph, model) -> None:
    """Repurpose the 'input' source (normally the token embedding hook_embed) to
    the POSITIONAL embedding (hook_pos_embed) on the LIVE graph only (never the
    vendored source), so attribution scores which edges route positional info
    (C4). The token embedding stays in the always-present residual baseline. The
    corrupted side must scramble positions (see scrambled_pos_corrupted) — clean
    and corrupted pos_embed are otherwise identical (position-only) so every
    score would be zero. Single-source by design (EAP has one input source);
    unlike ACDC's use_pos_embed this does NOT keep a separate token source."""
    if POS_EMBED_HOOK not in model.hook_dict:
        raise ValueError(
            f"model has no {POS_EMBED_HOOK} (positional_embedding_type="
            f"{model.cfg.positional_embedding_type!r}); --use-pos-embed needs a "
            "standard learned positional embedding")
    graph.nodes["input"].out_hook = POS_EMBED_HOOK


@contextlib.contextmanager
def scrambled_pos_corrupted(eap_mods, seed: int = 49) -> Iterator[None]:
    """Patch make_hooks_and_matrices in the vendored modules so the CORRUPTED
    forward runs on position-scrambled positional embeddings (mirrors ACDC's
    use_pos_embed scramble, seed 49): a shuffle hook on hook_pos_embed is
    prepended to the returned corrupted fwd-hook list, so it runs BEFORE the
    input(=pos) capture and the captured corrupted pos differs from the clean
    pos. eap_mods = (eap_utils, eap_attribute, eap_evaluate)."""
    orig = eap_mods[0].make_hooks_and_matrices

    def _scramble(z, hook):
        g = torch.Generator(device="cpu").manual_seed(seed)
        perm = torch.randperm(z.shape[1], generator=g).to(z.device)
        return z[:, perm]

    def patched(model, graph, batch_size, n_pos, scores):
        (fwd_corrupted, fwd_clean, bwd), ad = orig(model, graph, batch_size, n_pos, scores)
        return ([(POS_EMBED_HOOK, _scramble)] + fwd_corrupted, fwd_clean, bwd), ad

    for mod in eap_mods:
        mod.make_hooks_and_matrices = patched
    try:
        yield
    finally:
        for mod in eap_mods:
            mod.make_hooks_and_matrices = orig


# ---- Data: list-of-batches "dataloader" from a Segrev3Pairs -----------------
def build_dataloader_from_pairs(pairs, batch_size: int, labels=None) -> list:
    """labels[:, 0] = correct (clean answer), labels[:, 1] = incorrect;
    probe mode passes its own labels tensor (clean probe log-probs)."""
    n_examples = pairs.clean_tokens.shape[0]
    if labels is None:
        labels = pack_labels(pairs)
    batches = []
    for i in range(0, n_examples, batch_size):
        batches.append((
            pairs.clean_tokens[i:i + batch_size],
            pairs.corrupted_tokens[i:i + batch_size],
            labels[i:i + batch_size],
        ))
    return batches


def slice_pairs(pairs, start: int, stop: int) -> Segrev3Pairs:
    """A ``Segrev3Pairs`` with every tensor field sliced ``[start:stop]``."""
    sliced = {}
    for f in dataclasses.fields(pairs):
        v = getattr(pairs, f.name)
        sliced[f.name] = v[start:stop] if torch.is_tensor(v) else v
    return dataclasses.replace(pairs, **sliced)


def head_of(node_name: str) -> Optional[str]:
    """'a3.h1' -> 'a3.h1' ; 'input'/'m0'/'logits' -> None."""
    return node_name if node_name.startswith("a") and ".h" in node_name else None


def save_circuit_pt(model_or_cfg, nodes, edges, pt_path) -> str:
    """Write ``circuit.pt`` in the EAP-IG ``Graph`` format from a normalized
    ``(nodes, edges)`` circuit, so ACDC, Edge Pruning and the smallest-faithful
    circuit all emit the SAME loadable artifact EAP does
    (``eap.graph.Graph.from_pt`` round-trips it):
    ``{cfg, src_nodes, dst_nodes, edges_scores, edges_in_graph, nodes_in_graph}``.

    Builds the tensors NATIVELY (no ``eap`` import — edge_pruning runs on
    Modal, where the vendored EAP-IG package is NOT installed), replicating
    EAP-IG's forward/backward node indexing (github_repos/EAP-IG/src/eap/graph.py):
      n_forward  = 1 + n_layers*(n_heads+1)            # src nodes (logits excluded)
      n_backward = n_layers*(3*n_heads+1) + 1          # q/k/v + mlp dst, then logits
      input -> 0; a{l}.h{h} -> 1 + l*(H+1) + h; m{l} -> 1 + l*(H+1) + H
      a{l}.h{h}<qkv> -> l*(3H+1) + 'qkv'.index(qkv)*H + h; m{l} -> l*(3H+1)+3H;
      logits -> last backward index (-1)

    ``model_or_cfg`` is a HookedTransformer (``.cfg``) or a cfg dict (only
    n_layers/n_heads/parallel_attn_mlp/d_model are used); ``nodes`` are the
    in-circuit node names ('logits' is always in-graph and skipped); ``edges``
    are ``(parent, child, qkv, score)`` drawer tuples. Returns the path."""
    if hasattr(model_or_cfg, "cfg"):               # a HookedTransformer
        mc = model_or_cfg.cfg
        n_layers, n_heads, d_model = int(mc.n_layers), int(mc.n_heads), int(mc.d_model)
        parallel = bool(mc.parallel_attn_mlp)
    else:
        c = dict(model_or_cfg)
        n_layers, n_heads, d_model = int(c["n_layers"]), int(c["n_heads"]), int(c["d_model"])
        parallel = bool(c.get("parallel_attn_mlp", False))
    cfg = {"n_layers": n_layers, "n_heads": n_heads,
           "parallel_attn_mlp": parallel, "d_model": d_model}

    H = n_heads
    n_forward = 1 + n_layers * (H + 1)
    n_backward = n_layers * (3 * H + 1) + 1

    # src/dst node-name lists in EAP-IG's canonical order (Graph.to_pt / get_dst_nodes)
    src_nodes = ["input"]
    for l in range(n_layers):
        src_nodes += [f"a{l}.h{h}" for h in range(H)] + [f"m{l}"]
    dst_nodes = []
    for l in range(n_layers):
        for letter in "qkv":
            dst_nodes += [f"a{l}.h{h}<{letter}>" for h in range(H)]
        dst_nodes.append(f"m{l}")
    dst_nodes.append("logits")

    def _fwd(name):
        if name in ("input", "tok_embed", "pos_embed"):
            return 0                                    # pos/tok collapse onto the input source row
        if name[0] == "m":
            return 1 + int(name[1:]) * (H + 1) + H
        lp, hp = name.split(".")                    # a{l}.h{h}
        return 1 + int(lp[1:]) * (H + 1) + int(hp[1:])

    def _bwd(name, qkv):
        if name == "logits":
            return n_backward - 1                    # EAP-IG uses -1 (the last col)
        if name[0] == "m":
            return int(name[1:]) * (3 * H + 1) + 3 * H
        lp, hp = name.split(".")                    # a{l}.h{h}
        if qkv not in ("q", "k", "v"):
            raise ValueError(f"save_circuit_pt: edge into {name!r} needs qkv, got {qkv!r}")
        return int(lp[1:]) * (3 * H + 1) + "qkv".index(qkv) * H + int(hp[1:])

    edges_scores = torch.zeros((n_forward, n_backward), dtype=torch.float32)
    edges_in_graph = torch.zeros((n_forward, n_backward), dtype=torch.bool)
    nodes_in_graph = torch.zeros(n_forward, dtype=torch.bool)

    for parent, child, qkv, score in edges:
        fi, bi = _fwd(parent), _bwd(child, qkv)
        edges_in_graph[fi, bi] = True
        edges_scores[fi, bi] = float(score)
    for nm in nodes:
        if nm == "logits":                          # logits is a dst-only node
            continue
        nodes_in_graph[_fwd(nm)] = True

    torch.save({"cfg": cfg, "src_nodes": src_nodes, "dst_nodes": dst_nodes,
                "edges_scores": edges_scores, "edges_in_graph": edges_in_graph,
                "nodes_in_graph": nodes_in_graph}, pt_path)
    return pt_path


# --------------------------------------------------------------------------- #
# Dangling-node pruning (shared by ACDC + Edge Pruning + the smallest loader)  #
# --------------------------------------------------------------------------- #
_PRUNE_TERMINALS = {"input", "embed", "tok_embed", "pos_embed", "logits"}


def _edge_endpoints(e) -> tuple:
    """(parent, child) from a drawer tuple ``(p, c, qkv, score)`` OR a
    ``{"parent","child",...}`` dict — so prune works on either representation."""
    if isinstance(e, dict):
        return e["parent"], e["child"]
    return e[0], e[1]


def prune_dangling(nodes, edges) -> tuple:
    """Drop every head / MLP node that is NOT on an ``input``→``logits`` path —
    i.e. has no in-graph INCOMING edge or no in-graph OUTGOING edge — iterating to
    a fixpoint so cascades are removed (a head left dangling by an earlier removal
    goes too). ``input``/``embed``/``logits`` are terminals and always kept.

    This is exactly EAP-IG's ``Graph.prune()`` semantics; EAP already applies it
    (``apply_topn(..., prune=True)``), so calling this on an EAP circuit is a
    no-op. ACDC (greedy edge removal) and Edge Pruning leave dangling heads, which
    read pure mean under edge-level ablation (data-independent) yet inflate
    ``n_heads`` and the selection score — this removes them.

    Accepts drawer tuples or edge-dicts; returns ``(pruned_nodes, pruned_edges)``
    preserving the input edge representation and order. Idempotent."""
    nodes = list(nodes)
    edges = list(edges)
    while True:
        incoming, outgoing = set(), set()
        for e in edges:
            p, c = _edge_endpoints(e)
            outgoing.add(p)
            incoming.add(c)
        drop = {n for n in nodes
                if n not in _PRUNE_TERMINALS
                and (n not in incoming or n not in outgoing)}
        if not drop:
            return nodes, edges
        edges = [e for e in edges
                 if not ({*_edge_endpoints(e)} & drop)]
        nodes = [n for n in nodes if n not in drop]


def heads_mlps_from_nodes(nodes) -> tuple:
    """``(sorted [(l,h),...] heads, sorted [l,...] mlp layers)`` from a node-name
    list — the canonical re-derivation after pruning so circuit.json's ``heads`` /
    ``mlps`` match the surviving node set."""
    heads, mlps = [], []
    for nm in nodes:
        if nm.startswith("a") and ".h" in nm:
            l, h = nm[1:].split(".h")
            heads.append((int(l), int(h)))
        elif len(nm) > 1 and nm[0] == "m" and nm[1:].isdigit():
            mlps.append(int(nm[1:]))
    return sorted(heads), sorted(mlps)


def write_candidate_folder(model_name: str, circuit: str, circuit_name: str, *,
                           smoke: bool, meta: dict, nodes: list, edges: list,
                           summary: dict, model_or_cfg=None, graph=None,
                           circuit_json_method: Optional[str] = None,
                           uniform_penwidth: Optional[float] = None) -> dict:
    """Write ONE candidate-circuit folder — the uniform on-disk contract of every
    discovery method: circuits_data/<model>/<circuit>/circuits/<circuit_name>/
    {summary.json, circuit.json, circuit.pt, circuit.gv} + circuit.png under the
    mirrored figures tree. ``graph`` (a vendored EAP-IG Graph) writes
    circuit.json/.pt in its own format (eap); otherwise the normalized form +
    ``save_circuit_pt(model_or_cfg, ...)`` (acdc / edge_pruning). ``edges`` are
    drawer tuples ``(parent, child, qkv, score)``; ``summary`` carries the
    method-specific payload. Returns the summary dict written (meta + circuit +
    payload + files)."""
    ddir, fdir = circuit_dirs(model_name, circuit_name, circuit=circuit, smoke=smoke)
    os.makedirs(ddir, exist_ok=True)
    os.makedirs(fdir, exist_ok=True)
    summary_path = os.path.join(ddir, "summary.json")
    json_path = os.path.join(ddir, "circuit.json")
    pt_path = os.path.join(ddir, "circuit.pt")
    gv_path = os.path.join(ddir, "circuit.gv")
    png_path = os.path.join(fdir, "circuit.png")

    if graph is not None:                      # vendored EAP-IG Graph formats
        graph.to_json(json_path)
        graph.to_pt(pt_path)
    else:                                      # normalized circuit.json + native .pt
        heads, mlp_layers = heads_mlps_from_nodes(nodes)
        payload = {
            "meta": meta,
            "nodes": nodes,
            "heads": [f"L{l}H{h}" for (l, h) in heads],
            "mlps": [f"m{l}" for l in mlp_layers],
            "edges": [{"parent": p, "child": c, "qkv": q, "score": s}
                      for (p, c, q, s) in edges],
        }
        if circuit_json_method is not None:
            payload["method"] = circuit_json_method
        with open(json_path, "w") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        save_circuit_pt(model_or_cfg, nodes, edges, pt_path)

    png_ok, png_err = True, None
    try:
        draw_circuit_png(nodes, edges, png_path, gv_path, seed=0,
                         uniform_penwidth=uniform_penwidth)
    except (OSError, ValueError) as exc:       # dot/pygraphviz render failures
        png_ok, png_err = False, repr(exc)
        print(f"[{circuit_name}] PNG render skipped: {png_err}", flush=True)

    result = {"meta": meta, "circuit": circuit_name, **summary,
              "files": {"summary_json": repo_relpath(summary_path),
                        "circuit_json": repo_relpath(json_path),
                        "circuit_pt": repo_relpath(pt_path), "circuit_gv": repo_relpath(gv_path),
                        "circuit_png": repo_relpath(png_path) if png_ok else None}}
    if not png_ok:
        result["png_error"] = png_err
    with open(summary_path, "w") as f:         # written LAST (includes files)
        json.dump(result, f, indent=2, sort_keys=True)
    return result


# --------------------------------------------------------------------------- #
# Main pipeline                                                                #
# --------------------------------------------------------------------------- #
def _extract_candidate(graph) -> tuple:
    """(head (l,h) set, mlp l set, edge dicts) of the graph's current in-graph state —
    the same extraction the top-n sweep loop does, in the form the mean-ablation eval wants."""
    heads, mlps = set(), set()
    for nm, nd in graph.nodes.items():
        if not nd.in_graph:
            continue
        if head_of(nm) is not None:
            l, h = nm[1:].split(".h")
            heads.add((int(l), int(h)))
        elif len(nm) > 1 and nm[0] == "m" and nm[1:].isdigit():
            mlps.add(int(nm[1:]))
    edges = [{"parent": e.parent.name, "child": e.child.name, "qkv": e.qkv,
              "score": float(e.score)} for e in graph.edges.values() if e.in_graph]
    edges.sort(key=lambda d: -abs(d["score"]))
    return heads, mlps, edges


def _rungs(lo: int, hi: int, ratio: float) -> list:
    """Geometric rung set covering ``[lo, hi]`` inclusive (always at least [hi])."""
    out, n = [], max(1, lo)
    while n < hi:
        out.append(n)
        n = max(n + 1, math.ceil(n * ratio))
    return sorted(set(out + [hi]))


def _search_smallest_topn(model, graph, model_name: str, circuit: str, target: float,
                          live_edges: int, device, seed: int,
                          search_n: Optional[int] = None) -> tuple:
    """Find the EAP-IG top-n MAXIMIZING the smallest-faithful size score
    (1 - n_heads/n_heads_full)(1 - n_edges/n_edges_full) subject to the STRICT floor:
    edge-level per-position mean-ablation WHOLE-SEQUENCE accuracy > ``target``.

    Faithfulness is NOT monotone in top-n (one more edge can break a circuit), so this
    SCANS a declared geometric rung set and takes the objective's argmax over the passing
    rungs — a bisection on a non-monotone predicate can return a non-minimal n or miss a
    passing one entirely. Coarse sweep at SEARCH_COARSE_RATIO up to the FIRST passing rung
    (higher rungs only add edges, and the objective is decreasing in n_edges), then a
    SEARCH_FINE_RATIO sweep of the bracket below it. The metric + score are the smallest
    stage's own (imported, never reimplemented); the SELECTION pool window keeps these rows
    disjoint from both the attribution set and the rows the smallest stage finally scores on,
    and the edge denominator is the method-independent live budget.
    Returns (winner|None, {n: result})."""
    import smallest_faithful_circuit as SFC       # exact metric + score, no duplication
    import model_registry as MG
    n_eval = SFC.SMALLEST_EVAL_N if search_n is None else int(search_n)
    pairs, _ = MG.load_circuit_pairs(model_name, circuit, n=n_eval, device=str(device),
                                     consumer="selection")
    outs = pairs.correct.to(pairs.clean_tokens.device)
    attn_only = bool(model.cfg.attn_only)
    L, H = int(model.cfg.n_layers), int(model.cfg.n_heads)
    full_heads = L * H + (0 if attn_only else L)          # matches smallest's n_heads_full
    full_edges = MG.live_edges(MG.get_spec(model_name))
    means = SFC._edge_site_means(model, pairs.clean_tokens[:SFC.MEAN_N_PROD], SFC.BATCH_CAP_PROD)
    cache: dict = {}

    def ev(n: int) -> dict:
        n = max(1, min(int(n), live_edges))
        if n in cache:
            return cache[n]
        with force_zeros_device(device):
            graph.apply_topn(n, absolute=True, level="edge", reset=True, prune=True)
        heads, mlps, edges = _extract_candidate(graph)
        # attn_only drops the graph's phantom m{l} edges (same filter as eval_child)
        declared = SFC._edges_to_declared(edges, attn_only)
        fwd = lambda t: SFC._edge_forward(model, t, declared, means, heads, mlps, L, H, attn_only)
        seq = SFC._perf(fwd, pairs, outs, SFC.BATCH_CAP_PROD)[1]
        nh, ne = len(heads), len(edges)
        cache[n] = {"n": n, "seq_acc": seq, "n_heads": nh, "n_edges": ne,
                    "score": (1.0 - nh / full_heads) * (1.0 - ne / full_edges),
                    "passes": seq > target}
        return cache[n]

    coarse = _rungs(1, live_edges, SEARCH_COARSE_RATIO)
    first = None
    for n in coarse:
        if ev(n)["passes"]:
            first = n
            break
    if first is None:
        return None, cache                       # no rung, up to the full graph, clears the floor
    below = [n for n in coarse if n < first]
    for n in _rungs(max(below, default=0) + 1, first, SEARCH_FINE_RATIO):
        ev(n)
    winner = max((r for r in cache.values() if r["passes"]),
                 key=lambda r: (r["score"], -r["n_edges"], -r["n"]))
    return winner, cache


def _print_search_table(cache: dict, target: float, winner: Optional[dict]) -> None:
    print(f"\n== smallest-faithful top-n scan (mean-ablation whole-seq acc > {target:.2f}; "
          f"argmax (1-h/H)(1-e/E) over {len(cache)} rungs) ==")
    print(f"    {'top-n':>6} {'edges':>6} {'heads':>5} {'seq_acc':>8} {'score':>10}  gate")
    for n in sorted(cache):
        r = cache[n]
        mark = "PASS" if r["passes"] else "fail"
        star = "  <-- WINNER" if winner and r["n"] == winner["n"] else ""
        print(f"    {r['n']:>6} {r['n_edges']:>6} {r['n_heads']:>5} {r['seq_acc']:>8.4f} "
              f"{r['score']:>10.5f}  {mark}{star}")


def run_eap(model_name: str, topns, circuit: str, smoke: bool = False,
            device=None, seed: int = DEFAULT_SEED,
            topn_regime: str = CANON_TOPN_REGIME,
            method: str = "EAP-IG-inputs", probe: Optional[str] = None,
            probe_layer: Optional[int] = None,
            probes_root: Optional[str] = None,
            zero_ablation: bool = False, use_pos_embed: bool = False,
            target_seq_acc: Optional[float] = None,
            search_n: Optional[int] = None,
            probe_readout: str = "pooled") -> dict:
    if probe is not None and use_pos_embed:
        raise ValueError("--probe with --use-pos-embed is not supported: the "
                         "probe metric already retargets the input reader")
    ckpt = set_checkpoint_env(model_name)
    device = resolve_device(device)
    probe_art = (load_probe_artifact(model_name, probe, layer=probe_layer,
                                     probes_root=probes_root, readout=probe_readout)
                 if probe is not None else None)

    # vendored stack — imported AFTER the env export, then monkeypatched
    eap_utils, eap_attribute, eap_evaluate = _import_vendored_eap()
    from eap.graph import Graph
    from eap.attribute import attribute
    from eap.evaluate import evaluate_graph, evaluate_baseline
    _install_tokenize_patch(eap_utils, eap_attribute, eap_evaluate)
    from metrics import logit_diff as task_logit_diff

    attr_bs, attr_n, ig_steps = ATTR_BS, ATTR_N, IG_STEPS   # PINNED (see top)
    eval_n = SMOKE_EVAL_N if smoke else EVAL_N
    eval_bs = EVAL_BS
    spec = model_registry.get_spec(model_name)
    live_edges = model_registry.live_edges(spec)      # THE method-independent edge budget
    ladder = (model_registry.topns(spec, topn_regime) if topns is None
              else TopnLadder(sorted(set(int(n) for n in topns)), "explicit", live_edges))
    topns = ladder.topns
    over = [n for n in topns if n > live_edges]
    if over:
        # The absolute [50,200,800] ladder aliased eap_top800 to the FULL unpruned graph at
        # small (800 = 988% of its 81 live edges) and credited output-constant scopes.
        raise ValueError(
            f"top-n rungs {over} exceed {model_name}'s live edge budget {live_edges} "
            f"({spec.n_layers}L x {spec.n_heads}H attn_only={spec.attn_only}): such a rung "
            f"is the whole unpruned graph, not a circuit. Use the 'frac' regime.")
    # C4: zero-ablation drops the corrupted patch. EAP-IG-inputs is patching-only
    # (vendored assert), so zero uses EAP-IG-activations (the IG method that
    # supports intervention='zero'); evaluate_graph honors 'zero' for any method.
    attr_method = "EAP-IG-activations" if zero_ablation else method
    intervention = "zero" if zero_ablation else "patching"
    cand_suffix = ("_zero" if zero_ablation else "") + ("_pos" if use_pos_embed else "")

    print("=" * 78)
    print(f"EAP-IG circuit discovery | model={model_name} method={method}")
    print(f"  circuit = {circuit}")
    print(f"  checkpoint = {ckpt}")
    print(f"  PINNED attribution params: attr_bs={attr_bs} attr_n={attr_n} "
          f"ig_steps={ig_steps}")
    print(f"  eval: n={eval_n} bs={eval_bs} | smoke={smoke}")
    print(f"  ladder[{ladder.regime}]: topns={topns} of {live_edges} live edges "
          f"({spec.n_layers}L x {spec.n_heads}H attn_only={spec.attn_only})")
    if zero_ablation or use_pos_embed:
        print(f"  C4 flags: zero_ablation={zero_ablation} use_pos_embed="
              f"{use_pos_embed} | attr_method={attr_method} intervention={intervention}")
    if probe_art is not None:
        print(f"  probe: {probe} layer={int(probe_art['layer'])} "
              f"hook={probe_art['hook']} accuracy={float(probe_art['accuracy']):.4f}\n"
              f"         artifact = {probe_art['path']}")
    print("=" * 78)

    wait_for_headroom(context=f"eap.py before attribution ({model_name})")
    model = model_registry.load_model(model_name, device=device)
    _dev = next(model.parameters()).device
    # CRITICAL flags for EAP-IG hook support (set on this local instance).
    model.cfg.use_attn_result = True
    model.cfg.use_split_qkv_input = True
    model.cfg.use_hook_mlp_in = True
    capture = None
    if probe_art is not None:
        check_probe_layer(probe_art, int(model.cfg.n_layers))
        capture = install_probe_capture(model, probe_capture_hook(probe_art))

    # ---- data: ONE dataset feeds the whole run (attribution shared by all top-n)
    # Pairs whose clean and corrupt outputs agree everywhere are dropped at load (pairs-v2),
    # so a low-entropy output (e.g. a 0/1 latch) can leave fewer than attr_n + eval_n of the
    # first APP_SIZES["eap"] pairs: read a longer prefix of the same pool (surviving pairs keep
    # their order), never reaching the selection window.
    n_pairs = model_registry.APP_SIZES["eap"]
    while True:
        all_pairs, prov = load_run_pairs(
            model_name, circuit, num_examples=n_pairs,
            seed=seed, smoke=smoke, device=str(_dev))
        n_avail = all_pairs.clean_tokens.shape[0]
        if (n_avail >= attr_n + eval_n
                or 2 * n_pairs > model_registry.POOL_WINDOWS["selection"]):
            break
        n_pairs *= 2
    data_path = prov["path"]
    if n_avail < attr_n + eval_n:
        raise RuntimeError(f"dataset too small: {n_avail} < attr {attr_n} + "
                           f"eval {eval_n}")
    attr_pairs = slice_pairs(all_pairs, 0, attr_n)
    eval_pairs = slice_pairs(all_pairs, attr_n, attr_n + eval_n)
    attr_labels = eval_labels = None
    if probe_art is not None:
        attr_labels = clean_probe_reference(model, capture,
                                            attr_pairs.clean_tokens,
                                            probe_art, attr_bs)
        eval_labels = clean_probe_reference(model, capture,
                                            eval_pairs.clean_tokens,
                                            probe_art, eval_bs)
    attr_dl = build_dataloader_from_pairs(attr_pairs, attr_bs, labels=attr_labels)
    eval_dl = build_dataloader_from_pairs(eval_pairs, eval_bs, labels=eval_labels)
    print(f"data: {'in-memory ' + circuit if data_path is None else data_path} | "
          f"attribution {attr_n} ex / bs {attr_bs} ({len(attr_dl)} batches); "
          f"eval {eval_n} ex / bs {eval_bs} ({len(eval_dl)} batches)")

    # ---- metric + self-check ----
    # Answer positions come from the loaded pairs (variable length / subset).
    answer_positions = tuple(int(p) for p in all_pairs.answer_positions[0].tolist())
    if probe_art is None:
        # sanity check vs the task's own logit_diff
        metric = make_logit_diff_metric(answer_positions)
        with torch.no_grad():
            cl = model(attr_pairs.clean_tokens)
        mine = metric(cl, None, None, pack_labels(attr_pairs),
                      mean=True, loss=False).item()
        theirs = task_logit_diff(cl, attr_pairs).item()
        print(f"metric self-check: mine={mine:+.4f}  task.logit_diff={theirs:+.4f}  "
              f"(|diff|={abs(mine - theirs):.2e})")
        assert abs(mine - theirs) < 1e-3, "metric disagrees with metrics.logit_diff"
    else:
        metric = make_probe_metric(probe_art, capture)
        with torch.no_grad():
            model(attr_pairs.clean_tokens)
        kl0 = metric(None, None, None, attr_labels, mean=True, loss=True).item()
        agr0 = metric(None, None, None, attr_labels, mean=True, loss=False).item()
        print(f"probe metric self-check: clean KL={kl0:.3e} (expect ~0)  "
              f"clean agreement={agr0:.4f} (expect 1.0)")
        assert kl0 < 1e-4, "probe clean reference disagrees with a fresh clean forward"

    # ---- graph + attribution ----
    graph = Graph.from_model(model)
    L, H = int(model.cfg.n_layers), int(model.cfg.n_heads)
    attn_only = bool(model.cfg.attn_only)
    graph_real_edges = int(graph.real_edge_mask.sum())     # EAP-IG's count, phantoms included
    if attn_only:      # the Graph always builds m{l}; those hooks never fire (scores == 0)
        rows, cols = phantom_mlp_indices(L, H)
        graph.real_edge_mask[rows, :] = False
        graph.real_edge_mask[:, cols] = False
    fwd_cut = bwd_cut = None
    if probe_art is not None:
        fwd_cut, bwd_cut = retarget_graph_to_probe(graph, probe_art)
    if use_pos_embed:
        retarget_graph_to_positional(graph, model)
    # What apply_topn can actually select: == live_edges normally, SMALLER under --probe
    # (retarget_graph_to_probe drops every edge that cannot reach the probe site).
    selectable = int(graph.real_edge_mask.sum())
    assert selectable == live_edges or probe_art is not None, \
        f"live graph holds {selectable} edges but the budget says {live_edges}"
    if probe_art is not None and ladder.regime == "frac":
        # Scale the frac rungs by the PROBE-REACHABLE budget: rungs of the full live budget
        # all clamp to `selectable` and collapse into one or two distinct circuits.
        ladder = topn_ladder(selectable, "frac")
        topns = ladder.topns
        print(f"  probe ladder[frac]: topns={topns} of {selectable} probe-reachable edges")
    graph_info = {"n_forward": int(graph.n_forward),
                  "n_backward": int(graph.n_backward),
                  "real_edges": live_edges,               # phantom m{l} slots excluded
                  "graph_real_edges": graph_real_edges,   # EAP-IG's inflated count
                  "selectable_edges": selectable,
                  "n_edges_universe": live_edges}
    print(f"graph: {graph_info}")

    # scramble the corrupted positions only when pos-embed patches (not zero mode)
    eap_mods = (eap_utils, eap_attribute, eap_evaluate)
    scramble = use_pos_embed and not zero_ablation
    def _scramble_ctx():
        return scrambled_pos_corrupted(eap_mods) if scramble else contextlib.nullcontext()

    with force_zeros_device(_dev), _scramble_ctx():
        attribute(model, graph, attr_dl, sum_reduced(metric), method=attr_method,
                  ig_steps=ig_steps, intervention=intervention, quiet=True)

    nz = int((graph.scores.abs() > 0).sum())
    print(f"attribution done (method={attr_method}); nonzero edge scores: "
          f"{nz} / {graph.scores.numel()}; max|score|={graph.scores.abs().max():.4f}")
    if attn_only:
        rows, cols = phantom_mlp_indices(L, H)
        assert not (graph.scores[rows, :].any() or graph.scores[:, cols].any()), \
            "phantom m{l} slots scored nonzero on an attn_only model"

    if probe_art is not None:
        # Causal cutoff: writers above the probe layer cannot feed the probe.
        spill = (float(graph.scores[fwd_cut:, :].abs().max())
                 if fwd_cut < graph.scores.shape[0] else 0.0)
        graph.scores[fwd_cut:, :] = 0
        graph.scores[:, bwd_cut:-1] = 0
        assert not graph.scores[~graph.real_edge_mask].any(), \
            "causal cutoff left nonzero scores outside the probe's edge universe"
        print(f"probe causal cutoff (writers above layer {int(probe_art['layer'])}): "
              f"pre-mask max|score|={spill:.3e}; all such edges now exactly 0")

    # ---- baselines (full model on clean; corrupted lower bound) ----
    metric_eval = partial(metric, loss=False, mean=False)
    if probe_art is None:
        with force_zeros_device(_dev):
            clean_baseline = evaluate_baseline(model, eval_dl, metric_eval,
                                               quiet=True).mean().item()
            corrupt_baseline = evaluate_baseline(model, eval_dl, metric_eval,
                                                 run_corrupted=True,
                                                 quiet=True).mean().item()
        print(f"baseline logit-diff: clean={clean_baseline:+.4f}  "
              f"corrupted={corrupt_baseline:+.4f}")
    else:
        clean_baseline, corrupt_baseline = probe_baselines(model, eval_dl,
                                                           metric_eval)
        print(f"baseline probe agreement: clean={clean_baseline:.4f}  "
              f"corrupted={corrupt_baseline:.4f}")

    # --target-seq-acc: search the top-n MAXIMIZING (1-h/H)(1-e/E) subject to the STRICT
    # edge-level mean-ablation whole-seq accuracy floor, then emit ONLY that winner below.
    strict_by_n: dict = {}
    if target_seq_acc is not None:
        if probe_art is not None:
            raise ValueError("--target-seq-acc is incompatible with --probe: the whole-sequence "
                             "mean-ablation floor is an OUTPUT-logit metric, not a probe readout")
        winner, search_cache = _search_smallest_topn(
            model, graph, model_name, circuit, target_seq_acc, selectable, _dev, seed,
            search_n=search_n)
        _print_search_table(search_cache, target_seq_acc, winner)
        if winner is None:
            best = max(search_cache.values(), key=lambda r: r["seq_acc"])
            print(f"\nNO top-n reaches whole-seq acc > {target_seq_acc:.2f} "
                  f"(best available {best['seq_acc']:.4f} at top-{best['n']}); nothing emitted.")
            return {}
        topns = [winner["n"]]
        # The rung is the SEARCH's argmax, not a frac/abs ladder rung: say so in the payload.
        ladder = TopnLadder(topns, "search", ladder.live_edges)
        strict_by_n[winner["n"]] = {
            "selection_target_seq_acc": target_seq_acc,
            "whole_sequence_absolute_accuracy": winner["seq_acc"],
            "selection_score": winner["score"],
            "selection_strategy": "(1 - n_heads/n_heads_full) * (1 - n_edges/n_edges_full)",
            "selection_metric": "edge_level_per_position_mean_ablation_whole_sequence_accuracy",
        }
        print(f"\n== winner: eap_top{winner['n']}  seq_acc={winner['seq_acc']:.4f}  "
              f"heads={winner['n_heads']} edges={winner['n_edges']}  score={winner['score']:.5f} ==")

    results = {}
    for n in tqdm(topns, desc="EAP-IG top-n sweep", unit="topn"):
        applied = min(n, selectable)              # only reachable under --probe (see above)
        if applied < n:
            tqdm.write(f"  note: top-{n} clamped to the {selectable} edges that can reach "
                       f"the probe site (of {live_edges} live).")
        with force_zeros_device(_dev), _scramble_ctx():
            graph.apply_topn(applied, absolute=True, level="edge", reset=True, prune=True)
            circ = evaluate_graph(model, graph, eval_dl, metric_eval,
                                  intervention=intervention, quiet=True).mean().item()
        n_edges = int(graph.count_included_edges())
        n_nodes = int(graph.count_included_nodes())
        ratio = recovery_ratio(circ, clean_baseline, corrupt_baseline)
        heads_n = sorted(nm for nm in graph.nodes
                         if head_of(nm) is not None and graph.nodes[nm].in_graph)
        surviving = [{"parent": e.parent.name, "child": e.child.name,
                      "qkv": e.qkv, "score": float(e.score)}
                     for e in graph.edges.values() if e.in_graph]
        surviving.sort(key=lambda d: -abs(d["score"]))
        nodes_in = [nm for nm, nd in graph.nodes.items() if nd.in_graph]
        mlps_in = sorted(nm for nm in nodes_in
                         if len(nm) > 1 and nm[0] == "m" and nm[1:].isdigit())
        edges_in = [(d["parent"], d["child"], d["qkv"], d["score"])
                    for d in surviving]

        payload = {
            "topn": n,
            "topn_ladder": {"regime": ladder.regime, "topns": topns,
                            "live_edges": ladder.live_edges},
            "attribution": {"method": method, "ig_steps": ig_steps,
                            "attr_examples": attr_n, "attr_batch_size": attr_bs,
                            # regime marker: batch-summed loss / total_items (see sum_reduced)
                            "score_normalization": "per_example_mean"},
            "evaluation": {"eval_examples": eval_n, "eval_batch_size": eval_bs},
            # Pool REGIME + row WINDOW the attribution and these baselines read. v1 rows
            # (unmasked pools, every consumer on [0:n]) are not comparable to these, and an
            # unstamped artifact is indistinguishable from one — see the regeneration ledger.
            "pairs_regime": all_pairs.extra["regime"],
            "pairs_window": all_pairs.extra["window"],
            "pairs_consumer": all_pairs.extra["consumer"],
            "graph": graph_info,
            "n_edges_universe": live_edges,          # method-independent denominator
            "edges_after_prune": n_edges,
            "nodes_after_prune": n_nodes,
            "heads": heads_n,
            "n_heads": len(heads_n),
            "mlps": mlps_in,                       # m{l} MLP nodes (attn_only=False)
            "n_mlps": len(mlps_in),
            "surviving_edges": surviving,
        }
        if probe_art is None:
            payload.update({
                "metric": "mean(correct-incorrect) logit over answer positions "
                          f"{list(answer_positions)}",
                "clean_baseline_logit_diff": clean_baseline,
                "corrupted_baseline_logit_diff": corrupt_baseline,
                "circuit_logit_diff": circ,
                "faithfulness_ratio": ratio,
            })
        else:
            payload.update({
                "metric": f"probe-readout KL at {probe_art['hook']} over "
                          f"positions {probe_art['positions'].tolist()} "
                          "(attribution); eval = probe argmax agreement",
                "faithfulness_metric": "probe-readout agreement ratio "
                                       "(circuit vs full vs corrupted)",
                "probe_agreement_clean_baseline": clean_baseline,
                "probe_agreement_corrupted_baseline": corrupt_baseline,
                "probe_agreement_circuit": circ,
                "faithfulness_ratio": ratio,
            })
        if zero_ablation:
            payload["zero_ablation"] = True
        if use_pos_embed:
            payload["use_pos_embed"] = True    # input source = positional embedding
        if n in strict_by_n:                   # --target-seq-acc winner: record the strict metric
            payload.update(strict_by_n[n])
        cname = (f"eap_probe_{probe}_top{n}" if probe_art is not None
                 else f"eap_top{n}") + cand_suffix
        meta = artifact_meta(model_name, method=f"eap[{attr_method}]",
                             n_examples=attr_n + eval_n, seed=seed,
                             scheme=prov["scheme"],
                             source_dataset_path=data_path)
        if probe_art is not None:
            meta["probe"] = probe_meta(probe_art)
        summary = write_candidate_folder(
            model_name, circuit, cname, smoke=smoke, meta=meta,
            nodes=nodes_in, edges=edges_in, summary=payload, graph=graph)
        results[n] = summary
        perf = (f"circuit_ld={circ:+.4f} faithfulness={ratio:6.1%}"
                if probe_art is None else
                f"probe_agreement={circ:.4f} R={ratio:6.1%}")
        print(f"  top-{n:>4}: edges={n_edges:>3} nodes={n_nodes:>2} "
              f"heads={len(heads_n):>2} {perf} -> "
              f"{os.path.dirname(summary['files']['summary_json'])}")

    print("DONE.")
    return results


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="EAP-IG circuit discovery (pipeline). "
                    "Attribution params attr_bs=50/attr_n=200/ig_steps=30 are "
                    "pinned for throughput and not CLI-exposed.")
    p.add_argument("--model", required=True,
                   help="model name (a DSL algorithm_id)")
    p.add_argument("--circuit", required=True,
                   help="circuit slug of the model to discover; reads its "
                        "inputs.json and writes under <model>/<circuit>/")
    ladder = p.add_mutually_exclusive_group()
    ladder.add_argument("--topn-regime", choices=("frac", "abs"), dest="topn_regime",
                        default=CANON_TOPN_REGIME,
                        help="how canon resolves the top-n ladder (default: %(default)s). "
                             "'frac' scales CANON_TOPN_FRACS by the model's LIVE edge "
                             "budget; 'abs' uses the fixed CANON_TOPNS and only reproduces "
                             "pre-frac artifacts")
    ladder.add_argument("--topns", type=int, nargs="+", default=None,
                        help="explicit top-n rungs, overriding --topn-regime. Rungs above "
                             "the model's live edge budget are a hard error (at 2L/4H, "
                             "top-800 IS the whole unpruned graph)")
    p.add_argument("--smoke", action="store_true",
                   help="smoke mode: outputs quarantined under .smoke/run3/, "
                        "pairs built in memory, eval set shrunk (NEVER the "
                        "pinned attribution params)")
    p.add_argument("--probe", default=None, metavar="VARIABLE",
                   help="probe-metric mode: score edges by probe-readout KL at "
                        "this variable's site (probes_data/<model>/<VARIABLE>/"
                        "site_L*.pt); candidates write under "
                        "eap_probe_<VARIABLE>_top<n>/")
    p.add_argument("--probe-layer", type=int, default=None,
                   help="probe site layer override (default: the site_L*.pt "
                        "with the highest stored accuracy)")
    p.add_argument("--probes-root", default=None,
                   help="probe artifacts root (default: $PROBES_DATA_ROOT, "
                        "else <pipeline>/probes_data; point at a fixture dir "
                        "for testing)")
    p.add_argument("--probe-readout", choices=("pooled", "position"), default="pooled",
                   help="probe the KL metric reads: the pooled probe W,b (default, the "
                        "finder contract) or each probed position's own per-position probe "
                        "(the probes whose accuracy picks the site)")
    p.add_argument("--zero-ablation", action="store_true",
                   help="C4: zero-ablate instead of resample-patching (uses "
                        "EAP-IG-activations attribution); folders gain a _zero suffix")
    p.add_argument("--use-pos-embed", action="store_true",
                   help="C4: score positional-embedding edges — the input source "
                        "becomes hook_pos_embed with the corrupted positions "
                        "scrambled; folders gain a _pos suffix")
    p.add_argument("--device", default="auto",
                   help="compute device: auto (default) / cuda / cpu")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--target-seq-acc", type=float, default=None, metavar="T",
                   help="smallest-faithful search mode: instead of the ladder sweep, reuse "
                        "ONE attribution to scan a geometric rung set for the top-n "
                        "MAXIMIZING (1-heads/H)(1-edges/E) whose edge-level per-position "
                        "mean-ablation WHOLE-SEQUENCE accuracy is strictly > T; emits only "
                        "that eap_top<n>")
    p.add_argument("--search-n", type=int, default=None,
                   help="eval-set size for the --target-seq-acc probes (default: the smallest "
                        "stage's SMALLEST_EVAL_N, for exact metric parity)")
    return p.parse_args(argv)


if __name__ == "__main__":
    a = parse_args()
    run_eap(a.model, a.topns, a.circuit, smoke=a.smoke, device=a.device,
            seed=a.seed, topn_regime=a.topn_regime,
            probe=a.probe, probe_layer=a.probe_layer,
            probes_root=a.probes_root, zero_ablation=a.zero_ablation,
            use_pos_embed=a.use_pos_embed,
            target_seq_acc=a.target_seq_acc, search_n=a.search_n,
            probe_readout=a.probe_readout)
