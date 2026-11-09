"""ACDC circuit discovery for the pipeline.

Runs the vendored ACDC (github_repos/Automatic-Circuit-Discovery) for ONE
model+circuit, one candidate folder per threshold tau:
circuits_data/<model>/<circuit>/circuits/acdc_thresh_<tau>/{summary.json,
circuit.json, circuit.pt, circuit.gv} + the PNG under circuits_figures/.
circuit.json/.pt use the normalized EAP vocabulary (write_candidate_folder /
save_circuit_pt from find_circuits/eap.py), so every method emits uniform,
Graph.from_pt-loadable artifacts. --smoke quarantines outputs under
.smoke/run3/ and builds the pairs in memory.

PERFORMANCE RULE: ACDC caches stay ON THE GPU — its default CPU-resident
caches are a measured ~120x per-step slowdown; never expose cache-to-CPU
options.

--probe <variable> swaps the metric for probe-readout KL at the variable's
site (read via a PERMANENT capture hook that survives the vendored
reset_hooks() calls); candidates write under acdc_probe_<variable>_thresh_<tau>.
--zero-ablation / --use-pos-embed thread the C4 positional options into the
vendored TLACDCExperiment; their candidate folders gain _zero / _pos suffixes
so canonical artifacts are never clobbered. See docs/reference/design.md "Probe-metric
discovery".

Must NOT be imported under the name 'acdc' (collides with the vendored
package); run as a script or load via importlib under a unique name.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import os
import sys
import time
import traceback
from typing import Dict, List, Optional, Tuple

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))        # .../find_circuits
PIPELINE_ROOT = os.path.dirname(_THIS_DIR)                    # .../pipeline
REPO_ROOT = os.path.dirname(PIPELINE_ROOT)                    # .../circuits
ACDC_REPO = os.path.join(REPO_ROOT, "github_repos", "Automatic-Circuit-Discovery")

if __name__ == "acdc":  # pragma: no cover - import-hygiene guard
    raise ImportError(
        "pipeline/find_circuits/acdc.py must not be imported as "
        "'acdc' — that name belongs to the vendored ACDC package "
        "(github_repos/Automatic-Circuit-Discovery/acdc). Run it as a script "
        "or load it via importlib under a unique module name.")

# ACDC_REPO + models dir go BEFORE the script dir so `import acdc.*` resolves
# to the vendored package, never to this file.
for _p in (os.path.join(PIPELINE_ROOT, "models"), ACDC_REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from tqdm import tqdm  # noqa: E402

import model_registry  # noqa: E402  (bootstraps interp_tools/data_gen/metrics onto sys.path)
from model_registry import resolve_device, wait_for_headroom  # noqa: E402
from canon import CANON_TAUS, CANON_ACDC_EXAMPLES, DEFAULT_SEED  # noqa: E402

GENERATED_BY = "pipeline/find_circuits/acdc.py"


# The shared helper hub is find_circuits/eap.py, loaded under a collision-proof
# module name (importing it as "eap" would shadow the vendored EAP-IG package).
def _load_fc_eap():
    name = "pipeline_find_circuits_eap"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            name, os.path.join(_THIS_DIR, "eap.py"))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return sys.modules[name]


_fc_eap = _load_fc_eap()
circuit_dirs = _fc_eap.circuit_dirs
set_checkpoint_env = _fc_eap.set_checkpoint_env
artifact_meta = _fc_eap.artifact_meta
load_run_pairs = _fc_eap.load_run_pairs
write_candidate_folder = _fc_eap.write_candidate_folder
load_probe_artifact = _fc_eap.load_probe_artifact
probe_logprobs = _fc_eap.probe_logprobs
probe_meta = _fc_eap.probe_meta
probe_capture_hook = _fc_eap.probe_capture_hook
check_probe_layer = _fc_eap.check_probe_layer


def _tau_str(tau: float) -> str:
    """Tidy threshold string for folders/labels, e.g. 0.055 -> '0.055'."""
    return "%g" % tau


def _circuit_name(tau: float, probe: Optional[str] = None,
                  zero_ablation: bool = False,
                  use_pos_embed: bool = False) -> str:
    base = (f"acdc_probe_{probe}_thresh_{_tau_str(tau)}" if probe
            else f"acdc_thresh_{_tau_str(tau)}")
    return base + ("_zero" if zero_ablation else "") + ("_pos" if use_pos_embed else "")


def _probe_kl(ref: torch.Tensor, cur: torch.Tensor) -> torch.Tensor:
    """Scalar KL(ref || cur), both [B, P, C] log-probs."""
    return (ref.exp() * (ref - cur)).sum(-1).mean()


def _make_probe_metric(model, ds: torch.Tensor, probe: Dict):
    """ACDC metric closure: probe-readout KL(clean || current) at the probe
    site, read from a PERMANENT capture hook (TL reset_hooks() keeps permanent
    hooks). TLACDCExperiment asserts ZERO fwd hooks at construction and only
    calls the metric afterwards, so the capture self-installs on the FIRST
    metric call (inside __init__) and re-runs the forward it missed."""
    site = probe_capture_hook(probe)      # == the probe-site residual value
    with torch.no_grad():
        _, cache = model.run_with_cache(ds, names_filter=site)
        ref = probe_logprobs(cache[site], probe)
    store = {"resid": None}

    def _capture(act, hook):
        store["resid"] = act

    def metric(logits: torch.Tensor) -> torch.Tensor:
        if store["resid"] is None:
            model.add_perma_hook(site, _capture)
            with torch.no_grad():
                model(ds)
        return _probe_kl(ref, probe_logprobs(store["resid"], probe))

    return metric


def _probe_direct_edges(model, things, probe: Dict, threshold: float,
                        zero_ablation: bool) -> List[Dict]:
    """Score direct writer->probe-site edges. The vendored graph has no
    receiver node at a mid-stream resid_post, so writers at layers <= the
    probe layer reach the probe only via this sweep: patch each writer's
    direct contribution out of the clean residual at the site (corrupted
    activations, or zeros under --zero-ablation) and keep the edge iff the
    probe-KL jump clears tau — ACDC's own acceptance rule."""
    l, site, H = int(probe["layer"]), probe["hook"], int(model.cfg.n_heads)
    senders = [("blocks.0.hook_resid_pre", 0, None, "embed", "embed")]
    for k in range(l + 1):
        senders += [(f"blocks.{k}.attn.hook_result", k, h, "result", f"a{k}.{h}")
                    for h in range(H)]
        if not model.cfg.attn_only:
            senders.append((f"blocks.{k}.hook_mlp_out", k, None, "mlp_out", f"m{k}"))
    names = {s[0] for s in senders} | {site}
    edges: List[Dict] = []
    with torch.no_grad():
        _, clean = model.run_with_cache(things.validation_data,
                                        names_filter=lambda n: n in names)
        corr = ({n: torch.zeros_like(clean[n]) for n in names} if zero_ablation
                else model.run_with_cache(things.validation_patch_data,
                                          names_filter=lambda n: n in names)[1])
        ref = probe_logprobs(clean[site], probe)
        for name, layer, head, kind, label in senders:
            delta = corr[name] - clean[name]
            if head is not None:
                delta = delta[:, :, head]
            effect = float(_probe_kl(ref, probe_logprobs(clean[site] + delta,
                                                         probe)))
            if effect < threshold:
                continue
            edges.append({
                "receiver": site, "receiver_index": [None],
                "receiver_label": "probe_site", "receiver_kind": "resid_post",
                "receiver_layer": l, "receiver_head": None,
                "sender": name,
                "sender_index": [None, None, head] if head is not None else [None],
                "sender_label": label, "sender_kind": kind,
                "sender_layer": layer, "sender_head": head,
                "edge_type": "ADDITION", "effect_size": effect,
                "direct_to_probe": True,
            })
    return edges


# cmapy-free patch for ACDC graphics: the vendored acdc.acdc_graphics imports
# cmapy (not installed) at module top; the stub keeps the ACDC core importable.
def _cmapy_free_color(colorscheme: str = "") -> str:
    rgb = np.random.randint(140, 256, size=3)
    return "#{:02x}{:02x}{:02x}".format(int(rgb[0]), int(rgb[1]), int(rgb[2]))


def _install_acdc_graphics_patches() -> bool:
    """Make ACDC's graphics layer importable and quiet. Returns True when the
    REAL pygraphviz-backed module loaded, False when the no-op stub was
    installed (either way the ACDC core runs; our circuit.png comes from
    draw_circuit_png, not from acdc_graphics.show)."""
    import types

    graphics_real = True
    try:
        import acdc.acdc_graphics as graphics  # noqa: F401
    except ImportError:
        graphics_real = False
        graphics = types.ModuleType("acdc.acdc_graphics")

        def _noop(*a, **k):
            return None

        graphics.show = _noop
        graphics.log_metrics_to_wandb = _noop
        graphics.generate_random_color = _cmapy_free_color
        graphics.__getattr__ = lambda name: _noop
        sys.modules["acdc.acdc_graphics"] = graphics
        import acdc as _acdc_pkg
        _acdc_pkg.acdc_graphics = graphics

    import acdc.TLACDCExperiment as tlexp

    graphics.generate_random_color = _cmapy_free_color
    # Disable the per-step PNG dump inside TLACDCExperiment.step() (slow; writes
    # ims/img_new_*.png into cwd). The final render uses the FROZEN drawer.
    tlexp.show = lambda *args, **kwargs: None
    return graphics_real


# --------------------------------------------------------------------------- #
# Node-name parsing -> (layer, head, channel)                                  #
# --------------------------------------------------------------------------- #
def _layer_of(name: str) -> Optional[int]:
    # "blocks.3.attn.hook_result" -> 3
    parts = name.split(".")
    if len(parts) >= 2 and parts[0] == "blocks":
        try:
            return int(parts[1])
        except ValueError:
            return None
    return None


def parse_node(name: str, index_tuple: Tuple) -> Dict:
    """Structured description of an ACDC graph node.

    keys: kind ('result'|'q'|'k'|'v'|'q_input'|...|'embed'|'resid_post'|'other'),
          layer, head (or None), label (short readable string).
    """
    head = None
    if isinstance(index_tuple, tuple) and len(index_tuple) >= 3 and index_tuple[2] is not None:
        head = int(index_tuple[2])
    layer = _layer_of(name)

    if name.endswith("hook_resid_post"):
        return {"kind": "resid_post", "layer": layer, "head": None, "label": "resid_post"}
    if name.endswith("hook_resid_pre"):
        return {"kind": "embed", "layer": layer, "head": None, "label": "embed"}
    if name.endswith("hook_embed"):
        return {"kind": "embed", "layer": layer, "head": None, "label": "tok_embed"}
    if name.endswith("hook_pos_embed"):
        return {"kind": "embed", "layer": layer, "head": None, "label": "pos_embed"}
    # MLP nodes (attn_only=False models): hook_mlp_out writes into the residual
    # (a sender), hook_mlp_in reads from it (a receiver); both map to node m{layer}.
    if name.endswith("hook_mlp_out"):
        return {"kind": "mlp_out", "layer": layer, "head": None, "label": f"m{layer}"}
    if name.endswith("hook_mlp_in"):
        return {"kind": "mlp_in", "layer": layer, "head": None, "label": f"m{layer}"}
    if name.endswith("attn.hook_result"):
        return {"kind": "result", "layer": layer, "head": head,
                "label": f"a{layer}.{head}"}
    for ch in ("q", "k", "v"):
        if name.endswith(f"attn.hook_{ch}"):
            return {"kind": ch, "layer": layer, "head": head,
                    "label": f"a{layer}.{head}_{ch}"}
        if name.endswith(f"hook_{ch}_input"):
            return {"kind": f"{ch}_input", "layer": layer, "head": head,
                    "label": f"a{layer}.{head}_{ch}in"}
    return {"kind": "other", "layer": layer, "head": head, "label": name}


def _head_str(layer: int, head: int) -> str:
    return f"L{layer}H{head}"


# --------------------------------------------------------------------------- #
# ACDC result -> FROZEN drawer form                                            #
# --------------------------------------------------------------------------- #
def acdc_to_drawer_graph(parsed_edges: List[Dict]) -> Tuple[List[str], list]:
    """Convert ACDC surviving edges into the EAP-IG node/edge vocabulary.

    The split-qkv ACDC graph has three edge classes: ADDITION edges
    (hook_result / embed -> hook_{q,k,v}_input / hook_resid_post) carry the
    circuit structure; DIRECT_COMPUTATION (hook_{q,k,v}_input -> hook_{q,k,v})
    and PLACEHOLDER (hook_{q,k,v} -> hook_result) are intra-head plumbing.
    We map structural edges to EAP names — sender embed -> "input", sender
    head result -> "a{l}.h{h}", receiver {q,k,v}(_input) -> the head with that
    qkv channel, receiver resid_post -> "logits" — and drop intra-head
    self-edges. Scores are the ACDC effect sizes.

    parsed_edges entries need: sender_kind/sender_layer/sender_head,
    receiver_kind/receiver_layer/receiver_head, effect_size.
    Returns (nodes, edges) for ``draw_circuit_png``: nodes ordered input ->
    heads by (layer, head) -> logits; edges [(parent, child, qkv, score)].
    """
    def node_of(kind, layer, head):
        if kind == "embed":
            return "input"
        if kind == "resid_post":
            return "logits"
        if kind in ("mlp_out", "mlp_in") and layer is not None:
            return f"m{layer}"                          # MLP node (attn_only=False)
        if head is not None and layer is not None:
            return f"a{layer}.h{head}"
        return None

    def qkv_of(kind):
        if kind in ("q", "k", "v"):
            return kind
        if kind in ("q_input", "k_input", "v_input"):
            return kind[0]
        return None                                     # MLP/resid edges have no qkv

    edges, internal = [], set()
    seen = set()
    for e in parsed_edges:
        parent = node_of(e["sender_kind"], e["sender_layer"], e["sender_head"])
        child = node_of(e["receiver_kind"], e["receiver_layer"], e["receiver_head"])
        if parent is None or child is None or parent == child:
            continue  # intra-head / intra-MLP plumbing / unmappable
        qkv = qkv_of(e["receiver_kind"])
        key = (parent, child, qkv)
        if key in seen:
            continue
        seen.add(key)
        edges.append((parent, child, qkv, float(e["effect_size"])))
        for nm in (parent, child):
            if nm not in ("input", "logits"):
                internal.add(nm)                        # heads (a{l}.h{h}) + MLPs (m{l})

    def node_key(nm):
        if nm.startswith("a"):                          # "a2.h7" -> (2, 0, 7)
            l, h = nm[1:].split(".h")
            return (int(l), 0, int(h))
        return (int(nm[1:]), 1, 0)                       # "m{l}" sorts after that layer's heads

    nodes = (["input"] if any(p == "input" for p, _c, _q, _s in edges) else [])
    nodes += sorted(internal, key=node_key)
    if any(c == "logits" for _p, c, _q, _s in edges):
        nodes.append("logits")
    return nodes, edges


# --------------------------------------------------------------------------- #
# ACDC run for a single threshold                                              #
# --------------------------------------------------------------------------- #
def run_acdc_for_threshold(
    things,
    threshold: float,
    model_name: str,
    circuit: str,
    provenance: Dict,
    full_ablation_kl: float,
    smoke: bool = False,
    seed: int = DEFAULT_SEED,
    max_steps: int = 100000,
    verbose: bool = True,
    zero_ablation: bool = False,
    use_pos_embed: bool = False,
    probe: Optional[Dict] = None,
) -> Dict:
    """Run ACDC edge-pruning at one threshold; write the candidate folder and
    return the result dict. ``probe`` is a loaded probe artifact (the caller
    must already have swapped ``things.validation_metric`` for the probe KL).
    ``full_ablation_kl`` is the metric with everything patched — the signal scale
    the ABSOLUTE threshold tau is judged against (measured once in ``main``, since
    the vendored experiment's hooks make the model unusable for a plain forward)."""
    from acdc.TLACDCExperiment import TLACDCExperiment
    from acdc.TLACDCEdge import EdgeType

    model = things.tl_model
    model.reset_hooks()

    t0 = time.time()

    # CRITICAL: keep caches ON GPU — ACDC's default CPU-resident caches (online_cache_cpu/corrupted_cache_cpu) are a measured ~120x per-step slowdown on these tiny models; never expose cache-to-CPU options.
    on_cuda = next(model.parameters()).is_cuda
    cache_cpu = not on_cuda

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):       # silence the hook-name dump
        exp = TLACDCExperiment(
            model=model,
            ds=things.validation_data,
            ref_ds=things.validation_patch_data,
            threshold=threshold,
            metric=things.validation_metric,
            second_metric=None,
            verbose=False,
            zero_ablation=zero_ablation,
            abs_value_threshold=False,
            indices_mode="reverse",
            names_mode="normal",
            online_cache_cpu=cache_cpu,
            corrupted_cache_cpu=cache_cpu,
            add_sender_hooks=True,
            add_receiver_hooks=False,
            use_pos_embed=use_pos_embed,
            show_full_index=False,
        )
    if probe is not None:
        # Instance-patch (wrapper-side): connectivity-to-LOGITS is the wrong
        # prune criterion for a mid-stream probe — the vendored check would
        # cascade-delete the probe circuit (and its metric-invariance assert
        # would fire). tau itself deletes the zero-effect downstream edges.
        exp.current_node_connected = lambda: True

    n_edges_total = exp.count_no_edges()
    init_metric = float(exp.cur_metric)
    if verbose:
        print(f"[tau={threshold}] graph built: {n_edges_total} prunable edges, "
              f"init KL={init_metric:.4g}, full-ablation KL={full_ablation_kl:.4g} "
              f"(tau is {threshold / full_ablation_kl:.1%} of it). Running steps...")
    if threshold >= full_ablation_kl:
        print(f"[tau={threshold}] WARNING: tau >= the full-ablation KL "
              f"({full_ablation_kl:.4g}) — no single edge can clear it, so expect an empty "
              f"circuit; and every accept/reject is a "
              f"{things.validation_data.shape[0]}-example estimate.", flush=True)

    steps = 0
    devnull = io.StringIO()
    _pbar = tqdm(total=max_steps, desc=f"ACDC steps [tau={threshold}]", unit="step",
                 leave=False, disable=not verbose)
    while exp.current_node is not None and steps < max_steps:
        with contextlib.redirect_stdout(devnull):   # chatty per-edge logs
            exp.step(testing=False)
        devnull.truncate(0)
        devnull.seek(0)
        steps += 1
        _pbar.update(1)
        if steps % 25 == 0:
            _pbar.set_postfix(edges=exp.count_no_edges(),
                              KL=round(float(exp.cur_metric), 4))
    _pbar.close()

    final_metric = float(exp.cur_metric)
    n_edges_final = exp.count_no_edges()
    elapsed = time.time() - t0
    if verbose:
        print(f"[tau={threshold}] DONE in {elapsed:.1f}s, {steps} steps. "
              f"edges {n_edges_total} -> {n_edges_final}, final KL={final_metric:.4g}")

    # ----- collect surviving edges -----
    surviving_edges: List[Dict] = []
    n_untested = 0                     # present, but ACDC never priced it (see below)
    head_roles: Dict[str, set] = {}    # "L3H1" -> {"result", "q", ...}
    for tupl, edge in exp.corr.all_edges().items():
        if not edge.present:
            continue
        if edge.edge_type == EdgeType.PLACEHOLDER:
            continue
        if edge.effect_size is None:
            # TLACDCExperiment sets effect_size ONLY on edges its greedy walk actually
            # tested (TLACDCExperiment.py:609), so an untested edge has no measured
            # effect: dropped rather than given a fabricated score — but never silently
            # (n_edges_untested below, reconciled against count_no_edges).
            n_untested += 1
            continue
        receiver_name, receiver_idx, sender_name, sender_idx = tupl
        r = parse_node(receiver_name, receiver_idx.hashable_tuple)
        s = parse_node(sender_name, sender_idx.hashable_tuple)
        surviving_edges.append({
            "receiver": receiver_name,
            "receiver_index": list(receiver_idx.hashable_tuple),
            "receiver_label": r["label"],
            "receiver_kind": r["kind"],
            "receiver_layer": r["layer"],
            "receiver_head": r["head"],
            "sender": sender_name,
            "sender_index": list(sender_idx.hashable_tuple),
            "sender_label": s["label"],
            "sender_kind": s["kind"],
            "sender_layer": s["layer"],
            "sender_head": s["head"],
            "edge_type": edge.edge_type.name,
            "effect_size": float(edge.effect_size),
        })
        for node in (r, s):
            if node["head"] is not None and node["layer"] is not None:
                hs = _head_str(node["layer"], node["head"])
                head_roles.setdefault(hs, set()).add(node["kind"])

    n_scored = len(surviving_edges)
    assert n_edges_final == n_scored + n_untested, (
        f"edge accounting broke: count_no_edges()={n_edges_final} but {n_scored} scored "
        f"+ {n_untested} untested")

    direct_edges: List[Dict] = []
    if probe is not None:
        model.reset_hooks()      # drop ACDC's hooks for the sweep (perma capture survives, harmless)
        direct_edges = _probe_direct_edges(model, things, probe, threshold,
                                           zero_ablation)
        surviving_edges += direct_edges
        for e in direct_edges:
            if e["sender_head"] is not None:
                head_roles.setdefault(
                    _head_str(e["sender_layer"], e["sender_head"]),
                    set()).add("result")
        if verbose:
            print(f"[tau={threshold}] direct-to-probe sweep: "
                  f"{len(direct_edges)} writer->site edges >= tau")

    surviving_edges.sort(key=lambda e: -abs(e["effect_size"]))

    # Normalize + PRUNE dangling heads/MLPs (EAP-IG Graph.prune semantics): ACDC's greedy edge removal leaves no-incoming heads (pure-mean, data-independent, inflate n_heads). All artifacts below use this pruned graph.
    nodes, drawer_edges = acdc_to_drawer_graph(surviving_edges)
    nodes, drawer_edges = _fc_eap.prune_dangling(nodes, drawer_edges)
    heads_sorted, _mlp_layers = _fc_eap.heads_mlps_from_nodes(nodes)
    mlps_in_circuit = [f"m{l}" for l in _mlp_layers]
    _kept_heads = set(heads_sorted)
    _kept_mlps = set(_mlp_layers)

    def _edge_survives_prune(e):
        for kind in ("sender", "receiver"):
            l, h, k = e[f"{kind}_layer"], e[f"{kind}_head"], e[f"{kind}_kind"]
            if h is not None and l is not None and (l, h) not in _kept_heads:
                return False
            if k in ("mlp_out", "mlp_in") and l not in _kept_mlps:
                return False
        return True
    n_pre_prune = len(surviving_edges)
    surviving_edges = [e for e in surviving_edges if _edge_survives_prune(e)]
    n_pruned_dangling = n_pre_prune - len(surviving_edges)

    head_summary = {_head_str(l, h): sorted(head_roles[_head_str(l, h)])
                    for (l, h) in heads_sorted}

    meta = artifact_meta(
        model_name, method="acdc", generated_by=GENERATED_BY,
        n_examples=int(things.validation_data.shape[0]), seed=seed,
        scheme=provenance["scheme"], source_dataset_path=provenance["path"])
    if probe is not None:
        meta["probe"] = probe_meta(probe)

    payload = {
        "threshold": threshold,
        "metric": (f"probe-readout KL(clean||patched) at {probe['hook']} over "
                   f"positions {probe['positions'].tolist()}" if probe else
                   "kl_divergence(clean||patched) over the answer positions"),
        "corruption_scheme": provenance["scheme"],
        "num_examples": int(things.validation_data.shape[0]),
        # method-INDEPENDENT denominator (canon.live_edge_budget), so a "fraction of the
        # graph" score means the same thing here, in eap and in edge_pruning
        "n_edges_universe": model_registry.live_edges(
            model_registry.get_spec(model_name)),
        "n_prunable_edges_initial": int(n_edges_total),   # ACDC's own split-qkv graph size
        "n_edges_surviving": len(surviving_edges),
        "n_edges_count_no_edges": int(n_edges_final),
        "n_edges_untested": n_untested,
        "n_edges_pruned_dangling": n_pruned_dangling,
        "init_kl": init_metric,
        "final_kl": final_metric,
        "kl_full_ablation": full_ablation_kl,
        "tau_over_kl_full_ablation": threshold / full_ablation_kl,
        "steps": steps,
        "elapsed_sec": round(elapsed, 2),
        "heads_in_circuit": [_head_str(l, h) for (l, h) in heads_sorted],
        "n_heads": len(heads_sorted),
        "mlps_in_circuit": mlps_in_circuit,
        "n_mlps": len(mlps_in_circuit),
        "head_roles": head_summary,
        "surviving_edges": surviving_edges,
        "data": provenance,
    }
    # flags recorded only when they deviate from canonical (keeps canonical summaries byte-identical)
    if zero_ablation:
        payload["zero_ablation"] = True
    if use_pos_embed:
        payload["use_pos_embed"] = True
    if probe is not None:
        payload["n_direct_probe_edges"] = len(direct_edges)

    result = write_candidate_folder(
        model_name, circuit,
        _circuit_name(threshold,
                      probe=probe["variable"] if probe else None,
                      zero_ablation=zero_ablation, use_pos_embed=use_pos_embed),
        smoke=smoke, meta=meta, nodes=nodes, edges=drawer_edges,
        model_or_cfg=model, summary=payload)

    model.reset_hooks()                       # free hooks for the next tau
    return result


# --------------------------------------------------------------------------- #
# Per-threshold runner                                                         #
# --------------------------------------------------------------------------- #
def _build_and_run_threshold(tau: float, model_name: str, circuit: str,
                             num_examples: int, device: str, seed: int,
                             smoke: bool, full_ablation_kl: float,
                             verbose: bool = True,
                             probe: Optional[Dict] = None,
                             zero_ablation: bool = False,
                             use_pos_embed: bool = False) -> None:
    """Load data, build the ACDC bundle and run ACDC for ONE threshold. Each
    tau writes its own candidate folder, so thresholds are fully independent."""
    set_checkpoint_env(model_name)              # before the vendored stack
    _install_acdc_graphics_patches()
    from acdc.segrev3 import get_all_segrev3_things

    pairs, provenance = load_run_pairs(model_name, circuit, num_examples, seed,
                                       smoke, device)
    things = get_all_segrev3_things(
        num_examples=num_examples, device=device, seed=seed, pairs=pairs)
    if probe is not None:
        things.validation_metric = _make_probe_metric(
            things.tl_model, things.validation_data, probe)
    if verbose:
        src = provenance["path"] or "in-memory pairs"
        print(f"[tau={tau}] data: {src}", flush=True)
    run_acdc_for_threshold(things, threshold=tau, model_name=model_name,
                           circuit=circuit, provenance=provenance,
                           full_ablation_kl=full_ablation_kl,
                           smoke=smoke, seed=seed, verbose=verbose,
                           zero_ablation=zero_ablation,
                           use_pos_embed=use_pos_embed, probe=probe)


# --------------------------------------------------------------------------- #
# main                                                                         #
# --------------------------------------------------------------------------- #
def main(model_name: str, circuit: str, taus: Optional[List[float]] = None,
         num_examples: int = CANON_ACDC_EXAMPLES,
         device: Optional[str] = None, seed: int = DEFAULT_SEED,
         smoke: bool = False, probe: Optional[str] = None,
         probe_layer: Optional[int] = None, probes_root: Optional[str] = None,
         zero_ablation: bool = False, use_pos_embed: bool = False) -> List[Dict]:
    if taus is None:
        taus = list(CANON_TAUS)
    if probe is not None and use_pos_embed:
        raise ValueError("--probe with --use-pos-embed is not supported: the "
                         "direct-to-probe sweep has no pos-embed corruption "
                         "reference")
    device = resolve_device(device)
    ckpt = set_checkpoint_env(model_name)
    probe_art = (load_probe_artifact(model_name, probe, layer=probe_layer,
                                     probes_root=probes_root)
                 if probe is not None else None)

    wait_for_headroom(context=f"acdc.py ({model_name})")
    print(f"ACDC | model={model_name} circuit={circuit} device={device} "
          f"taus={taus} examples={num_examples} smoke={smoke}")
    print(f"  checkpoint = {ckpt}")
    if zero_ablation or use_pos_embed:
        print(f"  C4 flags: zero_ablation={zero_ablation} "
              f"use_pos_embed={use_pos_embed}")
    if probe_art is not None:
        print(f"  probe: {probe} layer={int(probe_art['layer'])} "
              f"hook={probe_art['hook']} accuracy={float(probe_art['accuracy']):.4f}\n"
              f"         artifact = {probe_art['path']}")

    graphics_real = _install_acdc_graphics_patches()
    if not graphics_real:
        print("  [note] acdc_graphics stubbed (cmapy/pygraphviz import failed) — "
              "harmless: circuit PNGs come from the FROZEN drawer.")
    from acdc.segrev3 import get_all_segrev3_things

    # Sanity (once): the metric is ~0 on clean data (no patching applied yet).
    pairs0, _prov0 = load_run_pairs(model_name, circuit, num_examples, seed,
                                    smoke, device)
    things0 = get_all_segrev3_things(num_examples=num_examples, device=device,
                                     seed=seed, pairs=pairs0)
    if probe_art is not None:
        # (the final layer is doubly out: it is also ACDC's intervened output node)
        check_probe_layer(probe_art, int(things0.tl_model.cfg.n_layers))
        things0.validation_metric = _make_probe_metric(
            things0.tl_model, things0.validation_data, probe_art)
    with torch.no_grad():
        clean_metric = float(things0.validation_metric(
            things0.tl_model(things0.validation_data)))
        # tau is an ABSOLUTE KL threshold, so record the signal scale it is judged
        # against: the metric with everything patched. Measured on the throwaway
        # things0 — once the vendored experiment installs its sender/receiver hooks the
        # model can no longer be run on a different input.
        full_ablation_kl = float(things0.validation_metric(
            things0.tl_model(things0.validation_patch_data)))
    where = (f"at {probe_art['hook']}" if probe_art is not None
             else "over the answer positions")
    print(f"sanity: KL(clean||clean) {where} = {clean_metric:.3e} (expect ~0)")
    if full_ablation_kl <= 0.0:
        raise ValueError(
            f"full-ablation metric is {full_ablation_kl:.3e}: clean and patched runs are "
            f"indistinguishable, so no edge carries evidence and every tau accepts or "
            f"rejects noise (degenerate constant-output scope?)")
    print(f"signal scale: KL(clean||fully-patched) = {full_ablation_kl:.4g}; taus as a "
          f"fraction of it: {[round(t / full_ablation_kl, 4) for t in taus]} "
          f"(each decision is a {num_examples}-example estimate)")
    del things0, pairs0

    # Aggregate-failures semantics: run EVERY tau, collect failures, raise once.
    failed: List[float] = []
    for tau in taus:
        try:
            _build_and_run_threshold(tau, model_name, circuit, num_examples,
                                     device, seed, smoke, full_ablation_kl,
                                     verbose=True, probe=probe_art,
                                     zero_ablation=zero_ablation,
                                     use_pos_embed=use_pos_embed)
        except Exception as e:
            traceback.print_exc()
            print(f"[tau={tau}] FAILED: {type(e).__name__}: {e}", flush=True)
            failed.append(tau)

    # Re-read the per-tau summaries for the return value.
    results: List[Dict] = []
    for tau in taus:
        ddir, _fdir = circuit_dirs(
            model_name,
            _circuit_name(tau, probe=probe, zero_ablation=zero_ablation,
                          use_pos_embed=use_pos_embed),
            circuit=circuit, smoke=smoke)
        sp = os.path.join(ddir, "summary.json")
        if os.path.exists(sp):
            with open(sp) as f:
                results.append(json.load(f))

    print("\n=== ACDC per-threshold summary ===")
    for r in sorted(results, key=lambda r: -float(r["threshold"])):
        print(f"  tau={r['threshold']:<6} heads={r['n_heads']:>2} "
              f"edges={r['n_edges_surviving']:>3}/{r['n_edges_universe']} "
              f"untested={r['n_edges_untested']:>3} "
              f"dangling={r['n_edges_pruned_dangling']:>3} "
              f"final_kl={r['final_kl']:.4g} steps={r['steps']}")
    if failed:
        # exit nonzero so orchestrators see the failure instead of a silent
        # partial run (the completed taus' folders remain valid).
        raise SystemExit(f"ACDC failed for {len(failed)} threshold(s): {failed}")
    return results


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ACDC circuit discovery (pipeline). Caches stay "
                    "on the GPU (no cache-to-CPU options — measured 120x slower).")
    p.add_argument("--model", required=True,
                   help="model name (a DSL algorithm_id)")
    p.add_argument("--circuit", required=True,
                   help="circuit slug of the model to discover; reads its "
                        "inputs.json and writes under <model>/<circuit>/")
    p.add_argument("--taus", type=float, nargs="+", default=list(CANON_TAUS),
                   help="thresholds to run ACDC at, one candidate folder per "
                        "tau (default: %(default)s)")
    p.add_argument("--examples", type=int, default=CANON_ACDC_EXAMPLES,
                   help="contrast examples (default: %(default)s, owned by canon; ACDC "
                        "runtime scales with it and EVERY greedy accept/reject is an "
                        "estimate on this many rows)")
    p.add_argument("--smoke", action="store_true",
                   help="smoke mode: outputs quarantined under .smoke/run3/, "
                        "pairs built in memory")
    p.add_argument("--probe", default=None, metavar="VARIABLE",
                   help="probe-metric mode: prune by probe-readout KL at this "
                        "variable's site (probes_data/<model>/<VARIABLE>/"
                        "site_L*.pt); candidates write under "
                        "acdc_probe_<VARIABLE>_thresh_<tau>")
    p.add_argument("--probe-layer", type=int, default=None,
                   help="probe site layer override (default: the site_L*.pt "
                        "with the highest stored accuracy)")
    p.add_argument("--probes-root", default=None,
                   help="probe artifacts root (default: $PROBES_DATA_ROOT, "
                        "else <pipeline>/probes_data; point at a fixture dir "
                        "for testing)")
    p.add_argument("--zero-ablation", action="store_true",
                   help="C4: zero-ablate instead of resample-patching; "
                        "candidate folders gain a _zero suffix")
    p.add_argument("--use-pos-embed", action="store_true",
                   help="C4: split tok/pos embeddings into separate ACDC "
                        "senders (pos_embed gets scrambled, not resampled); "
                        "candidate folders gain a _pos suffix")
    p.add_argument("--device", default="auto",
                   help="compute device: auto (default) / cuda / cpu")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return p.parse_args(argv)


if __name__ == "__main__":
    a = parse_args()
    main(a.model, a.circuit, taus=a.taus, num_examples=a.examples,
         device=a.device, seed=a.seed, smoke=a.smoke, probe=a.probe,
         probe_layer=a.probe_layer, probes_root=a.probes_root,
         zero_ablation=a.zero_ablation, use_pos_embed=a.use_pos_embed)
