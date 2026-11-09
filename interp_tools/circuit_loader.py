"""
circuit_loader.py
=================================================================================
Shared foundation for circuit-scoped visualization of a DISCOVERED circuit
(ACDC / EAP / manual) on segrev3 (4-layer, 8-head, attention-only). It loads a
discovered subgraph (heads + edges) into one uniform ``Circuit`` object and
provides the two helpers the per-analysis modules (vis_patching_dla, vis_edge,
the driver) share:

  * ``draw_circuit_metric`` -- render the circuit as a layered head graph with
    node face-colors from a per-head metric (+ optional per-edge coloring), the
    same layered layout as circuit_summary.draw().
  * ``ablate_complement``   -- a context manager that mean/zero-ablates every head
    NOT in a keep-set for the duration of a ``with`` block (the faithfulness
    "circuit-only" intervention), via permanent hooks on each layer's hook_z.

Circuit sources (read from ``<repo_root>/visualizations/``):

  acdc   -> acdc/<name>/summary.json
            heads from ``heads_in_circuit`` ("L0H0" labels); edges built head->head
            from ``surviving_edges`` (sender_label/receiver_label in the "a0.7_v"
            dialect), channel from the receiver's _q/_k/_v (..in) suffix; non-head
            endpoints (embed/resid/logits) dropped; roles from ``head_roles``.
  eap    -> eap/<name>/<name>.json   (cfg/nodes/edges, each carrying ``in_graph``);
            heads = in_graph head nodes ("a0.h0" dialect), edges = in_graph
            head->head edges ("a0.h7<v>->a3.h1<q>"). Falls back to
            eap/eap_all_topn_summary.json (circuit_heads_top_export_n + top30_edges)
            if the per-topn file is absent.
  manual -> rough_circuit.json (circuit_heads / edges / roles); may not exist.

Head labels are normalized via parse_head_label, which accepts every dialect
seen on disk ("L0H7", "a0.7", "a0.h7", "a0.7_v", "a0.h7<v>") and returns None for
non-head nodes ("embed", "input", "resid_post", "logits", "m0"..).
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import sys
from dataclasses import dataclass, field

# Same sys.path bootstrap as circuit_summary.py so ``from model_loading import ...``
# works whether this module is run directly or imported by a consumer.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS)            # repo root = two dirs up from interp_tools/


# --------------------------------------------------------------------------- #
# Circuit abstraction                                                         #
# --------------------------------------------------------------------------- #


@dataclass
class Circuit:
    name: str                 # e.g. "acdc_thresh_0.055", "eap_top200", "rough"
    app: str                  # "acdc" | "eap" | "manual"
    heads: list               # sorted list of (layer:int, head:int) tuples
    edges: list               # [{"sender":"L0H4","receiver":"L3H1","channel":"q"|"k"|"v"|None}]
    roles: dict = field(default_factory=dict)   # {"L3H1":"mover"/..} best-effort; {} ok


# --------------------------------------------------------------------------- #
# Label parsing                                                               #
# --------------------------------------------------------------------------- #

# Robust head matcher (cf. convergence_summary.py): handles "L0H7", "a0.7",
# "a0.h7", "a0.7_v", "a0.h7<v>", "a0.7_vin", etc. -- a leading optional 'a'/'L',
# the layer digits, one or more separators ('.', 'h', 'H'), then the head digits.
_HEAD_RE = re.compile(r"^[aL]?(\d+)[.hH]+(\d+)")
_CHANNEL_RE = re.compile(r"[_<]?([qkv])(?:in)?[>_]?$")
# tokens that are explicitly NOT heads
_NON_HEAD = {"embed", "input", "logits", "output"}


def parse_head_label(s):
    """Normalize ANY head dialect to (layer, head); return None for non-heads.

    Accepts "L0H7", "a0.7", "a0.h7", "a0.7_v", "a0.h7<v>", "a0.7_vin", ... and
    returns None for non-head nodes ("embed", "input", "resid_post", "logits",
    "m0".."m3", etc.).
    """
    if s is None:
        return None
    s = str(s).strip()
    if not s:
        return None
    low = s.lower()
    if low in _NON_HEAD:
        return None
    # resid_*, hook_*, blocks.*.hook_resid_pre, m0/m1 (MLP) -> not a head
    if low.startswith(("resid", "hook", "m", "blocks")) and not low.startswith("blocks.0.attn"):
        # "m0".."m3" and resid/hook tokens are non-heads
        if not _HEAD_RE.match(low):
            return None
    m = _HEAD_RE.match(s)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def _channel_of(label):
    """Extract the q/k/v channel from a receiver label suffix (_q/_k/_v/_qin..,
    <q>/<k>/<v>), else None."""
    if label is None:
        return None
    m = _CHANNEL_RE.search(str(label).strip().lower())
    return m.group(1) if m else None


def _hlabel(lh):
    """(layer, head) -> 'L{l}H{h}'."""
    return f"L{lh[0]}H{lh[1]}"


def head_tuples(circuit: "Circuit") -> list:
    """Sorted [(l, h), ...] for the circuit's heads."""
    return sorted(circuit.heads)


def head_labels(circuit: "Circuit") -> list:
    """['L0H7', ...] for the circuit's heads (sorted)."""
    return [_hlabel(lh) for lh in sorted(circuit.heads)]


def _dedup_edges(edges):
    """De-dup edges by (sender, receiver, channel), preserving order."""
    seen = set()
    out = []
    for e in edges:
        key = (e["sender"], e["receiver"], e["channel"])
        if key not in seen:
            seen.add(key)
            out.append(e)
    return out


# --------------------------------------------------------------------------- #
# Per-source loaders                                                           #
# --------------------------------------------------------------------------- #


def _load_json(path):
    with open(path) as f:
        return json.load(f)


def _circuit_from_acdc_summary(d, name):
    """Build a Circuit from an ACDC summary dict (heads_in_circuit/surviving_edges)."""
    heads = set()
    for lab in d.get("heads_in_circuit", []):
        lh = parse_head_label(lab)
        if lh is not None:
            heads.add(lh)

    edges = []
    for e in d.get("surviving_edges", []):
        s = parse_head_label(e.get("sender_label"))
        r = parse_head_label(e.get("receiver_label"))
        if s is None or r is None:        # drop edges touching embed/resid/logits
            continue
        ch = _channel_of(e.get("receiver_label"))
        edges.append({"sender": _hlabel(s), "receiver": _hlabel(r), "channel": ch})
    edges = _dedup_edges(edges)

    # roles: keep the raw head_roles channel list (best-effort), keyed by label.
    roles = {}
    for lab, role in d.get("head_roles", {}).items():
        lh = parse_head_label(lab)
        if lh is not None:
            roles[_hlabel(lh)] = role

    return Circuit(name=name, app="acdc", heads=sorted(heads), edges=edges, roles=roles)


def _load_acdc(name, viz):
    path = os.path.join(viz, "acdc", name, "summary.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"ACDC circuit summary not found: {path}")
    return _circuit_from_acdc_summary(_load_json(path), name)


def _circuit_from_eap_graph(d, name):
    """Build a Circuit from an EAP graph dict (cfg/nodes/edges carrying in_graph)."""
    nodes = d.get("nodes", {})
    edges_raw = d.get("edges", {})

    def _in(v):
        # node/edge values are dicts with "in_graph"; be defensive if they're bare.
        return v.get("in_graph", True) if isinstance(v, dict) else bool(v)

    heads = set()
    for node, v in nodes.items():
        if not _in(v):
            continue
        lh = parse_head_label(node)
        if lh is not None:
            heads.add(lh)

    edges = []
    for estr, v in edges_raw.items():
        if not _in(v):
            continue
        e = _parse_eap_edge_str(estr)
        if e is not None:
            edges.append(e)
    edges = _dedup_edges(edges)

    return Circuit(name=name, app="eap", heads=sorted(heads), edges=edges, roles={})


def _load_eap_pertopn(name, viz):
    """Per-topn eap/<name>/<name>.json with cfg/nodes/edges carrying in_graph."""
    path = os.path.join(viz, "eap", name, f"{name}.json")
    if not os.path.exists(path):
        return None
    return _circuit_from_eap_graph(_load_json(path), name)


def _parse_eap_edge_str(estr):
    """'a0.h7<v>->a3.h1<q>' / 'input->a0.h7<v>' -> head->head edge dict or None."""
    if "->" not in str(estr):
        return None
    src, dst = str(estr).split("->", 1)
    s = parse_head_label(src)
    r = parse_head_label(dst)
    if s is None or r is None:
        return None
    return {"sender": _hlabel(s), "receiver": _hlabel(r), "channel": _channel_of(dst)}


def _circuit_from_eap_summary(d, name):
    """Build a Circuit from an EAP all-topn summary dict
    (circuit_heads_top_export_n + top30_edges)."""
    heads = set()
    for lab in d.get("circuit_heads_top_export_n", []):
        lh = parse_head_label(lab)
        if lh is not None:
            heads.add(lh)

    edges = []
    for e in d.get("top30_edges", []):
        estr = e.get("edge") if isinstance(e, dict) else e
        parsed = _parse_eap_edge_str(estr)
        if parsed is not None:
            edges.append(parsed)
    edges = _dedup_edges(edges)

    return Circuit(name=name, app="eap", heads=sorted(heads), edges=edges, roles={})


def _load_eap_summary(name, viz):
    """Fallback: eap/eap_all_topn_summary.json (circuit_heads_top_export_n +
    top30_edges)."""
    path = os.path.join(viz, "eap", "eap_all_topn_summary.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"EAP circuit not found for {name!r}: neither "
            f"{os.path.join(viz, 'eap', name, name + '.json')} nor {path} exists")
    return _circuit_from_eap_summary(_load_json(path), name)


def _load_eap(name, viz):
    c = _load_eap_pertopn(name, viz)
    if c is not None:
        return c
    return _load_eap_summary(name, viz)


def _circuit_from_manual(d, name):
    """Build a Circuit from a manual rough-circuit dict (circuit_heads/edges/roles)."""
    heads = set()
    for lab in d.get("circuit_heads", []):
        lh = parse_head_label(lab)
        if lh is not None:
            heads.add(lh)

    edges = []
    for e in d.get("edges", []):
        s = parse_head_label(e.get("sender"))
        r = parse_head_label(e.get("receiver"))
        if s is None or r is None:
            continue
        edges.append({"sender": _hlabel(s), "receiver": _hlabel(r),
                      "channel": e.get("channel")})
    edges = _dedup_edges(edges)

    roles = dict(d.get("roles", {}))
    for h in d.get("movers", []):
        roles.setdefault(h, "mover")
    for h in d.get("detectors", []):
        roles.setdefault(h, "detector")

    return Circuit(name=name or "rough", app="manual", heads=sorted(heads),
                   edges=edges, roles=roles)


def _load_manual(name, viz):
    path = os.path.join(viz, "rough_circuit.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"manual circuit not found: {path} (run circuit_summary.py to build it)")
    return _circuit_from_manual(_load_json(path), name)


def load_circuit_from_file(path: str, name: str = None) -> "Circuit":
    """Load a circuit straight from a graph/summary JSON file, sniffing the
    dialect: an ACDC ``summary.json`` (heads_in_circuit / surviving_edges), an
    EAP graph json (nodes/edges carrying ``in_graph``), an EAP all-topn summary
    (circuit_heads_top_export_n), or a manual ``rough_circuit.json``
    (circuit_heads). ``name`` defaults to the file's stem. This is how a circuit
    produced by ACDC / EAP-IG (or built by hand) is fed to the manual analyses
    without it living under ``visualizations/``."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"circuit file not found: {path}")
    d = _load_json(path)
    if name is None:
        name = os.path.splitext(os.path.basename(path))[0]
        if name == "summary":          # ACDC circuits live at <name>/summary.json
            parent = os.path.basename(os.path.dirname(os.path.abspath(path)))
            name = parent or name
    if "heads_in_circuit" in d or "surviving_edges" in d:
        return _circuit_from_acdc_summary(d, name)
    if isinstance(d.get("nodes"), dict) and "edges" in d:
        return _circuit_from_eap_graph(d, name)
    if "circuit_heads_top_export_n" in d:
        return _circuit_from_eap_summary(d, name)
    if "circuit_heads" in d:
        return _circuit_from_manual(d, name)
    raise ValueError(
        f"unrecognized circuit file format: {path} (expected an ACDC summary.json, "
        "an EAP graph json, an EAP all-topn summary, or a manual rough_circuit.json)")


def load_circuit(app: str, name: str, repo_root=None) -> "Circuit":
    """Load a discovered circuit (acdc/eap/manual) into a uniform ``Circuit``.

    Reads from ``<repo_root>/visualizations/`` (repo_root defaults to two dirs up
    from this file).
    """
    root = repo_root or _ROOT
    viz = os.path.join(root, "visualizations")
    app = app.lower()
    if app == "acdc":
        return _load_acdc(name, viz)
    if app == "eap":
        return _load_eap(name, viz)
    if app == "manual":
        return _load_manual(name, viz)
    raise ValueError(f"unknown app {app!r} (expected 'acdc' | 'eap' | 'manual')")


# --------------------------------------------------------------------------- #
# Complement ablation (context manager)                                       #
# --------------------------------------------------------------------------- #


def _mean_z(model, n=2048, seed=0):
    """Dataset-mean z per layer -> {l: [seq, head, d_head]} (cf. validation._mean_z;
    reimplemented here to avoid coupling to the validation package)."""
    import torch
    from data_gen import segrev3_batch

    toks, _, _ = segrev3_batch(n, seed=seed)
    names = [f"blocks.{l}.attn.hook_z" for l in range(model.cfg.n_layers)]
    with torch.no_grad():
        _, cache = model.run_with_cache(toks, names_filter=names)
    return {l: cache[f"blocks.{l}.attn.hook_z"].mean(0)
            for l in range(model.cfg.n_layers)}


@contextlib.contextmanager
def ablate_complement(model, keep_heads, mode: str = "mean", n: int = 2048, seed: int = 0):
    """Mean- (or zero-) ablate every head NOT in ``keep_heads`` for the duration of
    the ``with`` block, via permanent hooks on each layer's hook_z; yields model.

    ``keep_heads`` is any iterable of (layer, head) tuples. On exit the hooks are
    removed with ``model.reset_hooks()``.
    """
    keep = set(tuple(h) for h in keep_heads)
    mean_cache = _mean_z(model, n=n, seed=seed) if mode == "mean" else None

    # heads to ablate per layer = complement of keep within that layer
    ablate_by_layer = {}
    for l in range(model.cfg.n_layers):
        ablate_by_layer[l] = [h for h in range(model.cfg.n_heads) if (l, h) not in keep]

    try:
        for l in range(model.cfg.n_layers):
            hs = ablate_by_layer[l]
            if not hs:
                continue

            def hook(act, hook, l=l, hs=hs):
                for h in hs:
                    if mode == "zero":
                        act[:, :, h] = 0.0
                    else:                                  # mean
                        act[:, :, h] = mean_cache[l][:, h]
                return act

            model.add_hook(f"blocks.{l}.attn.hook_z", hook)
        yield model
    finally:
        model.reset_hooks()


# --------------------------------------------------------------------------- #
# Circuit-graph drawing                                                        #
# --------------------------------------------------------------------------- #


def _normalize_head_value_keys(values):
    """Accept dict keyed by (l, h) tuples or 'L{l}H{h}' strings -> {(l,h): v}."""
    out = {}
    for k, v in (values or {}).items():
        if isinstance(k, str):
            lh = parse_head_label(k)
        else:
            lh = tuple(k)
        if lh is not None:
            out[lh] = v
    return out


def _normalize_edge_value_keys(values):
    """Accept dict keyed by ('L..','L..') or 'L..->L..' -> {('L..','L..'): v}."""
    out = {}
    for k, v in (values or {}).items():
        if isinstance(k, str) and "->" in k:
            a, b = k.split("->", 1)
            out[(a.strip(), b.strip())] = v
        elif isinstance(k, (tuple, list)) and len(k) == 2:
            out[(str(k[0]), str(k[1]))] = v
    return out


def draw_circuit_metric(circuit: "Circuit", head_values: dict, title: str, path: str,
                        edge_values=None, cmap: str = "RdBu_r",
                        diverging: bool = True, value_fmt: str = "{:+.2f}",
                        subtitle: str = None) -> None:
    """Render the circuit as a layered graph (y = layer 0..3 top->bottom, heads
    spread on x), with node face-color from ``head_values`` and optional edges
    colored by ``edge_values``; save to ``path``.

    ``head_values`` keys may be (layer, head) tuples or 'L{l}H{h}' strings.
    ``edge_values`` keys may be ('L..','L..') tuples or 'L..->L..' strings.
    When ``diverging`` the node colormap is symmetric about 0 (vmax = max|value|);
    otherwise a magma scale from 0.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.cm as mcm
    from matplotlib.colors import Normalize

    def _get_cmap(n):
        # non-deprecated access (matplotlib>=3.7), with a fallback.
        try:
            return matplotlib.colormaps[n]
        except Exception:
            return mcm.get_cmap(n)

    hvals = _normalize_head_value_keys(head_values)
    evals = _normalize_edge_value_keys(edge_values) if edge_values is not None else None

    n_layers = 4
    heads = sorted(circuit.heads)

    # layered layout: x spread within a layer, y = layer (cf. circuit_summary.draw).
    by_layer = {}
    for (l, h) in heads:
        by_layer.setdefault(l, []).append((l, h))
    pos = {}
    for l, hs in by_layer.items():
        hs_sorted = sorted(hs, key=lambda x: x[1])
        for i, lh in enumerate(hs_sorted):
            x = (i - (len(hs_sorted) - 1) / 2) * 1.6
            pos[lh] = (x, l)

    # node colormap normalization
    vlist = [v for v in hvals.values() if v is not None]
    if diverging:
        vmax = max((abs(v) for v in vlist), default=1.0) or 1.0
        norm = Normalize(vmin=-vmax, vmax=vmax)
        node_cmap = _get_cmap(cmap)
    else:
        vmax = max(vlist, default=1.0) or 1.0
        norm = Normalize(vmin=0.0, vmax=vmax)
        node_cmap = _get_cmap("magma")

    fig, ax = plt.subplots(figsize=(11, 7))

    # edges first (under the nodes)
    e_norm = None
    if evals:
        emax = max((abs(v) for v in evals.values()), default=1.0) or 1.0
        e_norm = Normalize(vmin=-emax, vmax=emax)
        edge_cmap = _get_cmap("RdBu_r")
    for e in circuit.edges:
        s, r = e["sender"], e["receiver"]
        slh, rlh = parse_head_label(s), parse_head_label(r)
        if slh not in pos or rlh not in pos:
            continue
        x0, y0 = pos[slh]
        x1, y1 = pos[rlh]
        if evals is not None and (s, r) in evals:
            val = evals[(s, r)]
            color = edge_cmap(e_norm(val))
            lw = 0.6 + 5 * abs(val) / (max((abs(v) for v in evals.values()), default=1.0) or 1.0)
            alpha = 0.85
        else:
            color = "#888888"
            lw = 0.8
            alpha = 0.5
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color=color,
                                    lw=min(lw, 5), alpha=alpha))

    # nodes
    for lh, (x, y) in pos.items():
        v = hvals.get(lh)
        color = node_cmap(norm(v)) if v is not None else "#cccccc"
        ax.scatter([x], [y], s=1500, c=[color], edgecolors="k", zorder=3)
        label = _hlabel(lh)
        txt = label if v is None else f"{label}\n{value_fmt.format(v)}"
        ax.text(x, y, txt, ha="center", va="center", fontsize=7.5, zorder=4,
                weight="bold")

    ax.set_yticks(range(n_layers))
    ax.set_yticklabels([f"Layer {l}" for l in range(n_layers)])
    ax.set_ylim(-0.6, n_layers - 0.4)
    ax.invert_yaxis()
    ax.set_xticks([])

    sm = mcm.ScalarMappable(norm=norm, cmap=node_cmap)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04, label="head value")

    ax.set_title(title, fontsize=11)
    fig.tight_layout()
    # Mark the number of 3s (segrev3 boundary digit) in the inputs behind this
    # figure -- a batch 3-count caption for data-dependent metrics, or an
    # input-independent sentinel for weight-based wiring (see n_threes.py).
    if subtitle:
        fig.text(0.995, 0.005, subtitle, ha="right", va="bottom", fontsize=7,
                 color="#444444", style="italic")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Smoke test                                                                  #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    sys.path.insert(0, os.path.join(_ROOT, "data_gen"))
    sys.path.insert(0, os.path.join(_ROOT, "metrics"))

    print("== parse_head_label ==")
    for s in ["L0H7", "a0.7", "a0.h7", "a0.7_v", "a0.h7<v>", "a0.7_vin",
              "embed", "input", "resid_post", "logits", "m0"]:
        print(f"  {s!r:14} -> {parse_head_label(s)}")

    print("\n== load acdc ==")
    c = load_circuit("acdc", "acdc_thresh_0.02")
    print("heads:", len(c.heads), c.heads[:4])
    print("edges:", len(c.edges), c.edges[:2])

    print("\n== load eap ==")
    e = load_circuit("eap", "eap_top200")
    print("heads:", len(e.heads), e.heads[:4])
    print("edges:", len(e.edges), e.edges[:2])

    print("\n== draw ==")
    hv = {h: (h[0] - 1.5) for h in c.heads}
    draw_circuit_metric(c, hv, "circuit_loader smoke test", "/tmp/circ_test.png")
    print("drew png:", os.path.getsize("/tmp/circ_test.png") > 0)
