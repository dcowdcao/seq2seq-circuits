"""
circuit_summary.py
=================================================================================
Synthesize the manually-found ("rough") segrev3 circuit from the manual-interp
analysis results, and render a layered circuit diagram: a heuristic head set plus
per-head function hypotheses (docs/segrev3_case_study.md §5.3), grounded in the numbers
from visualizations/manual_circuit_analysis/{vis_patching_dla,vis_edge}.py
(figs_data/data/<dataset>/patching_dla_results.json [canonical: random] +
figs_data/data/edge_results.json).

Outputs (under visualizations/):
  rough_circuit.json   -- selected heads, roles, edges, the selection thresholds
  rough_circuit.png    -- layered diagram (layer on y, heads as nodes; edges =
                          Q/K/V composition colored by channel)
  rough_circuit.md     -- short narrative mapping heads -> function

Head selection (principled, from the data):
  * movers           : top heads by |DLA| (direct logit writers)
  * detectors        : top heads by activation-patching necessity that are NOT
                       movers (the necessity signal that replaced whole-head knockout)
  * the union is the circuit head-set; edges are the strongest Q/K/V composition
    and path-patching edges among circuit heads.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_loading import load_segrev3                       # noqa: E402
import n_threes as N3                                        # noqa: E402

VIZ = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "visualizations")
# The manual-interp results now live in the combined manual_circuit_analysis area,
# produced by visualizations/manual_circuit_analysis/{vis_patching_dla,vis_edge}.py.
DATA = os.path.join(VIZ, "manual_circuit_analysis", "figs_data", "data")
# vis_patching_dla writes its (data-dependent) results into a PER-DATASET subfolder.
# The rough circuit is synthesized from a single canonical contrast dataset -- the
# global "random" contrast, matching the old phase-2 default. edge_results.json
# (weight-based, input-independent) stays flat.
CANON_DATASET = "random"   # label = datasets/manual/random.json stem (was "random_example")


def _load(p):
    with open(p) as f:
        return json.load(f)


def _hl(label):
    l = int(label[1:label.index("H")]); h = int(label[label.index("H") + 1:])
    return l, h


def _top_edges_from_matrix(mat, k=6, n_heads=8):
    """Top-k |effect| sender->receiver edges from a 32x32 (sender x receiver) path
    matrix (nested list). Returns [{sender,receiver,effect}] with signed effect."""
    flat = []
    for s in range(len(mat)):
        for r in range(len(mat[s])):
            v = mat[s][r]
            if v != 0:
                flat.append((abs(v), s, r, v))
    flat.sort(reverse=True)
    return [{"sender": f"L{s // n_heads}H{s % n_heads}",
             "receiver": f"L{r // n_heads}H{r % n_heads}", "effect": v}
            for _, s, r, v in flat[:k]]


def build_circuit(dla_top_k=6, detector_top_k=6, comp_edges=10, dataset=CANON_DATASET):
    # Manual-interp results from the combined manual_circuit_analysis area.
    p2 = _load(os.path.join(DATA, dataset, "patching_dla_results.json"))  # vis_patching_dla.py (per-dataset)
    p3 = _load(os.path.join(DATA, "edge_results.json"))                   # vis_edge.py (weight-based, flat)

    movers = [d["head"] for d in p2["dla_top"][:dla_top_k]]
    # Detectors: top heads by ACTIVATION-PATCHING importance (necessity) that are
    # not movers. (This replaces the old whole-head-knockout signal, which the new
    # vis_patching_dla pipeline no longer computes; activation patching noising is
    # the same necessity measure.)
    detect = [d["head"] for d in p2["activation_top"][:detector_top_k]]
    detectors = [h for h in detect if h not in movers]
    # intermediate heads that appear as composition targets/sources but aren't
    # already movers/detectors
    circuit = list(dict.fromkeys(movers + detectors))

    # edges: strongest Q/K/V composition among (sender,receiver), plus path edges.
    # composition_top8 is keyed by uppercase channel (Q/K/V); path edges are derived
    # per-channel from the 32x32 path_matrices.
    edges = []
    for ch in ("q", "k", "v"):
        for e in p3["composition_top8"][ch.upper()][:comp_edges]:
            edges.append({"sender": e["sender"], "receiver": e["receiver"],
                          "channel": ch, "score": e["score"], "kind": "composition"})
    for ch in ("q", "k", "v"):
        for e in _top_edges_from_matrix(p2["path_matrices"][ch], k=6):
            edges.append({"sender": e["sender"], "receiver": e["receiver"],
                          "channel": ch, "score": e["effect"], "kind": "path"})

    # add composition endpoints to the circuit head-set
    for e in edges:
        for k in ("sender", "receiver"):
            if e[k] not in circuit and e["kind"] == "composition" and abs(e["score"]) > 0.15:
                circuit.append(e[k])

    roles = {}
    for h in circuit:
        l, _ = _hl(h)
        if h in movers:
            roles[h] = "mover / copy head (writes the answer digit to the logits; high DLA)"
        elif l == 0:
            roles[h] = "L0 detector (boundary/position; high activation-patching necessity, feeds composition)"
        else:
            roles[h] = "intermediate composition head (reflect-index / routing)"

    return {
        "selection": {"dla_top_k": dla_top_k, "detector_top_k": detector_top_k},
        "movers": movers, "detectors": detectors,
        "circuit_heads": circuit, "roles": roles,
        "edges": edges,
        "copy_scores": {h: p3["copy_scores"][h] for h in circuit},
        "notes": [
            "The model is CAUSAL: output position 9+k attends only to positions <=9+k. "
            "The reversal mirror is the CAUSAL reflection diagonal (dest 9+k -> reflected "
            "input src 9-k). The positional QK weight circuit shows a weak reflection "
            "signal concentrated in the L0 heads (L0H7/H1/H2 ~0.02-0.03); the STRONG "
            "realized reflection (L0H7 attends ~0.9 from each output position to its "
            "reflected input position on no-3 inputs) is emergent from positional + "
            "content + composition + softmax, and is dynamically modulated by the "
            "3-boundaries (on inputs with a 3, L0H7 instead attends to the 3).",
            "L0 heads have high OV copy scores (~0.9): they propagate token identity. "
            "L0H4 is the dominant Q-composition hub into the movers; L0H7 the V hub into L1.",
        ],
    }


def draw(circuit, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    model_layers = 4
    heads = circuit["circuit_heads"]
    # position nodes: x spread within a layer, y = layer
    by_layer = {}
    for h in heads:
        l, hd = _hl(h)
        by_layer.setdefault(l, []).append(h)
    pos = {}
    for l, hs in by_layer.items():
        hs_sorted = sorted(hs, key=lambda x: _hl(x)[1])
        for i, h in enumerate(hs_sorted):
            x = (i - (len(hs_sorted) - 1) / 2) * 1.6
            pos[h] = (x, l)

    fig, ax = plt.subplots(figsize=(11, 7))
    ccolor = {"q": "#c0392b", "k": "#27ae60", "v": "#2980b9"}
    # edges
    for e in circuit["edges"]:
        s, r = e["sender"], e["receiver"]
        if s in pos and r in pos:
            x0, y0 = pos[s]; x1, y1 = pos[r]
            lw = 0.6 + 6 * abs(e["score"]) if e["kind"] == "composition" else 0.5 + 30 * abs(e["score"])
            ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                        arrowprops=dict(arrowstyle="-|>", color=ccolor[e["channel"]],
                                        lw=min(lw, 5), alpha=0.55))
    # nodes
    for h, (x, y) in pos.items():
        is_mover = h in circuit["movers"]
        color = "#f1c40f" if is_mover else ("#e67e22" if _hl(h)[0] == 0 else "#9b59b6")
        ax.scatter([x], [y], s=1400, c=color, edgecolors="k", zorder=3)
        ax.text(x, y, h, ha="center", va="center", fontsize=8, zorder=4, weight="bold")
    ax.set_yticks(range(model_layers)); ax.set_yticklabels([f"Layer {l}" for l in range(model_layers)])
    ax.set_ylim(-0.6, model_layers - 0.4); ax.invert_yaxis()
    ax.set_xticks([])
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    legend = [Line2D([0], [0], color=ccolor[c], lw=3, label=f"{c.upper()}-composition") for c in "qkv"]
    legend += [Patch(facecolor="#f1c40f", edgecolor="k", label="mover/copy (high DLA)"),
               Patch(facecolor="#e67e22", edgecolor="k", label="L0 detector (high necessity)"),
               Patch(facecolor="#9b59b6", edgecolor="k", label="intermediate")]
    ax.legend(handles=legend, loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8)
    ax.set_title("segrev3 heuristic manual circuit (rough)\nedges: top Q/K/V composition + path patching", fontsize=11)
    fig.tight_layout(); N3.annotate(fig, N3.STRUCTURAL)
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)


def write_md(circuit, path):
    sel = circuit["selection"]
    lines = ["# segrev3 — heuristic manual circuit (rough)\n",
             "Synthesized from `visualizations/manual_circuit_analysis/figs_data/data/"
             f"{CANON_DATASET}/patching_dla_results.json` + `.../edge_results.json`. "
             f"Heuristic selection: movers = top-{sel['dla_top_k']} heads by |DLA|, "
             f"detectors = top-{sel['detector_top_k']} by activation-patching necessity, "
             "plus endpoints of strong Q/K/V composition edges.\n",
             "## Circuit heads & hypothesized roles\n"]
    for h in circuit["circuit_heads"]:
        cs = circuit["copy_scores"].get(h)
        lines.append(f"- **{h}** — {circuit['roles'][h]} (OV copy score {cs:+.2f})")
    lines.append("\n## Mechanism (hypothesis)\n")
    lines.append(
        "Input digits → **L0 detector / reflection heads** (causal: each output "
        "position 9+k attends back to its reflected input position 9-k; on inputs "
        "with a 3 they attend to the boundary instead — highest necessity, high OV "
        "copy) → **composition** (L0H4 routes the *where* into the movers' queries; "
        "L0H7 routes value/identity into L1) → **reflect-index computed from the "
        "input + the autoregressive output prefix** (a weak positional reflection "
        "backbone in L0, modulated dynamically by the 3-boundaries) → **L2/L3 mover "
        "heads** copy the digit at the reflected index to the logits (highest DLA).")
    lines.append("\n## Key empirical notes\n")
    for n in circuit["notes"]:
        lines.append(f"- {n}")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    circuit = build_circuit()
    with open(os.path.join(VIZ, "rough_circuit.json"), "w") as f:
        json.dump(circuit, f, indent=2)
    draw(circuit, os.path.join(VIZ, "rough_circuit.png"))
    write_md(circuit, os.path.join(VIZ, "rough_circuit.md"))
    print("circuit heads:", circuit["circuit_heads"])
    print("movers:", circuit["movers"])
    print("detectors:", circuit["detectors"])
    print(f"edges: {len(circuit['edges'])}")
    print("wrote rough_circuit.{json,png,md} to", VIZ)


if __name__ == "__main__":
    main()
