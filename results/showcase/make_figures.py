#!/usr/bin/env python
"""Draw each showcase circuit on its model's architecture grid, plus a five-panel overview.

Reads results/showcase/<example>/circuit.json and writes <example>/figure.png and
results/showcase/overview.png. Deterministic (no randomness; fixed layout and DPI).

Layout: one row per layer (layer 0 at the bottom), attention heads left to right, then
the layer's MLP / MoE block (drawn slightly above the heads because it reads them). The
input embedding is the bar at the bottom; the logits (or, for the probe-based classes,
the probe readout at its layer) is the bar at the top. Circuit nodes and edges are drawn
in the class colour, everything else in faint gray. An edge into a head lands on its
Q, K or V input (left, middle, right of the head's lower edge); an edge that would cross
a row of nodes is routed up the gap beside its writer.

    python results/showcase/make_figures.py            # all five + overview
    python results/showcase/make_figures.py --only 01_domain_schmitt_id__moe8
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, PathPatch, Rectangle  # noqa: E402
from matplotlib.path import Path  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from showcase_spec import (EXAMPLES, FAINT_EDGE, FAINT_FILL, HERE, INK, INK_2,  # noqa: E402
                           MUTED)

ROW = 2.0          # vertical distance between layers
HEAD_W = 0.62      # head square side
MLP_W = 2.8        # MLP / MoE block width
MLP_H = 0.56       # MLP / MoE block height
MLP_DY = 0.85      # an MLP sits this far above its layer's heads
PROBE_GAP = 0.9    # extra room above the probe readout bar
LANE_DY = 1.5      # edges spanning more than this are routed along a lane
PORT_DX = {"q": -0.2, "k": 0.0, "v": 0.2, None: 0.0}
HEAD_RE = re.compile(r"^a(\d+)\.h(\d+)$")
MLP_RE = re.compile(r"^m(\d+)$")
EDGE_RE = re.compile(r"^(?P<p>[^>]+)->(?P<c>[^<]+)(?:<(?P<q>[qkv])>)?$")


def norm_edge(e) -> tuple:
    """(parent, child, qkv) from a dict, a [parent, child(<q>)] / [parent, child, qkv]
    list, or a 'parent->child<q>' string."""
    if isinstance(e, dict):
        return e["parent"], e["child"], e.get("qkv")
    if isinstance(e, str):
        m = EDGE_RE.match(e)
        return m["p"], m["c"], m["q"]
    if len(e) > 2:
        return e[0], e[1], e[2]
    m = EDGE_RE.match(f"{e[0]}->{e[1]}")          # [writer, "reader<q>"]
    return m["p"], m["c"], m["q"]


def fringe_edges(circ: dict) -> set | None:
    """The fringe (edges beyond the class's base circuit) when the evaluator reported one."""
    f = circ.get("fringe") or {}
    return {norm_edge(e) for e in f["edges"]} if f.get("available") else None


class Grid:
    """Node coordinates for one architecture (and, for probe classes, the probe layer)."""

    def __init__(self, arch: dict, probe_layer: int | None):
        self.L, self.H = int(arch["n_layers"]), int(arch["n_heads"])
        self.has_mlp = not arch.get("attn_only", False)
        self.n_experts = arch.get("num_experts") or 0
        self.top_k = arch.get("experts_per_token") or 1
        self.probe_layer = probe_layer
        self.mlp_x0 = self.H - 0.5 + 0.55
        self.x_min = -0.6
        self.x_max = self.mlp_x0 + (MLP_W if self.has_mlp else 0) + 0.1
        top = self.L - 1 if probe_layer is None else probe_layer
        self.y_term = self.y_head(top) + (MLP_DY if self.has_mlp else 0) + 0.75

    def y_head(self, l: int) -> float:
        lift = PROBE_GAP if self.probe_layer is not None and l > self.probe_layer else 0.0
        return 1.0 + l * ROW + lift

    def y_mlp(self, l: int) -> float:
        return self.y_head(l) + MLP_DY

    def x_mlp(self) -> float:
        return self.mlp_x0 + MLP_W / 2

    def anchor_x(self, name: str):
        """Horizontal position of a writer (None for the input bar, which spans the row)."""
        m = HEAD_RE.match(name)
        if m:
            return float(m[2])
        return self.x_mlp() if MLP_RE.match(name) else None

    def edge_path(self, p: str, c: str, q) -> Path:
        """Writer top -> reader input, straight Bezier or lane-routed when it would
        cross a row of nodes."""
        wx = self.anchor_x(p)
        if c == "logits":
            x1, y1 = (self.x_min + 0.3 if wx is None else wx), self.y_term - 0.22
        elif HEAD_RE.match(c):
            l, h = map(int, HEAD_RE.match(c).groups())
            x1, y1 = h + PORT_DX[q], self.y_head(l) - HEAD_W / 2
        else:
            src = self.x_mlp() if wx is None else wx
            frac = (src - self.x_min) / (self.x_max - self.x_min)   # fan in along the bottom
            x1 = self.mlp_x0 + 0.15 + frac * (MLP_W - 0.3)
            y1 = self.y_mlp(int(MLP_RE.match(c)[1])) - MLP_H / 2
        if p == "input":
            x0, y0, lane = x1, 0.22, (self.mlp_x0 - 0.25 if MLP_RE.match(c) else x1 - 0.5)
        elif HEAD_RE.match(p):
            x0, y0 = wx, self.y_head(int(HEAD_RE.match(p)[1])) + HEAD_W / 2
            lane = wx + 0.5
        else:
            x0, y0 = wx, self.y_mlp(int(MLP_RE.match(p)[1])) + MLP_H / 2
            lane = self.mlp_x0 + MLP_W + 0.15
        dy = y1 - y0
        if dy <= LANE_DY:
            return Path([(x0, y0), (x0, y0 + 0.5 * dy), (x1, y1 - 0.5 * dy), (x1, y1)],
                        [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4])
        a, b = y0 + 0.35, y1 - 0.35
        return Path([(x0, y0), (x0, y0 + 0.2), (lane, a - 0.15), (lane, a),
                     (lane, b), (lane, b + 0.15), (x1, y1 - 0.2), (x1, y1)],
                    [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4,
                     Path.LINETO, Path.CURVE4, Path.CURVE4, Path.CURVE4])

    def top(self) -> float:
        return max(self.y_term, self.y_mlp(self.L - 1) if self.has_mlp
                   else self.y_head(self.L - 1)) + 0.45


def draw_panel(ax, circ: dict, color: str, compact: bool = False) -> None:
    arch = circ["architecture"]
    probe = circ.get("probe") or None
    g = Grid(arch, int(probe["layer"]) if probe else None)
    edges = [norm_edge(e) for e in circ["edges"]]
    nodes = set(circ["nodes"]) | {p for p, _, _ in edges} | {c for _, c, _ in edges}
    fringe = fringe_edges(circ)
    fs = 6.5 if compact else 8
    lw = 0.7 if compact else 0.9

    for e in edges:                                       # edges under the nodes
        bold = fringe is None or e in fringe
        ax.add_patch(PathPatch(g.edge_path(*e), fc="none", ec=color,
                               lw=lw * (1.6 if fringe is not None and bold else 1.0),
                               alpha=0.8 if bold else 0.22, zorder=2.2 if bold else 2))

    def box(x, y, w, h, on, ctx=False, rounded=False):
        kw = dict(fc=color if on else FAINT_FILL, ec=INK if on else FAINT_EDGE, lw=0.6,
                  alpha=0.35 if ctx else 1.0, zorder=3)
        if rounded:
            ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.08",
                                        **kw))
        else:
            ax.add_patch(Rectangle((x, y), w, h, **kw))

    width = g.x_max - g.x_min
    box(g.x_min, -0.2, width, 0.4, "input" in nodes, rounded=True)
    ax.text(g.x_min + width / 2, 0.0, "input embedding (token + position)",
            ha="center", va="center", fontsize=fs, color=INK, zorder=4)
    box(g.x_min, g.y_term - 0.2, width, 0.4, "logits" in nodes, rounded=True)
    term = f"probe readout (layer {g.probe_layer} residual stream)" if probe else "logits"
    ax.text(g.x_min + width / 2, g.y_term, term, ha="center", va="center", fontsize=fs,
            color=INK, zorder=4)

    for l in range(g.L):
        ctx = probe is not None and l > g.probe_layer   # cannot influence the probe
        for h in range(g.H):
            box(h - HEAD_W / 2, g.y_head(l) - HEAD_W / 2, HEAD_W, HEAD_W,
                f"a{l}.h{h}" in nodes, ctx)
        if g.has_mlp:
            on, y = f"m{l}" in nodes, g.y_mlp(l)
            box(g.mlp_x0, y - MLP_H / 2, MLP_W, MLP_H, on, ctx)
            for k in range(1, g.n_experts):              # expert partitions of the block
                xk = g.mlp_x0 + k * MLP_W / g.n_experts
                ax.plot([xk, xk], [y - MLP_H / 2, y + MLP_H / 2], color=INK if on else FAINT_EDGE,
                        lw=0.3, alpha=0.35 if ctx else 0.8, zorder=3.5)
        ax.text(g.x_min - 0.25, g.y_head(l), f"L{l}", ha="right", va="center", fontsize=fs,
                color=MUTED if ctx else INK_2)

    for h in range(g.H):                                  # column labels under the input bar
        ax.text(h, -0.32, f"h{h}", ha="center", va="top", fontsize=fs - 1, color=MUTED)
    if g.has_mlp:
        lab = f"MoE: {g.n_experts} experts, top-{g.top_k}" if g.n_experts else "MLP"
        ax.text(g.x_mlp(), -0.32, lab, ha="center", va="top", fontsize=fs - 1, color=MUTED)
    if probe and g.probe_layer < g.L - 1:
        ax.text(g.x_min, g.top() - 0.15, "layers above the probe cannot affect it (context)",
                ha="left", va="top", fontsize=fs - 1, color=MUTED, style="italic")

    ax.set_xlim(g.x_min - 0.9, g.x_max + 0.25)
    ax.set_ylim(-0.75, g.top())
    ax.set_aspect("equal")
    ax.axis("off")
    return (g.top() + 0.75) / (g.x_max + 0.25 - g.x_min + 0.9)      # height / width


def headline(c: dict) -> str:
    return f"{c['class']}  ·  {c['model']}"


def subtitle(c: dict, compact: bool = False) -> str:
    """Circuit id, size, fringe (if any) and faithfulness; ``compact`` stacks the parts on
    short lines so they fit an overview panel."""
    verdict = "above floor" if c["passes"] else "BELOW FLOOR"
    f = c.get("fringe") or {}
    size = f"nodes {c['n_nodes']}/{c['n_nodes_full']}  ·  edges {c['n_edges']}/{c['n_edges_full']}"
    fringe = f"bold: {f['n_edges']} edges beyond {f['base_circuit']}" if f.get("available") else ""
    faith = f"faithfulness a = {c['a']:.3f}  (floor {c['floor']:.2f}, {verdict})"
    if compact:
        return "\n".join(p for p in (f"circuit {c['circuit']}", size, fringe, faith) if p)
    return (f"circuit {c['circuit']}  ·  {size}" + (f"  ·  {fringe}" if fringe else "")
            + f"\n{faith}")


def load(ex) -> dict | None:
    path = os.path.join(ex.dir, "circuit.json")
    return json.load(open(path)) if os.path.exists(path) else None


def figure_one(ex, circ: dict) -> str:
    w, head = 7.0, 0.95                                  # inches: width, title block
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    plot_h = (w - 0.1) * draw_panel(ax, circ, ex.color)
    h = plot_h + head + 0.1
    fig.set_size_inches(w, h)
    ax.set_position([0.05 / w, 0.05 / h, (w - 0.1) / w, plot_h / h])
    fig.text(0.2 / w, 1 - 0.15 / h, headline(circ), fontsize=12, color=INK, va="top")
    fig.text(0.2 / w, 1 - 0.47 / h, subtitle(circ), fontsize=9, color=INK_2, va="top")
    out = os.path.join(ex.dir, "figure.png")
    fig.savefig(out, dpi=160, facecolor="white", metadata={"Software": None})
    plt.close(fig)
    return out


def figure_overview(pairs: list) -> str:
    n, pw, head = len(pairs), 4.4, 1.15                  # inches: panel width, title block
    fig = plt.figure()
    axes = [fig.add_axes([i / n, 0, 1 / n, 1]) for i in range(n)]
    aspect = max(draw_panel(ax, circ, ex.color, compact=True)
                 for ax, (ex, circ) in zip(axes, pairs))
    plot_h = (pw - 0.1) * aspect
    h = plot_h + head + 0.1
    fig.set_size_inches(pw * n, h)
    for i, (ax, (ex, circ)) in enumerate(zip(axes, pairs)):
        x0 = i * pw
        ax.set_position([(x0 + 0.05) / (pw * n), 0.05 / h, (pw - 0.1) / (pw * n), plot_h / h])
        fig.text((x0 + 0.12) / (pw * n), 1 - 0.1 / h, headline(circ), fontsize=9.5,
                 color=INK, va="top")
        fig.text((x0 + 0.12) / (pw * n), 1 - 0.34 / h, subtitle(circ, compact=True),
                 fontsize=7.5, color=INK_2, va="top", linespacing=1.3)
    out = os.path.join(HERE, "overview.png")
    fig.savefig(out, dpi=140, facecolor="white", metadata={"Software": None})
    plt.close(fig)
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--only", nargs="*", default=None, help="example directory names")
    a = ap.parse_args(argv)
    pairs = []
    for ex in EXAMPLES:
        circ = load(ex)
        if circ is None:
            print(f"skip {ex.key}: no circuit.json yet")
            continue
        if a.only is None or ex.key in a.only:
            print("wrote", os.path.relpath(figure_one(ex, circ), HERE))
        pairs.append((ex, circ))
    if a.only is None and pairs:
        print("wrote", os.path.relpath(figure_overview(pairs), HERE))


if __name__ == "__main__":
    main()
