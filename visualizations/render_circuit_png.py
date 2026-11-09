#!/usr/bin/env python
"""Render a saved EAP-IG circuit (.pt) to the .png that Graph.to_image() would
have produced, WITHOUT pygraphviz. Emits the equivalent DOT, then renders it
with the graphviz ``dot`` binary (install graphviz so ``dot`` is on PATH).

Replicates github_repos/EAP-IG/src/eap/graph.py:to_image exactly (the defaults
eap.py uses: colorscheme=Pastel2, seed=0, penwidth 0.6..5.0): same node
iteration order for the seeded colors, same |score| min/max normalization over
the FULL scores tensor, same qkv edge colors, multi-edges preserved.

Usage:
    python visualizations/render_circuit_png.py visualizations/eap/eap_top75/eap_top75.pt
    # -> writes eap_top75.png next to the .pt
"""
import argparse
import os
import shutil
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "github_repos", "EAP-IG", "src"))

import numpy as np  # noqa: E402

from eap.graph import Graph  # noqa: E402
from eap.visualization import generate_random_color, get_color  # noqa: E402


def to_dot(g: Graph, colorscheme="Pastel2", minimum_penwidth=0.6,
           maximum_penwidth=5.0, seed=0) -> str:
    if seed is not None:
        np.random.seed(seed)
    colors = {n.name: generate_random_color(colorscheme) for n in g.nodes.values()}

    lines = ["digraph {", '\tgraph [bgcolor=white, overlap=false, splines=true];']
    for node in g.nodes.values():
        if node.in_graph:
            lines.append(
                f'\t"{node.name}" [fillcolor="{colors[node.name]}", color=black, '
                f'style="filled, rounded", shape=box, fontname=Helvetica];')
    scores = g.scores.view(-1).abs()
    max_s, min_s = scores.max().item(), scores.min().item()
    for edge in g.edges.values():
        if edge.in_graph:
            norm = (abs(edge.score) - min_s) / (max_s - min_s) \
                if max_s != min_s else abs(edge.score)
            penwidth = max(minimum_penwidth, norm * maximum_penwidth)
            lines.append(
                f'\t"{edge.parent.name}" -> "{edge.child.name}" '
                f'[penwidth="{penwidth}", color="{get_color(edge.qkv, edge.score)}"];')
    lines.append("}")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("pt_path", help="circuit .pt file (Graph.from_pt format)")
    ap.add_argument("-o", "--out", default=None,
                    help="output .png (default: alongside the .pt)")
    a = ap.parse_args()

    g = Graph.from_pt(a.pt_path)
    dot = to_dot(g)
    out = a.out or os.path.splitext(a.pt_path)[0] + ".png"

    # dot -Tpng; stdin/stdout streaming.
    if not shutil.which("dot"):
        raise SystemExit("graphviz `dot` not found on PATH (apt-get install graphviz)")
    png = subprocess.run(["dot", "-Tpng"], input=dot.encode(), capture_output=True,
                         check=True).stdout
    if not png.startswith(b"\x89PNG"):
        raise SystemExit(f"dot did not return a PNG:\n{png[:500]!r}")
    with open(out, "wb") as f:
        f.write(png)
    n_nodes = sum(1 for n in g.nodes.values() if n.in_graph)
    n_edges = sum(1 for e in g.edges.values() if e.in_graph)
    print(f"wrote {out}  ({n_nodes} nodes, {n_edges} edges)")


if __name__ == "__main__":
    main()
