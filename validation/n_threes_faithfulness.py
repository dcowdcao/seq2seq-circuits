"""
n_threes_faithfulness.py
=================================================================================
FAITHFULNESS of the discovered ACDC / EAP-IG circuits as a function of the
NUMBER OF 3s (boundary digits) in the input, k = 0..10.

The number of 3s sets the segment structure of segrev3, so this asks: does the
circuit recover the model's behavior uniformly across structural regimes, or
only on the typical inputs (k~1) the discovery datasets are dominated by?

Data: the single (non-paired) edge datasets ``datasets/edge/n_threes_<k>.json``
(every input has EXACTLY k 3s; see data_gen/dataset_store.py). Because these
carry no clean/corrupted contrast, the paired logit-diff of validation.py does
not apply; the metric is TOKEN ACCURACY (argmax over the answer positions 9..18
vs segrev3(input)), reported for three conditions per bin:

  * full_acc    -- the unablated model (ceiling).
  * empty_acc   -- ALL 32 heads mean-ablated (floor; direct path + constants).
  * circuit_acc -- keep ONLY the circuit's heads, mean-ablate the rest.

Mean ablation uses the GLOBAL random-data mean z (validation._mean_z),
identical for every bin, so bins differ only in their inputs.

Outputs (grouped like the other validation artifacts):
  validation/figs_data/data/<method>/faithfulness_by_n_threes_<method>.json
  validation/figs_data/figs/<method>/faithfulness_by_n_threes_<method>.png
  (method = acdc | eap)

Run:  python validation/n_threes_faithfulness.py [--circuits NAME ...]
(--circuits restricts to the named circuits; default = every discovered one,
which is a dense sweep since the min-faithful-circuit search.)
"""

from __future__ import annotations

import json
import os
import sys

import torch

_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS)
sys.path.insert(0, _THIS)
import validation as V                                  # noqa: E402  (bootstraps interp_tools/data_gen/metrics)
import circuit_loader as CL                             # noqa: E402
import dataset_store as store                           # noqa: E402
import n_threes as N3                                   # noqa: E402
from model_loading import ANSWER_POSITIONS, load_segrev3  # noqa: E402

N_THREES_RANGE = list(range(0, 11))
METHODS = ("acdc", "eap")


def _acc_single(logits, outputs):
    """Token accuracy over the answer positions for SINGLE (non-paired) data."""
    pos = torch.tensor(ANSWER_POSITIONS, device=logits.device)
    return (logits[:, pos, :].argmax(-1) == outputs).float().mean().item()


def _eval_keep_many_single(model, keep_sets, tokens, outputs, mean_cache):
    """validation._eval_keep_many for SINGLE data, mean mode only: one batched
    forward evaluates every keep-set (group-major replication + per-group
    head-ablation masks on each layer's hook_z). Returns [acc, ...]."""
    G = len(keep_sets)
    nL, nH = model.cfg.n_layers, model.cfg.n_heads
    B, S = tokens.shape

    ablate_mask = torch.ones(nL, G, nH, dtype=torch.bool, device=tokens.device)
    for g, keep in enumerate(keep_sets):
        for (l, h) in keep:
            ablate_mask[l, g, h] = False

    per_chunk = max(1, V._MAX_GROUP_ROWS // max(1, B))
    out = []
    for c0 in range(0, G, per_chunk):
        c1 = min(G, c0 + per_chunk)
        gc = c1 - c0
        toks_rep = tokens.repeat(gc, 1)

        hooks = []
        for l in range(nL):
            m_l = ablate_mask[l, c0:c1]                        # [gc, nH]

            def hook(act, hook, l=l, m_l=m_l):
                dH = act.shape[-1]
                av = act.view(gc, B, S, nH, dH)
                mean = mean_cache[l].view(1, 1, S, nH, dH)
                av = torch.where(m_l.view(gc, 1, 1, nH, 1), mean, av)
                return av.reshape(gc * B, S, nH, dH)

            hooks.append((f"blocks.{l}.attn.hook_z", hook))

        with torch.no_grad():
            logits = model.run_with_hooks(toks_rep, fwd_hooks=hooks)
        logits = logits.view(gc, B, S, -1)
        for g in range(gc):
            out.append(_acc_single(logits[g], outputs))
    return out


def compute(model=None, circuits_filter=None):
    """Per-bin (full, empty, circuit) accuracy for every discovered acdc/eap
    circuit. All circuits and both baselines for one bin share a single batched
    forward pass. Returns ({method: {name: {...}}}, per-bin data provenance).

    ``circuits_filter`` (iterable of circuit folder names) restricts the run to
    those circuits; None = every discovered circuit. Since the min-faithful-
    circuit search added a dense tau/top-n sweep to visualizations/, the
    unfiltered set is large -- pass an explicit list for a readable figure."""
    if model is None:
        model = load_segrev3()
    mean_cache = V._mean_z(model)
    all_heads = [(l, h) for l in range(model.cfg.n_layers) for h in range(model.cfg.n_heads)]

    wanted = set(circuits_filter) if circuits_filter is not None else None
    circuits = {}                      # name -> (method, heads)
    for method in METHODS:
        for name in V.discovered_circuit_names(method):
            if wanted is not None and name not in wanted:
                continue
            heads = CL.head_tuples(CL.load_circuit(method, name))
            circuits[name] = (method, sorted(heads))
    if wanted is not None and wanted - set(circuits):
        raise SystemExit(f"✗ circuits not found on disk: {sorted(wanted - set(circuits))} "
                         "(expected acdc_thresh_<tau> / eap_top<n> folders under "
                         "visualizations/)")
    names = sorted(circuits)

    results = {m: {} for m in METHODS}
    data_paths = {}
    for k in N_THREES_RANGE:
        single, path = store.load_single(k)
        data_paths[k] = {"path": os.path.relpath(path, _ROOT),
                         "n_examples": int(single.tokens.shape[0])}
        toks = single.tokens.to(next(model.parameters()).device)
        outs = single.outputs.to(toks.device)

        keep_sets = [all_heads, []] + [circuits[n][1] for n in names]
        evals = _eval_keep_many_single(model, keep_sets, toks, outs, mean_cache)
        full_acc, empty_acc = evals[0], evals[1]

        for i, name in enumerate(names):
            method, heads = circuits[name]
            results[method].setdefault(name, {
                "n_heads": len(heads),
                "heads": [f"L{l}H{h}" for (l, h) in heads],
                "per_n_threes": [],
            })["per_n_threes"].append({
                "n_threes": k,
                "full_acc": full_acc,
                "empty_acc": empty_acc,
                "circuit_acc": evals[2 + i],
            })
        print(f"  k={k:2d}  full_acc={full_acc:.3f}  empty_acc={empty_acc:.3f}  "
              f"({data_paths[k]['n_examples']} examples)")

    return results, data_paths


def write_json(results, data_paths, tag=None):
    """``tag`` suffixes every output filename (``..._<tag>.json``) so a filtered
    run (e.g. --circuits eap_top87 --tag top87) does not overwrite the canonical
    all-circuits artifacts. Methods with no selected circuits are skipped."""
    suffix = f"_{tag}" if tag else ""
    paths = []
    for method in METHODS:
        if not results[method]:
            continue
        out = {
            "method": method,
            "mode": "mean",
            "definitions": {
                "full_acc": "token accuracy of the unablated model over answer "
                            "positions 9..18",
                "empty_acc": "token accuracy with ALL 32 heads mean-ablated",
                "circuit_acc": "token accuracy keeping ONLY the circuit heads "
                               "(every other head mean-ablated)",
                "ablation_reference": "global random-data mean z (validation._mean_z), "
                                      "identical across bins",
            },
            "data": {"source": "datasets/edge/n_threes_<k>.json (every input has "
                               "EXACTLY k 3s)",
                     "per_bin": data_paths},
            "circuits": results[method],
        }
        d = os.path.join(V.DATA_DIR, method)
        os.makedirs(d, exist_ok=True)
        p = os.path.join(d, f"faithfulness_by_n_threes_{method}{suffix}.json")
        with open(p, "w") as f:
            json.dump(out, f, indent=2)
        paths.append(p)
    return paths


def _draw_method(ax, method, rows):
    """Circuit-only accuracy lines for one method + the full/empty references
    (circuit-independent per bin, so drawn once)."""
    any_c = next(iter(rows.values()))["per_n_threes"]
    ks = [p["n_threes"] for p in any_c]
    ax.plot(ks, [p["full_acc"] for p in any_c], "-", color="k", lw=1.4,
            label="full model")
    ax.plot(ks, [p["empty_acc"] for p in any_c], "--", color="grey", lw=1.2,
            label="empty (all heads ablated)")
    for name in sorted(rows, key=lambda n: rows[n]["n_heads"]):
        r = rows[name]
        ax.plot([p["n_threes"] for p in r["per_n_threes"]],
                [p["circuit_acc"] for p in r["per_n_threes"]],
                "-o", ms=3.5, label=f"{name} ({r['n_heads']} heads)")
    ax.set_xticks(N_THREES_RANGE)
    ax.set_xlabel("number of 3s in the input (exact)")
    ax.set_ylabel("token accuracy (answer positions 9..18)")
    ax.set_title(f"{'ACDC' if method == 'acdc' else 'EAP-IG'}: circuit-only "
                 "accuracy vs #3s  (mean ablation; full/empty = ceiling/floor)")
    ax.legend(fontsize=7)


def plot(results, tag=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    suffix = f"_{tag}" if tag else ""
    methods = [m for m in METHODS if results[m]]   # skip a method with no circuits
    caption = "n_3s: x-axis (exact per bin; datasets/edge/n_threes_<k>.json)"
    pngs = []
    for method in methods:
        fig, ax = plt.subplots(figsize=(7.5, 4.4))
        _draw_method(ax, method, results[method])
        fig.tight_layout()
        N3.annotate(fig, caption)
        d = os.path.join(V.FIG_DIR, method)
        os.makedirs(d, exist_ok=True)
        p = os.path.join(d, f"faithfulness_by_n_threes_{method}{suffix}.png")
        fig.savefig(p, dpi=140, bbox_inches="tight")
        plt.close(fig)
        pngs.append(p)

    # Combined ACDC vs EAP-IG view; spans both methods, so it lives at the figs
    # root rather than in either method folder.
    if len(methods) > 1:
        fig, axes = plt.subplots(1, len(methods), figsize=(13, 4.4), sharey=True)
        for ax, method in zip(axes, methods):
            _draw_method(ax, method, results[method])
        fig.tight_layout()
        N3.annotate(fig, caption)
        p = os.path.join(V.FIG_DIR, f"faithfulness_by_n_threes_combined{suffix}.png")
        fig.savefig(p, dpi=140, bbox_inches="tight")
        plt.close(fig)
        pngs.append(p)
    return pngs


def main(circuit_names=None, tag=None):
    print("=== faithfulness by n_threes (acdc + eap circuits, mean ablation) ===")
    if circuit_names:
        print(f"  circuits: {sorted(circuit_names)}" + (f"  tag={tag}" if tag else ""))
    results, data_paths = compute(circuits_filter=circuit_names)
    jsons = write_json(results, data_paths, tag=tag)
    pngs = plot(results, tag=tag)
    for p in jsons + pngs:
        print(f"  wrote {os.path.relpath(p, _ROOT)}")
    return results


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Faithfulness of discovered ACDC/EAP circuits vs the number "
                    "of 3s in the input (k=0..10).")
    ap.add_argument("--circuits", nargs="+", default=None, metavar="NAME",
                    help="circuit folder names to include (acdc_thresh_<tau> / "
                         "eap_top<n>). Default: every discovered acdc/eap circuit "
                         "-- a LOT since the min-faithful-circuit search's dense "
                         "sweep; pass an explicit list for a readable figure.")
    ap.add_argument("--tag", default=None,
                    help="suffix for every output filename (e.g. --tag top87 -> "
                         "faithfulness_by_n_threes_eap_top87.png), so a filtered "
                         "run does not overwrite the canonical artifacts")
    a = ap.parse_args()
    main(circuit_names=a.circuits, tag=a.tag)
