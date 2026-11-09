"""
validation.py
=================================================================================
Circuit-validation pipeline for segrev3 (docs/segrev3_runbook.md §6):
faithfulness, completeness, minimality. Works on a head-level circuit
(a set of (layer, head) attention heads); edges from ACDC/EAP can be reduced to
the heads they touch.

Ablation = replace a head's z (output, all positions) with a reference value:
  * "mean"     : the dataset-mean z for that head/position (default; standard
                 mean ablation -- removes input-specific info, keeps the average).
  * "zero"     : set z to 0.
  * "resample" : z from an independent random batch.

Definitions (operationalized):
  * FAITHFULNESS. Ablate every head NOT in the circuit; the circuit should retain
    the model's behavior. Recovered fraction
        R = (perf(circuit_only) - perf(empty)) / (perf(full) - perf(empty))
    reported on task token-accuracy and on the logit-difference metric. R~1 =>
    faithful.
  * COMPLETENESS. No important component lives outside the circuit. We report
    (a) perf(complement) -- keeping ONLY the non-circuit heads -- which should be
    low (the complement alone is not sufficient), and (b) the incompleteness gap
        max_v |perf(circuit) - perf(circuit + {v})|  over heads v outside C,
    which should be small (no outside head meaningfully changes the circuit).
  * MINIMALITY. No circuit head is dead weight: within the circuit-only model,
    ablating each circuit head individually should drop performance. We report
    each head's individual contribution; ~0 flags a removable head.

Outputs (grouped by circuit family <grp> = acdc/<name> | eap/<name> | manual):
  validation/figs_data/data/<grp>/<name>_validation.json   -- all numbers
  validation/figs_data/figs/<grp>/<name>_minimality.png    -- per-head contribution bar chart
  validation/figs_data/figs/<grp>/<name>_summary.png       -- faithfulness / completeness bars
  (e.g. acdc/acdc_thr0p020/..., eap/eap_top200/..., manual/rough_*, manual/faithfulness_curve.*)

DATA PROVENANCE / SIZE NUANCE
  The contrast (clean) data is no longer generated on the fly here -- it is LOADED
  from the central dataset store (data_gen/dataset_store.py) at
  ``datasets/<app>/<param>/<scheme>.json``. ``validate(circuit, ...)`` loads the
  SAME data family as the circuit it validates: an ``acdc_*`` circuit is validated
  on the acdc store dataset for its tau, an ``eap_*`` circuit on its top-n eap
  dataset, everything else on the manual dataset (see ``_circuit_data_spec``).
  Every output JSON carries a ``"data"`` block recording the store label, file
  path, scheme and example count, so each run is traceable to a file.

  CONSEQUENCE: the dataset size is whatever the stored file contains -- e.g. the
  acdc store default is small (N=20). The ``n=`` parameter is kept only for
  backward compatibility and is NO LONGER used to size the contrast set (the store
  load takes the file as-is). To validate on more examples, regenerate that folder,
  e.g.:  ``python data_gen/generate_datasets.py --apps acdc --schemes random --n 512``
  then re-run validation. (The mean/zero/resample ablation reference batches are
  separate reference statistics and are still generated independently.)
"""

from __future__ import annotations

import glob
import json
import os
import re
import sys
from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
from tqdm import tqdm

_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS)
sys.path.insert(0, os.path.join(_ROOT, "interp_tools"))
sys.path.insert(0, os.path.join(_ROOT, "data_gen"))
sys.path.insert(0, os.path.join(_ROOT, "metrics"))
from model_loading import load_segrev3, ANSWER_POSITIONS, repo_relpath         # noqa: E402
import n_threes as N3                                                          # noqa: E402
from data_gen import make_clean_corrupted                                      # noqa: E402
import dataset_store as store                                                  # noqa: E402
from segrev3_algorithm import segrev3_tensor                                   # noqa: E402
from metrics import logit_diff                                                 # noqa: E402

# Outputs live under validation/figs_data/{data (JSON), figs (PNG)}, grouped by
# circuit family: acdc/<name> and eap/<name> (one folder per tau / top-n), and
# manual/ for the rough circuit + general curves -- see _group_subdir().
FIGS_DATA = os.path.join(_THIS, "figs_data")
DATA_DIR = os.path.join(FIGS_DATA, "data")
FIG_DIR = os.path.join(FIGS_DATA, "figs")
Head = tuple


def _group_subdir(name: str) -> str:
    """Per-circuit output subfolder under data/ and figs/: ``acdc/<name>`` or
    ``eap/<name>`` (one per tau / top-n, e.g. acdc/acdc_thr0p020, eap/eap_top200),
    else ``manual`` (the rough circuit and general curves like faithfulness_curve)."""
    if name.startswith("acdc"):
        return os.path.join("acdc", name)
    if name.startswith("eap"):
        return os.path.join("eap", name)
    return "manual"


def _circuit_data_spec(name: str):
    """Map a circuit `name` to the central-store (app, kwargs) it should be
    validated on -- so validation runs on the SAME data family as the circuit.

      acdc...  -> ("acdc", {"tau": <float>})   tau parsed from the trailing number,
                  'p' read as '.', e.g. "acdc_thr0p055" -> 0.055,
                  "acdc_thresh_0.1" -> 0.1.
      eap...   -> ("eap", {"topn": <int>})     trailing integer, e.g. "eap_top200" -> 200.
      else     -> ("manual", {})               e.g. "rough".
    """
    if name.startswith("acdc"):
        import re
        m = re.search(r"[0-9pP.]+$", name)
        if not m:
            raise ValueError(f"cannot parse tau from acdc circuit name {name!r}")
        tau = float(m.group(0).replace("p", ".").replace("P", "."))
        return "acdc", {"tau": tau}
    if name.startswith("eap"):
        import re
        m = re.search(r"[0-9]+$", name)
        if not m:
            raise ValueError(f"cannot parse top-n from eap circuit name {name!r}")
        return "eap", {"topn": int(m.group(0))}
    return "manual", {}


# --------------------------------------------------------------------------- #
# Circuit abstraction                                                         #
# --------------------------------------------------------------------------- #


@dataclass
class Circuit:
    name: str
    heads: set                       # set of (layer, head)

    @staticmethod
    def from_labels(name: str, labels: Iterable[str]) -> "Circuit":
        hs = set()
        for s in labels:
            l = int(s[1:s.index("H")]); h = int(s[s.index("H") + 1:])
            hs.add((l, h))
        return Circuit(name, hs)


# --------------------------------------------------------------------------- #
# Ablation machinery                                                          #
# --------------------------------------------------------------------------- #


def _mean_z(model, n=2048, seed=0):
    """Dataset-mean z per layer -> [seq, head, d_head]."""
    toks, _, _ = __segrev3_batch(n, seed=seed)
    names = [f"blocks.{l}.attn.hook_z" for l in range(model.cfg.n_layers)]
    with torch.no_grad():
        _, cache = model.run_with_cache(toks, names_filter=names)
    return {l: cache[f"blocks.{l}.attn.hook_z"].mean(0) for l in range(model.cfg.n_layers)}


def _resample_z(model, batch, seed=123):
    rt, _, _ = __segrev3_batch(batch, seed=seed)
    names = [f"blocks.{l}.attn.hook_z" for l in range(model.cfg.n_layers)]
    with torch.no_grad():
        _, cache = model.run_with_cache(rt, names_filter=names)
    return {l: cache[f"blocks.{l}.attn.hook_z"] for l in range(model.cfg.n_layers)}


def run_with_ablation(model, tokens, ablate_heads: Sequence[Head], mode="mean",
                      mean_cache=None, resample_cache=None):
    """Forward pass with `ablate_heads` replaced by the reference value."""
    by_layer = {}
    for (l, h) in ablate_heads:
        by_layer.setdefault(l, []).append(h)
    hooks = []
    for l, hs in by_layer.items():
        name = f"blocks.{l}.attn.hook_z"
        def hook(act, hook, l=l, hs=hs):
            for h in hs:
                if mode == "zero":
                    act[:, :, h] = 0.0
                elif mode == "mean":
                    act[:, :, h] = mean_cache[l][:, h]
                elif mode == "resample":
                    act[:, :, h] = resample_cache[l][:, :, h]
            return act
        hooks.append((name, hook))
    with torch.no_grad():
        return model.run_with_hooks(tokens, fwd_hooks=hooks)


# --------------------------------------------------------------------------- #
# Performance metrics                                                          #
# --------------------------------------------------------------------------- #


def _perf(logits, outputs, pairs):
    pos = torch.tensor(ANSWER_POSITIONS)
    pred = logits[:, pos, :].argmax(-1)
    acc = (pred == outputs).float().mean().item()
    ld = logit_diff(logits, pairs).item()
    return acc, ld


def _eval_keep(model, keep_heads, tokens, outputs, pairs, mode, mean_cache, resample_cache):
    """Performance keeping ONLY `keep_heads` (ablate the complement)."""
    keep = set(keep_heads)
    all_heads = [(l, h) for l in range(model.cfg.n_layers) for h in range(model.cfg.n_heads)]
    ablate = [hd for hd in all_heads if hd not in keep]
    logits = run_with_ablation(model, tokens, ablate, mode, mean_cache, resample_cache)
    return _perf(logits, outputs, pairs)


# Cap on replicated sequences per batched forward (groups x examples). The model
# is tiny, so this is generous; it only bounds peak memory.
_MAX_GROUP_ROWS = 1 << 16


def _eval_keep_many(model, keep_sets, tokens, outputs, pairs, mode,
                    mean_cache, resample_cache):
    """Vectorized ``_eval_keep`` over MANY circuits at once.

    Given a list of ``keep_sets`` (each an iterable of ``(layer, head)`` to KEEP),
    return ``[(acc, ld), ...]`` -- numerically identical to calling ``_eval_keep``
    on each set, but computed in a SINGLE forward pass per chunk. We replicate the
    token batch once per keep-set (group-major) and apply, in one ``hook_z`` hook
    per layer, a per-group head-ablation mask. This turns the O(#heads) sequential
    forwards in faithfulness/completeness/minimality (and the O(n^2) greedy search)
    into one (chunked) batched forward -- the dominant validation speed-up on GPU.
    """
    G = len(keep_sets)
    if G == 0:
        return []
    nL, nH = model.cfg.n_layers, model.cfg.n_heads
    B, S = tokens.shape
    dev = tokens.device

    # Per-layer ablate mask [G, nH]: True where a head is NOT kept (so it is ablated).
    ablate_mask = torch.ones(nL, G, nH, dtype=torch.bool, device=dev)
    for g, keep in enumerate(keep_sets):
        for (l, h) in keep:
            ablate_mask[l, g, h] = False

    # Chunk groups so groups*examples stays within the memory budget.
    per_chunk = max(1, _MAX_GROUP_ROWS // max(1, B))
    out = []
    for c0 in range(0, G, per_chunk):
        c1 = min(G, c0 + per_chunk)
        gc = c1 - c0
        toks_rep = tokens.repeat(gc, 1)                       # [gc*B, S] group-major

        hooks = []
        for l in range(nL):
            name = f"blocks.{l}.attn.hook_z"
            m_l = ablate_mask[l, c0:c1]                       # [gc, nH]

            def hook(act, hook, l=l, m_l=m_l):
                # act: [gc*B, S, nH, dH]
                dH = act.shape[-1]
                av = act.view(gc, B, S, nH, dH)
                mask = m_l.view(gc, 1, 1, nH, 1)
                if mode == "zero":
                    av = av.masked_fill(mask, 0.0)
                elif mode == "mean":
                    mean = mean_cache[l].view(1, 1, S, nH, dH)
                    av = torch.where(mask, mean, av)
                elif mode == "resample":
                    rc = resample_cache[l][:B].reshape(1, B, S, nH, dH)
                    av = torch.where(mask, rc, av)
                return av.reshape(gc * B, S, nH, dH)

            hooks.append((name, hook))

        with torch.no_grad():
            logits = model.run_with_hooks(toks_rep, fwd_hooks=hooks)
        logits = logits.view(gc, B, S, -1)
        for g in range(gc):
            out.append(_perf(logits[g], outputs, pairs))
    return out


# --------------------------------------------------------------------------- #
# The three tests                                                             #
# --------------------------------------------------------------------------- #


def validate(circuit: Circuit, model=None, n=512, mode="mean", seed=0,
             scheme="random", mean_cache=None, resample_cache=None) -> dict:
    if model is None:
        model = load_segrev3()
    all_heads = [(l, h) for l in range(model.cfg.n_layers) for h in range(model.cfg.n_heads)]
    circuit_heads = sorted(circuit.heads)
    outside = [hd for hd in all_heads if hd not in circuit.heads]

    # Load the contrast (clean) data from the central store, on the SAME data
    # family as the circuit being validated (acdc/<tau>, eap/<topn>, or manual).
    # NB: `n` is no longer used to size the contrast set -- the store takes the
    # stored file as-is (see module docstring).
    app, kw = _circuit_data_spec(circuit.name)
    pairs, data_path = store.load(app, scheme, **kw)
    toks = pairs.clean_tokens
    outs = segrev3_tensor(pairs.clean_inputs)        # true outputs for these inputs

    # Caches depend only on (model, mode) -- not on the circuit -- so callers
    # validating many circuits can compute them once and pass them in.
    if mode == "mean" and mean_cache is None:
        mean_cache = _mean_z(model)
    if mode == "resample" and resample_cache is None:
        resample_cache = _resample_z(model, n)

    # Build EVERY keep-set the three tests need and evaluate them in ONE batched
    # forward pass (group-major over keep-sets). Order:
    #   0: full (keep all heads -> no ablation, == plain forward)
    #   1: empty (keep none)
    #   2: circuit-only        3: complement-only
    #   4 ..              : minimality (circuit minus each circuit head)
    #   4+|C| ..          : completeness (circuit plus each outside head)
    keep_sets = [all_heads, [], circuit_heads, outside]
    keep_sets += [[x for x in circuit_heads if x != hd] for hd in circuit_heads]
    keep_sets += [circuit_heads + [hd] for hd in outside]
    evals = _eval_keep_many(model, keep_sets, toks, outs, pairs, mode,
                            mean_cache, resample_cache)
    (full_acc, full_ld) = evals[0]
    (empty_acc, empty_ld) = evals[1]
    (circ_acc, circ_ld) = evals[2]
    (comp_acc, comp_ld) = evals[3]

    def recovered(x, full, empty):
        denom = (full - empty)
        return (x - empty) / denom if abs(denom) > 1e-9 else float("nan")

    faithfulness = {
        "full_acc": full_acc, "empty_acc": empty_acc, "circuit_acc": circ_acc,
        "full_logit_diff": full_ld, "empty_logit_diff": empty_ld, "circuit_logit_diff": circ_ld,
        "recovered_acc": recovered(circ_acc, full_acc, empty_acc),
        "recovered_logit_diff": recovered(circ_ld, full_ld, empty_ld),
        "n_circuit_heads": len(circuit_heads), "n_total_heads": len(all_heads),
    }

    # minimality: ablate each circuit head within the circuit-only model
    minimality = {}
    off = 4
    for i, hd in enumerate(circuit_heads):
        a, ld = evals[off + i]
        minimality[f"L{hd[0]}H{hd[1]}"] = {
            "acc_drop_vs_circuit": circ_acc - a,
            "logit_diff_drop_vs_circuit": circ_ld - ld,
        }

    # completeness: complement perf + incompleteness gap from adding each outside head
    gaps = {}
    off = 4 + len(circuit_heads)
    for i, hd in enumerate(outside):
        a, ld = evals[off + i]
        gaps[f"L{hd[0]}H{hd[1]}"] = {"acc_change": a - circ_acc, "logit_diff_change": ld - circ_ld}
    max_gap_head = max(gaps, key=lambda k: abs(gaps[k]["logit_diff_change"])) if gaps else None
    completeness = {
        "complement_acc": comp_acc, "complement_logit_diff": comp_ld,
        "max_incompleteness_head": max_gap_head,
        "max_incompleteness_logit_diff": (gaps[max_gap_head]["logit_diff_change"] if max_gap_head else 0.0),
        "per_outside_head": gaps,
    }

    data_block = {
        "store_label": store.dataset_label(app, scheme, **kw),
        "path": repo_relpath(data_path),
        "scheme": pairs.scheme,
        "n_examples": int(pairs.clean_tokens.shape[0]),
        # number of 3s (segrev3 boundary digit) in the clean inputs behind this run
        "n_threes_caption": N3.caption(pairs.clean_inputs, prefix="clean n_3s"),
    }

    return {"name": circuit.name, "mode": mode,
            "data": data_block,
            "faithfulness": faithfulness, "minimality": minimality,
            "completeness": completeness}


# --------------------------------------------------------------------------- #
# Figures + driver                                                            #
# --------------------------------------------------------------------------- #


def _figs(res, out_prefix):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n3 = res.get("data", {}).get("n_threes_caption")
    mini = res["minimality"]
    labels = list(mini.keys())
    drops = [mini[k]["logit_diff_drop_vs_circuit"] for k in labels]
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.4), 3.4))
    ax.bar(range(len(labels)), drops, color="#2980b9")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.axhline(0, color="k", lw=0.6)
    full_model = (res["faithfulness"]["n_circuit_heads"]
                  == res["faithfulness"]["n_total_heads"])
    ax.set_title(f"{res['name']}: minimality — logit-diff drop when each head is "
                 "ablated (full model, no circuit selection)" if full_model else
                 f"{res['name']}: minimality — logit-diff drop when each circuit head is ablated")
    ax.set_ylabel("logit-diff drop")
    fig.tight_layout(); N3.annotate(fig, n3)
    fig.savefig(out_prefix + "_minimality.png", dpi=140, bbox_inches="tight"); plt.close(fig)

    f = res["faithfulness"]; c = res["completeness"]
    fig, ax = plt.subplots(figsize=(6, 3.6))
    bars = {"full": f["full_logit_diff"], "circuit": f["circuit_logit_diff"],
            "complement": c["complement_logit_diff"], "empty": f["empty_logit_diff"]}
    ax.bar(range(len(bars)), list(bars.values()),
           color=["#2c3e50", "#27ae60", "#e67e22", "#c0392b"])
    ax.set_xticks(range(len(bars))); ax.set_xticklabels(list(bars.keys()))
    ax.set_ylabel("logit diff")
    ax.set_title(f"{res['name']}: faithfulness  (rec_LD={f['recovered_logit_diff']:.2f}, "
                 f"rec_acc={f['recovered_acc']:.2f}, |C|={f['n_circuit_heads']})")
    fig.tight_layout(); N3.annotate(fig, n3)
    fig.savefig(out_prefix + "_summary.png", dpi=140, bbox_inches="tight"); plt.close(fig)


def faithfulness_curve(model, ranking: Sequence[Head], n=512, mode="mean", seed=0):
    """Recovered accuracy/logit-diff as the circuit grows to the top-k heads of a
    given importance `ranking` (e.g. whole-head knockout order). Shows how
    DISTRIBUTED the computation is."""
    pairs, data_path = store.load("manual", "random")
    toks = pairs.clean_tokens
    outs = segrev3_tensor(pairs.clean_inputs)
    mean_cache = _mean_z(model) if mode == "mean" else None
    resample_cache = _resample_z(model, n) if mode == "resample" else None
    all_heads = [(l, h) for l in range(model.cfg.n_layers) for h in range(model.cfg.n_heads)]
    # full (keep all), empty (keep none), then the top-k prefixes -- one batched pass.
    keep_sets = [all_heads, []] + [list(ranking[:k]) for k in range(1, len(ranking) + 1)]
    evals = _eval_keep_many(model, keep_sets, toks, outs, pairs, mode, mean_cache, resample_cache)
    (full_acc, full_ld) = evals[0]
    (empty_acc, empty_ld) = evals[1]
    ks, racc, rld = [], [], []
    for k in range(1, len(ranking) + 1):
        a, ld = evals[1 + k]
        ks.append(k)
        racc.append((a - empty_acc) / (full_acc - empty_acc) if full_acc != empty_acc else float("nan"))
        rld.append((ld - empty_ld) / (full_ld - empty_ld) if full_ld != empty_ld else float("nan"))
    return {"k": ks, "recovered_acc": racc, "recovered_logit_diff": rld,
            "full_acc": full_acc, "empty_acc": empty_acc,
            "data": {"store_label": store.dataset_label("manual", "random"),
                     "path": repo_relpath(data_path), "scheme": pairs.scheme,
                     "n_examples": int(pairs.clean_tokens.shape[0]),
                     "n_threes_caption": N3.caption(pairs.clean_inputs, prefix="clean n_3s")}}


def plot_faithfulness_curve(model=None, mode="mean"):
    """Sweep top-k heads by whole-head knockout importance; plot recovered vs k."""
    if model is None:
        model = load_segrev3()
    # Head order: the whole-head knockout ranking stored with the committed curve
    # (the knockout results file it was originally derived from is not shipped).
    rank = json.load(open(os.path.join(DATA_DIR, "manual", "faithfulness_curve.json")))["ranking"]
    ranking = [(int(h[1:h.index("H")]), int(h[h.index("H") + 1:])) for h in rank]
    cur = faithfulness_curve(model, ranking, mode=mode)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(cur["k"], cur["recovered_acc"], "-o", ms=3, label="recovered token-acc")
    ax.plot(cur["k"], cur["recovered_logit_diff"], "-s", ms=3, label="recovered logit-diff")
    ax.axhline(0.9, color="grey", ls="--", lw=0.8)
    ax.set_xlabel("# heads kept (top-k by whole-head knockout importance)")
    ax.set_ylabel("recovered fraction")
    ax.set_title("segrev3 faithfulness vs circuit size (mean ablation)\nthe computation is distributed: many heads needed for full recovery")
    ax.legend(); fig.tight_layout()
    N3.annotate(fig, cur.get("data", {}).get("n_threes_caption"))
    # faithfulness_curve is a general (manual-pipeline) output -> manual/ group.
    man_fig = os.path.join(FIG_DIR, "manual")
    man_data = os.path.join(DATA_DIR, "manual")
    os.makedirs(man_fig, exist_ok=True)
    os.makedirs(man_data, exist_ok=True)
    fig.savefig(os.path.join(man_fig, "faithfulness_curve.png"), dpi=140, bbox_inches="tight")
    plt.close(fig)
    with open(os.path.join(man_data, "faithfulness_curve.json"), "w") as f:
        json.dump({"ranking": [f"L{l}H{h}" for (l, h) in ranking], **cur}, f, indent=2)
    return cur


def _aphr_to_lh(label: str):
    """'a0.h7' -> 'L0H7' (EAP node-name format -> validation head label)."""
    m = re.match(r"a(\d+)\.h(\d+)", label)
    return f"L{m.group(1)}H{m.group(2)}" if m else None


def _method_circuits(method: str):
    """Return [(param_value, circuit_name, [head_labels]), ...] for a discovery
    method, read from its per-param discovery outputs. param = tau (acdc) | top-n (eap)."""
    viz = os.path.join(_ROOT, "visualizations")
    out = []
    if method == "acdc":
        for sp in glob.glob(os.path.join(viz, "acdc", "acdc_thresh_*", "summary.json")):
            d = json.load(open(sp))
            tau = float(d["threshold"])
            # name encodes tau so validate() loads datasets/acdc/acdc_thresh_<tau>/random.json
            out.append((tau, f"acdc_thresh_{('%g' % tau)}", list(d.get("heads_in_circuit", []))))
    elif method == "eap":
        s = json.load(open(os.path.join(viz, "eap", "eap_all_topn_summary.json")))
        for k, v in s.get("per_topn", {}).items():
            n = int(k)
            heads = [h for h in (_aphr_to_lh(x) for x in v.get("circuit_heads", [])) if h]
            out.append((n, f"eap_top{n}", heads))
    else:
        raise ValueError(f"method must be 'acdc' or 'eap', got {method!r}")
    out.sort(key=lambda r: r[0])
    return out


def method_faithfulness_curve(method: str, mode: str = "mean", model=None):
    """Validation FAITHFULNESS vs the discovery parameter (tau for ACDC, top-n for
    EAP): validate each per-param discovered circuit and plot recovered token-acc +
    logit-diff against the parameter. Written to the method's validation group:
        validation/figs_data/figs/<method>/<method>_faithfulness_curve.png
        validation/figs_data/data/<method>/<method>_faithfulness_curve.json
    (Each point uses that circuit's own store dataset, via validate().)"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if model is None:
        model = load_segrev3()

    circuits = _method_circuits(method)
    xs, acc, ld, rows = [], [], [], []
    for param, name, heads in circuits:
        res = validate(Circuit.from_labels(name, heads), model=model, mode=mode)
        fa = res["faithfulness"]
        xs.append(param)
        acc.append(fa["recovered_acc"])
        ld.append(fa["recovered_logit_diff"])
        rows.append({"param": param, "circuit": name, "n_heads": fa["n_circuit_heads"],
                     "recovered_acc": fa["recovered_acc"],
                     "recovered_logit_diff": fa["recovered_logit_diff"],
                     "data": res.get("data")})

    pname = "tau" if method == "acdc" else "top-n"
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(xs, acc, "-o", label="recovered token-acc")
    ax.plot(xs, ld, "-s", label="recovered logit-diff")
    ax.axhline(0.9, color="grey", ls="--", lw=0.8)
    if method == "acdc":
        ax.set_xscale("log")
    ax.set_xlabel(f"{pname}  ({'ACDC threshold' if method=='acdc' else 'EAP edges kept'})")
    ax.set_ylabel("recovered fraction (vs full / empty)")
    ax.set_title(f"{method.upper()} faithfulness vs {pname}  (validation, {mode} ablation)")
    ax.legend(); fig.tight_layout()
    # Each point uses its own per-circuit random dataset; mark the 3-count of one as
    # representative (all are the 'random' scheme; per-point detail in the JSON).
    _eg = next((r["data"]["n_threes_caption"] for r in rows
                if r.get("data", {}).get("n_threes_caption")), None)
    N3.annotate(fig, (f"n_3s per point (random scheme); e.g. {_eg}" if _eg else None))

    # png -> figs_data/figs/<method>/ ; json -> figs_data/data/<method>/
    fig_dir = os.path.join(FIG_DIR, method)
    data_dir = os.path.join(DATA_DIR, method)
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    out_png = os.path.join(fig_dir, f"{method}_faithfulness_curve.png")
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)
    with open(os.path.join(data_dir, f"{method}_faithfulness_curve.json"), "w") as f:
        json.dump({"method": method, "param": pname, "mode": mode, "points": rows}, f, indent=2)
    print(f"  {method} faithfulness curve ({len(xs)} pts) -> {os.path.relpath(out_png, _ROOT)}")
    return out_png


def greedy_head_circuit(model=None, mode="mean", thresh=0.95, seed=11, n=512,
                        candidates=None):
    """Greedy HEAD-level circuit discovery — the head-level analog of ACDC.

    Repeatedly remove the least-damaging head while the head-level recovered
    token-accuracy stays >= `thresh` x full. Returns the minimal faithful head set.
    Because it optimizes head importance directly (not edge-KL), it gives tighter
    and better-chosen small head sets than projecting an ACDC/EAP edge circuit down
    to heads. `candidates` (list of (l,h)) restricts the search to a candidate pool
    (e.g. the union of the ACDC/EAP heads — the "work off the discovery result"
    workflow); default = all heads.
    """
    if model is None:
        model = load_segrev3()
    nL, nH = model.cfg.n_layers, model.cfg.n_heads
    all_heads = [(l, h) for l in range(nL) for h in range(nH)]
    cand = list(candidates) if candidates is not None else all_heads
    pairs, data_path = store.load("manual", "random")
    toks = pairs.clean_tokens
    outs = segrev3_tensor(pairs.clean_inputs)
    mean_cache = _mean_z(model) if mode == "mean" else None
    resample_cache = _resample_z(model, n) if mode == "resample" else None

    def acc(keep):
        a, _ = _eval_keep(model, keep, toks, outs, pairs, mode, mean_cache, resample_cache)
        return a

    full = acc(all_heads)
    target = thresh * full
    keep = list(cand)
    removal = []
    _pbar = tqdm(total=max(0, len(keep) - 1), desc="greedy head-ablation",
                 unit="head", leave=False)
    while len(keep) > 1:
        # Evaluate removing EACH remaining head in one batched forward (was one
        # forward per candidate -> turns the O(n^2) search into O(n) passes).
        cand_sets = [[x for x in keep if x != hd] for hd in keep]
        accs = [a for (a, _ld) in _eval_keep_many(
            model, cand_sets, toks, outs, pairs, mode, mean_cache, resample_cache)]
        bi = max(range(len(keep)), key=lambda i: accs[i])
        best, best_acc = keep[bi], accs[bi]
        if best_acc >= target:
            keep.remove(best)
            removal.append({"removed": f"L{best[0]}H{best[1]}", "acc_after": round(best_acc, 4)})
            _pbar.update(1)
        else:
            break
    _pbar.close()
    return {"mode": mode, "thresh": thresh, "seed": seed, "n": n,
            "data": {"store_label": store.dataset_label("manual", "random"),
                     "path": repo_relpath(data_path), "scheme": pairs.scheme,
                     "n_examples": int(pairs.clean_tokens.shape[0])},
            "full_acc": round(full, 4), "final_acc": round(acc(keep), 4),
            "n_heads": len(keep),
            "heads": sorted(f"L{l}H{h}" for l, h in keep),
            "removal_order": removal}


def main(circuit_labels=None, name="rough", mode="mean", model=None,
         mean_cache=None, resample_cache=None):
    sub = _group_subdir(name)
    data_dir = os.path.join(DATA_DIR, sub)
    fig_dir = os.path.join(FIG_DIR, sub)
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)
    if circuit_labels is None:
        # FULL-MODEL default: the manual pipeline no longer synthesizes a rough
        # circuit, so the no-args validation covers EVERY head (4 layers x 8
        # heads). Minimality then reports the logit-diff drop of ablating each
        # of the 32 heads individually; faithfulness/completeness are trivial
        # (circuit == full model, complement == empty) and kept for format parity.
        circuit_labels = [f"L{l}H{h}" for l in range(4) for h in range(8)]
    circuit = Circuit.from_labels(name, circuit_labels)
    res = validate(circuit, model=model, mode=mode,
                   mean_cache=mean_cache, resample_cache=resample_cache)
    json_path = os.path.join(data_dir, name + "_validation.json")
    with open(json_path, "w") as f:
        json.dump(res, f, indent=2)
    _figs(res, os.path.join(fig_dir, name))
    fa = res["faithfulness"]; co = res["completeness"]
    print(f"=== validation: {name} (mode={mode}) ===")
    print(f"  heads in circuit: {fa['n_circuit_heads']}/{fa['n_total_heads']}")
    print(f"  FAITHFULNESS  recovered_acc={fa['recovered_acc']:.3f}  recovered_LD={fa['recovered_logit_diff']:.3f}")
    print(f"    full_acc={fa['full_acc']:.3f}  circuit_acc={fa['circuit_acc']:.3f}  empty_acc={fa['empty_acc']:.3f}")
    print(f"  COMPLETENESS  complement_acc={co['complement_acc']:.3f}  "
          f"max_incompleteness_LD={co['max_incompleteness_logit_diff']:.3f} ({co['max_incompleteness_head']})")
    mini = res["minimality"]
    worst = sorted(mini.items(), key=lambda kv: kv[1]["logit_diff_drop_vs_circuit"])
    print("  MINIMALITY smallest contributors:",
          [(k, round(v['logit_diff_drop_vs_circuit'], 3)) for k, v in worst[:4]])
    print(f"  wrote {json_path} + figures under {fig_dir}")
    return res


# --------------------------------------------------------------------------- #
# Discovered-circuit validation driver                                        #
# --------------------------------------------------------------------------- #
# Validate the circuits PRODUCED by the discovery runners (acdc.py / eap.py /
# the manual synthesis) without hand-writing head lists. This replaces the
# ad-hoc ``python -c "...json.load...V.main(circuit_labels=...)"`` one-liners
# that used to be documented by hand -- the circuit's heads are read
# straight from its on-disk artifact via interp_tools/circuit_loader, and the
# output is grouped exactly like the discovery folders (acdc/<name>, eap/<name>,
# manual) so validation, post-analysis and the discovery outputs all line up.


def discovered_circuit_names(method: str, repo_root: str = _ROOT) -> list:
    """Folder names of the discovered circuits for ``method``.

    acdc -> ['acdc_thresh_<tau>', ...]   eap -> ['eap_top<n>', ...]
    manual -> ['rough'] (iff visualizations/rough_circuit.json exists).
    """
    import glob
    viz = os.path.join(repo_root, "visualizations")
    if method == "acdc":
        return sorted(os.path.basename(d)
                      for d in glob.glob(os.path.join(viz, "acdc", "acdc_thresh_*"))
                      if os.path.isdir(d))
    if method == "eap":
        return sorted(os.path.basename(d)
                      for d in glob.glob(os.path.join(viz, "eap", "eap_top*"))
                      if os.path.isdir(d))
    if method == "manual":
        return ["rough"] if os.path.exists(os.path.join(viz, "rough_circuit.json")) else []
    raise ValueError(f"unknown method {method!r} (expected acdc|eap|manual)")


def validate_circuit(method: str, name: str, mode: str = "mean", model=None,
                     mean_cache=None, resample_cache=None) -> dict:
    """Load a discovered circuit's heads (via circuit_loader) and run the full
    faithfulness/completeness/minimality validation through ``main()`` so the
    JSON + figures land under the circuit's group folder."""
    sys.path.insert(0, os.path.join(_ROOT, "interp_tools"))
    import circuit_loader as CL  # noqa: E402  (interp_tools on sys.path)
    circuit = CL.load_circuit(method, name)
    labels = CL.head_labels(circuit)
    return main(circuit_labels=labels, name=name, mode=mode, model=model,
                mean_cache=mean_cache, resample_cache=resample_cache)


def validate_discovered(method: str, name: str = "all", mode: str = "mean") -> list:
    """Validate every discovered circuit of ``method`` (or a single ``name``).

    ``method='all'`` validates acdc + eap + manual. Failures on one circuit are
    reported and skipped so the rest still run. Returns the list of result dicts.
    """
    methods = ["acdc", "eap", "manual"] if method == "all" else [method]
    # Load the model + ablation caches ONCE and reuse for every circuit (was a fresh
    # load_segrev3 + _mean_z per circuit inside main()/validate()).
    model = load_segrev3()
    mean_cache = _mean_z(model) if mode == "mean" else None
    results = []
    for m in methods:
        names = discovered_circuit_names(m) if name == "all" else [name]
        if not names:
            print(f"  (no discovered {m} circuits to validate)")
            continue
        for nm in names:
            try:
                results.append(validate_circuit(m, nm, mode=mode, model=model,
                                                mean_cache=mean_cache))
            except Exception as e:  # keep going across circuits
                print(f"⚠ validation failed for {m}/{nm}: {e}")
    return results


# segrev3_batch imported lazily to avoid a name clash with the private helpers above
from data_gen import segrev3_batch as __segrev3_batch                          # noqa: E402

if __name__ == "__main__":
    import argparse

    import canon  # noqa: E402  (data_gen/ is on sys.path above)

    ap = argparse.ArgumentParser(
        description="Circuit validation (faithfulness / completeness / minimality). "
                    "Select what to validate with ONE --circuit selector; with no "
                    "arguments, validates the FULL model (all 32 heads) under the name "
                    "'rough' -- note --circuit rough (the heuristic circuit) writes the "
                    "same manual/rough_validation.json.")
    ap.add_argument("--circuit", default=None,
                    help="'all' | a method (manual|acdc|eap = all of its circuits) | "
                         "a circuit folder name (rough | acdc_thresh_<tau> | "
                         "eap_top<n>; the method is inferred). Preferred selector.")
    ap.add_argument("--method", choices=["acdc", "eap", "manual", "all"], default=None,
                    help="(legacy spelling) validate this method's circuits; "
                         "--circuit overrides it.")
    ap.add_argument("--name", default="all",
                    help="(legacy spelling) a specific circuit folder name or 'all'.")
    ap.add_argument("--mode", default="mean", choices=["mean", "zero", "resample"],
                    help="ablation reference (default: mean).")
    a = ap.parse_args()

    if a.circuit is not None:
        try:
            method, name = canon.parse_circuit(a.circuit)
        except ValueError as e:
            ap.error(str(e))
        validate_discovered(method, name=name, mode=a.mode)
    elif a.method is None:
        main(mode=a.mode)                       # no args: full model, saved as 'rough' (legacy name)
    else:
        validate_discovered(a.method, name=a.name, mode=a.mode)
