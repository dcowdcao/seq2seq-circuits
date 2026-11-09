#!/usr/bin/env python
"""Write results/showcase/<example>/README.md for each example from its circuit.json.

Every number on a card is read from circuit.json / candidates.json / the checkpoint, or
computed here by running the DSL reference and the model on inputs drawn with a fixed
seed. Only the prose (what the algorithm does, how faithfulness is defined) is written
by hand, in ALGORITHMS and CLASS_TEXT below.

    python results/showcase/make_cards.py [--only <example dir> ...] [--device cpu]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import showcase_spec as S  # noqa: E402

for _p in S.PY_PATHS:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402

import algorithms  # noqa: E402
import gen_circuit_data as GCD  # noqa: E402
import model_registry as MR  # noqa: E402
import train_probes as TP  # noqa: E402
import variables as V  # noqa: E402
from make_figures import norm_edge  # noqa: E402

SEED = 0          # seed for every example input drawn on a card
N_EXAMPLES = 4

ALGORITHMS = {
    "schmitt_id__moe8": (
        "A Schmitt trigger: a switch with a high and a low threshold plus memory. The model "
        "reads the digits left to right and keeps a state that starts high (9). A digit of 3 "
        "or less switches the state low (0), a digit of 6 or more switches it high (9), and "
        "the digits 4 and 5 leave it unchanged. Each output digit is the current state.\n\n"
        "The circuit is conditioned on inputs that contain no 4 or 5 (`schmitt_noband`). On "
        "that domain the memory is never consulted: every output digit is a threshold test of "
        "the input digit at the same position. The DomainCircuit is that memoryless threshold "
        "detector."),
    "mexhat_id__moe8": (
        "A dip detector (a \"Mexican hat\" veto). Every interior digit is compared with its two "
        "neighbours: if the neighbours' sum exceeds twice the digit by 4 or more (a sharp local "
        "dip), the output at that position is 0; otherwise the digit is copied. The first and "
        "last digits are always copied.\n\n"
        "`mexhat_fill` pairs a random input with a one-digit edit of it that changes the "
        "output. The reported ContrastCircuit is the selected candidate under interchange "
        "patching: the part of the model that carries the effect of that edit to the output "
        "positions where the two outputs disagree. Its fringe beyond the domain circuit "
        "`mexhat_full` is listed alongside."),
    "mux2.pmstep-id.sr12-id__moe8": (
        "A multiplexer over two algorithms. The first token is a cue that selects which "
        "algorithm is applied to the ten payload digits after it. Cue 10 selects task 0, "
        "`pmstep`: add 1 to every even digit and subtract 1 from every odd digit (mod 10). "
        "Cue 11 selects task 1, `sr12`: a set/reset latch whose bit is switched on by a 1, "
        "switched off by a 2 and held by every other digit (it starts off); the output is the "
        "bit at each position.\n\n"
        "`suppress_t1_over_t0` pairs the latch cue with the parity-step cue on the same "
        "payload. Where the two algorithms disagree, the model must print the latch's answer "
        "under the latch cue and the parity step's answer under the parity cue. The reported "
        "SuppressionCircuit is the whole selected candidate: every row is scored against its "
        "own task's output, so the circuit has to route on the cue and run both tasks. Its "
        "fringe beyond the latch's own circuit (`task1`) is listed alongside as the "
        "suppression-specific part."),
    "graviclump_id__moe4": (
        "Gravity toward the centre of mass. The model takes the average of the ten input "
        "digits, rounded to the nearest integer (halves round to the even integer), and moves "
        "every digit one step toward it: digits below the average go up by one, digits above "
        "it go down by one, digits equal to it stay.\n\n"
        "The average is never printed. `graviclump_mean` is that hidden value, read by a "
        "linear probe at the last input position (the first position that has seen every "
        "digit). The ProbeCircuit is the part of the model that computes it."),
    "psum_novel_id__moe8": (
        "A composition of two operators. The inner one flags novelty: a position gets 1 if its "
        "digit has not appeared earlier in the input and 0 if it is a repeat. The outer one "
        "prints the running sum of those flags modulo 10, i.e. how many distinct digits have "
        "been seen so far.\n\n"
        "The novelty flags themselves are never printed. `bridge1_novel_id` is that "
        "intermediate value, read by a linear probe at each input position where it is not "
        "constant. The BridgeCircuit is the part of the model that computes \"is this digit "
        "new?\" on the way to the running count."),
}

CLASS_TEXT = {
    "DomainCircuit": (
        "Faithfulness `a` is the circuit's whole-sequence accuracy on inputs drawn from the "
        "conditioned domain, under edge-level mean ablation: every reader (each head's Q, K "
        "and V input, each MLP input, the logits) receives the full model's per-position mean "
        "residual plus the live contribution of exactly the circuit's senders, and circuit "
        "nodes are recomputed from their declared input edges."),
    "ContrastCircuit": (
        "Faithfulness is measured by interchange patching on clean/corrupt pairs that share "
        "one teacher-forced prefix (the clean input's true output). Each row's ablation "
        "baseline is its partner's unablated activations, and the labels are the full model's "
        "own outputs. Forward: start from the corrupt run, patch the clean row in through the "
        "circuit's edges, and require the full model's clean output (`a_clean`). Reverse: the "
        "mirror image (`a_corrupt`). `a = a_clean × a_corrupt`. Pairs whose outputs do not "
        "differ, or differ only by the edited digit copied straight through, are dropped. The "
        "empty circuit scores 0 and the full model 1 by construction, so the circuit is the "
        "path that carries the edit's effect, and parts of the model the edit never changes "
        "drop out without subtracting the domain circuit; the reported circuit is the "
        "selected candidate itself, and its fringe beyond the declared domain circuit is "
        "listed alongside. Interchange faithfulness is not "
        "comparable with mean-ablation faithfulness (the latter is listed as secondary)."),
    "SuppressionCircuit": (
        "Faithfulness is measured on alternating clean/corrupt pairs: clean = the cued task's "
        "cue plus a payload, corrupt = the other task's cue plus the same payload (payloads on "
        "which the two tasks disagree). Every row is scored against its own reference output "
        "under edge-level mean ablation (per-position means over the mixed clean and corrupt "
        "rows); `a = a_clean × a_corrupt`, so the evaluated circuit has to route the cue and "
        "run both tasks. Each EAP-IG candidate was evaluated as-is and united with the cued "
        "task's own circuit (`task1`, stored in `base_circuit.json`); the best passing "
        "selection score over both modes was kept. The reported circuit (size, `a`) is that "
        "whole selected candidate. Its fringe beyond `task1` (everything the candidate adds to "
        "the latch's own circuit: the routing and the parity step's execution) is listed below and drawn bold in "
        "the figure."),
    "ProbeCircuit": (
        "Faithfulness is probe accuracy with the model ablated down to the circuit (edge-level "
        "mean ablation, the graph cut at the probe layer), normalized so that 0 is the "
        "per-position majority class (the best constant prediction) and 1 is the full model, "
        "clamped at 0. It is measured on clean inputs (`a_clean`) and on a "
        "corrupt member that is the single-digit edit making the probe's correct class most "
        "strongly decodable (`a_corrupt`); `a = a_clean × a_corrupt`. Components above the "
        "probe layer cannot affect the probe, so they are never part of the graph. Discovery "
        "and evaluation both read each probed position's own linear probe, at the layer whose "
        "per-position probes are most accurate."),
    "BridgeCircuit": (
        "Found and scored exactly like a ProbeCircuit, with the probe reading the "
        "intermediate value of the composition: normalized probe accuracy (0 = majority "
        "class, 1 = full model, clamped at 0) on clean inputs and on the single-digit edit "
        "that makes the probe's correct class most strongly decodable (per-position probes at "
        "the layer where they are most accurate, as in the probe-metric EAP-IG run); "
        "`a = a_clean × a_corrupt`."),
}


#: how each probe-metric EAP-IG pair pool builds its corrupt member (circuits.py)
POOLS = {
    "graviclump_full": "clean and corrupt are two independent uniform random inputs",
    "graviclump_edit": "the corrupt input is the clean one with a single digit changed so that "
                       "the output changes",
    "psum_novel_whole": "clean and corrupt are two independent uniform random inputs",
    "psum_novel_fresh": "the corrupt input is the clean one with one repeated digit replaced by "
                        "a digit that does not occur in it (a repeat becomes novel)",
}
GENERATED = "generated by `reproduce.py --full`, gitignored"
READOUT = {"position": "per-position probes", "pooled": "pooled probe"}


def pools_text(ex, cj: dict) -> list:
    """Which pair pools probe-metric EAP-IG ran on, how each builds its corrupt member, and
    which pool the selected candidate came from."""
    import reproduce
    pools = [argv[argv.index("--circuit") + 1] for _, argv in reproduce.DISCOVERY[ex.key]
             if "--probe" in argv]
    pools = list(dict.fromkeys(pools))
    won = cj["candidate"].split("/")[3]
    lines = [f"- `{p}`: {POOLS[p]}" + (" (the selected candidate comes from this pool)"
                                       if p == won else "") for p in pools]
    return (["Discovery ran probe-metric EAP-IG on these clean/corrupt pair pools, and every "
             "candidate from every pool entered one selection:", ""] + lines +
            ["", "The evaluation's corrupt member differs from both: for each clean input it "
             "is the single-digit edit that makes the probe's correct class most strongly "
             "decodable.", ""])


def fmt_seq(t) -> str:
    return " ".join(str(int(v)) for v in t)


def greedy(model, inputs: torch.Tensor, out_len: int) -> torch.Tensor:
    """The model's own greedy outputs for ``inputs`` [B, in_len]."""
    toks = inputs.clone()
    with torch.no_grad():
        for _ in range(out_len):
            toks = torch.cat([toks, model(toks)[:, -1].argmax(-1, keepdim=True)], dim=1)
    return toks[:, inputs.shape[1]:]


def circuit_obj(model: str, cid: str):
    return next(c for c in GCD.circuits_for(model) if c.circuit_id == cid)


def io_domain(ex, model, spec) -> list[str]:
    circ = circuit_obj(ex.model, ex.circuit)
    xs = circ.sample(N_EXAMPLES, SEED)
    ys, ps = spec.task_fn(xs), greedy(model, xs, spec.n_ctx - spec.input_len)
    rows = ["| input (domain) | algorithm output | model output | match |", "|---|---|---|---|"]
    rows += [f"| `{fmt_seq(x)}` | `{fmt_seq(y)}` | `{fmt_seq(p)}` | "
             f"{'yes' if torch.equal(y, p) else 'no'} |" for x, y, p in zip(xs, ys, ps)]
    return [f"Inputs drawn from the circuit's domain (`{ex.circuit}.sample({N_EXAMPLES}, "
            f"seed={SEED})`), the reference output and the model's greedy output:", ""] + rows


def io_contrast(ex, model, spec) -> list[str]:
    circ = circuit_obj(ex.model, ex.circuit)
    clean, corrupt = circ.sample_pairs(N_EXAMPLES, SEED)
    out_len = spec.n_ctx - spec.input_len
    rows = ["| pair | member | input | algorithm output | model output |", "|---|---|---|---|---|"]
    for i in range(N_EXAMPLES):
        pos = int((clean[i] != corrupt[i]).nonzero()[0])
        for name, x in (("clean", clean[i]), ("corrupt", corrupt[i])):
            y = spec.task_fn(x[None])[0]
            p = greedy(model, x[None], out_len)[0]
            rows.append(f"| {i} (edit at {pos}) | {name} | `{fmt_seq(x)}` | `{fmt_seq(y)}` | "
                        f"`{fmt_seq(p)}` |")
    return [f"Clean/corrupt pairs from `{ex.circuit}.sample_pairs({N_EXAMPLES}, seed={SEED})`; "
            "each corrupt input is the clean one with a single digit changed (position given, "
            "counting from 0):", ""] + rows


def io_suppression(ex, model, spec) -> list[str]:
    circ = circuit_obj(ex.model, ex.circuit)
    clean, corrupt = circ.sample_pairs(N_EXAMPLES, SEED)
    out_len = spec.n_ctx - spec.input_len
    rows = ["| payload | cue | algorithm output | model output | positions where the tasks differ |",
            "|---|---|---|---|---|"]
    for i in range(N_EXAMPLES):
        differ = circ.affected_outputs(clean[i], corrupt[i]).nonzero().flatten().tolist()
        for x in (clean[i], corrupt[i]):
            y = spec.task_fn(x[None])[0]
            p = greedy(model, x[None], out_len)[0]
            rows.append(f"| `{fmt_seq(x[1:])}` | {int(x[0])} (task {int(x[0]) - 10}) | "
                        f"`{fmt_seq(y)}` | `{fmt_seq(p)}` | {differ} |")
    return [f"The same payload under both cues (pairs from `{ex.circuit}.sample_pairs("
            f"{N_EXAMPLES}, seed={SEED})`; cue {10 + circ.cued_task} = the cued task "
            f"{circ.cued_task}, cue {10 + circ.wrong_task} = the task to suppress):", ""] + rows


def io_probe(ex, model, spec, circ_json: dict) -> list[str]:
    var = V.get(ex.model, ex.circuit)()
    pos = var.valid_positions()
    g = torch.Generator().manual_seed(SEED)
    xs = torch.randint(0, 10, (N_EXAMPLES, spec.input_len), generator=g)
    labels = var.labels(xs)
    ys, ps = spec.task_fn(xs), greedy(model, xs, spec.n_ctx - spec.input_len)
    probe = circ_json["probe"]
    path = os.path.join(ex.dir, "probes", ex.model, probe["variable"], f"site_L{probe['layer']}.pt")
    art = torch.load(path, map_location="cpu", weights_only=False)
    keep = [pos.tolist().index(p) for p in probe["positions"]]
    labels = labels[:, keep]
    with torch.no_grad():
        resid = model(xs, stop_at_layer=probe["layer"] + 1)
    x = TP.ln(resid[:, probe["positions"]])           # the evaluator's readout
    if probe["readout"] == "pooled":
        reading = (x @ art["W"].T + art["b"]).argmax(-1)
    else:
        reading = (torch.einsum("bpd,pcd->bpc", x, art["W_pos"]) + art["b_pos"]).argmax(-1)
    rows = ["| input | output (algorithm / model) | variable at positions "
            f"{probe['positions']} | probe reading (layer {probe['layer']}) |", "|---|---|---|---|"]
    rows += [f"| `{fmt_seq(xs[i])}` | `{fmt_seq(ys[i])}` / `{fmt_seq(ps[i])}` | "
             f"`{fmt_seq(labels[i])}` | `{fmt_seq(reading[i])}` |" for i in range(N_EXAMPLES)]
    return [f"Uniform random inputs (seed {SEED}), the outputs, the hidden variable "
            f"`{ex.circuit}` at its probed input positions (computed by replaying the "
            "algorithm, never the model), and the linear probe's reading of the full model's "
            f"residual stream after layer {probe['layer']} ({READOUT[probe['readout']]}):",
            ""] + rows


def arch_rows(cfg: dict) -> list[str]:
    ff = "none (attention only)"
    if not cfg.get("attn_only", True):
        ff = (f"mixture of experts: {cfg['num_experts']} experts of width {cfg['d_mlp']}, "
              f"top-{cfg['experts_per_token']} routing, {cfg['act_fn']}"
              if cfg.get("num_experts") else f"MLP of width {cfg['d_mlp']}, {cfg['act_fn']}")
    return ["| | |", "|---|---|",
            f"| layers | {cfg['n_layers']} |",
            f"| attention heads per layer | {cfg['n_heads']} |",
            f"| hidden size | {cfg['d_model']} |",
            f"| head size | {cfg['d_head']} |",
            f"| feed-forward | {ff} |",
            f"| vocabulary | {cfg['d_vocab']} tokens |",
            f"| context | {cfg['n_ctx']} positions |"]


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def edge_str(e, terminal: str = "logits") -> str:
    p, c, q = norm_edge(e)
    return f"{p} → {terminal if c == 'logits' else c}" + (f" ({q.upper()})" if q else "")


def heldout_row(ex, cj: dict) -> list:
    """The held-out re-evaluation (seed 1: fresh rows and fresh mean-reference rows)."""
    p = os.path.join(ex.dir, "heldout_seed1.json")
    if not os.path.exists(p):
        return []
    h = json.load(open(p))
    parts = (f" = {h['a_clean']:.4f} × {h['a_corrupt']:.4f}"
             if h.get("a_corrupt") is not None else "")
    verdict = "above" if h["passes"] else "**below**"
    return [f"| held-out `a` (seed 1: fresh rows) | {h['a']:.4f}{parts} ({verdict} the floor; "
            "see [heldout_seed1.json](heldout_seed1.json)) |"]


def card(ex, device: str) -> str:
    cj = json.load(open(os.path.join(ex.dir, "circuit.json")))
    cand_path = os.path.join(ex.dir, "candidates.json")
    cands = json.load(open(cand_path)) if os.path.exists(cand_path) else None
    ck = torch.load(ex.checkpoint, map_location="cpu", weights_only=False)
    cfg = dict(ck["config"])
    spec = MR.get_spec(ex.model)
    model = MR.load_model(ex.model, device=device)
    expr = algorithms.decode_algorithm_id(algorithms.base_algorithm_id(ex.model))

    if ex.cls == "DomainCircuit":
        io = io_domain(ex, model, spec)
    elif ex.cls == "ContrastCircuit":
        io = io_contrast(ex, model, spec)
    elif ex.cls == "SuppressionCircuit":
        io = io_suppression(ex, model, spec)
    else:
        io = io_probe(ex, model, spec, cj)

    term = f"probe (layer {cj['probe']['layer']})" if cj.get("probe") else "logits"
    heads = sorted(n for n in cj["nodes"] if n.startswith("a"))
    mlps = sorted(n for n in cj["nodes"] if n.startswith("m"))
    verdict = "above" if cj["passes"] else "**below**"
    warn = ([f"> **Below the floor.** No EAP-IG candidate reached `a` > {cj['floor']:.2f} for "
             f"this class; the circuit shown is the most faithful candidate (`a` = "
             f"{cj['a']:.4f}). See [candidates.json](candidates.json).", ""]
            if not cj["passes"] else [])
    L = [f"# {ex.cls}: `{ex.model}` / `{ex.circuit}`", ""] + warn + [
         f"![{ex.cls} on {ex.model}](figure.png)", "",
         "## The algorithm", "", ALGORITHMS[ex.model], "",
         f"Reference expression: `{expr}`.", "",
         "## The model", "",
         f"Checkpoint: `{S.rel(ex.checkpoint)}` "
         f"(not included in the repository; sha256 `{sha256(ex.checkpoint)[:16]}…`).", ""] + arch_rows(cfg) + [
         "", "## Inputs and outputs", ""] + io + [
         "", "## The circuit", "", CLASS_TEXT[ex.cls], ""] + (
         pools_text(ex, cj) if cj.get("probe") else []) + [
         "| | |", "|---|---|",
         f"| nodes | {cj['n_nodes']} of {cj['n_nodes_full']} "
         f"({len(heads)} heads, {len(mlps)} MLP/MoE block{'s' if len(mlps) != 1 else ''}) |",
         f"| edges | {cj['n_edges']} of {cj['n_edges_full']} |",
         f"| faithfulness `a` | {cj['a']:.4f}"
         + (f" = {cj['a_clean']:.4f} (`a_clean`) × {cj['a_corrupt']:.4f} (`a_corrupt`)"
            if cj.get("a_corrupt") is not None else "") + " |",
         f"| floor | {cj['floor']:.2f} ({verdict} the floor) |"] + heldout_row(ex, cj) + [
         f"| selection score (1 − nodes/full)·(1 − edges/full) | {cj['selection_score']:.4f} |",
         f"| selected candidate | `{cj['candidate']}` ({GENERATED})"
         + ("" if cj["selection_mode"] == "candidate" else
            f" united with the base circuit (mode `{cj['selection_mode']}`)") + " |"]
    if cands is not None:
        n = sum(len(m["candidates"]) for m in cands["modes"])
        modes = " and ".join(f"`{m['mode']}`" for m in cands["modes"])
        L.append(f"| candidates evaluated | {n} (selection modes: {modes}; see "
                 f"[candidates.json](candidates.json), whose candidate paths are {GENERATED}) |")
    ref = cj.get("reference") or {}
    if ref:
        full = ref.get("model", ref.get("full"))
        L.append(f"| reference: unablated model / empty circuit | `a` = {full['a']:.4f} / "
                 f"{ref['empty']['a']:.4f} |")
    L.append(f"| ablation | {cj['ablation']} |")
    if cj.get("secondary"):
        sec = cj["secondary"]
        L.append(f"| {sec['ablation']}-ablation `a` (secondary, not comparable) | "
                 f"{sec[sec['ablation'] + '_ablation_a']:.4f} |")
    if cj.get("pairs_kept") is not None:
        drop = ", ".join(f"{k.replace('_', ' ')} {v}" for k, v in cj.get("pairs_dropped", {}).items())
        L.append(f"| pairs evaluated | {cj['pairs_kept']} of {cj['n_pairs']} drawn"
                 + (f" (dropped: {drop})" if drop else "") + " |")
    if cj.get("probe"):
        p = cj["probe"]
        L += [f"| probe | layer {p['layer']} ({READOUT[p['readout']]}), input positions "
              f"{p['positions']}, {p['n_classes']} classes |",
              f"| probe accuracy, full model (clean / corrupt) | {p['full_acc_clean']:.4f} / "
              f"{p['full_acc_corrupt']:.4f} |",
              f"| majority-class accuracy (clean / corrupt) | {p['majority_acc_clean']:.4f} / "
              f"{p['majority_acc_corrupt']:.4f} |"]
    if ex.probe_topns:
        L.append(f"| EAP-IG rungs | the canonical frac ladder plus explicit top-n "
                 f"{', '.join(map(str, ex.probe_topns))} on each pair pool: the 4-rung frac "
                 f"ladder over the {cj['n_edges_full']} probe-reachable edges is too coarse |")
    f = cj.get("fringe") or {}
    if f.get("available"):
        L.append(f"| fringe beyond `{f['base_circuit']}` | {f['n_nodes']} nodes, "
                 f"{f['n_edges']} edges (base: [`base_circuit.json`](base_circuit.json)) |")
    L += ["", "### Components", "",
          "Nodes: " + ", ".join(f"`{n}`" for n in cj["nodes"]) + ".", ""]
    if f.get("available"):
        L += [f"Fringe edges (not in `{f['base_circuit']}`; drawn bold in the figure):", ""]
        L += [f"- `{edge_str(e, term)}`" for e in f["edges"]] + [""]
    L += ["<details><summary>All edges (writer → reader, with the head input they enter)"
          "</summary>", ""]
    L += [f"- `{edge_str(e, term)}`" for e in cj["edges"]]
    ho = os.path.join(ex.dir, "heldout_seed1.json")
    L += ["", "</details>", "", "## Regenerate", "",
          "Re-evaluate the stored circuit (CPU, seconds to minutes):", "",
          "```bash", cj["command"], "```", ""]
    if os.path.exists(ho):
        L += ["Held-out re-evaluation (seed 1):", "", "```bash", json.load(open(ho))["command"],
              "```", ""]
    L += [
          "Re-run discovery and selection from the checkpoint: see "
          "[`../COMMANDS.md`](../COMMANDS.md) or `python results/showcase/reproduce.py --full`.",
          ""]
    return "\n".join(L)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args(argv)
    torch.manual_seed(SEED)
    for ex in S.EXAMPLES:
        if a.only is not None and ex.key not in a.only:
            continue
        if not os.path.exists(os.path.join(ex.dir, "circuit.json")):
            print(f"skip {ex.key}: no circuit.json yet")
            continue
        with open(os.path.join(ex.dir, "README.md"), "w") as f:
            f.write(card(ex, a.device))
        print("wrote", f"{ex.key}/README.md")


if __name__ == "__main__":
    main()
