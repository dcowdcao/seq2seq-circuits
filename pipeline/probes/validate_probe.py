#!/usr/bin/env python3
"""Selectivity + causal gates for trained probes. REPORT ONLY — no hard gates.

Per (model, variable), at the probe's evaluation layer (first_decodable_layer when the
scan verdict is "decodable", else the best-accuracy layer, flagged), on the scan's
causally-trimmed positions with the PER-POSITION probes (W_pos/b_pos):
  (a) permutation control  — per-position refit on per-position-shuffled labels;
                             position-mean accuracy should sit at chance.
  (b) leakage matrix       — probe V's per-position predictions mapped (train-fit majority
                             map) onto every sibling variable's labels; vs that sibling's
                             majority rate. Row-split, like every other fit here.
  (c) causal gate          — directional patch along W_pos[p][target]-W_pos[p][source] at
                             the probe site on matched rows, scaled until the probe itself
                             reads the target class post-LN; outputs scored against the DSL
                             counterfactual replay. TWO rates come out and both are always
                             recorded:
                               steered_rate  the PROBE moved. Steering along the probe's own
                                             weight vector and reading the result back with
                                             that same probe is near-tautological — it
                                             measures the patch, not the model.
                               flip_rate     the MODEL'S OUTPUT moved at the toggled slot.
                                             This is the honest measurement and the one the
                                             docs promise (>=0.80 flip, <=0.05 collateral).
                             ``--gate`` picks which one ``gate.passed`` enforces; it defaults
                             to ``output``, the doc-promised gate, which is also the one
                             repo_tools/gates.py accepts rows on.
                             Skipped (with reason) when the variable defines no counterfactual.
  (d) probe robustness     — 3 per-position refits across seeds/regularizations; agreement
                             of (verdict, first_decodable_layer). Resid sites only.

Everything here is measured at INPUT positions, so the forward runs over the input tokens
alone: appending output tokens cannot change a residual at an input position under a causal
mask (verified — max |free - teacher| = 0 at every input position over 4096 rows). The
generation-position scan lives in train_probes.py, whose ``regime`` block is copied into
this report as ``fit_regime``.

Writes probes_data/<model>/<variable>/validation.json. CLI: --model, --variables, --n,
--n-pairs, --seed, --device, --layer, --gate {steered,output,both}, --out
(probes root, default probes_data/).
"""
import argparse, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

import train_probes as tp
import variables as varlib
import model_registry

ROBUSTNESS_FITS = [(1, 1e-3), (2, 1e-2), (3, 1e-4)]   # (seed, weight decay)
ALPHAS = (0.5, 1.0, 2.0, 4.0, 8.0)                    # patch magnitude, x row resid norm
#: the documented output gate. Mirrors repo_tools/gates.FLIP_GATE/COLLATERAL_GATE, which
#: is the canonical copy: importing it here would make a GPU producer depend on the
#: repo tooling. Change both or neither.
FLIP_GATE, COLLATERAL_GATE = 0.80, 0.05
GATE_RULES = {"steered": "steered_rate >= 0.80 (the probe moved)",
              "output": "flip_rate >= 0.80 and collateral_rate <= 0.05 (the model moved)",
              "both": "steered_rate >= 0.80 AND flip_rate >= 0.80 AND collateral_rate <= 0.05"}


def eval_layer(scan: dict) -> tuple[int, bool]:
    """(layer, from_first_decodable). Best-layer fallback when absent/trivial_from_input."""
    if scan["first_decodable_layer"] is not None:
        return scan["first_decodable_layer"], True
    best = max(scan["per_layer_accuracy"], key=scan["per_layer_accuracy"].get)
    return int(best), False


def permutation_control(X: torch.Tensor, y: torch.Tensor, n_classes: int, seed: int) -> float:
    """Position-mean accuracy of per-position refits on per-position-shuffled labels."""
    g = torch.Generator().manual_seed(int(seed) + 7)
    perm = torch.stack([torch.randperm(y.shape[0], generator=g) for _ in range(y.shape[1])], dim=1)
    _, _, accs = tp.fit_positions(X, y.gather(0, perm), n_classes, seed=seed)
    return sum(accs) / len(accs)


def mapped_accuracy(preds: torch.Tensor, labels: torch.Tensor, seed: int) -> float:
    """Best train-fit class-mapping accuracy of ``preds`` [N,K] against ``labels`` [N,K]
    (leakage measure that works across differing class spaces); ROW-split holdout."""
    tr, te = tp.split(preds.shape[0], seed)
    ptr, ltr = preds[tr].reshape(-1), labels[tr].reshape(-1)
    mapping = {}
    for c in ptr.unique().tolist():
        sel = ltr[ptr == c]
        mapping[c] = int(sel.mode().values) if sel.numel() else 0
    pte, lte = preds[te].reshape(-1), labels[te].reshape(-1)
    mapped = torch.tensor([mapping.get(int(c), -1) for c in pte])   # unmapped class = wrong
    return (mapped == lte).float().mean().item()


def leakage_matrix(model_name: str, var_cls: type, layer: int, resids: dict,
                   inputs: torch.Tensor, seed: int) -> dict:
    art = torch.load(os.path.join(tp.probe_dir(model_name, var_cls.variable_id),
                                  f"site_L{layer}.pt"), weights_only=False)
    Wp, bp = art["W_pos"], art["b_pos"]
    fitpos = art["positions"].tolist()
    out = {}
    for other in varlib.for_model(model_name):
        o = other()
        o_pos = o.valid_positions().tolist()
        common = [p for p in fitpos if p in o_pos]
        if not common:
            continue
        j = torch.tensor([fitpos.index(p) for p in common])
        X = resids[art["hook"]][:, common]                          # [N, K, d]
        preds = (torch.einsum("nkd,kcd->nkc", X, Wp[j]) + bp[j]).argmax(-1)
        labs = o.labels(inputs)[:, [o_pos.index(p) for p in common]]
        majority = (labs.reshape(-1).bincount().max() / labs.numel()).item()
        out[other.variable_id] = {"mapped_accuracy": round(mapped_accuracy(preds, labs, seed), 4),
                                  "majority_rate": round(majority, 4)}
    return out


def gate_verdict(rates: dict, mode: str) -> dict:
    """{mode, rule, passed} for the selected gate. A gate with no measured output rate
    (skipped variable, or no row that steered) cannot pass any mode."""
    if rates.get("flip_rate") is None:
        return {"mode": mode, "rule": GATE_RULES[mode], "passed": False}
    output_ok = rates["flip_rate"] >= FLIP_GATE and rates["collateral_rate"] <= COLLATERAL_GATE
    passed = {"steered": rates["steered_rate"] >= FLIP_GATE, "output": output_ok,
              "both": rates["steered_rate"] >= FLIP_GATE and output_ok}[mode]
    return {"mode": mode, "rule": GATE_RULES[mode], "passed": passed}


def causal_gate(model, spec, var, layer: int, n_pairs: int, seed: int, mode: str) -> dict:
    """Directional patching at (layer, p) with position p's probe: push toward class
    b=(a+1)%K, score model outputs against the variable's DSL counterfactual replay.
    Records BOTH the probe-side (steered) and the model-side (flip) rate."""
    hook_name = f"blocks.{layer}.hook_resid_post"
    art = torch.load(os.path.join(tp.probe_dir(spec.name, var.variable_id),
                                  f"site_L{layer}.pt"), weights_only=False)
    dev = model.cfg.device
    Wp, bp = art["W_pos"].to(dev), art["b_pos"].to(dev)
    fitpos = art["positions"].tolist()
    declared = var.valid_positions().tolist()
    g = torch.Generator().manual_seed(int(seed) + 13)
    inputs = tp.sample_inputs(n_pairs, seed + 13)
    labs = var.labels(inputs)
    mid = [p for p in fitpos if 2 <= p <= 8] or fitpos
    ps = torch.tensor(mid)[torch.randint(len(mid), (n_pairs,), generator=g)]
    a_cls = labs[torch.arange(n_pairs), torch.tensor([declared.index(int(p)) for p in ps])]
    b_cls = (a_cls + 1) % var.n_classes
    cfs = [var.counterfactual(inputs[i], int(ps[i]), int(b_cls[i])) for i in range(n_pairs)]
    keep = [i for i, c in enumerate(cfs) if c is not None]   # per-row undefined counterfactuals drop out
    if len(keep) < max(8, n_pairs // 4):
        return {"skipped": f"counterfactual undefined on {n_pairs - len(keep)}/{n_pairs} rows",
                "gate": gate_verdict({}, mode)}
    inputs = inputs[keep]
    ps, a_cls, b_cls = ps[keep], a_cls[keep], b_cls[keep]
    n_pairs = len(keep)
    cf = torch.stack([cfs[i] for i in keep])
    outs = spec.task_fn(inputs)
    toks = torch.cat([inputs, outs], dim=1).to(dev)
    ans = torch.tensor(spec.answer_positions, device=dev)

    with torch.no_grad():
        base_logits, cache = model.run_with_cache(toks, names_filter=lambda n: n == hook_name)
        site = cache[hook_name]                                    # [n, seq, d] raw resid
    base_pred = base_logits[:, ans].argmax(-1).cpu()
    rows = torch.arange(n_pairs, device=dev)
    r = site[rows, ps.to(dev)]                                     # [n, d] raw resid at (l, p)
    jrow = torch.tensor([fitpos.index(int(p)) for p in ps], device=dev)
    Wrow, brow = Wp[jrow], bp[jrow]                                # [n, C, d], [n, C]
    read = lambda x: (torch.einsum("nd,ncd->nc", tp.ln(x), Wrow) + brow).argmax(-1)
    a_dev, b_dev = a_cls.to(dev), b_cls.to(dev)
    reads_a = (read(r) == a_dev).cpu()
    delta_dir = torch.nn.functional.normalize(Wrow[rows, b_dev] - Wrow[rows, a_dev], dim=-1)
    scale = r.norm(dim=-1, keepdim=True)

    steered_alpha = torch.full((n_pairs,), float("nan"))
    preds = base_pred.clone()
    for alpha in ALPHAS:
        delta = alpha * scale * delta_dir
        reads_b = (read(r + delta) == b_dev).cpu()
        todo = reads_b & steered_alpha.isnan()
        if not bool(todo.any()):
            continue
        full = torch.zeros_like(site)
        full[rows, ps.to(dev)] = delta

        def add_delta(resid, hook):
            return resid + full[:resid.shape[0]]

        with torch.no_grad():
            logits = model.run_with_hooks(toks, fwd_hooks=[(hook_name, add_delta)])
        preds[todo] = logits[:, ans].argmax(-1).cpu()[todo]
        steered_alpha[todo] = alpha

    steered = ~steered_alpha.isnan() & reads_a   # matched rows: probe read the source class pre-patch
    n_s = int(steered.sum())
    rec = {"n_pairs": n_pairs, "n_matched_steered": n_s, "layer": layer,
           "scoring": "teacher_forced_next_token",   # preds read off the DSL suffix, patched at (l, p)
           "base_model_seq_acc": round((base_pred == outs).all(dim=1).float().mean().item(), 4),
           "probe_reads_source_rate": round(reads_a.float().mean().item(), 4),
           "steered_rate": round(n_s / n_pairs, 4),
           "alpha_applied": {str(a): int((steered_alpha[steered] == a).sum()) for a in ALPHAS},
           "mean_alpha": round(steered_alpha[steered].mean().item(), 3) if n_s else None,
           "docs_gate": "flip >= 0.80 and collateral <= 0.05"}
    if n_s == 0:
        rec.update(flip_rate=None, suffix_consistent_rate=None, collateral_rate=None,
                   note="no matched+steered rows (probe never read source then target class)")
        rec["gate"] = gate_verdict(rec, mode)
        return rec
    sp, spred, scf, sbase = ps[steered], preds[steered], cf[steered], base_pred[steered]
    flip = torch.tensor([int(spred[i, sp[i]] == scf[i, sp[i]]) for i in range(n_s)])
    suffix = torch.tensor([int(bool((spred[i, sp[i]:] == scf[i, sp[i]:]).all())) for i in range(n_s)])
    collat = torch.tensor([int(bool((spred[i, :sp[i]] != sbase[i, :sp[i]]).any())) for i in range(n_s)])
    rec.update(flip_rate=round(flip.float().mean().item(), 4),
               suffix_consistent_rate=round(suffix.float().mean().item(), 4),
               collateral_rate=round(collat.float().mean().item(), 4))
    rec["gate"] = gate_verdict(rec, mode)
    return rec


def robustness(y: torch.Tensor, model, resids: dict, pos: torch.Tensor,
               n_classes: int, majority: float) -> dict:
    outs = []
    for seed, wd in ROBUSTNESS_FITS:
        per_layer, baseline = {}, None
        for key, hook in tp.sites_for(model, tp.RESID):
            _, _, accs = tp.fit_positions(resids[hook][:, pos], y, n_classes, seed=seed, wd=wd)
            mean = sum(accs) / len(accs)
            if key == "baseline":
                baseline = mean
            else:
                per_layer[int(hook.split(".")[1])] = mean
        outs.append(tp.verdict(per_layer, baseline, majority))
    return {"fits": [{"seed": s, "wd": w, "verdict": v, "first_decodable_layer": f}
                     for (s, w), (v, f) in zip(ROBUSTNESS_FITS, outs)],
            "agree": len(set(outs)) == 1}


def validate_variable(model, spec, cls: type, resids: dict, inputs: torch.Tensor,
                      seed: int, n_pairs: int, gate: str, layer_override=None) -> dict:
    scan_path = os.path.join(tp.probe_dir(spec.name, cls.variable_id), "site_scan.json")
    if not os.path.exists(scan_path):
        raise SystemExit(f"no site_scan.json for {spec.name}/{cls.variable_id}; "
                         f"run train_probes.py first")
    scan = json.load(open(scan_path))
    if "gen_verdict" not in scan:
        raise SystemExit(f"{spec.name}/{cls.variable_id}: site_scan.json predates the "
                         f"input+generation fit (no gen_verdict); re-run train_probes.py")
    layer, from_first = (layer_override, False) if layer_override is not None else eval_layer(scan)
    var = cls()
    declared = var.valid_positions().tolist()
    pos = torch.tensor(scan["positions"], dtype=torch.long)        # causally trimmed at fit time
    y = var.labels(inputs)[:, [declared.index(p) for p in pos.tolist()]]
    X = resids[f"blocks.{layer}.hook_resid_post"][:, pos]          # [N, P, d]

    report = {"model": spec.name, "variable": cls.variable_id, "eval_layer": layer,
              "eval_layer_is_first_decodable": from_first,
              "verdict": scan["verdict"], "gen_verdict": scan["gen_verdict"],
              "fit_regime": scan["regime"],
              "probe_accuracy": scan["per_layer_accuracy"][str(layer)],
              "permutation_control": {
                  "accuracy": round(permutation_control(X, y, cls.n_classes, seed), 4),
                  "chance_uniform": round(1 / cls.n_classes, 4),
                  "chance_majority": scan["majority_class_rate"]},
              "leakage": leakage_matrix(spec.name, cls, layer, resids, inputs, seed),
              "robustness": robustness(y, model, resids, pos, cls.n_classes,
                                       scan["majority_class_rate"])}
    if var.counterfactual(inputs[0], int(pos[len(pos) // 2]), 0) is None:
        report["causal_gate"] = {"skipped": "counterfactual undefined for this variable",
                                 "gate": gate_verdict({}, gate)}
    else:
        report["causal_gate"] = causal_gate(model, spec, var, layer, n_pairs, seed, gate)
    out = os.path.join(tp.probe_dir(spec.name, cls.variable_id), "validation.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--variables", nargs="*", default=None)
    ap.add_argument("--n", type=int, default=4096)
    ap.add_argument("--n-pairs", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--gate", choices=tuple(GATE_RULES), default="output",
                    help="which causal rate gate.passed enforces; 'output' is the documented one")
    ap.add_argument("--layer", type=int, default=None,
                    help="evaluate at this layer instead of the first-decodable/best layer")
    ap.add_argument("--out", default=tp.PROBES_ROOT,
                    help="probes root to read fits from / write validation.json to")
    args = ap.parse_args()
    tp.PROBES_ROOT = args.out

    spec = model_registry.get_spec(args.model)
    classes = ([tp.resolve_variable(args.model, n) for n in args.variables]
               if args.variables else varlib.for_model(args.model))
    if not classes:
        raise SystemExit(f"no ProbeVariables registered for {args.model!r}")
    model = model_registry.load_model(args.model, device=args.device)
    inputs = tp.sample_inputs(args.n, args.seed)
    resids = tp.cached_residuals(model, inputs, tp.RESID)   # input positions only (see docstring)
    for cls in classes:
        rep = validate_variable(model, spec, cls, resids, inputs, args.seed, args.n_pairs,
                                args.gate, layer_override=args.layer)
        print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
