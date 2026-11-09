#!/usr/bin/env python3
"""Fit LINEAR probes for a model's ProbeVariables on the LN'd residual stream.

One cached forward over the FULL sequence (uniform inputs + the DECODED output), then per
(variable, site, POSITION) a multinomial logistic probe: a single probe pooled across
positions provably hides position-rotating codes (pmaxv9_shadow 0.77 pooled vs 0.95+
per-position), so per-position fits carry the verdict. Linearity is load-bearing: no MLP
probes ever. Fits run on CPU: the problems are tiny and convex, so a GPU only adds
launch round-trips (measured 4.4x faster per fit on CPU).

Two position sets are scanned:
  INPUT   the causally-trimmed declared positions -> ``verdict`` (the reported reading:
          is the variable linearly present while the input is being read?)
  OUTPUT  input_len..n_ctx-1, one fit per (declared position, decode step) ->
          ``gen_verdict``. Without it the generation phase is unobserved, so a variable
          that only materialises while decoding is structurally undetectable and every
          ``absent`` verdict is an inference about a phase nobody measured.

Decode regime (``--decode``): ``free`` (default) feeds the model its own argmax — what
inference actually does; ``teacher`` feeds the DSL ground truth, which leaks the answer
into the residual being probed. ``decode_seq_acc`` records how far the two diverge (on an
accurate model they barely do). The output-position layer-0 ``resid_pre`` baseline is the
control for the leak either way: the emitted token IS in that baseline, so gen
decodability above it means "computed", not "read back the token just written".

Splits are on ROW index, never on flattened (row, position) examples. This only ever bit
the POOLED fit, which turns each row into P examples: its flattened holdout shared a row
(and often a label) with train for 100% of held-out items. The per-position fits are one
example per row by construction, so the verdict was never affected.

Positions whose label never varies in the sample are recorded (``constant_positions``)
and optionally dropped (``--drop-constant-positions``): a constant label is trivially
predictable and otherwise averages into the reported accuracy.

Declared positions are causally TRIMMED before fitting (variables.causal_trim): a
position whose labels move under suffix resampling cannot live in a causal residual
stream. Trimming that empties a variable fails loud — the variable is unanswerable.
Every variable's declaration is checked (variables.validate) before it is fitted.

Artifact contract (consumed by the finder plumbing) — probes_data/<model>/<variable>/:
  site_L{l}.pt   {"W": [n_classes, d_model], "b": [n_classes] (probe pooled across
                  positions — the finder readout contract),
                  "W_pos": [P, n_classes, d_model], "b_pos": [P, n_classes]
                  (per-position probes), "layer": l,
                  "hook": "blocks.{l}.hook_resid_post", "positions": LongTensor [P]
                  (causally trimmed), "n_classes": int,
                  "accuracy": float (position-mean per-position held-out acc),
                  "pooled_accuracy": float, "per_position_accuracy": {pos: acc},
                  "W_gen"/"b_gen": [P, n_classes, d_model]/[P, n_classes] (each declared
                  position's probe at its best-scoring generation position),
                  "gen_positions": LongTensor [Q], "gen_argmax_position": [P],
                  "per_gen_position_accuracy": {pos: {gen_pos: acc}},
                  "gen_accuracy": float, "normalize": "ln"}
  site_baseline.pt        same keys at "blocks.0.hook_resid_pre" (the layer-0 baseline)
  site_attnL{l}/mlpL{l}   the input-position keys only (no gen scan: P*Q fits) at
  site_headL{l}H{h}       hook_attn_out / hook_mlp_out (--sites components, default) and
                          per-head hook_z slices (--sites heads). Component sites never
                          carry the verdict; they localise it. Named off the site_L{l}
                          glob on purpose (eap.load_probe_artifact).
  site_scan.json          position-mean accuracy per layer + baseline for both position
                          sets, per-position detail, per-position majority rates,
                          declared/trimmed/constant positions, the regime block, and
                          verdict + first_decodable_layer (input) / gen_verdict (output)

Verdict (three-way, on position-mean accuracies of the per-position probes):
  trivial_from_input  the baseline probe alone decodes the variable (acc >= TRIVIAL_ACC
                      and clears the majority floor) — layer probes add nothing
  decodable           first layer whose probe error <= ERR_REDUCTION x baseline error
                      AND clears the majority floor; floor = halving the majority-class
                      error too (acc >= 1 - ERR_REDUCTION*(1-majority)), which stays
                      satisfiable at any majority rate, unlike a fixed additive margin
  absent              neither, at any layer (first_decodable_layer stays null)
``gen_verdict`` applies the same rule to the generation-position accuracies, where each
declared position takes its BEST decode step — an optimistic statistic, so a gen
``absent`` is the strong result. Its baseline is the layer-0 embedding of the model's own
emitted token, so the trivial case is reported as ``trivial_from_emitted_output``.

CLI: --model, --variables (default: all registered for the model; bridge{k} names are
created on demand via variables.bridge_variable), --n, --seed, --decode {free,teacher},
--sites {resid,components,heads}, --drop-constant-positions, --out (artifact root,
default probes_data/), --site-scan (run all variables and print the accuracy map).
"""
import argparse, json, os, re, sys
from dataclasses import dataclass

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PIPELINE_ROOT = os.path.dirname(_THIS_DIR)
for _p in (_THIS_DIR, os.path.join(_PIPELINE_ROOT, "models")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
import torch.nn.functional as F

import algorithms
import model_registry
import variables as varlib

PROBES_ROOT = os.path.join(_PIPELINE_ROOT, "probes_data")
TRIVIAL_ACC = 0.9        # trivial_from_input: the baseline alone decodes this well
ERR_REDUCTION = 0.5      # decodable: probe error must halve baseline AND majority error
HOLDOUT_FRAC = 0.25
LN_EPS = 1e-5
RESID, COMPONENTS, HEADS = "resid", "components", "heads"
GEN_TRIVIAL = "trivial_from_emitted_output"   # the gen-space reading of trivial_from_input


@dataclass(frozen=True, kw_only=True)
class FitConfig:
    seed: int = 0
    decode: str = "free"          # free = the model's own argmax; teacher = DSL truth
    sites: str = COMPONENTS
    drop_constant: bool = False


def probe_dir(model: str, variable: str) -> str:
    return os.path.join(PROBES_ROOT, model, variable)


def ln(x: torch.Tensor, eps: float = LN_EPS) -> torch.Tensor:
    """LayerNormPre: center + scale (what the next block / unembed actually reads)."""
    x = x - x.mean(-1, keepdim=True)
    return x / (x.pow(2).mean(-1, keepdim=True) + eps).sqrt()


def sample_inputs(n: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(int(seed))
    return torch.randint(0, 10, (n, varlib.INPUT_LEN), generator=g)


def decode_outputs(model, spec, inputs: torch.Tensor, batch: int = 1024) -> torch.Tensor:
    """Free-running greedy decode [N, out_len]: the model's own argmax, fed back."""
    out_len = spec.n_ctx - spec.input_len
    toks = torch.zeros(inputs.shape[0], spec.n_ctx, dtype=torch.long)
    toks[:, :spec.input_len] = inputs
    with torch.no_grad():
        for k in range(out_len):
            end = spec.input_len + k
            for i in range(0, toks.shape[0], batch):
                logits = model(toks[i:i + batch, :end].to(model.cfg.device))
                toks[i:i + batch, end] = logits[:, spec.answer_positions[k]].argmax(-1).cpu()
    return toks[:, spec.input_len:]


def full_sequence(model, spec, inputs: torch.Tensor, decode: str) -> tuple[torch.Tensor, float]:
    """(tokens [N, n_ctx] = inputs + decoded output, free-run exact-match rate). The free
    decode runs in both regimes so the divergence is always recorded."""
    truth = spec.task_fn(inputs)
    free = decode_outputs(model, spec, inputs)
    out = free if decode == "free" else truth
    return torch.cat([inputs, out], dim=1), (free == truth).all(1).float().mean().item()


def sites_for(model, level: str = COMPONENTS) -> list[tuple[str, str]]:
    """(site_key, hook) pairs. ``resid`` = the VERDICT sites (layer-0 baseline +
    resid_post per layer); ``components`` adds attn_out/mlp_out, ``heads`` adds per-head
    hook_z slices (key ``<hook>[h]``). Component/head sites localise a verdict layer to
    attention vs MLP vs head; they never carry the verdict themselves, and their ``ln`` is
    a probe-side rescale (the model LNs the residual stream, not a block's output)."""
    sites = [("baseline", "blocks.0.hook_resid_pre")] + \
            [(f"L{l}", f"blocks.{l}.hook_resid_post") for l in range(model.cfg.n_layers)]
    if level == RESID:
        return sites
    for l in range(model.cfg.n_layers):
        sites.append((f"attnL{l}", f"blocks.{l}.hook_attn_out"))
        if not model.cfg.attn_only:
            sites.append((f"mlpL{l}", f"blocks.{l}.hook_mlp_out"))
    if level == HEADS:
        sites += [(f"headL{l}H{h}", f"blocks.{l}.attn.hook_z[{h}]")
                  for l in range(model.cfg.n_layers) for h in range(model.cfg.n_heads)]
    return sites


def hook_head(key: str) -> tuple[str, int | None]:
    """``blocks.0.attn.hook_z[3]`` -> (hook, 3); a plain hook name -> (hook, None)."""
    m = re.fullmatch(r"(.+)\[(\d+)\]", key)
    return (m.group(1), int(m.group(2))) if m else (key, None)


def cached_residuals(model, toks: torch.Tensor, level: str = COMPONENTS,
                     batch: int = 1024) -> dict:
    """{site hook key: LN'd activation [N, n_ctx, d]} ON CPU, one batched cached forward."""
    keys = [h for _, h in sites_for(model, level)]
    names = {hook_head(k)[0] for k in keys}
    out = {k: [] for k in keys}
    with torch.no_grad():
        for i in range(0, toks.shape[0], batch):
            _, cache = model.run_with_cache(toks[i:i + batch].to(model.cfg.device),
                                            names_filter=lambda n: n in names)
            for k in keys:
                name, head = hook_head(k)
                a = cache[name].float()
                out[k].append(ln(a if head is None else a[:, :, head]).cpu())
    return {k: torch.cat(v) for k, v in out.items()}


def split(n_rows: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Train/holdout ROW indices. Splitting rows rather than flattened (row, position)
    examples is what stops a POOLED fit from testing on a row it trained on."""
    g = torch.Generator().manual_seed(int(seed))
    perm = torch.randperm(n_rows, generator=g)
    n_te = int(HOLDOUT_FRAC * n_rows)
    return perm[n_te:], perm[:n_te]


def fit_probe(X: torch.Tensor, y: torch.Tensor, n_classes: int, seed: int = 0,
              wd: float = 1e-3) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Multinomial logistic regression via LBFGS on CPU -> (W [C,d], b [C], held-out acc).
    X [N, d], or [N, E, d] with y [N, E] to pool E examples per row into one probe (the
    finder readout contract). The holdout is a ROW split, so a pooled fit can't test on a
    row it trained on. Deterministic given (X, y, seed): zero init, full batch, no sampling.
    CPU on purpose: 1e5-FLOP convex problems, so a GPU only adds launch round-trips."""
    X, y = X.cpu(), y.cpu()
    tr, te = split(X.shape[0], seed)
    d = X.shape[-1]
    Xtr, ytr = X[tr].reshape(-1, d), y[tr].reshape(-1)
    W = torch.zeros(n_classes, d, requires_grad=True)
    b = torch.zeros(n_classes, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], max_iter=200, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(Xtr @ W.T + b, ytr) + wd * W.pow(2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        acc = ((X[te].reshape(-1, d) @ W.T + b).argmax(-1) == y[te].reshape(-1)).float().mean()
    return W.detach(), b.detach(), acc.item()


def fit_positions(X: torch.Tensor, y: torch.Tensor, n_classes: int, seed: int = 0,
                  wd: float = 1e-3) -> tuple[torch.Tensor, torch.Tensor, list[float]]:
    """Independent probe per position: X [N,P,d], y [N,P] -> (W [P,C,d], b [P,C], accs [P])."""
    fits = [fit_probe(X[:, j], y[:, j], n_classes, seed=seed, wd=wd) for j in range(X.shape[1])]
    return (torch.stack([W for W, _, _ in fits]), torch.stack([b for _, b, _ in fits]),
            [acc for _, _, acc in fits])


def fit_generation(X: torch.Tensor, y: torch.Tensor, n_classes: int, seed: int = 0,
                   wd: float = 1e-3) -> tuple[torch.Tensor, torch.Tensor, list[int], list[list[float]]]:
    """Every (declared position, generation position) pair: X [N,Q,d] at the output
    positions, y [N,P] -> (W [P,C,d], b [P,C]) at each declared position's BEST decode
    step, that step's index [P], and the full [P][Q] accuracy matrix. P*Q fits, which is
    why this runs at the verdict (resid) sites only."""
    Ws, bs, argmax, accs = [], [], [], []
    for j in range(y.shape[1]):
        fits = [fit_probe(X[:, q], y[:, j], n_classes, seed=seed, wd=wd)
                for q in range(X.shape[1])]
        row = [a for _, _, a in fits]
        best = max(range(len(row)), key=row.__getitem__)
        Ws.append(fits[best][0])
        bs.append(fits[best][1])
        argmax.append(best)
        accs.append(row)
    return torch.stack(Ws), torch.stack(bs), argmax, accs


def constant_columns(y: torch.Tensor) -> list[int]:
    """Columns of y [N, P] whose label never varies in the sample (trivially predictable)."""
    return [j for j in range(y.shape[1]) if bool((y[:, j] == y[0, j]).all())]


def resolve_variable(model: str, name: str) -> type:
    if m := re.fullmatch(r"bridge(\d+)", name):
        return varlib.bridge_variable(model, int(m.group(1)))
    return varlib.get(model, name)


def verdict(per_layer: dict, baseline: float, majority: float) -> tuple[str, int | None]:
    """Three-way gate on position-mean accuracies (bars documented in the module docstring)."""
    floor = 1 - ERR_REDUCTION * (1 - majority)
    if baseline >= TRIVIAL_ACC and baseline >= floor:
        return "trivial_from_input", None
    hits = [l for l, acc in sorted(per_layer.items())
            if 1 - acc <= ERR_REDUCTION * (1 - baseline) and acc >= floor]
    return ("decodable", hits[0]) if hits else ("absent", None)


def train_variable(model, model_name: str, cls: type, resids: dict, inputs: torch.Tensor,
                   gen_pos: torch.Tensor, decode_acc: float, cfg: FitConfig) -> dict:
    """Fit every site at the input AND generation positions; write site_*.pt + site_scan.json."""
    varlib.validate(cls)
    var = cls()
    pos, cols, dropped = varlib.causal_trim(var)
    y = var.labels(inputs)[:, cols]                                # [N, P] causally-held only
    constant = [int(pos[j]) for j in constant_columns(y)]
    if cfg.drop_constant and constant:
        keep = [j for j, p in enumerate(pos.tolist()) if p not in constant]
        if not keep:
            raise varlib.ProbeError(f"{cls.variable_id}: every causally-held position has a "
                                    "constant label in this sample — nothing to fit")
        pos, y = pos[keep], y[:, keep]
    out_dir = probe_dir(model_name, cls.variable_id)
    os.makedirs(out_dir, exist_ok=True)
    resid_keys = {k for k, _ in sites_for(model, RESID)}
    per_layer, per_layer_pos, pooled, baseline_acc = {}, {}, {}, None
    gen_layer, gen_pos_acc, gen_baseline, comps = {}, {}, None, {}
    for key, hook in sites_for(model, cfg.sites):
        Xin = resids[hook][:, pos]
        W, b, pool_acc = fit_probe(Xin, y, cls.n_classes, seed=cfg.seed)
        W_pos, b_pos, accs = fit_positions(Xin, y, cls.n_classes, seed=cfg.seed)
        mean_acc = sum(accs) / len(accs)
        per_pos = {int(p): round(a, 4) for p, a in zip(pos.tolist(), accs)}
        art = {"W": W, "b": b, "W_pos": W_pos, "b_pos": b_pos,
               "layer": int(hook.split(".")[1]), "head": hook_head(hook)[1], "hook": hook,
               "positions": pos.clone(), "n_classes": cls.n_classes, "accuracy": mean_acc,
               "pooled_accuracy": pool_acc, "per_position_accuracy": per_pos,
               "normalize": "ln", "variable_id": cls.variable_id, "model": model_name,
               "n": int(inputs.shape[0]), "seed": cfg.seed, "decode": cfg.decode}
        if key in resid_keys:            # the generation scan is P*Q fits: verdict sites only
            W_gen, b_gen, g_argmax, g_accs = fit_generation(resids[hook][:, gen_pos], y,
                                                            cls.n_classes, seed=cfg.seed)
            gen_best = [max(row) for row in g_accs]
            gen_acc = sum(gen_best) / len(gen_best)
            per_gen = {str(int(p)): {str(int(q)): round(a, 4)
                                     for q, a in zip(gen_pos.tolist(), row)}
                       for p, row in zip(pos.tolist(), g_accs)}
            art.update(W_gen=W_gen, b_gen=b_gen, gen_positions=gen_pos.clone(),
                       gen_argmax_position=[int(gen_pos[q]) for q in g_argmax],
                       per_gen_position_accuracy=per_gen, gen_accuracy=gen_acc)
            gen_pos_acc[key] = per_gen
            if key == "baseline":
                baseline_acc, gen_baseline = mean_acc, gen_acc
            else:
                per_layer[art["layer"]], gen_layer[art["layer"]] = mean_acc, gen_acc
        else:
            comps[key] = round(mean_acc, 4)
        torch.save(art, os.path.join(out_dir, f"site_{key}.pt"))
        per_layer_pos[key], pooled[key] = per_pos, round(pool_acc, 4)
    maj_pos = {int(p): (y[:, j].bincount().max() / y.shape[0]).item()
               for j, p in enumerate(pos.tolist())}
    majority = sum(maj_pos.values()) / len(maj_pos)
    v, first = verdict(per_layer, baseline_acc, majority)
    gv, gfirst = verdict(gen_layer, gen_baseline, majority)
    gv = GEN_TRIVIAL if gv == "trivial_from_input" else gv   # gen baseline = the emitted token
    scan = {"model": model_name, "variable": cls.variable_id, "n_classes": cls.n_classes,
            "n": int(inputs.shape[0]), "seed": cfg.seed,
            "regime": {"split": "row_grouped", "decode": cfg.decode, "fit_device": "cpu",
                       "positions": "input+generation", "sites": cfg.sites,
                       "constant_positions": "dropped" if cfg.drop_constant else "recorded"},
            "decode_seq_acc": round(decode_acc, 4),
            "declared_positions": var.valid_positions().tolist(),
            "positions": pos.tolist(), "trimmed_positions": dropped,
            "generation_positions": gen_pos.tolist(),
            "constant_positions": constant,
            "majority_class_rate": round(majority, 4),
            "per_position_majority": {str(p): round(r, 4) for p, r in maj_pos.items()},
            "baseline_accuracy": round(baseline_acc, 4),
            "per_layer_accuracy": {str(l): round(a, 4) for l, a in sorted(per_layer.items())},
            "per_position_accuracy": per_layer_pos,
            "per_component_accuracy": comps,
            "pooled_accuracy": pooled,
            "gen_baseline_accuracy": round(gen_baseline, 4),
            "per_layer_gen_accuracy": {str(l): round(a, 4) for l, a in sorted(gen_layer.items())},
            "per_gen_position_accuracy": gen_pos_acc,
            "verdict": v, "first_decodable_layer": first,
            "gen_verdict": gv, "first_gen_decodable_layer": gfirst}
    with open(os.path.join(out_dir, "site_scan.json"), "w") as f:
        json.dump(scan, f, indent=2)
    return scan


def print_scan_table(scans: list, n_layers: int) -> None:
    cols = ["base"] + [f"L{l}" for l in range(n_layers)]
    width = max(len(s["variable"]) for s in scans)
    print(f"\n{'variable':<{width}}  {'  '.join(f'{c:>6}' for c in cols)}  first  chance  "
          f"trimmed  const  verdict  |  genbase  genbest  gen_verdict")
    for s in scans:
        accs = [s["baseline_accuracy"]] + [s["per_layer_accuracy"][str(l)] for l in range(n_layers)]
        fd, gen = s["first_decodable_layer"], s["per_layer_gen_accuracy"]
        print(f"{s['variable']:<{width}}  {'  '.join(f'{a:6.3f}' for a in accs)}  "
              f"{('L' + str(fd)) if fd is not None else 'none':>5}  {s['majority_class_rate']:.3f}  "
              f"{len(s['trimmed_positions']):>7}  {len(s['constant_positions']):>5}  "
              f"{s['verdict']:<8}  |  {s['gen_baseline_accuracy']:7.3f}  "
              f"{max(gen.values()):7.3f}  {s['gen_verdict']}")


def main() -> None:
    global PROBES_ROOT
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--variables", nargs="*", default=None,
                    help="variable_ids (default: all registered for the model)")
    ap.add_argument("--n", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--decode", choices=("free", "teacher"), default="free",
                    help="output tokens fed to the probed forward (default: the model's own argmax)")
    ap.add_argument("--sites", choices=(RESID, COMPONENTS, HEADS), default=COMPONENTS,
                    help="which activations to probe (default: + attn_out/mlp_out per layer)")
    ap.add_argument("--drop-constant-positions", action="store_true",
                    help="exclude positions whose label is constant in the sample from the fits")
    ap.add_argument("--out", default=PROBES_ROOT,
                    help="artifact root (default: probes_data/)")
    ap.add_argument("--site-scan", action="store_true",
                    help="run ALL variables for the model and print the accuracy map")
    args = ap.parse_args()
    PROBES_ROOT = args.out
    cfg = FitConfig(seed=args.seed, decode=args.decode, sites=args.sites,
                    drop_constant=args.drop_constant_positions)

    base = algorithms.base_algorithm_id(args.model)          # __<variant> probes the base's vars
    if base not in model_registry.all_models():
        raise SystemExit(f"model {args.model!r} (base {base!r}) not in models/models.txt; "
                         f"know {model_registry.all_models()}")
    names = args.variables
    if args.site_scan or names is None:
        classes = varlib.for_model(args.model)
        if not classes:
            raise SystemExit(f"no ProbeVariables registered for {args.model!r}")
    else:
        classes = [resolve_variable(args.model, n) for n in names]

    spec = model_registry.get_spec(args.model)
    if spec.input_len != varlib.INPUT_LEN:
        raise SystemExit(f"{args.model}: input_len {spec.input_len} != variables.INPUT_LEN "
                         f"{varlib.INPUT_LEN}; the label positions would not line up")
    model = model_registry.load_model(args.model, device=args.device)
    inputs = sample_inputs(args.n, args.seed)
    toks, decode_acc = full_sequence(model, spec, inputs, args.decode)
    resids = cached_residuals(model, toks, args.sites)
    gen_pos = torch.arange(spec.input_len, spec.n_ctx)
    scans = [train_variable(model, args.model, cls, resids, inputs, gen_pos, decode_acc, cfg)
             for cls in classes]
    print_scan_table(scans, model.cfg.n_layers)
    print(f"\nartifacts under {os.path.join(PROBES_ROOT, args.model)}/")


if __name__ == "__main__":
    main()
