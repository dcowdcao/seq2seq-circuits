#!/usr/bin/env python3
"""Router-imbalance census over every trained MoE checkpoint, joined to its circuits.

Reads blocks.{l}.mlp.hook_expert_indices on uniform random inputs and reports, per
(model, layer): expert shares, normalized entropy (1 = balanced, 0 = collapsed), the
dominant expert's share, effective expert count, and how much of the routing decision
position alone explains. Then joins each model's mean imbalance to its faithful
DomainCircuit sizes so imbalance can be correlated with circuit shape.

Nothing else in the repo reads these hooks; the EAP graph collapses every expert into a
single m{l} node, so circuit artifacts cannot distinguish experts on their own.

  python repo_tools/moe_routing_census.py --out repo_tools/moe_routing_census.json
"""
import argparse, glob, json, math, os, statistics as st, sys

PIPE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(PIPE, "models"), os.path.join(os.path.dirname(PIPE), "interp_tools")]

import torch
import model_registry

CKPTS = os.path.join(PIPE, "models", "trained_models_pt")
CIRCUITS = os.path.join(PIPE, "circuits_data")


def routing(name: str, n: int, seed: int) -> list[dict]:
    """Per-layer routing stats for one checkpoint."""
    model = model_registry.load_model(name, device="cpu")
    E, L = int(model.cfg.num_experts), int(model.cfg.n_layers)
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(0, 10, (n, 10), generator=g)
    hooks = {f"blocks.{l}.mlp.hook_expert_indices" for l in range(L)}
    with torch.no_grad():
        _, cache = model.run_with_cache(x, names_filter=lambda h: h in hooks)
    out = []
    for l in range(L):
        top1 = cache[f"blocks.{l}.mlp.hook_expert_indices"].reshape(n, 10, -1)[:, :, 0]
        share = (torch.bincount(top1.flatten(), minlength=E).float() / top1.numel())
        nz = share[share > 0]
        entropy = float(-(nz * nz.log()).sum() / math.log(E))
        modes = [int(top1[:, p].mode().values) for p in range(10)]
        locked = float(torch.stack([(top1[:, p] == modes[p]).float().mean() for p in range(10)]).mean())
        out.append({"layer": l, "shares": [round(float(v), 4) for v in share],
                    "entropy": round(entropy, 4), "max_share": round(float(share.max()), 4),
                    "effective_experts": int((share >= 0.01).sum()),
                    "position_locked": round(locked, 4), "modal_expert_per_position": modes})
    return out


def circuit_sizes(name: str) -> dict:
    """Faithful DomainCircuit sizes for one model, plus m-node incidence."""
    h, e, with_m, tot = [], [], 0, 0
    for p in glob.glob(os.path.join(CIRCUITS, name, "*", "smallest_circuit.json")):
        d = json.load(open(p))
        if d.get("mode") in ("differential", "suppression"):
            continue
        tot += 1
        if d.get("no_faithful_circuit"):
            continue
        h.append(d["n_heads"]); e.append(d["n_edges"])
        if any(str(x.get(k, "")).startswith("m") for x in (d.get("edges") or []) for k in ("parent", "child")):
            with_m += 1
    return {"n_domain_built": tot, "n_domain_faithful": len(h),
            "median_heads": st.median(h) if h else None,
            "median_edges": st.median(e) if e else None,
            "faithful_with_mlp_node": with_m}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=256, help="random inputs per model")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None, help="first N checkpoints per tier (smoke)")
    ap.add_argument("--out", default=os.path.join(PIPE, "repo_tools", "moe_routing_census.json"))
    a = ap.parse_args()

    records = []
    for tier in ("moe4", "moe8"):
        names = sorted(os.path.basename(p)[:-3] for p in glob.glob(os.path.join(CKPTS, f"*__{tier}.pt")))
        for i, name in enumerate(names[: a.limit]):
            layers = routing(name, a.n, a.seed)
            records.append({"model": name, "tier": tier, "layers": layers,
                            "mean_entropy": round(st.mean(x["entropy"] for x in layers), 4),
                            "mean_max_share": round(st.mean(x["max_share"] for x in layers), 4),
                            "mean_position_locked": round(st.mean(x["position_locked"] for x in layers), 4),
                            "circuits": circuit_sizes(name)})
            if (i + 1) % 25 == 0:
                print(f"  {tier}: {i + 1}/{len(names[: a.limit])}", flush=True)
    json.dump({"n_inputs": a.n, "seed": a.seed, "models": records}, open(a.out, "w"), indent=1)
    print(f"wrote {a.out}  ({len(records)} models)")


if __name__ == "__main__":
    main()
