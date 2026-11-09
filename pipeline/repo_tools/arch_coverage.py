#!/usr/bin/env python3
"""Regenerate the tables in docs/reference/circuit_types_across_architectures.md: which circuit types
exist at which architecture, why, and the checkpoint-quality control that rewrites the rates.

Sources:
  models/taxonomy.json          built/faithful verdicts per (model, circuit) and probe scan
  models/trained_models_pt/*.pt realized geometry + the eval accuracy recorded at promotion
  models/trained_models_pt_grid raw size-grid attempts (pre-promotion), for never-promoted models
  models/models.txt             run-list, for the mux universe
  task_router/<model>/           suppression's router-diagnostic prerequisite

The checkpoint stores (trained_models_pt/, trained_models_pt_grid/) are the full model zoo,
which is not shipped in this repository; the quality tables cover only the checkpoints
present locally.

Usage:
    python arch_coverage.py                 # every section
    python arch_coverage.py --section c1     # one section
"""
import argparse, json, statistics as st
from collections import Counter, defaultdict
from pathlib import Path

import torch

PIPELINE = Path(__file__).parents[1]
TAXONOMY = PIPELINE / "models/taxonomy.json"
CKPTS = PIPELINE / "models/trained_models_pt"
GRID = PIPELINE / "models/trained_models_pt_grid"

ARCHES = ["small", "large", "wide", "deep", "xl", "mlp", "moe4", "moe8"]
#: canonical (n_layers, n_heads, d_model) per arch label; a ckpt off this escalated (UPGRADE_LADDER).
CANON = {"small": (2, 4, 64), "large": (4, 8, 128), "wide": (4, 16, 256), "deep": (8, 8, 128),
         "xl": (8, 16, 256), "mlp": (4, 8, 128), "moe4": (4, 8, 128), "moe8": (4, 8, 128)}
GATE = 0.95   # run_model_pipeline.SEQ_ACC_GATE


def arch(model: str) -> str:
    return model.split("__")[1] if "__" in model else "large"


def base(model: str) -> str:
    return model.split("__")[0]


def checkpoints() -> dict:
    """name -> {"acc": recorded eval accuracy or None, "geom": (L, H, d_model)}."""
    out = {}
    for p in sorted(CKPTS.glob("*.pt")):
        d = torch.load(p, map_location="cpu", weights_only=False, mmap=True)
        c = d["config"]
        out[p.stem] = {"acc": (d.get("metrics", {}).get("eval") or {}).get("accuracy"),
                       "geom": (c["n_layers"], c["n_heads"], c["d_model"])}
    return out


def built(tax: dict, section: str) -> list:
    """Rows the pipeline actually searched (winner | no_faithful), tagged with arch."""
    return [r for r in tax[section].values() if r["verdict"] in ("winner", "no_faithful")]


def acc(ck: dict, model: str) -> float | None:
    """Eval accuracy recorded at promotion; None for legacy checkpoints that stored no metric."""
    return ck[model]["acc"] if model in ck else None


def gate_ok(ck: dict, model: str) -> bool:
    a = acc(ck, model)
    return a is not None and a >= GATE


def below(ck: dict, model: str, t: float) -> bool:
    """Recorded accuracy strictly below t. False when no metric was stored — never assume."""
    a = acc(ck, model)
    return a is not None and a < t


def rate(rows: list) -> str:
    if not rows:
        return f"{'-':>16}"
    w = sum(1 for r in rows if r["verdict"] == "winner")
    return f"{w:4}/{len(rows):4} {w/len(rows):.3f}"


def paired(rows: list, ck: dict, require_gate: bool) -> tuple:
    """(definitions present at all 8 arches, per-arch faithful counts) — the like-for-like view."""
    by = defaultdict(dict)
    for r in rows:
        by[(base(r["model"]), r["circuit"])][arch(r["model"])] = r
    full = [v for v in by.values() if all(a in v for a in ARCHES)
            and (not require_gate or all(gate_ok(ck, v[a]["model"]) for a in ARCHES))]
    return full, {a: sum(1 for v in full if v[a]["verdict"] == "winner") for a in ARCHES}


def head(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def sec_presence(tax: dict, ck: dict) -> None:
    head("1. Presence map — where each type is built, and the universe it is drawn from")
    for s in ("domain", "contrast", "suppression"):
        rows = built(tax, s)
        defs = {(base(r["model"]), r["circuit"]) for r in rows}
        at = {a for a in ARCHES if any(arch(r["model"]) == a for r in rows)}
        fa = {a for a in ARCHES
              if any(arch(r["model"]) == a and r["verdict"] == "winner" for r in rows)}
        n_arch = Counter(sum(1 for a in ARCHES
                             if any(arch(r["model"]) == a and (base(r["model"]), r["circuit"]) == d
                                    for r in rows)) for d in defs)
        print(f"  {s:12} {len(defs):4} defs / {len({base(r['model']) for r in rows}):3} models  "
              f"built@{len(at)} arches  faithful@{sorted(fa)}")
        print(f"  {'':12} defs by #arches built: {dict(sorted(n_arch.items()))}")
    pr = list(tax["probes"].values())
    pairs = {(base(r["model"]), r["variable"]) for r in pr}
    brid = {p for p in pairs if p[1].startswith("bridge")}
    print(f"  {'probe':12} {len(pairs):4} pairs / {len({base(r['model']) for r in pr}):3} models  "
          f"(of which bridge: {len(brid)} over {len({b for b, _ in brid})} composites)")


def sec_quality(tax: dict, ck: dict) -> None:
    head("2. Checkpoint-quality control — promoted checkpoints that failed the trainer's own gate")
    rows_on = Counter()
    for rows in [built(tax, s) for s in ("domain", "contrast", "suppression")] + [tax["probes"].values()]:
        for r in rows:
            if below(ck, r["model"], GATE):
                rows_on[arch(r["model"])] += 1
    print(f"  {'arch':6} {'ckpts':>6} {'acc<0.95':>9} {'acc<0.50':>9} {'no metric':>10} "
          f"{'rows built on them':>19} {'off-geometry':>13}")
    for a in ARCHES:
        v = [ck[k] for k in ck if arch(k) == a]
        accs = [x["acc"] for x in v if x["acc"] is not None]
        print(f"  {a:6} {len(v):6} {sum(1 for x in accs if x < GATE):9} "
              f"{sum(1 for x in accs if x < 0.50):9} {len(v) - len(accs):10} {rows_on[a]:19} "
              f"{sum(1 for x in v if x['geom'] != CANON[a]):13}")
    print(f"\n  total rows on sub-gate checkpoints: {sum(rows_on.values())}")
    grid = {p.stem for p in GRID.glob("*.pt")}
    attn = [a for a in ARCHES if a not in ("mlp", "moe4", "moe8")]
    broken = sorted({base(k) for k in ck
                     if arch(k) in ("mlp", "moe4", "moe8") and below(ck, k, 0.50)})
    print("\n  models with a sub-0.50 feedforward checkpoint, against the attn-only size grid.")
    print("  The grid store is one run's raw output, so 'attempts' is a lower bound.")
    print(f"  {'base':30} {'attn-only grid attempts':26} {'attn-only promoted':20}")
    dead = []
    for b in broken:
        att = [a for a in attn if (b if a == "large" else f"{b}__{a}") in grid]
        pro = [a for a in attn if (b if a == "large" else f"{b}__{a}") in ck]
        if len(att) == len(attn) and not pro:
            dead.append(b)
        print(f"  {b[:30]:30} {','.join(att) or '(none)':26} {','.join(pro) or '(none)':20}")
    print(f"\n  untrainable at every architecture (all {len(attn)} attn-only sizes attempted, none "
          f"promoted, feedforward < 0.50): {len(dead)} models\n  {dead}")


def _cov_row(a: str, ck: dict, rows: list) -> str:
    mine = [r for r in rows if arch(r["model"]) == a]
    good = [r for r in mine if gate_ok(ck, r["model"])]
    n_ck = sum(1 for k in ck if arch(k) == a)
    return (f"  {a:6} ckpts={n_ck:4} defs={len(mine):4} "
            f"all={rate(mine)}  gate-passing={rate(good)}")


def sec_c0(tax: dict, ck: dict) -> None:
    head("3. C0 domain — built wherever the model trained; existence is arch-independent")
    rows = built(tax, "domain")
    for a in ARCHES:
        print(_cov_row(a, ck, rows))
    for require in (False, True):
        full, f = paired(rows, ck, require)
        print(f"  paired over {len(full)} defs built at all 8 arches"
              f"{' with every ckpt >= 0.95' if require else ''}: "
              + " ".join(f"{a}={f[a]}" for a in ARCHES))
    w = [r for r in rows if r["verdict"] == "winner"]
    full, _ = paired(rows, ck, True)
    won = [v for v in full if all(v[a]["verdict"] == "winner" for a in ARCHES)]
    #: h_frac counts attention heads only; node_frac is what selection_score uses (heads + MLPs),
    #: so on the feedforward tiers a circuit can look compact in h_frac and keep the whole stack.
    print(f"\n  compactness of the winner — median node fraction (heads+MLPs, the quantity "
          f"selection_score uses) and median head fraction, all winners then the {len(won)} "
          f"defs faithful at all 8:")
    for a in ARCHES:
        nf = [r["node_frac"] for r in w if arch(r["model"]) == a and r["node_frac"] is not None]
        hf = [r["h_frac"] for r in w if arch(r["model"]) == a and r["h_frac"] is not None]
        pn = [x[a]["node_frac"] for x in won if x[a]["node_frac"] is not None]
        allh = sum(1 for r in w if arch(r["model"]) == a and r["h_frac"] == 1.0)
        alln = sum(1 for r in w if arch(r["model"]) == a and r["node_frac"] == 1.0)
        print(f"  {a:6} node all={st.median(nf):.3f} paired={st.median(pn):.3f} | "
              f"head all={st.median(hf):.3f}  keeps every head: {allh}  every node: {alln}")


def sec_c1(tax: dict, ck: dict) -> None:
    head("4. C1 contrast — presence tracks C0; findability collapses on wide/deep/xl")
    rows = built(tax, "contrast")
    for a in ARCHES:
        print(_cov_row(a, ck, rows))
    for require in (False, True):
        full, f = paired(rows, ck, require)
        print(f"  paired over {len(full)} defs{' (gated)' if require else ''}: "
              + " ".join(f"{a}={f[a]}" for a in ARCHES))
    print("\n  why the unfaithful rows failed (faithful base vs cascade):")
    for a in ARCHES:
        nf = [r for r in rows if arch(r["model"]) == a and r["verdict"] == "no_faithful"]
        if not nf:
            continue
        su = [r["s_union"] for r in nf if r["s_union"] is not None]
        print(f"  {a:6} no_faithful={len(nf):3} unfaithful_base={sum(1 for r in nf if r['base_faithful'] is False):3} "
              f"empty_fringe={sum(1 for r in nf if r['n_fringe_edges'] == 0):3} "
              f"median s_union={st.median(su) if su else float('nan'):.3f}")
    print("\n  necessity of the reported fringe (paired winners):")
    full, _ = paired(rows, ck, True)
    for a in ARCHES:
        w = [v[a] for v in full if v[a]["verdict"] == "winner"]
        nec = sum(1 for r in w if r["necessity_class"] == "necessary")
        print(f"  {a:6} necessary={nec:3}/{len(w):3} ({nec/len(w):.2f})")


def sec_c3(tax: dict, ck: dict) -> None:
    head("5. C3 suppression — mux-only, and the mux cohort was trained at four arches")
    #: base ids only — run-list entries carry __arch suffixes that must not be counted as `large`.
    mux = sorted({base(l.split("#")[0].strip())
                  for l in (PIPELINE / "models/models.txt").read_text().splitlines()
                  if l.split("#")[0].strip().startswith("mux")})
    rows = built(tax, "suppression")
    router = defaultdict(set)
    for p in (PIPELINE / "task_router").glob("*/diagnostic.json"):
        router[arch(p.parts[-2])].add(base(p.parts[-2]))
    print(f"  run-list mux models: {len(mux)}")
    print(f"  {'arch':6} {'mux ckpts':>9} {'router diag':>11} {'C3 models':>9} {'defs':>5} "
          f"{'all':>16} {'gate-passing':>16}")
    for a in ARCHES:
        have = sum(1 for m in mux if (m if a == "large" else f"{m}__{a}") in ck)
        mine = [r for r in rows if arch(r["model"]) == a]
        good = [r for r in mine if gate_ok(ck, r["model"])]
        print(f"  {a:6} {have:9} {len(router[a]):11} "
              f"{len({base(r['model']) for r in mine}):9} {len(mine):5} {rate(mine)} {rate(good)}")
    print("\n  mux checkpoint accuracy per arch (blank = not trained):")
    for m in sorted({base(r["model"]) for r in rows}):
        cells = []
        for a in ARCHES:
            k = m if a == "large" else f"{m}__{a}"
            cells.append(f"{ck[k]['acc']:.3f}" if k in ck and ck[k]["acc"] is not None else "  -  ")
        print(f"  {m[:52]:52} " + " ".join(cells))


def sec_c2(tax: dict, ck: dict) -> None:
    head("6. C2 probe — non-default stage; decodability halves on feedforward blocks")
    pr = list(tax["probes"].values())
    declared = {base(r["model"]) for r in pr}
    print(f"  {'arch':6} {'scans':>6} {'of models w/ ckpt':>18} {'decodable':>9} {'absent':>7} "
          f"{'rate':>6} {'gated rate':>11}")
    for a in ARCHES:
        mine = [r for r in pr if arch(r["model"]) == a]
        good = [r for r in mine if gate_ok(ck, r["model"])]
        have = sum(1 for b in declared if (b if a == "large" else f"{b}__{a}") in ck)

        def sh(rs):
            c = Counter(r["verdict"] for r in rs)
            n = c["decodable"] + c["absent"]
            return (c["decodable"], c["absent"], c["decodable"] / n if n else float("nan"))
        d0, a0, r0 = sh(mine)
        _, _, r1 = sh(good)
        print(f"  {a:6} {len(mine):6} {have:18} {d0:9} {a0:7} {r0:6.2f} {r1:11.2f}")
    print("\n  paired against large, both checkpoints gate-passing:")
    by = defaultdict(dict)
    for r in pr:
        by[(base(r["model"]), r["variable"])][arch(r["model"])] = r
    for a in ARCHES:
        if a == "large":
            continue
        p = [(v["large"], v[a]) for v in by.values() if "large" in v and a in v
             and gate_ok(ck, v["large"]["model"]) and gate_ok(ck, v[a]["model"])]
        lost = sum(1 for x, y in p if x["verdict"] == "decodable" and y["verdict"] == "absent")
        gain = sum(1 for x, y in p if x["verdict"] == "absent" and y["verdict"] == "decodable")
        print(f"  large vs {a:5} n={len(p):3} decodable->absent={lost:3} absent->decodable={gain:3}")
    print(f"\n  probe-metric circuits built, any arch: "
          f"{sum(len(r['probe_circuits']) for r in pr)}")


def sec_bridge(tax: dict, ck: dict) -> None:
    head("7. C2 bridge — composites only; absent at xl because no composite has an xl checkpoint")
    br = [r for r in tax["probes"].values() if r["variable"].startswith("bridge")]
    comps = sorted({base(r["model"]) for r in br})
    print(f"  composites: {len(comps)}")
    print(f"  {'arch':6} {'scanned':>7} {'composites w/ ckpt':>19}")
    for a in ARCHES:
        n = len({(base(r["model"]), r["variable"]) for r in br if arch(r["model"]) == a})
        have = sum(1 for b in comps if (b if a == "large" else f"{b}__{a}") in ck)
        print(f"  {a:6} {n:7} {have:19}")
    print(f"\n  {'composite':30} " + " ".join(f"{a[:5]:>6}" for a in ARCHES))
    for b in comps:
        cells = []
        for a in ARCHES:
            mine = [r for r in br if base(r["model"]) == b and arch(r["model"]) == a]
            cells.append("/".join(str(r["first_decodable_layer"]) if r["verdict"] == "decodable"
                                  else "abs" for r in mine) if mine else "-")
        print(f"  {b[:30]:30} " + " ".join(f"{c:>6}" for c in cells))


def sec_c4(tax: dict, ck: dict) -> None:
    head("8. C4 positional — three pilot models, trained at large only")
    pilots = sorted({p.parts[-4] for p in PIPELINE.glob("circuits_data/*/*/circuits/*_zero_pos")})
    cands = sorted({(p.parts[-4], p.parts[-3]) for p in
                    PIPELINE.glob("circuits_data/*/*/circuits/*_zero_pos")})
    print(f"  models with positional candidates: {pilots}")
    print(f"  candidate circuits: {len(cands)}")
    for b in sorted({base(m) for m in pilots}):
        print(f"  {b:12} " + " ".join(
            f"{a}={'Y' if (b if a == 'large' else f'{b}__{a}') in ck else '-'}" for a in ARCHES))
    print("  faithful C4 at any arch: 0 (the smallest stage drops positional candidates as not"
          " output-metric comparable)")


SECTIONS = {"presence": sec_presence, "quality": sec_quality, "c0": sec_c0, "c1": sec_c1,
            "c3": sec_c3, "c2": sec_c2, "bridge": sec_bridge, "c4": sec_c4}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--section", choices=sorted(SECTIONS), action="append",
                    help="repeatable; default is every section")
    a = ap.parse_args()
    tax = json.loads(TAXONOMY.read_text())
    ck = checkpoints()
    print(f"taxonomy: {TAXONOMY.relative_to(PIPELINE)}   checkpoints: {len(ck)}")
    for name in (a.section or list(SECTIONS)):
        SECTIONS[name](tax, ck)


if __name__ == "__main__":
    main()
