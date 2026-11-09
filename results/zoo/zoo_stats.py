#!/usr/bin/env python3
"""Zoo-wide statistics, regenerated from committed files ONLY: the three indexes, the run
list (pipeline/models/models.txt) and the showcase BridgeCircuit's site_scan.json.

    python results/zoo/zoo_stats.py            # reads results/zoo/*_index.csv, writes stats.md

Deterministic: same inputs in, byte-identical stats.md out. Every table names its source
file and the row filter it applies. The indexes themselves come from build_index.py.
"""
import argparse
import csv
import glob
import json
import os
import statistics as st
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
RUN_LIST = os.path.join(REPO, "pipeline", "models", "models.txt")
FINDERS = ["eap", "acdc", "edge_pruning"]
#: the zoo BridgeCircuit row compared against its showcase record (current BridgeVariable)
BRIDGE_EXAMPLE = ("psum_novel_id__moe8", "bridge1_novel_id")
ARCHES = ["small", "large", "wide", "deep", "xl", "mlp", "moe4", "moe8"]
FFN = ["mlp", "moe4", "moe8"]
CLASSES = ["DomainCircuit", "ContrastCircuit", "SuppressionCircuit"]
CANON_TIER = {a: ("xlarge" if a == "xl" else a) for a in ARCHES}
MAX_LAYERS = 8
TEXT_COLS = {"mlp_layers", "finder_param"}     # never parsed as numbers ("0" is a layer list)


def load(name):
    with open(os.path.join(HERE, name)) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in r.items():
            if v == "":
                r[k] = None
            elif k not in TEXT_COLS:
                try:
                    r[k] = int(v)
                except ValueError:
                    try:
                        r[k] = float(v)
                    except ValueError:
                        pass
    return rows


def f2(x, nd=2):
    return "–" if x is None else f"{x:.{nd}f}"


def pct(num, den):
    return "–" if not den else f"{100 * num / den:.1f}%"


def mean(xs):
    xs = [x for x in xs if x is not None]
    return st.mean(xs) if xs else None


def median(xs):
    xs = [x for x in xs if x is not None]
    return st.median(xs) if xs else None


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def src(csv_name, filt):
    return f"*Source: `{csv_name}`; filter: {filt}.*"


def split_model(model):
    """'foo_id__moe4' -> ('foo_id', 'moe4'); 'foo_id' -> ('foo_id', 'large') (build_index.py)."""
    base, _, suffix = model.partition("__")
    return base, suffix or "large"


def run_list():
    """The run list, parsed like run_model_pipeline.list_models: text before '#', stripped,
    blank lines skipped, first occurrence kept."""
    with open(RUN_LIST) as f:
        return list(dict.fromkeys(n for n in (line.split("#", 1)[0].strip() for line in f) if n))


def bridge_note(probes):
    """About-this-snapshot note: the zoo BridgeCircuit rows vs the current BridgeVariable."""
    bridge = [p for p in probes if p["class"] == "BridgeCircuit"]
    model, var = BRIDGE_EXAMPLE
    row = next(p for p in bridge if p["model"] == model and p["variable"] == var)
    path, = sorted(glob.glob(os.path.join(REPO, "results", "showcase", "*", "probes", model,
                                          var, "site_scan.json")))
    with open(path) as f:
        cur = json.load(f)

    def one(col):
        vals = sorted({p[col] for p in bridge})
        return ", ".join(str(v) for v in vals)

    pos = cur["positions"]
    pos = (f"{pos[0]}–{pos[-1]}" if pos == list(range(pos[0], pos[-1] + 1))
           else ", ".join(map(str, pos)))
    return [f"All {len(bridge)} BridgeCircuit rows of `probes_index.csv` record `n_classes` = "
            f"{one('n_classes')} (`chance_uniform` {one('chance_uniform')}) over "
            f"{one('n_positions')} input positions. `build_index.py` reads `n_classes` and the "
            "positions from the probe records, which were written under an earlier "
            "BridgeVariable definition with 10 classes at every input position, position 0 "
            "included. The current "
            "`bridge_variable` (`pipeline/probes/variables.py`) sets `n_classes` to the measured "
            "label cardinality and keeps only non-constant, load-bearing positions: for "
            f"`{model}` / `{var}` the zoo row has {row['n_classes']} classes at "
            f"{row['n_positions']} positions (best accuracy {f2(row['best_acc'], 3)} at "
            f"L{row['best_layer']}), while the showcase record "
            f"(`{os.path.relpath(path, REPO)}`) has {cur['n_classes']} classes at positions "
            f"{pos}. The verdicts do not use `n_classes` (they compare against the "
            "embedding baseline and the majority class), but where the current definition "
            "drops a position, the zoo row's position means still include it.", ""]


def run_list_coverage(models, circuits, probes):
    """§1 subsection: run-list entries vs the checkpoints of this snapshot (survivorship)."""
    listed = run_list()
    ck = {m["model"] for m in models}
    nock_rows = {c["model"] for c in circuits if c["model"] not in ck}
    rows = []
    for a in ARCHES + [None]:
        xs = [n for n in listed if a is None or split_model(n)[1] == a]
        mux = [n for n in xs if n.startswith("mux")]
        have, mux_have = sum(n in ck for n in xs), sum(n in ck for n in mux)
        rows.append([a or "all", len(xs), have, pct(have, len(xs)),
                     sum(n in nock_rows for n in xs), len(mux), mux_have, pct(mux_have, len(mux))])
    tiers = Counter(split_model(n)[0] for n in listed)
    ck_tiers = Counter(m["base"] for m in models)
    mux_bases = sorted(b for b in tiers if b.startswith("mux"))
    unlisted = sorted({r["model"] for r in models + circuits + probes} - set(listed))
    return ["### Run-list coverage", "",
            src("pipeline/models/models.txt` + `models_index.csv` + `circuits_index.csv",
                "every run-list entry, grouped by the model-name suffix"), "",
            "`pipeline/models/models.txt` is the run list: one model id per line, `<base>` for "
            "Large and `<base>__<arch>` otherwise (text after `#` is a comment). A run-list "
            "entry without a checkpoint means that no checkpoint for that model is in this "
            "snapshot. The mux columns count the multiplexed (`mux*`) subset of each row.", "",
            table(["arch", "run-list entries", "with checkpoint", "coverage",
                   "no checkpoint, with circuit rows", "mux entries", "mux with checkpoint",
                   "mux coverage"], rows), "",
            f"Base algorithms: {len(tiers)} in the run list "
            f"({sum(v == len(ARCHES) for v in tiers.values())} listed at all "
            f"{len(ARCHES)} architectures), {len(ck_tiers)} with at least one checkpoint "
            f"({sum(v == len(ARCHES) for v in ck_tiers.values())} with a checkpoint at all "
            f"{len(ARCHES)}); {sum(b not in ck_tiers for b in tiers)} listed base algorithms "
            "have no checkpoint at any architecture. Multiplexed (`mux*`) base algorithms: "
            f"{len(mux_bases)} in the run list, {sum(b in ck_tiers for b in mux_bases)} with at "
            "least one checkpoint. Models in the three indexes that are not in the run list: "
            f"{', '.join(f'`{m}`' for m in unlisted) if unlisted else 'none'}.", ""]


def below_gate(models, circuits, probes):
    """§1 subsection: the checkpoints whose recorded eval accuracy is below the 0.95 gate."""
    low = sorted((m for m in models if m["eval_seq_acc"] is not None and m["eval_seq_acc"] < 0.95),
                 key=lambda m: (ARCHES.index(m["arch"]), m["model"]))
    rows = []
    for m in low:
        cs = [c for c in circuits if c["model"] == m["model"]]
        rows.append([f"`{m['model']}`", m["arch"], m["tier"], m["escalated"],
                     f2(m["eval_seq_acc"], 4), f2(m["train_acc"], 4), m["metrics_layout"],
                     m["epoch"], f"{len(cs)} ({sum(c['faithful'] == 1 for c in cs)})",
                     sum(p["model"] == m["model"] for p in probes)])
    return ["### Checkpoints below the 0.95 gate", "",
            src("models_index.csv` + `circuits_index.csv` + `probes_index.csv",
                "`eval_seq_acc < 0.95`"), "",
            table(["model", "arch", "tier", "escalated", "eval seq-acc", "train acc",
                   "metrics layout", "epoch", "circuit rows (faithful)", "probe rows"], rows), "",
            f"The 0.95 gate (`SEQ_ACC_GATE` in `pipeline/run_model_pipeline.py`, applied by "
            "`pipeline/models/train_models/promote.py`) reads the same recorded eval accuracy "
            "that `eval_seq_acc` reports and would reject these checkpoints; `promote.py` "
            "records that earlier versions of the training scripts wrote checkpoints into the "
            f"checkpoint tree without this gate. The {len(low)} checkpoints and their circuit "
            "and probe rows are included in every table.", ""]


def winning_finder(circuits):
    """§3 subsection: which candidate finder produced each faithful winner."""
    win = [c for c in circuits if c["faithful"] == 1]
    rows = []
    for k in CLASSES:
        for r in sorted({c["run"] for c in win if c["class"] == k}):
            xs = [c for c in win if c["class"] == k and c["run"] == r]
            rows.append([k, r] + [sum(c["finder"] == f for c in xs) for f in FINDERS] + [len(xs)])
    other = sorted((c for c in win if c["finder"] != "eap"),
                   key=lambda c: (c["finder"], c["model"], c["circuit"]))
    by = Counter(c["finder"] for c in other)

    def only(col):
        return " / ".join(sorted({str(c[col]) for c in other}))

    return ["### Winning finder", "",
            src("circuits_index.csv", "`faithful == 1`; `finder` = family of the winning "
                "candidate"), "",
            table(["class", "run"] + FINDERS + ["total"], rows), "",
            f"The {len(other)} faithful winners not found by EAP-IG are {only('class')} rows of "
            f"the {only('run')} run at {only('arch')}: "
            f"{', '.join(f'{f} {by[f]}' for f in FINDERS if by[f])} (`finder_param` = ACDC "
            "threshold τ or Edge Pruning target sparsity).", "",
            table(["model", "arch", "run", "circuit", "finder", "param"],
                  [[f"`{c['model']}`", c["arch"], c["run"], f"`{c['circuit']}`", c["finder"],
                    c["finder_param"]] for c in other]), ""]


def decodable_share_of_all(probes):
    """§7 subsection: the first decodable layer as a share of ALL probed variables."""
    rows = []
    for a in ARCHES + [None]:
        xs = [p for p in probes if a is None or p["arch"] == a]
        cnt = Counter(p["first_decodable_layer"] for p in xs if p["verdict"] == "decodable")
        rows.append([a or "all", len(xs)] + [pct(cnt.get(l, 0), len(xs)) for l in range(MAX_LAYERS)]
                    + [pct(len(xs) - sum(cnt.values()), len(xs))])
    dec = [p for p in probes if p["verdict"] == "decodable"]
    l0 = sum(p["first_decodable_layer"] == 0 for p in dec)
    return ["### First decodable layer, share of all probed variables", "",
            src("probes_index.csv", "all rows; `not decodable` = `trivial_from_input` or "
                "`absent`"), "",
            table(["arch", "n"] + [f"L{l}" for l in range(MAX_LAYERS)] + ["not decodable"], rows),
            "", f"Layer 0 is the first decodable layer of {pct(l0, len(dec))} of the decodable "
            f"variables ({l0} of {len(dec)}) and of {pct(l0, len(probes))} of all "
            f"{len(probes)} probed variables.", ""]


def about_snapshot(models, rows):
    """The 'About this snapshot' section: which runs the records come from and which
    selection criteria produced the verdicts (every number computed from the CSVs)."""
    runs = {}
    for c in rows:
        runs.setdefault(c["run"], []).append(c)

    def run_line(name):
        xs = runs.get(name, [])
        by_class = Counter(c["class"] for c in xs)
        floors = sorted({(c["class"], c["floor"]) for c in xs}, key=lambda t: CLASSES.index(t[0]))
        n_test = sorted({c["n_test"] for c in xs})
        return (f"- **{name}** run ({', '.join(f'{n:,}' for n in n_test)} test rows per circuit): "
                f"{len(xs):,} rows ({', '.join(f'{k} {by_class[k]:,}' for k in CLASSES if by_class[k])}); "
                f"floors {', '.join(f'{k} {fl:.2f}' for k, fl in floors)}.")

    supp = [c for c in rows if c["class"] == "SuppressionCircuit" and c["faithful"] == 1]
    tonic = sum(c["release_reference"] == "tonic" for c in supp)
    low = [m for m in models if m["eval_seq_acc"] is not None and m["eval_seq_acc"] < 0.95]
    no_ck = [c for c in rows if c["has_checkpoint"] == 0]
    return ["## About this snapshot", "",
            "The circuit records come from two discovery runs of the pipeline "
            "(`run` column of `circuits_index.csv`):", "",
            run_line("early"), run_line("main"),
            f"- {len(runs.get(None, []))} further rows are markers written without an "
            "evaluation (e.g. the base circuit had no faithful winner); they carry no run.", "",
            "Every verdict is the zoo-wide selection of "
            "`pipeline/find_circuits/smallest_faithful_circuit.py`, under edge-level "
            "per-position mean ablation:", "",
            "- **DomainCircuit**: absolute whole-sequence accuracy of the circuit ≥ the "
            "row's floor, plus the stage's other eligibility conditions and its winner rule "
            "(docs/circuits.md §6.2; each record states the `selection_regime` it was "
            "selected under).",
            "- **ContrastCircuit**: differential mode: `s_union` (whole-sequence accuracy of "
            "candidate ∪ DomainCircuit) ≥ the row's floor, with a non-empty fringe.",
            f"- **SuppressionCircuit**: severing the fringe releases the wrong task "
            f"(release ≥ 0.5) with collateral ≤ 0.05, under the cued-row or the wrong-cue "
            f"(tonic) sever reference; "
            f"{'all' if tonic == len(supp) else f'{tonic} of the'} {len(supp)} faithful "
            "SuppressionCircuits pass only via the wrong-cue (tonic) reference.", "",
            "These are not the per-class faithfulness definitions and floors, nor the "
            "interchange patching for ContrastCircuits, used for the showcase circuits "
            "(see [docs/circuits.md](../../docs/circuits.md)); numbers from the two are not "
            "comparable. Edge fractions below use the live-edge denominator.", "",
            f"{len(low)} checkpoints score below the 0.95 eval sequence-accuracy gate; they "
            "and their circuits are included in every table. "
            f"{len(no_ck)} circuit rows ({len({c['model'] for c in no_ck})} models) have no "
            "checkpoint in the indexed set; they are included in the counts and faithful "
            "rates, and their full node/edge budgets use the architecture's default geometry.",
            ""]


def main():
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args()
    models, circuits, probes = (load(f"{n}_index.csv") for n in ("models", "circuits", "probes"))
    tier = {m["model"]: m["tier"] for m in models}
    pub = [c for c in circuits if c["arch"] in ARCHES]
    canon = [c for c in pub if tier.get(c["model"], CANON_TIER[c["arch"]]) == CANON_TIER[c["arch"]]]
    evaluated = [c for c in pub if c["not_faithful_reason"] in (None, "below_floor")]
    win = [c for c in canon if c["faithful"] == 1]
    md = ["# Zoo statistics", "",
          "Generated by `python results/zoo/zoo_stats.py` from `models_index.csv`, "
          "`circuits_index.csv` and `probes_index.csv` (built by `build_index.py` from the "
          "pipeline artifacts). Do not edit by hand.", "",
          "The run-list coverage table also reads `pipeline/models/models.txt`, and the "
          "BridgeCircuit note reads the showcase BridgeCircuit's `site_scan.json`.", ""]
    md += about_snapshot(models, pub)
    md += bridge_note(probes)
    md += ["## Conventions", "",
           "- **Architectures** are the 8 zoo variants, keyed by the model-name suffix "
           "(no suffix = Large).",
           "- **Size** of a ContrastCircuit / SuppressionCircuit is its *fringe* (candidate "
           "edges minus the base circuit's edges). Edge fractions use the live edge budget "
           "(`n_edges_live / n_edges_full`; attention-only models exclude the EAP graph's "
           "phantom MLP edges). Size tables keep only models whose checkpoint has its "
           "architecture's canonical geometry (escalated checkpoints are excluded).",
           "- Fringe metrics (`d_*`, release, collateral) in rows without a recorded metric "
           "unit are token-level deltas (the pipeline's pre-`2026-07-v2` convention).", ""]

    # ---------------------------------------------------------------- 1. checkpoints
    bases = {m["base"] for m in models}
    md += ["## 1. Checkpoints", "",
           f"{len(models)} checkpoints over {len(bases)} base algorithms.", ""]
    rows = []
    for a in ARCHES:
        ms = [m for m in models if m["arch"] == a]
        acc = [m["eval_seq_acc"] for m in ms if m["eval_seq_acc"] is not None]
        rows.append([a, len(ms), sum(m["escalated"] == 1 for m in ms), f2(min(acc), 3),
                     f2(median(acc), 4), f2(mean(acc), 4), pct(sum(x >= 0.95 for x in acc), len(acc)),
                     pct(sum(x >= 0.99 for x in acc), len(acc))])
    md += [src("models_index.csv", "all rows, grouped by `arch`"), "",
           table(["arch", "checkpoints", "escalated", "min eval seq-acc", "median", "mean",
                  "≥ 0.95", "≥ 0.99"], rows), "",
           "`escalated` = the checkpoint's geometry is a larger tier than its suffix "
           "(e.g. an `__mlp` model trained at 8 layers × 16 heads).", ""]
    md += run_list_coverage(models, pub, probes)
    md += below_gate(models, pub, probes)

    # ---------------------------------------------------------------- 2. circuit records
    md += ["## 2. Circuit records", "", src("circuits_index.csv", "all rows"), ""]
    other = Counter(c["arch"] for c in circuits if c["arch"] not in ARCHES)
    other_note = (f"{sum(other.values())} rows of other variants "
                  f"({', '.join(f'{k}: {v}' for k, v in sorted(other.items()))}) are excluded "
                  "below. " if other else "")
    rows = [[k] + [sum(c["class"] == k and c["arch"] == a for c in pub) for a in ARCHES]
            + [sum(c["class"] == k for c in pub)] for k in CLASSES]
    md += [table(["class"] + ARCHES + ["total"], rows), "",
           f"Total {len(circuits)} rows. {other_note}"
           f"{sum(c['has_checkpoint'] == 0 for c in pub)} rows have no checkpoint in the "
           "indexed checkpoint set.", ""]
    reasons = Counter((c["class"], c["not_faithful_reason"]) for c in pub if c["faithful"] == 0)
    md += ["Rows that are not faithful, by recorded reason (`below_floor` = evaluated, no "
           "candidate cleared the floor; the others were not evaluated):", "",
           table(["class", "reason", "rows"], [[k, r, n] for (k, r), n in sorted(reasons.items())]),
           ""]
    runs = Counter((c["run"], c["floor"], c["class"]) for c in pub)
    md += ["Selection runs present (`run` is derived from `n_test`: 5000 → main, "
           "20000 → early):", "",
           table(["run", "floor", "class", "rows"],
                 [[r or "–", fl if fl is not None else "–", k, n]
                  for (r, fl, k), n in sorted(runs.items(), key=lambda t: str(t[0]))]), ""]

    # ---------------------------------------------------------------- 3. faithful rate
    md += ["## 3. Faithful rate", "",
           src("circuits_index.csv", "`not_faithful_reason` empty or "
               "`below_floor` (evaluated rows)"), ""]
    rows = []
    for k in CLASSES:
        r = [k]
        for a in ARCHES + [None]:
            xs = [c for c in evaluated if c["class"] == k and (a is None or c["arch"] == a)]
            r.append(f"{pct(sum(c['faithful'] for c in xs), len(xs))} ({sum(c['faithful'] for c in xs)}/{len(xs)})")
        rows.append(r)
    hi = [c for c in pub if c["class"] == "DomainCircuit" and c["faithful"] == 1]
    rows.append(["DomainCircuit winners with accuracy ≥ 0.90"]
                + [pct(sum(c["abs_seq_acc"] >= 0.90 for c in hi if c["arch"] == a),
                       sum(c["arch"] == a for c in hi)) for a in ARCHES]
                + [pct(sum(c["abs_seq_acc"] >= 0.90 for c in hi), len(hi))])
    md += [table(["class"] + ARCHES + ["all"], rows), "",
           "The selection keeps the sparsest candidate above the floor, so a winner's "
           "accuracy sits close to the floor; the last row is the share of DomainCircuit "
           "winners whose recorded accuracy also clears 0.90.", ""]
    md += winning_finder(pub)

    # ---------------------------------------------------------------- 4. circuit size
    md += ["## 4. Circuit size (faithful circuits)", "",
           src("circuits_index.csv", "`faithful == 1`, canonical "
               "geometry"), ""]
    for k in CLASSES:
        rows = []
        for a in ARCHES:
            xs = [c for c in win if c["class"] == k and c["arch"] == a]
            if not xs:
                rows.append([a, 0] + ["–"] * 10)
                continue
            H, L = xs[0]["n_heads_full"], xs[0]["n_mlp_full"]
            ef = [c["n_edges_live"] / c["n_edges_full"] for c in xs]
            rows.append([a, len(xs), f"{f2(mean([c['n_heads'] for c in xs]), 1)} / {H}",
                         f"{f2(median([c['n_heads'] for c in xs]), 1)} / {H}",
                         pct(sum(c["n_heads"] == c["n_heads_full"] for c in xs), len(xs)),
                         f2(mean([c["n_edges_live"] for c in xs]), 1),
                         f2(median([c["n_edges_live"] for c in xs]), 1),
                         pct(mean(ef), 1), pct(median(ef), 1),
                         f"{f2(mean([c['n_mlp_nodes'] for c in xs]), 2)} / {L}" if L else "–",
                         pct(sum(c["n_mlp_nodes"] >= 1 for c in xs), len(xs)) if L else "–",
                         pct(sum(c["n_mlp_nodes"] == L for c in xs), len(xs)) if L else "–"])
        md += [f"### {k}", "",
               table(["arch", "n", "heads mean", "heads median", "all heads kept",
                      "edges mean", "edges median", "edge frac mean", "edge frac median",
                      "MLP nodes mean", "≥ 1 MLP node", "all MLP nodes"], rows), ""]

    # heads vs edges
    rows = []
    for k in CLASSES:
        xs = [c for c in win if c["class"] == k]
        hf = [c["n_heads"] / c["n_heads_full"] for c in xs]
        ef = [c["n_edges_live"] / c["n_edges_full"] for c in xs]
        rows.append([k, len(xs), pct(mean(hf), 1), pct(mean(ef), 1),
                     pct(sum(e < h for h, e in zip(hf, ef)), len(xs))])
    md += ["### Heads vs edges", "",
           src("circuits_index.csv", "as above, pooled over architectures"), "",
           table(["class", "n", "mean head fraction kept", "mean edge fraction kept",
                  "edge fraction < head fraction"], rows), ""]

    # MLP / expert-block layers
    rows = []
    for a in FFN:
        xs = [c for c in win if c["class"] == "DomainCircuit" and c["arch"] == a]
        dropped = [sorted(set(range(c["n_mlp_full"])) - {int(t) for t in (c["mlp_layers"] or "").split(";") if t})
                   for c in xs]
        some = [d for d in dropped if d]
        layer = Counter(l for d in some for l in d)
        suffix = sum(d == list(range(4 - len(d), 4)) for d in some)   # canonical FFN tiers: 4 layers
        rows.append([a, len(xs), len(some), pct(suffix, len(some))]
                    + [layer.get(l, 0) for l in range(4)])
    md += ["### Dropped MLP / MoE blocks (DomainCircuits)", "",
           src("circuits_index.csv", "DomainCircuit, `faithful == 1`, canonical geometry, "
               "MLP/MoE variants; dropped = layers absent from `mlp_layers`"), "",
           table(["arch", "circuits", "drop ≥ 1 block", "dropped set = last k layers",
                  "L0 dropped", "L1 dropped", "L2 dropped", "L3 dropped"], rows), ""]

    # ---------------------------------------------------------------- 5. contrast
    rows = []
    for a in ARCHES:
        xs = [c for c in win if c["class"] == "ContrastCircuit" and c["arch"] == a]
        allx = [c for c in pub if c["class"] == "ContrastCircuit" and c["arch"] == a]
        rows.append([a, len(xs), sum(c["n_edges_live"] == 0 for c in xs),
                     sum(c["n_heads"] == 0 for c in xs),
                     pct(sum(c["d_affected"] < 0.02 for c in xs), len(xs)),
                     pct(sum(c["d_affected"] >= 0.10 for c in xs), len(xs)),
                     sum(c["base_faithful"] == 0 for c in allx)])
    md += ["## 5. ContrastCircuit fringes", "",
           src("circuits_index.csv", "ContrastCircuit; first six columns `faithful == 1`, "
               "canonical geometry; last column all rows"), "",
           table(["arch", "faithful", "empty (0 live edges)", "0 heads",
                  "d_affected < 0.02", "d_affected ≥ 0.10", "rows with unfaithful base"], rows),
           "", "`d_affected` = drop in accuracy on the edit-affected outputs when the fringe "
           "alone is severed; near 0 means the base DomainCircuit already carries the "
           "predicate.", ""]

    # ---------------------------------------------------------------- 6. suppression
    rows = []
    for a in ARCHES:
        mux = [m for m in models if m["arch"] == a and str(m["base"]).startswith("mux")]
        xs = [c for c in pub if c["class"] == "SuppressionCircuit" and c["arch"] == a]
        ok = [c for c in xs if c["faithful"] == 1]
        refs = Counter(c["release_reference"] for c in ok)
        rows.append([a, len(mux), len({c["model"] for c in xs}), len(xs), len(ok),
                     ", ".join(f"{k}: {v}" for k, v in sorted(refs.items())) or "–"])
    md += ["## 6. SuppressionCircuit coverage", "",
           src("models_index.csv` + `circuits_index.csv", "multiplexed (`mux*`) checkpoints; "
               "SuppressionCircuit rows"), "",
           table(["arch", "mux checkpoints", "mux models with rows", "rows", "faithful",
                  "passing sever reference"], rows), ""]

    # ---------------------------------------------------------------- 7. probes
    md += ["## 7. Probe decodability", "",
           "Linear probes at the input positions of every layer's residual stream "
           "(`train_probes.py`). `verdict`: `decodable` = first layer whose probe error is at "
           "most half the embedding-baseline error and half the majority-class error; "
           "`trivial_from_input` = the embedding alone decodes it (≥ 0.9); else `absent`.", "",
           src("probes_index.csv", "all rows"), ""]
    rows = []
    for a in ARCHES + [None]:
        xs = [p for p in probes if a is None or p["arch"] == a]
        v = Counter(p["verdict"] for p in xs)
        rows.append([a or "all", len(xs), sum(p["class"] == "BridgeCircuit" for p in xs),
                     pct(v["decodable"], len(xs)), pct(v["trivial_from_input"], len(xs)),
                     pct(v["absent"], len(xs)),
                     pct(sum(p["best_acc"] > p["majority_rate"] for p in xs), len(xs))])
    md += [table(["arch", "variables", "of which BridgeCircuit", "decodable",
                  "trivial_from_input", "absent", "best layer beats majority class"], rows), ""]
    for col, title, filt in (("first_decodable_layer", "First decodable layer",
                              "`verdict == decodable`"),
                             ("best_layer", "Layer of highest probe accuracy",
                              "all rows (ties → lowest layer)")):
        rows = []
        for a in ARCHES + [None]:
            xs = [p[col] for p in probes if (a is None or p["arch"] == a) and p[col] is not None]
            cnt = Counter(xs)
            rows.append([a or "all", len(xs)] + [pct(cnt.get(l, 0), len(xs)) for l in range(MAX_LAYERS)])
        n_dec = sum(p[col] is not None for p in probes)
        note = ([f"Percentages are shares of the `n` decodable variables ({n_dec} of "
                 f"{len(probes)} probed variables); the next table gives them as shares of all "
                 "probed variables.", ""] if col == "first_decodable_layer" else [])
        md += [f"### {title}", "", src("probes_index.csv", filt), ""] + note + [
               table(["arch", "n"] + [f"L{l}" for l in range(MAX_LAYERS)], rows), ""]
        if col == "first_decodable_layer":
            md += decodable_share_of_all(probes)
    gate = Counter(p["causal_gate"] for p in probes)
    passed = sum(p["gate_passed"] == 1 for p in probes)
    md += ["### Causal gate", "", src("probes_index.csv", "all rows"), "",
           f"Directional patching along the probe direction (`validate_probe.py`): measured on "
           f"{gate['measured']} variables, skipped on {gate['skipped']} (no counterfactual "
           f"defined), no steerable rows on {gate['no_steered_rows']}. {passed} pass the "
           "output gate (flip rate ≥ 0.80 and collateral ≤ 0.05).", ""]

    # ---------------------------------------------------------------- 8. training recipe
    layouts = sorted({m["metrics_layout"] for m in models})
    rows = []
    for a in ARCHES:
        ms = [m for m in models if m["arch"] == a]
        rows.append([a] + [sum(m["metrics_layout"] == l for m in ms) for l in layouts]
                    + [f2(median([m["epoch"] for m in ms if m["metrics_layout"] == "steps+eval"]), 0),
                       f2(median([m["epoch"] for m in ms if m["metrics_layout"] == "epoch+train+eval"]), 0)])
    recipe = Counter((m["lr"], m["batch_size"], m["max_steps"]) for m in models if m["lr"] is not None)
    dtypes = Counter(m["param_dtype"] for m in models)
    md += ["## 8. Training metadata recorded in the checkpoints", "",
           src("models_index.csv", "all rows"), "",
           table(["arch"] + layouts + ["median `epoch` (steps+eval)", "median `epoch` (epoch+train+eval)"], rows), "",
           "`steps+eval`: `epoch` holds the optimizer step count and only eval accuracy is "
           "stored (the stacked size-grid trainer). `epoch+train+eval`: `epoch` is an epoch "
           "index and train loss/accuracy are stored (the per-model trainer used for every "
           "MLP/MoE model and for the multiplexed models). `flat+recipe`: the recipe is "
           "written into the checkpoint: "
           + "; ".join(f"lr {lr}, batch {int(b)}, max steps {int(s)} ({n} checkpoints)"
                       for (lr, b, s), n in sorted(recipe.items())) + ". "
           f"Stored parameter dtypes: {', '.join(f'{k} ({v})' for k, v in sorted(dtypes.items()))}. "
           "No other learning-rate, batch-size or precision field is recorded; the "
           "per-architecture recipes are documented in "
           "`pipeline/models/train_models/hparams.py`.", ""]

    with open(os.path.join(HERE, "stats.md"), "w") as f:
        f.write("\n".join(md))
    print(f"wrote {os.path.relpath(os.path.join(HERE, 'stats.md'))}")


if __name__ == "__main__":
    main()
