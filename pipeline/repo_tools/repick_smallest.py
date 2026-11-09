#!/usr/bin/env python3
"""Offline re-pick of plain-mode smallest_circuit.json winners from cached evals.

Re-runs the smallest stage's eligibility + selection (imported from
find_circuits/smallest_faithful_circuit.py — including the relative recovery floor,
the empty-baseline guard and the edges-aware tie-break) over an existing
circuits_data tree WITHOUT any GPU work: each circuit's
``.smallest_circuit_evals/evals/*.json`` caches are read directly by filename (their
config keys embed absolute machine-specific paths and are never re-derived) and BOTH full
budgets come from the model's GEOMETRY via ``model_registry.get_spec`` — nodes
``= L*H + (0 if attn_only else L)``, edges ``= model_registry.live_edges(spec)`` —
never from a finder summary. Differential / suppression / marker rows are skipped
(re-picking them needs GPU re-eval).

A cache is re-pickable ONLY when it carries the current pool regime (``REGIME_KEYS``
— the machine-independent half of the stage's ``_eval_config`` match) and its
candidate dir still parses, under the stage's own ``discover_candidates`` so the
attn-only phantom-m{l} strip applies, to exactly the edge set the cache scored.
Anything else is a ``stale_regime`` skip: reported, never re-picked, because those
numbers describe a pool/ablation/edge-set the current pipeline no longer produces.
The winner is written through the stage's own ``build_payload``, so a re-picked row
carries the same keys (selection_regime / candidate_set_sig / min_relative_recovery)
a stage-written one does.

Safety: for every row the recorded winner must be reproduced by the OLD tie-break
(or already equal the new pick — idempotent re-runs), and the recorded
selection_score must match the recomputed one. Two provenance drifts are
classified instead of fatal, because they say the recorded json is out of date
rather than that the re-pick is wrong: ``stale_recorded`` (both keys agree on a
candidate the winner json never saw) and ``stale_denominator`` (the recorded score
reproduces at one of the FINDER-REPORTED edge totals — edge_pruning/acdc/eap all
disagree at identical geometry — instead of at the geometric budget). Anything
else is an error and is never rewritten.

    python repo_tools/repick_smallest.py --root circuits_data_run2/_raw --dry-run
"""

from __future__ import annotations

import argparse, glob, json, os, shutil, sys
from types import SimpleNamespace

_THIS = os.path.dirname(os.path.abspath(__file__))
PIPELINE_ROOT = os.path.dirname(_THIS)
sys.path.insert(0, os.path.join(PIPELINE_ROOT, "find_circuits"))

import smallest_faithful_circuit as SFC          # noqa: E402  (selection rules, imported not duplicated)
import model_registry as MG                      # noqa: E402  (geometry budgets; SFC's import put models/ on sys.path)

SCORE_TOL = 5e-6                                 # recorded scores are rounded to 6dp
MARKER_REASONS = ("base_not_faithful", "positional_only_candidates")
#: Where each finder records its full-edge total. They disagree at identical geometry
#: (1184/1377/1519 at 4L x 8H), which is why the budget is geometric now; a recorded score
#: that reproduces at one of these was written before that fix.
ARTIFACT_TOTAL_KEYS = (("edge_pruning_sparsity_*", ("n_edges_total",)),
                       ("acdc_thresh_*", ("n_prunable_edges_initial",)),
                       ("eap_top*", ("graph", "real_edges")))
#: Eval-cache keys whose VALUE repick can verify offline — the pool-regime half of the
#: stage's ``_eval_config``. A v1 cache (unmasked pools, rows [0:n]) fails these.
REGIME_KEYS = {"pairs_regime": MG.CANON_PAIRS_REGIME,
               "pairs_window": [MG.POOL_WINDOWS[SFC.SCORING_CONSUMER],
                                MG.POOL_WINDOWS[SFC.SCORING_CONSUMER] + SFC.SMALLEST_EVAL_N],
               "pairs_consumer": SFC.SCORING_CONSUMER,
               "n_requested": SFC.SMALLEST_EVAL_N,
               "ablation": SFC.EDGE_ABLATION_TAG}
#: Keys the selection + the payload read that only a current-schema cache carries.
#: (checkpoint_sig/dataset_sig are machine-dependent, so presence is all repick can check.)
REQUIRED_KEYS = ("checkpoint_sig", "dataset_sig", "edges_sig", "n_examples",
                 "R_sequence_accuracy", "perf_full", "perf_empty")


class RowError(Exception):
    pass


class StaleEvalCache(RowError):
    """The cache predates the current pool regime / phantom-edge strip: not re-pickable."""


def _load(path: str):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _old_pick(results: dict, target: float, edge_counts: dict, full_heads: int,
              full_edges: int, relative_floor: float) -> dict | None:
    """The pre-2026-07-16 tie-break (-score, -acc, nodes, name), under the CURRENT
    eligibility rule — parity must isolate the tie-break, not mix in a second change."""
    elig = [r for r in results.values() if SFC._passes(r, target, relative_floor)]
    if not elig:
        return None
    return sorted(elig, key=lambda r: (-SFC._selection_score(r, edge_counts[r["name"]], full_heads, full_edges),
                                       -r["perf_circuit"]["sequence_accuracy"],
                                       SFC._n_nodes(r),
                                       r["name"]))[0]


def _check_regime(e: dict, fname: str) -> None:
    """Refuse an eval cache the smallest stage's ``_cached_eval`` would refuse."""
    missing = [k for k in (*REGIME_KEYS, *REQUIRED_KEYS) if k not in e]
    if missing:
        raise StaleEvalCache(f"{fname}: pre-regime eval cache (no {', '.join(missing)})")
    off = [f"{k}={e[k]!r} != {v!r}" for k, v in REGIME_KEYS.items() if e[k] != v]
    if off:
        raise StaleEvalCache(f"{fname}: eval cache from another regime ({'; '.join(off)})")


def _artifact_edge_totals(cpath: str) -> set[int]:
    """Full-edge totals the FINDERS report here — the denominators a pre-geometry
    recorded score could have been written at."""
    totals = set()
    for pat, keys in ARTIFACT_TOTAL_KEYS:
        for sp in sorted(glob.glob(os.path.join(cpath, "circuits", pat, "summary.json"))):
            v = _load(sp)                      # _load is None on a missing/corrupt summary
            for k in keys:
                v = v[k] if isinstance(v, dict) and k in v else None
            if v is not None:
                totals.add(int(v))
    return totals


def _repick(cpath: str, sj: dict, model: str, circuit: str, relative_floor: float) -> dict:
    """Selection decision for one plain row: recorded / old-key / new-key picks."""
    evals = {}
    for ep in sorted(glob.glob(os.path.join(cpath, ".smallest_circuit_evals", "evals", "*.json"))):
        e = _load(ep)
        if e and "name" in e and "perf_circuit" in e:
            _check_regime(e, os.path.basename(ep))
            evals[e["name"]] = e
    if not evals:
        raise RowError("no eval caches")
    spec = MG.get_spec(model)
    full_heads = spec.n_layers * spec.n_heads + (0 if spec.attn_only else spec.n_layers)
    full_edges = MG.live_edges(spec)              # geometry, never a finder-reported total
    cand_dir = os.path.join(cpath, "circuits")
    # Re-parse through the STAGE's loader so the attn-only phantom-m{l} strip applies here
    # too; a cache scored on the unstripped edge set is stale, not re-pickable.
    cands = {c.name: c for c in SFC.discover_candidates([cand_dir], spec.attn_only)}
    if set(cands) != set(evals):
        raise RowError(f"candidate set {sorted(set(cands) ^ set(evals))} is not the set the "
                       f"eval caches cover, so candidate_set_sig would misdescribe the pick")
    drifted = [n for n, c in cands.items()
               if SFC._edges_sig(c.edges) != evals[n]["edges_sig"]
               or c.n_nodes != SFC._n_nodes(evals[n])]
    if drifted:
        raise StaleEvalCache(f"candidates {sorted(drifted)} now parse to a different circuit "
                             f"than their cached eval scored (phantom-edge strip / "
                             f"re-discovery)")
    target = sj["min_accuracy_threshold"]
    edge_counts = {n: len(c.edges) for n, c in cands.items()}

    notes = []                                    # (class, message) provenance drifts
    recorded = sj.get("circuit_name")
    if recorded is not None and recorded not in evals:
        raise RowError(f"recorded winner {recorded!r} has no eval cache")
    if recorded is not None:                      # recorded-score tripwire
        w = evals[recorded]
        got = SFC._selection_score(w, edge_counts[recorded], full_heads, full_edges)
        if abs(got - sj["selection_score"]) > SCORE_TOL:
            artifact = next((e for e in sorted(_artifact_edge_totals(cpath))
                             if abs(SFC._selection_score(w, edge_counts[recorded], full_heads, e)
                                    - sj["selection_score"]) <= SCORE_TOL), None)
            if artifact is None:
                raise RowError(f"score mismatch for {recorded}: recorded "
                               f"{sj['selection_score']:.6f} vs recomputed {got:.6f} "
                               f"(H={full_heads} E={full_edges})")
            notes.append(("stale_denominator", f"{recorded} score {sj['selection_score']:.6f} "
                          f"was written at the finder total E={artifact}, not the geometric "
                          f"E={full_edges} (recomputes to {got:.6f}); re-picking at E={full_edges}"))

    new = SFC.pick_winner(evals, target, edge_counts, full_heads, full_edges, relative_floor)
    old = _old_pick(evals, target, edge_counts, full_heads, full_edges, relative_floor)
    new_name = new["name"] if new else None
    old_name = old["name"] if old else None
    if recorded not in (old_name, new_name):
        if old_name != new_name:                  # tie-break actually moved the winner: ambiguous
            raise RowError(f"parity failure: recorded {recorded!r} != old-key pick "
                           f"{old_name!r} (new-key {new_name!r})")
        if new_name is not None:                  # both keys pick None = nothing eligible, not a drift
            notes.append(("stale_recorded", f"recorded {recorded!r} predates candidates now in the "
                          f"eval cache; both keys pick {new_name!r}"))
    args = SimpleNamespace(model=model, circuit=circuit, base_circuit=None,
                           suppression=False, base_fallback=None, target=target,
                           relative_floor=relative_floor,
                           candidate_set_sig=SFC._candidate_set_sig([cand_dir]))
    return {"evals": evals, "cands": cands, "edge_counts": edge_counts, "H": full_heads,
            "E": full_edges, "recorded": recorded, "old": old_name, "args": args,
            "new": new, "new_name": new_name, "notes": notes}


def _rewrite(cpath: str, sj_path: str, d: dict) -> None:
    """Write the re-picked winner (+ its .pt) through the stage's OWN build_payload, so a
    re-picked row is schema-identical to a stage-written one."""
    if d["new_name"] is None:
        # Retracting a recorded winner also drops its .pt/.png: a stage decision, not a
        # cache-only one.
        raise RowError(f"no candidate is eligible any more; re-run the smallest stage "
                       f"for this row (recorded winner {d['recorded']!r} left untouched)")
    SFC._dump_json(SFC.build_payload(d["evals"][d["new_name"]], d["cands"][d["new_name"]],
                                     d["args"], d["edge_counts"], d["H"], d["E"]), sj_path)
    if d["new_name"] != d["recorded"]:
        src_pt = os.path.join(cpath, "circuits", d["new_name"], "circuit.pt")
        out_pt = os.path.splitext(sj_path)[0] + ".pt"
        if os.path.isfile(src_pt):
            shutil.copyfile(src_pt, out_pt)
        else:
            print(f"    WARNING: {src_pt} missing; smallest_circuit.pt left stale")


def main() -> int:
    ap = argparse.ArgumentParser(description="Re-run smallest-circuit selection "
                                 "offline from cached candidate evals.")
    ap.add_argument("--root", default=os.path.join(PIPELINE_ROOT, "circuits_data"),
                    help="circuits_data tree (default: %(default)s)")
    ap.add_argument("--dry-run", action="store_true",
                    help="report picks without rewriting anything")
    ap.add_argument("--relative-floor", type=float, default=SFC.DEFAULT_RELATIVE_FLOOR,
                    help="floor on R_sequence_accuracy (default: %(default)s)")
    ap.add_argument("--json", help="write the per-row result manifest here")
    args = ap.parse_args()

    counts = {"rows": 0, "repicked": 0, "marker": 0, "mode": 0, "no_sj": 0,
              "errors": 0, "changed": 0, "smaller_edges": 0, "no_faithful": 0,
              "rewritten": 0, "old_match": 0, "stale_recorded": 0,
              "stale_denominator": 0, "stale_regime": 0}
    changes, errors, stale, manifest = [], [], [], []
    for mdir in sorted(os.listdir(args.root)):
        mpath = os.path.join(args.root, mdir)
        if not os.path.isdir(mpath):
            continue
        for cdir in sorted(os.listdir(mpath)):
            cpath = os.path.join(mpath, cdir)
            if not os.path.isdir(cpath):
                continue
            counts["rows"] += 1
            row = f"{mdir}/{cdir}"
            sj_path = os.path.join(cpath, "smallest_circuit.json")
            sj = _load(sj_path)
            if sj is None:
                counts["no_sj"] += 1
                continue
            if sj.get("reason") in MARKER_REASONS:
                counts["marker"] += 1
                continue
            if sj.get("mode") in ("differential", "suppression"):
                counts["mode"] += 1
                continue
            try:
                d = _repick(cpath, sj, mdir, cdir, args.relative_floor)
                counts["old_match"] += d["old"] == d["recorded"]
                classes = [c for c, _ in d["notes"]]
                for cls, msg in d["notes"]:
                    counts[cls] += 1
                    stale.append(f"{cls.upper():17s} {row}: {msg}")
                if d["new_name"] != d["recorded"]:
                    counts["changed"] += 1
                    classes.append("changed")
                    o = d["evals"].get(d["recorded"])   # None when the row had no winner
                    o_txt = (f"{d['recorded']} ({o['n_heads']}h/{o['n_edges']}e "
                             f"acc={o['perf_circuit']['sequence_accuracy']:.3f})"
                             if o else str(d["recorded"]))
                    if d["new_name"] is None:
                        counts["no_faithful"] += 1
                        classes.append("no_faithful")
                        changes.append(f"{row}: {o_txt} -> no_faithful_circuit")
                    else:
                        n = d["evals"][d["new_name"]]
                        if o and n["n_edges"] < o["n_edges"]:
                            counts["smaller_edges"] += 1
                        changes.append(f"{row}: {o_txt} -> {d['new_name']} "
                                       f"({n['n_heads']}h/{n['n_edges']}e "
                                       f"acc={n['perf_circuit']['sequence_accuracy']:.3f})")
                if not args.dry_run:
                    _rewrite(cpath, sj_path, d)
                    counts["rewritten"] += 1
                counts["repicked"] += 1
                if classes:
                    manifest.append({"row": row, "classes": classes, "recorded": d["recorded"],
                                     "picked": d["new_name"], "old_key_pick": d["old"],
                                     "messages": [m for _, m in d["notes"]]})
            except StaleEvalCache as e:
                counts["stale_regime"] += 1
                stale.append(f"{'STALE_REGIME':17s} {row}: {e}")
                manifest.append({"row": row, "classes": ["stale_regime"],
                                 "recorded": sj.get("circuit_name"), "picked": None,
                                 "old_key_pick": None, "messages": [str(e)]})
            except (RowError, FileNotFoundError, KeyError) as e:
                counts["errors"] += 1
                errors.append(f"{row}: {type(e).__name__}: {e}")
                manifest.append({"row": row, "classes": ["error"],
                                 "recorded": sj.get("circuit_name"), "picked": None,
                                 "old_key_pick": None,
                                 "messages": [f"{type(e).__name__}: {e}"]})

    for line in changes:
        print("CHANGED " + line)
    for line in stale:
        print(line)
    for line in errors:
        print("ERROR   " + line, file=sys.stderr)
    print(f"\nrows={counts['rows']}  plain re-picked={counts['repicked']}  "
          f"skipped: marker={counts['marker']} diff/supp={counts['mode']} "
          f"no-json={counts['no_sj']} stale-regime={counts['stale_regime']}  "
          f"errors={counts['errors']}  R_seq floor={args.relative_floor}")
    print(f"stale_regime={counts['stale_regime']} (eval caches predate the current pool regime "
          f"{REGIME_KEYS['pairs_regime']} {REGIME_KEYS['pairs_window']} or the attn-only "
          "phantom-edge strip; NOT re-picked — re-run the smallest stage for those rows)")
    print(f"parity: OLD tie-break reproduces the recorded pick on "
          f"{counts['old_match']}/{counts['repicked']} re-picked rows "
          "(must be all on a pristine tree; re-picked rows drop out after a rewrite)")
    print(f"stale_recorded={counts['stale_recorded']} (recorded winner predates the eval "
          "cache; both tie-break keys agree, so the row is re-picked, not failed)")
    print(f"stale_denominator={counts['stale_denominator']} (recorded score reproduces at a "
          "FINDER-reported edge total; re-picked at the geometric one)")
    print(f"winners changed={counts['changed']} (strictly fewer edges: "
          f"{counts['smaller_edges']}; now no_faithful: {counts['no_faithful']})"
          + ("  [dry-run: nothing written]" if args.dry_run
             else f"  rewritten={counts['rewritten']}"))
    if not args.dry_run and counts["changed"]:
        print("note: smallest_circuit.png/.gv not regenerated (offline tool); "
              "figures refresh on the next pipeline pass")
    if args.json:
        SFC._dump_json({"root": os.path.abspath(args.root), "dry_run": args.dry_run,
                        "counts": counts, "rows": manifest}, args.json)
        print(f"wrote {args.json}")
    return 1 if counts["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
