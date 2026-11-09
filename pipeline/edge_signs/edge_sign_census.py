"""Edge-sign census: negative-score edge mass per model x method across all
finder candidates in circuits_data/. Suppression circuits show up as
negative-score edges, so this measures how much of each candidate's score mass
is negative, aggregated per (model, method) with base-vs-__variant tier tags.

Score semantics per method (verified against real artifacts + finder sources):
  acdc          circuit.json edges list; score = signed metric delta (ACDC
                effect_size = evaluated - old metric), but the keep-rule only
                retains edges above a positive threshold -> negatives censored.
  eap           circuit.json is the raw EAP graph (nodes/edges dicts); in-graph
                edges carry signed attribution scores -> negatives survive.
  edge_pruning  circuit.json edges list; score = L0 gate value in [0.5, 1.0],
                a magnitude with no sign -> recorded as unsigned.
"""

import argparse, json, os, sys
from dataclasses import dataclass
from pathlib import Path

PIPELINE_DIR = Path(__file__).resolve().parent.parent
METHOD_PREFIXES = {"acdc_thresh_": "acdc", "eap_top": "eap",
                   "edge_pruning_sparsity_": "edge_pruning"}
SCORE_SEMANTICS = {"acdc": "signed_metric_delta", "eap": "signed_attribution",
                   "edge_pruning": "unsigned_gate"}


class CensusError(Exception):
    pass


def die(msg: str) -> None:
    raise CensusError(msg)


@dataclass
class CandidateStats:
    model: str
    circuit: str
    method: str
    candidate: str
    n_edges: int
    n_negative: int
    neg_score_mass: float


def _method_of(candidate: str) -> str | None:
    for prefix, method in METHOD_PREFIXES.items():
        if candidate.startswith(prefix):
            return method
    return None


def _candidate_scores(raw: dict, path: str) -> list[float]:
    """Signed scores of the kept edges, from either finder dialect of circuit.json."""
    edges = raw.get("edges")
    if isinstance(edges, dict):  # eap_graph: keep only in-graph edges
        kept = [v for v in edges.values()
                if (v.get("in_graph", True) if isinstance(v, dict) else bool(v))]
        return [v["score"] for v in kept if isinstance(v, dict) and v.get("score") is not None]
    if isinstance(edges, list):  # uniform list dialect (acdc / edge_pruning)
        return [e["score"] for e in edges if isinstance(e, dict) and e.get("score") is not None]
    die(f"unparseable circuit.json (no edges list/dict): {path}")


def census_candidate(model: str, circuit: str, method: str, candidate: str,
                     path: str) -> CandidateStats:
    try:
        raw = json.load(open(path))
    except (OSError, json.JSONDecodeError) as e:
        die(f"cannot read {path}: {e}")
    scores = _candidate_scores(raw, path)
    neg = [s for s in scores if s < 0]
    total_mass = sum(abs(s) for s in scores)
    return CandidateStats(model, circuit, method, candidate, len(scores), len(neg),
                          sum(abs(s) for s in neg) / total_mass if total_mass else 0.0)


def walk_candidates(root: Path) -> list[CandidateStats]:
    if not root.is_dir():
        die(f"root not found: {root}")
    out = []
    for dirpath, _dirnames, filenames in os.walk(root):
        if "circuit.json" not in filenames:
            continue
        candidate = os.path.basename(dirpath)
        method = _method_of(candidate)
        if method is None:  # not a finder candidate dir (e.g. edges/, heads_only/)
            continue
        parts = Path(dirpath).relative_to(root).parts
        if len(parts) != 4 or parts[2] != "circuits":
            die(f"unexpected candidate layout {dirpath}: expected <model>/<circuit>/circuits/<candidate>")
        out.append(census_candidate(parts[0], parts[1], method, candidate,
                                    os.path.join(dirpath, "circuit.json")))
    if not out:
        die(f"no finder candidates found under {root}")
    return sorted(out, key=lambda c: (c.model, c.circuit, c.method, c.candidate))


def build_census(cands: list[CandidateStats]) -> dict:
    models: dict = {}
    for c in cands:
        m = models.setdefault(c.model, {"tier": c.model.partition("__")[2] or "base",
                                        "methods": {}})
        meth = m["methods"].setdefault(c.method, {"signed": SCORE_SEMANTICS[c.method] != "unsigned_gate",
                                                  "per_candidate": []})
        meth["per_candidate"].append({"circuit": c.circuit, "candidate": c.candidate,
                                      "n_edges": c.n_edges, "n_negative": c.n_negative,
                                      "neg_score_mass": c.neg_score_mass})
    for m in models.values():
        for meth in m["methods"].values():
            fracs = [p["neg_score_mass"] for p in meth["per_candidate"]]
            meth["candidates"] = len(fracs)
            meth["mean_neg_frac"] = sum(fracs) / len(fracs)
            meth["max_neg_frac"] = max(fracs)
    return {"score_semantics": SCORE_SEMANTICS, "models": models}


def print_table(census: dict) -> None:
    hdr = f"{'model':<22} {'tier':<5} {'method':<13} {'candidates':>10} {'mean_neg_frac':>14} {'max_neg_frac':>13}"
    print(hdr)
    print("-" * len(hdr))
    for model in sorted(census["models"]):
        m = census["models"][model]
        for method in sorted(m["methods"]):
            s = m["methods"][method]
            print(f"{model:<22} {m['tier']:<5} {method:<13} {s['candidates']:>10} "
                  f"{s['mean_neg_frac']:>14.4f} {s['max_neg_frac']:>13.4f}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, default=PIPELINE_DIR / "circuits_data")
    ap.add_argument("--out", type=Path, default=PIPELINE_DIR / "edge_signs" / "census.json")
    args = ap.parse_args()
    try:
        census = build_census(walk_candidates(args.root))
    except CensusError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(census, indent=2) + "\n")
    print_table(census)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
