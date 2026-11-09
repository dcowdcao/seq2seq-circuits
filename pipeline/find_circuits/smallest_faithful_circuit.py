"""Select the SMALLEST faithful circuit among one circuit's discovered candidates.

Reads every candidate under ``circuits_data/<model>/<circuit>/circuits/`` (ACDC
``acdc_thresh_<tau>``, EAP-IG ``eap_top<n>``, Edge Pruning
``edge_pruning_sparsity_<s>``; formats sniffed circuit_loader-style) and scores
each one's faithfulness under EDGE-LEVEL per-position mean ablation on the
circuit's dataset (``model_registry.load_circuit_pairs(consumer="scoring")`` <-
the ``[10000:15000)`` window of ``gen_data/<model>/<circuit>/inputs.json``, DISJOINT
from the finders' attribution rows and from the rows eap.py's ladder search picks
its winner on): each kept head's q/k/v input, each
kept MLP input, and the final logit read is rebuilt as
``recon = m_site + Σ_declared(live_s − mean_s)`` over the candidate's DECLARED
edges; every undeclared edge carries that sender's per-position dataset mean
(severed, on-distribution — not zeroed). See the ``_edge_*`` helpers.

SELECTION (regime ``2026-07-v2``, stamped as ``selection_regime`` on every
output): a candidate is ELIGIBLE iff ALL THREE hold —

  * its ABSOLUTE whole-sequence exact-match accuracy under this ablation
    (``perf_circuit.sequence_accuracy`` — every answer token correct) is
    ``>= --target`` (default 0.85 = DEFAULT_TARGET),
  * its RELATIVE recovery ``R_sequence_accuracy = (circ - empty)/(full - empty)``
    is a NUMBER and ``>= --relative-floor`` (default 0.85), so the verdict
    measures the circuit and not the checkpoint; when full == empty the ratio is
    0/0 and the field carries the ``zero_contrast`` marker instead of a number
    (model_registry.recovery_ratio raises rather than inventing a denominator),
    which fails this gate,
  * the EMPTY baseline (every edge severed: ``perf_empty.sequence_accuracy``) is
    BELOW ``--target`` — a near-constant-output isolation whose fully
    mean-ablated model already clears the floor has nothing to attribute.

Among the eligible candidates the winner MAXIMIZES

    (1 - n_nodes_circuit/n_nodes_full) * (1 - n_edges_circuit/n_edges_full)

with ``n_nodes_full = n_layers * (n_heads + 0 if attn_only else 1)`` and
``n_edges_full = model_registry.live_edges(spec)`` — the LIVE attribution-graph
edge count computed from the geometry (never read from a finder's summary:
edge_pruning/acdc/eap report 1184/1377/1519 at 4L x 8H), with the phantom m{l}
edges of an attn-only model excluded because the ablation cannot reconstruct
them. Ties: fewer edges, then fewer
nodes, then higher absolute accuracy, then name (eap.py's search tie-break:
prefer the smaller circuit). No qualifying candidate -> a
``no_faithful_circuit`` marker (no .pt / PNG). Optionally refines between
adjacent candidate sizes by spawning the discovery runners for midpoint
parameters (--refine-rounds). ``--search`` runs FIRST and drives eap.py's
one-attribution ``--target-seq-acc`` search, which emits the score-maximizing
EAP candidate that then wins the normal selection below; it is what produces
almost every plain row.

DIFFERENTIAL MODE (``--base-circuit``, for ContrastCircuit contrast-pair
circuits): eligibility is the candidate∪base union reaching --target whole-seq
accuracy with a non-empty fringe; the severing metrics (d_affected/d_unaffected/
d_control) are ALWAYS computed and recorded but never gate (classification lives
in models/taxonomy.json). The winner maximizes
``(1 + clamp(D_affected,0,1)) * (1 - (h/H)*(e/E))``. Severing uses the DUAL
edge forward (declared edges carry the mean, everything else live).

SUPPRESSION MODE (``--suppression`` + ``--base-circuit``, C3 cue-flip TaskMux
circuits): the base is the CUED task's executor; the reported circuit is the
FRINGE beyond it. Eligible iff the fringe is non-empty and severing it RELEASES
the wrong task (``release_rate >= --release-floor``) with collateral
``<= --epsilon`` on the agree positions, under EITHER sever reference — cued-row
means or wrong-cue "tonic" means (a cue-CONSTANT suppressor equals its own cued
mean, so the cued reference is blind to it). The score reads whichever reference
PASSED; both are recorded.

In fringe modes the reported edge set is the FRINGE, so the payload carries the
fringe's own counts plus ``source_candidate_absolute_accuracy`` (the WHOLE
candidate's accuracy — it does NOT describe the fringe) instead of
``whole_sequence_absolute_accuracy``. All differential/suppression metrics are
WHOLE-SEQUENCE exact match over the relevant positions, matching the plain gate
and the thresholds models/taxonomy.json applies to them. Design:
docs/reference/circuit_types.md §4 (C1) and §5 (C3).

In BOTH differential and suppression mode an UNFAITHFUL base never blocks the
row: its edge set falls back to the base's best-available candidate
(``best_available_candidate``) and the output — winner or no_faithful — records
``base_faithful: false`` + ``base_accuracy`` + ``base_candidate``; only a base
with zero candidates raises (stage-ordering bug).

Outputs (paths via model_registry builders ONLY):
    circuits_data/<model>/<circuit>/smallest_circuit.json   slim payload (docs/reference/design.md)
    circuits_data/<model>/<circuit>/smallest_circuit.pt     winner's circuit.pt, copied
    circuits_figures/<model>/<circuit>/smallest_circuit.png (+ .gv)  eap.draw_circuit_png

MEMORY SAFETY: the parent process never touches the GPU; every candidate eval is a
separate subprocess (``--eval-child``) with ``wait_for_headroom`` before each
spawn and sleep-60 retries (max 5) on OOM-ish failures. ``--jobs`` (default 2)
is the concurrency cap and is NOT clamped — run_model_pipeline passes 4.

Run:
    ./.venv/bin/python pipeline/find_circuits/smallest_faithful_circuit.py \
        --model dblodd_id --circuit all_odd --target 0.85 --jobs 2
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from types import SimpleNamespace

# --------------------------------------------------------------------------- #
# sys.path bootstrap: model_registry is the single import site (model loading / rails / data / paths); importing it also puts interp_tools/, data_gen/, metrics/ on sys.path (so circuit_loader + metrics resolve).
# --------------------------------------------------------------------------- #
_THIS = os.path.dirname(os.path.abspath(__file__))            # .../find_circuits
PIPELINE_ROOT = os.path.dirname(_THIS)                        # .../pipeline
REPO_ROOT = os.path.dirname(PIPELINE_ROOT)                    # .../circuits
_MODELS_DIR = os.path.join(PIPELINE_ROOT, "models")
if _MODELS_DIR not in sys.path:
    sys.path.insert(0, _MODELS_DIR)

import model_registry as MG                                      # noqa: E402
import circuit_loader as CL                                   # noqa: E402  (repo interp_tools, via MG bootstrap)

PY = sys.executable
SMOKE_DIR = os.path.join(PIPELINE_ROOT, ".smoke", "run4")

#: 0.85 ABSOLUTE whole-sequence exact-match accuracy floor (>=); see docs/reference/design.md.
#: (0.90 -> 0.85 with the 0.95 train gate: preserves the old relative slack vs 0.999 models.)
DEFAULT_TARGET = 0.85
#: RELATIVE floor on R_sequence_accuracy, conjoined with DEFAULT_TARGET. Same number as the
#: absolute floor: on a healthy checkpoint (full ~ 1, empty ~ 0) abs >= 0.85 already implies
#: R >= 0.85, so this binds exactly on the high-empty-baseline / weak-checkpoint rows the
#: absolute floor cannot see -- where the verdict would otherwise measure the checkpoint.
DEFAULT_RELATIVE_FLOOR = 0.85
DEFAULT_RELEASE_FLOOR, DEFAULT_EPSILON = 0.5, 0.05      # suppression release / collateral gates
#: Selection-semantics regime stamped on every output: empty-baseline guard + relative floor +
#: spec-derived edge budget + attn-only phantom-edge strip + whole-sequence fringe metrics.
#: Rows from different regimes are NOT comparable; a regime bump re-selects every row.
SELECTION_REGIME = "2026-07-v2"
#: Faithfulness ablation scheme. Tagged into every eval result so legacy
#: head-level caches (which lack it) are never reused (see _cached_eval).
EDGE_ABLATION_TAG = "edge_level_per_position_mean"
#: E3 suppression tonic sever reference. Tagged into suppression eval results so
#: pre-tonic caches (which lack release_rate_tonic) are never reused.
TONIC_REFERENCE_TAG = "wrong_cue_per_position_mean"
#: Unit of every differential/suppression metric. Tagged into those eval results so
#: token-accuracy caches (the pre-2026-07-v2 mix) are never reused.
FRINGE_METRIC_TAG = "whole_sequence_exact_match"
#: Tolerance of the full-declared identity assert (reconstruction == real model). The
#: reconstruction re-sums the residual stream, so it is float-noise, not exact: measured
#: 3e-6 on CPU fp32 (small..deep, attn-only and mlp) and ~5e-3 on the TF32 GPU path a real
#: run uses. A broken reconstruction is O(1), so 1e-2 separates the two without flapping.
FULL_DECLARED_TOL = 1e-2
#: This stage reports the FINAL faithfulness numbers, so it reads the SCORING window of
#: the pool (model_registry.POOL_WINDOWS) — disjoint from the attribution rows the finders
#: used and from the selection rows eap.py's --target-seq-acc ladder search picks on.
SCORING_CONSUMER = "scoring"
#: R_* value written when the model and the fully-severed baseline are indistinguishable:
#: the ratio is 0/0, so the scope is UNSCORABLE. A marker, never a number (a number here
#: is what the old ``or 1.0`` denominator guard invented).
ZERO_CONTRAST = "zero_contrast"
MEAN_N_PROD, MEAN_N_SMOKE = 2048, 64          # mean-ablation reference batch
BATCH_CAP_PROD, BATCH_CAP_SMOKE = 512, 64     # max rows per forward (politeness)
SMOKE_EVAL_N = 64                             # smoke eval size (pairs)
#: Production faithfulness eval-set size (PAIRS). 5000 (was 20000): ±0.4% std on whole-seq acc, fine for the 0.85 floor; reads the SCORING window of inputs.json, so the pool needs POOL_WINDOWS["scoring"] + SMALLEST_EVAL_N = 15000 pairs (gen_circuit_data writes 20000 by default).
SMALLEST_EVAL_N = 5000
MAX_OOM_RETRIES = 5
OOM_SLEEP_S = 60

_OOM_MARKERS = ("out of memory", "memory headroom", "cuda error",
                "cublas", "hip error", "memoryerror")
#: model_registry.dataset_to_pairs raises a bare ValueError with this text when EVERY row of
#: the pool agrees with its resample reference: the logit diff is identically 0, so nothing
#: is scorable. Sniffed from the child log (as OOM is) because the registry does not raise a
#: dedicated class — that verdict must be RECORDED, not left as a stale winner on disk.
_ZERO_CONTRAST_MARKER = "zero-contrast isolation"


class ZeroContrastDataset(RuntimeError):
    """The circuit's dataset carries no clean/corrupt contrast at all (unscorable scope)."""


def _load_fc_eap():
    """Import the sibling eap.py once (drawer + prune helpers), cached in sys.modules."""
    name = "pipeline_find_circuits_eap"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            name, os.path.join(_THIS, "eap.py"))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return sys.modules[name]


def _json_ok(path) -> bool:
    """True iff ``path`` exists AND parses as JSON (a killed run can leave a
    truncated artifact -- that must count as not-done). [min_faithful :110]"""
    try:
        with open(path) as f:
            json.load(f)
        return True
    except (OSError, json.JSONDecodeError):
        return False


def _content_sig(path: str) -> str:
    """md5 of a file's CONTENT — regenerating inputs.json IN PLACE must miss the eval cache
    (the path alone cannot see it)."""
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def _stat_sig(path: str) -> str:
    """(mtime, size) identity of a big file — retraining to the same .pt must miss the cache."""
    st = os.stat(path)
    return f"{int(st.st_mtime)}:{st.st_size}"


def _round_floats(obj, nd=6):
    """Fixed float rounding for byte-stable JSON artifacts."""
    if isinstance(obj, float):
        return round(obj, nd)
    if isinstance(obj, dict):
        return {k: _round_floats(v, nd) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_round_floats(v, nd) for v in obj]
    return obj


def _dump_json(payload, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(_round_floats(payload), f, indent=2, sort_keys=True)
    return path


# --------------------------------------------------------------------------- #
# Candidate discovery + format sniffing (circuit_loader dialects)             #
# --------------------------------------------------------------------------- #


@dataclass
class Candidate:
    name: str                 # dir name, e.g. "acdc_thresh_0.1", "eap_top200"
    source: str               # abs path of the circuit JSON that was parsed
    dialect: str              # acdc_summary | eap_graph | eap_topn_summary | manual
    heads: list               # sorted [(l, h), ...] via circuit_loader
    edges: list = field(default_factory=list)
    #     full node-level edges [{"parent","child","qkv","score"}] in the EAP
    #     node dialect ("input", "a0.h7", "m1", "logits") -- for the JSON
    #     payload and the drawer.
    mlps: list = field(default_factory=list)
    #     sorted [l, ...] MLP layer indices kept (attn+MLP only; ALWAYS empty for attention-only models -> keep-set/ablation/node-count/JSON all collapse to legacy head-only behavior, byte-for-byte).

    @property
    def n_heads(self) -> int:
        return len(self.heads)

    @property
    def n_mlps(self) -> int:
        return len(self.mlps)

    @property
    def n_nodes(self) -> int:
        """Total kept node count = heads + MLPs. For attention-only models
        (no kept MLPs) this is identical to ``n_heads`` -> the size-based
        ranking/selection is UNCHANGED."""
        return len(self.heads) + len(self.mlps)

    def head_labels(self) -> list:
        return [f"L{l}H{h}" for (l, h) in self.heads]

    def mlp_labels(self) -> list:
        return [f"m{l}" for l in self.mlps]


def _sniff_dialect(d):
    """The exact dialect checks of circuit_loader.load_circuit_from_file:310."""
    if not isinstance(d, dict):
        return None
    if "heads_in_circuit" in d or "surviving_edges" in d:
        return "acdc_summary"
    if isinstance(d.get("nodes"), dict) and "edges" in d:
        return "eap_graph"
    if "circuit_heads_top_export_n" in d:
        return "eap_topn_summary"
    if "circuit_heads" in d:
        return "manual"
    return None


#: when several JSONs in one candidate dir parse, prefer the richest dialect
_DIALECT_PREF = {"eap_graph": 0, "acdc_summary": 1, "eap_topn_summary": 2,
                 "manual": 3}


def _node_name(label):
    """Normalize ANY node label to the EAP node dialect the drawer takes:
    'input' | 'logits' | 'a{l}.h{h}' | 'm{n}' | None (unrecognized)."""
    if label is None:
        return None
    low = str(label).strip().lower()
    if low in ("embed", "input", "resid_pre", "hook_embed", "tok_embeds"):
        return "input"
    if low in ("resid_post", "logits", "output"):
        return "logits"
    if re.fullmatch(r"m\d+", low):
        return low
    lh = CL.parse_head_label(label)
    if lh is not None:
        return f"a{lh[0]}.h{lh[1]}"
    return None


def _add_edge(out, seen, parent, child, qkv, score):
    """De-dup on (parent, child, qkv), keeping the max-|score| witness."""
    key = (parent, child, qkv)
    if key in seen:
        if score is not None and abs(score) > abs(out[seen[key]]["score"]):
            out[seen[key]]["score"] = float(score)
        return
    seen[key] = len(out)
    out.append({"parent": parent, "child": child, "qkv": qkv,
                "score": float(score) if score is not None else 0.0})


def _parse_edge_str(estr):
    """EAP edge string 'src->dst[<q|k|v>]' -> (parent, child, qkv) in the node
    dialect; None when ``estr`` is not an edge string."""
    if not estr or "->" not in str(estr):
        return None
    src, dst = str(estr).split("->", 1)
    m = re.match(r"(.+?)<([qkv])>$", dst.strip())
    child, qkv = (m.group(1), m.group(2)) if m else (dst.strip(), None)
    return _node_name(src) or src.strip(), _node_name(child) or child, qkv


def _full_edges(raw: dict, dialect: str) -> list:
    """Node-level edges (with qkv channel + score) from a raw candidate dict,
    normalized to the EAP node dialect. Unlike circuit_loader's head-level
    reduction, input/logits/MLP endpoints are KEPT (the drawer needs them)."""
    out, seen = [], {}
    if dialect == "acdc_summary":
        for e in raw.get("surviving_edges", []):
            p = _node_name(e.get("sender_label"))
            c = _node_name(e.get("receiver_label"))
            if p is None or c is None:
                continue
            if p == c:
                # DIRECT_COMPUTATION plumbing (a0.4_qin -> a0.4_q) collapses to
                # a self-edge; the channel info already lives on the ADDITION
                # edge INTO the qkv input, so self-edges are dropped.
                continue
            qkv = CL._channel_of(e.get("receiver_label"))
            _add_edge(out, seen, p, c, qkv, e.get("effect_size"))
    elif dialect == "eap_graph":
        for estr, v in raw.get("edges", {}).items():
            in_graph = v.get("in_graph", True) if isinstance(v, dict) else bool(v)
            parsed = _parse_edge_str(estr)
            if not in_graph or parsed is None:
                continue
            score = v.get("score") if isinstance(v, dict) else None
            _add_edge(out, seen, *parsed, score)
    elif dialect == "eap_topn_summary":
        for e in raw.get("top30_edges", []):
            parsed = _parse_edge_str(e.get("edge") if isinstance(e, dict) else e)
            if parsed is None:
                continue
            score = e.get("score") if isinstance(e, dict) else None
            _add_edge(out, seen, *parsed, score)
    else:                                       # manual: sender/receiver/channel
        for e in raw.get("edges", []):
            p = _node_name(e.get("sender"))
            c = _node_name(e.get("receiver"))
            if p is None or c is None:
                continue
            _add_edge(out, seen, p, c, e.get("channel"), e.get("score"))
    out.sort(key=lambda e: (e["parent"], e["child"], e["qkv"] or ""))
    return out


def _mlp_layer(label):
    """'m{l}' -> int layer index l; None for any non-MLP label."""
    if label is None:
        return None
    m = re.fullmatch(r"m(\d+)", str(label).strip().lower())
    return int(m.group(1)) if m else None


def _mlp_layers(raw: dict, dialect: str) -> list:
    """Sorted MLP layer indices a candidate KEEPS, parsed from its raw circuit
    dict (the MLP analog of circuit_loader's head reduction). Mirrors each
    dialect's node/membership convention:

      * acdc_summary    -> ``mlps_in_circuit`` (["m0","m1"]); falls back to MLP
                           endpoints appearing in ``surviving_edges``.
      * eap_graph       -> ``nodes`` dict entries 'm{l}' with in_graph truthy.
      * eap_topn_summary-> ``circuit_mlps`` if present, else MLP endpoints in
                           ``top30_edges``.
      * manual          -> ``circuit_mlps`` / ``mlps`` if present, else MLP
                           endpoints in ``edges``.

    For attention-only candidates there are no MLP nodes anywhere -> [].
    """
    found = set()

    def _add_from_labels(labels):
        for lab in labels or []:
            l = _mlp_layer(lab)
            if l is not None:
                found.add(l)

    def _add_from_edge_endpoints(edges, parent_key, child_key):
        for e in edges or []:
            if not isinstance(e, dict):
                continue
            for l in (_mlp_layer(e.get(parent_key)), _mlp_layer(e.get(child_key))):
                if l is not None:
                    found.add(l)

    if dialect == "acdc_summary":
        if isinstance(raw.get("mlps_in_circuit"), list):
            _add_from_labels(raw["mlps_in_circuit"])
        else:
            _add_from_edge_endpoints(raw.get("surviving_edges"),
                                     "sender_label", "receiver_label")
    elif dialect == "eap_graph":
        nodes = raw.get("nodes", {})
        if isinstance(nodes, dict):
            for node, v in nodes.items():
                in_graph = v.get("in_graph", True) if isinstance(v, dict) else bool(v)
                if not in_graph:
                    continue
                l = _mlp_layer(node)
                if l is not None:
                    found.add(l)
    elif dialect == "eap_topn_summary":
        if isinstance(raw.get("circuit_mlps"), list):
            _add_from_labels(raw["circuit_mlps"])
        else:
            for e in raw.get("top30_edges", []):
                parsed = _parse_edge_str(e.get("edge") if isinstance(e, dict) else e)
                if parsed is not None:
                    _add_from_labels(parsed[:2])
    else:                                            # manual
        if isinstance(raw.get("circuit_mlps"), list):
            _add_from_labels(raw["circuit_mlps"])
        elif isinstance(raw.get("mlps"), list):
            _add_from_labels(raw["mlps"])
        else:
            _add_from_edge_endpoints(raw.get("edges"), "sender", "receiver")
    return sorted(found)


def _sniff_candidate_dir(subdir: str, name: str, attn_only: bool = False):
    """Find + parse the best circuit JSON in one candidate dir. Returns a
    Candidate or None. ``attn_only`` (callers that know the arch MUST pass
    ``spec.attn_only``) strips the EAP graph's phantom m{l} edges."""
    try:
        files = sorted(os.listdir(subdir))
    except OSError:
        return None
    parsed = []                                  # (dialect_pref, fname_pref, ...)
    for fname in files:
        if not fname.endswith(".json"):
            continue
        path = os.path.join(subdir, fname)
        try:
            with open(path) as f:
                raw = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        dialect = _sniff_dialect(raw)
        if dialect is None:
            continue
        # priority: dialect richness, then the conventional file names
        fname_pref = {"summary.json": 0, "circuit.json": 1,
                      f"{name}.json": 2}.get(fname, 9)
        parsed.append((_DIALECT_PREF[dialect], fname_pref, path, dialect, raw))
    if not parsed:
        return None
    parsed.sort(key=lambda t: (t[0], t[1]))
    _, _, path, dialect, raw = parsed[0]
    circ = CL.load_circuit_from_file(path, name=name)   # the sniffing loader
    cand = Candidate(name=name, source=path, dialect=dialect,
                     heads=CL.head_tuples(circ), edges=_full_edges(raw, dialect),
                     mlps=_mlp_layers(raw, dialect))
    return _prune_candidate(cand, attn_only)


def _is_mlp_edge(e: dict) -> bool:
    return _mlp_layer(e["parent"]) is not None or _mlp_layer(e["child"]) is not None


def _prune_candidate(cand: "Candidate", attn_only: bool = False) -> "Candidate":
    """Drop DANGLING heads/MLPs — nodes with no in-graph incoming OR outgoing edge
    (EAP-IG Graph.prune semantics) — from a freshly-loaded candidate, so n_heads /
    n_edges / the selection score / the edge-level eval all use the connected
    circuit. ACDC leaves no-incoming heads (read pure mean) and Edge Pruning leaves
    no-outgoing heads; this removes them. EAP circuits are already pruned (no-op).

    On ``attn_only`` models the m{l} edges come FIRST: EAP-IG instantiates MLP graph
    nodes for every model, so a selected m{l} edge is phantom (scores exactly 0.0, no
    MLP to reconstruct, dropped at eval time by _edges_to_declared). Stripping them
    before the prune keeps n_edges / the score / the PNG / the payload honest and lets
    a head whose ONLY edge fed a phantom MLP die as the dangling node it is."""
    eap = _load_fc_eap()
    if attn_only:
        cand.edges = [e for e in cand.edges if not _is_mlp_edge(e)]
        cand.mlps = []
    nodes = ["input", "logits"]
    nodes += [f"a{l}.h{h}" for (l, h) in cand.heads]
    nodes += [f"m{l}" for l in cand.mlps]
    seen = set(nodes)
    for e in cand.edges:                          # include any edge-only endpoints
        for nm in (e["parent"], e["child"]):
            if nm not in seen:
                seen.add(nm)
                nodes.append(nm)
    pnodes, pedges = eap.prune_dangling(nodes, cand.edges)
    cand.heads, cand.mlps = eap.heads_mlps_from_nodes(pnodes)
    cand.edges = pedges
    return cand


def _is_variant_candidate_dir(sub: str) -> bool:
    """C4 positional/zero-ablation (``*_zero``/``*_pos``) or probe-metric (``*_probe_*``) candidate
    dir — NOT output-metric comparable, so excluded from the smallest-faithful search."""
    return "_probe_" in sub or sub.endswith(("_zero", "_pos"))


def discover_candidates(dirs: list, attn_only: bool = False) -> list:
    """All parseable candidates under the given dirs (one candidate per subdir).
    ``attn_only`` strips phantom MLP edges (see _prune_candidate)."""
    cands, skipped = [], []
    for d in dirs:
        if not os.path.isdir(d):
            print(f"    WARNING: candidates dir missing: {d}")
            continue
        for sub in sorted(os.listdir(d)):
            subdir = os.path.join(d, sub)
            if not os.path.isdir(subdir):
                continue
            if _is_variant_candidate_dir(sub):     # positional/probe: not output-metric comparable
                continue
            cand = _sniff_candidate_dir(subdir, sub, attn_only)
            if cand is None:
                skipped.append(sub)
                continue
            cands.append(cand)
    if skipped:
        print(f"    (skipped {len(skipped)} dirs with no parseable circuit "
              f"JSON: {skipped[:6]}{'...' if len(skipped) > 6 else ''})")
    # de-dup by name (a name appearing in two dirs keeps the first occurrence)
    seen, uniq = set(), []
    for c in cands:
        if c.name in seen:
            continue
        seen.add(c.name)
        uniq.append(c)
    return uniq


def best_available_candidate(circuit_dir: str,
                             attn_only: bool = False) -> tuple[Candidate, float | None]:
    """Unfaithful-base fallback (shared with task_router/router_diagnostic.py): among
    ``circuit_dir``'s candidates pick the highest cached mean-ablation whole-seq accuracy
    (tie -> fewer edges -> name). Accuracy reads .smallest_circuit_evals/evals/<name>.json
    (edges-sig-matched; None when uncached, ranked last). Raises FileNotFoundError when
    the circuit has NO candidates at all — its finder stage never ran (stage-ordering bug)."""
    cands = discover_candidates([os.path.join(circuit_dir, "circuits")], attn_only)
    if not cands:
        raise FileNotFoundError(
            f"best-available fallback: no candidate circuits under "
            f"{os.path.join(circuit_dir, 'circuits')} — the finder (eap) stage never "
            "produced any; run the finders for that circuit first")

    def acc(c: Candidate) -> float | None:
        p = os.path.join(circuit_dir, ".smallest_circuit_evals", "evals", f"{c.name}.json")
        if not _json_ok(p):
            return None
        d = json.load(open(p))
        ok = d.get("edges_sig") == _edges_sig(c.edges) and "perf_circuit" in d
        return d["perf_circuit"]["sequence_accuracy"] if ok else None

    scored = [(c, acc(c)) for c in cands]
    scored.sort(key=lambda ca: (-(ca[1] if ca[1] is not None else -1.0),
                                len(ca[0].edges), ca[0].name))
    return scored[0]


def _pick_candidate_subset(cands: list, k: int) -> list:
    """Deterministic --max-candidates subset: the smallest EVALUABLE (>0 heads)
    candidate (exercises the fail side of the target) plus the LARGEST candidate,
    drawn from a DIFFERENT dialect when possible (so a 2-candidate run exercises
    both parsers' head lists end-to-end); k>2 adds evenly spaced sizes between."""
    by_size = sorted(cands, key=lambda c: (c.n_heads, c.name))
    if k >= len(by_size):
        return by_size
    nonzero = [c for c in by_size if c.n_heads > 0] or by_size
    pool = [c for c in by_size if c.dialect != nonzero[0].dialect] or by_size
    largest = max(pool, key=lambda c: (c.n_heads, c.name))
    if k == 1:
        return [largest]                         # most likely to pass
    picks = {nonzero[0].name: nonzero[0], largest.name: largest}
    mids = [c for c in nonzero if c.name not in picks]
    for i in range(k - 2):
        if not mids:
            break
        c = mids[round(i * (len(mids) - 1) / max(1, k - 3))]
        picks.setdefault(c.name, c)
    return sorted(picks.values(), key=lambda c: (c.n_heads, c.name))[:k]


# --------------------------------------------------------------------------- #
# CHILD MODE: evaluate ONE candidate's faithfulness in its own process        #
# --------------------------------------------------------------------------- #
# Loads via model_registry (LNPre + per-process CUDA cap; rails inherited), scores full / circuit-only / empty under EDGE-LEVEL per-position mean ablation (_edge_* helpers), writes one JSON, exits. No multi-circuit sweep ever happens in one process (§0.6).


def _parse_heads_arg(s: str) -> list:
    if not s:
        return []
    out = []
    for lab in s.split(","):
        lh = CL.parse_head_label(lab.strip())
        if lh is None:
            raise ValueError(f"bad head label {lab!r}")
        out.append(lh)
    return sorted(set(out))


def _parse_mlps_arg(s: str) -> list:
    """'m0,m1' (or bare layer ints) -> sorted [0, 1]; empty for attn-only."""
    if not s:
        return []
    out = []
    for lab in s.split(","):
        lab = lab.strip()
        if not lab:
            continue
        l = _mlp_layer(lab)
        if l is None and re.fullmatch(r"\d+", lab):
            l = int(lab)
        if l is None:
            raise ValueError(f"bad mlp label {lab!r}")
        out.append(l)
    return sorted(set(out))


# =========================================================================== #
# EDGE-LEVEL per-position mean ablation  (the canonical faithfulness)          #
# Each receiver input (head q/k/v, MLP in, final read) is overwritten with recon = m_site + Σ_declared(live_s − mean_s) = Σ_declared live_s + Σ_undeclared mean_s: declared edges pass the REAL activation, undeclared paths carry the per-position MEAN (severed, on-distribution, not zeroed). See module docstring + the _edge_* helpers.
# Validated tensor-identical to the dual (value − Σ_undeclared(live−mean)) and, with EVERY causal edge declared, to the real model's logits.
# =========================================================================== #
def _edge_site_means(model, toks, batch_cap):
    """Per-position dataset means (real-model forward over ``toks`` — a prefix of
    the SAME clean eval tokens) for the edge-level reconstruction. Returns a dict:
        m_rpre {l:[S,d]}      site mean for head q/k/v inputs (resid_pre[l])
        m_rpost   [S,d]       site mean for the final logit read (resid_post)
        mean_input [S,d]      sender mean of the input/(embed+pos) stream
        mean_head {l:[S,H,d]} sender mean of each head's output (hook_result)
      attn+MLP models also get:
        m_rmid {l:[S,d]}      site mean for MLP inputs (hook_mlp_in)
        mean_mlp {l:[S,d]}    sender mean of each MLP output (hook_mlp_out)"""
    import torch
    L = model.cfg.n_layers
    attn_only = bool(model.cfg.attn_only)
    names = ([f"blocks.{l}.hook_resid_pre" for l in range(L)]
             + [f"blocks.{l}.attn.hook_result" for l in range(L)]
             + [f"blocks.{L - 1}.hook_resid_post"])
    if not attn_only:
        names += [f"blocks.{l}.hook_mlp_in" for l in range(L)]
        names += [f"blocks.{l}.hook_mlp_out" for l in range(L)]
    acc = {nm: None for nm in names}
    with torch.no_grad():
        for i0 in range(0, toks.shape[0], batch_cap):
            _, cache = model.run_with_cache(toks[i0:i0 + batch_cap],
                                            names_filter=names)
            for nm in names:
                s = cache[nm].sum(0)             # sum over batch -> keep [S, ...]
                acc[nm] = s if acc[nm] is None else acc[nm] + s
    B = toks.shape[0]
    m = {nm: acc[nm] / B for nm in names}
    out = {
        "m_rpre": {l: m[f"blocks.{l}.hook_resid_pre"] for l in range(L)},
        "m_rpost": m[f"blocks.{L - 1}.hook_resid_post"],
        "mean_input": m["blocks.0.hook_resid_pre"],          # embed+pos stream
        "mean_head": {l: m[f"blocks.{l}.attn.hook_result"] for l in range(L)},
    }
    if not attn_only:
        out["m_rmid"] = {l: m[f"blocks.{l}.hook_mlp_in"] for l in range(L)}
        out["mean_mlp"] = {l: m[f"blocks.{l}.hook_mlp_out"] for l in range(L)}
    return out


def full_declared_edges(n_layers: int, n_heads: int, attn_only: bool) -> dict:
    """EVERY causal edge of the geometry as ``{reader-site: {sender-id}}`` (the
    _edges_to_declared shape), in EAP-IG's residual order: a reader takes every writer
    at/below it. ``attn_only`` drops the m{l} nodes the ablation cannot reconstruct.
    This IS the edge universe: |pairs| == model_registry.live_edges(spec), asserted in
    assert_full_declared so the denominator and the ablation can never drift apart."""
    resid, declared = ["input"], {}
    for l in range(n_layers):
        for h in range(n_heads):
            for chan in "qkv":
                declared[(l, h, chan)] = set(resid)
        resid += [(l, h) for h in range(n_heads)]
        if not attn_only:
            declared[("m", l)] = set(resid)
            resid.append(("m", l))
    declared["final"] = set(resid)
    return declared


def _edges_to_declared(edges, attn_only=False) -> dict:
    """Edge list [{parent, child, qkv, score}] -> {reader-site: set(sender-id)}.
    sender-id: 'input' | (l,h) head | ('m', l) MLP.
    reader-site: (l,h,'q'|'k'|'v') head input | ('m', l) MLP input | 'final'.
    attn_only drops MLP-touching edges: EAP-IG carries m{l} graph nodes even for
    attn-only models, so a selected m{l} edge is phantom (no MLP to reconstruct)."""
    def sender(p):
        if p in ("input", "embed"):
            return "input"
        ml = _mlp_layer(p)
        if ml is not None:
            return ("m", ml)
        l, h = p[1:].split(".h")                  # a{l}.h{h}
        return (int(l), int(h))

    def site(child, qkv):
        if child == "logits":
            return "final"
        ml = _mlp_layer(child)
        if ml is not None:
            return ("m", ml)
        l, h = child[1:].split(".h")
        return (int(l), int(h), qkv)

    def is_mlp(x):
        return isinstance(x, tuple) and x[0] == "m"

    d = {}
    for e in edges:
        s, snd = site(e["child"], e["qkv"]), sender(e["parent"])
        if attn_only and (is_mlp(s) or is_mlp(snd)):
            continue
        d.setdefault(s, set()).add(snd)
    return d


def _edges_sig(edges) -> str:
    """Order-independent signature of a circuit's edge set (cache-match key)."""
    keys = sorted(f"{e['parent']}>{e['child']}:{e['qkv']}" for e in edges)
    return hashlib.md5("|".join(keys).encode()).hexdigest()


def _load_edges(path: str) -> list:
    """The candidate's declared edges [{parent, child, qkv, score}], written by
    _spawn_eval (a JSON list). Empty/absent path -> no edges (empty circuit)."""
    if not path or not os.path.exists(path):
        return []
    with open(path) as f:
        return json.load(f)


def _sender_delta(sid, live, means):
    """live_s − mean_s for ONE sender -> [B,S,d] (shared by both edge forwards)."""
    if sid == "input":
        return live["input"] - means["mean_input"].unsqueeze(0)
    if isinstance(sid, tuple) and sid[0] == "m":
        l2 = sid[1]
        return live["mlp"][l2] - means["mean_mlp"][l2].unsqueeze(0)
    l2, h2 = sid
    return (live["head"][l2][:, :, h2, :]
            - means["mean_head"][l2][:, h2, :].unsqueeze(0))


def _edge_forward(model, tokens, declared, means, circuit_heads, circuit_mlps,
                  L, H, attn_only):
    """One edge-ablated forward -> logits. Captures live writers (input,
    head_result, mlp_out) and overwrites each circuit receiver input + the final
    read with ``recon = m_site + Σ_declared(live − mean)``."""
    import torch
    live = {"input": None, "head": {}, "mlp": {}}

    def cap_input(value, hook):
        live["input"] = value                    # [B,S,d] resid_pre[0] = embed+pos
        return value

    def cap_result(value, hook, l):
        live["head"][l] = value                  # [B,S,H,d]
        return value

    def cap_mlp(value, hook, l):
        live["mlp"][l] = value                    # [B,S,d]
        return value

    def recon(base, decl, B, S, d):
        r = base.unsqueeze(0).expand(B, S, d).clone()   # m_site, [B,S,d]
        for sid in decl:
            r = r + _sender_delta(sid, live, means)
        return r

    def ow_qkv(value, hook, l, chan):
        B, S, _, d = value.shape
        for h in range(H):
            if (l, h) in circuit_heads:
                value[:, :, h, :] = recon(means["m_rpre"][l],
                                          declared.get((l, h, chan), ()), B, S, d)
        return value

    def ow_mlp_in(value, hook, l):
        if l not in circuit_mlps:
            return value
        B, S, d = value.shape
        return recon(means["m_rmid"][l], declared.get(("m", l), ()), B, S, d)

    def ow_final(value, hook):
        B, S, d = value.shape
        return recon(means["m_rpost"], declared.get("final", ()), B, S, d)

    hooks = [("blocks.0.hook_resid_pre", cap_input)]
    for l in range(L):
        hooks.append((f"blocks.{l}.hook_q_input",
                      lambda v, hook, l=l: ow_qkv(v, hook, l, "q")))
        hooks.append((f"blocks.{l}.hook_k_input",
                      lambda v, hook, l=l: ow_qkv(v, hook, l, "k")))
        hooks.append((f"blocks.{l}.hook_v_input",
                      lambda v, hook, l=l: ow_qkv(v, hook, l, "v")))
        hooks.append((f"blocks.{l}.attn.hook_result",
                      lambda v, hook, l=l: cap_result(v, hook, l)))
        if not attn_only:
            hooks.append((f"blocks.{l}.hook_mlp_in",
                          lambda v, hook, l=l: ow_mlp_in(v, hook, l)))
            hooks.append((f"blocks.{l}.hook_mlp_out",
                          lambda v, hook, l=l: cap_mlp(v, hook, l)))
    hooks.append((f"blocks.{L - 1}.hook_resid_post", ow_final))
    return model.run_with_hooks(tokens, return_type="logits", fwd_hooks=hooks)


def assert_full_declared(model, tokens, means, spec, L, H, attn_only) -> float:
    """The invariant the whole ablation rests on: with EVERY causal edge declared the
    reconstruction reproduces the real model, and the declared-edge universe is exactly the
    scoring denominator. Returns the max |dlogit| gap.
    (The old check severed ZERO edges, which never touches a tensor -- it asserted the
    model equals itself and left this, the real reconstruction path, unmeasured.)"""
    import torch
    declared = full_declared_edges(L, H, attn_only)
    n_declared = sum(len(v) for v in declared.values())
    assert n_declared == MG.live_edges(spec), (
        f"the ablation's edge universe ({n_declared}) != the scoring denominator "
        f"model_registry.live_edges ({MG.live_edges(spec)}) at {L}L x {H}H attn_only={attn_only}")
    heads = {(l, h) for l in range(L) for h in range(H)}
    mlps = set() if attn_only else set(range(L))
    with torch.no_grad():
        gap = (_edge_forward(model, tokens, declared, means, heads, mlps, L, H, attn_only)
               - model(tokens, return_type="logits")).abs().max().item()
    assert gap < FULL_DECLARED_TOL, (
        f"full-declared reconstruction != real model (max|dlogit|={gap:.2e} >= "
        f"{FULL_DECLARED_TOL:g}): the edge-level ablation is not measuring this model")
    return gap


def _edge_sever_forward(model, tokens, declared, means, L, H, attn_only):
    """DUAL of ``_edge_forward``: the DECLARED edges carry the per-position mean
    and everything else stays live — each declared receiver input becomes
    ``live_site − Σ_declared(live_s − mean_s)``. Zero declared edges == full model."""
    live = {"input": None, "head": {}, "mlp": {}}

    def cap_input(value, hook):
        live["input"] = value
        return value

    def cap_result(value, hook, l):
        live["head"][l] = value
        return value

    def cap_mlp(value, hook, l):
        live["mlp"][l] = value
        return value

    def sever(value, decl):
        for sid in decl:
            value = value - _sender_delta(sid, live, means)
        return value

    def ow_qkv(value, hook, l, chan):
        for h in range(H):
            decl = declared.get((l, h, chan))
            if decl:
                value[:, :, h, :] = sever(value[:, :, h, :], decl)
        return value

    def ow_mlp_in(value, hook, l):
        decl = declared.get(("m", l))
        return sever(value, decl) if decl else value

    def ow_final(value, hook):
        decl = declared.get("final")
        return sever(value, decl) if decl else value

    hooks = [("blocks.0.hook_resid_pre", cap_input)]
    for l in range(L):
        for chan in ("q", "k", "v"):
            hooks.append((f"blocks.{l}.hook_{chan}_input",
                          lambda v, hook, l=l, c=chan: ow_qkv(v, hook, l, c)))
        hooks.append((f"blocks.{l}.attn.hook_result",
                      lambda v, hook, l=l: cap_result(v, hook, l)))
        if not attn_only:
            hooks.append((f"blocks.{l}.hook_mlp_in",
                          lambda v, hook, l=l: ow_mlp_in(v, hook, l)))
            hooks.append((f"blocks.{l}.hook_mlp_out",
                          lambda v, hook, l=l: cap_mlp(v, hook, l)))
    hooks.append((f"blocks.{L - 1}.hook_resid_post", ow_final))
    return model.run_with_hooks(tokens, return_type="logits", fwd_hooks=hooks)


def _score_logits(logits, pairs, i0, i1, pos, outs):
    """(#token-correct, #tokens, #seq-correct, Σ logit_diff) over a logits chunk.
    A sequence counts only when ALL its VALID answer positions are correct;
    ``answer_mask`` present -> per-row position gather, masked slots only."""
    from metrics import logit_diff
    mask = None if pairs.answer_mask is None else pairs.answer_mask[i0:i1]
    if mask is None:
        pred = logits[:, pos, :].argmax(-1)
        ok = pred == outs[i0:i1]                  # [chunk, P]
        tok_c, tok_n, seq_c = ok.sum().item(), pred.numel(), ok.all(dim=1).sum().item()
    else:
        # An all-False row would satisfy the 0 == 0 seq test below = a FREE exact-match win.
        # model_registry drops those rows; assert it rather than trust it.
        assert bool((mask.sum(dim=1) >= 1).all()), \
            "a scored row has an EMPTY answer_mask: it would count as a free sequence-accuracy win"
        pred = logits.argmax(-1).gather(1, pairs.answer_positions[i0:i1])
        ok = (pred == outs[i0:i1]) & mask
        tok_c, tok_n = ok.sum().item(), int(mask.sum().item())
        seq_c = (ok.sum(dim=1) == mask.sum(dim=1)).sum().item()
    chunk = SimpleNamespace(answer_positions=pairs.answer_positions[i0:i1],
                            correct=pairs.correct[i0:i1],
                            incorrect=pairs.incorrect[i0:i1],
                            answer_mask=mask)
    return (tok_c, tok_n, seq_c,
            logit_diff(logits, chunk, per_example=True).sum().item())


def _perf(forward, pairs, outs, batch_cap):
    """(token_accuracy, sequence_accuracy, logit_diff) of ``forward`` (a
    tokens->logits callable) over the eval pairs, batched by ``batch_cap``."""
    import torch
    toks = pairs.clean_tokens
    B = toks.shape[0]
    pos = pairs.answer_positions[0]              # shared-position fast path (mask None)
    n_tok_c = n_tok = n_seq_c = 0
    ld_sum = 0.0
    with torch.no_grad():
        for i0 in range(0, B, batch_cap):
            i1 = min(B, i0 + batch_cap)
            logits = forward(toks[i0:i1])
            tc, nt, sc, ld = _score_logits(logits, pairs, i0, i1, pos, outs)
            n_tok_c += tc
            n_tok += nt
            n_seq_c += sc
            ld_sum += ld
    return (n_tok_c / max(1, n_tok), n_seq_c / max(1, B), ld_sum / max(1, B))


# --------------------------------------------------------------------------- #
# Differential mode (--base-circuit): S_union / D_affected / D_unaffected /   #
# D_control for contrast-pair circuits (docs/reference/design.md, Differential mode) #
# --------------------------------------------------------------------------- #


def _union_edges(a: list, b: list) -> list:
    """Union of two edge lists, de-duped on (parent, child, qkv) via _add_edge."""
    out, seen = [], {}
    for e in a + b:
        _add_edge(out, seen, e["parent"], e["child"], e["qkv"], e["score"])
    out.sort(key=lambda e: (e["parent"], e["child"], e["qkv"] or ""))
    return out


def _at(pred, positions):
    """Whole-seq predictions [B,S] gathered at positions ([P] shared | [B,P] per-row)."""
    return pred[:, positions] if positions.ndim == 1 else pred.gather(1, positions)


def _tok_acc(pred, targets, mask=None) -> float:
    """Token accuracy of pred vs targets [B,P] over mask-True slots (None = all)."""
    ok = pred == targets
    if mask is None:
        return ok.sum().item() / max(1, ok.numel())
    return (ok & mask).sum().item() / max(1, int(mask.sum().item()))


def _seq_acc(pred, targets, mask=None) -> float:
    """Whole-sequence exact-match accuracy: every VALID slot of a row correct."""
    ok = pred == targets
    if mask is None:
        return ok.all(dim=1).sum().item() / max(1, ok.shape[0])
    return ((ok & mask).sum(dim=1) == mask.sum(dim=1)).sum().item() / max(1, ok.shape[0])


def _complement_mask(all_positions, affected_positions):
    """[B, P_all] bool: True where a model answer position is NOT in the row's
    affected set (padded slots repeat a real affected position, so no mask needed)."""
    return (all_positions.view(1, -1, 1) != affected_positions.unsqueeze(1)).all(-1)


def _collect_argmax(forward, toks, batch_cap: int):
    """Whole-sequence argmax predictions of a tokens->logits callable: [B,S] on CPU."""
    import torch
    preds = []
    with torch.no_grad():
        for i0 in range(0, toks.shape[0], batch_cap):
            preds.append(forward(toks[i0:i0 + batch_cap]).argmax(-1).cpu())
    return torch.cat(preds)


def _differential_metrics(model, spec, pairs, edges, args, means, L, H,
                          attn_only, device) -> dict:
    """The differential result block for ONE candidate: the contrast circuit is
    scored by what severing it BREAKS on the affected positions, guarded by the
    union sufficiency test and two collateral controls. EVERY metric here is
    WHOLE-SEQUENCE exact match over its position set (``FRINGE_METRIC_TAG``) — the
    unit of the plain gate and of the thresholds taxonomy.json applies to d_*."""
    import torch
    eap = _load_fc_eap()
    base_edges = _load_edges(args.base_edges_json)
    toks, cap = pairs.clean_tokens, args.batch_cap

    # FRINGE = candidate minus base executor: the detector hypothesis. Whole
    # candidates contain the executor outright (top-N attribution ranks the
    # movers highest), so severing THEM fails d_control by construction —
    # the severing knife is the fringe; the union test is set-identical.
    base_keys = {(e["parent"], e["child"], e["qkv"]) for e in base_edges}
    fringe = [e for e in edges if (e["parent"], e["child"], e["qkv"]) not in base_keys]
    f_heads, f_mlps = eap.heads_mlps_from_nodes(
        sorted({e["parent"] for e in fringe} | {e["child"] for e in fringe}))

    union = _union_edges(edges, base_edges)
    u_heads, u_mlps = eap.heads_mlps_from_nodes(
        sorted({e["parent"] for e in union} | {e["child"] for e in union}))
    union_fwd = lambda t: _edge_forward(model, t, _edges_to_declared(union, attn_only), means,
                                        set(u_heads), set(u_mlps), L, H, attn_only)
    sever_fwd = lambda t: _edge_sever_forward(model, t, _edges_to_declared(fringe, attn_only),
                                              means, L, H, attn_only)
    full_fwd = lambda t: model(t, return_type="logits")

    pos_all = torch.tensor(list(spec.answer_positions), dtype=torch.long)
    tgt_all = toks[:, pos_all.to(toks.device) + 1].cpu()   # teacher-forced targets
    aff_pos, tgt_aff = pairs.answer_positions.cpu(), pairs.correct.cpu()
    mask = None if pairs.answer_mask is None else pairs.answer_mask.cpu()
    comp = _complement_mask(pos_all, aff_pos)

    pred_full = _collect_argmax(full_fwd, toks, cap)
    pred_sever = _collect_argmax(sever_fwd, toks, cap)
    s_union = _seq_acc(_at(_collect_argmax(union_fwd, toks, cap), pos_all), tgt_all)
    d_affected = (_seq_acc(_at(pred_full, aff_pos), tgt_aff, mask)
                  - _seq_acc(_at(pred_sever, aff_pos), tgt_aff, mask))
    d_unaffected = (_seq_acc(_at(pred_full, pos_all), tgt_all, comp)
                    - _seq_acc(_at(pred_sever, pos_all), tgt_all, comp))

    # D_control: severing the candidate must not hurt the BASE circuit's own pool.
    bpairs, _ = MG.load_circuit_pairs(args.model, args.base_circuit, n=args.n,
                                      device=device, consumer=SCORING_CONSUMER)
    bmeans = _edge_site_means(model, bpairs.clean_tokens[:args.mean_n], cap)
    bsever = lambda t: _edge_sever_forward(model, t, _edges_to_declared(fringe, attn_only),
                                           bmeans, L, H, attn_only)
    bpos, btgt = bpairs.answer_positions.cpu(), bpairs.correct.cpu()
    bmask = None if bpairs.answer_mask is None else bpairs.answer_mask.cpu()
    d_control = (_seq_acc(_at(_collect_argmax(full_fwd, bpairs.clean_tokens, cap), bpos), btgt, bmask)
                 - _seq_acc(_at(_collect_argmax(bsever, bpairs.clean_tokens, cap), bpos), btgt, bmask))

    return {"mode": "differential", "base_circuit": args.base_circuit,
            "base_edges_sig": _edges_sig(base_edges), "n_union_edges": len(union),
            "sever": "fringe", "metric_unit": FRINGE_METRIC_TAG,
            "fringe_edges": fringe, "fringe_heads": f_heads, "fringe_mlps": f_mlps,
            "n_fringe_edges": len(fringe), "n_fringe_heads": len(f_heads),
            "n_fringe_mlps": len(f_mlps),
            "s_union": s_union, "d_affected": d_affected,
            "d_unaffected": d_unaffected, "d_control": d_control}


def _mode(args) -> str:
    """'suppression' (C3, cue-flip) | 'differential' (contrast) | 'plain'."""
    if getattr(args, "suppression", False):
        return "suppression"
    return "differential" if args.base_circuit else "plain"


def _suppression_metrics(model, spec, pairs, edges, args, means, L, H,
                         attn_only, device) -> dict:
    """The suppression result for ONE candidate (cue-flip pair: clean=cued-cued,
    corrupt=wrong-cued). The suppressor is the FRINGE beyond the cued task's own
    executor; it is scored by how much severing it RELEASES the wrong task on the
    affected positions (B-wins rate rises) while leaving the agree positions intact
    (collateral ~ 0). pairs.correct = cued output A(x); pairs.incorrect = wrong
    output B(x); both aligned to pairs.answer_positions (the disagreeing slots).
    EVERY rate here is WHOLE-SEQUENCE exact match over its position set
    (``FRINGE_METRIC_TAG``): a release counts when ALL the row's affected slots flip to
    the wrong task, the unit the plain gate and taxonomy.json's thresholds assume."""
    import torch
    eap = _load_fc_eap()
    base_edges = _load_edges(args.base_edges_json)
    toks, cap = pairs.clean_tokens, args.batch_cap

    base_keys = {(e["parent"], e["child"], e["qkv"]) for e in base_edges}
    fringe = [e for e in edges if (e["parent"], e["child"], e["qkv"]) not in base_keys]
    f_heads, f_mlps = eap.heads_mlps_from_nodes(
        sorted({e["parent"] for e in fringe} | {e["child"] for e in fringe}))

    sever_fwd = lambda t: _edge_sever_forward(model, t, _edges_to_declared(fringe, attn_only),
                                              means, L, H, attn_only)
    full_fwd = lambda t: model(t, return_type="logits")

    pos_all = torch.tensor(list(spec.answer_positions), dtype=torch.long)
    aff_pos = pairs.answer_positions.cpu()          # [B,P] disagreeing (affected) slots
    a_tgt, b_tgt = pairs.correct.cpu(), pairs.incorrect.cpu()   # A(x), B(x) at those slots
    mask = None if pairs.answer_mask is None else pairs.answer_mask.cpu()
    comp = _complement_mask(pos_all, aff_pos)       # non-affected (A==B) model positions
    tgt_all = toks[:, pos_all.to(toks.device) + 1].cpu()

    pred_full = _collect_argmax(full_fwd, toks, cap)
    pred_sever = _collect_argmax(sever_fwd, toks, cap)
    full_aff, sev_aff = _at(pred_full, aff_pos), _at(pred_sever, aff_pos)

    b_full = _seq_acc(full_aff, b_tgt, mask)        # wrong-task wins, full model (~0)
    b_sev = _seq_acc(sev_aff, b_tgt, mask)          # wrong-task wins, suppressor severed
    a_full = _seq_acc(full_aff, a_tgt, mask)        # cued task correct, full (~1)
    a_sev = _seq_acc(sev_aff, a_tgt, mask)          # cued task, suppressor severed (drops = release)
    release_rate = b_sev - b_full                   # > 0: severing releases the wrong answer
    collateral = (_seq_acc(_at(pred_full, pos_all), tgt_all, comp)
                  - _seq_acc(_at(pred_sever, pos_all), tgt_all, comp))

    # E3 tonic reference: a cue-CONSTANT suppressor equals its own cued-row mean, so the
    # cued-reference sever above is a no-op on it (release == 0 blind spot). Sever again
    # against WRONG-CUE (corrupt-row) means, where a tonic suppressor's mean differs.
    means_tonic = _edge_site_means(model, pairs.corrupted_tokens[:args.mean_n], cap)
    sever_tonic = lambda t: _edge_sever_forward(model, t, _edges_to_declared(fringe, attn_only),
                                                means_tonic, L, H, attn_only)
    pred_sever_t = _collect_argmax(sever_tonic, toks, cap)
    sev_aff_t = _at(pred_sever_t, aff_pos)
    b_sev_t = _seq_acc(sev_aff_t, b_tgt, mask)
    a_sev_t = _seq_acc(sev_aff_t, a_tgt, mask)
    collateral_tonic = (_seq_acc(_at(pred_full, pos_all), tgt_all, comp)
                        - _seq_acc(_at(pred_sever_t, pos_all), tgt_all, comp))

    return {"mode": "suppression", "base_circuit": args.base_circuit,
            "base_edges_sig": _edges_sig(base_edges), "sever": "fringe",
            "tonic_reference": TONIC_REFERENCE_TAG, "metric_unit": FRINGE_METRIC_TAG,
            "fringe_edges": fringe, "fringe_heads": f_heads, "fringe_mlps": f_mlps,
            "n_fringe_edges": len(fringe), "n_fringe_heads": len(f_heads),
            "n_fringe_mlps": len(f_mlps),
            "release_rate": release_rate, "b_wins_full": b_full,
            "b_wins_severed": b_sev, "a_correct_full": a_full,
            "a_correct_severed": a_sev, "collateral": collateral,
            "release_rate_tonic": b_sev_t - b_full,
            "b_wins_severed_tonic": b_sev_t, "a_correct_severed_tonic": a_sev_t,
            "collateral_tonic": collateral_tonic}


def _recovery(x: float, full: float, empty: float):
    """R = (x - empty)/(full - empty), or the ZERO_CONTRAST marker when the denominator
    vanishes. The marker propagates: _passes REJECTS a non-numeric R, so a scope whose
    fully-severed model already matches the full model can never mint a winner."""
    try:
        return MG.recovery_ratio(x, full, empty)
    except MG.ZeroContrastError:
        return ZERO_CONTRAST


def _fmt_r(v) -> str:
    """R_* for a log line: 4dp when numeric, the marker verbatim when not."""
    return f"{v:.4f}" if isinstance(v, (int, float)) else str(v)


def eval_child(args) -> int:
    """--eval-child entry: score ONE candidate circuit; write JSON to --out."""
    spec = MG.get_spec(args.model)
    MG.apply_gpu_memory_cap()                    # also applied inside load_model
    model = MG.load_model(args.model)
    device = str(next(model.parameters()).device)

    pairs, ds_path = MG.load_circuit_pairs(args.model, args.circuit, n=args.n,
                                           device=device, consumer=SCORING_CONSUMER)
    # Per-position correct answers ALIGNED to pairs.answer_positions (width P): equals task_fn(clean_inputs) for the whole circuit; for a position-partition circuit (even/odd subset) the loader already subset both answer_positions AND pairs.correct, so this stays consistent.
    outs = pairs.correct.to(pairs.clean_tokens.device)

    attn_only = bool(model.cfg.attn_only)

    # ---- EDGE-LEVEL per-position mean ablation -----------------------------
    # Expose the per-head q/k/v inputs (use_split_qkv_input), per-head outputs
    # (use_attn_result), and per-layer MLP input (use_hook_mlp_in) the
    # reconstruction reads/overwrites.
    model.cfg.use_attn_result = True
    model.cfg.use_split_qkv_input = True
    if not attn_only:
        model.cfg.use_hook_mlp_in = True
    L, H = int(model.cfg.n_layers), int(model.cfg.n_heads)

    keep = _parse_heads_arg(args.heads)
    circuit_heads = set(keep)
    keep_mlps = [] if attn_only else _parse_mlps_arg(args.mlps)
    circuit_mlps = set(keep_mlps)

    edges = _load_edges(args.edges_json)         # the circuit's declared edges
    declared = _edges_to_declared(edges, attn_only)
    # Mean-ablation reference = the first mean_n rows of the SAME eval clean tokens.
    means = _edge_site_means(model, pairs.clean_tokens[:args.mean_n],
                             args.batch_cap)
    gap = assert_full_declared(model, pairs.clean_tokens[:min(32, pairs.clean_tokens.shape[0])],
                               means, spec, L, H, attn_only)

    def ablated(decl):
        return lambda t: _edge_forward(model, t, decl, means, circuit_heads,
                                       circuit_mlps, L, H, attn_only)

    t0 = time.time()
    # full = the real model (every causal edge declared -> corrections vanish);
    # circuit = the candidate's declared edges; empty = no edges (every receiver
    # input is the pure positional mean).
    full_acc, full_seq, full_ld = _perf(lambda t: model(t, return_type="logits"),
                                        pairs, outs, args.batch_cap)
    circ_acc, circ_seq, circ_ld = _perf(ablated(declared), pairs, outs,
                                        args.batch_cap)
    empty_acc, empty_seq, empty_ld = _perf(ablated({}), pairs, outs,
                                           args.batch_cap)

    result = {
        "name": args.name,
        "heads": [f"L{l}H{h}" for (l, h) in keep],
        "n_heads": len(keep),
        "ablation": EDGE_ABLATION_TAG,           # invalidates legacy head-level caches
        "n_edges": len(edges),
        "edges_sig": _edges_sig(edges),
        # MLP fields (attn+MLP models only): kept MLP nodes + total node count.
        # OMITTED for attention-only models so their eval JSON keeps the legacy
        # head-only shape.
        **({"mlps": [f"m{l}" for l in keep_mlps],
            "n_mlps": len(keep_mlps),
            "n_nodes": len(keep) + len(keep_mlps)} if not attn_only else {}),
        "n_examples": int(pairs.clean_tokens.shape[0]),
        "n_requested": args.n,
        # The pool REGIME + the disjoint row WINDOW this stage scored on: v1 numbers (no
        # answer_mask, every consumer reading rows [0:n]) are not comparable to these, and
        # the cache key must see the difference (same path, same n, different rows).
        "pairs_regime": pairs.extra["regime"],
        "pairs_window": pairs.extra["window"],
        "pairs_consumer": pairs.extra["consumer"],
        "n_agreeing_rows_dropped": pairs.extra["n_agreeing_rows_dropped"],
        "mean_n": args.mean_n,
        "checkpoint": spec.checkpoint,
        "checkpoint_sig": _stat_sig(spec.checkpoint),
        "dataset_path": ds_path,
        "dataset_sig": _content_sig(ds_path),
        "full_declared_gap": gap,
        "perf_full": {"accuracy": full_acc, "sequence_accuracy": full_seq,
                      "logit_diff": full_ld},
        "perf_circuit": {"accuracy": circ_acc, "sequence_accuracy": circ_seq,
                         "logit_diff": circ_ld},
        "perf_empty": {"accuracy": empty_acc, "sequence_accuracy": empty_seq,
                       "logit_diff": empty_ld},
        "R_accuracy": _recovery(circ_acc, full_acc, empty_acc),
        "R_sequence_accuracy": _recovery(circ_seq, full_seq, empty_seq),
        "R_logit_diff": _recovery(circ_ld, full_ld, empty_ld),
        "elapsed_sec": time.time() - t0,
    }
    if args.base_edges_json:                     # differential / suppression (--base-circuit)
        metrics_fn = (_suppression_metrics if _mode(args) == "suppression"
                      else _differential_metrics)
        result.update(metrics_fn(model, spec, pairs, edges, args,
                                 means, L, H, attn_only, device))
        result["elapsed_sec"] = time.time() - t0
    _dump_json(result, args.out)
    mlp_note = "" if attn_only else f" mlps={len(keep_mlps)}"
    print(f"[eval-child] {args.name}: heads={len(keep)}{mlp_note} "
          f"R_seq={_fmt_r(result['R_sequence_accuracy'])} "
          f"R_acc={_fmt_r(result['R_accuracy'])} R_ld={_fmt_r(result['R_logit_diff'])} "
          f"(full_seq={full_seq:.4f} circ_seq={circ_seq:.4f} "
          f"empty_seq={empty_seq:.4f}; full-declared gap={gap:.2e}; "
          f"pool {pairs.extra['regime']} rows {pairs.extra['window']})")
    if args.base_edges_json and _mode(args) == "suppression":
        print(f"[eval-child] {args.name} suppression vs {args.base_circuit}: "
              f"release={result['release_rate']:+.4f} "
              f"tonic={result['release_rate_tonic']:+.4f} "
              f"(B-wins {result['b_wins_full']:.3f}->{result['b_wins_severed']:.3f}) "
              f"collateral={result['collateral']:.4f}/{result['collateral_tonic']:.4f}")
    elif args.base_edges_json:
        print(f"[eval-child] {args.name} differential vs {args.base_circuit}: "
              f"s_union={result['s_union']:.4f} d_aff={result['d_affected']:.4f} "
              f"d_unaff={result['d_unaffected']:.4f} d_ctl={result['d_control']:.4f}")
    return 0


# --------------------------------------------------------------------------- #
# PARENT: subprocess scheduler (jobs <= 2, headroom assert before EVERY spawn) #
# --------------------------------------------------------------------------- #


def _child_env(jobs: int):
    """Inherited env + per-child BLAS caps (the circuit_discovery/main.py
    run_parallel pattern -- keeps concurrent children from oversubscribing). The
    parent's own budget (an inherited OMP_NUM_THREADS, else all cores) is SPLIT
    across the ``jobs`` children, never inherited whole by each."""
    env = os.environ.copy()
    budget = int(os.environ.get("OMP_NUM_THREADS") or os.cpu_count() or 4)
    threads = str(max(1, budget // max(1, jobs)))
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS"):
        env[var] = threads
    return env


def _eval_cache_path(work_dir: str, name: str) -> str:
    return os.path.join(work_dir, "evals", f"{name}.json")


def _eval_config(cand: Candidate, args, ckpt: str) -> dict:
    """The config a cached eval must match to be reused: each key equals the
    SAME-NAMED field of the child's result JSON. The kept-MLP set is part of the
    match ONLY for candidates that keep MLP nodes (attn+MLP models). The dataset
    CONTENT hash and the checkpoint (mtime,size) are part of the match because both
    paths are regenerated in place — a path-only key calls a stale cache fresh."""
    ds = MG.circuit_dataset_path(args.model, args.circuit)
    n = SMOKE_EVAL_N if args.smoke else SMALLEST_EVAL_N
    off = MG.POOL_WINDOWS[SCORING_CONSUMER]
    cfg = {"checkpoint": ckpt,
           "checkpoint_sig": _stat_sig(ckpt),
           "dataset_path": ds,
           "dataset_sig": _content_sig(ds),
           "n_requested": n,
           # v1 caches (unmasked pools, rows [0:n]) score DIFFERENT rows under DIFFERENT
           # metrics at the same path+n, so the regime and the window are part of the key.
           "pairs_regime": MG.CANON_PAIRS_REGIME,
           "pairs_window": [off, off + n],
           "pairs_consumer": SCORING_CONSUMER,
           "mean_n": MEAN_N_SMOKE if args.smoke else MEAN_N_PROD,
           "ablation": EDGE_ABLATION_TAG,        # head-level caches never match
           "edges_sig": _edges_sig(cand.edges),  # re-discovered edges -> recompute
           "heads": cand.head_labels()}
    if cand.mlps:
        cfg["mlps"] = cand.mlp_labels()
    if args.base_circuit:                        # a new base winner invalidates the cache
        cfg["base_circuit"] = args.base_circuit
        cfg["base_edges_sig"] = args.base_edges_sig
        cfg["sever"] = "fringe"                  # severing-semantics change -> recompute
        cfg["mode"] = _mode(args)                # differential vs suppression: distinct metrics
        cfg["metric_unit"] = FRINGE_METRIC_TAG   # token-accuracy caches never match
        if _mode(args) == "suppression":         # pre-tonic caches lack release_rate_tonic
            cfg["tonic_reference"] = TONIC_REFERENCE_TAG
    return cfg


def _cached_eval(cand: Candidate, args, ckpt: str, work_dir: str):
    path = _eval_cache_path(work_dir, cand.name)
    if not _json_ok(path):
        return None
    with open(path) as f:
        res = json.load(f)
    want = _eval_config(cand, args, ckpt)
    if "R_sequence_accuracy" not in res or any(k not in res for k in want):
        return None                              # legacy cache schema: recompute
    return res if {k: res[k] for k in want} == want else None


def _spawn_eval(cand: Candidate, args, work_dir: str, jobs: int):
    """Headroom-assert (visible, EVERY spawn) then launch one --eval-child."""
    out = _eval_cache_path(work_dir, cand.name)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    log = os.path.join(work_dir, "logs", f"{cand.name}.log")
    os.makedirs(os.path.dirname(log), exist_ok=True)
    # The candidate's DECLARED EDGES drive the edge-level ablation — serialize
    # them next to the eval cache and hand the child their path.
    edges_path = os.path.join(work_dir, "edges", f"{cand.name}.json")
    os.makedirs(os.path.dirname(edges_path), exist_ok=True)
    with open(edges_path, "w") as f:
        json.dump(cand.edges, f)
    gib = MG.wait_for_headroom(
        context=f"smallest_faithful_circuit spawning eval-child for {cand.name}",
        retries=MAX_OOM_RETRIES, sleep_s=OOM_SLEEP_S)
    mlp_note = f" + {cand.n_mlps} mlps" if cand.mlps else ""
    print(f"    [headroom OK: {gib:.1f} GiB free] spawning eval-child "
          f"{cand.name} ({cand.n_heads} heads{mlp_note}, {len(cand.edges)} edges)")
    cmd = [PY, os.path.abspath(__file__), "--eval-child",
           "--model", args.model, "--circuit", args.circuit, "--name", cand.name,
           "--heads", ",".join(cand.head_labels()),
           "--edges-json", edges_path,
           "--n", str(SMOKE_EVAL_N if args.smoke else SMALLEST_EVAL_N),
           "--mean-n", str(MEAN_N_SMOKE if args.smoke else MEAN_N_PROD),
           "--batch-cap", str(BATCH_CAP_SMOKE if args.smoke else BATCH_CAP_PROD),
           "--out", out]
    if cand.mlps:                                # attn+MLP candidate keep-set
        cmd += ["--mlps", ",".join(cand.mlp_labels())]
    if args.base_circuit:                        # differential / suppression flags for the child
        cmd += ["--base-circuit", args.base_circuit,
                "--base-edges-json", args.base_edges_json]
        if _mode(args) == "suppression":
            cmd += ["--suppression"]
    lf = open(log, "w")
    proc = subprocess.Popen(cmd, cwd=REPO_ROOT, env=_child_env(jobs),
                            stdout=lf, stderr=subprocess.STDOUT)
    return {"cand": cand, "proc": proc, "out": out, "log": log, "lf": lf}


def run_evaluations(cands: list, args, ckpt: str, work_dir: str) -> dict:
    """Evaluate every candidate, one SUBPROCESS each, <= args.jobs concurrent.
    Skip-if-done via the per-candidate eval cache. Fails LOUD on worker errors
    (except OOM-ish failures, which get sleep-60 re-spawns, max 5 per cand)."""
    jobs = max(1, args.jobs)
    results, queue, running = {}, list(cands), []
    attempts = {c.name: 0 for c in cands}
    while queue or running:
        while queue and len(running) < jobs:
            cand = queue.pop(0)
            if not args.force:
                cached = _cached_eval(cand, args, ckpt, work_dir)
                if cached is not None:
                    print(f"    [skip-if-done] {cand.name}: cached eval reused "
                          f"(R_seq={_fmt_r(cached['R_sequence_accuracy'])})")
                    results[cand.name] = cached
                    continue
            attempts[cand.name] += 1
            running.append(_spawn_eval(cand, args, work_dir, jobs))
        if not running:
            continue
        time.sleep(2.0)
        still = []
        for r in running:
            if r["proc"].poll() is None:
                still.append(r)
                continue
            r["lf"].close()
            rc = r["proc"].returncode
            name = r["cand"].name
            with open(r["log"]) as f:
                log_txt = f.read()
            if rc == 0 and _json_ok(r["out"]):
                with open(r["out"]) as f:
                    res = json.load(f)
                results[name] = res
                print(f"    [eval done] {name}: heads={res['n_heads']} "
                      f"R_seq={_fmt_r(res['R_sequence_accuracy'])} "
                      f"R_acc={_fmt_r(res['R_accuracy'])} "
                      f"R_ld={_fmt_r(res['R_logit_diff'])} "
                      f"({res['elapsed_sec']:.0f}s)")
                continue
            if _ZERO_CONTRAST_MARKER in log_txt:
                raise ZeroContrastDataset(
                    f"{name}: the circuit's dataset has no clean/corrupt contrast at any "
                    f"scored position, so nothing here is measurable; log {r['log']}")
            oomish = any(m in log_txt.lower() for m in _OOM_MARKERS)
            if oomish and attempts[name] < MAX_OOM_RETRIES:
                print(f"    [eval OOM-ish failure] {name} (exit {rc}, attempt "
                      f"{attempts[name]}/{MAX_OOM_RETRIES}); sleeping "
                      f"{OOM_SLEEP_S}s then re-queueing")
                time.sleep(OOM_SLEEP_S)
                queue.insert(0, r["cand"])
                continue
            tail = "\n".join(log_txt.strip().splitlines()[-25:])
            raise RuntimeError(
                f"eval-child for {name} failed (exit {rc}); log tail "
                f"({r['log']}):\n{tail}")
        running = still
    return results


# --------------------------------------------------------------------------- #
# Selection + (optional) refinement between adjacent candidate sizes          #
# --------------------------------------------------------------------------- #


def _n_nodes(res) -> int:
    """Total kept-node count of an eval result = heads + MLPs. For
    attention-only models the result has no ``n_nodes`` field and this falls
    back to ``n_heads`` -> the size used for ranking/selection is UNCHANGED
    (node count == head count when there are no MLPs)."""
    return res.get("n_nodes", res["n_heads"])


def _fringe_nodes(res: dict, attn_only: bool) -> int:
    """Kept fringe NODE count. On attn-only models the m{l} fringe nodes are phantom
    (EAP-IG graph artefacts), and counting them pushed nodes/full_heads over 1 -> a
    NEGATIVE score that sank real suppressors to the bottom of the ranking."""
    return res["n_fringe_heads"] + (0 if attn_only else res["n_fringe_mlps"])


def _passes(res, target, relative_floor: float = DEFAULT_RELATIVE_FLOOR) -> bool:
    """Selection eligibility, ALL of:
      * a non-empty circuit (>=1 kept NODE = head or MLP),
      * ABSOLUTE per-sequence exact-match accuracy under edge-level mean ablation
        (perf_circuit.sequence_accuracy) >= target,
      * a NUMERIC relative recovery R_sequence_accuracy >= relative_floor, so a scope whose
        checkpoint is itself near the floor cannot mint a "faithful" circuit, and a
        ZERO_CONTRAST scope (R undefined) is unscorable rather than eligible,
      * the EMPTY baseline (perf_empty.sequence_accuracy, every edge severed) BELOW
        target: when the fully mean-ablated model already clears the floor the circuit
        is credited for a constant output it does not compute.
    Among the eligible candidates the winner is the MAXIMUM selection score (see
    _selection_score / pick_winner) -- not the smallest one."""
    r_seq = res["R_sequence_accuracy"]
    return (_n_nodes(res) > 0
            and res["perf_circuit"]["sequence_accuracy"] >= target
            and isinstance(r_seq, (int, float)) and r_seq >= relative_floor
            and res["perf_empty"]["sequence_accuracy"] < target)


#: Smallest-most-faithful SELECTION SCORE (product of complements: small in BOTH heads and edges); see docs/reference/design.md. Changing this string invalidates old outputs via the _output_done match.
SELECTION_STRATEGY = ("(1 - n_nodes_circuit/n_nodes_full) * (1 - n_edges_circuit/n_edges_full) "
                      "with n_edges_full = model_registry.live_edges(spec) (geometry, never a "
                      "finder artifact); "
                      "eligible iff abs whole-seq acc >= target AND a NUMERIC "
                      "R_sequence_accuracy >= relative_floor AND the empty (fully severed) "
                      "baseline is below target; "
                      "ties -> fewer edges, fewer nodes, higher accuracy, name")

#: Differential-mode SELECTION SCORE (contrast-pair circuits): necessity-weighted smallness. A NEW string so _output_done auto-invalidates non-differential outputs.
# d_affected/d_unaffected/d_control are MEASUREMENTS, not gates (by design):
# classification (necessary/redundant, local/global) lives in models/taxonomy.json.
SELECTION_STRATEGY_DIFFERENTIAL = (
    "fringe = candidate edges minus base edges; severing tests sever the FRINGE; "
    "all metrics are WHOLE-SEQUENCE exact match over their position set; "
    "(1 + clamp(d_affected,0,1)) * (1 - (n_fringe_nodes/n_nodes_full)*(n_fringe_edges/n_edges_full)); "
    "eligible iff s_union >= target and fringe non-empty; d_* reported, not gated; "
    "unfaithful base = best-available fallback")


#: Suppression-mode SELECTION SCORE (C3, cue-flip contrast): release-weighted smallness. A NEW string so _output_done auto-invalidates non-suppression outputs.
SELECTION_STRATEGY_SUPPRESSION = (
    "fringe = candidate edges minus cued-task base edges; severing tests sever the FRINGE; "
    "release_rate = wrong-task-wins(sever) - wrong-task-wins(full) on affected positions, "
    "WHOLE-SEQUENCE exact match (every affected slot flips) like every other rate here; "
    "sever references: cued-row means (release_rate) AND wrong-cue-row means (release_rate_tonic, "
    "catches cue-constant tonic suppressors that equal their own cued mean); "
    "clamp(release,0,1) * (1 - (n_fringe_nodes/n_nodes_full)*(n_fringe_edges/n_edges_full)) where "
    "release is the reference that PASSED (larger when both; recorded as release_reference); "
    "eligible iff fringe non-empty and EITHER release_rate >= release_floor with collateral <= epsilon "
    "OR release_rate_tonic >= release_floor with collateral_tonic <= epsilon; "
    "unfaithful base = best-available fallback")


def _release_reference(res: dict, release_floor: float, epsilon: float) -> str | None:
    """The sever reference that qualifies this fringe as a suppressor -- 'cued' or 'tonic',
    the LARGER release when both pass -- or None when neither does."""
    ok = [(r, name) for (name, r, c) in
          (("cued", res["release_rate"], res["collateral"]),
           ("tonic", res["release_rate_tonic"], res["collateral_tonic"]))
          if r >= release_floor and c <= epsilon]
    return max(ok)[1] if ok else None


def _scored_release(res: dict, release_floor: float, epsilon: float) -> tuple[str, float]:
    """(reference, release) the score reads: the PASSING reference, else the larger release
    (rank-only, for the table). Scoring the CUED rate unconditionally was scoring a number
    every earlier suppression winner failed on -- all of them passed on tonic."""
    ref = _release_reference(res, release_floor, epsilon)
    if ref is None:
        ref = "cued" if res["release_rate"] >= res["release_rate_tonic"] else "tonic"
    return ref, (res["release_rate"] if ref == "cued" else res["release_rate_tonic"])


def _suppression_score(res: dict, full_heads: int, full_edges: int, attn_only: bool = False,
                       release_floor: float = DEFAULT_RELEASE_FLOOR,
                       epsilon: float = DEFAULT_EPSILON) -> float:
    """Release dominates (bounded [0,1]); smaller suppressor wins ties of equal release."""
    if full_heads < 1 or full_edges < 1:
        return 0.0
    release = _scored_release(res, release_floor, epsilon)[1]
    return (min(1.0, max(0.0, release))
            * (1.0 - (_fringe_nodes(res, attn_only) / full_heads)
               * (res["n_fringe_edges"] / full_edges)))


def _passes_suppression(res: dict, release_floor: float, epsilon: float) -> bool:
    """Suppression eligibility: a non-empty fringe whose severing RELEASES the wrong
    task without disturbing the agree positions, under EITHER sever reference —
    cued-row means (release_rate) or wrong-cue tonic means (release_rate_tonic, E3);
    the collateral cap applies to whichever reference passed."""
    return (res["n_fringe_edges"] > 0
            and _release_reference(res, release_floor, epsilon) is not None)


def pick_winner_suppression(results: dict, release_floor: float, epsilon: float,
                            full_heads: int, full_edges: int,
                            attn_only: bool = False) -> dict | None:
    """MAX suppression score among eligible candidates. Ties -> higher release,
    then smaller fringe, then name. None when none qualifies."""
    elig = [r for r in results.values() if _passes_suppression(r, release_floor, epsilon)]
    if not elig:
        return None
    return sorted(elig, key=lambda r: (
        -_suppression_score(r, full_heads, full_edges, attn_only, release_floor, epsilon),
        -_scored_release(r, release_floor, epsilon)[1],
        r["n_fringe_edges"], r["name"]))[0]


def _selection_score(res: dict, n_edges: int, full_heads: int, full_edges: int) -> float:
    """(1 - nodes/nodes_full) * (1 - edges/edges_full); nodes == heads on attn-only."""
    if full_heads < 1 or full_edges < 1:
        return 0.0
    return (1.0 - _n_nodes(res) / full_heads) * (1.0 - n_edges / full_edges)


def _differential_score(res: dict, full_heads: int, full_edges: int,
                        attn_only: bool = False) -> float:
    """Size dominates; necessity is a bounded bonus (at most 2x, never a gatekeeper)."""
    if full_heads < 1 or full_edges < 1:
        return 0.0
    return ((1.0 + min(1.0, max(0.0, res["d_affected"])))
            * (1.0 - (_fringe_nodes(res, attn_only) / full_heads)
               * (res["n_fringe_edges"] / full_edges)))


def _passes_differential(res: dict, target: float) -> bool:
    """Differential eligibility: a NON-EMPTY fringe whose union reaches the floor.
    The d_* severing metrics are reported, never gated."""
    return res["n_fringe_edges"] > 0 and res["s_union"] >= target


def pick_winner_differential(results: dict, target: float, full_heads: int, full_edges: int,
                             attn_only: bool = False) -> dict | None:
    """MAX differential score among eligible candidates. Ties -> higher
    d_affected, then smaller fringe, then name. None when none qualifies."""
    elig = [r for r in results.values()
            if _passes_differential(r, target)]
    if not elig:
        return None
    return sorted(elig, key=lambda r: (-_differential_score(r, full_heads, full_edges, attn_only),
                                       -r["d_affected"], r["n_fringe_edges"], r["name"]))[0]


def pick_winner(results: dict, target: float, edge_counts: dict, full_heads: int,
                full_edges: int,
                relative_floor: float = DEFAULT_RELATIVE_FLOOR) -> dict | None:
    """Smallest-most-faithful = MAX selection score among the candidates _passes accepts.
    Ties -> FEWER EDGES, then fewer nodes, then higher abs seq-acc, then name (the old
    -acc-first key let the LARGER of two score-tied candidates win). None when no
    candidate qualifies. ``edge_counts`` is indexed, never .get-defaulted: a result with
    no parsed candidate used to score 0 edges = maximal score = an unbuildable winner."""
    elig = [r for r in results.values() if _passes(r, target, relative_floor)]
    if not elig:
        return None
    return sorted(elig, key=lambda r: (-_selection_score(r, edge_counts[r["name"]], full_heads, full_edges),
                                       edge_counts[r["name"]],
                                       _n_nodes(r),
                                       -r["perf_circuit"]["sequence_accuracy"],
                                       r["name"]))[0]


def _input_data_description(model: str, circuit) -> str:
    """Human description of the discovery input distribution for (model, circuit), from human_circuit_finding.json."""
    cf = os.path.join(PIPELINE_ROOT, "models", "human_circuit_finding.json")
    for r in json.load(open(cf)).get("requests", []):
        if r.get("model") == model and r.get("slug") == circuit:
            detail = r.get("isolate_by", {}).get("detail", "")
            return f"random digit sequences restricted so that {detail}: {r.get('isolate','')}".strip()
    return "random digit sequences"


def _contrast_data_description(base_circuit: str) -> str:
    return (f"minimal-edit clean/corrupt contrast pairs (differential vs base "
            f"{base_circuit!r}; per-pair masks recomputed at load)")


def _suppression_data_description(base_circuit: str) -> str:
    return (f"cue-flip pairs (clean cued task vs corrupt wrong task, shared payload; "
            f"suppressor = fringe beyond cued executor {base_circuit!r} whose severing "
            f"releases the wrong task; per-pair masks = disagreeing positions)")


def _param_of(name: str):
    """('acdc', tau) | ('eap', topn) | None from a candidate dir name."""
    m = re.fullmatch(r"acdc_thresh_([0-9.]+)", name)
    if m:
        return "acdc", float(m.group(1))
    m = re.fullmatch(r"eap_top(\d+)", name)
    if m:
        return "eap", int(m.group(1))
    return None


def refine_proposals(results: dict, passes) -> dict:
    """Midpoint parameters between the smallest PASSING candidate (per the
    ``passes`` predicate) and its FAILING neighbor, per method
    (min_faithful refine_proposals:301)."""
    props = {"taus": [], "topns": []}
    fams = {"acdc": [], "eap": []}
    for r in results.values():
        p = _param_of(r["name"])
        if p:
            fams[p[0]].append((p[1], r))
    acdc = sorted(fams["acdc"], key=lambda t: t[0])
    passing = [p for p, r in acdc if passes(r)]
    if passing:
        t_pass = max(passing)                    # largest tau that passes
        bigger = [p for p, r in acdc if p > t_pass]
        if bigger:
            mid = float("%.2g" % math.sqrt(t_pass * min(bigger)))
            if all(abs(mid - p) > 1e-12 for p, _ in acdc):
                props["taus"].append(mid)
    eap = sorted(fams["eap"], key=lambda t: t[0])
    passing = [p for p, r in eap if passes(r)]
    if passing:
        n_pass = min(passing)                    # smallest topn that passes
        smaller = [p for p, r in eap if p < n_pass]
        if smaller:
            mid = (n_pass + max(smaller)) // 2
            if max(smaller) < mid < n_pass:
                props["topns"].append(mid)
    return props


def _run_discovery(method: str, value, args, jobs: int) -> bool:
    """Spawn one discovery runner (acdc/eap) for ONE midpoint parameter --
    subprocess-only, serial, headroom-asserted. Non-fatal on failure."""
    script = os.path.join(_THIS, f"{method}.py")
    cmd = [PY, script, "--model", args.model, "--circuit", args.circuit]
    cmd += (["--taus", "%g" % value] if method == "acdc"
            else ["--topns", str(int(value))])
    MG.wait_for_headroom(context=f"refinement discovery {method} {value}",
                         retries=MAX_OOM_RETRIES, sleep_s=OOM_SLEEP_S)
    print(f"    [refine] running: {' '.join(cmd[1:])}")
    r = subprocess.run(cmd, cwd=REPO_ROOT, env=_child_env(jobs),
                       capture_output=True, text=True)
    if r.returncode != 0:
        tail = "\n".join((r.stdout + "\n" + r.stderr).strip().splitlines()[-20:])
        print(f"    BLOCKED(non-fatal): refinement discovery failed "
              f"(exit {r.returncode}):\n{tail}")
        return False
    return True


def _run_eap_search(args) -> None:
    """--search: drive eap.py's ONE-attribution --target-seq-acc search to emit the EAP
    candidate MAXIMIZING (1-h/H)(1-e/E) above the --target floor, so the normal selection
    below writes the canonical smallest_circuit.json from it. EAP is the default (only)
    finder; any opt-in acdc/edge_pruning candidates already on disk join the same selection
    unchanged (the (1-h/H)(1-e/E) > target rule is method-agnostic)."""
    cmd = [PY, os.path.join(_THIS, "eap.py"), "--model", args.model,
           "--circuit", args.circuit, "--target-seq-acc", "%g" % args.target]
    if args.search_n:
        cmd += ["--search-n", str(int(args.search_n))]
    if args.smoke:
        cmd += ["--smoke"]
    MG.wait_for_headroom(context=f"--search eap discovery ({args.model}/{args.circuit})",
                         retries=MAX_OOM_RETRIES, sleep_s=OOM_SLEEP_S)
    print(f"    [search] {' '.join(cmd[1:])}")
    r = subprocess.run(cmd, cwd=REPO_ROOT, env=_child_env(args.jobs),
                       capture_output=True, text=True)
    for line in r.stdout.splitlines():                  # echo the search table + winner line
        if any(k in line for k in ("top-n search", "WINNER", "winner:", "PASS", "NO top-n")):
            print("    " + line.rstrip())
    if r.returncode != 0:
        tail = "\n".join((r.stdout + "\n" + r.stderr).strip().splitlines()[-25:])
        raise SystemExit(f"--search: eap --target-seq-acc failed (exit {r.returncode}):\n{tail}")


def refine(results: dict, args, ckpt: str, work_dir: str, canonical_dir: str) -> None:
    """Up to --refine-rounds rounds of midpoint discovery+evaluation. Only in
    production against the canonical circuits_data/<m>/circuits tree."""
    passes = {"suppression": lambda r: _passes_suppression(r, args.release_floor, args.epsilon),
              "differential": lambda r: _passes_differential(r, args.target),
              "plain": lambda r: _passes(r, args.target, args.relative_floor)}[_mode(args)]
    for rnd in range(args.refine_rounds):
        props = refine_proposals(results, passes)
        todo = ([("acdc", t, f"acdc_thresh_{'%g' % t}") for t in props["taus"]]
                + [("eap", n, f"eap_top{n}") for n in props["topns"]])
        todo = [(m, v, nm) for (m, v, nm) in todo if nm not in results]
        if not todo:
            break
        print(f"== refinement round {rnd + 1}: {[nm for _, _, nm in todo]} ==")
        for method, value, name in todo:
            sub = os.path.join(canonical_dir, name)
            if not os.path.isdir(sub):            # skip-if-done on discovery too
                if not _run_discovery(method, value, args, jobs=1):
                    return
            cand = _sniff_candidate_dir(sub, name, args.attn_only)
            if cand is None:
                print(f"    BLOCKED(non-fatal): {name} discovery left no "
                      "parseable circuit JSON; skipping")
                continue
            results.update(run_evaluations([cand], args, ckpt, work_dir))


# --------------------------------------------------------------------------- #
# Rendering (the drawer from find_circuits/eap.py)                             #
# --------------------------------------------------------------------------- #


def render_winner(cand_heads: list, edges: list, out_png: str, out_gv: str,
                  seed: int = 0, source_name: str = None) -> None:
    """Render via eap.draw_circuit_png. ``source_name`` is the winning circuit's
    name; when it names an Edge Pruning circuit, edges render at the drawer's
    uniform width instead of the attribution scaling -- the gate score (~0.8-1.0)
    would otherwise pin every edge at the penwidth ceiling."""
    drawer = _load_fc_eap()
    uniform = (drawer.EDGE_PRUNING_PENWIDTH
               if str(source_name or "").startswith("edge_pruning") else None)
    nodes = sorted({e["parent"] for e in edges} | {e["child"] for e in edges}
                   | {f"a{l}.h{h}" for (l, h) in cand_heads})
    tuples = [(e["parent"], e["child"], e["qkv"], e["score"]) for e in edges]
    os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(out_gv)), exist_ok=True)
    drawer.draw_circuit_png(nodes, tuples, out_png, out_gv, seed=seed,
                            uniform_penwidth=uniform)
    print(f"    drew {out_png} (+ {out_gv})")


# --------------------------------------------------------------------------- #
# Reporting + the headline artifact                                           #
# --------------------------------------------------------------------------- #


def _print_table(results: dict, target: float, edge_counts: dict, full_heads: int,
                 full_edges: int, relative_floor: float = DEFAULT_RELATIVE_FLOOR) -> None:
    # Rows ordered by SELECTION SCORE (best first); edges/score columns make it auditable.
    has_mlp = any("mlps" in r for r in results.values())
    mlp_hdr = f" {'mlps':>5}" if has_mlp else ""
    print(f"    {'candidate':<22} {'heads':>5}{mlp_hdr} {'edges':>5} {'R_seq':>8} "
          f"{'R_acc':>8} {'R_ld':>8} {'circ_seq':>9} {'empty':>7} {'score':>10}  "
          f"circ_seq >= {target:.2f} & R_seq >= {relative_floor:.2f} & empty < {target:.2f}?")
    def _row_key(r):
        return (-_selection_score(r, edge_counts[r["name"]], full_heads, full_edges),
                edge_counts[r["name"]], _n_nodes(r), r["name"])
    for r in sorted(results.values(), key=_row_key):
        e = edge_counts[r["name"]]
        mark = "PASS" if _passes(r, target, relative_floor) else "fail"
        mlp_col = f" {r.get('n_mlps', 0):>5}" if has_mlp else ""
        print(f"    {r['name']:<22} {r['n_heads']:>5}{mlp_col} {e:>5} "
              f"{_fmt_r(r['R_sequence_accuracy']):>8} {_fmt_r(r['R_accuracy']):>8} "
              f"{_fmt_r(r['R_logit_diff']):>8} "
              f"{r['perf_circuit']['sequence_accuracy']:>9.3f} "
              f"{r['perf_empty']['sequence_accuracy']:>7.3f} "
              f"{_selection_score(r, e, full_heads, full_edges):>10.5f}  {mark}")
        for key in ("R_sequence_accuracy", "R_accuracy", "R_logit_diff"):
            if not isinstance(r[key], (int, float)):
                print(f"      UNSCORABLE: {r['name']}.{key}={r[key]} (full == empty: this "
                      "isolation has no contrast to recover)")
            elif not (-0.05 <= r[key] <= 1.25) and not math.isnan(r[key]):
                print(f"      WARNING: {r['name']}.{key}={r[key]:.3f} outside "
                      "the sane faithfulness range [~0, ~1.2]")


def _print_differential_table(results: dict, args, full_heads: int,
                              full_edges: int) -> None:
    # Differential columns; ELIG = s_union floor + non-empty fringe (d_* reported only).
    def score(r):
        return _differential_score(r, full_heads, full_edges, args.attn_only)
    print(f"    {'candidate':<22} {'fringe':>6} {'s_union':>8} {'d_aff':>8} {'d_unaff':>8} "
          f"{'d_ctl':>8} {'score':>10}  s_union >= {args.target:.2f}? (d_* reported, not gated)")
    for r in sorted(results.values(), key=lambda r: (-score(r), r["n_fringe_edges"], r["name"])):
        mark = "ELIG" if _passes_differential(r, args.target) else "fail"
        print(f"    {r['name']:<22} {r['n_fringe_edges']:>6} {r['s_union']:>8.3f} "
              f"{r['d_affected']:>8.3f} {r['d_unaffected']:>8.3f} {r['d_control']:>8.3f} "
              f"{score(r):>10.5f}  {mark}")


def _print_suppression_table(results: dict, args, full_heads: int,
                             full_edges: int) -> None:
    # Suppression columns; ELIG = non-empty fringe + EITHER release (cued/tonic) >= floor with its collateral <= eps.
    def score(r):
        return _suppression_score(r, full_heads, full_edges, args.attn_only,
                                  args.release_floor, args.epsilon)
    print(f"    {'candidate':<22} {'fringe':>6} {'release':>8} {'rel_ton':>8} {'ref':>5} "
          f"{'wrongF':>7} {'wrongS':>7} {'collat':>7} {'col_ton':>7} {'score':>10}  "
          f"EITHER release >= {args.release_floor:.2f} & its collat <= {args.epsilon:.2f}?")
    for r in sorted(results.values(), key=lambda r: (-score(r), r["n_fringe_edges"], r["name"])):
        mark = "ELIG" if _passes_suppression(r, args.release_floor, args.epsilon) else "fail"
        print(f"    {r['name']:<22} {r['n_fringe_edges']:>6} {r['release_rate']:>+8.3f} "
              f"{r['release_rate_tonic']:>+8.3f} "
              f"{_scored_release(r, args.release_floor, args.epsilon)[0]:>5} "
              f"{r['b_wins_full']:>7.3f} {r['b_wins_severed']:>7.3f} {r['collateral']:>7.3f} "
              f"{r['collateral_tonic']:>7.3f} {score(r):>10.5f}  {mark}")


def build_payload(winner: dict, cand: Candidate, args, edge_counts: dict,
                  full_heads: int, full_edges: int) -> dict:
    """Slim smallest_circuit.json: winner heads/edges + summary stats only (see docs/reference/design.md).

    In fringe modes the reported heads/edges are the FRINGE, so the source candidate's own
    accuracy is named ``source_candidate_absolute_accuracy``: it belongs to a DIFFERENT edge
    set than the ``edges`` beside it (it used to be written as
    ``whole_sequence_absolute_accuracy``, which readers take for the reported circuit's, and
    can sit below the row's own floor). The fringe is a DETECTOR/SUPPRESSOR, never a
    sufficient circuit, so it has no standalone task accuracy to report -- its evidence is
    s_union (C1) / release+collateral (C3)."""
    n_edges = edge_counts[winner["name"]]
    acc_key = ("whole_sequence_absolute_accuracy" if _mode(args) == "plain"
               else "source_candidate_absolute_accuracy")
    payload = {
        "circuit_name": winner["name"],
        "heads": winner["heads"],
        "edges": cand.edges,
        "n_heads": winner["n_heads"],
        "n_edges": n_edges,
        "n_heads_full": full_heads,
        "n_edges_full": full_edges,
        "n_test": winner["n_examples"],          # rows actually scored (agreeing rows dropped)
        "n_requested": winner["n_requested"],
        "pairs_regime": winner["pairs_regime"],
        "pairs_window": winner["pairs_window"],
        "input_sequence_data": (_contrast_data_description(args.base_circuit)
                                if args.base_circuit
                                else _input_data_description(args.model, args.circuit)),
        acc_key: winner["perf_circuit"]["sequence_accuracy"],
        # The model's own rate and the fully-severed baseline: without them no reader can
        # tell a real circuit from a constant-output isolation or a weak checkpoint.
        "perf_full": winner["perf_full"],
        "perf_empty": winner["perf_empty"],
        "relative_recovery_sequence_accuracy": winner["R_sequence_accuracy"],
        "min_accuracy_threshold": args.target,
        "min_relative_recovery": args.relative_floor,
        "selection_regime": SELECTION_REGIME,
        "candidate_set_sig": args.candidate_set_sig,
        "selection_strategy": SELECTION_STRATEGY,
        "selection_score": _selection_score(winner, n_edges, full_heads, full_edges),
    }
    if _mode(args) == "suppression":
        ref, release = _scored_release(winner, args.release_floor, args.epsilon)
        payload.update({
            "mode": "suppression",
            "base_circuit": args.base_circuit,
            "base_edges_sig": winner["base_edges_sig"],
            # The reported circuit IS the fringe suppressor; circuit_name keeps the source.
            "heads": winner["fringe_heads"],
            "edges": winner["fringe_edges"],
            "n_heads": winner["n_fringe_heads"],
            "n_edges": winner["n_fringe_edges"],
            "n_fringe_mlps": winner["n_fringe_mlps"],
            "input_sequence_data": _suppression_data_description(args.base_circuit),
            "metric_unit": winner["metric_unit"],
            "release": {"release_rate": winner["release_rate"],
                        "wrong_wins_full": winner["b_wins_full"],
                        "wrong_wins_severed": winner["b_wins_severed"],
                        "cued_correct_full": winner["a_correct_full"],
                        "cued_correct_severed": winner["a_correct_severed"],
                        "release_rate_tonic": winner["release_rate_tonic"],
                        "wrong_wins_severed_tonic": winner["b_wins_severed_tonic"],
                        "cued_correct_severed_tonic": winner["a_correct_severed_tonic"],
                        # which sever reference the gate + score actually read
                        "release_reference": ref,
                        "release_scored": release},
            "collateral": winner["collateral"],
            "collateral_tonic": winner["collateral_tonic"],
            "min_accuracy_threshold": args.release_floor,
            "selection_strategy": SELECTION_STRATEGY_SUPPRESSION,
            "selection_score": _suppression_score(winner, full_heads, full_edges,
                                                  args.attn_only, args.release_floor,
                                                  args.epsilon),
        })
    elif args.base_circuit:
        payload.update({
            "mode": "differential",
            "base_circuit": args.base_circuit,
            "base_edges_sig": winner["base_edges_sig"],
            # The reported circuit IS the fringe (the detector); circuit_name keeps
            # the source candidate's identity.
            "heads": winner["fringe_heads"],
            "edges": winner["fringe_edges"],
            "n_heads": winner["n_fringe_heads"],
            "n_edges": winner["n_fringe_edges"],
            "n_fringe_mlps": winner["n_fringe_mlps"],
            "metric_unit": winner["metric_unit"],
            "sufficiency": {"s_union": winner["s_union"]},
            "necessity": {"d_affected": winner["d_affected"],
                          "d_unaffected": winner["d_unaffected"],
                          "d_control": winner["d_control"],
                          "local_predicate": not args.global_predicate},
            "selection_strategy": SELECTION_STRATEGY_DIFFERENTIAL,
            "selection_score": _differential_score(winner, full_heads, full_edges,
                                                   args.attn_only),
        })
    _record_base_provenance(payload, args)
    return payload


def _record_base_provenance(payload: dict, args) -> None:
    """Unfaithful-base fallback fields + the echoed router verdict (suppression rows)."""
    if getattr(args, "base_fallback", None):
        payload.update(args.base_fallback)   # base_faithful/base_accuracy/base_candidate
    if _mode(args) == "suppression" and args.router_verdict is not None:
        payload["router_verdict"] = args.router_verdict


def build_no_faithful_payload(results: dict, args) -> dict:
    """Slim no_faithful marker: honest record that no candidate cleared the floor."""
    best = max(results.values(),
               key=lambda r: (r["perf_circuit"]["sequence_accuracy"], -_n_nodes(r)))
    payload = {
        "circuit_name": None,
        "no_faithful_circuit": True,
        "n_test": best["n_examples"],
        "n_requested": best["n_requested"],
        "pairs_regime": best["pairs_regime"],
        "pairs_window": best["pairs_window"],
        "input_sequence_data": (_contrast_data_description(args.base_circuit)
                                if args.base_circuit
                                else _input_data_description(args.model, args.circuit)),
        "min_accuracy_threshold": args.target,
        "min_relative_recovery": args.relative_floor,
        "selection_regime": SELECTION_REGIME,
        "candidate_set_sig": args.candidate_set_sig,
        "selection_strategy": SELECTION_STRATEGY,
        "best_available_accuracy": best["perf_circuit"]["sequence_accuracy"],
        # perf_full/perf_empty are candidate-independent: they say whether the floor was
        # even REACHABLE (weak checkpoint / constant-output isolation) or the circuit missed.
        "perf_full": best["perf_full"],
        "perf_empty": best["perf_empty"],
        "best_available_relative_recovery": best["R_sequence_accuracy"],
    }
    if _mode(args) == "suppression":
        # The best-by-release fringe: its SIZE is the only record of what was tried, and it
        # is the number models/taxonomy.json reports for these rows (it had nothing to read).
        top = max(results.values(),
                  key=lambda r: _scored_release(r, args.release_floor, args.epsilon)[1])
        ref, release = _scored_release(top, args.release_floor, args.epsilon)
        payload.update({
            "mode": "suppression",
            "base_circuit": args.base_circuit,
            "base_edges_sig": args.base_edges_sig,
            "input_sequence_data": _suppression_data_description(args.base_circuit),
            "min_accuracy_threshold": args.release_floor,
            "selection_strategy": SELECTION_STRATEGY_SUPPRESSION,
            "best_available_release": max(r["release_rate"] for r in results.values()),
            "best_available_release_tonic": max(r["release_rate_tonic"]
                                                for r in results.values()),
            "best_available_candidate": top["name"],
            "best_available_release_reference": ref,
            "best_available_release_scored": release,
            "n_heads": top["n_fringe_heads"],
            "n_edges": top["n_fringe_edges"],
            "n_fringe_mlps": top["n_fringe_mlps"],
            "collateral": top["collateral"],
            "collateral_tonic": top["collateral_tonic"],
        })
    elif args.base_circuit:
        payload.update({
            "mode": "differential",
            "base_circuit": args.base_circuit,
            "base_edges_sig": args.base_edges_sig,
            "necessity": {"local_predicate": not args.global_predicate},
            "selection_strategy": SELECTION_STRATEGY_DIFFERENTIAL,
            "best_available_s_union": max(r["s_union"] for r in results.values()),
        })
    _record_base_provenance(payload, args)
    return payload


def _drop_stale(out_json: str, out_png: str, out_gv: str) -> None:
    """A no-winner verdict must not leave the PREVIOUS winner's .pt/PNG beside it."""
    for stale in (os.path.splitext(out_json)[0] + ".pt", out_png, out_gv):
        if os.path.exists(stale):
            os.remove(stale)


def _candidate_set_sig(dirs: list) -> str:
    """md5 of the sorted candidate-dir name set = the selection's INPUT UNIVERSE. Folded into
    the skip-if-done match because floor/strategy/n_test are all unchanged when a later run
    ADDS a candidate: the old winner then stands even though a smaller one now exists."""
    names = sorted({sub for d in dirs if os.path.isdir(d) for sub in os.listdir(d)
                    if os.path.isdir(os.path.join(d, sub))
                    and not _is_variant_candidate_dir(sub)})
    return hashlib.md5("|".join(names).encode()).hexdigest()


def _output_done(out_json: str, args, candidate_set_sig: str) -> bool:
    """Skip-if-done: slim file matches current regime/floor/strategy/n_test + the candidate
    set it was selected from (+ base sig in differential/suppression mode); a winner also
    needs its .pt."""
    if not _json_ok(out_json):
        return False
    with open(out_json) as f:
        d = json.load(f)
    mode = _mode(args)
    strategy = (SELECTION_STRATEGY_SUPPRESSION if mode == "suppression"
                else SELECTION_STRATEGY_DIFFERENTIAL if mode == "differential"
                else SELECTION_STRATEGY)
    floor = args.release_floor if mode == "suppression" else args.target
    n = SMOKE_EVAL_N if args.smoke else SMALLEST_EVAL_N
    off = MG.POOL_WINDOWS[SCORING_CONSUMER]
    # n_REQUESTED, not n_test: n_test is the kept-row count, which the v2 answer_mask makes
    # smaller than n on any pool with agreeing rows -- checking it would never be "done".
    if (d.get("min_accuracy_threshold") != floor
            or d.get("selection_strategy") != strategy
            or d.get("selection_regime") != SELECTION_REGIME
            or d.get("candidate_set_sig") != candidate_set_sig
            or d.get("n_requested") != n
            or d.get("pairs_regime") != MG.CANON_PAIRS_REGIME
            or d.get("pairs_window") != [off, off + n]):
        return False
    if args.base_circuit and (d.get("base_circuit") != args.base_circuit
                              or d.get("base_edges_sig") != args.base_edges_sig):
        return False                             # stale base winner
    if d.get("no_faithful_circuit"):
        return True
    return os.path.exists(os.path.splitext(out_json)[0] + ".pt")


# --------------------------------------------------------------------------- #
# Driver                                                                       #
# --------------------------------------------------------------------------- #


def main(args) -> int:
    spec = MG.get_spec(args.model)
    ckpt = spec.checkpoint
    args.attn_only = bool(spec.attn_only)        # phantom-MLP strip + fringe node counts
    args.jobs = max(1, args.jobs)
    if args.smoke:
        args.jobs = 1                            # one GPU process at a time

    canonical_dir = os.path.join(MG.data_dir(args.model, args.circuit), "circuits")
    cand_dirs = args.candidates_dir or [canonical_dir]
    if args.smoke:
        smoke_base = SMOKE_DIR
        os.makedirs(smoke_base, exist_ok=True)
        out_json = os.path.join(smoke_base, "smallest_circuit.json")
        out_png = os.path.join(smoke_base, "smallest_circuit.png")
        out_gv = os.path.join(smoke_base, "smallest_circuit.gv")
        work_dir = smoke_base
    else:
        out_json = os.path.join(MG.data_dir(args.model, args.circuit),
                                "smallest_circuit.json")
        out_png = os.path.join(MG.fig_dir(args.model, args.circuit),
                               "smallest_circuit.png")
        out_gv = os.path.join(MG.fig_dir(args.model, args.circuit),
                              "smallest_circuit.gv")
        work_dir = os.path.join(MG.data_dir(args.model, args.circuit),
                                ".smallest_circuit_evals")
    os.makedirs(work_dir, exist_ok=True)

    if args.base_circuit:
        # Differential/suppression: the BASE (cued-executor) winner edges form the fringe cut.
        mode = _mode(args)
        base_json = os.path.join(MG.data_dir(args.model, args.base_circuit),
                                 "smallest_circuit.json")
        if not os.path.isfile(base_json):
            raise FileNotFoundError(
                f"{mode} mode: base winner missing: {base_json} "
                f"(run smallest for {args.base_circuit!r} first)")
        with open(base_json) as f:
            base_payload = json.load(f)
        if base_payload.get("no_faithful_circuit"):
            # Unfaithful base: evaluate anyway against its best-available candidate;
            # recorded as base_faithful=false metadata, never a blocker (by design).
            bc, bacc = best_available_candidate(MG.data_dir(args.model, args.base_circuit),
                                                args.attn_only)
            base_edges = bc.edges
            args.base_fallback = {"base_faithful": False, "base_accuracy": bacc,
                                  "base_candidate": bc.name}
            print(f"== {mode} mode: base {args.base_circuit!r} has no faithful winner -> "
                  f"best-available fallback {bc.name} "
                  f"(acc={'?' if bacc is None else f'{bacc:.4f}'}, {len(base_edges)} edges) ==")
        else:
            base_edges = base_payload["edges"]   # KeyError = fail loud
        args.base_edges_sig = _edges_sig(base_edges)
        args.base_edges_json = os.path.join(work_dir, "edges", "_base.json")
        os.makedirs(os.path.dirname(args.base_edges_json), exist_ok=True)
        with open(args.base_edges_json, "w") as f:
            json.dump(base_edges, f)
        print(f"== {mode} mode: base={args.base_circuit} "
              f"({len(base_edges)} edges, sig={args.base_edges_sig[:8]}) ==")

    if not args.force and _output_done(out_json, args, _candidate_set_sig(cand_dirs)):
        print(f"[skip-if-done] {out_json} already complete for this "
              "regime/target/strategy/n_test/candidate-set (pass --force to redo).")
        if not os.path.exists(out_png):
            with open(out_json) as f:
                d = json.load(f)
            if not d.get("no_faithful_circuit"):  # no_faithful has no PNG by design
                heads = [CL.parse_head_label(h) for h in d["heads"]]
                render_winner(heads, d["edges"], out_png, out_gv, seed=args.seed,
                              source_name=d["circuit_name"])
        return 0

    if args.search:            # generate the score-maximizing EAP candidate before scanning
        _run_eap_search(args)

    print(f"== candidates: scanning {cand_dirs} ==")
    cands = discover_candidates(cand_dirs, args.attn_only)
    if not cands:
        subs = [s for d in cand_dirs if os.path.isdir(d) for s in os.listdir(d)
                if os.path.isdir(os.path.join(d, s))]
        if subs and all(_is_variant_candidate_dir(s) for s in subs):   # only positional/probe candidates
            payload = {"circuit_name": None, "no_faithful_circuit": True,
                       "selection_regime": SELECTION_REGIME,
                       "reason": "positional_only_candidates"}
            _dump_json(payload, out_json)
            _drop_stale(out_json, out_png, out_gv)
            print("    only positional/probe candidates (not output-metric comparable); "
                  f"smallest N/A -> wrote positional_only marker\n    wrote {out_json}")
            return 0
        print("ERROR: no parseable candidate circuits found.")
        return 2
    any_mlp = any(c.mlps for c in cands)
    print(f"    {len(cands)} candidates parsed:")
    for c in sorted(cands, key=lambda c: (c.n_nodes, c.name)):
        mlp_col = f" mlps={c.n_mlps:>2}" if any_mlp else ""
        print(f"      {c.name:<22} dialect={c.dialect:<17} heads={c.n_heads:>3}"
              f"{mlp_col} edges={len(c.edges):>4}  "
              f"({os.path.relpath(c.source, REPO_ROOT)})")

    if args.max_candidates and args.max_candidates < len(cands):
        cands = _pick_candidate_subset(cands, args.max_candidates)
        print(f"    evaluating subset of {len(cands)} (max-candidates="
              f"{args.max_candidates}): {[c.name for c in cands]}")

    print(f"== evaluation: {len(cands)} candidates, jobs={args.jobs}, "
          f"n={SMOKE_EVAL_N if args.smoke else SMALLEST_EVAL_N} pairs ==")
    by_name = {c.name: c for c in cands}
    try:
        results = run_evaluations(sorted(cands, key=lambda c: (c.n_nodes, c.name)),
                                  args, ckpt, work_dir)
    except ZeroContrastDataset as e:
        # The isolation itself is degenerate (every row agrees with its resample reference).
        # Record it as UNSCORABLE instead of leaving the old winner standing.
        _dump_json({"circuit_name": None, "no_faithful_circuit": True,
                    "selection_regime": SELECTION_REGIME,
                    "reason": "zero_contrast_dataset", "detail": str(e)}, out_json)
        _drop_stale(out_json, out_png, out_gv)
        print(f"\n=== UNSCORABLE: {e} ===\n    wrote {out_json} "
              "(zero_contrast_dataset; no winner, no PNG/.pt)")
        return 0

    if not args.smoke and args.refine_rounds > 0 and not args.candidates_dir:
        refine(results, args, ckpt, work_dir, canonical_dir)
        for name in results:
            if name not in by_name:
                sub = os.path.join(canonical_dir, name)
                c = _sniff_candidate_dir(sub, name, args.attn_only)
                if c is not None:
                    by_name[name] = c

    # Drawn-edge count per candidate (incl. input edges) -> n_edges_circuit in the selection
    # score. Comes from the parsed Candidate, not the eval JSON; a result with no candidate
    # is a bug (it cannot be built or drawn), so KeyError here instead of scoring it 0 edges.
    edge_counts = {name: len(by_name[name].edges) for name in results}
    # full prunable NODE budget: heads + one MLP node per layer when not attn-only
    full_heads = spec.n_layers * spec.n_heads + (0 if spec.attn_only else spec.n_layers)
    full_edges = MG.live_edges(spec)              # the ONE edge denominator (never an artifact)
    args.candidate_set_sig = _candidate_set_sig(cand_dirs)   # AFTER search/refine added dirs

    _print_table(results, args.target, edge_counts, full_heads, full_edges,
                 args.relative_floor)
    if _mode(args) == "suppression":
        _print_suppression_table(results, args, full_heads, full_edges)
        winner = pick_winner_suppression(results, args.release_floor, args.epsilon,
                                         full_heads, full_edges, args.attn_only)
    elif args.base_circuit:
        _print_differential_table(results, args, full_heads, full_edges)
        winner = pick_winner_differential(results, args.target, full_heads, full_edges,
                                          args.attn_only)
    else:
        winner = pick_winner(results, args.target, edge_counts, full_heads, full_edges,
                             args.relative_floor)
    if winner is None:
        # No candidate cleared the floor -> honest no_faithful marker; drop stale winner artifacts.
        payload = build_no_faithful_payload(results, args)
        _dump_json(payload, out_json)
        _drop_stale(out_json, out_png, out_gv)
        if _mode(args) == "suppression":
            print(f"\n=== NO suppression circuit @ release >= {args.release_floor:.2f} "
                  f"(collateral <= {args.epsilon:.2f}) ===")
            print(f"    best available release: {payload['best_available_release']:+.4f} "
                  f"(tonic: {payload['best_available_release_tonic']:+.4f}) "
                  "(no candidate releases the wrong task within collateral)")
        elif args.base_circuit:
            print(f"\n=== NO eligible differential circuit @ s_union >= "
                  f"{args.target:.2f} ===")
            print(f"    best available s_union: {payload['best_available_s_union']:.4f}")
        else:
            print(f"\n=== NO faithful circuit @ abs seq-acc >= {args.target:.2f}, "
                  f"R_seq >= {args.relative_floor:.2f}, empty < {args.target:.2f} ===")
            print(f"    best available abs seq-acc: {payload['best_available_accuracy']:.4f} "
                  f"(R_seq {_fmt_r(payload['best_available_relative_recovery'])}); "
                  f"full model {payload['perf_full']['sequence_accuracy']:.4f}, "
                  f"empty baseline {payload['perf_empty']['sequence_accuracy']:.4f}")
        print(f"    wrote {out_json}  (no_faithful_circuit=true; no PNG/.pt)")
        return 0

    cand = by_name[winner["name"]]
    payload = build_payload(winner, cand, args, edge_counts, full_heads, full_edges)
    _dump_json(payload, out_json)

    # smallest_circuit.pt: the smallest circuit IS the winning candidate, so copy its circuit.pt (emitted by eap/acdc/edge_pruning alike) next to the json — a self-contained, Graph.from_pt-loadable artifact with byte-identical provenance to the candidate it selected.
    out_pt = os.path.splitext(out_json)[0] + ".pt"          # .../smallest_circuit.pt
    src_pt = os.path.join(os.path.dirname(cand.source), "circuit.pt")
    if args.base_circuit:
        # Differential: the reported circuit is the FRINGE — build ITS .pt natively.
        nodes = sorted({e["parent"] for e in winner["fringe_edges"]}
                       | {e["child"] for e in winner["fringe_edges"]})
        tuples = [(e["parent"], e["child"], e["qkv"], e["score"])
                  for e in winner["fringe_edges"]]
        _load_fc_eap().save_circuit_pt(
            {"n_layers": spec.n_layers, "n_heads": spec.n_heads,
             "d_model": spec.d_model}, nodes, tuples, out_pt)
        print(f"    wrote {out_pt}  (fringe: {winner['name']} minus base "
              f"{args.base_circuit})")
    elif os.path.isfile(src_pt):
        shutil.copyfile(src_pt, out_pt)
        print(f"    wrote {out_pt}  (copied from "
              f"{os.path.relpath(src_pt, REPO_ROOT)})")
    else:
        print(f"    WARNING: no circuit.pt beside the winning candidate "
              f"({os.path.relpath(src_pt, REPO_ROOT)}); smallest_circuit.pt NOT "
              f"written. Re-run the {winner['name']!r} candidate stage (eap/acdc/"
              "edge_pruning) to emit it, then re-run --stages smallest.")

    mode = _mode(args)
    winner_edges = (winner["n_fringe_edges"] if args.base_circuit
                    else edge_counts[winner["name"]])
    winner_heads = winner["n_fringe_heads"] if args.base_circuit else winner["n_heads"]
    kind = {"suppression": "suppression", "differential": "most-faithful",
            "plain": "most-faithful"}[mode]
    print(f"\n=== smallest {kind} circuit: {winner['name']} ===")
    nodes_note = ""
    if "mlps" in winner:
        nodes_note = (f"  mlps={winner['mlps']}  "
                      f"n_nodes={winner['n_nodes']}")
    match mode:                       # lazy: each score reads mode-specific keys absent in other modes
        case "suppression": score = _suppression_score(winner, full_heads, full_edges,
                                                       args.attn_only, args.release_floor,
                                                       args.epsilon)
        case "differential": score = _differential_score(winner, full_heads, full_edges,
                                                         args.attn_only)
        case _: score = _selection_score(winner, winner_edges, full_heads, full_edges)
    strategy = {"suppression": SELECTION_STRATEGY_SUPPRESSION,
                "differential": SELECTION_STRATEGY_DIFFERENTIAL,
                "plain": SELECTION_STRATEGY}[mode]
    floor_note = {"suppression": f"release >= {args.release_floor}, collateral <= {args.epsilon}",
                  "differential": f"s_union >= {args.target}; d_* reported, not gated",
                  "plain": (f"abs seq-acc >= {args.target}, R_seq >= {args.relative_floor}, "
                            f"empty < {args.target}")}[mode]
    print(f"    heads={winner_heads}{nodes_note}  edges={winner_edges}  "
          f"R_sequence_accuracy={_fmt_r(winner['R_sequence_accuracy'])}  "
          f"R_accuracy={_fmt_r(winner['R_accuracy'])}  "
          f"R_logit_diff={_fmt_r(winner['R_logit_diff'])}\n"
          f"    selection score={score:.5f} "
          f"= {strategy}  (floor: {floor_note})")
    if mode == "suppression":
        print(f"    release={winner['release_rate']:+.4f}  "
              f"tonic={winner['release_rate_tonic']:+.4f}  "
              f"(scored on the {_scored_release(winner, args.release_floor, args.epsilon)[0]} "
              f"reference)  "
              f"(wrong-wins {winner['b_wins_full']:.3f}->{winner['b_wins_severed']:.3f}; "
              f"cued {winner['a_correct_full']:.3f}->{winner['a_correct_severed']:.3f})  "
              f"collateral={winner['collateral']:.4f}/{winner['collateral_tonic']:.4f}")
    elif args.base_circuit:
        print(f"    s_union={winner['s_union']:.4f}  d_affected={winner['d_affected']:.4f}  "
              f"d_unaffected={winner['d_unaffected']:.4f}  d_control={winner['d_control']:.4f}")
    print(f"    wrote {out_json}")
    render_winner(cand.heads, cand.edges, out_png, out_gv, seed=args.seed,
                  source_name=winner["name"])
    print(f"safety: jobs={args.jobs}; assert_memory_headroom before every spawn.")
    return 0


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Select the smallest faithful circuit among the discovered "
                    "candidates (edge-level per-position mean-ablation "
                    "faithfulness; every evaluation in its own subprocess, "
                    "--jobs at a time).")
    ap.add_argument("--model", required=True, help="model name (a DSL algorithm_id)")
    ap.add_argument("--circuit", required=True,
                    help="named circuit of the model: reads its inputs.json and "
                         "candidate folders under <model>/<circuit>/, writing "
                         "<model>/<circuit>/smallest_circuit.json")
    ap.add_argument("--target", type=float, default=DEFAULT_TARGET,
                    help="absolute eligibility floor: a candidate qualifies iff its "
                         "ABSOLUTE whole-sequence exact-match accuracy under "
                         "edge-level mean ablation (perf_circuit."
                         "sequence_accuracy: all answer tokens correct per "
                         "example) is >= this AND the fully-severed baseline "
                         "(perf_empty) is below it; among the "
                         "eligible the winner MAXIMIZES "
                         f"{SELECTION_STRATEGY} (default {DEFAULT_TARGET})")
    ap.add_argument("--relative-floor", type=float, default=DEFAULT_RELATIVE_FLOOR,
                    help="relative eligibility floor, conjoined with --target: minimum "
                         "R_sequence_accuracy = (circuit - empty)/(full - empty), so the "
                         "verdict measures the CIRCUIT and not the checkpoint (0 disables; "
                         f"default {DEFAULT_RELATIVE_FLOOR})")
    ap.add_argument("--search", action="store_true",
                    help="before scanning, drive eap.py's ONE-attribution --target-seq-acc "
                         "search to emit the EAP candidate maximizing (1-h/H)(1-e/E) above "
                         "--target, then select+write smallest_circuit.json from it (EAP is "
                         "the default finder; opt-in acdc/edge_pruning candidates still join "
                         "the same selection)")
    ap.add_argument("--search-n", type=int, default=None,
                    help="eval-set size for the --search eap probes (passed to eap --search-n; "
                         "default: exact parity with this stage's SMALLEST_EVAL_N)")
    ap.add_argument("--jobs", type=int, default=2,
                    help="max concurrent eval subprocesses (NOT clamped; "
                         "run_model_pipeline passes 4)")
    ap.add_argument("--smoke", action="store_true",
                    help=f"smoke: n={SMOKE_EVAL_N} pairs, tiny mean batch, "
                         "jobs=1, outputs quarantined under .smoke/run4/")
    ap.add_argument("--candidates-dir", nargs="+", default=None,
                    help="override candidate dirs (default: "
                         "circuits_data/<model>/<circuit>/circuits)")
    ap.add_argument("--seed", type=int, default=0,
                    help="figure layout seed (default 0)")
    ap.add_argument("--max-candidates", type=int, default=None,
                    help="evaluate at most this many candidates (smoke default 2)")
    ap.add_argument("--refine-rounds", type=int, default=1,
                    help="midpoint-refinement rounds between adjacent candidate "
                         "sizes via the discovery runners (production only; "
                         "default 1)")
    ap.add_argument("--force", action="store_true",
                    help="ignore cached evals / existing output")
    ap.add_argument("--base-circuit", default=None,
                    help="differential mode (contrast-pair circuits): load this "
                         "sibling circuit's smallest_circuit.json edges and select "
                         "by S_union / D_affected / D_unaffected / D_control "
                         "(docs/reference/circuit_types.md §4; default None)")
    ap.add_argument("--global-predicate", action="store_true",
                    help="the circuit's predicate is consumed at every position "
                         "(ContrastCircuit.local_predicate=False): d_unaffected is "
                         "reported but NOT gated")
    ap.add_argument("--suppression", action="store_true",
                    help="suppression mode (C3, cue-flip TaskMux circuits; "
                         "docs/reference/circuit_types.md §5): --base-circuit "
                         "is the cued task's executor; select the fringe suppressor by "
                         "release_rate (wrong-task-wins rise when severed) with collateral "
                         "<= --epsilon on the agree positions, under EITHER the cued-row or "
                         "the wrong-cue (tonic) sever reference")
    ap.add_argument("--release-floor", type=float, default=DEFAULT_RELEASE_FLOOR,
                    help="suppression eligibility: min release_rate (wrong-task-wins rise "
                         f"when severing the fringe) to count as a suppressor "
                         f"(default {DEFAULT_RELEASE_FLOOR})")
    ap.add_argument("--epsilon", type=float, default=DEFAULT_EPSILON,
                    help="suppression eligibility: max collateral (accuracy drift on the "
                         f"agree positions) when severing the fringe "
                         f"(default {DEFAULT_EPSILON})")
    ap.add_argument("--router-verdict", default=None,
                    help="suppression mode: the ROUTER diagnostic's (cued,wrong) verdict, "
                         "echoed verbatim into the output payload (recorded, never gates)")
    # ---- child mode (internal) ----
    ap.add_argument("--eval-child", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--name", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--heads", default="", help=argparse.SUPPRESS)
    ap.add_argument("--mlps", default="", help=argparse.SUPPRESS)
    ap.add_argument("--edges-json", default="", help=argparse.SUPPRESS)
    ap.add_argument("--base-edges-json", default="", help=argparse.SUPPRESS)
    ap.add_argument("--n", type=int, default=SMALLEST_EVAL_N, help=argparse.SUPPRESS)
    ap.add_argument("--mean-n", type=int, default=MEAN_N_PROD, help=argparse.SUPPRESS)
    ap.add_argument("--batch-cap", type=int, default=BATCH_CAP_PROD,
                    help=argparse.SUPPRESS)
    ap.add_argument("--out", default=None, help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    if a.smoke and a.max_candidates is None:
        a.max_candidates = 2
    if a.eval_child and not (a.name and a.out):
        ap.error("--eval-child requires --name and --out")
    if a.base_edges_json and not a.base_circuit:
        ap.error("--base-edges-json requires --base-circuit")
    if a.suppression and not a.base_circuit:
        ap.error("--suppression requires --base-circuit (the cued task's executor circuit)")
    if a.router_verdict is not None and not a.suppression:
        ap.error("--router-verdict only applies to --suppression rows")
    return a


if __name__ == "__main__":
    a = parse_args()
    sys.exit(eval_child(a) if a.eval_child else main(a))
