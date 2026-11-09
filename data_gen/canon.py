"""
canon.py
=================================================================================
The SINGLE SOURCE OF TRUTH for the pipeline's canonical parameters and the shared
CLI fragments, imported by the orchestrator (circuit_discovery/main.py), the data
generator (generate_datasets.py), the discovery runners (acdc.py / eap.py /
manual_circuit.py -- whose --circuit mode is also the circuit-scoped analysis)
and validation.

Pure stdlib (argparse only) -- NO torch / numpy imports -- so the thin
orchestrators can import it without paying the heavy-import startup cost.

Before this module existed the sweep constants were defined independently in
three places and had already drifted (bare ``eap.py`` used topns=[30,60,120] /
export-n=60 vs the canonical values below). Defining them once makes a bare
runner and an orchestrator run agree BY CONSTRUCTION.
"""

from __future__ import annotations

from dataclasses import dataclass

# --------------------------------------------------------------------------- #
# Canonical sweep parameters (what the committed results + docs use)           #
# --------------------------------------------------------------------------- #
CANON_TAUS = [0.2, 0.1, 0.055, 0.02]        # ACDC thresholds (one circuit per tau)
CANON_TOPNS = [50, 200, 800]                # ABS regime only: reproduces the pre-frac eap_top<n> artifacts
CANON_TOPN_FRACS = [0.02, 0.05, 0.15, 0.40]  # FRAC regime: rungs as fractions of the model's LIVE edge budget
CANON_TOPN_REGIME = "frac"                  # regime new runs use; "abs" only to reproduce old artifacts
CANON_EDGE_PRUNING_SPARSITIES = [0.7, 0.9, 0.95, 0.98]   # Edge Pruning target edge sparsities
DEFAULT_FINDERS = ("eap",)   # finders run by default; acdc/edge_pruning only when named in --stages
CANON_NTHREES = list(range(0, 11))          # edge datasets: 0..10 threes
CANON_IG_STEPS = 30                         # EAP integrated-gradient steps
CANON_EXPORT_N = 200                        # EAP headline circuit size
CANON_ACDC_EXAMPLES = 128                   # ACDC contrast-set size (was 20: 25.6x less data than EAP's 512)
CANON_MANUAL_EXAMPLES = 128                 # manual pipeline dataset size
CANON_EAP_EXAMPLES = 512                    # EAP contrast-set size: attr (200) + eval (256), with margin to slice both
# segrev3 case study only (repo-root runners): the committed datasets/acdc/ sets and EAP-IG sweep
SEGREV3_ACDC_EXAMPLES = 20
SEGREV3_TOPNS = [50, 75, 87, 100, 200, 400, 800]

#: Stamped into every ``Segrev3Pairs.extra`` (model_registry.dataset_to_pairs) so v1 and
#: v2 numbers are never compared. v1: plain pools scored at ALL output positions with no
#: answer_mask, and every consumer reading pool rows [0:n]. v2: plain pools masked to the
#: positions where the clean and resample labels DISAGREE (all-agreeing rows dropped), and
#: disjoint per-consumer pool windows (model_registry.POOL_WINDOWS).
CANON_PAIRS_REGIME = "pairs-v2"


# --------------------------------------------------------------------------- #
# The live edge universe + the scale-relative EAP top-n ladder                  #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class TopnLadder:
    """The resolved EAP rungs plus the provenance to record alongside them."""
    topns: list[int]
    regime: str
    live_edges: int


def live_edge_budget(n_layers: int, n_heads: int, attn_only: bool = True) -> int:
    """Edges in the model's LIVE attribution graph -- the ones an ablation can touch.

    EAP-IG's ``Graph.from_model`` wires an ``m{l}`` node per layer from
    ``(n_layers, n_heads, parallel_attn_mlp)`` alone and never reads ``attn_only``, so its
    ``real_edge_mask`` over-counts an attn-only model by every MLP-touching edge (110 vs 81
    at small, 1519 vs 1281 large, 5327 vs 4865 wide, 6637 vs 5633 deep, 23981 vs 22017 xl).
    Serial (``parallel_attn_mlp=False``) topology: every prior residual-stream writer feeds
    q/k/v of every head, then the MLP, then the logits."""
    L, H = n_layers, n_heads
    if attn_only:
        return 3 * H * (L + H * L * (L - 1) // 2) + 1 + L * H
    writers = L + (H + 1) * L * (L - 1) // 2          # summed residual-stream size per layer
    return 3 * H * writers + (writers + L * H) + (1 + L * H + L)


def topn_ladder(live_edges: int, regime: str = CANON_TOPN_REGIME) -> TopnLadder:
    """The canonical EAP top-n rungs for a model with ``live_edges`` live edges.

    ``"frac"`` scales CANON_TOPN_FRACS by that budget, so no rung can alias the full
    unpruned graph (the absolute [50, 200, 800] ladder covered 988% of the live graph at
    small -- two rungs over the whole graph -- and 3.6% at xl). ``"abs"`` returns the fixed
    CANON_TOPNS, for reproducing pre-frac artifacts only."""
    if live_edges < 1:
        raise ValueError(f"live_edges must be >= 1 (got {live_edges})")
    if regime == "abs":
        return TopnLadder(list(CANON_TOPNS), "abs", live_edges)
    if regime != "frac":
        raise ValueError(f"unknown top-n regime {regime!r}; expected 'abs' or 'frac'")
    if not all(0 < f <= 1 for f in CANON_TOPN_FRACS):
        raise ValueError(f"CANON_TOPN_FRACS must lie in (0, 1]: {CANON_TOPN_FRACS}")
    return TopnLadder(sorted({max(1, round(f * live_edges)) for f in CANON_TOPN_FRACS}),
                      "frac", live_edges)


# --------------------------------------------------------------------------- #
# Data-generation conventions (mirrored by data_gen.py / dataset_store.py)     #
# --------------------------------------------------------------------------- #
#: the three corruption schemes, named for WHICH POSITIONS GET REPLACED:
#:   replace_threes_to_nonthrees  -- n of the clean 3s -> non-3s (DELETES boundaries)
#:   replace_non_threes_to_threes -- n of the clean non-3s -> 3s (ADDS boundaries)
SCHEMES = ("random", "replace_threes_to_nonthrees", "replace_non_threes_to_threes")
DEFAULT_SEED = 0
DEFAULT_REPLACE_N = 1          # positions replaced for the replace_* schemes

# --------------------------------------------------------------------------- #
# Circuit selection -- ONE selector instead of --method + --name               #
# --------------------------------------------------------------------------- #
METHODS = ("manual", "acdc", "eap")


def parse_circuit(circuit: str):
    """Resolve a ``--circuit`` selector into ``(method, name)``.

    Accepts ``all`` (every method, every circuit), a method (``manual`` /
    ``acdc`` / ``eap`` -> all of its circuits), or a concrete circuit folder
    name (``rough`` / ``acdc_thresh_<tau>`` / ``eap_top<n>`` -- the owning
    method is inferred). Raises ValueError on anything else."""
    if circuit in (None, "all"):
        return "all", "all"
    if circuit in METHODS:
        return circuit, "all"
    if circuit == "rough":
        return "manual", "rough"
    if circuit.startswith("acdc_thresh_"):
        return "acdc", circuit
    if circuit.startswith("eap_top"):
        return "eap", circuit
    raise ValueError(
        f"unknown circuit {circuit!r}; expected 'all', a method "
        f"({'/'.join(METHODS)}), or a circuit folder name "
        f"(rough | acdc_thresh_<tau> | eap_top<n>)")


# --------------------------------------------------------------------------- #
# Single-sequence mode -- ONE flag set shared by every CLI that supports it    #
# --------------------------------------------------------------------------- #
# main.py (the ``single`` stage) and manual_circuit.py (--single, optionally
# combined with --circuit) take the same parameters; defining the argparse group
# + the guard checks + the forwarding argv HERE means the CLIs cannot drift apart.


def add_single_args(ap, switch: bool = False, scheme: bool = False,
                    seed: bool = False, title: str = "single-sequence mode"):
    """Add the single-sequence parameter group to ``ap``.

    ``switch`` also adds the ``--single`` on/off flag (the orchestrator uses the
    ``single`` stage word instead); ``scheme``/``seed`` add ``--scheme``/
    ``--seed`` for CLIs that do not already define them globally."""
    g = ap.add_argument_group(title)
    if switch:
        g.add_argument("--single", action="store_true",
                       help="analyse ONE custom sequence (pair) instead of the "
                            "batched datasets")
    g.add_argument("--n-threes", type=int, default=None, dest="n_threes",
                   help="EXACT number of 3s (k) in the clean and edge sequences "
                        "(required unless --clean is given)")
    if scheme:
        g.add_argument("--scheme", default="random", choices=list(SCHEMES),
                       help="how the corrupted side is built from the clean side "
                            "(default: random)")
    g.add_argument("--n-replace", type=int, default=DEFAULT_REPLACE_N,
                   dest="n_replace",
                   help="replaced-subset size n for the replace_* schemes -- "
                        "independent of k; needs n <= k (replace_threes_to_nonthrees) "
                        "or n <= 10-k (replace_non_threes_to_threes). Default 1.")
    g.add_argument("--clean", type=int, nargs="+", default=None, metavar="D",
                   help="literal clean input digits (10 digits 0-9); overrides generation")
    g.add_argument("--corrupted", type=int, nargs="+", default=None, metavar="D",
                   help="literal corrupted input digits (requires --clean)")
    g.add_argument("--edge-input", type=int, nargs="+", default=None,
                   dest="edge_input", metavar="D",
                   help="literal edge input digits (default: generated with the "
                        "clean side's 3-count)")
    if seed:
        g.add_argument("--seed", type=int, default=DEFAULT_SEED,
                       help="RNG seed if the single datasets must be "
                            "(re)generated (default: %(default)s)")
    return g


def check_single_args(a, error) -> None:
    """Shared guard checks for single-sequence mode; ``error`` is the reporter
    (e.g. ``parser.error`` or a SystemExit-raising lambda)."""
    if a.n_threes is None and a.clean is None:
        error("single mode needs --n-threes (exact clean 3-count) or a literal --clean")
    if a.corrupted is not None and a.clean is None:
        error("--corrupted requires --clean")
    if getattr(a, "scheme", None) == "all":
        error("single mode analyses ONE pair -- pick one --scheme (not 'all')")


def single_argv(a, switch: bool = True) -> list:
    """The CLI tail forwarding the single-sequence parameters (and any literal
    digit overrides) to manual_circuit.py."""
    argv = (["--single"] if switch else [])
    argv += ["--scheme", a.scheme, "--seed", str(a.seed),
             "--n-replace", str(a.n_replace)]
    if a.n_threes is not None:
        argv += ["--n-threes", str(a.n_threes)]
    for flag, vals in (("--clean", a.clean), ("--corrupted", a.corrupted),
                       ("--edge-input", a.edge_input)):
        if vals is not None:
            argv += [flag, *map(str, vals)]
    return argv
