"""
dataset_store.py
=================================================================================
Central, traceable dataset store for the segrev3 pipeline. ALL input data used
across the repo lives under ``datasets/<app>/<param>/<scheme>.json``, organised by
APPLICATION (manual / acdc / eap) and by the PARAMETER that indexes each run
(``tau`` for acdc, ``top-n`` for eap; manual has no per-run parameter):

    datasets/
      manual/                         <scheme>.json
      acdc/   acdc_thresh_<tau>/       <scheme>.json
      eap/    eap_top<n>/              <scheme>.json

``scheme`` is one of the three data-generation types in ``data_gen.py``:
``random`` | ``replace_threes_to_nonthrees`` | ``replace_non_threes_to_threes``.

Every pipeline LOADS its input from here (so the stored data IS what was used and
every run is traceable to a file), and ``data_gen/generate_datasets.py`` WRITES here. NB:
acdc feeds the same data to every tau and eap to every top-n within a run -- the
per-parameter folders make the provenance explicit and self-contained (the files
are identical across taus / top-ns unless you generate them with different seeds).

``load()`` (re)generates a missing file with the recorded defaults so a pipeline
run is never blocked by a missing dataset.
"""

from __future__ import annotations

import os
import sys

_THIS = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(_THIS)
sys.path.insert(0, _THIS)
from data_gen import (  # noqa: E402
    make_clean_corrupted, load_pairs, SCHEMES,
    make_single_dataset, load_single as _dg_load_single,
    make_pair_with_n_threes, count_threes, _as_input_rows,
)
from canon import (DEFAULT_SEED, DEFAULT_REPLACE_N,  # noqa: E402,F401  (canonical; re-exported)
                   SEGREV3_ACDC_EXAMPLES, CANON_EAP_EXAMPLES, CANON_MANUAL_EXAMPLES)

STORE = os.path.join(REPO, "datasets")
APPS = ("manual", "acdc", "eap", "edge")

# Default dataset size (examples) per app when a file must be generated on the fly.
# eap needs attr (200) + eval (256); 512 leaves margin to slice both from one file.
# edge does no patching -- it just needs single inputs for the realized-attention plots.
DEFAULT_N = {"manual": CANON_MANUAL_EXAMPLES, "acdc": SEGREV3_ACDC_EXAMPLES,
             "eap": CANON_EAP_EXAMPLES, "edge": 64}   #: canon is the single source

# The "edge" app is SPECIAL: its datasets are SINGLE (non-paired) inputs indexed by
# the NUMBER OF 3s rather than a corruption scheme -> datasets/edge/n_threes_<k>.json.
# (manual/acdc/eap are clean/corrupted PAIRS indexed by scheme.)


def _tau_str(tau) -> str:
    """Tidy tau string matching the discovery output folders, e.g. 0.055 -> '0.055'."""
    return "%g" % float(tau)


def param_folder(app: str, tau=None, topn=None) -> str:
    """Relative ``<app>/<param>`` folder (or just ``manual``)."""
    if app == "manual":
        return "manual"
    if app == "acdc":
        if tau is None:
            raise ValueError("acdc datasets are indexed by tau")
        return os.path.join("acdc", f"acdc_thresh_{_tau_str(tau)}")
    if app == "eap":
        if topn is None:
            raise ValueError("eap datasets are indexed by top-n")
        return os.path.join("eap", f"eap_top{int(topn)}")
    raise ValueError(f"unknown app {app!r}; expected one of {APPS}")


def app_param_dir(app: str, tau=None, topn=None) -> str:
    return os.path.join(STORE, param_folder(app, tau=tau, topn=topn))


def _scheme_stem(scheme: str) -> str:
    """File stem for a BATCHED paired dataset. The replace_* schemes carry an
    explicit ``batched_`` prefix to distinguish them at a glance from the
    single-sequence files (``datasets/single/single_<scheme>_k<k>...``)."""
    return f"batched_{scheme}" if scheme.startswith("replace") else scheme


def dataset_path(app: str, scheme: str = "random", tau=None, topn=None, n_threes=None) -> str:
    if app == "edge":
        if n_threes is None:
            raise ValueError("edge datasets are indexed by n_threes")
        return os.path.join(STORE, "edge", f"n_threes_{int(n_threes)}.json")
    return os.path.join(app_param_dir(app, tau=tau, topn=topn), f"{_scheme_stem(scheme)}.json")


def dataset_label(app: str, scheme: str = "random", tau=None, topn=None, n_threes=None) -> str:
    """Human-readable provenance label, e.g. ``acdc/acdc_thresh_0.02/random`` or
    ``edge/n_threes_3``."""
    if app == "edge":
        return os.path.join("edge", f"n_threes_{int(n_threes)}")
    return os.path.join(param_folder(app, tau=tau, topn=topn), _scheme_stem(scheme))


def generate(app: str, scheme: str = "random", tau=None, topn=None, n_threes=None,
             n=None, seed: int = DEFAULT_SEED, replace_n: int = DEFAULT_REPLACE_N) -> str:
    """Write a dataset and return its path. For ``edge`` this is a SINGLE dataset
    (``datasets/edge/n_threes_<k>.json``); otherwise a clean/corrupted PAIR
    (``datasets/<app>/<param>/<scheme>.json``)."""
    size = n if n is not None else DEFAULT_N[app]
    if app == "edge":
        p = dataset_path("edge", n_threes=n_threes)
        make_single_dataset(size, n_threes=n_threes, seed=seed, save_path=p)
        return p
    if scheme not in SCHEMES:
        raise ValueError(f"unknown scheme {scheme!r}; expected one of {SCHEMES}")
    if app == "manual":
        seed += SCHEMES.index(scheme)   # manual_circuit.py's convention: seed + i per scheme
    p = dataset_path(app, scheme, tau=tau, topn=topn)
    make_clean_corrupted(size, scheme=scheme, seed=seed, n=replace_n, save_path=p)
    return p


def load(app: str, scheme: str = "random", tau=None, topn=None, device=None,
         generate_if_missing: bool = True, n=None, seed: int = DEFAULT_SEED):
    """Return ``(pairs, path)`` for a PAIRED dataset (manual/acdc/eap); generate it
    if missing. For the ``edge`` (single, n_threes) app use ``load_single``."""
    if app == "edge":
        raise ValueError("edge datasets are single (non-paired) -- use load_single(n_threes=...)")
    p = dataset_path(app, scheme, tau=tau, topn=topn)
    if not os.path.exists(p):
        if not generate_if_missing:
            raise FileNotFoundError(
                f"dataset not found: {p}\n(run data_gen/generate_datasets.py first, "
                f"or pass generate_if_missing=True)")
        generate(app, scheme, tau=tau, topn=topn, n=n, seed=seed)
    return load_pairs(p, device=device), p


def load_single(n_threes: int, device=None, generate_if_missing: bool = True,
                n=None, seed: int = DEFAULT_SEED):
    """Return ``(single, path)`` for the edge dataset with exactly ``n_threes`` 3s;
    generate it (default size/seed) if missing. ``single`` is a Segrev3Single."""
    p = dataset_path("edge", n_threes=n_threes)
    if not os.path.exists(p):
        if not generate_if_missing:
            raise FileNotFoundError(
                f"dataset not found: {p}\n(run data_gen/generate_datasets.py "
                f"--apps edge first, or pass generate_if_missing=True)")
        generate("edge", n_threes=n_threes, n=n, seed=seed)
    return _dg_load_single(p, device=device), p


# --------------------------------------------------------------------------- #
# SINGLE-SEQUENCE datasets (manual_circuit.py --single)                        #
# --------------------------------------------------------------------------- #
# ONE custom clean/corrupted pair (patching/DLA) + ONE custom single input (the
# edge realized-attention), parameterized by the EXACT clean 3-count k and the
# replaced-subset size n (independent knobs; see data_gen.make_pair_with_n_threes):
#
#     datasets/single/single_<scheme>_k<k>[_n<n>].json   (n only for replace_*)
#     datasets/single/edge_k<k>.json
#
# Literal digit rows override generation (and force a rewrite, so the stored file
# always reflects them); the filename stem is the dataset LABEL that every figure
# and results folder carries downstream.

SINGLE_DIR = os.path.join(STORE, "single")


def _derive_n_threes(n_threes, literal_inputs):
    """Resolve the 3-count k: the explicit parameter, else counted from the
    literal digit row(s) (data_gen validates agreement when both are given)."""
    if n_threes is not None:
        return int(n_threes)
    if literal_inputs is None:
        raise ValueError("single datasets need n_threes or literal input digits")
    return count_threes(_as_input_rows(literal_inputs))[0]


def single_pair_path(scheme: str = "random", n_threes: int = 0,
                     replace_n: int = DEFAULT_REPLACE_N) -> str:
    """``datasets/single/single_<scheme>_k<k>[_n<n>].json`` (the replaced-subset
    size n is only meaningful -- and only in the name -- for the replace_* schemes)."""
    if scheme not in SCHEMES:
        raise ValueError(f"unknown scheme {scheme!r}; expected one of {SCHEMES}")
    stem = f"single_{scheme}_k{int(n_threes)}"
    if scheme != "random":
        stem += f"_n{int(replace_n)}"
    return os.path.join(SINGLE_DIR, stem + ".json")


def single_edge_path(n_threes: int) -> str:
    """``datasets/single/edge_k<k>.json`` -- ONE input with exactly k 3s."""
    return os.path.join(SINGLE_DIR, f"edge_k{int(n_threes)}.json")


def generate_single_pair(scheme: str = "random", n_threes=None,
                         replace_n: int = DEFAULT_REPLACE_N, seed: int = DEFAULT_SEED,
                         clean_inputs=None, corrupted_inputs=None) -> str:
    """Write the SINGLE custom pair dataset and return its path. The clean side
    has EXACTLY ``n_threes`` 3s (or is the literal ``clean_inputs``); corrupted
    follows ``scheme`` with exactly ``replace_n`` replacements (or is the literal
    ``corrupted_inputs``)."""
    k = _derive_n_threes(n_threes, clean_inputs)
    p = single_pair_path(scheme, k, replace_n)
    make_pair_with_n_threes(1, n_threes=n_threes, scheme=scheme, n=replace_n,
                            seed=seed, save_path=p,
                            clean_inputs=clean_inputs, corrupted_inputs=corrupted_inputs)
    return p


def generate_single_edge(n_threes=None, seed: int = DEFAULT_SEED, inputs=None) -> str:
    """Write the SINGLE edge dataset (one input with exactly ``n_threes`` 3s, or
    the literal ``inputs``) and return its path."""
    k = _derive_n_threes(n_threes, inputs)
    p = single_edge_path(k)
    make_single_dataset(1, n_threes=n_threes, seed=seed, save_path=p, inputs=inputs)
    return p


def load_single_pair(scheme: str = "random", n_threes=None,
                     replace_n: int = DEFAULT_REPLACE_N, seed: int = DEFAULT_SEED,
                     device=None, generate_if_missing: bool = True,
                     clean_inputs=None, corrupted_inputs=None):
    """Return ``(pairs, path)`` for the SINGLE custom pair. Literal digits always
    regenerate (the file must reflect them); otherwise an existing file is reused,
    so the manual and post-analysis stages see the SAME pair."""
    k = _derive_n_threes(n_threes, clean_inputs)
    p = single_pair_path(scheme, k, replace_n)
    if clean_inputs is not None or not os.path.exists(p):
        if not generate_if_missing and clean_inputs is None:
            raise FileNotFoundError(
                f"dataset not found: {p}\n(run circuit_discovery/manual_circuit.py "
                f"--single first, or pass generate_if_missing=True)")
        generate_single_pair(scheme, n_threes=n_threes, replace_n=replace_n, seed=seed,
                             clean_inputs=clean_inputs, corrupted_inputs=corrupted_inputs)
    return load_pairs(p, device=device), p


def load_single_edge(n_threes=None, seed: int = DEFAULT_SEED, device=None,
                     generate_if_missing: bool = True, inputs=None):
    """Return ``(single, path)`` for the SINGLE edge input (a ``Segrev3Single``).
    Literal digits always regenerate; otherwise an existing file is reused."""
    k = _derive_n_threes(n_threes, inputs)
    p = single_edge_path(k)
    if inputs is not None or not os.path.exists(p):
        if not generate_if_missing and inputs is None:
            raise FileNotFoundError(
                f"dataset not found: {p}\n(run circuit_discovery/manual_circuit.py "
                f"--single first, or pass generate_if_missing=True)")
        generate_single_edge(n_threes=n_threes, seed=seed, inputs=inputs)
    return _dg_load_single(p, device=device), p


if __name__ == "__main__":
    # smoke test: paths + a tiny generate/load round-trip (does not touch the real store).
    print("STORE:", STORE)
    print("manual/random ->", dataset_label("manual", "random"))
    print("acdc tau=0.02 ->", dataset_label("acdc", "random", tau=0.02))
    print("eap top=200   ->", dataset_label("eap", "replace_threes_to_nonthrees", topn=200))
    print("single pair   ->", os.path.relpath(single_pair_path("replace_threes_to_nonthrees", 3, 1), STORE))
    print("single edge   ->", os.path.relpath(single_edge_path(3), STORE))
    pairs, path = load("manual", "random", n=8)
    print("loaded", tuple(pairs.clean_tokens.shape), "from", os.path.relpath(path, REPO))
    print("OK")
