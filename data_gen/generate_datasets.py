"""
data_gen/generate_datasets.py
=================================================================================
Generate the THREE dataset types (``random`` / ``replace_threes_to_nonthrees`` /
``replace_non_threes_to_threes``) as ``.json`` datasets for any APPLICATION in this repo
(``manual`` / ``acdc`` / ``eap``) -- for one, a subset, or all of them -- into the
central ``datasets/`` store (see ``data_gen/dataset_store.py`` for the layout):

    datasets/manual/<scheme>.json
    datasets/acdc/acdc_thresh_<tau>/<scheme>.json     (one folder per tau)
    datasets/eap/eap_top<n>/<scheme>.json             (one folder per top-n)

These are exactly the files the pipelines load (manual viz, ACDC, EAP, validation),
so what is stored here IS what every run is computed on.

EXAMPLES (run from the repo root)
  python data_gen/generate_datasets.py
      # default: all 3 schemes for manual + every canonical tau (acdc) + top-n (eap)

  python data_gen/generate_datasets.py --apps acdc --schemes random --taus 0.055 0.02
      # only ACDC, only the random scheme, for two taus

  python data_gen/generate_datasets.py --apps manual eap --schemes all --topns 100 200 --examples 256
      # manual + EAP, all 3 schemes, custom top-ns, 256 examples each

  python data_gen/generate_datasets.py --apps all --schemes replace_threes_to_nonthrees --seed 7
"""

from __future__ import annotations

import argparse
import os
import sys

from tqdm import tqdm

_THIS = os.path.dirname(os.path.abspath(__file__))   # data_gen/
_REPO = os.path.dirname(_THIS)
sys.path.insert(0, _THIS)                            # dataset_store lives here
import dataset_store as store  # noqa: E402
# Canonical parameter sets: ONE source of truth, shared with the orchestrator
# and the discovery runners.
from canon import CANON_TAUS, SEGREV3_TOPNS, CANON_NTHREES  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Generate datasets into the central datasets/ store, per application. "
                    "manual/acdc/eap are clean/corrupted PAIRS (by scheme); edge is "
                    "SINGLE inputs indexed by the number of 3s.")
    ap.add_argument("--apps", nargs="+", default=["all"],
                    choices=["manual", "acdc", "eap", "edge", "all"],
                    help="which applications to generate for (default: all)")
    ap.add_argument("--schemes", nargs="+", default=["all"],
                    choices=list(store.SCHEMES) + ["all"],
                    help="which corruption schemes for the PAIRED apps (default: all 3)")
    ap.add_argument("--taus", type=float, nargs="+", default=CANON_TAUS,
                    help="ACDC thresholds to make per-tau dataset folders for")
    ap.add_argument("--topns", type=int, nargs="+", default=SEGREV3_TOPNS,
                    help="EAP top-n values to make per-top-n dataset folders for")
    ap.add_argument("--n-threes", type=int, nargs="+", default=CANON_NTHREES,
                    dest="n_threes",
                    help="EDGE: number(s) of 3s per input (one single dataset each; default 0..10)")
    ap.add_argument("-n", "--examples", type=int, default=None, dest="n",
                    help="examples per dataset (default: per-app -- "
                         f"{store.DEFAULT_N})")
    ap.add_argument("--seed", type=int, default=store.DEFAULT_SEED)
    ap.add_argument("--replace-n", type=int, default=store.DEFAULT_REPLACE_N,
                    help="# positions replaced for the replace_* schemes")
    a = ap.parse_args(argv)

    apps = list(store.APPS) if "all" in a.apps else a.apps
    schemes = list(store.SCHEMES) if "all" in a.schemes else a.schemes

    # total datasets to write (so the bar has an accurate length)
    _total = sum(
        len(a.n_threes) if app == "edge"
        else len(schemes) if app == "manual"
        else len(schemes) * len(a.taus) if app == "acdc"
        else len(schemes) * len(a.topns) if app == "eap"
        else 0
        for app in apps)

    written = []
    _pbar = tqdm(total=_total, desc="generating datasets", unit="ds")

    def _gen(*args, **kw):
        written.append(store.generate(*args, **kw))
        _pbar.update(1)

    for app in apps:
        if app == "edge":
            # single (non-paired) datasets indexed by n_threes -- no scheme axis.
            for k in a.n_threes:
                _gen("edge", n_threes=k, n=a.n, seed=a.seed)
            continue
        for scheme in schemes:
            if app == "manual":
                _gen(app, scheme, n=a.n, seed=a.seed, replace_n=a.replace_n)
            elif app == "acdc":
                for tau in a.taus:
                    _gen(app, scheme, tau=tau, n=a.n, seed=a.seed, replace_n=a.replace_n)
            elif app == "eap":
                for topn in a.topns:
                    _gen(app, scheme, topn=topn, n=a.n, seed=a.seed, replace_n=a.replace_n)
    _pbar.close()

    print(f"wrote {len(written)} datasets under {store.STORE}:")
    for p in written:
        print("   ", os.path.relpath(p, _REPO))
    return written


if __name__ == "__main__":
    main()
