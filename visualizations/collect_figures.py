"""
collect_figures.py
=================================================================================
Bundle every generated ``.png`` figure into ONE zip (preserving the repo-relative
folder structure, so the archive unzips into the same acdc/eap/manual/validation
tree) for easy download.

By default it collects PNGs under ``visualizations/`` and ``validation/figs_data/``
(the analysis-output trees) and writes ``figures.zip`` at the repo root. The
pre-pulled reference repos (``github_repos/``), ``__pycache__`` and ``.git`` are
always skipped.

EXAMPLES (run from the repo root)
  python visualizations/collect_figures.py
      # -> figures.zip with every png under visualizations/ + validation/figs_data/

  python visualizations/collect_figures.py --out my_figs.zip
  python visualizations/collect_figures.py --roots visualizations validation/figs_data
"""

from __future__ import annotations

import argparse
import os
import zipfile

_THIS = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(_THIS)

DEFAULT_ROOTS = ["visualizations", os.path.join("validation", "figs_data")]
EXCLUDE_DIRS = {"github_repos", "__pycache__", ".git"}


def collect_pngs(roots):
    """Return a sorted list of absolute paths to every .png under the given repo
    roots, skipping EXCLUDE_DIRS."""
    pngs = []
    for root in roots:
        base = os.path.join(REPO, root)
        if not os.path.isdir(base):
            continue
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]
            for f in filenames:
                if f.lower().endswith(".png"):
                    pngs.append(os.path.join(dirpath, f))
    return sorted(pngs)


def make_zip(roots=DEFAULT_ROOTS, out="figures.zip"):
    pngs = collect_pngs(roots)
    out_path = out if os.path.isabs(out) else os.path.join(REPO, out)
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as z:
        for p in pngs:
            # preserve the repo-relative path inside the archive
            z.write(p, arcname=os.path.relpath(p, REPO))
    return out_path, pngs


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Zip every .png figure (preserving folder structure) for download.")
    ap.add_argument("--roots", nargs="+", default=DEFAULT_ROOTS,
                    help="repo-relative dirs to scan (default: visualizations/ + validation/figs_data/)")
    ap.add_argument("--out", default="figures.zip",
                    help="output zip path (default: figures.zip at the repo root)")
    a = ap.parse_args(argv)

    out_path, pngs = make_zip(a.roots, a.out)

    # per-top-level-group counts, for a quick manifest
    groups = {}
    for p in pngs:
        rel = os.path.relpath(p, REPO)
        key = os.sep.join(rel.split(os.sep)[:2])   # e.g. visualizations/acdc
        groups[key] = groups.get(key, 0) + 1

    size_mb = os.path.getsize(out_path) / 1e6
    print(f"zipped {len(pngs)} PNGs -> {os.path.relpath(out_path, REPO)} ({size_mb:.1f} MB)")
    for key in sorted(groups):
        print(f"   {groups[key]:3d}  {key}/")
    return out_path


if __name__ == "__main__":
    main()
