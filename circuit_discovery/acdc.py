"""
acdc.py
=================================================================================
Run Automatic Circuit Discovery (ACDC) -- recursive edge pruning with the
KL-divergence metric over the answer positions 9..18 -- on the ``segrev3``
attention-only model, for a LIST of thresholds tau, and emit the pruned circuit
subgraph for each.

OUTPUT LAYOUT (visualizations/acdc/):
  * One folder PER threshold: ``acdc_thresh_<tau>/`` containing
        summary.json   -- the full per-circuit result (surviving edges, heads,
                          KL, steps, head roles) for that tau
        circuit.gv     -- graphviz source of the circuit
        circuit.pth    -- the saved ACDC subgraph
        circuit.png    -- the rendered circuit
  * One combined json ``acdc_all_thresholds_summary.json`` summarising EVERY tau
    run THUS FAR (it scans every ``acdc_thresh_<tau>/`` folder present, not just
    the thresholds of the current invocation).

Pipeline:

  1. Load segrev3 via the canonical loader; enable ``use_attn_result`` and
     ``use_split_qkv_input`` so ACDC can address per-head Q/K/V/result edges.
  2. Build the ACDC ``AllDataThings`` bundle (clean = ds, corrupted ``random``
     = ref_ds, metric = KL(clean||patched) over positions 9..18) via the new
     ``acdc.segrev3`` task package.
  3. For each threshold tau, construct a ``TLACDCExperiment`` and loop
     ``exp.step()`` until ``exp.current_node is None`` (recursive edge pruning).
  4. Export the surviving non-PLACEHOLDER edges to JSON, the subgraph dict to a
     ``.pth``, and a circuit PNG via ``acdc.acdc_graphics.show``.
  5. Summarise which attention heads survive and compare to the manually-found
     circuit (``visualizations/rough_circuit.json``).

Everything is CPU-friendly and self-contained: ``python circuit_discovery/acdc.py``.

Notes / deviations:
  * ACDC's ``step()`` renders an intermediate PNG every step via
    ``acdc.acdc_graphics.show`` and its colour helper imports ``cmapy`` (an extra
    dependency we do not require). We therefore (a) monkeypatch a
    cmapy-free pastel colour generator and (b) disable the per-step rendering
    (it is slow on CPU and clutters the cwd). The final per-threshold PNG is
    rendered once at the end with the same ``show`` machinery.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import time
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

# --------------------------------------------------------------------------- #
# Paths                                                                        #
# --------------------------------------------------------------------------- #
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(_THIS_DIR)
ACDC_REPO = os.path.join(REPO_ROOT, "github_repos", "Automatic-Circuit-Discovery")
OUT_DIR = os.path.join(REPO_ROOT, "visualizations", "acdc")
ROUGH_CIRCUIT_JSON = os.path.join(REPO_ROOT, "visualizations", "rough_circuit.json")
# One json (in OUT_DIR) summarising EVERY tau run thus far (one per acdc_thresh_<tau>/).
COMBINED_SUMMARY = os.path.join(OUT_DIR, "acdc_all_thresholds_summary.json")

for _p in (ACDC_REPO, os.path.join(REPO_ROOT, "interp_tools"),
           os.path.join(REPO_ROOT, "data_gen"), os.path.join(REPO_ROOT, "metrics")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


from model_loading import assert_memory_headroom, repo_relpath, resolve_device  # noqa: E402


def _load(p):
    with open(p) as f:
        return json.load(f)


def _tau_str(tau: float) -> str:
    """Tidy threshold string for folders/labels, e.g. 0.055 -> '0.055'."""
    return "%g" % tau


def _tau_folder(tau: float) -> str:
    """Per-threshold output folder: visualizations/acdc/acdc_thresh_<tau>/."""
    return os.path.join(OUT_DIR, f"acdc_thresh_{_tau_str(tau)}")


# --------------------------------------------------------------------------- #
# cmapy-free patch for ACDC graphics + disable per-step rendering             #
# --------------------------------------------------------------------------- #
def _cmapy_free_color(colorscheme: str = "") -> str:
    # Random light/pastel hex (cmapy "Pastel2" replacement).
    rgb = np.random.randint(140, 256, size=3)
    return "#{:02x}{:02x}{:02x}".format(int(rgb[0]), int(rgb[1]), int(rgb[2]))


def _install_acdc_graphics_patches() -> bool:
    """Make ACDC's graphics layer importable and quiet. Returns True if the REAL
    pygraphviz-backed renderer is available (circuit PNGs will render), False if we
    installed a no-op stub instead (PNGs skipped; everything else runs).

    The vendored ``acdc.TLACDCExperiment`` does ``from acdc.acdc_graphics import
    show`` at module top, and ``acdc_graphics`` imports ``pygraphviz`` (and
    ``cmapy``) at import time. If either is missing, that would make the ENTIRE ACDC
    core unimportable. So if the
    real module can't load, we register a no-op stub under ``acdc.acdc_graphics``
    in ``sys.modules`` BEFORE importing the experiment, so the core still loads and
    produces summary.json + circuit.pth (the real artifacts)."""
    import types

    graphics_real = True
    try:
        import acdc.acdc_graphics as graphics  # noqa: F401
    except Exception:
        graphics_real = False
        graphics = types.ModuleType("acdc.acdc_graphics")

        def _noop(*a, **k):
            return None

        graphics.show = _noop
        graphics.log_metrics_to_wandb = _noop
        graphics.generate_random_color = _cmapy_free_color
        # any other name the vendored code imports -> harmless no-op
        graphics.__getattr__ = lambda name: _noop
        sys.modules["acdc.acdc_graphics"] = graphics
        import acdc as _acdc_pkg
        _acdc_pkg.acdc_graphics = graphics

    import acdc.TLACDCExperiment as tlexp

    graphics.generate_random_color = _cmapy_free_color
    # Disable the per-step PNG dump inside TLACDCExperiment.step(): it is slow and
    # writes ims/img_new_*.png into the cwd. We render the final circuit ourselves
    # after the run completes (when the real renderer is available).
    tlexp.show = lambda *args, **kwargs: None
    return graphics_real


# --------------------------------------------------------------------------- #
# Node-name parsing -> (layer, head, channel)                                  #
# --------------------------------------------------------------------------- #
def _layer_of(name: str) -> Optional[int]:
    # "blocks.3.attn.hook_result" -> 3
    parts = name.split(".")
    if len(parts) >= 2 and parts[0] == "blocks":
        try:
            return int(parts[1])
        except ValueError:
            return None
    return None


def parse_node(name: str, index_tuple: Tuple) -> Dict:
    """Return a structured description of an ACDC graph node.

    keys: kind ('result'|'q'|'k'|'v'|'q_input'|...|'embed'|'resid_post'|'other'),
          layer, head (or None), label (short readable string).
    """
    head = None
    if isinstance(index_tuple, tuple) and len(index_tuple) >= 3 and index_tuple[2] is not None:
        head = int(index_tuple[2])
    layer = _layer_of(name)

    if name.endswith("hook_resid_post"):
        return {"kind": "resid_post", "layer": layer, "head": None, "label": "resid_post"}
    if name.endswith("hook_resid_pre"):
        return {"kind": "embed", "layer": layer, "head": None, "label": "embed"}
    if name.endswith("hook_embed"):
        return {"kind": "embed", "layer": layer, "head": None, "label": "tok_embed"}
    if name.endswith("hook_pos_embed"):
        return {"kind": "embed", "layer": layer, "head": None, "label": "pos_embed"}
    if name.endswith("attn.hook_result"):
        return {"kind": "result", "layer": layer, "head": head,
                "label": f"a{layer}.{head}"}
    for ch in ("q", "k", "v"):
        if name.endswith(f"attn.hook_{ch}"):
            return {"kind": ch, "layer": layer, "head": head,
                    "label": f"a{layer}.{head}_{ch}"}
        if name.endswith(f"hook_{ch}_input"):
            return {"kind": f"{ch}_input", "layer": layer, "head": head,
                    "label": f"a{layer}.{head}_{ch}in"}
    return {"kind": "other", "layer": layer, "head": head, "label": name}


def _head_str(layer: int, head: int) -> str:
    return f"L{layer}H{head}"


# --------------------------------------------------------------------------- #
# ACDC run for a single threshold                                             #
# --------------------------------------------------------------------------- #
def run_acdc_for_threshold(
    things,
    threshold: float,
    max_steps: int = 100000,
    verbose: bool = True,
    data_provenance: Optional[Dict] = None,
) -> Dict:
    """Run ACDC edge-pruning at one threshold; return a result dict + write files."""
    from acdc.TLACDCExperiment import TLACDCExperiment
    from acdc.TLACDCEdge import EdgeType
    import acdc.acdc_graphics as graphics

    model = things.tl_model
    model.reset_hooks()

    t0 = time.time()

    # CRITICAL PERFORMANCE FLAG. ACDC defaults to keeping its activation caches on
    # the CPU (online_cache_cpu / corrupted_cache_cpu). That makes sense for a large
    # model on a memory-constrained GPU, but for the tiny segrev3 model it forces a
    # CPU<->GPU round trip on essentially every edge test -- measured at ~6.8 s/step
    # vs ~0.06 s/step with the caches kept on-device (a ~120x per-step speed-up).
    # So when we are on CUDA we keep them on the GPU.
    on_cuda = next(model.parameters()).is_cuda
    cache_cpu = not on_cuda

    # Build the experiment (silence ACDC's very chatty hook-name dump).
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exp = TLACDCExperiment(
            model=model,
            ds=things.validation_data,
            ref_ds=things.validation_patch_data,
            threshold=threshold,
            metric=things.validation_metric,
            second_metric=None,
            verbose=False,
            zero_ablation=False,
            abs_value_threshold=False,
            indices_mode="reverse",
            names_mode="normal",
            online_cache_cpu=cache_cpu,
            corrupted_cache_cpu=cache_cpu,
            add_sender_hooks=True,
            add_receiver_hooks=False,
            use_pos_embed=False,
            show_full_index=False,
        )

    n_edges_total = exp.count_no_edges()
    init_metric = float(exp.cur_metric)
    if verbose:
        print(f"[tau={threshold}] graph built: {n_edges_total} prunable edges, "
              f"init KL={init_metric:.4g}. Running steps...")

    steps = 0
    devnull = io.StringIO()
    _pbar = tqdm(total=max_steps, desc=f"ACDC steps [tau={threshold}]", unit="step",
                 leave=False, disable=not verbose)
    while exp.current_node is not None and steps < max_steps:
        # ACDC's step() prints chatty per-edge logs; silence them but keep our
        # own periodic progress on the bar below.
        with contextlib.redirect_stdout(devnull):
            exp.step(testing=False)
        devnull.truncate(0)
        devnull.seek(0)
        steps += 1
        _pbar.update(1)
        if steps % 25 == 0:
            _pbar.set_postfix(edges=exp.count_no_edges(),
                              KL=round(float(exp.cur_metric), 4))
    _pbar.close()

    final_metric = float(exp.cur_metric)
    n_edges_final = exp.count_no_edges()
    elapsed = time.time() - t0
    if verbose:
        print(f"[tau={threshold}] DONE in {elapsed:.1f}s, {steps} steps. "
              f"edges {n_edges_total} -> {n_edges_final}, final KL={final_metric:.4g}")

    # ----- collect surviving edges -----
    surviving_edges: List[Dict] = []
    heads_in_circuit = set()           # (layer, head)
    head_roles: Dict[str, set] = {}    # "L3H1" -> {"result", "q", ...}
    for tupl, edge in exp.corr.all_edges().items():
        if not edge.present:
            continue
        if edge.edge_type == EdgeType.PLACEHOLDER:
            continue
        if edge.effect_size is None:
            continue
        receiver_name, receiver_idx, sender_name, sender_idx = tupl
        r = parse_node(receiver_name, receiver_idx.hashable_tuple)
        s = parse_node(sender_name, sender_idx.hashable_tuple)
        surviving_edges.append({
            "receiver": receiver_name,
            "receiver_index": list(receiver_idx.hashable_tuple),
            "receiver_label": r["label"],
            "sender": sender_name,
            "sender_index": list(sender_idx.hashable_tuple),
            "sender_label": s["label"],
            "edge_type": edge.edge_type.name,
            "effect_size": float(edge.effect_size),
        })
        for node in (r, s):
            if node["head"] is not None and node["layer"] is not None:
                heads_in_circuit.add((node["layer"], node["head"]))
                hs = _head_str(node["layer"], node["head"])
                head_roles.setdefault(hs, set()).add(node["kind"])

    surviving_edges.sort(key=lambda e: -abs(e["effect_size"]))

    heads_sorted = sorted(heads_in_circuit)
    head_summary = {
        _head_str(l, h): sorted(head_roles.get(_head_str(l, h), []))
        for (l, h) in heads_sorted
    }

    # ----- per-tau folder: summary.json + circuit.{gv,pth,png} -----
    folder = _tau_folder(threshold)
    os.makedirs(folder, exist_ok=True)
    json_path = os.path.join(folder, "summary.json")
    pth_path = os.path.join(folder, "circuit.pth")
    png_path = os.path.join(folder, "circuit.png")
    gv_path = os.path.join(folder, "circuit.gv")

    result = {
        "threshold": threshold,
        "tau_folder": os.path.basename(folder),
        "metric": "kl_divergence(clean||patched) over answer positions 9..18",
        "corruption_scheme": "random",
        "num_examples": int(things.validation_data.shape[0]),
        "n_prunable_edges_initial": int(n_edges_total),
        "n_edges_surviving": len(surviving_edges),
        "n_edges_count_no_edges": int(n_edges_final),
        "init_kl": init_metric,
        "final_kl": final_metric,
        "steps": steps,
        "heads_in_circuit": [_head_str(l, h) for (l, h) in heads_sorted],
        "head_roles": head_summary,
        "surviving_edges": surviving_edges,
    }
    if data_provenance is not None:
        result["data"] = data_provenance
    # subgraph .pth
    exp.save_subgraph(fpath=pth_path)

    # circuit.png + circuit.gv via ACDC's pygraphviz renderer (cmapy-free thanks to
    # the patch). show() writes "<base>.gv" (== circuit.gv) and draws "<base>.png",
    # and -- because it mkdir(exist_ok=True)s "<base>/" for per-colour-group graphs
    # (empty here, since colorscheme is a string) -- leaves an empty "circuit/" dir
    # we remove for a clean folder.
    png_ok = True
    gv_ok = False
    png_err = None
    try:
        buf2 = io.StringIO()
        with contextlib.redirect_stdout(buf2):
            graphics.show(exp.corr, fname=png_path, show_full_index=False, seed=0)
        # When graphics is the no-op stub (pygraphviz unavailable), show() returns
        # without writing files -> report PNG/GV as absent rather than claiming them.
        png_ok = os.path.exists(png_path)
        gv_ok = os.path.exists(gv_path)
        empty_dir = os.path.splitext(png_path)[0]      # ".../circuit"
        if os.path.isdir(empty_dir):
            try:
                os.rmdir(empty_dir)
            except OSError:
                pass
    except Exception as e:  # pragma: no cover - rendering is best-effort
        png_ok = False
        png_err = repr(e)
        if verbose:
            print(f"[tau={threshold}] PNG render failed: {png_err}")

    result["files"] = {
        "summary_json": repo_relpath(json_path),
        "circuit_gv": repo_relpath(gv_path) if gv_ok else None,
        "circuit_pth": repo_relpath(pth_path),
        "circuit_png": repo_relpath(png_path) if png_ok else None,
    }
    if not png_ok:
        result["png_error"] = png_err

    # Write summary.json LAST so it includes the files dict.
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)

    # Free hooks for the next threshold.
    model.reset_hooks()
    return result


# --------------------------------------------------------------------------- #
# Comparison with the manual / rough circuit                                  #
# --------------------------------------------------------------------------- #
def _parse_head_str(s: str) -> Optional[Tuple[int, int]]:
    # "L3H1" -> (3, 1)
    try:
        layer = int(s[1:s.index("H")])
        head = int(s[s.index("H") + 1:])
        return (layer, head)
    except (ValueError, IndexError):
        return None


def compare_to_manual(results: List[Dict]) -> Dict:
    manual = None
    if os.path.exists(ROUGH_CIRCUIT_JSON):
        with open(ROUGH_CIRCUIT_JSON) as f:
            manual = json.load(f)

    # Reference head sets from the manual analysis (movers + L0 detectors; see visualizations/rough_circuit.md).
    key_movers = {"L3H1", "L3H3", "L3H6", "L2H7"}
    key_detectors = {"L0H7", "L0H2", "L0H1", "L0H4", "L0H6"}
    manual_heads = set(manual.get("circuit_heads", [])) if manual else set()
    manual_movers = set(manual.get("movers", [])) if manual else set()
    manual_detectors = set(manual.get("detectors", [])) if manual else set()

    out = {
        "manual_circuit_heads": sorted(manual_heads),
        "key_movers": sorted(key_movers),
        "key_detectors": sorted(key_detectors),
        "per_threshold": {},
    }
    for r in results:
        acdc_heads = set(r["heads_in_circuit"])
        out["per_threshold"][str(r["threshold"])] = {
            "n_acdc_heads": len(acdc_heads),
            "acdc_heads": sorted(acdc_heads),
            "key_movers_recovered": sorted(acdc_heads & key_movers),
            "key_movers_missing": sorted(key_movers - acdc_heads),
            "key_detectors_recovered": sorted(acdc_heads & key_detectors),
            "key_detectors_missing": sorted(key_detectors - acdc_heads),
            "overlap_with_manual_full": sorted(acdc_heads & manual_heads),
            "acdc_only_heads": sorted(acdc_heads - manual_heads),
            "manual_only_heads": sorted(manual_heads - acdc_heads),
            "n_surviving_edges": r["n_edges_surviving"],
            "final_kl": r["final_kl"],
        }
    return out


# --------------------------------------------------------------------------- #
# Combined "thus far" summary (scans every acdc_thresh_<tau>/ folder)          #
# --------------------------------------------------------------------------- #
def build_combined_summary(config: Optional[Dict] = None) -> Dict:
    """Scan every ``acdc_thresh_<tau>/`` folder in OUT_DIR, load each ``summary.json``,
    and (re)write ``acdc_all_thresholds_summary.json`` covering ALL thresholds run
    thus far (sorted by tau descending = smallest circuit first)."""
    results: List[Dict] = []
    if os.path.isdir(OUT_DIR):
        for name in sorted(os.listdir(OUT_DIR)):
            if not name.startswith("acdc_thresh_"):
                continue
            sp = os.path.join(OUT_DIR, name, "summary.json")
            if os.path.exists(sp):
                try:
                    results.append(_load(sp))
                except Exception:
                    pass
    results.sort(key=lambda r: -float(r["threshold"]))

    comparison = compare_to_manual(results)
    summary = {
        "description": "Combined summary of ALL ACDC thresholds run thus far "
                       "(one acdc_thresh_<tau>/ folder each).",
        "config": config or {},
        "thresholds": [r["threshold"] for r in results],
        "per_threshold": [
            {
                "threshold": r["threshold"],
                "folder": r.get("tau_folder", os.path.basename(_tau_folder(r["threshold"]))),
                "n_prunable_edges_initial": r.get("n_prunable_edges_initial"),
                "n_edges_surviving": r.get("n_edges_surviving"),
                "final_kl": r.get("final_kl"),
                "n_heads": len(r.get("heads_in_circuit", [])),
                "heads_in_circuit": r.get("heads_in_circuit", []),
                "data": r.get("data", {}),
                "files": r.get("files", {}),
            }
            for r in results
        ],
        "comparison_to_manual_circuit": comparison,
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(COMBINED_SUMMARY, "w") as f:
        json.dump(summary, f, indent=2)
    return summary


# --------------------------------------------------------------------------- #
# Per-threshold runner (also the parallel-worker entry point)                  #
# --------------------------------------------------------------------------- #
def _build_and_run_threshold(tau: float, num_examples: int, device: str,
                             seed: int, scheme: str, verbose: bool = True) -> float:
    """Load data, build the ACDC bundle and run ACDC for ONE threshold. Defined at
    module top level (picklable) so it can run either in-process or as a separate
    process for ``--jobs > 1``. Each tau writes its own ``acdc_thresh_<tau>/`` folder,
    so the thresholds are fully independent and the combined summary is rebuilt from
    the folders afterwards."""
    _install_acdc_graphics_patches()
    from acdc.segrev3 import get_all_segrev3_things
    import dataset_store as store

    # Each tau loads ITS OWN dataset file so every run is traceable; the data is
    # identical across taus (one seed), so results are unchanged.
    pairs, data_path = store.load("acdc", scheme, tau=tau, device=device, n=num_examples)
    things = get_all_segrev3_things(
        num_examples=num_examples, device=device, seed=seed, pairs=pairs)
    if verbose:
        print(f"[tau={tau}] loaded data from {os.path.relpath(data_path, REPO_ROOT)}",
              flush=True)
    data_provenance = {
        "store_label": store.dataset_label("acdc", scheme, tau=tau),
        "path": repo_relpath(data_path),
        "scheme": scheme,
        "num_examples": int(things.validation_data.shape[0]),
    }
    run_acdc_for_threshold(things, threshold=tau, data_provenance=data_provenance,
                           verbose=verbose)
    return tau


# --------------------------------------------------------------------------- #
# main                                                                        #
# --------------------------------------------------------------------------- #
def main(
    thresholds: Optional[List[float]] = None,
    num_examples: int = 20,
    device: Optional[str] = None,
    seed: int = 0,
    scheme: str = "random",
    jobs: Optional[int] = None,
) -> Dict:
    if thresholds is None:
        thresholds = [0.2, 0.1, 0.055, 0.02]
    device = resolve_device(device)   # "auto"/None -> cuda when available, else cpu
    # The thresholds are INDEPENDENT (each writes its own folder), so by default run
    # them concurrently -- up to 4 at once -- which is a near-linear speed-up because
    # ACDC's recursive edge-pruning is Python-bound (separate processes => separate
    # GILs) and the model is tiny (cheap to replicate per worker).
    if jobs is None:
        jobs = min(len(thresholds), 4)
    jobs = max(1, min(jobs, len(thresholds)))
    # Refuse to fan out workers on an already-starved box. CPU threads per worker
    # follow OMP_NUM_THREADS (load_segrev3 uses every core when it is unset).
    assert_memory_headroom(context=f"acdc.py launching {jobs} worker(s)")
    print(f"ACDC device: {device}  | thresholds={thresholds}  | jobs={jobs}")

    graphics_real = _install_acdc_graphics_patches()
    if not graphics_real:
        print("  [note] pygraphviz unavailable -> circuit PNGs skipped "
              "(summary.json + circuit.pth still written).")
    from acdc.segrev3 import get_all_segrev3_things
    import dataset_store as store

    os.makedirs(OUT_DIR, exist_ok=True)

    # Sanity (once): metric is ~0 on clean data (no patching applied yet).
    # An existing store file is used as-is, so report ITS size, not the -n request.
    pairs0, _ = store.load("acdc", scheme, tau=thresholds[0], device=device, n=num_examples)
    num_examples = int(pairs0.clean_tokens.shape[0])
    print(f"Running ACDC PER TAU from the central store "
          f"(N={num_examples}, scheme={scheme})...")
    things0 = get_all_segrev3_things(num_examples=num_examples, device=device,
                                     seed=seed, pairs=pairs0)
    with torch.no_grad():
        clean_metric = float(things0.validation_metric(things0.tl_model(things0.validation_data)))
    print(f"sanity: KL(clean||clean) over positions 9..18 = {clean_metric:.3e} (expect ~0)")
    del things0, pairs0

    failed = []
    if jobs == 1:
        for tau in thresholds:
            _build_and_run_threshold(tau, num_examples, device, seed, scheme, verbose=True)
    else:
        import concurrent.futures
        import multiprocessing as mp
        ctx = mp.get_context("spawn")   # required for CUDA in subprocesses
        with concurrent.futures.ProcessPoolExecutor(max_workers=jobs, mp_context=ctx) as ex:
            futs = {ex.submit(_build_and_run_threshold, tau, num_examples, device,
                              seed, scheme, False): tau for tau in thresholds}
            for fut in concurrent.futures.as_completed(futs):
                tau = futs[fut]
                try:
                    fut.result()
                    print(f"[tau={tau}] done.", flush=True)
                except Exception as e:
                    print(f"[tau={tau}] FAILED: {type(e).__name__}: {e}", flush=True)
                    failed.append(tau)

    # Rebuild the combined summary from EVERY acdc_thresh_<tau>/ folder present (all
    # thresholds run thus far, not just this invocation's list).
    summary = build_combined_summary(config={
        "num_examples": num_examples,
        "device": device,
        "seed": seed,
        "corruption_scheme": scheme,
        "metric": "KL(clean||patched) over answer positions 9..18",
        "last_run_thresholds": thresholds,
    })
    print("\n=== ACDC summary (all thresholds thus far) ===")
    print(json.dumps(summary["comparison_to_manual_circuit"]["per_threshold"], indent=2))
    print(f"\nWrote combined summary -> {COMBINED_SUMMARY}")
    if failed:
        # exit nonzero so orchestrators (e.g. interp_tools/min_faithful_circuit)
        # see the failure instead of a silent partial run -- the combined summary
        # above still covers every tau that DID complete.
        raise SystemExit(f"ACDC failed for {len(failed)} threshold(s): {failed}")
    return summary


if __name__ == "__main__":
    import argparse

    from canon import CANON_TAUS, SEGREV3_ACDC_EXAMPLES, SCHEMES  # single source of truth

    ap = argparse.ArgumentParser(description="Run ACDC on segrev3.")
    ap.add_argument("--thresholds", type=float, nargs="+", default=CANON_TAUS,
                    help="list of tau thresholds to run ACDC at (one folder per tau)")
    ap.add_argument("-n", "--examples", type=int, default=SEGREV3_ACDC_EXAMPLES,
                    dest="num_examples",
                    help="contrast examples when a store file must be generated "
                         "(default: %(default)s; an existing file is used as-is)")
    ap.add_argument("--device", type=str, default="auto",
                    help="compute device: 'auto' (default; cuda when available), 'cuda', or 'cpu'")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scheme", default="random", choices=list(SCHEMES),
                    help="data-generation scheme to load from the central store")
    ap.add_argument("--jobs", type=int, default=None,
                    help="number of thresholds to run CONCURRENTLY (separate processes; "
                         "default: min(#thresholds, 4)). Use 1 to force sequential.")
    args = ap.parse_args()
    main(
        thresholds=args.thresholds,
        num_examples=args.num_examples,
        device=args.device,
        seed=args.seed,
        scheme=args.scheme,
        jobs=args.jobs,
    )
