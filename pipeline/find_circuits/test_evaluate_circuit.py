#!/usr/bin/env python3
"""Fast self-test of evaluate_circuit.py (CPU, about a minute; small row counts).

Checks, on the showcase checkpoints in ``pipeline/models/trained_models_pt/``:
  * the edge universe equals ``canon.live_edge_budget`` and the node universe heads + MLP blocks;
  * the local per-row forward (``_patched_forward``) with mean baselines reproduces the pipeline's
    ``smallest_faithful_circuit._edge_forward``, and the full circuit reproduces the model;
  * DomainCircuit: full circuit a == the model's whole-sequence accuracy, empty circuit a ~ 0;
  * ContrastCircuit (interchange): full circuit a == 1 and empty circuit a == 0 exactly;
  * SuppressionCircuit: full circuit a == the model's paired accuracy, empty circuit a ~ 0;
  * ProbeCircuit: normalization (full -> 1, constant predictor / empty circuit -> 0) on a probe
    fitted here with the probes stage's own fitter (``train_probes.fit_positions``);
  * determinism: two CLI runs under different hash seeds write identical JSON.

Usage:  python pipeline/find_circuits/test_evaluate_circuit.py [--checkpoint-dir DIR]
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time

_THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS)

import torch                                                   # noqa: E402

import evaluate_circuit as E                                   # noqa: E402

DOMAIN = ("schmitt_id__moe8", "schmitt_noband")
CONTRAST = ("mexhat_id__moe8", "mexhat_fill")
SUPPRESSION = ("mux2.pmstep-id.sr12-id__moe8", "suppress_t1_over_t0")
PROBE = ("graviclump_id__moe4", "graviclump_mean")


THREADS = ["--threads", "1"]        # robust on a shared machine; the CLI default is 4


def ckpt_args(model: str, ckpt_dir: str | None) -> list:
    path = os.path.join(ckpt_dir, f"{model}.pt") if ckpt_dir else E.MG.get_spec(model).checkpoint
    if not os.path.exists(path):
        raise SystemExit(f"missing checkpoint {path}; pass --checkpoint-dir")
    return ["--checkpoint", path]


def evaluate(model: str, circuit: str, candidate: str, extra: list, ckpt_dir, tmp) -> dict:
    out = os.path.join(tmp, f"{model}.{circuit}.{candidate}.json")
    payload, _ = E.run(E.parse_args(["--model", model, "--circuit", circuit, "--candidate",
                                     candidate, "--out", out] + extra + THREADS
                                    + ckpt_args(model, ckpt_dir)))
    return payload


def check_universe(ckpt_dir, tmp):
    edges = E.graph_edges(4, 8, False)
    assert len(edges) == E.live_edge_budget(4, 8, False) == 1519
    assert len(E.Circuit("full", edges).nodes) == 36           # 32 heads + 4 MLP/MoE blocks
    assert len(E.graph_edges(4, 8, True)) == E.live_edge_budget(4, 8, True)
    assert len(E.graph_edges(1, 8, False)) == E.live_edge_budget(1, 8, False)


def check_forward_identity(ckpt_dir, tmp):
    model = E.MG.load_segrev3(checkpoint_path=ckpt_args(DOMAIN[0], ckpt_dir)[1], device="cpu")
    model.cfg.use_attn_result = model.cfg.use_split_qkv_input = model.cfg.use_hook_mlp_in = True
    g = torch.Generator().manual_seed(0)
    toks = torch.randint(0, 10, (24, 20), generator=g)
    means = E.SFC._edge_site_means(model, toks, 24)
    base = E._mean_base(means, 3, 4)
    full_edges = E.graph_edges(4, 8, False)
    some = E.Circuit("some", [full_edges[i] for i in
                              torch.randperm(len(full_edges), generator=g)[:300].tolist()])
    with torch.no_grad():
        for c in (E.Circuit("full", full_edges), some, E.Circuit("empty", [])):
            ours = E._patched_forward(model, toks, c, base)
            theirs = E._mean_forward(model, toks, c, means)
            assert (ours - theirs).abs().max() < 1e-4, c.name
        gap = (E._patched_forward(model, toks, E.Circuit("full", full_edges), base)
               - model(toks)).abs().max()
        assert gap < 1e-4, gap


def check_domain(ckpt_dir, tmp):
    n = 200
    full = evaluate(*DOMAIN, "full", ["--n-rows", str(n)], ckpt_dir, tmp)
    assert full["class"] == "DomainCircuit" and full["floor"] == 0.9
    assert abs(full["a"] - full["model_seq_acc"]) <= 1 / n, (full["a"], full["model_seq_acc"])
    assert full["n_nodes"] == full["n_nodes_full"] and full["n_edges"] == full["n_edges_full"]
    assert full["reference"]["empty"]["a"] < 0.1
    assert full["reference"]["model"]["a"] == full["model_seq_acc"]
    return full


def check_contrast(ckpt_dir, tmp):
    full = evaluate(*CONTRAST, "full", ["--n-rows", "200", "--n-pairs", "120"], ckpt_dir, tmp)
    assert full["class"] == "ContrastCircuit" and full["ablation"] == "interchange"
    assert full["pairs_kept"] > 0
    assert full["a"] == full["a_clean"] == full["a_corrupt"] == 1.0, full["a"]
    assert full["reference"]["empty"]["a"] == 0.0
    assert "mean_ablation_a" in full["secondary"]
    return full


def check_suppression(ckpt_dir, tmp):
    full = evaluate(*SUPPRESSION, "full", ["--n-rows", "200", "--n-pairs", "100"], ckpt_dir, tmp)
    assert full["class"] == "SuppressionCircuit" and full["floor"] == 0.8
    assert full["a"] > 0.9 and full["reference"]["empty"]["a"] < 0.1, full["reference"]
    assert abs(full["a"] - full["reference"]["model"]["a"]) <= 0.02, full["reference"]
    return full


def fit_fixture_probe(ckpt_dir, root: str) -> None:
    """A layer-0 per-position probe written in the probes stage's artifact contract."""
    model_name, var_id = PROBE
    model = E.MG.load_segrev3(checkpoint_path=ckpt_args(model_name, ckpt_dir)[1], device="cpu")
    var = E.TP.resolve_variable(model_name, var_id)()
    pos = var.valid_positions()
    x = E.TP.sample_inputs(1024, 0)
    with torch.no_grad():
        X = E.TP.ln(model(x, stop_at_layer=1))[:, pos]
    y = var.labels(x)
    W, b, accs = E.TP.fit_positions(X, y, var.n_classes)
    d = os.path.join(root, model_name, var_id)
    os.makedirs(d, exist_ok=True)
    torch.save({"W": W.mean(0), "b": b.mean(0), "W_pos": W, "b_pos": b, "layer": 0,
                "hook": "blocks.0.hook_resid_post", "positions": pos, "n_classes": var.n_classes,
                "normalize": "ln", "accuracy": sum(accs) / len(accs)},
               os.path.join(d, "site_L0.pt"))
    with open(os.path.join(d, "site_scan.json"), "w") as f:
        json.dump({"per_layer_accuracy": {"0": sum(accs) / len(accs)}}, f)


def check_probe(ckpt_dir, tmp):
    root = os.path.join(tmp, "probes_data")
    fit_fixture_probe(ckpt_dir, root)
    full = evaluate(*PROBE, "full", ["--n-rows", "200", "--n-pairs", "40", "--probes-root", root],
                    ckpt_dir, tmp)
    assert full["class"] == "ProbeCircuit" and full["probe"]["layer"] == 0
    assert full["n_nodes_full"] == 9 and full["n_edges_full"] == E.live_edge_budget(1, 8, False)
    assert full["a_clean"] == full["a_corrupt"] == 1.0, (full["a_clean"], full["a_corrupt"])
    assert full["reference"]["empty"]["a"] == 0.0
    lab = torch.tensor([[1, 2], [1, 3], [2, 3], [1, 3]])
    assert E._majority(lab) == (3 + 3) / 8                       # best constant per position
    return full


def check_determinism(ckpt_dir, tmp):
    outs = []
    for hs in ("1", "2"):
        out = os.path.join(tmp, f"det{hs}", "evaluation.json")    # same basename: same command
        cmd = [sys.executable, os.path.join(_THIS, "evaluate_circuit.py"), "--model", CONTRAST[0],
               "--circuit", CONTRAST[1], "--candidate", "full", "--n-rows", "60", "--n-pairs",
               "40", "--seed", "3", "--out", out] + THREADS + ckpt_args(CONTRAST[0], ckpt_dir)
        subprocess.run(cmd, check=True, capture_output=True,
                       env=dict(os.environ, PYTHONHASHSEED=hs, PYTHONDONTWRITEBYTECODE="1"))
        outs.append(open(out).read())
    assert outs[0] == outs[1], "two runs with the same --seed differ"


CHECKS = [check_universe, check_forward_identity, check_domain, check_contrast,
          check_suppression, check_probe, check_determinism]


def run_all(ckpt_dir=None) -> None:
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory() as tmp:
        for fn in CHECKS:
            t = time.time()
            res = fn(ckpt_dir, tmp)
            summary = "" if not isinstance(res, dict) else \
                f" a={res['a']:.4f} (clean={res['a_clean']}, corrupt={res['a_corrupt']})"
            print(f"PASS {fn.__name__[6:]:<17}{summary} [{time.time() - t:.1f}s]", flush=True)


def test_evaluate_circuit():
    """pytest entry point."""
    run_all()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--checkpoint-dir", default=None,
                    help="dir holding <model>.pt (default: pipeline/models/trained_models_pt)")
    run_all(ap.parse_args().checkpoint_dir)
    print("evaluate_circuit self-test PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
