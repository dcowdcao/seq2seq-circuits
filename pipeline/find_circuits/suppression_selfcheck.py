#!/usr/bin/env python3
"""Model-free self-check for the C3 suppression finder (no trained mux needed).

Part A validates the cue-flip circuit against the DSL engine (models/circuits.py).
Part B validates the suppression SELECTION math on synthetic tensors: the release
formula in the unit production uses (WHOLE-SEQUENCE exact match, on rows where the
token rate DISAGREES), the eligibility gates (release floor + collateral cap +
non-empty fringe), which sever reference the score reads, and the winner ranking.
Part C validates the unfaithful-base best-available fallback rule (max cached
whole-seq acc; tie -> fewer edges -> name; stale caches ignored; zero candidates raise).
Part D validates the run-everything metadata: base_faithful/base_accuracy/base_candidate
+ the echoed --router-verdict land on BOTH output shapes, and the old
base_not_faithful / router_gating markers fail _output_done (reruns redo those rows).

The mux5 breadth model is now trained (eval 0.956) and listed in models/models.txt;
end-to-end validation on it is still pending. This exercises every pure branch it will hit.

    python find_circuits/suppression_selfcheck.py
"""
import contextlib, io, json, os, sys, tempfile

_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS)
for _p in (_THIS, os.path.join(_ROOT, "models")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch

import algorithms, circuits
import model_registry as MG
import smallest_faithful_circuit as S

MUX = "mux2.rev-id.dblodd-id"     # task0 = reverse, task1 = double-odd
#: The pool window the smallest stage scores on — imported so a constant bump fails here.
SCORING_WINDOW = [MG.POOL_WINDOWS[S.SCORING_CONSUMER],
                  MG.POOL_WINDOWS[S.SCORING_CONSUMER] + S.SMALLEST_EVAL_N]


def part_a() -> None:
    """Cue-flip circuit vs the DSL engine."""
    expr = algorithms.decode_algorithm_id(MUX)
    assert isinstance(expr, algorithms.TaskMux) and len(expr.members) == 2
    cls = circuits.suppression_circuit(MUX, cued=0, wrong=1)
    assert cls.circuit_id == "suppress_t0_over_t1" and cls.base_circuit == "task0"
    assert circuits.suppression_circuit(MUX, 0, 1) is cls, "factory must be idempotent"
    assert any(c is cls for c in circuits.CIRCUITS), "circuit must self-register"

    n = circuits.validate_pairs(cls, trials=512, seed=0)   # raises on any bad pair
    inst = circuits.build(cls)
    clean, corrupt = inst.sample_pairs(256, seed=1)
    assert torch.equal(clean[:, 0], torch.full((256,), 10))     # cued cue token
    assert torch.equal(corrupt[:, 0], torch.full((256,), 11))   # wrong cue token
    assert torch.equal(clean[:, 1:], corrupt[:, 1:]), "payload must be shared"
    # eval(clean)=A(payload)=reverse ; eval(corrupt)=B(payload)=double-odd ; masks non-empty
    for i in range(256):
        a, b = expr.eval(clean[i]), expr.eval(corrupt[i])
        assert bool((a != b).any()), "empty mask slipped past validation"
        assert torch.equal(inst.affected_outputs(clean[i], corrupt[i]), a != b)
    two = circuits.suppression_circuits_for(MUX)
    assert {c.circuit_id for c in two} == {"suppress_t0_over_t1", "suppress_t1_over_t0"}
    print(f"  part A OK: {n} pairs validated; cue-flip structure + affected_outputs exact; "
          f"both ordered pairs registered")


def _res(name, fringe_edges, release, collateral, b_full=0.0,
         release_tonic=None, collateral_tonic=None) -> dict:
    """A synthetic eval-result dict shaped like _suppression_metrics output.
    Tonic reference defaults to the cued numbers (a non-tonic suppressor)."""
    rt = release if release_tonic is None else release_tonic
    ct = collateral if collateral_tonic is None else collateral_tonic
    return {"name": name, "n_fringe_edges": fringe_edges, "n_fringe_heads": max(1, fringe_edges // 3),
            "n_fringe_mlps": 0, "release_rate": release, "collateral": collateral,
            "b_wins_full": b_full, "b_wins_severed": b_full + release,
            "a_correct_full": 1.0, "a_correct_severed": 1.0 - release,
            "release_rate_tonic": rt, "collateral_tonic": ct,
            "b_wins_severed_tonic": b_full + rt, "a_correct_severed_tonic": 1.0 - rt}


def part_b_selection() -> None:
    """Eligibility gates + winner ranking on synthetic result dicts."""
    H, E = 32, 1000
    good = _res("eap_top200", fringe_edges=12, release=0.80, collateral=0.01)   # the suppressor
    weak = _res("eap_top50", fringe_edges=6, release=0.20, collateral=0.01)     # release too low
    noisy = _res("eap_top400", fringe_edges=40, release=0.90, collateral=0.30)  # wrecks agree posns
    empty = _res("eap_top100", fringe_edges=0, release=0.99, collateral=0.0)    # fringe subset of base

    assert S._passes_suppression(good, 0.5, 0.05)
    assert not S._passes_suppression(weak, 0.5, 0.05), "release below floor must fail"
    assert not S._passes_suppression(noisy, 0.5, 0.05), "collateral above eps must fail"
    assert not S._passes_suppression(empty, 0.5, 0.05), "empty fringe must fail"

    results = {r["name"]: r for r in (good, weak, noisy, empty)}
    win = S.pick_winner_suppression(results, 0.5, 0.05, H, E)
    assert win is not None and win["name"] == "eap_top200", f"wrong winner: {win and win['name']}"
    # score favors high release and a small fringe
    assert S._suppression_score(good, H, E) > S._suppression_score(weak, H, E)
    # release clamps into [0,1]; a tiny suppressor beats a bloated one at equal release
    small = _res("s", 6, 0.8, 0.0); big = _res("b", 300, 0.8, 0.0)
    assert S._suppression_score(small, H, E) > S._suppression_score(big, H, E)
    assert S.pick_winner_suppression({"none": weak}, 0.5, 0.05, H, E) is None
    print("  part B OK: release floor / collateral cap / empty-fringe gates + ranking + clamp")


def part_b_tonic_gate() -> None:
    """E3: the wrong-cue (tonic) sever reference opens the cue-constant blind spot."""
    tonic = _res("t", 10, release=0.0, collateral=0.0, release_tonic=0.9, collateral_tonic=0.01)
    assert S._passes_suppression(tonic, 0.5, 0.05), "tonic release alone must pass"
    messy = _res("m", 10, release=0.0, collateral=0.0, release_tonic=0.9, collateral_tonic=0.30)
    assert not S._passes_suppression(messy, 0.5, 0.05), "a tonic pass needs TONIC collateral <= eps"
    cross = _res("c", 10, release=0.9, collateral=0.01, release_tonic=0.0, collateral_tonic=0.50)
    assert S._passes_suppression(cross, 0.5, 0.05), "a cued pass must not be vetoed by tonic collateral"
    half = _res("h", 10, release=0.9, collateral=0.30, release_tonic=0.7, collateral_tonic=0.01)
    assert S._passes_suppression(half, 0.5, 0.05), "collateral gate binds per-reference, not across"
    dead = _res("d", 10, release=0.2, collateral=0.0, release_tonic=0.2, collateral_tonic=0.0)
    assert not S._passes_suppression(dead, 0.5, 0.05), "both releases below floor must fail"
    win = S.pick_winner_suppression({r["name"]: r for r in (tonic, dead)}, 0.5, 0.05, 32, 1000)
    assert win is not None and win["name"] == "t", "tonic-only passer must be selectable"
    # the score reads whichever reference PASSED, not the cued rate (earlier suppression
    # winners passed on tonic and were scored on a cued 0.0 before this)
    assert S._scored_release(tonic, 0.5, 0.05) == ("tonic", 0.9)
    assert S._scored_release(cross, 0.5, 0.05) == ("cued", 0.9)
    both = _res("bo", 10, release=0.6, collateral=0.0, release_tonic=0.8, collateral_tonic=0.0)
    assert S._scored_release(both, 0.5, 0.05) == ("tonic", 0.8), "both pass -> the LARGER"
    assert S._scored_release(dead, 0.5, 0.05) == ("cued", 0.2), "neither passes -> larger, rank-only"
    assert S._suppression_score(tonic, 32, 1000) > S._suppression_score(dead, 32, 1000)
    print("  part B OK: tonic gate (EITHER reference passes with ITS collateral; cued numbers "
          "untouched) + the score reads the PASSING reference")


def part_b_release_math() -> None:
    """The release + collateral formula on constructed argmax predictions, in the unit
    _suppression_metrics uses: WHOLE-SEQUENCE exact match (S._seq_acc), where a release
    counts only when ALL of a row's affected slots flip to the wrong task.

    The rows are built so the token rate DISAGREES with the sequence rate — one row
    flips only part of its affected set, one row has a padded (masked-out) slot — which
    is the whole point: a token-accuracy check passes on both units and would not have
    caught the pre-2026-07-v2 mix.
    """
    torch.manual_seed(0)
    S_len = 12
    pos_all = torch.arange(2, 7)                                # 5 spec answer positions
    true_all = torch.randint(0, 10, (4, 5))                     # cued answers A(x) at pos_all
    aff_pos = torch.tensor([[2, 3, 4], [2, 3, 4], [2, 3, 4], [2, 3, 2]])   # row 3: 2 real + pad
    mask = torch.tensor([[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 0]], dtype=torch.bool)

    pred_full = torch.zeros(4, S_len, dtype=torch.long)
    pred_full[:, 2:7] = true_all                                # full model: cued answer, perfect
    a_tgt = S._at(pred_full, aff_pos)                           # A(x) at the affected slots
    b_tgt = (a_tgt + 1) % 10                                    # B(x) != A(x)
    pred_sever = pred_full.clone()
    for row, flip in ((0, (2, 3, 4)), (1, (2, 3, 4)), (2, (2,)), (3, (2, 3))):
        for p in flip:                                          # row 2 flips ONE of three slots
            pred_sever[row, p] = (pred_sever[row, p] + 1) % 10

    full_aff, sev_aff = S._at(pred_full, aff_pos), S._at(pred_sever, aff_pos)
    b_full, b_sev = S._seq_acc(full_aff, b_tgt, mask), S._seq_acc(sev_aff, b_tgt, mask)
    assert b_full == 0.0, f"full model must never win as the wrong task, got {b_full}"
    assert b_sev == 0.75, f"rows 0/1/3 fully release, row 2 partially: expect 0.75, got {b_sev}"
    assert b_sev - b_full == 0.75
    tok = S._tok_acc(sev_aff, b_tgt, mask)                      # 9 of 11 valid slots
    assert abs(tok - 9 / 11) < 1e-9 and tok != b_sev, \
        "the synthetic case must make token and sequence rates DISAGREE"
    a_sev = S._seq_acc(sev_aff, a_tgt, mask)
    assert a_sev == 0.0, "every row lost at least one cued slot"

    comp = S._complement_mask(pos_all, aff_pos)                 # the untouched agree positions
    def collateral(sever):
        return (S._seq_acc(S._at(pred_full, pos_all), true_all, comp)
                - S._seq_acc(S._at(sever, pos_all), true_all, comp))
    assert collateral(pred_sever) == 0.0, "agree positions were untouched"
    wrecked = pred_sever.clone()
    wrecked[0, 6] = (wrecked[0, 6] + 1) % 10                    # one agree slot, one row
    assert collateral(wrecked) == 0.25, "one wrecked row of four = 0.25 whole-sequence"
    assert S._tok_acc(S._at(wrecked, pos_all), true_all, comp) != 0.75, \
        "the collateral case must separate the token rate from the sequence rate too"

    assert S._seq_acc(S._at(pred_full, aff_pos), b_tgt, mask) - b_full == 0.0   # no-op sever
    assert S.FRINGE_METRIC_TAG == "whole_sequence_exact_match", \
        "the tag stamped on these results must name the unit this part exercises"
    print("  part B OK: whole-sequence release math (0->0.75 with a partial-flip and a padded "
          "row where the token rate says 9/11), collateral 0 / 0.25 on the agree posns")


def part_c_fallback() -> None:
    """best_available_candidate: max cached acc; tie -> fewer edges -> name; stale/missing
    caches rank last; a circuit with zero candidates raises (stage-ordering bug)."""
    with tempfile.TemporaryDirectory() as td:
        def cand(name, k):   # k connected heads -> 2k edges (input->head->logits, prune-proof)
            d = os.path.join(td, "circuits", name); os.makedirs(d)
            edges = ([{"sender": "input", "receiver": f"a1.h{i}", "channel": "q", "score": 1.0}
                      for i in range(k)]
                     + [{"sender": f"a1.h{i}", "receiver": "logits", "channel": None, "score": 1.0}
                        for i in range(k)])
            json.dump({"circuit_heads": [f"L1H{i}" for i in range(k)], "edges": edges},
                      open(os.path.join(d, "circuit.json"), "w"))
        for name, k in [("eap_top100", 5), ("eap_top50", 3), ("aap_tie", 3), ("eap_top10", 1)]:
            cand(name, k)
        parsed = {c.name: c for c in S.discover_candidates([os.path.join(td, "circuits")])}
        assert set(parsed) == {"eap_top100", "eap_top50", "aap_tie", "eap_top10"}
        ev = os.path.join(td, ".smallest_circuit_evals", "evals"); os.makedirs(ev)
        def cache(name, acc, sig=None):
            json.dump({"edges_sig": sig or S._edges_sig(parsed[name].edges),
                       "perf_circuit": {"sequence_accuracy": acc}},
                      open(os.path.join(ev, f"{name}.json"), "w"))
        cache("eap_top100", 0.95); cache("eap_top50", 0.90)
        best, acc = S.best_available_candidate(td)
        assert (best.name, acc) == ("eap_top100", 0.95), "highest cached acc must win"
        cache("eap_top50", 0.95)
        assert S.best_available_candidate(td)[0].name == "eap_top50", "acc tie -> fewer edges"
        cache("aap_tie", 0.95)
        assert S.best_available_candidate(td)[0].name == "aap_tie", "acc+edges tie -> name"
        cache("eap_top10", 0.99, sig="stale")
        assert S.best_available_candidate(td)[0].name == "aap_tie", \
            "a stale edges_sig cache must not count as accuracy"
    with tempfile.TemporaryDirectory() as td:
        os.makedirs(os.path.join(td, "circuits"))
        try:
            S.best_available_candidate(td)
            raise AssertionError("zero candidates must raise FileNotFoundError")
        except FileNotFoundError:
            pass
    print("  part C OK: fallback rule (max acc; ties fewer-edges/name; stale cache ignored; "
          "zero candidates raise)")


def part_d_recording() -> None:
    """Fallback + verdict metadata on both output shapes; old markers fail _output_done."""
    assert "unfaithful base = best-available fallback" in S.SELECTION_STRATEGY_SUPPRESSION
    assert "unfaithful base = best-available fallback" in S.SELECTION_STRATEGY_DIFFERENTIAL
    a = S.parse_args(["--model", MUX, "--circuit", "suppress_t0_over_t1",
                      "--base-circuit", "task0", "--suppression",
                      "--router-verdict", "gating"])
    a.base_edges_sig = "cafef00d"
    a.attn_only = False
    a.candidate_set_sig = "deadbeef"
    a.base_fallback = {"base_faithful": False, "base_accuracy": 0.71,
                       "base_candidate": "eap_top50"}
    fringe = [{"parent": "a0.h0", "child": "a1.h1", "qkv": "q", "score": 1.0}]
    # a TONIC-only suppressor: the recorded cued release_rate is 0.0, so the payload's
    # release_scored proves the score reads the reference that PASSED
    res = {**_res("eap_top200", 12, 0.0, 0.01, release_tonic=0.8, collateral_tonic=0.01),
           "n_examples": S.SMALLEST_EVAL_N, "n_heads": 4, "heads": ["L0H0"],
           "perf_circuit": {"sequence_accuracy": 0.91}, "base_edges_sig": "cafef00d",
           "R_sequence_accuracy": 1.0, "R_accuracy": 1.0, "R_logit_diff": 1.0,
           "n_requested": 200, "pairs_regime": MG.CANON_PAIRS_REGIME,
           "pairs_window": SCORING_WINDOW, "metric_unit": S.FRINGE_METRIC_TAG,
           "perf_full": {"sequence_accuracy": 1.0}, "perf_empty": {"sequence_accuracy": 0.0},
           "fringe_heads": ["L1H1"], "fringe_edges": fringe}
    nf = S.build_no_faithful_payload({"eap_top200": res}, a)
    assert nf["no_faithful_circuit"] and nf["base_faithful"] is False
    assert (nf["pairs_regime"], nf["pairs_window"]) == (MG.CANON_PAIRS_REGIME, SCORING_WINDOW)
    assert nf["base_candidate"] == "eap_top50" and nf["base_accuracy"] == 0.71
    assert nf["router_verdict"] == "gating", "no_faithful must still record the verdict"
    cand = S.Candidate(name="eap_top200", source="/dev/null", dialect="manual",
                       heads=[(0, 0)], edges=fringe)
    win = S.build_payload(res, cand, a, {"eap_top200": 1}, 32, 1000)
    assert win["mode"] == "suppression" and win["base_faithful"] is False
    assert win["base_candidate"] == "eap_top50" and win["router_verdict"] == "gating"
    assert (win["pairs_regime"], win["pairs_window"]) == (MG.CANON_PAIRS_REGIME, SCORING_WINDOW)
    assert (win["release"]["release_reference"], win["release"]["release_scored"],
            win["release"]["release_rate"]) == ("tonic", 0.8, 0.0), \
        "the winner must be scored on the PASSING reference, not the cued rate"
    assert win["selection_score"] == S._suppression_score(res, 32, 1000, a.attn_only,
                                                          a.release_floor, a.epsilon)
    assert (nf["best_available_release_reference"],
            nf["best_available_release_scored"]) == ("tonic", 0.8)
    # differential fallback records base_* too; no router_verdict outside suppression
    d = S.parse_args(["--model", "segrev3_id", "--circuit", "three_detector",
                      "--base-circuit", "reverse_only"])
    d.base_edges_sig = "cafef00d"
    d.attn_only = True
    d.candidate_set_sig = "deadbeef"
    d.base_fallback = dict(a.base_fallback)
    dres = {**res, "s_union": 0.5}
    dnf = S.build_no_faithful_payload({"eap_top200": dres}, d)
    assert dnf["base_faithful"] is False and "router_verdict" not in dnf
    # faithful-base rows carry none of the fallback keys
    d.base_fallback = None
    assert "base_faithful" not in S.build_no_faithful_payload({"eap_top200": dres}, d)
    # OLD markers (no thresholds/strategy) fail _output_done -> reruns redo those rows
    with tempfile.TemporaryDirectory() as td:
        mk = os.path.join(td, "smallest_circuit.json")
        json.dump({"circuit_name": None, "no_faithful_circuit": True, "mode": "suppression",
                   "reason": "base_not_faithful", "base_circuit": "task0"}, open(mk, "w"))
        assert not S._output_done(mk, a, "deadbeef"), "old base_not_faithful marker must be invalidated"
        json.dump({"circuit_name": None, "no_faithful_circuit": True, "mode": "suppression",
                   "reason": "router_gating", "verdict": "gating", "base_circuit": "task0"},
                  open(mk, "w"))
        assert not S._output_done(mk, a, "deadbeef"), "old router_gating marker must be invalidated"
    print("  part D OK: base_faithful/base_accuracy/base_candidate + router_verdict recorded "
          "on winner AND no_faithful; old markers invalidated via _output_done")


def main() -> int:
    print("suppression finder self-check (no trained model):")
    part_a()
    part_b_selection()
    part_b_tonic_gate()
    part_b_release_math()
    part_c_fallback()
    part_d_recording()
    # smoke the CLI surface: --suppression parses and requires --base-circuit
    a = S.parse_args(["--model", MUX, "--circuit", "suppress_t0_over_t1",
                      "--base-circuit", "task0", "--suppression"])
    assert S._mode(a) == "suppression" and a.release_floor == 0.5 and a.epsilon == 0.05
    assert a.router_verdict is None
    plain = S.parse_args(["--model", "dblodd_id", "--circuit", "all_odd"])
    assert S._mode(plain) == "plain"
    diff = S.parse_args(["--model", "segrev3_id", "--circuit", "three_detector",
                         "--base-circuit", "reverse_only"])
    assert S._mode(diff) == "differential"
    try:                                        # --router-verdict is suppression-only
        with contextlib.redirect_stderr(io.StringIO()):
            S.parse_args(["--model", "dblodd_id", "--circuit", "all_odd",
                          "--router-verdict", "gating"])
        raise AssertionError("--router-verdict without --suppression must be rejected")
    except SystemExit:
        pass
    print("  CLI OK: --suppression -> mode dispatch; --router-verdict suppression-only; "
          "plain/differential unchanged")
    print("ALL SUPPRESSION SELF-CHECKS PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
