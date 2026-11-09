#!/usr/bin/env python3
"""check_ckpt_gate.py <repo> <name> <ckpt> <gate> — exit 0 iff the checkpoint passes G-TRAIN.

Reads the recorded sequence accuracy via promote.recorded_seq_acc (the ONE gate's reader);
a checkpoint that records none gets a live eval on the DSL task (train.py's test-split
recipe). Run as a subprocess by run_model_pipeline's train skip-check so torch/GPU never
enter the orchestrator process.
"""
import os
import sys


def main() -> None:
    repo, name, ckpt, gate = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
    sys.path.insert(0, os.path.join(repo, "pipeline", "models"))
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import model_registry
    import promote
    import torch

    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    acc, src = promote.recorded_seq_acc(state)
    if acc is None:
        # No recorded sequence accuracy: recover a TRUE one with a live eval on the DSL
        # task — same recipe as train.py (test-split seed 3).
        spec = model_registry.get_spec(name)
        assert os.path.abspath(spec.checkpoint) == os.path.abspath(ckpt), \
            f"ckpt mismatch: {spec.checkpoint} != {ckpt}"
        model = model_registry.load_model(name)            # rails (mem cap) inherited
        import algorithms
        from metrics import task_accuracy
        g = torch.Generator().manual_seed(3)               # train.py test-split seed
        inp = torch.randint(0, 10, (4096, spec.input_len), generator=g)
        row_map = algorithms.decode_algorithm_id(name.partition("__")[0],
                                                 input_length=spec.input_len).sample_row
        inp = torch.stack([row_map(r) for r in inp])
        out = spec.task_fn(inp).long()
        dev = str(model.cfg.device)
        toks = torch.cat([inp, out], dim=1).long().to(dev)
        acc = float(task_accuracy(model, toks, out.to(dev),
                                  answer_positions=tuple(spec.answer_positions))["sequence_acc"])
        src = "live DSL eval (4096 examples, seed 3)"
    verdict = "PASS" if acc >= gate else "FAIL"
    print(f"CKPT-CHECK {verdict}: seq-acc={acc:.5f} (gate >= {gate}) from {src}")
    sys.exit(0 if acc >= gate else 1)


if __name__ == "__main__":
    main()
