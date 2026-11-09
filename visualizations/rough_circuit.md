# segrev3 — heuristic manual circuit (rough)

Synthesized from `visualizations/manual_circuit_analysis/figs_data/data/random/patching_dla_results.json` + `.../edge_results.json`. Heuristic selection: movers = top-6 heads by |DLA|, detectors = top-6 by activation-patching necessity, plus endpoints of strong Q/K/V composition edges.

## Circuit heads & hypothesized roles

- **L3H1** — mover / copy head (writes the answer digit to the logits; high DLA) (OV copy score -0.38)
- **L3H6** — mover / copy head (writes the answer digit to the logits; high DLA) (OV copy score -0.25)
- **L3H3** — mover / copy head (writes the answer digit to the logits; high DLA) (OV copy score -0.31)
- **L3H4** — mover / copy head (writes the answer digit to the logits; high DLA) (OV copy score -0.33)
- **L3H5** — mover / copy head (writes the answer digit to the logits; high DLA) (OV copy score -0.23)
- **L3H2** — mover / copy head (writes the answer digit to the logits; high DLA) (OV copy score -0.68)
- **L0H2** — L0 detector (boundary/position; high activation-patching necessity, feeds composition) (OV copy score +0.93)
- **L0H4** — L0 detector (boundary/position; high activation-patching necessity, feeds composition) (OV copy score +0.73)
- **L0H7** — L0 detector (boundary/position; high activation-patching necessity, feeds composition) (OV copy score +0.71)
- **L0H1** — L0 detector (boundary/position; high activation-patching necessity, feeds composition) (OV copy score +0.94)
- **L0H6** — L0 detector (boundary/position; high activation-patching necessity, feeds composition) (OV copy score +0.89)
- **L0H3** — L0 detector (boundary/position; high activation-patching necessity, feeds composition) (OV copy score +0.82)
- **L3H0** — intermediate composition head (reflect-index / routing) (OV copy score -0.20)
- **L2H7** — intermediate composition head (reflect-index / routing) (OV copy score -0.35)
- **L2H5** — intermediate composition head (reflect-index / routing) (OV copy score -0.28)
- **L2H6** — intermediate composition head (reflect-index / routing) (OV copy score -0.22)
- **L2H3** — intermediate composition head (reflect-index / routing) (OV copy score -0.24)
- **L2H1** — intermediate composition head (reflect-index / routing) (OV copy score -0.32)
- **L2H2** — intermediate composition head (reflect-index / routing) (OV copy score -0.46)
- **L2H0** — intermediate composition head (reflect-index / routing) (OV copy score +0.02)
- **L1H6** — intermediate composition head (reflect-index / routing) (OV copy score -0.17)

## Mechanism (hypothesis)

Input digits → **L0 detector / reflection heads** (causal: each output position 9+k attends back to its reflected input position 9-k; on inputs with a 3 they attend to the boundary instead — highest necessity, high OV copy) → **composition** (L0H4 routes the *where* into the movers' queries; L0H7 routes value/identity into L1) → **reflect-index computed from the input + the autoregressive output prefix** (a weak positional reflection backbone in L0, modulated dynamically by the 3-boundaries) → **L2/L3 mover heads** copy the digit at the reflected index to the logits (highest DLA).

## Key empirical notes

- The model is CAUSAL: output position 9+k attends only to positions <=9+k. The reversal mirror is the CAUSAL reflection diagonal (dest 9+k -> reflected input src 9-k). The positional QK weight circuit shows a weak reflection signal concentrated in the L0 heads (L0H7/H1/H2 ~0.02-0.03); the STRONG realized reflection (L0H7 attends ~0.9 from each output position to its reflected input position on no-3 inputs) is emergent from positional + content + composition + softmax, and is dynamically modulated by the 3-boundaries (on inputs with a 3, L0H7 instead attends to the 3).
- L0 heads have high OV copy scores (~0.9): they propagate token identity. L0H4 is the dominant Q-composition hub into the movers; L0H7 the V hub into L1.
