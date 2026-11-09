# Phase-2 patching + DLA report

**test_data:** random

- **Data**: `datasets/manual/random.json` (scheme=`random`, batch=128, path_sweep_batch=48, 4 layers x 8 heads)
- **Clean logit-diff**: +10.214  |  **Corrupted logit-diff**: -10.289

## Top direct logit writers (DLA, single-position contrast)
By |attribution|: `L3H1` (+0.884), `L3H6` (+0.859), `L3H3` (+0.783), `L3H4` (+0.671), `L3H5` (+0.655), `L3H2` (+0.593), `L2H7` (+0.558), `L2H1` (+0.523)

Most positive (write toward correct answer): `L3H1` (+0.884), `L3H6` (+0.859), `L3H3` (+0.783), `L3H4` (+0.671), `L3H5` (+0.655), `L3H2` (+0.593), `L2H7` (+0.558), `L2H1` (+0.523)

## Top heads by activation-patching importance (z, noising = necessity)
`L0H2` (+0.027), `L0H4` (+0.027), `L0H7` (+0.025), `L0H1` (+0.022), `L0H6` (+0.018), `L0H3` (+0.016), `L0H0` (+0.011), `L0H5` (+0.008)

## Top heads by attribution-patching importance (EAP-IG(8), denoising)
`L3H1` (+0.009), `L3H6` (+0.008), `L3H3` (+0.008), `L3H4` (+0.007), `L0H4` (+0.007), `L3H5` (+0.006), `L3H2` (+0.006), `L2H7` (+0.005)

## Top causal path edges (combined Q/K/V, noising)
`L0H7->L1H7` (+0.019), `L0H7->L1H1` (+0.016), `L0H2->L1H1` (+0.016), `L0H4->L1H6` (+0.015), `L0H2->L1H5` (+0.015), `L0H2->L1H4` (+0.014), `L0H2->L1H0` (+0.012), `L0H7->L1H4` (+0.012)

## Figures
- `activation_heatmap.png` -- 4x8 activation-patching (z, noising) head necessity, signed effect at the max-|effect| position.
- `attribution_heatmap.png` -- 4x8 EAP-IG(8) attribution-patching (z, denoising) head importance, signed likewise.
- `dla_heatmap.png` -- 4x8 direct-logit-attribution: each head's direct write onto the (correct-incorrect) logit direction.
- `path_heatmap.png` -- 32x32 sender->receiver path effect, combining Q/K/V by the strongest-|.| channel (signed, RdBu_r).
