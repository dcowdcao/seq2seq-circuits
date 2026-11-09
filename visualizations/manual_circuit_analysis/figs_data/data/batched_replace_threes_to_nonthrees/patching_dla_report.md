# Phase-2 patching + DLA report

**test_data:** batched_replace_threes_to_nonthrees

- **Data**: `datasets/manual/batched_replace_threes_to_nonthrees.json` (scheme=`replace_threes_to_nonthrees`, batch=128, path_sweep_batch=48, 4 layers x 8 heads)
- **Clean logit-diff**: +7.763  |  **Corrupted logit-diff**: -8.546

## Top direct logit writers (DLA, single-position contrast)
By |attribution|: `L3H6` (+0.893), `L3H1` (+0.855), `L3H4` (+0.760), `L3H3` (+0.736), `L0H4` (+0.653), `L3H5` (+0.647), `L3H2` (+0.622), `L2H7` (+0.517)

Most positive (write toward correct answer): `L3H6` (+0.893), `L3H1` (+0.855), `L3H4` (+0.760), `L3H3` (+0.736), `L0H4` (+0.653), `L3H5` (+0.647), `L3H2` (+0.622), `L2H7` (+0.517)

## Top heads by activation-patching importance (z, noising = necessity)
`L0H1` (+0.045), `L0H7` (+0.038), `L0H6` (+0.038), `L0H2` (+0.034), `L0H4` (+0.030), `L0H5` (+0.021), `L0H3` (+0.017), `L1H3` (+0.011)

## Top heads by attribution-patching importance (EAP-IG(8), denoising)
`L3H1` (+0.009), `L3H3` (+0.008), `L3H6` (+0.008), `L3H4` (+0.007), `L3H5` (+0.006), `L2H7` (+0.006), `L3H2` (+0.006), `L2H1` (+0.006)

## Top causal path edges (combined Q/K/V, noising)
`L0H1->L1H6` (+0.019), `L0H2->L1H1` (+0.018), `L0H1->L1H3` (+0.017), `L0H2->L1H0` (+0.014), `L0H4->L1H6` (+0.013), `L0H2->L1H7` (+0.013), `L0H2->L1H2` (+0.013), `L0H7->L1H0` (+0.012)

## Figures
- `activation_heatmap.png` -- 4x8 activation-patching (z, noising) head necessity, signed effect at the max-|effect| position.
- `attribution_heatmap.png` -- 4x8 EAP-IG(8) attribution-patching (z, denoising) head importance, signed likewise.
- `dla_heatmap.png` -- 4x8 direct-logit-attribution: each head's direct write onto the (correct-incorrect) logit direction.
- `path_heatmap.png` -- 32x32 sender->receiver path effect, combining Q/K/V by the strongest-|.| channel (signed, RdBu_r).
