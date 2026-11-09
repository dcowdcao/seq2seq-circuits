# Edge-circuit (Phase 3, weight-based) report

**test_data (attn):** n_threes_0, n_threes_1, n_threes_10, n_threes_2, n_threes_3, n_threes_4, n_threes_5, n_threes_6, n_threes_7, n_threes_8, n_threes_9

- **QK / OV / copying-score / composition are WEIGHT-BASED (independent of test_data); written once, flat.**
- The realized-attention plots/stats (`*_attn.png`) ARE data-dependent and were computed per dataset for: n_threes_0, n_threes_1, n_threes_10, n_threes_2, n_threes_3, n_threes_4, n_threes_5, n_threes_6, n_threes_7, n_threes_8, n_threes_9 (see `figs/edge/<label>/specific_heads/` and `attn_by_dataset`).

## Top OV copy-score heads  (weight-based)
- L0H1: +0.935
- L0H2: +0.931
- L0H0: +0.889
- L0H6: +0.886
- L0H3: +0.816
- L0H4: +0.727
- L0H7: +0.706
- L1H0: +0.155

## Top composition edges  (weight-based; random baseline = 0.088)
- Q: L0H4->L3H0 (0.208), L0H4->L3H5 (0.208), L0H4->L2H7 (0.208)
- K: L0H4->L3H4 (0.197), L0H4->L3H2 (0.197), L0H4->L2H3 (0.181)
- V: L0H4->L3H4 (0.170), L0H4->L1H6 (0.158), L0H4->L1H3 (0.145)

## Requested specific heads  (QK/OV weight-based)
- L3H1: copy=-0.378; QK top q1->k1 (0.01); OV top attend 7->out 2 (0.16)
- L3H3: copy=-0.310; QK top q2->k2 (0.01); OV top attend 9->out 5 (0.32)
- L3H6: copy=-0.253; QK top q7->k7 (0.01); OV top attend 7->out 1 (0.16)
- L2H7: copy=-0.347; QK top q2->k2 (0.01); OV top attend 3->out 1 (0.24)
- L0H7: copy=+0.706; QK top q2->k3 (0.01); OV top attend 8->out 8 (0.23)
- L0H4: copy=+0.727; QK top q1->k3 (0.00); OV top attend 3->out 3 (1.86)

## Realized attention: reflected position + 3-content split  (DATA-DEPENDENT, per dataset)
### test_data: n_threes_0
- L3H1: mean-mass-to-reflected-pos=0.217, mass-on-3-sources@answers=0.000 (uniform baseline 0.000), mean-max-attention=0.849
- L3H3: mean-mass-to-reflected-pos=0.256, mass-on-3-sources@answers=0.000 (uniform baseline 0.000), mean-max-attention=0.832
- L3H6: mean-mass-to-reflected-pos=0.210, mass-on-3-sources@answers=0.000 (uniform baseline 0.000), mean-max-attention=0.841
- L2H7: mean-mass-to-reflected-pos=0.384, mass-on-3-sources@answers=0.000 (uniform baseline 0.000), mean-max-attention=0.925
- L0H7: mean-mass-to-reflected-pos=0.898, mass-on-3-sources@answers=0.000 (uniform baseline 0.000), mean-max-attention=0.909
- L0H4: mean-mass-to-reflected-pos=0.048, mass-on-3-sources@answers=0.000 (uniform baseline 0.000), mean-max-attention=0.749

### test_data: n_threes_1
- L3H1: mean-mass-to-reflected-pos=0.043, mass-on-3-sources@answers=0.149 (uniform baseline 0.102), mean-max-attention=0.657
- L3H3: mean-mass-to-reflected-pos=0.036, mass-on-3-sources@answers=0.162 (uniform baseline 0.102), mean-max-attention=0.547
- L3H6: mean-mass-to-reflected-pos=0.041, mass-on-3-sources@answers=0.150 (uniform baseline 0.102), mean-max-attention=0.581
- L2H7: mean-mass-to-reflected-pos=0.057, mass-on-3-sources@answers=0.146 (uniform baseline 0.102), mean-max-attention=0.583
- L0H7: mean-mass-to-reflected-pos=0.189, mass-on-3-sources@answers=0.729 (uniform baseline 0.102), mean-max-attention=0.561
- L0H4: mean-mass-to-reflected-pos=0.045, mass-on-3-sources@answers=0.194 (uniform baseline 0.102), mean-max-attention=0.710

### test_data: n_threes_10
- L3H1: mean-mass-to-reflected-pos=0.099, mass-on-3-sources@answers=1.000 (uniform baseline 1.000), mean-max-attention=0.453
- L3H3: mean-mass-to-reflected-pos=0.092, mass-on-3-sources@answers=1.000 (uniform baseline 1.000), mean-max-attention=0.440
- L3H6: mean-mass-to-reflected-pos=0.089, mass-on-3-sources@answers=1.000 (uniform baseline 1.000), mean-max-attention=0.468
- L2H7: mean-mass-to-reflected-pos=0.086, mass-on-3-sources@answers=1.000 (uniform baseline 1.000), mean-max-attention=0.394
- L0H7: mean-mass-to-reflected-pos=0.104, mass-on-3-sources@answers=1.000 (uniform baseline 1.000), mean-max-attention=0.978
- L0H4: mean-mass-to-reflected-pos=0.025, mass-on-3-sources@answers=1.000 (uniform baseline 1.000), mean-max-attention=0.748

### test_data: n_threes_2
- L3H1: mean-mass-to-reflected-pos=0.052, mass-on-3-sources@answers=0.221 (uniform baseline 0.200), mean-max-attention=0.607
- L3H3: mean-mass-to-reflected-pos=0.071, mass-on-3-sources@answers=0.191 (uniform baseline 0.200), mean-max-attention=0.523
- L3H6: mean-mass-to-reflected-pos=0.054, mass-on-3-sources@answers=0.172 (uniform baseline 0.200), mean-max-attention=0.541
- L2H7: mean-mass-to-reflected-pos=0.089, mass-on-3-sources@answers=0.225 (uniform baseline 0.200), mean-max-attention=0.540
- L0H7: mean-mass-to-reflected-pos=0.218, mass-on-3-sources@answers=0.707 (uniform baseline 0.200), mean-max-attention=0.524
- L0H4: mean-mass-to-reflected-pos=0.044, mass-on-3-sources@answers=0.345 (uniform baseline 0.200), mean-max-attention=0.682

### test_data: n_threes_3
- L3H1: mean-mass-to-reflected-pos=0.040, mass-on-3-sources@answers=0.297 (uniform baseline 0.299), mean-max-attention=0.579
- L3H3: mean-mass-to-reflected-pos=0.049, mass-on-3-sources@answers=0.253 (uniform baseline 0.299), mean-max-attention=0.515
- L3H6: mean-mass-to-reflected-pos=0.036, mass-on-3-sources@answers=0.192 (uniform baseline 0.299), mean-max-attention=0.519
- L2H7: mean-mass-to-reflected-pos=0.065, mass-on-3-sources@answers=0.310 (uniform baseline 0.299), mean-max-attention=0.553
- L0H7: mean-mass-to-reflected-pos=0.230, mass-on-3-sources@answers=0.680 (uniform baseline 0.299), mean-max-attention=0.511
- L0H4: mean-mass-to-reflected-pos=0.048, mass-on-3-sources@answers=0.482 (uniform baseline 0.299), mean-max-attention=0.659

### test_data: n_threes_4
- L3H1: mean-mass-to-reflected-pos=0.040, mass-on-3-sources@answers=0.342 (uniform baseline 0.396), mean-max-attention=0.579
- L3H3: mean-mass-to-reflected-pos=0.051, mass-on-3-sources@answers=0.296 (uniform baseline 0.396), mean-max-attention=0.505
- L3H6: mean-mass-to-reflected-pos=0.033, mass-on-3-sources@answers=0.191 (uniform baseline 0.396), mean-max-attention=0.513
- L2H7: mean-mass-to-reflected-pos=0.066, mass-on-3-sources@answers=0.355 (uniform baseline 0.396), mean-max-attention=0.562
- L0H7: mean-mass-to-reflected-pos=0.240, mass-on-3-sources@answers=0.678 (uniform baseline 0.396), mean-max-attention=0.510
- L0H4: mean-mass-to-reflected-pos=0.045, mass-on-3-sources@answers=0.596 (uniform baseline 0.396), mean-max-attention=0.644

### test_data: n_threes_5
- L3H1: mean-mass-to-reflected-pos=0.045, mass-on-3-sources@answers=0.375 (uniform baseline 0.497), mean-max-attention=0.571
- L3H3: mean-mass-to-reflected-pos=0.048, mass-on-3-sources@answers=0.336 (uniform baseline 0.497), mean-max-attention=0.505
- L3H6: mean-mass-to-reflected-pos=0.031, mass-on-3-sources@answers=0.207 (uniform baseline 0.497), mean-max-attention=0.503
- L2H7: mean-mass-to-reflected-pos=0.073, mass-on-3-sources@answers=0.403 (uniform baseline 0.497), mean-max-attention=0.555
- L0H7: mean-mass-to-reflected-pos=0.226, mass-on-3-sources@answers=0.667 (uniform baseline 0.497), mean-max-attention=0.522
- L0H4: mean-mass-to-reflected-pos=0.045, mass-on-3-sources@answers=0.682 (uniform baseline 0.497), mean-max-attention=0.641

### test_data: n_threes_6
- L3H1: mean-mass-to-reflected-pos=0.038, mass-on-3-sources@answers=0.435 (uniform baseline 0.593), mean-max-attention=0.553
- L3H3: mean-mass-to-reflected-pos=0.037, mass-on-3-sources@answers=0.358 (uniform baseline 0.593), mean-max-attention=0.479
- L3H6: mean-mass-to-reflected-pos=0.018, mass-on-3-sources@answers=0.214 (uniform baseline 0.593), mean-max-attention=0.482
- L2H7: mean-mass-to-reflected-pos=0.063, mass-on-3-sources@answers=0.465 (uniform baseline 0.593), mean-max-attention=0.532
- L0H7: mean-mass-to-reflected-pos=0.218, mass-on-3-sources@answers=0.688 (uniform baseline 0.593), mean-max-attention=0.530
- L0H4: mean-mass-to-reflected-pos=0.040, mass-on-3-sources@answers=0.756 (uniform baseline 0.593), mean-max-attention=0.644

### test_data: n_threes_7
- L3H1: mean-mass-to-reflected-pos=0.043, mass-on-3-sources@answers=0.512 (uniform baseline 0.693), mean-max-attention=0.511
- L3H3: mean-mass-to-reflected-pos=0.045, mass-on-3-sources@answers=0.440 (uniform baseline 0.693), mean-max-attention=0.430
- L3H6: mean-mass-to-reflected-pos=0.017, mass-on-3-sources@answers=0.244 (uniform baseline 0.693), mean-max-attention=0.428
- L2H7: mean-mass-to-reflected-pos=0.067, mass-on-3-sources@answers=0.548 (uniform baseline 0.693), mean-max-attention=0.489
- L0H7: mean-mass-to-reflected-pos=0.197, mass-on-3-sources@answers=0.705 (uniform baseline 0.693), mean-max-attention=0.544
- L0H4: mean-mass-to-reflected-pos=0.038, mass-on-3-sources@answers=0.821 (uniform baseline 0.693), mean-max-attention=0.663

### test_data: n_threes_8
- L3H1: mean-mass-to-reflected-pos=0.047, mass-on-3-sources@answers=0.631 (uniform baseline 0.796), mean-max-attention=0.459
- L3H3: mean-mass-to-reflected-pos=0.050, mass-on-3-sources@answers=0.569 (uniform baseline 0.796), mean-max-attention=0.384
- L3H6: mean-mass-to-reflected-pos=0.017, mass-on-3-sources@answers=0.331 (uniform baseline 0.796), mean-max-attention=0.379
- L2H7: mean-mass-to-reflected-pos=0.065, mass-on-3-sources@answers=0.642 (uniform baseline 0.796), mean-max-attention=0.443
- L0H7: mean-mass-to-reflected-pos=0.163, mass-on-3-sources@answers=0.745 (uniform baseline 0.796), mean-max-attention=0.596
- L0H4: mean-mass-to-reflected-pos=0.035, mass-on-3-sources@answers=0.884 (uniform baseline 0.796), mean-max-attention=0.683

### test_data: n_threes_9
- L3H1: mean-mass-to-reflected-pos=0.062, mass-on-3-sources@answers=0.796 (uniform baseline 0.898), mean-max-attention=0.386
- L3H3: mean-mass-to-reflected-pos=0.065, mass-on-3-sources@answers=0.770 (uniform baseline 0.898), mean-max-attention=0.332
- L3H6: mean-mass-to-reflected-pos=0.031, mass-on-3-sources@answers=0.557 (uniform baseline 0.898), mean-max-attention=0.352
- L2H7: mean-mass-to-reflected-pos=0.066, mass-on-3-sources@answers=0.829 (uniform baseline 0.898), mean-max-attention=0.377
- L0H7: mean-mass-to-reflected-pos=0.140, mass-on-3-sources@answers=0.850 (uniform baseline 0.898), mean-max-attention=0.729
- L0H4: mean-mass-to-reflected-pos=0.030, mass-on-3-sources@answers=0.939 (uniform baseline 0.898), mean-max-attention=0.715

## Generated figures
Weight-based (flat, independent of test_data):
- figs/edge/copy_scores.png
- figs/edge/composition_q.png, composition_k.png, composition_v.png
- figs/edge/specific_heads/L3H1_QK.png, L3H1_OV.png
- figs/edge/specific_heads/L3H3_QK.png, L3H3_OV.png
- figs/edge/specific_heads/L3H6_QK.png, L3H6_OV.png
- figs/edge/specific_heads/L2H7_QK.png, L2H7_OV.png
- figs/edge/specific_heads/L0H7_QK.png, L0H7_OV.png
- figs/edge/specific_heads/L0H4_QK.png, L0H4_OV.png

Data-dependent realized attention (per dataset; *_attn = positional view, *_attn_content = 3-vs-non-3 source split):
- figs/edge/n_threes_0/specific_heads/: L3H1_attn{,_content}.png, L3H3_attn{,_content}.png, L3H6_attn{,_content}.png, L2H7_attn{,_content}.png, L0H7_attn{,_content}.png, L0H4_attn{,_content}.png
- figs/edge/n_threes_1/specific_heads/: L3H1_attn{,_content}.png, L3H3_attn{,_content}.png, L3H6_attn{,_content}.png, L2H7_attn{,_content}.png, L0H7_attn{,_content}.png, L0H4_attn{,_content}.png
- figs/edge/n_threes_10/specific_heads/: L3H1_attn{,_content}.png, L3H3_attn{,_content}.png, L3H6_attn{,_content}.png, L2H7_attn{,_content}.png, L0H7_attn{,_content}.png, L0H4_attn{,_content}.png
- figs/edge/n_threes_2/specific_heads/: L3H1_attn{,_content}.png, L3H3_attn{,_content}.png, L3H6_attn{,_content}.png, L2H7_attn{,_content}.png, L0H7_attn{,_content}.png, L0H4_attn{,_content}.png
- figs/edge/n_threes_3/specific_heads/: L3H1_attn{,_content}.png, L3H3_attn{,_content}.png, L3H6_attn{,_content}.png, L2H7_attn{,_content}.png, L0H7_attn{,_content}.png, L0H4_attn{,_content}.png
- figs/edge/n_threes_4/specific_heads/: L3H1_attn{,_content}.png, L3H3_attn{,_content}.png, L3H6_attn{,_content}.png, L2H7_attn{,_content}.png, L0H7_attn{,_content}.png, L0H4_attn{,_content}.png
- figs/edge/n_threes_5/specific_heads/: L3H1_attn{,_content}.png, L3H3_attn{,_content}.png, L3H6_attn{,_content}.png, L2H7_attn{,_content}.png, L0H7_attn{,_content}.png, L0H4_attn{,_content}.png
- figs/edge/n_threes_6/specific_heads/: L3H1_attn{,_content}.png, L3H3_attn{,_content}.png, L3H6_attn{,_content}.png, L2H7_attn{,_content}.png, L0H7_attn{,_content}.png, L0H4_attn{,_content}.png
- figs/edge/n_threes_7/specific_heads/: L3H1_attn{,_content}.png, L3H3_attn{,_content}.png, L3H6_attn{,_content}.png, L2H7_attn{,_content}.png, L0H7_attn{,_content}.png, L0H4_attn{,_content}.png
- figs/edge/n_threes_8/specific_heads/: L3H1_attn{,_content}.png, L3H3_attn{,_content}.png, L3H6_attn{,_content}.png, L2H7_attn{,_content}.png, L0H7_attn{,_content}.png, L0H4_attn{,_content}.png
- figs/edge/n_threes_9/specific_heads/: L3H1_attn{,_content}.png, L3H3_attn{,_content}.png, L3H6_attn{,_content}.png, L2H7_attn{,_content}.png, L0H7_attn{,_content}.png, L0H4_attn{,_content}.png
