# expT Results — Temperature Scaling at Eval (pre-quant)

**Date**: 2026-04-19
**Checkpoint**: `h100_deploy/run10/final_model_r10.pt` (post-EMA, 5051 H100 steps, SP8192)
**Baseline**: val_bpb=1.1033 (T=1.0 control reproduces exactly)

## Result: NEGATIVE

Temperature scaling does not help. T=1.0 is the optimum to grid resolution 0.05.

| T | val_loss | val_bpb | Δ vs baseline |
|---|---------:|--------:|--------------:|
| 0.85 | 2.9099 | 1.1265 | +0.0232 |
| 0.90 | 2.8762 | 1.1135 | +0.0102 |
| 0.95 | 2.8570 | 1.1060 | +0.0027 |
| **1.00** | **2.8501** | **1.1033** | 0.0000 (control) |
| 1.05 | 2.8539 | 1.1048 | +0.0015 |
| 1.10 | 2.8671 | 1.1099 | +0.0066 |
| 1.15 | 2.8884 | 1.1182 | +0.0149 |

**Curve shape**: clean V centered at T=1.00. Minimum is baseline.

## Interpretation

The model is **already calibrated** for cross-entropy on its validation distribution. Post-hoc temperature is a free lever in models trained with objectives that aren't pure CE (focal loss, label smoothing, classification with one-hot-hard targets). r10 was trained with bog-standard mean CE for 5000+ steps with EMA — it converges to CE-optimal calibration by construction.

**Tiny asymmetry**: T=1.05 (Δ+0.0015) slightly beats T=0.95 (Δ+0.0027). Possible interpretation: if anything, model is a hair overconfident on correct tokens. But the effect is within the curve's noise floor and not worth pursuing.

## What this tells us about the other ideas

- **NEW-3 variant (b) "online T during TTT"**: predicted gain ≤ 0.003 based on this data. Not worth the 40 lines.
- **Label-smoothing variant of NEW-2**: unlikely to help via calibration — but NEW-2's p_train blend is a distributional shift, not calibration, so still worth testing.
- **TTT gain of -0.0047 from baseline is NOT due to calibration** — it must be from actual adaptation to val tokens. Confirms TTT adapts parameters, not just post-hoc decision rules.

## Test-time cost on H100

Per-eval overhead of `logits / T`: ~20ms for 40M val tokens (the extra division). Negligible.

## Deployment decision: DO NOT DEPLOY

Adding EVAL_TEMP to the H100 submission with T=1.00 is a no-op and adds code size. Remove from the final stack.

## Next experiment

Moving to **NEW-2 (train distribution blend)**. Unlike temperature, this is a *distributional* prior shift — it changes what probability mass the model assigns to rare vs common tokens. Temperature only rescales; blend reallocates.
