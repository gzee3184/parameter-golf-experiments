# expTq Results — Post-Quantization Temperature Scaling

**Date**: 2026-04-19
**Baseline**: Run #10c pre-quant 1.1033 (H100), local post-quant with 39% pruning

## Result: NEGATIVE

V-curve minimum at T=1.00 for BOTH pre-quant and post-quant. Symmetric within ±0.025 of T=1.

| T | pre_quant | post_quant (local) | Δ vs T=1 (post) |
|---|----------:|-------------------:|-----------------:|
| 0.90 | 1.1135 | 1.20292605 | +0.00792 |
| 0.95 | 1.1060 | 1.19632734 | +0.00132 |
| 0.975 | 1.1041 | 1.19505284 | +0.00005 |
| **1.00** | **1.1033** | **1.19500289** | **baseline** |
| 1.025 | 1.1036 | 1.19609254 | +0.00109 |
| 1.05 | 1.1048 | 1.19824309 | +0.00324 |
| 1.10 | 1.1099 | 1.20543967 | +0.01044 |

## Key finding

Post-quant model shows the same flat V at T=1 as pre-quant. Quantization damage (~0.09 local BPB) is distributed evenly across the distribution — no asymmetric miscalibration that T could fix.

## Quant damage vs T (pre → post gap)
- T=0.90: +0.0894
- T=1.00: +0.0917
- T=1.05: +0.0934

Slight trend: sharper distributions tolerate quantization marginally better. Not enough to overcome sharpness penalty.

## Note on local vs H100

Local shows 39% pruning and ~0.09 BPB quant damage. H100 has 2.1% pruning and ~0.002 BPB quant damage. Absolute numbers don't match, but **relative T comparison is valid**. The conclusion "T=1 is optimal" transfers.

## Deployment decision: DO NOT DEPLOY

Same as expT. Adding EVAL_TEMP to final stack is a no-op.

## Next

Moving to **expZ** (Z-Loss regularization during training). First experiment that actually modifies training, not just eval. Requires retraining from scratch.
