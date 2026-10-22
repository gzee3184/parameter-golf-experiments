# expDS Results — Deep Supervision at Eval (Multi-Layer Logit Ensemble)

**Date**: 2026-04-19
**Checkpoint**: `h100_deploy/run10/final_model_r10.pt`
**Baseline**: val_bpb=1.1033

## Result: NEGATIVE (strong)

Every ensemble variant underperforms baseline. Adding ANY weight to intermediate-layer logits strictly hurts.

| Arm | DS_N | Weights | val_loss | val_bpb | Δ vs baseline |
|-----|------|---------|---------:|--------:|--------------:|
| control_N0 | 0 | baseline path | 2.8501 | 1.1033 | 0.0000 |
| singleN1 | 1 | last only | 2.8501 | 1.1033 | 0.0000 (control via new path) |
| weighted_N3b | 3 | 0.1 / 0.2 / 0.7 | 2.9061 | 1.1251 | +0.0218 |
| uniformN2 | 2 | 0.5 / 0.5 | 2.9688 | 1.1493 | +0.0460 |
| weighted_N3a | 3 | 0.2 / 0.3 / 0.5 | 3.0121 | 1.1661 | +0.0628 |
| weighted_N4a | 4 | 0.1 / 0.15 / 0.25 / 0.5 | 3.0463 | 1.1793 | +0.0760 |
| uniformN3 | 3 | 1/3 each | 3.1740 | 1.2288 | +0.1255 |
| uniformN4 | 4 | 1/4 each | 3.4707 | 1.3436 | +0.2403 |

## Interpretation

**Intermediate hidden states are NOT a noisier version of the final state.** The tied embedding matrix applied to the output of any non-final decoder layer produces meaningfully worse predictions. This strongly contradicts the "residual stream as semantic basis" assumption for this architecture.

Likely cause: parallel residual lanes (psl=7) and the `self.lm` mixing scalar make the last layer's transformation qualitatively non-monotonic. Intermediate states aren't aimed at the vocabulary manifold; only the final state is.

## What this tells us about related ideas

- **DS-at-eval is dead** for this model. No weighting scheme recovers baseline.
- **Adaptive compute / early-exit** (my category C idea) is almost certainly dead too — exiting before the final layer would use exactly these "bad" intermediate predictions.
- **BUT**: this suggests **deep supervision DURING TRAINING** (adding CE loss at intermediate layers) might help significantly. The current architecture allows intermediate states to drift from the vocabulary manifold because there's no supervisory signal pulling them back. Adding intermediate CE (expDS-train) would force the residual stream to progressively refine toward the answer.

## Next

Moving to **expTq** (post-quant temperature scaling) — smaller hypothesis, faster test. Then **expDS-train** (deep supervision during training, unlike the failed eval-time version) if expTq is also dead.

## Not deploying

Do not stack expDS in H100 runs. Remove from consideration.
