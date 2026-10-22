# expZ Results — Z-Loss Regularization (Phase 1, short A/B)

**Date**: 2026-04-19
**Phase 1**: from-scratch 500 steps, control vs Z_LOSS_WEIGHT=1e-4

## Result: NEGATIVE at short-run scale

| Arm | final train_loss (step 500) | step-500 val_bpb | post_EMA val_bpb |
|-----|-----------------------------|------------------|------------------|
| control (Z=0) | 3.2023 | 1.2453 | 1.4274 |
| zloss_1em4 | 3.2275 | 1.2521 | 1.4780 |
| **Δ** | **+0.025 worse** | **+0.0068 worse** | **+0.0506 worse** |

## Caveats

1. **Far from convergence**: BPB=1.25 is nowhere near converged BPB=1.10. At this stage the model is still learning basic patterns. Regularization benefits (if any) only show near convergence.
2. **EMA not yet meaningful**: post-EMA BPB > current-state BPB suggests EMA weights are still dominated by init noise at step 500 with EMA_DECAY=0.9965.
3. **Ternary competitor uses Z-Loss at convergence** (5000+ steps). Not directly comparable to our 500-step test.

## What we learn

- Z-Loss doesn't provide early-training benefit for our architecture
- At this scale, the regularization may actually impede learning (extra gradient term competes with CE)
- Whether Z-Loss helps at FULL convergence requires a proper ~3000-step or longer retrain

## Decision: DO NOT escalate to Phase 2 locally

Rationale: local 3000-step A/B would cost ~3 hours × 2 arms = 6 hours, high risk of another negative. The Phase 1 signal isn't encouraging.

**Better path**: if we do test Z-Loss on H100, test it directly as part of an integrated H100 run with other changes, rather than burning local compute on isolated A/B that may not transfer.

## Next

Retry retraining experiments with a clearer signal-per-hour value. Options on the table:
- **expDual** (dual-stream byte+token embeddings) — architectural, high ceiling
- **expKD** (pretrained teacher distillation) — different training signal entirely
- Accept that cheap modifications are exhausted and go to H100 directly

## Recommendation for session

We've exhausted eval-only tricks (3 negatives) and the cheapest training-tweak (Z-Loss, marginal negative). Further local experiments need bigger changes (architectural/data). Consider: the gap to competitors may be **irreducible without a full redesign**, not a single clever addition.
