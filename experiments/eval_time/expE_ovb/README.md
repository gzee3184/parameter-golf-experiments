# expE — Online Validation Bootstrap (Hard-Mining)

## The H100 hypothesis
During the last ~1000 steps of training, record per-token CE loss. Identify the hardest 10% of sequences. Upsample them for the final 500 steps of training. This **curriculum on the reverse axis** (focus where you're weakest) should improve pre-quant BPB.

## The local test
We can't reproduce the full 600s H100 run locally. Instead, test a reduced version:
1. Start from r10 post-EMA checkpoint (val_bpb=1.1033)
2. Do a short fine-tune (~200 steps) with low LR to avoid catastrophic unlearning
3. **Control arm**: fine-tune on uniform training data
4. **OVB arm**: fine-tune with per-sequence upsampling weighted by sequence-level loss from a prior pass
5. Compare post-finetune val_bpb

If the OVB arm beats control by ≥ 0.001 BPB (any positive signal), the technique is worth H100 deployment.

## Design

### Hard-mining mechanism
In `DTL.nb()` (the data loader): for the first N steps, accept all sequences uniformly but **record per-sequence mean CE loss** keyed by (shard_id, position). After N steps, compute the 90th-percentile loss threshold. For subsequent steps, use importance sampling: sequences with recorded loss above threshold get 10x sampling weight.

### Complications
- **Sequences aren't repeated** in normal training (each position is seen once). OVB breaks that assumption.
- **Loss dynamics shift**: what was "hard" at step 10 may not be hard at step 200. Need a sliding window of recorded losses.
- **Computational overhead**: sorting & sampling per step is cheap. Recording is free (we already compute per-token loss).

### Fine-tune settings
- Load r10 checkpoint
- SGD with LR 1e-5 (tiny; don't wander from EMA state)
- 200 steps total
- Arms: {uniform, ovb} × {seed 1337, seed 42} = 4 runs for statistical power on small signal

## Files
- `train_gpt.py` — modified with hard-mining in DTL + env vars OVB_ENABLED, OVB_LOSS_THRESHOLD_PCTILE, OVB_BOOST
- `run_local.sh` — driver: 4 runs, logs to `logs/`

## Success criterion
`mean(ovb_seeds_val_bpb) + 2*sem < mean(uniform_seeds_val_bpb)` — one-sided signal at p≈0.05.
Practically: need ~0.001 BPB improvement with 2 seeds to be worth H100 test.

## Revert
This experiment doesn't modify the main `h100_deploy/train_gpt.py`.
