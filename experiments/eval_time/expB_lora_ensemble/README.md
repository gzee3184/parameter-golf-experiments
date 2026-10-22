# expB — LoRA Ensemble at Eval

## Hypothesis
Three independently-trained LoRA adapters over the frozen r10 backbone, with logit averaging at eval time, will reduce val_bpb by 0.005-0.015 via ensembling variance reduction.

**Why this might work**: Each LoRA sees a different permutation of training data and starts from a different init. The backbone is fixed, so they converge to different "modes" of residual correction. Averaging their output distributions cuts independent noise by ~1/√3.

**Why it might not**: LoRAs over the same backbone may converge to highly correlated updates (same gradient landscape from the frozen base). Expected benefit could be near zero if correlation > 0.9.

## Method
1. Load `h100_deploy/run10/final_model_r10.pt` (135MB EMA state dict, pre-quant 1.1033)
2. Freeze everything except LoRA params
3. Train 3 LoRAs, seeds {1337, 42, 7}, rank 16, ~200 steps each (fast)
4. At eval: for each token, run forward through all 3 LoRAs, average logits in log-space, compute CE.

## Storage cost (for eventual H100 submission)
- 3× (rank 16 × 512d × 2 matrices × 11 layers) = ~500KB at int6 = negligible

## Legality
Pure train-side ensemble. No val leakage. Similar in spirit to snapshot ensembles. Legal.

## Test plan (local)
1. Load r10 checkpoint, evaluate baseline → should reproduce 1.1033
2. Train LoRA-0 (seed 1337) for 200 steps → eval
3. Train LoRA-1 (seed 42) for 200 steps → eval
4. Train LoRA-2 (seed 7) for 200 steps → eval
5. Ensemble eval: average log-probs across all 3

## Success criterion
Ensemble pre-quant val_bpb ≤ **1.1003** (baseline 1.1033 − 0.003).

## Files
- `eval_from_checkpoint.py` — loads r10, runs eval with {1, 2, 3} LoRAs
- `run_local.sh` — driver script

## Revert
```bash
rm -rf ./lora_*.pt  # only local checkpoints
```

No train_gpt.py changes for this exp — it's a standalone eval harness.
