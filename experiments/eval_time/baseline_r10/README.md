# baseline_r10 — FROZEN REFERENCE

**DO NOT MODIFY**. This directory is the frozen baseline for experiment comparisons.

## What this is
Exact copy of `h100_deploy/train_gpt.py`, `train_gpt_lzma.py`, `run_h100.sh` as they existed at the start of the Run #11 experimentation phase (2026-04-19).

This code produced **Run #10c**: TTT **1.1009**, sliding **1.1056**, pre-quant **1.1033**.

## Baseline metrics (Run #10c on 8×H100)
| Metric | Value |
|--------|-------|
| Pre-quant BPB (post-EMA) | 1.1033 |
| Quant BPB (roundtrip) | 1.1056 |
| Sliding BPB (stride=64) | 1.1056 |
| TTT BPB | 1.1009 |
| Quant gap | 0.0023 |
| Pruning | 2.1% |
| Artifact | 15,991,240 bytes |

## Associated checkpoint
`h100_deploy/run10/final_model_r10.pt` (135MB EMA state dict).
Used by experiments B, C which evaluate without retraining.

## Revert command
From any experiment directory:
```bash
cp ../baseline_r10/train_gpt.py ../../train_gpt.py
cp ../baseline_r10/train_gpt_lzma.py ../../train_gpt_lzma.py
cp ../baseline_r10/run_h100.sh ../../run_h100.sh
```
