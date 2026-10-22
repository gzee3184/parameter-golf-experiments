# expT — Temperature Scaling at Eval

## Hypothesis
The trained model's logits may be slightly miscalibrated — too confident or too underconfident. Dividing logits by a scalar T before softmax can reduce CE loss **without changing the model**. This is "classical" temperature scaling from Guo et al. 2017.

**Why this could work on r10**:
- The model has `LOGIT_SOFTCAP=20`, which bounds logits to [-20, 20] via `tanh`. Competitors use softcap=30. Ours may be **under-confident** relative to them because softer bound → softer distribution. T < 1 would sharpen.
- Conversely, GPTQ quantization causes ~+0.002 damage; the quantized model may be over-confident on rare tokens (clipping error spreads mass). T > 1 would smooth.
- Either direction, some T ≠ 1.0 should beat T = 1.0.

## Control expectation
T = 1.00 must reproduce baseline: **val_bpb:1.1033** (post_ema, pre-quant).
If it doesn't, the harness has a bug — abort and investigate.

## What we're testing
Three-phase experiment:

**Phase A: Grid sweep (safe, reliable)**
- Run the canonical `train_gpt.py RESUME_QUANT=1` path
- Sweep T over {0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15} at the pre-quant eval step
- Each T costs ~70s eval time; 7 values ≈ 8 minutes total
- T=1.00 acts as our control

**Phase B: Optimal T via optimization (higher precision)**
- After grid narrows the range, fit T by minimizing CE on the SAME val (this is information leakage for testing purposes only — we'd need a held-out train chunk for a real submission)
- Report this as the "ceiling" — best achievable BPB if T were perfectly tuned on val

**Phase C: Train-side T (legal submission version)**
- Fit T on 1M tokens of TRAINING data (not val), minimize CE
- Report this as the "realizable" BPB for submission

If Phase C beats baseline by >= 0.003 BPB, the technique is worth deploying on H100.

## Implementation plan
Modify a COPY of `train_gpt.py` at `expT_temp_scaling/train_gpt.py`:
1. Add env var `EVAL_TEMP` (default 1.0)
2. In `GPT.forward`, divide logits by `EVAL_TEMP` when model is in eval mode, AFTER softcap
3. In `GPT.fwl` (used by sliding eval), same change
4. Verify T=1.0 produces 1.1033

The modification is ~5 lines. All other baseline behavior identical.

## Files
- `train_gpt.py` — modified with EVAL_TEMP env var
- `run_sweep.sh` — sweeps EVAL_TEMP over grid
- `fit_T_traindata.py` — standalone: fits T on training data tokens by gradient descent
- `results.md` — final results table (filled in as phases complete)

## Revert
```bash
# This experiment doesn't modify the main train_gpt.py — it's all in expT_temp_scaling/
# Nothing to revert.
```

## Success criterion
Phase C result ≥ 0.003 BPB better than 1.1033. Legal (train-side T).
