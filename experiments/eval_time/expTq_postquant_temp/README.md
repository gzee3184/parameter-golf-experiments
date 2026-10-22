# expTq — Post-Quantization Temperature Scaling

## Hypothesis
Pre-quant temperature showed a clean V at T=1.00 (expT). But GPTQ quantization causes ~+0.002 BPB damage. This damage often introduces asymmetric miscalibration — the quantized model may be over/under-confident in structured ways that a scalar T can partly correct. Post-quant optimum may be T≠1.

## Known limitation
Local compression/pruning differs from H100. **Absolute post-quant BPB won't match Run #10c's 1.1056**. But **relative T comparison is still valid**, because all T values share the same quant pipeline.

## Design
- Reuse expT's `EVAL_TEMP` env var in `GPT.forward` and `GPT.fwl` (already applies to post-quant eval path)
- Add `EXPTQ_EARLY_EXIT` to stop after roundtrip eval (skip sliding + TTT to save time)
- Grid T in {0.90, 0.95, 0.975, 1.00, 1.025, 1.05, 1.10}
- Each run: ~3-4 min (GPTQ is ~2 min + eval ~50s)

## Control
T=1.00 establishes the LOCAL post-quant baseline (probably around 1.13-1.15 due to local quant damage, not comparable to H100's 1.1056).

## Success criterion
Some T≠1.00 gives Δ < -0.001 vs T=1.00. That signal should be even better on H100 because H100 has smaller quant gap (0.0023 vs local's much larger).

## Files
- `train_gpt.py` — copy of expT's with EXPTQ_EARLY_EXIT added
- `run_sweep.sh` — runs the grid
