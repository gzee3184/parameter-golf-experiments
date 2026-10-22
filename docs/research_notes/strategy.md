# Strategy: adopt the converged recipe or build something new

This note records a strategy debate from April 2026 and how it ended. The debate was about how to close the gap to the leading entries of the competition.

## Starting point

On 2026-04-06 the best local model had a quantized BPB of 1.1226 without TTT. The leading entries were at 1.0896 and 1.0900 (see [PR #1326](https://github.com/openai/parameter-golf/pull/1326) and [PR #1331](https://github.com/openai/parameter-golf/pull/1331)). The gap was about 0.02 BPB. The leading entries shared several features that the local model did not have:

- Depth recurrence on a few middle layers, switched on late in training.
- Parallel residuals in the last layers.
- Value embeddings.
- Sigmoid-gated skip connections.
- A modified Muon optimizer (MuonEq-R).
- Smaller code, from 20 KB to 25 KB after minification.

The local code was 79 KB.

## Two positions

### Position A: adopt the proven recipe

- Add recurrence, parallel residuals, value embeddings, skip gates and MuonEq-R.
- Set weight decay to 0.090 and the QK gain to 5.0.
- Add TTT.
- The estimated total gain was 0.022 to 0.042 BPB. The largest single items were recurrence (0.010 to 0.015) and parallel residuals (0.005 to 0.010).

The objections to A were these:

- The added features raise the code size to 90 KB to 100 KB. Each extra kilobyte costs about 1000 int6 parameters.
- Copying reaches the place where others were, not a better place.
- The earlier local tests that rejected recurrence were not valid evidence, because they were too short for a feature that starts at step 3000.

### Position B: reduce code size first, then add targeted features

- Minify the code from 79 KB to about 25 KB. This gives about 54,000 more int6 parameters at no BPB cost.
- Test attention residuals (AttnRes), a learned routing over earlier layer outputs. It needs 1 KB to 2 KB of code. Its queries start at zero, so it starts as a plain residual connection.
- Improve `torch.compile` use, which costs zero bytes.
- Add value embeddings, late-start recurrence and parallel residuals only after tests at H100-equivalent step counts.

The measure for B was BPB gain per byte of code.

## Points both positions accepted

1. The earlier tests of recurrence and parallel residuals were flawed, so they gave no evidence against these features.
2. MuonEq-R is free. It needs three lines.
3. Weight decay and QK gain tuning cost no code.
4. The TTT gain shrinks as the model improves. It was 0.030 BPB on the local model and 0.0008 on a competitor model.
5. A model with 0% pruning is a real gain. Keep the budget formula.
6. Minification helps under both positions.

## How it resolved

The project did parts of both positions in this order.

1. **Minification and the LZMA wrapper were adopted.** The code went from 79 KB to about 66 KB raw and about 23 KB as counted in the artifact. See [../infrastructure.md](../infrastructure.md).
2. **AttnRes was tested and dropped.** In a one-hour local test it was 23% slower per step and reached a post-EMA BPB of 1.2019, against 1.1605 for the control. The test showed 1512 steps against 1862. The queries could not learn useful routing in about 500 steps after recurrence started.
3. **Recurrence and parallel residuals were adopted.** Recurrence on layers 3, 4 and 5 and parallel residuals from layer 7 are in the final model. A triple loop (three passes) was 6% slower and slightly worse in a one-hour test, so the final model uses two passes.
4. **Several of the free changes were adopted.** These were MuonEq-R, weight decay 0.090 and QK gain 5.0.
5. **The "match and layer" attempt failed.** Run #5 copied the full hyperparameter set of PR #1326 and added the local quantization changes. Pre-quantization BPB got worse by 0.0127. The project concluded that the hyperparameters of another entry depend on its architecture. The next runs returned to the local hyperparameters and kept only changes that were independent of the architecture, such as the SDClip quantization clipping and score-first TTT.
6. **Late quantization-aware training was dropped.** It raised the pre-quantization BPB by 0.0171 in Run #6.
7. **The vocabulary moved from 4096 to 8192.** This followed the leading entry and gave the biggest single gain: Run #10c reached 1.1056 sliding BPB and 1.1009 with TTT. Run #3, the best SP4096 run, had 1.1131 sliding BPB.

Details of each run are in [../h100_runs.md](../h100_runs.md).

## The decision left open

After Run #10c the gap to the leader ([PR #1541](https://github.com/openai/parameter-golf/pull/1541), 1.07785 BPB) was about 0.02 BPB. The top four entries then shared a common recipe. It had these parts:

- Squared leaky ReLU with slope 0.5.
- Muon momentum warmup from 0.92 to 0.99 over 1500 steps.
- GPTQ with columns ordered by descending Hessian diagonal.
- Partial RoPE on 16 of 64 head dimensions.
- A per-dimension mix of the current state with the first embedding.
- Layer-scaled norm outputs and logit softcap 30.
- Three passes through layers 3 to 5, starting at 35% of training.
- Parallel residuals with cross-lane routing (see [PR #1541](https://github.com/openai/parameter-golf/pull/1541)).
- Int6 matrices with int8 embeddings, Brotli with byte shuffle, and TTT with a cosine learning-rate decay.

The local code had some of these and not others. Two options remained for a next run:

- **Option A.** Start from the decompressed code of PR #1541. This gives a clean base of a proven design. It loses the local code history and the local H100 fixes.
- **Option B.** Change the local code toward the recipe. This keeps the history and the fixes. It needs about 15 edits in minified code.

The record ends before this choice was made. In April 2026 the project instead ran evaluation-only experiments on the saved Run #10 checkpoint: temperature scaling, deep supervision at evaluation, and a z-loss regularizer test. All gave no gain. See [../experiments/README.md](../experiments/README.md).

## Rules that the project kept

- Validate on a local run before spending an H100 session.
- Do not trust short-run results for features that start late in training.
- Do not assume that a TTT gain carries over to a better model.
- Measure the code size cost of each feature.
