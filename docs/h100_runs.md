# 8xH100 runs

This file lists every run on a rented 8xH100 node. The competition limits training to 600 seconds and the artifact to 16,000,000 bytes (code plus compressed weights). The score is val_bpb on the FineWeb validation set. Lower is better.

Each full session on the rented node cost about $23. Resume-only sessions (quantization and evaluation from a saved checkpoint) cost less. The logs record $4.50 for Run 5 and $11 for Run 6.

## Terms

| Term | Meaning |
|------|---------|
| BPB | Bits per byte. The tokenizer-independent score. |
| Pre-quant | BPB of the EMA weights before quantization. Written "pre-quant" in the tables. |
| Roundtrip | BPB after quantization and dequantization, with a standard (non-sliding) evaluation. |
| Sliding | BPB with sliding-window evaluation, stride 64. |
| TTT | Test-time training. The model adapts to validation chunks that it has already scored. |
| EMA | Exponential moving average of the weights. |
| GPTQ | A post-training quantization method that uses second-order (Hessian) statistics. |
| SDClip | Clipping rule for GPTQ: clip = k x row standard deviation. From [PR #1394](https://github.com/openai/parameter-golf/pull/1394). |
| QAT | Quantization-aware training. |
| VE | Value embeddings. |
| LoRA | Low-rank adapter. Here it gives the repeated MLP a small per-pass difference. |
| Pruning | The share of +-1 quantized values that the size fitter set to zero to meet the byte budget. |
| SP4096, SP8192 | SentencePiece BPE tokenizers with 4096 and 8192 tokens. |

## Summary table

All values come from the run logs in `runs/` unless a note says otherwise. A dash means the value does not exist. "Not recorded" means the log does not contain it.

| Run | Date (2026) | Vocab | Key change | Steps | Pre-quant | Roundtrip | Sliding | TTT | Pruning | Artifact (bytes) | Outcome |
|-----|-------------|-------|-----------|-------|-----------|-----------|---------|-----|---------|------------------|---------|
| 0 | Apr 5 | 1024 | 13 layers, MLP 4x, first full pipeline | 5511 | 1.1204 | 1.1375 | 1.1461 | 1.1166 | 9.3% | 16,671,332 | Baseline. Size above the limit as logged. |
| 1a | Apr 5 | 4096 | Same model, SP4096 | 5348 | 1.1029 | 1.1740 | 1.1818 | 1.1138 | 55.3% | 16,669,166 | Heavy pruning ruined quantization. |
| 1b | Apr 5 | 4096 | Re-run of Run 1a | 5347 (1) | 1.1063 | 1.1376 | 1.1413 | 1.1110 | 19.4% (1) | 16,671,205 | Size above the limit as logged. |
| 2 | Apr 9 | 4096 | 11 layers, recurrence, parallel residuals, minified code | 5194 (1) | 1.1113 | 1.1634 | 1.1443 | not recorded | 40.6% | 15,997,662 | Pruning failure. |
| 3 | Apr 9 | 4096 | Run 2 plus int5 attention | 5209 | 1.1100 | 1.1316 | 1.1131 | not run | 0% | 15,232,121 | Best SP4096 result. |
| 4 | Apr 9 | 4096 | 12 layers | not recorded | 1.1110 | 1.1445 | crashed | not run | 12.5% | 15,999,081 | Worse than Run 3. |
| 5 | Apr 11 | 4096 | Competitor hyperparameters, SDClip, late QAT | 5577 (1) | 1.1113 | 1.1333 | 1.1148 | 1.1109 | 0% | 15,951,253 | Regression. |
| 6 | Apr 11 | 4096 | Run 3 recipe, SDClip, late QAT | 5182 | 1.1157 | 1.1386 | 1.1198 | 1.1151 | 0% | 15,902,041 | Worst of the SP4096 runs. |
| 7 | Apr 11 | 4096 | Run 6 without late QAT | 5182 | 1.1149 | 1.1347 | 1.1161 | not recorded | 0% | 15,956,661 | Still below Run 3. |
| 8 | Apr 11 | 4096 | Run 3 weights, new quantization code, AdamW TTT (no training) | - | 1.1098 | 1.1313 | 1.1126 | 1.1114 | 0% | 15,894,086 | Test of the resume path. |
| 9 | Apr 11 | 4096 | Run 3 recipe restored (repeat-pass scale, RoPE base 10000) | 5174 | 1.1088 | not run | not run | not run | - | - | Stopped during GPTQ. |
| 10a | Apr 11 | 8192 | First SP8192 run | 5051 | 1.1045 (2) | - | - | - | - | - | GPTQ crashed on an NCCL timeout. |
| 10b | Apr 11 | 8192 | Resume from the Run 10 checkpoint, int8 embedding | - | 1.1033 | not recorded | not recorded | not run | 41.1% | not recorded | Embedding too large. |
| 10c | Apr 11 | 8192 | Resume, int6 embedding | - | 1.1033 | 1.1226 | 1.1056 | 1.1009 | 2.1% | 15,991,240 | Best result. |

Notes on the table:

1. The step count and pruning share for Runs 1b and 2, and the step count for Run 5, come from the written experiment log. The shipped logs for those runs are partial. Run 1 also has an earlier attempt (`runs/run01_sp4096/run_v2_output.log`) that stopped at 4374 steps with 28.7% pruning and then crashed in the evaluation code.
2. Run 10a reports 1.1045 at the end of training. The resumed runs 10b and 10c re-evaluated the saved EMA checkpoint and report 1.1033. The 0.0012 difference is not explained in the logs. All later numbers use 1.1033, as the 10c log does.
3. The Run 10c quantization gap is 0.0023 when measured as sliding minus pre-quant. Measured as roundtrip minus pre-quant, it is 0.0193. See `docs/findings.md`, finding 4, for the reason this matters.

Leaderboard reference at the time: [PR #1541](https://github.com/openai/parameter-golf/pull/1541) held first place at 1.07785 BPB. Run 10c finished 0.0231 BPB above it with TTT.

## Run 0: SP1024 baseline (2026-04-05)

**What changed.** First full pipeline on 8xH100: 13 layers, MLP 4x, Muon optimizer, GPTQ int6, Brotli, sliding evaluation and TTT.

**What happened.** Training reached 1.1204 BPB in 5511 steps. Quantization needed 9.3% pruning. TTT improved sliding BPB from 1.1461 to 1.1166. The logged size of 16,671,332 bytes is above the limit.

**Log.** `runs/run00_sp1024_baseline/run_20260405_023212.log`

## Run 1: SP4096 (2026-04-05)

**What changed.** Vocabulary 4096 instead of 1024. The model had 39.6M parameters.

**What happened.** Pre-quant BPB was better (1.1029 against 1.1204), but the larger embedding made the model 17.68MB before pruning. The fitter pruned 55.3% of the +-1 values. Roundtrip BPB rose to 1.1740. A second attempt (Run 1b) pruned 19.4% and reached 1.1376 roundtrip and 1.1110 with TTT.

**Root cause.** The unpruned model did not fit the byte budget. Pruning damage grew with the pruned share (see the table in `docs/experiment_log.md`, section 8).

**Logs.** `runs/run01_sp4096/`

## Run 2: pruning failure (2026-04-09)

**What changed.** The local 11-layer design: depth recurrence on layers 3, 4 and 5, parallel residuals from layer 7, LoRA for the repeated MLP, LN scale off, softcap 20, warmdown 3500 iterations. The code was minified.

**What happened.** The first attempt crashed at start. The minifier renamed a Muon parameter (`weight_decay`) but not the call site. After a hot fix, training finished (5194 steps, 1.1113 BPB). GPTQ then crashed, and a resume run completed the quantization. The model compressed to 16.21MB against a 15.20MB budget. The fitter pruned 40.6% of the values. Roundtrip BPB was 1.1634.

**Root cause.** The model was too large for the budget. The same configuration compressed to 15.62MB locally. The pre-quant BPB on the node was 0.0127 worse than the local run (1.1113 against 1.0986), and the weights compressed less well.

**Logs.** `runs/run02/`

## Run 3: int5 attention (2026-04-09)

**What changed.** Attention weights use 5 bits. The ROPE base returned to 10000. The code stayed at about 60KB (an LZMA wrapper would save about 38KB).

**What happened.** Training ran 5209 steps at 115 ms per step after recurrence started. Pre-quant BPB was 1.1100. The model compressed to 14.47MB with no pruning. Sliding BPB was 1.1131.

**Root cause of the fix.** Int5 attention saves about 0.68MB at no measured BPB cost (see `docs/findings.md`, finding 1). The model then fits with 770KB of headroom against the limit.

**Log.** `runs/run03/train.log`

## Run 4: 12 layers (2026-04-09)

**What changed.** A 12th physical layer. Local 3-hour tests had shown a 0.0064 BPB gain for 12 layers.

**What happened.** Pre-quant BPB was 1.1110, which is 0.0010 worse than Run 3. The unpruned model was 15.62MB. The fitter pruned 12.5%. Roundtrip BPB was 1.1445 against 1.1316. The sliding evaluation crashed because the pod code called a function name that the minifier had changed.

**Root cause.** A 12-layer model needs about 126 ms per step against 115 ms. It trains fewer steps in 600 seconds. It is also larger, so it needs pruning. The loss from fewer steps and pruning outweighed the gain from depth.

**Log.** `runs/run04/resume_quant.log`

## Run 5: competitor hyperparameters (2026-04-11)

**What changed.** I copied the main hyperparameters of [PR #1326](https://github.com/openai/parameter-golf/pull/1326): recurrence on layers 4 and 5 only, softcap 30, EMA 0.997, recurrence start at step 3000, ROPE base 10000. I also added SDClip, late QAT and restored pre-damping in the Hessian step.

**What happened.** Pre-quant BPB was 1.1113, which is 0.0013 worse than Run 3. Sliding BPB was 1.1148 against 1.1131. TTT gave 1.1109. Pruning was 0%.

**Root cause.** The hyperparameters suited another architecture. Layers 3, 4 and 5 gave more virtual depth in my model than layers 4 and 5. A missing `fwl()` method in the Hessian model class crashed GPTQ once. The fix and a resume-only run cost about $4.50.

**Log.** `runs/run05/resume_quant.log`

## Run 6: Run 3 recipe with late QAT (2026-04-11)

**What changed.** The Run 3 training recipe, plus SDClip and late QAT (noise scale 0.10 in the last 10% of steps).

**What happened.** Pre-quant BPB was 1.1157, which is 0.0057 worse than Run 3. Sliding BPB was 1.1198. TTT gave 1.1151. Steps fell to 5182.

**Root cause.** The first reading blamed late QAT. Run 7 showed that this was only a small part (next section).

**Log.** `runs/run06/train.log`

## Run 7: late QAT removed (2026-04-11)

**What changed.** Run 6 with late QAT off.

**What happened.** Pre-quant BPB was 1.1149, which is only 0.0008 better than Run 6. Sliding BPB was 1.1161. The TTT stage did not finish in the log.

**Root cause.** See the forensics below.

**Log.** `runs/run07/train.log`

## Runs 3 to 7: regression forensics

Run 3 stayed the best SP4096 result although Runs 5, 6 and 7 added quantization improvements. A comparison of the logs found these differences between Run 3 and Runs 6 and 7.

| Setting | Run 3 | Runs 6 and 7 |
|---------|-------|--------------|
| `REPEAT_PASS_SCALE` | 1 | 0 |
| `ROPE_BASE` | 10000 (code default) | 22000 (set in the launch script) |
| Parameters | 34,524,255 | 34,524,252 |

`REPEAT_PASS_SCALE=1` adds three learned scalars that set the strength of the recurrence on layers 3, 4 and 5. This explains the three-parameter difference. The launch script of Runs 6 and 7 set ROPE base 22000 although it claimed to use the Run 3 recipe. A local comparison put the cost of that setting near 0.0006 BPB.

Validation BPB at step 4000 shows the gap before any quantization or late QAT could matter.

| Run | Code | ROPE base | Repeat-pass scale | Step-4000 BPB |
|-----|------|-----------|-------------------|---------------|
| 3 | old | 10000 | 1 | 1.1550 |
| 6 | new | 22000 | 0 | 1.1591 |
| 7 | new | 22000 | 0 | 1.1599 |
| 9 | new | 10000 | 1 | 1.1521 |

Run 9 used the new code with both settings restored. It reached 1.1521 at step 4000 and 1.1088 pre-quant. That is better than Run 3 (1.1100). This points to the two settings as the cause, and away from the code edits themselves.

A third suspect was the `torch.compile` graph. The code edits (a `fwl()` method, resume logic, removal of dead features) changed the compiled graph. Step 1 took 119 ms in Run 3 and 4461 to 4920 ms in Runs 6, 7 and 9. Run 9 has the slow first step and still recovered. The compile graph is therefore not the main cause. The byte-shuffle option also differed, but it acts only after training.

Two numbers in the written log disagree with the run logs. The written log lists Run 3 pre-quant as 1.0986. That value came from the local run of the same configuration. The node log shows 1.1100. With 1.1100, the pre-quant regression of Run 6 is 0.0057 and not 0.0171.

## Run 8: resume test (2026-04-11)

**What changed.** No training. The Run 3 checkpoint went through the new quantization code (SDClip) and AdamW TTT (learning rate 1e-4, 3 epochs).

**What happened.** The pre-quant BPB was 1.1098. Roundtrip BPB was 1.1313 and sliding BPB was 1.1126. TTT gave 1.1114, a gain of 0.0012.

**Root cause and use.** The run proved that the checkpoint-resume path works. It also gave a clean comparison of quantization code on fixed weights: SDClip improved sliding BPB by 0.0005 over Run 3. It was not a valid submission, because the weights came from Run 3.

**Log.** `runs/run08/train.log`

## Run 9: recipe restored (2026-04-11)

**What changed.** Fresh training with `REPEAT_PASS_SCALE=1` and ROPE base 10000 on SP4096, with SDClip and real-data Hessians.

**What happened.** Pre-quant BPB was 1.1088 after 5174 steps. The run stopped during GPTQ and did not produce a submission. The next run changed the vocabulary to SP8192.

**Log.** `runs/run09/train.log`

## Pivot from SP4096 to SP8192

Two top entries at the time used an 8192-token vocabulary. [PR #1394](https://github.com/openai/parameter-golf/pull/1394) set a base at 1.0856 BPB and [PR #1423](https://github.com/openai/parameter-golf/pull/1423) reached 1.0791. Earlier analysis had assumed they used SP4096. A check of the PRs showed SP8192. The larger vocabulary gives a better compression ratio per token (about 1.57x in my estimate), but the embedding table is twice as large.

Changes for Run 10:

- Vocabulary 8192.
- Value embeddings off, to save bytes.
- A separate SDClip constant for the embedding (k = 20.0).
- Warmdown as a fraction of training (0.72).
- TTT learning rate 0.005.
- Real-data Hessians, including a Hessian for the embedding from the final norm output.

At step 4000 the SP8192 run was 0.0094 BPB better than Run 9 (1.1427 against 1.1521).

## Run 10: SP8192 (2026-04-11)

### Run 10a: NCCL timeout

Training reached 1.1045 BPB at 5051 steps (118 ms per step). GPTQ then crashed. Only rank 0 ran GPTQ with 24,512 calibration sequences. Ranks 1 to 7 waited at a barrier until the NCCL watchdog (600 seconds) killed them. The EMA checkpoint was saved, so the run did not need retraining.

Fixes:

- NCCL timeout raised to 1800 seconds.
- The whole quantize, prune and save block runs on rank 0 only.
- Calibration capped at 256 sequences.
- A bug where `QUANT_BITS_EMBED=8` skipped GPTQ for the embedding (the test was `<= 6`) was corrected to `<= 8`.
- A mismatch in which skip-connection activations the Hessian model pushed was corrected.

### Run 10b: embedding too large

The resume run finished GPTQ. The unpruned model was 16.35MB against a 15.24MB budget, so the fitter pruned 41.1% of the values. The int8 embedding (8192 x 512, about 4MB) used too much of the budget. I stopped the run before evaluation.

### Run 10c: int6 embedding

Resuming with `QUANT_BITS_EMBED=6` saved about 1MB. Only 2.1% pruning was needed. Results:

| Metric | Run 10c | Run 3 |
|--------|---------|-------|
| Pre-quant | 1.1033 | 1.1100 |
| Roundtrip | 1.1226 | 1.1316 |
| Sliding | 1.1056 | 1.1131 |
| TTT (SGD, lr 0.005) | 1.1009 | - |
| Artifact (bytes) | 15,991,240 | 15,232,121 |
| Code (LZMA wrapper, bytes) | 23,524 | about 22,000 |

The artifact is 8,760 bytes under the limit. All results use one seed (1337).

**Logs.** `runs/run10/`, `runs/run10c/train.log`

## What the remaining gap contained

After Run 10c, the written log listed features that the top entries shared and that this code lacked. Examples are GPTQ column reordering by Hessian diagonal, squared leaky-ReLU, softcap 30, three recurrence loops and cross-lane parallel residual routing ([PR #1541](https://github.com/openai/parameter-golf/pull/1541)). I did not run those changes on the 8xH100 node. Run 10c is the last 8xH100 run in this repository.
