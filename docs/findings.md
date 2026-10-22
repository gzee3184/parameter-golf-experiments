# Findings

This file lists the findings of the project in three groups: non-obvious findings, negative results, and the eval-time experiment set. Each finding has a claim, the evidence with file paths, and a confidence note.

Rules for the confidence notes: "H100" means a 600-second run on 8xH100. "Local" means a run on a 2-GPU local workstation (RTX PRO 6000 Blackwell, sm_120) with one GPU and a different attention kernel. Unless a note says otherwise, every result comes from one seed.

Terms: BPB is bits per byte. TTT is test-time training. GPTQ is a post-training quantization method that uses Hessian statistics. EMA is an exponential moving average of weights. QAT is quantization-aware training. XSA is exclusive self attention. SDClip is the clipping rule clip = k x row standard deviation ([PR #1394](https://github.com/openai/parameter-golf/pull/1394)). RTN is round-to-nearest quantization. LoRA is a low-rank adapter.

## A. Non-obvious findings

### 1. Int5 attention costs about zero BPB and saves 0.7MB to 0.9MB

**Claim.** Storing attention matrices with 5 bits instead of 6 did not change BPB beyond noise. The compressed model was 0.7MB to 0.9MB smaller.

**Evidence.** Local 3-hour A/B, same 11-layer SP4096 model (`sweeps/local/v10_int5attn/`):

| Arm | Steps | Pre-quant | Roundtrip | Unpruned model | Pruning |
|-----|-------|-----------|-----------|----------------|---------|
| Int5 attention | 5552 | 1.0981 | 1.1210 | 14.68MB | 0% |
| Int6 attention | 5349 | 1.0991 | 1.1279 | 15.62MB | 20.0% |

Earlier same-model comparisons gave BPB differences of -0.000005 and +0.000034 for 0.69MB and 0.68MB of savings (`docs/experiment_log.md`, section 26). H100 Run 3 used int5 attention and fit with 0% pruning (`runs/run03/train.log`).

**Confidence.** Medium. The local A/B also changed the step count. No H100 A/B exists. A 972-step test showed large damage for int5 (+0.625 against +0.337), so the result holds only for converged models.

### 2. Local compression and pruning results did not predict H100 results

**Claim.** The compressed size of the same architecture differed between local and H100 runs. I stopped drawing size conclusions from local runs.

**Evidence.** A local checkpoint compressed to 16.25MB with 43% pruning. RTN and two GPTQ variants gave the same 16.25MB (`experiments/quantization/ablation_results.log`), so GPTQ did not drive the size. H100 Run 3 compressed to 14.47MB. The size search used Brotli quality 6. At quality 11 the local model was 15.38MB.

**Confidence.** Low to medium for the cause, high for the practice. The local int5 run in finding 1 compressed to 14.68MB, close to Run 3. The 16.25MB checkpoint came from other settings, such as an int8 embedding, that I could not match. Only pre-quant BPB was comparable between local and H100 runs.

### 3. Gradient analysis of the recurrent and parallel model

**Claim.** In a trained model with recurrence on layers 3, 4, 5 and parallel residuals from layer 7, gradients were uneven across layers. Layer 5 attention received no gradient on the repeat pass.

**Evidence** (`analysis/gradient_analysis.py`, `docs/experiment_log.md`, section 20):

| Layer group | Mean MLP-up gradient norm | Relative |
|-------------|---------------------------|----------|
| Recurrent (3, 4, 5) | 0.415 | 2.9x |
| Serial (1, 2, 6) | 0.301 | 2.1x |
| Parallel (7, 8, 9) | 0.142 | 1.0x |
| Output (10) | 0.245 | 1.7x |

Layer 5 attention output and value gradients were 0.000 on the repeat pass. Layer 7 had the smallest scalar gradients. The lane-merge scalar had the largest scalar gradient (0.0875). The layer-0 MLP scale gradient (0.247) was the largest block scalar.

**Confidence.** Medium for the measurement, low for any benefit. It used one checkpoint and 8 batches. The fixes built from it did not give a valid gain (negative results 12 to 15).

### 4. The "quantization gap" depends on how it is measured

**Claim.** The first reported SDClip effect (gap 0.0131 down to 0.0012, a 91% cut) came from comparing a local baseline with H100 numbers. In the run logs, SDClip improved results by about 0.0005 BPB on fixed weights.

**Evidence.**

| Run | Sliding minus pre-quant | Roundtrip minus pre-quant |
|-----|-------------------------|---------------------------|
| 3 (no SDClip) | 0.0031 | 0.0216 |
| 5 | 0.0035 | 0.0220 |
| 6 | 0.0041 | 0.0229 |
| 7 | 0.0012 | 0.0198 |
| 8 (Run 3 weights, SDClip) | 0.0028 | 0.0215 |
| 10c (SP8192) | 0.0023 | 0.0193 |

Run 8 quantized the Run 3 weights with the new code. Sliding BPB went from 1.1131 to 1.1126 and roundtrip from 1.1316 to 1.1313 (`runs/run03/train.log`, `runs/run08/train.log`). "Sliding minus pre-quant" is small because the sliding window gains about 0.019 BPB.

**Confidence.** High for the arithmetic. The SDClip effect on fixed weights (0.0003 to 0.0005) is one measurement.

### 5. Real-data Hessians for GPTQ, with pre-damping

**Claim.** Hessians from real training data need the pre-damping term (0.01 x mean diagonal). Without it, quantization failed.

**Evidence.** Real data without pre-damping gave 43.3% pruning and roundtrip 1.1756 against 1.1117 (`experiments/quantization/gptq_fix_long_test.log`). The SP8192 runs (real-data Hessians plus an embedding Hessian) reached roundtrip minus pre-quant 0.0193 against 0.0216 for Run 3 (`runs/run10c/train.log`). A bug skipped GPTQ for an int8 embedding (`<= 6` fixed to `<= 8`).

**Confidence.** Low for the benefit. No run isolates real-data Hessians on the same weights. The Run 10c difference mixes vocabulary, embedding bits and Hessian changes.

### 6. Embedding precision decided the SP8192 budget

**Claim.** At vocabulary 8192 an int8 embedding does not fit. Int6 removed almost all pruning.

**Evidence.** Run 10b: unpruned 16.35MB, pruning 41.1%. Run 10c with int6 embedding: 15.36MB unpruned, pruning 2.1%, artifact 15,991,240 bytes (`runs/run10/run10b_full.log`, `runs/run10c/train.log`).

**Confidence.** High for the size effect. The BPB cost of the int6 embedding was not measured separately.

### 7. Grid-pushing during training conflicts with GPTQ

**Claim.** Pulling weights toward the RTN grid during training did not help a model that GPTQ quantizes later. A strong pull destroyed compressibility. Late QAT had a small effect.

**Evidence.** CAGE-lite (a small pull toward the RTN grid in the last 15% of training, after [arXiv 2510.18784](https://arxiv.org/abs/2510.18784)) kept pre-quant BPB (1.1001 against 1.1000). It raised pruning to 92.6% and roundtrip from 1.1131 to 1.2428 in a 3-hour local run (`experiments/quantization/cage_lite_long_test.log`). Late QAT in H100 Runs 6 and 7: pre-quant 1.1157 with it, 1.1149 without. Sliding 1.1198 with it, 1.1161 without. A local late-QAT test gave 1.1003 against 1.1006 (noise, `docs/experiment_log.md`, section 33).

**Confidence.** Medium for CAGE-lite, low for late QAT. My first notes blamed late QAT for the Run 6 regression. Run 7 showed it explains at most 0.0008 pre-quant (finding 9).

### 8. Copying hyperparameters from another architecture hurt

**Claim.** Run 5 copied main hyperparameters from [PR #1326](https://github.com/openai/parameter-golf/pull/1326). Pre-quant BPB got worse.

**Evidence.** Recurrence on layers 4, 5 instead of 3, 4, 5; softcap 30 instead of 20; EMA 0.997 instead of 0.9965; recurrence start 3000 instead of 2000. Pre-quant went 1.1100 to 1.1113, sliding 1.1131 to 1.1148 (`runs/run05/resume_quant.log`).

**Confidence.** Low to medium. Seven settings changed at once and none was tested alone. The effect (0.0013) is small for one seed. A local run of the PR code reached 1.0959 pre-quant against 1.0991 for mine (`docs/experiment_log.md`, section 18).

### 9. Silent launch-script differences caused the Run 3 to 7 regression

**Claim.** `REPEAT_PASS_SCALE` (1 against 0) and ROPE base (10000 against 22000) explain most of the regression, not the quantization changes.

**Evidence.** Validation BPB at step 4000: 1.1550 (Run 3), 1.1591 (Run 6), 1.1599 (Run 7), 1.1521 (Run 9, both settings restored). Details are in `docs/h100_runs.md`.

**Confidence.** Medium. Run 9 restored both settings together and changed other items, so the two effects are not separated.

### 10. Warmdown length changes compressibility

**Claim.** A longer warmdown made weights compress better. A warmdown set in steps can exceed the wall-clock budget.

**Evidence.** Warmdown 1500 instead of 3500 made the compressed model about 0.75MB larger (16.88MB against 15.88MB). A name pattern that kept LoRA tensors in float32 added about 0.25MB. A 3500-step warmdown with a 2-hour limit put the schedule in decay from step 0, which invalidated that run (`docs/experiment_log.md`, section 22).

**Confidence.** Medium. The two size causes were not fully separated.

### 11. A 12th layer won locally and lost on H100

**Claim.** More depth gained in a long local run and lost in a 600-second run.

**Evidence.** Local: pre-quant 1.0922 for 12 layers against 1.0986 for 11 (`sweeps/local/v11_scale/`). H100 Run 4: pre-quant 1.1110 against 1.1100, pruning 12.5%, roundtrip 1.1445 against 1.1316 (`runs/run04/resume_quant.log`).

**Confidence.** Medium. One seed. The step-time mechanism (about 126 ms against 115 ms) is an estimate; Run 4 steps were not recorded.

### 12. AdamW for TTT: large local gain, small H100 gain

**Claim.** AdamW (learning rate 1e-4) beat SGD by 0.0107 BPB in a local sweep. The H100 gain was small.

**Evidence.** Local sweep on one quantized Run 3 model with 33.3% local pruning (`sweeps/local/ttt_sweep/`): AdamW 1.1173 to 1.1174 (3 epochs), 1.1239 (5 epochs); SGD 1.1280 (lr 0.002), 1.1312 (lr 0.001). Frozen blocks did not matter. On H100, AdamW on Run 3 weights gained 0.0012 (`runs/run08/train.log`). SGD gained 0.0039 (Run 5), 0.0047 (Run 6), 0.0047 (Run 10c, lr 0.005).

**Confidence.** Low for any H100 ranking. The checkpoints differ. The local gap includes repair of local pruning damage.

### 13. TTT gain shrinks as the base model improves

**Evidence.** A 924-step local model gained 0.215 BPB. The 5347-step SP4096 model gained 0.030. Run 10c gained 0.0047.

**Confidence.** Medium. These are three points from different setups.

### 14. Softcap 20 gave the largest local tuning gain

**Evidence** (`sweeps/local/v8_sweep_r1/` to `v8_sweep_r4/`): v7 B 1.1015 / 1.1144 (pre-quant / roundtrip); softcap 20 with ROPE 22000 1.0991 / 1.1119; with ROPE 10000 1.0997 / 1.1127; plus LN scale off 1.0986 / 1.1117; parallel start at layer 9 1.1014 / 1.1167.

**Confidence.** Low to medium. Differences are 0.0005 to 0.0024 with one seed. Run 5 later showed that softcap 30 did not transfer cleanly.

## B. Negative results

| # | Idea | Result | Evidence |
|---|------|--------|----------|
| 1 | Attention residuals (AttnRes) | 23% slower. +0.041 pre-quant, +0.086 roundtrip (1-hour local) | `sweeps/local/v6_sweep/` |
| 2 | Three recurrence loops | 6% fewer steps, 1.1635 against 1.1605 | `sweeps/local/v6_sweep/` |
| 3 | Fully untied repeat MLP | Pre-quant 1.0957 (-0.0029), 56.6% pruning, roundtrip 1.1953 | `sweeps/local/v11_scale/` |
| 4 | Untied repeat MLP in v6 | Pre-quant 1.0996, 77.5% pruning, roundtrip 1.1540 | `sweeps/local/v6_long/` |
| 5 | 12-layer model | See finding 11 | `runs/run04/` |
| 6 | MLP weight sharing (5 pairs) | 0% pruning, pre-quant 0.017 worse | `sweeps/local/v3_mlp_share/` |
| 7 | CAGE-lite | Short: +0.0139 post-EMA. Long: 92.6% pruning | `experiments/quantization/cage_lite_*.log` |
| 8 | Late QAT | Within 0.0008 pre-quant; sliding 0.0037 worse | `runs/run06/`, `runs/run07/` |
| 9 | Remove GPTQ pre-damping | 43.3% pruning, roundtrip +0.0639 | `experiments/quantization/gptq_fix_long_test.log` |
| 10 | Parallel start at layer 9 | Roundtrip +0.0023 | `sweeps/local/v8_sweep_r2/` |
| 11 | Data-ramp warmdown coupling | +0.0001 (flat) | `sweeps/local/v8_sweep_r3/` |
| 12 | Disable layer-0 MLP | +0.0006 pre-quant, +0.0013 roundtrip | `sweeps/local/v8.5_sweep/` |
| 13 | Skip layer-5 repeat attention | +0.0029 pre-quant, +0.0035 roundtrip | `sweeps/local/v8.5_sweep/` |
| 14 | Per-layer lane merge with LoRA rank 32 | 14.8% pruning, over budget | `sweeps/local/v8_sweep_r5/` |
| 15 | Attention-query LoRA, parallel gradient scale | Invalid (warmdown bug), not re-run | `sweeps/local/v8.5_sweep_INVALID_warmdown/` |
| 16 | Z-loss 1e-4 | 500-step A/B: +0.0068 val BPB, +0.0506 post-EMA | `experiments/eval_time/expZ_zloss/results.md` |
| 17 | Temperature scaling, pre-quant | Minimum at T = 1.00 | `experiments/eval_time/expT_temp_scaling/results.md` |
| 18 | Temperature scaling, post-quant | Minimum at T = 1.00 (local, 39% pruning) | `experiments/eval_time/expTq_postquant_temp/results.md` |
| 19 | Multi-layer logit averaging | Every arm worse (+0.0218 to +0.2403) | `experiments/eval_time/expDS_deep_supervision/results.md` |
| 20 | Inductor configs, training-loop options | None beat the 199.7 ms baseline; CUDA graphs failed on the RoPE cache | `perf/RESULTS.md` |
| 21 | Stack of mish-squared, XSA on 6 layers, multi-token prediction | Never beat baseline in 1050 local steps | `sweeps/early_sp1024/` |
| 22 | Mish-squared against leaky-ReLU at MLP 4x, SP4096 | Leaky-ReLU won (3.423 against 3.428 at 35 steps) | `docs/research_notes/activation_functions.md` |
| 23 | Weight decay, query gain, batch sweeps at 35 to 70 steps | Flat or dominated by step count; short runs do not inform 5000-step behavior | `docs/research_notes/activation_functions.md` |

Rows 21 to 23 use short local runs and may not hold at convergence. A GPU-contention artifact misled an early activation comparison. Two jobs on the two local GPUs added about 93 ms per step to one activation, but single-GPU runs did not show this.

## C. Eval-time experiment set (April 2026)

These experiments used the saved Run 10 checkpoint (1.1033 pre-quant). The frozen reference code is in `experiments/eval_time/baseline_r10/`.

| Experiment | Idea | Result | Decision |
|------------|------|--------|----------|
| expT | Divide logits by T | V curve, minimum at T = 1.00 (T = 0.95: +0.0027, T = 1.05: +0.0015, T = 0.85: +0.0232) | Not deployed |
| expTq | Same, after quantization | Minimum at T = 1.00 | Not deployed |
| expDS | Average logits from last N decoder states | All arms worse; N = 2 uniform 1.1493, N = 4 uniform 1.3436 | Not deployed |
| expZ | Z-loss during training | Worse at 500 steps (post-EMA 1.4780 against 1.4274) | Not escalated |
| expA | Compile the Hessian model | Downgraded before any run | No result |
| expB | Three LoRA adapters, logit averaging | Design only | Result not recorded |
| expE | Resample hard sequences near end of training | Design only | Result not recorded |

Observations:

- The model is already calibrated for cross-entropy. Thus the TTT gain comes from weight adaptation, not from calibration.
- Intermediate hidden states are not noisier copies of the final state. Cross-entropy at intermediate layers during training is not tested.
- A standalone loader evaluated the same checkpoint at 1.4956 BPB (`experiments/eval_time/verify_r10_checkpoint.log`). The standard resume path reproduced 1.1033 (`verify_r10_canonical.log`). The standalone loader has a mismatch that I did not find. Only the standard path is valid.

**Confidence.** High that the three eval-only ideas give no gain on this checkpoint (one checkpoint, one seed).
