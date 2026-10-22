# Experiment log

This is a condensed, chronological log of the experiments. It covers local runs on a 2-GPU local workstation (RTX PRO 6000 Blackwell, sm_120) and runs on rented 8xH100 nodes. Raw numbers and tables are kept. Status notes were removed. An entry whose end state was not written down says "result not recorded".

Dates run from late March 2026 to 2026-04-19. Local logs are under `sweeps/local/`, `experiments/` and `runs/`. Per-run H100 detail is in `docs/h100_runs.md`. Curated conclusions are in `docs/findings.md`.

Terms: BPB is bits per byte (lower is better). "Post-EMA" is BPB of the exponential-moving-average weights. "Quant" or "roundtrip" is BPB after quantization. TTT is test-time training. GPTQ is a Hessian-based post-training quantization method. QAT is quantization-aware training. VE is value embeddings. LoRA is a low-rank adapter. XSA is exclusive self attention. SDClip is a GPTQ clipping rule from [PR #1394](https://github.com/openai/parameter-golf/pull/1394). "Pruning" is the share of +-1 quantized values set to zero to fit the byte budget. Local runs use one GPU, so a "3-hour" local run gives about 5500 steps, close to the step count of a 600-second H100 run.

---

## 1. Early 8xH100 baselines (2026-04-05)

Logs: `runs/run00_sp1024_baseline/`, `runs/run01_sp4096/`.

Run 0 used 13 layers, MLP 4x, vocabulary 1024. Run 1 changed the vocabulary to 4096. The config of the best Run 1 attempt: Muon+, GPTQ int6, Brotli, TTT, weight decay 0.12, matrix learning rate 0.02, QK gain 4.0, EMA 0.997, gradient clip 0.3.

| Metric | Run 0 (SP1024) | Run 1a (SP4096) | Run 1b (SP4096) |
|--------|----------------|-----------------|-----------------|
| Steps | 5511 | 5348 | 5347 |
| ms per step | about 109 | about 112 | 112 |
| Post-EMA BPB | 1.1204 | 1.1029 | 1.1063 |
| Roundtrip BPB | 1.1375 | 1.1740 | 1.1376 |
| Sliding BPB | 1.1461 | 1.1818 | 1.1413 |
| TTT BPB | 1.1166 | 1.1138 | 1.1110 |
| Pruning | 9.3% | 55.3% | 19.4% |
| Artifact (bytes) | 16,671,332 | 16,669,166 | 16,671,205 |

Pruning of 19.4% caused +0.031 BPB of quantization damage in Run 1b. This was the main bottleneck. The code was about 104KB, which left little room for weights.

---

## 2. v3 MLP sharing, 5-hour local test (2026-04-05)

Log: `sweeps/local/v3_mlp_share/`. Config: 13 layers, 5 MLP-sharing pairs (layer 0 with 12, 1 with 11, and so on), 29.1M parameters, 8 unique MLPs of 13.

| Metric | Value |
|--------|-------|
| Steps | 9208 |
| ms per step | 1955 |
| Post-EMA BPB | 1.1232 |
| Quant BPB | 1.1383 (+0.015) |
| Pruning | 0% |
| Artifact | 14,622,840 bytes |

Sharing removed the pruning problem but lost 0.017 BPB before quantization against the H100 baseline. It was a net loss against simply reducing code size.

## 3. v3 sweep, 30-minute screening (2026-04-05)

Log: `sweeps/local/v3_sweep/`. Duration 1800 s each.

| Test | Config | Steps | ms per step | Post-EMA | Quant | Artifact |
|------|--------|-------|-------------|----------|-------|----------|
| A | 5 pairs, recurrence always on | 862 | 2089 | 1.4295 | 1.6229 | 8.9MB |
| B | 3 pairs (10 unique MLPs) | 923 | 1952 | 1.3900 | 1.5797 | 9.4MB |
| C | 5 pairs, TTT | about 924 | about 1952 | about 1.3940 | 1.5817, TTT 1.3667 | 9.1MB |

Test A turned recurrence on from step 0. The entries I compared against start recurrence late (step 3000 in their code). A 900-step test never reaches that point, so Test A says nothing about late-start recurrence. TTT gained 0.215 BPB on this under-trained model.

## 4. v4 max-parameters, 30-minute test (2026-04-05)

Log: `sweeps/local/v4_maxparams/`. Config: 13 layers, MLP 4x, leaky-ReLU 0.5, 39.6M parameters, stripped code (76.9KB), no TTT, recurrence or sharing.

| Metric | Value |
|--------|-------|
| Steps | 892 |
| ms per step | 2019 |
| Pre-EMA / post-EMA BPB | 1.2772 / 1.4062 |
| Quant BPB | 1.6305 |
| Pruning | 0% (model 9.07MB, budget 15.93MB) |
| Artifact | 9,590,000 bytes |

The budget formula (16,000,000 minus code size) gave 0% pruning.

## 5. v4 long test, 3 hours (2026-04-06)

Log: `sweeps/local/v4_long/`. Same config as section 4 with a 10,800 s limit.

| Metric | Value |
|--------|-------|
| Steps | 5522 |
| ms per step | 1956 |
| Pre-EMA / post-EMA BPB | 1.1080 / 1.1070 |
| Quant BPB | 1.1226 (+0.016) |
| Pruning | 0% (model 15.79MB, budget 15.93MB) |
| Artifact | 16,631,638 bytes |

Pre-quant BPB matched the H100 baseline (1.1070 against 1.1063). Quant BPB was better (1.1226 against 1.1376) because of the missing pruning. The architectural gap to the then-leader (1.0900) was 0.033 BPB at quant. TTT was not run.

## 6. AttnRes short sweep and step overhead (2026-04-06)

Log: `sweeps/local/attnres_sweep/`. AttnRes is a learned depth-wise routing of earlier layer outputs, with zero-initialized queries. Each config ran 900 s.

| Config | Layers | Dim | AttnRes | Steps | ms per step | Post-EMA | Artifact |
|--------|--------|-----|---------|-------|-------------|----------|----------|
| A | 13 | 512 | yes | 304 | 2963 | 3.8736 | 6.6MB |
| D | 15 | 512 | yes | 246 | 3664 | 3.4754 | 7.2MB |
| B | 15 | 480 | yes | about 263 | 3418 | result not recorded | - |
| Control | 13 | 512 | no | about 444 | 2027 | result not recorded | - |

At 250 to 300 steps the models are too under-trained to rank. Step overhead was the usable result: 47% for 13 layers and 81% for 15 layers (estimated steps in 600 s on H100: 5350 without AttnRes, 3660 with it for 13 layers).

## 7. TTT gain against model quality (analysis)

| Model | Base BPB | After TTT | Gain |
|-------|----------|-----------|------|
| v3 sweep C (924 steps) | 1.5817 | 1.3667 | -0.215 |
| H100 Run 1b (5347 steps) | 1.1413 (sliding) | 1.1110 | -0.030 |
| Leader PR #1326 (about 5300 steps) | 1.0904 (sliding) | 1.0896 | -0.0008 |

TTT gain falls as the model improves.

## 8. Pruning damage curve

| Pruning | Quant damage (BPB) | Source |
|---------|--------------------|--------|
| 0% | +0.008 | 11-layer model, 2026-03-23 |
| 0% | +0.015 | v3 MLP sharing |
| 0% | +0.016 | v4 long |
| 0% | +0.031 | v6 1-hour sweep control (under-trained) |
| 19.4% | +0.031 | H100 Run 1b |
| 55.3% | +0.071 | H100 Run 1a (log: 1.1740 against 1.1029 post-EMA) |
| 77.5% | +0.054 | v6 long (untied MLP overflow) |

Damage is not monotone in the pruning share. The damage also depends on how well trained the model is.

## 9. v5 long test, 3 hours (2026-04-06)

Log: `sweeps/local/v5_long/`. Config: 11 layers, recurrence on layers 4 and 5, VE on layers 9 and 10, MuonEq-R, weight decay 0.09, QK gain 5.0, skip gates, no parallel residuals.

| Metric | Value |
|--------|-------|
| Steps | about 5600 |
| ms per step | about 1900 after recurrence starts |
| Post-EMA / quant BPB | 1.1146 / 1.1146 (damage 0.000) |
| Pruning | 0% |

First implementation of the competitor features. It closed 24% of the gap to the leader.

## 10. v6 long test, 3 hours (2026-04-06)

Log: `sweeps/local/v6_long/`. Config: 11 layers, recurrence on layers 3, 4, 5 (two loops), VE on layers 9 and 10, parallel residuals from layer 7, MuonEq-R, untied MLP down-projection for repeats, layer-0 attention disabled, skip gates, multi-shard loader.

| Metric | Value |
|--------|-------|
| Steps | 5720 |
| ms per step | 1604 before recurrence, 1888 after |
| Pre-EMA / post-EMA BPB | 1.1004 / 1.0996 |
| Quant BPB | 1.1540 (+0.054) |
| Pruning | 77.5% |
| Unpruned model | 17.76MB (budget 15.91MB) |
| Code | 93.9KB |
| Artifact | 16.78MB |

Pre-quant 1.0996 was the best at the time. The untied repeat MLP added 3.15M parameters and overflowed the budget by 1.85MB. Without it the model fit at about 14.6MB. Disabling layer-0 attention saved about 4% of compute.

## 11. v6 one-hour sweep (2026-04-06)

Log: `sweeps/local/v6_sweep/`. Duration 3600 s each, recurrence start at step 1000.

| Test | Steps | ms per step after recurrence | Post-EMA | Quant | Pruning |
|------|-------|------------------------------|----------|-------|---------|
| Control (2 loops) | 1862 | 1934 | 1.1605 | 1.1918 | 0% |
| Three loops | 1753 | 2054 | 1.1635 | 1.1955 | 0% |
| AttnRes | 1512 | 2377 | 1.2019 | 1.2778 | 0% |

Three loops cost 6% of the steps and were slightly worse. AttnRes cost 23% per step, and BPB was +0.041 (post-EMA) and +0.086 (quant) worse. EMA made AttnRes worse (pre-EMA 1.1727, post-EMA 1.2019). The zero-initialized queries did not learn useful routing in about 500 post-recurrence steps.

## 12. Findings from competitor PRs (2026-04-07)

- [PR #1420](https://github.com/openai/parameter-golf/pull/1420) (triple loop, fused kernels, 1.08014 BPB) found a PyTorch 2.11 inductor behavior: `mix_order_reduction` over-fuses the RMSNorm backward pass and adds 8.8% overhead. The fix is `torch._inductor.config.mix_order_reduction = False`. My build did not have that option. The PR also found that L2 weight decay is scale-invariant under max-abs quantization.
- [PR #1423](https://github.com/openai/parameter-golf/pull/1423) (1.0791 BPB) trains on the validation data with AdamW for 6 epochs before GPTQ ("pre-quant TTT", from [PR #1364](https://github.com/openai/parameter-golf/pull/1364)). This differs from score-first TTT, which I used. I did not implement the pre-quant variant.

## 13. Speed comparison across versions

| Version | ms per step before / after recurrence | Steps in 3 hours | Relative |
|---------|---------------------------------------|------------------|----------|
| v4 (13 layers, no recurrence) | 1956 / 1956 | 5522 | 1.00 |
| v5 (11 layers, recurrence 4,5) | 1665 / about 1900 | about 5600 | 1.01 |
| v6 (recurrence 3,4,5, parallel, layer-0 attention off) | 1604 / 1888 | 5720 | 1.04 |
| v6 three loops | 1652 / 2054 | about 5250 (estimate) | 0.95 |
| v6 AttnRes | 2052 / 2377 | about 4536 (estimate) | 0.82 |

Parallel residuals and disabled layer-0 attention paid for the cost of recurrence.

## 14. v7 sweep, LoRA and bundle (2026-04-07)

Log: `sweeps/local/v7_sweep/`. Duration 10,800 s each. Code: 98,583 bytes. Both tests replace the untied repeat MLP with a rank-16 LoRA.

| Metric | Test A (LoRA baseline) | Test B (LoRA and bundle) |
|--------|------------------------|--------------------------|
| Bundle | - | EMA 0.9965, warmdown 3500, SWA off, recurrence start 2000, int8 embedding, per-pass scale |
| Steps | 5836 | 5407 |
| ms per step | 1851 | 1998 |
| Pre-EMA / post-EMA | 1.1045 / 1.1037 | 1.1027 / 1.1015 |
| Quant | 1.1154 | 1.1144 |
| Quant damage | +0.0117 | +0.0129 |
| Pruning | 0% | 0% |
| Artifact (bytes) | 16,763,273 | 16,749,271 |

LoRA cost 0.004 BPB pre-quant against the full untied bank (1.1037 against 1.0996) but improved quant by 0.039 BPB (no pruning). The bundle gained 0.0022 post-EMA and 0.0010 quant, despite 429 fewer steps. Estimated per-item gains (EMA 0.9965 about -0.001, warmdown 3500 about -0.0005, per-pass scale about -0.0005) come from other PRs and are not isolated here.

## 15. v8 round 1: RoPE fix and LN scale (2026-04-07)

Log: `sweeps/local/v8_sweep_r1/`. Duration 10,800 s each.

| Test | Steps | Post-EMA | Quant | Delta vs v7 B (pre / quant) |
|------|-------|----------|-------|-----------------------------|
| RoPE base 10,000 (fix) | 5579 | 1.1000 | 1.1131 | -0.0015 / -0.0013 |
| LN scale off | 5372 | 1.0999 | 1.1130 | -0.0016 / -0.0014 |

An earlier code path used an accidental RoPE base of 22,000 (from NTK scaling). The corrected base converged better. Turning LN scale off was neutral on its own.

## 16. v8 round 2: skip topology and parallel start (2026-04-07)

Log: `sweeps/local/v8_sweep_r2/`. Common base: v7 B with ROPE base 22,000.

| Test | Steps | Post-EMA | Quant | Delta vs v7 B (pre / quant) |
|------|-------|----------|-------|-----------------------------|
| Physical-layer skip topology | 5720 | 1.1009 | 1.1143 | -0.0006 / -0.0001 |
| Parallel start at layer 9 | 5390 | 1.1014 | 1.1167 | -0.0001 / +0.0023 |

Neither change helped. More sequential layers produced weights that quantize worse.

## 17. v8 round 3: softcap and data ramp (2026-04-07)

Log: `sweeps/local/v8_sweep_r3/`. Common base: v7 B with ROPE base 22,000.

| Test | Steps | Post-EMA | Quant | Delta vs v7 B (pre / quant) |
|------|-------|----------|-------|-----------------------------|
| Logit softcap 20 (from 30) | 5583 | 1.0991 | 1.1119 | -0.0024 / -0.0025 |
| Data ramp with warmdown | 5401 | 1.1016 | 1.1147 | +0.0001 / +0.0003 |

Softcap 20 was the largest local gain in the tuning rounds. The data ramp had no effect.

## 18. v8 round 4: my combination against PR #1326 (2026-04-07)

Log: `sweeps/local/v8_sweep_r4/`. Duration 10,800 s each.

| Test | Steps | Post-EMA | Quant | Pruning |
|------|-------|----------|-------|---------|
| Mine (ROPE 10,000, softcap 20) | 5586 | 1.0997 | 1.1127 | 0% |
| [PR #1326](https://github.com/openai/parameter-golf/pull/1326) code, default settings | 5572 | 1.0959 | 1.1144 | 7.2% |

PR #1326 was 0.0032 better before quantization. Mine was 0.0017 better after it, because the PR code needed pruning. Setting ROPE base 10,000 with softcap 20 was 0.0006 worse than ROPE base 22,000 with softcap 20 (section 17). Differences in the PR's settings: embedding learning rate 0.6 (17x mine), softcap 30, recurrence on layers 4 and 5 only, partial RoPE (16 dimensions), EMA 0.997, recurrence start at step 3000. Its local result of 1.0959 compared with 1.0896 on H100.

## 19. v8 round 5: skip, lane merge and LoRA rank (2026-04-07)

Log: `sweeps/local/v8_sweep_r5/`. Three changes together: skip connection 4 targets the live lane in parallel mode (it was dead code), a per-layer lane-merge weight, and LoRA rank 32.

| Metric | Value |
|--------|-------|
| Steps | about 5500 |
| Post-EMA | 1.1013 |
| Quant | 1.1217 |
| Pruning | 14.8% |

The rank-32 LoRA exceeded the budget. The three changes were not separated.

## 20. Gradient analysis (2026-04-07)

Script: `analysis/gradient_analysis.py`. Checkpoint: the v7/v8 EMA weights. Config: 11 layers, recurrence on layers 3, 4, 5 (14 virtual layers), parallel residuals from layer 7, LoRA rank 16, softcap 30. Method: 8 batches of 16 sequences of 2048 tokens from the validation set, mean gradient norms. The script must switch recurrence on before the analysis.

MLP-up gradient norm by layer:

| Layer | Norm | Note |
|-------|------|------|
| 0 | 0.084 | attention disabled |
| 1 | 0.268 | |
| 2 | 0.384 | |
| 3 | 0.370 | recurrent |
| 4 | 0.429 | recurrent |
| 5 | 0.447 | recurrent, peak |
| 6 | 0.252 | last serial |
| 7 | 0.177 | first parallel |
| 8 | 0.123 | parallel |
| 9 | 0.125 | parallel |
| 10 | 0.245 | output |

| Layer group | Mean gradient | Relative |
|-------------|---------------|----------|
| Recurrent (3, 4, 5) | 0.415 | 2.9x |
| Serial (1, 2, 6) | 0.301 | 2.1x |
| Parallel (7, 8, 9) | 0.142 | 1.0x |
| Output (10) | 0.245 | 1.7x |

Other observations:

- Layer 5 attention output and value gradients were exactly 0.000 on the repeat pass.
- Layer 7 had the lowest gradients on every metric (attention scale 0.007, residual mix 0.037, query gain 0.002).
- The lane-merge scalar had the largest scalar gradient (0.0875). The repeat-pass scale gradients were 0.006, 0.023 and 0.076 for layers 3, 4, 5.
- The layer-0 MLP scale gradient (0.247) was the largest block scalar. The model pushed that MLP toward zero.
- Skip connection gradients (weight / gate): skip 0 0.020 / 0.003, skip 1 0.026 / 0.012, skip 2 0.052 / 0.036, skip 3 0.063 / 0.048, skip 4 0.053 / 0.020. The skip 4 gradient is non-zero only after the fix in section 19.
- The check loss was 5.05, higher than expected, probably from sequential data loading with EMA weights. Relative patterns should hold.

## 21. v8.5 architectural ideas and sweep (2026-04-07 to 2026-04-08)

Ideas from the gradient analysis:

| Idea | Code switch | Note |
|------|-------------|------|
| A: attention-query LoRA on repeat passes (rank 8) | `REPEAT_ATTN_LORA` | 24,576 parameters |
| B: disable layer-0 MLP | `DISABLE_LAYER0_MLP` | |
| C: skip repeat attention | `SKIP_REPEAT_ATTN` | exclusive with A |
| E: scale gradients of parallel-block scalars by 2.5 | `PARALLEL_SCALAR_GRAD_SCALE` | 16 parameters |
| F: LN scale off and softcap 20 | - | |

XSA here means: y = y - (y . v) v, with v the normalized value vector. It removes the value-parallel part of the attention output. It is not a sliding window. Ideas that I rejected before testing: short convolution before attention (kept as a candidate, not run), sliding-window attention (a misreading of XSA), wider embeddings with a projection (breaks tied embeddings).

First round (2 hours): invalid, see section 22. Round 2 (3 hours, warmdown 1500): all results invalid because of two bugs (section 22):

| Test | Steps | Post-EMA | Quant | Pruning |
|------|-------|----------|-------|---------|
| F: LN scale off, softcap 20 | 5541 | 1.1066 | 1.1282 | 45.4% |
| E: parallel gradient scale 2.5 | about 5400 | 1.1085 | 1.1310 | 48.0% |
| A: attention-query LoRA | about 5200 | 1.1110 | 1.1426 | 66.4% |
| B, C | killed before GPTQ | - | - | - |

Round 3, with the bugs fixed (warmdown 3500), `sweeps/local/v8.5_sweep/`:

| Test | Steps | Post-EMA | Quant | Delta vs best | Pruning |
|------|-------|----------|-------|---------------|---------|
| Ffix: LN scale off, softcap 20 | 5551 | 1.0986 | 1.1117 | - | 0% |
| Bfix: disable layer-0 MLP | not recorded | 1.0992 | 1.1130 | +0.0013 | 0% |
| R3 re-run | 5371 | 1.1000 | 1.1129 | +0.0012 | 0% |
| CL5: skip layer-5 repeat attention | not recorded | 1.1015 | 1.1152 | +0.0035 | 0% |
| E-fix (parallel gradient scale) | killed at step 4000 of 20000 | - | - | - | - |

Ffix became the local best (compressed 15.62MB, headroom 0.28MB). The tests of ideas A, C (all layers), D and E-fix with valid settings were not completed.

## 22. Warmdown bug and compression catastrophe (2026-04-08)

In the first v8.5 run, the warmdown was set in steps (3500) and the wall-clock limit was 7200 s. The schedule is wall-clock based, so the learning rate was in decay from step 0 (77% of peak at step 100, 35% at step 2000). Test A's 1.1293 BPB at 3249 steps was invalid. Results were archived in `sweeps/local/v8.5_sweep_INVALID_warmdown/`.

The second round used warmdown 1500, based on an assumed step time. Compressed models were 16.88MB against 15.88MB for round 3 of v8. Two causes:

1. Warmdown 1500 instead of 3500 left 27.1% of training in warmdown instead of 62.7%. The weights were rougher and about 0.75MB larger after Brotli.
2. A name pattern kept `repeat_lora_B` (98,304 parameters) in float32 (393KB) instead of int8 (98KB). This cost about 0.25MB.

Both fixes were applied (warmdown 3500, pattern removed).

## 23. Analysis of PR #1423 and PR #1326 from the H100 view (2026-04-08)

[PR #1423](https://github.com/openai/parameter-golf/pull/1423) uses an 8192 vocabulary, not 4096. It builds on the SP8192 base of [PR #1394](https://github.com/openai/parameter-golf/pull/1394) and the pre-quant TTT of [PR #1364](https://github.com/openai/parameter-golf/pull/1364).

| Stage | PR #1423 (SP8192, H100) | PR #1326 (SP4096, H100) | Mine (local, round 3) |
|-------|-------------------------|-------------------------|-----------------------|
| Post-EMA | 1.1026 | 1.0967 | 1.0991 |
| Pre-quant TTT | 1.0687 | - | - |
| Roundtrip | 1.0946 | 1.1082 | 1.1119 |
| Quant degradation | +0.026 | +0.012 | +0.013 |
| Sliding (stride 64) | 1.0802 | 1.0896 | - |
| Score-first TTT | - | 1.0889 | - |
| Final | 1.0791 | 1.0896 | 1.1119 |

H100 notes: PR #1326 ran 5440 steps in 590 s (about 108 ms before and 140 ms after recurrence), with LZMA-wrapped code of 24,671 bytes and 0.4% pruning. PR #1423 ran 5965 steps at about 100 ms per step, with SWA from step 5200 and late QAT from step 5376, and 116,724 bytes of code.

Techniques I had not used at that time: late QAT, SDClip, an LZMA code wrapper (about 100KB of code against 25KB).

Estimated gap decomposition (my 1.1119 against 1.0791, 0.033 BPB): sliding evaluation -0.015 to -0.019, code size -0.002 to -0.004, local-to-H100 quality about -0.002, SP4096 to SP8192 about -0.004, pre-quant TTT -0.006 to -0.010. These are estimates.

## 24. H100 deploy package and Run 2 (2026-04-08 to 2026-04-09)

Changes in the deploy code: sliding-window evaluation (stride 64), artifact limit corrected to 16,000,000 bytes (it was 16,777,216), gradient clipping moved after the all-reduce (it had been about 2.8x weaker on 8 GPUs), dead SWA code removed, warmdown default 3500, code minified from 112,669 to 75,421 bytes (60,686 after hot fixes on the node).

Run 2 (seed 1337): the first attempt crashed because the minifier renamed a Muon parameter (`weight_decay` to `wdy`) without updating the call site. After a hot fix, training finished and GPTQ crashed. A resume run completed it.

| Metric | Run 2 | Local Ffix | PR #1326 (H100) |
|--------|-------|------------|-----------------|
| Steps | 5194 | about 5550 | 5440 |
| Post-EMA | 1.1113 | 1.0986 | 1.0967 |
| Roundtrip | 1.1634 | 1.1117 | 1.1082 |
| Sliding | 1.1443 | - | 1.0896 |
| Pruning | 40.6% | 0% | 0.4% |
| Code (bytes) | 60,686 | 60,015 | 24,671 |

The model compressed to 16.21MB against a 15.20MB budget. Step time was 93.5 ms before and 115 ms after recurrence. Weaker pre-quant quality gave rougher weights and worse compression.

## 25. Compression pipeline analysis (2026-04-09)

My GPTQ and the one in PR #1326 use the same algorithm: Cholesky-based, column reordering, a five-candidate clip search, per-row float16 scales, block size 128, Brotli with byte shuffle. TurboQuant targets KV-cache compression and does not apply here. Levers listed for the next run, with estimated size effect: LZMA code wrapper (about -38KB), late QAT (-0.2 to -0.5MB), SWA (-0.1 to -0.3MB), SDClip (faster, same size). Late QAT was later tested and did not help (sections 33 and 35).

## 26. v8.5 final results and the int5 attention decision (2026-04-09)

All LoRA-16 models compressed to 15.84MB to 15.89MB across a pre-quant range of 1.099 to 1.104. With a 60KB code size the budget is 15.94MB and about 28% pruning would be needed.

Earlier same-model comparisons with different quantization (`rel_mlp3.5` series): attention at 5 bits against 6 bits gave BPB differences of -0.000005 (MLP 6 bits) and +0.000034 (MLP 8 bits). The size savings were 0.69MB and 0.68MB. I chose int5 attention.

Local confirmation (`sweeps/local/v10_int5attn/`, 3 hours each, same code):

| Arm | Steps | Post-EMA | Roundtrip | Unpruned model | Pruning | Artifact (bytes) |
|-----|-------|----------|-----------|----------------|---------|------------------|
| Int5 attention | 5552 | 1.0981 | 1.1210 | 14.68MB | 0% | 15,505,554 |
| Int6 attention (control) | 5349 | 1.0991 | 1.1279 | 15.62MB | 20.0% | 15,999,977 |

Tools added: late QAT (`LATE_QAT`), SWA weight choice, SDClip (`SDCLIP_K` 12.85), LZMA wrapper (`src/build_lzma_wrapper.py`; 60KB to 22KB). Late QAT test v9 F was killed at step 500 of 20000.

## 27. H100 Run 3 (2026-04-09)

Log: `runs/run03/train.log`. Config: 11 layers, LN scale off, softcap 20, warmdown 3500, ROPE base 10,000, int5 attention.

| Metric | Value |
|--------|-------|
| Steps | 5209 (115 ms per step) |
| Post-EMA | 1.1100 |
| Roundtrip | 1.1316 |
| Sliding | 1.1131 |
| Pruning | 0% |
| Model compressed | 14.47MB |
| Artifact | 15,232,121 bytes |

Best SP4096 result. The gap to the then-leader (1.0896) was 0.0235.

## 28. v11 scale test, local 3 hours (2026-04-09)

Log: `sweeps/local/v11_scale/`. Baseline: 11 layers, Ffix (1.0986 post-EMA, 1.1117 roundtrip).

| Metric | 12 layers | Full untied repeat MLP |
|--------|-----------|------------------------|
| Parameters | 37,410,919 | 37,547,103 |
| Steps | 5285 | 5344 |
| ms per step | 2044 | 2021 |
| Post-EMA | 1.0922 (-0.0064) | 1.0957 (-0.0029) |
| Roundtrip | 1.1268 | 1.1953 |
| Pruning | 18.7% | 56.6% |
| Unpruned model | 15.81MB | 16.88MB |

My projection for 12 layers on H100 was about 1.100 sliding. See section 29 for the real result. The untied MLP was not viable.

## 29. H100 Run 4: 12 layers (2026-04-09)

Log: `runs/run04/resume_quant.log`. Pre-quant 1.1110 (Run 3: 1.1100), pruning 12.5% (unpruned 15.62MB), roundtrip 1.1445 (Run 3: 1.1316), artifact 15,999,081 bytes. The sliding evaluation crashed on a function name that the minifier had changed. Cause of the loss: a slower step (about 126 ms against 115 ms, estimated), a larger model and pruning damage.

## 30. Int6 TTT and PR #1326 comparison, local 30 minutes (2026-04-10)

Logs: `experiments/quantization/ttt_int6_test.log`, `ttt_test.log`, `pr1326_test.log`.

| Metric | Mine (int5 attention) | Mine (int6 attention) | PR #1326 (int6) |
|--------|-----------------------|-----------------------|-----------------|
| Steps | 972 | 972 | 1025 |
| ms per step after recurrence | 1853 | 1853 | about 1750 |
| Pre-stop val BPB | 1.2500 | 1.2500 | 1.1902 |
| Post-EMA | 1.2934 | 1.2899 | 1.2397 |
| Roundtrip | 1.9183 | 1.6270 | 1.2674 |
| Quant degradation | +0.625 | +0.337 | +0.028 |
| Sliding | 1.9179 | 1.6121 | 1.2490 |
| TTT | 1.4988 | 1.4132 | 1.2358 |
| TTT gain from sliding | -0.419 | -0.199 | -0.013 |

At 972 steps the models are too immature to compare quantizers. The degradation comes from weight immaturity. The run does show a 0.050 post-EMA gap between the codes at equal bits.

## 31. CAGE-lite (2026-04-10)

CAGE-lite pulls each 2D weight toward its RTN grid point in the last 15% of training. The pull weight grows linearly from 0 to 0.01. It is applied before the EMA update. The idea follows arXiv 2510.18784.

| Metric | Control, short (972 steps) | CAGE-lite, short (940 steps) |
|--------|----------------------------|------------------------------|
| Pre-stop val BPB | 1.2500 | 1.2580 |
| Post-EMA | 1.2899 | 1.3038 (+0.0139) |
| Roundtrip | 1.6270 | 1.6633 (+0.0363) |

Logs: `experiments/quantization/cage_lite_test.log`, `cage_lite_long_test.log`.

Long test (3 hours):

| Metric | v8 baseline | CAGE-lite |
|--------|-------------|-----------|
| Steps | 5579 | 5368 |
| Post-EMA | 1.1000 | 1.1001 |
| Roundtrip | 1.1131 | 1.2428 |
| Quant degradation | +0.0131 | +0.1427 |
| Pruning | 0% | 92.6% |
| Sliding | - | 1.2217 |

Pre-quant quality was unchanged. The pull moved weights onto a grid and made the weights harder to compress with Brotli. I removed CAGE-lite.

## 32. GPTQ real-data Hessian tests (2026-04-10)

Logs: `experiments/quantization/gptq_fix_long_test.log`, `gptq_realdata_damped_test.log`.

Test 1 changed two things: Hessians from real training data (instead of generated sequences), and no pre-damping (0.01 x mean diagonal) in the Hessian step. The aim was to match the PR #1326 values.

| Metric | Ffix | Real data, no pre-damping |
|--------|------|---------------------------|
| Steps | 5551 | 5566 |
| Post-EMA | 1.0986 | 1.0997 |
| Unpruned model | about 14.5MB | 16.25MB |
| Pruning | 0% | 43.3% |
| Roundtrip | 1.1117 | 1.1756 |
| Quant degradation | +0.0131 | +0.0759 |

Real-data Hessians are less well conditioned. Without pre-damping, the GPTQ retry loop used larger damping (0.05 to 0.5), which weakened the correction. Pre-damping is needed. The separate test of real data with pre-damping restored has no final result in the log. Section 33 reports a later test with that setting.

## 33. GPTQ ablation and compression investigation (2026-04-10)

Log: `experiments/quantization/ablation_results.log` and related test logs.

| Test (same local checkpoint) | Post-EMA | Sliding | Pruning | Unpruned |
|------------------------------|----------|---------|---------|----------|
| Late QAT with real-data GPTQ (24K sequences) | 1.1003 | 1.1624 | 43.3% | 16.25MB |
| Control with real-data GPTQ | 1.1006 | 1.1571 | 44.2% | 16.27MB |

| Variant | Compressed size | Pruning |
|---------|-----------------|---------|
| RTN (no GPTQ) | 16.25MB | 43% |
| Real-data GPTQ | 16.25MB | 43% |
| Generated-sequence GPTQ | 16.25MB | 43% |

RTN gives the same size as every GPTQ variant. Three more changes (recurrence in the Hessian model, a fresh permutation, no pre-damping) also left the size unchanged. Comparison: the H100 Run 3 model was 14.47MB at Brotli quality 6. The local model was 16.25MB at quality 6 and 15.38MB at quality 11. The size search used quality 6 and overestimated the size by about 0.87MB. Byte shuffle added 0.02MB.

Conclusions: local runs cannot validate compression changes, and late QAT cannot be judged locally (0.003 BPB, noise). The ablation script failed on a state-dict shape mismatch for the plain "original" arm, so that arm has no result.

## 34. Competitor comparison and Run 5 preparation (2026-04-11)

Gap to [PR #1326](https://github.com/openai/parameter-golf/pull/1326) and [PR #1331](https://github.com/openai/parameter-golf/pull/1331), measured at Run 3: pre-quant 1.1100 against about 1.0850 (0.025), which is the whole gap. Quantization loss was no worse than theirs. Possible causes in the training recipe: my 14 virtual layers (recurrence on 3, 4, 5) against their 13, softcap 20 against 30, EMA 0.9965 against 0.997, ROPE base.

Run 5 plan: match the competitor recipe and add my quantization tools. Changes: recurrence on layers 4 and 5, softcap 30, EMA 0.997, ROPE base 10,000, recurrence start 3000, SDClip, pre-damping restored, late QAT in the last 10%, byte shuffle off, generated-sequence Hessians. Code: 64,166 bytes, 23,049 bytes with the LZMA wrapper. Projection was 1.085 to 1.105 sliding. That projection was wrong (section 35).

## 35. H100 Runs 5, 6 and 7 (2026-04-11)

Logs: `runs/run05/`, `runs/run06/`, `runs/run07/`. Pre-quant values below come from the run logs.

| Metric | Run 3 | Run 5 | Run 6 | Run 7 |
|--------|-------|-------|-------|-------|
| Strategy | baseline | competitor recipe, SDClip, late QAT | Run 3 recipe, SDClip, late QAT | Run 3 recipe, SDClip |
| Steps | 5209 | 5577 | 5182 | 5182 |
| Post-EMA | 1.1100 | 1.1113 | 1.1157 | 1.1149 |
| Roundtrip | 1.1316 | 1.1333 | 1.1386 | 1.1347 |
| Sliding | 1.1131 | 1.1148 | 1.1198 | 1.1161 |
| TTT | - | 1.1109 | 1.1151 | not recorded |
| Sliding minus post-EMA | 0.0031 | 0.0035 | 0.0041 | 0.0012 |
| Pruning | 0% | 0% | 0% | 0% |
| Artifact (bytes) | 15,232,121 | 15,951,253 | 15,902,041 | 15,956,661 |

Run 5: a missing `fwl()` method in the Hessian model class crashed GPTQ. A resume run (cost about $4.50) finished it. TTT gained 0.0039. Run 6: TTT gained 0.0047. Cost about $11.

The written log used 1.0986 as the Run 3 pre-quant value and 0.0131 as its quantization gap. Those values came from the local run. With the node values above, the Run 6 regression is 0.0057 and not 0.0171, and the change in quantization loss from SDClip is smaller than first stated (see `docs/findings.md`, finding 4).

Forensics (see `docs/h100_runs.md`): `REPEAT_PASS_SCALE` was 1 in Run 3 and 0 in Runs 6 and 7. ROPE base was 10,000 in Run 3 and 22,000 in Runs 6 and 7. Validation BPB at step 4000: 1.1550 (Run 3), 1.1591 (Run 6), 1.1599 (Run 7), 1.1521 (Run 9).

## 36. Local TTT optimizer sweep (2026-04-11)

Log: `sweeps/local/ttt_sweep/`. Setup: the Run 3 checkpoint on the local workstation, one quantized model for all configs (sliding 1.1549, 33.3% local pruning, pre-quant 1.1098, roundtrip 1.1739).

| Rank | Config | Optimizer | LR | Frozen blocks | Epochs | TTT BPB | Gain | Versus SGD baseline |
|------|--------|-----------|----|---------------|--------|---------|------|---------------------|
| 1 | adamw_f0 | AdamW | 1e-4 | 0 | 3 | 1.1173 | -0.0376 | +0.0107 |
| 2 | adamw_con | AdamW | 1e-4 | 2 | 3 | 1.1174 | -0.0375 | +0.0106 |
| 3 | adamw_e5 | AdamW | 1e-4 | 2 | 5 | 1.1239 | -0.0310 | +0.0041 |
| 4 | sgd_baseline | SGD | 0.002 | 0 | 3 | 1.1280 | -0.0269 | - |
| 5 | sgd_low | SGD | 0.001 | 0 | 3 | 1.1312 | -0.0237 | -0.0032 |

Frozen blocks did not matter. Five epochs overfit. A lower SGD learning rate was worse. The local gain was expected to shrink on H100.

## 37. H100 Runs 8 and 9 (2026-04-11)

Run 8 loaded the Run 3 checkpoint, quantized it with SDClip, and ran AdamW TTT (learning rate 1e-4, weight decay 0.01, 3 epochs). Results: post-EMA 1.1098, roundtrip 1.1313, sliding 1.1126, TTT 1.1114 (gain 0.0012), artifact 15,894,086 bytes. It tested the resume path and was not a submission. Run 9 trained fresh with `REPEAT_PASS_SCALE=1`, ROPE base 10,000 and SGD TTT. Post-EMA was 1.1088 after 5174 steps and validation at step 4000 was 1.1521. The run stopped during GPTQ.

## 38. H100 Run 10, SP8192 (2026-04-11)

Logs: `runs/run10/`, `runs/run10c/`. Changes against Run 9: vocabulary 8192, VE off, a separate SDClip constant for the embedding (20.0), warmdown fraction 0.72, TTT learning rate 0.005, real-data Hessians including one for the embedding, three code fixes (the `<= 6` test for the embedding became `<= 8`, a skip-push mismatch in the Hessian model, and an unused environment variable).

| Stage | Steps | Post-EMA | Result |
|-------|-------|----------|--------|
| 10a (fresh train) | 5051 at 118 ms per step | 1.1045 at end of training; 1.1033 when re-evaluated from the saved checkpoint | GPTQ crashed on an NCCL timeout |
| 10b (resume, int8 embedding) | - | 1.1033 | 16.35MB unpruned against 15.24MB budget, 41.1% pruning |
| 10c (resume, int6 embedding) | - | 1.1033 | Best result |

Crash cause in 10a: only rank 0 ran GPTQ (24,512 calibration sequences), while ranks 1 to 7 waited at a barrier until the 600 s watchdog fired. Fixes: NCCL timeout 1800 s, quantization gated to rank 0, calibration capped at 256 sequences.

| Metric | Run 10c | Run 3 |
|--------|---------|-------|
| Post-EMA | 1.1033 | 1.1100 |
| Roundtrip | 1.1226 | 1.1316 |
| Sliding | 1.1056 | 1.1131 |
| TTT | 1.1009 | - |
| Pruning | 2.1% | 0% |
| Artifact (bytes) | 15,991,240 | 15,232,121 |
| Model compressed (bytes) | 15,967,716 | - |
| Code with LZMA wrapper (bytes) | 23,524 | about 22,000 |

How others handled the budget: the entry at 1.0810 needed no pruning (int6 matrices, int8 embedding). [PR #1331](https://github.com/openai/parameter-golf/pull/1331) used Hessian-sensitivity mixed int5 and int6. [PR #1326](https://github.com/openai/parameter-golf/pull/1326) used a one-shot heuristic. Mine used a binary search on +-1 values sorted by scale-squared error.

## 39. Eval-time experiments on the Run 10 checkpoint (2026-04-19)

Details are in `docs/findings.md`, section C, and in `experiments/eval_time/`. All used the saved Run 10 EMA checkpoint (1.1033).

| Experiment | Result |
|------------|--------|
| Temperature scaling, pre-quant | Minimum at T = 1.00 (V curve) |
| Temperature scaling, post-quant (local, 39% pruning) | Minimum at T = 1.00 |
| Multi-layer logit averaging | Every arm worse (+0.0218 to +0.2403) |
| Z-loss, 500-step A/B | Worse (post-EMA 1.4780 against 1.4274) |
| Hessian-model compile | Not run |
| LoRA ensemble, hard-sequence resampling | Result not recorded |
