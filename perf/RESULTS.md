# Optimization Experiment Results — 2026-04-05

## Test Environment
- GPU: NVIDIA RTX PRO 6000 Blackwell Max-Q (sm_120, 96GB)
- PyTorch 2.11.0+cu130, Python 3.12
- Model: 512-dim, 11 layers, GQA (8 heads, 4 KV), 4096 vocab
- Config: batch=48, seq=2048 (98,304 tok/step), 20 measured steps, 3 warmup

## Inductor Config Sweep

Baseline: **199.7 ms/step, 492,317 tok/s**

| Config | ms/step | Speedup | Status |
|--------|---------|---------|--------|
| baseline | 199.7 | 1.000x | OK |
| epilogue_fusion_off | 199.5 | 1.001x | OK (noise) |
| cutlass_epilogue | 200.3 | 0.997x | OK |
| aggressive_fusion | 200.8 | 0.995x | OK |
| b2b_gemm | 200.9 | 0.994x | OK |
| max_fusion_128 | 201.9 | 0.989x | OK |
| permute_fusion | 202.0 | 0.989x | OK |
| combo_kernels | 202.5 | 0.986x | OK |
| fusion_combo | 202.8 | 0.985x | OK |
| cooperative_reductions | 203.8 | 0.980x | OK |
| max_autotune_gemm | 203.8 | 0.980x | OK |
| max_autotune_pointwise | 204.4 | 0.977x | OK |
| coord_descent | 204.8 | 0.975x | OK |
| cudagraphs | N/A | N/A | FAILED (RoPE cache overwrite) |

**Conclusion: No inductor config improves over baseline.** All are within noise or slower.
CUDA graphs failed due to rotary embedding cache being a static tensor overwritten between steps.

## Training Loop Optimizations

Baseline: **199.4 ms/step, 492,957 tok/s**

| Config | ms/step | Speedup | Peak MiB | Notes |
|--------|---------|---------|----------|-------|
| baseline | 199.4 | 1.000x | 26,513 | |
| tf32_matmul | 199.4 | 1.000x | 26,516 | Already set to "highest" |
| grad_accum_2x | 199.8 | 0.998x | **13,743** | Halves memory! |
| cudnn_benchmark | 200.6 | 0.994x | 26,516 | |
| prefetch_stream | 201.0 | 0.992x | 26,516 | |
| set_grad_none | 201.7 | 0.988x | 26,515 | |
| combined_best | 202.8 | 0.983x | 26,515 | Combining adds overhead |

**Conclusion: No training loop optimization improves speed.** 
Notable: grad_accum_2x halves peak memory with no speed penalty — useful if memory-constrained.

## Why Nothing Helps

1. **torch.compile already maximally fuses** — inductor's default config captures all profitable fusions
2. **The model is compute-bound, not launch-bound** — kernel launch overhead is negligible at 98K tok/step
3. **TF32 is already enabled** — the training script sets highest precision
4. **GEMM shapes are standard** — default cuBLAS/ATEN kernels are already near-optimal for 512×2048
5. **Overhead sources (data gen, optimizer) are tiny** — randint + AdamW are <1% of step time

## H100-Specific Optimization Vectors (researched, not tested)

### DDP Communication — NOT WORTH IT
- Model ~8M params, BF16 gradients ~16MB. NVLink allreduce: ~0.03ms vs ~25ms step. <0.2%.
- `gradient_as_bucket_view=True` — safe to add, immeasurable gain.
- `static_graph=True` — INCOMPATIBLE with Muon's rank-sharded update pattern.
- DDP allreduce + Muon allreduce are NOT redundant (different purposes).

### Gradient Compression — NOT APPLICABLE
- Muon already communicates BF16 (line 889 of train_gpt_modified.py).
- 14MB on NVLink = 0.03ms. PowerSGD overhead would exceed the savings.
- Lead: Newton-Schulz `backend_steps` reduction (10→5-6) saves actual compute.

### Batch Schedule — RISKY
- Shape changes break `torch.compile(fullgraph=True)` — recompilation penalty.
- RoPE configured for seq_len=2048, shorter seqs need reconfiguration.
- Lead: `warmdown_frac=0.667` is very aggressive (LR decay starts at 33% through
  training). Tuning this hyperparameter could improve BPB without code changes.

### CUDA Graphs — REQUIRES CODE CHANGE
- Failed due to RoPE cache overwrite. Fix: add `cudagraph_mark_step_begin()` or
  clone cached cos/sin tensors. Even then, launch overhead is <0.5% of step time.

## Actionable Next Steps (ranked by expected impact)
1. **Tune warmdown_frac** — currently 0.667, try 0.3-0.5 (hyperparameter sweep)
2. **Reduce Newton-Schulz steps** — try 5-6 instead of 10 (saves compute per step)
3. **Add gradient_as_bucket_view=True** — free, safe (line 1566)
4. **Everything else** — not worth the engineering effort for this model size
