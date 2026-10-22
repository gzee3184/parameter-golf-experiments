#!/usr/bin/env python3
"""Profile torch.compile'd GPT model to find kernel-level bottlenecks.

Usage:
    CUDA_VISIBLE_DEVICES=0 python profile_compiled.py
"""

import os
import sys
import torch
import torch.nn.functional as F
from torch.profiler import profile, ProfilerActivity, schedule

# Make sure we can import from the training script directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'baselines'))
# Also ensure kernels dir is on path (for fused_mlp if needed)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from train_gpt_modified import GPT, Hyperparameters

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEVICE = "cuda"
BATCH_SIZE = 48
SEQ_LEN = 2048
WARMUP_STEPS = 3      # profiler warmup (no tracing)
ACTIVE_STEPS = 3      # steps we actually trace
DTYPE = torch.bfloat16

# ---------------------------------------------------------------------------
# Build model
# ---------------------------------------------------------------------------
print("=== Building model ===")
h = Hyperparameters()
model = GPT(h).to(device=DEVICE, dtype=DTYPE)

# Enable recurrence so we profile the full 13-virtual-layer path
if h.recur_layers:
    model.set_recurrence_active(True)
    print(f"  Depth recurrence ON: recur_layers={h.recur_layers}")

num_params = sum(p.numel() for p in model.parameters())
print(f"  Parameters: {num_params:,} ({num_params/1e6:.1f}M)")
print(f"  Virtual layers: {model._get_virtual_layers()}")

# Compile
print("=== Compiling model ===")
compiled_model = torch.compile(model, dynamic=False, fullgraph=True)

# Dummy optimizer (so we profile backward + optimizer step too)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# ---------------------------------------------------------------------------
# Warmup: ensure compilation happens before profiling
# ---------------------------------------------------------------------------
print(f"=== Warming up ({WARMUP_STEPS} compilation steps) ===")
for i in range(WARMUP_STEPS):
    input_ids = torch.randint(0, h.vocab_size, (BATCH_SIZE, SEQ_LEN), device=DEVICE)
    target_ids = torch.randint(0, h.vocab_size, (BATCH_SIZE, SEQ_LEN), device=DEVICE)
    loss = compiled_model(input_ids, target_ids)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    print(f"  warmup step {i}: loss={loss.item():.4f}")
torch.cuda.synchronize()
print("  Compilation done.")

# ---------------------------------------------------------------------------
# Profile
# ---------------------------------------------------------------------------
print(f"\n=== Profiling {ACTIVE_STEPS} steps ===")

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=schedule(wait=0, warmup=1, active=ACTIVE_STEPS, repeat=1),
    record_shapes=True,
    with_stack=False,
    profile_memory=False,
) as prof:
    for step in range(1 + ACTIVE_STEPS):
        input_ids = torch.randint(0, h.vocab_size, (BATCH_SIZE, SEQ_LEN), device=DEVICE)
        target_ids = torch.randint(0, h.vocab_size, (BATCH_SIZE, SEQ_LEN), device=DEVICE)

        loss = compiled_model(input_ids, target_ids)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        torch.cuda.synchronize()

        prof.step()
        if step >= 1:
            print(f"  profiled step {step}: loss={loss.item():.4f}")

# ---------------------------------------------------------------------------
# Report: Top-25 CUDA kernels by total GPU time
# ---------------------------------------------------------------------------
print("\n" + "=" * 120)
print("TOP-25 CUDA KERNELS BY TOTAL GPU TIME (self_cuda_time_total)")
print("=" * 120)

event_list = prof.key_averages()
# Sort by self_device_time_total descending (self CUDA time)
event_list_sorted = sorted(event_list, key=lambda e: e.self_device_time_total, reverse=True)

total_cuda_us = sum(e.self_device_time_total for e in event_list_sorted)

print(f"\n{'Rank':<5} {'%GPU':>6} {'Self CUDA (ms)':>15} {'Count':>7} {'Avg (ms)':>10}  {'Kernel / Op Name'}")
print("-" * 120)

for rank, evt in enumerate(event_list_sorted[:25], 1):
    pct = 100.0 * evt.self_device_time_total / total_cuda_us if total_cuda_us > 0 else 0
    self_ms = evt.self_device_time_total / 1000.0
    avg_ms = self_ms / evt.count if evt.count > 0 else 0
    name = evt.key
    # Truncate long names
    if len(name) > 80:
        name = name[:77] + "..."
    shapes_str = ""
    if evt.input_shapes:
        shapes_str = f"  shapes={evt.input_shapes}"
        if len(shapes_str) > 60:
            shapes_str = shapes_str[:57] + "..."
    print(f"{rank:<5} {pct:>5.1f}% {self_ms:>15.3f} {evt.count:>7} {avg_ms:>10.3f}  {name}{shapes_str}")

print("-" * 120)
print(f"{'':5} {'100%':>6} {total_cuda_us/1000:.3f} ms total CUDA time across {ACTIVE_STEPS} steps")

# ---------------------------------------------------------------------------
# Report: Inductor-generated kernels (triton_*)
# ---------------------------------------------------------------------------
print("\n" + "=" * 120)
print("INDUCTOR-GENERATED KERNELS (triton_* / inductor)")
print("=" * 120)

inductor_kernels = [e for e in event_list_sorted
                    if any(tag in e.key.lower() for tag in ("triton_", "inductor", "triton_poi", "triton_red", "triton_per"))]

if not inductor_kernels:
    # Fallback: look for any compiled kernel patterns
    inductor_kernels = [e for e in event_list_sorted if "Compiled" in e.key or "triton" in e.key.lower()]

if inductor_kernels:
    print(f"\n{'Rank':<5} {'%GPU':>6} {'Self CUDA (ms)':>15} {'Count':>7}  {'Kernel Name'}")
    print("-" * 120)
    for rank, evt in enumerate(inductor_kernels[:30], 1):
        pct = 100.0 * evt.self_device_time_total / total_cuda_us if total_cuda_us > 0 else 0
        self_ms = evt.self_device_time_total / 1000.0
        name = evt.key[:100]
        print(f"{rank:<5} {pct:>5.1f}% {self_ms:>15.3f} {evt.count:>7}  {name}")
    print("-" * 120)
else:
    print("\n  No inductor-generated kernels found in trace.")
    print("  Listing all unique kernel names for inspection:")
    for rank, evt in enumerate(event_list_sorted[:40], 1):
        self_ms = evt.self_device_time_total / 1000.0
        print(f"  {rank:>3}. [{self_ms:>8.3f} ms]  {evt.key[:120]}")

# ---------------------------------------------------------------------------
# Breakdown by category
# ---------------------------------------------------------------------------
print("\n" + "=" * 120)
print("TIME BREAKDOWN BY CATEGORY")
print("=" * 120)

categories = {
    "Attention (SDPA/flash)": [],
    "GEMM (matmul/linear)": [],
    "Elementwise / fused": [],
    "Reduction (norm/softmax)": [],
    "Optimizer (Adam)": [],
    "Other": [],
}

for evt in event_list_sorted:
    k = evt.key.lower()
    t = evt.self_device_time_total
    if any(x in k for x in ("sdp", "flash", "attention", "fmha")):
        categories["Attention (SDPA/flash)"].append(t)
    elif any(x in k for x in ("gemm", "matmul", "mm_", "cutlass", "cublas", "linear")):
        categories["GEMM (matmul/linear)"].append(t)
    elif any(x in k for x in ("adam", "optimizer", "fused_adam")):
        categories["Optimizer (Adam)"].append(t)
    elif any(x in k for x in ("norm", "softmax", "reduce", "layernorm", "rmsnorm")):
        categories["Reduction (norm/softmax)"].append(t)
    elif any(x in k for x in ("elementwise", "pointwise", "triton_poi", "triton_per", "copy", "fill")):
        categories["Elementwise / fused"].append(t)
    else:
        categories["Other"].append(t)

print(f"\n{'Category':<30} {'Total (ms)':>12} {'%GPU':>7} {'#Ops':>6}")
print("-" * 60)
for cat, times in sorted(categories.items(), key=lambda x: -sum(x[1])):
    total_ms = sum(times) / 1000.0
    pct = 100.0 * sum(times) / total_cuda_us if total_cuda_us > 0 else 0
    print(f"{cat:<30} {total_ms:>12.3f} {pct:>6.1f}% {len(times):>6}")

# Export chrome trace for detailed inspection
trace_path = os.path.join(os.path.dirname(__file__), "profile_trace.json")
prof.export_chrome_trace(trace_path)
print(f"\nChrome trace exported to: {trace_path}")
print("Open in chrome://tracing or https://ui.perfetto.dev/")
