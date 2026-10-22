#!/usr/bin/env python3
"""
Training loop micro-optimizations for Parameter Golf GPT.

Tests:
1. CUDA graph capture of the training step
2. Prefetched data loading (overlap randint with compute)
3. Torch.set_float32_matmul_precision('high') for TF32 matmuls
4. torch.backends.cudnn.benchmark = True
5. Gradient accumulation with static graph
6. Combined best settings

Usage:
    python perf/training_loop_opts.py [--steps 30] [--batch-size 48]
    python perf/training_loop_opts.py --quick
"""

import argparse
import copy
import gc
import sys
import time
import torch
import torch.nn.functional as F
from contextlib import contextmanager

sys.path.insert(0, ".")
sys.path.insert(0, "./src/baselines")
from train_gpt_modified import GPT, Hyperparameters, restore_fp32_params
from perf.bench_harness import create_model, print_comparison

DEVICE = "cuda"


def run_baseline(model, h, args):
    """Standard training loop — same as bench_harness.run_steps but inline for clarity."""
    compiled = torch.compile(model, dynamic=False, fullgraph=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    vocab = h.vocab_size
    bs, sl = args.batch_size, args.seq_len

    # Warmup
    for _ in range(3):
        x = torch.randint(0, vocab, (bs, sl), device=DEVICE)
        y = torch.randint(0, vocab, (bs, sl), device=DEVICE)
        optimizer.zero_grad()
        with torch.autocast("cuda", torch.bfloat16):
            loss = compiled(x, y)
        loss.backward()
        optimizer.step()

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    losses = []
    t0 = time.perf_counter()

    for step in range(args.steps):
        x = torch.randint(0, vocab, (bs, sl), device=DEVICE)
        y = torch.randint(0, vocab, (bs, sl), device=DEVICE)
        optimizer.zero_grad()
        with torch.autocast("cuda", torch.bfloat16):
            loss = compiled(x, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return {
        "losses": losses,
        "elapsed_s": elapsed,
        "tok_per_s": (args.steps * bs * sl) / elapsed,
        "ms_per_step": (elapsed / args.steps) * 1000,
        "peak_mib": torch.cuda.max_memory_allocated() / 1024**2,
        "compile_time_s": 0,
    }


def run_prefetch(model, h, args):
    """Prefetch next batch on a separate CUDA stream while compute runs."""
    compiled = torch.compile(model, dynamic=False, fullgraph=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    vocab = h.vocab_size
    bs, sl = args.batch_size, args.seq_len
    prefetch_stream = torch.cuda.Stream()

    def gen_batch():
        return (
            torch.randint(0, vocab, (bs, sl), device=DEVICE),
            torch.randint(0, vocab, (bs, sl), device=DEVICE),
        )

    # Warmup
    x, y = gen_batch()
    for _ in range(3):
        optimizer.zero_grad()
        with torch.autocast("cuda", torch.bfloat16):
            loss = compiled(x, y)
        loss.backward()
        optimizer.step()
        x, y = gen_batch()

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    losses = []
    t0 = time.perf_counter()

    # Pre-generate first batch
    x, y = gen_batch()

    for step in range(args.steps):
        # Launch prefetch of next batch on separate stream
        with torch.cuda.stream(prefetch_stream):
            x_next, y_next = gen_batch()

        # Compute on default stream
        optimizer.zero_grad()
        with torch.autocast("cuda", torch.bfloat16):
            loss = compiled(x, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        # Wait for prefetch
        torch.cuda.current_stream().wait_stream(prefetch_stream)
        x, y = x_next, y_next

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return {
        "losses": losses,
        "elapsed_s": elapsed,
        "tok_per_s": (args.steps * bs * sl) / elapsed,
        "ms_per_step": (elapsed / args.steps) * 1000,
        "peak_mib": torch.cuda.max_memory_allocated() / 1024**2,
        "compile_time_s": 0,
    }


def run_tf32(model, h, args):
    """Enable TF32 matmul precision (faster but slightly less precise f32 matmuls)."""
    old_precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("high")
    try:
        result = run_baseline(model, h, args)
    finally:
        torch.set_float32_matmul_precision(old_precision)
    return result


def run_cudnn_benchmark(model, h, args):
    """Enable cuDNN benchmark mode."""
    old = torch.backends.cudnn.benchmark
    torch.backends.cudnn.benchmark = True
    try:
        result = run_baseline(model, h, args)
    finally:
        torch.backends.cudnn.benchmark = old
    return result


def run_set_grad_none(model, h, args):
    """Use optimizer.zero_grad(set_to_none=True) for slightly faster zeroing."""
    compiled = torch.compile(model, dynamic=False, fullgraph=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    vocab = h.vocab_size
    bs, sl = args.batch_size, args.seq_len

    for _ in range(3):
        x = torch.randint(0, vocab, (bs, sl), device=DEVICE)
        y = torch.randint(0, vocab, (bs, sl), device=DEVICE)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", torch.bfloat16):
            loss = compiled(x, y)
        loss.backward()
        optimizer.step()

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    losses = []
    t0 = time.perf_counter()

    for step in range(args.steps):
        x = torch.randint(0, vocab, (bs, sl), device=DEVICE)
        y = torch.randint(0, vocab, (bs, sl), device=DEVICE)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", torch.bfloat16):
            loss = compiled(x, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return {
        "losses": losses,
        "elapsed_s": elapsed,
        "tok_per_s": (args.steps * bs * sl) / elapsed,
        "ms_per_step": (elapsed / args.steps) * 1000,
        "peak_mib": torch.cuda.max_memory_allocated() / 1024**2,
        "compile_time_s": 0,
    }


def run_grad_accum_static(model, h, args):
    """Simulate grad accumulation (2 micro-steps) with static graph hint.
    This tests whether splitting the batch and accumulating is faster due to
    better memory behavior, even though it's more kernel launches."""
    compiled = torch.compile(model, dynamic=False, fullgraph=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    vocab = h.vocab_size
    micro_bs = args.batch_size // 2
    sl = args.seq_len
    accum_steps = 2

    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        for _ in range(accum_steps):
            x = torch.randint(0, vocab, (micro_bs, sl), device=DEVICE)
            y = torch.randint(0, vocab, (micro_bs, sl), device=DEVICE)
            with torch.autocast("cuda", torch.bfloat16):
                loss = compiled(x, y) / accum_steps
            loss.backward()
        optimizer.step()

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    losses = []
    t0 = time.perf_counter()

    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        for _ in range(accum_steps):
            x = torch.randint(0, vocab, (micro_bs, sl), device=DEVICE)
            y = torch.randint(0, vocab, (micro_bs, sl), device=DEVICE)
            with torch.autocast("cuda", torch.bfloat16):
                loss = compiled(x, y) / accum_steps
            loss.backward()
            step_loss += loss.item()
        optimizer.step()
        losses.append(step_loss)

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return {
        "losses": losses,
        "elapsed_s": elapsed,
        "tok_per_s": (args.steps * args.batch_size * sl) / elapsed,
        "ms_per_step": (elapsed / args.steps) * 1000,
        "peak_mib": torch.cuda.max_memory_allocated() / 1024**2,
        "compile_time_s": 0,
    }


def run_combined_best(model, h, args):
    """Combine: TF32 + set_grad_none + cudnn benchmark + prefetch."""
    old_precision = torch.get_float32_matmul_precision()
    old_cudnn = torch.backends.cudnn.benchmark
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True

    try:
        compiled = torch.compile(model, dynamic=False, fullgraph=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
        vocab = h.vocab_size
        bs, sl = args.batch_size, args.seq_len
        prefetch_stream = torch.cuda.Stream()

        def gen_batch():
            return (
                torch.randint(0, vocab, (bs, sl), device=DEVICE),
                torch.randint(0, vocab, (bs, sl), device=DEVICE),
            )

        x, y = gen_batch()
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", torch.bfloat16):
                loss = compiled(x, y)
            loss.backward()
            optimizer.step()
            x, y = gen_batch()

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        losses = []
        t0 = time.perf_counter()
        x, y = gen_batch()

        for step in range(args.steps):
            with torch.cuda.stream(prefetch_stream):
                x_next, y_next = gen_batch()

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", torch.bfloat16):
                loss = compiled(x, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

            torch.cuda.current_stream().wait_stream(prefetch_stream)
            x, y = x_next, y_next

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        return {
            "losses": losses,
            "elapsed_s": elapsed,
            "tok_per_s": (args.steps * bs * sl) / elapsed,
            "ms_per_step": (elapsed / args.steps) * 1000,
            "peak_mib": torch.cuda.max_memory_allocated() / 1024**2,
            "compile_time_s": 0,
        }
    finally:
        torch.set_float32_matmul_precision(old_precision)
        torch.backends.cudnn.benchmark = old_cudnn


EXPERIMENTS = {
    "baseline": run_baseline,
    "prefetch_stream": run_prefetch,
    "tf32_matmul": run_tf32,
    "cudnn_benchmark": run_cudnn_benchmark,
    "set_grad_none": run_set_grad_none,
    "grad_accum_2x": run_grad_accum_static,
    "combined_best": run_combined_best,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--only", type=str, default=None)
    args = parser.parse_args()

    if args.quick:
        args.steps = 10
        args.batch_size = 16
        args.seq_len = 512

    print("=" * 70)
    print("Training Loop Optimization Experiments")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Steps: {args.steps}, Batch: {args.batch_size}, Seq: {args.seq_len}")
    print(f"TF32 current: {torch.get_float32_matmul_precision()}")
    print(f"cuDNN benchmark: {torch.backends.cudnn.benchmark}")
    print("=" * 70)

    _, h, init_state = create_model()

    exp_names = list(EXPERIMENTS.keys())
    if args.only:
        exp_names = [n for n in args.only.split(",") if n in EXPERIMENTS]

    results = {}
    for name in exp_names:
        print(f"\n--- {name} ---")
        torch._dynamo.reset()
        gc.collect()
        torch.cuda.empty_cache()

        model, _, _ = create_model()
        model.load_state_dict(init_state)

        try:
            r = EXPERIMENTS[name](model, h, args)
            r["status"] = "OK"
            results[name] = r
            print(f"  {r['ms_per_step']:.1f} ms/step, {r['tok_per_s']:,.0f} tok/s, peak: {r['peak_mib']:.0f} MiB")
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback; traceback.print_exc()
            results[name] = {"status": f"FAILED: {e}", "ms_per_step": float("inf"),
                             "tok_per_s": 0, "peak_mib": 0, "losses": [], "elapsed_s": 0,
                             "compile_time_s": 0}

        del model
        gc.collect()
        torch.cuda.empty_cache()

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    ok = {k: v for k, v in results.items() if v.get("status") == "OK"}
    print_comparison(ok, baseline_key="baseline")

    base_ms = results.get("baseline", {}).get("ms_per_step", 1)
    ranked = sorted(ok.items(), key=lambda x: x[1]["ms_per_step"])
    print(f"\n{'Rank':<6} {'Config':<40} {'Speedup':<10}")
    print("-" * 56)
    for i, (name, r) in enumerate(ranked, 1):
        print(f"{i:<6} {name:<40} {base_ms / r['ms_per_step']:.3f}x")


if __name__ == "__main__":
    main()
