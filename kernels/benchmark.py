#!/usr/bin/env python3
"""
Integration benchmark: drop fused Triton kernels into the actual GPT model,
run N training steps unfused vs fused, compare loss curves and tok/s.

Usage:
    python kernels/benchmark.py [--steps 50] [--seq-len 512] [--batch-size 16]
"""

import argparse
import copy
import math
import sys
import time

import torch
import torch.nn.functional as F
from torch import Tensor

sys.path.insert(0, ".")
sys.path.insert(0, "./src/baselines")

# Import model components from training script
from train_gpt_modified import (
    GPT, Hyperparameters, Block, MLP, RMSNorm, CastedLinear,
    restore_fp32_params,
)

from kernels.fused_mlp import (
    leaky05_sq,
    rms_norm_scale_autograd,
    residual_scale_add,
    FusedMLP,
)


DEVICE = "cuda"


def patch_model_level1(model: GPT) -> GPT:
    """Replace MLP activation with fused leaky05² (Level 1)."""
    for block in model.blocks:
        original_mlp = block.mlp
        block.mlp = FusedMLP(original_mlp, level=1)
    return model


def patch_model_level2(model: GPT) -> GPT:
    """Replace MLP forward path with Level 2 fused kernels.

    Patches Block.forward to use:
    - Kernel A: fused rms_norm * scale_factor
    - Kernel B: fused leaky05² activation
    - Kernel C: fused residual add with per-dim scale
    """
    # First apply Level 1 (fused activation in MLP)
    patch_model_level1(model)

    # Now monkey-patch Block.forward to fuse norm + residual add
    original_forward = Block.forward

    def fused_block_forward(self, x: Tensor, x0: Tensor,
                            v_embed: Tensor | None = None,
                            resid_mix_override: Tensor | None = None) -> Tensor:
        mix = (resid_mix_override if resid_mix_override is not None else self.resid_mix).to(dtype=x.dtype)
        x_in = mix[0][None, None, :] * x + mix[1][None, None, :] * x0

        # Attention path (unchanged)
        attn_out = self.attn(self.attn_norm(x_in) * self.ln_scale_factor, v_embed=v_embed)
        x_out = x_in + self.attn_scale.to(dtype=x_in.dtype)[None, None, :] * attn_out

        # MLP path with fused kernels:
        # Kernel A: rms_norm * scale_factor
        normed = rms_norm_scale_autograd(x_out, self.ln_scale_factor)
        # MLP forward (uses fused activation from Level 1)
        mlp_out = self.mlp(normed)
        # Kernel C: residual add with per-dim scale
        x_out = torch.ops.fused_mlp.residual_scale_add(
            x_out, mlp_out, self.mlp_scale
        )
        return x_out

    Block.forward = fused_block_forward
    return model


def run_training_steps(model, num_steps, seq_len, batch_size, vocab_size, device,
                       use_compile=False):
    """Run training steps with synthetic data, return losses and timing."""
    if use_compile:
        model = torch.compile(model, dynamic=False, fullgraph=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    losses = []
    tokens_processed = 0

    # Warmup
    for _ in range(3):
        x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        y = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = model(x, y)
        loss.backward()
        optimizer.zero_grad()

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()

    for step in range(num_steps):
        x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        y = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

        optimizer.zero_grad()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = model(x, y)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        tokens_processed += batch_size * seq_len

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    peak_mem = torch.cuda.max_memory_allocated() / 1024**2

    tok_per_sec = tokens_processed / elapsed
    ms_per_step = (elapsed / num_steps) * 1000

    return losses, elapsed, tok_per_sec, ms_per_step, peak_mem


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--compile", action="store_true", help="Use torch.compile")
    args = parser.parse_args()

    print("=" * 70)
    print("Integration Benchmark: Fused Triton Kernels in GPT Model")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Steps: {args.steps}, Seq: {args.seq_len}, Batch: {args.batch_size}")
    print(f"Compile: {args.compile}")
    print("=" * 70)

    h = Hyperparameters()
    h.train_seq_len = args.seq_len
    vocab_size = h.vocab_size

    # --- Baseline (unfused) ---
    print("\n--- Baseline (unfused) ---")
    torch.manual_seed(42)
    model_base = GPT(h).to(DEVICE).bfloat16()
    restore_fp32_params(model_base)

    # Save initial state for fair comparison
    init_state = copy.deepcopy(model_base.state_dict())

    base_losses, base_time, base_tps, base_ms, base_mem = run_training_steps(
        model_base, args.steps, args.seq_len, args.batch_size, vocab_size, DEVICE,
        use_compile=args.compile,
    )
    print(f"  Time: {base_time:.2f}s ({base_ms:.1f} ms/step)")
    print(f"  Tok/s: {base_tps:,.0f}")
    print(f"  Peak mem: {base_mem:.0f} MiB")
    print(f"  Final loss: {base_losses[-1]:.4f}")

    del model_base
    torch.cuda.empty_cache()

    # --- Level 1 (fused activation only) ---
    print("\n--- Level 1 (fused leaky05² activation) ---")
    torch.manual_seed(42)
    model_l1 = GPT(h).to(DEVICE).bfloat16()
    restore_fp32_params(model_l1)
    model_l1.load_state_dict(init_state)
    patch_model_level1(model_l1)

    l1_losses, l1_time, l1_tps, l1_ms, l1_mem = run_training_steps(
        model_l1, args.steps, args.seq_len, args.batch_size, vocab_size, DEVICE,
        use_compile=args.compile,
    )
    print(f"  Time: {l1_time:.2f}s ({l1_ms:.1f} ms/step)")
    print(f"  Tok/s: {l1_tps:,.0f}")
    print(f"  Peak mem: {l1_mem:.0f} MiB")
    print(f"  Final loss: {l1_losses[-1]:.4f}")

    del model_l1
    torch.cuda.empty_cache()

    # --- Level 2 (fused norm + activation + residual) ---
    # Note: Level 2 patches Block.forward, so we need to restore it after
    original_block_forward = Block.forward

    print("\n--- Level 2 (fused norm + activation + residual add) ---")
    torch.manual_seed(42)
    model_l2 = GPT(h).to(DEVICE).bfloat16()
    restore_fp32_params(model_l2)
    model_l2.load_state_dict(init_state)
    patch_model_level2(model_l2)

    l2_losses, l2_time, l2_tps, l2_ms, l2_mem = run_training_steps(
        model_l2, args.steps, args.seq_len, args.batch_size, vocab_size, DEVICE,
        use_compile=args.compile,
    )
    print(f"  Time: {l2_time:.2f}s ({l2_ms:.1f} ms/step)")
    print(f"  Tok/s: {l2_tps:,.0f}")
    print(f"  Peak mem: {l2_mem:.0f} MiB")
    print(f"  Final loss: {l2_losses[-1]:.4f}")

    # Restore original Block.forward
    Block.forward = original_block_forward

    # --- Summary ---
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Config':<35} {'ms/step':<10} {'tok/s':<12} {'Speedup':<10} {'Peak MiB':<10}")
    print("-" * 77)
    print(f"{'Baseline (unfused)':<35} {base_ms:<10.1f} {base_tps:<12,.0f} {'1.00x':<10} {base_mem:<10.0f}")
    print(f"{'Level 1 (fused activation)':<35} {l1_ms:<10.1f} {l1_tps:<12,.0f} {base_ms/l1_ms:<10.2f}x {l1_mem:<10.0f}")
    print(f"{'Level 2 (fused norm+act+resid)':<35} {l2_ms:<10.1f} {l2_tps:<12,.0f} {base_ms/l2_ms:<10.2f}x {l2_mem:<10.0f}")

    # Loss curve comparison
    print(f"\n{'Step':<8} {'Baseline':<12} {'Level 1':<12} {'Level 2':<12} {'L1 diff':<12} {'L2 diff':<12}")
    print("-" * 68)
    for i in [0, 4, 9, 19, 29, 39, 49]:
        if i < len(base_losses):
            l1_diff = l1_losses[i] - base_losses[i]
            l2_diff = l2_losses[i] - base_losses[i]
            print(f"{i+1:<8} {base_losses[i]:<12.4f} {l1_losses[i]:<12.4f} {l2_losses[i]:<12.4f} {l1_diff:<+12.4f} {l2_diff:<+12.4f}")

    max_l1_diff = max(abs(l1 - b) for l1, b in zip(l1_losses, base_losses))
    max_l2_diff = max(abs(l2 - b) for l2, b in zip(l2_losses, base_losses))
    print(f"\nMax loss diff: L1={max_l1_diff:.4f}, L2={max_l2_diff:.4f}")

    if max_l1_diff < 0.1 and max_l2_diff < 0.5:
        print("Loss curves match within tolerance — kernels are training-safe.")
    else:
        print("WARNING: Loss curves diverge significantly — investigate!")


if __name__ == "__main__":
    main()
