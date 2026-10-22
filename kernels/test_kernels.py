#!/usr/bin/env python3
"""
Test harness for fused Triton kernels.

For each fusion level:
  1. Forward correctness vs PyTorch reference
  2. Backward correctness vs PyTorch reference
  3. Timing (100 iters) → speedup report

Usage:
    python kernels/test_kernels.py
"""

import sys
import time
import torch
import torch.nn.functional as F
from torch import Tensor

# Add parent to path so we can import kernels
sys.path.insert(0, ".")
from kernels.fused_mlp import (
    leaky05_sq,
    _leaky05_sq_fwd,
    _leaky05_sq_bwd,
    rms_norm_scale_autograd,
    _rms_norm_scale_fwd,
    _rms_norm_scale_bwd,
    residual_scale_add,
    _residual_scale_add_fwd,
)

DEVICE = "cuda"
DTYPE = torch.bfloat16

# Model dimensions
B_T = 384 * 2048  # flattened batch*seq for full training batch (786432)
# Use smaller size for quick testing, full size for benchmark
B_T_TEST = 2048 * 8  # 16384 — enough to test correctness
B_T_BENCH = B_T  # full training batch for benchmarking
DIM = 512
HIDDEN = 2048
ATOL = 1e-2  # bf16 tolerance (1e-2 is generous for bf16 rounding)
RTOL = 1e-2
TIMING_ITERS = 100
WARMUP_ITERS = 10


def check_close(name: str, ref: Tensor, test: Tensor, atol=ATOL, rtol=RTOL) -> bool:
    """Check if tensors are close. Returns True if pass."""
    if ref.shape != test.shape:
        print(f"  FAIL {name}: shape mismatch {ref.shape} vs {test.shape}")
        return False
    ref_f = ref.float()
    test_f = test.float()
    max_diff = (ref_f - test_f).abs().max().item()
    rel_diff = ((ref_f - test_f).abs() / (ref_f.abs() + 1e-8)).max().item()
    passed = torch.allclose(ref_f, test_f, atol=atol, rtol=rtol)
    status = "PASS" if passed else "FAIL"
    print(f"  {status} {name}: max_abs_diff={max_diff:.6f}, max_rel_diff={rel_diff:.6f}")
    return passed


def time_fn(fn, *args, iters=TIMING_ITERS, warmup=WARMUP_ITERS):
    """Time a function with CUDA synchronization."""
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn(*args)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return elapsed / iters


# =============================================================================
# Level 1: Fused leaky_relu(x, 0.5).square()
# =============================================================================

def ref_leaky05_sq(x: Tensor) -> Tensor:
    return F.leaky_relu(x, negative_slope=0.5).square()


def test_level1_fwd():
    print("\n=== Level 1: leaky05² Forward ===")
    x = torch.randn(B_T_TEST, HIDDEN, device=DEVICE, dtype=DTYPE)
    ref = ref_leaky05_sq(x)
    out = _leaky05_sq_fwd(x)
    return check_close("leaky05_sq_fwd", ref, out)


def test_level1_bwd():
    print("\n=== Level 1: leaky05² Backward ===")
    x = torch.randn(B_T_TEST, HIDDEN, device=DEVICE, dtype=DTYPE, requires_grad=True)
    x_ref = x.detach().clone().requires_grad_(True)
    x_test = x.detach().clone().requires_grad_(True)

    # Reference backward
    ref_out = ref_leaky05_sq(x_ref)
    dy = torch.randn_like(ref_out)
    ref_out.backward(dy)

    # Fused backward
    test_out = torch.ops.fused_mlp.leaky05_sq(x_test)
    test_out.backward(dy)

    ok = check_close("leaky05_sq_bwd dx", x_ref.grad, x_test.grad)
    return ok


def bench_level1():
    print("\n=== Level 1: leaky05² Benchmark ===")
    x = torch.randn(B_T_BENCH, HIDDEN, device=DEVICE, dtype=DTYPE)

    t_ref = time_fn(ref_leaky05_sq, x)
    t_fused = time_fn(_leaky05_sq_fwd, x)
    speedup = t_ref / t_fused
    print(f"  PyTorch:  {t_ref*1000:.3f} ms")
    print(f"  Fused:    {t_fused*1000:.3f} ms")
    print(f"  Speedup:  {speedup:.2f}x")

    # Backward benchmark
    x_ref = torch.randn(B_T_BENCH, HIDDEN, device=DEVICE, dtype=DTYPE, requires_grad=True)
    x_fused = x_ref.detach().clone().requires_grad_(True)
    dy = torch.randn(B_T_BENCH, HIDDEN, device=DEVICE, dtype=DTYPE)

    def ref_bwd():
        out = ref_leaky05_sq(x_ref)
        out.backward(dy, retain_graph=True)

    def fused_bwd():
        out = torch.ops.fused_mlp.leaky05_sq(x_fused)
        out.backward(dy, retain_graph=True)

    t_ref_bwd = time_fn(ref_bwd)
    t_fused_bwd = time_fn(fused_bwd)
    speedup_bwd = t_ref_bwd / t_fused_bwd
    print(f"  PyTorch bwd: {t_ref_bwd*1000:.3f} ms")
    print(f"  Fused bwd:   {t_fused_bwd*1000:.3f} ms")
    print(f"  Speedup bwd: {speedup_bwd:.2f}x")
    return t_ref, t_fused, t_ref_bwd, t_fused_bwd


# =============================================================================
# Level 2A: Fused RMS norm * scale_factor
# =============================================================================

def ref_rms_norm_scale(x: Tensor, scale_factor: float) -> Tensor:
    return F.rms_norm(x, (x.size(-1),)) * scale_factor


def test_level2a_fwd():
    print("\n=== Level 2A: rms_norm*scale Forward ===")
    x = torch.randn(B_T_TEST, DIM, device=DEVICE, dtype=DTYPE)
    scale = 1.0 / (5.0 ** 0.5)  # ln_scale_factor for layer 4
    ref = ref_rms_norm_scale(x, scale)
    out, _ = _rms_norm_scale_fwd(x, scale)
    return check_close("rms_norm_scale_fwd", ref, out)


def test_level2a_bwd():
    print("\n=== Level 2A: rms_norm*scale Backward ===")
    scale = 1.0 / (5.0 ** 0.5)
    x_ref = torch.randn(B_T_TEST, DIM, device=DEVICE, dtype=DTYPE, requires_grad=True)
    x_test = x_ref.detach().clone().requires_grad_(True)

    ref_out = ref_rms_norm_scale(x_ref, scale)
    dy = torch.randn_like(ref_out)
    ref_out.backward(dy)

    test_out = rms_norm_scale_autograd(x_test, scale)
    test_out.backward(dy)

    return check_close("rms_norm_scale_bwd dx", x_ref.grad, x_test.grad)


def bench_level2a():
    print("\n=== Level 2A: rms_norm*scale Benchmark ===")
    x = torch.randn(B_T_BENCH, DIM, device=DEVICE, dtype=DTYPE)
    scale = 1.0 / (5.0 ** 0.5)

    t_ref = time_fn(ref_rms_norm_scale, x, scale)
    t_fused = time_fn(lambda: _rms_norm_scale_fwd(x, scale))
    speedup = t_ref / t_fused
    print(f"  PyTorch:  {t_ref*1000:.3f} ms")
    print(f"  Fused:    {t_fused*1000:.3f} ms")
    print(f"  Speedup:  {speedup:.2f}x")
    return t_ref, t_fused


# =============================================================================
# Level 2C: Fused residual add with per-dim scale
# =============================================================================

def ref_residual_scale_add(x: Tensor, proj_out: Tensor, scale: Tensor) -> Tensor:
    # Compute in f32 to match Triton kernel (which loads bf16→f32, computes, stores bf16)
    return (x.float() + scale * proj_out.float()).to(x.dtype)


def test_level2c_fwd():
    print("\n=== Level 2C: residual_scale_add Forward ===")
    x = torch.randn(B_T_TEST, DIM, device=DEVICE, dtype=DTYPE)
    p = torch.randn(B_T_TEST, DIM, device=DEVICE, dtype=DTYPE)
    s = torch.randn(DIM, device=DEVICE, dtype=torch.float32)
    ref = ref_residual_scale_add(x, p, s)
    out = _residual_scale_add_fwd(x, p, s)
    return check_close("residual_scale_add_fwd", ref, out, atol=1e-3)


def test_level2c_bwd():
    print("\n=== Level 2C: residual_scale_add Backward ===")
    x_ref = torch.randn(B_T_TEST, DIM, device=DEVICE, dtype=DTYPE, requires_grad=True)
    p_ref = torch.randn(B_T_TEST, DIM, device=DEVICE, dtype=DTYPE, requires_grad=True)
    s_ref = torch.randn(DIM, device=DEVICE, dtype=torch.float32, requires_grad=True)

    x_test = x_ref.detach().clone().requires_grad_(True)
    p_test = p_ref.detach().clone().requires_grad_(True)
    s_test = s_ref.detach().clone().requires_grad_(True)

    # Reference: use f32-matched version for dx/dp, but PyTorch autograd for ds
    ref_out = ref_residual_scale_add(x_ref, p_ref, s_ref)
    dy = torch.randn_like(ref_out)
    ref_out.backward(dy)

    # Fused
    test_out = torch.ops.fused_mlp.residual_scale_add(x_test, p_test, s_test)
    test_out.backward(dy)

    ok1 = check_close("residual_scale_add_bwd dx", x_ref.grad, x_test.grad)
    ok2 = check_close("residual_scale_add_bwd dp", p_ref.grad, p_test.grad)
    # ds sums over B_T_TEST rows of bf16→f32 products; large accumulation has some variance
    ok3 = check_close("residual_scale_add_bwd ds", s_ref.grad, s_test.grad, atol=2.0, rtol=0.1)
    return ok1 and ok2 and ok3


def bench_level2c():
    print("\n=== Level 2C: residual_scale_add Benchmark ===")
    x = torch.randn(B_T_BENCH, DIM, device=DEVICE, dtype=DTYPE)
    p = torch.randn(B_T_BENCH, DIM, device=DEVICE, dtype=DTYPE)
    s = torch.randn(DIM, device=DEVICE, dtype=torch.float32)

    t_ref = time_fn(ref_residual_scale_add, x, p, s)
    t_fused = time_fn(_residual_scale_add_fwd, x, p, s)
    speedup = t_ref / t_fused
    print(f"  PyTorch:  {t_ref*1000:.3f} ms")
    print(f"  Fused:    {t_fused*1000:.3f} ms")
    print(f"  Speedup:  {speedup:.2f}x")
    return t_ref, t_fused


# =============================================================================
# Level 2 End-to-End: Full MLP path fused vs unfused
# =============================================================================

def test_level2_e2e():
    """Test the full MLP forward: norm*scale -> fc -> leaky05² -> proj -> residual add."""
    print("\n=== Level 2 E2E: Full fused MLP path ===")
    torch.manual_seed(42)

    # Create MLP components
    fc_weight = torch.randn(HIDDEN, DIM, device=DEVICE, dtype=DTYPE) * 0.01
    proj_weight = torch.randn(DIM, HIDDEN, device=DEVICE, dtype=DTYPE) * 0.01
    mlp_scale = torch.ones(DIM, device=DEVICE, dtype=torch.float32)
    ln_scale_factor = 1.0 / (5.0 ** 0.5)

    x = torch.randn(1024, DIM, device=DEVICE, dtype=DTYPE, requires_grad=True)
    x_ref = x.detach().clone().requires_grad_(True)

    # Reference path
    normed_ref = F.rms_norm(x_ref, (DIM,)) * ln_scale_factor
    h_ref = F.linear(normed_ref, fc_weight)
    h_ref = F.leaky_relu(h_ref, negative_slope=0.5).square()
    out_ref = F.linear(h_ref, proj_weight)
    y_ref = x_ref + mlp_scale.to(DTYPE) * out_ref

    # Fused path
    normed = rms_norm_scale_autograd(x, ln_scale_factor)
    h = F.linear(normed, fc_weight)
    h = torch.ops.fused_mlp.leaky05_sq(h)
    out = F.linear(h, proj_weight)
    y = torch.ops.fused_mlp.residual_scale_add(x, out, mlp_scale)

    ok_fwd = check_close("level2_e2e_fwd", y_ref, y)

    # Backward
    dy = torch.randn_like(y_ref)
    y_ref.backward(dy)
    y.backward(dy)
    ok_bwd = check_close("level2_e2e_bwd dx", x_ref.grad, x.grad)

    return ok_fwd and ok_bwd


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 60)
    print("Fused Kernel Test Harness — Parameter Golf LM")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Test batch: {B_T_TEST}, Bench batch: {B_T_BENCH}")
    print(f"DIM={DIM}, HIDDEN={HIDDEN}, dtype=bf16")
    print("=" * 60)

    results = {}

    # --- Level 1 ---
    results["L1_fwd"] = test_level1_fwd()
    results["L1_bwd"] = test_level1_bwd()

    # --- Level 2A ---
    results["L2A_fwd"] = test_level2a_fwd()
    results["L2A_bwd"] = test_level2a_bwd()

    # --- Level 2C ---
    results["L2C_fwd"] = test_level2c_fwd()
    results["L2C_bwd"] = test_level2c_bwd()

    # --- Level 2 E2E ---
    results["L2_e2e"] = test_level2_e2e()

    # --- Benchmarks ---
    print("\n" + "=" * 60)
    print("BENCHMARKS (full training batch size)")
    print("=" * 60)

    l1_times = bench_level1()
    l2a_times = bench_level2a()
    l2c_times = bench_level2c()

    # --- Summary ---
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Test':<30} {'Status':<8}")
    print("-" * 38)
    all_pass = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"{name:<30} {status:<8}")

    print("-" * 38)
    print(f"\n{'Kernel':<25} {'PyTorch ms':<12} {'Fused ms':<12} {'Speedup':<8}")
    print("-" * 57)
    if l1_times:
        t_ref, t_fused, t_ref_bwd, t_fused_bwd = l1_times
        print(f"{'leaky05² fwd':<25} {t_ref*1000:<12.3f} {t_fused*1000:<12.3f} {t_ref/t_fused:<8.2f}x")
        print(f"{'leaky05² fwd+bwd':<25} {t_ref_bwd*1000:<12.3f} {t_fused_bwd*1000:<12.3f} {t_ref_bwd/t_fused_bwd:<8.2f}x")
    if l2a_times:
        t_ref, t_fused = l2a_times
        print(f"{'rms_norm*scale fwd':<25} {t_ref*1000:<12.3f} {t_fused*1000:<12.3f} {t_ref/t_fused:<8.2f}x")
    if l2c_times:
        t_ref, t_fused = l2c_times
        print(f"{'residual_scale_add fwd':<25} {t_ref*1000:<12.3f} {t_fused*1000:<12.3f} {t_ref/t_fused:<8.2f}x")

    print()
    if all_pass:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
