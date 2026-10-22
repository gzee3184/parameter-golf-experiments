"""
Triton fused kernels for the MLP block in Parameter Golf LM.

Level 1: Fused activation  — leaky_relu(x, 0.5).square()  (fwd + bwd)
Level 2: Fused MLP pre/post:
    Kernel A: rms_norm(x) * scale_factor  (fwd + bwd)
    Kernel B: same as Level 1
    Kernel C: x + mlp_scale * proj_out    (fwd + bwd)

All kernels:
- bf16 in/out, f32 accumulation where needed
- Portable Triton (no TMA, no make_block_ptr)
- Compatible with torch.compile(fullgraph=True) via torch.library.custom_op
"""

import torch
import triton
import triton.language as tl
from torch import Tensor

LIBRARY = torch.library.Library("fused_mlp", "DEF")

# =============================================================================
# Level 1: Fused leaky_relu(x, 0.5).square()  —  forward + backward
# =============================================================================

@triton.jit
def _leaky05_sq_fwd_kernel(
    X_ptr, Y_ptr,
    N,  # number of elements
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X_ptr + offs, mask=mask).to(tl.float32)
    # leaky_relu(x, 0.5) = x if x >= 0 else 0.5*x
    lr = tl.where(x >= 0.0, x, x * 0.5)
    y = lr * lr
    tl.store(Y_ptr + offs, y.to(tl.bfloat16), mask=mask)


@triton.jit
def _leaky05_sq_bwd_kernel(
    X_ptr, DY_ptr, DX_ptr,
    N,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X_ptr + offs, mask=mask).to(tl.float32)
    dy = tl.load(DY_ptr + offs, mask=mask).to(tl.float32)
    # Forward: y = leaky_relu(x, 0.5)^2
    # Let lr = leaky_relu(x, 0.5)
    # dy/dx = 2*lr * d(lr)/dx
    # d(lr)/dx = 1 if x >= 0 else 0.5
    lr = tl.where(x >= 0.0, x, x * 0.5)
    dlr_dx = tl.where(x >= 0.0, 1.0, 0.5)
    dx = dy * 2.0 * lr * dlr_dx
    tl.store(DX_ptr + offs, dx.to(tl.bfloat16), mask=mask)


def _leaky05_sq_fwd(x: Tensor) -> Tensor:
    assert x.is_contiguous() and x.dtype == torch.bfloat16
    y = torch.empty_like(x)
    N = x.numel()
    BLOCK = 1024
    grid = ((N + BLOCK - 1) // BLOCK,)
    _leaky05_sq_fwd_kernel[grid](x, y, N, BLOCK=BLOCK)
    return y


def _leaky05_sq_bwd_impl(x: Tensor, dy: Tensor) -> Tensor:
    assert x.is_contiguous() and dy.is_contiguous()
    dx = torch.empty_like(x)
    N = x.numel()
    BLOCK = 1024
    grid = ((N + BLOCK - 1) // BLOCK,)
    _leaky05_sq_bwd_kernel[grid](x, dy, dx, N, BLOCK=BLOCK)
    return dx


# --- torch.library.custom_op wrappers for fullgraph compatibility ---

@torch.library.custom_op("fused_mlp::leaky05_sq", mutates_args=())
def leaky05_sq(x: Tensor) -> Tensor:
    return _leaky05_sq_fwd(x)


@leaky05_sq.register_fake
def _leaky05_sq_fake(x: Tensor) -> Tensor:
    return torch.empty_like(x)


# Backward also needs to be a custom_op for torch.compile(fullgraph=True)
@torch.library.custom_op("fused_mlp::leaky05_sq_bwd", mutates_args=())
def leaky05_sq_bwd(x: Tensor, dy: Tensor) -> Tensor:
    return _leaky05_sq_bwd_impl(x, dy)


@leaky05_sq_bwd.register_fake
def _leaky05_sq_bwd_fake(x: Tensor, dy: Tensor) -> Tensor:
    return torch.empty_like(x)


def _leaky05_sq_setup_context(ctx, inputs, output):
    (x,) = inputs
    ctx.save_for_backward(x)


def _leaky05_sq_backward(ctx, dy):
    (x,) = ctx.saved_tensors
    return torch.ops.fused_mlp.leaky05_sq_bwd(x, dy)


leaky05_sq.register_autograd(
    _leaky05_sq_backward, setup_context=_leaky05_sq_setup_context
)


# =============================================================================
# Level 2A: Fused RMS norm * scale_factor
# rms_norm(x, dim=-1) * scale_factor
# Input: [*, D] bf16, scale_factor: float scalar
# Output: [*, D] bf16
# =============================================================================

@triton.jit
def _rms_norm_scale_fwd_kernel(
    X_ptr, Y_ptr, Rrms_ptr,
    scale_factor,
    stride_row, D,
    EPS: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_D)
    mask = offs < D
    x = tl.load(X_ptr + row * stride_row + offs, mask=mask, other=0.0).to(tl.float32)
    # RMS norm
    ms = tl.sum(x * x, axis=0) / D
    rrms = 1.0 / tl.sqrt(ms + EPS)
    y = x * rrms * scale_factor
    tl.store(Y_ptr + row * stride_row + offs, y.to(tl.bfloat16), mask=mask)
    # Save 1/rms for backward
    tl.store(Rrms_ptr + row, rrms.to(tl.float32))


@triton.jit
def _rms_norm_scale_bwd_kernel(
    DY_ptr, X_ptr, Rrms_ptr, DX_ptr,
    scale_factor,
    stride_row, D,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_D)
    mask = offs < D

    dy = tl.load(DY_ptr + row * stride_row + offs, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(X_ptr + row * stride_row + offs, mask=mask, other=0.0).to(tl.float32)
    rrms = tl.load(Rrms_ptr + row).to(tl.float32)

    # Forward: y = x * rrms * scale_factor
    # dy/dx = scale_factor * (rrms * I - rrms^3 * x * x^T / D)
    # Simplified: dx = scale_factor * rrms * (dy - x * (sum(dy * x) / D) * rrms^2)
    x_hat = x * rrms
    dot = tl.sum(dy * x_hat, axis=0) / D
    dx = scale_factor * rrms * (dy - x_hat * dot)
    tl.store(DX_ptr + row * stride_row + offs, dx.to(tl.bfloat16), mask=mask)


def _rms_norm_scale_fwd(x: Tensor, scale_factor: float) -> tuple[Tensor, Tensor]:
    assert x.is_contiguous() and x.dtype == torch.bfloat16
    shape = x.shape
    D = shape[-1]
    x_2d = x.reshape(-1, D)
    rows = x_2d.shape[0]
    y = torch.empty_like(x_2d)
    rrms = torch.empty(rows, dtype=torch.float32, device=x.device)
    BLOCK_D = triton.next_power_of_2(D)
    _rms_norm_scale_fwd_kernel[(rows,)](
        x_2d, y, rrms, scale_factor, D, D, EPS=1e-6, BLOCK_D=BLOCK_D,
    )
    return y.reshape(shape), rrms


def _rms_norm_scale_bwd(dy: Tensor, x: Tensor, rrms: Tensor, scale_factor: float) -> Tensor:
    shape = dy.shape
    D = shape[-1]
    dy_2d = dy.reshape(-1, D)
    x_2d = x.reshape(-1, D)
    rows = dy_2d.shape[0]
    dx = torch.empty_like(dy_2d)
    BLOCK_D = triton.next_power_of_2(D)
    _rms_norm_scale_bwd_kernel[(rows,)](
        dy_2d, x_2d, rrms, dx, scale_factor, D, D, BLOCK_D=BLOCK_D,
    )
    return dx.reshape(shape)


@torch.library.custom_op("fused_mlp::rms_norm_scale", mutates_args=())
def rms_norm_scale(x: Tensor, scale_factor: float) -> Tensor:
    y, _ = _rms_norm_scale_fwd(x, scale_factor)
    return y


@rms_norm_scale.register_fake
def _rms_norm_scale_fake(x: Tensor, scale_factor: float) -> Tensor:
    return torch.empty_like(x)


# For autograd we need to save rrms, so use a wrapper that returns both
# but the custom_op only exposes the tensor output.
# We use a manual autograd function here since we need the intermediate rrms.

class _RMSNormScaleAutograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, scale_factor: float) -> Tensor:
        y, rrms = _rms_norm_scale_fwd(x, scale_factor)
        ctx.save_for_backward(x, rrms)
        ctx.scale_factor = scale_factor
        return y

    @staticmethod
    def backward(ctx, dy: Tensor):
        x, rrms = ctx.saved_tensors
        dx = _rms_norm_scale_bwd(dy.contiguous(), x, rrms, ctx.scale_factor)
        return dx, None


def rms_norm_scale_autograd(x: Tensor, scale_factor: float) -> Tensor:
    """Use this for training (has backward). The custom_op version is for compile tracing."""
    return _RMSNormScaleAutograd.apply(x, scale_factor)


# =============================================================================
# Level 2C: Fused residual add with per-dim scale
# y = x + scale * proj_out
# Input: x [*, D] bf16, proj_out [*, D] bf16, scale [D] f32
# Output: [*, D] bf16
# =============================================================================

@triton.jit
def _residual_scale_add_fwd_kernel(
    X_ptr, P_ptr, S_ptr, Y_ptr,
    N, D,
    stride_row,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_D)
    mask = offs < D
    x = tl.load(X_ptr + row * stride_row + offs, mask=mask).to(tl.float32)
    p = tl.load(P_ptr + row * stride_row + offs, mask=mask).to(tl.float32)
    s = tl.load(S_ptr + offs, mask=mask).to(tl.float32)
    y = x + s * p
    tl.store(Y_ptr + row * stride_row + offs, y.to(tl.bfloat16), mask=mask)


@triton.jit
def _residual_scale_add_bwd_kernel(
    DY_ptr, S_ptr, DX_ptr, DP_ptr, DS_ptr,
    N, D,
    stride_row,
    BLOCK_D: tl.constexpr,
):
    """Backward for y = x + s * p.
    dx = dy, dp = dy * s, ds = sum_over_rows(dy * p) — ds accumulated via atomics."""
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_D)
    mask = offs < D
    dy = tl.load(DY_ptr + row * stride_row + offs, mask=mask).to(tl.float32)
    s = tl.load(S_ptr + offs, mask=mask).to(tl.float32)
    # dx = dy (just copy)
    tl.store(DX_ptr + row * stride_row + offs, dy.to(tl.bfloat16), mask=mask)
    # dp = dy * s
    dp = dy * s
    tl.store(DP_ptr + row * stride_row + offs, dp.to(tl.bfloat16), mask=mask)
    # ds partial: we need to load p and accumulate
    # But for simplicity, we'll handle ds in the Python wrapper using torch


def _residual_scale_add_fwd(x: Tensor, proj_out: Tensor, scale: Tensor) -> Tensor:
    assert x.is_contiguous() and proj_out.is_contiguous()
    shape = x.shape
    D = shape[-1]
    x_2d = x.reshape(-1, D)
    p_2d = proj_out.reshape(-1, D)
    rows = x_2d.shape[0]
    y = torch.empty_like(x_2d)
    BLOCK_D = triton.next_power_of_2(D)
    _residual_scale_add_fwd_kernel[(rows,)](
        x_2d, p_2d, scale, y, rows, D, D, BLOCK_D=BLOCK_D,
    )
    return y.reshape(shape)


@torch.library.custom_op("fused_mlp::residual_scale_add", mutates_args=())
def residual_scale_add(x: Tensor, proj_out: Tensor, scale: Tensor) -> Tensor:
    return _residual_scale_add_fwd(x, proj_out, scale)


@residual_scale_add.register_fake
def _residual_scale_add_fake(x: Tensor, proj_out: Tensor, scale: Tensor) -> Tensor:
    return torch.empty_like(x)


def _residual_scale_add_setup_context(ctx, inputs, output):
    x, proj_out, scale = inputs
    ctx.save_for_backward(proj_out, scale)


def _residual_scale_add_backward(ctx, dy):
    proj_out, scale = ctx.saved_tensors
    dy = dy.contiguous()
    # dx = dy
    dx = dy
    # dp = dy * scale[None, ...]
    scale_bf16 = scale.to(dy.dtype)
    dp = dy * scale_bf16
    # ds = sum over batch dims of (dy * proj_out)
    shape = dy.shape
    D = shape[-1]
    dy_2d = dy.reshape(-1, D).float()
    p_2d = proj_out.reshape(-1, D).float()
    ds = (dy_2d * p_2d).sum(dim=0)
    return dx, dp, ds


residual_scale_add.register_autograd(
    _residual_scale_add_backward, setup_context=_residual_scale_add_setup_context
)


# =============================================================================
# Convenience: FusedMLP drop-in replacement
# =============================================================================

class FusedMLP(torch.nn.Module):
    """Drop-in replacement for the original MLP using fused Triton kernels.

    Level 1: fuses leaky_relu(x, 0.5).square()
    Level 2: also fuses rms_norm*scale pre-processing and residual add post-processing
    """
    def __init__(self, original_mlp, level: int = 1):
        super().__init__()
        self.fc = original_mlp.fc
        self.proj = original_mlp.proj
        self.level = level

    def forward(self, x: Tensor) -> Tensor:
        h = self.fc(x)
        h = torch.ops.fused_mlp.leaky05_sq(h)
        return self.proj(h)


def fused_mlp_forward_level2(
    x: Tensor,
    mlp_norm_fn,
    ln_scale_factor: float,
    fc: torch.nn.Module,
    proj: torch.nn.Module,
    mlp_scale: Tensor,
    x_residual: Tensor,
) -> Tensor:
    """Level 2 fused MLP: norm*scale -> fc -> leaky05² -> proj -> residual add.

    Args:
        x: input to norm (before mlp_norm)
        mlp_norm_fn: not used (norm is fused)
        ln_scale_factor: scalar multiplied after rms_norm
        fc: the fc linear layer
        proj: the proj linear layer
        mlp_scale: per-dim scale [D] for residual add
        x_residual: the tensor to add residual to (may be same as x)
    """
    # Kernel A: rms_norm * scale_factor
    normed = rms_norm_scale_autograd(x, ln_scale_factor)
    # fc matmul (cuBLAS)
    h = fc(normed)
    # Kernel B: leaky05²
    h = torch.ops.fused_mlp.leaky05_sq(h)
    # proj matmul (cuBLAS)
    out = proj(h)
    # Kernel C: residual add with scale
    return torch.ops.fused_mlp.residual_scale_add(x_residual, out, mlp_scale)
