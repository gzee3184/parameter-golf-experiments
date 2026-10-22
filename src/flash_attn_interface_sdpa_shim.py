"""
Compatibility shim: wraps PyTorch SDPA to match the FlashAttention 3 API.

FA3 API:  flash_attn_func(q, k, v, causal=True)
  - q: (batch, seqlen, nheads, headdim)
  - k: (batch, seqlen, nkv_heads, headdim)
  - v: (batch, seqlen, nkv_heads, headdim)
  - returns: (batch, seqlen, nheads, headdim)

Used on GPUs where FA3/FA4 kernels are unavailable (e.g. consumer Blackwell sm_120).
"""

import torch
import torch.nn.functional as F


def flash_attn_func(q, k, v, causal=False, **kwargs):
    # FA3 layout: (B, T, H, D) -> SDPA layout: (B, H, T, D)
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    gqa = q.shape[1] != k.shape[1]
    out = F.scaled_dot_product_attention(
        q, k, v, is_causal=causal, enable_gqa=gqa
    )
    return out.transpose(1, 2)
