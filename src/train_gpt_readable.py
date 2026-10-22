from __future__ import annotations
import copy
import glob
import io
import lzma
try:
    import brotli
    _HAS_BROTLI = True
except ImportError:
    _HAS_BROTLI = False
import math
import os
import random
import subprocess
import sys
import time
import uuid
from pathlib import Path
import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP
from flash_attn_interface import flash_attn_func as flash_attn_3_func
torch._dynamo.config.cache_size_limit = 128
if hasattr(torch._inductor.config, 'mix_order_reduction'):
    torch._inductor.config.mix_order_reduction = False  # PyTorch 2.11 bug: over-fused RMSNorm backward spills registers
class Hyperparameters:
    data_path = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    train_files = os.path.join(data_path, "fineweb_train_*.bin")
    val_files = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
    run_id = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed = int(os.environ.get("SEED", 1337))
    val_batch_size = int(os.environ.get("VAL_BATCH_SIZE", 524_288))
    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 4000))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 500))
    iterations = int(os.environ.get("ITERATIONS", 20000))
    warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 4000))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 20))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 786_432))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 2048))
    eval_seq_len = int(os.environ.get("EVAL_SEQ_LEN", 2048))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))
    qk_gain_init = float(os.environ.get("QK_GAIN_INIT", 1.5))
    vocab_size = int(os.environ.get("VOCAB_SIZE", 1024))
    num_layers = int(os.environ.get("NUM_LAYERS", 11))
    num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 4))
    model_dim = int(os.environ.get("MODEL_DIM", 512))
    num_heads = int(os.environ.get("NUM_HEADS", 8))
    mlp_mult = float(os.environ.get("MLP_MULT", 3.0))
    tie_embeddings = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    rope_base = float(os.environ.get("ROPE_BASE", 10000.0))
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 30.0))
    embed_lr = float(os.environ.get("EMBED_LR", 0.6))
    head_lr = float(os.environ.get("HEAD_LR", 0.008))
    tied_embed_lr = float(os.environ.get("TIED_EMBED_LR", 0.035))
    tied_embed_init_std = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))
    matrix_lr = float(os.environ.get("MATRIX_LR", 0.025))
    scalar_lr = float(os.environ.get("SCALAR_LR", 0.025))
    muon_momentum = float(os.environ.get("MUON_MOMENTUM", 0.99))
    muon_plus = bool(int(os.environ.get("MUON_PLUS", "0")))
    muon_backend_steps = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    muon_momentum_warmup_start = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.92))
    muon_momentum_warmup_steps = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 1500))
    beta1 = float(os.environ.get("BETA1", 0.9))
    beta2 = float(os.environ.get("BETA2", 0.95))
    adam_eps = float(os.environ.get("ADAM_EPS", 1e-8))
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 0.3))
    muon_beta2 = float(os.environ.get("MUON_BETA2", 0.95))
    swa_enabled = bool(int(os.environ.get("SWA_ENABLED", "1")))
    swa_every = int(os.environ.get("SWA_EVERY", 50))
    muon_wd = float(os.environ.get("MUON_WD", 0.04))
    adam_wd = float(os.environ.get("ADAM_WD", 0.04))
    embed_wd = float(os.environ.get("EMBED_WD", -1))  # -1 means use adam_wd
    xsa_last_n = int(os.environ.get("XSA_LAST_N", 11))
    rope_dims = int(os.environ.get("ROPE_DIMS", 16))
    ln_scale = bool(int(os.environ.get("LN_SCALE", "1")))
    quant_bits_mlp = int(os.environ.get("QUANT_BITS_MLP", 6))
    quant_bits_attn = int(os.environ.get("QUANT_BITS_ATTN", 6))
    quant_bits_embed = int(os.environ.get("QUANT_BITS_EMBED", 6))
    compressor = os.environ.get("COMPRESSOR", "brotli")
    byte_shuffle = bool(int(os.environ.get("BYTE_SHUFFLE", "1")))
    sigmoid_skip = bool(int(os.environ.get("SIGMOID_SKIP", "1")))
    coprime_stride = bool(int(os.environ.get("COPRIME_STRIDE", "1")))
    gptq_enabled = bool(int(os.environ.get("GPTQ_ENABLED", "1")))
    gptq_calib_seqs = int(os.environ.get("GPTQ_CALIB_SEQS", 64))
    gptq_calib_len = int(os.environ.get("GPTQ_CALIB_LEN", 2048))
    gptq_temperature = float(os.environ.get("GPTQ_TEMPERATURE", 0.8))
    recur_layers = os.environ.get("RECUR_LAYERS", "4,5")
    recur_start_step = int(os.environ.get("RECUR_START_STEP", 3000))
    recur_num_loops = int(os.environ.get("RECUR_NUM_LOOPS", 2))  # total passes through recur layers (1=no repeat, 2=1 extra, 3=2 extra)
    ve_enabled = bool(int(os.environ.get("VE_ENABLED", "1")))
    ve_dim = int(os.environ.get("VE_DIM", 128))
    ve_layers = os.environ.get("VE_LAYERS", "9,10")
    parallel_start_layer = int(os.environ.get("PARALLEL_START_LAYER", 7))
    repeat_untie_mlp = os.environ.get("REPEAT_UNTIE_MLP", "lora")  # none, down, lora
    repeat_lora_rank = int(os.environ.get("REPEAT_LORA_RANK", "16"))
    disable_layer0_attn = bool(int(os.environ.get("DISABLE_LAYER0_ATTN", "1")))
    repeat_pass_scale = bool(int(os.environ.get("REPEAT_PASS_SCALE", "0")))
    recur_skip_physical = bool(int(os.environ.get("RECUR_SKIP_PHYSICAL", "0")))
    per_layer_lane_merge = bool(int(os.environ.get("PER_LAYER_LANE_MERGE", "0")))
    repeat_attn_lora = bool(int(os.environ.get("REPEAT_ATTN_LORA", "0")))
    repeat_attn_lora_rank = int(os.environ.get("REPEAT_ATTN_LORA_RANK", "8"))
    disable_layer0_mlp = bool(int(os.environ.get("DISABLE_LAYER0_MLP", "0")))
    skip_repeat_attn = bool(int(os.environ.get("SKIP_REPEAT_ATTN", "0")))
    skip_repeat_attn_layers = set(int(x) for x in os.environ.get("SKIP_REPEAT_ATTN_LAYERS", "").split(",") if x.strip()) if os.environ.get("SKIP_REPEAT_ATTN_LAYERS", "") else set()
    parallel_scalar_grad_scale = float(os.environ.get("PARALLEL_SCALAR_GRAD_SCALE", "0"))
    attn_res = bool(int(os.environ.get("ATTN_RES", "0")))
    # Late QAT: inject quantization noise in last portion of training for smoother weights
    late_qat = bool(int(os.environ.get("LATE_QAT", "0")))
    qat_start_frac = float(os.environ.get("QAT_START_FRAC", 0.90))  # fraction of training before QAT starts
    qat_noise_scale = float(os.environ.get("QAT_NOISE_SCALE", 0.10))  # conservative (competitor uses 0.15)
    # SDClip: entropy-optimal clipping via clip = k * std(row), replaces 5-candidate search
    sdclip = bool(int(os.environ.get("SDCLIP", "0")))
    sdclip_k = float(os.environ.get("SDCLIP_K", 12.85))  # optimal for int6 (±31 levels)

def zeropower_via_newtonschulz5(G: Tensor, steps: int = 5, eps: float = 1e-7) -> Tensor:
    a, b, c = (3.4445, -4.7750, 2.0315)
    was_2d = G.ndim == 2
    if was_2d:
        G = G.unsqueeze(0)
    X = G.bfloat16()
    transposed = X.size(-2) > X.size(-1)
    if transposed:
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + eps)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.mT
    if was_2d:
        X = X.squeeze(0)
    return X

class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr: float, momentum: float, backend_steps: int,
                 nesterov: bool = True, weight_decay: float = 0.0,
                 post_norm: bool = False):
        super().__init__(
            params,
            dict(lr=lr, momentum=momentum, backend_steps=backend_steps,
                 nesterov=nesterov, weight_decay=weight_decay),
        )
        self.post_norm = post_norm
        self._built = False

    def _build(self):
        self._distributed = dist.is_available() and dist.is_initialized()
        self._world_size = dist.get_world_size() if self._distributed else 1
        self._rank = dist.get_rank() if self._distributed else 0
        ws = self._world_size

        self._bank_meta = []
        for group in self.param_groups:
            for p in group["params"]:
                B = p.shape[0]
                padded_B = ((B + ws - 1) // ws) * ws
                shard_B = padded_B // ws
                tail = p.shape[1:]
                dev = p.device
                self._bank_meta.append({
                    'p': p,
                    'B': B,
                    'padded_grad': torch.zeros(padded_B, *tail, device=dev, dtype=torch.bfloat16),
                    'shard': torch.zeros(shard_B, *tail, device=dev, dtype=torch.bfloat16),
                    'shard_mom': torch.zeros(shard_B, *tail, device=dev, dtype=torch.bfloat16),
                    'full_update': torch.zeros(padded_B, *tail, device=dev, dtype=torch.bfloat16),
                    'scale': max(1, p.shape[-2] / p.shape[-1]) ** 0.5,
                })
        self._bank_meta.sort(key=lambda m: -m['p'].numel())
        self._built = True

    def launch_reduce_scatters(self):
        if not self._built:
            self._build()
        if not self._distributed:
            return
        self._rs_futures = []
        for m in self._bank_meta:
            p = m['p']
            if p.grad is None:
                self._rs_futures.append(None)
                continue
            pg = m['padded_grad']
            pg[:m['B']].copy_(p.grad.bfloat16())
            if pg.shape[0] > m['B']:
                pg[m['B']:].zero_()
            fut = dist.reduce_scatter_tensor(m['shard'], pg, op=dist.ReduceOp.AVG, async_op=True)
            self._rs_futures.append(fut)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        if not self._built:
            self._build()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            backend_steps = group["backend_steps"]
            nesterov = group["nesterov"]
            wd = group.get("weight_decay", 0.0)

            prev_ag_handle = None
            prev_m = None

            sharded = self._distributed and hasattr(self, '_rs_futures')

            for i, m in enumerate(self._bank_meta):
                p = m['p']
                if p.grad is None:
                    continue

                if prev_ag_handle is not None:
                    prev_ag_handle.wait()
                    pp = prev_m['p']
                    upd = prev_m['full_update'][:prev_m['B']]
                    if wd > 0.0:
                        pp.data.mul_(1.0 - lr * wd)
                    pp.add_(upd.to(dtype=pp.dtype), alpha=-lr * prev_m['scale'])

                if sharded and self._rs_futures[i] is not None:
                    self._rs_futures[i].wait()
                    g = m['shard']
                    buf = m['shard_mom']
                else:
                    g = p.grad.bfloat16()
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]

                buf.mul_(momentum).add_(g)
                if nesterov:
                    update = g.add(buf, alpha=momentum)
                else:
                    update = buf

                row_norms = update.float().norm(dim=-1, keepdim=True).clamp_min(1e-7)
                update = update / row_norms.to(update.dtype)
                update = zeropower_via_newtonschulz5(update, steps=backend_steps)
                if self.post_norm:
                    target = (min(update.shape[-2], update.shape[-1])) ** 0.5
                    update = update * (target / (update.norm(dim=(-2, -1), keepdim=True) + 1e-7))

                if sharded:
                    prev_ag_handle = dist.all_gather_into_tensor(
                        m['full_update'], update, async_op=True)
                    prev_m = m
                else:
                    if wd > 0.0:
                        p.data.mul_(1.0 - lr * wd)
                    p.add_(update.to(dtype=p.dtype), alpha=-lr * m['scale'])

            if prev_ag_handle is not None:
                prev_ag_handle.wait()
                pp = prev_m['p']
                upd = prev_m['full_update'][:prev_m['B']]
                if wd > 0.0:
                    pp.data.mul_(1.0 - lr * wd)
                pp.add_(upd.to(dtype=pp.dtype), alpha=-lr * prev_m['scale'])

            if hasattr(self, '_rs_futures'):
                del self._rs_futures

        return loss

def build_sentencepiece_luts(
    sp: spm.SentencePieceProcessor, vocab_size: int, device: torch.device
) -> tuple[Tensor, Tensor, Tensor]:
    sp_vocab_size = int(sp.vocab_size())
    table_size = max(sp_vocab_size, vocab_size)
    base_bytes_np = np.zeros((table_size,), dtype=np.int16)
    has_leading_space_np = np.zeros((table_size,), dtype=np.bool_)
    is_boundary_token_np = np.ones((table_size,), dtype=np.bool_)
    for token_id in range(sp_vocab_size):
        if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id):
            continue
        is_boundary_token_np[token_id] = False
        if sp.is_byte(token_id):
            base_bytes_np[token_id] = 1
            continue
        piece = sp.id_to_piece(token_id)
        if piece.startswith("\u2581"):
            has_leading_space_np[token_id] = True
            piece = piece[1:]
        base_bytes_np[token_id] = len(piece.encode("utf-8"))
    return (
        torch.tensor(base_bytes_np, dtype=torch.int16, device=device),
        torch.tensor(has_leading_space_np, dtype=torch.bool, device=device),
        torch.tensor(is_boundary_token_np, dtype=torch.bool, device=device),
    )
def load_validation_tokens(pattern: str, seq_len: int) -> Tensor:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    tokens = torch.cat([load_data_shard(file) for file in files]).contiguous()
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    if usable <= 0:
        raise ValueError(f"Validation split is too short for TRAIN_SEQ_LEN={seq_len}")
    return tokens[: usable + 1]
def eval_val(
    args: Hyperparameters,
    model: nn.Module,
    rank: int,
    world_size: int,
    device: torch.device,
    grad_accum_steps: int,
    val_tokens: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
    eval_seq_len: int | None = None,
) -> tuple[float, float]:
    seq_len = eval_seq_len or args.train_seq_len
    local_batch_tokens = args.val_batch_size // (world_size * grad_accum_steps)
    if local_batch_tokens < seq_len:
        raise ValueError(
            "VAL_BATCH_SIZE must provide at least one sequence per rank; "
            f"got VAL_BATCH_SIZE={args.val_batch_size}, WORLD_SIZE={world_size}, "
            f"GRAD_ACCUM_STEPS={grad_accum_steps}, seq_len={seq_len}"
        )
    local_batch_seqs = local_batch_tokens // seq_len
    total_seqs = (val_tokens.numel() - 1) // seq_len
    seq_start = (total_seqs * rank) // world_size
    seq_end = (total_seqs * (rank + 1)) // world_size
    val_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count = torch.zeros((), device=device, dtype=torch.float64)
    model.eval()
    with torch.inference_mode():
        for batch_seq_start in range(seq_start, seq_end, local_batch_seqs):
            batch_seq_end = min(batch_seq_start + local_batch_seqs, seq_end)
            raw_start = batch_seq_start * seq_len
            raw_end = batch_seq_end * seq_len + 1
            local = val_tokens[raw_start:raw_end].to(device=device, dtype=torch.int64, non_blocking=True)
            x = local[:-1].reshape(-1, seq_len)
            y = local[1:].reshape(-1, seq_len)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                batch_loss = model(x, y).detach()
            batch_token_count = float(y.numel())
            val_loss_sum += batch_loss.to(torch.float64) * batch_token_count
            val_token_count += batch_token_count
            prev_ids = x.reshape(-1)
            tgt_ids = y.reshape(-1)
            token_bytes = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
            token_bytes += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(dtype=torch.int16)
            val_byte_count += token_bytes.to(torch.float64).sum()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(val_loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_byte_count, op=dist.ReduceOp.SUM)
    val_loss = val_loss_sum / val_token_count
    bits_per_token = val_loss.item() / math.log(2.0)
    tokens_per_byte = val_token_count.item() / val_byte_count.item()
    model.train()
    return float(val_loss.item()), float(bits_per_token * tokens_per_byte)

CONTROL_TENSOR_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "CONTROL_TENSOR_NAME_PATTERNS",
        "attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,q_gain,skip_weight,skip_weights,skip_gates,ve_layer_scales,ve_shared.scale,lane_merge,repeat_pass_scale",
    ).split(",")
    if pattern
)
INT8_CLIP_Q = 99.99984 / 100.0
def quantize_float_tensor(t: Tensor) -> tuple[Tensor, Tensor]:
    t32 = t.float()
    if t32.ndim == 2:
        clip_abs = (
            torch.quantile(t32.abs(), INT8_CLIP_Q, dim=1)
            if t32.numel()
            else torch.empty((t32.shape[0],), dtype=torch.float32)
        )
        clipped = torch.maximum(torch.minimum(t32, clip_abs[:, None]), -clip_abs[:, None])
        scale = (clip_abs / 127.0).clamp_min(1.0 / 127.0)
        q = torch.clamp(torch.round(clipped / scale[:, None]), -127, 127).to(torch.int8).contiguous()
        return q, scale.to(dtype=torch.float16).contiguous()
    clip_abs = float(torch.quantile(t32.abs().flatten(), INT8_CLIP_Q).item()) if t32.numel() else 0.0
    scale = torch.tensor(clip_abs / 127.0 if clip_abs > 0 else 1.0, dtype=torch.float32)
    q = torch.clamp(torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale), -127, 127).to(torch.int8).contiguous()
    return q, scale

def load_data_shard(file: Path) -> Tensor:
    header_bytes = 256 * np.dtype("<i4").itemsize
    token_bytes = np.dtype("<u2").itemsize
    header = np.fromfile(file, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {file}")
    num_tokens = int(header[2])
    expected_size = header_bytes + num_tokens * token_bytes
    if file.stat().st_size != expected_size:
        raise ValueError(f"Shard size mismatch for {file}: expected {expected_size} bytes")
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes)
    if tokens_np.size != num_tokens:
        raise ValueError(f"Short read for {file}")
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))
_SHARD_HEADER_BYTES = 256 * np.dtype("<i4").itemsize
_SHARD_NTOKENS_CACHE: dict[str, int] = {}
_MMAP_CACHE: dict[str, np.memmap] = {}
def _read_num_tokens(file: Path) -> int:
    key = str(file)
    cached = _SHARD_NTOKENS_CACHE.get(key)
    if cached is not None:
        return cached
    header = np.fromfile(file, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {file}")
    n = int(header[2])
    _SHARD_NTOKENS_CACHE[key] = n
    return n
def _get_shard_memmap(file: Path) -> np.memmap:
    key = str(file)
    mm = _MMAP_CACHE.get(key)
    if mm is not None:
        return mm
    n = _read_num_tokens(file)
    mm = np.memmap(file, mode="r", dtype="<u2", offset=_SHARD_HEADER_BYTES, shape=(n,))
    _MMAP_CACHE[key] = mm
    return mm
class DistributedTokenLoader:
    def __init__(self, pattern: str, rank: int, world_size: int, device: torch.device):
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files found for pattern: {pattern}")
        self._num_tokens = np.array([_read_num_tokens(f) for f in self.files], dtype=np.int64)
        seed = 0
        for f in self.files:
            for b in str(f).encode():
                seed = ((seed ^ b) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
        self._rng = np.random.Generator(np.random.PCG64(seed))
        self._cfg = None
        self._eligible_shards = None
        self._base_block_counts = None
        n = len(self.files)
        self._cursor_phase = np.zeros(n, dtype=np.int64)
        self._cursor_block_count = np.zeros(n, dtype=np.int64)
        self._cursor_next = np.zeros(n, dtype=np.int64)
        self._cursor_start = np.zeros(n, dtype=np.int64)
        self._cursor_stride = np.ones(n, dtype=np.int64)
        self._cursor_init = np.zeros(n, dtype=np.bool_)
        self._batches_built = 0
        self._data_ramp_steps = float(os.environ.get("DATA_RAMP_STEPS", "1800"))
        self._data_warmdown_coupling = bool(int(os.environ.get("DATA_WARMDOWN_COUPLING", "0")))
        self._warmdown_frac = 0.0
    def set_warmdown_frac(self, frac: float) -> None:
        self._warmdown_frac = frac
    def _pick_coprime_stride(self, n: int) -> int:
        if n <= 1:
            return 1
        while True:
            s = int(self._rng.integers(1, n))
            if math.gcd(s, n) == 1:
                return s
    def _reset_cursor(self, si: int, seq_len: int) -> None:
        nt = int(self._num_tokens[si])
        max_phase = min(seq_len - 1, max(0, nt - seq_len - 1))
        phase = int(self._rng.integers(max_phase + 1)) if max_phase > 0 else 0
        bc = (nt - 1 - phase) // seq_len
        self._cursor_phase[si] = phase
        self._cursor_block_count[si] = bc
        self._cursor_next[si] = 0
        self._cursor_start[si] = int(self._rng.integers(bc)) if bc > 1 else 0
        self._cursor_stride[si] = self._pick_coprime_stride(bc)
        self._cursor_init[si] = True
    def _ensure_cursor(self, si: int, seq_len: int) -> None:
        if not self._cursor_init[si] or self._cursor_next[si] >= self._cursor_block_count[si]:
            self._reset_cursor(si, seq_len)
    def _take_from_shard(self, si: int, seq_len: int, count: int, out: list) -> None:
        rem = count
        while rem > 0:
            self._ensure_cursor(si, seq_len)
            bc = int(self._cursor_block_count[si])
            ni = int(self._cursor_next[si])
            take = min(rem, bc - ni)
            phase = int(self._cursor_phase[si])
            start = int(self._cursor_start[si])
            stride = int(self._cursor_stride[si])
            for j in range(take):
                bi = (start + (ni + j) * stride) % bc
                out.append((si, phase + bi * seq_len))
            self._cursor_next[si] = ni + take
            rem -= take
    def _init_pipeline(self, global_tokens: int, seq_len: int, grad_accum_steps: int) -> None:
        local_tokens = global_tokens // (self.world_size * grad_accum_steps)
        num_seqs = local_tokens // seq_len
        global_num_seqs = num_seqs * self.world_size
        self._cfg = (local_tokens, seq_len, num_seqs, global_num_seqs)
        bbc = (self._num_tokens - 1) // seq_len
        eligible = bbc > 0
        self._eligible_shards = np.nonzero(eligible)[0].astype(np.int64)
        self._base_block_counts = bbc[self._eligible_shards].astype(np.int64)
    def _sample_global_windows(self) -> list:
        _, seq_len, _, gns = self._cfg
        ec = int(self._eligible_shards.size)
        ramp_progress = min(self._batches_built / max(self._data_ramp_steps, 1.0), 1.0)
        base_alpha = 0.90 - 0.40 * ramp_progress
        if self._data_warmdown_coupling:
            decay_alpha = self._warmdown_frac * 0.30
            alpha = base_alpha + decay_alpha
        else:
            alpha = base_alpha
        remaining = np.empty(ec, dtype=np.float64)
        for i, si in enumerate(self._eligible_shards.tolist()):
            if self._cursor_init[si]:
                r = int(self._cursor_block_count[si]) - int(self._cursor_next[si])
                remaining[i] = float(max(r, 1))
            else:
                remaining[i] = float(self._base_block_counts[i])
        weights = np.power(remaining, alpha)
        ws = float(weights.sum())
        if not np.isfinite(ws) or ws <= 0.0:
            weights = np.ones(ec, dtype=np.float64)
            ws = float(weights.sum())
        probs = weights / ws
        low = min(max(8, self.world_size), ec, gns)
        high = min(max(32, self.world_size * 8), ec, gns)
        ramp_mix = low + ramp_progress * (high - low)
        if self._data_warmdown_coupling:
            decay_mix = self._warmdown_frac * (ramp_mix - low) * 0.5
            mix = max(1, min(int(round(ramp_mix - decay_mix)), ec, gns))
        else:
            mix = max(1, min(int(round(ramp_mix)), ec, gns))
        cp = self._rng.choice(ec, size=mix, replace=False, p=probs)
        cs = self._eligible_shards[cp]
        cpr = probs[cp].copy()
        cpr /= cpr.sum()
        counts = np.ones(mix, dtype=np.int64)
        extra = gns - mix
        if extra > 0:
            counts += self._rng.multinomial(extra, cpr).astype(np.int64)
        perm = self._rng.permutation(mix)
        cs, counts = cs[perm], counts[perm]
        buckets = []
        for si, cnt in zip(cs.tolist(), counts.tolist()):
            b = []
            self._take_from_shard(int(si), seq_len, int(cnt), b)
            if b:
                if len(b) > 1:
                    bp = self._rng.permutation(len(b))
                    b = [b[int(k)] for k in bp.tolist()]
                buckets.append(b)
        windows = []
        active = [i for i, bk in enumerate(buckets) if bk]
        while active:
            order = self._rng.permutation(len(active))
            new_active = []
            for oi in order.tolist():
                bi = active[oi]
                if buckets[bi]:
                    windows.append(buckets[bi].pop())
                if buckets[bi]:
                    new_active.append(bi)
            active = new_active
        return windows
    def next_batch(self, global_tokens: int, seq_len: int, grad_accum_steps: int) -> tuple[Tensor, Tensor]:
        if self._cfg is None:
            self._init_pipeline(global_tokens, seq_len, grad_accum_steps)
        _, _, num_seqs, _ = self._cfg
        gw = self._sample_global_windows()
        local_w = gw[self.rank::self.world_size]
        x = torch.empty((num_seqs, seq_len), dtype=torch.int64)
        y = torch.empty((num_seqs, seq_len), dtype=torch.int64)
        for slot, (si, pos) in enumerate(local_w):
            mm = _get_shard_memmap(self.files[si])
            window = torch.as_tensor(np.array(mm[pos:pos + seq_len + 1], dtype=np.int64))
            x[slot] = window[:-1]
            y[slot] = window[1:]
        self._batches_built += 1
        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)

class RMSNorm(nn.Module):
    def __init__(self, eps: float | None = None):
        super().__init__()
        self.eps = eps
    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)
class CastedLinear(nn.Linear):
    _qat_enabled: bool = False
    def forward(self, x: Tensor) -> Tensor:
        w = self.weight.to(x.dtype)
        if CastedLinear._qat_enabled and self.training and w.ndim == 2:
            with torch.no_grad():
                w32 = self.weight.float()
                row_max = w32.abs().amax(dim=1)
                scale = (row_max / 31.0).clamp_min(1.0 / 31.0)
                w_q = (torch.clamp(torch.round(w32 / scale[:, None]), -32, 31) * scale[:, None]).to(x.dtype)
            w = w + (w_q - w).detach()
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, w, bias)
def restore_low_dim_params_to_fp32(module: nn.Module) -> None:
    with torch.no_grad():
        for name, param in module.named_parameters():
            if (param.ndim < 2 or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)) and param.dtype != torch.float32:
                param.data = param.data.float()
class Rotary(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0, train_seq_len: int = 1024, rope_dims: int = 0):
        super().__init__()
        self.dim = dim
        self.base = base
        self.train_seq_len = train_seq_len
        self.rope_dims = rope_dims if rope_dims > 0 else dim
        inv_freq = 1.0 / (base ** (torch.arange(0, self.rope_dims, 2, dtype=torch.float32) / self.rope_dims))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached: Tensor | None = None
        self._sin_cached: Tensor | None = None
    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        if (
            self._cos_cached is None
            or self._sin_cached is None
            or self._seq_len_cached != seq_len
            or self._cos_cached.device != device
        ):
            rd = self.rope_dims
            if seq_len > self.train_seq_len:
                scale = seq_len / self.train_seq_len
                new_base = self.base * (scale ** (rd / (rd - 2)))
                inv_freq = 1.0 / (new_base ** (torch.arange(0, rd, 2, dtype=torch.float32, device=device) / rd))
            else:
                inv_freq = self.inv_freq.to(device)
            t = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
            freqs = torch.outer(t, inv_freq)
            self._cos_cached = freqs.cos()[None, :, None, :]
            self._sin_cached = freqs.sin()[None, :, None, :]
            self._seq_len_cached = seq_len
        return self._cos_cached.to(dtype=dtype), self._sin_cached.to(dtype=dtype)
def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor, rope_dims: int = 0) -> Tensor:
    if rope_dims > 0 and rope_dims < x.size(-1):
        x_rope, x_pass = x[..., :rope_dims], x[..., rope_dims:]
        half = rope_dims // 2
        x1, x2 = x_rope[..., :half], x_rope[..., half:]
        x_rope = torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)
        return torch.cat((x_rope, x_pass), dim=-1)
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)

class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, rope_base: float, qk_gain_init: float, train_seq_len: int = 2048):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.rope_dims = 0
        self.rotary = Rotary(self.head_dim, base=rope_base, train_seq_len=train_seq_len)
        self.use_xsa = False
    def _xsa_efficient(self, y: Tensor, v: Tensor) -> Tensor:
        B, T, H, D = y.shape
        Hkv = v.size(-2)
        group = H // Hkv
        y_g = y.reshape(B, T, Hkv, group, D)        # [B, T, Hkv, group, D]
        vn = F.normalize(v, dim=-1).unsqueeze(-2)    # [B, T, Hkv, 1, D] -- broadcast ready
        proj = (y_g * vn).sum(dim=-1, keepdim=True) * vn
        return (y_g - proj).reshape(B, T, H, D)
    def forward(self, x: Tensor, q_w: Tensor, k_w: Tensor, v_w: Tensor, out_w: Tensor, v_embed: Tensor | None = None) -> Tensor:
        bsz, seqlen, dim = x.shape
        q = F.linear(x, q_w.to(x.dtype)).reshape(bsz, seqlen, self.num_heads, self.head_dim)
        k = F.linear(x, k_w.to(x.dtype)).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
        v = F.linear(x, v_w.to(x.dtype))
        if v_embed is not None:
            v = v + v_embed
        v = v.reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin, self.rope_dims)
        k = apply_rotary_emb(k, cos, sin, self.rope_dims)
        q = q * self.q_gain.to(dtype=q.dtype)[None, None, :, None]
        y = flash_attn_3_func(q, k, v, causal=True)
        if self.use_xsa:
            y = self._xsa_efficient(y, v)
        y = y.reshape(bsz, seqlen, dim)
        return F.linear(y, out_w.to(x.dtype))

_ACT_MAP = {
    "leaky05": lambda x: F.leaky_relu(x, 0.5).square(),
    "mish2":   lambda x: F.mish(x).square(),
}
_MLP_ACTIVATION = os.environ.get("MLP_ACTIVATION", "leaky05")
assert _MLP_ACTIVATION in _ACT_MAP, f"Unknown MLP_ACTIVATION={_MLP_ACTIVATION!r}, valid: {list(_ACT_MAP)}"
_MLP_ACT_FN = _ACT_MAP[_MLP_ACTIVATION]

class MLP(nn.Module):
    def __init__(self, dim: int, mlp_mult: int):
        super().__init__()
        self.act_fn = _MLP_ACT_FN
    def forward(self, x: Tensor, up_w: Tensor, down_w: Tensor) -> Tensor:
        _raw = F.linear(x, up_w.to(x.dtype))
        return F.linear(self.act_fn(_raw), down_w.to(x.dtype))

class ValueEmbedding(nn.Module):
    def __init__(self, vocab_size: int, ve_dim: int, target_dim: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, ve_dim)
        nn.init.normal_(self.embed.weight, std=0.01)
        self.proj = CastedLinear(ve_dim, target_dim, bias=False) if ve_dim != target_dim else None
        if self.proj is not None:
            nn.init.zeros_(self.proj.weight)
        self.scale = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
    def forward(self, token_ids: Tensor) -> Tensor:
        h = self.embed(token_ids)
        if self.proj is not None:
            h = self.proj(h)
        return h * self.scale.to(dtype=h.dtype)

class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, mlp_mult: int,
                 rope_base: float, qk_gain_init: float, layer_idx: int = 0, ln_scale: bool = False,
                 train_seq_len: int = 2048):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init, train_seq_len=train_seq_len)
        self.mlp = MLP(dim, mlp_mult)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())
        self.ln_scale_factor = 1.0 / math.sqrt(layer_idx + 1) if ln_scale else 1.0
    def forward(self, x: Tensor, x0: Tensor, q_w: Tensor, k_w: Tensor, v_w: Tensor, out_w: Tensor, up_w: Tensor, down_w: Tensor, v_embed: Tensor | None = None, down_w_override: Tensor | None = None, skip_attn: bool = False) -> Tensor:
        mix = self.resid_mix.to(dtype=x.dtype)
        x_in = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        x_out = x_in
        if not (getattr(self, 'disable_attn', False) or skip_attn):
            attn_out = self.attn(self.attn_norm(x_in) * self.ln_scale_factor, q_w, k_w, v_w, out_w, v_embed=v_embed)
            x_out = x_out + self.attn_scale.to(dtype=x_in.dtype)[None, None, :] * attn_out
        if not getattr(self, 'disable_mlp', False):
            dw = down_w_override if down_w_override is not None else down_w
            x_out = x_out + self.mlp_scale.to(dtype=x_out.dtype)[None, None, :] * self.mlp(self.mlp_norm(x_out) * self.ln_scale_factor, up_w, dw)
        return x_out

class GPT(nn.Module):
    def __init__(self, vocab_size: int, num_layers: int, model_dim: int, num_heads: int,
                 num_kv_heads: int, mlp_mult: int, tie_embeddings: bool, tied_embed_init_std: float,
                 logit_softcap: float, rope_base: float, qk_gain_init: float,
                 xsa_last_n: int = 0, rope_dims: int = 0, ln_scale: bool = False,
                 sigmoid_skip: bool = False, recur_layers: list[int] | None = None,
                 ve_enabled: bool = False, ve_dim: int = 128, ve_layers: list[int] | None = None,
                 parallel_start_layer: int = -1, repeat_untie_mlp: str = 'none',
                 disable_layer0_attn: bool = False, recur_num_loops: int = 2,
                 attn_res: bool = False, lora_rank: int = 0,
                 repeat_pass_scale: bool = False, train_seq_len: int = 2048,
                 recur_skip_physical: bool = False, per_layer_lane_merge: bool = False,
                 repeat_attn_lora: bool = False, attn_lora_rank: int = 0,
                 disable_layer0_mlp: bool = False, skip_repeat_attn: bool = False,
                 skip_repeat_attn_layers: set = None):
        super().__init__()
        if skip_repeat_attn_layers is None:
            skip_repeat_attn_layers = set()
        self.recur_skip_physical = recur_skip_physical
        self.tie_embeddings = tie_embeddings
        self.tied_embed_init_std = tied_embed_init_std
        self.logit_softcap = logit_softcap
        self.sigmoid_skip = sigmoid_skip
        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = nn.Parameter(torch.ones(self.num_skip_weights, model_dim, dtype=torch.float32))
        self.skip_gates = nn.Parameter(torch.zeros(self.num_skip_weights, model_dim, dtype=torch.float32)) if sigmoid_skip else None
        kv_dim = num_kv_heads * (model_dim // num_heads)
        mlp_dim = int(mlp_mult * model_dim)
        self.num_layers = num_layers
        self.qo_bank = nn.Parameter(torch.empty(2 * num_layers, model_dim, model_dim))
        self.kv_bank = nn.Parameter(torch.empty(2 * num_layers, kv_dim, model_dim))
        self.mlp_up_bank = nn.Parameter(torch.empty(num_layers, mlp_dim, model_dim))
        self.mlp_down_bank = nn.Parameter(torch.empty(num_layers, model_dim, mlp_dim))
        self.blocks = nn.ModuleList([
            Block(model_dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init,
                  layer_idx=i, ln_scale=ln_scale, train_seq_len=train_seq_len)
            for i in range(num_layers)
        ])
        if rope_dims > 0:
            for block in self.blocks:
                block.attn.rope_dims = rope_dims
                block.attn.rotary = Rotary(model_dim // num_heads, base=rope_base, train_seq_len=train_seq_len, rope_dims=rope_dims)
        self.final_norm = RMSNorm()
        self.lm_head = None if tie_embeddings else CastedLinear(model_dim, vocab_size, bias=False)
        if self.lm_head is not None:
            self.lm_head._zero_init = True
        if xsa_last_n > 0:
            for i in range(max(0, num_layers - xsa_last_n), num_layers):
                self.blocks[i].attn.use_xsa = True
        # Depth recurrence
        self.recur_layers = sorted(recur_layers) if recur_layers else []
        self.recur_num_loops = recur_num_loops
        self._recurrence_active = False
        # Value Embedding
        self.ve_layer_indices = sorted(ve_layers) if ve_enabled and ve_layers else []
        if self.ve_layer_indices:
            self.ve_shared = ValueEmbedding(vocab_size, ve_dim, kv_dim)
            self.ve_layer_scales = nn.ParameterList(
                [nn.Parameter(torch.ones(1, dtype=torch.float32)) for _ in self.ve_layer_indices]
            )
        else:
            self.ve_shared = None
            self.ve_layer_scales = nn.ParameterList()
        # Parallel residuals
        self.parallel_start_layer = parallel_start_layer
        self.per_layer_lane_merge = per_layer_lane_merge
        if 0 < parallel_start_layer < num_layers:
            n_parallel = num_layers - parallel_start_layer
            if per_layer_lane_merge:
                self.lane_merge = nn.Parameter(torch.full((n_parallel,), 0.5, dtype=torch.float32))
            else:
                self.lane_merge = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))
        else:
            self.lane_merge = None
        # Untied MLP down-projection for recurrence repeat pass
        self.repeat_untie_mlp = repeat_untie_mlp
        self.lora_rank = lora_rank
        n_recur = len(self.recur_layers) if self.recur_layers else 0
        if repeat_untie_mlp == 'lora' and n_recur > 0 and lora_rank > 0:
            # LoRA delta: effective_down = base_down + A @ B
            self.repeat_lora_A = nn.Parameter(torch.empty(n_recur, model_dim, lora_rank))
            self.repeat_lora_B = nn.Parameter(torch.zeros(n_recur, lora_rank, mlp_dim))
            nn.init.kaiming_uniform_(self.repeat_lora_A, a=math.sqrt(5))
            self.repeat_down_bank = None
        elif repeat_untie_mlp == 'down' and n_recur > 0:
            self.repeat_down_bank = nn.Parameter(torch.zeros(n_recur, model_dim, mlp_dim))
            self.repeat_lora_A = None
            self.repeat_lora_B = None
        else:
            self.repeat_down_bank = None
            self.repeat_lora_A = None
            self.repeat_lora_B = None
        # LoRA for repeat attention Q projection
        if repeat_attn_lora and n_recur > 0 and attn_lora_rank > 0:
            self.repeat_attn_lora_A = nn.Parameter(torch.empty(n_recur, model_dim, attn_lora_rank))
            self.repeat_attn_lora_B = nn.Parameter(torch.zeros(n_recur, attn_lora_rank, model_dim))
            nn.init.kaiming_uniform_(self.repeat_attn_lora_A, a=math.sqrt(5))
        else:
            self.repeat_attn_lora_A = None
            self.repeat_attn_lora_B = None
        # Per-pass recurrence scaling: learned scalar per recur layer for repeat passes
        if repeat_pass_scale and n_recur > 0:
            self.repeat_pass_scale = nn.Parameter(torch.ones(n_recur, dtype=torch.float32))
        else:
            self.repeat_pass_scale = None
        # Disable attention at layer 0
        if disable_layer0_attn and len(self.blocks) > 0:
            self.blocks[0].disable_attn = True
        # Disable MLP at layer 0 (gradient analysis: model pushing L0 MLP toward zero)
        if disable_layer0_mlp and len(self.blocks) > 0:
            self.blocks[0].disable_mlp = True
        # Skip attention during recurrence repeat passes (all layers or specific ones)
        self.skip_repeat_attn = skip_repeat_attn
        self.skip_repeat_attn_layers = skip_repeat_attn_layers
        # AttnRes: learned depth attention over residuals
        self.attn_res = attn_res
        if attn_res:
            max_virtual = num_layers + len(self.recur_layers) * max(self.recur_num_loops - 1, 0)
            self.ar_queries = nn.Parameter(torch.zeros(max_virtual, model_dim, dtype=torch.float32))
            self.ar_key_norm = RMSNorm()
        self._init_weights()
    def _init_weights(self) -> None:
        if self.tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=self.tied_embed_init_std)
        n = self.num_layers
        proj_scale = 1.0 / math.sqrt(2 * n)
        for i in range(n):
            nn.init.orthogonal_(self.qo_bank.data[i], gain=1.0)
            nn.init.zeros_(self.qo_bank.data[n + i])
            nn.init.orthogonal_(self.kv_bank.data[i], gain=1.0)
            nn.init.orthogonal_(self.kv_bank.data[n + i], gain=1.0)
            nn.init.orthogonal_(self.mlp_up_bank.data[i], gain=1.0)
            nn.init.zeros_(self.mlp_down_bank.data[i])
            self.qo_bank.data[n + i].mul_(proj_scale)
            self.mlp_down_bank.data[i].mul_(proj_scale)
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if getattr(module, "_zero_init", False):
                    nn.init.zeros_(module.weight)
                elif module.weight.ndim == 2 and module.weight.shape[0] >= 64 and module.weight.shape[1] >= 64:
                    nn.init.orthogonal_(module.weight, gain=1.0)
    def set_recurrence_active(self, active: bool) -> None:
        self._recurrence_active = active
    def prime_repeat_mlp(self) -> None:
        """Copy base MLP down weights to repeat bank at recurrence activation."""
        if self.repeat_down_bank is None:
            return
        with torch.no_grad():
            for ri, phys_idx in enumerate(self.recur_layers):
                self.repeat_down_bank.data[ri].copy_(self.mlp_down_bank.data[phys_idx])
    def _get_virtual_layers(self) -> list[tuple[int, bool]]:
        """Returns list of (physical_idx, is_repeat) tuples."""
        n = len(self.blocks)
        if not self._recurrence_active or not self.recur_layers:
            return [(i, False) for i in range(n)]
        virtual: list[tuple[int, bool]] = []
        inserted = False
        extra_loops = self.recur_num_loops - 1  # number of extra passes
        for i in range(n):
            virtual.append((i, False))
            if not inserted and i == self.recur_layers[-1]:
                for _ in range(extra_loops):
                    for rl in self.recur_layers:
                        virtual.append((rl, True))
                inserted = True
        return virtual
    def _bank_weights(self, phys_idx: int):
        n = self.num_layers
        return (self.qo_bank[phys_idx], self.kv_bank[phys_idx], self.kv_bank[n + phys_idx],
                self.qo_bank[n + phys_idx], self.mlp_up_bank[phys_idx], self.mlp_down_bank[phys_idx])
    def _get_repeat_down_w(self, phys_idx: int) -> Tensor | None:
        if phys_idx not in self.recur_layers:
            return None
        idx = self.recur_layers.index(phys_idx)
        if self.repeat_lora_A is not None:
            base_down = self.mlp_down_bank[phys_idx]
            delta = self.repeat_lora_A[idx] @ self.repeat_lora_B[idx]
            return base_down + delta
        if self.repeat_down_bank is not None:
            return self.repeat_down_bank[idx]
        return None
    def _get_repeat_q_w(self, phys_idx: int) -> Tensor | None:
        if self.repeat_attn_lora_A is None or phys_idx not in self.recur_layers:
            return None
        idx = self.recur_layers.index(phys_idx)
        base_q = self.qo_bank[phys_idx]
        delta = self.repeat_attn_lora_A[idx] @ self.repeat_attn_lora_B[idx]
        return base_q + delta
    def _get_ve(self, phys_idx: int, input_ids: Tensor, ve_cache: dict) -> Tensor | None:
        if self.ve_shared is None or phys_idx not in self.ve_layer_indices:
            return None
        if 've' not in ve_cache:
            ve_cache['ve'] = self.ve_shared(input_ids)
        ve_base = ve_cache['ve']
        ve_idx = self.ve_layer_indices.index(phys_idx)
        return ve_base * self.ve_layer_scales[ve_idx].to(dtype=ve_base.dtype)
    def _attn_res_input(self, layer_out_list: list[Tensor], layer_idx: int) -> Tensor:
        stacked = torch.stack(layer_out_list)
        keys = self.ar_key_norm(stacked)
        q = self.ar_queries[layer_idx].to(dtype=keys.dtype)
        logits = torch.einsum('d, l b t d -> l b t', q, keys)
        weights = F.softmax(logits, dim=0)
        return torch.einsum('l b t, l b t d -> b t d', weights, stacked)
    def _apply_repeat_scale(self, x_before: Tensor, x_after: Tensor, phys_idx: int) -> Tensor:
        """Scale the residual delta for repeat passes."""
        if self.repeat_pass_scale is None:
            return x_after
        ri = self.recur_layers.index(phys_idx)
        s = self.repeat_pass_scale[ri].to(dtype=x_after.dtype)
        return x_before + s * (x_after - x_before)
    def _forward_body(self, input_ids: Tensor) -> Tensor:
        x = self.tok_emb(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x
        virtual_layers = self._get_virtual_layers()
        num_virtual = len(virtual_layers)
        num_enc = num_virtual // 2
        num_dec = num_virtual - num_enc
        skips: list[Tensor] = []
        ve_cache: dict = {}
        parallel_thresh = self.parallel_start_layer if self.lane_merge is not None else num_virtual + 1
        is_parallel = False
        lane0: Tensor | None = None
        lane1: Tensor | None = None
        ar_layers: list[Tensor] = [x] if self.attn_res else []
        for vi in range(num_enc):
            phys_idx, is_repeat = virtual_layers[vi]
            if self.attn_res:
                x = self._attn_res_input(ar_layers, vi)
            q_w, k_w, v_w, out_w, up_w, down_w = self._bank_weights(phys_idx)
            if is_repeat:
                qw_o = self._get_repeat_q_w(phys_idx)
                if qw_o is not None: q_w = qw_o
            ve = self._get_ve(phys_idx, input_ids, ve_cache)
            dw_override = self._get_repeat_down_w(phys_idx) if is_repeat else None
            x_prev = x if is_repeat else None
            _skip = is_repeat and (self.skip_repeat_attn or phys_idx in self.skip_repeat_attn_layers)
            x = self.blocks[phys_idx](x, x0, q_w, k_w, v_w, out_w, up_w, down_w, v_embed=ve, down_w_override=dw_override, skip_attn=_skip)
            if is_repeat and x_prev is not None:
                x = self._apply_repeat_scale(x_prev, x, phys_idx)
            if self.attn_res:
                ar_layers.append(x)
            if not (is_repeat and self.recur_skip_physical):
                skips.append(x)
        skip_idx = 0
        for vi in range(num_dec):
            phys_idx, is_repeat = virtual_layers[num_enc + vi]
            skip_this = is_repeat and self.recur_skip_physical
            if skips and skip_idx < self.num_skip_weights and not skip_this:
                target = lane0 if is_parallel else x
                scaled_skip = self.skip_weights[skip_idx].to(dtype=target.dtype)[None, None, :] * skips.pop()
                if self.skip_gates is not None:
                    g = torch.sigmoid(self.skip_gates[skip_idx].to(dtype=target.dtype))[None, None, :]
                    target = torch.lerp(scaled_skip, target, g)
                else:
                    target = target + scaled_skip
                if is_parallel:
                    lane0 = target
                else:
                    x = target
                skip_idx += 1
            if self.attn_res:
                x = self._attn_res_input(ar_layers, num_enc + vi)
            # Enter parallel mode when physical layer >= threshold
            if phys_idx >= parallel_thresh and not is_parallel:
                lane0 = x
                lane1 = x
                is_parallel = True
            if is_parallel:
                block = self.blocks[phys_idx]
                q_w, k_w, v_w, out_w, up_w, down_w = self._bank_weights(phys_idx)
                if is_repeat:
                    qw_o = self._get_repeat_q_w(phys_idx)
                    if qw_o is not None: q_w = qw_o
                ve = self._get_ve(phys_idx, input_ids, ve_cache)
                dw_override = self._get_repeat_down_w(phys_idx) if is_repeat else None
                effective_down_w = dw_override if dw_override is not None else down_w
                if self.per_layer_lane_merge:
                    # Single-stream: per-layer merge of attn and MLP contributions
                    mix = block.resid_mix.to(dtype=x.dtype)
                    attn_in = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
                    attn_out = block.attn(block.attn_norm(attn_in) * block.ln_scale_factor, q_w, k_w, v_w, out_w, v_embed=ve)
                    attn_contrib = block.attn_scale.to(dtype=x.dtype)[None, None, :] * attn_out
                    mlp_in = block.mlp_norm(x) * block.ln_scale_factor
                    mlp_out = block.mlp(mlp_in, up_w, effective_down_w)
                    mlp_contrib = block.mlp_scale.to(dtype=x.dtype)[None, None, :] * mlp_out
                    m = torch.sigmoid(self.lane_merge[phys_idx - parallel_thresh]).to(dtype=x.dtype)
                    x = x + m * attn_contrib + (1 - m) * mlp_contrib
                else:
                    # Dual-lane: separate accumulation, single final merge
                    mix = block.resid_mix.to(dtype=lane0.dtype)
                    attn_in = mix[0][None, None, :] * lane0 + mix[1][None, None, :] * x0
                    attn_out = block.attn(block.attn_norm(attn_in) * block.ln_scale_factor, q_w, k_w, v_w, out_w, v_embed=ve)
                    lane0 = attn_in + block.attn_scale.to(dtype=attn_in.dtype)[None, None, :] * attn_out
                    mlp_in = block.mlp_norm(lane1) * block.ln_scale_factor
                    mlp_out = block.mlp(mlp_in, up_w, effective_down_w)
                    lane1 = lane1 + block.mlp_scale.to(dtype=lane1.dtype)[None, None, :] * mlp_out
                if self.attn_res:
                    if self.per_layer_lane_merge:
                        ar_layers.append(x)
                    else:
                        lm = self.lane_merge
                        m_ar = lm.to(dtype=lane0.dtype)
                        ar_layers.append(m_ar * lane0 + (1 - m_ar) * lane1)
            else:
                q_w, k_w, v_w, out_w, up_w, down_w = self._bank_weights(phys_idx)
                if is_repeat:
                    qw_o = self._get_repeat_q_w(phys_idx)
                    if qw_o is not None: q_w = qw_o
                ve = self._get_ve(phys_idx, input_ids, ve_cache)
                dw_override = self._get_repeat_down_w(phys_idx) if is_repeat else None
                x_prev = x if is_repeat else None
                _skip = is_repeat and (self.skip_repeat_attn or phys_idx in self.skip_repeat_attn_layers)
                x = self.blocks[phys_idx](x, x0, q_w, k_w, v_w, out_w, up_w, down_w, v_embed=ve, down_w_override=dw_override, skip_attn=_skip)
                if is_repeat and x_prev is not None:
                    x = self._apply_repeat_scale(x_prev, x, phys_idx)
                if self.attn_res:
                    ar_layers.append(x)
        if is_parallel and not self.per_layer_lane_merge:
            m = self.lane_merge.to(dtype=lane0.dtype)
            x = m * lane0 + (1 - m) * lane1
        return self.final_norm(x)
    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        x = self._forward_body(input_ids)
        x_flat = x.reshape(-1, x.size(-1))
        targets = target_ids.reshape(-1)
        if self.tie_embeddings:
            logits_proj = F.linear(x_flat, self.tok_emb.weight)
        else:
            logits_proj = self.lm_head(x_flat)
        logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
        return F.cross_entropy(logits.float(), targets, reduction="mean")
    def forward_logits(self, input_ids: Tensor) -> Tensor:
        x = self._forward_body(input_ids)
        if self.tie_embeddings:
            logits_proj = F.linear(x, self.tok_emb.weight)
        else:
            logits_proj = self.lm_head(x)
        return self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)

def _classify_param(name: str) -> str:
    if "tok_emb" in name or "lm_head" in name:
        return "embed"
    if ".mlp." in name:
        return "mlp"
    if ".attn." in name or (".proj." in name and ".mlp." not in name):
        return "attn"
    return "other"
def quantize_int6_per_row(t: Tensor, clip_range: int = 31, sdclip_k: float = 0.0) -> tuple[Tensor, Tensor]:
    t32 = t.float()
    if t32.ndim == 2:
        if sdclip_k > 0:
            # SDClip: clip = k * std(row) — entropy-optimal, no search needed
            row_clip = sdclip_k * t32.std(dim=1).clamp_min(1e-12)
            row_clip = torch.minimum(row_clip, t32.abs().amax(dim=1))  # never exceed actual max
            s = (row_clip / clip_range).clamp_min(1.0 / clip_range).to(torch.float16)
            q = torch.clamp(torch.round(t32 / s.float()[:, None]), -clip_range, clip_range).to(torch.int8)
            return q, s
        best_q, best_s, best_err = None, None, float('inf')
        for pct in [0.9990, 0.9995, 0.9999, 0.99999, 1.0]:
            if pct < 1.0:
                row_clip = torch.quantile(t32.abs(), pct, dim=1)
            else:
                row_clip = t32.abs().amax(dim=1)
            s = (row_clip / clip_range).clamp_min(1.0 / clip_range).to(torch.float16)
            q = torch.clamp(torch.round(t32 / s.float()[:, None]), -clip_range, clip_range).to(torch.int8)
            recon = q.float() * s.float()[:, None]
            err = (t32 - recon).pow(2).mean().item()
            if err < best_err:
                best_q, best_s, best_err = q, s, err
        return best_q, best_s
    amax = t32.abs().max().item()
    scale = torch.tensor(amax / clip_range if amax > 0 else 1.0, dtype=torch.float16)
    q = torch.clamp(torch.round(t32 / scale.float()), -clip_range, clip_range).to(torch.int8)
    return q, scale

def quantize_int6_gptq(weight: Tensor, hessian: Tensor, clip_range: int = 31,
                        block_size: int = 128, sdclip_k: float = 0.0) -> tuple[Tensor, Tensor]:
    t32 = weight.float()
    if t32.ndim != 2:
        return quantize_int6_per_row(t32, clip_range)
    rows, cols = t32.shape
    H = hessian.float().clone()
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    diag_mean = torch.mean(torch.diag(H))
    perm = None
    Hinv = None
    for attempt, damp_frac in enumerate([0.01, 0.05, 0.1, 0.5, 1.0]):
        try:
            Htry = H.clone()
            damp = damp_frac * diag_mean
            Htry[torch.arange(cols), torch.arange(cols)] += damp
            if perm is None:
                perm = torch.argsort(torch.diag(Htry), descending=True)
            Hp = Htry[perm][:, perm]
            L = torch.linalg.cholesky(Hp)
            Hinv_full = torch.cholesky_inverse(L)
            Hinv = torch.linalg.cholesky(Hinv_full, upper=True)
            break
        except torch._C._LinAlgError:
            if attempt == 4:
                return quantize_int6_per_row(t32, clip_range)
    inv_perm = torch.argsort(perm)
    W = t32[:, perm].clone()
    W[:, dead[perm]] = 0
    # Determine clip candidates
    if sdclip_k > 0:
        # SDClip: single entropy-optimal clip, no search
        row_clip = sdclip_k * t32.std(dim=1).clamp_min(1e-12)
        row_clip = torch.minimum(row_clip, t32.abs().amax(dim=1))
        clip_candidates = [(row_clip, "sdclip")]
    else:
        clip_candidates = []
        for pct in [0.9990, 0.9995, 0.9999, 0.99999, 1.0]:
            if pct < 1.0:
                rc = torch.quantile(t32.abs(), pct, dim=1)
            else:
                rc = t32.abs().amax(dim=1)
            clip_candidates.append((rc, f"p{pct}"))
    best_q, best_scale, best_err = None, None, float('inf')
    for row_clip, _label in clip_candidates:
        s = (row_clip / clip_range).clamp_min(1.0 / clip_range).to(torch.float16)
        sf = s.float()
        Q = torch.zeros_like(W, dtype=torch.int8)
        W_work = W.clone()
        for i1 in range(0, cols, block_size):
            i2 = min(i1 + block_size, cols)
            count = i2 - i1
            W1 = W_work[:, i1:i2].clone()
            Q1 = torch.zeros(rows, count, dtype=torch.int8)
            Err1 = torch.zeros(rows, count)
            Hinv1 = Hinv[i1:i2, i1:i2]
            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]
                q = torch.clamp(torch.round(w / sf), -clip_range, clip_range).to(torch.int8)
                Q1[:, i] = q
                err = (w - q.float() * sf) / d
                W1[:, i:] -= err.unsqueeze(1) * Hinv1[i, i:].unsqueeze(0)
                Err1[:, i] = err
            Q[:, i1:i2] = Q1
            if i2 < cols:
                W_work[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]
        recon = Q.float() * sf[:, None]
        mse = (W - recon).pow(2).mean().item()
        if mse < best_err:
            best_q, best_scale, best_err = Q, s, mse
    best_q = best_q[:, inv_perm]
    return best_q, best_scale

def generate_autoregressive_calib(model, device, num_seqs=64, seq_len=2048,
                                   vocab_size=1024, temperature=0.8, batch_size=8, seed=42):
    model.eval()
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)
    all_tokens = []
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for batch_start in range(0, num_seqs, batch_size):
            bs = min(batch_size, num_seqs - batch_start)
            tokens = torch.randint(0, vocab_size, (bs, 1), device=device, generator=rng)
            for pos in range(seq_len - 1):
                logits = model.forward_logits(tokens)
                next_logit = logits[:, -1, :]
                probs = torch.softmax(next_logit / temperature, dim=-1)
                next_tok = torch.multinomial(probs, 1, generator=rng)
                tokens = torch.cat([tokens, next_tok], dim=1)
            for i in range(bs):
                all_tokens.append(tokens[i:i+1])
    return all_tokens

def collect_hessians_from_tokens(hessian_model, token_seqs, device):
    hessians = {}
    hooks = []
    for name, module in hessian_model.named_modules():
        if isinstance(module, CastedLinear):
            param_name = name + ".weight"
            cols = module.weight.shape[1]
            hessians[param_name] = torch.zeros(cols, cols, dtype=torch.float32, device='cpu')
            def make_hook(pname):
                def hook_fn(module, input, output):
                    x = input[0].detach().float()
                    if x.ndim == 3:
                        x = x.reshape(-1, x.shape[-1])
                    hessians[pname] += (x.T @ x).cpu()
                return hook_fn
            h = module.register_forward_hook(make_hook(param_name))
            hooks.append(h)
    hessian_model.eval()
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for seq in token_seqs:
            x = seq[:, :-1].to(device)
            y = seq[:, 1:].to(device)
            hessian_model(x, y)
    for h in hooks:
        h.remove()
    num_batches = len(token_seqs)
    for name in hessians:
        H = hessians[name]
        H /= num_batches
        damp = 0.01 * torch.diag(H).mean().clamp_min(1e-6)
        H += damp * torch.eye(H.shape[0])
        hessians[name] = H
    return hessians

def _unbank_state_dict(sd: dict[str, Tensor], num_layers: int) -> dict[str, Tensor]:
    out: dict[str, Tensor] = {}
    n = num_layers
    for name, tensor in sd.items():
        if name == "qo_bank":
            for i in range(n):
                out[f"blocks.{i}.attn.c_q.weight"] = tensor[i]
                out[f"blocks.{i}.attn.proj.weight"] = tensor[n + i]
        elif name == "kv_bank":
            for i in range(n):
                out[f"blocks.{i}.attn.c_k.weight"] = tensor[i]
                out[f"blocks.{i}.attn.c_v.weight"] = tensor[n + i]
        elif name == "mlp_up_bank":
            for i in range(n):
                out[f"blocks.{i}.mlp.fc.weight"] = tensor[i]
        elif name == "mlp_down_bank":
            for i in range(n):
                out[f"blocks.{i}.mlp.proj.weight"] = tensor[i]
        else:
            out[name] = tensor
    return out

def _rebank_state_dict(sd: dict[str, Tensor], num_layers: int, template_sd: dict[str, Tensor]) -> dict[str, Tensor]:
    out: dict[str, Tensor] = {}
    n = num_layers
    qo_slices = [None] * (2 * n)
    kv_slices = [None] * (2 * n)
    up_slices = [None] * n
    down_slices = [None] * n
    consumed = set()
    for i in range(n):
        for key, lst, idx in [
            (f"blocks.{i}.attn.c_q.weight", qo_slices, i),
            (f"blocks.{i}.attn.proj.weight", qo_slices, n + i),
            (f"blocks.{i}.attn.c_k.weight", kv_slices, i),
            (f"blocks.{i}.attn.c_v.weight", kv_slices, n + i),
            (f"blocks.{i}.mlp.fc.weight", up_slices, i),
            (f"blocks.{i}.mlp.proj.weight", down_slices, i),
        ]:
            if key in sd:
                lst[idx] = sd[key]
                consumed.add(key)
    out["qo_bank"] = torch.stack(qo_slices).to(dtype=template_sd["qo_bank"].dtype)
    out["kv_bank"] = torch.stack(kv_slices).to(dtype=template_sd["kv_bank"].dtype)
    out["mlp_up_bank"] = torch.stack(up_slices).to(dtype=template_sd["mlp_up_bank"].dtype)
    out["mlp_down_bank"] = torch.stack(down_slices).to(dtype=template_sd["mlp_down_bank"].dtype)
    for name, tensor in sd.items():
        if name not in consumed:
            out[name] = tensor
    return out

class _HessianAttn(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, rope_base, qk_gain_init, rope_dims=0, train_seq_len=2048):
        super().__init__()
        self.num_heads, self.num_kv_heads = num_heads, num_kv_heads
        self.head_dim = dim // num_heads
        kv_dim = num_kv_heads * self.head_dim
        self.c_q = CastedLinear(dim, dim, bias=False)
        self.c_k = CastedLinear(dim, kv_dim, bias=False)
        self.c_v = CastedLinear(dim, kv_dim, bias=False)
        self.proj = CastedLinear(dim, dim, bias=False)
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.rope_dims = rope_dims
        self.rotary = Rotary(self.head_dim, base=rope_base, train_seq_len=train_seq_len, rope_dims=rope_dims)
        self.use_xsa = False
    def _xsa_efficient(self, y, v):
        B, T, H, D = y.shape; Hkv = v.size(-2); group = H // Hkv
        y_g = y.reshape(B, T, Hkv, group, D)
        vn = F.normalize(v, dim=-1).unsqueeze(-2)
        proj = (y_g * vn).sum(dim=-1, keepdim=True) * vn
        return (y_g - proj).reshape(B, T, H, D)
    def forward(self, x, v_embed=None):
        bsz, seqlen, dim = x.shape
        q = self.c_q(x).reshape(bsz, seqlen, self.num_heads, self.head_dim)
        k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
        v = self.c_v(x)
        if v_embed is not None:
            v = v + v_embed
        v = v.reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin, self.rope_dims)
        k = apply_rotary_emb(k, cos, sin, self.rope_dims)
        q = q * self.q_gain.to(dtype=q.dtype)[None, None, :, None]
        y = flash_attn_3_func(q, k, v, causal=True)
        if self.use_xsa:
            y = self._xsa_efficient(y, v)
        return self.proj(y.reshape(bsz, seqlen, dim))

class _HessianMLP(nn.Module):
    def __init__(self, dim, mlp_mult):
        super().__init__()
        self.fc = CastedLinear(dim, int(mlp_mult * dim), bias=False)
        self.proj = CastedLinear(int(mlp_mult * dim), dim, bias=False)
        self.act_fn = _MLP_ACT_FN
    def forward(self, x):
        return self.proj(self.act_fn(self.fc(x)))

class _HessianBlock(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, mlp_mult, rope_base,
                 qk_gain_init, layer_idx=0, ln_scale=False, rope_dims=0, train_seq_len=2048):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = _HessianAttn(dim, num_heads, num_kv_heads, rope_base, qk_gain_init, rope_dims=rope_dims, train_seq_len=train_seq_len)
        self.mlp = _HessianMLP(dim, mlp_mult)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())
        self.ln_scale_factor = 1.0 / math.sqrt(layer_idx + 1) if ln_scale else 1.0
    def forward(self, x, x0, v_embed=None):
        mix = self.resid_mix.to(dtype=x.dtype)
        x_in = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        x_out = x_in
        if not getattr(self, 'disable_attn', False):
            attn_out = self.attn(self.attn_norm(x_in) * self.ln_scale_factor, v_embed=v_embed)
            x_out = x_out + self.attn_scale.to(dtype=x_in.dtype)[None, None, :] * attn_out
        return x_out + self.mlp_scale.to(dtype=x_out.dtype)[None, None, :] * self.mlp(self.mlp_norm(x_out) * self.ln_scale_factor)

class _HessianGPT(nn.Module):
    def __init__(self, vocab_size, num_layers, model_dim, num_heads, num_kv_heads,
                 mlp_mult, tie_embeddings, logit_softcap, rope_base, qk_gain_init,
                 xsa_last_n=0, rope_dims=0, ln_scale=False, sigmoid_skip=False,
                 ve_enabled=False, ve_dim=128, ve_layers=None, parallel_start_layer=-1,
                 disable_layer0_attn=False, train_seq_len=2048, per_layer_lane_merge=False):
        super().__init__()
        self.tie_embeddings = tie_embeddings
        self.logit_softcap = logit_softcap
        self.sigmoid_skip = sigmoid_skip
        self.num_layers = num_layers
        self.per_layer_lane_merge = per_layer_lane_merge
        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = nn.Parameter(torch.ones(self.num_skip_weights, model_dim, dtype=torch.float32))
        self.skip_gates = nn.Parameter(torch.zeros(self.num_skip_weights, model_dim, dtype=torch.float32)) if sigmoid_skip else None
        self.blocks = nn.ModuleList([
            _HessianBlock(model_dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init,
                          layer_idx=i, ln_scale=ln_scale, rope_dims=rope_dims, train_seq_len=train_seq_len)
            for i in range(num_layers)
        ])
        if xsa_last_n > 0:
            for i in range(max(0, num_layers - xsa_last_n), num_layers):
                self.blocks[i].attn.use_xsa = True
        self.final_norm = RMSNorm()
        self.lm_head = None if tie_embeddings else CastedLinear(model_dim, vocab_size, bias=False)
        kv_dim = num_kv_heads * (model_dim // num_heads)
        self.ve_layer_indices = sorted(ve_layers) if ve_enabled and ve_layers else []
        if self.ve_layer_indices:
            self.ve_shared = ValueEmbedding(vocab_size, ve_dim, kv_dim)
            self.ve_layer_scales = nn.ParameterList(
                [nn.Parameter(torch.ones(1, dtype=torch.float32)) for _ in self.ve_layer_indices]
            )
        else:
            self.ve_shared = None
            self.ve_layer_scales = nn.ParameterList()
        self.parallel_start_layer = parallel_start_layer
        self.per_layer_lane_merge = per_layer_lane_merge
        if 0 < parallel_start_layer < num_layers:
            n_parallel = num_layers - parallel_start_layer
            if per_layer_lane_merge:
                self.lane_merge = nn.Parameter(torch.full((n_parallel,), 0.5, dtype=torch.float32))
            else:
                self.lane_merge = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))
        else:
            self.lane_merge = None
        if disable_layer0_attn and len(self.blocks) > 0:
            self.blocks[0].disable_attn = True
    def _get_ve(self, phys_idx, input_ids, ve_cache):
        if self.ve_shared is None or phys_idx not in self.ve_layer_indices:
            return None
        if 've' not in ve_cache:
            ve_cache['ve'] = self.ve_shared(input_ids)
        ve_base = ve_cache['ve']
        ve_idx = self.ve_layer_indices.index(phys_idx)
        return ve_base * self.ve_layer_scales[ve_idx].to(dtype=ve_base.dtype)
    def forward(self, input_ids, target_ids):
        x = self.tok_emb(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x
        skips = []
        ve_cache = {}
        parallel_thresh = self.parallel_start_layer if self.lane_merge is not None else self.num_layers + 1
        is_parallel = False
        lane0 = None
        lane1 = None
        for i in range(self.num_encoder_layers):
            ve = self._get_ve(i, input_ids, ve_cache)
            x = self.blocks[i](x, x0, v_embed=ve)
            skips.append(x)
        for i in range(self.num_decoder_layers):
            bi = self.num_encoder_layers + i
            if skips:
                target = lane0 if is_parallel else x
                scaled_skip = self.skip_weights[i].to(dtype=target.dtype)[None, None, :] * skips.pop()
                if self.skip_gates is not None:
                    g = torch.sigmoid(self.skip_gates[i].to(dtype=target.dtype))[None, None, :]
                    target = torch.lerp(scaled_skip, target, g)
                else:
                    target = target + scaled_skip
                if is_parallel:
                    lane0 = target
                else:
                    x = target
            if bi >= parallel_thresh and not is_parallel:
                lane0 = x
                lane1 = x
                is_parallel = True
            if is_parallel:
                block = self.blocks[bi]
                ve = self._get_ve(bi, input_ids, ve_cache)
                if self.per_layer_lane_merge:
                    mix = block.resid_mix.to(dtype=x.dtype)
                    attn_in = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
                    attn_out = block.attn(block.attn_norm(attn_in) * block.ln_scale_factor, v_embed=ve)
                    attn_contrib = block.attn_scale.to(dtype=x.dtype)[None, None, :] * attn_out
                    mlp_in = block.mlp_norm(x) * block.ln_scale_factor
                    mlp_out = block.mlp(mlp_in)
                    mlp_contrib = block.mlp_scale.to(dtype=x.dtype)[None, None, :] * mlp_out
                    m = torch.sigmoid(self.lane_merge[bi - parallel_thresh]).to(dtype=x.dtype)
                    x = x + m * attn_contrib + (1 - m) * mlp_contrib
                else:
                    mix = block.resid_mix.to(dtype=lane0.dtype)
                    attn_in = mix[0][None, None, :] * lane0 + mix[1][None, None, :] * x0
                    attn_out = block.attn(block.attn_norm(attn_in) * block.ln_scale_factor, v_embed=ve)
                    lane0 = attn_in + block.attn_scale.to(dtype=attn_in.dtype)[None, None, :] * attn_out
                    mlp_in = block.mlp_norm(lane1) * block.ln_scale_factor
                    mlp_out = block.mlp(mlp_in)
                    lane1 = lane1 + block.mlp_scale.to(dtype=lane1.dtype)[None, None, :] * mlp_out
            else:
                ve = self._get_ve(bi, input_ids, ve_cache)
                x = self.blocks[bi](x, x0, v_embed=ve)
        if is_parallel and not self.per_layer_lane_merge:
            m = self.lane_merge.to(dtype=lane0.dtype)
            x = m * lane0 + (1 - m) * lane1
        x = self.final_norm(x)
        x_flat = x.reshape(-1, x.size(-1))
        targets = target_ids.reshape(-1)
        logits_proj = F.linear(x_flat, self.tok_emb.weight) if self.tie_embeddings else self.lm_head(x_flat)
        logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
        return F.cross_entropy(logits.float(), targets, reduction="mean")

def _bits_to_clip_range(bits: int) -> int:
    return {5: 15, 6: 31, 8: 127}[bits]

def mixed_quantize_int6(state_dict: dict[str, Tensor], int6_cats: set[str],
                        hessians: dict[str, Tensor] | None = None,
                        bits_mlp: int = 6, bits_attn: int = 6, bits_embed: int = 6,
                        sdclip_k: float = 0.0):
    result: dict[str, Tensor] = {}
    meta: dict[str, object] = {}
    for name, tensor in state_dict.items():
        t = tensor.detach().cpu().contiguous()
        cat = _classify_param(name)
        if not t.is_floating_point() or t.numel() <= 65536:
            result[name] = t.to(torch.float16) if t.is_floating_point() else t
            meta[name] = "passthrough"
            continue
        if any(p in name for p in CONTROL_TENSOR_NAME_PATTERNS):
            result[name] = t.float()
            meta[name] = "passthrough_ctrl"
            continue
        if cat in int6_cats and t.ndim >= 1:
            bits = bits_mlp if cat == "mlp" else (bits_embed if cat == "embed" else bits_attn)
            clip = _bits_to_clip_range(bits)
            H = hessians.get(name) if hessians else None
            if H is not None and t.ndim == 2:
                q, s = quantize_int6_gptq(t, hessian=H, clip_range=clip, sdclip_k=sdclip_k)
            else:
                q, s = quantize_int6_per_row(t, clip_range=clip, sdclip_k=sdclip_k)
            result[name + ".q"] = q
            result[name + ".scale"] = s
            meta[name] = {"type": f"int{bits}"}
        else:
            q, s = quantize_float_tensor(t)
            result[name + ".q"] = q
            result[name + ".scale"] = s
            meta[name] = {"type": "int8"}
    return result, meta

def dequantize_mixed_int6(result: dict[str, Tensor], meta: dict[str, object],
                          template_sd: dict[str, Tensor]) -> dict[str, Tensor]:
    out: dict[str, Tensor] = {}
    for name, orig in template_sd.items():
        info = meta.get(name)
        if info is None:
            continue
        orig_dtype = orig.dtype
        if info in ("passthrough", "passthrough_ctrl", "passthrough_fp16"):
            t = result[name]
            if t.dtype == torch.float16 and orig_dtype in (torch.float32, torch.bfloat16):
                t = t.to(orig_dtype)
            out[name] = t
            continue
        q, s = result[name + ".q"], result[name + ".scale"]
        if s.ndim > 0:
            deq = (q.float() * s.float().view(q.shape[0], *([1] * (q.ndim - 1))))
        else:
            deq = (q.float() * float(s.item()))
        out[name] = deq.to(orig_dtype)
    return out

def main() -> None:
    code = Path(__file__).read_text(encoding="utf-8")
    args = Hyperparameters()
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size <= 0:
        raise ValueError(f"WORLD_SIZE must be positive, got {world_size}")
    if 8 % world_size != 0:
        raise ValueError(f"WORLD_SIZE={world_size} must divide 8 so grad_accum_steps stays integral")
    grad_accum_steps = 8 // world_size
    grad_scale = 1.0 / grad_accum_steps
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    master_process = rank == 0
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp
    enable_cudnn_sdp(False)
    enable_flash_sdp(True)
    enable_mem_efficient_sdp(False)
    enable_math_sdp(False)
    logfile = None
    if master_process:
        os.makedirs("logs", exist_ok=True)
        logfile = f"logs/{args.run_id}.txt"
        print(logfile)
    def log0(msg: str, console: bool = True) -> None:
        if not master_process:
            return
        if console:
            print(msg)
        if logfile is not None:
            with open(logfile, "a", encoding="utf-8") as f:
                print(msg, file=f)
    log0(code, console=False)
    log0("=" * 100, console=False)
    log0(f"Running Python {sys.version}", console=False)
    log0(f"Running PyTorch {torch.__version__}", console=False)
    log0(
        subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False).stdout,
        console=False,
    )
    log0("=" * 100, console=False)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if not args.tokenizer_path.endswith(".model"):
        raise ValueError(f"Script only setup for SentencePiece .model file: {args.tokenizer_path}")
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    if int(sp.vocab_size()) != args.vocab_size:
        raise ValueError(
            f"VOCAB_SIZE={args.vocab_size} does not match tokenizer vocab_size={int(sp.vocab_size())}"
        )
    dataset_dir = Path(args.data_path).resolve()
    actual_train_files = len(list(dataset_dir.glob("fineweb_train_*.bin")))
    effective_eval_seq_len = args.eval_seq_len if args.eval_seq_len > 0 else args.train_seq_len
    val_seq_len = max(args.train_seq_len, effective_eval_seq_len)
    val_tokens = load_validation_tokens(args.val_files, val_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
        sp, args.vocab_size, device
    )
    log0(f"val_bpb:enabled tokenizer_kind=sentencepiece tokenizer_path={args.tokenizer_path}")
    log0(f"train_loader:dataset:{dataset_dir.name} train_shards:{actual_train_files}")
    log0(f"val_loader:shards pattern={args.val_files} tokens:{val_tokens.numel() - 1}")
    recur_layers_list = [int(x) for x in args.recur_layers.split(",") if x.strip()] if args.recur_layers.strip() else []
    ve_layers_list = [int(x) for x in args.ve_layers.split(",") if x.strip()] if args.ve_enabled else []
    base_model = GPT(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings,
        tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap,
        rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init,
        xsa_last_n=args.xsa_last_n,
        rope_dims=args.rope_dims,
        ln_scale=args.ln_scale,
        sigmoid_skip=args.sigmoid_skip,
        recur_layers=recur_layers_list,
        ve_enabled=args.ve_enabled,
        ve_dim=args.ve_dim,
        ve_layers=ve_layers_list,
        parallel_start_layer=args.parallel_start_layer,
        repeat_untie_mlp=args.repeat_untie_mlp,
        disable_layer0_attn=args.disable_layer0_attn,
        recur_num_loops=args.recur_num_loops,
        attn_res=args.attn_res,
        lora_rank=args.repeat_lora_rank,
        repeat_pass_scale=args.repeat_pass_scale,
        train_seq_len=args.train_seq_len,
        recur_skip_physical=args.recur_skip_physical,
        per_layer_lane_merge=args.per_layer_lane_merge,
        repeat_attn_lora=args.repeat_attn_lora,
        attn_lora_rank=args.repeat_attn_lora_rank,
        disable_layer0_mlp=args.disable_layer0_mlp,
        skip_repeat_attn=args.skip_repeat_attn,
        skip_repeat_attn_layers=args.skip_repeat_attn_layers,
    ).to(device).bfloat16()
    base_model.qo_bank.data = base_model.qo_bank.data.float()
    base_model.kv_bank.data = base_model.kv_bank.data.float()
    base_model.mlp_up_bank.data = base_model.mlp_up_bank.data.float()
    base_model.mlp_down_bank.data = base_model.mlp_down_bank.data.float()
    if base_model.repeat_down_bank is not None:
        base_model.repeat_down_bank.data = base_model.repeat_down_bank.data.float()
    if base_model.repeat_lora_A is not None:
        base_model.repeat_lora_A.data = base_model.repeat_lora_A.data.float()
        base_model.repeat_lora_B.data = base_model.repeat_lora_B.data.float()
    if base_model.repeat_attn_lora_A is not None:
        base_model.repeat_attn_lora_A.data = base_model.repeat_attn_lora_A.data.float()
        base_model.repeat_attn_lora_B.data = base_model.repeat_attn_lora_B.data.float()
    for module in base_model.modules():
        if isinstance(module, CastedLinear):
            module.float()
    restore_low_dim_params_to_fp32(base_model)
    compiled_model = torch.compile(base_model, dynamic=False, fullgraph=True)
    model = compiled_model

    matrix_params = [
        base_model.qo_bank, base_model.kv_bank,
        base_model.mlp_up_bank, base_model.mlp_down_bank,
    ]
    if base_model.repeat_down_bank is not None:
        matrix_params.append(base_model.repeat_down_bank)
    block_named_params = list(base_model.blocks.named_parameters())
    scalar_params = [
        p
        for name, p in block_named_params
        if p.ndim < 2 or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)
    ]
    if base_model.skip_weights.numel() > 0:
        scalar_params.append(base_model.skip_weights)
    if base_model.skip_gates is not None:
        scalar_params.append(base_model.skip_gates)
    if base_model.ve_shared is not None:
        for p in base_model.ve_layer_scales.parameters():
            scalar_params.append(p)
        scalar_params.append(base_model.ve_shared.scale)
    if base_model.lane_merge is not None:
        scalar_params.append(base_model.lane_merge)
    if hasattr(base_model, 'ar_queries') and base_model.attn_res:
        scalar_params.append(base_model.ar_queries)
    if base_model.repeat_lora_A is not None:
        scalar_params.append(base_model.repeat_lora_A)
        scalar_params.append(base_model.repeat_lora_B)
    if base_model.repeat_attn_lora_A is not None:
        scalar_params.append(base_model.repeat_attn_lora_A)
        scalar_params.append(base_model.repeat_attn_lora_B)
    if base_model.repeat_pass_scale is not None:
        scalar_params.append(base_model.repeat_pass_scale)
    token_lr = args.tied_embed_lr if args.tie_embeddings else args.embed_lr
    tok_params = [{"params": [base_model.tok_emb.weight], "lr": token_lr, "base_lr": token_lr}]
    embed_wd_val = args.embed_wd if args.embed_wd >= 0 else args.adam_wd
    if base_model.ve_shared is not None:
        ve_matrix_params = [base_model.ve_shared.embed.weight]
        if base_model.ve_shared.proj is not None:
            ve_matrix_params.append(base_model.ve_shared.proj.weight)
        tok_params.append({"params": ve_matrix_params, "lr": token_lr, "base_lr": token_lr})
    optimizer_tok = torch.optim.AdamW(
        tok_params,
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        weight_decay=embed_wd_val,
        fused=True,
    )
    optimizer_muon = Muon(
        matrix_params,
        lr=args.matrix_lr,
        momentum=args.muon_momentum,
        backend_steps=args.muon_backend_steps,
        weight_decay=args.muon_wd,
        post_norm=args.muon_plus,
    )
    for group in optimizer_muon.param_groups:
        group["base_lr"] = args.matrix_lr
    optimizer_scalar = torch.optim.AdamW(
        [{"params": scalar_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr}],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        weight_decay=args.adam_wd,
        fused=True,
    )
    replicated_params = list(optimizer_tok.param_groups[0]["params"])
    for pg in optimizer_tok.param_groups[1:]:
        replicated_params.extend(pg["params"])
    replicated_params.extend(scalar_params)

    optimizer_head = None
    if base_model.lm_head is not None:
        optimizer_head = torch.optim.Adam(
            [{"params": [base_model.lm_head.weight], "lr": args.head_lr, "base_lr": args.head_lr}],
            betas=(args.beta1, args.beta2),
            eps=args.adam_eps,
            fused=True,
        )
        replicated_params.append(base_model.lm_head.weight)
    optimizers: list[torch.optim.Optimizer] = [optimizer_tok, optimizer_muon, optimizer_scalar]
    if optimizer_head is not None:
        optimizers.append(optimizer_head)
    n_params = sum(p.numel() for p in base_model.parameters())
    log0(f"model_params:{n_params}")
    log0(f"arch_config: vocab_size={args.vocab_size} num_layers={args.num_layers} model_dim={args.model_dim} mlp_mult={args.mlp_mult} "
         f"muon_wd={args.muon_wd} adam_wd={args.adam_wd} embed_wd={embed_wd_val} matrix_lr={args.matrix_lr} "
         f"sigmoid_skip={args.sigmoid_skip} byte_shuffle={args.byte_shuffle} "
         f"qk_gain_init={args.qk_gain_init} "
         f"recur_layers={recur_layers_list} recur_start_step={args.recur_start_step} "
         f"ve_enabled={args.ve_enabled} ve_dim={args.ve_dim} ve_layers={ve_layers_list} "
         f"parallel_start_layer={args.parallel_start_layer}")
    xsa_layers = [i for i, b in enumerate(base_model.blocks) if b.attn.use_xsa]
    log0(f"XSA:last_{args.xsa_last_n} active_layers:{xsa_layers}")
    log0(f"world_size:{world_size} grad_accum_steps:{grad_accum_steps}")
    log0("sdp_backends:cudnn=False flash=True mem_efficient=False math=False")
    log0(f"attention_mode:gqa num_heads:{args.num_heads} num_kv_heads:{args.num_kv_heads}")
    log0(
        f"tie_embeddings:{args.tie_embeddings} embed_lr:{token_lr} "
        f"head_lr:{args.head_lr if base_model.lm_head is not None else 0.0} "
        f"matrix_lr:{args.matrix_lr} scalar_lr:{args.scalar_lr}"
    )
    log0(
        f"train_batch_tokens:{args.train_batch_tokens} train_seq_len:{args.train_seq_len} "
        f"iterations:{args.iterations} warmup_steps:{args.warmup_steps} "
        f"max_wallclock_seconds:{args.max_wallclock_seconds:.3f}"
    )
    log0(f"seed:{args.seed}")
    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)
    def zero_grad_all() -> None:
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)
    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None
    def lr_mul(step: int, elapsed_ms: float) -> float:
        if args.warmdown_iters <= 0:
            return 1.0
        if max_wallclock_ms is None:
            warmdown_start = max(args.iterations - args.warmdown_iters, 0)
            return max((args.iterations - step) / max(args.warmdown_iters, 1), 0.0) if warmdown_start <= step < args.iterations else 1.0
        step_ms = elapsed_ms / max(step, 1)
        warmdown_ms = args.warmdown_iters * step_ms
        remaining_ms = max(max_wallclock_ms - elapsed_ms, 0.0)
        return remaining_ms / max(warmdown_ms, 1e-9) if remaining_ms <= warmdown_ms else 1.0
    if args.warmup_steps > 0:
        initial_model_state = {name: tensor.detach().cpu().clone() for name, tensor in base_model.state_dict().items()}
        initial_optimizer_states = [copy.deepcopy(opt.state_dict()) for opt in optimizers]
        model.train()
        for warmup_step in range(args.warmup_steps):
            zero_grad_all()
            for micro_step in range(grad_accum_steps):
                x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    warmup_loss = model(x, y)
                (warmup_loss * grad_scale).backward()
            if distributed:
                for p in base_model.parameters():
                    if p.grad is not None:
                        dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
            for opt in optimizers:
                opt.step()
            zero_grad_all()
            if args.warmup_steps <= 20 or (warmup_step + 1) % 10 == 0 or warmup_step + 1 == args.warmup_steps:
                log0(f"warmup_step:{warmup_step + 1}/{args.warmup_steps}")
        base_model.load_state_dict(initial_model_state, strict=True)
        for opt, state in zip(optimizers, initial_optimizer_states, strict=True):
            opt.load_state_dict(state)
        zero_grad_all()
        train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)
    swa_state: dict[str, Tensor] | None = None
    swa_count = 0
    ema_state = {name: t.detach().float().clone() for name, t in base_model.state_dict().items()}
    ema_decay = float(os.environ.get("EMA_DECAY", 0.997))
    training_time_ms = 0.0
    stop_after_step: int | None = None
    # Collect parallel block scalar params for gradient scaling
    _parallel_scalar_params: list[nn.Parameter] = []
    if args.parallel_scalar_grad_scale > 0 and args.parallel_start_layer > 0:
        for li in range(args.parallel_start_layer, args.num_layers):
            blk = base_model.blocks[li]
            _parallel_scalar_params.extend([blk.attn_scale, blk.mlp_scale, blk.resid_mix])
            if hasattr(blk.attn, 'q_gain'):
                _parallel_scalar_params.append(blk.attn.q_gain)
        log0(f"parallel_scalar_grad_scale={args.parallel_scalar_grad_scale} on {len(_parallel_scalar_params)} params (blocks {args.parallel_start_layer}-{args.num_layers-1})")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    step = 0
    while True:
        last_step = step == args.iterations or (stop_after_step is not None and step >= stop_after_step)
        # Activate recurrence at recur_start_step
        if recur_layers_list and step == args.recur_start_step and not base_model._recurrence_active:
            base_model.prime_repeat_mlp()
            base_model.set_recurrence_active(True)
            log0(f"recurrence:activated at step {step}, virtual_layers={base_model._get_virtual_layers()}")
        should_validate = last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0)
        if should_validate:
            torch.cuda.synchronize()
            training_time_ms += 1000.0 * (time.perf_counter() - t0)
            val_loss, val_bpb = eval_val(
                args,
                model,
                rank,
                world_size,
                device,
                grad_accum_steps,
                val_tokens,
                base_bytes_lut,
                has_leading_space_lut,
                is_boundary_token_lut,
            )
            log0(
                f"step:{step}/{args.iterations} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} "
                f"train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms / max(step, 1):.2f}ms"
            )
            torch.cuda.synchronize()
            t0 = time.perf_counter()
        if last_step:
            if stop_after_step is not None and step < args.iterations:
                log0(
                    f"stopping_early: wallclock_cap train_time:{training_time_ms:.0f}ms "
                    f"step:{step}/{args.iterations}"
                )
            break
        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        scale = lr_mul(step, elapsed_ms)
        train_loader.set_warmdown_frac(max(1.0 - scale, 0.0))
        zero_grad_all()
        train_loss = torch.zeros((), device=device)
        for micro_step in range(grad_accum_steps):
            x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                loss = model(x, y)
            train_loss += loss.detach()
            (loss * grad_scale).backward()
        train_loss /= grad_accum_steps
        frac = min(step / args.muon_momentum_warmup_steps, 1.0) if args.muon_momentum_warmup_steps > 0 else 1.0
        muon_momentum = (1 - frac) * args.muon_momentum_warmup_start + frac * args.muon_momentum
        for group in optimizer_muon.param_groups:
            group["momentum"] = muon_momentum
        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["base_lr"] * scale
        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)
        if _parallel_scalar_params:
            with torch.no_grad():
                for p in _parallel_scalar_params:
                    if p.grad is not None:
                        p.grad.mul_(args.parallel_scalar_grad_scale)
        optimizer_muon.launch_reduce_scatters()
        if distributed:
            for p in replicated_params:
                if p.grad is not None:
                    dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
        optimizer_tok.step()
        optimizer_scalar.step()
        if optimizer_head is not None:
            optimizer_head.step()
        optimizer_muon.step()
        zero_grad_all()
        # Late QAT: add quantization-grid-aligned noise to 2D weight matrices
        if args.late_qat and scale <= (1.0 - args.qat_start_frac):
            with torch.no_grad():
                for p in base_model.parameters():
                    if p.ndim == 2 and p.numel() >= 256:
                        row_max = p.float().abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)
                        grid_step = row_max / 31.0  # int6 has 2^5=32 positive levels
                        noise = (torch.rand_like(p, dtype=torch.float32) - 0.5) * grid_step * args.qat_noise_scale
                        p.add_(noise.to(p.dtype))
        with torch.no_grad():
            for name, t in base_model.state_dict().items():
                ema_state[name].mul_(ema_decay).add_(t.detach().float(), alpha=1.0 - ema_decay)
        step += 1
        approx_training_time_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        if args.swa_enabled and scale < 0.2 and step % args.swa_every == 0:
            if swa_state is None:
                swa_state = {name: t.detach().cpu().clone() for name, t in base_model.state_dict().items()}
                swa_count = 1
                log0(f"swa:start step:{step}")
            else:
                for name, t in base_model.state_dict().items():
                    swa_state[name] += t.detach().cpu()
                swa_count += 1
        should_log_train = (
            args.train_log_every > 0
            and (step <= 10 or step % args.train_log_every == 0 or stop_after_step is not None)
        )
        if should_log_train:
            log0(
                f"step:{step}/{args.iterations} train_loss:{train_loss.item():.4f} "
                f"train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms / step:.2f}ms"
            )
        reached_cap = max_wallclock_ms is not None and approx_training_time_ms >= max_wallclock_ms
        if distributed and max_wallclock_ms is not None:
            reached_cap_tensor = torch.tensor(int(reached_cap), device=device)
            dist.all_reduce(reached_cap_tensor, op=dist.ReduceOp.MAX)
            reached_cap = bool(reached_cap_tensor.item())
        if stop_after_step is None and reached_cap:
            stop_after_step = step
    log0(
        f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
        f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB"
    )
    # Apply final weights: EMA (default) or SWA (if USE_SWA_WEIGHTS=1 and enough snapshots)
    use_swa = bool(int(os.environ.get("USE_SWA_WEIGHTS", "0")))
    current_state = base_model.state_dict()
    if use_swa and swa_state is not None and swa_count >= 5:
        log0(f"swa:applying {swa_count} snapshots as final weights")
        avg_state = {}
        for name in current_state:
            if name in swa_state:
                avg_state[name] = (swa_state[name].float() / swa_count).to(dtype=current_state[name].dtype)
            else:
                avg_state[name] = ema_state[name].to(dtype=current_state[name].dtype)
    else:
        if use_swa:
            log0(f"swa:requested but only {swa_count} snapshots — falling back to EMA")
        log0("ema:applying EMA weights")
        avg_state = {name: t.to(dtype=current_state[name].dtype) for name, t in ema_state.items()}
    base_model.load_state_dict(avg_state, strict=True)
    torch.cuda.synchronize()
    t_diag = time.perf_counter()
    diag_val_loss, diag_val_bpb = eval_val(
        args, compiled_model, rank, world_size, device, grad_accum_steps,
        val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
    )
    torch.cuda.synchronize()
    weight_type = "swa" if (use_swa and swa_state is not None and swa_count >= 5) else "ema"
    log0(
        f"DIAGNOSTIC post_{weight_type} val_loss:{diag_val_loss:.4f} val_bpb:{diag_val_bpb:.4f} "
        f"eval_time:{1000.0 * (time.perf_counter() - t_diag):.0f}ms"
    )
    full_state_dict = base_model.state_dict()
    export_sd = dict(full_state_dict)
    if master_process:
        torch.save(export_sd, "final_model.pt")
        model_bytes = os.path.getsize("final_model.pt")
        code_bytes = len(code.encode("utf-8"))
        log0(f"Serialized model: {model_bytes} bytes")
        log0(f"Code size: {code_bytes} bytes")
    sd_cpu = {k: v.detach().cpu() for k, v in export_sd.items()}
    unbanked_sd = _unbank_state_dict(sd_cpu, args.num_layers)
    hessians = None
    if args.gptq_enabled and master_process:
        log0("gptq:building non-banked model for Hessian collection...")
        hessian_model = _HessianGPT(
            vocab_size=args.vocab_size, num_layers=args.num_layers, model_dim=args.model_dim,
            num_heads=args.num_heads, num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
            tie_embeddings=args.tie_embeddings, logit_softcap=args.logit_softcap,
            rope_base=args.rope_base, qk_gain_init=args.qk_gain_init,
            xsa_last_n=args.xsa_last_n, rope_dims=args.rope_dims, ln_scale=args.ln_scale,
            sigmoid_skip=args.sigmoid_skip,
            ve_enabled=args.ve_enabled, ve_dim=args.ve_dim, ve_layers=ve_layers_list,
            parallel_start_layer=args.parallel_start_layer,
            disable_layer0_attn=args.disable_layer0_attn,
            train_seq_len=args.train_seq_len,
            per_layer_lane_merge=args.per_layer_lane_merge,
        ).to(device).bfloat16()
        for m in hessian_model.modules():
            if isinstance(m, CastedLinear):
                m.float()
        restore_low_dim_params_to_fp32(hessian_model)
        hessian_model.load_state_dict(
            {k: v.to(device) for k, v in unbanked_sd.items() if k in hessian_model.state_dict()},
            strict=False,
        )
        log0(f"gptq:generating AR calibration data ({args.gptq_calib_seqs} seqs x {args.gptq_calib_len} tokens, temp={args.gptq_temperature})...")
        t_gen = time.perf_counter()
        ar_tokens = generate_autoregressive_calib(
            base_model, device, num_seqs=args.gptq_calib_seqs, seq_len=args.gptq_calib_len,
            vocab_size=args.vocab_size, temperature=args.gptq_temperature, batch_size=8, seed=args.seed,
        )
        log0(f"gptq:generated {len(ar_tokens)} sequences in {time.perf_counter()-t_gen:.1f}s")
        log0("gptq:collecting hessians from AR data...")
        hessians = collect_hessians_from_tokens(hessian_model, ar_tokens, device)
        log0(f"gptq:collected hessians for {len(hessians)} layers")
        del ar_tokens, hessian_model
        torch.cuda.empty_cache()
    int6_cats = {"mlp", "attn"}
    if args.quant_bits_embed <= 6:
        int6_cats.add("embed")
    log0(f"quant_config: bits_mlp={args.quant_bits_mlp} bits_attn={args.quant_bits_attn} bits_embed={args.quant_bits_embed}")
    quant_result, quant_meta = mixed_quantize_int6(
        unbanked_sd, int6_cats, hessians=hessians,
        bits_mlp=args.quant_bits_mlp, bits_attn=args.quant_bits_attn, bits_embed=args.quant_bits_embed,
        sdclip_k=args.sdclip_k if args.sdclip else 0.0,
    )
    code_bytes_est = int(os.environ.get("CODE_SIZE_OVERRIDE", 0)) or len(code.encode("utf-8"))
    ARTIFACT_LIMIT = 16_000_000  # 16 MB decimal (competition rule)
    target_bytes = ARTIFACT_LIMIT - code_bytes_est
    log0(f"budget: artifact_limit={ARTIFACT_LIMIT} code={code_bytes_est} model_budget={target_bytes} ({target_bytes/(1024*1024):.2f}MB)")
    ones_info = []
    for name, info in quant_meta.items():
        if not (isinstance(info, dict) and info.get("type", "").startswith("int")):
            continue
        qk, sk = name + ".q", name + ".scale"
        if qk not in quant_result or sk not in quant_result:
            continue
        q, s = quant_result[qk], quant_result[sk]
        if s.ndim > 0:
            ones_mask = (q.abs() == 1)
            if ones_mask.any():
                row_idx = torch.arange(q.shape[0]).unsqueeze(1).expand_as(q)[ones_mask]
                flat_idx = torch.arange(q.numel()).reshape(q.shape)[ones_mask]
                errors = s.float()[row_idx].pow(2)
                for fi, err in zip(flat_idx.tolist(), errors.tolist()):
                    ones_info.append((qk, fi, err))
    def _byte_shuffle(data: bytes) -> bytes:
        arr = np.frombuffer(data, dtype=np.uint8)
        pad = (4 - len(arr) % 4) % 4
        if pad:
            arr = np.concatenate([arr, np.zeros(pad, dtype=np.uint8)])
        return arr.reshape(-1, 4).T.tobytes() + pad.to_bytes(1, 'little')

    def _byte_unshuffle(data: bytes) -> bytes:
        pad = data[-1]
        arr = np.frombuffer(data[:-1], dtype=np.uint8)
        n_elements = len(arr) // 4
        result = arr.reshape(4, n_elements).T.tobytes()
        if pad:
            result = result[:-pad]
        return result

    def _compress(data: bytes) -> bytes:
        if args.byte_shuffle:
            data = _byte_shuffle(data)
        if args.compressor == "brotli" and _HAS_BROTLI:
            return brotli.compress(data, quality=11)
        return lzma.compress(data, preset=9)

    def _decompress(data: bytes) -> bytes:
        if args.compressor == "brotli" and _HAS_BROTLI:
            raw = brotli.decompress(data)
        else:
            raw = lzma.decompress(data)
        if args.byte_shuffle:
            raw = _byte_unshuffle(raw)
        return raw

    if ones_info:
        ones_info.sort(key=lambda x: x[2])
        def _try_prune(n):
            tmp = {k: v.clone() for k, v in quant_result.items()}
            for i in range(min(n, len(ones_info))):
                tmp[ones_info[i][0]].view(-1)[ones_info[i][1]] = 0
            buf = io.BytesIO()
            torch.save({"w": tmp, "m": quant_meta}, buf)
            return len(_compress(buf.getvalue())), tmp
        no_sz, _ = _try_prune(0)
        log0(f"selective_prune: {len(ones_info)} +/-1 candidates, model_unpruned={no_sz/(1024*1024):.2f}MB model_budget={target_bytes/(1024*1024):.2f}MB")
        if no_sz <= target_bytes:
            log0("selective_prune: already fits, no pruning needed")
        else:
            full_sz, _ = _try_prune(len(ones_info))
            log0(f"selective_prune: full +/-1 prune={full_sz/(1024*1024):.2f}MB")
            if full_sz > target_bytes:
                log0("selective_prune: even full prune not enough, applying all")
                _, quant_result = _try_prune(len(ones_info))
            else:
                lo, hi = 0, len(ones_info)
                while lo < hi:
                    mid = (lo + hi) // 2
                    sz, _ = _try_prune(mid)
                    if sz <= target_bytes:
                        hi = mid
                    else:
                        lo = mid + 1
                log0(f"selective_prune: pruning {lo}/{len(ones_info)} +/-1 values ({100*lo/len(ones_info):.1f}%)")
                _, quant_result = _try_prune(lo)
    quant_buf = io.BytesIO()
    torch.save({"w": quant_result, "m": quant_meta}, quant_buf)
    quant_raw = quant_buf.getvalue()
    quant_blob = _compress(quant_raw)
    if master_process:
        with open("final_model.int6.ptz", "wb") as f:
            f.write(quant_blob)
        quant_file_bytes = len(quant_blob)
        code_bytes = len(code.encode("utf-8"))
        log0(f"Serialized model {args.compressor}: {quant_file_bytes} bytes")
        log0(f"Total submission size {args.compressor}: {quant_file_bytes + code_bytes} bytes")
    if distributed:
        dist.barrier()
    with open("final_model.int6.ptz", "rb") as f:
        quant_blob_disk = f.read()
    quant_state = torch.load(
        io.BytesIO(_decompress(quant_blob_disk)),
        map_location="cpu",
    )
    deq_unbanked = dequantize_mixed_int6(quant_state["w"], quant_state["m"], unbanked_sd)
    deq_state = _rebank_state_dict(deq_unbanked, args.num_layers, sd_cpu)
    eval_model = GPT(
        vocab_size=args.vocab_size, num_layers=args.num_layers, model_dim=args.model_dim,
        num_heads=args.num_heads, num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings, tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap, rope_base=args.rope_base, qk_gain_init=args.qk_gain_init,
        xsa_last_n=args.xsa_last_n, rope_dims=args.rope_dims, ln_scale=args.ln_scale,
        sigmoid_skip=args.sigmoid_skip,
        recur_layers=recur_layers_list,
        ve_enabled=args.ve_enabled, ve_dim=args.ve_dim, ve_layers=ve_layers_list,
        parallel_start_layer=args.parallel_start_layer,
        repeat_untie_mlp=args.repeat_untie_mlp,
        disable_layer0_attn=args.disable_layer0_attn,
        recur_num_loops=args.recur_num_loops,
        attn_res=args.attn_res,
        lora_rank=args.repeat_lora_rank,
        repeat_pass_scale=args.repeat_pass_scale,
        train_seq_len=args.train_seq_len,
        recur_skip_physical=args.recur_skip_physical,
        per_layer_lane_merge=args.per_layer_lane_merge,
        repeat_attn_lora=args.repeat_attn_lora,
        attn_lora_rank=args.repeat_attn_lora_rank,
        disable_layer0_mlp=args.disable_layer0_mlp,
        skip_repeat_attn=args.skip_repeat_attn,
        skip_repeat_attn_layers=args.skip_repeat_attn_layers,
    ).to(device).bfloat16()
    eval_model.qo_bank.data = eval_model.qo_bank.data.float()
    eval_model.kv_bank.data = eval_model.kv_bank.data.float()
    eval_model.mlp_up_bank.data = eval_model.mlp_up_bank.data.float()
    eval_model.mlp_down_bank.data = eval_model.mlp_down_bank.data.float()
    if eval_model.repeat_down_bank is not None:
        eval_model.repeat_down_bank.data = eval_model.repeat_down_bank.data.float()
    if eval_model.repeat_lora_A is not None:
        eval_model.repeat_lora_A.data = eval_model.repeat_lora_A.data.float()
        eval_model.repeat_lora_B.data = eval_model.repeat_lora_B.data.float()
    if eval_model.repeat_attn_lora_A is not None:
        eval_model.repeat_attn_lora_A.data = eval_model.repeat_attn_lora_A.data.float()
        eval_model.repeat_attn_lora_B.data = eval_model.repeat_attn_lora_B.data.float()
    for m in eval_model.modules():
        if isinstance(m, CastedLinear):
            m.float()
    restore_low_dim_params_to_fp32(eval_model)
    eval_model.load_state_dict(deq_state, strict=True)
    eval_model.set_recurrence_active(base_model._recurrence_active)
    compiled_eval = torch.compile(eval_model, dynamic=False, fullgraph=True)
    torch.cuda.synchronize()
    t_qeval = time.perf_counter()
    q_val_loss, q_val_bpb = eval_val(
        args, compiled_eval, rank, world_size, device, grad_accum_steps,
        val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
        eval_seq_len=effective_eval_seq_len,
    )
    torch.cuda.synchronize()
    log0(
        f"final_int6_roundtrip val_loss:{q_val_loss:.4f} val_bpb:{q_val_bpb:.4f} "
        f"eval_time:{1000.0 * (time.perf_counter() - t_qeval):.0f}ms"
    )
    log0(f"final_int6_roundtrip_exact val_loss:{q_val_loss:.8f} val_bpb:{q_val_bpb:.8f}")
    if distributed:
        dist.destroy_process_group()
if __name__ == "__main__":
    main()
