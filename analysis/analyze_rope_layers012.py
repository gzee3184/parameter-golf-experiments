"""
Analyze how much layers 0-2 depend on positional information.

Three tasks:
1. RoPE fraction calculation (analytic).
2. q_gain inspection for layers 0-2.
3. Forward-pass experiment: compare layer-2 output with and without RoPE
   on layers 0-2 (keeping RoPE active for layers 3+).
"""
from __future__ import annotations
import os, sys, math
import torch
import torch.nn.functional as F
from torch import Tensor, nn
import numpy as np
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["DATA_PATH"] = "./parameter-golf/data/datasets/fineweb10B_sp4096"
os.environ["TOKENIZER_PATH"] = "./parameter-golf/data/tokenizers/fineweb_4096_bpe.model"
os.environ["VOCAB_SIZE"] = "4096"
os.environ["MLP_MULT"] = "4"
os.environ["RECUR_LAYERS"] = "3,4,5"

sys.path.insert(0, "./h100_final")
sys.path.insert(0, "./venv/lib/python3.12/site-packages")

from train_gpt_v6 import GPT, Hyperparameters

ROPE_DIMS = 16
HEAD_DIM   = 64

rope_frac = ROPE_DIMS / HEAD_DIM
print("=" * 70)
print("SECTION 1 - RoPE fraction analysis")
print("=" * 70)
print(f"rope_dims={ROPE_DIMS}, head_dim={HEAD_DIM}")
print(f"Position-sensitive QK dims: {ROPE_DIMS}/{HEAD_DIM} = {rope_frac:.4f}  ({rope_frac*100:.1f}%)")
print(f"Position-independent QK dims: {HEAD_DIM - ROPE_DIMS}/{HEAD_DIM} = {(HEAD_DIM-ROPE_DIMS)/HEAD_DIM:.4f}  ({(1-rope_frac)*100:.1f}%)")

print()
print("=" * 70)
print("SECTION 2 - q_gain inspection (layers 0-2)")
print("=" * 70)

device = torch.device("cuda", 0)

args = Hyperparameters()
recur_layers_list = [int(x) for x in args.recur_layers.split(",") if x.strip()]
ve_layers_list    = [int(x) for x in args.ve_layers.split(",") if x.strip()] if args.ve_enabled else []

model = GPT(
    vocab_size          = args.vocab_size,
    num_layers          = args.num_layers,
    model_dim           = args.model_dim,
    num_heads           = args.num_heads,
    num_kv_heads        = args.num_kv_heads,
    mlp_mult            = args.mlp_mult,
    tie_embeddings      = args.tie_embeddings,
    tied_embed_init_std = args.tied_embed_init_std,
    logit_softcap       = args.logit_softcap,
    rope_base           = args.rope_base,
    qk_gain_init        = args.qk_gain_init,
    xsa_last_n          = args.xsa_last_n,
    rope_dims           = args.rope_dims,
    ln_scale            = args.ln_scale,
    sigmoid_skip        = args.sigmoid_skip,
    recur_layers        = recur_layers_list,
    ve_enabled          = args.ve_enabled,
    ve_dim              = args.ve_dim,
    ve_layers           = ve_layers_list,
    parallel_start_layer = args.parallel_start_layer,
    repeat_untie_mlp    = args.repeat_untie_mlp,
    disable_layer0_attn = args.disable_layer0_attn,
    recur_num_loops     = args.recur_num_loops,
    attn_res            = args.attn_res,
    lora_rank           = args.repeat_lora_rank,
    repeat_pass_scale   = args.repeat_pass_scale,
).to(device).bfloat16()

ckpt_path = "./parameter-golf/final_model.pt"
state = torch.load(ckpt_path, map_location=device, weights_only=True)
missing, unexpected = model.load_state_dict(state, strict=False)
print(f"Checkpoint loaded. Missing keys: {len(missing)}, Unexpected: {len(unexpected)}")
if missing:
    print(f"  First 5 missing: {missing[:5]}")

model.eval()

for layer_idx in range(3):
    block = model.blocks[layer_idx]
    attn_disabled = getattr(block, 'disable_attn', False)
    if attn_disabled:
        print(f"Layer {layer_idx}: attention DISABLED (q_gain not used)")
    else:
        gain = block.attn.q_gain.detach().float()
        print(f"Layer {layer_idx}: q_gain  min={gain.min().item():.4f}  max={gain.max().item():.4f}  "
              f"mean={gain.mean().item():.4f}  std={gain.std().item():.4f}")
        print(f"           per-head: {[f'{v:.3f}' for v in gain.tolist()]}")
        if gain.mean().item() > 3.0:
            print(f"           => SHARP attention (local/induction-head style)")
        elif gain.mean().item() > 1.5:
            print(f"           => MODERATE sharpness")
        else:
            print(f"           => DIFFUSE attention (global)")

# Also show a few other layers for context
print()
print("Context: q_gain means for all layers:")
for layer_idx in range(args.num_layers):
    block = model.blocks[layer_idx]
    attn_disabled = getattr(block, 'disable_attn', False)
    if attn_disabled:
        print(f"  Layer {layer_idx:2d}: ATTN_DISABLED")
    else:
        gain = block.attn.q_gain.detach().float()
        print(f"  Layer {layer_idx:2d}: mean={gain.mean().item():.4f}  min={gain.min().item():.4f}  max={gain.max().item():.4f}")

print()
print("=" * 70)
print("SECTION 3 - RoPE ablation experiment (layers 0-2)")
print("=" * 70)

val_file = Path("./parameter-golf/data/datasets/fineweb10B_sp4096/fineweb_val_000000.bin")
header_bytes = 256 * np.dtype("<i4").itemsize
header = np.fromfile(val_file, dtype="<i4", count=256)
num_tokens = int(header[2])
tokens_np = np.fromfile(val_file, dtype="<u2", count=num_tokens, offset=header_bytes)
tokens = torch.from_numpy(tokens_np.astype(np.int64))
SEQ_LEN = 2048
N_SEQS = 4
x_val = tokens[:N_SEQS * SEQ_LEN].reshape(N_SEQS, SEQ_LEN).to(device)
print(f"Using {N_SEQS} sequences of length {SEQ_LEN} from validation set")
print(f"Token range: [{x_val.min().item()}, {x_val.max().item()}] (vocab_size={args.vocab_size})")


def run_forward_capture_layer2(model_obj, x_in, ablate_rope_layers):
    """
    Run forward pass, capturing output after physical layer 2.
    For ablate_rope_layers: temporarily replace rotary cos/sin with identity.
    Returns (final_normed_output, layer2_output).
    """
    # First: populate rotary cache for all layers
    T = x_in.shape[1]
    half = ROPE_DIMS // 2
    for li in range(model_obj.num_layers):
        model_obj.blocks[li].attn.rotary(T, device, torch.bfloat16)

    # Save and patch
    saved = {}
    for li in ablate_rope_layers:
        rot = model_obj.blocks[li].attn.rotary
        saved[li] = (rot._cos_cached, rot._sin_cached)
        rot._cos_cached = torch.ones(1, T, 1, half, dtype=torch.bfloat16, device=device)
        rot._sin_cached = torch.zeros(1, T, 1, half, dtype=torch.bfloat16, device=device)

    layer2_out = [None]

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        x = model_obj.tok_emb(x_in)
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x

        virtual_layers = model_obj._get_virtual_layers()
        num_virtual = len(virtual_layers)
        num_enc = num_virtual // 2
        skips = []
        ve_cache = {}
        ar_layers = [x] if model_obj.attn_res else []

        for vi in range(num_enc):
            phys_idx, is_repeat = virtual_layers[vi]
            if model_obj.attn_res:
                x = model_obj._attn_res_input(ar_layers, vi)
            q_w, k_w, v_w, out_w, up_w, down_w = model_obj._bank_weights(phys_idx)
            ve = model_obj._get_ve(phys_idx, x_in, ve_cache)
            dw_override = model_obj._get_repeat_down_w(phys_idx) if is_repeat else None
            x_prev = x if is_repeat else None
            x = model_obj.blocks[phys_idx](x, x0, q_w, k_w, v_w, out_w, up_w, down_w, v_embed=ve, down_w_override=dw_override)
            if is_repeat and x_prev is not None:
                x = model_obj._apply_repeat_scale(x_prev, x, phys_idx)
            if model_obj.attn_res:
                ar_layers.append(x)
            skips.append(x)
            if phys_idx == 2:
                layer2_out[0] = x.detach().float().clone()

        for vi in range(num_virtual - num_enc):
            phys_idx, is_repeat = virtual_layers[num_enc + vi]
            if skips and vi < model_obj.num_skip_weights:
                scaled_skip = model_obj.skip_weights[vi].to(dtype=x.dtype)[None, None, :] * skips.pop()
                if model_obj.skip_gates is not None:
                    g = torch.sigmoid(model_obj.skip_gates[vi].to(dtype=x.dtype))[None, None, :]
                    x = torch.lerp(scaled_skip, x, g)
                else:
                    x = x + scaled_skip
            if model_obj.attn_res:
                x = model_obj._attn_res_input(ar_layers, num_enc + vi)
            q_w, k_w, v_w, out_w, up_w, down_w = model_obj._bank_weights(phys_idx)
            ve = model_obj._get_ve(phys_idx, x_in, ve_cache)
            dw_override = model_obj._get_repeat_down_w(phys_idx) if is_repeat else None
            x_prev = x if is_repeat else None
            x = model_obj.blocks[phys_idx](x, x0, q_w, k_w, v_w, out_w, up_w, down_w, v_embed=ve, down_w_override=dw_override)
            if is_repeat and x_prev is not None:
                x = model_obj._apply_repeat_scale(x_prev, x, phys_idx)
            if model_obj.attn_res:
                ar_layers.append(x)

        x = model_obj.final_norm(x)

    # Restore
    for li, (cos_orig, sin_orig) in saved.items():
        rot = model_obj.blocks[li].attn.rotary
        rot._cos_cached = cos_orig
        rot._sin_cached = sin_orig

    return x.detach().float().clone(), layer2_out[0]


print("\nRunning normal forward (RoPE enabled everywhere)...")
out_normal, layer2_normal = run_forward_capture_layer2(model, x_val, ablate_rope_layers=[])
print(f"layer2_normal shape: {layer2_normal.shape}")
print(f"layer2_normal norm (mean): {layer2_normal.norm(dim=-1).mean().item():.4f}")

print("\nRunning ablated forward (RoPE DISABLED for layers 0-2)...")
out_ablated, layer2_ablated = run_forward_capture_layer2(model, x_val, ablate_rope_layers=[0, 1, 2])
print(f"layer2_ablated norm (mean): {layer2_ablated.norm(dim=-1).mean().item():.4f}")

D = layer2_normal.shape[-1]

# Layer-2 comparison
diff_l2 = (layer2_normal - layer2_ablated)
rms_diff_l2 = diff_l2.pow(2).mean().sqrt().item()
rms_normal_l2 = layer2_normal.pow(2).mean().sqrt().item()
nrmsd_l2 = rms_diff_l2 / rms_normal_l2
sim_l2 = F.cosine_similarity(
    layer2_normal.reshape(-1, D),
    layer2_ablated.reshape(-1, D),
    dim=-1
)
mad_l2 = diff_l2.abs().mean().item()

print()
print("Layer-2 output: normal vs ablated")
print(f"  MAD:                               {mad_l2:.6f}")
print(f"  RMS diff:                          {rms_diff_l2:.6f}")
print(f"  RMS normal:                        {rms_normal_l2:.6f}")
print(f"  NRMSD (rms_diff/rms_normal):       {nrmsd_l2:.6f}  ({100*nrmsd_l2:.2f}%)")
print(f"  Cosine similarity (mean):          {sim_l2.mean().item():.6f}")
print(f"  Cosine similarity (min):           {sim_l2.min().item():.6f}")
print(f"  Cosine similarity (std):           {sim_l2.std().item():.6f}")

# Final output comparison
diff_fin = (out_normal - out_ablated)
rms_diff_fin = diff_fin.pow(2).mean().sqrt().item()
rms_normal_fin = out_normal.pow(2).mean().sqrt().item()
nrmsd_fin = rms_diff_fin / rms_normal_fin
sim_fin = F.cosine_similarity(
    out_normal.reshape(-1, D),
    out_ablated.reshape(-1, D),
    dim=-1
)

print()
print("FINAL output (after all layers): normal vs ablated")
print(f"  NRMSD (rms_diff/rms_normal):       {nrmsd_fin:.6f}  ({100*nrmsd_fin:.2f}%)")
print(f"  Cosine similarity (mean):          {sim_fin.mean().item():.6f}")

# Positional variance analysis
pos_var_normal  = layer2_normal.var(dim=1).mean().item()
pos_var_ablated = layer2_ablated.var(dim=1).mean().item()
print()
print("Positional variance in layer-2 output (variance over token positions):")
print(f"  Normal:   {pos_var_normal:.6f}")
print(f"  Ablated:  {pos_var_ablated:.6f}")
print(f"  Ratio (ablated/normal): {pos_var_ablated/pos_var_normal:.4f}")
print()
print("  If ablating RoPE collapses positional variance, that indicates layers 0-2")
print("  rely heavily on RoPE for position discrimination.")

print()
print("=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"1. RoPE covers {ROPE_DIMS}/{HEAD_DIM} = {100*rope_frac:.0f}% of head dimensions.")
print(f"   25% of QK dot-product is position-sensitive; 75% is position-independent.")
print()
print("2. q_gain for active attention layers in 0-2:")
for layer_idx in range(3):
    block = model.blocks[layer_idx]
    if getattr(block, 'disable_attn', False):
        print(f"   Layer {layer_idx}: ATTENTION DISABLED")
    else:
        gain = block.attn.q_gain.detach().float()
        print(f"   Layer {layer_idx}: q_gain mean={gain.mean().item():.3f}  "
              f"range=[{gain.min().item():.3f}, {gain.max().item():.3f}]")
print()
print("3. RoPE ablation (layers 0-2 only):")
print(f"   Layer-2 output NRMSD:  {nrmsd_l2:.4f} ({100*nrmsd_l2:.2f}%)")
print(f"   Layer-2 cos-sim mean:  {sim_l2.mean().item():.6f}")
print(f"   Final output NRMSD:    {nrmsd_fin:.4f} ({100*nrmsd_fin:.2f}%)")
print(f"   Final cos-sim mean:    {sim_fin.mean().item():.6f}")
if nrmsd_l2 < 0.05:
    verdict = "NOT significantly position-dependent (NRMSD < 5%)"
elif nrmsd_l2 < 0.15:
    verdict = "MILDLY position-dependent (NRMSD 5-15%)"
else:
    verdict = "SIGNIFICANTLY position-dependent (NRMSD > 15%)"
print(f"   => Layers 0-2 are {verdict}.")
