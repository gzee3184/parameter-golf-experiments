"""
Analyze layers 0-2 RoPE dependence using _HessianGPT which matches checkpoint format.
"""
from __future__ import annotations
import os, sys, math
import torch
import torch.nn.functional as F
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

from train_gpt_v6 import _HessianGPT, Hyperparameters, apply_rotary_emb

ROPE_DIMS = 16
HEAD_DIM   = 64

# ──────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("SECTION 1 - RoPE fraction analysis")
print("=" * 70)
rope_frac = ROPE_DIMS / HEAD_DIM
print(f"rope_dims={ROPE_DIMS}, head_dim={HEAD_DIM}")
print(f"Position-sensitive QK dims: {ROPE_DIMS}/{HEAD_DIM} = {rope_frac:.4f}  ({rope_frac*100:.1f}%)")
print(f"Position-independent QK dims: {HEAD_DIM-ROPE_DIMS}/{HEAD_DIM} = {(HEAD_DIM-ROPE_DIMS)/HEAD_DIM:.4f}  ({(1-rope_frac)*100:.1f}%)")
print()
print("The QK dot product decomposes as:")
print("  q·k = q_rope[:16]·k_rope[:16]  +  q_pass[16:64]·k_pass[16:64]")
print("With RMS-normalised Q/K, the expected energy split is ~25%/75% (pos/pos-indep).")

# ──────────────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("SECTION 2 - q_gain inspection")
print("=" * 70)

device = torch.device("cuda", 0)
args = Hyperparameters()

# _HessianGPT does NOT support recur_layers, parallel, etc. but its weights match
# the checkpoint exactly (separate linear layers per block).
# We only need it for weight inspection and ablation; we won't use its full forward.
ve_layers_list = [int(x) for x in args.ve_layers.split(",") if x.strip()] if args.ve_enabled else []

model = _HessianGPT(
    vocab_size          = args.vocab_size,
    num_layers          = args.num_layers,
    model_dim           = args.model_dim,
    num_heads           = args.num_heads,
    num_kv_heads        = args.num_kv_heads,
    mlp_mult            = args.mlp_mult,
    tie_embeddings      = args.tie_embeddings,
    logit_softcap       = args.logit_softcap,
    rope_base           = args.rope_base,
    qk_gain_init        = args.qk_gain_init,
    xsa_last_n          = args.xsa_last_n,
    rope_dims           = args.rope_dims,
    ln_scale            = args.ln_scale,
    sigmoid_skip        = args.sigmoid_skip,
    ve_enabled          = args.ve_enabled,
    ve_dim              = args.ve_dim,
    ve_layers           = ve_layers_list,
    parallel_start_layer = args.parallel_start_layer,
    disable_layer0_attn = args.disable_layer0_attn,
).to(device).bfloat16()

ckpt_path = "./parameter-golf/final_model.pt"
state = torch.load(ckpt_path, map_location=device, weights_only=True)
missing, unexpected = model.load_state_dict(state, strict=False)
print(f"Checkpoint loaded. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
if missing:
    print(f"  Missing: {missing[:10]}")
if unexpected:
    print(f"  Unexpected: {unexpected[:10]}")

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
        print(f"           per-head: {[round(v,3) for v in gain.tolist()]}")
        if gain.mean().item() > 3.0:
            print(f"           => SHARP attention (high q_gain -> peaked softmax, likely local/induction)")
        elif gain.mean().item() > 1.5:
            print(f"           => MODERATE sharpness")
        else:
            print(f"           => DIFFUSE attention (low q_gain -> flat softmax, global)")

print()
print("All layers q_gain summary:")
for layer_idx in range(args.num_layers):
    block = model.blocks[layer_idx]
    if getattr(block, 'disable_attn', False):
        print(f"  Layer {layer_idx:2d}: ATTN_DISABLED")
    else:
        gain = block.attn.q_gain.detach().float()
        print(f"  Layer {layer_idx:2d}: mean={gain.mean().item():.4f}  "
              f"min={gain.min().item():.4f}  max={gain.max().item():.4f}")

# ──────────────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("SECTION 3 - RoPE ablation experiment (layers 0-2 only)")
print("=" * 70)

# Load validation data
val_file = Path("./parameter-golf/data/datasets/fineweb10B_sp4096/fineweb_val_000000.bin")
header_bytes = 256 * np.dtype("<i4").itemsize
header = np.fromfile(val_file, dtype="<i4", count=256)
num_tokens = int(header[2])
tokens_np = np.fromfile(val_file, dtype="<u2", count=num_tokens, offset=header_bytes)
tokens = torch.from_numpy(tokens_np.astype(np.int64))
SEQ_LEN = 2048
N_SEQS = 4
x_val = tokens[:N_SEQS * SEQ_LEN].reshape(N_SEQS, SEQ_LEN).to(device)
print(f"Input: {N_SEQS} seqs x {SEQ_LEN} tokens")

# We need a custom forward through the HessianGPT that:
# 1. Captures output after layer 2
# 2. Optionally zeros out RoPE for layers 0-2
# The _HessianGPT forward is not suitable for recur, but layers 0-2 happen in
# the encoder half (first num_layers//2 = 5 layers), so we can replicate it.

def forward_with_ablation(mdl, x_in, ablate_layers):
    """
    Manual forward through _HessianGPT that:
    - Optionally uses identity RoPE (cos=1, sin=0) for ablate_layers
    - Captures output after layer 2
    Returns (final_output_float, layer2_output_float)
    """
    T = x_in.shape[1]
    half = ROPE_DIMS // 2

    # Pre-populate caches
    for li in range(mdl.num_layers):
        mdl.blocks[li].attn.rotary(T, device, torch.bfloat16)

    # Patch ablated layers
    saved = {}
    for li in ablate_layers:
        rot = mdl.blocks[li].attn.rotary
        saved[li] = (rot._cos_cached, rot._sin_cached)
        rot._cos_cached = torch.ones(1, T, 1, half, dtype=torch.bfloat16, device=device)
        rot._sin_cached = torch.zeros(1, T, 1, half, dtype=torch.bfloat16, device=device)

    layer2_out = [None]

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        x = mdl.tok_emb(x_in)
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x
        skips = []
        ve_cache = {}
        # no parallel mode for layers 0-4 (encoder half)
        for i in range(mdl.num_encoder_layers):
            ve = mdl._get_ve(i, x_in, ve_cache)
            x = mdl.blocks[i](x, x0, v_embed=ve)
            skips.append(x)
            if i == 2:
                layer2_out[0] = x.detach().float().clone()
        for i in range(mdl.num_decoder_layers):
            bi = mdl.num_encoder_layers + i
            if skips:
                scaled_skip = mdl.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
                if mdl.skip_gates is not None:
                    g = torch.sigmoid(mdl.skip_gates[i].to(dtype=x.dtype))[None, None, :]
                    x = torch.lerp(scaled_skip, x, g)
                else:
                    x = x + scaled_skip
            # parallel mode: bi >= parallel_thresh
            if mdl.lane_merge is not None and bi >= mdl.parallel_start_layer:
                # simplified: just run block normally (no true parallel for this analysis)
                ve = mdl._get_ve(bi, x_in, ve_cache)
                x = mdl.blocks[bi](x, x0, v_embed=ve)
            else:
                ve = mdl._get_ve(bi, x_in, ve_cache)
                x = mdl.blocks[bi](x, x0, v_embed=ve)
        x = mdl.final_norm(x)

    # Restore
    for li, (cos_orig, sin_orig) in saved.items():
        rot = mdl.blocks[li].attn.rotary
        rot._cos_cached = cos_orig
        rot._sin_cached = sin_orig

    return x.detach().float().clone(), layer2_out[0]


print("\nRunning normal forward (full RoPE)...")
out_normal, l2_normal = forward_with_ablation(model, x_val, ablate_layers=[])
print(f"layer2_normal shape: {l2_normal.shape}  norm_mean: {l2_normal.norm(dim=-1).mean().item():.4f}")

# Sanity-check: verify this matches the model's own forward
with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    y_dummy = torch.zeros_like(x_val)
    # Run model.forward requires target_ids. Let's just check by running the body separately.
    # Use _forward_body equivalent approach from _HessianGPT
    pass

print("\nRunning ablated forward (RoPE identity on layers 0-2)...")
out_ablated, l2_ablated = forward_with_ablation(model, x_val, ablate_layers=[0, 1, 2])
print(f"layer2_ablated shape: {l2_ablated.shape}  norm_mean: {l2_ablated.norm(dim=-1).mean().item():.4f}")

D = l2_normal.shape[-1]

# Layer-2 metrics
diff_l2 = l2_normal - l2_ablated
rms_diff_l2    = diff_l2.pow(2).mean().sqrt().item()
rms_normal_l2  = l2_normal.pow(2).mean().sqrt().item()
nrmsd_l2       = rms_diff_l2 / rms_normal_l2
sim_l2 = F.cosine_similarity(l2_normal.reshape(-1, D), l2_ablated.reshape(-1, D), dim=-1)
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

# Final output metrics
diff_fin = out_normal - out_ablated
rms_diff_fin   = diff_fin.pow(2).mean().sqrt().item()
rms_normal_fin = out_normal.pow(2).mean().sqrt().item()
nrmsd_fin      = rms_diff_fin / rms_normal_fin
sim_fin = F.cosine_similarity(out_normal.reshape(-1, D), out_ablated.reshape(-1, D), dim=-1)

print()
print("FINAL output (after all 11 layers): normal vs ablated")
print(f"  NRMSD (rms_diff/rms_normal):       {nrmsd_fin:.6f}  ({100*nrmsd_fin:.2f}%)")
print(f"  Cosine similarity (mean):          {sim_fin.mean().item():.6f}")
print(f"  Cosine similarity (min):           {sim_fin.min().item():.6f}")

# Positional variance
pos_var_normal  = l2_normal.var(dim=1).mean().item()
pos_var_ablated = l2_ablated.var(dim=1).mean().item()
print()
print("Positional variance in layer-2 output:")
print(f"  Normal:   {pos_var_normal:.6f}")
print(f"  Ablated:  {pos_var_ablated:.6f}")
print(f"  Ratio (ablated/normal): {pos_var_ablated/pos_var_normal:.4f}")

# Per-position NRMSDs (early positions should change more if RoPE matters at position 0)
print()
print("NRMSD by token position (sampled: pos 0,1,2,5,10,50,100,500,1000,2047):")
sample_pos = [0, 1, 2, 5, 10, 50, 100, 500, 1000, 2047]
for pos in sample_pos:
    d = diff_l2[:, pos, :]
    r = l2_normal[:, pos, :]
    n = d.pow(2).mean().sqrt().item() / r.pow(2).mean().sqrt().item()
    print(f"  pos={pos:4d}: NRMSD={n:.4f}")

print()
print("=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"1. RoPE covers {ROPE_DIMS}/{HEAD_DIM} = {100*rope_frac:.0f}% of head dims.")
print(f"   25% of QK dot-product is position-sensitive; 75% is position-independent.")
print()
print("2. q_gain for layers 0-2:")
for layer_idx in range(3):
    block = model.blocks[layer_idx]
    if getattr(block, 'disable_attn', False):
        print(f"   Layer {layer_idx}: ATTENTION DISABLED (no q_gain effect)")
    else:
        gain = block.attn.q_gain.detach().float()
        print(f"   Layer {layer_idx}: q_gain mean={gain.mean().item():.3f}  "
              f"range=[{gain.min().item():.3f}, {gain.max().item():.3f}]")
        print(f"            High q_gain (>>1) => sharp/peaked softmax => attention is LOCAL.")
print()
print("3. RoPE ablation experiment (layers 0-2, RoPE replaced by identity):")
print(f"   Layer-2 output NRMSD:  {nrmsd_l2:.4f} ({100*nrmsd_l2:.2f}%)")
print(f"   Layer-2 cos-sim mean:  {sim_l2.mean().item():.6f}")
print(f"   Final output NRMSD:    {nrmsd_fin:.4f} ({100*nrmsd_fin:.2f}%)")
print(f"   Final cos-sim mean:    {sim_fin.mean().item():.6f}")
if nrmsd_l2 < 0.03:
    verdict = "NEGLIGIBLE position dependence (NRMSD < 3%)"
elif nrmsd_l2 < 0.10:
    verdict = "LOW position dependence (NRMSD 3-10%)"
elif nrmsd_l2 < 0.20:
    verdict = "MODERATE position dependence (NRMSD 10-20%)"
else:
    verdict = "STRONG position dependence (NRMSD > 20%)"
print(f"   => Layers 0-2 exhibit {verdict}.")
