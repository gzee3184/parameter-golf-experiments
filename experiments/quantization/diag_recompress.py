#!/usr/bin/env python3
"""Re-quantize and re-compress the model with repeat_lora_B as int8 instead of float32.
Compare compressed sizes to identify the root cause of the 1MB gap."""
import sys, io, os, torch, brotli
sys.path.insert(0, "./parameter-golf")
os.chdir("./parameter-golf")

# Load the float model and run full quantization pipeline with OLD vs NEW control patterns
float_model_path = "final_model.pt"
print(f"Loading float model from {float_model_path}...")
sd = torch.load(float_model_path, map_location="cpu", weights_only=True)
print(f"Float model: {os.path.getsize(float_model_path):,} bytes, {len(sd)} keys")

# Import quantization functions from v8_experimental
import importlib.util
spec = importlib.util.spec_from_file_location("train",
    "./h100_final/train_gpt_v8_experimental.py")
# Can't import the module directly (it runs main). Let's just copy the key functions.

# ---- INLINE THE KEY FUNCTIONS ----

INT8_CLIP_Q = 99.99984 / 100.0

def quantize_float_tensor(t):
    t32 = t.float()
    if t32.ndim == 2:
        clip_abs = (torch.quantile(t32.abs(), INT8_CLIP_Q, dim=1) if t32.numel()
                    else torch.empty((t32.shape[0],), dtype=torch.float32))
        clipped = torch.maximum(torch.minimum(t32, clip_abs[:, None]), -clip_abs[:, None])
        scale = (clip_abs / 127.0).clamp_min(1.0 / 127.0)
        q = torch.clamp(torch.round(clipped / scale[:, None]), -127, 127).to(torch.int8).contiguous()
        return q, scale.to(dtype=torch.float16).contiguous()
    clip_abs = float(torch.quantile(t32.abs().flatten(), INT8_CLIP_Q).item()) if t32.numel() else 0.0
    scale = torch.tensor(clip_abs / 127.0 if clip_abs > 0 else 1.0, dtype=torch.float32)
    q = torch.clamp(torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale), -127, 127).to(torch.int8).contiguous()
    return q, scale

def quantize_int6_per_row(t, clip_range=31):
    t32 = t.float()
    best_q, best_s, best_mse = None, None, float("inf")
    for pct in [99.90, 99.95, 99.99, 99.995, 100.0]:
        pf = pct / 100.0
        if t32.ndim >= 2:
            clip_abs = torch.quantile(t32.abs(), pf, dim=-1) if pf < 1.0 else t32.abs().amax(dim=-1)
        else:
            clip_abs = torch.quantile(t32.abs().flatten(), pf) if pf < 1.0 else t32.abs().max()
        if t32.ndim >= 2:
            scale = (clip_abs / float(clip_range)).clamp_min(1e-12)
            q = torch.clamp(torch.round(t32 / scale.unsqueeze(-1)), -clip_range, clip_range).to(torch.int8)
        else:
            scale = (clip_abs / float(clip_range)).clamp_min(1e-12)
            q = torch.clamp(torch.round(t32 / float(scale)), -clip_range, clip_range).to(torch.int8)
        if t32.ndim >= 2:
            recon = q.float() * scale.unsqueeze(-1)
        else:
            recon = q.float() * float(scale)
        mse = ((t32 - recon) ** 2).mean().item()
        if mse < best_mse:
            best_q, best_s, best_mse = q.contiguous(), scale.to(torch.float16).contiguous(), mse
    return best_q, best_s

def _classify_param(name):
    if "tok_emb" in name or "lm_head" in name: return "embed"
    if ".mlp." in name: return "mlp"
    if ".attn." in name or (".proj." in name and ".mlp." not in name): return "attn"
    return "other"

def _bits_to_clip_range(bits):
    return {5: 15, 6: 31, 8: 127}[bits]

def _unbank_state_dict(sd, num_layers):
    out = {}
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

def _byte_shuffle(data):
    n = len(data)
    pad = (4 - n % 4) % 4
    padded = data + b'\x00' * pad
    n2 = len(padded)
    cols = 4
    rows = n2 // cols
    shuffled = bytearray(n2)
    for c in range(cols):
        for r in range(rows):
            shuffled[c * rows + r] = padded[r * cols + c]
    return bytes(shuffled) + bytes([pad])

def _compress(data):
    data = _byte_shuffle(data)
    return brotli.compress(data, quality=11)

def mixed_quantize(state_dict, ctrl_patterns, int6_cats, bits_mlp=6, bits_attn=6, bits_embed=6):
    result, meta = {}, {}
    for name, tensor in state_dict.items():
        t = tensor.detach().cpu().contiguous()
        cat = _classify_param(name)
        if not t.is_floating_point() or t.numel() <= 65536:
            result[name] = t.to(torch.float16) if t.is_floating_point() else t
            meta[name] = "passthrough"
            continue
        if any(p in name for p in ctrl_patterns):
            result[name] = t.float()
            meta[name] = "passthrough_ctrl"
            continue
        if cat in int6_cats and t.ndim >= 1:
            bits = bits_mlp if cat == "mlp" else (bits_embed if cat == "embed" else bits_attn)
            clip = _bits_to_clip_range(bits)
            q, s = quantize_int6_per_row(t, clip_range=clip)
            result[name + ".q"] = q
            result[name + ".scale"] = s
            meta[name] = {"type": f"int{bits}"}
        else:
            q, s = quantize_float_tensor(t)
            result[name + ".q"] = q
            result[name + ".scale"] = s
            meta[name] = {"type": "int8"}
    return result, meta

# Unbank
print("Unbanking state dict...")
unbanked = _unbank_state_dict({k: v.detach().cpu() for k, v in sd.items()}, 11)
print(f"Unbanked: {len(unbanked)} keys")

# OLD control patterns (R3 version, 100KB code)
OLD_CTRL = ("attn_scale", "attn_scales", "mlp_scale", "mlp_scales",
            "resid_mix", "resid_mixes", "q_gain", "skip_weight", "skip_weights")

# NEW control patterns (current 108KB code)
NEW_CTRL = OLD_CTRL + ("skip_gates", "ve_layer_scales", "ve_shared.scale",
                        "lane_merge", "repeat_lora", "repeat_attn_lora", "repeat_pass_scale")

int6_cats = {"mlp", "attn", "embed"}

# Quantize with OLD patterns (what R3 did)
print("\n" + "="*60)
print("Quantizing with OLD control patterns (R3 style)...")
old_result, old_meta = mixed_quantize(unbanked, OLD_CTRL, int6_cats, bits_mlp=6, bits_attn=6, bits_embed=8)
old_buf = io.BytesIO()
torch.save({"w": old_result, "m": old_meta}, old_buf)
old_raw = old_buf.getvalue()
print(f"Raw quantized (old): {len(old_raw):,} bytes ({len(old_raw)/1024/1024:.2f} MB)")
old_compressed = _compress(old_raw)
print(f"Compressed (old): {len(old_compressed):,} bytes ({len(old_compressed)/1024/1024:.2f} MB)")

# Check what changed for each tensor
print("\nDifferences in quantization treatment:")
for name in sorted(unbanked.keys()):
    o = old_meta.get(name, "?")
    n = None  # will compute below
    # Check new treatment
    t = unbanked[name].detach().cpu().contiguous()
    cat = _classify_param(name)
    if not t.is_floating_point() or t.numel() <= 65536:
        n = "passthrough"
    elif any(p in name for p in NEW_CTRL):
        n = "passthrough_ctrl"
    elif cat in int6_cats and t.ndim >= 1:
        n = "int6/custom"
    else:
        n = "int8"
    if str(o) != str(n):
        print(f"  {name:50s}  OLD={o}  NEW={n}  shape={list(t.shape)}")

# Quantize with NEW patterns (current code)
print("\n" + "="*60)
print("Quantizing with NEW control patterns (current code)...")
new_result, new_meta = mixed_quantize(unbanked, NEW_CTRL, int6_cats, bits_mlp=6, bits_attn=6, bits_embed=8)
new_buf = io.BytesIO()
torch.save({"w": new_result, "m": new_meta}, new_buf)
new_raw = new_buf.getvalue()
print(f"Raw quantized (new): {len(new_raw):,} bytes ({len(new_raw)/1024/1024:.2f} MB)")
new_compressed = _compress(new_raw)
print(f"Compressed (new): {len(new_compressed):,} bytes ({len(new_compressed)/1024/1024:.2f} MB)")

print("\n" + "="*60)
print("COMPARISON:")
print(f"  OLD patterns compressed: {len(old_compressed):>12,} bytes ({len(old_compressed)/1024/1024:.2f} MB)")
print(f"  NEW patterns compressed: {len(new_compressed):>12,} bytes ({len(new_compressed)/1024/1024:.2f} MB)")
diff = len(new_compressed) - len(old_compressed)
print(f"  Difference:              {diff:>+12,} bytes ({diff/1024/1024:+.2f} MB)")
print(f"\n  R3 had:     15.88 MB compressed (from log)")
print(f"  F/E had:    ~16.88-16.92 MB compressed (from logs)")
print(f"  Gap:        ~1.00 MB")
print(f"  This fix:   {diff/1024/1024:+.2f} MB")
if abs(diff) > 500000:
    print(f"  --> This explains {abs(diff)/1024/1024/1.0*100:.0f}% or more of the gap!")
else:
    print(f"  --> This explains only {abs(diff)/1024/1024/1.0*100:.0f}% of the gap. Other causes exist!")

# Budget check with old patterns + code minification
code_size_current = 107945
code_size_minified_est = 60000  # aggressive minification estimate
artifact_limit = 16777216
model_budget_old = artifact_limit - code_size_current
model_budget_minified = artifact_limit - code_size_minified_est

print(f"\n  Budget with current code ({code_size_current} bytes): model_budget={model_budget_old:,} ({model_budget_old/1024/1024:.2f} MB)")
print(f"  Budget with minified code (~{code_size_minified_est} bytes): model_budget={model_budget_minified:,} ({model_budget_minified/1024/1024:.2f} MB)")
print(f"  OLD patterns model: {len(old_compressed):,} bytes")
if len(old_compressed) <= model_budget_old:
    print(f"  --> FITS even WITHOUT minification! No pruning needed!")
elif len(old_compressed) <= model_budget_minified:
    print(f"  --> Fits WITH minification. No pruning needed after minification!")
else:
    print(f"  --> Still over budget. Need {len(old_compressed)-model_budget_minified:,} bytes of additional savings.")
