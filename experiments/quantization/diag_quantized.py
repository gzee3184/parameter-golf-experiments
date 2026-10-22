#!/usr/bin/env python3
"""Diagnose quantized model artifact: show all keys, types, sizes, and control vs quantized."""
import sys, io, torch, brotli

ptz_path = "./parameter-golf/final_model.int6.ptz"
print(f"Loading {ptz_path}...")
with open(ptz_path, "rb") as f:
    blob = f.read()
print(f"Compressed: {len(blob):,} bytes ({len(blob)/1024/1024:.2f} MB)")

# Decompress (byte_unshuffle + brotli)
raw = brotli.decompress(blob)
# byte unshuffle
n = len(raw) - 1  # last byte is padding length
stride = raw[-1]  # padding byte count
data_len = n - stride
cols = 4
rows = (n + cols - 1) // cols
arr = bytearray(n)
for c in range(cols):
    col_start = c * rows
    for r in range(rows):
        idx = r * cols + c
        if idx < n:
            arr[idx] = raw[col_start + r]
unshuffled = bytes(arr[:data_len])

print(f"Decompressed: {len(unshuffled):,} bytes ({len(unshuffled)/1024/1024:.2f} MB)")

state = torch.load(io.BytesIO(unshuffled), map_location="cpu", weights_only=True)
w = state["w"]
m = state["m"]

print(f"\n{'Key':<55} {'Type/Meta':<20} {'Shape':<25} {'Bytes':>12} {'Dtype'}")
print("=" * 130)

# OLD patterns (R3 version that worked)
OLD_CTRL = {"attn_scale", "attn_scales", "mlp_scale", "mlp_scales",
            "resid_mix", "resid_mixes", "q_gain", "skip_weight", "skip_weights"}
# NEW patterns (current v8_experimental)
NEW_CTRL = OLD_CTRL | {"skip_gates", "ve_layer_scales", "ve_shared.scale",
                        "lane_merge", "repeat_lora", "repeat_attn_lora", "repeat_pass_scale"}

total_bytes = 0
ctrl_float32_bytes = 0
ctrl_float32_names = []
quantized_bytes = 0
passthrough_bytes = 0

for name in sorted(w.keys()):
    t = w[name]
    nbytes = t.numel() * t.element_size()
    total_bytes += nbytes

    # Determine what this key represents
    base_name = name.replace(".q", "").replace(".scale", "")
    meta_info = m.get(base_name, m.get(name, "?"))

    # Check if this matches new-only control patterns
    is_new_ctrl = any(p in name for p in NEW_CTRL - OLD_CTRL)
    is_old_ctrl = any(p in name for p in OLD_CTRL)

    marker = ""
    if isinstance(meta_info, str) and "passthrough_ctrl" in meta_info:
        ctrl_float32_bytes += nbytes
        if is_new_ctrl and not is_old_ctrl:
            marker = " <<<< NEW CTRL (was quantized in R3)"
            ctrl_float32_names.append((name, nbytes, str(list(t.shape))))
        elif is_new_ctrl:
            marker = " (ctrl in both)"
    elif isinstance(meta_info, str) and "passthrough" in meta_info:
        passthrough_bytes += nbytes
    else:
        quantized_bytes += nbytes

    print(f"{name:<55} {str(meta_info):<20} {str(list(t.shape)):<25} {nbytes:>12,} {t.dtype}{marker}")

print("=" * 130)
print(f"Total raw bytes in quantized dict: {total_bytes:,} ({total_bytes/1024/1024:.2f} MB)")
print(f"  Quantized (int6/int8): {quantized_bytes:,} ({quantized_bytes/1024/1024:.2f} MB)")
print(f"  Passthrough (small/fp16): {passthrough_bytes:,} ({passthrough_bytes/1024/1024:.2f} MB)")
print(f"  Control tensors (float32): {ctrl_float32_bytes:,} ({ctrl_float32_bytes/1024/1024:.2f} MB)")

if ctrl_float32_names:
    print(f"\n{'='*80}")
    print(f"NEW CONTROL TENSORS (would have been quantized in R3 version):")
    print(f"{'='*80}")
    total_new_ctrl = 0
    for name, nbytes, shape in ctrl_float32_names:
        print(f"  {name:<50} {shape:<20} {nbytes:>10,} bytes")
        total_new_ctrl += nbytes
    print(f"  {'TOTAL':<50} {'':20} {total_new_ctrl:>10,} bytes ({total_new_ctrl/1024/1024:.2f} MB)")

    # Estimate savings if these were int8 quantized instead
    est_int8 = 0
    for name, nbytes, shape in ctrl_float32_names:
        # float32 → int8 saves 3 bytes per element (plus small scale overhead)
        n_elem = nbytes // 4  # float32 = 4 bytes
        est_int8 += n_elem + 4  # int8 bytes + tiny scale
    print(f"  If quantized to int8: ~{est_int8:,} bytes ({est_int8/1024/1024:.2f} MB)")
    print(f"  Raw savings: ~{total_new_ctrl - est_int8:,} bytes ({(total_new_ctrl - est_int8)/1024/1024:.2f} MB)")
    print(f"  After compression (est ~1.5x), savings: ~{(total_new_ctrl - est_int8)*1.5/1024/1024:.2f} MB")
