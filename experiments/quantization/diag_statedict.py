#!/usr/bin/env python3
"""Quick diagnostic: compare state_dict keys between R3 config and current code."""
import sys, os
sys.path.insert(0, "./parameter-golf")
os.chdir("./parameter-golf")

# Set R3's config (softcap=20, ROPE_BASE=22000, all toggles off)
for k, v in {
    "VOCAB_SIZE": "4096", "NUM_LAYERS": "11", "MODEL_DIM": "512",
    "NUM_HEADS": "8", "NUM_KV_HEADS": "4", "MLP_MULT": "4",
    "RECUR_LAYERS": "3,4,5", "RECUR_NUM_LOOPS": "2",
    "VE_ENABLED": "1", "VE_DIM": "128", "VE_LAYERS": "9,10",
    "PARALLEL_START_LAYER": "7", "DISABLE_LAYER0_ATTN": "1",
    "SIGMOID_SKIP": "1", "TIE_EMBEDDINGS": "1",
    "REPEAT_UNTIE_MLP": "lora", "REPEAT_LORA_RANK": "16",
    "REPEAT_PASS_SCALE": "1", "LOGIT_SOFTCAP": "20",
    "ROPE_BASE": "22000", "LN_SCALE": "1",
    "PER_LAYER_LANE_MERGE": "0", "REPEAT_ATTN_LORA": "0",
    "DISABLE_LAYER0_MLP": "0", "SKIP_REPEAT_ATTN": "0",
    "PARALLEL_SCALAR_GRAD_SCALE": "0",
    "MLP_ACTIVATION": "leaky05", "QK_GAIN_INIT": "5.0",
    "TRAIN_SEQ_LEN": "2048",
}.items():
    os.environ[k] = v

import torch
# Import after setting env vars
import importlib
spec = importlib.util.spec_from_file_location("train_gpt",
    "./h100_final/train_gpt_v8_experimental.py")
# Can't easily import the whole module. Let's just check file content for serialize differences.

# Instead, load the last saved model and check its keys
model_path = "./parameter-golf/final_model.pt"
sd = torch.load(model_path, map_location="cpu", weights_only=True)
print(f"State dict keys ({len(sd)}):")
total_params = 0
total_bytes = 0
for k in sorted(sd.keys()):
    t = sd[k]
    params = t.numel()
    bytes_size = t.numel() * t.element_size()
    total_params += params
    total_bytes += bytes_size
    print(f"  {k:50s}  {str(list(t.shape)):20s}  {params:>10,} params  {bytes_size:>10,} bytes  {t.dtype}")
print(f"\nTotal: {total_params:,} params, {total_bytes:,} bytes ({total_bytes/1024/1024:.2f} MB)")
