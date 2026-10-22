"""Gradient analysis of trained model — measures per-layer gradient norms."""
import os, sys, torch, torch.nn.functional as F, numpy as np

# Use GPU 1 (GPU 0 busy with R4)
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# Set our config
for k, v in {
    'VOCAB_SIZE': '4096', 'NUM_LAYERS': '11', 'MODEL_DIM': '512',
    'NUM_HEADS': '8', 'NUM_KV_HEADS': '4', 'MLP_MULT': '4',
    'MLP_ACTIVATION': 'leaky05', 'TIE_EMBEDDINGS': '1',
    'LOGIT_SOFTCAP': '30', 'ROPE_BASE': '10000', 'QK_GAIN_INIT': '5.0',
    'XSA_LAST_N': '11', 'ROPE_DIMS': '0', 'LN_SCALE': '1',
    'SIGMOID_SKIP': '1', 'RECUR_LAYERS': '3,4,5', 'RECUR_NUM_LOOPS': '2',
    'VE_ENABLED': '1', 'VE_DIM': '128', 'VE_LAYERS': '9,10',
    'PARALLEL_START_LAYER': '7', 'DISABLE_LAYER0_ATTN': '1',
    'REPEAT_UNTIE_MLP': 'lora', 'REPEAT_LORA_RANK': '16',
    'ATTN_RES': '0', 'RECUR_SKIP_PHYSICAL': '0',
    'REPEAT_PASS_SCALE': '1', 'PER_LAYER_LANE_MERGE': '0',
    'BYTE_SHUFFLE': '1', 'COPRIME_STRIDE': '1',
    'RECUR_START_STEP': '2000', 'EMA_DECAY': '0.9965',
    'MUON_PLUS': '1',
    'DATA_PATH': './data/datasets/fineweb10B_sp4096',
    'TOKENIZER_PATH': './data/tokenizers/fineweb_4096_bpe.model',
    'TRAIN_SEQ_LEN': '2048', 'EVAL_SEQ_LEN': '2048',
    'TRAIN_BATCH_TOKENS': '786432',
}.items():
    os.environ[k] = v

# Load the training code (just model definitions)
script_path = 'src/train_gpt_readable.py'
with open(script_path) as f:
    code = f.read()
code = code.split('if __name__')[0]
exec(code)

device = torch.device('cuda')

recur_layers_list = [3, 4, 5]
ve_layers_list = [9, 10]
model = GPT(
    vocab_size=4096, num_layers=11, model_dim=512, num_heads=8, num_kv_heads=4,
    mlp_mult=4, tie_embeddings=True, tied_embed_init_std=0.02,
    logit_softcap=30, rope_base=10000, qk_gain_init=5.0,
    xsa_last_n=11, rope_dims=0, ln_scale=True, sigmoid_skip=True,
    ve_enabled=True, ve_dim=128, ve_layers=ve_layers_list,
    parallel_start_layer=7, repeat_untie_mlp='lora', disable_layer0_attn=True,
    recur_layers=recur_layers_list, recur_num_loops=2, attn_res=False,
    lora_rank=16, repeat_pass_scale=True, train_seq_len=2048,
    recur_skip_physical=False, per_layer_lane_merge=False
).to(device).bfloat16()

# Keep bank params in float32 (like training code does)
model.qo_bank.data = model.qo_bank.data.float()
model.kv_bank.data = model.kv_bank.data.float()
model.mlp_up_bank.data = model.mlp_up_bank.data.float()
model.mlp_down_bank.data = model.mlp_down_bank.data.float()
if model.repeat_down_bank is not None:
    model.repeat_down_bank.data = model.repeat_down_bank.data.float()
if model.repeat_lora_A is not None:
    model.repeat_lora_A.data = model.repeat_lora_A.data.float()
    model.repeat_lora_B.data = model.repeat_lora_B.data.float()

# Load trained weights
ckpt = torch.load('./parameter-golf/final_model.pt',
                  map_location=device, weights_only=True)
missing, unexpected = model.load_state_dict(ckpt, strict=False)
print("Loaded checkpoint: %d missing, %d unexpected" % (len(missing), len(unexpected)))
if missing:
    print("  Missing:", missing[:10])

# CRITICAL: activate recurrence (it's off by default, training loop turns it on at recur_start_step)
model.set_recurrence_active(True)
vl = model._get_virtual_layers()
print("Virtual layers (%d): %s" % (len(vl), vl))

# Verify key params loaded correctly
print("lane_merge: %.4f (sigmoid=%.4f)" % (model.lane_merge.item(), torch.sigmoid(model.lane_merge.detach()).item()))
print("repeat_pass_scale:", model.repeat_pass_scale.data.tolist() if model.repeat_pass_scale is not None else "None")
print("repeat_lora_A shape:", model.repeat_lora_A.shape if model.repeat_lora_A is not None else "None")
print("skip_gates[4] mean:", model.skip_gates[4].mean().item() if model.skip_gates is not None else "None")

seq_len = 2048
num_batches = 8
batch_size = 16

# Load validation data
repo_dir = './parameter-golf'
val_path = os.path.join(repo_dir, 'data/datasets/fineweb10B_sp4096/fineweb_val_000000.bin')
val_data = np.memmap(val_path, dtype=np.uint16, mode='r')
print("Val data: %d tokens" % len(val_data))

# Quick loss sanity check
with torch.no_grad():
    test_tokens = torch.tensor(val_data[:batch_size * (seq_len + 1)].astype(np.int64),
                              device=device).reshape(batch_size, seq_len + 1).clamp(0, 4095)
    test_loss = model(test_tokens[:, :seq_len], test_tokens[:, 1:seq_len + 1])
    print("Sanity check loss: %.4f (expect ~0.76 for BPB ~1.10)" % test_loss.item())

# Accumulate per-layer gradient norms across batches
mlp_up_norms_accum = [[] for _ in range(11)]
mlp_down_norms_accum = [[] for _ in range(11)]
qo_norms_accum = [[] for _ in range(22)]
kv_norms_accum = [[] for _ in range(22)]
scalar_grad_accum = {}

model.train()
np.random.seed(42)

for batch_idx in range(num_batches):
    start = np.random.randint(0, len(val_data) - batch_size * (seq_len + 1))
    tokens = torch.tensor(val_data[start:start + batch_size * (seq_len + 1)].astype(np.int64),
                         device=device).reshape(batch_size, seq_len + 1).clamp(0, 4095)
    x = tokens[:, :seq_len]
    y = tokens[:, 1:seq_len + 1]

    model.zero_grad()
    loss = model(x, y)
    loss.backward()

    if batch_idx == 0:
        print("Batch 0 loss: %.4f" % loss.item())

    # Per-layer bank gradients
    mlp_up_g = model.mlp_up_bank.grad.float()
    mlp_down_g = model.mlp_down_bank.grad.float()
    for i in range(11):
        mlp_up_norms_accum[i].append(mlp_up_g[i].norm().item())
        mlp_down_norms_accum[i].append(mlp_down_g[i].norm().item())

    qo_g = model.qo_bank.grad.float()
    kv_g = model.kv_bank.grad.float()
    for i in range(22):
        qo_norms_accum[i].append(qo_g[i].norm().item())
    for i in range(kv_g.shape[0]):
        kv_norms_accum[i].append(kv_g[i].norm().item())

    # All named param gradients
    for name, p in model.named_parameters():
        if p.grad is not None:
            gn = p.grad.float().norm().item()
            if name not in scalar_grad_accum:
                scalar_grad_accum[name] = []
            scalar_grad_accum[name].append(gn)

print("\nAveraged over %d batches" % num_batches)
print("=" * 80)

# === MLP_UP gradient norms by layer ===
print("\n## MLP_UP Gradient Norms by Layer")
print("(Higher = more learning signal flowing to that layer's MLP)")
print()
up_norms = [np.mean(mlp_up_norms_accum[i]) for i in range(11)]
max_norm = max(up_norms)
for i in range(11):
    bar_len = int(40 * up_norms[i] / max_norm)
    bar = '\u2588' * bar_len
    role = ""
    if i == 0: role = "<- disabled L0"
    elif i in [3,4,5]: role = "<- recurrence"
    elif i >= 7 and i <= 9: role = "<- parallel"
    elif i == 10: role = "<- output (near loss)"
    elif i == 6: role = "<- last serial"
    print("  L%2d: %6.3f  %s  %s" % (i, up_norms[i], bar, role))

recur_avg = np.mean([up_norms[i] for i in [3,4,5]])
serial_avg = np.mean([up_norms[i] for i in [1,2,6]])
parallel_avg = np.mean([up_norms[i] for i in [7,8,9]])
output_avg = up_norms[10]

print()
print("  +----------------+-----------+-----------+-----------------------------+")
print("  | Layer Group     |   Avg     | Relative  |             Why             |")
print("  |                 | Gradient  | to Para.  |                             |")
print("  +----------------+-----------+-----------+-----------------------------+")
print("  | Recurrence      | %7.3f   | %.1fx      | Weights used TWICE          |" % (recur_avg, recur_avg/parallel_avg))
print("  | (3,4,5)         |           |           | gradients accumulate        |")
print("  | Serial          | %7.3f   | %.1fx      | Normal single-pass          |" % (serial_avg, serial_avg/parallel_avg))
print("  | (1,2,6)         |           |           |                             |")
print("  | Parallel        | %7.3f   | 1.0x      | Split across lanes +        |" % parallel_avg)
print("  | (7,8,9)         |           | (lowest)  | ln_scale attenuation        |")
print("  | Output (10)     | %7.3f   | %.1fx      | Closest to loss             |" % (output_avg, output_avg/parallel_avg))
print("  +----------------+-----------+-----------+-----------------------------+")

# === MLP_DOWN ===
print("\n## MLP_DOWN Gradient Norms by Layer")
down_norms = [np.mean(mlp_down_norms_accum[i]) for i in range(11)]
max_dn = max(down_norms)
for i in range(11):
    bar_len = int(40 * down_norms[i] / max_dn)
    bar = '\u2588' * bar_len
    print("  L%2d: %6.3f  %s" % (i, down_norms[i], bar))

# === QO bank (attention Q and O projections) ===
print("\n## QO Bank (Attention Q/O) Gradient Norms by Layer")
for i in range(0, 22, 2):
    layer = i // 2
    q_norm = np.mean(qo_norms_accum[i])
    o_norm = np.mean(qo_norms_accum[i+1])
    role = ""
    if layer == 0: role = "<- disabled"
    elif layer in [3,4,5]: role = "<- recurrence"
    elif layer >= 7: role = "<- parallel"
    print("  L%2d: Q=%6.3f  O=%6.3f  %s" % (layer, q_norm, o_norm, role))

# === KV bank ===
print("\n## KV Bank Gradient Norms by Layer")
for i in range(0, min(22, len(kv_norms_accum)), 2):
    layer = i // 2
    k_norm = np.mean(kv_norms_accum[i])
    v_norm = np.mean(kv_norms_accum[i+1])
    print("  L%2d: K=%6.3f  V=%6.3f" % (layer, k_norm, v_norm))

# === Skip connection gradients ===
print("\n## Skip Connection Gradients")
if 'skip_weights' in scalar_grad_accum:
    # Need per-index, re-check from last backward
    sw_g = model.skip_weights.grad.float()
    sg_g = model.skip_gates.grad.float() if model.skip_gates is not None and model.skip_gates.grad is not None else None
    for i in range(sw_g.shape[0]):
        wg = sw_g[i].norm().item()
        gg = sg_g[i].norm().item() if sg_g is not None else 0
        status = " <-- DEAD (zero gradient)" if wg < 1e-8 else ""
        print("  skip[%d]: weight_grad=%.4f  gate_grad=%.4f%s" % (i, wg, gg, status))

# === Lane merge gradient ===
print("\n## Lane Merge Gradient")
if model.lane_merge is not None and model.lane_merge.grad is not None:
    lm_grad = np.mean(scalar_grad_accum.get('lane_merge', [0]))
    lm_val = model.lane_merge.item()
    print("  lane_merge value: %.4f (sigmoid=%.4f)" % (lm_val, torch.sigmoid(torch.tensor(lm_val)).item()))
    print("  lane_merge avg gradient norm: %.4f" % lm_grad)

# === Repeat pass scale gradient ===
print("\n## Repeat Pass Scale Gradients")
if model.repeat_pass_scale is not None and model.repeat_pass_scale.grad is not None:
    rps_grad = model.repeat_pass_scale.grad.float()
    rps_val = model.repeat_pass_scale.data
    for i in range(len(rps_val)):
        print("  repeat_pass_scale[%d] (L%d): value=%.4f  grad=%.4f" % (i, recur_layers_list[i], rps_val[i].item(), rps_grad[i].item()))

# === LoRA gradients ===
print("\n## LoRA Gradients")
if model.repeat_lora_A is not None and model.repeat_lora_A.grad is not None:
    la_grad = model.repeat_lora_A.grad.float()
    lb_grad = model.repeat_lora_B.grad.float()
    for i in range(la_grad.shape[0]):
        print("  LoRA bank[%d] (L%d): A_grad=%.4f  B_grad=%.4f" % (i, recur_layers_list[i], la_grad[i].norm().item(), lb_grad[i].norm().item()))

# === Per-block scalar param gradients ===
print("\n## Per-Block Scalar Parameter Gradients")
for i in range(11):
    block = model.blocks[i]
    parts = []
    if block.attn_scale.grad is not None:
        parts.append("attn_s=%.4f" % block.attn_scale.grad.float().norm().item())
    if block.mlp_scale.grad is not None:
        parts.append("mlp_s=%.4f" % block.mlp_scale.grad.float().norm().item())
    if block.resid_mix.grad is not None:
        parts.append("resid=%.4f" % block.resid_mix.grad.float().norm().item())
    if block.attn.q_gain.grad is not None:
        parts.append("q_gain=%.4f" % block.attn.q_gain.grad.float().norm().item())
    print("  L%2d: %s" % (i, ', '.join(parts)))

# === VE gradients ===
print("\n## Value Embedding Gradients")
if model.ve_shared is not None:
    if model.ve_shared.embed.weight.grad is not None:
        print("  ve_embed: %.4f" % model.ve_shared.embed.weight.grad.float().norm().item())
    if model.ve_shared.scale.grad is not None:
        print("  ve_scale: %.4f" % model.ve_shared.scale.grad.float().norm().item())
    for idx, p in enumerate(model.ve_layer_scales.parameters()):
        if p.grad is not None:
            print("  ve_layer_scales[%d] (L%d): %.4f" % (idx, ve_layers_list[idx], p.grad.float().norm().item()))

# === Token embedding gradient ===
print("\n## Token Embedding Gradient")
if model.tok_emb.weight.grad is not None:
    print("  tok_emb: %.4f" % model.tok_emb.weight.grad.float().norm().item())

# === Top 25 scalar param gradients ===
print("\n## Top 25 Scalar Param Gradient Norms (sorted)")
all_scalar = []
for name, p in model.named_parameters():
    if name in scalar_grad_accum and p.numel() <= 100:
        gn = np.mean(scalar_grad_accum[name])
        all_scalar.append((gn, name, p.numel()))
all_scalar.sort(reverse=True)
for gn, name, numel in all_scalar[:25]:
    print("  %8.4f  %s (numel=%d)" % (gn, name, numel))

# === ln_scale_factor analysis ===
print("\n## ln_scale_factor Analysis (1/sqrt(layer+1))")
for i in range(11):
    block = model.blocks[i]
    lsf = block.ln_scale_factor.item() if hasattr(block.ln_scale_factor, 'item') else block.ln_scale_factor
    print("  L%2d: ln_scale_factor=%.4f  (effective gradient mult = %.4f)" % (i, lsf, lsf))

print("\n=== GRADIENT ANALYSIS COMPLETE ===")
