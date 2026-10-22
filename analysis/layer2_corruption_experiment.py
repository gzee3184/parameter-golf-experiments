"""
Layer-2 corruption experiment.

Measures how sensitive layers 3-10 are to the quality of layers 0-2's output.
We intercept the hidden state after virtual layer 2 (physical layer 2), corrupt it
at various levels, then feed the corrupted state through the rest of the network.
"""

from __future__ import annotations
import os
import sys
import math

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["VOCAB_SIZE"] = "4096"
os.environ["MLP_MULT"] = "4"
os.environ["RECUR_LAYERS"] = "3,4,5"
os.environ["RECUR_NUM_LOOPS"] = "2"
os.environ["NUM_LAYERS"] = "11"
os.environ["VE_ENABLED"] = "1"
os.environ["VE_DIM"] = "128"
os.environ["VE_LAYERS"] = "9,10"
os.environ["PARALLEL_START_LAYER"] = "7"
os.environ["REPEAT_UNTIE_MLP"] = "lora"
os.environ["REPEAT_LORA_RANK"] = "16"
os.environ["DISABLE_LAYER0_ATTN"] = "1"
os.environ["SIGMOID_SKIP"] = "1"
os.environ["ATTN_RES"] = "0"
os.environ["REPEAT_PASS_SCALE"] = "0"
os.environ["NUM_KV_HEADS"] = "4"
os.environ["NUM_HEADS"] = "8"
os.environ["MODEL_DIM"] = "512"

sys.path.insert(0, "./h100_final")

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from train_gpt_v6 import GPT, restore_low_dim_params_to_fp32, CastedLinear, _rebank_state_dict

CHECKPOINT = "./parameter-golf/final_model.pt"
DEVICE = torch.device("cuda", 0)
SEED = 42
B, T = 4, 512
NUM_LAYERS = 11


def build_model() -> GPT:
    model = GPT(
        vocab_size=4096,
        num_layers=NUM_LAYERS,
        model_dim=512,
        num_heads=8,
        num_kv_heads=4,
        mlp_mult=4,
        tie_embeddings=True,
        tied_embed_init_std=0.005,
        logit_softcap=30.0,
        rope_base=10000.0,
        qk_gain_init=1.5,
        xsa_last_n=11,
        rope_dims=16,
        ln_scale=False,
        sigmoid_skip=True,
        recur_layers=[3, 4, 5],
        recur_num_loops=2,
        ve_enabled=True,
        ve_dim=128,
        ve_layers=[9, 10],
        parallel_start_layer=7,
        repeat_untie_mlp="lora",
        disable_layer0_attn=True,
        attn_res=False,
        lora_rank=16,
        repeat_pass_scale=False,
    ).to(DEVICE).bfloat16()
    model.qo_bank.data = model.qo_bank.data.float()
    model.kv_bank.data = model.kv_bank.data.float()
    model.mlp_up_bank.data = model.mlp_up_bank.data.float()
    model.mlp_down_bank.data = model.mlp_down_bank.data.float()
    if model.repeat_lora_A is not None:
        model.repeat_lora_A.data = model.repeat_lora_A.data.float()
        model.repeat_lora_B.data = model.repeat_lora_B.data.float()
    for m in model.modules():
        if isinstance(m, CastedLinear):
            m.float()
    restore_low_dim_params_to_fp32(model)
    return model


def load_checkpoint(model: GPT) -> None:
    print(f"Loading checkpoint from {CHECKPOINT}...")
    # The checkpoint is in "unbanked" format (per-block weights).
    # We need to rebank it to match the GPT model's state dict format.
    raw_sd = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    print(f"  Checkpoint keys: {len(raw_sd)}")
    
    # Check format: if 'qo_bank' present -> already banked; else -> unbanked
    if 'qo_bank' in raw_sd:
        print("  Format: banked (direct load)")
        sd = raw_sd
    else:
        print("  Format: unbanked (converting to banked format)...")
        template_sd = model.state_dict()
        sd = _rebank_state_dict(raw_sd, NUM_LAYERS, template_sd)
        print(f"  Converted to {len(sd)} banked keys")

    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"  Missing keys ({len(missing)}): {missing[:8]}")
    if unexpected:
        print(f"  Unexpected keys ({len(unexpected)}): {unexpected[:5]}")
    print("  Checkpoint loaded.")


def forward_full(model: GPT, input_ids: Tensor, target_ids: Tensor) -> float:
    with torch.inference_mode():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = model(input_ids, target_ids)
    return loss.item()


def split_forward_at_layer2(model: GPT, input_ids: Tensor):
    """Run virtual layers 0, 1, 2 and return the hidden state + accumulated state."""
    with torch.inference_mode():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            x = model.tok_emb(input_ids)
            x = F.rms_norm(x, (x.size(-1),))
            x0 = x.clone()

            virtual_layers = model._get_virtual_layers()
            ve_cache = {}
            skips = []

            for vi in range(3):  # virtual indices 0, 1, 2 = physical layers 0, 1, 2
                phys_idx, is_repeat = virtual_layers[vi]
                q_w, k_w, v_w, out_w, up_w, down_w = model._bank_weights(phys_idx)
                ve = model._get_ve(phys_idx, input_ids, ve_cache)
                dw_override = model._get_repeat_down_w(phys_idx) if is_repeat else None
                x = model.blocks[phys_idx](x, x0, q_w, k_w, v_w, out_w, up_w, down_w,
                                           v_embed=ve, down_w_override=dw_override)
                skips.append(x.clone())

            return (x.detach().clone(), x0.detach().clone(),
                    [s.detach().clone() for s in skips],
                    {k: v.detach().clone() for k, v in ve_cache.items()})


def forward_from_layer3(model: GPT, input_ids: Tensor, h: Tensor, x0: Tensor,
                         skips_in: list, ve_cache_in: dict, target_ids: Tensor) -> float:
    """Continue forward from virtual layer 3 with a (possibly corrupted) h."""
    with torch.inference_mode():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            virtual_layers = model._get_virtual_layers()
            num_virtual = len(virtual_layers)
            num_enc = num_virtual // 2

            x = h.clone()
            x0_use = x0.clone()

            ve_cache = {}
            if 've' in ve_cache_in:
                ve_cache['ve'] = ve_cache_in['ve'].clone()

            skips = [s.clone() for s in skips_in]

            is_parallel = False
            lane0 = None
            lane1 = None
            parallel_thresh = model.parallel_start_layer if model.lane_merge is not None else num_virtual + 1

            # Remaining encoder layers: vi = 3 .. num_enc-1
            for vi in range(3, num_enc):
                phys_idx, is_repeat = virtual_layers[vi]
                q_w, k_w, v_w, out_w, up_w, down_w = model._bank_weights(phys_idx)
                ve = model._get_ve(phys_idx, input_ids, ve_cache)
                dw_override = model._get_repeat_down_w(phys_idx) if is_repeat else None
                x_prev = x if is_repeat else None
                x = model.blocks[phys_idx](x, x0_use, q_w, k_w, v_w, out_w, up_w, down_w,
                                            v_embed=ve, down_w_override=dw_override)
                if is_repeat and x_prev is not None:
                    x = model._apply_repeat_scale(x_prev, x, phys_idx)
                skips.append(x.clone())

            # Decoder phase
            for vi in range(num_virtual - num_enc):
                phys_idx, is_repeat = virtual_layers[num_enc + vi]

                if skips and vi < model.num_skip_weights:
                    scaled_skip = model.skip_weights[vi].to(dtype=x.dtype)[None, None, :] * skips.pop()
                    if model.skip_gates is not None:
                        g = torch.sigmoid(model.skip_gates[vi].to(dtype=x.dtype))[None, None, :]
                        x = torch.lerp(scaled_skip, x, g)
                    else:
                        x = x + scaled_skip

                if phys_idx >= parallel_thresh and not is_parallel:
                    lane0 = x
                    lane1 = x
                    is_parallel = True

                if is_parallel:
                    block = model.blocks[phys_idx]
                    q_w, k_w, v_w, out_w, up_w, down_w = model._bank_weights(phys_idx)
                    ve = model._get_ve(phys_idx, input_ids, ve_cache)
                    dw_override = model._get_repeat_down_w(phys_idx) if is_repeat else None
                    effective_down_w = dw_override if dw_override is not None else down_w

                    mix = block.resid_mix.to(dtype=lane0.dtype)
                    attn_in = mix[0][None, None, :] * lane0 + mix[1][None, None, :] * x0_use
                    attn_out = block.attn(block.attn_norm(attn_in) * block.ln_scale_factor,
                                          q_w, k_w, v_w, out_w, v_embed=ve)
                    lane0 = attn_in + block.attn_scale.to(dtype=attn_in.dtype)[None, None, :] * attn_out

                    mlp_in = block.mlp_norm(lane1) * block.ln_scale_factor
                    mlp_out = block.mlp(mlp_in, up_w, effective_down_w)
                    lane1 = lane1 + block.mlp_scale.to(dtype=lane1.dtype)[None, None, :] * mlp_out
                else:
                    q_w, k_w, v_w, out_w, up_w, down_w = model._bank_weights(phys_idx)
                    ve = model._get_ve(phys_idx, input_ids, ve_cache)
                    dw_override = model._get_repeat_down_w(phys_idx) if is_repeat else None
                    x_prev = x if is_repeat else None
                    x = model.blocks[phys_idx](x, x0_use, q_w, k_w, v_w, out_w, up_w, down_w,
                                               v_embed=ve, down_w_override=dw_override)
                    if is_repeat and x_prev is not None:
                        x = model._apply_repeat_scale(x_prev, x, phys_idx)

            if is_parallel:
                m = model.lane_merge.to(dtype=lane0.dtype)
                x = m * lane0 + (1 - m) * lane1

            x = model.final_norm(x)
            x_flat = x.reshape(-1, x.size(-1))
            targets = target_ids.reshape(-1)

            if model.tie_embeddings:
                logits_proj = F.linear(x_flat, model.tok_emb.weight)
            else:
                logits_proj = model.lm_head(x_flat)

            logits = model.logit_softcap * torch.tanh(logits_proj / model.logit_softcap)
            loss = F.cross_entropy(logits.float(), targets, reduction="mean")
            return loss.item()


def corrupt_gaussian(h: Tensor, noise_std_fraction: float, rng: torch.Generator) -> Tensor:
    h_std = h.float().std().item()
    noise_std = noise_std_fraction * h_std
    noise = torch.randn(h.shape, dtype=torch.float32, device=h.device, generator=rng)
    return (h.float() + noise * noise_std).to(dtype=h.dtype)


def corrupt_random_same_norm(h: Tensor, rng: torch.Generator) -> Tensor:
    h_f = h.float()
    norms = h_f.norm(dim=-1, keepdim=True)
    random_vec = torch.randn(h.shape, dtype=torch.float32, device=h.device, generator=rng)
    random_norms = random_vec.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return ((random_vec / random_norms) * norms).to(dtype=h.dtype)


def main():
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    print("=" * 70)
    print("Layer-2 Corruption Sensitivity Experiment")
    print("=" * 70)

    model = build_model()
    load_checkpoint(model)
    model.set_recurrence_active(True)
    model.eval()

    vl = model._get_virtual_layers()
    print(f"\nVirtual layer sequence ({len(vl)} total):")
    for vi, (phys, rep) in enumerate(vl):
        enc_dec = "ENC" if vi < len(vl) // 2 else "DEC"
        print(f"  vi={vi:2d} [{enc_dec}] phys={phys} repeat={rep}")

    rng = torch.Generator(device=DEVICE)
    rng.manual_seed(SEED)
    input_ids = torch.randint(0, 4096, (B, T), device=DEVICE, generator=rng)
    target_ids = torch.randint(0, 4096, (B, T), device=DEVICE, generator=rng)
    print(f"\nInput: B={B}, T={T}, vocab=4096")

    # Clean full forward pass
    clean_loss_full = forward_full(model, input_ids, target_ids)
    print(f"\nClean full forward pass loss: {clean_loss_full:.6f}")

    # Split pass: run layers 0-2, intercept
    h_real, x0, skips, ve_cache = split_forward_at_layer2(model, input_ids)
    print(f"\nh_real (hidden state after layer 2):")
    print(f"  shape: {h_real.shape}")
    print(f"  mean:  {h_real.float().mean():.6f}")
    print(f"  std:   {h_real.float().std():.6f}")
    print(f"  per-token norm (avg): {h_real.float().norm(dim=-1).mean():.4f}")

    clean_split_loss = forward_from_layer3(model, input_ids, h_real, x0, skips, ve_cache, target_ids)
    print(f"\nClean split pass loss:  {clean_split_loss:.6f}")
    print(f"Full vs split diff:     {abs(clean_split_loss - clean_loss_full):.8f}")

    # Corruption experiments
    print("\n" + "=" * 70)
    print("Corruption experiments (5 trials each, varying RNG seed):")
    print("=" * 70)

    NUM_TRIALS = 5
    corruption_levels = [
        ("clean (split baseline)",             None,          None),
        ("10% Gaussian noise",                 "gaussian",    0.10),
        ("50% Gaussian noise",                 "gaussian",    0.50),
        ("100% corruption (random same norm)", "random_norm", None),
    ]

    results = {}
    for label, corruption_type, param in corruption_levels:
        trial_losses = []
        for trial in range(NUM_TRIALS):
            trial_rng = torch.Generator(device=DEVICE)
            trial_rng.manual_seed(SEED + trial * 1000)

            if corruption_type is None:
                h_corr = h_real.clone()
            elif corruption_type == "gaussian":
                h_corr = corrupt_gaussian(h_real, param, trial_rng)
            else:  # random_norm
                h_corr = corrupt_random_same_norm(h_real, trial_rng)

            loss = forward_from_layer3(model, input_ids, h_corr, x0, skips, ve_cache, target_ids)
            trial_losses.append(loss)

        avg_loss = sum(trial_losses) / len(trial_losses)
        std_loss = (sum((l - avg_loss)**2 for l in trial_losses) / len(trial_losses)) ** 0.5
        results[label] = avg_loss
        print(f"\n{label}:")
        print(f"  Avg loss = {avg_loss:.6f}  std = {std_loss:.6f}")
        print(f"  Per-trial: {[f'{l:.4f}' for l in trial_losses]}")

    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    baseline = results["clean (split baseline)"]
    print(f"{'Corruption Level':<45} {'Loss':>10}  {'Delta':>10}  {'Ratio':>8}")
    print("-" * 75)
    for label, _, _ in corruption_levels:
        loss = results[label]
        delta = loss - baseline
        ratio = loss / baseline
        sign = "+" if delta >= 0 else ""
        print(f"{label:<45} {loss:>10.6f}  {sign}{delta:>9.6f}  {ratio:>8.4f}x")

    print(f"\nFull forward (reference):  {clean_loss_full:.6f}")
    print(f"Split baseline (layer3+):  {clean_split_loss:.6f}")

    r10  = results["10% Gaussian noise"] / baseline
    r50  = results["50% Gaussian noise"] / baseline
    r100 = results["100% corruption (random same norm)"] / baseline

    print("\nSensitivity analysis:")
    print(f"  10% noise  -> {r10:.4f}x loss  ({(r10-1)*100:+.2f}%)")
    print(f"  50% noise  -> {r50:.4f}x loss  ({(r50-1)*100:+.2f}%)")
    print(f"  100% corr  -> {r100:.4f}x loss  ({(r100-1)*100:+.2f}%)")

    if r10 > 2.0:
        verdict = "STRONGLY AMPLIFIED -- 10% noise more than doubles the loss"
    elif r10 > 1.5:
        verdict = "HIGHLY SENSITIVE (amplified) -- small corruption causes large downstream damage"
    elif r10 > 1.1:
        verdict = "MODERATELY SENSITIVE -- layers 3+ somewhat depend on early layer quality"
    elif r10 > 1.01:
        verdict = "WEAKLY SENSITIVE -- minor degradation from 10% noise"
    else:
        verdict = "ROBUST -- 10% noise causes negligible loss increase"

    print(f"\nVerdict: {verdict}")


if __name__ == "__main__":
    main()
