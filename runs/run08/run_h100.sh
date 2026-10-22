#!/bin/bash
# H100 8×GPU Run #8 — RESUME_QUANT: Run #3 model + SDClip + TTT
#
# History:
#   Run #3: 1.1131 sliding, post-EMA 1.1100, quant gap 0.0031 — BEST
#   Run #5: 1.1148 sliding (RESUME_QUANT of R3, competitor arch) — SDClip quant gap 0.0035
#   Run #6: 1.1198 sliding (fresh train, new code) — WORST, code regression +0.006
#   Run #7: 1.1161 sliding (fresh train, new code) — SDClip quant gap 0.0012 (91% reduction)
#
# Root cause analysis (deep forensic, 2026-04-11):
#   Runs 6/7 regressed +0.004 BPB vs Run #3 due to:
#   1. REPEAT_PASS_SCALE=0 (was 1 in R3) — removed learnable recurrence gating
#   2. ROPE_BASE=22000 (was 10000 in R3) — wrong positional encoding
#   3. Code edits (fwl fix, RESUME_QUANT) changed torch.compile graph
#
# Strategy: Skip training entirely. Load Run #3's final_model.pt
#   and re-quantize with SDClip (proven 91% quant gap reduction) + TTT
#   - RESUME_QUANT=1: loads final_model.pt, skips training loop
#   - REPEAT_PASS_SCALE=1: REQUIRED for state dict compatibility (R3 has rps params)
#   - ROPE_BASE=10000: Run #3's actual value (not 22000!)
#   - SDCLIP=1 + TTT: post-training improvements
#
# Projected: post-EMA 1.1100 + quant gap ~0.001 → sliding ~1.111
#            + TTT -0.004 → ~1.107
set -uo pipefail

# Architecture (unchanged)
export VOCAB_SIZE=4096 NUM_LAYERS=11 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=4
export MLP_ACTIVATION=leaky05
export DATA_PATH="./data/datasets/fineweb10B_sp4096"
export TOKENIZER_PATH="./data/tokenizers/fineweb_4096_bpe.model"
export MAX_WALLCLOCK_SECONDS=600
export SEED=1337 TRAIN_BATCH_TOKENS=786432 TRAIN_SEQ_LEN=2048 EVAL_SEQ_LEN=2048
export ITERATIONS=20000 WARMUP_STEPS=20
export VAL_LOSS_EVERY=4000 TRAIN_LOG_EVERY=100

# Optimizer (Run #3 proven recipe)
export MUON_PLUS=1 MATRIX_LR=0.02 SCALAR_LR=0.02
export MUON_WD=0.090 ADAM_WD=0.02 EMBED_WD=0.090
export QK_GAIN_INIT=5.0 GRAD_CLIP_NORM=0.3

# Skip / recurrence / structure (Run #3 recipe)
export SIGMOID_SKIP=1 COPRIME_STRIDE=1
export RECUR_LAYERS="3,4,5" RECUR_NUM_LOOPS=2 RECUR_START_STEP=2000
export VE_ENABLED=1 VE_DIM=128 VE_LAYERS="9,10"
export PARALLEL_START_LAYER=7 DISABLE_LAYER0_ATTN=1
export XSA_LAST_N=11 TIE_EMBEDDINGS=1
export REPEAT_PASS_SCALE=1

# EMA and warmdown (Run #3 recipe)
export EMA_DECAY=0.9965 WARMDOWN_ITERS=3500

# Training recipe (Run #3 ACTUAL values — verified from run3_output.log)
export ROPE_BASE=10000
export LOGIT_SOFTCAP=20

# LoRA repeat MLP
export REPEAT_UNTIE_MLP=lora REPEAT_LORA_RANK=16

# Quantization config
export QUANT_BITS_MLP=6 QUANT_BITS_ATTN=5 QUANT_BITS_EMBED=8
export GPTQ_ENABLED=1 GPTQ_CALIB_SEQS=64 COMPRESSOR=brotli
export GPTQ_REAL_DATA=0
export BYTE_SHUFFLE=0

# RESUME_QUANT: Skip training, load Run #3's model, do GPTQ+eval+TTT only
export RESUME_QUANT=1

# Late QAT: DISABLED — minimal effect (Run #7 proved only 0.0008 delta)
export LATE_QAT=0

# SDClip: entropy-optimal clipping (post-training, confirmed 69-73% quant gap reduction)
export SDCLIP=1 SDCLIP_K=12.85

# Sliding window eval (stride=64)
export SLIDING_WINDOW_EVAL=1 EVAL_STRIDE=64

# LZMA wrapper size
export CODE_SIZE_OVERRIDE=23294

# TTT — AdamW lr=1e-4 (local sweep 2026-04-11: +0.0107 BPB vs SGD baseline)
# Sweep results (local, 33% pruned model — relative rankings transfer):
#   adamw_f0  (AdamW lr=1e-4 freeze=0 ep=3): 1.1173 ← BEST
#   adamw_con (AdamW lr=1e-4 freeze=2 ep=3): 1.1174
#   sgd_base  (SGD   lr=0.002       ep=3):   1.1280 (current H100 default)
export TTT_ENABLED=1 TTT_OPTIMIZER=adamw TTT_LR=0.0001 TTT_WD=0.01
export TTT_EPOCHS=3 TTT_CHUNK_TOKENS=32768
export TTT_FREEZE_BLOCKS=0 TTT_MOMENTUM=0.9 TTT_BATCH_SEQS=32 TTT_GRAD_CLIP=1.0

echo "$(date) ── H100 8×GPU Run #8 (RESUME_QUANT: Run #3 model + SDClip + TTT) ──"
echo "Key: RESUME_QUANT=1 RPS=1 ROPE=10K + SDClip + TTT"
echo "NOTE: Requires final_model.pt on pod (copy from final_model_r3.pt, 133MB, 60 keys)"
torchrun --standalone --nproc_per_node=8 train_gpt.py 2>&1 | tee run.log
echo "$(date) ── Run #8 COMPLETE ──"
