#!/bin/bash
# H100 8×GPU Run #9 — Fresh training, single seed first
#
# Config: Run #3 proven recipe + fixes
#   - REPEAT_PASS_SCALE=1: learnable recurrence gating (was missing in Runs 6/7)
#   - ROPE_BASE=10000: Run #3's actual value (was incorrectly 22000 in Runs 6/7)
#   - SDCLIP=1: 91% quant gap reduction (proven in Run #7)
#   - SGD TTT (AdamW tested on Run #8: only -0.0012 vs SGD's -0.004 on H100)
#   - GPTQ_REAL_DATA=1: real Hessians (both #1 and #2 competitors use this)
#   - BYTE_SHUFFLE=1: better brotli compression (Run #3 had it, competitors have it)
#   - No RESUME_QUANT, no Late QAT
#
# Timing: ~17 min per seed (10 min train + 7 min eval)
#   If BPB is good, run seeds 42 and 2025 manually afterward
#
# History:
#   Run #3: 1.1131 sliding — BEST (old code, RPS=1, ROPE=10K)
#   Run #7: 1.1161 sliding — code regression +0.003, but SDClip quant gap 0.0012
#   Run #8: RESUME_QUANT test (not a valid submission)
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

# Training recipe (Run #3 ACTUAL values)
export ROPE_BASE=10000
export LOGIT_SOFTCAP=20

# LoRA repeat MLP
export REPEAT_UNTIE_MLP=lora REPEAT_LORA_RANK=16

# Quantization config
export QUANT_BITS_MLP=6 QUANT_BITS_ATTN=5 QUANT_BITS_EMBED=8
export GPTQ_ENABLED=1 GPTQ_CALIB_SEQS=64 COMPRESSOR=brotli
# Both top competitors use real training data for GPTQ Hessians (not synthetic)
export GPTQ_REAL_DATA=1
# Byte shuffle before brotli — Run #3 had it on, competitors have it on
export BYTE_SHUFFLE=1

# NO RESUME_QUANT — fresh training from scratch
export RESUME_QUANT=0

# Late QAT: DISABLED
export LATE_QAT=0

# SDClip: entropy-optimal clipping
export SDCLIP=1 SDCLIP_K=12.85

# Sliding window eval (stride=64)
export SLIDING_WINDOW_EVAL=1 EVAL_STRIDE=64

# LZMA wrapper size
export CODE_SIZE_OVERRIDE=23294

# TTT — SGD (AdamW lr=1e-4 tested on Run #8: only -0.0012 gain vs SGD's -0.004 on H100)
export TTT_ENABLED=1 TTT_OPTIMIZER=sgd TTT_LR=0.002 TTT_WD=0.01
export TTT_EPOCHS=3 TTT_CHUNK_TOKENS=32768
export TTT_FREEZE_BLOCKS=0 TTT_MOMENTUM=0.9 TTT_BATCH_SEQS=32 TTT_GRAD_CLIP=1.0

echo "$(date) ── H100 8×GPU Run #9 (fresh train, seed=$SEED) ──"
echo "Config: RPS=1 ROPE=10K SDClip SGD_TTT GPTQ_REAL_DATA BYTE_SHUFFLE"
torchrun --standalone --nproc_per_node=8 train_gpt.py 2>&1 | tee run9_seed${SEED}.log
echo "$(date) ── Run #9 seed=$SEED COMPLETE ──"
