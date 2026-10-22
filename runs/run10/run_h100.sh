#!/bin/bash
# H100 8×GPU Run #10 — SP8192 + proven improvements
#
# Key change: SP8192 vocab (from 4096) — new #1 (1.0810) uses SP8192
#
# Config: Run #9 proven recipe adapted for SP8192 + competitor improvements
#   - VOCAB_SIZE=8192: larger vocab, ~1.57× better tokenizer compression
#   - REPEAT_PASS_SCALE=1: learnable recurrence gating
#   - ROPE_BASE=10000: proven value
#   - SDCLIP=1: entropy-optimal clipping (K=12.85 for matrices)
#   - SDCLIP_K_EMBED=20.0: separate higher K for embeddings (from competitor)
#   - GPTQ_REAL_DATA=1: real Hessians + embedding Hessian via final_norm hook
#   - BYTE_SHUFFLE=1: better brotli compression
#   - WARMDOWN_FRAC=0.72: 72% warmdown (from competitor, was ~67%)
#   - TTT_LR=0.005: 2.5× higher (competitor's value, eval-time only)
#   - VE_ENABLED=0: dropped — VE embedding grows with vocab, budget too tight
#   - SGD TTT (proven on H100, AdamW only -0.0012 vs SGD's -0.004)
#
# Budget concern:
#   SP8192 embedding is +2MB raw vs SP4096. With VE off + byte_shuffle,
#   estimated ~15.5-15.9MB. If over 16MB, try QUANT_BITS_EMBED=6.
#
# History:
#   Run #3: 1.1131 sliding — BEST SP4096 (old code, RPS=1, ROPE=10K)
#   Run #9: post_ema=1.1088 (better than R3's 1.1100!) — stopped during GPTQ
#   Competitor #1: 1.0810 sliding (SP8192, similar arch)
#
# Data: SP8192 must be downloaded on pod first:
#   cd parameter-golf && rm -f data/manifest.json
#   MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf \
#     python3 data/cached_challenge_fineweb.py --variant sp8192 --train-shards 143
set -uo pipefail

# Architecture (SP8192 adaptation)
export VOCAB_SIZE=8192 NUM_LAYERS=11 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=4
export MLP_ACTIVATION=leaky05
export DATA_PATH="./data/datasets/fineweb10B_sp8192"
export TOKENIZER_PATH="./data/tokenizers/fineweb_8192_bpe.model"
export MAX_WALLCLOCK_SECONDS=600
export SEED=1337 TRAIN_BATCH_TOKENS=786432 TRAIN_SEQ_LEN=2048 EVAL_SEQ_LEN=2048
export ITERATIONS=20000 WARMUP_STEPS=20
export VAL_LOSS_EVERY=4000 TRAIN_LOG_EVERY=100

# Optimizer (Run #3 proven recipe)
export MUON_PLUS=1 MATRIX_LR=0.02 SCALAR_LR=0.02
export MUON_WD=0.090 ADAM_WD=0.02 EMBED_WD=0.090
export QK_GAIN_INIT=5.0 GRAD_CLIP_NORM=0.3

# Skip / recurrence / structure (Run #3 recipe, VE disabled for SP8192 budget)
export SIGMOID_SKIP=1
export RECUR_LAYERS="3,4,5" RECUR_NUM_LOOPS=2 RECUR_START_STEP=2000
export VE_ENABLED=0
export PARALLEL_START_LAYER=7 DISABLE_LAYER0_ATTN=1
export XSA_LAST_N=11 TIE_EMBEDDINGS=1
export REPEAT_PASS_SCALE=1

# EMA and warmdown (72% fraction — from competitor, was ~67% at 3500 iters)
export EMA_DECAY=0.9965 WARMDOWN_FRAC=0.72 WARMDOWN_ITERS=3500

# Training recipe (proven values)
export ROPE_BASE=10000
export LOGIT_SOFTCAP=20

# LoRA repeat MLP
export REPEAT_UNTIE_MLP=lora REPEAT_LORA_RANK=16

# Quantization config
export QUANT_BITS_MLP=6 QUANT_BITS_ATTN=5 QUANT_BITS_EMBED=8
export GPTQ_ENABLED=1 GPTQ_CALIB_SEQS=64 COMPRESSOR=brotli
export GPTQ_REAL_DATA=1
export BYTE_SHUFFLE=1

# Fresh training from scratch
export RESUME_QUANT=0

# Late QAT: DISABLED
export LATE_QAT=0

# SDClip: entropy-optimal clipping (separate K for embeddings)
export SDCLIP=1 SDCLIP_K=12.85 SDCLIP_K_EMBED=20.0

# Sliding window eval (stride=64)
export SLIDING_WINDOW_EVAL=1 EVAL_STRIDE=64

# LZMA wrapper size
export CODE_SIZE_OVERRIDE=23524

# TTT — SGD with higher LR (competitor uses 0.005, 2.5× our old 0.002)
export TTT_ENABLED=1 TTT_OPTIMIZER=sgd TTT_LR=0.005 TTT_WD=0.01
export TTT_EPOCHS=3 TTT_CHUNK_TOKENS=32768
export TTT_FREEZE_BLOCKS=0 TTT_MOMENTUM=0.9 TTT_BATCH_SEQS=32 TTT_GRAD_CLIP=1.0

echo "$(date) ── H100 8×GPU Run #10 (SP8192, seed=$SEED) ──"
echo "Config: SP8192 RPS=1 ROPE=10K SDClip(12.85/20.0) WARMDOWN=72% TTT_LR=0.005 GPTQ_REAL VE=OFF"
torchrun --standalone --nproc_per_node=8 train_gpt.py 2>&1 | tee run10_seed${SEED}.log
echo "$(date) ── Run #10 seed=$SEED COMPLETE ──"
