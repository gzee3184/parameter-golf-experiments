#!/bin/bash
# H100 8×GPU Run #10c — SP8192 + QUANT_BITS_EMBED=6 (RESUME from Run #10 checkpoint)
#
# Run #10 results: post_ema=1.1033 (best ever), but GPTQ crashed twice:
#   10a: NCCL timeout — ranks 1-7 died waiting for rank 0 during GPTQ (24K calib seqs)
#   10b: GPTQ completed but 41.1% pruning (16.35MB unpruned, 1.11MB over budget)
#        Root cause: SP8192 int8 embedding too large for 16MB budget
#
# Fix for 10c: QUANT_BITS_EMBED=6 (saves ~1MB, should eliminate pruning)
# Code fixes: NCCL timeout 1800s, quant gated behind ms, calib capped at 256 seqs
#
# History:
#   Run #3:  1.1131 sliding — BEST SP4096 (0% pruning)
#   Run #10: post_ema=1.1033 — BEST pre-quant ever (SP8192)
#   Run #10b: 41.1% pruning killed BPB — int8 embed too large
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
export QUANT_BITS_MLP=6 QUANT_BITS_ATTN=5 QUANT_BITS_EMBED=6
export GPTQ_ENABLED=1 GPTQ_CALIB_SEQS=64 COMPRESSOR=brotli
export GPTQ_REAL_DATA=1
export BYTE_SHUFFLE=1

# Resume from Run #10 checkpoint (skip training, do GPTQ + eval only)
export RESUME_QUANT=${RESUME_QUANT:-0}  # 1 = re-quantize an existing final_model.pt without training

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

echo "$(date) ── H100 8×GPU Run #10c (SP8192, EMBED=int6, RESUME, seed=$SEED) ──"
echo "Config: SP8192 RPS=1 ROPE=10K SDClip(12.85/20.0) WARMDOWN=72% TTT_LR=0.005 GPTQ_REAL VE=OFF EMBED=int6"
torchrun --standalone --nproc_per_node=8 train_gpt.py 2>&1 | tee run10c_seed${SEED}.log
echo "$(date) ── Run #10c seed=$SEED COMPLETE ──"
