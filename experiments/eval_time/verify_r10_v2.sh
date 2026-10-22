#!/bin/bash
# Run verify_r10_checkpoint.py under the EXACT environment that run10c used
# (minus distributed / TTT / quant settings).
set -eo pipefail

cd ${PROJECT_ROOT:-.}
source venv/bin/activate

# Architecture (SP8192 adaptation) — matches run_h100.sh for run10c
export VOCAB_SIZE=8192 NUM_LAYERS=11 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=4
export MLP_ACTIVATION=leaky05
export DATA_PATH="./parameter-golf/data/datasets/fineweb10B_sp8192"
export TOKENIZER_PATH="./parameter-golf/data/tokenizers/fineweb_8192_bpe.model"
export SEED=1337 TRAIN_SEQ_LEN=2048 EVAL_SEQ_LEN=2048
export VAL_BATCH_SIZE=524288

# Optimizer (for completeness, won't affect eval)
export MUON_PLUS=1 MATRIX_LR=0.02 SCALAR_LR=0.02
export MUON_WD=0.090 ADAM_WD=0.02 EMBED_WD=0.090
export QK_GAIN_INIT=5.0 GRAD_CLIP_NORM=0.3

# Architecture options used at Run #10c
export SIGMOID_SKIP=1
export RECUR_LAYERS="3,4,5" RECUR_NUM_LOOPS=2 RECUR_START_STEP=2000
export VE_ENABLED=0
export PARALLEL_START_LAYER=7 DISABLE_LAYER0_ATTN=1
export XSA_LAST_N=11 TIE_EMBEDDINGS=1
export REPEAT_PASS_SCALE=1

# EMA / warmdown
export EMA_DECAY=0.9965 WARMDOWN_FRAC=0.72 WARMDOWN_ITERS=3500
export ROPE_BASE=10000
export LOGIT_SOFTCAP=20

# LoRA repeat MLP
export REPEAT_UNTIE_MLP=lora REPEAT_LORA_RANK=16

# Rope dims (partial rope), NOT in run_h100.sh — using model default (16)
# Actually check default: baseline run_h100.sh does not set ROPE_DIMS, so default 16.

CUDA_VISIBLE_DEVICES=0 nice -n 10 python3 h100_deploy/experiments/verify_r10_checkpoint.py
