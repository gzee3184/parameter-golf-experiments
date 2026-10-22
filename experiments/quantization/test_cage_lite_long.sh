#!/bin/bash
# Long CAGE-lite test: 3 hours to get well-trained model (~5000+ steps)
# Short 30-min test showed CAGE hurt at 940 steps (undertrained).
# Hypothesis: CAGE-lite needs mature weights to be effective.
# Control comparison: v8 baseline results from EXPERIMENT_LOG section 17
#   (v8 baseline: 5579 steps, post-EMA 1.1000, quant 1.1131)
set -uo pipefail

source ${VENV:-./venv}/bin/activate
export CUDA_VISIBLE_DEVICES=1
export VOCAB_SIZE=4096 NUM_LAYERS=11 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=4
export MLP_ACTIVATION=leaky05
export DATA_PATH="./data/datasets/fineweb10B_sp4096"
export TOKENIZER_PATH="./data/tokenizers/fineweb_4096_bpe.model"
export MAX_WALLCLOCK_SECONDS=10800
export SEED=1337 TRAIN_BATCH_TOKENS=786432 TRAIN_SEQ_LEN=2048 EVAL_SEQ_LEN=2048
export ITERATIONS=20000 WARMUP_STEPS=20
export VAL_LOSS_EVERY=4000 TRAIN_LOG_EVERY=100
export MUON_PLUS=1 MATRIX_LR=0.02 SCALAR_LR=0.02
export MUON_WD=0.090 ADAM_WD=0.02 EMBED_WD=0.090
export QK_GAIN_INIT=5.0 GRAD_CLIP_NORM=0.3
export SIGMOID_SKIP=1 BYTE_SHUFFLE=1 COPRIME_STRIDE=1
export RECUR_LAYERS="3,4,5" RECUR_NUM_LOOPS=2
export VE_ENABLED=1 VE_DIM=128 VE_LAYERS="9,10"
export PARALLEL_START_LAYER=7 DISABLE_LAYER0_ATTN=1
export XSA_LAST_N=11 TIE_EMBEDDINGS=1
export ATTN_RES=0 RECUR_SKIP_PHYSICAL=0
export EMA_DECAY=0.9965 WARMDOWN_ITERS=3500 SWA_ENABLED=0
export RECUR_START_STEP=2000 QUANT_BITS_EMBED=8 REPEAT_PASS_SCALE=1
export GPTQ_ENABLED=1 GPTQ_CALIB_SEQS=64 COMPRESSOR=brotli
# INT6 for both (matches existing long-run baselines)
export QUANT_BITS_MLP=6 QUANT_BITS_ATTN=6
export ROPE_BASE=10000
export LOGIT_SOFTCAP=20
export DATA_WARMDOWN_COUPLING=0
export LN_SCALE=0
export REPEAT_UNTIE_MLP=lora REPEAT_LORA_RANK=16
export PER_LAYER_LANE_MERGE=0
export REPEAT_ATTN_LORA=0 REPEAT_ATTN_LORA_RANK=8
export DISABLE_LAYER0_MLP=0 SKIP_REPEAT_ATTN=0
export SKIP_REPEAT_ATTN_LAYERS=""
export SLIDING_WINDOW_EVAL=1 EVAL_STRIDE=64
# NO TTT — keep eval fast, we only care about quant roundtrip delta
export TTT_ENABLED=0
# CAGE-lite: push weights toward quant grid in last 15%
export CAGE_LITE=1 CAGE_LITE_FRAC=0.15 CAGE_LITE_MAX=0.01

cd ${PROJECT_ROOT:-.}/parameter-golf
echo "$(date) ── CAGE-lite LONG test (3hr train + eval, GPU 1) ──"
echo "Config: 11L int6 + CAGE_LITE=1, frac=0.15, max=0.01"
echo "Compare vs v8 baseline: 5579 steps, post-EMA 1.1000, quant 1.1131"
PYTHONUNBUFFERED=1 nice -n 10 torchrun --standalone --nproc_per_node=1 \
    ${PROJECT_ROOT:-.}/h100_deploy/train_gpt.py 2>&1 | tee ${PROJECT_ROOT:-.}/h100_deploy/cage_lite_long_test.log
echo "$(date) ── COMPLETE ──"
