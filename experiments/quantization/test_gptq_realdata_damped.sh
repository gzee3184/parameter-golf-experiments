#!/bin/bash
# Long test: GPTQ real-data Hessians WITH pre-damping restored
# Isolates the effect of real data alone (vs previous test which also removed damping)
# Previous GPTQ-fix test (no damping): 43.3% prune, 1.1756 quant — CATASTROPHIC
# Ffix baseline: 0% prune, 1.1117 quant
# This test: real data + pre-damping (0.01*diag_mean*I in _cht) — should fix conditioning
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
# Ffix config
export QUANT_BITS_MLP=6 QUANT_BITS_ATTN=5
export ROPE_BASE=22000
export LOGIT_SOFTCAP=20
export DATA_WARMDOWN_COUPLING=0
export LN_SCALE=0
export REPEAT_UNTIE_MLP=lora REPEAT_LORA_RANK=16
export PER_LAYER_LANE_MERGE=0
export REPEAT_ATTN_LORA=0 REPEAT_ATTN_LORA_RANK=8
export DISABLE_LAYER0_MLP=0 SKIP_REPEAT_ATTN=0
export SKIP_REPEAT_ATTN_LAYERS=""
export SLIDING_WINDOW_EVAL=1 EVAL_STRIDE=64
# No TTT
export TTT_ENABLED=0
# No CAGE-lite
export CAGE_LITE=0

cd ${PROJECT_ROOT:-.}/parameter-golf
echo "$(date) ── GPTQ real-data + pre-damping test (3hr, GPU 1) ──"
echo "Isolating real-data effect: real data Hessians WITH 0.01*diag_mean*I pre-damping"
echo "Compare vs Ffix baseline: post-EMA 1.0986, quant 1.1117"
echo "Compare vs GPTQ-fix (no damping): quant 1.1756 (BROKEN)"
PYTHONUNBUFFERED=1 nice -n 10 torchrun --standalone --nproc_per_node=1 \
    ${PROJECT_ROOT:-.}/h100_deploy/train_gpt.py 2>&1 | tee ${PROJECT_ROOT:-.}/h100_deploy/gptq_realdata_damped_test.log
echo "$(date) ── COMPLETE ──"
