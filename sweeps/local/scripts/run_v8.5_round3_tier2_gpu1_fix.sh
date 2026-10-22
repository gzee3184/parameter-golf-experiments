#!/bin/bash
# v8.5 Round 3 Tier 2 — GPU 1 (relaunch): C-L5only then E-fix
# C-L5only: Skip repeat attention ONLY on L5 (confirmed dead: grad=0.000)
# E-fix: Parallel grad scale 2.5× (was killed, needs full run)
set -uo pipefail
source ${VENV:-./venv}/bin/activate

REPO_DIR="${PROJECT_ROOT:-.}/parameter-golf"
SCRIPT_DIR="${PROJECT_ROOT:-.}/h100_final"
RESULTS_DIR="${PROJECT_ROOT:-.}/local_tests/v8.5_sweep"
mkdir -p "$RESULTS_DIR"
cd "$REPO_DIR"

cp "$SCRIPT_DIR/train_gpt_v8_experimental.py" "$REPO_DIR/train_gpt.py"

# F-fix base config (our current best)
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
export QUANT_BITS_MLP=6 QUANT_BITS_ATTN=6
export ROPE_BASE=22000
export LOGIT_SOFTCAP=20
export DATA_WARMDOWN_COUPLING=0
export LN_SCALE=0
export REPEAT_UNTIE_MLP=lora REPEAT_LORA_RANK=16
export PER_LAYER_LANE_MERGE=0
export REPEAT_ATTN_LORA=0 REPEAT_ATTN_LORA_RANK=8
export DISABLE_LAYER0_MLP=0 SKIP_REPEAT_ATTN=0
export PARALLEL_SCALAR_GRAD_SCALE=0

# ── Test C-L5only: Skip repeat attention ONLY on layer 5 ──
export SKIP_REPEAT_ATTN_LAYERS="5"
LOGFILE="$RESULTS_DIR/CL5_skip_repeat_attn_l5only_$(date +%Y%m%d_%H%M%S).log"
echo "$(date) ── Test C-L5only: Skip Repeat Attn L5 only (on F-fix base) GPU 1 ──"
echo "Log: $LOGFILE"
nice -n 10 torchrun --standalone --nproc_per_node=1 train_gpt.py 2>&1 | tee "$LOGFILE"
echo "$(date) Test C-L5only complete"
echo ""

# ── Test E-fix: Parallel Scalar Gradient Scaling (2.5×) ──
export SKIP_REPEAT_ATTN_LAYERS=""
export PARALLEL_SCALAR_GRAD_SCALE=2.5
LOGFILE="$RESULTS_DIR/Efix_parallel_grad_scale_$(date +%Y%m%d_%H%M%S).log"
echo "$(date) ── Test E-fix: Parallel Grad Scale 2.5× (on F-fix base) GPU 1 ──"
echo "Log: $LOGFILE"
nice -n 10 torchrun --standalone --nproc_per_node=1 train_gpt.py 2>&1 | tee "$LOGFILE"
echo "$(date) Test E-fix complete"

echo "$(date) ═══ GPU 1 (C-L5only + E-fix) COMPLETE ═══"
