#!/bin/bash
# v8 R5: All three fixes combined (skip[4] + per-layer lane_merge + LoRA r=32)
# Baseline comparison: R4 combo (already ran — same config without these fixes)
# Single GPU only — other GPU reserved for v8.5 development
set -uo pipefail
source ${VENV:-./venv}/bin/activate

REPO_DIR="${PROJECT_ROOT:-.}/parameter-golf"
SCRIPT_DIR="${PROJECT_ROOT:-.}/h100_final"
RESULTS_DIR="${PROJECT_ROOT:-.}/local_tests/v8_sweep_r5"
mkdir -p "$RESULTS_DIR"
cd "$REPO_DIR"

cp "$SCRIPT_DIR/train_gpt_v8_experimental.py" "$REPO_DIR/train_gpt.py"
export CUDA_VISIBLE_DEVICES=0
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
export XSA_LAST_N=11 LN_SCALE=1 TIE_EMBEDDINGS=1
export ATTN_RES=0 RECUR_SKIP_PHYSICAL=0
export EMA_DECAY=0.9965 WARMDOWN_ITERS=3500 SWA_ENABLED=0
export RECUR_START_STEP=2000 QUANT_BITS_EMBED=8 REPEAT_PASS_SCALE=1
export GPTQ_ENABLED=1 GPTQ_CALIB_SEQS=64 COMPRESSOR=brotli
export QUANT_BITS_MLP=6 QUANT_BITS_ATTN=6
# R4 winning settings
export ROPE_BASE=10000
export LOGIT_SOFTCAP=20
export DATA_WARMDOWN_COUPLING=0
# === R5 FIXES ===
# Fix 1: skip[4] dead code fix — baked into the new code
# Fix 2: per-layer lane_merge (single-stream parallel)
export PER_LAYER_LANE_MERGE=1
# Fix 3: LoRA rank 16 → 32
export REPEAT_UNTIE_MLP=lora REPEAT_LORA_RANK=32

LOGFILE="$RESULTS_DIR/all_fixes_$(date +%Y%m%d_%H%M%S).log"
echo "$(date) R5: ALL FIXES (skip[4] + per-layer lane_merge + LoRA r=32) on GPU 0"
echo "Log: $LOGFILE"
nice -n 10 torchrun --standalone --nproc_per_node=1 train_gpt.py 2>&1 | tee "$LOGFILE"
echo "$(date) R5 complete"
