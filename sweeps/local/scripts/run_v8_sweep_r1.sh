#!/bin/bash
# v8 sweep Round 1: Baseline (with RoPE fix) + AREA 2 (LN_SCALE=0)
# Both use v7 Test B config + RoPE fix as common base
# GPU 0: v8 baseline (control — same as v7 Test B but with correct RoPE)
# GPU 1: v8 baseline + LN_SCALE=0 (test if 1/sqrt(depth) hurts under recurrence)
set -uo pipefail
source ${VENV:-./venv}/bin/activate

REPO_DIR="${PROJECT_ROOT:-.}/parameter-golf"
SCRIPT_DIR="${PROJECT_ROOT:-.}/h100_final"
RESULTS_DIR="${PROJECT_ROOT:-.}/local_tests/v8_sweep_r1"
mkdir -p "$RESULTS_DIR"
cd "$REPO_DIR"
cp "$SCRIPT_DIR/train_gpt_v8_experimental.py" "$REPO_DIR/train_gpt.py"

# ===== COMMON ENV (v7 Test B config + RoPE fix) =====
export VOCAB_SIZE=4096 NUM_LAYERS=11 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=4
export MLP_ACTIVATION=leaky05
export DATA_PATH="./data/datasets/fineweb10B_sp4096"
export TOKENIZER_PATH="./data/tokenizers/fineweb_4096_bpe.model"
export MAX_WALLCLOCK_SECONDS=10800
export SEED=1337 TRAIN_BATCH_TOKENS=786432 TRAIN_SEQ_LEN=2048 EVAL_SEQ_LEN=2048
export ITERATIONS=20000 WARMUP_STEPS=20
export VAL_LOSS_EVERY=4000 TRAIN_LOG_EVERY=100
# Optimizer
export MUON_PLUS=1 MATRIX_LR=0.02 SCALAR_LR=0.02
export MUON_WD=0.090 ADAM_WD=0.02 EMBED_WD=0.090
export QK_GAIN_INIT=5.0 GRAD_CLIP_NORM=0.3
# Features
export SIGMOID_SKIP=1 BYTE_SHUFFLE=1 COPRIME_STRIDE=1
export RECUR_LAYERS="3,4,5" RECUR_NUM_LOOPS=2
export VE_ENABLED=1 VE_DIM=128 VE_LAYERS="9,10"
export PARALLEL_START_LAYER=7 DISABLE_LAYER0_ATTN=1
export LOGIT_SOFTCAP=30.0 XSA_LAST_N=11 LN_SCALE=1 TIE_EMBEDDINGS=1
export REPEAT_UNTIE_MLP=lora REPEAT_LORA_RANK=16
export ATTN_RES=0
# v7 Test B improvements
export EMA_DECAY=0.9965 WARMDOWN_ITERS=3500 SWA_ENABLED=0
export RECUR_START_STEP=2000 QUANT_BITS_EMBED=8 REPEAT_PASS_SCALE=1
# AREA 9 off by default
export RECUR_SKIP_PHYSICAL=0
# Quantization
export GPTQ_ENABLED=1 GPTQ_CALIB_SEQS=64 COMPRESSOR=brotli
export QUANT_BITS_MLP=6 QUANT_BITS_ATTN=6

# ===== TEST: v8 baseline (RoPE fix, everything else same as v7 Test B) =====
run_baseline() {
    export CUDA_VISIBLE_DEVICES=0
    # LN_SCALE=1 (default, inherited from common)
    LOGFILE="$RESULTS_DIR/baseline_$(date +%Y%m%d_%H%M%S).log"
    echo "$(date) v8 BASELINE: v7 Test B + RoPE fix (LN_SCALE=1)"
    echo "Log: $LOGFILE"
    nice -n 10 torchrun --standalone --nproc_per_node=1 train_gpt.py 2>&1 | tee "$LOGFILE"
    echo "$(date) v8 baseline complete"
}

# ===== TEST: AREA 2 — LN_SCALE=0 (disable 1/sqrt(depth) scaling) =====
run_area2() {
    export CUDA_VISIBLE_DEVICES=1
    export LN_SCALE=0
    LOGFILE="$RESULTS_DIR/area2_ln_scale_0_$(date +%Y%m%d_%H%M%S).log"
    echo "$(date) AREA 2: LN_SCALE=0 (disable depth scaling under recurrence)"
    echo "Log: $LOGFILE"
    nice -n 10 torchrun --standalone --nproc_per_node=1 train_gpt.py 2>&1 | tee "$LOGFILE"
    echo "$(date) AREA 2 complete"
}

# Launch both
run_baseline &
PID_BASE=$!
sleep 5
run_area2 &
PID_A2=$!

echo "$(date) Round 1 launched: baseline=$PID_BASE (GPU 0), area2=$PID_A2 (GPU 1)"
echo "Monitor: tail -f $RESULTS_DIR/*.log"
wait $PID_BASE
echo "$(date) Baseline finished (exit $?)"
wait $PID_A2
echo "$(date) AREA 2 finished (exit $?)"
echo "$(date) === Round 1 complete ==="
