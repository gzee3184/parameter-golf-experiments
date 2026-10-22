#!/bin/bash
# v7 sweep: 2 tests in parallel on 2 GPUs, 3 hours each (H100 equivalent)
# GPU 0: Test A — LoRA baseline (isolate LoRA vs v6 full bank)
# GPU 1: Test B — LoRA + all improvements (EMA 0.9965, warmdown 3500, embed int8,
#                 per-pass scale, recur start 2000, SWA off)
set -uo pipefail
source ${VENV:-./venv}/bin/activate

REPO_DIR="${PROJECT_ROOT:-.}/parameter-golf"
SCRIPT_DIR="${PROJECT_ROOT:-.}/h100_final"
RESULTS_DIR="${PROJECT_ROOT:-.}/local_tests/v7_sweep"
mkdir -p "$RESULTS_DIR"
cd "$REPO_DIR"
cp "$SCRIPT_DIR/train_gpt_v6.py" "$REPO_DIR/train_gpt.py"

# ===== COMMON ENV (shared by both tests) =====
# Architecture
export VOCAB_SIZE=4096 NUM_LAYERS=11 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=4
export MLP_ACTIVATION=leaky05
export DATA_PATH="./data/datasets/fineweb10B_sp4096"
export TOKENIZER_PATH="./data/tokenizers/fineweb_4096_bpe.model"
export MAX_WALLCLOCK_SECONDS=10800
export SEED=1337 TRAIN_BATCH_TOKENS=786432 TRAIN_SEQ_LEN=2048 EVAL_SEQ_LEN=2048
export ITERATIONS=20000 WARMUP_STEPS=20
export VAL_LOSS_EVERY=4000 TRAIN_LOG_EVERY=100
# Optimizer base
export MUON_PLUS=1 MATRIX_LR=0.02 SCALAR_LR=0.02
export MUON_WD=0.090 ADAM_WD=0.02 EMBED_WD=0.090
export QK_GAIN_INIT=5.0 GRAD_CLIP_NORM=0.3
# Features (shared)
export SIGMOID_SKIP=1 BYTE_SHUFFLE=1 COPRIME_STRIDE=1
export RECUR_LAYERS="3,4,5"
export VE_ENABLED=1 VE_DIM=128 VE_LAYERS="9,10"
export PARALLEL_START_LAYER=7 DISABLE_LAYER0_ATTN=1
export LOGIT_SOFTCAP=30.0 XSA_LAST_N=11 LN_SCALE=1 TIE_EMBEDDINGS=1
# LoRA (both tests)
export REPEAT_UNTIE_MLP=lora REPEAT_LORA_RANK=16
# Quantization base
export GPTQ_ENABLED=1 GPTQ_CALIB_SEQS=64 COMPRESSOR=brotli
export QUANT_BITS_MLP=6 QUANT_BITS_ATTN=6
# AttnRes off (confirmed bad)
export ATTN_RES=0 RECUR_NUM_LOOPS=2

# ===== TEST A: LoRA baseline (match v6 long settings except LoRA) =====
run_test_a() {
    export CUDA_VISIBLE_DEVICES=0
    # Same as v6 long except REPEAT_UNTIE_MLP=lora instead of down
    export EMA_DECAY=0.997
    export WARMDOWN_ITERS=2500
    export SWA_ENABLED=1
    export RECUR_START_STEP=3000
    export QUANT_BITS_EMBED=6
    export REPEAT_PASS_SCALE=0
    LOGFILE="$RESULTS_DIR/test_a_lora_baseline_$(date +%Y%m%d_%H%M%S).log"
    echo "$(date) TEST A: LoRA baseline — same as v6 long except lora r=16 instead of full bank"
    echo "Log: $LOGFILE"
    nice -n 10 torchrun --standalone --nproc_per_node=1 train_gpt.py 2>&1 | tee "$LOGFILE"
    echo "$(date) TEST A complete"
}

# ===== TEST B: LoRA + all improvements =====
run_test_b() {
    export CUDA_VISIBLE_DEVICES=1
    # All improvements bundled
    export EMA_DECAY=0.9965       # PR #1421/#1435 finding
    export WARMDOWN_ITERS=3500    # Match competitor ~67% warmdown fraction
    export SWA_ENABLED=0          # Disable dead SWA code
    export RECUR_START_STEP=2000  # Earlier activation (~0.38 fraction)
    export QUANT_BITS_EMBED=8     # Embed int8 (matching pr1326)
    export REPEAT_PASS_SCALE=1    # Per-pass recurrence scaling
    LOGFILE="$RESULTS_DIR/test_b_all_improvements_$(date +%Y%m%d_%H%M%S).log"
    echo "$(date) TEST B: LoRA + all improvements (EMA 0.9965, warmdown 3500, embed int8, per-pass scale, recur start 2000, SWA off)"
    echo "Log: $LOGFILE"
    nice -n 10 torchrun --standalone --nproc_per_node=1 train_gpt.py 2>&1 | tee "$LOGFILE"
    echo "$(date) TEST B complete"
}

# Launch both in parallel
run_test_a &
PID_A=$!
sleep 5  # stagger to avoid file system contention
run_test_b &
PID_B=$!

echo "$(date) Both tests launched: A=$PID_A (GPU 0), B=$PID_B (GPU 1)"
echo "Monitor: tail -f $RESULTS_DIR/test_a_*.log $RESULTS_DIR/test_b_*.log"
wait $PID_A
echo "$(date) Test A finished (exit $?)"
wait $PID_B
echo "$(date) Test B finished (exit $?)"
echo "$(date) === All sweep tests complete ==="
