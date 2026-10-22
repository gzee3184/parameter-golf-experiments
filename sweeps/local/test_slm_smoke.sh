#!/bin/bash
# Smoke tests for SLM (interleaving) + shared attention implementation
# Tests backward compat, SLM forward pass, shared attention, and roundtrip
# Run on any GPU: bash test_slm_smoke.sh [GPU_ID]

cd ${PROJECT_ROOT:-.}/parameter-golf
source ${VENV:-./venv}/bin/activate

SCRIPT="${PROJECT_ROOT:-.}/local_tests/train_gpt_slm.py"
RESULTS="${PROJECT_ROOT:-.}/local_tests"

GPU=${1:-0}
PORT=$((29600 + GPU))
export CUDA_VISIBLE_DEVICES=$GPU

# Shared base config (minimal — just enough to train + quant)
export DATA_PATH="./data/datasets/fineweb10B_sp4096"
export TOKENIZER_PATH="./data/tokenizers/fineweb_4096_bpe.model"
export VOCAB_SIZE=4096
export MODEL_DIM=512
export MLP_MULT=4
export NUM_HEADS=8
export NUM_KV_HEADS=4
export SIGMOID_SKIP=1
export SMEAR_ENABLED=0
export BYTE_SHUFFLE=1
export COPRIME_STRIDE=1
export COMPRESSOR=brotli
export MUON_WD=0.085
export ADAM_WD=0.02
export EMBED_WD=0.085
export MATRIX_LR=0.02
export SCALAR_LR=0.02
export QK_GAIN_INIT=4.0
export LOGIT_SOFTCAP=30.0
export MLP_ACTIVATION=leaky05
export BIGRAM_VOCAB_SIZE=0
export VE_ENABLED=0
export QAT_ENABLED=0
export LATE_QAT_THRESHOLD=0
export GATED_ATTENTION=0
export VALUE_RESIDUAL=0
export MTP_NUM_HEADS=0
export QUANT_BITS_MLP=6
export QUANT_BITS_ATTN=5
export GPTQ_ENABLED=1
export GPTQ_CALIB_SEQS=4
export TARGET_MB=15.9
export SEED=42
export EVAL_SEQ_LEN=2048
export SKIP_SLIDING=1
export TTT_ENABLED=0
export NGRAM_CACHE=0
export MUON_PLUS=1
export EMA_DECAY=0.997
export MAX_WALLCLOCK_SECONDS=60

FAIL=0

run_test() {
    local NAME="$1"
    local LOGFILE="$RESULTS/slm_test_${NAME}.log"
    echo ""
    echo "$(date) ═══════════════════════════════════════"
    echo "  TEST: $NAME"
    echo "═══════════════════════════════════════"
    nice -n 10 torchrun --standalone --nproc_per_node=1 --master_port=$PORT "$SCRIPT" \
        > "$LOGFILE" 2>&1
    RC=$?
    if [ $RC -ne 0 ]; then
        echo "  FAILED (exit code $RC)"
        tail -20 "$LOGFILE"
        FAIL=1
        return 1
    fi
    local VAL=$(grep -oP 'DIAGNOSTIC post_ema.*val_bpb:\K[0-9.]+' "$LOGFILE" 2>/dev/null)
    local QVAL=$(grep -oP 'roundtrip_exact.*val_bpb:\K[0-9.]+' "$LOGFILE" 2>/dev/null | tail -1)
    local STEPS=$(grep -oP 'step:\K[0-9]+(?=/)' "$LOGFILE" 2>/dev/null | tail -1)
    local SIZE=$(grep "Total submission size" "$LOGFILE" 2>/dev/null | grep -oP '[0-9]+(?= bytes)')
    local PARAMS=$(grep -oP 'model_params:\K[0-9]+' "$LOGFILE" 2>/dev/null)
    local SLM_LOG=$(grep "slm_config:" "$LOGFILE" 2>/dev/null)
    if [ -n "$SIZE" ]; then local SIZE_MB=$(echo "scale=2; $SIZE / 1048576" | bc); else local SIZE_MB="—"; fi
    echo "  $NAME: prequant=$VAL quant=$QVAL steps=$STEPS art=${SIZE_MB}MB params=$PARAMS"
    echo "  $SLM_LOG"
    return 0
}

# ═══════════════════════════════════════
# TEST 1: Backward compatibility (no SLM env vars)
# Should produce identical behavior to original train_gpt.py
# ═══════════════════════════════════════
export NUM_LAYERS=5
export XSA_LAST_N=5
# Ensure SLM vars are unset
unset TOTAL_LAYERS
unset ATTN_PATTERN
unset NUM_UNIQUE_ATTN
run_test "backward_compat_5L"

# ═══════════════════════════════════════
# TEST 2: SLM interleaving — 7 total layers, 4 with attention (uniform:2)
# Layers 0,2,4,6 have attention; layers 1,3,5 are MLP-only
# ═══════════════════════════════════════
export TOTAL_LAYERS=7
export ATTN_PATTERN="uniform:2"
export NUM_UNIQUE_ATTN=0
export XSA_LAST_N=4
run_test "slm_7L_uniform2"

# ═══════════════════════════════════════
# TEST 3: SLM sandwich pattern — 7 total, first 2 + last 2 have attention
# ═══════════════════════════════════════
export TOTAL_LAYERS=7
export ATTN_PATTERN="sandwich:2"
export NUM_UNIQUE_ATTN=0
export XSA_LAST_N=4
run_test "slm_7L_sandwich2"

# ═══════════════════════════════════════
# TEST 4: Shared attention — 7 layers, all have attention, but only 3 unique sets
# ═══════════════════════════════════════
unset TOTAL_LAYERS
export NUM_LAYERS=7
export ATTN_PATTERN=""
export NUM_UNIQUE_ATTN=3
export XSA_LAST_N=7
run_test "shared_attn_7L_3unique"

# ═══════════════════════════════════════
# TEST 5: Combined SLM + shared attention — 9 total, 6 attn, 3 unique attn sets
# ═══════════════════════════════════════
export TOTAL_LAYERS=9
export ATTN_PATTERN="uniform:2"  # attn at 0,2,4,6,8 = 5 layers
export NUM_UNIQUE_ATTN=3
export XSA_LAST_N=5
run_test "combined_9L_5attn_3unique"

echo ""
echo "$(date) ═══════════════════════════════════════"
if [ $FAIL -eq 0 ]; then
    echo "  ALL TESTS PASSED"
else
    echo "  SOME TESTS FAILED"
fi
echo "═══════════════════════════════════════"
