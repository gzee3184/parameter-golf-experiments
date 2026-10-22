#!/bin/bash
# Sequential SLM experiments — one at a time, no GPU contention
# Uses GPU0 only, full 12000s wallclock each

cd ${PROJECT_ROOT:-.}/parameter-golf
source ${VENV:-./venv}/bin/activate

SCRIPT="${PROJECT_ROOT:-.}/local_tests/train_gpt_slm.py"
RESULTS="${PROJECT_ROOT:-.}/local_tests"
GPU=0
PORT=29600
export CUDA_VISIBLE_DEVICES=$GPU

# Base config (matches 13L baseline exactly)
export DATA_PATH="./data/datasets/fineweb10B_sp4096"
export TOKENIZER_PATH="./data/tokenizers/fineweb_4096_bpe.model"
export VOCAB_SIZE=4096 MODEL_DIM=512 MLP_MULT=4 NUM_HEADS=8 NUM_KV_HEADS=4
export SIGMOID_SKIP=1 SMEAR_ENABLED=0 BYTE_SHUFFLE=1 COPRIME_STRIDE=1
export COMPRESSOR=brotli MUON_WD=0.085 ADAM_WD=0.02 EMBED_WD=0.085
export MATRIX_LR=0.02 SCALAR_LR=0.02 QK_GAIN_INIT=4.0 LOGIT_SOFTCAP=30.0
export MLP_ACTIVATION=leaky05 BIGRAM_VOCAB_SIZE=0 VE_ENABLED=0
export QAT_ENABLED=0 LATE_QAT_THRESHOLD=0 GATED_ATTENTION=0 VALUE_RESIDUAL=0
export MTP_NUM_HEADS=0 QUANT_BITS_MLP=6 QUANT_BITS_ATTN=5
export GPTQ_ENABLED=1 GPTQ_CALIB_SEQS=8 TARGET_MB=15.9
export SEED=1337 EVAL_SEQ_LEN=2048 SKIP_SLIDING=1
export TTT_ENABLED=0 NGRAM_CACHE=0 MUON_PLUS=1 EMA_DECAY=0.997
export MAX_WALLCLOCK_SECONDS=12000

run_test() {
    local NAME="$1"
    local LOGFILE="$RESULTS/slm_long_${NAME}.log"
    echo ""
    echo "$(date) ═══════════════════════════════════════"
    echo "  RUN: $NAME"
    echo "═══════════════════════════════════════"
    nice -n 10 torchrun --standalone --nproc_per_node=1 --master_port=$PORT "$SCRIPT" \
        > "$LOGFILE" 2>&1
    RC=$?
    if [ $RC -ne 0 ]; then
        echo "  FAILED (exit code $RC)"
        tail -20 "$LOGFILE"
        return 1
    fi
    local VAL=$(grep -oP 'DIAGNOSTIC post_ema.*val_bpb:\K[0-9.]+' "$LOGFILE" 2>/dev/null)
    local QVAL=$(grep -oP 'roundtrip_exact.*val_bpb:\K[0-9.]+' "$LOGFILE" 2>/dev/null | tail -1)
    local STEPS=$(grep -oP 'step:\K[0-9]+(?=/)' "$LOGFILE" 2>/dev/null | tail -1)
    local SAVG=$(grep -oP 'step_avg:\K[0-9.]+' "$LOGFILE" 2>/dev/null | tail -1)
    local SIZE=$(grep "Total submission size" "$LOGFILE" 2>/dev/null | grep -oP '[0-9]+(?= bytes)')
    local PARAMS=$(grep -oP 'model_params:\K[0-9]+' "$LOGFILE" 2>/dev/null)
    local PRUNE=$(grep "selective_prune:.*pruning\|no pruning" "$LOGFILE" 2>/dev/null | tail -1)
    if [ -n "$SIZE" ]; then local SIZE_MB=$(echo "scale=2; $SIZE / 1048576" | bc); else local SIZE_MB="—"; fi
    echo "  $NAME: prequant=$VAL quant=$QVAL steps=$STEPS ms/step=$SAVG art=${SIZE_MB}MB params=$PARAMS"
    echo "  $PRUNE"
    return 0
}

# ═══════════════════════════════════════
# RUN 1: 15L uniform:2 — 8 attn, 7 MLP-only (~40M params, fair vs 13L)
# ═══════════════════════════════════════
export TOTAL_LAYERS=15
export ATTN_PATTERN="uniform:2"
export NUM_UNIQUE_ATTN=0
export XSA_LAST_N=8
run_test "15L_uniform2_8attn"

# ═══════════════════════════════════════
# RUN 2: 19L uniform:2 — 10 attn, 9 MLP-only (~50M params, depth test)
# ═══════════════════════════════════════
export TOTAL_LAYERS=19
export ATTN_PATTERN="uniform:2"
export NUM_UNIQUE_ATTN=0
export XSA_LAST_N=10
run_test "19L_uniform2_10attn"

echo ""
echo "$(date) ═══════════════════════════════════════"
echo "  ALL DONE"
echo "═══════════════════════════════════════"
