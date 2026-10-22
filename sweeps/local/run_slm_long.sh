#!/bin/bash
# Long SLM experiment — simulated H100 run (12000s wallclock)
# Tests whether SLM interleaving beats 13L baseline after full training
#
# Rationale for config:
#   13L baseline: 39.6M params, 6125 steps, 1959ms/step, quant=1.1335
#   SLM 19L uniform:2: ~39M params (10 attn + 9 MLP-only), ~1000ms/step
#     → ~12000 steps in same wallclock = 2x more training
#   If MLP-only layers contribute, this should beat 13L.
#   If they don't, the extra steps alone might help.

cd ${PROJECT_ROOT:-.}/parameter-golf
source ${VENV:-./venv}/bin/activate

SCRIPT="${PROJECT_ROOT:-.}/local_tests/train_gpt_slm.py"
RESULTS="${PROJECT_ROOT:-.}/local_tests"

GPU=${1:-0}
PORT=$((29600 + GPU))
export CUDA_VISIBLE_DEVICES=$GPU

# Same base config as 13L baseline
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
export GPTQ_CALIB_SEQS=8
export TARGET_MB=15.9
export SEED=1337
export EVAL_SEQ_LEN=2048
export SKIP_SLIDING=1
export TTT_ENABLED=0
export NGRAM_CACHE=0
export MUON_PLUS=1
export EMA_DECAY=0.997
export MAX_WALLCLOCK_SECONDS=12000

FAIL=0

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
        FAIL=1
        return 1
    fi
    local VAL=$(grep -oP 'DIAGNOSTIC post_ema.*val_bpb:\K[0-9.]+' "$LOGFILE" 2>/dev/null)
    local QVAL=$(grep -oP 'roundtrip_exact.*val_bpb:\K[0-9.]+' "$LOGFILE" 2>/dev/null | tail -1)
    local STEPS=$(grep -oP 'step:\K[0-9]+(?=/)' "$LOGFILE" 2>/dev/null | tail -1)
    local SIZE=$(grep "Total submission size" "$LOGFILE" 2>/dev/null | grep -oP '[0-9]+(?= bytes)')
    local PARAMS=$(grep -oP 'model_params:\K[0-9]+' "$LOGFILE" 2>/dev/null)
    local PRUNE=$(grep "selective_prune:.*pruning\|no pruning" "$LOGFILE" 2>/dev/null | tail -1)
    local SLM_LOG=$(grep "slm_config:" "$LOGFILE" 2>/dev/null)
    if [ -n "$SIZE" ]; then local SIZE_MB=$(echo "scale=2; $SIZE / 1048576" | bc); else local SIZE_MB="—"; fi
    echo "  $NAME: prequant=$VAL quant=$QVAL steps=$STEPS art=${SIZE_MB}MB params=$PARAMS"
    echo "  $SLM_LOG"
    echo "  $PRUNE"
    return 0
}

# ═══════════════════════════════════════
# RUN 1: SLM 19L uniform:2 — 10 attn layers, 9 MLP-only
# Target: same param count as 13L (~39M), 2x steps
# ═══════════════════════════════════════
export TOTAL_LAYERS=19
export ATTN_PATTERN="uniform:2"
export NUM_UNIQUE_ATTN=0
export XSA_LAST_N=10
run_test "19L_uniform2_10attn"

# ═══════════════════════════════════════
# RUN 2: SLM 19L uniform:2 + shared attention (5 unique)
# Same layout but shared attn = smaller artifact
# ═══════════════════════════════════════
export TOTAL_LAYERS=19
export ATTN_PATTERN="uniform:2"
export NUM_UNIQUE_ATTN=5
export XSA_LAST_N=10
run_test "19L_uniform2_10attn_5shared"

echo ""
echo "$(date) ═══════════════════════════════════════"
if [ $FAIL -eq 0 ]; then
    echo "  ALL RUNS COMPLETE"
else
    echo "  SOME RUNS FAILED"
fi
echo "═══════════════════════════════════════"
