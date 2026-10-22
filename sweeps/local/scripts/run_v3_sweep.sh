#!/bin/bash
# =============================================================================
# v3 Sweep — 3 configs × 1800s each on 2 GPUs
#   Test A: v3 + recurrence (5 shared pairs + RECUR_LAYERS=5)
#   Test B: v3 with 3 pairs (more capacity, still fits 16MB?)
#   Test C: v3 5 pairs + TTT enabled
# =============================================================================
set -uo pipefail
source ${VENV:-./venv}/bin/activate

REPO_DIR="${PROJECT_ROOT:-.}/parameter-golf"
SWEEP_DIR="${PROJECT_ROOT:-.}/local_tests/v3_sweep"
SCRIPT_DIR="${PROJECT_ROOT:-.}/h100_final"
mkdir -p "$SWEEP_DIR"

cd "$REPO_DIR"
cp "$SCRIPT_DIR/train_gpt.py" "$REPO_DIR/train_gpt.py"

# ── Common env ──
common_env() {
    export VOCAB_SIZE=4096
    export NUM_LAYERS=13
    export MODEL_DIM=512
    export NUM_HEADS=8
    export NUM_KV_HEADS=4
    export MLP_MULT=4
    export MLP_ACTIVATION=leaky05
    export DATA_PATH="./data/datasets/fineweb10B_sp4096"
    export TOKENIZER_PATH="./data/tokenizers/fineweb_4096_bpe.model"
    export MAX_WALLCLOCK_SECONDS=1800
    export SEED=1337
    export TRAIN_BATCH_TOKENS=786432
    export TRAIN_SEQ_LEN=2048
    export EVAL_SEQ_LEN=2048
    export ITERATIONS=20000
    export WARMUP_STEPS=20
    export WARMDOWN_ITERS=2500
    export VAL_LOSS_EVERY=4000
    export TRAIN_LOG_EVERY=100
    export MUON_PLUS=1
    export MATRIX_LR=0.02
    export SCALAR_LR=0.02
    export MUON_WD=0.12
    export ADAM_WD=0.02
    export EMBED_WD=0.12
    export QK_GAIN_INIT=4.0
    export GRAD_CLIP_NORM=0.3
    export EMA_DECAY=0.997
    export SIGMOID_SKIP=1
    export BYTE_SHUFFLE=1
    export COPRIME_STRIDE=1
    export LOGIT_SOFTCAP=30.0
    export XSA_LAST_N=13
    export LN_SCALE=1
    export TIE_EMBEDDINGS=1
    export QUANT_MODE=int6
    export QUANT_BITS_MLP=6
    export QUANT_BITS_ATTN=6
    export QUANT_BITS_EMBED=6
    export GPTQ_ENABLED=1
    export GPTQ_CALIB_SEQS=64
    export COMPRESSOR=brotli
    export TARGET_MB=15.9
    export TTT_ENABLED=0
    export SKIP_SLIDING=1
}

run_test() {
    local NAME=$1
    local GPU=$2
    local LOG="$SWEEP_DIR/${NAME}.log"
    echo "$(date) ▶ START $NAME on GPU $GPU → $LOG"
    CUDA_VISIBLE_DEVICES=$GPU torchrun --standalone --nproc_per_node=1 \
        --master_port=$((29500 + GPU)) train_gpt.py 2>&1 | tee "$LOG"
    echo "$(date) ■ DONE $NAME"
    echo "--- $NAME RESULTS ---"
    grep "DIAGNOSTIC post_ema" "$LOG" 2>/dev/null || echo "  (no pre-quant eval)"
    grep "selective_prune:" "$LOG" 2>/dev/null | tail -1
    grep "Total submission size" "$LOG" 2>/dev/null
    grep "final_int6_roundtrip_exact" "$LOG" 2>/dev/null
    echo ""
}

# ═══════════════════════════════════════
# All tests sequential on GPU 0
# ═══════════════════════════════════════

# Test A: 5 shared pairs + recurrence on layer 5
echo "$(date) === Test A: 5 pairs + recurrence ==="
(
    common_env
    export MLP_SHARE_PAIRS="0:12,1:11,2:10,3:9,4:8"
    export RECUR_LAYERS="5"
    export RECUR_START_STEP=200
    run_test "A_5pairs_recur" 0
)

# Test B: 3 shared pairs (more capacity) — no recurrence
echo ""
echo "$(date) === Test B: 3 pairs (more capacity) ==="
(
    common_env
    export MLP_SHARE_PAIRS="0:12,1:11,2:10"
    export RECUR_LAYERS=""
    export RECUR_START_STEP=200
    run_test "B_3pairs" 0
)

# Test C: 5 shared pairs + TTT
echo ""
echo "$(date) === Test C: 5 pairs + TTT ==="
(
    common_env
    export MLP_SHARE_PAIRS="0:12,1:11,2:10,3:9,4:8"
    export RECUR_LAYERS=""
    export RECUR_START_STEP=200
    export TTT_ENABLED=1
    export TTT_OPTIMIZER=adamw
    export TTT_LR=0.0001
    export TTT_EPOCHS=3
    export TTT_FREEZE_BLOCKS=2
    export TTT_BATCH_SEQS=32
    export TTT_GRAD_CLIP=1.0
    export TTT_WEIGHT_DECAY=0.01
    export TTT_QTTT=0
    export TTT_SPECTRAL_K=0
    export TTT_CHUNK_TOKENS=32768
    export SKIP_SLIDING=0
    run_test "C_5pairs_ttt" 0
)

# ═══════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════
echo ""
echo "$(date) ════════════════════════════════════════"
echo "  SWEEP SUMMARY"
echo "════════════════════════════════════════"
for f in "$SWEEP_DIR"/{A,B,C}_*.log; do
    NAME=$(basename "$f" .log)
    PREQUANT=$(grep -oP 'DIAGNOSTIC post_ema.*val_bpb:\K[0-9.]+' "$f" 2>/dev/null || echo "N/A")
    QUANT=$(grep -oP 'final_int6_roundtrip_exact.*val_bpb:\K[0-9.]+' "$f" 2>/dev/null || echo "N/A")
    TTT_SCORE=$(grep -oP 'legal_ttt_exact.*val_bpb:\K[0-9.]+' "$f" 2>/dev/null || echo "N/A")
    ARTSIZE=$(grep "Total submission size" "$f" 2>/dev/null | grep -oP '[0-9]+(?= bytes)' || echo "N/A")
    STEPS=$(grep "^step:" "$f" 2>/dev/null | grep "train_time" | tail -1 | grep -oP 'step:\K[0-9]+' || echo "N/A")
    PRUNE=$(grep "selective_prune:" "$f" 2>/dev/null | tail -1 | grep -oP 'pruned=\K[0-9.]+' || echo "0")%
    echo "  $NAME: steps=$STEPS pre=$PREQUANT quant=$QUANT ttt=$TTT_SCORE art=${ARTSIZE}B prune=$PRUNE"
done
echo "════════════════════════════════════════"
echo "$(date) === ALL DONE ==="
