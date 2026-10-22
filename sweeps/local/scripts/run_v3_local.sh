#!/bin/bash
# =============================================================================
# v3 Local Test — RTX PRO 6000 (simulates 8×H100 with longer wallclock)
#
# v3 changes:
#   - MLP sharing (5 pairs): 39.6M → 29.1M params (26.5% reduction)
#   - Static-graph recurrence (gate-based, no recompilation)
#   - Parallel lanes removed
#   - cache_size_limit = 64
# =============================================================================
set -uo pipefail

source ${VENV:-./venv}/bin/activate

REPO_DIR="${PROJECT_ROOT:-.}/parameter-golf"
SCRIPT_DIR="${PROJECT_ROOT:-.}/h100_final"
RESULTS_DIR="${PROJECT_ROOT:-.}/local_tests/v3_mlp_share"
mkdir -p "$RESULTS_DIR"

cd "$REPO_DIR"

# Install our train_gpt.py
cp "$SCRIPT_DIR/train_gpt.py" "$REPO_DIR/train_gpt.py"
echo "Installed v3 train_gpt.py"

# --- Model architecture: 13L, vocab4096, MLP4x with sharing ---
export VOCAB_SIZE=4096
export NUM_LAYERS=13
export MODEL_DIM=512
export NUM_HEADS=8
export NUM_KV_HEADS=4
export MLP_MULT=4
export MLP_ACTIVATION=leaky05

# --- v3: MLP sharing (mirror pairs, 13→8 unique MLPs) ---
export MLP_SHARE_PAIRS="0:12,1:11,2:10,3:9,4:8"

# --- v3: Depth recurrence (static graph, activates at step 70% through) ---
# Disabled for first test — test MLP sharing alone first
export RECUR_LAYERS=""
export RECUR_START_STEP=200

# --- Data: sp4096 ---
export DATA_PATH="./data/datasets/fineweb10B_sp4096"
export TOKENIZER_PATH="./data/tokenizers/fineweb_4096_bpe.model"

# --- Training: 12000s local (simulates ~6000 H100 steps) ---
export MAX_WALLCLOCK_SECONDS=18000
export SEED=1337
export TRAIN_BATCH_TOKENS=786432
export TRAIN_SEQ_LEN=2048
export EVAL_SEQ_LEN=2048
export ITERATIONS=20000
export WARMUP_STEPS=20
export WARMDOWN_ITERS=2500
export VAL_LOSS_EVERY=4000
export TRAIN_LOG_EVERY=500

# --- Optimizer: Muon+ ---
export MUON_PLUS=1
export MATRIX_LR=0.02
export SCALAR_LR=0.02
export MUON_WD=0.12
export ADAM_WD=0.02
export EMBED_WD=0.12
export QK_GAIN_INIT=4.0
export GRAD_CLIP_NORM=0.3
export EMA_DECAY=0.997

# --- Architecture features ---
export SIGMOID_SKIP=1
export BYTE_SHUFFLE=1
export COPRIME_STRIDE=1
export LOGIT_SOFTCAP=30.0
export XSA_LAST_N=13
export LN_SCALE=1
export TIE_EMBEDDINGS=1

# --- Quantization: int6 + GPTQ ---
export QUANT_MODE=int6
export QUANT_BITS_MLP=6
export QUANT_BITS_ATTN=6
export QUANT_BITS_EMBED=6
export GPTQ_ENABLED=1
export GPTQ_CALIB_SEQS=64
export COMPRESSOR=brotli
export TARGET_MB=15.9

# --- TTT: off for local test (too slow locally) ---
export TTT_ENABLED=0
export SKIP_SLIDING=1

LOGFILE="$RESULTS_DIR/run_$(date +%Y%m%d_%H%M%S).log"
echo ""
echo "$(date) ══════════════════════════════════════════"
echo "  v3: 13L MLP4x + MLP_SHARE(5 pairs) = 29.1M params"
echo "  Wallclock: ${MAX_WALLCLOCK_SECONDS}s  TTT: off"
echo "  Log: $LOGFILE"
echo "══════════════════════════════════════════"
echo ""

torchrun --standalone --nproc_per_node=1 train_gpt.py 2>&1 | tee "$LOGFILE"
RC=${PIPESTATUS[0]}

echo ""
echo "$(date) ══════════════════════════════════════════"
if [ $RC -ne 0 ]; then
    echo "  TRAINING FAILED (exit code $RC)"
else
    echo "  RESULTS:"
    grep "DIAGNOSTIC post_ema" "$LOGFILE" 2>/dev/null || true
    grep "selective_prune:" "$LOGFILE" 2>/dev/null | tail -1 || true
    grep "Total submission size" "$LOGFILE" 2>/dev/null || true
    grep "final_int6_roundtrip_exact" "$LOGFILE" 2>/dev/null || true
    PREQUANT=$(grep -oP 'DIAGNOSTIC post_ema.*val_bpb:\K[0-9.]+' "$LOGFILE" 2>/dev/null || echo "N/A")
    QUANT=$(grep -oP 'final_int6_roundtrip_exact.*val_bpb:\K[0-9.]+' "$LOGFILE" 2>/dev/null || echo "N/A")
    ARTSIZE=$(grep "Total submission size" "$LOGFILE" 2>/dev/null | grep -oP '[0-9]+(?= bytes)' || echo "N/A")
    echo "    Pre-quant BPB:  $PREQUANT"
    echo "    Quant BPB:      $QUANT"
    echo "    Artifact bytes: $ARTSIZE"
fi
echo "══════════════════════════════════════════"

for f in final_model.int6.ptz final_model.pt; do
    [ -f "$f" ] && cp "$f" "$RESULTS_DIR/" && echo "Saved: $RESULTS_DIR/$f"
done
cp "$LOGFILE" "$RESULTS_DIR/latest.log" 2>/dev/null || true

echo "$(date) === DONE ==="
