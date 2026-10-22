#!/bin/bash
# =============================================================================
# RunPod 8×H100 — Parameter Golf Final Run (sp4096)
#
# Usage:
#   bash run.sh              # Full: setup + train + eval + TTT
#   bash run.sh train        # Skip setup (if already done)
#   bash run.sh nottt        # Train + quant only, no TTT (debug)
# =============================================================================
set -uo pipefail

MODE="${1:-full}"
REPO_DIR="/workspace/parameter-golf"
RESULTS_DIR="/workspace/results"
mkdir -p "$RESULTS_DIR"

# ═══════════════════════════════════════
# STEP 0: FIX HF CACHE (avoid filling container disk)
# ═══════════════════════════════════════
mkdir -p /workspace/.cache/huggingface
rm -rf /root/.cache/huggingface 2>/dev/null
ln -sf /workspace/.cache/huggingface /root/.cache/huggingface
# export HF_TOKEN=<your token>  (only needed for dataset download)

# ═══════════════════════════════════════
# STEP 1: SETUP (clone repo, download sp4096 data)
# ═══════════════════════════════════════
if [ "$MODE" != "train" ] && [ "$MODE" != "nottt" ] || [ ! -d "$REPO_DIR/data/datasets/fineweb10B_sp4096" ]; then
    echo "$(date) === SETUP ==="

    # Clone repo if not present
    if [ ! -f "$REPO_DIR/train_gpt.py" ]; then
        cd /workspace
        rm -rf parameter-golf 2>/dev/null
        git clone https://github.com/openai/parameter-golf.git
        cd parameter-golf
    else
        cd "$REPO_DIR"
    fi

    # Download sp4096 from kevclark's HF repo
    if [ ! -f "data/datasets/fineweb10B_sp4096/fineweb_val_000000.bin" ]; then
        echo "$(date) Downloading sp4096 data (143 shards, ~27GB)..."
        rm -f data/manifest.json
        MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf \
            python3 data/cached_challenge_fineweb.py --variant sp4096 --train-shards 143
    else
        echo "$(date) sp4096 data already present, skipping download"
    fi

    # Verify data
    python3 -c "
import glob
train = sorted(glob.glob('data/datasets/fineweb10B_sp4096/fineweb_train_*.bin'))
val = sorted(glob.glob('data/datasets/fineweb10B_sp4096/fineweb_val_*.bin'))
print(f'Train shards: {len(train)}  Val shards: {len(val)}')
assert len(train) >= 143, f'Expected 143, got {len(train)}'
print('Data: OK')
"

    # GPU check
    python3 -c "
import torch
n = torch.cuda.device_count()
for i in range(n):
    print(f'GPU {i}: {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_mem // 1024**3}GB)')
print(f'Total GPUs: {n}')
"

    # Clean HF cache to free disk
    rm -rf /workspace/.cache/huggingface 2>/dev/null

    echo "$(date) === SETUP DONE ==="
fi

cd "$REPO_DIR"

# Install our train_gpt.py (replaces default)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if [ -f "$SCRIPT_DIR/train_gpt.py" ]; then
    cp "$SCRIPT_DIR/train_gpt.py" "$REPO_DIR/train_gpt.py"
    echo "Installed custom train_gpt.py"
fi

N_GPUS=$(python3 -c "import torch; print(torch.cuda.device_count())")
echo "$(date) Using $N_GPUS GPUs"

# ═══════════════════════════════════════
# STEP 2: CONFIGURE (sp4096, 13L, Muon+, GPTQ, TTT)
# ═══════════════════════════════════════

# --- Model architecture: 13L, vocab4096, dim512 ---
export VOCAB_SIZE=4096
export NUM_LAYERS=13
export MODEL_DIM=512
export NUM_HEADS=8
export NUM_KV_HEADS=4
export MLP_MULT=4
export MLP_ACTIVATION=leaky05

# --- Data: sp4096 ---
export DATA_PATH="./data/datasets/fineweb10B_sp4096"
export TOKENIZER_PATH="./data/tokenizers/fineweb_4096_bpe.model"

# --- Training ---
export MAX_WALLCLOCK_SECONDS=600
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

# --- v2: Depth recurrence + Lane-based parallel residuals ---
export RECUR_LAYERS="4,5"
export RECUR_START_STEP=200
export PARALLEL_START_LAYER=7

# --- Quantization: int6/int5 + GPTQ ---
export QUANT_MODE=int6
export QUANT_BITS_MLP=6
export QUANT_BITS_ATTN=6
export QUANT_BITS_EMBED=6
export GPTQ_ENABLED=1
export GPTQ_CALIB_SEQS=64
export COMPRESSOR=brotli
export TARGET_MB=15.9

# --- TTT: AdamW conservative ---
if [ "$MODE" = "nottt" ]; then
    export TTT_ENABLED=0
    export SKIP_SLIDING=1
    echo "$(date) MODE: no-TTT (train + quant only)"
else
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
    echo "$(date) MODE: full (train + quant + sliding + TTT)"
fi

# ═══════════════════════════════════════
# STEP 3: LAUNCH
# ═══════════════════════════════════════
LOGFILE="$RESULTS_DIR/run_$(date +%Y%m%d_%H%M%S).log"
echo ""
echo "$(date) ══════════════════════════════════════════"
echo "  CONFIG: 13L vocab4096 leaky05 Muon+ int6/int5 GPTQ+TTT"
echo "  GPUs: $N_GPUS  Wallclock: 600s  TTT: ${TTT_ENABLED:-1}"
echo "  Log: $LOGFILE"
echo "══════════════════════════════════════════"
echo ""

torchrun --standalone --nproc_per_node=$N_GPUS train_gpt.py 2>&1 | tee "$LOGFILE"
RC=${PIPESTATUS[0]}

# ═══════════════════════════════════════
# STEP 4: RESULTS
# ═══════════════════════════════════════
echo ""
echo "$(date) ══════════════════════════════════════════"
if [ $RC -ne 0 ]; then
    echo "  TRAINING FAILED (exit code $RC)"
    echo "  Check log: $LOGFILE"
else
    echo "  RESULTS:"
    grep "DIAGNOSTIC post_ema" "$LOGFILE" 2>/dev/null || true
    grep "selective_prune:" "$LOGFILE" 2>/dev/null | tail -1 || true
    grep "Total submission size" "$LOGFILE" 2>/dev/null || true
    grep "final_int6_roundtrip_exact" "$LOGFILE" 2>/dev/null || true
    grep "final_int6_sliding_window_exact" "$LOGFILE" 2>/dev/null || true
    grep "legal_ttt_exact" "$LOGFILE" 2>/dev/null || true
    echo ""
    PREQUANT=$(grep -oP 'DIAGNOSTIC post_ema.*val_bpb:\K[0-9.]+' "$LOGFILE" 2>/dev/null || echo "N/A")
    QUANT=$(grep -oP 'final_int6_roundtrip_exact.*val_bpb:\K[0-9.]+' "$LOGFILE" 2>/dev/null || echo "N/A")
    SLIDING=$(grep -oP 'final_int6_sliding_window_exact.*val_bpb:\K[0-9.]+' "$LOGFILE" 2>/dev/null || echo "N/A")
    TTT_SCORE=$(grep -oP 'legal_ttt_exact.*val_bpb:\K[0-9.]+' "$LOGFILE" 2>/dev/null || echo "N/A")
    ARTSIZE=$(grep "Total submission size" "$LOGFILE" 2>/dev/null | grep -oP '[0-9]+(?= bytes)' || echo "N/A")
    echo "    Pre-quant BPB:  $PREQUANT"
    echo "    Quant BPB:      $QUANT"
    echo "    Sliding BPB:    $SLIDING"
    echo "    TTT BPB:        $TTT_SCORE"
    echo "    Artifact bytes: $ARTSIZE"
    echo "PREQUANT=$PREQUANT QUANT=$QUANT SLIDING=$SLIDING TTT=$TTT_SCORE ARTIFACT=$ARTSIZE" > "$RESULTS_DIR/summary.txt"
fi
echo "══════════════════════════════════════════"

# Copy artifacts
for f in final_model.int6.ptz final_model.pt; do
    [ -f "$f" ] && cp "$f" "$RESULTS_DIR/" && echo "Saved: $RESULTS_DIR/$f"
done
cp "$LOGFILE" "$RESULTS_DIR/latest.log" 2>/dev/null || true

echo ""
echo "$(date) === DONE ==="
echo "Results in: $RESULTS_DIR/"
