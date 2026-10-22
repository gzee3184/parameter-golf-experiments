#!/bin/bash
# Vocab4096 baseline smoke test — quick 60s run to validate everything works
# Based on kevclark's 1.098 BPB entry + our mish2 advantage

cd ${PROJECT_ROOT:-.}/parameter-golf
source ${VENV:-./venv}/bin/activate

SCRIPT="records/track_10min_16mb/2026-03-27_ActivationSearch/train_gpt.py"
RESULTS="${PROJECT_ROOT:-.}/local_tests"

GPU=${1:-0}
PORT=$((29500 + GPU))
export CUDA_VISIBLE_DEVICES=$GPU

# === Vocab4096 baseline config ===
# Data
export DATA_PATH="./data/datasets/fineweb10B_sp4096"
export TOKENIZER_PATH="./data/tokenizers/fineweb_4096_bpe.model"

# Architecture
export VOCAB_SIZE=4096
export NUM_LAYERS=11
export MODEL_DIM=512
export MLP_MULT=4
export NUM_HEADS=8
export NUM_KV_HEADS=4

# New features from kevclark entry
export SIGMOID_SKIP=1         # sigmoid-gated UNet skip connections
export SMEAR_ENABLED=0        # no SmearGate (redundant with vocab4096)
export BYTE_SHUFFLE=1         # byte-shuffle before brotli
export COPRIME_STRIDE=1       # coprime-stride data loader
export COMPRESSOR=brotli      # brotli > lzma for compression with byte-shuffle

# Weight decay as compression lever (from kevclark R²≈0.99 finding)
export MUON_WD=0.085
export ADAM_WD=0.02
export EMBED_WD=0.085

# Learning rates (lower for larger effective model)
export MATRIX_LR=0.02
export SCALAR_LR=0.02

# Attention config
export QK_GAIN_INIT=4.0
export XSA_LAST_N=11          # XSA in all layers
export LOGIT_SOFTCAP=30.0

# Our mish2 advantage (proven better than LeakyReLU(0.5)²)
export MLP_ACTIVATION=mish2

# Simplifications (remove features that don't help at vocab4096)
export BIGRAM_VOCAB_SIZE=0    # no BigramHash
export VE_ENABLED=0           # no value embeddings (test later)
export QAT_ENABLED=0          # no QAT
export LATE_QAT_THRESHOLD=0   # disable late QAT trigger
export GATED_ATTENTION=0      # no gated attention
export VALUE_RESIDUAL=0       # no value residuals
export MTP_NUM_HEADS=0        # no MTP

# Quantization (start with default, optimize from sweep data)
export QUANT_BITS_MLP=8
export QUANT_BITS_ATTN=6
export GPTQ_ENABLED=1
export GPTQ_CALIB_SEQS=8
export TARGET_MB=15.9

# Training config
export SEED=1337
export EVAL_SEQ_LEN=2048
export MAX_WALLCLOCK_SECONDS=60
export SKIP_SLIDING=1         # fast screening mode
export TTT_ENABLED=0
export NGRAM_CACHE=0
export EMA_DECAY=0.997        # will optimize later based on step count

# Muon optimizer
export MUON_PLUS=1

LABEL="v4096_smoke"
echo "$(date) === $LABEL === (GPU $GPU)"
nice -n 10 torchrun --standalone --nproc_per_node=1 --master_port=$PORT "$SCRIPT" \
    > "$RESULTS/${LABEL}.log" 2>&1

VAL=$(grep -oP 'DIAGNOSTIC post_ema.*val_bpb:\K[0-9.]+' "$RESULTS/${LABEL}.log" 2>/dev/null)
QVAL=$(grep -oP 'roundtrip_exact.*val_bpb:\K[0-9.]+' "$RESULTS/${LABEL}.log" 2>/dev/null | tail -1)
STEP=$(grep -oP 'step_avg:\K[0-9.]+' "$RESULTS/${LABEL}.log" 2>/dev/null | tail -1)
STEPS=$(grep -oP 'step:\K[0-9]+(?=/)' "$RESULTS/${LABEL}.log" 2>/dev/null | tail -1)
SIZE=$(grep "Total submission size" "$RESULTS/${LABEL}.log" 2>/dev/null | grep -oP '[0-9]+(?= bytes)')
PARAMS=$(grep -oP 'model_params:\K[0-9]+' "$RESULTS/${LABEL}.log" 2>/dev/null)
if [ -n "$SIZE" ]; then SIZE_MB=$(echo "scale=2; $SIZE / 1048576" | bc); else SIZE_MB="—"; fi
echo "  $LABEL: prequant=$VAL quant=$QVAL steps=$STEPS step_ms=$STEP art=${SIZE_MB}MB params=$PARAMS"
echo "$(date) === DONE ==="
