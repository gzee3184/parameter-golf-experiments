#!/bin/bash
# v9 Compression Quality Sweep — test Late QAT and SWA
# GPU 0 only (GPU 1 has v8.5 E-fix running)
# 2 tests × 3hr each = ~6hrs sequential
#
# Both tests use our best config (R4 combo: softcap=20, rope_base=10K)
# + R5 fixes (per_layer_lane_merge, LoRA r=32, skip[4] fix)
#
# Test F: Late QAT (inject quant noise last 10% of training)
#   - QAT makes weights robust to int6 quantization → better compression
#   - From #1423: competitor uses scale=0.15 at 90%. We use conservative 0.10
# Test G: Late QAT + SWA (averaged weights from warmdown)
#   - SWA averages last ~14 snapshots → smoother weights → better brotli compression
#   - Combined with QAT for maximum compression improvement
set -uo pipefail
source ${VENV:-./venv}/bin/activate

REPO_DIR="${PROJECT_ROOT:-.}/parameter-golf"
SCRIPT_DIR="${PROJECT_ROOT:-.}/h100_final"
RESULTS_DIR="${PROJECT_ROOT:-.}/local_tests/v9_compression"
mkdir -p "$RESULTS_DIR"
cd "$REPO_DIR"

cp "$SCRIPT_DIR/train_gpt_v8_experimental.py" "$REPO_DIR/train_gpt.py"

# Common config: our best (R4 combo + R5 fixes)
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
export EMA_DECAY=0.9965 WARMDOWN_ITERS=1500
export RECUR_START_STEP=2000 QUANT_BITS_EMBED=8 REPEAT_PASS_SCALE=1
export GPTQ_ENABLED=1 GPTQ_CALIB_SEQS=64 COMPRESSOR=brotli
export QUANT_BITS_MLP=6 QUANT_BITS_ATTN=6
# R4 winning settings
export ROPE_BASE=10000
export LOGIT_SOFTCAP=20
export DATA_WARMDOWN_COUPLING=0
# R5 fixes
export PER_LAYER_LANE_MERGE=1
export REPEAT_UNTIE_MLP=lora REPEAT_LORA_RANK=32
# v8.5 toggles off by default
export REPEAT_ATTN_LORA=0 DISABLE_LAYER0_MLP=0 SKIP_REPEAT_ATTN=0
export PARALLEL_SCALAR_GRAD_SCALE=0
# New v9 defaults
export SDCLIP=0
export USE_SWA_WEIGHTS=0

# ── Test F: Late QAT only ──
export LATE_QAT=1 QAT_START_FRAC=0.90 QAT_NOISE_SCALE=0.10
export SWA_ENABLED=0 USE_SWA_WEIGHTS=0
LOGFILE="$RESULTS_DIR/F_late_qat_$(date +%Y%m%d_%H%M%S).log"
echo "$(date) ── Test F: Late QAT (scale=0.10, start=90%) on GPU 0 ──"
echo "Log: $LOGFILE"
nice -n 10 torchrun --standalone --nproc_per_node=1 train_gpt.py 2>&1 | tee "$LOGFILE"
echo "$(date) Test F complete"
echo ""

# ── Test G: Late QAT + SWA ──
export LATE_QAT=1 QAT_START_FRAC=0.90 QAT_NOISE_SCALE=0.10
export SWA_ENABLED=1 SWA_EVERY=50 USE_SWA_WEIGHTS=1
LOGFILE="$RESULTS_DIR/G_qat_swa_$(date +%Y%m%d_%H%M%S).log"
echo "$(date) ── Test G: Late QAT + SWA on GPU 0 ──"
echo "Log: $LOGFILE"
nice -n 10 torchrun --standalone --nproc_per_node=1 train_gpt.py 2>&1 | tee "$LOGFILE"
echo "$(date) Test G complete"
echo ""

echo "=== v9 Compression Sweep Complete ==="
echo "Compare Test F (QAT-only) vs Test G (QAT+SWA):"
echo "  Look at: post_ema/post_swa BPB, quant BPB, compressed model size, prune %"
echo "  grep 'DIAGNOSTIC\|prune\|budget\|sliding' $RESULTS_DIR/*.log"
