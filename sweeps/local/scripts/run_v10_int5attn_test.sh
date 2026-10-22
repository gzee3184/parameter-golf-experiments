#!/bin/bash
# v10 Int5 Attention Test — verify int5 attn saves ~0.68MB with zero BPB cost
# GPU 0: int5 attn (QUANT_BITS_ATTN=5)
# GPU 1: int6 attn control (QUANT_BITS_ATTN=6)
# Both use Ffix config (our best: LN_SCALE=0 + softcap20 + WARMDOWN_ITERS=3500)
# Uses CORRECT artifact limit: 16,000,000 bytes (16MB decimal)
set -uo pipefail
source ${VENV:-./venv}/bin/activate

REPO_DIR="${PROJECT_ROOT:-.}/parameter-golf"
SCRIPT_DIR="${PROJECT_ROOT:-.}/h100_final"
RESULTS_DIR="${PROJECT_ROOT:-.}/local_tests/v10_int5attn"
mkdir -p "$RESULTS_DIR"
cd "$REPO_DIR"

cp "$SCRIPT_DIR/train_gpt_v8_experimental.py" "$REPO_DIR/train_gpt.py"

# Common Ffix config (our current best)
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
export XSA_LAST_N=11 TIE_EMBEDDINGS=1
export ATTN_RES=0 RECUR_SKIP_PHYSICAL=0
export EMA_DECAY=0.9965
export RECUR_START_STEP=2000 REPEAT_PASS_SCALE=1
export GPTQ_ENABLED=1 GPTQ_CALIB_SEQS=64 COMPRESSOR=brotli
export QUANT_BITS_MLP=6
# Ffix winning settings
export ROPE_BASE=10000 LOGIT_SOFTCAP=20 LN_SCALE=0 WARMDOWN_ITERS=3500
export DATA_WARMDOWN_COUPLING=0
# LoRA r=16 (not r=32, saves ~350KB compressed)
export PER_LAYER_LANE_MERGE=0
export REPEAT_UNTIE_MLP=lora REPEAT_LORA_RANK=16
# v8.5 toggles off
export REPEAT_ATTN_LORA=0 DISABLE_LAYER0_MLP=0 SKIP_REPEAT_ATTN=0
export PARALLEL_SCALAR_GRAD_SCALE=0
# New features off (testing quant, not training changes)
export LATE_QAT=0 SWA_ENABLED=0 SDCLIP=0

# ── GPU 0: int5 attention ──
export CUDA_VISIBLE_DEVICES=0
export QUANT_BITS_ATTN=5 QUANT_BITS_EMBED=8
LOGFILE="$RESULTS_DIR/int5_attn_$(date +%Y%m%d_%H%M%S).log"
echo "$(date) ── int5 attn test on GPU 0 ──"
echo "Log: $LOGFILE"
nice -n 10 torchrun --standalone --nproc_per_node=1 train_gpt.py 2>&1 | tee "$LOGFILE" &
PID0=$!

# ── GPU 1: int6 attention (control) ──
export CUDA_VISIBLE_DEVICES=1
export QUANT_BITS_ATTN=6 QUANT_BITS_EMBED=8
LOGFILE="$RESULTS_DIR/int6_attn_control_$(date +%Y%m%d_%H%M%S).log"
echo "$(date) ── int6 attn control on GPU 1 ──"
echo "Log: $LOGFILE"
nice -n 10 torchrun --standalone --nproc_per_node=1 train_gpt.py 2>&1 | tee "$LOGFILE" &
PID1=$!

echo "Waiting for both to complete... GPU0=$PID0 GPU1=$PID1"
wait $PID0 $PID1
echo ""
echo "=== Both complete ==="
echo "Compare:"
echo "  grep 'DIAGNOSTIC\|prune\|budget\|submission\|roundtrip_exact\|sliding' $RESULTS_DIR/*.log"
