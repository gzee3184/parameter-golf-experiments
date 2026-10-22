#!/bin/bash
# v8.5 Architectural Sweep — gradient-analysis-driven improvements
# GPU 1 only (GPU 0 reserved for R5)
# 5 tests × 3hr each = ~15hrs sequential
# Baseline: R4 best (base=10K, softcap=20) + R5 fixes (skip[4], per-layer lane_merge, LoRA r=32)
# NOTE: WARMDOWN_ITERS=1500 (not 3500) because experimental code runs ~2220ms/step.
#   At 3500, warmdown_window (7.8s) > wall_time (7.2s) → LR decays from step 0. INVALID.
#   At 1500, warmdown starts ~step 2800, LR=100% through recurrence activation (step 2000).
#
# Tests:
#   A: Repeat Attention Q LoRA (rank 8) — give repeat pass different attention queries
#   B: Disable L0 MLP — gradient shows model pushing L0 MLP toward zero (0.247 highest block scalar grad)
#   C: Skip repeat attention — save compute by not computing attention on repeat passes (L5 attn dead)
#   D: Combo A+B — attn Q LoRA + disable L0 MLP (non-conflicting)
#   E: Parallel scalar gradient scaling (2.5x) — compensate 2.9x gradient deficit on Adam-trained scalars
#
# Note: A (attn LoRA) and C (skip repeat attn) are mutually exclusive approaches
set -uo pipefail
source ${VENV:-./venv}/bin/activate

REPO_DIR="${PROJECT_ROOT:-.}/parameter-golf"
SCRIPT_DIR="${PROJECT_ROOT:-.}/h100_final"
RESULTS_DIR="${PROJECT_ROOT:-.}/local_tests/v8.5_sweep"
mkdir -p "$RESULTS_DIR"
cd "$REPO_DIR"

cp "$SCRIPT_DIR/train_gpt_v8_experimental.py" "$REPO_DIR/train_gpt.py"

# Common config: matches R5 baseline (R4 best + fixes)
export CUDA_VISIBLE_DEVICES=1
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
export EMA_DECAY=0.9965 WARMDOWN_ITERS=1500 SWA_ENABLED=0
export RECUR_START_STEP=2000 QUANT_BITS_EMBED=8 REPEAT_PASS_SCALE=1
export GPTQ_ENABLED=1 GPTQ_CALIB_SEQS=64 COMPRESSOR=brotli
export QUANT_BITS_MLP=6 QUANT_BITS_ATTN=6
# R4 winning settings
export ROPE_BASE=10000
export LOGIT_SOFTCAP=20
export DATA_WARMDOWN_COUPLING=0
# R5 fixes (baked into all v8.5 tests)
export PER_LAYER_LANE_MERGE=1
export REPEAT_UNTIE_MLP=lora REPEAT_LORA_RANK=32
# Default: all v8.5 toggles off
export REPEAT_ATTN_LORA=0 REPEAT_ATTN_LORA_RANK=8
export DISABLE_LAYER0_MLP=0 SKIP_REPEAT_ATTN=0
export PARALLEL_SCALAR_GRAD_SCALE=0

# ── Test A: Repeat Attention Q LoRA (rank 8) ──
export REPEAT_ATTN_LORA=1 REPEAT_ATTN_LORA_RANK=8
export DISABLE_LAYER0_MLP=0 SKIP_REPEAT_ATTN=0
LOGFILE="$RESULTS_DIR/A_attn_qlora_$(date +%Y%m%d_%H%M%S).log"
echo "$(date) ── Test A: Repeat Attn Q LoRA (rank=8) on GPU 1 ──"
echo "Log: $LOGFILE"
nice -n 10 torchrun --standalone --nproc_per_node=1 train_gpt.py 2>&1 | tee "$LOGFILE"
echo "$(date) Test A complete"
echo ""

# ── Test B: Disable L0 MLP ──
export REPEAT_ATTN_LORA=0 REPEAT_ATTN_LORA_RANK=8
export DISABLE_LAYER0_MLP=1 SKIP_REPEAT_ATTN=0
LOGFILE="$RESULTS_DIR/B_disable_l0_mlp_$(date +%Y%m%d_%H%M%S).log"
echo "$(date) ── Test B: Disable L0 MLP on GPU 1 ──"
echo "Log: $LOGFILE"
nice -n 10 torchrun --standalone --nproc_per_node=1 train_gpt.py 2>&1 | tee "$LOGFILE"
echo "$(date) Test B complete"
echo ""

# ── Test C: Skip Repeat Attention ──
export REPEAT_ATTN_LORA=0 REPEAT_ATTN_LORA_RANK=8
export DISABLE_LAYER0_MLP=0 SKIP_REPEAT_ATTN=1
LOGFILE="$RESULTS_DIR/C_skip_repeat_attn_$(date +%Y%m%d_%H%M%S).log"
echo "$(date) ── Test C: Skip Repeat Attention on GPU 1 ──"
echo "Log: $LOGFILE"
nice -n 10 torchrun --standalone --nproc_per_node=1 train_gpt.py 2>&1 | tee "$LOGFILE"
echo "$(date) Test C complete"
echo ""

# ── Test D: Combo (Attn Q LoRA + Disable L0 MLP) ──
export REPEAT_ATTN_LORA=1 REPEAT_ATTN_LORA_RANK=8
export DISABLE_LAYER0_MLP=1 SKIP_REPEAT_ATTN=0
export PARALLEL_SCALAR_GRAD_SCALE=0
LOGFILE="$RESULTS_DIR/D_combo_qlora_nol0mlp_$(date +%Y%m%d_%H%M%S).log"
echo "$(date) ── Test D: Combo (Attn Q LoRA + Disable L0 MLP) on GPU 1 ──"
echo "Log: $LOGFILE"
nice -n 10 torchrun --standalone --nproc_per_node=1 train_gpt.py 2>&1 | tee "$LOGFILE"
echo "$(date) Test D complete"
echo ""

# ── Test E: Parallel Scalar Gradient Scaling (2.5x) ──
export REPEAT_ATTN_LORA=0 REPEAT_ATTN_LORA_RANK=8
export DISABLE_LAYER0_MLP=0 SKIP_REPEAT_ATTN=0
export PARALLEL_SCALAR_GRAD_SCALE=2.5
LOGFILE="$RESULTS_DIR/E_parallel_grad_scale_$(date +%Y%m%d_%H%M%S).log"
echo "$(date) ── Test E: Parallel Scalar Grad Scale (2.5x) on GPU 1 ──"
echo "Log: $LOGFILE"
nice -n 10 torchrun --standalone --nproc_per_node=1 train_gpt.py 2>&1 | tee "$LOGFILE"
echo "$(date) Test E complete"
echo ""

echo "$(date) ═══ v8.5 SWEEP COMPLETE ═══"
echo "Results in: $RESULTS_DIR/"
