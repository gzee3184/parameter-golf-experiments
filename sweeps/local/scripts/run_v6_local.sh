#!/bin/bash
# v6: 11L + recurrence(3,4,5) + VE(9,10) + parallel(7+) + MuonEqR + WD0.09 + QK5.0 + skip gates + inductor fix
# Matches competitor architecture fully. 3-hour local test = H100 equivalent.
set -uo pipefail
source ${VENV:-./venv}/bin/activate

REPO_DIR="${PROJECT_ROOT:-.}/parameter-golf"
SCRIPT_DIR="${PROJECT_ROOT:-.}/h100_final"
RESULTS_DIR="${PROJECT_ROOT:-.}/local_tests/v6_long"
mkdir -p "$RESULTS_DIR"
cd "$REPO_DIR"
cp "$SCRIPT_DIR/train_gpt_v6.py" "$REPO_DIR/train_gpt.py"

# Architecture: 11L physical + recurrence [3,4,5] = 14 virtual, parallel from layer 7
export VOCAB_SIZE=4096 NUM_LAYERS=11 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=4
export MLP_ACTIVATION=leaky05
export DATA_PATH="./data/datasets/fineweb10B_sp4096"
export TOKENIZER_PATH="./data/tokenizers/fineweb_4096_bpe.model"
export MAX_WALLCLOCK_SECONDS=${1:-10800}
export SEED=1337 TRAIN_BATCH_TOKENS=786432 TRAIN_SEQ_LEN=2048 EVAL_SEQ_LEN=2048
export ITERATIONS=20000 WARMUP_STEPS=20 WARMDOWN_ITERS=2500
export VAL_LOSS_EVERY=4000 TRAIN_LOG_EVERY=100

# Optimizer: match competitor #1
export MUON_PLUS=1 MATRIX_LR=0.02 SCALAR_LR=0.02
export MUON_WD=0.090 ADAM_WD=0.02 EMBED_WD=0.090
export QK_GAIN_INIT=5.0 GRAD_CLIP_NORM=0.3 EMA_DECAY=0.997

# Features
export SIGMOID_SKIP=1 BYTE_SHUFFLE=1 COPRIME_STRIDE=1
export RECUR_LAYERS="3,4,5" RECUR_START_STEP=3000
export VE_ENABLED=1 VE_DIM=128 VE_LAYERS="9,10"
export PARALLEL_START_LAYER=7
export REPEAT_UNTIE_MLP=lora REPEAT_LORA_RANK=16
export DISABLE_LAYER0_ATTN=1

# Other
export LOGIT_SOFTCAP=30.0 XSA_LAST_N=11 LN_SCALE=1 TIE_EMBEDDINGS=1
export QUANT_BITS_MLP=6 QUANT_BITS_ATTN=6 QUANT_BITS_EMBED=6
export GPTQ_ENABLED=1 GPTQ_CALIB_SEQS=64 COMPRESSOR=brotli

LOGFILE="$RESULTS_DIR/run_$(date +%Y%m%d_%H%M%S).log"
echo "$(date) v6: 11L+recur(3,4,5)+VE(9,10)+parallel(7+)+MuonEqR+WD0.09+QK5.0+inductor_fix, wallclock=${MAX_WALLCLOCK_SECONDS}s"
echo "Log: $LOGFILE"

nice -n 10 torchrun --standalone --nproc_per_node=1 train_gpt.py 2>&1 | tee "$LOGFILE"
