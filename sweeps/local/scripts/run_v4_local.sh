#!/bin/bash
# v4: Max params, no sharing, no TTT, no recurrence. Pruning fills 16MiB budget.
set -uo pipefail
source ${VENV:-./venv}/bin/activate

REPO_DIR="${PROJECT_ROOT:-.}/parameter-golf"
SCRIPT_DIR="${PROJECT_ROOT:-.}/h100_final"
RESULTS_DIR="${PROJECT_ROOT:-.}/local_tests/v4_maxparams"
mkdir -p "$RESULTS_DIR"
cd "$REPO_DIR"
cp "$SCRIPT_DIR/train_gpt.py" "$REPO_DIR/train_gpt.py"

export VOCAB_SIZE=4096 NUM_LAYERS=13 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=4
export MLP_ACTIVATION=leaky05
export DATA_PATH="./data/datasets/fineweb10B_sp4096"
export TOKENIZER_PATH="./data/tokenizers/fineweb_4096_bpe.model"
export MAX_WALLCLOCK_SECONDS=${1:-1800}
export SEED=1337 TRAIN_BATCH_TOKENS=786432 TRAIN_SEQ_LEN=2048 EVAL_SEQ_LEN=2048
export ITERATIONS=20000 WARMUP_STEPS=20 WARMDOWN_ITERS=2500
export VAL_LOSS_EVERY=4000 TRAIN_LOG_EVERY=100
export MUON_PLUS=1 MATRIX_LR=0.02 SCALAR_LR=0.02
export MUON_WD=0.12 ADAM_WD=0.02 EMBED_WD=0.12
export QK_GAIN_INIT=4.0 GRAD_CLIP_NORM=0.3 EMA_DECAY=0.997
export SIGMOID_SKIP=1 BYTE_SHUFFLE=1 COPRIME_STRIDE=1
export LOGIT_SOFTCAP=30.0 XSA_LAST_N=13 LN_SCALE=1 TIE_EMBEDDINGS=1
export QUANT_BITS_MLP=6 QUANT_BITS_ATTN=6 QUANT_BITS_EMBED=6
export GPTQ_ENABLED=1 GPTQ_CALIB_SEQS=64 COMPRESSOR=brotli

LOGFILE="$RESULTS_DIR/run_$(date +%Y%m%d_%H%M%S).log"
echo "$(date) v4: 13L MLP4x leaky05 39.6M params, wallclock=${MAX_WALLCLOCK_SECONDS}s"
echo "Log: $LOGFILE"

torchrun --standalone --nproc_per_node=1 train_gpt.py 2>&1 | tee "$LOGFILE"
RC=${PIPESTATUS[0]}

echo ""
if [ $RC -ne 0 ]; then
    echo "FAILED (exit $RC)"
else
    grep "DIAGNOSTIC post_ema" "$LOGFILE" 2>/dev/null
    grep "budget:" "$LOGFILE" 2>/dev/null
    grep "selective_prune:" "$LOGFILE" 2>/dev/null | tail -1
    grep "Total submission size" "$LOGFILE" 2>/dev/null
    grep "final_int6_roundtrip_exact" "$LOGFILE" 2>/dev/null
fi

for f in final_model.int6.ptz final_model.pt; do
    [ -f "$f" ] && cp "$f" "$RESULTS_DIR/"
done
cp "$LOGFILE" "$RESULTS_DIR/latest.log" 2>/dev/null
echo "$(date) DONE"
