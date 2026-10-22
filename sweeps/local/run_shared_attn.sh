#!/bin/bash
# Shared attention depth test — more attention layers without pruning cliff
# Run: bash run_shared_attn.sh <gpu> <name> <total_layers> <num_unique_attn>

cd ${PROJECT_ROOT:-.}/parameter-golf
source ${VENV:-./venv}/bin/activate

SCRIPT="${PROJECT_ROOT:-.}/local_tests/train_gpt_slm.py"
RESULTS="${PROJECT_ROOT:-.}/local_tests"

GPU=${1}
NAME=${2}
TOTAL=${3}
UNIQUE=${4}
PORT=$((29600 + GPU))
export CUDA_VISIBLE_DEVICES=$GPU

# Same base config as 13L baseline
export DATA_PATH="./data/datasets/fineweb10B_sp4096"
export TOKENIZER_PATH="./data/tokenizers/fineweb_4096_bpe.model"
export VOCAB_SIZE=4096 MODEL_DIM=512 MLP_MULT=4 NUM_HEADS=8 NUM_KV_HEADS=4
export SIGMOID_SKIP=1 SMEAR_ENABLED=0 BYTE_SHUFFLE=1 COPRIME_STRIDE=1
export COMPRESSOR=brotli MUON_WD=0.085 ADAM_WD=0.02 EMBED_WD=0.085
export MATRIX_LR=0.02 SCALAR_LR=0.02 QK_GAIN_INIT=4.0 LOGIT_SOFTCAP=30.0
export MLP_ACTIVATION=leaky05 BIGRAM_VOCAB_SIZE=0 VE_ENABLED=0
export QAT_ENABLED=0 LATE_QAT_THRESHOLD=0 GATED_ATTENTION=0 VALUE_RESIDUAL=0
export MTP_NUM_HEADS=0 QUANT_BITS_MLP=6 QUANT_BITS_ATTN=5
export GPTQ_ENABLED=1 GPTQ_CALIB_SEQS=8 TARGET_MB=15.9
export SEED=1337 EVAL_SEQ_LEN=2048 SKIP_SLIDING=1
export TTT_ENABLED=0 NGRAM_CACHE=0 MUON_PLUS=1 EMA_DECAY=0.997
export MAX_WALLCLOCK_SECONDS=12000

# All layers have attention (no SLM interleaving)
export TOTAL_LAYERS=$TOTAL
export ATTN_PATTERN=""
export NUM_UNIQUE_ATTN=$UNIQUE
export XSA_LAST_N=$TOTAL

echo "$(date) === $NAME on GPU$GPU ==="
echo "  total_layers=$TOTAL num_unique_attn=$UNIQUE"
nice -n 10 torchrun --standalone --nproc_per_node=1 --master_port=$PORT "$SCRIPT" \
    > "$RESULTS/shared_long_${NAME}.log" 2>&1
RC=$?
if [ $RC -ne 0 ]; then
    echo "  FAILED (exit code $RC)"
    tail -20 "$RESULTS/shared_long_${NAME}.log"
else
    VAL=$(grep -oP 'DIAGNOSTIC post_ema.*val_bpb:\K[0-9.]+' "$RESULTS/shared_long_${NAME}.log")
    QVAL=$(grep -oP 'roundtrip_exact.*val_bpb:\K[0-9.]+' "$RESULTS/shared_long_${NAME}.log" | tail -1)
    STEPS=$(grep -oP 'step:\K[0-9]+(?=/)' "$RESULTS/shared_long_${NAME}.log" | tail -1)
    SAVG=$(grep -oP 'step_avg:\K[0-9.]+' "$RESULTS/shared_long_${NAME}.log" | tail -1)
    PRUNE=$(grep "selective_prune:.*pruning\|no pruning" "$RESULTS/shared_long_${NAME}.log" | tail -1)
    PARAMS=$(grep -oP 'model_params:\K[0-9]+' "$RESULTS/shared_long_${NAME}.log")
    echo "  $NAME: prequant=$VAL quant=$QVAL steps=$STEPS ms/step=$SAVG params=$PARAMS"
    echo "  $PRUNE"
fi
echo "$(date) === $NAME DONE ==="
