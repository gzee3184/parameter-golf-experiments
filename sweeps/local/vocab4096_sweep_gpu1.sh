#!/bin/bash
# Vocab4096 hyperparameter sweep — GPU1
# Tests batch size, quant combos, layers×WD interactions, mish2 vs leaky

cd ${PROJECT_ROOT:-.}/parameter-golf
source ${VENV:-./venv}/bin/activate

SCRIPT="records/track_10min_16mb/2026-03-27_ActivationSearch/train_gpt.py"
RESULTS="${PROJECT_ROOT:-.}/local_tests"

GPU=${1:-1}
PORT=$((29500 + GPU))
export CUDA_VISIBLE_DEVICES=$GPU

# === Vocab4096 base config ===
export DATA_PATH="./data/datasets/fineweb10B_sp4096"
export TOKENIZER_PATH="./data/tokenizers/fineweb_4096_bpe.model"
export VOCAB_SIZE=4096
export NUM_LAYERS=11
export MODEL_DIM=512
export MLP_MULT=4
export NUM_HEADS=8
export NUM_KV_HEADS=4
export SIGMOID_SKIP=1
export SMEAR_ENABLED=0
export BYTE_SHUFFLE=1
export COPRIME_STRIDE=1
export COMPRESSOR=brotli
export MUON_WD=0.085
export ADAM_WD=0.02
export EMBED_WD=0.085
export MATRIX_LR=0.02
export SCALAR_LR=0.02
export QK_GAIN_INIT=4.0
export XSA_LAST_N=11
export LOGIT_SOFTCAP=30.0
export MLP_ACTIVATION=mish2
export BIGRAM_VOCAB_SIZE=0
export VE_ENABLED=0
export QAT_ENABLED=0
export LATE_QAT_THRESHOLD=0
export GATED_ATTENTION=0
export VALUE_RESIDUAL=0
export MTP_NUM_HEADS=0
export QUANT_BITS_MLP=8
export QUANT_BITS_ATTN=6
export GPTQ_ENABLED=1
export GPTQ_CALIB_SEQS=8
export TARGET_MB=15.9
export SEED=1337
export EVAL_SEQ_LEN=2048
export MAX_WALLCLOCK_SECONDS=60
export SKIP_SLIDING=1
export TTT_ENABLED=0
export NGRAM_CACHE=0
export EMA_DECAY=0.997
export MUON_PLUS=1

run_one() {
    local LABEL="$1"
    echo "$(date) === $LABEL ==="
    nice -n 10 torchrun --standalone --nproc_per_node=1 --master_port=$PORT "$SCRIPT" \
        > "$RESULTS/${LABEL}.log" 2>&1 || echo "$LABEL FAILED"
    local VAL=$(grep -oP 'DIAGNOSTIC post_ema.*val_bpb:\K[0-9.]+' "$RESULTS/${LABEL}.log" 2>/dev/null)
    local QVAL=$(grep -oP 'roundtrip_exact.*val_bpb:\K[0-9.]+' "$RESULTS/${LABEL}.log" 2>/dev/null | tail -1)
    local STEP=$(grep -oP 'step_avg:\K[0-9.]+' "$RESULTS/${LABEL}.log" 2>/dev/null | tail -1)
    local STEPS=$(grep -oP 'step:\K[0-9]+(?=/)' "$RESULTS/${LABEL}.log" 2>/dev/null | tail -1)
    local SIZE=$(grep "Total submission size" "$RESULTS/${LABEL}.log" 2>/dev/null | grep -oP '[0-9]+(?= bytes)')
    local PARAMS=$(grep -oP 'model_params:\K[0-9]+' "$RESULTS/${LABEL}.log" 2>/dev/null)
    if [ -n "$SIZE" ]; then local SIZE_MB=$(echo "scale=2; $SIZE / 1048576" | bc); else local SIZE_MB="—"; fi
    echo "  $LABEL: prequant=$VAL quant=$QVAL steps=$STEPS step_ms=$STEP art=${SIZE_MB}MB params=$PARAMS"
}

echo "$(date) === VOCAB4096 SWEEP GPU$GPU ==="

# --- Batch size sweep ---
for batch in 393216 524288 786432 1048576; do
    export TRAIN_BATCH_TOKENS=$batch
    export NUM_LAYERS=11
    export MLP_MULT=4
    run_one "v4_batch${batch}"
done
export TRAIN_BATCH_TOKENS=786432  # reset

# --- Mish2 vs LeakyReLU(0.5)² comparison ---
export MLP_ACTIVATION=leaky05
run_one "v4_leaky05"
export MLP_ACTIVATION=mish2
run_one "v4_mish2_control"

# --- Quant sweep (for artifact size optimization) ---
for combo in "8_6" "8_5" "6_6" "6_5" "8_8"; do
    IFS='_' read -r BMLP BATTN <<< "$combo"
    export QUANT_BITS_MLP=$BMLP
    export QUANT_BITS_ATTN=$BATTN
    run_one "v4_q${BMLP}${BATTN}"
done
export QUANT_BITS_MLP=8
export QUANT_BITS_ATTN=6

# --- 12L with high WD (filling budget with more layers) ---
export NUM_LAYERS=12
export XSA_LAST_N=12
for wd in 0.085 0.12 0.15; do
    export MUON_WD=$wd
    export EMBED_WD=$wd
    run_one "v4_12L_wd${wd}"
done
export NUM_LAYERS=11
export XSA_LAST_N=11
export MUON_WD=0.085
export EMBED_WD=0.085

# --- 13L with high WD ---
export NUM_LAYERS=13
export XSA_LAST_N=13
for wd in 0.12 0.15 0.20; do
    export MUON_WD=$wd
    export EMBED_WD=$wd
    run_one "v4_13L_wd${wd}"
done
export NUM_LAYERS=11
export XSA_LAST_N=11
export MUON_WD=0.085
export EMBED_WD=0.085

# --- qk_gain sweep (extending range from our earlier sweep) ---
for qk in 2.0 4.0 6.0 8.0; do
    export QK_GAIN_INIT=$qk
    run_one "v4_qk${qk}"
done
export QK_GAIN_INIT=4.0

echo ""
echo "$(date) === VOCAB4096 SWEEP GPU$GPU COMPLETE ==="
