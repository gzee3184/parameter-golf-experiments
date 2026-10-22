#!/bin/bash
# Zero-cost experiments on GPU1: 14L standard + WD at scale
# No new code needed — just env var changes

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
export LOGIT_SOFTCAP=30.0
export MLP_ACTIVATION=leaky05
export BIGRAM_VOCAB_SIZE=0
export VE_ENABLED=0
export QAT_ENABLED=0
export LATE_QAT_THRESHOLD=0
export GATED_ATTENTION=0
export VALUE_RESIDUAL=0
export MTP_NUM_HEADS=0
export GPTQ_ENABLED=1
export GPTQ_CALIB_SEQS=8
export TARGET_MB=15.9
export SEED=1337
export EVAL_SEQ_LEN=2048
export SKIP_SLIDING=0
export TTT_ENABLED=0
export NGRAM_CACHE=0
export MUON_PLUS=1
export EMA_DECAY=0.997

# Submission-ready quant
export QUANT_BITS_MLP=6
export QUANT_BITS_ATTN=5

run_one() {
    local LABEL="$1"
    echo "$(date) === $LABEL ==="
    nice -n 10 torchrun --standalone --nproc_per_node=1 --master_port=$PORT "$SCRIPT" \
        > "$RESULTS/${LABEL}.log" 2>&1 || echo "$LABEL FAILED"
    cp final_model.int6.ptz "$RESULTS/${LABEL}.ptz" 2>/dev/null
    local VAL=$(grep -oP 'DIAGNOSTIC post_ema.*val_bpb:\K[0-9.]+' "$RESULTS/${LABEL}.log" 2>/dev/null)
    local QVAL=$(grep -oP 'roundtrip_exact.*val_bpb:\K[0-9.]+' "$RESULTS/${LABEL}.log" 2>/dev/null | tail -1)
    local SVAL=$(grep -oP 'sliding_exact.*val_bpb:\K[0-9.]+' "$RESULTS/${LABEL}.log" 2>/dev/null | tail -1)
    local STEP=$(grep -oP 'step_avg:\K[0-9.]+' "$RESULTS/${LABEL}.log" 2>/dev/null | tail -1)
    local STEPS=$(grep -oP 'step:\K[0-9]+(?=/)' "$RESULTS/${LABEL}.log" 2>/dev/null | tail -1)
    local SIZE=$(grep "Total submission size" "$RESULTS/${LABEL}.log" 2>/dev/null | grep -oP '[0-9]+(?= bytes)')
    local PARAMS=$(grep -oP 'model_params:\K[0-9]+' "$RESULTS/${LABEL}.log" 2>/dev/null)
    if [ -n "$SIZE" ]; then local SIZE_MB=$(echo "scale=2; $SIZE / 1048576" | bc); else local SIZE_MB="—"; fi
    echo "  $LABEL: prequant=$VAL quant=$QVAL sliding=$SVAL steps=$STEPS step_ms=$STEP art=${SIZE_MB}MB params=$PARAMS"
}

run_ttt() {
    local LABEL="$1"
    echo "$(date) === $LABEL ==="
    nice -n 10 torchrun --standalone --nproc_per_node=1 --master_port=$PORT "$SCRIPT" \
        > "$RESULTS/${LABEL}.log" 2>&1 || echo "$LABEL FAILED"
    local TVAL=$(grep -oP 'ttt_exact.*val_bpb:\K[0-9.]+' "$RESULTS/${LABEL}.log" 2>/dev/null | tail -1)
    local QVAL=$(grep -oP 'roundtrip_exact.*val_bpb:\K[0-9.]+' "$RESULTS/${LABEL}.log" 2>/dev/null | tail -1)
    local SVAL=$(grep -oP 'sliding_exact.*val_bpb:\K[0-9.]+' "$RESULTS/${LABEL}.log" 2>/dev/null | tail -1)
    echo "  $LABEL: ttt=$TVAL roundtrip=$QVAL sliding=$SVAL"
}

export MAX_WALLCLOCK_SECONDS=12000

echo "$(date) ═══════════════════════════════════════"
echo "  RUN 1: 14L standard (int6/int5) — does one more layer fit?"
echo "═══════════════════════════════════════"

export NUM_LAYERS=14
export XSA_LAST_N=14
run_one "v4R_14L_q65"

echo ""
echo "$(date) ═══════════════════════════════════════"
echo "  RUN 2: 13L + WD 0.12 (int6/int5) — does higher WD improve compression?"
echo "═══════════════════════════════════════"

export NUM_LAYERS=13
export XSA_LAST_N=13
export MUON_WD=0.12
export EMBED_WD=0.12
run_one "v4R_13L_wd012_q65"

echo ""
echo "$(date) ═══════════════════════════════════════"
echo "  TTT on best model"
echo "═══════════════════════════════════════"

# Pick best of the two by post-quant BPB
BPB_14L=$(grep -oP 'roundtrip_exact.*val_bpb:\K[0-9.]+' "$RESULTS/v4R_14L_q65.log" 2>/dev/null | tail -1)
BPB_13LWD=$(grep -oP 'roundtrip_exact.*val_bpb:\K[0-9.]+' "$RESULTS/v4R_13L_wd012_q65.log" 2>/dev/null | tail -1)

echo "14L: $BPB_14L BPB | 13L+wd012: $BPB_13LWD BPB"

if [ -n "$BPB_14L" ] && [ -n "$BPB_13LWD" ] && [ "$(echo "$BPB_14L < $BPB_13LWD" | bc -l)" -eq 1 ]; then
    BEST_PTZ="$RESULTS/v4R_14L_q65.ptz"
    BEST_NAME="14L"
    export NUM_LAYERS=14
    export XSA_LAST_N=14
    export MUON_WD=0.085
    export EMBED_WD=0.085
elif [ -n "$BPB_13LWD" ]; then
    BEST_PTZ="$RESULTS/v4R_13L_wd012_q65.ptz"
    BEST_NAME="13L_wd012"
    export NUM_LAYERS=13
    export XSA_LAST_N=13
    export MUON_WD=0.12
    export EMBED_WD=0.12
else
    BEST_PTZ="$RESULTS/v4R_14L_q65.ptz"
    BEST_NAME="14L"
    export NUM_LAYERS=14
    export XSA_LAST_N=14
    export MUON_WD=0.085
    export EMBED_WD=0.085
fi

echo "Best: $BEST_NAME → $BEST_PTZ"

export MAX_WALLCLOCK_SECONDS=1
export TTT_ONLY="$BEST_PTZ"
export TTT_ENABLED=1
export TTT_EPOCHS=3
export TTT_CHUNK_TOKENS=32768
export TTT_FREEZE_BLOCKS=2

# AdamW conservative (proven winner)
export TTT_OPTIMIZER=adamw
export TTT_LR=0.0001
export TTT_WEIGHT_DECAY=0.01
run_ttt "v4R_g1_ttt_adamw_con"

# AdamW moderate (new data point)
export TTT_LR=0.0002
export TTT_WEIGHT_DECAY=0.01
run_ttt "v4R_g1_ttt_adamw_mod"

# AdamW fine-grained (between con and mod)
export TTT_LR=0.00015
export TTT_WEIGHT_DECAY=0.01
run_ttt "v4R_g1_ttt_adamw_mid"

echo ""
echo "$(date) ═══════════════════════════════════════"
echo "  ALL DONE"
echo "═══════════════════════════════════════"
