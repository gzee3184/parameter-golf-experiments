#!/bin/bash
# Vocab4096 long runs — GPU1 (complementary configs)
# Phase 1: 600s comparison runs for configs not on GPU0
# Phase 2: Pick top 2 from GPU1 set, run ~7000 steps
# Phase 3: TTT on best GPU1 model

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
export MLP_ACTIVATION=leaky05
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
export SKIP_SLIDING=1
export TTT_ENABLED=0
export NGRAM_CACHE=0
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

echo "$(date) ═══════════════════════════════════════"
echo "  PHASE 1: 600s runs (GPU $GPU) — complementary configs"
echo "═══════════════════════════════════════"

export MAX_WALLCLOCK_SECONDS=600

# F: Higher WD (0.15 — further compression)
export EMA_DECAY=0.997
export NUM_LAYERS=11
export XSA_LAST_N=11
export MUON_WD=0.15
export EMBED_WD=0.15
export MLP_ACTIVATION=leaky05
run_one "v4L_wd015"

# G: 12L (midpoint between 11L and 13L)
export MUON_WD=0.085
export EMBED_WD=0.085
export NUM_LAYERS=12
export XSA_LAST_N=12
run_one "v4L_12L"

# H: 12L + wd0.12 (budget-filling combo)
export NUM_LAYERS=12
export XSA_LAST_N=12
export MUON_WD=0.12
export EMBED_WD=0.12
run_one "v4L_12L_wd012"

# I: MLP 4.5× (wider MLP, testing if artifact fits under 16MB)
export NUM_LAYERS=11
export XSA_LAST_N=11
export MUON_WD=0.085
export EMBED_WD=0.085
export MLP_MULT=4.5
run_one "v4L_mlp45"

# J: MLP 4.5× + wd0.12 (use WD to compress wider MLP)
export MLP_MULT=4.5
export MUON_WD=0.12
export EMBED_WD=0.12
run_one "v4L_mlp45_wd012"

# K: EMA 0.993 at 600s (interpolating between 0.99 and 0.995)
export MLP_MULT=4
export MUON_WD=0.085
export EMBED_WD=0.085
export EMA_DECAY=0.993
run_one "v4L_ema993"

echo ""
echo "$(date) ═══════════════════════════════════════"
echo "  PHASE 1 COMPLETE — selecting top 2 configs"
echo "═══════════════════════════════════════"

# Collect all Phase 1 results and pick top 2 by post-quant BPB
BEST_FILE=$(mktemp)
for f in "$RESULTS"/v4L_wd015.log "$RESULTS"/v4L_12L.log "$RESULTS"/v4L_12L_wd012.log \
         "$RESULTS"/v4L_mlp45.log "$RESULTS"/v4L_mlp45_wd012.log "$RESULTS"/v4L_ema993.log; do
    [ -f "$f" ] || continue
    name=$(basename "$f" .log)
    qval=$(grep -oP 'roundtrip_exact.*val_bpb:\K[0-9.]+' "$f" 2>/dev/null | tail -1)
    preval=$(grep -oP 'DIAGNOSTIC post_ema.*val_bpb:\K[0-9.]+' "$f" 2>/dev/null)
    steps=$(grep -oP 'step:\K[0-9]+(?=/)' "$f" 2>/dev/null | tail -1)
    step_ms=$(grep -oP 'step_avg:\K[0-9.]+' "$f" 2>/dev/null | tail -1)
    size=$(grep "Total submission size" "$f" 2>/dev/null | grep -oP '[0-9]+(?= bytes)')
    params=$(grep -oP 'model_params:\K[0-9]+' "$f" 2>/dev/null)
    [ -n "$qval" ] && echo "$qval $name $preval $steps $step_ms $size $params" >> "$BEST_FILE"
done

echo ""
echo "All Phase 1 GPU1 results (sorted by post-quant BPB):"
sort -n "$BEST_FILE" | while read qval name preval steps step_ms size params; do
    if [ -n "$size" ]; then size_mb=$(echo "scale=2; $size / 1048576" | bc); else size_mb="—"; fi
    echo "  $name: prequant=$preval quant=$qval steps=$steps step_ms=${step_ms}ms art=${size_mb}MB"
done

TOP1=$(sort -n "$BEST_FILE" | head -1)
TOP2=$(sort -n "$BEST_FILE" | head -2 | tail -1)
TOP1_NAME=$(echo "$TOP1" | awk '{print $2}')
TOP2_NAME=$(echo "$TOP2" | awk '{print $2}')
TOP1_BPB=$(echo "$TOP1" | awk '{print $1}')
TOP2_BPB=$(echo "$TOP2" | awk '{print $1}')

echo ""
echo "TOP 1: $TOP1_NAME ($TOP1_BPB BPB)"
echo "TOP 2: $TOP2_NAME ($TOP2_BPB BPB)"
rm "$BEST_FILE"

echo ""
echo "$(date) ═══════════════════════════════════════"
echo "  PHASE 2: ~7000-step runs on top 2 configs"
echo "═══════════════════════════════════════"

export MAX_WALLCLOCK_SECONDS=12000
export SKIP_SLIDING=0

set_config() {
    local name="$1"
    export NUM_LAYERS=11
    export XSA_LAST_N=11
    export MUON_WD=0.085
    export EMBED_WD=0.085
    export MLP_MULT=4
    export MLP_ACTIVATION=leaky05
    export EMA_DECAY=0.997

    case "$name" in
        *wd015*)  export MUON_WD=0.15; export EMBED_WD=0.15 ;;
        *wd012*)  export MUON_WD=0.12; export EMBED_WD=0.12 ;;
    esac
    case "$name" in
        *12L*)    export NUM_LAYERS=12; export XSA_LAST_N=12 ;;
        *13L*)    export NUM_LAYERS=13; export XSA_LAST_N=13 ;;
    esac
    case "$name" in
        *mlp45*)  export MLP_MULT=4.5 ;;
    esac
    case "$name" in
        *ema993*) export EMA_DECAY=0.993 ;;
    esac
}

set_config "$TOP1_NAME"
run_one "v4F_${TOP1_NAME#v4L_}"
cp ${PROJECT_ROOT:-.}/parameter-golf/final_model.int6.ptz \
   "$RESULTS/v4F_${TOP1_NAME#v4L_}.ptz" 2>/dev/null

set_config "$TOP2_NAME"
run_one "v4F_${TOP2_NAME#v4L_}"
cp ${PROJECT_ROOT:-.}/parameter-golf/final_model.int6.ptz \
   "$RESULTS/v4F_${TOP2_NAME#v4L_}.ptz" 2>/dev/null

echo ""
echo "$(date) ═══════════════════════════════════════"
echo "  PHASE 2 COMPLETE — picking best for TTT"
echo "═══════════════════════════════════════"

FULL1_BPB=$(grep -oP 'roundtrip_exact.*val_bpb:\K[0-9.]+' "$RESULTS/v4F_${TOP1_NAME#v4L_}.log" 2>/dev/null | tail -1)
FULL2_BPB=$(grep -oP 'roundtrip_exact.*val_bpb:\K[0-9.]+' "$RESULTS/v4F_${TOP2_NAME#v4L_}.log" 2>/dev/null | tail -1)
echo "Full run 1 (${TOP1_NAME#v4L_}): $FULL1_BPB BPB"
echo "Full run 2 (${TOP2_NAME#v4L_}): $FULL2_BPB BPB"

if [ "$(echo "$FULL1_BPB < $FULL2_BPB" | bc -l)" -eq 1 ]; then
    BEST_CONFIG="${TOP1_NAME#v4L_}"
else
    BEST_CONFIG="${TOP2_NAME#v4L_}"
fi
BEST_PTZ="$RESULTS/v4F_${BEST_CONFIG}.ptz"
echo "Best: $BEST_CONFIG → $BEST_PTZ"

echo ""
echo "$(date) ═══════════════════════════════════════"
echo "  PHASE 3: TTT evaluation on best model"
echo "═══════════════════════════════════════"

set_config "v4L_${BEST_CONFIG}"
export SKIP_SLIDING=0
export MAX_WALLCLOCK_SECONDS=1

# TTT config 1: SGD baseline
export TTT_ONLY="$BEST_PTZ"
export TTT_ENABLED=1
export TTT_OPTIMIZER=sgd
export TTT_LR=0.002
export TTT_WEIGHT_DECAY=0.0
export TTT_EPOCHS=3
export TTT_CHUNK_TOKENS=32768
export TTT_FREEZE_BLOCKS=2
export TTT_MOMENTUM=0.9
run_one "v4T_g1_sgd"

# TTT config 2: AdamW conservative
export TTT_OPTIMIZER=adamw
export TTT_LR=0.0001
export TTT_WEIGHT_DECAY=0.01
export TTT_EPOCHS=3
run_one "v4T_g1_adamw_con"

# TTT config 3: AdamW aggressive
export TTT_LR=0.0005
export TTT_WEIGHT_DECAY=0.01
export TTT_EPOCHS=3
run_one "v4T_g1_adamw_agg"

echo ""
echo "$(date) ═══════════════════════════════════════"
echo "  ALL PHASES COMPLETE (GPU $GPU)"
echo "═══════════════════════════════════════"
echo ""
echo "=== FINAL SUMMARY ==="
for f in "$RESULTS"/v4L_wd015.log "$RESULTS"/v4L_12L.log "$RESULTS"/v4L_12L_wd012.log \
         "$RESULTS"/v4L_mlp45.log "$RESULTS"/v4L_mlp45_wd012.log "$RESULTS"/v4L_ema993.log \
         "$RESULTS"/v4F_*.log "$RESULTS"/v4T_g1_*.log; do
    [ -f "$f" ] || continue
    name=$(basename "$f" .log)
    qval=$(grep -oP 'roundtrip_exact.*val_bpb:\K[0-9.]+' "$f" 2>/dev/null | tail -1)
    sval=$(grep -oP 'final_int6_sliding_window.*val_bpb:\K[0-9.]+' "$f" 2>/dev/null | tail -1)
    tval=$(grep -oP 'ttt_.*val_bpb:\K[0-9.]+' "$f" 2>/dev/null | tail -1)
    echo "  $name: roundtrip=$qval sliding=$sval ttt=$tval"
done
