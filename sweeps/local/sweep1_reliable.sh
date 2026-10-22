#!/bin/bash
# Sweep 1 Reliability Rerun — MLP expansion + quant combos
# Runs SEQUENTIALLY on ONE GPU to eliminate contention noise
# Uses SKIP_SLIDING=1 for fast screening

cd ${PROJECT_ROOT:-.}/parameter-golf
source ${VENV:-./venv}/bin/activate

SCRIPT="records/track_10min_16mb/2026-03-27_ActivationSearch/train_gpt.py"
RESULTS="${PROJECT_ROOT:-.}/local_tests"

GPU=${1:-0}
PORT=$((29500 + GPU))
export CUDA_VISIBLE_DEVICES=$GPU

# Base config
export MLP_ACTIVATION=mish2
export MUON_PLUS=1
export XSA_LAST_N=11
export MTP_NUM_HEADS=0
export BIGRAM_VOCAB_SIZE=3072
export BIGRAM_DIM=112
export WARMDOWN_ITERS=4000
export GPTQ_ENABLED=1
export GPTQ_CALIB_SEQS=8
export TARGET_MB=15.9
export NGRAM_CACHE=0
export SEED=1337
export EVAL_SEQ_LEN=2048
export MAX_WALLCLOCK_SECONDS=60
export COMPRESSOR=lzma
export MUON_WD=0.04
export TTT_ENABLED=0
export SKIP_SLIDING=1

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

echo "$(date) === RELIABLE MLP/QUANT SWEEP (GPU $GPU, sequential) ==="

# All MLP expansion × quant combos on same GPU, no contention changes
for combo in "3.0_8_6" "3.5_8_6" "3.5_8_5" "3.5_6_6" "3.5_6_5" "3.5_8_8" "4.0_8_6" "4.0_8_5" "4.0_6_5" "4.0_8_8" "4.5_8_6" "4.5_8_5" "4.5_6_5"; do
    IFS='_' read -r MULT BMLP BATTN <<< "$combo"
    export MLP_MULT=$MULT
    export QUANT_BITS_MLP=$BMLP
    export QUANT_BITS_ATTN=$BATTN
    run_one "rel_mlp${MULT}_q${BMLP}${BATTN}"
done

echo ""
echo "$(date) === RELIABLE SWEEP COMPLETE ==="
echo "All results (sorted by prequant BPB):"
for f in "$RESULTS"/rel_*.log; do
    [ -f "$f" ] || continue
    name=$(basename "$f" .log)
    val=$(grep -oP 'DIAGNOSTIC post_ema.*val_bpb:\K[0-9.]+' "$f" 2>/dev/null)
    qval=$(grep -oP 'roundtrip_exact.*val_bpb:\K[0-9.]+' "$f" 2>/dev/null | tail -1)
    step=$(grep -oP 'step_avg:\K[0-9.]+' "$f" 2>/dev/null | tail -1)
    steps=$(grep -oP 'step:\K[0-9]+(?=/)' "$f" 2>/dev/null | tail -1)
    size=$(grep "Total submission size" "$f" 2>/dev/null | grep -oP '[0-9]+(?= bytes)')
    params=$(grep -oP 'model_params:\K[0-9]+' "$f" 2>/dev/null)
    if [ -n "$size" ]; then size_mb=$(echo "scale=2; $size / 1048576" | bc); else size_mb="—"; fi
    echo "  $name: prequant=$val quant=$qval steps=$steps step_ms=${step}ms art=${size_mb}MB params=$params"
done | sort
