#!/bin/bash
# Grid Search GPU 1 — Fast screening (NO TTT)
# Tests MLP expansion + quantization combos. ~3 min per run.

cd ${PROJECT_ROOT:-.}/parameter-golf
source ${VENV:-./venv}/bin/activate

SCRIPT="records/track_10min_16mb/2026-03-27_ActivationSearch/train_gpt.py"
RESULTS="${PROJECT_ROOT:-.}/local_tests"

# Common config
export CUDA_VISIBLE_DEVICES=1
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

# Screening mode — skip sliding window + TTT evals (saves ~20min per run)
export TTT_ENABLED=0
export SKIP_SLIDING=1

echo "$(date) === GRID SEARCH GPU1 START (no TTT) ==="

# ─── Block 1: MLP expansion + quantization combos ───
for combo in "3.0_8_6" "3.5_8_6" "3.5_8_5" "3.5_6_6" "3.5_6_5" "4.0_8_6" "4.0_8_5" "4.0_6_5" "4.5_8_5" "4.5_6_5" "3.5_8_8" "4.0_8_8"; do
    IFS='_' read -r MULT BMLP BATTN <<< "$combo"
    export MLP_MULT=$MULT
    export QUANT_BITS_MLP=$BMLP
    export QUANT_BITS_ATTN=$BATTN
    LABEL="screen_mlp${MULT}_q${BMLP}${BATTN}"
    echo "$(date) === $LABEL ==="
    nice -n 10 torchrun --standalone --nproc_per_node=1 --master_port=29501 "$SCRIPT" \
        > "$RESULTS/${LABEL}.log" 2>&1 || echo "$LABEL FAILED"
    VAL=$(grep -oP 'val_bpb:\K[0-9.]+' "$RESULTS/${LABEL}.log" 2>/dev/null | tail -1 || echo "—")
    SLID=$(grep "sliding_window_exact" "$RESULTS/${LABEL}.log" 2>/dev/null | grep -oP 'val_bpb:\K[0-9.]+' || echo "—")
    SIZE=$(grep "Total submission size" "$RESULTS/${LABEL}.log" 2>/dev/null | grep -oP '[0-9]+(?= bytes)' || echo "—")
    if [ "$SIZE" != "—" ]; then SIZE_MB=$(echo "scale=2; $SIZE / 1048576" | bc); else SIZE_MB="—"; fi
    echo "  $LABEL: val=$VAL sliding=$SLID artifact=${SIZE_MB}MB"
done

# ─── Block 2: Compressor comparison ───
export MLP_MULT=3.5
export QUANT_BITS_MLP=8
export QUANT_BITS_ATTN=6
for COMP in lzma brotli; do
    export COMPRESSOR=$COMP
    LABEL="screen_compress_${COMP}"
    echo "$(date) === $LABEL ==="
    nice -n 10 torchrun --standalone --nproc_per_node=1 --master_port=29501 "$SCRIPT" \
        > "$RESULTS/${LABEL}.log" 2>&1 || echo "$LABEL FAILED"
    SLID=$(grep "sliding_window_exact" "$RESULTS/${LABEL}.log" 2>/dev/null | grep -oP 'val_bpb:\K[0-9.]+' || echo "—")
    SIZE=$(grep "Total submission size" "$RESULTS/${LABEL}.log" 2>/dev/null | grep -oP '[0-9]+(?= bytes)' || echo "—")
    if [ "$SIZE" != "—" ]; then SIZE_MB=$(echo "scale=2; $SIZE / 1048576" | bc); else SIZE_MB="—"; fi
    echo "  $LABEL: sliding=$SLID artifact=${SIZE_MB}MB"
done

# ─── Block 3: GPTQ calibration sequences sweep ───
export COMPRESSOR=lzma
for CALIB in 4 8 16 32; do
    export GPTQ_CALIB_SEQS=$CALIB
    LABEL="screen_gptq_calib${CALIB}"
    echo "$(date) === $LABEL ==="
    nice -n 10 torchrun --standalone --nproc_per_node=1 --master_port=29501 "$SCRIPT" \
        > "$RESULTS/${LABEL}.log" 2>&1 || echo "$LABEL FAILED"
    SLID=$(grep "sliding_window_exact" "$RESULTS/${LABEL}.log" 2>/dev/null | grep -oP 'val_bpb:\K[0-9.]+' || echo "—")
    echo "  $LABEL: sliding=$SLID"
done

export GPTQ_CALIB_SEQS=8  # reset

echo ""
echo "$(date) === GRID SEARCH GPU1 COMPLETE ==="
echo "Results:"
for f in "$RESULTS"/screen_*.log; do
    [ -f "$f" ] || continue
    name=$(basename "$f" .log)
    val=$(grep -oP 'val_bpb:\K[0-9.]+' "$f" 2>/dev/null | tail -1 || echo "—")
    slid=$(grep "sliding_window_exact" "$f" 2>/dev/null | grep -oP 'val_bpb:\K[0-9.]+' || echo "—")
    size=$(grep "Total submission size" "$f" 2>/dev/null | grep -oP '[0-9]+(?= bytes)' || echo "—")
    if [ "$size" != "—" ]; then size_mb=$(echo "scale=2; $size / 1048576" | bc); else size_mb="—"; fi
    echo "  $name: val=$val sliding=$slid artifact=${size_mb}MB"
done
