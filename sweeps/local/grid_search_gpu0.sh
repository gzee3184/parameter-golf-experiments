#!/bin/bash
# Grid Search GPU 0 — Fast screening (NO TTT)
# Compares val_bpb and sliding_bpb only. ~3 min per run.

cd ${PROJECT_ROOT:-.}/parameter-golf
source ${VENV:-./venv}/bin/activate

SCRIPT="records/track_10min_16mb/2026-03-27_ActivationSearch/train_gpt.py"
RESULTS="${PROJECT_ROOT:-.}/local_tests"

# Common config
export CUDA_VISIBLE_DEVICES=0
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
export MLP_MULT=3.5
export QUANT_BITS_MLP=8
export QUANT_BITS_ATTN=6
export COMPRESSOR=lzma
export MUON_WD=0.04

# Screening mode — skip sliding window + TTT evals (saves ~20min per run)
export TTT_ENABLED=0
export SKIP_SLIDING=1

echo "$(date) === GRID SEARCH GPU0 START (no TTT) ==="

# ─── Block 1: TTT LR sweep — skip, no TTT in this round ───

# ─── Block 2: Muon Weight Decay sweep ───
for MWD in 0.02 0.04 0.06 0.08 0.1 0.15 0.2; do
    export MUON_WD=$MWD
    LABEL="screen_muon_wd${MWD}"
    echo "$(date) === $LABEL ==="
    nice -n 10 torchrun --standalone --nproc_per_node=1 --master_port=29500 "$SCRIPT" \
        > "$RESULTS/${LABEL}.log" 2>&1 || echo "$LABEL FAILED"
    VAL=$(grep -oP 'val_bpb:\K[0-9.]+' "$RESULTS/${LABEL}.log" 2>/dev/null | tail -1 || echo "—")
    SLID=$(grep "sliding_window_exact" "$RESULTS/${LABEL}.log" 2>/dev/null | grep -oP 'val_bpb:\K[0-9.]+' || echo "—")
    SIZE=$(grep "Total submission size" "$RESULTS/${LABEL}.log" 2>/dev/null | grep -oP '[0-9]+(?= bytes)' || echo "—")
    echo "  $LABEL: val=$VAL sliding=$SLID size=$SIZE"
done

export MUON_WD=0.04  # reset

# ─── Block 3: Bigram config sweep ───
for BCOMBO in "3072_112" "4096_96" "4096_112" "2048_128" "3072_128"; do
    IFS='_' read -r BV BD <<< "$BCOMBO"
    export BIGRAM_VOCAB_SIZE=$BV
    export BIGRAM_DIM=$BD
    LABEL="screen_bigram_v${BV}_d${BD}"
    echo "$(date) === $LABEL ==="
    nice -n 10 torchrun --standalone --nproc_per_node=1 --master_port=29500 "$SCRIPT" \
        > "$RESULTS/${LABEL}.log" 2>&1 || echo "$LABEL FAILED"
    VAL=$(grep -oP 'val_bpb:\K[0-9.]+' "$RESULTS/${LABEL}.log" 2>/dev/null | tail -1 || echo "—")
    SLID=$(grep "sliding_window_exact" "$RESULTS/${LABEL}.log" 2>/dev/null | grep -oP 'val_bpb:\K[0-9.]+' || echo "—")
    SIZE=$(grep "Total submission size" "$RESULTS/${LABEL}.log" 2>/dev/null | grep -oP '[0-9]+(?= bytes)' || echo "—")
    echo "  $LABEL: val=$VAL sliding=$SLID size=$SIZE"
done

export BIGRAM_VOCAB_SIZE=3072
export BIGRAM_DIM=112

# ─── Block 4: Warmdown iters sweep ───
for WD in 1000 2000 4000 6000 8000; do
    export WARMDOWN_ITERS=$WD
    LABEL="screen_warmdown${WD}"
    echo "$(date) === $LABEL ==="
    nice -n 10 torchrun --standalone --nproc_per_node=1 --master_port=29500 "$SCRIPT" \
        > "$RESULTS/${LABEL}.log" 2>&1 || echo "$LABEL FAILED"
    VAL=$(grep -oP 'val_bpb:\K[0-9.]+' "$RESULTS/${LABEL}.log" 2>/dev/null | tail -1 || echo "—")
    SLID=$(grep "sliding_window_exact" "$RESULTS/${LABEL}.log" 2>/dev/null | grep -oP 'val_bpb:\K[0-9.]+' || echo "—")
    echo "  $LABEL: val=$VAL sliding=$SLID"
done

export WARMDOWN_ITERS=4000  # reset

echo ""
echo "$(date) === GRID SEARCH GPU0 COMPLETE ==="
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
