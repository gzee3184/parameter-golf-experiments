#!/bin/bash
# Sweep 2 GPU 0 — Architecture & Training Dynamics (NO TTT)
# Tests: num_layers, EMA decay, QK gain × softcap, batch size
# ~3 min per run, ~25 runs ≈ 75 min

cd ${PROJECT_ROOT:-.}/parameter-golf
source ${VENV:-./venv}/bin/activate

SCRIPT="records/track_10min_16mb/2026-03-27_ActivationSearch/train_gpt.py"
RESULTS="${PROJECT_ROOT:-.}/local_tests"

# Base config (best known)
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
export TTT_ENABLED=0
export SKIP_SLIDING=1

run_one() {
    local LABEL="$1"
    echo "$(date) === $LABEL ==="
    nice -n 10 torchrun --standalone --nproc_per_node=1 --master_port=29500 "$SCRIPT" \
        > "$RESULTS/${LABEL}.log" 2>&1 || echo "$LABEL FAILED"
    local VAL=$(grep -oP 'val_bpb:\K[0-9.]+' "$RESULTS/${LABEL}.log" 2>/dev/null | tail -1 || echo "—")
    local SLID=$(grep "sliding_window_exact" "$RESULTS/${LABEL}.log" 2>/dev/null | grep -oP 'val_bpb:\K[0-9.]+' || echo "—")
    local SIZE=$(grep "Total submission size" "$RESULTS/${LABEL}.log" 2>/dev/null | grep -oP '[0-9]+(?= bytes)' || echo "—")
    local STEP=$(grep -oP 'step_avg:\K[0-9.]+' "$RESULTS/${LABEL}.log" 2>/dev/null | tail -1 || echo "—")
    local PARAMS=$(grep -oP 'model_params:\K[0-9]+' "$RESULTS/${LABEL}.log" 2>/dev/null || echo "—")
    if [ "$SIZE" != "—" ]; then local SIZE_MB=$(echo "scale=2; $SIZE / 1048576" | bc); else local SIZE_MB="—"; fi
    echo "  $LABEL: val=$VAL sliding=$SLID artifact=${SIZE_MB}MB step_ms=$STEP params=$PARAMS"
}

echo "$(date) === SWEEP 2 GPU0 START ==="

# ─── Block 1: Number of layers ───
# Interaction: more layers → more params → bigger artifact → may need different quant
for NL in 9 10 11 12 13 14; do
    export NUM_LAYERS=$NL
    export NUM_LAYERS=$NL
    run_one "s2_layers${NL}"
done
export NUM_LAYERS=11

# ─── Block 2: QK Gain × Logit Softcap (interacting) ───
# Both control attention sharpness — sweep jointly
for QK in 1.0 1.5 2.0 2.5; do
    for SC in 20.0 30.0 50.0; do
        export QK_GAIN_INIT=$QK
        export LOGIT_SOFTCAP=$SC
        run_one "s2_qk${QK}_sc${SC}"
    done
done
export QK_GAIN_INIT=1.5
export LOGIT_SOFTCAP=30.0

# ─── Block 3: EMA decay ───
# On H100 (5732 steps) optimal is likely higher than on RTX (384 steps)
for EMA in 0.99 0.995 0.997 0.999 0.9995; do
    export EMA_DECAY=$EMA
    run_one "s2_ema${EMA}"
done
export EMA_DECAY=0.997

# ─── Block 4: Training batch size ───
# Smaller batch → more steps in same time, but noisier gradients
for BT in 393216 524288 786432 1048576; do
    export TRAIN_BATCH_TOKENS=$BT
    run_one "s2_batch${BT}"
done
export TRAIN_BATCH_TOKENS=786432

echo ""
echo "$(date) === SWEEP 2 GPU0 COMPLETE ==="
echo "All results:"
for f in "$RESULTS"/s2_*.log; do
    [ -f "$f" ] || continue
    name=$(basename "$f" .log)
    val=$(grep -oP 'val_bpb:\K[0-9.]+' "$f" 2>/dev/null | tail -1 || echo "—")
    slid=$(grep "sliding_window_exact" "$f" 2>/dev/null | grep -oP 'val_bpb:\K[0-9.]+' || echo "—")
    size=$(grep "Total submission size" "$f" 2>/dev/null | grep -oP '[0-9]+(?= bytes)' || echo "—")
    step=$(grep -oP 'step_avg:\K[0-9.]+' "$f" 2>/dev/null | tail -1 || echo "—")
    params=$(grep -oP 'model_params:\K[0-9]+' "$f" 2>/dev/null || echo "—")
    if [ "$size" != "—" ]; then size_mb=$(echo "scale=2; $size / 1048576" | bc); else size_mb="—"; fi
    echo "  $name: val=$val sliding=$slid artifact=${size_mb}MB step_ms=$step params=$params"
done
