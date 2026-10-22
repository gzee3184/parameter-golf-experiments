#!/bin/bash
# Sweep 2 GPU 1 — Quantization & TTT Prep (NO TTT eval, but TTT params logged)
# Tests: layer-adaptive mixed quant, TTT chunk size (dry), spectral K configs
# ~3 min per run, ~25 runs ≈ 75 min

cd ${PROJECT_ROOT:-.}/parameter-golf
source ${VENV:-./venv}/bin/activate

SCRIPT="records/track_10min_16mb/2026-03-27_ActivationSearch/train_gpt.py"
RESULTS="${PROJECT_ROOT:-.}/local_tests"

# Base config (best known)
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
    nice -n 10 torchrun --standalone --nproc_per_node=1 --master_port=29501 "$SCRIPT" \
        > "$RESULTS/${LABEL}.log" 2>&1 || echo "$LABEL FAILED"
    local VAL=$(grep -oP 'val_bpb:\K[0-9.]+' "$RESULTS/${LABEL}.log" 2>/dev/null | tail -1 || echo "—")
    local SLID=$(grep "sliding_window_exact" "$RESULTS/${LABEL}.log" 2>/dev/null | grep -oP 'val_bpb:\K[0-9.]+' || echo "—")
    local SIZE=$(grep "Total submission size" "$RESULTS/${LABEL}.log" 2>/dev/null | grep -oP '[0-9]+(?= bytes)' || echo "—")
    local STEP=$(grep -oP 'step_avg:\K[0-9.]+' "$RESULTS/${LABEL}.log" 2>/dev/null | tail -1 || echo "—")
    local PARAMS=$(grep -oP 'model_params:\K[0-9]+' "$RESULTS/${LABEL}.log" 2>/dev/null || echo "—")
    if [ "$SIZE" != "—" ]; then local SIZE_MB=$(echo "scale=2; $SIZE / 1048576" | bc); else local SIZE_MB="—"; fi
    echo "  $LABEL: val=$VAL sliding=$SLID artifact=${SIZE_MB}MB step_ms=$step params=$PARAMS"
}

echo "$(date) === SWEEP 2 GPU1 START ==="

# ─── Block 1: Layer-adaptive mixed quantization ───
# Abay showed: MLP 4× more quant-sensitive, layer 9 most sensitive
# Test: give MLP more bits, attention fewer; test with more layers too

# Standard combos (already tested some, but with new context)
for combo in "8_6" "8_5" "6_6" "6_5" "8_4" "8_8"; do
    IFS='_' read -r BMLP BATTN <<< "$combo"
    export QUANT_BITS_MLP=$BMLP
    export QUANT_BITS_ATTN=$BATTN
    run_one "s2_quant_m${BMLP}_a${BATTN}"
done

# 12 layers × quant combos (interaction: more layers needs more compression)
export NUM_LAYERS=12
for combo in "8_6" "8_5" "6_5" "8_4"; do
    IFS='_' read -r BMLP BATTN <<< "$combo"
    export QUANT_BITS_MLP=$BMLP
    export QUANT_BITS_ATTN=$BATTN
    run_one "s2_12L_m${BMLP}_a${BATTN}"
done
export NUM_LAYERS=11
export QUANT_BITS_MLP=8
export QUANT_BITS_ATTN=6

# 13 layers × quant (push further)
export NUM_LAYERS=13
for combo in "8_5" "6_5" "8_4"; do
    IFS='_' read -r BMLP BATTN <<< "$combo"
    export QUANT_BITS_MLP=$BMLP
    export QUANT_BITS_ATTN=$BATTN
    run_one "s2_13L_m${BMLP}_a${BATTN}"
done
export NUM_LAYERS=11
export QUANT_BITS_MLP=8
export QUANT_BITS_ATTN=6

# ─── Block 2: MLP expansion × layers (budget-filling) ───
# How big can we go while fitting in 16MB?
for combo in "3.5_11" "3.5_12" "3.5_13" "4.0_11" "4.0_12" "3.0_13" "3.0_14"; do
    IFS='_' read -r MULT NL <<< "$combo"
    export MLP_MULT=$MULT
    export NUM_LAYERS=$NL
    # Use aggressive quant to fit larger models
    export QUANT_BITS_MLP=8
    export QUANT_BITS_ATTN=5
    run_one "s2_mlp${MULT}_${NL}L_q85"
done
export MLP_MULT=3.5
export NUM_LAYERS=11
export QUANT_BITS_MLP=8
export QUANT_BITS_ATTN=6

# ─── Block 3: Compressor × model size (does Brotli help bigger models more?) ───
for COMP in lzma brotli; do
    export COMPRESSOR=$COMP
    export NUM_LAYERS=12
    export QUANT_BITS_MLP=8
    export QUANT_BITS_ATTN=5
    run_one "s2_12L_${COMP}"
done
export COMPRESSOR=lzma
export NUM_LAYERS=11
export QUANT_BITS_ATTN=6

echo ""
echo "$(date) === SWEEP 2 GPU1 COMPLETE ==="
echo "All results:"
for f in "$RESULTS"/s2_*.log; do
    [ -f "$f" ] || continue
    name=$(basename "$f" .log)
    val=$(grep -oP 'val_bpb:\K[0-9.]+' "$f" 2>/dev/null | tail -1 || echo "—")
    slid=$(grep "sliding_window_exact" "$f" 2>/dev/null | grep -oP 'val_bpb:\K[0-9.]+' || echo "—")
    size=$(grep "Total submission size" "$f" 2>/dev/null | grep -oP '[0-9]+(?= bytes)' || echo "—")
    params=$(grep -oP 'model_params:\K[0-9]+' "$f" 2>/dev/null || echo "—")
    if [ "$size" != "—" ]; then size_mb=$(echo "scale=2; $size / 1048576" | bc); else size_mb="—"; fi
    echo "  $name: val=$val sliding=$slid artifact=${size_mb}MB params=$params"
done
