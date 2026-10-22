#!/usr/bin/env bash
set -euo pipefail

# Head-to-head: mish2 vs gelu2 on identical local conditions
# Runs both in parallel on GPU0 and GPU1
# Uses the activation-patched train_gpt.py with _ACT_MAP dispatch

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TRAIN_SCRIPT="$SCRIPT_DIR/2026-03-27_ActivationSearch/train_gpt.py"
RESULTS_DIR="$SCRIPT_DIR/mish2_vs_gelu2_results"
VENV="${VENV:-./venv}"

mkdir -p "$RESULTS_DIR"

source "$VENV/bin/activate"

# Matching overnight run conditions exactly: 2400s wallclock, nproc=1, seed 1337
COMMON_VARS=(
    SEED=1337
    NUM_LAYERS=11 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=3.0
    BIGRAM_VOCAB_SIZE=1536 BIGRAM_DIM=128
    XSA_LAST_N=4 ROPE_DIMS=16 LN_SCALE=1
    VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10
    SWA_ENABLED=1 SWA_EVERY=50
    MUON_WD=0.04 ADAM_WD=0.04
    MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035
    MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 MUON_MOMENTUM_WARMUP_STEPS=1500
    WARMDOWN_ITERS=3500 ITERATIONS=9000
    MAX_WALLCLOCK_SECONDS=2400
    LATE_QAT_THRESHOLD=0.15 EVAL_STRIDE=64
    TTT_ENABLED=0
    VAL_LOSS_EVERY=50 TRAIN_LOG_EVERY=10
)

run_variant() {
    local name="$1" gpu="$2" port="$3"
    local logfile="$RESULTS_DIR/${name}.log"

    # Skip if already completed
    if [[ -f "$logfile" ]] && grep -q "final_int6_roundtrip_exact" "$logfile"; then
        echo "[$(date '+%H:%M:%S')] SKIP $name (already done)"
        return 0
    fi

    echo "[$(date '+%H:%M:%S')] START $name on GPU$gpu → $logfile"
    local t0=$SECONDS

    env "${COMMON_VARS[@]}" MLP_ACTIVATION="$name" \
        CUDA_VISIBLE_DEVICES="$gpu" \
        nice -n 10 \
        torchrun --standalone --nproc_per_node=1 --master_port="$port" \
        "$TRAIN_SCRIPT" > "$logfile" 2>&1
    local rc=$?

    local elapsed=$(( SECONDS - t0 ))
    local val_bpb=$(grep -o 'val_bpb:[0-9.]*' "$logfile" | tail -1 | cut -d: -f2)
    local step_avg=$(grep -o 'step_avg:[0-9.]*' "$logfile" | tail -1 | cut -d: -f2)
    local steps=$(grep -o 'step:[0-9]*/9000' "$logfile" | tail -1 | grep -o '^[^/]*' | cut -d: -f2)
    echo "[$(date '+%H:%M:%S')] DONE  $name — val_bpb=$val_bpb step_avg=${step_avg}ms steps=$steps elapsed=${elapsed}s exit=$rc"
}

echo "============================================="
echo "  mish2 vs gelu2 — Head-to-Head Comparison"
echo "  $(date)"
echo "============================================="
echo "Training script: $TRAIN_SCRIPT"
echo "Wallclock: 2400s | Seed: 1337 | TTT: off"
echo ""

# Run in parallel: mish2 on GPU0, gelu2 on GPU1
run_variant "mish2" 0 29500 &
PID_MISH=$!

run_variant "gelu2" 1 29501 &
PID_GELU=$!

# Wait for both
wait $PID_MISH
wait $PID_GELU

echo ""
echo "============================================="
echo "  RESULTS SUMMARY"
echo "============================================="

for name in mish2 gelu2; do
    logfile="$RESULTS_DIR/${name}.log"
    if [[ -f "$logfile" ]]; then
        val_bpb=$(grep -o 'val_bpb:[0-9.]*' "$logfile" | tail -1 | cut -d: -f2)
        sliding=$(grep -o 'final_int6_sliding_window_exact.*val_bpb:[0-9.]*' "$logfile" | grep -o '[0-9.]*$')
        roundtrip=$(grep -o 'final_int6_roundtrip_exact.*val_bpb:[0-9.]*' "$logfile" | grep -o '[0-9.]*$')
        step_avg=$(grep -o 'step_avg:[0-9.]*' "$logfile" | tail -1 | cut -d: -f2)
        steps=$(grep -o 'step:[0-9]*/9000' "$logfile" | tail -1 | grep -o '^[^/]*' | cut -d: -f2)
        artifact=$(grep -o 'Total submission size int6+lzma: [0-9]* bytes' "$logfile" | grep -o '[0-9]*')
        printf "  %-8s  val_bpb=%-12s  sliding=%-12s  roundtrip=%-12s  steps=%-5s  avg=%-8s  artifact=%s\n" \
            "$name" "$val_bpb" "${sliding:-N/A}" "${roundtrip:-N/A}" "$steps" "${step_avg}ms" "${artifact:-N/A}"
    fi
done

echo ""
echo "Previous overnight gelu2 result: val_bpb=1.36577345, step_avg=1426.74ms, steps=1683"
echo "Done at $(date)"
