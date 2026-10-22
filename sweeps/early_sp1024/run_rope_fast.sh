#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TRAIN_SCRIPT="$SCRIPT_DIR/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py"
RESULTS_DIR="$SCRIPT_DIR/rope_fast_results"
VENV="${VENV:-./venv}"

mkdir -p "$RESULTS_DIR"
source "$VENV/bin/activate"

WALLCLOCK=1200

COMMON_VARS=(
    SEED=1337
    NUM_LAYERS=11 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=3.0
    BIGRAM_VOCAB_SIZE=1536 BIGRAM_DIM=128
    XSA_LAST_N=4 LN_SCALE=1
    VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10
    SWA_ENABLED=1 SWA_EVERY=50
    MUON_WD=0.04 ADAM_WD=0.04
    MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035
    MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 MUON_MOMENTUM_WARMUP_STEPS=1500
    WARMDOWN_ITERS=3500 ITERATIONS=9000
    MAX_WALLCLOCK_SECONDS=$WALLCLOCK
    LATE_QAT_THRESHOLD=0.15 EVAL_STRIDE=64
    TTT_ENABLED=0
    VAL_LOSS_EVERY=50 TRAIN_LOG_EVERY=10
)

run_variant() {
    local dims="$1" gpu="$2" port="$3"
    local name="rope_${dims}"
    local logfile="$RESULTS_DIR/${name}.log"

    if [[ -f "$logfile" ]] && grep -q "final_int6_roundtrip_exact" "$logfile"; then
        echo "[$(date '+%H:%M:%S')] SKIP $name (already done)"
        return 0
    fi

    echo "[$(date '+%H:%M:%S')] START $name on GPU$gpu → $logfile"
    local t0=$SECONDS

    env "${COMMON_VARS[@]}" ROPE_DIMS="$dims" \
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
echo "  RoPE Dimension Sweep (fast: ${WALLCLOCK}s)"
echo "  $(date)"
echo "============================================="

# Batch 1: rope_8 + rope_24 in parallel
echo "--- Batch 1: rope_8 (GPU0) + rope_24 (GPU1) ---"
run_variant 8  0 29500 &
PID1=$!
run_variant 24 1 29501 &
PID2=$!
wait $PID1
wait $PID2

# Batch 2: rope_32 on GPU0
echo "--- Batch 2: rope_32 (GPU0) ---"
run_variant 32 0 29500

echo ""
echo "============================================="
echo "  RESULTS SUMMARY"
echo "============================================="
for dims in 8 24 32; do
    logfile="$RESULTS_DIR/rope_${dims}.log"
    if [[ -f "$logfile" ]]; then
        val_bpb=$(grep -o 'val_bpb:[0-9.]*' "$logfile" | tail -1 | cut -d: -f2)
        roundtrip=$(grep -o 'final_int6_roundtrip_exact.*val_bpb:[0-9.]*' "$logfile" | grep -o '[0-9.]*$')
        step_avg=$(grep -o 'step_avg:[0-9.]*' "$logfile" | tail -1 | cut -d: -f2)
        steps=$(grep -o 'step:[0-9]*/9000' "$logfile" | tail -1 | grep -o '^[^/]*' | cut -d: -f2)
        printf "  rope_%-3s  val_bpb=%-12s  roundtrip=%-12s  steps=%-5s  avg=%s\n" \
            "$dims" "$val_bpb" "${roundtrip:-N/A}" "$steps" "${step_avg}ms"
    fi
done
echo ""
echo "Compare against overnight rope_16: val_bpb=1.40835665 at step 1610 (2400s)"
echo "For fair comparison, check val_bpb at step ~800 across all variants."
echo "Done at $(date)"
