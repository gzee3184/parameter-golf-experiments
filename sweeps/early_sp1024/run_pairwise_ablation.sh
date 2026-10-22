#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TRAIN_SCRIPT="$SCRIPT_DIR/2026-03-27_ActivationSearch/train_gpt.py"
RESULTS_DIR="$SCRIPT_DIR/pairwise_ablation_results"
VENV="${VENV:-./venv}"

mkdir -p "$RESULTS_DIR"
source "$VENV/bin/activate"

COMMON_VARS=(
    SEED=1337
    NUM_LAYERS=11 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=3.0
    BIGRAM_VOCAB_SIZE=1536 BIGRAM_DIM=128
    ROPE_DIMS=16 LN_SCALE=1
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
    shift 3
    local extra_vars=("$@")
    local logfile="$RESULTS_DIR/${name}.log"

    if [[ -f "$logfile" ]] && grep -q "final_int6_roundtrip_exact" "$logfile"; then
        echo "[$(date '+%H:%M:%S')] SKIP $name (already done)"
        return 0
    fi

    echo "[$(date '+%H:%M:%S')] START $name on GPU$gpu → $logfile"
    local t0=$SECONDS

    env "${COMMON_VARS[@]}" "${extra_vars[@]}" \
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
echo "  Pairwise Ablation: mish2 + one change"
echo "  $(date)"
echo "============================================="
echo "  GPU0: mish2 + XSA=6  (MTP off, XSA_LAST_N=6)"
echo "  GPU1: mish2 + MTP    (XSA=4, MTP 1h w=0.05)"
echo ""

run_variant "mish2_xsa6" 0 29500 \
    MLP_ACTIVATION=mish2 XSA_LAST_N=6 MTP_NUM_HEADS=0 &
PID1=$!

run_variant "mish2_mtp" 1 29501 \
    MLP_ACTIVATION=mish2 XSA_LAST_N=4 MTP_NUM_HEADS=1 MTP_LOSS_WEIGHT=0.05 &
PID2=$!

wait $PID1
wait $PID2

echo ""
echo "============================================="
echo "  RESULTS — compare against mish2-only"
echo "============================================="
echo "  mish2-only (prior run): sliding_bpb=1.3652, steps=1673, avg=1435ms"
echo ""

for name in mish2_xsa6 mish2_mtp; do
    logfile="$RESULTS_DIR/${name}.log"
    if [[ -f "$logfile" ]]; then
        val_bpb=$(grep "step:.*val_loss" "$logfile" | tail -1 | grep -o 'val_bpb:[0-9.]*' | cut -d: -f2)
        sliding=$(grep "final_int6_sliding_window_exact" "$logfile" | grep -o 'val_bpb:[0-9.]*' | cut -d: -f2)
        roundtrip=$(grep "final_int6_roundtrip_exact" "$logfile" | grep -o 'val_bpb:[0-9.]*' | cut -d: -f2)
        step_avg=$(grep -o 'step_avg:[0-9.]*' "$logfile" | tail -1 | cut -d: -f2)
        steps=$(grep "step:.*val_loss" "$logfile" | tail -1 | grep -o 'step:[0-9]*' | cut -d: -f2)
        printf "  %-15s  train_bpb=%-10s  sliding=%-10s  roundtrip=%-10s  steps=%-5s  avg=%s\n" \
            "$name" "$val_bpb" "${sliding:-pending}" "${roundtrip:-pending}" "$steps" "${step_avg}ms"
    fi
done

echo ""
echo "Done at $(date)"
