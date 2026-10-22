#!/usr/bin/env bash
# ============================================================================
# Overnight Local Runner — 2 GPUs, Phase 1 screening
# ============================================================================
# Runs all Phase 1 experiments (A/C/D/E) across 2 GPUs in parallel.
# Each experiment variant gets ~1690 steps (~40 min) — enough to compare.
#
# Resource-conscious: nice'd, 1 process per GPU, no more than 2 concurrent.
# Idempotent: completed runs are skipped.
#
# Usage (from parameter-golf/):
#   tmux new -s overnight
#   bash records/track_10min_16mb/run_overnight.sh
#   # Then Ctrl-B D to detach. Reconnect with: tmux attach -t overnight
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
source "$SCRIPT_DIR/sweep_common.sh"

RESULTS_DIR="$SCRIPT_DIR/overnight_results"
mkdir -p "$RESULTS_DIR"

# --- Local overrides ---
export NPROC=1
LOCAL_WALLCLOCK=2400  # 40 min per run (~1690 steps at 1420ms/step)

SCREEN_VARS=(SEED=1337 MAX_WALLCLOCK_SECONDS=$LOCAL_WALLCLOCK TTT_ENABLED=0 VAL_LOSS_EVERY=400 TRAIN_LOG_EVERY=50)

# ============================================================================
# Job definitions: (label, logfile, extra_vars...)
# ============================================================================
# We write jobs to two FIFO queues, one per GPU.
# Jobs are balanced: ~9 per GPU queue.

GPU0_JOBS=()
GPU1_JOBS=()

add_job() {
    local gpu="$1"; shift
    if [ "$gpu" = "0" ]; then
        GPU0_JOBS+=("$*")
    else
        GPU1_JOBS+=("$*")
    fi
}

# Exp A: Activation Search (5 runs) — uses custom train_gpt.py
A_SCRIPT="$SCRIPT_DIR/2026-03-27_ActivationSearch/train_gpt.py"
A_DIR="$RESULTS_DIR/exp_a"
mkdir -p "$A_DIR"
add_job 0 "leaky05|$A_DIR/leaky05.log|$A_SCRIPT|MLP_ACTIVATION=leaky05"
add_job 1 "silu2|$A_DIR/silu2.log|$A_SCRIPT|MLP_ACTIVATION=silu2"
add_job 0 "gelu2|$A_DIR/gelu2.log|$A_SCRIPT|MLP_ACTIVATION=gelu2"
add_job 1 "leaky03|$A_DIR/leaky03.log|$A_SCRIPT|MLP_ACTIVATION=leaky03"
add_job 0 "leaky07|$A_DIR/leaky07.log|$A_SCRIPT|MLP_ACTIVATION=leaky07"

# Exp C: XSA Sweep (4 runs) — uses base train_gpt.py
C_DIR="$RESULTS_DIR/exp_c"
mkdir -p "$C_DIR"
add_job 1 "xsa_4|$C_DIR/xsa4.log|$BASE_TRAIN_SCRIPT|XSA_LAST_N=4"
add_job 0 "xsa_6|$C_DIR/xsa6.log|$BASE_TRAIN_SCRIPT|XSA_LAST_N=6"
add_job 1 "xsa_8|$C_DIR/xsa8.log|$BASE_TRAIN_SCRIPT|XSA_LAST_N=8"
add_job 0 "xsa_11|$C_DIR/xsa11.log|$BASE_TRAIN_SCRIPT|XSA_LAST_N=11"

# Exp D: MTP Sweep (4 runs)
D_DIR="$RESULTS_DIR/exp_d"
mkdir -p "$D_DIR"
add_job 1 "mtp_off|$D_DIR/mtp_off.log|$BASE_TRAIN_SCRIPT|MTP_NUM_HEADS=0"
add_job 0 "mtp_1h_w05|$D_DIR/mtp_1h_w05.log|$BASE_TRAIN_SCRIPT|MTP_NUM_HEADS=1 MTP_LOSS_WEIGHT=0.05"
add_job 1 "mtp_1h_w10|$D_DIR/mtp_1h_w10.log|$BASE_TRAIN_SCRIPT|MTP_NUM_HEADS=1 MTP_LOSS_WEIGHT=0.10"
add_job 0 "mtp_2h_w05|$D_DIR/mtp_2h_w05.log|$BASE_TRAIN_SCRIPT|MTP_NUM_HEADS=2 MTP_LOSS_WEIGHT=0.05"

# Exp E: RoPE Sweep (4 runs)
E_DIR="$RESULTS_DIR/exp_e"
mkdir -p "$E_DIR"
add_job 1 "rope_16|$E_DIR/rope16.log|$BASE_TRAIN_SCRIPT|ROPE_DIMS=16"
add_job 0 "rope_8|$E_DIR/rope8.log|$BASE_TRAIN_SCRIPT|ROPE_DIMS=8"
add_job 1 "rope_24|$E_DIR/rope24.log|$BASE_TRAIN_SCRIPT|ROPE_DIMS=24"
add_job 0 "rope_32|$E_DIR/rope32.log|$BASE_TRAIN_SCRIPT|ROPE_DIMS=32"

# ============================================================================
# Runner
# ============================================================================
run_gpu_queue() {
    local gpu_id="$1"
    shift
    local jobs=("$@")

    for job_str in "${jobs[@]}"; do
        IFS='|' read -r label logfile script extra_vars <<< "$job_str"

        if [ -f "$logfile" ] && grep -q "stopping_early\|step:9000" "$logfile"; then
            log "[GPU$gpu_id] SKIP $label — already completed"
            continue
        fi

        log "[GPU$gpu_id] START $label → $logfile"
        local t_start=$SECONDS

        # shellcheck disable=SC2086
        nice -n 10 env CUDA_VISIBLE_DEVICES="$gpu_id" \
            "${COMMON_VARS[@]}" ${extra_vars} "${SCREEN_VARS[@]}" \
            torchrun --standalone --nproc_per_node=1 \
            --master_port=$((29500 + gpu_id)) \
            "$script" \
            > "$logfile" 2>&1
        local rc=$?
        local elapsed=$(( SECONDS - t_start ))

        local bpb
        bpb=$(extract_last_val_bpb "$logfile" 2>/dev/null || echo "N/A")
        local sa
        sa=$(extract_step_avg "$logfile" 2>/dev/null || echo "N/A")
        log "[GPU$gpu_id] DONE  $label — val_bpb=$bpb step_avg=${sa}ms elapsed=${elapsed}s exit=$rc"
    done
}

log "=============================================="
log "  OVERNIGHT LOCAL RUN — Phase 1 Screening"
log "  GPUs: 0, 1 (1 process each, nice=10)"
log "  Wallclock: ${LOCAL_WALLCLOCK}s per run"
log "  Jobs: GPU0=${#GPU0_JOBS[@]}, GPU1=${#GPU1_JOBS[@]}"
log "  Estimated total: ~$(( (${#GPU0_JOBS[@]} > ${#GPU1_JOBS[@]} ? ${#GPU0_JOBS[@]} : ${#GPU1_JOBS[@]}) * LOCAL_WALLCLOCK / 60 )) min"
log "=============================================="

# Launch both GPU queues in parallel
run_gpu_queue 0 "${GPU0_JOBS[@]}" &
PID0=$!
run_gpu_queue 1 "${GPU1_JOBS[@]}" &
PID1=$!

log "GPU0 queue PID: $PID0"
log "GPU1 queue PID: $PID1"

# Wait for both
wait $PID0
RC0=$?
wait $PID1
RC1=$?

log ""
log "=============================================="
log "  ALL JOBS COMPLETE (GPU0 exit=$RC0, GPU1 exit=$RC1)"
log "=============================================="

# ============================================================================
# Print summary table
# ============================================================================
log ""
log "========== OVERNIGHT RESULTS SUMMARY =========="

log ""
log "--- Exp A: Activation Search ---"
print_results_table extract_last_val_bpb \
    "leaky05:$A_DIR/leaky05.log" \
    "silu2:$A_DIR/silu2.log" \
    "gelu2:$A_DIR/gelu2.log" \
    "leaky03:$A_DIR/leaky03.log" \
    "leaky07:$A_DIR/leaky07.log"

log ""
log "--- Exp C: XSA Sweep ---"
print_results_table extract_last_val_bpb \
    "xsa_4:$C_DIR/xsa4.log" \
    "xsa_6:$C_DIR/xsa6.log" \
    "xsa_8:$C_DIR/xsa8.log" \
    "xsa_11:$C_DIR/xsa11.log"

log ""
log "--- Exp D: MTP Sweep ---"
print_results_table extract_last_val_bpb \
    "mtp_off:$D_DIR/mtp_off.log" \
    "mtp_1h_w05:$D_DIR/mtp_1h_w05.log" \
    "mtp_1h_w10:$D_DIR/mtp_1h_w10.log" \
    "mtp_2h_w05:$D_DIR/mtp_2h_w05.log"

log ""
log "--- Exp E: RoPE Sweep ---"
print_results_table extract_last_val_bpb \
    "rope_16:$E_DIR/rope16.log" \
    "rope_8:$E_DIR/rope8.log" \
    "rope_24:$E_DIR/rope24.log" \
    "rope_32:$E_DIR/rope32.log"

log ""
log "Results saved in: $RESULTS_DIR/"
log "To view: cat $RESULTS_DIR/exp_*/\*.log | grep 'val_bpb:' | tail"
