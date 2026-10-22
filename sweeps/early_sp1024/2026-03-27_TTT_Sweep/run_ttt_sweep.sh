#!/usr/bin/env bash
# ============================================================================
# Experiment B: TTT Protocol Optimization
# ============================================================================
# Tests chunk size, epoch count, and block freezing.
# All variants share identical training — only TTT eval differs.
# Phase 1: seed 1337, full 600s + TTT for all variants (~12 min × 7 = ~84 min)
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$(dirname "$SCRIPT_DIR")/sweep_common.sh"

TRAIN_SCRIPT="$BASE_TRAIN_SCRIPT"
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

RUN_VARS=(SEED=1337 MAX_WALLCLOCK_SECONDS=600 VAL_LOSS_EVERY=2000 TRAIN_LOG_EVERY=500)

log "========== EXPERIMENT B: TTT Protocol Sweep =========="

run_variant "baseline_ttt" "$LOG_DIR/baseline_seed1337.log"  "${TTT_VARS[@]}" "${RUN_VARS[@]}"
run_variant "chunk_16k"    "$LOG_DIR/chunk16k_seed1337.log"  "${TTT_VARS[@]}" TTT_CHUNK_TOKENS=16384 "${RUN_VARS[@]}"
run_variant "chunk_64k"    "$LOG_DIR/chunk64k_seed1337.log"  "${TTT_VARS[@]}" TTT_CHUNK_TOKENS=65536 "${RUN_VARS[@]}"
run_variant "epochs_5"     "$LOG_DIR/epochs5_seed1337.log"   "${TTT_VARS[@]}" TTT_EPOCHS=5 "${RUN_VARS[@]}"
run_variant "freeze_2"     "$LOG_DIR/freeze2_seed1337.log"   "${TTT_VARS[@]}" TTT_FREEZE_BLOCKS=2 "${RUN_VARS[@]}"
run_variant "freeze_4"     "$LOG_DIR/freeze4_seed1337.log"   "${TTT_VARS[@]}" TTT_FREEZE_BLOCKS=4 "${RUN_VARS[@]}"

log "========== RESULTS =========="
print_results_table extract_ttt_bpb \
    "baseline:$LOG_DIR/baseline_seed1337.log" \
    "chunk_16k:$LOG_DIR/chunk16k_seed1337.log" \
    "chunk_64k:$LOG_DIR/chunk64k_seed1337.log" \
    "epochs_5:$LOG_DIR/epochs5_seed1337.log" \
    "freeze_2:$LOG_DIR/freeze2_seed1337.log" \
    "freeze_4:$LOG_DIR/freeze4_seed1337.log"

log ""
log "TTT Timing Check (must be < 490s):"
for f in "$LOG_DIR"/*_seed1337.log; do
    label=$(basename "$f" _seed1337.log)
    ttt_time=$(extract_ttt_time "$f" 2>/dev/null || echo "N/A")
    printf "  %-15s %sms\n" "$label" "${ttt_time:-N/A}"
done

log "========== EXPERIMENT B COMPLETE =========="
