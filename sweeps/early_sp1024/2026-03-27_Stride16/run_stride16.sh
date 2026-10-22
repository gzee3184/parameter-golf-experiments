#!/usr/bin/env bash
# ============================================================================
# Experiment G: Stride-16 Evaluation
# ============================================================================
# Stride-16 gives 4x more context windows than stride-64.
# CRITICAL: Never run TTT itself at stride-16 (~4x over budget).
#
# Step 1: Baseline stride=64 + TTT
# Step 2: Stride-16 without TTT (timing check)
# Step 3: If stride-16 eval < 200s, run stride-16 with TTT (TTT uses stride=64 internally)
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$(dirname "$SCRIPT_DIR")/sweep_common.sh"

TRAIN_SCRIPT="$BASE_TRAIN_SCRIPT"
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

RUN_VARS=(SEED=1337 MAX_WALLCLOCK_SECONDS=600 VAL_LOSS_EVERY=2000 TRAIN_LOG_EVERY=500)

log "========== EXPERIMENT G: Stride-16 Evaluation =========="

run_variant "stride64_ttt"    "$LOG_DIR/stride64_ttt_seed1337.log"    "${TTT_VARS[@]}" EVAL_STRIDE=64 "${RUN_VARS[@]}"
run_variant "stride16_no_ttt" "$LOG_DIR/stride16_nottt_seed1337.log"  TTT_ENABLED=0 EVAL_STRIDE=16 "${RUN_VARS[@]}"

# Check stride-16 eval timing from the sliding window eval line
if [ -f "$LOG_DIR/stride16_nottt_seed1337.log" ]; then
    eval_time_ms=$(grep -oP 'final_int6_sliding_window .*eval_time:\K[0-9.]+' "$LOG_DIR/stride16_nottt_seed1337.log" | head -1 || echo "999999")
    eval_sec=$(( ${eval_time_ms%%.*} / 1000 ))
    log "Stride-16 sliding window eval time: ${eval_sec}s (${eval_time_ms}ms)"

    if [ "${eval_time_ms%%.*}" -lt 200000 ] 2>/dev/null; then
        log "Stride-16 eval < 200s — safe to combine with TTT"
        # TTT still runs at its own stride internally; EVAL_STRIDE=16 only affects sliding window eval
        run_variant "stride16_ttt" "$LOG_DIR/stride16_ttt_seed1337.log" "${TTT_VARS[@]}" EVAL_STRIDE=16 "${RUN_VARS[@]}"
    else
        log "WARNING: Stride-16 eval took ${eval_sec}s — too slow to combine with TTT"
    fi
fi

log "========== RESULTS =========="
print_results_table extract_sliding_bpb \
    "stride64+ttt:$LOG_DIR/stride64_ttt_seed1337.log" \
    "stride16_no_ttt:$LOG_DIR/stride16_nottt_seed1337.log"

if [ -f "$LOG_DIR/stride16_ttt_seed1337.log" ]; then
    ttt_bpb=$(extract_ttt_bpb "$LOG_DIR/stride16_ttt_seed1337.log" 2>/dev/null || echo "N/A")
    log "Stride-16 + TTT BPB: ${ttt_bpb:-N/A}"
fi

log "========== EXPERIMENT G COMPLETE =========="
