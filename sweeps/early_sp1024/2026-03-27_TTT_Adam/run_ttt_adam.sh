#!/usr/bin/env bash
# ============================================================================
# Experiment F: Adam TTT Optimizer
# ============================================================================
# Replaces SGD with Adam in TTT eval phase.
# Uses modified train_gpt.py in this directory (adds TTT_OPTIMIZER env var).
# Phase 1: seed 1337, SGD baseline vs Adam at two LRs (~12 min × 3 = ~36 min)
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$(dirname "$SCRIPT_DIR")/sweep_common.sh"

TRAIN_SCRIPT="$SCRIPT_DIR/train_gpt.py"
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

RUN_VARS=(SEED=1337 MAX_WALLCLOCK_SECONDS=600 VAL_LOSS_EVERY=2000 TRAIN_LOG_EVERY=500)

log "========== EXPERIMENT F: Adam TTT Optimizer =========="

run_variant "ttt_sgd"         "$LOG_DIR/ttt_sgd_seed1337.log"       "${TTT_VARS[@]}" TTT_OPTIMIZER=sgd "${RUN_VARS[@]}"
run_variant "ttt_adam_lr5e4"  "$LOG_DIR/ttt_adam_lr5e4_seed1337.log" "${TTT_VARS[@]}" TTT_OPTIMIZER=adam TTT_LR=0.0005 "${RUN_VARS[@]}"
run_variant "ttt_adam_lr1e3"  "$LOG_DIR/ttt_adam_lr1e3_seed1337.log" "${TTT_VARS[@]}" TTT_OPTIMIZER=adam TTT_LR=0.001 "${RUN_VARS[@]}"

log "========== RESULTS =========="
print_results_table extract_ttt_bpb \
    "ttt_sgd:$LOG_DIR/ttt_sgd_seed1337.log" \
    "ttt_adam_lr5e4:$LOG_DIR/ttt_adam_lr5e4_seed1337.log" \
    "ttt_adam_lr1e3:$LOG_DIR/ttt_adam_lr1e3_seed1337.log"

log ""
log "TTT Timing Check (must be < 490s):"
for f in "$LOG_DIR"/*_seed1337.log; do
    label=$(basename "$f" _seed1337.log)
    ttt_time=$(extract_ttt_time "$f" 2>/dev/null || echo "N/A")
    printf "  %-20s %sms\n" "$label" "${ttt_time:-N/A}"
done

log "========== EXPERIMENT F COMPLETE =========="
