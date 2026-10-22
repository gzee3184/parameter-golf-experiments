#!/usr/bin/env bash
# ============================================================================
# Experiment E: Partial RoPE Dimension Sweep
# ============================================================================
# Tests how many of 64 head dims get rotary encoding.
# Current: ROPE_DIMS=16. Tests: 8, 24, 32.
# Phase 1: 300s screen, no TTT, seed 1337 (~5 min × 4 = ~20 min)
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$(dirname "$SCRIPT_DIR")/sweep_common.sh"

TRAIN_SCRIPT="$BASE_TRAIN_SCRIPT"
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

SCREEN_VARS=(SEED=1337 MAX_WALLCLOCK_SECONDS=300 TTT_ENABLED=0 VAL_LOSS_EVERY=500 TRAIN_LOG_EVERY=100)

log "========== EXPERIMENT E: Partial RoPE Sweep =========="

for rd in 16 8 24 32; do
    run_variant "rope_$rd" "$LOG_DIR/rope${rd}_seed1337.log" ROPE_DIMS="$rd" "${SCREEN_VARS[@]}"
done

log "========== RESULTS =========="
print_results_table extract_last_val_bpb \
    "rope_16:$LOG_DIR/rope16_seed1337.log" \
    "rope_8:$LOG_DIR/rope8_seed1337.log" \
    "rope_24:$LOG_DIR/rope24_seed1337.log" \
    "rope_32:$LOG_DIR/rope32_seed1337.log"

log "========== EXPERIMENT E COMPLETE =========="
