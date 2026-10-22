#!/usr/bin/env bash
# ============================================================================
# Experiment D: MTP (Multi-Token Prediction) Auxiliary Loss
# ============================================================================
# MTP heads stripped at export — free training signal, zero artifact cost.
# Phase 1: 300s screen, no TTT, seed 1337 (~5 min × 4 = ~20 min)
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$(dirname "$SCRIPT_DIR")/sweep_common.sh"

TRAIN_SCRIPT="$BASE_TRAIN_SCRIPT"
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

SCREEN_VARS=(SEED=1337 MAX_WALLCLOCK_SECONDS=300 TTT_ENABLED=0 VAL_LOSS_EVERY=500 TRAIN_LOG_EVERY=100)

log "========== EXPERIMENT D: MTP Sweep =========="

run_variant "mtp_off"     "$LOG_DIR/mtp_off_seed1337.log"     MTP_NUM_HEADS=0 "${SCREEN_VARS[@]}"
run_variant "mtp_1h_w05"  "$LOG_DIR/mtp_1h_w05_seed1337.log"  MTP_NUM_HEADS=1 MTP_LOSS_WEIGHT=0.05 "${SCREEN_VARS[@]}"
run_variant "mtp_1h_w10"  "$LOG_DIR/mtp_1h_w10_seed1337.log"  MTP_NUM_HEADS=1 MTP_LOSS_WEIGHT=0.10 "${SCREEN_VARS[@]}"
run_variant "mtp_2h_w05"  "$LOG_DIR/mtp_2h_w05_seed1337.log"  MTP_NUM_HEADS=2 MTP_LOSS_WEIGHT=0.05 "${SCREEN_VARS[@]}"

log "========== RESULTS =========="
print_results_table extract_last_val_bpb \
    "mtp_off:$LOG_DIR/mtp_off_seed1337.log" \
    "mtp_1h_w05:$LOG_DIR/mtp_1h_w05_seed1337.log" \
    "mtp_1h_w10:$LOG_DIR/mtp_1h_w10_seed1337.log" \
    "mtp_2h_w05:$LOG_DIR/mtp_2h_w05_seed1337.log"

log "========== EXPERIMENT D COMPLETE =========="
