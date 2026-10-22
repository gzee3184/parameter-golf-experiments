#!/usr/bin/env bash
# ============================================================================
# Experiment C: Extended XSA Depth Sweep
# ============================================================================
# XSA adds no parameters. Tests XSA_LAST_N = 4 (baseline), 6, 8, 11.
# Phase 1: 300s screen, no TTT, seed 1337 (~5 min × 4 = ~20 min)
# Check step_avg — XSA adds compute per layer, must stay < 88ms.
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$(dirname "$SCRIPT_DIR")/sweep_common.sh"

TRAIN_SCRIPT="$BASE_TRAIN_SCRIPT"
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

SCREEN_VARS=(SEED=1337 MAX_WALLCLOCK_SECONDS=300 TTT_ENABLED=0 VAL_LOSS_EVERY=500 TRAIN_LOG_EVERY=100)

log "========== EXPERIMENT C: XSA Depth Sweep =========="

for n in 4 6 8 11; do
    run_variant "xsa_$n" "$LOG_DIR/phase1_xsa${n}_seed1337.log" XSA_LAST_N="$n" "${SCREEN_VARS[@]}"
done

log "========== RESULTS =========="
print_results_table extract_last_val_bpb \
    "xsa_4:$LOG_DIR/phase1_xsa4_seed1337.log" \
    "xsa_6:$LOG_DIR/phase1_xsa6_seed1337.log" \
    "xsa_8:$LOG_DIR/phase1_xsa8_seed1337.log" \
    "xsa_11:$LOG_DIR/phase1_xsa11_seed1337.log"

log ""
log "Step Time Check (must be < 88ms on H100):"
for n in 4 6 8 11; do
    sa=$(extract_step_avg "$LOG_DIR/phase1_xsa${n}_seed1337.log" 2>/dev/null || echo "N/A")
    printf "  XSA_LAST_N=%-3s step_avg=%sms\n" "$n" "${sa:-N/A}"
done

log "========== EXPERIMENT C COMPLETE =========="
