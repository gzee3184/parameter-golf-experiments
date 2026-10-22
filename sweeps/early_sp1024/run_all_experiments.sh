#!/usr/bin/env bash
# ============================================================================
# Master Experiment Runner — Parameter Golf
# ============================================================================
# Phase 1 (independent):  A, C, D, E  (~90 min)
# Phase 2 (sequential):   B, F, G     (~130 min)
# Total: ~3.5 hours on 8×H100 (~$35–42)
#
# Usage:
#   cd parameter-golf/
#   bash records/track_10min_16mb/run_all_experiments.sh [--phase1-only]
# ============================================================================
set -euo pipefail

BASE="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
source "$BASE/sweep_common.sh"

PHASE1_ONLY="${1:-}"

run_experiment() {
    local name="$1" script="$2"
    log ">>> $name"
    if bash "$script" 2>&1 | tee "$script.out"; then
        log "<<< $name completed"
    else
        log "<<< $name FAILED (exit=${PIPESTATUS[0]})"
    fi
    echo ""
}

log "=============================================="
log "  PARAMETER GOLF — FULL EXPERIMENT SUITE"
log "=============================================="

log "===== PHASE 1: Independent Experiments ====="
run_experiment "Exp A: Activation Search" "$BASE/2026-03-27_ActivationSearch/run_activation_sweep.sh"
run_experiment "Exp C: XSA Depth Sweep"   "$BASE/2026-03-27_XSA_Sweep/run_xsa_sweep.sh"
run_experiment "Exp D: MTP Sweep"          "$BASE/2026-03-27_MTP_Sweep/run_mtp_sweep.sh"
run_experiment "Exp E: RoPE Sweep"         "$BASE/2026-03-27_RoPE_Sweep/run_rope_sweep.sh"
log "===== PHASE 1 COMPLETE ====="

if [ "$PHASE1_ONLY" = "--phase1-only" ]; then
    log "Stopping after Phase 1 (--phase1-only)."
    exit 0
fi

log "===== PHASE 2: TTT & Eval Experiments ====="
run_experiment "Exp B: TTT Protocol Sweep" "$BASE/2026-03-27_TTT_Sweep/run_ttt_sweep.sh"
run_experiment "Exp F: TTT Adam Optimizer" "$BASE/2026-03-27_TTT_Adam/run_ttt_adam.sh"
run_experiment "Exp G: Stride-16 Eval"     "$BASE/2026-03-27_Stride16/run_stride16.sh"

log "===== ALL EXPERIMENTS COMPLETE ====="
log "Review results, combine best settings, run 3-seed confirmation."
