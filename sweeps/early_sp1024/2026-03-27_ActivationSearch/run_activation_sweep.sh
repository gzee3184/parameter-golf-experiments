#!/usr/bin/env bash
# ============================================================================
# Experiment A: Activation Search — H100 Test Script
# ============================================================================
# Phase 1: 300s screen (no TTT, seed 1337) for all variants      ~30 min
# Phase 2: Full 600s + TTT for top 2 (if ≥0.003 BPB at 300s)    ~24 min
# Phase 3: 3-seed confirmation for winner (if ≥0.0005 BPB)       ~36 min
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$(dirname "$SCRIPT_DIR")/sweep_common.sh"

TRAIN_SCRIPT="$SCRIPT_DIR/train_gpt.py"
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

ACTIVATIONS=("leaky05" "silu2" "gelu2" "leaky03" "leaky07")
SCREEN_VARS=(SEED=1337 MAX_WALLCLOCK_SECONDS=300 TTT_ENABLED=0 VAL_LOSS_EVERY=500 TRAIN_LOG_EVERY=100)
FULL_VARS=(MAX_WALLCLOCK_SECONDS=600 "${TTT_VARS[@]}" VAL_LOSS_EVERY=1000 TRAIN_LOG_EVERY=200)

# ============================================================================
# PHASE 1: Quick Screen
# ============================================================================
log "========== PHASE 1: Quick Screen (300s, no TTT) =========="

declare -A PHASE1_BPB
for act in "${ACTIVATIONS[@]}"; do
    logfile="$LOG_DIR/phase1_${act}_seed1337.log"
    run_variant "$act" "$logfile" MLP_ACTIVATION="$act" "${SCREEN_VARS[@]}"
    PHASE1_BPB[$act]=$(extract_last_val_bpb "$logfile" 2>/dev/null || echo "")
done

log "========== PHASE 1 RESULTS =========="
entries=()
for act in "${ACTIVATIONS[@]}"; do
    entries+=("$act:$LOG_DIR/phase1_${act}_seed1337.log")
done
print_results_table extract_last_val_bpb "${entries[@]}"

BASELINE_BPB="${PHASE1_BPB[leaky05]}"
if [ -z "$BASELINE_BPB" ]; then
    log "ERROR: No baseline BPB — cannot proceed"
    exit 1
fi

# Select top 2 candidates that improve ≥0.003 over baseline
PHASE2_CANDIDATES=()
for act in "${ACTIVATIONS[@]}"; do
    [ "$act" = "leaky05" ] && continue
    bpb="${PHASE1_BPB[$act]}"
    [ -z "$bpb" ] && continue
    if float_ge "$(float_sub "$BASELINE_BPB" "$bpb")" 0.003; then
        PHASE2_CANDIDATES+=("$act")
    fi
done
# Sort by BPB and take top 2
if [ ${#PHASE2_CANDIDATES[@]} -gt 2 ]; then
    PHASE2_CANDIDATES=($(for act in "${PHASE2_CANDIDATES[@]}"; do echo "$act ${PHASE1_BPB[$act]}"; done | sort -k2 -n | head -2 | awk '{print $1}'))
fi

if [ ${#PHASE2_CANDIDATES[@]} -eq 0 ]; then
    log "No variants improved by ≥0.003 BPB. Baseline leaky05 remains optimal."
    log "========== SWEEP COMPLETE — NO WINNER =========="
    exit 0
fi
log "Phase 2 candidates: ${PHASE2_CANDIDATES[*]}"

# ============================================================================
# PHASE 2: Full Run (600s + TTT, seed 1337)
# ============================================================================
log "========== PHASE 2: Full Run (600s + TTT) =========="

PHASE2_ALL=("leaky05" "${PHASE2_CANDIDATES[@]}")
declare -A PHASE2_BPB

for act in "${PHASE2_ALL[@]}"; do
    logfile="$LOG_DIR/phase2_${act}_seed1337.log"
    run_variant "$act" "$logfile" MLP_ACTIVATION="$act" SEED=1337 "${FULL_VARS[@]}"
    PHASE2_BPB[$act]=$(extract_ttt_bpb "$logfile" 2>/dev/null || echo "")
done

log "========== PHASE 2 RESULTS =========="
entries=()
for act in "${PHASE2_ALL[@]}"; do
    entries+=("$act:$LOG_DIR/phase2_${act}_seed1337.log")
done
print_results_table extract_ttt_bpb "${entries[@]}"

BASELINE_TTT="${PHASE2_BPB[leaky05]}"
WINNER="" WINNER_BPB=""
for act in "${PHASE2_CANDIDATES[@]}"; do
    bpb="${PHASE2_BPB[$act]}"
    [ -z "$bpb" ] && continue
    improvement=$(float_sub "$BASELINE_TTT" "$bpb")
    if float_ge "$improvement" 0.0005; then
        if [ -z "$WINNER" ] || float_lt "$bpb" "$WINNER_BPB"; then
            WINNER="$act"
            WINNER_BPB="$bpb"
        fi
    fi
done

if [ -z "$WINNER" ]; then
    log "No variant improved by ≥0.0005 BPB with TTT. Baseline remains optimal."
    log "========== SWEEP COMPLETE — NO WINNER =========="
    exit 0
fi
log "WINNER: $WINNER (BPB: $WINNER_BPB, improvement: $(float_sub "$BASELINE_TTT" "$WINNER_BPB"))"

# ============================================================================
# PHASE 3: 3-Seed Confirmation
# ============================================================================
log "========== PHASE 3: 3-Seed Confirmation ($WINNER) =========="

declare -A PHASE3_BPB
for seed in 1337 42 2025; do
    logfile="$LOG_DIR/phase3_${WINNER}_seed${seed}.log"
    run_variant "${WINNER}_s${seed}" "$logfile" MLP_ACTIVATION="$WINNER" SEED="$seed" "${FULL_VARS[@]}"
    PHASE3_BPB[$seed]=$(extract_ttt_bpb "$logfile" 2>/dev/null || echo "")
done

log "========== FINAL RESULTS =========="
awk "BEGIN{
    split(\"${PHASE3_BPB[1337]} ${PHASE3_BPB[42]} ${PHASE3_BPB[2025]}\", a)
    sum=0; for(i=1;i<=3;i++) sum+=a[i]
    mean=sum/3
    printf \"  seed 1337: %.4f\n  seed 42:   %.4f\n  seed 2025: %.4f\n  mean:      %.4f\n  vs #1 (1.1194): %+.4f\n\", a[1], a[2], a[3], mean, 1.1194-mean
    exit(mean < 1.1194 ? 0 : 1)
}" && log "NEW RECORD! 3-seed mean beats #1 (1.1194)." \
   || log "3-seed mean does not beat #1. Consider combining with other experiments."

log "========== SWEEP COMPLETE =========="
