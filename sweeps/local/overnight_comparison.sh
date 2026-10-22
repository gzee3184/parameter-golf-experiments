#!/bin/bash
# Overnight head-to-head: Leaderboard base vs Our Best (mish2 + Muon+)
# Launches two separate scripts, one per GPU
# Expected: ~5-6 hours total

RESULTS="${PROJECT_ROOT:-.}/local_tests"
WALLCLOCK="${1:-14400}"

echo "$(date) ====== OVERNIGHT COMPARISON START ======"
echo "Wallclock per run: ${WALLCLOCK}s"
echo ""

# Launch GPU 0: Leaderboard baseline
echo "$(date) Launching GPU 0: Leaderboard baseline (leaky05, no Muon+)..."
bash "$RESULTS/run_gpu0.sh" "$WALLCLOCK" > "$RESULTS/comparison_leaderboard.log" 2>&1 &
PID_LB=$!
echo "  PID: $PID_LB"

sleep 10

# Launch GPU 1: Our best
echo "$(date) Launching GPU 1: Our best (mish2 + Muon+)..."
bash "$RESULTS/run_gpu1.sh" "$WALLCLOCK" > "$RESULTS/comparison_ours.log" 2>&1 &
PID_OURS=$!
echo "  PID: $PID_OURS"

echo ""
echo "$(date) Both runs launched."
echo "  Monitor GPU 0: tail -f $RESULTS/comparison_leaderboard.log"
echo "  Monitor GPU 1: tail -f $RESULTS/comparison_ours.log"
echo ""

# Wait
wait $PID_LB
RC_LB=$?
echo "$(date) GPU 0 (leaderboard) finished — exit code $RC_LB"

wait $PID_OURS
RC_OURS=$?
echo "$(date) GPU 1 (ours) finished — exit code $RC_OURS"

# Summary
echo ""
echo "$(date) ====== RESULTS SUMMARY ======"
echo ""

for entry in "LEADERBOARD(leaky05,noMuon+):$RESULTS/comparison_leaderboard.log" \
             "OURS(mish2+Muon+):$RESULTS/comparison_ours.log"; do
    label="${entry%%:*}"
    f="${entry##*:}"
    [ ! -f "$f" ] && echo "$label: LOG MISSING" && continue

    steps=$(grep "stopping_early" "$f" | grep -oP 'step:\K[0-9]+' 2>/dev/null || echo "?")
    pre_quant=$(grep "^step:.*val_bpb" "$f" | tail -1 | grep -oP 'val_bpb:\K[0-9.]+' 2>/dev/null || echo "?")
    post_ema=$(grep "DIAGNOSTIC post_ema" "$f" | grep -oP 'val_bpb:\K[0-9.]+' 2>/dev/null || echo "?")
    sliding=$(grep "final_int6_sliding_window_exact" "$f" | grep -oP 'val_bpb:\K[0-9.]+' 2>/dev/null || echo "?")
    ttt=$(grep "legal_ttt_exact" "$f" | grep -oP 'val_bpb:\K[0-9.]+' 2>/dev/null || echo "?")
    ttt_time=$(grep "ttt_sliding:done" "$f" | grep -oP 'elapsed=\K[0-9.]+' 2>/dev/null || echo "?")
    artifact=$(grep "Total submission size" "$f" | grep -oP ': \K[0-9]+' 2>/dev/null || echo "?")

    echo "  $label"
    echo "    steps=$steps  pre_quant=$pre_quant  post_ema=$post_ema"
    echo "    sliding=$sliding  ttt=$ttt  ttt_time=${ttt_time}s"
    echo "    artifact=$artifact bytes"
    echo ""
done

echo "Reference: Leaderboard #1 = 1.1147 BPB (8xH100)"
echo "Reference: Leaderboard #2 = 1.1194 BPB (8xH100)"
echo "$(date) ====== DONE ======"
