#!/bin/bash
# Launcher: Wait for R5 (GPU 0) and v8.5-A (GPU 1) to finish,
# kill the sequential sweep, then launch B+D and C+E in parallel.
#
# R5 python3 PID: 1618678 (GPU 0)
# v8.5-A python3 PID: 1629455 (GPU 1)
# v8.5 sweep bash PID: 1629357 (will try to start Test B after A)
set -uo pipefail

R5_PID=1618678
V85A_PID=1629455
SWEEP_BASH_PID=1629357

SCRIPT_DIR="${PROJECT_ROOT:-.}/h100_final"
LOG_DIR="${PROJECT_ROOT:-.}/local_tests/v8.5_sweep"

echo "$(date) Waiting for R5 (PID $R5_PID) and v8.5-A (PID $V85A_PID) to finish..."

# Wait for both training processes to die
while kill -0 $R5_PID 2>/dev/null || kill -0 $V85A_PID 2>/dev/null; do
    # Show which are still running
    R5_STATUS="done"
    V85A_STATUS="done"
    kill -0 $R5_PID 2>/dev/null && R5_STATUS="running"
    kill -0 $V85A_PID 2>/dev/null && V85A_STATUS="running"
    echo "$(date) R5=$R5_STATUS, v8.5-A=$V85A_STATUS — sleeping 60s"
    sleep 60
done

echo "$(date) Both R5 and v8.5-A finished!"

# Kill the sweep's process group to prevent Test B from starting
# (or kill Test B if it just started during the race)
if kill -0 $SWEEP_BASH_PID 2>/dev/null; then
    echo "$(date) Killing sweep process group (PGID $SWEEP_BASH_PID)..."
    kill -- -$SWEEP_BASH_PID 2>/dev/null || true
    sleep 3
    # Force kill if still alive
    if kill -0 $SWEEP_BASH_PID 2>/dev/null; then
        echo "$(date) Force killing sweep..."
        kill -9 -- -$SWEEP_BASH_PID 2>/dev/null || true
    fi
fi

# Also kill any stray torchrun/train_gpt processes (safety net)
pkill -f "train_gpt.py" -u $(whoami) 2>/dev/null || true
sleep 5

# Verify GPUs are free
echo "$(date) Verifying GPUs are free..."
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader

# Launch both GPU scripts in parallel
echo "$(date) ═══ Launching v8.5 B+D on GPU 0 and C+E on GPU 1 ═══"

nohup bash "$SCRIPT_DIR/run_v8.5_gpu0.sh" > "$LOG_DIR/gpu0_wrapper.log" 2>&1 &
GPU0_PID=$!
echo "$(date) GPU 0 (B+D) launched, wrapper PID: $GPU0_PID"

nohup bash "$SCRIPT_DIR/run_v8.5_gpu1.sh" > "$LOG_DIR/gpu1_wrapper.log" 2>&1 &
GPU1_PID=$!
echo "$(date) GPU 1 (C+E) launched, wrapper PID: $GPU1_PID"

echo ""
echo "$(date) Both launched! Monitor with:"
echo "  tail -f $LOG_DIR/gpu0_wrapper.log  # B+D"
echo "  tail -f $LOG_DIR/gpu1_wrapper.log  # C+E"
echo "  ls -lt $LOG_DIR/  # check for result files"
echo ""
echo "Expected completion: ~6hrs from now (2 tests × 3hr each, sequential per GPU)"
