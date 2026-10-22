#!/bin/bash
# v8 sweep: Run R2 immediately, then R3 after R2 completes
# R1 is already running separately — this chains R2+R3
set -uo pipefail

echo "$(date) Waiting for R1 to finish (checking for torchrun processes)..."
while pgrep -f "torchrun.*train_gpt" > /dev/null 2>&1; do
    sleep 60
done
echo "$(date) R1 done. Starting R2 in 10s..."
sleep 10

echo "$(date) === LAUNCHING R2 ==="
bash ${PROJECT_ROOT:-.}/h100_final/run_v8_sweep_r2.sh
echo "$(date) === R2 COMPLETE ==="

sleep 10

echo "$(date) === LAUNCHING R3 ==="
bash ${PROJECT_ROOT:-.}/h100_final/run_v8_sweep_r3.sh
echo "$(date) === R3 COMPLETE ==="

echo "$(date) === ALL SWEEPS DONE ==="
