#!/bin/bash
# Run aryanbhosale's PR #1326 locally on GPU 1 for 30 min as reference
set -uo pipefail

export CUDA_VISIBLE_DEVICES=1
export DATA_DIR="./data/"
export MAX_WALLCLOCK_SECONDS=1800
export SEED=1337
export TRAIN_LOG_EVERY=100
export VAL_LOSS_EVERY=4000
# Use their defaults for everything else (already in the code)
# TTT enabled for comparison
export TTT_ENABLED=1

cd ${PROJECT_ROOT:-.}/parameter-golf
echo "$(date) ── PR #1326 reference run (30min, GPU 1) ──"
PYTHONUNBUFFERED=1 nice -n 10 torchrun --standalone --nproc_per_node=1 \
    ${PROJECT_ROOT:-.}/competitor_code/pr1326_decompressed.py 2>&1 | tee ${PROJECT_ROOT:-.}/h100_deploy/pr1326_test.log
echo "$(date) ── COMPLETE ──"
