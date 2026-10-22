#!/bin/bash
# Test: Extended eval context vs TTT variants
# Each run trains from scratch (600s) then evals with different strategies
# GPU 1 only, sequential
#
# Runs:
#   1. Baseline: EVAL_SEQ_LEN=2048, no TTT (control)
#   2. Extended: EVAL_SEQ_LEN=4096, no TTT
#   3. Extended: EVAL_SEQ_LEN=8192, no TTT
#   4. qTTT: query-only TTT with AdamW
#   5. Standard TTT (for comparison)

set -e
cd ${PROJECT_ROOT:-.}/parameter-golf
source ${VENV:-./venv}/bin/activate

SCRIPT="${PROJECT_ROOT:-.}/parameter-golf/records/track_10min_16mb/2026-03-27_ActivationSearch/train_gpt.py"
RESULTS="${PROJECT_ROOT:-.}/local_tests"

# Common config
export CUDA_VISIBLE_DEVICES=1
export MLP_ACTIVATION=mish2
export MUON_PLUS=1
export XSA_LAST_N=11
export MTP_NUM_HEADS=0
export BIGRAM_VOCAB_SIZE=3072
export BIGRAM_DIM=112
export WARMDOWN_ITERS=4000
export GPTQ_ENABLED=1
export GPTQ_CALIB_SEQS=32
export TARGET_MB=15.9
export NGRAM_CACHE=0
export SEED=1337
export MAX_WALLCLOCK_SECONDS=600

# Default TTT params (overridden per-run)
export TTT_LR=0.002
export TTT_EPOCHS=3
export TTT_FREEZE_BLOCKS=0
export TTT_MOMENTUM=0.9
export TTT_BATCH_SEQS=32
export TTT_GRAD_CLIP=1.0
export TTT_QTTT=0
export TTT_OPTIMIZER=sgd
export TTT_WEIGHT_DECAY=0.0

echo "$(date) ====== EVAL CONTEXT TEST START ======"

# --- Run 1: Baseline (2048, no TTT) ---
echo ""
echo "$(date) === RUN 1/5: EVAL_SEQ_LEN=2048, no TTT ==="
export EVAL_SEQ_LEN=2048
export TTT_ENABLED=0
nice -n 10 torchrun --standalone --nproc_per_node=1 --master_port=29501 "$SCRIPT" \
    > "$RESULTS/eval_ctx_2048_nottt.log" 2>&1
echo "$(date) === RUN 1 DONE ==="

# --- Run 2: Extended 4096, no TTT ---
echo ""
echo "$(date) === RUN 2/5: EVAL_SEQ_LEN=4096, no TTT ==="
export EVAL_SEQ_LEN=4096
export TTT_ENABLED=0
nice -n 10 torchrun --standalone --nproc_per_node=1 --master_port=29501 "$SCRIPT" \
    > "$RESULTS/eval_ctx_4096_nottt.log" 2>&1
echo "$(date) === RUN 2 DONE ==="

# --- Run 3: Extended 8192, no TTT ---
echo ""
echo "$(date) === RUN 3/5: EVAL_SEQ_LEN=8192, no TTT ==="
export EVAL_SEQ_LEN=8192
export TTT_ENABLED=0
nice -n 10 torchrun --standalone --nproc_per_node=1 --master_port=29501 "$SCRIPT" \
    > "$RESULTS/eval_ctx_8192_nottt.log" 2>&1
echo "$(date) === RUN 3 DONE ==="

# --- Run 4: qTTT (query-only, AdamW) ---
echo ""
echo "$(date) === RUN 4/5: qTTT (query-only, AdamW) ==="
export EVAL_SEQ_LEN=2048
export TTT_ENABLED=1
export TTT_QTTT=1
export TTT_OPTIMIZER=adamw
export TTT_WEIGHT_DECAY=0.01
export TTT_LR=0.001
nice -n 10 torchrun --standalone --nproc_per_node=1 --master_port=29501 "$SCRIPT" \
    > "$RESULTS/eval_ctx_2048_qttt.log" 2>&1
echo "$(date) === RUN 4 DONE ==="

# --- Run 5: Standard full TTT (reference) ---
echo ""
echo "$(date) === RUN 5/5: Standard TTT (SGD, all params) ==="
export EVAL_SEQ_LEN=2048
export TTT_ENABLED=1
export TTT_QTTT=0
export TTT_OPTIMIZER=sgd
export TTT_WEIGHT_DECAY=0.0
export TTT_LR=0.002
nice -n 10 torchrun --standalone --nproc_per_node=1 --master_port=29501 "$SCRIPT" \
    > "$RESULTS/eval_ctx_2048_ttt.log" 2>&1
echo "$(date) === RUN 5 DONE ==="

# --- Summary ---
echo ""
echo "$(date) ====== RESULTS ======"
echo ""
for f in "$RESULTS/eval_ctx_2048_nottt.log" \
         "$RESULTS/eval_ctx_4096_nottt.log" \
         "$RESULTS/eval_ctx_8192_nottt.log" \
         "$RESULTS/eval_ctx_2048_qttt.log" \
         "$RESULTS/eval_ctx_2048_ttt.log"; do
    name=$(basename "$f" .log)
    steps=$(grep "stopping_early" "$f" | grep -oP 'step:\K[0-9]+' 2>/dev/null || echo "?")
    avg=$(grep "stopping_early" "$f" | grep -oP 'step_avg:\K[0-9.]+' 2>/dev/null || echo "?")
    pre=$(grep "^step:.*val_bpb" "$f" | tail -1 | grep -oP 'val_bpb:\K[0-9.]+' 2>/dev/null || echo "?")
    sliding=$(grep "final_int6_sliding_window_exact" "$f" | grep -oP 'val_bpb:\K[0-9.]+' 2>/dev/null || echo "?")
    ttt=$(grep "legal_ttt_exact" "$f" | grep -oP 'val_bpb:\K[0-9.]+' 2>/dev/null || echo "?")
    echo "  $name: steps=$steps avg=${avg}ms pre=$pre sliding=$sliding ttt=$ttt"
done

echo ""
echo "$(date) ====== DONE ======"
