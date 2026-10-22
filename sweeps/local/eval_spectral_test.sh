#!/bin/bash
# Test: Spectral surgery TTT variants
# Run AFTER eval_context_test.sh completes (or on a separate GPU)
# GPU 1 only

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
export EVAL_SEQ_LEN=2048

# TTT defaults
export TTT_ENABLED=1
export TTT_EPOCHS=3
export TTT_FREEZE_BLOCKS=0
export TTT_BATCH_SEQS=32
export TTT_GRAD_CLIP=1.0

echo "$(date) ====== SPECTRAL SURGERY TEST START ======"

# --- Run 1: Spectral surgery k=32, full params, AdamW ---
echo ""
echo "$(date) === RUN 1/3: Spectral k=32, AdamW, all params ==="
export TTT_QTTT=0
export TTT_OPTIMIZER=adamw
export TTT_LR=0.001
export TTT_WEIGHT_DECAY=0.01
export TTT_MOMENTUM=0.9
export TTT_SPECTRAL_K=32
nice -n 10 torchrun --standalone --nproc_per_node=1 --master_port=29501 "$SCRIPT" \
    > "$RESULTS/eval_spectral_k32_adamw.log" 2>&1
echo "$(date) === RUN 1 DONE ==="

# --- Run 2: Spectral surgery k=64, qTTT (Q-only), AdamW ---
echo ""
echo "$(date) === RUN 2/3: Spectral k=64, qTTT, AdamW ==="
export TTT_QTTT=1
export TTT_OPTIMIZER=adamw
export TTT_LR=0.001
export TTT_WEIGHT_DECAY=0.01
export TTT_SPECTRAL_K=64
nice -n 10 torchrun --standalone --nproc_per_node=1 --master_port=29501 "$SCRIPT" \
    > "$RESULTS/eval_spectral_k64_qttt.log" 2>&1
echo "$(date) === RUN 2 DONE ==="

# --- Run 3: Spectral surgery k=128, full params, SGD no momentum ---
echo ""
echo "$(date) === RUN 3/3: Spectral k=128, SGD (no momentum), all params ==="
export TTT_QTTT=0
export TTT_OPTIMIZER=sgd
export TTT_LR=0.002
export TTT_WEIGHT_DECAY=0.0
export TTT_MOMENTUM=0.0
export TTT_SPECTRAL_K=128
nice -n 10 torchrun --standalone --nproc_per_node=1 --master_port=29501 "$SCRIPT" \
    > "$RESULTS/eval_spectral_k128_sgd.log" 2>&1
echo "$(date) === RUN 3 DONE ==="

# --- Summary ---
echo ""
echo "$(date) ====== RESULTS ======"
echo ""
for f in "$RESULTS/eval_spectral_k32_adamw.log" \
         "$RESULTS/eval_spectral_k64_qttt.log" \
         "$RESULTS/eval_spectral_k128_sgd.log"; do
    name=$(basename "$f" .log)
    steps=$(grep "stopping_early" "$f" | grep -oP 'step:\K[0-9]+' 2>/dev/null || echo "?")
    avg=$(grep "stopping_early" "$f" | grep -oP 'step_avg:\K[0-9.]+' 2>/dev/null || echo "?")
    sliding=$(grep "final_int6_sliding_window_exact" "$f" | grep -oP 'val_bpb:\K[0-9.]+' 2>/dev/null || echo "?")
    ttt=$(grep "legal_ttt_exact" "$f" | grep -oP 'val_bpb:\K[0-9.]+' 2>/dev/null || echo "?")
    echo "  $name: steps=$steps avg=${avg}ms sliding=$sliding ttt=$ttt"
done

echo ""
echo "$(date) ====== DONE ======"
