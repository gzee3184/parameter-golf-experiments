#!/bin/bash
# Spectral surgery + TTT variant tests — GPU 1, sequential
# Shorter runs (300s training) for quick comparison
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
export MAX_WALLCLOCK_SECONDS=300
export EVAL_SEQ_LEN=2048
export TTT_ENABLED=1
export TTT_EPOCHS=3
export TTT_FREEZE_BLOCKS=0
export TTT_BATCH_SEQS=32
export TTT_GRAD_CLIP=1.0

echo "$(date) ====== SPECTRAL + TTT GPU1 START ======"

# Run 1: Spectral k=32, all params, AdamW
echo ""
echo "$(date) === RUN 1/3: Spectral k=32, AdamW, all params ==="
export TTT_QTTT=0
export TTT_OPTIMIZER=adamw
export TTT_LR=0.001
export TTT_WEIGHT_DECAY=0.01
export TTT_MOMENTUM=0.9
export TTT_SPECTRAL_K=32
nice -n 10 torchrun --standalone --nproc_per_node=1 --master_port=29501 "$SCRIPT" \
    > "$RESULTS/ttt_spectral_k32_adamw.log" 2>&1 || echo "RUN 1 FAILED"
echo "$(date) === RUN 1 DONE ==="

# Run 2: Spectral k=64, qTTT, AdamW
echo ""
echo "$(date) === RUN 2/3: Spectral k=64, qTTT, AdamW ==="
export TTT_QTTT=1
export TTT_OPTIMIZER=adamw
export TTT_LR=0.001
export TTT_WEIGHT_DECAY=0.01
export TTT_SPECTRAL_K=64
nice -n 10 torchrun --standalone --nproc_per_node=1 --master_port=29501 "$SCRIPT" \
    > "$RESULTS/ttt_spectral_k64_qttt.log" 2>&1 || echo "RUN 2 FAILED"
echo "$(date) === RUN 2 DONE ==="

# Run 3: Full params, AdamW, no spectral (AdamW-only comparison)
echo ""
echo "$(date) === RUN 3/3: Full TTT, AdamW (no spectral) ==="
export TTT_QTTT=0
export TTT_OPTIMIZER=adamw
export TTT_LR=0.001
export TTT_WEIGHT_DECAY=0.01
export TTT_SPECTRAL_K=0
nice -n 10 torchrun --standalone --nproc_per_node=1 --master_port=29501 "$SCRIPT" \
    > "$RESULTS/ttt_full_adamw.log" 2>&1 || echo "RUN 3 FAILED"
echo "$(date) === RUN 3 DONE ==="

echo ""
echo "$(date) ====== GPU1 RESULTS ======"
for f in "$RESULTS/ttt_spectral_k32_adamw.log" \
         "$RESULTS/ttt_spectral_k64_qttt.log" \
         "$RESULTS/ttt_full_adamw.log"; do
    name=$(basename "$f" .log)
    steps=$(grep "stopping_early" "$f" 2>/dev/null | grep -oP 'step:\K[0-9]+' || echo "?")
    sliding=$(grep "final_int6_sliding_window_exact" "$f" 2>/dev/null | grep -oP 'val_bpb:\K[0-9.]+' || echo "?")
    ttt=$(grep "legal_ttt_exact" "$f" 2>/dev/null | grep -oP 'val_bpb:\K[0-9.]+' || echo "N/A")
    echo "  $name: steps=$steps sliding=$sliding ttt=$ttt"
done
echo "$(date) ====== DONE ======"
