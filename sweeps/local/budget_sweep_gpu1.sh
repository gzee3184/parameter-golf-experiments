#!/bin/bash
# Budget filling experiments — GPU 1
# Test reverse quant strategy + LR sweep for AdamW TTT
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
# TTT with AdamW
export TTT_ENABLED=1
export TTT_EPOCHS=3
export TTT_FREEZE_BLOCKS=0
export TTT_BATCH_SEQS=32
export TTT_GRAD_CLIP=1.0
export TTT_OPTIMIZER=adamw
export TTT_WEIGHT_DECAY=0.01
export TTT_QTTT=0
export TTT_SPECTRAL_K=0

echo "$(date) ====== BUDGET SWEEP GPU1 START ======"

# Run 1: MLP 3.5x, int8 MLP / int5 attn (give MLP MORE bits, Abay's key insight)
echo ""
echo "$(date) === RUN 1/3: MLP 3.5x, int8 MLP / int5 attn ==="
export MLP_MULT=3.5
export QUANT_BITS_MLP=8
export QUANT_BITS_ATTN=5
export TTT_LR=0.001
nice -n 10 torchrun --standalone --nproc_per_node=1 --master_port=29501 "$SCRIPT" \
    > "$RESULTS/budget_mlp35_int8_int5.log" 2>&1 || echo "RUN 1 FAILED"
echo "$(date) === RUN 1 DONE ==="

# Run 2: MLP 3.5x, int6/int6, TTT lr=0.002 (match old SGD lr, might be better for AdamW too)
echo ""
echo "$(date) === RUN 2/3: MLP 3.5x, int6/int6, TTT lr=0.002 ==="
export MLP_MULT=3.5
export QUANT_BITS_MLP=6
export QUANT_BITS_ATTN=6
export TTT_LR=0.002
nice -n 10 torchrun --standalone --nproc_per_node=1 --master_port=29501 "$SCRIPT" \
    > "$RESULTS/budget_mlp35_int6_lr002.log" 2>&1 || echo "RUN 2 FAILED"
echo "$(date) === RUN 2 DONE ==="

# Run 3: MLP 3.5x, int6/int5, TTT lr=0.002 (combined best config candidate)
echo ""
echo "$(date) === RUN 3/3: MLP 3.5x, int6 MLP / int5 attn, TTT lr=0.002 ==="
export MLP_MULT=3.5
export QUANT_BITS_MLP=6
export QUANT_BITS_ATTN=5
export TTT_LR=0.002
nice -n 10 torchrun --standalone --nproc_per_node=1 --master_port=29501 "$SCRIPT" \
    > "$RESULTS/budget_mlp35_int6_int5_lr002.log" 2>&1 || echo "RUN 3 FAILED"
echo "$(date) === RUN 3 DONE ==="

echo ""
echo "$(date) ====== GPU1 RESULTS ======"
for f in "$RESULTS/budget_mlp35_int8_int5.log" \
         "$RESULTS/budget_mlp35_int6_lr002.log" \
         "$RESULTS/budget_mlp35_int6_int5_lr002.log"; do
    name=$(basename "$f" .log)
    params=$(grep "model_params" "$f" 2>/dev/null | grep -oP '[0-9]+' || echo "?")
    steps=$(grep "stopping_early" "$f" 2>/dev/null | grep -oP 'step:\K[0-9]+' || echo "?")
    size=$(grep "Total submission size" "$f" 2>/dev/null | grep -oP '[0-9]+' || echo "?")
    sliding=$(grep "final_int6_sliding_window_exact" "$f" 2>/dev/null | grep -oP 'val_bpb:\K[0-9.]+' || echo "?")
    ttt=$(grep "legal_ttt_exact" "$f" 2>/dev/null | grep -oP 'val_bpb:\K[0-9.]+' || echo "N/A")
    size_mb=$(echo "scale=2; $size / 1048576" | bc 2>/dev/null || echo "?")
    echo "  $name: params=$params steps=$steps artifact=${size_mb}MB sliding=$sliding ttt=$ttt"
done
echo "$(date) ====== DONE ======"
