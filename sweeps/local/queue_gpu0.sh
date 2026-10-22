#!/bin/bash
# Queued experiments for GPU 0 — runs after budget_sweep_gpu0.sh finishes
# 1. Long 600s training run (MLP 3.5×, int8/int5) — test AdamW TTT on better-trained model
# 2. Same config with Brotli-11 compression — compare artifact size vs LZMA

echo "$(date) Starting queued experiments (GPU0)."

cd ${PROJECT_ROOT:-.}/parameter-golf
source ${VENV:-./venv}/bin/activate

SCRIPT="${PROJECT_ROOT:-.}/parameter-golf/records/track_10min_16mb/2026-03-27_ActivationSearch/train_gpt.py"
RESULTS="${PROJECT_ROOT:-.}/local_tests"

# Common config
export CUDA_VISIBLE_DEVICES=0
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
export EVAL_SEQ_LEN=2048
# TTT AdamW config
export TTT_ENABLED=1
export TTT_EPOCHS=3
export TTT_FREEZE_BLOCKS=0
export TTT_BATCH_SEQS=32
export TTT_GRAD_CLIP=1.0
export TTT_OPTIMIZER=adamw
export TTT_LR=0.001
export TTT_WEIGHT_DECAY=0.01
export TTT_QTTT=0
export TTT_SPECTRAL_K=0

# ============================================================
# RUN 1: Long training (600s) — MLP 3.5×, int8 MLP / int5 attn, LZMA
# Purpose: Does 2× more training improve AdamW TTT results?
# ============================================================
echo ""
echo "$(date) === LONG RUN 1/3: MLP 3.5x, int8/int5, 600s, LZMA ==="
export MLP_MULT=3.5
export QUANT_BITS_MLP=8
export QUANT_BITS_ATTN=5
export MAX_WALLCLOCK_SECONDS=600
export COMPRESSOR=lzma
nice -n 10 torchrun --standalone --nproc_per_node=1 --master_port=29500 "$SCRIPT" \
    > "$RESULTS/long_mlp35_int8_int5_600s.log" 2>&1 || echo "LONG RUN 1 FAILED"
echo "$(date) === LONG RUN 1 DONE ==="

# ============================================================
# RUN 2: Same as Run 1 but with Brotli-11 compression
# Purpose: Does Brotli give smaller artifact than LZMA?
# ============================================================
echo ""
echo "$(date) === LONG RUN 2/3: MLP 3.5x, int8/int5, 600s, BROTLI ==="
export MLP_MULT=3.5
export QUANT_BITS_MLP=8
export QUANT_BITS_ATTN=5
export MAX_WALLCLOCK_SECONDS=600
export COMPRESSOR=brotli
nice -n 10 torchrun --standalone --nproc_per_node=1 --master_port=29500 "$SCRIPT" \
    > "$RESULTS/long_mlp35_int8_int5_600s_brotli.log" 2>&1 || echo "LONG RUN 2 FAILED"
echo "$(date) === LONG RUN 2 DONE ==="

# ============================================================
# RUN 3: MLP 4×, int6/int5, 600s, Brotli — max capacity
# Purpose: Push model size up, see if 4× MLP fills budget better
# ============================================================
echo ""
echo "$(date) === LONG RUN 3/3: MLP 4x, int6/int5, 600s, BROTLI ==="
export MLP_MULT=4.0
export QUANT_BITS_MLP=6
export QUANT_BITS_ATTN=5
export MAX_WALLCLOCK_SECONDS=600
export COMPRESSOR=brotli
nice -n 10 torchrun --standalone --nproc_per_node=1 --master_port=29500 "$SCRIPT" \
    > "$RESULTS/long_mlp4_int6_int5_600s_brotli.log" 2>&1 || echo "LONG RUN 3 FAILED"
echo "$(date) === LONG RUN 3 DONE ==="

echo ""
echo "$(date) ====== GPU0 QUEUED RESULTS ======"
for f in "$RESULTS/long_mlp35_int8_int5_600s.log" \
         "$RESULTS/long_mlp35_int8_int5_600s_brotli.log" \
         "$RESULTS/long_mlp4_int6_int5_600s_brotli.log"; do
    name=$(basename "$f" .log)
    sliding=$(grep "final_int6_sliding_window_exact" "$f" 2>/dev/null | grep -oP 'val_bpb:\K[0-9.]+' || echo "—")
    ttt=$(grep "legal_ttt_exact" "$f" 2>/dev/null | grep -oP 'val_bpb:\K[0-9.]+' || echo "—")
    size=$(grep "Total submission size" "$f" 2>/dev/null | grep -oP '[0-9]+(?= bytes)' || echo "—")
    if [ "$size" != "—" ]; then size_mb=$(echo "scale=2; $size / 1048576" | bc); else size_mb="—"; fi
    echo "  $name: sliding=$sliding ttt=$ttt artifact=${size_mb}MB"
done
echo "$(date) ====== GPU0 QUEUE DONE ======"
