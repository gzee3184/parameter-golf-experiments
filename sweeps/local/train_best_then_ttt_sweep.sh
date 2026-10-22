#!/bin/bash
# Step 1: Train best config for 600s to produce a good .ptz model
# Step 2: Run TTT sweep on that model
# Run on GPU 1 (GPU 0 runs reliable MLP sweep)

cd ${PROJECT_ROOT:-.}/parameter-golf
source ${VENV:-./venv}/bin/activate

SCRIPT="records/track_10min_16mb/2026-03-27_ActivationSearch/train_gpt.py"
RESULTS="${PROJECT_ROOT:-.}/local_tests"
GPU=${1:-1}
PORT=$((29500 + GPU))
export CUDA_VISIBLE_DEVICES=$GPU

echo "$(date) === STEP 1: Training best config for 600s ==="

# Best known config
export MLP_ACTIVATION=mish2
export MUON_PLUS=1
export XSA_LAST_N=11
export MTP_NUM_HEADS=0
export BIGRAM_VOCAB_SIZE=3072
export BIGRAM_DIM=112
export WARMDOWN_ITERS=4000
export GPTQ_ENABLED=1
export GPTQ_CALIB_SEQS=8
export TARGET_MB=15.9
export NGRAM_CACHE=0
export SEED=1337
export EVAL_SEQ_LEN=2048
export MAX_WALLCLOCK_SECONDS=600
export MLP_MULT=3.5
export QUANT_BITS_MLP=8
export QUANT_BITS_ATTN=6
export COMPRESSOR=lzma
export MUON_WD=0.04
export TTT_ENABLED=0
export SKIP_SLIDING=1

nice -n 10 torchrun --standalone --nproc_per_node=1 --master_port=$PORT "$SCRIPT" \
    > "$RESULTS/train_600s_for_ttt.log" 2>&1

# Save the .ptz for TTT sweep
PTZ_SRC="final_model.int6.ptz"
PTZ_DEST="$RESULTS/best_600s.ptz"
if [ -f "$PTZ_SRC" ]; then
    cp "$PTZ_SRC" "$PTZ_DEST"
    echo "$(date) Model saved to $PTZ_DEST ($(wc -c < "$PTZ_DEST") bytes)"
else
    echo "$(date) ERROR: $PTZ_SRC not found!"
    exit 1
fi

echo "$(date) === STEP 2: Running TTT sweep ==="
bash "$RESULTS/sweep3_ttt.sh" "$GPU" "$PTZ_DEST"

echo "$(date) === ALL DONE ==="
