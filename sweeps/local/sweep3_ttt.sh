#!/bin/bash
# Sweep 3 — TTT-specific sweep using TTT_ONLY mode
# Loads a pre-quantized model and sweeps TTT configs WITHOUT retraining
# Each run takes ~40-50 min (sliding + TTT eval only, no training)
# Run on whichever GPU is free

cd ${PROJECT_ROOT:-.}/parameter-golf
source ${VENV:-./venv}/bin/activate

SCRIPT="records/track_10min_16mb/2026-03-27_ActivationSearch/train_gpt.py"
RESULTS="${PROJECT_ROOT:-.}/local_tests"

GPU=${1:-0}
PORT=$((29500 + GPU))
export CUDA_VISIBLE_DEVICES=$GPU

# Model to test against — UPDATE THIS to a 600s run .ptz
MODEL_PTZ="${2:-${PROJECT_ROOT:-.}/local_tests/best_600s.ptz}"

if [ ! -f "$MODEL_PTZ" ]; then
    echo "ERROR: Model file not found: $MODEL_PTZ"
    echo "Run a 600s training first, then copy final_model.int6.ptz here"
    exit 1
fi

# Architecture must match the model that was quantized
export MLP_ACTIVATION=mish2
export MUON_PLUS=1
export XSA_LAST_N=11
export MTP_NUM_HEADS=0
export BIGRAM_VOCAB_SIZE=3072
export BIGRAM_DIM=112
export NGRAM_CACHE=0
export SEED=1337
export EVAL_SEQ_LEN=2048
export MLP_MULT=3.5
export NUM_LAYERS=11
export COMPRESSOR=lzma
export QUANT_BITS_MLP=8
export QUANT_BITS_ATTN=6

# TTT base config
export TTT_ENABLED=1
export TTT_FREEZE_BLOCKS=0
export TTT_BATCH_SEQS=32
export TTT_GRAD_CLIP=1.0
export TTT_OPTIMIZER=adamw
export TTT_WEIGHT_DECAY=0.01
export TTT_QTTT=0

# TTT_ONLY mode — skip training, load quantized model directly
export TTT_ONLY="$MODEL_PTZ"

run_one() {
    local LABEL="$1"
    echo "$(date) === $LABEL ==="
    nice -n 10 torchrun --standalone --nproc_per_node=1 --master_port=$PORT "$SCRIPT" \
        > "$RESULTS/${LABEL}.log" 2>&1 || echo "$LABEL FAILED"
    local TTT=$(grep "legal_ttt_exact" "$RESULTS/${LABEL}.log" 2>/dev/null | grep -oP 'val_bpb:\K[0-9.]+' || echo "—")
    local SLID=$(grep "sliding_window_exact" "$RESULTS/${LABEL}.log" 2>/dev/null | grep -oP 'val_bpb:\K[0-9.]+' || echo "—")
    echo "  $LABEL: ttt=$TTT sliding=$SLID"
}

echo "$(date) === SWEEP 3 TTT (GPU $GPU, model=$MODEL_PTZ) ==="

# ─── Block 1: TTT Chunk Size (KEY SWEEP) ───
for CT in 4096 8192 16384 32768 65536 131072; do
    export TTT_CHUNK_TOKENS=$CT
    export TTT_LR=0.001
    export TTT_EPOCHS=3
    export TTT_SPECTRAL_K=0
    run_one "s3_chunk${CT}"
done
export TTT_CHUNK_TOKENS=32768

# ─── Block 2: Spectral Surgery ───
for SK in 0 2 4 8 16; do
    export TTT_SPECTRAL_K=$SK
    export TTT_LR=0.001
    export TTT_EPOCHS=3
    run_one "s3_spectral${SK}"
done
export TTT_SPECTRAL_K=0

# ─── Block 3: Spectral × Chunk interaction ───
for CT in 8192 16384; do
    for SK in 4 8; do
        export TTT_CHUNK_TOKENS=$CT
        export TTT_SPECTRAL_K=$SK
        export TTT_LR=0.001
        export TTT_EPOCHS=3
        run_one "s3_chunk${CT}_spec${SK}"
    done
done
export TTT_CHUNK_TOKENS=32768
export TTT_SPECTRAL_K=0

# ─── Block 4: TTT LR × Epochs ───
for LR in 0.0003 0.0005 0.001 0.002 0.003; do
    for EP in 1 2 3 5; do
        export TTT_LR=$LR
        export TTT_EPOCHS=$EP
        run_one "s3_lr${LR}_ep${EP}"
    done
done
export TTT_LR=0.001
export TTT_EPOCHS=3

# ─── Block 5: TTT Weight Decay ───
for WD in 0.0 0.001 0.005 0.01 0.02 0.05; do
    export TTT_WEIGHT_DECAY=$WD
    run_one "s3_tttwd${WD}"
done

echo ""
echo "$(date) === SWEEP 3 TTT COMPLETE ==="
echo "All results (sorted by TTT BPB):"
for f in "$RESULTS"/s3_*.log; do
    [ -f "$f" ] || continue
    name=$(basename "$f" .log)
    ttt=$(grep "legal_ttt_exact" "$f" 2>/dev/null | grep -oP 'val_bpb:\K[0-9.]+' || echo "—")
    slid=$(grep "sliding_window_exact" "$f" 2>/dev/null | grep -oP 'val_bpb:\K[0-9.]+' || echo "—")
    echo "$ttt $name sliding=$slid"
done | sort -n
