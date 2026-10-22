#!/bin/bash
# AttnRes sweep: 4 configs, 2 rounds of 2 parallel runs (one per GPU)
# Round 1: A (13L/512d + AttnRes) on GPU0, D (15L/512d + AttnRes) on GPU1
# Round 2: B (15L/480d + AttnRes) on GPU0, Control (13L/512d no AttnRes) on GPU1
set -uo pipefail
source ${VENV:-./venv}/bin/activate

REPO_DIR="${PROJECT_ROOT:-.}/parameter-golf"
SCRIPT_DIR="${PROJECT_ROOT:-.}/h100_final"
RESULTS_DIR="${PROJECT_ROOT:-.}/local_tests/attnres_sweep"
mkdir -p "$RESULTS_DIR"
cd "$REPO_DIR"
cp "$SCRIPT_DIR/train_gpt.py" "$REPO_DIR/train_gpt.py"

WALLCLOCK=900  # 15 min per config

# Common env vars
export DATA_PATH="./data/datasets/fineweb10B_sp4096"
export TOKENIZER_PATH="./data/tokenizers/fineweb_4096_bpe.model"
export VOCAB_SIZE=4096 MLP_MULT=4 MLP_ACTIVATION=leaky05
export SEED=1337 TRAIN_BATCH_TOKENS=786432 TRAIN_SEQ_LEN=2048 EVAL_SEQ_LEN=2048
export ITERATIONS=20000 WARMUP_STEPS=20 WARMDOWN_ITERS=2500
export VAL_LOSS_EVERY=4000 TRAIN_LOG_EVERY=100
export MUON_PLUS=1 MATRIX_LR=0.02 SCALAR_LR=0.02
export MUON_WD=0.12 ADAM_WD=0.02 EMBED_WD=0.12
export QK_GAIN_INIT=4.0 GRAD_CLIP_NORM=0.3 EMA_DECAY=0.997
export SIGMOID_SKIP=1 BYTE_SHUFFLE=1 COPRIME_STRIDE=1
export LOGIT_SOFTCAP=30.0 XSA_LAST_N=13 LN_SCALE=1 TIE_EMBEDDINGS=1
export QUANT_BITS_MLP=6 QUANT_BITS_ATTN=6 QUANT_BITS_EMBED=6
export GPTQ_ENABLED=1 GPTQ_CALIB_SEQS=64 COMPRESSOR=brotli
export MAX_WALLCLOCK_SECONDS=$WALLCLOCK

run_config() {
    local name=$1 gpu=$2 layers=$3 dim=$4 heads=$5 kv_heads=$6 attn_res=$7
    local logfile="$RESULTS_DIR/${name}.log"
    echo "$(date) Starting $name: ${layers}L/${dim}d/${heads}h attn_res=$attn_res on GPU $gpu"
    CUDA_VISIBLE_DEVICES=$gpu NUM_LAYERS=$layers MODEL_DIM=$dim NUM_HEADS=$heads NUM_KV_HEADS=$kv_heads \
        ATTN_RES=$attn_res XSA_LAST_N=$layers \
        torchrun --standalone --nproc_per_node=1 train_gpt.py 2>&1 | tee "$logfile"
    echo "$(date) Finished $name"
    echo "---"
    grep "DIAGNOSTIC post_ema" "$logfile" 2>/dev/null
    grep "budget:" "$logfile" 2>/dev/null
    grep "selective_prune:" "$logfile" 2>/dev/null | tail -1
    grep "Total submission size" "$logfile" 2>/dev/null
    grep "final_int6_roundtrip_exact" "$logfile" 2>/dev/null
    echo "==="
}

echo "========== ROUND 1: A + D =========="
echo "$(date) Round 1 starting"

# A: 13L/512d + AttnRes (baseline comparison)
run_config "A_13L_512d_attnres" 0 13 512 8 4 1 &
PID_A=$!

# D: 15L/512d + AttnRes (deeper, same width, ideal head_dim=64)
run_config "D_15L_512d_attnres" 1 15 512 8 4 1 &
PID_D=$!

wait $PID_A $PID_D
echo "$(date) Round 1 done"
echo ""

echo "========== ROUND 2: B + Control =========="
echo "$(date) Round 2 starting"

# B: 15L/480d + AttnRes (deeper, narrower, head_dim=60)
run_config "B_15L_480d_attnres" 0 15 480 8 4 1 &
PID_B=$!

# Control: 13L/512d WITHOUT AttnRes (current v4)
run_config "Control_13L_512d_noattnres" 1 13 512 8 4 0 &
PID_C=$!

wait $PID_B $PID_C
echo "$(date) Round 2 done"
echo ""

echo "========== SWEEP COMPLETE =========="
echo "$(date) All 4 configs done. Results in $RESULTS_DIR/"
echo ""
for f in "$RESULTS_DIR"/*.log; do
    name=$(basename "$f" .log)
    echo "--- $name ---"
    grep "model_params:" "$f" 2>/dev/null
    grep "step_avg:" "$f" 2>/dev/null | tail -1
    grep "DIAGNOSTIC post_ema" "$f" 2>/dev/null
    grep "final_int6_roundtrip_exact" "$f" 2>/dev/null
    grep "Total submission size" "$f" 2>/dev/null
    echo ""
done
