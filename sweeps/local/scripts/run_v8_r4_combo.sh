#!/bin/bash
# v8 R4: Our combo (RoPE fix + SOFTCAP=20) vs PR #1326 (aryanbhosale, #1 on leaderboard)
# GPU 0: Our v8 combo — best winning env vars from R1-R3
# GPU 1: PR #1326 — competitor's code with their defaults, local wallclock
set -uo pipefail
source ${VENV:-./venv}/bin/activate

REPO_DIR="${PROJECT_ROOT:-.}/parameter-golf"
SCRIPT_DIR="${PROJECT_ROOT:-.}/h100_final"
RESULTS_DIR="${PROJECT_ROOT:-.}/local_tests/v8_sweep_r4"
mkdir -p "$RESULTS_DIR"
cd "$REPO_DIR"

# ===== OUR COMBO: RoPE fix + SOFTCAP=20 (GPU 0) =====
run_our_combo() {
    cp "$SCRIPT_DIR/train_gpt_v8_experimental.py" "$REPO_DIR/train_gpt.py"
    export CUDA_VISIBLE_DEVICES=0
    # v7 Test B config + winning changes
    export VOCAB_SIZE=4096 NUM_LAYERS=11 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=4
    export MLP_ACTIVATION=leaky05
    export DATA_PATH="./data/datasets/fineweb10B_sp4096"
    export TOKENIZER_PATH="./data/tokenizers/fineweb_4096_bpe.model"
    export MAX_WALLCLOCK_SECONDS=10800
    export SEED=1337 TRAIN_BATCH_TOKENS=786432 TRAIN_SEQ_LEN=2048 EVAL_SEQ_LEN=2048
    export ITERATIONS=20000 WARMUP_STEPS=20
    export VAL_LOSS_EVERY=4000 TRAIN_LOG_EVERY=100
    export MUON_PLUS=1 MATRIX_LR=0.02 SCALAR_LR=0.02
    export MUON_WD=0.090 ADAM_WD=0.02 EMBED_WD=0.090
    export QK_GAIN_INIT=5.0 GRAD_CLIP_NORM=0.3
    export SIGMOID_SKIP=1 BYTE_SHUFFLE=1 COPRIME_STRIDE=1
    export RECUR_LAYERS="3,4,5" RECUR_NUM_LOOPS=2
    export VE_ENABLED=1 VE_DIM=128 VE_LAYERS="9,10"
    export PARALLEL_START_LAYER=7 DISABLE_LAYER0_ATTN=1
    export XSA_LAST_N=11 LN_SCALE=1 TIE_EMBEDDINGS=1
    export REPEAT_UNTIE_MLP=lora REPEAT_LORA_RANK=16
    export ATTN_RES=0 RECUR_SKIP_PHYSICAL=0
    export EMA_DECAY=0.9965 WARMDOWN_ITERS=3500 SWA_ENABLED=0
    export RECUR_START_STEP=2000 QUANT_BITS_EMBED=8 REPEAT_PASS_SCALE=1
    export GPTQ_ENABLED=1 GPTQ_CALIB_SEQS=64 COMPRESSOR=brotli
    export QUANT_BITS_MLP=6 QUANT_BITS_ATTN=6
    # === R4 WINNING CHANGES ===
    export ROPE_BASE=10000        # RoPE fix: proper base (was accidentally 22K)
    export LOGIT_SOFTCAP=20       # AREA 10 winner: -0.0024 pre-quant, -0.0025 quant
    export DATA_WARMDOWN_COUPLING=0  # AREA 7 was flat, keep default
    LOGFILE="$RESULTS_DIR/our_combo_$(date +%Y%m%d_%H%M%S).log"
    echo "$(date) OUR COMBO: v8 + RoPE fix (base=10K) + SOFTCAP=20"
    echo "Log: $LOGFILE"
    nice -n 10 torchrun --standalone --nproc_per_node=1 train_gpt.py 2>&1 | tee "$LOGFILE"
    echo "$(date) OUR COMBO complete"
}

# ===== PR #1326: aryanbhosale (GPU 1) =====
run_pr1326() {
    cp /tmp/pr1326_decompressed.py "$REPO_DIR/train_gpt.py"
    export CUDA_VISIBLE_DEVICES=1
    # Their defaults — only override wallclock/logging for local run
    export MAX_WALLCLOCK_SECONDS=10800
    export VAL_LOSS_EVERY=4000
    export TRAIN_LOG_EVERY=100
    export VOCAB_SIZE=4096
    export GPTQ_ENABLED=1
    # Unset our env vars that don't apply to their code
    unset DATA_PATH TOKENIZER_PATH MLP_ACTIVATION MUON_PLUS SIGMOID_SKIP BYTE_SHUFFLE
    unset COPRIME_STRIDE RECUR_NUM_LOOPS DISABLE_LAYER0_ATTN REPEAT_UNTIE_MLP REPEAT_LORA_RANK
    unset ATTN_RES RECUR_SKIP_PHYSICAL WARMDOWN_ITERS SWA_ENABLED QUANT_BITS_EMBED
    unset REPEAT_PASS_SCALE GPTQ_CALIB_SEQS QUANT_BITS_MLP QUANT_BITS_ATTN
    unset DATA_WARMDOWN_COUPLING DATA_RAMP_STEPS ROPE_BASE
    LOGFILE="$RESULTS_DIR/pr1326_aryanbhosale_$(date +%Y%m%d_%H%M%S).log"
    echo "$(date) PR #1326 (aryanbhosale): Their code, their defaults, local wallclock"
    echo "Log: $LOGFILE"
    nice -n 10 torchrun --standalone --nproc_per_node=1 train_gpt.py 2>&1 | tee "$LOGFILE"
    echo "$(date) PR #1326 complete"
}

# Launch both
run_our_combo &
PID_OURS=$!
sleep 10
run_pr1326 &
PID_PR=$!

echo "$(date) R4 launched: our_combo=$PID_OURS (GPU 0), pr1326=$PID_PR (GPU 1)"
echo "Monitor: tail -f $RESULTS_DIR/*.log"
wait $PID_OURS
echo "$(date) Our combo finished (exit $?)"
wait $PID_PR
echo "$(date) PR #1326 finished (exit $?)"
echo "$(date) === R4 complete ==="
