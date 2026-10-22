#!/bin/bash
# v8 sweep Round 3: AREA 10 (softcap=20) + AREA 7 (data ramp + warmdown coupling)
# GPU 0: AREA 10 — LOGIT_SOFTCAP=20
# GPU 1: AREA 7 — Faster ramp (1000 steps) + LR-coupled diversity warmdown
set -uo pipefail
source ${VENV:-./venv}/bin/activate

REPO_DIR="${PROJECT_ROOT:-.}/parameter-golf"
SCRIPT_DIR="${PROJECT_ROOT:-.}/h100_final"
RESULTS_DIR="${PROJECT_ROOT:-.}/local_tests/v8_sweep_r3"
mkdir -p "$RESULTS_DIR"
cd "$REPO_DIR"
cp "$SCRIPT_DIR/train_gpt_v8_experimental.py" "$REPO_DIR/train_gpt.py"

# ===== COMMON ENV (v7 Test B config + RoPE fix + ROPE_BASE=22000) =====
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
export LOGIT_SOFTCAP=30.0 XSA_LAST_N=11 LN_SCALE=1 TIE_EMBEDDINGS=1
export REPEAT_UNTIE_MLP=lora REPEAT_LORA_RANK=16
export ATTN_RES=0 RECUR_SKIP_PHYSICAL=0
export EMA_DECAY=0.9965 WARMDOWN_ITERS=3500 SWA_ENABLED=0
export RECUR_START_STEP=2000 QUANT_BITS_EMBED=8 REPEAT_PASS_SCALE=1
export GPTQ_ENABLED=1 GPTQ_CALIB_SEQS=64 COMPRESSOR=brotli
export QUANT_BITS_MLP=6 QUANT_BITS_ATTN=6
export ROPE_BASE=22000

# ===== TEST: AREA 10 — Lower logit softcap =====
run_area10() {
    export CUDA_VISIBLE_DEVICES=0
    export LOGIT_SOFTCAP=20
    # Data loader: default settings (control)
    export DATA_RAMP_STEPS=1800
    export DATA_WARMDOWN_COUPLING=0
    LOGFILE="$RESULTS_DIR/area10_softcap20_$(date +%Y%m%d_%H%M%S).log"
    echo "$(date) AREA 10: LOGIT_SOFTCAP=20 + ROPE_BASE=22000"
    echo "Log: $LOGFILE"
    nice -n 10 torchrun --standalone --nproc_per_node=1 train_gpt.py 2>&1 | tee "$LOGFILE"
    echo "$(date) AREA 10 complete"
}

# ===== TEST: AREA 7 — Faster data ramp + LR-coupled diversity warmdown =====
# Theory: Current loader holds max diversity during entire warmdown (75% of training).
# LR is decaying → gradient noise dominates → max diversity hurts convergence.
# Fix: couple data diversity to LR schedule via warmdown_frac.
# Ramp: 1000 steps (vs 1800) to reach peak diversity faster.
# Warmdown: alpha rises from 0.50→0.80, mix shrinks by 50% — tracks LR decay.
run_area7() {
    export CUDA_VISIBLE_DEVICES=1
    # Faster ramp to peak diversity (1000 vs 1800 steps)
    export DATA_RAMP_STEPS=1000
    # Enable LR-coupled diversity warmdown
    export DATA_WARMDOWN_COUPLING=1
    LOGFILE="$RESULTS_DIR/area7_dataramp_warmdown_$(date +%Y%m%d_%H%M%S).log"
    echo "$(date) AREA 7: DATA_RAMP_STEPS=1000, DATA_WARMDOWN_COUPLING=1 (alpha 0.50->0.80, mix -50% during warmdown)"
    echo "Log: $LOGFILE"
    nice -n 10 torchrun --standalone --nproc_per_node=1 train_gpt.py 2>&1 | tee "$LOGFILE"
    echo "$(date) AREA 7 complete"
}

# Launch both
run_area10 &
PID_A10=$!
sleep 5
run_area7 &
PID_A7=$!

echo "$(date) Round 3 launched: area10=$PID_A10 (GPU 0), area7=$PID_A7 (GPU 1)"
echo "Monitor: tail -f $RESULTS_DIR/*.log"
wait $PID_A10
echo "$(date) AREA 10 finished (exit $?)"
wait $PID_A7
echo "$(date) AREA 7 finished (exit $?)"
echo "$(date) === Round 3 complete (combo will be run manually) ==="
