#!/bin/bash
# expZ Phase 1 (scratch A/B): short from-scratch training locally, both arms,
# compare post-EMA val_bpb. Cheap signal-check before committing to full retraining.
set -uo pipefail

cd ${PROJECT_ROOT:-.}
source venv/bin/activate

export VOCAB_SIZE=8192 NUM_LAYERS=11 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=4
export MLP_ACTIVATION=leaky05
export DATA_PATH="${PROJECT_ROOT:-.}/parameter-golf/data/datasets/fineweb10B_sp8192"
export TOKENIZER_PATH="${PROJECT_ROOT:-.}/parameter-golf/data/tokenizers/fineweb_8192_bpe.model"
export SEED=1337 TRAIN_SEQ_LEN=2048 EVAL_SEQ_LEN=2048 TRAIN_BATCH_TOKENS=786432
export VAL_BATCH_SIZE=524288
# Short-run config: 500 steps, wallclock cap disabled
export MAX_WALLCLOCK_SECONDS=0
export ITERATIONS=500
export WARMUP_STEPS=20
export VAL_LOSS_EVERY=0  # eval only at end
export TRAIN_LOG_EVERY=50
# Warmdown: make it kick in at step 350 of 500 (70%) so final 150 are decaying
export WARMDOWN_FRAC=0
export WARMDOWN_ITERS=150

export MUON_PLUS=1 MATRIX_LR=0.02 SCALAR_LR=0.02
export MUON_WD=0.090 ADAM_WD=0.02 EMBED_WD=0.090
export QK_GAIN_INIT=5.0 GRAD_CLIP_NORM=0.3

export SIGMOID_SKIP=1
# Disable recurrence for short run (RECUR_START_STEP higher than iterations)
export RECUR_LAYERS="3,4,5" RECUR_NUM_LOOPS=2 RECUR_START_STEP=999999
export VE_ENABLED=0
export PARALLEL_START_LAYER=7 DISABLE_LAYER0_ATTN=1
export XSA_LAST_N=11 TIE_EMBEDDINGS=1 REPEAT_PASS_SCALE=1

export EMA_DECAY=0.9965
export ROPE_BASE=10000 LOGIT_SOFTCAP=20
export REPEAT_UNTIE_MLP=lora REPEAT_LORA_RANK=16

# Skip GPTQ/compression/TTT
export QUANT_BITS_MLP=6 QUANT_BITS_ATTN=5 QUANT_BITS_EMBED=6
export GPTQ_ENABLED=0 COMPRESSOR=brotli BYTE_SHUFFLE=1
export RESUME_QUANT=0  # from-scratch training
export LATE_QAT=0
export SDCLIP=1 SDCLIP_K=12.85 SDCLIP_K_EMBED=20.0
export SLIDING_WINDOW_EVAL=0 EVAL_STRIDE=64
export CODE_SIZE_OVERRIDE=23524 TTT_ENABLED=0

EXP_DIR=h100_deploy/experiments/expZ_zloss
LOG_DIR="$EXP_DIR/logs"
mkdir -p "$LOG_DIR"
cd h100_deploy

# Two arms: control (ZW=0) and treatment (ZW=1e-4 as Ternary competitor uses)
for ARM in "control:0" "zloss_1em4:0.0001"; do
  NAME="${ARM%:*}"
  ZW="${ARM#*:}"
  LOG="../${EXP_DIR}/logs/scratch_${NAME}.log"
  if [ -s "$LOG" ] && grep -q "post_ema val_bpb:" "$LOG"; then
    echo "  [skip] $NAME already has result"
    continue
  fi
  echo ""
  echo "$(date) ── Running $NAME (Z_LOSS_WEIGHT=$ZW) ──"
  Z_LOSS_WEIGHT="$ZW" CUDA_VISIBLE_DEVICES=0 nice -n 10 \
    torchrun --standalone --nproc_per_node=1 \
      experiments/expZ_zloss/train_gpt.py 2>&1 \
    | tee "$LOG" \
    | grep -E "step:[0-9]+/[0-9]+ (train_loss|val_loss)|post_ema|Traceback|RuntimeError" | head -40
done

echo ""
echo "$(date) ── expZ scratch A/B DONE ──"
echo ""
echo "Results:"
for ARM in "control:0" "zloss_1em4:0.0001"; do
  NAME="${ARM%:*}"
  LOG="../${EXP_DIR}/logs/scratch_${NAME}.log"
  PE=$(grep 'post_ema val_loss' "$LOG" 2>/dev/null | head -1 | grep -oE 'val_bpb:[0-9]+\.[0-9]+' | cut -d: -f2)
  TRAIN_LOSS=$(grep -oE 'step:500/500 train_loss:[0-9]+\.[0-9]+' "$LOG" 2>/dev/null | head -1 | grep -oE '[0-9]+\.[0-9]+' | head -1)
  printf "  %-15s final_train_loss=%-7s  post_ema_val_bpb=%s\n" "$NAME" "${TRAIN_LOSS:-MISS}" "${PE:-MISS}"
done
