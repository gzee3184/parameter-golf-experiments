#!/bin/bash
# expT sweep: grid T over {0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15}.
# Each T runs as a fresh torchrun process to guarantee a clean torch.compile state.
# T=1.00 is the CONTROL — must produce val_bpb=1.1033.
set -uo pipefail

cd ${PROJECT_ROOT:-.}
source venv/bin/activate

export VOCAB_SIZE=8192 NUM_LAYERS=11 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=4
export MLP_ACTIVATION=leaky05
export DATA_PATH="${PROJECT_ROOT:-.}/parameter-golf/data/datasets/fineweb10B_sp8192"
export TOKENIZER_PATH="${PROJECT_ROOT:-.}/parameter-golf/data/tokenizers/fineweb_8192_bpe.model"
export SEED=1337 TRAIN_SEQ_LEN=2048 EVAL_SEQ_LEN=2048 TRAIN_BATCH_TOKENS=786432
export VAL_BATCH_SIZE=524288 MAX_WALLCLOCK_SECONDS=600
export ITERATIONS=20000 WARMUP_STEPS=20
export VAL_LOSS_EVERY=0 TRAIN_LOG_EVERY=100

export MUON_PLUS=1 MATRIX_LR=0.02 SCALAR_LR=0.02
export MUON_WD=0.090 ADAM_WD=0.02 EMBED_WD=0.090
export QK_GAIN_INIT=5.0 GRAD_CLIP_NORM=0.3

export SIGMOID_SKIP=1
export RECUR_LAYERS="3,4,5" RECUR_NUM_LOOPS=2 RECUR_START_STEP=2000
export VE_ENABLED=0
export PARALLEL_START_LAYER=7 DISABLE_LAYER0_ATTN=1
export XSA_LAST_N=11 TIE_EMBEDDINGS=1 REPEAT_PASS_SCALE=1

export EMA_DECAY=0.9965 WARMDOWN_FRAC=0.72 WARMDOWN_ITERS=3500
export ROPE_BASE=10000 LOGIT_SOFTCAP=20

export REPEAT_UNTIE_MLP=lora REPEAT_LORA_RANK=16

# Skip everything after post-EMA eval: no GPTQ, no sliding, no TTT.
export QUANT_BITS_MLP=6 QUANT_BITS_ATTN=5 QUANT_BITS_EMBED=6
export GPTQ_ENABLED=0 COMPRESSOR=brotli BYTE_SHUFFLE=1
export RESUME_QUANT=1 LATE_QAT=0
export SDCLIP=1 SDCLIP_K=12.85 SDCLIP_K_EMBED=20.0
export SLIDING_WINDOW_EVAL=0 EVAL_STRIDE=64
export CODE_SIZE_OVERRIDE=23524 TTT_ENABLED=0

# Ensure final_model.pt symlink points to r10 checkpoint (canonical setup)
if [ ! -L h100_deploy/final_model.pt ]; then
  echo "ERROR: h100_deploy/final_model.pt is not a symlink — aborting to avoid overwriting real checkpoints"
  exit 1
fi
TARGET=$(readlink h100_deploy/final_model.pt)
if [ "$TARGET" != "run10/final_model_r10.pt" ]; then
  echo "ERROR: h100_deploy/final_model.pt points to '$TARGET', expected run10/final_model_r10.pt"
  exit 1
fi

EXP_DIR=h100_deploy/experiments/expT_temp_scaling
LOG_DIR="$EXP_DIR/logs"
mkdir -p "$LOG_DIR"

# Run the EXPERIMENT train_gpt.py (has EVAL_TEMP modification), not the root one.
# We cd into h100_deploy/ so final_model.pt resolves correctly, but point torchrun at expT/train_gpt.py
cd h100_deploy

echo "$(date) ── expT sweep START ──"
echo "Results will be in: ${EXP_DIR}/logs/T_<VAL>.log"

for T in 0.85 0.90 0.95 1.00 1.05 1.10 1.15; do
  LOG="../${EXP_DIR}/logs/T_${T}.log"
  if [ -s "$LOG" ] && grep -q "post_ema val_bpb:" "$LOG"; then
    echo "  [skip] T=$T already has result: $(grep -o 'post_ema val_bpb:[0-9.]*' "$LOG")"
    continue
  fi
  echo "  running T=$T ..."
  EVAL_TEMP="$T" EXPT_EARLY_EXIT=1 CUDA_VISIBLE_DEVICES=0 nice -n 10 \
    torchrun --standalone --nproc_per_node=1 \
      experiments/expT_temp_scaling/train_gpt.py 2>&1 \
    | tee "$LOG" \
    | grep -E "^(resume:|post_ema|expT:|RuntimeError|Error|FAILED)" || true
  RESULT=$(grep -o 'post_ema val_bpb:[0-9.]*' "$LOG" | head -1)
  echo "    -> $RESULT"
done

echo ""
echo "$(date) ── expT sweep DONE ──"
echo ""
echo "Results summary:"
for T in 0.85 0.90 0.95 1.00 1.05 1.10 1.15; do
  LOG="../${EXP_DIR}/logs/T_${T}.log"
  BPB=$(grep -oE 'post_ema val_bpb:[0-9.]+' "$LOG" 2>/dev/null | head -1 | cut -d: -f2)
  echo "  T=$T  val_bpb=${BPB:-MISSING}"
done
