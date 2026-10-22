#!/bin/bash
# expDS sweep: ensemble logits from last N intermediate hidden states.
# DS_N=0 is control (must reproduce 1.1033)
# Each run is fresh torchrun process -> fresh compile cache
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

export QUANT_BITS_MLP=6 QUANT_BITS_ATTN=5 QUANT_BITS_EMBED=6
export GPTQ_ENABLED=0 COMPRESSOR=brotli BYTE_SHUFFLE=1
export RESUME_QUANT=1 LATE_QAT=0
export SDCLIP=1 SDCLIP_K=12.85 SDCLIP_K_EMBED=20.0
export SLIDING_WINDOW_EVAL=0 EVAL_STRIDE=64
export CODE_SIZE_OVERRIDE=23524 TTT_ENABLED=0

# Verify r10 symlink
if [ ! -L h100_deploy/final_model.pt ] || [ "$(readlink h100_deploy/final_model.pt)" != "run10/final_model_r10.pt" ]; then
  echo "ERROR: h100_deploy/final_model.pt must be symlink to run10/final_model_r10.pt"; exit 1
fi

EXP_DIR=h100_deploy/experiments/expDS_deep_supervision
LOG_DIR="$EXP_DIR/logs"
mkdir -p "$LOG_DIR"
cd h100_deploy

# Each row: NAME  DS_N  DS_W
# DS_N=0 is control (straight baseline path, no ensembling)
# DS_N=1 with DS_W=1.0 should give ~same as baseline (only last layer, just through new branch)
# DS_N>=2 ensembles
declare -a ARMS=(
  "control_N0     0   "
  "singleN1       1   "
  "uniformN2      2   "
  "uniformN3      3   "
  "uniformN4      4   "
  "weighted_N3a   3   0.2,0.3,0.5"
  "weighted_N3b   3   0.1,0.2,0.7"
  "weighted_N4a   4   0.1,0.15,0.25,0.5"
)

echo "$(date) ── expDS sweep START ──"
for ARM in "${ARMS[@]}"; do
  read -r NAME NV WS <<< "$ARM"
  LOG="../${EXP_DIR}/logs/${NAME}.log"
  if [ -s "$LOG" ] && grep -q "post_ema val_bpb:" "$LOG"; then
    echo "  [skip] $NAME"; continue
  fi
  echo "  running $NAME  DS_N=$NV  DS_W=${WS:-uniform}"
  DS_N="$NV" DS_W="${WS:-}" EXPT_EARLY_EXIT=1 CUDA_VISIBLE_DEVICES=0 nice -n 10 \
    torchrun --standalone --nproc_per_node=1 \
      experiments/expDS_deep_supervision/train_gpt.py 2>&1 \
    | tee "$LOG" \
    | grep -E "^(resume:|post_ema|expDS:|RuntimeError|Traceback|Error|FAILED)" || true
done

echo ""
echo "$(date) ── expDS sweep DONE ──"
echo ""
echo "Results summary:"
for ARM in "${ARMS[@]}"; do
  read -r NAME NV WS <<< "$ARM"
  LOG="../${EXP_DIR}/logs/${NAME}.log"
  LINE=$(grep 'post_ema val_loss' "$LOG" 2>/dev/null | head -1)
  LOSS=$(echo "$LINE" | grep -oE 'val_loss:[0-9]+\.[0-9]+' | cut -d: -f2)
  BPB=$(echo "$LINE"  | grep -oE 'val_bpb:[0-9]+\.[0-9]+'  | cut -d: -f2)
  if [ -n "$BPB" ]; then
    DELTA=$(python3 -c "print(f'{float($BPB)-1.1033:+.4f}')" 2>/dev/null)
  else
    DELTA="MISSING"
  fi
  printf "  %-15s DS_N=%s  %-22s loss=%-7s bpb=%-7s delta=%s\n" "$NAME" "$NV" "W=${WS:-uniform}" "$LOSS" "$BPB" "$DELTA"
done
