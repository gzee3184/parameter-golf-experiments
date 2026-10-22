#!/bin/bash
# expTq sweep: temperature scaling applied to POST-QUANT model.
# Each run: full GPTQ + roundtrip eval, then early-exit.
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

# expTq needs GPTQ on (vs expT which disabled it)
export QUANT_BITS_MLP=6 QUANT_BITS_ATTN=5 QUANT_BITS_EMBED=6
export GPTQ_ENABLED=1 COMPRESSOR=brotli BYTE_SHUFFLE=1
export GPTQ_CALIB_SEQS=64 GPTQ_CALIB_LEN=2048 GPTQ_CALIB_BATCHES=64 GPTQ_REAL_DATA=1
export RESUME_QUANT=1 LATE_QAT=0
export SDCLIP=1 SDCLIP_K=12.85 SDCLIP_K_EMBED=20.0
export SLIDING_WINDOW_EVAL=0 EVAL_STRIDE=64
export CODE_SIZE_OVERRIDE=23524 TTT_ENABLED=0
export EXPT_EARLY_EXIT=0  # DO NOT early-exit at post-EMA
export EXPTQ_EARLY_EXIT=1  # DO early-exit after roundtrip eval

if [ ! -L h100_deploy/final_model.pt ] || [ "$(readlink h100_deploy/final_model.pt)" != "run10/final_model_r10.pt" ]; then
  echo "ERROR: h100_deploy/final_model.pt must be symlink to run10/final_model_r10.pt"; exit 1
fi

EXP_DIR=h100_deploy/experiments/expTq_postquant_temp
LOG_DIR="$EXP_DIR/logs"
mkdir -p "$LOG_DIR"
cd h100_deploy

echo "$(date) ── expTq sweep START ──"
for T in 0.90 0.95 0.975 1.00 1.025 1.05 1.10; do
  LOG="../${EXP_DIR}/logs/Tq_${T}.log"
  if [ -s "$LOG" ] && grep -q "final_int6_roundtrip val_bpb:" "$LOG"; then
    echo "  [skip] T=$T already has result"
    continue
  fi
  echo "  running T=$T ..."
  EVAL_TEMP="$T" CUDA_VISIBLE_DEVICES=0 nice -n 10 \
    torchrun --standalone --nproc_per_node=1 \
      experiments/expTq_postquant_temp/train_gpt.py 2>&1 \
    | tee "$LOG" \
    | grep -E "^(resume:|post_ema|final_int6|expTq:|prune:|RuntimeError|Traceback|FAILED)" || true
  RESULT=$(grep -o 'final_int6_roundtrip val_bpb:[0-9.]*' "$LOG" | head -1)
  echo "    -> $RESULT"
done

echo ""
echo "$(date) ── expTq sweep DONE ──"
echo ""
echo "Results (pre-quant baseline = 1.1033; post-quant BASELINE varies with local compression):"
for T in 0.90 0.95 0.975 1.00 1.025 1.05 1.10; do
  LOG="../${EXP_DIR}/logs/Tq_${T}.log"
  PE_LINE=$(grep 'post_ema val_loss' "$LOG" 2>/dev/null | head -1)
  PE_BPB=$(echo "$PE_LINE" | grep -oE 'val_bpb:[0-9]+\.[0-9]+' | cut -d: -f2)
  RT_LINE=$(grep 'final_int6_roundtrip val_loss' "$LOG" 2>/dev/null | head -1)
  RT_LOSS=$(echo "$RT_LINE" | grep -oE 'val_loss:[0-9]+\.[0-9]+' | cut -d: -f2)
  RT_BPB=$(echo "$RT_LINE" | grep -oE 'val_bpb:[0-9]+\.[0-9]+' | cut -d: -f2)
  PRUNE=$(grep -oE 'prune: [0-9]+/[0-9]+ \([0-9.]+%' "$LOG" 2>/dev/null | head -1 || echo "prune_line_missing")
  printf "  T=%-6s  pre_quant=%-7s  post_quant_rt=%-7s  %s\n" "$T" "${PE_BPB:-MISS}" "${RT_BPB:-MISS}" "$PRUNE"
done
