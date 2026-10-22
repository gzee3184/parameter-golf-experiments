#!/bin/bash
# Run the canonical train_gpt.py with RESUME_QUANT=1 on 1 local GPU.
# This uses the EXACT same code path as H100, just on our machine.
# Gets us: the real local-reproducible baseline BPB.
set -eo pipefail

cd ${PROJECT_ROOT:-.}
source venv/bin/activate

# Match run10c environment exactly
export VOCAB_SIZE=8192 NUM_LAYERS=11 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=4
export MLP_ACTIVATION=leaky05
export DATA_PATH="${PROJECT_ROOT:-.}/parameter-golf/data/datasets/fineweb10B_sp8192"
export TOKENIZER_PATH="${PROJECT_ROOT:-.}/parameter-golf/data/tokenizers/fineweb_8192_bpe.model"
export SEED=1337 TRAIN_SEQ_LEN=2048 EVAL_SEQ_LEN=2048
export TRAIN_BATCH_TOKENS=786432
export VAL_BATCH_SIZE=524288
export MAX_WALLCLOCK_SECONDS=600
export ITERATIONS=20000 WARMUP_STEPS=20
export VAL_LOSS_EVERY=0 TRAIN_LOG_EVERY=100

export MUON_PLUS=1 MATRIX_LR=0.02 SCALAR_LR=0.02
export MUON_WD=0.090 ADAM_WD=0.02 EMBED_WD=0.090
export QK_GAIN_INIT=5.0 GRAD_CLIP_NORM=0.3

export SIGMOID_SKIP=1
export RECUR_LAYERS="3,4,5" RECUR_NUM_LOOPS=2 RECUR_START_STEP=2000
export VE_ENABLED=0
export PARALLEL_START_LAYER=7 DISABLE_LAYER0_ATTN=1
export XSA_LAST_N=11 TIE_EMBEDDINGS=1
export REPEAT_PASS_SCALE=1

export EMA_DECAY=0.9965 WARMDOWN_FRAC=0.72 WARMDOWN_ITERS=3500
export ROPE_BASE=10000 LOGIT_SOFTCAP=20

export REPEAT_UNTIE_MLP=lora REPEAT_LORA_RANK=16

export QUANT_BITS_MLP=6 QUANT_BITS_ATTN=5 QUANT_BITS_EMBED=6
export GPTQ_ENABLED=0   # Skip GPTQ; we only want the post-EMA eval line
export COMPRESSOR=brotli
export BYTE_SHUFFLE=1

export RESUME_QUANT=1   # Short-circuit to post-EMA eval using final_model.pt

export LATE_QAT=0
export SDCLIP=1 SDCLIP_K=12.85 SDCLIP_K_EMBED=20.0
export SLIDING_WINDOW_EVAL=0  # Skip sliding for this test
export EVAL_STRIDE=64
export CODE_SIZE_OVERRIDE=23524
export TTT_ENABLED=0  # Skip TTT

cd h100_deploy
echo "$(date) ── verify_r10_canonical: train_gpt.py RESUME_QUANT=1 on 1 GPU ──"
CUDA_VISIBLE_DEVICES=0 nice -n 10 \
  torchrun --standalone --nproc_per_node=1 train_gpt.py 2>&1 \
  | tee experiments/verify_r10_canonical.log
echo "$(date) ── verify_r10_canonical COMPLETE ──"
