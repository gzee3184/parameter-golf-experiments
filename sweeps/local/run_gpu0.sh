#!/bin/bash
# GPU 0: Leaderboard baseline (leaky05, no Muon+)
cd ${PROJECT_ROOT:-.}/parameter-golf
source ${VENV:-./venv}/bin/activate

export CUDA_VISIBLE_DEVICES=0
export MLP_ACTIVATION=leaky05
export MUON_PLUS=0
export XSA_LAST_N=11
export MTP_NUM_HEADS=0
export BIGRAM_VOCAB_SIZE=3072
export BIGRAM_DIM=112
export WARMDOWN_ITERS=4000
export GPTQ_ENABLED=1
export GPTQ_CALIB_SEQS=32
export TARGET_MB=15.9
export NGRAM_CACHE=0
export TTT_ENABLED=1
export TTT_LR=0.002
export TTT_EPOCHS=3
export TTT_FREEZE_BLOCKS=0
export TTT_MOMENTUM=0.9
export TTT_BATCH_SEQS=32
export TTT_GRAD_CLIP=1.0
export SEED=1337
export MAX_WALLCLOCK_SECONDS="${1:-14400}"

exec nice -n 10 torchrun --standalone --nproc_per_node=1 --master_port=29500 \
    ${PROJECT_ROOT:-.}/parameter-golf/records/track_10min_16mb/2026-03-27_ActivationSearch/train_gpt.py
