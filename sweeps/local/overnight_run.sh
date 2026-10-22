#!/bin/bash
# Overnight comparison tests — Muon+ and TTT tuning
# Run from: ${PROJECT_ROOT:-.}/parameter-golf
# Expected total time: ~4-5 hours (4 runs × ~60-75 min each)

set -e
cd ${PROJECT_ROOT:-.}/parameter-golf
source ${VENV:-./venv}/bin/activate

SCRIPT="records/track_10min_16mb/2026-03-27_ActivationSearch/train_gpt.py"
RESULTS="${PROJECT_ROOT:-.}/local_tests"

# Common env vars for all runs
export MLP_ACTIVATION=mish2
export XSA_LAST_N=11
export MTP_NUM_HEADS=0
export BIGRAM_VOCAB_SIZE=3072
export BIGRAM_DIM=112
export WARMDOWN_ITERS=4000
export GPTQ_ENABLED=1
export GPTQ_CALIB_SEQS=32
export TARGET_MB=15.9
export NGRAM_CACHE=0
export MAX_WALLCLOCK_SECONDS=600
export SEED=1337

# ============================================================
# RUN 1: Control — current config, no Muon+
# ============================================================
echo "$(date) === RUN 1/4: Control (no Muon+, TTT epochs=3, freeze=0) ==="
export MUON_PLUS=0
export TTT_ENABLED=1
export TTT_LR=0.002
export TTT_EPOCHS=3
export TTT_FREEZE_BLOCKS=0
export TTT_MOMENTUM=0.9
export TTT_BATCH_SEQS=32
export TTT_GRAD_CLIP=1.0

nice -n 10 torchrun --standalone --nproc_per_node=1 "$SCRIPT" \
    2>&1 | tee "$RESULTS/overnight_control.log"
echo "$(date) === RUN 1 DONE ==="

# ============================================================
# RUN 2: Muon+ only
# ============================================================
echo "$(date) === RUN 2/4: Muon+ (TTT epochs=3, freeze=0) ==="
export MUON_PLUS=1
export TTT_ENABLED=1
export TTT_LR=0.002
export TTT_EPOCHS=3
export TTT_FREEZE_BLOCKS=0

nice -n 10 torchrun --standalone --nproc_per_node=1 "$SCRIPT" \
    2>&1 | tee "$RESULTS/overnight_muon_plus.log"
echo "$(date) === RUN 2 DONE ==="

# ============================================================
# RUN 3: Muon+ + TTT tuning (fewer epochs, some freezing)
# ============================================================
echo "$(date) === RUN 3/4: Muon+ + TTT tuned (epochs=2, freeze=4) ==="
export MUON_PLUS=1
export TTT_ENABLED=1
export TTT_LR=0.001
export TTT_EPOCHS=2
export TTT_FREEZE_BLOCKS=4

nice -n 10 torchrun --standalone --nproc_per_node=1 "$SCRIPT" \
    2>&1 | tee "$RESULTS/overnight_muon_plus_ttt_tuned.log"
echo "$(date) === RUN 3 DONE ==="

# ============================================================
# RUN 4: Muon+ + aggressive TTT (higher LR, query-only feel)
# ============================================================
echo "$(date) === RUN 4/4: Muon+ + TTT aggressive (epochs=1, lr=0.003, freeze=6) ==="
export MUON_PLUS=1
export TTT_ENABLED=1
export TTT_LR=0.003
export TTT_EPOCHS=1
export TTT_FREEZE_BLOCKS=6

nice -n 10 torchrun --standalone --nproc_per_node=1 "$SCRIPT" \
    2>&1 | tee "$RESULTS/overnight_muon_plus_ttt_aggressive.log"
echo "$(date) === RUN 4 DONE ==="

# ============================================================
# Summary
# ============================================================
echo ""
echo "$(date) ====== ALL RUNS COMPLETE ======"
echo ""
for f in "$RESULTS"/overnight_*.log; do
    name=$(basename "$f" .log)
    # Extract key metrics
    pre_quant=$(grep "stopping_early" "$f" | head -1)
    val_bpb=$(grep "^step:.*val_bpb" "$f" | tail -1 | grep -oP 'val_bpb:\K[0-9.]+')
    sliding=$(grep "final_int6_sliding_window_exact" "$f" | grep -oP 'val_bpb:\K[0-9.]+')
    ttt=$(grep "legal_ttt_exact" "$f" | grep -oP 'val_bpb:\K[0-9.]+')
    artifact=$(grep "Total submission size" "$f" | grep -oP ': \K[0-9]+')
    echo "$name: val_bpb=$val_bpb sliding=$sliding ttt=$ttt artifact=$artifact"
done
echo ""
echo "Results saved to: $RESULTS/overnight_*.log"
