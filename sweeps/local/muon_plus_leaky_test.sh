#!/bin/bash
# Test Muon+ with leaky05 — single GPU, 600s wallclock each, 3 configs
# Expected: ~3 runs × (~10min train + ~20min eval + ~5min GPTQ) ≈ 1.5-2 hours total

cd ${PROJECT_ROOT:-.}/parameter-golf
source ${VENV:-./venv}/bin/activate

SCRIPT="${PROJECT_ROOT:-.}/parameter-golf/records/track_10min_16mb/2026-03-27_ActivationSearch/train_gpt.py"
RESULTS="${PROJECT_ROOT:-.}/local_tests"

export CUDA_VISIBLE_DEVICES=0
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
export MAX_WALLCLOCK_SECONDS=600

echo "$(date) ====== MUON+ LEAKY05 TEST START ======"
echo ""

# Run 1: leaky05, no Muon+ (control)
echo "$(date) === RUN 1/3: leaky05, no Muon+ (control) ==="
export MLP_ACTIVATION=leaky05
export MUON_PLUS=0
nice -n 10 torchrun --standalone --nproc_per_node=1 --master_port=29500 "$SCRIPT" \
    > "$RESULTS/leaky05_control.log" 2>&1
echo "$(date) === RUN 1 DONE ==="

# Run 2: leaky05, Muon+ (THE NEW TEST)
echo "$(date) === RUN 2/3: leaky05 + Muon+ ==="
export MLP_ACTIVATION=leaky05
export MUON_PLUS=1
nice -n 10 torchrun --standalone --nproc_per_node=1 --master_port=29500 "$SCRIPT" \
    > "$RESULTS/leaky05_muon_plus.log" 2>&1
echo "$(date) === RUN 2 DONE ==="

# Run 3: mish2, Muon+ (reference, single GPU)
echo "$(date) === RUN 3/3: mish2 + Muon+ (reference) ==="
export MLP_ACTIVATION=mish2
export MUON_PLUS=1
nice -n 10 torchrun --standalone --nproc_per_node=1 --master_port=29500 "$SCRIPT" \
    > "$RESULTS/mish2_muon_plus_solo.log" 2>&1
echo "$(date) === RUN 3 DONE ==="

# Summary
echo ""
echo "$(date) ====== RESULTS ======"
echo ""

for f in "$RESULTS/leaky05_control.log" \
         "$RESULTS/leaky05_muon_plus.log" \
         "$RESULTS/mish2_muon_plus_solo.log"; do
    name=$(basename "$f" .log)
    steps=$(grep "stopping_early" "$f" | grep -oP 'step:\K[0-9]+' 2>/dev/null || echo "?")
    avg=$(grep "stopping_early" "$f" | grep -oP 'step_avg:\K[0-9.]+' 2>/dev/null || echo "?")
    pre=$(grep "^step:.*val_bpb" "$f" | tail -1 | grep -oP 'val_bpb:\K[0-9.]+' 2>/dev/null || echo "?")
    ema=$(grep "DIAGNOSTIC post_ema" "$f" | grep -oP 'val_bpb:\K[0-9.]+' 2>/dev/null || echo "?")
    sliding=$(grep "final_int6_sliding_window_exact" "$f" | grep -oP 'val_bpb:\K[0-9.]+' 2>/dev/null || echo "?")
    ttt=$(grep "legal_ttt_exact" "$f" | grep -oP 'val_bpb:\K[0-9.]+' 2>/dev/null || echo "?")
    echo "  $name: steps=$steps avg=${avg}ms pre=$pre ema=$ema sliding=$sliding ttt=$ttt"
done

echo ""
echo "$(date) ====== DONE ======"
