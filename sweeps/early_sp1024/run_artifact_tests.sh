#!/usr/bin/env bash
set -euo pipefail

# Quick artifact size tests: measure compression with Int5-MLP and Hadamard+Int5
# Short runs (600s) — we only care about artifact size, not final BPB

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TRAIN_SCRIPT="$SCRIPT_DIR/2026-03-27_ActivationSearch/train_gpt.py"
RESULTS_DIR="$SCRIPT_DIR/artifact_test_results"
VENV="${VENV:-./venv}"

mkdir -p "$RESULTS_DIR"
source "$VENV/bin/activate"
cd ${PROJECT_ROOT:-.}/parameter-golf

COMMON_VARS=(
    SEED=1337
    NUM_LAYERS=11 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=3.0
    BIGRAM_VOCAB_SIZE=1536 BIGRAM_DIM=128
    ROPE_DIMS=16 LN_SCALE=1
    VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10
    SWA_ENABLED=1 SWA_EVERY=50
    MUON_WD=0.04 ADAM_WD=0.04
    MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035
    MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 MUON_MOMENTUM_WARMUP_STEPS=1500
    WARMDOWN_ITERS=3500 ITERATIONS=9000
    MAX_WALLCLOCK_SECONDS=2400
    LATE_QAT_THRESHOLD=0.15 EVAL_STRIDE=64
    TTT_ENABLED=0 MTP_NUM_HEADS=0 XSA_LAST_N=4
    VAL_LOSS_EVERY=50 TRAIN_LOG_EVERY=10
    MLP_ACTIVATION=mish2
)

run_variant() {
    local name="$1" gpu="$2" port="$3"
    shift 3
    local extra_vars=("$@")
    local logfile="$RESULTS_DIR/${name}.log"

    echo "[$(date '+%H:%M:%S')] START $name on GPU$gpu"
    env "${COMMON_VARS[@]}" "${extra_vars[@]}" \
        CUDA_VISIBLE_DEVICES="$gpu" \
        nice -n 10 \
        torchrun --standalone --nproc_per_node=1 --master_port="$port" \
        "$TRAIN_SCRIPT" > "$logfile" 2>&1

    local artifact=$(grep "Total submission size" "$logfile" | grep -oP '\d+ bytes' | grep -oP '\d+')
    local roundtrip=$(grep "final_int6_roundtrip_exact" "$logfile" | grep -oP 'val_bpb:\K[\d.]+')
    local sliding=$(grep "final_int6_sliding_window_exact" "$logfile" | grep -oP 'val_bpb:\K[\d.]+')
    echo "[$(date '+%H:%M:%S')] DONE $name — artifact=$artifact roundtrip=$roundtrip sliding=$sliding"
}

echo "============================================="
echo "  Artifact Size Tests"
echo "  $(date)"
echo "============================================="

# GPU0: mish2 + Int5-MLP (no Hadamard)
# GPU1: mish2 + Int5-MLP + Hadamard
run_variant "mish2_int5mlp" 0 29500 \
    MLP_QUANT_BITS=5 HADAMARD_QUANT=0 &

run_variant "mish2_int5mlp_had" 1 29501 \
    MLP_QUANT_BITS=5 HADAMARD_QUANT=1 &

wait

echo ""
echo "============================================="
echo "  ARTIFACT RESULTS SUMMARY"
echo "============================================="
echo "  Reference: mish2 Int6 (A100 full): artifact=16,648,933 sliding=1.1445"
echo "  Reference: mish2 Int6 (local): artifact=~9,300,000 sliding=1.3652"
echo "  Reference: mish2 Int6+Had (local): artifact=9,671,645 sliding=1.3643"
echo ""
for name in mish2_int5mlp mish2_int5mlp_had; do
    logfile="$RESULTS_DIR/${name}.log"
    artifact=$(grep "Total submission size" "$logfile" | grep -oP '\d+ bytes' | grep -oP '\d+')
    roundtrip=$(grep "final_int6_roundtrip_exact" "$logfile" | grep -oP 'val_bpb:\K[\d.]+')
    sliding=$(grep "final_int6_sliding_window_exact" "$logfile" | grep -oP 'val_bpb:\K[\d.]+')
    printf "  %-25s  artifact=%s  roundtrip=%s  sliding=%s\n" "$name" "${artifact:-N/A}" "${roundtrip:-N/A}" "${sliding:-N/A}"
done

echo ""
echo "Done at $(date)"
