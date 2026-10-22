#!/bin/bash
# TTT Hyperparameter Sweep — Local testing on Run #3's final_model_r3.pt
#
# Uses RESUME_QUANT to skip training, loads Run #3's model, runs GPTQ+eval+TTT
# with different TTT configs. Two RTX PRO 6000 GPUs run configs in parallel.
#
# Past data context:
#   v4T model (sp4096, well-trained):
#     AdamW lr=1e-4 freeze=2: 1.1026 BPB (best ever)
#     SGD lr=0.002 freeze=2:  1.1048 BPB (+0.002 worse)
#     AdamW lr=2e-4:          1.1136 BPB (too high LR → catastrophic)
#   H100 Run #5: SGD lr=0.002 → TTT gain -0.0039
#
# NOTE: Local quantization always gives ~43% pruning (weight entropy artifact).
# Absolute TTT BPB values will NOT match H100. We're measuring RELATIVE
# improvement between TTT configs — those rankings should transfer.
#
# Estimated time: ~30 min per config, 2 GPUs parallel → ~1.5h total
set -uo pipefail

BASEDIR=${PROJECT_ROOT:-.}/h100_deploy
SWEEPDIR=${PROJECT_ROOT:-.}/local_tests/ttt_sweep
SCRIPT=$BASEDIR/train_gpt.py
MODEL=$BASEDIR/final_model_r3.pt
DATA=${PROJECT_ROOT:-.}/parameter-golf/data/datasets/fineweb10B_sp4096
TOKENIZER=${PROJECT_ROOT:-.}/parameter-golf/data/tokenizers/fineweb_4096_bpe.model

if [ ! -f "$MODEL" ]; then
    echo "ERROR: $MODEL not found"; exit 1
fi
if [ ! -d "$DATA" ]; then
    echo "ERROR: $DATA not found"; exit 1
fi

echo "$(date) ── TTT Sweep Start ──"
echo "Model: $MODEL ($(du -h "$MODEL" | cut -f1))"
echo "Data: $DATA"
echo ""

# Common env vars — matches Run #8 / Run #3 config exactly
export VOCAB_SIZE=4096 NUM_LAYERS=11 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=4
export MLP_ACTIVATION=leaky05
export DATA_PATH="$DATA"
export TOKENIZER_PATH="$TOKENIZER"
export MAX_WALLCLOCK_SECONDS=600
export SEED=1337 TRAIN_BATCH_TOKENS=786432 TRAIN_SEQ_LEN=2048 EVAL_SEQ_LEN=2048
export ITERATIONS=20000 WARMUP_STEPS=20
export VAL_LOSS_EVERY=4000 TRAIN_LOG_EVERY=100
export MUON_PLUS=1 MATRIX_LR=0.02 SCALAR_LR=0.02
export MUON_WD=0.090 ADAM_WD=0.02 EMBED_WD=0.090
export QK_GAIN_INIT=5.0 GRAD_CLIP_NORM=0.3
export SIGMOID_SKIP=1
export RECUR_LAYERS="3,4,5" RECUR_NUM_LOOPS=2 RECUR_START_STEP=2000
export VE_ENABLED=1 VE_DIM=128 VE_LAYERS="9,10"
export PARALLEL_START_LAYER=7 DISABLE_LAYER0_ATTN=1
export XSA_LAST_N=11 TIE_EMBEDDINGS=1
export REPEAT_PASS_SCALE=1
export EMA_DECAY=0.9965 WARMDOWN_ITERS=3500
export ROPE_BASE=10000
export LOGIT_SOFTCAP=20
export REPEAT_UNTIE_MLP=lora REPEAT_LORA_RANK=16
export QUANT_BITS_MLP=6 QUANT_BITS_ATTN=5 QUANT_BITS_EMBED=8
export GPTQ_ENABLED=1 GPTQ_CALIB_SEQS=64 COMPRESSOR=brotli
export GPTQ_REAL_DATA=0
export BYTE_SHUFFLE=0
export RESUME_QUANT=1
export LATE_QAT=0
export SDCLIP=1 SDCLIP_K=12.85
export SLIDING_WINDOW_EVAL=1 EVAL_STRIDE=64
export CODE_SIZE_OVERRIDE=23229
export TTT_ENABLED=1
export TTT_CHUNK_TOKENS=32768
export TTT_BATCH_SEQS=32
export TTT_GRAD_CLIP=1.0

# ── Sweep Configurations ──
# Format: NAME|OPTIMIZER|LR|MOMENTUM|WD|EPOCHS|FREEZE_BLOCKS
CONFIGS=(
  "sgd_baseline|sgd|0.002|0.9|0.01|3|0"       # Current H100 config (control)
  "adamw_con|adamw|0.0001|0.0|0.01|3|2"        # Best from v4T data
  "adamw_f0|adamw|0.0001|0.0|0.01|3|0"         # No freeze variant
  "adamw_e5|adamw|0.0001|0.0|0.01|5|2"         # More epochs
  "sgd_low|sgd|0.001|0.9|0.01|3|0"             # Lower LR SGD
)

run_config() {
    local cfg="$1"
    local gpu="$2"
    IFS='|' read -r name optim lr mom wd epochs freeze <<< "$cfg"

    local workdir="$SWEEPDIR/$name"
    local logfile="$SWEEPDIR/${name}.log"
    mkdir -p "$workdir"

    # Symlink final_model.pt (RESUME_QUANT loads from CWD)
    ln -sf "$MODEL" "$workdir/final_model.pt"

    echo "$(date) [GPU $gpu] Starting: $name (opt=$optim lr=$lr mom=$mom wd=$wd ep=$epochs freeze=$freeze)"

    # Run from workdir (RESUME_QUANT loads final_model.pt from CWD)
    cd "$workdir"

    CUDA_VISIBLE_DEVICES=$gpu \
    TTT_OPTIMIZER=$optim \
    TTT_LR=$lr \
    TTT_MOMENTUM=$mom \
    TTT_WD=$wd \
    TTT_EPOCHS=$epochs \
    TTT_FREEZE_BLOCKS=$freeze \
    nice -n 10 torchrun --standalone --nproc_per_node=1 "$SCRIPT" \
        2>&1 | tee "$logfile"

    local exit_code=${PIPESTATUS[0]}
    echo "$(date) [GPU $gpu] Done: $name (exit=$exit_code)"

    # Extract key results
    local sliding=$(grep "final_int6_sliding_window_exact" "$logfile" | tail -1 | grep -oP 'val_bpb:\K[0-9.]+')
    local ttt=$(grep "final_int6_ttt_exact" "$logfile" | tail -1 | grep -oP 'val_bpb:\K[0-9.]+')
    local prune=$(grep "prune:" "$logfile" | tail -1)
    echo "  → $name: sliding=$sliding ttt=$ttt $prune"
}

# ── Run pairs in parallel (2 GPUs) ──
echo ""
echo "=== Round 1/3: sgd_baseline (GPU 0) + adamw_con (GPU 1) ==="
(run_config "${CONFIGS[0]}" 0) &
PID0=$!
(run_config "${CONFIGS[1]}" 1) &
PID1=$!
wait $PID0 $PID1

echo ""
echo "=== Round 2/3: adamw_f0 (GPU 0) + adamw_e5 (GPU 1) ==="
(run_config "${CONFIGS[2]}" 0) &
PID0=$!
(run_config "${CONFIGS[3]}" 1) &
PID1=$!
wait $PID0 $PID1

echo ""
echo "=== Round 3/3: sgd_low (GPU 0) ==="
(run_config "${CONFIGS[4]}" 0)

# ── Summary ──
echo ""
echo "============================================"
echo "          TTT SWEEP RESULTS SUMMARY"
echo "============================================"
echo ""
printf "%-15s %-8s %-10s %-5s %-5s %-8s %-12s %-12s\n" \
    "Config" "Optim" "LR" "Ep" "Frz" "Sliding" "TTT_BPB" "TTT_Gain"
printf "%-15s %-8s %-10s %-5s %-5s %-8s %-12s %-12s\n" \
    "-------" "-----" "------" "---" "---" "-------" "-------" "--------"

for cfg in "${CONFIGS[@]}"; do
    IFS='|' read -r name optim lr mom wd epochs freeze <<< "$cfg"
    logfile="$SWEEPDIR/${name}.log"
    sliding=$(grep "final_int6_sliding_window_exact" "$logfile" 2>/dev/null | tail -1 | grep -oP 'val_bpb:\K[0-9.]+' || echo "—")
    ttt=$(grep "final_int6_ttt_exact" "$logfile" 2>/dev/null | tail -1 | grep -oP 'val_bpb:\K[0-9.]+' || echo "—")
    if [[ "$sliding" != "—" && "$ttt" != "—" ]]; then
        gain=$(python3 -c "print(f'{float(\"$ttt\")-float(\"$sliding\"):.6f}')" 2>/dev/null || echo "—")
    else
        gain="—"
    fi
    printf "%-15s %-8s %-10s %-5s %-5s %-8s %-12s %-12s\n" \
        "$name" "$optim" "$lr" "$epochs" "$freeze" "$sliding" "$ttt" "$gain"
done

echo ""
echo "$(date) ── TTT Sweep Complete ──"
echo "Logs: $SWEEPDIR/*.log"
