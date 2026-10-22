#!/bin/bash
# v11 Scaling Test — fill 0.77MB+ budget headroom with LZMA wrapper
# GPU 0: 12 layers (11→12), est -0.008 to -0.012 BPB
# GPU 1: REPEAT_UNTIE_MLP=down (full untied down-proj, competitor approach), est -0.003 to -0.006 BPB
# Both use LZMA code budget: 16,000,000 - 22,000 = 15,978,000 bytes
# Both use Ffix config + int5 attn (our Run #3 winning config)
set -uo pipefail
source ${VENV:-./venv}/bin/activate

REPO_DIR="${PROJECT_ROOT:-.}/parameter-golf"
SCRIPT_DIR="${PROJECT_ROOT:-.}/h100_final"
RESULTS_DIR="${PROJECT_ROOT:-.}/local_tests/v11_scale"
mkdir -p "$RESULTS_DIR"
cd "$REPO_DIR"

cp "$SCRIPT_DIR/train_gpt_v8_experimental.py" "$REPO_DIR/train_gpt.py"

# Common Ffix + int5 attn config (Run #3 winning config)
export VOCAB_SIZE=4096 NUM_LAYERS=11 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=4
export MLP_ACTIVATION=leaky05
export DATA_PATH="./data/datasets/fineweb10B_sp4096"
export TOKENIZER_PATH="./data/tokenizers/fineweb_4096_bpe.model"
export MAX_WALLCLOCK_SECONDS=10800
export SEED=1337 TRAIN_BATCH_TOKENS=786432 TRAIN_SEQ_LEN=2048 EVAL_SEQ_LEN=2048
export ITERATIONS=20000 WARMUP_STEPS=20
export VAL_LOSS_EVERY=4000 TRAIN_LOG_EVERY=100
export MUON_PLUS=1 MATRIX_LR=0.02 SCALAR_LR=0.02
export MUON_WD=0.090 ADAM_WD=0.02 EMBED_WD=0.090
export QK_GAIN_INIT=5.0 GRAD_CLIP_NORM=0.3
export SIGMOID_SKIP=1 BYTE_SHUFFLE=1 COPRIME_STRIDE=1
export RECUR_LAYERS="3,4,5" RECUR_NUM_LOOPS=2
export VE_ENABLED=1 VE_DIM=128 VE_LAYERS="9,10"
export PARALLEL_START_LAYER=7 DISABLE_LAYER0_ATTN=1
export XSA_LAST_N=11 TIE_EMBEDDINGS=1
export ATTN_RES=0 RECUR_SKIP_PHYSICAL=0
export EMA_DECAY=0.9965
export RECUR_START_STEP=2000 REPEAT_PASS_SCALE=1
export GPTQ_ENABLED=1 GPTQ_CALIB_SEQS=64 COMPRESSOR=brotli
export QUANT_BITS_MLP=6 QUANT_BITS_ATTN=5 QUANT_BITS_EMBED=8
# Ffix winning settings
export ROPE_BASE=10000 LOGIT_SOFTCAP=20 LN_SCALE=0 WARMDOWN_ITERS=3500
export DATA_WARMDOWN_COUPLING=0
# Default: LoRA r=16
export PER_LAYER_LANE_MERGE=0
export REPEAT_UNTIE_MLP=lora REPEAT_LORA_RANK=16
export REPEAT_ATTN_LORA=0 DISABLE_LAYER0_MLP=0 SKIP_REPEAT_ATTN=0
export PARALLEL_SCALAR_GRAD_SCALE=0
export LATE_QAT=0 SWA_ENABLED=0 SDCLIP=0
# Sliding window eval
export SLIDING_WINDOW_EVAL=1 EVAL_STRIDE=64

# ── GPU 0: 12 layers (budget with LZMA code) ──
(
  export CUDA_VISIBLE_DEVICES=0
  export NUM_LAYERS=12
  # 12L: adjust VE_LAYERS to last 2 = "10,11", PARALLEL_START_LAYER=8, XSA_LAST_N=12
  export VE_LAYERS="10,11"
  export PARALLEL_START_LAYER=8
  export XSA_LAST_N=12
  # LZMA code budget: 22KB code
  export CODE_SIZE_OVERRIDE=22000
  LOGFILE="$RESULTS_DIR/12layers_$(date +%Y%m%d_%H%M%S).log"
  echo "$(date) ── 12 layers test on GPU 0 ──"
  echo "Config: NUM_LAYERS=12, VE_LAYERS=10,11, PARALLEL_START=8, XSA_LAST_N=12"
  echo "Log: $LOGFILE"
  nice -n 10 torchrun --standalone --nproc_per_node=1 --master_port=29500 train_gpt.py 2>&1 | tee "$LOGFILE"
) &
PID0=$!

sleep 5  # let first process bind its port

# ── GPU 1: Full untied MLP repeat (budget with LZMA code) ──
(
  export CUDA_VISIBLE_DEVICES=1
  export REPEAT_UNTIE_MLP=down
  # Keep LoRA rank setting (ignored when untie=down, but harmless)
  # LZMA code budget: 22KB code
  export CODE_SIZE_OVERRIDE=22000
  LOGFILE="$RESULTS_DIR/untie_down_$(date +%Y%m%d_%H%M%S).log"
  echo "$(date) ── untied MLP down test on GPU 1 ──"
  echo "Config: REPEAT_UNTIE_MLP=down (full untied down-projection)"
  echo "Log: $LOGFILE"
  nice -n 10 torchrun --standalone --nproc_per_node=1 --master_port=29501 train_gpt.py 2>&1 | tee "$LOGFILE"
) &
PID1=$!

echo "Waiting for both to complete... GPU0(12L)=$PID0 GPU1(untie_down)=$PID1"
wait $PID0 $PID1
echo ""
echo "=== Both complete ==="
echo "Compare results:"
echo "  grep -E 'model_params|prune|budget|submission|roundtrip_exact|sliding_window_exact|val_bpb' $RESULTS_DIR/*.log"
