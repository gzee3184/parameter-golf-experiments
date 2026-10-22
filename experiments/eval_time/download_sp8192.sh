#!/bin/bash
# Wrapper: sets HF env vars BEFORE Python imports huggingface_hub,
# so the cache lives on scratch, not home.
export HF_HOME=${HF_HOME:-~/.cache/huggingface}
export HUGGINGFACE_HUB_CACHE=${HF_HOME:-~/.cache/huggingface}/hub
export HF_HUB_CACHE=${HF_HOME:-~/.cache/huggingface}/hub
mkdir -p "$HUGGINGFACE_HUB_CACHE"

cd ${PROJECT_ROOT:-.}
source venv/bin/activate
N_SHARDS="${N_SHARDS:-10}" nice -n 10 python3 h100_deploy/experiments/download_sp8192.py
