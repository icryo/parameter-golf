#!/bin/bash
# CLEAN deploy: PR #1060's proven code + FA3 + tighter GPTQ
# NO ported innovations (EngramLite, gated skips, etc. — they HURT)
#
# Expected: ~1.1115-1.1122 BPP (PR #1060 published 1.1122)
# With GPTQ_RESERVE_MS=9000: +57 steps → potentially better
set -euo pipefail

echo "=== Clean PR #1060 Deploy + FA3 ==="
nvidia-smi -L | head -1
NGPU=$(nvidia-smi -L | wc -l)

python3 -m venv .venv
source .venv/bin/activate
pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128 2>&1 | tail -1
pip install wheel packaging ninja numpy sentencepiece huggingface-hub datasets tqdm 2>&1 | tail -1
pip install --no-cache-dir "https://download.pytorch.org/whl/cu128/flash_attn_3-3.0.0-cp39-abi3-manylinux_2_28_x86_64.whl" 2>&1 | tail -1

# Remove any FA shim directory
rm -rf flash_attn_interface/

python3 -c "
import torch; print(f'PyTorch: {torch.__version__}')
from flash_attn_interface import flash_attn_func
q = torch.randn(1,64,8,64,device='cuda',dtype=torch.bfloat16)
print(f'FA3: {flash_attn_func(q,q[:,:,:4],q[:,:,:4],causal=True).shape} OK')
"

python3 data/cached_challenge_fineweb.py --variant sp1024 2>&1 | tail -1
echo "Shards: $(ls data/datasets/fineweb10B_sp1024/fineweb_train_*.bin | wc -l)"

echo "=== Training: PR #1060 exact config + GPTQ_RESERVE=9s ==="
export SEED="${SEED:-1337}"
export RUN_ID="clean_s${SEED}"
export DATA_PATH="./data/datasets/fineweb10B_sp1024"
export TOKENIZER_PATH="./data/tokenizers/fineweb_1024_bpe.model"
export MAX_WALLCLOCK_SECONDS=600
export ITERATIONS=9000 WARMUP_STEPS=20 WARMDOWN_ITERS=3500
export TRAIN_BATCH_TOKENS=786432 TRAIN_SEQ_LEN=2048
export EVAL_SEQ_LEN=2048 EVAL_STRIDE=64
export NUM_LAYERS=11 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=3
export TIE_EMBEDDINGS=1 ROPE_DIMS=16 LN_SCALE=1
export VE_ENABLED=1 VE_DIM=128 VE_LAYERS="9,10"
export LOGIT_SOFTCAP=30.0
export MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035
export MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 MUON_MOMENTUM_WARMUP_STEPS=1500
export MUON_WD=0.04 ADAM_WD=0.04 GRAD_CLIP_NORM=0.3
export SWA_ENABLED=1 SWA_EVERY=50
export LATE_QAT_THRESHOLD=0.15
# PR #1060 runtime config
export XSA_LAST_N=11
export BIGRAM_VOCAB_SIZE=2816
export BIGRAM_DIM=112
export USE_GPTQ=1
export GPTQ_RESERVE_MS=9000  # tighter than their 14000
export TTT_ENABLED=0

# Use PR #1060's ORIGINAL code (not our merged version)
torchrun --standalone --nproc_per_node=$NGPU train_gpt_pr1060.py

echo "=== Complete ==="
grep "final_int6_sliding_window_exact" logs/clean_s${SEED}.txt 2>/dev/null
