#!/bin/bash
# Deploy with FlashAttention 3 (pre-built wheel, no compilation)
#
# Key: FA3 = flash_attn_3 package (NOT flash-attn which is FA2)
# Pre-built wheel: https://download.pytorch.org/whl/cu128/flash_attn_3-3.0.0-cp39-abi3-manylinux_2_28_x86_64.whl
# Requires: PyTorch 2.9.x+cu128 (NOT 2.11)
#
# Usage:
#   git clone https://github.com/icryo/parameter-golf.git
#   cd parameter-golf && git checkout experiments/triton-kernels-qat-fix
#   bash deploy_fa3.sh 2>&1 | tee deploy.log
set -euo pipefail

echo "=== Parameter Golf Deploy (FA3 pre-built wheel) ==="
echo "$(date)"

nvidia-smi -L
NGPU=$(nvidia-smi -L | wc -l)
echo "GPUs: $NGPU"

# Setup venv with PyTorch 2.9+cu128 (proven working config for FA3)
echo ""
echo "=== Installing PyTorch 2.9+cu128 + FA3 ==="
python3 -m venv .venv
source .venv/bin/activate

pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128 2>&1 | tail -2
pip install wheel packaging ninja numpy sentencepiece huggingface-hub datasets tqdm brotli 2>&1 | tail -2

# Install FA3 pre-built wheel (seconds, no compilation)
pip install --no-cache-dir "https://download.pytorch.org/whl/cu128/flash_attn_3-3.0.0-cp39-abi3-manylinux_2_28_x86_64.whl" 2>&1 | tail -2

# Remove any flash_attn_interface shim directory (we want the real module from flash_attn_3)
rm -rf flash_attn_interface/

# Verify
echo ""
echo "=== Verification ==="
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}, {torch.cuda.get_device_name(0)}')
from flash_attn_interface import flash_attn_func
print('FA3 (flash_attn_interface): OK')
import torch
q = torch.randn(1, 64, 8, 64, device='cuda', dtype=torch.bfloat16)
k = torch.randn(1, 64, 4, 64, device='cuda', dtype=torch.bfloat16)
v = torch.randn(1, 64, 4, 64, device='cuda', dtype=torch.bfloat16)
out = flash_attn_func(q, k, v, causal=True)
print(f'FA3 forward: {out.shape} OK')
try:
    from triton.tools.tensor_descriptor import TensorDescriptor
    print('Triton TMA: OK (fused kernel available)')
except ImportError:
    print('Triton TMA: NO (fused kernel disabled)')
"

# Download data
echo ""
echo "=== Downloading data (80 shards) ==="
python3 data/cached_challenge_fineweb.py --variant sp1024 2>&1 | tail -1
echo "Shards: $(ls data/datasets/fineweb10B_sp1024/fineweb_train_*.bin | wc -l)"

# Run
echo ""
echo "=== Training (8xH100, expect ~85ms/step with FA3) ==="
export SEED="${SEED:-1337}"
export RUN_ID="fa3_s${SEED}"
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
# Innovations ported from PR #1089 (#1 submission, 1.1086 BPB):
export MUON_BACKEND_STEPS=4        # 4 NS steps (Turbo-Muon)
export NGRAM_BUCKETS=8192          # EngramLite: multi-head bigram+trigram hash
export NGRAM_HEADS=2               # 2 hash heads
export NGRAM_ORDERS=2              # bigram + trigram
export NGRAM_DIM_PER_HEAD=32       # 32-dim per head
export NEGATIVE_SLOPE=0.3          # LeakyReLU(0.3)² (not 0.5)
export LR_FLOOR=0.05              # Warmdown doesn't reach zero
export MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 MUON_MOMENTUM_WARMUP_STEPS=1500
export MUON_WD=0.04 ADAM_WD=0.04 GRAD_CLIP_NORM=0.3
export SWA_ENABLED=1 SWA_EVERY=50
export LATE_QAT_THRESHOLD=0.15
export XSA_LAST_N=11
export BIGRAM_VOCAB_SIZE=2816 BIGRAM_DIM=112
export USE_GPTQ=1 GPTQ_RESERVE_MS=9000
export MIXED_PRECISION=1 TARGET_BYTES=16000000
export TTT_ENABLED=0

torchrun --standalone --nproc_per_node=$NGPU train_gpt_merged.py

echo ""
echo "=== Complete ==="
grep "step_avg" logs/fa3_s${SEED}.txt 2>/dev/null | tail -1
grep "final_int6_sliding_window_exact" logs/fa3_s${SEED}.txt 2>/dev/null
