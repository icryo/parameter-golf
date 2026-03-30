#!/bin/bash
# Quick deploy for vast.ai PyTorch template (should have FA3 pre-installed)
# Usage: ssh into node, then: curl -sL <raw_url> | bash
# Or: git clone ... && bash deploy.sh
set -euo pipefail

echo "=== Parameter Golf Deployment ==="
echo "$(date)"

# Check GPUs
echo ""
echo "GPUs:"
nvidia-smi -L
NGPU=$(nvidia-smi -L | wc -l)
echo "Count: $NGPU"

# Check PyTorch + FA3
echo ""
echo "Checking environment..."
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
try:
    from flash_attn_interface import flash_attn_func
    print('FlashAttention 3: YES (native)')
except ImportError:
    try:
        from flash_attn import flash_attn_func
        print('FlashAttention 2: YES (will create FA3 shim)')
    except ImportError:
        print('FlashAttention: NONE (will use SDPA shim)')
try:
    from triton.tools.tensor_descriptor import TensorDescriptor
    print('Triton TMA: YES (fused kernel available)')
except ImportError:
    print('Triton TMA: NO (fused kernel will be disabled)')
"

# Install missing deps if needed
echo ""
echo "Installing dependencies..."
pip install sentencepiece huggingface-hub datasets numpy tqdm 2>/dev/null | tail -1

# Create FA3 shim if native FA3 not available
python3 -c "from flash_attn_interface import flash_attn_func" 2>/dev/null || {
    echo "Creating FA3 compatibility shim..."
    mkdir -p flash_attn_interface
    python3 -c "
try:
    # Try FA2 -> FA3 shim
    from flash_attn import flash_attn_func
    shim = '''
from flash_attn import flash_attn_func as _fa2
def flash_attn_func(q, k, v, causal=False):
    return _fa2(q, k, v, causal=causal)
'''
    print('Using FA2 backend')
except ImportError:
    # SDPA fallback
    shim = '''
import torch, torch.nn.functional as F
def flash_attn_func(q, k, v, causal=False):
    B, T, H, D = q.shape
    Hkv = k.shape[2]
    group = H // Hkv
    if group > 1:
        k = k.unsqueeze(3).expand(B, T, Hkv, group, D).reshape(B, T, H, D)
        v = v.unsqueeze(3).expand(B, T, Hkv, group, D).reshape(B, T, H, D)
    q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    return F.scaled_dot_product_attention(q, k, v, is_causal=causal).transpose(1, 2)
'''
    print('Using SDPA fallback (slower)')
with open('flash_attn_interface/__init__.py', 'w') as f:
    f.write(shim)
"
}

# Download data
echo ""
echo "Downloading data (80 train shards + 1 val)..."
python3 data/cached_challenge_fineweb.py --variant sp1024 2>&1 | tail -1
NSHARDS=$(ls data/datasets/fineweb10B_sp1024/fineweb_train_*.bin 2>/dev/null | wc -l)
echo "Train shards: $NSHARDS"

# Run
echo ""
echo "=== Starting training ==="
echo "GPUs: $NGPU | Script: train_gpt_merged.py"
echo ""

export SEED="${SEED:-1337}"
export RUN_ID="run_s${SEED}"
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
export XSA_LAST_N=11
export BIGRAM_VOCAB_SIZE=2816 BIGRAM_DIM=112
export USE_GPTQ=1 GPTQ_RESERVE_MS=14000
export TTT_ENABLED=0

torchrun --standalone --nproc_per_node=$NGPU train_gpt_merged.py 2>&1 | tee training.log

echo ""
echo "=== Complete ==="
echo "Check logs/ for detailed output"
grep "final_int6_sliding_window_exact" logs/run_s${SEED}.txt 2>/dev/null || echo "Check logs/run_s${SEED}.txt manually"
