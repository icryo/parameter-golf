#!/bin/bash
# Deploy script that builds FlashAttention 3 from source
# For vast.ai or any 8xH100 node with CUDA 12.x
#
# FA3 install: clone flash-attention repo, build hopper/ subdirectory
# This creates the flash_attn_3 package + flash_attn_interface.py
#
# Usage: ssh into node, then:
#   git clone https://github.com/icryo/parameter-golf.git
#   cd parameter-golf && git checkout experiments/triton-kernels-qat-fix
#   bash deploy_fa3.sh 2>&1 | tee deploy.log
set -euo pipefail

echo "=== Parameter Golf Deploy (with FA3 build) ==="
echo "$(date)"

# Check GPUs
nvidia-smi -L
NGPU=$(nvidia-smi -L | wc -l)
echo "GPUs: $NGPU"

# Find system CUDA version
CUDA_VER=$(nvcc --version 2>/dev/null | grep "release" | sed 's/.*release //' | sed 's/,.*//' || echo "unknown")
echo "System CUDA: $CUDA_VER"

# Create venv
echo ""
echo "=== Setting up Python environment ==="
python3 -m venv .venv
source .venv/bin/activate

# Install PyTorch matching system CUDA
# Map CUDA version to PyTorch index
case "$CUDA_VER" in
    12.6*|12.7*|12.8*)
        PIP_INDEX="https://download.pytorch.org/whl/cu126"
        ;;
    12.4*|12.5*)
        PIP_INDEX="https://download.pytorch.org/whl/cu124"
        ;;
    13.*)
        # CUDA 13 — use cu126 and patch the version check
        PIP_INDEX="https://download.pytorch.org/whl/cu126"
        ;;
    *)
        PIP_INDEX="https://download.pytorch.org/whl/cu126"
        echo "WARNING: Unknown CUDA $CUDA_VER, trying cu126"
        ;;
esac

pip install torch --index-url "$PIP_INDEX" 2>&1 | tail -2
pip install wheel packaging ninja numpy sentencepiece huggingface-hub datasets tqdm 2>&1 | tail -2

TORCH_VER=$(python3 -c "import torch; print(torch.__version__)")
TORCH_CUDA=$(python3 -c "import torch; print(torch.version.cuda)")
echo "PyTorch: $TORCH_VER (CUDA $TORCH_CUDA)"

# Patch CUDA version check if system CUDA != PyTorch CUDA
if [ "$CUDA_VER" != "$TORCH_CUDA" ]; then
    echo "Patching CUDA version check (system=$CUDA_VER, torch=$TORCH_CUDA)..."
    CPEXT=$(python3 -c "import torch.utils.cpp_extension as c; import inspect; print(inspect.getfile(c))")
    sed -i 's/raise RuntimeError(CUDA_MISMATCH_MESSAGE/pass  # patched: raise RuntimeError(CUDA_MISMATCH_MESSAGE/' "$CPEXT"
fi

# Build FlashAttention 3 from source
echo ""
echo "=== Building FlashAttention 3 (Hopper kernels) ==="
cd /tmp
if [ ! -d flash-attention ]; then
    git clone https://github.com/Dao-AILab/flash-attention.git --depth 1
fi
cd flash-attention/hopper

# Build and install
pip install . --no-build-isolation 2>&1 | tail -5
FA3_OK=$(python3 -c "import flash_attn_3; print('YES')" 2>/dev/null || echo "NO")
echo "flash_attn_3 package: $FA3_OK"

if [ "$FA3_OK" = "YES" ]; then
    # Copy flash_attn_interface.py to our project (it imports flash_attn_3)
    cp /tmp/flash-attention/hopper/flash_attn_interface.py /workspace/parameter-golf/flash_attn_interface.py
    echo "flash_attn_interface.py copied"
    python3 -c "
import sys; sys.path.insert(0, '/workspace/parameter-golf')
from flash_attn_interface import flash_attn_func
import torch
q = torch.randn(1, 64, 8, 64, device='cuda', dtype=torch.bfloat16)
k = torch.randn(1, 64, 4, 64, device='cuda', dtype=torch.bfloat16)
v = torch.randn(1, 64, 4, 64, device='cuda', dtype=torch.bfloat16)
out = flash_attn_func(q, k, v, causal=True)
print(f'FA3 forward test: {out.shape} OK')
" && echo "=== FA3 WORKING ===" || echo "=== FA3 FAILED, falling back to shim ==="
fi

cd /workspace/parameter-golf

# If FA3 didn't work, create fallback shim
if [ ! -f flash_attn_interface.py ] || ! python3 -c "from flash_attn_interface import flash_attn_func" 2>/dev/null; then
    echo "Creating FA2/SDPA fallback shim..."
    mkdir -p flash_attn_interface
    # Try FA2 first, then SDPA
    python3 -c "from flash_attn import flash_attn_func; print('FA2 available')" 2>/dev/null && {
        cat > flash_attn_interface/__init__.py << 'SHIMEOF'
from flash_attn import flash_attn_func as _fa2
def flash_attn_func(q, k, v, causal=False):
    return _fa2(q, k, v, causal=causal)
SHIMEOF
        echo "Using FA2 shim"
    } || {
        cat > flash_attn_interface/__init__.py << 'SHIMEOF'
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
SHIMEOF
        echo "Using SDPA shim (slowest)"
    }
fi

# Download data
echo ""
echo "=== Downloading data ==="
python3 data/cached_challenge_fineweb.py --variant sp1024 2>&1 | tail -1
NSHARDS=$(ls data/datasets/fineweb10B_sp1024/fineweb_train_*.bin 2>/dev/null | wc -l)
echo "Train shards: $NSHARDS"

# Final environment check
echo ""
echo "=== Environment Summary ==="
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
try:
    from flash_attn_interface import flash_attn_func
    # Check if it's real FA3 or a shim
    import inspect
    src = inspect.getfile(flash_attn_func)
    if 'flash_attn_3' in open(src).read():
        print('Attention: FA3 (Hopper native) ← FAST')
    elif 'flash_attn' in open(src).read():
        print('Attention: FA2 shim ← MEDIUM')
    else:
        print('Attention: SDPA shim ← SLOW')
except:
    print('Attention: UNKNOWN')
try:
    from triton.tools.tensor_descriptor import TensorDescriptor
    print('Fused MLP: TMA available')
except:
    print('Fused MLP: TMA not available')
"

# Run training
echo ""
echo "=== Starting training ==="

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
export MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 MUON_MOMENTUM_WARMUP_STEPS=1500
export MUON_WD=0.04 ADAM_WD=0.04 GRAD_CLIP_NORM=0.3
export SWA_ENABLED=1 SWA_EVERY=50
export LATE_QAT_THRESHOLD=0.15
export XSA_LAST_N=11
export BIGRAM_VOCAB_SIZE=2816 BIGRAM_DIM=112
export USE_GPTQ=1 GPTQ_RESERVE_MS=14000
export TTT_ENABLED=0

torchrun --standalone --nproc_per_node=$NGPU train_gpt_merged.py

echo ""
echo "=== Complete ==="
grep "final_int6_sliding_window_exact" logs/fa3_s${SEED}.txt 2>/dev/null || echo "Check logs/ manually"
