#!/bin/bash
# Deploy: Simplified Scylla stack (PR #1218 philosophy)
# Strip complexity, bigger MLP, higher weight decay for better compression
#
# Key changes from our 0.9485 run:
#   MLP_MULT: 3 → 4 (bigger model)
#   MUON_WD: 0.04 → 0.085 (better compression, fits in 16MB)
#   ADAM_WD: 0.04 → 0.02 (scalars don't need heavy regularization)
#   BIGRAM_VOCAB_SIZE: 2816 → 0 (removed — frees params)
#   VE_ENABLED: 1 → 0 (removed — frees params)
#   QK_GAIN_INIT: 1.5 → 4.0 (validated in PR #1125)
set -euo pipefail

PHASE="${1:-train}"
SEED="${SEED:-1337}"
NGPU=$(nvidia-smi -L 2>/dev/null | wc -l)

if [ ! -d .venv ]; then
    python3 -m venv .venv
    source .venv/bin/activate
    pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128 -q
    pip install wheel packaging ninja numpy sentencepiece huggingface-hub datasets tqdm tokenmonster -q
    pip install --no-cache-dir "https://download.pytorch.org/whl/cu128/flash_attn_3-3.0.0-cp39-abi3-manylinux_2_28_x86_64.whl" -q
    rm -rf flash_attn_interface/
    python3 -c "from flash_attn_interface import flash_attn_func; print('FA3 OK')"
else
    source .venv/bin/activate
fi

case "$PHASE" in
    train)
        echo "=== Simplified Scylla: MLP4x + WD=0.085 ==="
        echo "Seed: $SEED | GPUs: $NGPU"

        if [ ! -d data/datasets/fineweb10B_scylla ]; then
            echo "ERROR: Scylla data not found. Transfer from local."
            exit 1
        fi

        export RUN_ID="simple_s${SEED}"
        export SEED="$SEED"
        export DATA_PATH="./data/datasets/fineweb10B_scylla"
        export TOKENIZER_PATH="./data/datasets/fineweb10B_scylla/candidate.vocab"
        export TOKENIZER_META_PATH="./data/datasets/fineweb10B_scylla/candidate.meta.npz"
        export VOCAB_SIZE=998
        export MAX_WALLCLOCK_SECONDS=600
        export ITERATIONS=9000 WARMUP_STEPS=20 WARMDOWN_ITERS=3500
        export TRAIN_BATCH_TOKENS=786432 TRAIN_SEQ_LEN=2048
        export EVAL_SEQ_LEN=2048 EVAL_STRIDE=64
        export NUM_LAYERS=11 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4
        export MLP_MULT=4              # bigger MLP (was 3)
        export TIE_EMBEDDINGS=1 ROPE_DIMS=16 LN_SCALE=1
        export VE_ENABLED=0            # removed (was 1)
        export LOGIT_SOFTCAP=30.0
        export MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035
        export MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 MUON_MOMENTUM_WARMUP_STEPS=1500
        export MUON_WD=0.085           # higher WD for compression (was 0.04)
        export ADAM_WD=0.02            # lower for scalars (was 0.04)
        export GRAD_CLIP_NORM=0.3
        export SWA_ENABLED=1 SWA_EVERY=50
        export LATE_QAT_THRESHOLD=0.15
        export XSA_LAST_N=11
        export BIGRAM_VOCAB_SIZE=0     # removed (was 2816)
        export BIGRAM_DIM=112
        export USE_GPTQ=1 GPTQ_RESERVE_MS=9000
        export QK_GAIN_INIT=4.0        # validated
        export TTT_ENABLED=0

        torchrun --standalone --nproc_per_node=$NGPU train_gpt_scylla_stack.py
        ;;
    *)
        echo "Usage: SEED=1337 bash deploy_simplified.sh train"
        ;;
esac
