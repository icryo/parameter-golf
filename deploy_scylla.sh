#!/bin/bash
# Deploy: Scylla tokenizer + PR #1060 stack + FA3
# Two phases: retokenize (once), then train
#
# Phase 1: Retokenize FineWeb with Scylla vocab (~2-3 hours)
# Phase 2: Train with retokenized data (~15 min)
#
# Usage:
#   bash deploy_scylla.sh retokenize   # Phase 1 (do this first, once)
#   bash deploy_scylla.sh train        # Phase 2 (repeat for each seed)
set -euo pipefail

PHASE="${1:-train}"
SEED="${SEED:-1337}"
NGPU=$(nvidia-smi -L 2>/dev/null | wc -l)

# Setup venv if not exists
if [ ! -d .venv ]; then
    python3 -m venv .venv
    source .venv/bin/activate
    pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128 2>&1 | tail -1
    pip install wheel packaging ninja numpy sentencepiece huggingface-hub datasets tqdm tokenmonster 2>&1 | tail -1
    pip install --no-cache-dir "https://download.pytorch.org/whl/cu128/flash_attn_3-3.0.0-cp39-abi3-manylinux_2_28_x86_64.whl" 2>&1 | tail -1
    rm -rf flash_attn_interface/
    python3 -c "from flash_attn_interface import flash_attn_func; print('FA3 OK')"
else
    source .venv/bin/activate
fi

case "$PHASE" in
    retokenize)
        echo "=== Phase 1: Retokenize FineWeb with Scylla vocab ==="
        echo "This takes 2-3 hours. Run once, then use 'train' for each seed."

        # Download raw docs first (if not already)
        if [ ! -f data/raw/docs_selected.jsonl ]; then
            echo "Downloading raw FineWeb docs..."
            python3 data/cached_challenge_fineweb.py --variant sp1024 --with-docs 2>&1 | tail -3
        fi

        # Also download sp1024 shards (for fallback testing)
        python3 data/cached_challenge_fineweb.py --variant sp1024 2>&1 | tail -1

        # Retokenize with Scylla vocab
        echo "Retokenizing with Scylla vocab (998 tokens)..."
        python3 retokenize.py \
            --vocab scylla_candidate.vocab \
            --vocab-name scylla_tm0054 \
            --output-dir data/datasets/fineweb10B_scylla \
            --cache-dir data/raw

        # Copy the validated metadata
        cp scylla_candidate.meta.npz data/datasets/fineweb10B_scylla/candidate.meta.npz
        cp scylla_candidate.vocab data/datasets/fineweb10B_scylla/candidate.vocab

        echo "=== Retokenization complete ==="
        echo "Train shards: $(ls data/datasets/fineweb10B_scylla/fineweb_train_*.bin 2>/dev/null | wc -l)"
        echo "Val shards: $(ls data/datasets/fineweb10B_scylla/fineweb_val_*.bin 2>/dev/null | wc -l)"
        ;;

    train)
        echo "=== Phase 2: Train with Scylla tokenizer + PR #1060 stack ==="
        echo "Seed: $SEED | GPUs: $NGPU"

        # Check retokenized data exists
        if [ ! -d data/datasets/fineweb10B_scylla ]; then
            echo "ERROR: Retokenized data not found. Run: bash deploy_scylla.sh retokenize"
            exit 1
        fi

        export RUN_ID="scylla_s${SEED}"
        export SEED="$SEED"
        export DATA_PATH="./data/datasets/fineweb10B_scylla"
        export TOKENIZER_PATH="./data/datasets/fineweb10B_scylla/candidate.vocab"
        export TOKENIZER_META_PATH="./data/datasets/fineweb10B_scylla/candidate.meta.npz"
        export VOCAB_SIZE=998
        export MAX_WALLCLOCK_SECONDS=600
        export ITERATIONS=9000 WARMUP_STEPS=20 WARMDOWN_ITERS=3500
        export TRAIN_BATCH_TOKENS=786432 TRAIN_SEQ_LEN=2048
        export EVAL_SEQ_LEN=2048 EVAL_STRIDE=64
        export NUM_LAYERS=11 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=3
        export TIE_EMBEDDINGS=1 ROPE_DIMS=16 LN_SCALE=1
        export VE_ENABLED=1 VE_DIM=128 VE_LAYERS="7,9,10"
        export LOGIT_SOFTCAP=30.0
        export MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035
        export MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 MUON_MOMENTUM_WARMUP_STEPS=1500
        export MUON_WD=0.04 ADAM_WD=0.04 GRAD_CLIP_NORM=0.3
        export SWA_ENABLED=1 SWA_EVERY=50
        export LATE_QAT_THRESHOLD=0.15
        export XSA_LAST_N=11
        export BIGRAM_VOCAB_SIZE=4096
        export BIGRAM_DIM=112
        export USE_GPTQ=1 GPTQ_RESERVE_MS=9000
        # Tuned from local sweeps (10.5% BPB improvement locally):
        export TIED_EMBED_INIT_STD=0.015
        export QK_GAIN_INIT=4.0
        # No TTT (neutral on this stack)
        export TTT_ENABLED=0

        torchrun --standalone --nproc_per_node=$NGPU train_gpt_scylla_stack.py
        ;;

    *)
        echo "Usage: bash deploy_scylla.sh [retokenize|train]"
        exit 1
        ;;
esac
