#!/bin/bash
# Deploy: Scylla + Window Attention + Mixed Seq_Len + Eval at 6144
#
# Window attention on layers 2,4,6,8,10 (size=512)
# Mixed training: 5 GPUs at 2048, 3 GPUs at 6144
# Eval at seq_len=6144, stride=128
set -euo pipefail

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

echo "=== Scylla + Window Attention + Mixed Seq_Len ==="
echo "Seed: $SEED | GPUs: $NGPU"

if [ ! -d data/datasets/fineweb10B_scylla ]; then
    echo "ERROR: Scylla data not found. Transfer from local."
    exit 1
fi

export RUN_ID="window_s${SEED}"
export SEED="$SEED"
export DATA_PATH="./data/datasets/fineweb10B_scylla"
export TOKENIZER_PATH="./data/datasets/fineweb10B_scylla/candidate.vocab"
export TOKENIZER_META_PATH="./data/datasets/fineweb10B_scylla/candidate.meta.npz"
export VOCAB_SIZE=998
export MAX_WALLCLOCK_SECONDS=600
export ITERATIONS=9000 WARMUP_STEPS=20 WARMDOWN_ITERS=3500
export TRAIN_BATCH_TOKENS=786432 TRAIN_SEQ_LEN=2048
export EVAL_SEQ_LEN=6144 EVAL_STRIDE=128
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
export USE_GPTQ=1 GPTQ_RESERVE_MS=9000
export TTT_ENABLED=0

# Window attention config
export WINDOW_ATTN_SIZE=512
export WINDOW_ATTN_LAYERS="2,4,6,8,10"

# Mixed seq_len: 5 GPUs at 2048, 3 GPUs at 6144
export TRAIN_SEQ_LEN_LONG=6144
export NUM_GPUS_LONG=3

torchrun --standalone --nproc_per_node=$NGPU train_gpt_window.py
