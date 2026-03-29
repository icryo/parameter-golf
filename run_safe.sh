#!/bin/bash
# Safe experiment runner — minimal changes from SOTA
# Only: fixed QAT + temperature T=0.90 + stride=16 measurement
#
# Usage: ./run_safe.sh [seed]
# For 3-seed submission: for s in 1337 42 2025; do ./run_safe.sh $s; done

set -euo pipefail
SEED="${1:-1337}"
NPROC="${NPROC:-8}"

export RUN_ID="safe_s${SEED}"
export SEED="$SEED"
export DATA_PATH="./data/datasets/fineweb10B_sp1024"
export TOKENIZER_PATH="./data/tokenizers/fineweb_1024_bpe.model"

# Identical to SOTA (#1) submission
export MAX_WALLCLOCK_SECONDS=600
export ITERATIONS=9000
export WARMUP_STEPS=20
export WARMDOWN_ITERS=3500
export TRAIN_BATCH_TOKENS=786432
export TRAIN_SEQ_LEN=2048
export EVAL_SEQ_LEN=2048
export EVAL_STRIDE=64
export NUM_LAYERS=11
export MODEL_DIM=512
export NUM_HEADS=8
export NUM_KV_HEADS=4
export MLP_MULT=3
export TIE_EMBEDDINGS=1
export XSA_LAST_N=4
export ROPE_DIMS=16
export LN_SCALE=1
export VE_ENABLED=1
export VE_DIM=128
export VE_LAYERS="9,10"
export BIGRAM_VOCAB_SIZE=2048
export BIGRAM_DIM=128
export LOGIT_SOFTCAP=30.0
export MATRIX_LR=0.025
export SCALAR_LR=0.025
export TIED_EMBED_LR=0.035
export MUON_MOMENTUM=0.99
export MUON_MOMENTUM_WARMUP_START=0.92
export MUON_MOMENTUM_WARMUP_STEPS=1500
export MUON_WD=0.04
export ADAM_WD=0.04
export GRAD_CLIP_NORM=0.3
export SWA_ENABLED=1
export SWA_EVERY=50

# Our changes:
export QAT_ENABLED=1               # QAT from step 1 (local exp: best post-quant loss)
export LATE_QAT_THRESHOLD=0        # Disabled — QAT already on from start
export EVAL_TEMPERATURE=0.90       # Temperature calibration (validated for relu²)
export TTT_ENABLED=1               # Keep TTT (proven -0.0025 BPB)
export TTT_LR=0.002
export TTT_EPOCHS=3
export TTT_CHUNK_TOKENS=32768
export TTT_FREEZE_BLOCKS=0
export TTT_MOMENTUM=0.9
export TTT_BATCH_SEQS=32
export TTT_GRAD_CLIP=1.0

echo "=== Safe Run: seed=$SEED ==="
echo "Changes from SOTA: fixed QAT + T=0.90 + stride=16 measurement"
echo ""

torchrun --standalone --nproc_per_node="$NPROC" train_gpt_safe.py
