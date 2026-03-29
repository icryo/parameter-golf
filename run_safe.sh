#!/bin/bash
# Information-maximizing single-seed run
# Trains identically to SOTA, then sweeps temperatures + measures gaps
#
# Expected output (key lines to look for in logs):
#   BEST TEMPERATURE: T=X.XX
#   TEMPERATURE EFFECT: delta=X.XXXXXXXX
#   quant_gap: gap=X.XXXXXXXX
#   legal_ttt_exact val_bpb:X.XXXXXXXX
#
# Usage: ./run_safe.sh [seed]
# Cost: ~$5 on 8xH100 (~20 min total)
set -euo pipefail
SEED="${1:-1337}"
NPROC="${NPROC:-8}"

export RUN_ID="sweep_s${SEED}"
export SEED="$SEED"
export DATA_PATH="./data/datasets/fineweb10B_sp1024"
export TOKENIZER_PATH="./data/tokenizers/fineweb_1024_bpe.model"

# Training: identical to SOTA
export MAX_WALLCLOCK_SECONDS=600
export ITERATIONS=9000 WARMUP_STEPS=20 WARMDOWN_ITERS=3500
export TRAIN_BATCH_TOKENS=786432 TRAIN_SEQ_LEN=2048
export EVAL_SEQ_LEN=2048 EVAL_STRIDE=64
export NUM_LAYERS=11 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=3
export TIE_EMBEDDINGS=1 XSA_LAST_N=4 ROPE_DIMS=16 LN_SCALE=1
export VE_ENABLED=1 VE_DIM=128 VE_LAYERS="9,10"
export BIGRAM_VOCAB_SIZE=2048 BIGRAM_DIM=128 LOGIT_SOFTCAP=30.0
export MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035
export MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 MUON_MOMENTUM_WARMUP_STEPS=1500
export MUON_WD=0.04 ADAM_WD=0.04 GRAD_CLIP_NORM=0.3
export SWA_ENABLED=1 SWA_EVERY=50
export LATE_QAT_THRESHOLD=0.15

# Eval: temperature grid search built into script (no env var needed)
export EVAL_TEMPERATURE=0  # 0 = use internal grid search

# TTT: same as SOTA
export TTT_ENABLED=1
export TTT_LR=0.002 TTT_EPOCHS=3 TTT_CHUNK_TOKENS=32768
export TTT_FREEZE_BLOCKS=0 TTT_MOMENTUM=0.9 TTT_BATCH_SEQS=32 TTT_GRAD_CLIP=1.0

echo "=== Information sweep: seed=$SEED ==="
echo "Training: identical to SOTA"
echo "Eval: temperature grid [0.85,0.88,0.90,0.92,0.95,1.00] + sliding + TTT"
echo ""
torchrun --standalone --nproc_per_node="$NPROC" train_gpt_safe.py
