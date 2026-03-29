#!/bin/bash
# Quick 1xH100 validation (~15 min, ~$1-2)
# Confirms: fused kernel works, measures ms/step, tests GPTQ + TTT reset
# Does NOT produce competitive BPB (too few steps on 1 GPU)
set -euo pipefail
SEED="${1:-1337}"

echo "=== 1xH100 Validation Test ==="

export SEED="$SEED" RUN_ID="test1gpu_s${SEED}"
export DATA_PATH="./data/datasets/fineweb10B_sp1024"
export TOKENIZER_PATH="./data/tokenizers/fineweb_1024_bpe.model"
export MAX_WALLCLOCK_SECONDS=300  # 5 min training (enough for ~500 steps)
export ITERATIONS=2000
export WARMUP_STEPS=5 WARMDOWN_ITERS=500
export TRAIN_BATCH_TOKENS=98304  # Smaller batch for 1 GPU (96K vs 786K)
export TRAIN_SEQ_LEN=2048 EVAL_SEQ_LEN=2048 EVAL_STRIDE=64
export VAL_BATCH_SIZE=65536
export TRAIN_LOG_EVERY=50 VAL_LOSS_EVERY=500
export NUM_LAYERS=11 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=3
export TIE_EMBEDDINGS=1 ROPE_DIMS=16 LN_SCALE=1
export VE_ENABLED=1 VE_DIM=128 VE_LAYERS="9,10"
export LOGIT_SOFTCAP=30.0
export MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035
export MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 MUON_MOMENTUM_WARMUP_STEPS=100
export MUON_WD=0.04 ADAM_WD=0.04 GRAD_CLIP_NORM=0.3
export SWA_ENABLED=1 SWA_EVERY=50
export LATE_QAT_THRESHOLD=0.15
export XSA_LAST_N=11
export BIGRAM_VOCAB_SIZE=2816 BIGRAM_DIM=112
export USE_GPTQ=1 GPTQ_RESERVE_MS=10000
# Enable TTT with reset for validation
export TTT_ENABLED=1 TTT_LR=0.0025 TTT_EPOCHS=2
export TTT_FREEZE_BLOCKS=0 TTT_MOMENTUM=0.9 TTT_BATCH_SEQS=16
export TTT_GRAD_CLIP=1.0 TTT_RESET_EVERY=50
export TTT_CHUNK_TOKENS=32768

echo "Key things to check in output:"
echo "  1. 'HAS_FUSED_MLP' — should be True on H100"
echo "  2. 'step_avg:XXms' — target ~70ms with fused kernel"
echo "  3. 'gptq:' lines — Full GPTQ calibration works"
echo "  4. 'ttt_reset' lines — periodic reset fires"
echo "  5. No crashes or NaN losses"
echo ""

python3 train_gpt_merged.py
