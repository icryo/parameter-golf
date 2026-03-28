#!/bin/bash
# Parameter Golf experiment runner
# Usage: ./run_experiments.sh <experiment_name> [seed]
#
# Experiments:
#   baseline    - Reproduce SOTA (PR #518) as reference
#   kernels     - SOTA + Triton kernels (speed test)
#   noisy_qat   - SOTA + kernels + Noisy QAT (fixes broken QAT)
#   full        - All optimizations + TTT + temp calibration
#   ablation_qat - Compare STE vs Noisy QAT
#   ablation_bigram - Test BigramHash 10240 vs 2048

set -euo pipefail

EXPERIMENT="${1:-baseline}"
SEED="${2:-1337}"
NPROC="${NPROC:-8}"

# Common settings (from SOTA submission)
export DATA_PATH="./data/datasets/fineweb10B_sp1024"
export TOKENIZER_PATH="./data/tokenizers/fineweb_1024_bpe.model"
export SEED="$SEED"
export MAX_WALLCLOCK_SECONDS=600
export ITERATIONS=9000
export WARMUP_STEPS=20
export WARMDOWN_ITERS=3500
export TRAIN_BATCH_TOKENS=786432
export TRAIN_SEQ_LEN=2048
export EVAL_SEQ_LEN=2048
export EVAL_STRIDE=64

# Architecture (11L, 512d, 8H/4KV, 3x MLP)
export NUM_LAYERS=11
export MODEL_DIM=512
export NUM_HEADS=8
export NUM_KV_HEADS=4
export MLP_MULT=3
export TIE_EMBEDDINGS=1

# Features (all from SOTA)
export XSA_LAST_N=4
export ROPE_DIMS=16
export LN_SCALE=1
export VE_ENABLED=1
export VE_DIM=128
export VE_LAYERS="9,10"
export BIGRAM_VOCAB_SIZE=2048
export BIGRAM_DIM=128
export LOGIT_SOFTCAP=30.0

# Optimizer
export MATRIX_LR=0.025
export SCALAR_LR=0.025
export TIED_EMBED_LR=0.035
export MUON_MOMENTUM=0.99
export MUON_MOMENTUM_WARMUP_START=0.92
export MUON_MOMENTUM_WARMUP_STEPS=1500
export MUON_WD=0.04
export ADAM_WD=0.04
export GRAD_CLIP_NORM=0.3

# Weight averaging
export SWA_ENABLED=1
export SWA_EVERY=50

# QAT defaults
export QAT_ENABLED=0
export LATE_QAT_THRESHOLD=0.15
export NOISY_QAT=1

# TTT defaults
export TTT_ENABLED=0
export TTT_LR=0.002
export TTT_EPOCHS=3
export TTT_CHUNK_TOKENS=32768
export TTT_FREEZE_BLOCKS=0
export TTT_MOMENTUM=0.9
export TTT_BATCH_SEQS=32
export TTT_GRAD_CLIP=1.0

# Temperature
export EVAL_TEMPERATURE=0

case "$EXPERIMENT" in

  baseline)
    # Reproduce SOTA (PR #518) for reference
    export RUN_ID="baseline_s${SEED}"
    export TTT_ENABLED=1
    SCRIPT="records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py"
    echo "=== Baseline (SOTA reproduction) seed=$SEED ==="
    ;;

  kernels)
    # SOTA + Triton kernels (measure speed gain, should match BPB)
    export RUN_ID="kernels_s${SEED}"
    export TTT_ENABLED=1
    SCRIPT="train_gpt_kernels.py"
    echo "=== Triton Kernels seed=$SEED ==="
    ;;

  noisy_qat)
    # SOTA + kernels + Noisy QAT (key innovation: fixes broken QAT)
    export RUN_ID="noisy_qat_s${SEED}"
    export NOISY_QAT=1
    export TTT_ENABLED=1
    SCRIPT="train_gpt_kernels.py"
    echo "=== Noisy QAT + Kernels seed=$SEED ==="
    ;;

  full)
    # All optimizations stacked
    export RUN_ID="full_s${SEED}"
    export NOISY_QAT=1
    export TTT_ENABLED=1
    export BIGRAM_VOCAB_SIZE=3072  # Larger bigram (from SOTA ablation: -0.0009)
    export EVAL_TEMPERATURE=0     # Auto-calibrate temperature
    SCRIPT="train_gpt_kernels.py"
    echo "=== Full Optimization Stack seed=$SEED ==="
    ;;

  ablation_qat)
    # A/B: STE QAT vs Noisy QAT
    export RUN_ID="ablation_ste_qat_s${SEED}"
    export NOISY_QAT=0
    export TTT_ENABLED=0
    SCRIPT="train_gpt_kernels.py"
    echo "=== Ablation: STE QAT seed=$SEED ==="
    echo "Run again with NOISY_QAT=1 for comparison"
    ;;

  ablation_bigram)
    # Test larger BigramHash
    export RUN_ID="ablation_bigram10k_s${SEED}"
    export BIGRAM_VOCAB_SIZE=10240
    export NOISY_QAT=1
    export TTT_ENABLED=0
    SCRIPT="train_gpt_kernels.py"
    echo "=== Ablation: BigramHash 10240 seed=$SEED ==="
    ;;

  *)
    echo "Unknown experiment: $EXPERIMENT"
    echo "Available: baseline, kernels, noisy_qat, full, ablation_qat, ablation_bigram"
    exit 1
    ;;
esac

echo "Script: $SCRIPT"
echo "Run ID: $RUN_ID"
echo "Seed: $SEED"
echo "GPUs: $NPROC"
echo ""

torchrun --standalone --nproc_per_node="$NPROC" "$SCRIPT"
