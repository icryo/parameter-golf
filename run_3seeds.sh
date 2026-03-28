#!/bin/bash
# Run 3 seeds for submission validation
# Usage: ./run_3seeds.sh <experiment_name>
set -euo pipefail

EXPERIMENT="${1:-full}"

echo "========================================"
echo "Parameter Golf: 3-seed run for $EXPERIMENT"
echo "========================================"

for SEED in 1337 42 2025; do
    echo ""
    echo "========== SEED $SEED =========="
    ./run_experiments.sh "$EXPERIMENT" "$SEED" 2>&1 | tee "logs/${EXPERIMENT}_s${SEED}.log"
    echo "========== SEED $SEED DONE =========="
    echo ""
done

echo "========================================"
echo "All 3 seeds complete. Check logs/ for results."
echo "Grep for final BPB:"
echo "========================================"
for SEED in 1337 42 2025; do
    echo -n "  Seed $SEED: "
    grep -o 'final_int8_zlib_roundtrip_exact val_bpb:[0-9.]*' "logs/${EXPERIMENT}_s${SEED}.log" | tail -1 || echo "NOT FOUND"
done
