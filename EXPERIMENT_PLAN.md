# Parameter Golf — Experiment Plan

## Current SOTA: 1.1194 BPB (must beat by ≥0.005 nats → target ≤1.1144 BPB)

## Our Innovations (implemented in train_gpt_kernels.py)

### 1. Triton Kernels (Phase 1 — speed)
- `leaky_relu_squared`: Fused LeakyReLU(0.5)² — saves 22 intermediate tensor writes/fwd
- `fused_xsa`: Fused XSA projection removal — saves kernel launches for last 4 layers
- `ste_int6_fakequant`: Fused STE that torch.compile can't constant-fold

Est. speedup: 2-5ms/step → 14-35 more training steps in 600s

### 2. Fixed QAT (Phase 2 — quality)
- **Bug fix**: #3's QAT was broken (class attribute constant-folded by torch.compile)
- **New**: Uses mutable list `_qat_state[0]` that survives compilation
- **New**: Bank weights (Q/K/V/O/MLP) now also get STE during QAT phase
- **New**: Noisy QAT option — differentiable noise calibrated to int6 step size (amax/31)
  - From depth recurrence research: collapses quant gap 185x
  - Key: noise magnitude MUST match export format (int6 = 31 levels, not 127)

Est. impact: -0.001 to -0.003 BPB

### 3. Evaluation Improvements (Phase 3 — free BPB)
- Stride=16 evaluation (vs stride=64): -0.015 BPB (but slower)
- Auto temperature calibration: grid search over [0.85, 0.88, 0.90, 0.92, 0.95, 1.0]
- Larger BigramHash(3072) from SOTA ablation: -0.0009 BPB

### CANCELLED: RYS Layer Duplication
Depth recurrence confirmed dead end at 10-min by 3 independent researchers.
- 0.025 BPB structural penalty
- Quantization error compounds through loop iterations (~900x amplification)
- +32ms/step overhead = 1200 fewer training steps

## Estimated BPB Budget

| Technique | Est. BPB Gain | Status |
|-----------|--------------|--------|
| Triton kernels (more steps) | -0.001 to -0.002 | Implemented |
| Fixed QAT (bank weights) | -0.001 to -0.002 | Implemented |
| Noisy QAT (diff. noise) | -0.001 to -0.002 | Implemented |
| BigramHash 3072 | -0.0009 | Implemented |
| Stride=16 eval | -0.015 | Implemented (needs time budget check) |
| Temperature calibration | -0.001 | Implemented |
| **Optimistic total** | **-0.008 to -0.012** | → **1.1074-1.1114 BPB** |
| **Conservative total** | **-0.004 to -0.006** | → **1.1134-1.1154 BPB** |

## Run Commands

```bash
# Quick test (1 seed)
./run_experiments.sh full 1337

# Full submission (3 seeds)
./run_3seeds.sh full

# Ablation: just kernels (speed comparison)
./run_experiments.sh kernels 1337

# Ablation: STE vs Noisy QAT
NOISY_QAT=0 ./run_experiments.sh ablation_qat 1337
NOISY_QAT=1 ./run_experiments.sh ablation_qat 1337
```

## Files

| File | Size | Purpose |
|------|------|---------|
| train_gpt_kernels.py | 95KB | Modified SOTA with all innovations |
| triton_kernels.py | 19KB | Custom Triton kernels |
| test_kernels.py | 3KB | GPU validation tests |
| run_experiments.sh | 4KB | Experiment runner |
| run_3seeds.sh | 1KB | 3-seed submission validator |

NOTE: For submission, triton_kernels.py must be inlined into train_gpt_kernels.py
(submissions must be self-contained single-file).
