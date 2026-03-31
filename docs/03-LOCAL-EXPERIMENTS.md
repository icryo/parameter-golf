# Local Experiments (RTX 4090)

All experiments run on a single RTX 4090 (24GB) with 2 training shards.
Purpose: validate mechanisms before spending money on H100s.

## QAT Constant-Folding Discovery ★

**The find:** `torch.compile` constant-folds `CastedLinear._qat_enabled: bool = False`. When Late QAT flips it to `True` mid-training, the compiled model ignores the change. **QAT never actually runs in any top submission.**

**Proof:**
```
Class bool (original):    Output diff after enabling QAT: 0.000000e+00  ← BROKEN
Mutable list (our fix):   Output diff after enabling QAT: 9.921875e-01  ← WORKS
```

**Recompilation cost:** 311ms one-time, 0.004ms/step ongoing. Negligible.

**But then:** Muon's quant gap is only 0.07% (vs AdamW's 10.07%). QAT is irrelevant with Muon — the broken QAT never mattered.

## QAT A/B Experiments (AdamW)

| Config | Pre-quant | Post-quant | Gap |
|--------|-----------|------------|-----|
| No QAT | 4.6007 | 4.8125 | +0.2118 (+4.60%) |
| **STE QAT** | 4.7986 | **4.6383** | **-0.1603 (-3.34%)** |
| Noisy QAT | 4.6044 | 4.9213 | +0.3169 (+6.88%) |

STE inverts the gap (quantized model better than float). Noisy QAT makes it worse.

## QAT Timing Sweep (AdamW)

| QAT Start | Gap % |
|-----------|-------|
| No QAT | +4.60% |
| QAT@15% (current SOTA) | -3.70% |
| QAT@25% | -4.36% |
| QAT@50% | -4.76% |
| **QAT@100%** | **-11.60%** |

Earlier QAT = better. But irrelevant with Muon.

## Muon vs AdamW Quant Gap

| Optimizer | Quant Gap |
|-----------|-----------|
| **Muon** | **0.07%** |
| AdamW | 10.07% |

**179x smaller gap with Muon.** This killed our entire QAT strategy.

## Triton Kernel Benchmarks (4090)

| Kernel | PyTorch | Triton | Speedup |
|--------|---------|--------|---------|
| LeakyReLU² (activation only) | 0.037ms | 0.053ms | **0.70x (SLOWER)** |
| STE int6 | 0.119ms | 0.048ms | **2.47x** |
| XSA projection | 0.089ms | 0.050ms | **1.76x** |
| LeakyReLU² (torch.compile) | 0.041ms | 0.053ms | **0.77x (compile wins)** |

**Lesson:** Don't write custom kernels for operations torch.compile already fuses.

## Fused Matmul+Activation (4090, non-TMA)

| | Time |
|---|------|
| Unfused | 0.238ms |
| Our Triton kernel | 0.199ms |
| Speedup | 1.20x |

Modest improvement on 4090. The real PR #1072 kernel uses H100 TMA for much larger gains.

## Bank QAT Investigation

**Discovery:** Bank weights (96% of params) were never QAT'd — they go through `F.linear(x, w.to(x.dtype))` not `CastedLinear`. STE on fp32 bank weights has no effect because `w.to(bf16)` wipes out the perturbation.

**Fix:** Cast to bf16 first, then quantize. Verified at block level (diff=2896). But irrelevant with Muon.

## Temperature Scaling

No effect at 400 training steps. The model needs full convergence for temperature to matter. Ternary/binary submissions report T=0.90 optimal for relu² activations.
