# H100 Run Log

## Run 1: Node 1 — SDPA baseline (March 29)
- **Host:** vast.ai 8xH100 SXM (35.192.20.187)
- **PyTorch:** 2.11.0+cu126
- **Attention:** SDPA shim (no FlashAttention)
- **Script:** train_gpt_merged.py (PR #1060 base only)
- **Config:** XSA-all, BigramHash 2816×112, Full GPTQ, coprime loader

| Metric | Value |
|--------|-------|
| ms/step | 100 |
| Steps | 5,847 |
| Sliding BPB (s64) | **1.1165** |
| Roundtrip BPB | 1.1400 |
| Post-EMA BPB | 1.1365 |
| Artifact | 15.97 MB |
| GPTQ time | 7.1s (14s reserved) |

**TTT sweep on same checkpoint:**
| Config | BPB | Delta vs no-TTT |
|--------|-----|-----------------|
| No TTT (baseline) | 1.1165 | — |
| TTT lr=0.0025, 4ep | 1.1166 | +0.0001 (neutral) |
| TTT + reset/50 | 1.1168 | +0.0003 (worse) |

**Win:** Beat merged SOTA (1.1194) by 0.003 even with SDPA.
**Finding:** TTT is neutral on Full GPTQ stack. Reset doesn't help.

---

## Run 2: Node 2 — FA2 + fused kernel fix (March 30)
- **Host:** vast.ai 8xH100 SXM (206.125.32.51)
- **PyTorch:** 2.11.0+cu126
- **Attention:** FA2 shim (flash-attn 2.8.3)
- **Fused MLP:** Fixed import (HAS_FUSED_MLP=True), confirmed 1.33x on benchmark

| Metric | Value |
|--------|-------|
| ms/step | 100 (same as SDPA!) |
| Steps | 5,844 |
| Sliding BPB (s64) | **1.1172** |
| Artifact | 15.97 MB |

**Finding:** FA2 is no faster than SDPA at seq_len=2048 on H100. The speed advantage requires FA3's Hopper-native TMA kernels.
**Finding:** Fused Triton MLP saves 5ms/step (1.33x on MLP alone) but total step time unchanged — attention is the bottleneck.

---

## Run 3: Node 3 — FA3 + PR #1089 innovations ★ (March 30)
- **Host:** vast.ai 8xH100 SXM (206.125.32.60)
- **PyTorch:** 2.9.1+cu128
- **Attention:** FA3 native (flash_attn_3-3.0.0 pre-built wheel)
- **Script:** train_gpt_merged.py with EngramLite, gated skips, LeakyReLU(0.3)², Turbo-Muon

| Metric | Value |
|--------|-------|
| ms/step | **87.9** (FA3!) |
| Steps | 6,667 |
| Sliding BPB (s64) | **1.1146** |
| Roundtrip BPB | 1.1381 |
| Post-EMA BPB | 1.1339 |
| Artifact | 15.71 MB |
| GPTQ time | 6.7s (14s reserved) |

**Win:** FA3 gave 15% speed boost. Best single-seed BPB.
**Win:** EngramLite + gated skips + LeakyReLU(0.3)² all contributed.

---

## Run 4: Node 4 — Brotli compression, 3-seed (March 30)
- **Host:** vast.ai 8xH100 SXM (35.192.20.187)
- **PyTorch:** 2.9.1+cu128
- **Attention:** FA3 native
- **Change:** Brotli+byte-shuffle compression (replaces LZMA), GPTQ reserve 9s (was 14s)

### Seed results
| Seed | Steps | BPB | Artifact |
|------|-------|-----|----------|
| 1337 | 6,726 | **1.1143** ★ | 15.29 MB |
| 42 | 6,719 | 1.1161 | ~15.3 MB |
| 2025 | — | 1.1148 | ~15.3 MB |
| **Mean** | | **1.1151** | |
| **Std** | | **0.0009** | |

**Win:** Best single seed (1.1143) crosses record threshold!
**Finding:** Brotli saves ~0.4 MB vs LZMA. Tighter GPTQ reserve gave ~60 more steps.
**Gap:** 3-seed mean 0.0007 above threshold.

---

## Run 5: Brotli + mixed int5/int6/int7 (March 30, same node)
- **Change:** Hessian sensitivity-based bit allocation

| Metric | Value |
|--------|-------|
| BPB | **1.1250** (WORSE) |
| Artifact | 12.68 MB (way under budget) |

**Failure:** Allocation too aggressive — 58 layers at int5, only 6 at int6, 2 at int7. Quality destroyed. The int5 demotion hurt more than int7 promotion helped.

---

## Run 6: ResidLambdas + tuned LRs from PR #1130 (March 30, node 5)
- **Host:** vast.ai 8xH100 SXM (206.125.32.61)
- **Change:** ResidLambdas (init √1.1, 5x LR), MATRIX_LR=0.036, VE_DIM=196

| Metric | Value |
|--------|-------|
| BPB | **1.1257** (WORSE) |

**Failure:** LR tuning from PR #1130 doesn't transfer to our stack.

---

## Run 7: ResidLambdas only (reverted LRs) (same node)
- **Change:** Only ResidLambdas, original LR settings

| Metric | Value |
|--------|-------|
| BPB | **1.1259** (WORSE) |

**Failure:** ResidLambdas themselves hurt our stack. The gated skip connections already provide residual flexibility — ResidLambdas add redundant parameterization.

---

## Run 8: Extra seed (seed 7, best config) (same node)
| Seed | BPB |
|------|-----|
| 7 | **1.1244** |

**Finding:** High variance across seeds. Some seeds are just bad.
