# Innovations: What Worked, What Didn't

## ✅ What Worked

### FlashAttention 3 (pre-built wheel)
- **Impact:** 100ms → 85ms/step (15% faster, ~850 more training steps)
- **How:** `pip install "https://download.pytorch.org/whl/cu128/flash_attn_3-3.0.0-cp39-abi3-manylinux_2_28_x86_64.whl"`
- **Requires:** PyTorch 2.9.1+cu128 (NOT 2.11)
- **The key we missed for 2 days:** The pre-built wheel URL was in our own repo's setup.sh the entire time.

### EngramLite (from PR #1089)
- **Impact:** Replaces BigramHash with multi-head bigram+trigram hashing
- **Config:** 8192 buckets, 2 heads, 2 orders, 32 dim/head, learned sigmoid gate
- **Why it works:** Captures both bigram and trigram patterns with multiple hash functions, reducing collision rate.

### Sigmoid-gated skip connections (from PR #1089)
- **Impact:** ~0.001-0.002 BPB improvement over additive skips
- **How:** `gate = sigmoid(skip_gates[i]); x = lerp(skip_weight*skip, x, gate)`
- **Why it works:** Lets the model learn how much encoder information to inject at each decoder layer.

### LeakyReLU(0.3)² (from PR #1089)
- **Impact:** Replaces LeakyReLU(0.5)²
- **Why:** Different negative slope changes the dead neuron recovery dynamics. 0.3 was empirically better in PR #1089's extensive testing.

### Turbo-Muon (4 NS steps) (from PR #1089)
- **Impact:** ~1-2ms/step faster optimizer
- **How:** `MUON_BACKEND_STEPS=4` instead of 5
- **Why:** 4 Newton-Schulz iterations are sufficient for the orthogonalization quality needed.

### Coprime-stride data loader (from PR #1060)
- **Impact:** Better data diversity per batch
- **How:** Samples blocks from multiple shards using coprime strides, ensuring full coverage before repetition.

### Full Hessian GPTQ (from PR #1060)
- **Impact:** ~0.003 BPB better quantization than GPTQ-lite
- **How:** Cholesky error compensation with column reordering, 64-batch calibration in ~7s
- **Budget legal:** Reserves 9-14s from training budget.

### XSA on all 11 layers (from PR #1060)
- **Impact:** ~0.002 BPB over XSA on last 4 only
- **How:** `XSA_LAST_N=11`

### Brotli + byte-shuffle compression
- **Impact:** ~0.4 MB smaller artifact than LZMA
- **How:** Brotli quality=11 with stride=2 byte interleaving pre-filter
- **Requires:** `pip install brotli`

### Tighter GPTQ reserve (9s vs 14s)
- **Impact:** ~57 more training steps
- **How:** `GPTQ_RESERVE_MS=9000`. Actual GPTQ takes ~7s, so 9s gives 2s safety margin.

### LR floor 0.05
- **Impact:** Warmdown doesn't reach zero — maintains some learning capacity in final steps
- **From:** PR #1089

## ❌ What Didn't Work

### ResidLambdas (from PR #1130)
- **Expected:** ~0.001-0.002 BPB (PR #1130 achieved 1.1140 with them)
- **Actual:** 1.1257 BPB — **0.011 WORSE**
- **Why:** Our stack already has sigmoid-gated skip connections which provide similar residual flexibility. ResidLambdas add redundant parameterization that hurts convergence.
- **Lesson:** Innovations don't transfer between stacks blindly.

### PR #1130's tuned LRs (MATRIX_LR=0.036, etc.)
- **Actual:** Made ResidLambdas result even worse
- **Why:** Hyperparameters are tuned for their specific architecture. Different activation functions, different embedding types, different skip mechanisms all change the loss landscape.

### Mixed int5/int6/int7 quantization
- **Expected:** Better precision for sensitive layers
- **Actual:** 1.1250 BPB — **0.010 WORSE**
- **Why:** The allocation algorithm was too aggressive. 58/66 layers got int5 (15 levels), losing critical precision. The budget estimation assumed worse compression than Brotli actually achieves.
- **Fixable:** Need a less aggressive allocation that starts at int6 baseline and only promotes to int7, never demotes to int5.

### TTT (Test-Time Training) on Full GPTQ stack
- **Expected:** ~0.002 BPP improvement (as on the old SOTA)
- **Actual:** +0.00004 (neutral) to +0.0003 (slightly worse with reset)
- **Why:** Full GPTQ already recovers the quantization error that TTT was compensating for. TTT's SGD adaptation pushes weights away from the GPTQ-optimized point.

### TTT periodic reset
- **Expected:** Prevent TTT drift, recover early-chunk quality
- **Actual:** Slightly worse than no-reset TTT
- **Why:** Each reset cycle's first ~10 chunks are scored with the unadapted model. The "drift" in running BPP was actually data difficulty variation, not model degradation.

### Fused Triton LeakyReLU² kernel (activation only)
- **Expected:** 1-3ms/step savings
- **Actual:** 0.70x SLOWER than torch.compile
- **Why:** torch.compile already fuses leaky_relu + square. Our kernel added launch overhead.

### Fused Triton matmul+activation kernel (from PR #1072)
- **Expected:** 70ms/step (the PR's claim)
- **Actual:** Only saves 5ms/step (2.01ms → 1.51ms per MLP call × 11 layers)
- **Why:** Works correctly but total step time still dominated by attention. Need FA3 for the attention speedup. Combined FA3+fused would give ~80ms/step but fused kernel requires Triton TMA (3.3+) which conflicts with some PyTorch versions.

### QAT (Quantization-Aware Training) with Muon
- **Expected:** Fix broken QAT for better quantization
- **Actual:** Muon produces 179x smaller quant gap than AdamW. QAT is irrelevant.
- **Discovery value:** Confirmed that every top submission's Late QAT was broken (class bool constant-folded by torch.compile). But it didn't matter because Muon's Newton-Schulz orthogonalization naturally produces quantization-friendly weights.

### Noisy QAT (differentiable noise)
- **Expected:** Better than STE for training robustness
- **Actual:** +6.88% quant gap (WORSE than no QAT)
- **Why:** Designed for depth-recurrent architectures with catastrophic quant gaps. On flat architectures with small gaps, the noise just hurts convergence.

### Bigger Value Embeddings (VE_DIM=196, VE_LAYERS=5,9,10)
- **Actual:** Part of the failed PR #1130 tuning — contributed to regression
- **Why:** More VE parameters compete with the main model for the 16MB budget.

### Mamba/SSM architecture
- **Researched but not tested**
- **Conclusion:** Loses to transformers below 350M params. No torch.compile support. No FA3 equivalent.

### Depth recurrence (RYS-style layer duplication)
- **Researched (from depth recurrence PR #363)**
- **Conclusion:** Structural 0.025 BPB penalty from quantization compounding + step-time overhead. Three independent researchers confirmed. Dead end at 10-min constraint.

### FA4 (FlashAttention 4)
- **Researched**
- **Conclusion:** Backward pass NOT optimized for Hopper — falls back to Ampere code. Forward-only optimization. Useless for training.
