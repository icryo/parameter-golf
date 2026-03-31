# Confirmed Dead Ends

## 1. QAT with Muon Optimizer
**Time wasted:** ~4 hours of local experiments
**Why it seemed promising:** Every top submission's Late QAT was provably broken (class bool constant-folded by torch.compile). Fixing it should improve BPB.
**Why it failed:** Muon produces 179x smaller quantization gap than AdamW. The "broken" QAT never mattered because Muon's Newton-Schulz orthogonalization naturally produces quantization-friendly weight distributions.
**Lesson:** Validate assumptions on the actual optimizer before investing in fixes.

## 2. Triton Kernels for Operations torch.compile Already Fuses
**Time wasted:** ~3 hours
**Why it seemed promising:** Fusing LeakyReLU+square should eliminate intermediate tensor writes.
**Why it failed:** torch.compile already fuses these ops and does it better (0.041ms vs our 0.053ms). Custom kernels add launch overhead that exceeds the memory savings at these tensor sizes.
**Lesson:** Profile torch.compile's output before writing custom kernels. The only kernel that actually helps fuses the MATMUL with the activation (PR #1072's approach).

## 3. TTT (Test-Time Training) on Full GPTQ Stack
**Time wasted:** ~30 minutes of H100 time (~$10)
**Why it seemed promising:** TTT gives -0.0025 BPP on the old SOTA. PR #1043 reported early TTT chunks hitting 1.109 BPB.
**Why it failed:** Full GPTQ already recovers the quantization error that TTT was compensating for. TTT's SGD pushes weights away from the GPTQ-optimized point. The "early chunk quality" was just easier validation data, not a TTT effect.
**Lesson:** When a better quantization method replaces the old one, eval-time adaptations tuned for the old method become useless.

## 4. TTT Periodic Reset (Anti-Drift)
**Time wasted:** ~15 minutes of H100 time
**Why it seemed promising:** Running BPP showed drift from 1.105 (chunk 51) to 1.126 (late chunks). Resetting should recover early-chunk quality.
**Why it failed:** The "drift" was data difficulty variation in the validation set, not model degradation. Each reset cycle's first chunks scored without any TTT benefit, dragging down the average.
**Lesson:** Distinguish data effects from model effects before building solutions.

## 5. Depth Recurrence / RYS-style Layer Duplication
**Time wasted:** Research only (saved by reading PR #363's 250-hour failure report)
**Why it seemed promising:** Share weights → more effective depth per byte.
**Why it failed:** Two structural penalties: (1) quantization error compounds through loop iterations (~900x amplification), (2) +32ms/step overhead = 1200 fewer training steps. Three independent researchers confirmed.
**Lesson:** Read the competition's negative results before investing in approaches others have exhaustively tested.

## 6. Mamba / State-Space Models
**Time wasted:** Research only
**Why it seemed promising:** Linear attention = no O(n²) cost, potentially more layers per 16MB.
**Why it failed:** Mamba loses to transformers below 350M params. At 25M params and seq_len=2048, attention's direct token access is more valuable than SSM's linear scan. No torch.compile support, no FA3 equivalent.
**Lesson:** Architecture advantages depend on scale regime. What works at 1B params may fail at 25M.

## 7. FlashAttention 4 (FA4)
**Time wasted:** Research only
**Why it seemed promising:** Pure-Python wheel, no compilation needed.
**Why it failed:** Backward pass NOT optimized for Hopper — falls back to Ampere code. Forward-only optimization is useless for training.
**Lesson:** Check backward pass support before adopting new attention libraries for training.

## 8. Noisy QAT (Differentiable Noise)
**Time wasted:** ~30 minutes of local experiments
**Why it seemed promising:** Collapses quantization gap 185x on depth-recurrent architectures.
**Why it failed:** Designed for catastrophic quant gaps (0.37 BPP). Flat architecture gap is 0.006. Adding noise to fix a 0.006 gap just hurts convergence.
**Lesson:** Solutions designed for extreme conditions don't necessarily help moderate conditions.

## 9. ResidLambdas on Our Stack
**Time wasted:** ~30 minutes of H100 time (~$10)
**Why it seemed promising:** PR #1130 achieved 1.1140 (12-seed!) with them.
**Why it failed:** Our stack has sigmoid-gated skip connections that already provide residual flexibility. ResidLambdas add redundant parameterization. The tuned LRs from #1130 also don't transfer to our architecture.
**Lesson:** Architecture innovations interact non-additively. What helps one stack can hurt another.

## 10. Building FlashAttention from Source on CUDA 13.0
**Time wasted:** ~1 hour across two nodes
**Why it failed:** CUDA version mismatch (system 13.0 vs PyTorch 12.6/12.8). Even patching the version check doesn't always work. FA3's pre-built wheel on cu128 was the solution all along.
**Lesson:** Always check for pre-built wheels before attempting source builds.

## 11. Bigger Value Embeddings (VE_DIM=196, VE_LAYERS=5,9,10)
**Time wasted:** Part of failed ResidLambdas run
**Why it failed:** More VE parameters compete with the main model for artifact budget. The improvement in one area doesn't compensate for the loss elsewhere.

## 12. FA2 as FA3 Replacement
**Time wasted:** ~20 minutes of H100 time
**Why it seemed promising:** FA2 on H100 should be faster than SDPA.
**Why it failed:** At seq_len=2048, FA2 and SDPA have identical performance on H100. The O(n²) cost at n=2048 is negligible. Only FA3's Hopper-native TMA kernels (which bypass HBM for small sequences) give a speed advantage.
