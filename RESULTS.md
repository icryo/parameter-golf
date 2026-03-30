# Experiment Results — March 29-30, 2026

## 8xH100 Run (vast.ai, SDPA shim — no FlashAttention 3)

### Training (Phase 1)
- **Script**: `train_gpt_merged.py` (PR #1060 + PR #1072 fused kernel)
- **Config**: 11L 512d, 8H/4KV, 3x MLP LeakyReLU(0.5)², XSA-all-11, BigramHash(2816×112)
- **Coprime-stride loader**: 80 shards
- **Full Hessian GPTQ**: 64-batch calibration, 7.1s (within 14s budget)
- **Fused Triton MLP**: `HAS_FUSED_MLP = True` (TMA kernel loaded)
- **Attention**: SDPA shim (F.scaled_dot_product_attention), NOT FlashAttention 3
- **Steps**: 5,847 (wallclock cap at 586s)
- **ms/step**: 100ms (bottlenecked by SDPA — FA3 would give ~70-85ms)
- **Seed**: 1337

### Results

| Metric | BPB | Notes |
|--------|-----|-------|
| Val BPB @ step 4000 | 1.1869 | Mid-training checkpoint |
| Val BPB @ step 5847 (final) | 1.1373 | Pre-quant, pre-EMA |
| Post-EMA diagnostic | 1.1365 | |
| **Int6 roundtrip** | **1.1400** | Post GPTQ int6+LZMA |
| **Sliding window s64** | **1.1165** | **Main metric** |
| TTT (lr=0.0025, 4ep, no reset) | 1.1166 | Neutral (+0.00004) |
| TTT (lr=0.0025, 4ep, reset/50) | 1.1168 | Slightly worse (+0.0002) |

### Key Findings
1. **1.1165 BPB beats merged SOTA (1.1194) by 0.003** even with SDPA
2. **TTT is neutral on Full GPTQ stack** — confirms PR #1060's finding independently
3. **Periodic TTT reset does NOT help** — drift wasn't the issue; TTT is simply unnecessary
4. **Fused Triton kernel loaded** but speedup masked by SDPA attention bottleneck
5. **Artifact size**: 15.97 MB (under 16 MB limit)
6. **GPTQ calibration**: 7.1s (well within 14s budget)

### Comparison to Competition

| Submission | BPB | Our delta | Notes |
|-----------|-----|-----------|-------|
| PR #1089 (Turbo-Muon+EngramLite) | 1.1086 | +0.0079 | Uses innovations we don't have |
| PR #1060 (coprime+GPTQ+XSA-all) | 1.1122 | +0.0043 | Same config, but they have FA3 |
| **Our run (SDPA, 5847 steps)** | **1.1165** | — | |
| Merged SOTA (PR #549) | 1.1194 | **-0.0029** | We beat this |
| Record threshold (vs SOTA) | 1.1144 | +0.0021 | Need FA3 to cross this |

### Projection with FlashAttention 3
- FA3 gives ~85ms/step → ~6,894 steps (18% more)
- Fused kernel + FA3 gives ~70ms/step → ~8,371 steps (43% more)
- Projected BPB with FA3: **~1.112-1.114** (crosses record threshold)
- Projected BPB with fused+FA3: **~1.108-1.112** (competitive with PR #1060)

## Local Experiments (RTX 4090)

### QAT Fix Validation
- `torch.compile` constant-folds `_qat_enabled: bool = False` — QAT never activates
- Mutable list `_qat_state: list[bool] = [False]` survives compilation
- **Every top submission's Late QAT is broken** (confirmed on real hardware)
- However: **Muon produces 179x smaller quant gap than AdamW** — QAT is unnecessary

### Triton Kernel Benchmarks (4090)
- Fused LeakyReLU²: **0.70x SLOWER** than torch.compile (don't use)
- Fused STE int6: **2.47x faster** than PyTorch
- Fused XSA: **1.76x faster** (marginal, 0.15ms across 4 layers)
- Fused matmul+activation (PR #1072): requires TMA (H100 only)

### QAT Experiments
- Muon quant gap: 0.07% (vs AdamW 10.07%) — QAT irrelevant with Muon
- STE QAT inverts gap on AdamW (-8.9%) but Muon doesn't need it
- Noisy QAT: WORSE than STE on flat architecture (+6.9% gap)

## Blockers for Next Run
1. **FlashAttention 3** — pip install fails on PyTorch 2.11 (build deps issue)
   - Need a pre-built image with FA3 or PyTorch 2.9 + matching flash-attn
2. **Record threshold** — need ≤1.1144 BPB (0.005 nats below SOTA)
   - FA3 alone should get us there (~1.112-1.114)

## Files
- `train_gpt_merged.py` — merged PR #1060 + PR #1072 (fused kernel + coprime loader + Full GPTQ)
- `run_ttt_only.py` — standalone TTT eval on saved checkpoint
- `run_h100.sh` — full experiment runner
- `run_1gpu_test.sh` — 1xH100 validation script

## Run 2: 8xH100 Node 2 (FA2 + Fused Kernel Fix)

- **Attention**: FA2 (flash-attn 2.8.3 via patched CUDA check)
- **Fused MLP**: Fixed import from separate file — loaded and verified 1.33x speedup
  - But only saves 5ms/step (MLP not the bottleneck)
- **ms/step**: 100ms (still SDPA/FA2-limited)
- **Steps**: 5,844
- **Sliding BPB**: **1.1172**

### Fused Kernel Benchmark (H100)
- Unfused MLP: 2.01 ms/call
- Fused MLP:   1.51 ms/call  
- Speedup: 1.33x (0.49ms × 11 layers = 5.4ms/step)
- Total step still 100ms — attention is the bottleneck, not MLP

## Final Assessment

Two independent 8xH100 runs confirm **~1.117 BPB** with SDPA/FA2 attention.
The gap to FA3 (which gives ~85ms/step) is ~15ms/step = ~1000 more training steps.
With those extra steps, projected BPB: **~1.112-1.114** (crosses record threshold).

**Blocker**: flash_attn_interface (FA3 Hopper kernels) cannot be pip-installed on
PyTorch 2.11 due to CUDA version mismatch. Needs either:
1. A pre-built image with FA3 + PyTorch 2.9+
2. Building from Dao-AILab/flash-attention Hopper branch with matching CUDA
