# Next Steps: Research Directions

## Gap Analysis

Our best: **1.1151 BPB** (3-seed mean). Record threshold: **1.1144**. Gap: **0.0007**.
Top open PR: **1.0806 BPB** (#1143 Scylla). Gap to that: **0.0345**.

## Tier 1: Close the 0.0007 Gap (Record vs Merged SOTA)

### Fix Mixed int6/int7 Quantization
**Current problem:** Our implementation demotes too many layers to int5. 58/66 layers at int5, only 6 at int6, 2 at int7. Artifact 12.68 MB (3.3 MB under budget).
**Fix:** Start at int6 baseline (not int5), only PROMOTE the most sensitive to int7. Never demote below int6. With Brotli compression, uniform int6 already fits under 16MB.
**Expected impact:** Should at least match our 1.1143 single-seed with the budget headroom used for int7 on the most sensitive layers.
**Effort:** 1-2 hours code change.

### Online N-gram Agreement (PR #1145)
**What:** Eval-time ensemble that maintains causal prefix experts (token n-gram, within-word, word-start). Boosts model distribution for agreed tokens.
**Impact:** PR #1145 achieved 1.1109 with this on the same base stack.
**Risk:** May conflict with our gated skips / EngramLite.
**Effort:** 2-3 hours to port and test.

### Disable Coprime Stride (Test)
**Why:** Community feedback says coprime stride "had a slightly negative impact" on some stacks.
**Effort:** Need to add a toggle env var and test. 30 min code + 15 min run.

### Try More Seeds
**Why:** With std=0.0009, a lucky 3-seed combination could cross threshold.
**Math:** P(3-seed mean < 1.1144 | true mean = 1.1151, std = 0.0009) ≈ 22%.
**Cost:** ~$5 per 3-seed batch.

## Tier 2: Reach 1.110 BPB (Competitive with Top PRs)

### Scylla Tokenizer (PR #1143)
**Impact:** 0.028 BPB improvement — the single largest lever in the competition.
**What's needed:**
1. Install TokenMonster (`pip install tokenmonster`)
2. Download raw FineWeb docs (~50GB)
3. Retokenize all 80 shards with Scylla vocab (998 tokens)
4. Export in competition binary format
5. Update byte accounting to use pre-computed metadata
**Time:** 3-5 hours (mostly retokenization)
**Risk:** Medium — the retokenization pipeline needs careful implementation.

### CUTLASS EVT Backward Fusion (PR #1105)
**Impact:** -3.7% step time, +500 training steps
**What:** Fuses `(grad @ W_down) * act_grad` into GEMM epilogue. The intermediate never touches HBM.
**Effort:** Hard — requires CUTLASS/CuTe programming.

### MUD Optimizer
**Impact:** 1.3-2.6x faster peak tokens/s than Muon. Drop-in replacement.
**What:** Triangular whitening instead of Newton-Schulz.
**Source:** arXiv:2603.17970
**Effort:** Moderate — need to implement or find existing code.

### Soft-Round QAT (from PR #1089)
**Impact:** Unknown on our stack, but PR #1089 includes it.
**What:** Replace STE with sigmoid-based soft rounding (α ramp 1→16). Smoother gradients than hard STE.
**Effort:** 1-2 hours.

### AOL Polar Express Coefficients (from PR #1089)
**Impact:** Potentially better NS approximation per iteration.
**What:** Different polynomial coefficients for Newton-Schulz. PR #1089 uses `_AOL_POLAR_COEFFS` with tuned values per iteration.
**Effort:** 30 min to port.

## Tier 3: Reach 1.080 BPB (Competition Leader)

### Scylla + Our Full Stack
**What:** Combine the best tokenizer (Scylla, 998 tokens) with our full innovation stack (EngramLite, gated skips, Full GPTQ, FA3, etc.)
**Expected:** ~1.085 BPB (better than either alone)
**Why:** #1143 uses the old SOTA base (no EngramLite, no GPTQ, no gated skips). Our stack improvements should transfer to a better tokenizer.

### Tokenizer Search
**What:** Run our own autoresearch loop to find an even better tokenizer than Scylla.
**Why:** #1143 found Scylla through iterative search from TokenMonster english-1024-clean-v1. The search space is likely not exhausted.
**Effort:** Days of compute for the search loop.

## Infrastructure Lessons for Next Deploy

1. **Always use PyTorch 2.9.1+cu128** — it's the proven working config for FA3.
2. **FA3 pre-built wheel:** `pip install "https://download.pytorch.org/whl/cu128/flash_attn_3-3.0.0-cp39-abi3-manylinux_2_28_x86_64.whl"` — installs in seconds.
3. **Avoid CUDA 13.0 nodes** if possible — can cause `cudaErrorDevicesUnavailable`.
4. **GPTQ_RESERVE_MS=9000** — actual GPTQ takes ~7s, 14s wastes training budget.
5. **Brotli > LZMA** for compression (saves ~0.4 MB).
6. **Don't mix innovations from different stacks without testing** — ResidLambdas and tuned LRs from PR #1130 hurt our stack.
7. **test one change at a time** on H100 — each run costs ~$5 and takes 15 min.

## Cost-Effective Experiment Design

**Per run cost:** ~$5 (15 min on 8xH100 at $20/hr)
**Per 3-seed cost:** ~$15
**Budget for next session:** Plan for 6-10 runs ($30-50)

**Recommended experiment order:**
1. Fixed mixed int6/int7 (int6 baseline, promote sensitive to int7) — 1 seed
2. If better: 3-seed confirmation
3. If not: online n-gram agreement — 1 seed
4. If better: 3-seed confirmation
5. If neither works: Scylla tokenizer investment
