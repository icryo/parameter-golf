# Competitive Landscape (as of March 30, 2026)

## Leaderboard (Open PRs)

| # | PR | BPB | Seeds | Key Innovation | Status |
|---|-----|-----|-------|----------------|--------|
| 1 | #1143 | **1.0806** | 3 | **Scylla tokenizer** (TokenMonster) + TTT | Under review |
| 2 | #1089 | 1.1086 | 3 | Turbo-Muon + EngramLite + int6/int7 + Brotli | Open |
| 3 | #1145 | 1.1109 | 4 | Online n-gram agreement augmentation | Open |
| 4 | #1135 | 1.1116 | 3 | Fused Triton MLP + Full GPTQ + coprime | Open |
| 5 | #1060 | 1.1122 | 3 | Coprime loader + Full GPTQ + XSA-all | Open |
| 6 | #1130 | 1.1140 | 12 | ResidLambdas + Split-LR + GPTQ | Open |
| 7 | **#1122 (ours)** | **1.1151** | 3 | EngramLite + gated skips + GPTQ + FA3 | Open |
| — | *Merged SOTA* | *1.1194* | *3* | *LeakyReLU² + TTT + Parallel Muon* | *Merged* |

## Technique Evolution

### Phase 1: Architecture (Mar 18-20)
- 9L → 11L transformer (+0.02 BPB)
- Int8 → Int6 quantization (+0.015 BPB)
- 3x MLP expansion (+0.020 BPB)
- GQA, U-Net skips, SmearGate

### Phase 2: Optimization (Mar 20-23)
- EMA weight averaging (+0.0006)
- Partial RoPE, LN Scale (+0.002 each)
- XSA on last 4 layers (+0.002)
- LeakyReLU(0.5)² (+0.003)
- Legal TTT (+0.0025)
- Parallel Muon / Parameter Banking (speed, quality-neutral)

### Phase 3: Quantization (Mar 23-28)
- GPTQ-lite (5 percentile search) (+0.0006)
- Full Hessian GPTQ (+0.003 over lite)
- Mixed int6/int7 (PR #1089)
- Brotli + byte-shuffle compression

### Phase 4: Data & Tokenization (Mar 28-30)
- Coprime-stride data loading (+0.002)
- **Scylla tokenizer** (+0.028 — biggest single innovation)
- Online n-gram augmentation (+0.001-0.003)

## Key Insight: The Tokenizer Frontier

PR #1143 (Scylla) achieved 1.0806 BPB — 0.028 better than the best pure-architecture submission (#1089 at 1.1086). **The tokenizer is the single biggest lever remaining.** All architecture/quantization/optimizer innovations combined give ~0.01 BPB improvement. The tokenizer gives 0.028 alone.

TokenMonster's ungreedy multi-branch search produces ~37.5% fewer tokens at equivalent vocab size compared to BPE. The Scylla variant (998 tokens, pruned from english-1024-clean-v1) was discovered through iterative autoresearch.

## Untried Combinations (Potential for Record)

| Combination | Est. BPB | Difficulty |
|-------------|---------|------------|
| Scylla tokenizer + our stack (EngramLite+GPTQ+XSA-all) | ~1.085 | High (retokenization needed) |
| Our stack + online n-gram agreement (#1145) | ~1.112 | Moderate |
| Our stack + fixed mixed int6/int7 | ~1.112 | Moderate |
| Our stack + MUD optimizer (faster than Muon) | ~1.113 | Easy |
| Our stack + CUTLASS EVT backward fusion | ~1.112 | Hard |

## The N-gram Cache Wave (Mar 25-27)

A wave of submissions used n-gram eval-time caches to achieve impossibly low BPB (<0.5). 33+ PRs were closed on Mar 27 after a normalization bug was identified. The partition function inflation issue (see PR #1147) means hashed n-gram caches produce invalid BPB scores.

Post-enforcement, legitimate approaches include:
- Legal backward-looking TTT
- Causal n-gram agreement (PR #1145's approach — boosting model distribution, not replacing it)
- Tokenizer optimization (Scylla)
