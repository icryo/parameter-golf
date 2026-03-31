# Parameter Golf: The Full Journey

## Overview

**Goal:** Beat the merged SOTA (1.1194 BPB) by ≥0.005 nats on the OpenAI Parameter Golf challenge.
**Result:** 1.1151 BPB (3-seed mean). 0.0043 below SOTA but 0.0007 short of formal record threshold.
**Best single seed:** 1.1143 BPB (crosses threshold individually).
**Spend:** ~$100-150 across 5 vast.ai 8xH100 sessions.
**Duration:** ~24 hours of active work across March 28-30, 2026.

## Timeline

| Phase | Time | What | Outcome |
|-------|------|------|---------|
| Research | 8h | Deep dive on competition, top 3 PRs, TTT, QAT, Mamba | Mapped the landscape |
| Local experiments | 4h | QAT fix, Triton kernels, Muon interaction | QAT irrelevant with Muon, kernels can't beat torch.compile |
| H100 Run 1 (SDPA) | 20min | PR #1060 base, no FA3 | 1.1165 BPB — beat SOTA but SDPA bottleneck |
| H100 Run 2 (FA2) | 20min | Same + FA2 shim | 1.1172 BPB — FA2 no faster than SDPA at seq2048 |
| TTT experiments | 30min | TTT sweep + reset | TTT neutral on Full GPTQ stack |
| PR research pivot | 2h | Discovered PRs #1060, #1072, #1089 moved the frontier | Decoded #1 submission (PR #1089) |
| H100 Run 3 (FA3!) | 15min | Merged stack + FA3 + PR #1089 innovations | **1.1146 BPB** — near record! |
| 3-seed confirmation | 30min | Seeds 42, 2025 | Mean 1.1152, std 0.0007 |
| Brotli compression | 15min | Replace LZMA with Brotli | 1.1143 (seed 1337) — best single seed |
| Brotli 3-seed | 30min | Full confirmation | Mean 1.1151 — 0.0007 from record |
| ResidLambdas attempt | 30min | Ported from PR #1130 | 1.1257 — WORSE, reverted |
| More seeds | 15min | Seed 7 | 1.1244 — high variance |

## Key Metrics Across All Runs

See [01-RUNS.md](01-RUNS.md) for detailed per-run data.

## Documents in this series

1. [00-JOURNEY.md](00-JOURNEY.md) — This overview
2. [01-RUNS.md](01-RUNS.md) — Every H100 run with exact numbers
3. [02-INNOVATIONS.md](02-INNOVATIONS.md) — What we tried, what worked, what didn't
4. [03-LOCAL-EXPERIMENTS.md](03-LOCAL-EXPERIMENTS.md) — RTX 4090 local validation results
5. [04-COMPETITIVE-LANDSCAPE.md](04-COMPETITIVE-LANDSCAPE.md) — Analysis of top PRs
6. [05-DEAD-ENDS.md](05-DEAD-ENDS.md) — Confirmed failures and why
7. [06-NEXT-STEPS.md](06-NEXT-STEPS.md) — Research directions for the next attempt
