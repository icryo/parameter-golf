# Local Sweep Results (RTX 4090, 300 steps, AdamW, Scylla data)

## Confirmed Improvements (env vars only)

| Env Var | Default | Optimal | Local Impact (nats) |
|---------|---------|---------|---------------------|
| TIED_EMBED_INIT_STD | 0.005 | **0.015** | **-0.50** |
| BIGRAM_VOCAB_SIZE | 2816 | **4096** | -0.26 |
| QK_GAIN_INIT | 1.5 | **4.0** | -0.08 |
| VE_LAYERS | 9,10 | **7,9,10** | -0.002 |
| LOGIT_SOFTCAP | 30 | 40 | -0.001 (noise) |

## Confirmed No-Change (defaults are optimal)

| Param | Tested Values | Result |
|-------|--------------|--------|
| BIGRAM_DIM | 64, 96, 112, 128, 160 | 112 is best |
| VE_DIM | 128, 196 | 128 is best |
| NUM_KV_HEADS | 2, 4, 8 | 4 (GQA) is best |
| MLP_MULT | 3.0, 3.5 | 3.0 (3.5 doesn't fit 16MB) |
| ROPE_DIMS | 16, 32 | 16 is best |
| XSA_LAST_N | 7, 11 | 11 is best |

## QK-Gain Sweep
1.5→4.0: monotonic improvement. 4.0 is peak. 5.0+ gets worse.

## Embed Init Sweep
0.002→0.015: monotonic improvement. 0.015 is peak. 0.02+ gets worse.

## Tokenizer Search
- english-2048-clean: +20% bytes/token but harder prediction. Untested on H100.
- All other TokenMonster families (code, fiction) worse on FineWeb.
- Pruning 2048→1949 doesn't change efficiency.
- Custom training not available in tokenmonster 1.1.12.

## Training Dynamics (AdamW — may not transfer to Muon)

| Param | Tested | Best | Local Impact |
|-------|--------|------|-------------|
| Warmdown fraction | 0.3-0.7 | **0.3** | -0.055 nats |
| Learning rate | 0.0005-0.002 | 0.0008 | -0.003 (AdamW-specific) |
| Weight decay | 0.01-0.1 | 0.1 | -0.005 (AdamW-specific) |
| Grad clip | 0.1-2.0 | all similar | No effect |

Note: LR and WD are optimizer-specific and won't transfer to Muon.
Warmdown fraction translates to WARMDOWN_ITERS on H100 — try 2500 instead of 3500.

## Embed Init Persistence Across Training Length

| Steps | embed=0.005 | embed=0.015 | Delta |
|-------|-------------|-------------|-------|
| 100 | 5.433 | 4.972 | -0.461 |
| 200 | 4.885 | 4.244 | -0.641 |
| 400 | 4.235 | 3.679 | -0.556 |
| 600 | 3.874 | 3.454 | -0.420 |

Advantage is consistent — NOT just faster warmup. Strongly suggests transfer to H100.

## Depth/Width Tradeoffs (200 steps, smaller batch)

| Config | Loss | Params | Artifact | Fits? |
|--------|------|--------|----------|-------|
| 10L 512d | 4.2395 | 24.8M | 14.3MB | ✓ Best |
| 11L 512d | 4.2453 | 27.2M | 15.7MB | ✓ |
| 12L 512d | 4.2498 | 29.5M | 17.1MB | ✗ |
| 13L 448d | 4.3634 | 24.6M | 14.2MB | ✓ |

11L 512d is near-optimal. 10L saves artifact space but marginal quality diff.

## Softcap Fine Sweep

35-45 range: all within 0.009 nats. 42 marginally best. Not worth tuning.

## Caveat
All results are 300-step AdamW on 1 shard. Muon at 7000 steps on 80+ shards
may behave differently. The embed init finding is the highest risk to not transfer.
