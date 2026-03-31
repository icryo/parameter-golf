# Tokenizer Plan: Path to #1

## Architecture

```
[Scylla tokenizer (998 tokens)]
  + [PR #1060 training stack (GPTQ, XSA-all, coprime loader)]
  + [FA3 attention]
  + [Legal TTT (works with custom tokenizers per #1143)]
  = projected ~1.070-1.080 BPP
```

## What We Need

### Already Have
- [x] Scylla vocab file (`scylla_candidate.vocab`, 25KB)
- [x] Scylla metadata (`scylla_candidate.meta.npz`, validated)
- [x] PR #1060 training code (`train_gpt_pr1060.py`)
- [x] PR #1143 training code (`train_gpt_scylla.py`, has metadata loading)
- [x] TokenMonster installed locally
- [x] FA3 deployment recipe (proven)
- [x] Retokenization script scaffolded (`retokenize.py`)

### Need to Build
1. **Retokenize FineWeb with Scylla vocab** (~3 hours on H100)
   - Download raw docs (~50GB) via `--with-docs` flag
   - Run retokenize.py with Scylla vocab
   - Output: 79 train shards + 1 val shard in `fineweb10B_scylla/`

2. **Merge PR #1060 stack + Scylla metadata loading**
   - Graft `load_tokenizer_luts()` from train_gpt_scylla.py into train_gpt_pr1060.py
   - Add TOKENIZER_META_PATH env var
   - Update VOCAB_SIZE default to 998
   - Test with standard sp1024 data first (fallback path)

3. **Validate byte accounting**
   - Run a quick eval with Scylla tokenizer
   - Compare BPP numbers against PR #1143's published results

## Execution Plan

### Phase 1: Code prep (local, no GPU)
- Merge tokenizer loading code
- Update retokenize.py to match PR #1143's shard format exactly
- Test locally with small data sample

### Phase 2: Retokenization (1xH100 or CPU, ~3 hours)
- Download raw docs
- Retokenize all 6.3M documents
- Validate shard format

### Phase 3: Training (8xH100, ~15 min)
- Run PR #1060 stack with Scylla data
- Expected: ~1.083 BPP (sliding, no TTT)
- With TTT: ~1.080 BPP (matching or beating #1143)

### Phase 4: Stack improvements on Scylla base
- Try EngramLite (if it helps on Scylla vocab)
- Optimize bigram hash for 998-token vocab
- Tune hyperparameters

## Cost Estimate
- Retokenization: ~$5 (1 GPU × 3 hours)
- Training runs: ~$15-30 (3 seeds × $5 each)
- Total: ~$20-35 for the full attempt

## Risk Assessment
- **High confidence:** Retokenization works (standard pipeline, just different tokenizer)
- **High confidence:** Training runs (PR #1060's code is proven)
- **Medium confidence:** BPP improvement over #1143 (they use old stack, we use modern stack)
- **Low risk:** Byte accounting (using their validated metadata)
