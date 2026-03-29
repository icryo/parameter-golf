#!/bin/bash
# DEPLOY: Merged PR #1060 + PR #1072 fused kernel + TTT reset
#
# Train once with fused MLP (expected ~70ms/step → ~7900 steps)
# Then sweep 4 TTT configs on the same checkpoint
#
# Usage:
#   pip install sentencepiece huggingface-hub datasets flash-attn triton
#   python3 data/cached_challenge_fineweb.py --variant sp1024
#   ./run_h100.sh 2>&1 | tee experiment.log
set -euo pipefail
SEED="${1:-1337}"
NPROC=8

echo "============================================================"
echo "MERGED: PR1060 (coprime+GPTQ) + PR1072 (fused kernel) + TTT reset"
echo "Seed: $SEED | GPUs: $NPROC | $(date)"
echo "============================================================"

# Common env
export SEED="$SEED"
export DATA_PATH="./data/datasets/fineweb10B_sp1024"
export TOKENIZER_PATH="./data/tokenizers/fineweb_1024_bpe.model"
export MAX_WALLCLOCK_SECONDS=600
export ITERATIONS=9000 WARMUP_STEPS=20 WARMDOWN_ITERS=3500
export TRAIN_BATCH_TOKENS=786432 TRAIN_SEQ_LEN=2048
export EVAL_SEQ_LEN=2048 EVAL_STRIDE=64
export NUM_LAYERS=11 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=3
export TIE_EMBEDDINGS=1 ROPE_DIMS=16 LN_SCALE=1
export VE_ENABLED=1 VE_DIM=128 VE_LAYERS="9,10"
export LOGIT_SOFTCAP=30.0
export MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035
export MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 MUON_MOMENTUM_WARMUP_STEPS=1500
export MUON_WD=0.04 ADAM_WD=0.04 GRAD_CLIP_NORM=0.3
export SWA_ENABLED=1 SWA_EVERY=50
export LATE_QAT_THRESHOLD=0.15
export XSA_LAST_N=11
export BIGRAM_VOCAB_SIZE=2816 BIGRAM_DIM=112
export USE_GPTQ=1 GPTQ_RESERVE_MS=14000
export TTT_ENABLED=0
export RUN_ID="merged_s${SEED}"

echo "=== Phase 1: Training (fused kernel + coprime loader + Full GPTQ) ==="
torchrun --standalone --nproc_per_node=$NPROC train_gpt_merged.py

echo ""
echo "=== Phase 2: TTT sweep (loads saved checkpoint, no re-training) ==="

cat > /tmp/ttt_sweep.py << 'PYEOF'
import torch, time, io, lzma, sys, os
import torch.distributed as dist
sys.path.insert(0, '.')
from train_gpt_merged import (
    Hyperparameters, GPT, CastedLinear, eval_val_sliding_ttt,
    dequantize_mixed_int6, _rebank_state_dict, _unbank_state_dict,
    build_sentencepiece_luts, load_validation_tokens, restore_low_dim_params_to_fp32,
)
import sentencepiece as spm
args = Hyperparameters()
rank = int(os.environ.get('RANK', '0'))
world_size = int(os.environ.get('WORLD_SIZE', '1'))
local_rank = int(os.environ.get('LOCAL_RANK', '0'))
device = torch.device('cuda', local_rank)
torch.cuda.set_device(device)
if world_size > 1: dist.init_process_group(backend='nccl', device_id=device)
torch.backends.cuda.matmul.allow_tf32 = True
master = rank == 0
def log(msg):
    if master: print(msg, flush=True)
sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
luts = build_sentencepiece_luts(sp, args.vocab_size, device)
with open('final_model.int6.ptz', 'rb') as f: qblob = f.read()
qs = torch.load(io.BytesIO(lzma.decompress(qblob)), map_location='cpu')
sd = {k:v.detach().cpu() for k,v in torch.load('final_model.pt', map_location='cpu', weights_only=True).items()}
ub = _unbank_state_dict(sd, args.num_layers)
deq = dequantize_mixed_int6(qs['w'], qs['m'], ub)
rb = _rebank_state_dict(deq, args.num_layers, sd)
def fresh():
    m = GPT(vocab_size=args.vocab_size, num_layers=args.num_layers, model_dim=args.model_dim,
        num_heads=args.num_heads, num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings, tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap, rope_base=args.rope_base, qk_gain_init=args.qk_gain_init,
        mtp_num_heads=0, mtp_loss_weight=0.0, bigram_vocab_size=args.bigram_vocab_size,
        bigram_dim=args.bigram_dim, xsa_last_n=args.xsa_last_n, rope_dims=args.rope_dims,
        ln_scale=args.ln_scale, dtg=args.dtg_enabled, ve_enabled=args.ve_enabled,
        ve_dim=args.ve_dim, ve_layers=args.ve_layers, gated_attention=args.gated_attention,
        value_residual=args.value_residual).to(device).bfloat16()
    for b in ['qo_bank','kv_bank','mlp_up_bank','mlp_down_bank']:
        getattr(m,b).data = getattr(m,b).data.float()
    for mod in m.modules():
        if isinstance(mod, CastedLinear): mod.float()
    restore_low_dim_params_to_fp32(m)
    m.load_state_dict(rb, strict=True)
    return m
def ttt(label, lr, epochs, reset):
    log(f'  [{label}] lr={lr} ep={epochs} reset={reset}...')
    m = fresh()
    args.ttt_lr, args.ttt_epochs, args.ttt_freeze_blocks = lr, epochs, 0
    args.ttt_chunk_tokens, args.ttt_momentum = 32768, 0.9
    args.ttt_batch_seqs, args.ttt_grad_clip, args.ttt_reset_every = 32, 1.0, reset
    torch.cuda.synchronize(); t0 = time.perf_counter()
    loss, bpb = eval_val_sliding_ttt(args, m, rank, world_size, device, val_tokens, *luts, stride=64, log0=log)
    torch.cuda.synchronize()
    log(f'  RESULT [{label}] val_bpb={bpb:.8f} time={time.perf_counter()-t0:.1f}s')
    del m; torch.cuda.empty_cache()
    return bpb
log('\n' + '='*60)
log('TTT SWEEP on merged model checkpoint')
log('='*60)
r = {}
r['sota'] = ttt('sota_ttt', 0.002, 3, 0)
r['pr1039'] = ttt('pr1039', 0.0025, 4, 0)
r['reset100'] = ttt('reset100', 0.0025, 4, 100)
r['reset50'] = ttt('reset50', 0.0025, 4, 50)
log('\n' + '='*60)
log('SUMMARY')
log('='*60)
for k,v in sorted(r.items(), key=lambda x:x[1]):
    log(f'  {k:<12} bpb={v:.8f} vs_PR1060_noTTT={v-1.1122:+.6f}')
best = min(r.values())
log(f'\nBest: {best:.8f}  Record threshold: 1.1072  Gap: {best-1.1072:+.6f}')
if world_size > 1: dist.destroy_process_group()
PYEOF

torchrun --standalone --nproc_per_node=$NPROC /tmp/ttt_sweep.py

echo ""
echo "============================================================"
echo "Done. $(date)"
echo "============================================================"
