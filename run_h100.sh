#!/bin/bash
# Full H100 experiment: PR #1060 base (1.1122) + TTT with reset
#
# What we test (train once, eval many):
#   1. PR #1060 baseline reproduction (no TTT): ~87s eval
#   2. TTT with SOTA config (lr=0.002, 3ep): ~410s
#   3. TTT with PR #1039 config (lr=0.0025, 4ep): ~410s
#   4. TTT with periodic reset every 100 chunks: ~410s
#   5. TTT with periodic reset every 50 chunks: ~410s
#
# Total: ~10min train + 87s baseline + 4×410s TTT ≈ 37 min
# Cost: ~$15-20 on 8xH100
#
# Usage:
#   git clone https://github.com/icryo/parameter-golf.git
#   cd parameter-golf && git checkout experiments/triton-kernels-qat-fix
#   pip install sentencepiece huggingface-hub datasets flash-attn
#   python3 data/cached_challenge_fineweb.py --variant sp1024
#   ./run_h100.sh 2>&1 | tee full_experiment.log
set -euo pipefail
SEED="${1:-1337}"
NPROC=8

echo "============================================================"
echo "PARAMETER GOLF: PR #1060 base + TTT reset experiments"
echo "Seed: $SEED | GPUs: $NPROC | $(date)"
echo "============================================================"

# === PHASE 1: Train with PR #1060 config ===
echo ""
echo "=== PHASE 1: Training (PR #1060: coprime loader + Full GPTQ + XSA-all) ==="

export RUN_ID="pr1060_ttt_s${SEED}"
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
export BIGRAM_VOCAB_SIZE=2816
export BIGRAM_DIM=112
export USE_GPTQ=1
export GPTQ_RESERVE_MS=14000
export TTT_ENABLED=0

torchrun --standalone --nproc_per_node=$NPROC train_gpt_ours.py

echo ""
echo "=== Training complete. Now running TTT sweep. ==="

# === PHASE 2: TTT sweep on saved checkpoint ===
cat > /tmp/ttt_sweep.py << 'PYEOF'
import torch, time, math, io, lzma, sys, os
import torch.distributed as dist
sys.path.insert(0, '.')
from train_gpt_ours import (
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
if world_size > 1:
    dist.init_process_group(backend='nccl', device_id=device)
torch.backends.cuda.matmul.allow_tf32 = True
master = (rank == 0)
def log(msg):
    if master: print(msg, flush=True)

sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
    sp, args.vocab_size, device)

with open('final_model.int6.ptz', 'rb') as f:
    quant_blob = f.read()
quant_state = torch.load(io.BytesIO(lzma.decompress(quant_blob)), map_location='cpu')
sd_cpu = {k: v.detach().cpu() for k, v in
          torch.load('final_model.pt', map_location='cpu', weights_only=True).items()}
unbanked = _unbank_state_dict(sd_cpu, args.num_layers)
deq = dequantize_mixed_int6(quant_state['w'], quant_state['m'], unbanked)
deq_banked = _rebank_state_dict(deq, args.num_layers, sd_cpu)

def load_fresh():
    m = GPT(
        vocab_size=args.vocab_size, num_layers=args.num_layers, model_dim=args.model_dim,
        num_heads=args.num_heads, num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings, tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap, rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init, mtp_num_heads=0, mtp_loss_weight=0.0,
        bigram_vocab_size=args.bigram_vocab_size, bigram_dim=args.bigram_dim,
        xsa_last_n=args.xsa_last_n, rope_dims=args.rope_dims, ln_scale=args.ln_scale,
        dtg=args.dtg_enabled, ve_enabled=args.ve_enabled, ve_dim=args.ve_dim,
        ve_layers=args.ve_layers, gated_attention=args.gated_attention,
        value_residual=args.value_residual,
    ).to(device).bfloat16()
    m.qo_bank.data = m.qo_bank.data.float()
    m.kv_bank.data = m.kv_bank.data.float()
    m.mlp_up_bank.data = m.mlp_up_bank.data.float()
    m.mlp_down_bank.data = m.mlp_down_bank.data.float()
    for mod in m.modules():
        if isinstance(mod, CastedLinear): mod.float()
    restore_low_dim_params_to_fp32(m)
    m.load_state_dict(deq_banked, strict=True)
    return m

def run_ttt(label, ttt_lr, ttt_epochs, freeze_blocks=0, reset_every=0):
    log(f'  [{label}] lr={ttt_lr} ep={ttt_epochs} reset={reset_every}...')
    model = load_fresh()
    args.ttt_lr = ttt_lr
    args.ttt_epochs = ttt_epochs
    args.ttt_freeze_blocks = freeze_blocks
    args.ttt_chunk_tokens = 32768
    args.ttt_momentum = 0.9
    args.ttt_batch_seqs = 32
    args.ttt_grad_clip = 1.0
    args.ttt_reset_every = reset_every
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    loss, bpb = eval_val_sliding_ttt(
        args, model, rank, world_size, device, val_tokens,
        base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
        stride=64, log0=log,
    )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    log(f'  [{label}] val_bpb={bpb:.8f} time={elapsed:.1f}s')
    del model; torch.cuda.empty_cache()
    return bpb

log('')
log('=' * 60)
log('TTT SWEEP on PR #1060 (Full GPTQ + XSA-all) quantized model')
log('=' * 60)
results = {}

results['sota_ttt'] = run_ttt('sota_ttt', 0.002, 3)
results['pr1039'] = run_ttt('pr1039', 0.0025, 4)
results['reset100'] = run_ttt('reset100', 0.0025, 4, reset_every=100)
results['reset50'] = run_ttt('reset50', 0.0025, 4, reset_every=50)

log('')
log('=' * 60)
log('RESULTS')
log('=' * 60)
log(f'PR #1060 no-TTT baseline: 1.1122 (their submission)')
log('')
for label, bpb in sorted(results.items(), key=lambda x: x[1]):
    log(f'  {label:<20} bpb={bpb:.8f}  vs_noTTT={bpb-1.1122:+.6f}  vs_merged_SOTA={bpb-1.1194:+.6f}')
best = min(results, key=results.get)
log(f'\nBest: {best} = {results[best]:.8f}')
log(f'Record threshold (vs PR #1060): <= 1.1072')
log(f'Gap: {results[best] - 1.1072:+.8f}')

if world_size > 1:
    dist.destroy_process_group()
PYEOF

torchrun --standalone --nproc_per_node=$NPROC /tmp/ttt_sweep.py

echo ""
echo "============================================================"
echo "All experiments complete. $(date)"
echo "============================================================"
