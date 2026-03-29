#!/bin/bash
# Full H100 experiment: train once, eval many configs
# Maximizes information per dollar on rented 8xH100
#
# Budget: ~1 hour total (~$25-30)
#   - Setup: 10 min
#   - Training: 10 min (identical to SOTA)
#   - Eval sweep: 40 min (6-8 configs on same model)
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
echo "PARAMETER GOLF: Full H100 Experiment Suite"
echo "Seed: $SEED | GPUs: $NPROC | $(date)"
echo "============================================================"

# === PHASE 1: Train (identical to SOTA, ~10 min) ===
echo ""
echo "=== PHASE 1: Training (SOTA reproduction) ==="
export RUN_ID="h100_s${SEED}"
export SEED="$SEED"
export DATA_PATH="./data/datasets/fineweb10B_sp1024"
export TOKENIZER_PATH="./data/tokenizers/fineweb_1024_bpe.model"
export MAX_WALLCLOCK_SECONDS=600
export ITERATIONS=9000 WARMUP_STEPS=20 WARMDOWN_ITERS=3500
export TRAIN_BATCH_TOKENS=786432 TRAIN_SEQ_LEN=2048
export EVAL_SEQ_LEN=2048 EVAL_STRIDE=64
export NUM_LAYERS=11 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=3
export TIE_EMBEDDINGS=1 XSA_LAST_N=4 ROPE_DIMS=16 LN_SCALE=1
export VE_ENABLED=1 VE_DIM=128 VE_LAYERS="9,10"
export BIGRAM_VOCAB_SIZE=2048 BIGRAM_DIM=128 LOGIT_SOFTCAP=30.0
export MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035
export MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 MUON_MOMENTUM_WARMUP_STEPS=1500
export MUON_WD=0.04 ADAM_WD=0.04 GRAD_CLIP_NORM=0.3
export SWA_ENABLED=1 SWA_EVERY=50
export LATE_QAT_THRESHOLD=0.15
export EVAL_TEMPERATURE=0
# Disable TTT in training script — we'll run it manually below
export TTT_ENABLED=0

torchrun --standalone --nproc_per_node=$NPROC train_gpt_safe.py

echo ""
echo "=== Training complete. Model saved to final_model.int6.ptz ==="
echo "=== Now running eval sweep on saved checkpoint ==="

# === PHASE 2: Eval sweep (multiple configs, ~40 min) ===
# Each eval_val_sliding takes ~74s, each TTT takes ~410s
# Uses all 8 GPUs for fast eval via torchrun

cat > /tmp/eval_sweep.py << 'PYEOF'
import torch, time, math, io, lzma, sys, os
import torch.distributed as dist

sys.path.insert(0, '.')
from train_gpt_safe import (
    Hyperparameters, GPT, CastedLinear, eval_val_sliding, eval_val_sliding_ttt,
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

# Load tokenizer and validation data
sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
    sp, args.vocab_size, device)

# Load quantized model
with open('final_model.int6.ptz', 'rb') as f:
    quant_blob = f.read()
quant_state = torch.load(io.BytesIO(lzma.decompress(quant_blob)), map_location='cpu')
sd_cpu = {k: v.detach().cpu() for k, v in torch.load('final_model.pt', map_location='cpu', weights_only=True).items()}
unbanked = _unbank_state_dict(sd_cpu, args.num_layers)
deq = dequantize_mixed_int6(quant_state['w'], quant_state['m'], unbanked)
deq_banked = _rebank_state_dict(deq, args.num_layers, sd_cpu)

def load_eval_model(softcap_scale=1.0):
    m = GPT(
        vocab_size=args.vocab_size, num_layers=args.num_layers, model_dim=args.model_dim,
        num_heads=args.num_heads, num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings, tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap * softcap_scale, rope_base=args.rope_base,
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

def run_sliding(label, temp=1.0, stride=64):
    log(f'  [{label}] T={temp:.2f} stride={stride}...')
    model = load_eval_model(softcap_scale=temp)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    loss, bpb = eval_val_sliding(
        args, model, rank, world_size, device, val_tokens,
        base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
        stride=stride, eval_seq_len=args.eval_seq_len,
    )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    log(f'  [{label}] val_bpb={bpb:.8f} time={elapsed:.1f}s')
    del model; torch.cuda.empty_cache()
    return bpb, elapsed

def run_ttt(label, temp=1.0, ttt_lr=0.002, ttt_epochs=3, freeze_blocks=0, stride=64):
    log(f'  [{label}] T={temp:.2f} lr={ttt_lr} ep={ttt_epochs} freeze={freeze_blocks}...')
    model = load_eval_model(softcap_scale=temp)
    args.ttt_lr = ttt_lr
    args.ttt_epochs = ttt_epochs
    args.ttt_freeze_blocks = freeze_blocks
    args.ttt_chunk_tokens = 32768
    args.ttt_momentum = 0.9
    args.ttt_batch_seqs = 32
    args.ttt_grad_clip = 1.0
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    loss, bpb = eval_val_sliding_ttt(
        args, model, rank, world_size, device, val_tokens,
        base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
        stride=stride, log0=log,
    )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    log(f'  [{label}] val_bpb={bpb:.8f} time={elapsed:.1f}s')
    del model; torch.cuda.empty_cache()
    return bpb, elapsed

log('')
log('=' * 60)
log('EVAL SWEEP: Temperature + Stride + TTT configs')
log('=' * 60)
results = {}

# --- Block 1: Temperature sweep at stride=64 (~6 x 74s = 444s) ---
log('')
log('--- Temperature sweep (stride=64, no TTT) ---')
for t in [0.85, 0.88, 0.90, 0.92, 0.95, 1.00]:
    bpb, _ = run_sliding(f'temp_{t:.2f}', temp=t, stride=64)
    results[f'sliding_T{t:.2f}_s64'] = bpb

# Find best temperature
best_t = min([0.85, 0.88, 0.90, 0.92, 0.95, 1.00],
             key=lambda t: results[f'sliding_T{t:.2f}_s64'])
best_bpb = results[f'sliding_T{best_t:.2f}_s64']
baseline_bpb = results['sliding_T1.00_s64']
temp_delta = best_bpb - baseline_bpb
log(f'  >>> Best T={best_t:.2f} bpb={best_bpb:.8f} delta={temp_delta:+.8f} <<<')

# --- Block 2: TTT sweep with best temperature (~3 x 410s = 1230s) ---
log('')
log(f'--- TTT sweep (T={best_t:.2f}, stride=64) ---')

# Config A: SOTA TTT (baseline)
bpb_a, t_a = run_ttt('ttt_sota', temp=best_t, ttt_lr=0.002, ttt_epochs=3, freeze_blocks=0)
results['ttt_sota'] = bpb_a

# Config B: PR #1039 recipe (claimed 1.1184 BPB — potential record)
bpb_b, t_b = run_ttt('ttt_pr1039', temp=best_t, ttt_lr=0.0025, ttt_epochs=4, freeze_blocks=0)
results['ttt_pr1039'] = bpb_b

# Config C: More epochs (deeper adaptation)
bpb_c, t_c = run_ttt('ttt_5ep', temp=best_t, ttt_lr=0.002, ttt_epochs=5, freeze_blocks=0)
results['ttt_5ep'] = bpb_c

# Config D: Higher LR + more epochs (aggressive)
bpb_d, t_d = run_ttt('ttt_hiLR_4ep', temp=best_t, ttt_lr=0.003, ttt_epochs=4, freeze_blocks=0)
results['ttt_hiLR_4ep'] = bpb_d

# --- Summary ---
log('')
log('=' * 60)
log('RESULTS SUMMARY')
log('=' * 60)

log('')
log('Temperature sweep (stride=64, no TTT):')
for t in [0.85, 0.88, 0.90, 0.92, 0.95, 1.00]:
    bpb = results[f'sliding_T{t:.2f}_s64']
    delta = bpb - baseline_bpb
    marker = ' <<<' if t == best_t else ''
    log(f'  T={t:.2f}  bpb={bpb:.8f}  delta={delta:+.8f}{marker}')

log('')
log('TTT configurations:')
log(f'  SOTA (lr=0.002, 3ep):         bpb={bpb_a:.8f}')
log(f'  PR1039 (lr=0.0025, 4ep):      bpb={bpb_b:.8f}  delta={bpb_b-bpb_a:+.8f}')
log(f'  5 epochs (lr=0.002, 5ep):     bpb={bpb_c:.8f}  delta={bpb_c-bpb_a:+.8f}')
log(f'  Aggressive (lr=0.003, 4ep):   bpb={bpb_d:.8f}  delta={bpb_d-bpb_a:+.8f}')

log('')
best_ttt = min(bpb_a, bpb_b, bpb_c, bpb_d)
log(f'SOTA reference (seed 1337): 1.11922988')
log(f'Our best result:            {best_ttt:.8f}')
log(f'Delta vs SOTA:              {best_ttt - 1.11922988:+.8f}')
log(f'Record threshold:           1.1144')
log(f'Gap to record:              {best_ttt - 1.1144:+.8f}')

if best_ttt < 1.1144:
    log('>>> RECORD TERRITORY! Run 2 more seeds to confirm. <<<')
elif best_ttt < 1.1192:
    log('>>> Better than SOTA seed, but need 3-seed mean. Worth 2 more seeds. <<<')
elif best_ttt < 1.1200:
    log('>>> Close to SOTA. Temperature or TTT tuning helping marginally. <<<')
else:
    log('>>> At or worse than SOTA. No path to record from these changes. <<<')

if world_size > 1:
    dist.destroy_process_group()
PYEOF

torchrun --standalone --nproc_per_node=$NPROC /tmp/eval_sweep.py

echo ""
echo "============================================================"
echo "All experiments complete. $(date)"
echo "============================================================"
