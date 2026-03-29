#!/usr/bin/env python3
"""Local QAT experiment: measure quantization gap with and without QAT.

Trains two models (~300 steps each) on 1x4090, then quantizes both to int6
and measures the BPB degradation. QAT should reduce the gap.

Usage: source .venv/bin/activate && python3 experiment_qat.py
"""
import copy
import io
import math
import os
import sys
import time

os.environ["PYTHONPATH"] = os.path.dirname(__file__) + ":" + os.environ.get("PYTHONPATH", "")

# Ensure flash_attn shim exists
shim_dir = os.path.join(os.path.dirname(__file__), "flash_attn_interface")
os.makedirs(shim_dir, exist_ok=True)
shim_file = os.path.join(shim_dir, "__init__.py")
if not os.path.exists(shim_file) or os.path.getsize(shim_file) < 100:
    with open(shim_file, "w") as f:
        f.write("""
import torch, torch.nn.functional as F
def flash_attn_func(q, k, v, causal=False):
    B, T, H, D = q.shape
    Hkv = k.shape[2]
    group = H // Hkv
    if group > 1:
        k = k.unsqueeze(3).expand(B, T, Hkv, group, D).reshape(B, T, H, D)
        v = v.unsqueeze(3).expand(B, T, Hkv, group, D).reshape(B, T, H, D)
    q, k, v = q.transpose(1,2), k.transpose(1,2), v.transpose(1,2)
    return F.scaled_dot_product_attention(q, k, v, is_causal=causal).transpose(1,2)
""")

import torch
import torch.nn.functional as F
import numpy as np

# Now import our modules
from train_gpt_kernels import (
    GPT, CastedLinear, Muon, Hyperparameters,
    quantize_int6_per_row, _unbank_state_dict, mixed_quantize_int6,
    dequantize_mixed_int6, load_data_shard, build_sentencepiece_luts,
    restore_low_dim_params_to_fp32,
)
import sentencepiece as spm
from pathlib import Path
import glob


def create_model(device):
    model = GPT(
        vocab_size=1024, num_layers=11, model_dim=512,
        num_heads=8, num_kv_heads=4, mlp_mult=3,
        tie_embeddings=True, tied_embed_init_std=0.005,
        logit_softcap=30.0, rope_base=10000.0, qk_gain_init=1.5,
        bigram_vocab_size=2048, bigram_dim=128, xsa_last_n=4,
        rope_dims=16, ln_scale=True, ve_enabled=True, ve_dim=128, ve_layers="9,10",
    ).to(device).bfloat16()
    model.qo_bank.data = model.qo_bank.data.float()
    model.kv_bank.data = model.kv_bank.data.float()
    model.mlp_up_bank.data = model.mlp_up_bank.data.float()
    model.mlp_down_bank.data = model.mlp_down_bank.data.float()
    for m in model.modules():
        if isinstance(m, CastedLinear):
            m.float()
    restore_low_dim_params_to_fp32(model)
    return model


def eval_loss(model, val_x, val_y, device):
    """Quick eval on a batch."""
    model.eval()
    with torch.inference_mode(), torch.autocast("cuda", torch.bfloat16):
        losses = []
        bs = 4
        for i in range(0, val_x.shape[0], bs):
            loss = model(val_x[i:i+bs].to(device), val_y[i:i+bs].to(device))
            losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def quantize_and_measure(model, val_x, val_y, device, label):
    """Quantize model to int6, dequantize, measure loss gap."""
    # Pre-quantization loss
    pre_loss = eval_loss(model, val_x, val_y, device)

    # Unbank -> int6 quantize -> dequantize -> rebank
    sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    unbanked = _unbank_state_dict(sd, 11)
    quant_result, quant_meta = mixed_quantize_int6(unbanked, {"mlp", "attn"})

    # Measure compressed size
    buf = io.BytesIO()
    torch.save({"w": quant_result, "m": quant_meta}, buf)
    raw_bytes = len(buf.getvalue())

    deq = dequantize_mixed_int6(quant_result, quant_meta, unbanked)

    # Rebank
    from train_gpt_kernels import _rebank_state_dict
    deq_banked = _rebank_state_dict(deq, 11, sd)

    # Load dequantized weights
    eval_model = create_model(device)
    eval_model.load_state_dict(deq_banked, strict=True)

    # Post-quantization loss
    post_loss = eval_loss(eval_model, val_x, val_y, device)

    gap = post_loss - pre_loss
    print(f"  {label}:")
    print(f"    Pre-quant loss:  {pre_loss:.4f}")
    print(f"    Post-quant loss: {post_loss:.4f}")
    print(f"    Quant gap:       {gap:+.4f} ({gap/pre_loss*100:+.2f}%)")
    print(f"    Artifact size:   {raw_bytes/1e6:.2f} MB")
    return pre_loss, post_loss, gap


def train_model(model, train_tokens, val_x, val_y, device, qat_mode, steps=400):
    """Train for N steps with specified QAT mode."""
    torch.manual_seed(42)
    np.random.seed(42)

    seq_len = 1024
    batch_tokens = 32768
    batch_seqs = batch_tokens // seq_len

    # Optimizer setup (simplified — no Muon, just Adam for local testing)
    all_params = list(model.parameters())
    optimizer = torch.optim.AdamW(all_params, lr=0.001, betas=(0.9, 0.95), weight_decay=0.04)

    # QAT configuration
    CastedLinear._qat_state[0] = False
    CastedLinear._noisy_qat[0] = (qat_mode == "noisy")
    qat_start_step = int(steps * 0.5) if qat_mode != "none" else steps + 1

    model.train()
    pos = 0
    t0 = time.perf_counter()

    for step in range(1, steps + 1):
        # Enable QAT at the right step
        if step == qat_start_step and qat_mode != "none":
            CastedLinear._qat_state[0] = True
            print(f"    QAT enabled at step {step} (mode={qat_mode})")

        # LR schedule: warmup 10 steps, linear decay in last 50%
        if step <= 10:
            lr_scale = step / 10
        elif step >= steps // 2:
            lr_scale = max(0.0, 2.0 * (1.0 - step / steps))
        else:
            lr_scale = 1.0

        for pg in optimizer.param_groups:
            pg['lr'] = 0.001 * lr_scale

        # Get batch
        if pos + batch_seqs * seq_len + 1 > train_tokens.numel():
            pos = 0
        x = train_tokens[pos:pos + batch_seqs * seq_len].reshape(batch_seqs, seq_len).to(device, dtype=torch.int64)
        y = train_tokens[pos + 1:pos + batch_seqs * seq_len + 1].reshape(batch_seqs, seq_len).to(device, dtype=torch.int64)
        pos += batch_seqs * seq_len

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", torch.bfloat16):
            loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(all_params, 1.0)
        optimizer.step()

        if step % 100 == 0 or step == 1:
            elapsed = time.perf_counter() - t0
            print(f"    step {step}/{steps}  loss={loss.item():.4f}  lr={lr_scale:.3f}  "
                  f"time={elapsed:.1f}s  ms/step={elapsed/step*1000:.1f}")

    CastedLinear._qat_state[0] = False
    elapsed = time.perf_counter() - t0
    print(f"    Training done: {elapsed:.1f}s ({elapsed/steps*1000:.1f} ms/step)")
    return model


def main():
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True

    print("=" * 60)
    print("QAT EXPERIMENT: Measuring quantization gap reduction")
    print("=" * 60)

    # Load data
    data_path = "./data/datasets/fineweb10B_sp1024"
    train_file = sorted(glob.glob(os.path.join(data_path, "fineweb_train_*.bin")))[0]
    val_file = sorted(glob.glob(os.path.join(data_path, "fineweb_val_*.bin")))[0]

    train_tokens = load_data_shard(Path(train_file))
    val_tokens_raw = load_data_shard(Path(val_file))

    # Prepare small val set
    seq_len = 1024
    n_val = min(64, (val_tokens_raw.numel() - 1) // seq_len)
    val_x = val_tokens_raw[:n_val * seq_len].reshape(n_val, seq_len).to(dtype=torch.int64)
    val_y = val_tokens_raw[1:n_val * seq_len + 1].reshape(n_val, seq_len).to(dtype=torch.int64)

    print(f"Train tokens: {train_tokens.numel():,}")
    print(f"Val sequences: {n_val}")

    STEPS = 400
    results = {}

    for qat_mode in ["none", "ste", "noisy"]:
        print(f"\n{'='*60}")
        print(f"EXPERIMENT: QAT mode = {qat_mode}")
        print(f"{'='*60}")

        # Fresh model with same init
        torch.manual_seed(1337)
        model = create_model(device)

        print(f"\n  Training ({STEPS} steps)...")
        model = train_model(model, train_tokens, val_x, val_y, device, qat_mode, steps=STEPS)

        print(f"\n  Quantization measurement...")
        pre, post, gap = quantize_and_measure(model, val_x, val_y, device, qat_mode)
        results[qat_mode] = {"pre": pre, "post": post, "gap": gap}

        del model
        torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"{'Mode':<10} {'Pre-quant':<12} {'Post-quant':<12} {'Gap':<12} {'Gap %':<10}")
    print(f"{'-'*56}")
    for mode, r in results.items():
        print(f"{mode:<10} {r['pre']:<12.4f} {r['post']:<12.4f} {r['gap']:<+12.4f} {r['gap']/r['pre']*100:<+10.2f}%")

    baseline_gap = results["none"]["gap"]
    for mode in ["ste", "noisy"]:
        if mode in results:
            reduction = baseline_gap - results[mode]["gap"]
            print(f"\n  {mode} QAT gap reduction: {reduction:+.4f} ({reduction/abs(baseline_gap)*100:.1f}% of baseline gap)")


if __name__ == "__main__":
    main()
