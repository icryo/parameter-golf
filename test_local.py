#!/usr/bin/env python3
"""Local 1xGPU test: compare original SOTA QAT (broken) vs fixed QAT.

Runs ~500 steps on 1x4090 with 2 shards. Takes ~5-10 minutes.
Validates:
  1. QAT fix actually activates (check logs for "late_qat:enabled")
  2. Temperature scaling changes eval BPB
  3. Both versions train without errors

Usage:
  source .venv/bin/activate
  python3 test_local.py
"""
import os
import subprocess
import sys
import re

def run_experiment(script: str, name: str, extra_env: dict = None):
    """Run a training script and extract key metrics."""
    env = os.environ.copy()
    # Small local config
    env.update({
        "DATA_PATH": "./data/datasets/fineweb10B_sp1024",
        "TOKENIZER_PATH": "./data/tokenizers/fineweb_1024_bpe.model",
        "SEED": "42",
        "MAX_WALLCLOCK_SECONDS": "180",  # 3 minutes max
        "ITERATIONS": "500",
        "WARMUP_STEPS": "5",
        "WARMDOWN_ITERS": "200",
        "TRAIN_BATCH_TOKENS": "65536",  # Small batch for 1 GPU
        "TRAIN_SEQ_LEN": "1024",
        "EVAL_SEQ_LEN": "1024",
        "EVAL_STRIDE": "64",
        "VAL_LOSS_EVERY": "250",
        "VAL_BATCH_SIZE": "65536",
        "TRAIN_LOG_EVERY": "50",
        # Architecture
        "NUM_LAYERS": "11",
        "MODEL_DIM": "512",
        "NUM_HEADS": "8",
        "NUM_KV_HEADS": "4",
        "MLP_MULT": "3",
        "TIE_EMBEDDINGS": "1",
        "XSA_LAST_N": "4",
        "ROPE_DIMS": "16",
        "LN_SCALE": "1",
        "VE_ENABLED": "1",
        "VE_DIM": "128",
        "VE_LAYERS": "9,10",
        "BIGRAM_VOCAB_SIZE": "2048",
        "BIGRAM_DIM": "128",
        "LOGIT_SOFTCAP": "30.0",
        # Optimizer
        "MATRIX_LR": "0.025",
        "SCALAR_LR": "0.025",
        "TIED_EMBED_LR": "0.035",
        "MUON_MOMENTUM": "0.99",
        "MUON_MOMENTUM_WARMUP_START": "0.92",
        "MUON_MOMENTUM_WARMUP_STEPS": "100",
        "MUON_WD": "0.04",
        "ADAM_WD": "0.04",
        "GRAD_CLIP_NORM": "0.3",
        "SWA_ENABLED": "1",
        "SWA_EVERY": "50",
        # QAT
        "QAT_ENABLED": "0",
        "LATE_QAT_THRESHOLD": "0.15",
        # TTT off for local test (too slow)
        "TTT_ENABLED": "0",
        "RUN_ID": name,
    })
    if extra_env:
        env.update(extra_env)

    print(f"\n{'='*60}")
    print(f"Running: {name}")
    print(f"Script: {script}")
    print(f"{'='*60}\n")

    proc = subprocess.run(
        [sys.executable, script],
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )

    output = proc.stdout + proc.stderr
    print(output[-3000:] if len(output) > 3000 else output)

    # Extract metrics
    metrics = {}
    for line in output.split('\n'):
        if 'late_qat:enabled' in line:
            metrics['qat_activated'] = True
        m = re.search(r'final_int6_roundtrip.*val_bpb:([0-9.]+)', line)
        if m:
            metrics['roundtrip_bpb'] = float(m.group(1))
        m = re.search(r'final_int6_sliding_window_exact.*val_bpb:([0-9.]+)', line)
        if m:
            metrics['sliding_bpb'] = float(m.group(1))
        m = re.search(r'temperature:applied.*T=([0-9.]+)', line)
        if m:
            metrics['temperature'] = float(m.group(1))
        m = re.search(r'step:(\d+)/\d+ train_loss:([0-9.]+).*step_avg:([0-9.]+)ms', line)
        if m:
            metrics['final_step'] = int(m.group(1))
            metrics['final_loss'] = float(m.group(2))
            metrics['step_avg_ms'] = float(m.group(3))

    if proc.returncode != 0:
        metrics['error'] = True
        print(f"\nERROR: {name} failed with return code {proc.returncode}")

    return metrics


def main():
    # Check we can import the patched module
    print("Checking flash_attn availability...")
    try:
        from flash_attn_interface import flash_attn_func
        print("  FlashAttention 3 available (H100)")
        has_fa3 = True
    except ImportError:
        print("  FlashAttention 3 NOT available (expected on 4090)")
        print("  Patching scripts to use F.scaled_dot_product_attention...")
        has_fa3 = False

    if not has_fa3:
        # Create a shim module so the import doesn't fail
        shim_dir = os.path.join(os.path.dirname(__file__), "flash_attn_interface")
        os.makedirs(shim_dir, exist_ok=True)
        with open(os.path.join(shim_dir, "__init__.py"), "w") as f:
            f.write("""
import torch
import torch.nn.functional as F

def flash_attn_func(q, k, v, causal=False):
    # q: [B, T, H, D], k: [B, T, Hkv, D], v: [B, T, Hkv, D]
    B, T, H, D = q.shape
    Hkv = k.shape[2]
    group = H // Hkv

    # Expand KV heads for GQA
    if group > 1:
        k = k.unsqueeze(3).expand(B, T, Hkv, group, D).reshape(B, T, H, D)
        v = v.unsqueeze(3).expand(B, T, Hkv, group, D).reshape(B, T, H, D)

    # [B, T, H, D] -> [B, H, T, D] for SDPA
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    out = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
    return out.transpose(1, 2)  # back to [B, T, H, D]
""")
        print("  Created flash_attn_interface shim")

    # --- Run original SOTA (broken QAT) ---
    original = run_experiment(
        "records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py",
        "original_sota",
    )

    # --- Run safe version (fixed QAT + temperature) ---
    safe = run_experiment(
        "train_gpt_safe.py",
        "safe_fixed_qat",
        extra_env={"EVAL_TEMPERATURE": "0.90"},
    )

    # --- Results ---
    print("\n" + "="*60)
    print("RESULTS COMPARISON")
    print("="*60)

    for name, m in [("Original SOTA", original), ("Safe (QAT fix + T=0.90)", safe)]:
        print(f"\n{name}:")
        print(f"  QAT activated:  {m.get('qat_activated', False)}")
        print(f"  Temperature:    {m.get('temperature', 'N/A')}")
        print(f"  Final step:     {m.get('final_step', 'N/A')}")
        print(f"  Final loss:     {m.get('final_loss', 'N/A')}")
        print(f"  Step avg (ms):  {m.get('step_avg_ms', 'N/A')}")
        print(f"  Roundtrip BPB:  {m.get('roundtrip_bpb', 'N/A')}")
        print(f"  Sliding BPB:    {m.get('sliding_bpb', 'N/A')}")
        if m.get('error'):
            print(f"  *** ERROR ***")

    # Key comparison
    if 'sliding_bpb' in original and 'sliding_bpb' in safe:
        delta = safe['sliding_bpb'] - original['sliding_bpb']
        print(f"\nDelta (safe - original): {delta:+.4f} BPB")
        if delta < 0:
            print("  -> Safe version is BETTER (lower BPB)")
        elif delta > 0:
            print("  -> Safe version is WORSE (may need tuning)")
        else:
            print("  -> No difference (QAT may not have activated)")

    # QAT activation check
    if safe.get('qat_activated') and not original.get('qat_activated'):
        print("\nQAT FIX VALIDATED: Safe version activated QAT, original did not!")
    elif safe.get('qat_activated') and original.get('qat_activated'):
        print("\nBoth activated QAT (original may work on non-compiled path)")
    elif not safe.get('qat_activated'):
        print("\nWARNING: QAT did not activate in safe version — warmdown too short?")
        print("  (500 iters with threshold=0.15 may not reach the activation point)")
        print("  Try increasing ITERATIONS or lowering LATE_QAT_THRESHOLD")


if __name__ == "__main__":
    main()
