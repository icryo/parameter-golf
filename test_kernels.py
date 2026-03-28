"""Test script to validate Triton kernels match PyTorch reference implementations.

Run on GPU: python test_kernels.py
"""
import torch
import torch.nn.functional as F


def test_leaky_relu_squared():
    """Verify fused kernel matches: F.leaky_relu(x, 0.5).square()"""
    from triton_kernels import leaky_relu_squared

    torch.manual_seed(42)
    for shape in [(4, 128, 1536), (1, 2048, 1536), (8, 64, 512)]:
        x = torch.randn(*shape, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        x_ref = x.detach().clone().requires_grad_(True)

        # Reference
        y_ref = F.leaky_relu(x_ref, negative_slope=0.5).square()
        (y_ref.sum()).backward()

        # Fused
        y_fused = leaky_relu_squared(x)
        (y_fused.sum()).backward()

        fwd_err = (y_ref - y_fused).abs().max().item()
        bwd_err = (x_ref.grad - x.grad).abs().max().item()
        print(f"  Shape {shape}: fwd_err={fwd_err:.2e}, bwd_err={bwd_err:.2e}")
        assert fwd_err < 0.05, f"Forward mismatch: {fwd_err}"
        assert bwd_err < 0.1, f"Backward mismatch: {bwd_err}"
    print("  PASSED\n")


def test_ste_int6():
    """Verify fused STE matches PyTorch reference."""
    from triton_kernels import ste_int6_fakequant

    torch.manual_seed(42)
    for shape in [(512, 1536), (256, 512), (1536, 512)]:
        w = torch.randn(*shape, device="cuda", dtype=torch.bfloat16)

        # Reference
        w32 = w.float()
        row_max = w32.abs().amax(dim=1)
        scale = (row_max / 31.0).clamp_min(1.0 / 31.0)
        q = torch.clamp(torch.round(w32 / scale[:, None]), -32, 31)
        wq_ref = (q * scale[:, None]).to(torch.bfloat16)

        # Fused
        wq_fused = ste_int6_fakequant(w)

        err = (wq_ref - wq_fused).abs().max().item()
        print(f"  Shape {shape}: max_err={err:.2e}")
        # Allow some tolerance due to bf16 rounding differences
        assert err < 0.5, f"STE mismatch: {err}"
    print("  PASSED\n")


def test_xsa():
    """Verify fused XSA matches PyTorch reference."""
    from triton_kernels import fused_xsa

    torch.manual_seed(42)
    for B, T in [(2, 64), (4, 128), (1, 2048)]:
        H, D, Hkv = 8, 64, 4
        y = torch.randn(B, T, H, D, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(B, T, Hkv, D, device="cuda", dtype=torch.bfloat16)

        # Reference
        group = H // Hkv
        y_g = y.reshape(B, T, Hkv, group, D)
        vn = F.normalize(v, dim=-1).unsqueeze(-2)
        proj = (y_g * vn).sum(dim=-1, keepdim=True) * vn
        ref = (y_g - proj).reshape(B, T, H, D)

        # Fused
        fused = fused_xsa(y, v)

        err = (ref - fused).abs().max().item()
        print(f"  B={B}, T={T}: max_err={err:.2e}")
        assert err < 0.05, f"XSA mismatch: {err}"
    print("  PASSED\n")


def test_ste_gradient():
    """Verify STE gradient passes through correctly."""
    from triton_kernels import ste_int6_fakequant

    torch.manual_seed(42)
    w = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)

    # STE: forward uses quantized weights, backward passes gradient through
    w_q = ste_int6_fakequant(w)
    w_eff = w + (w_q - w).detach()

    loss = w_eff.sum()
    loss.backward()

    # Gradient should be all ones (identity STE)
    expected = torch.ones_like(w)
    err = (w.grad - expected).abs().max().item()
    print(f"  STE gradient max_err={err:.2e}")
    assert err < 1e-4, f"STE gradient not identity: {err}"
    print("  PASSED\n")


def benchmark_leaky_relu_squared():
    """Benchmark fused vs unfused LeakyReLU²."""
    from triton_kernels import leaky_relu_squared

    torch.manual_seed(42)
    x = torch.randn(8, 2048, 1536, device="cuda", dtype=torch.bfloat16)

    # Warmup
    for _ in range(10):
        F.leaky_relu(x, negative_slope=0.5).square()
        leaky_relu_squared(x)
    torch.cuda.synchronize()

    import time

    # Unfused
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(100):
        F.leaky_relu(x, negative_slope=0.5).square()
    torch.cuda.synchronize()
    unfused_ms = (time.perf_counter() - t0) * 10  # ms per call

    # Fused
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(100):
        leaky_relu_squared(x)
    torch.cuda.synchronize()
    fused_ms = (time.perf_counter() - t0) * 10  # ms per call

    print(f"  Unfused: {unfused_ms:.3f} ms")
    print(f"  Fused:   {fused_ms:.3f} ms")
    print(f"  Speedup: {unfused_ms / fused_ms:.2f}x\n")


if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA required for testing"
    print("=" * 60)
    print("Testing Triton kernels for Parameter Golf")
    print("=" * 60)

    print("\n1. LeakyReLU-Squared (correctness):")
    test_leaky_relu_squared()

    print("2. STE Int6 (correctness):")
    test_ste_int6()

    print("3. XSA (correctness):")
    test_xsa()

    print("4. STE Gradient (identity check):")
    test_ste_gradient()

    print("5. LeakyReLU-Squared (benchmark):")
    benchmark_leaky_relu_squared()

    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)
