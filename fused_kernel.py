"""Fused MLP kernel: matmul + LeakyReLU(0.5) + square in one GPU pass.
Must be in its own .py file for Triton 3.6 JIT (requires real file path).
From PR #1072 by the parameter-golf community."""
import torch
import torch.nn.functional as F
from torch import Tensor
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

@triton.jit
def _fused_leaky_relu_sq_kernel(a_desc, b_desc, c_desc, aux_desc,
                                 M, N, K,
                                 BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
                                 BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr,
                                 NUM_SMS: tl.constexpr, FORWARD: tl.constexpr):
    dtype = tl.bfloat16
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n
    tile_id_c = start_pid - NUM_SMS
    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, flatten=True):
        pid_m = tile_id // num_pid_n
        pid_n = tile_id % num_pid_n
        offs_am = pid_m * BLOCK_SIZE_M
        offs_bn = pid_n * BLOCK_SIZE_N
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for ki in range(k_tiles):
            offs_k = ki * BLOCK_SIZE_K
            a = a_desc.load([offs_am, offs_k])
            b = b_desc.load([offs_bn, offs_k])
            accumulator = tl.dot(a, b.T, accumulator)
        tile_id_c += NUM_SMS
        pid_m = tile_id // num_pid_n
        pid_n = tile_id % num_pid_n
        offs_am_c = pid_m * BLOCK_SIZE_M
        offs_bn_c = pid_n * BLOCK_SIZE_N
        acc = tl.reshape(accumulator, (BLOCK_SIZE_M, 2, BLOCK_SIZE_N // 2))
        acc = tl.permute(acc, (0, 2, 1))
        acc0, acc1 = tl.split(acc)
        c0 = acc0.to(dtype)
        if not FORWARD:
            c0_pre = aux_desc.load([offs_am_c, offs_bn_c])
            c0 = c0 * tl.where(c0_pre > 0, 2.0 * c0_pre, 0.5 * c0_pre)
        c_desc.store([offs_am_c, offs_bn_c], c0)
        if FORWARD:
            c0_post = tl.where(c0 > 0, c0, 0.5 * c0)
            c0_post = c0_post * c0_post
            aux_desc.store([offs_am_c, offs_bn_c], c0_post)
        c1 = acc1.to(dtype)
        if not FORWARD:
            c1_pre = aux_desc.load([offs_am_c, offs_bn_c + BLOCK_SIZE_N // 2])
            c1 = c1 * tl.where(c1_pre > 0, 2.0 * c1_pre, 0.5 * c1_pre)
        c_desc.store([offs_am_c, offs_bn_c + BLOCK_SIZE_N // 2], c1)
        if FORWARD:
            c1_post = tl.where(c1 > 0, c1, 0.5 * c1)
            c1_post = c1_post * c1_post
            aux_desc.store([offs_am_c, offs_bn_c + BLOCK_SIZE_N // 2], c1_post)

def _fused_leaky_relu_sq(a, b, aux=None):
    M, K = a.shape; N, K2 = b.shape; assert K == K2
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    FORWARD = aux is None
    if FORWARD: aux = torch.empty((M, N), device=a.device, dtype=a.dtype)
    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
    BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K = 128, 256, 64
    a_desc = TensorDescriptor.from_tensor(a, [BLOCK_SIZE_M, BLOCK_SIZE_K])
    b_desc = TensorDescriptor.from_tensor(b, [BLOCK_SIZE_N, BLOCK_SIZE_K])
    c_desc = TensorDescriptor.from_tensor(c, [BLOCK_SIZE_M, BLOCK_SIZE_N // 2])
    aux_desc = TensorDescriptor.from_tensor(aux, [BLOCK_SIZE_M, BLOCK_SIZE_N // 2])
    grid = lambda META: (min(NUM_SMS, triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N)),)
    _fused_leaky_relu_sq_kernel[grid](
        a_desc, b_desc, c_desc, aux_desc, M, N, K,
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=1, NUM_SMS=NUM_SMS, FORWARD=FORWARD,
        num_stages=4 if FORWARD else 3, num_warps=8)
    return (c, aux) if FORWARD else c

class FusedLeakyReLUSqMLP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, up_w, down_w):
        x_flat = x.view(-1, x.shape[-1])
        pre, post = _fused_leaky_relu_sq(x_flat, up_w)
        out = F.linear(post, down_w)
        ctx.save_for_backward(x_flat, up_w, down_w, pre, post)
        return out.view(x.shape)
    @staticmethod
    def backward(ctx, grad_output):
        x_flat, up_w, down_w, pre, post = ctx.saved_tensors
        go = grad_output.view(-1, grad_output.shape[-1])
        dW2 = go.T @ post
        dpre = _fused_leaky_relu_sq(go, down_w.T.contiguous(), aux=pre)
        dW1 = dpre.T @ x_flat
        dx = dpre @ up_w
        return dx.view(grad_output.shape), dW1, dW2
