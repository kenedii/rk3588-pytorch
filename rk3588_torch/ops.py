"""NPU-backed matmul primitives with autograd support.

Every GEMM in forward and backward (y = x W^T, dX = dY W, dW = dY^T X) goes through
the NPU. Operand transposes are passed as strides, never materialized.
"""
import os

import torch

from ._lib import default_npu

# Below this many multiply-adds the CPU wins (submit + packing overhead dominates).
# Overridable with RKNPU_MIN_MACS; tuned from bench/bench_gemm.py on a Rock 5B.
MIN_MACS = int(os.environ.get("RKNPU_MIN_MACS", str(8_000_000)))

# Precision used for NPU GEMMs: "fp16" (default), "fp16x3" (near-fp32 accuracy, ~3x NPU work),
# "int8" (fastest MACs, quantized: approximate), "fp16_out16" (fp16 results), "bf16" (if supported).
_precision = os.environ.get("RKNPU_PRECISION", "fp16")


def set_precision(name):
    """Select the NPU GEMM precision for subsequent calls. Returns the previous one."""
    global _precision
    from ._lib import PRECISIONS
    if name not in PRECISIONS:
        raise ValueError(f"unknown precision {name!r}; choose from {sorted(PRECISIONS)}")
    prev, _precision = _precision, name
    return prev


def get_precision():
    return _precision


_stats = {"npu_calls": 0, "cpu_calls": 0, "npu_macs": 0, "cpu_macs": 0}


def stats(reset=False):
    out = dict(_stats)
    if reset:
        for k in _stats:
            _stats[k] = 0
    return out


def _as_f32(t):
    return t if t.dtype == torch.float32 else t.float()


def _usable(*ts):
    for t in ts:
        if t.device.type != "cpu" or t.dtype not in (torch.float32, torch.float16, torch.bfloat16):
            return False
        if any(s < 0 for s in t.stride()):
            return False
    return True


def mm_raw(a, b, out=None, accumulate=False, precision=None):
    """C = a @ b for 2-D CPU tensors, computed on the NPU (no autograd)."""
    precision = precision or _precision
    assert a.dim() == 2 and b.dim() == 2 and a.shape[1] == b.shape[0], (a.shape, b.shape)
    M, K = a.shape
    N = b.shape[1]
    if M * N * K < MIN_MACS:
        _stats["cpu_calls"] += 1
        _stats["cpu_macs"] += M * N * K
        r = _as_f32(a) @ _as_f32(b)
        if out is None:
            return r
        if accumulate:
            out += r
        else:
            out.copy_(r)
        return out
    a = _as_f32(a)
    b = _as_f32(b)
    if out is None:
        out = torch.empty((M, N), dtype=torch.float32)
        accumulate = False
    assert out.is_contiguous() and out.dtype == torch.float32
    if a.stride(0) == 0 or b.stride(1) == 0:
        a, b = a.contiguous(), b.contiguous()
    npu = default_npu()
    npu.gemm_ptr(M, N, K, a.data_ptr(), a.stride(0), a.stride(1),
                 b.data_ptr(), b.stride(0), b.stride(1), out.data_ptr(), out.stride(0), accumulate,
                 precision=precision)
    _stats["npu_calls"] += 1
    _stats["npu_macs"] += M * N * K
    return out


class NPUMatmul(torch.autograd.Function):
    """2-D a @ b with NPU forward and backward."""

    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return mm_raw(a.detach(), b.detach()).to(torch.promote_types(a.dtype, b.dtype))

    @staticmethod
    def backward(ctx, g):
        a, b = ctx.saved_tensors
        ga = gb = None
        g = g.detach()
        if ctx.needs_input_grad[0]:
            ga = mm_raw(g, b.detach().t()).to(a.dtype)       # dA = dC @ B^T
        if ctx.needs_input_grad[1]:
            gb = mm_raw(a.detach().t(), g).to(b.dtype)       # dB = A^T @ dC
        return ga, gb


class NPULinearFn(torch.autograd.Function):
    """y = x @ W^T + bias, x is (M, K), W is (N, K)."""

    @staticmethod
    def forward(ctx, x, weight, bias):
        ctx.save_for_backward(x, weight)
        ctx.has_bias = bias is not None
        y = mm_raw(x.detach(), weight.detach().t())
        if bias is not None:
            y += bias.detach().float()
        return y.to(x.dtype)

    @staticmethod
    def backward(ctx, g):
        x, w = ctx.saved_tensors
        g = g.detach().contiguous()
        gx = gw = gb = None
        if ctx.needs_input_grad[0]:
            gx = mm_raw(g, w.detach()).to(x.dtype)            # dX = dY @ W
        if ctx.needs_input_grad[1]:
            gw = mm_raw(g.t(), x.detach()).to(w.dtype)        # dW = dY^T @ X
        if ctx.has_bias and ctx.needs_input_grad[2]:
            gb = g.sum(0).to(w.dtype)
        return gx, gw, gb


def matmul2d(a, b):
    return NPUMatmul.apply(a, b)


def linear(x, weight, bias=None):
    shape = x.shape
    x2 = x.reshape(-1, shape[-1])
    y = NPULinearFn.apply(x2, weight, bias)
    return y.reshape(*shape[:-1], weight.shape[0])


def bmm(a, b):
    # one GEMM per batch element; the NPU context serializes them anyway
    return torch.stack([NPUMatmul.apply(a[i], b[i]) for i in range(a.shape[0])])


def matmul(a, b):
    """torch.matmul semantics for the common cases; returns None if unsupported."""
    if a.dim() == 2 and b.dim() == 2:
        return NPUMatmul.apply(a, b)
    if a.dim() >= 3 and b.dim() == 2:
        lead = a.shape[:-1]
        return NPUMatmul.apply(a.reshape(-1, a.shape[-1]), b).reshape(*lead, b.shape[-1])
    if a.dim() >= 3 and b.dim() >= 3 and a.shape[:-2] == b.shape[:-2]:
        lead = a.shape[:-2]
        a3 = a.reshape(-1, *a.shape[-2:])
        b3 = b.reshape(-1, *b.shape[-2:])
        return bmm(a3, b3).reshape(*lead, a.shape[-2], b.shape[-1])
    return None


def conv2d(x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    """conv2d lowered to im2col + NPU GEMM. Returns None if unsupported."""
    if groups != 1 or isinstance(padding, str) or x.dim() != 4:
        return None
    B, C, H, W = x.shape
    O, _, kh, kw = weight.shape
    st = (stride, stride) if isinstance(stride, int) else tuple(stride)
    pd = (padding, padding) if isinstance(padding, int) else tuple(padding)
    dl = (dilation, dilation) if isinstance(dilation, int) else tuple(dilation)
    oh = (H + 2 * pd[0] - dl[0] * (kh - 1) - 1) // st[0] + 1
    ow = (W + 2 * pd[1] - dl[1] * (kw - 1) - 1) // st[1] + 1
    cols = torch.nn.functional.unfold(x, (kh, kw), dilation=dl, padding=pd, stride=st)  # B, C*kh*kw, L
    cols = cols.transpose(1, 2).reshape(B * oh * ow, C * kh * kw)
    y = linear(cols, weight.reshape(O, -1), bias)                                     # B*L, O
    return y.reshape(B, oh * ow, O).transpose(1, 2).reshape(B, O, oh, ow)
