"""Fused NEON kernels for the element-wise ops of a training step: GELU (erf and tanh), SiLU and RMSNorm,
forward and backward, fp32 or bf16, run on the library's CPU thread pool.

PyTorch's aarch64 builds run these through generic kernels: on three A76 cores GELU forward + backward of a
2048 x 1536 activation takes 56 ms and RMSNorm (four separate ops) 29 ms, a large share of a transformer
training step once the matmuls are on the NPU. Under npu_mode() F.gelu, F.silu and F.rms_norm (and so nn.GELU,
nn.SiLU, nn.RMSNorm) use these automatically; convert() also swaps Hugging Face-style *RMSNorm modules.
RKNPU_ELTWISE=0 turns them off. Each function returns None when it doesn't apply (other devices / dtypes,
small tensors), and the caller falls back to PyTorch.
"""
import os

import torch

ENABLED = os.environ.get("RKNPU_ELTWISE", "1") != "0"
MIN_ELEMS = int(os.environ.get("RKNPU_ELTWISE_MIN", "32768"))
GELU_FWD, GELU_BWD, GELU_TANH_FWD, GELU_TANH_BWD, SILU_FWD, SILU_BWD, RMS_FWD, RMS_BWD = range(8)

_npu = None


def _lib():
    global _npu
    if _npu is None:
        from ._lib import default_npu
        _npu = default_npu()
    return _npu


def _ok(*ts):
    return ENABLED and all(t is None or (t.device.type == "cpu" and t.dtype in (torch.float32, torch.bfloat16)
                                         and t.dtype == ts[0].dtype) for t in ts)


def _run(op, x, g=None):
    x = x.contiguous()
    g = g.contiguous() if g is not None else None
    y = torch.empty_like(x)
    _lib().eltwise(op, x.dtype == torch.bfloat16, x.data_ptr(), g.data_ptr() if g is not None else 0, 0,
                   y.data_ptr(), 0, 0, 1, x.numel())
    return y


class _Pointwise(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, fwd, bwd):
        ctx.save_for_backward(x)
        ctx.bwd = bwd
        return _run(fwd, x)

    @staticmethod
    def backward(ctx, gy):
        (x,) = ctx.saved_tensors
        if gy.dtype != x.dtype:
            gy = gy.to(x.dtype)
        return _run(ctx.bwd, x, gy), None, None


def gelu(x, approximate="none"):
    if not _ok(x) or x.numel() < MIN_ELEMS or approximate not in ("none", "tanh"):
        return None
    if approximate == "tanh":
        return _Pointwise.apply(x, GELU_TANH_FWD, GELU_TANH_BWD)
    return _Pointwise.apply(x, GELU_FWD, GELU_BWD)


def silu(x):
    if not _ok(x) or x.numel() < MIN_ELEMS:
        return None
    return _Pointwise.apply(x, SILU_FWD, SILU_BWD)


class _RMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, eps):
        D = x.shape[-1]
        xc = x.contiguous()
        wc = w.contiguous() if w is not None else None
        rows = xc.numel() // D
        y = torch.empty_like(xc)
        rstd = torch.empty(rows, dtype=torch.float32)
        _lib().eltwise(RMS_FWD, xc.dtype == torch.bfloat16, xc.data_ptr(), 0, wc.data_ptr() if wc is not None else 0,
                       y.data_ptr(), rstd.data_ptr(), 0, rows, D, eps)
        ctx.save_for_backward(xc, wc, rstd) if wc is not None else ctx.save_for_backward(xc, rstd)
        ctx.has_w = wc is not None
        return y

    @staticmethod
    def backward(ctx, gy):
        if ctx.has_w:
            x, w, rstd = ctx.saved_tensors
        else:
            (x, rstd), w = ctx.saved_tensors, None
        gy = gy.contiguous().to(x.dtype)
        D = x.shape[-1]
        rows = x.numel() // D
        gx = torch.empty_like(x)
        need_w = w is not None and ctx.needs_input_grad[1]
        dw = torch.empty(D, dtype=torch.float32) if need_w else None
        _lib().eltwise(RMS_BWD, x.dtype == torch.bfloat16, x.data_ptr(), gy.data_ptr(),
                       w.data_ptr() if w is not None else 0, gx.data_ptr(), rstd.data_ptr(),
                       dw.data_ptr() if dw is not None else 0, rows, D, 0.0)
        return gx, (dw.to(w.dtype) if dw is not None else None), None


def rms_norm(x, normalized_shape, weight=None, eps=None):
    ns = (normalized_shape,) if isinstance(normalized_shape, int) else tuple(normalized_shape)
    if len(ns) != 1 or x.shape[-1] != ns[0] or not _ok(x, weight) or x.numel() < MIN_ELEMS:
        return None
    if eps is None:
        eps = torch.finfo(x.dtype).eps
    return _RMSNorm.apply(x, weight, float(eps))


class RMSNorm(torch.nn.Module):
    """Drop-in for Hugging Face-style RMSNorm modules (weight * x / rms(x), computed in fp32)."""

    def __init__(self, weight, eps):
        super().__init__()
        self.weight = weight
        self.variance_epsilon = eps

    def forward(self, x):
        y = rms_norm(x, (x.shape[-1],), self.weight if self.weight.dtype == x.dtype else self.weight.to(x.dtype),
                     self.variance_epsilon)
        if y is not None:
            return y
        xf = x.float()
        y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)
        return self.weight * y.to(x.dtype)


_TABLE = {
    "gelu (erf)": lambda v: gelu(v), "gelu (tanh)": lambda v: gelu(v, "tanh"), "silu": silu,
    "rms_norm": lambda v: rms_norm(v, (v.shape[-1],), torch.ones(v.shape[-1], dtype=v.dtype, requires_grad=True)),
    "swiglu x*silu(x)": lambda v: v * silu(v),
}


def supports(name):
    return name in _TABLE


def get(name):
    return _TABLE[name]
