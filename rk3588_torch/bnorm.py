"""Training-mode BatchNorm (NCHW fp32) on the library's CPU kernels.

PyTorch's aarch64 BatchNorm runs through generic kernels; on MobileNet / EfficientNet-style models the forward and
backward of the BatchNorm layers were 20-25% of a training step once the convolutions ran on the NPU. rk_batchnorm
computes each channel's statistics and normalizes it in one pass over data that is still in the cache.
Under npu_mode() F.batch_norm (so nn.BatchNorm1d/2d/3d) uses it in training mode; convert() swaps the modules so
it also works without npu_mode(). RKNPU_BNORM=0 turns it off. Every function returns None when it doesn't apply
(evaluation mode, other dtypes / layouts, small tensors) and the caller falls back to PyTorch.
"""
import os

import torch

ENABLED = os.environ.get("RKNPU_BNORM", "1") != "0"
MIN_ELEMS = int(os.environ.get("RKNPU_BNORM_MIN", "32768"))
MIN_HW = 16
_npu = None


def _lib():
    global _npu
    if _npu is None:
        from ._lib import default_npu
        _npu = default_npu()
    return _npu


def _p(t):
    return t.data_ptr() if t is not None else 0


class _BN(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, gamma, beta, eps):
        N, C = x.shape[:2]
        HW = x.numel() // (N * C)
        xc = x.contiguous()
        gc = gamma.contiguous() if gamma is not None else None
        bc = beta.contiguous() if beta is not None else None
        y = torch.empty_like(xc)
        mean, rstd, var = (torch.empty(C, dtype=torch.float32) for _ in range(3))
        _lib().batchnorm(0, xc.data_ptr(), 0, _p(gc), _p(bc), 0, 0, y.data_ptr(), mean.data_ptr(), rstd.data_ptr(),
                         var.data_ptr(), 0, 0, N, C, HW, eps)
        ctx.save_for_backward(xc, gc, mean, rstd)
        ctx.has_beta = beta is not None
        ctx.dims = (N, C, HW)
        ctx.mark_non_differentiable(mean, var)
        return y, mean, var

    @staticmethod
    def backward(ctx, gy, _gm, _gv):
        x, gamma, mean, rstd = ctx.saved_tensors
        N, C, HW = ctx.dims
        g = gy.contiguous()
        if g.dtype != torch.float32:
            g = g.float()
        need_x, need_g, need_b = ctx.needs_input_grad[0], ctx.needs_input_grad[1], ctx.needs_input_grad[2]
        dx = torch.empty_like(x) if need_x else None
        dgamma = torch.empty(C, dtype=torch.float32) if (need_g and gamma is not None) else None
        dbeta = torch.empty(C, dtype=torch.float32) if (need_b and ctx.has_beta) else None
        _lib().batchnorm(1, x.data_ptr(), g.data_ptr(), _p(gamma), 0, mean.data_ptr(), rstd.data_ptr(), _p(dx), 0, 0, 0,
                         _p(dgamma), _p(dbeta), N, C, HW, 0.0)
        return dx, dgamma, dbeta, None


def batch_norm(x, running_mean, running_var, weight, bias, training, momentum, eps):
    """F.batch_norm in training mode through rk_batchnorm; None when it doesn't apply."""
    if not (ENABLED and training) or x.dim() < 3 or x.device.type != "cpu" or x.dtype != torch.float32:
        return None
    if x.numel() < MIN_ELEMS or x.numel() // (x.shape[0] * x.shape[1]) < MIN_HW or not x.is_contiguous():
        return None
    for t in (weight, bias, running_mean, running_var):
        if t is not None and (t.dtype != torch.float32 or t.device.type != "cpu"):
            return None
    y, mean, var = _BN.apply(x, weight, bias, float(eps))
    if running_mean is not None and running_var is not None:
        M = x.numel() // x.shape[1]
        with torch.no_grad():
            m = 0.1 if momentum is None else float(momentum)
            running_mean.mul_(1.0 - m).add_(mean, alpha=m)
            running_var.mul_(1.0 - m).add_(var * (M / max(M - 1, 1)), alpha=m)
    return y
