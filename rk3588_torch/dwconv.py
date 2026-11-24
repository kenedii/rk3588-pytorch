"""Depthwise convolutions (groups == in_channels == out_channels) on the library's CPU kernels.

The NPU's depthwise mode doesn't work in fp16 yet, so these stay on the CPU, but PyTorch's own CPU depthwise
backward is slow on the RK3588 (295 ms for a 32x256x32x32 3x3 layer against 50 ms forward): MobileNet-style
models spent about half of a training step there. rk_dwconv does the forward, input gradient and weight gradient
as vectorized row loops on the library's thread pool. Used by ops.conv2d (so by npu_mode() and NPUConv2d) for
depthwise convolutions with dilation 1; RKNPU_DWCONV=0 turns it off.
"""
import os

import torch

ENABLED = os.environ.get("RKNPU_DWCONV", "1") != "0"
MIN_ELEMS = int(os.environ.get("RKNPU_DWCONV_MIN", "16384"))
_npu = None


def _lib():
    global _npu
    if _npu is None:
        from ._lib import default_npu
        _npu = default_npu()
    return _npu


def _call(op, x, w, dy, out, geo):
    _lib().dwconv(op, x.data_ptr() if x is not None else 0, w.data_ptr() if w is not None else 0,
                  dy.data_ptr() if dy is not None else 0, out.data_ptr(), *geo)
    return out


class _DW(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, st, pt, pl, OH, OW):
        N, C, H, W = x.shape
        KH, KW = w.shape[2:]
        geo = (N, C, H, W, KH, KW, st[0], st[1], pt, pl, OH, OW)
        x32 = x.detach().float().contiguous()
        w32 = w.detach().float().contiguous()
        y = _call(0, x32, w32, None, torch.empty(N, C, OH, OW), geo)
        ctx.save_for_backward(x32, w32)
        ctx.geo, ctx.dt = geo, (x.dtype, w.dtype)
        return y.to(x.dtype)

    @staticmethod
    def backward(ctx, g):
        x32, w32 = ctx.saved_tensors
        geo = ctx.geo
        g32 = g.float().contiguous()
        dx = dw = None
        if ctx.needs_input_grad[0]:
            dx = _call(1, None, w32, g32, torch.empty_like(x32), geo).to(ctx.dt[0])
        if ctx.needs_input_grad[1]:
            dw = _call(2, x32, None, g32, torch.empty_like(w32), geo).to(ctx.dt[1])
        return dx, dw, None, None, None, None, None


def conv2d(x, w, bias, st, pads, act=None):
    """Depthwise conv with padding (top, bottom, left, right). Returns None when it doesn't apply."""
    if not ENABLED or x.dim() != 4 or w.dim() != 4 or x.device.type != "cpu" or w.device.type != "cpu":
        return None
    N, C, H, W = x.shape
    if w.shape[0] != C or w.shape[1] != 1 or x.numel() < MIN_ELEMS:
        return None
    if x.dtype not in (torch.float32, torch.bfloat16) or w.dtype not in (torch.float32, torch.bfloat16):
        return None
    t, b, l, r = pads
    KH, KW = w.shape[2:]
    OH = (H + t + b - KH) // st[0] + 1
    OW = (W + l + r - KW) // st[1] + 1
    if OH < 1 or OW < 1 or min(pads) < 0:
        return None
    with torch._C.DisableTorchFunction():
        y = _DW.apply(x, w, tuple(st), t, l, OH, OW)
        if bias is not None:
            y = y + bias.view(1, C, 1, 1).to(y.dtype)
        if act == "relu":
            y = torch.relu(y)
    return y
