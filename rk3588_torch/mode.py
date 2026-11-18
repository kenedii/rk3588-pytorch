"""Transparent routing of PyTorch matmul-type ops to the NPU.

    with rk3588_torch.npu_mode():
        loss = model(x).sum()
        loss.backward()

Inside the context, F.linear / matmul / mm / bmm / addmm / conv2d on CPU float tensors
run their GEMMs (forward and backward) on the NPU; every other op stays on the CPU.
"""
import torch
import torch.nn.functional as F
from torch.overrides import TorchFunctionMode

from . import ops

_MM_FUNCS = {torch.mm, torch.Tensor.mm, torch.matmul, torch.Tensor.matmul, torch.Tensor.__matmul__}
_BMM_FUNCS = {torch.bmm, torch.Tensor.bmm}
_ADDMM_FUNCS = {torch.addmm, torch.Tensor.addmm}


def _tensors_ok(*ts):
    return all(isinstance(t, torch.Tensor) for t in ts) and ops._usable(*ts)


def _big_enough(m, n, k):
    return m * n * k >= ops.MIN_MACS


class NPUMode(TorchFunctionMode):
    def __init__(self, linear=True, matmul=True, conv=True):
        super().__init__()
        self.linear = linear
        self.matmul = matmul
        self.conv = conv

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        try:
            out = self._route(func, args, kwargs)
        except _Fallback:
            out = None
        if out is not None:
            return out
        return func(*args, **kwargs)

    def _route(self, func, args, kwargs):
        if self.linear and func is F.linear:
            x, w = args[0], args[1]
            b = args[2] if len(args) > 2 else kwargs.get("bias")
            if _tensors_ok(x, w) and (b is None or _tensors_ok(b)) and x.dim() >= 1:
                m = x.numel() // x.shape[-1]
                if _big_enough(m, w.shape[0], w.shape[1]):
                    return ops.linear(x, w, b)
            return None
        if self.matmul and func in _MM_FUNCS and not kwargs:
            a, b = args[0], args[1]
            if _tensors_ok(a, b) and a.dim() >= 2 and b.dim() >= 2:
                m = a.numel() // a.shape[-1]
                if _big_enough(m, b.shape[-1], a.shape[-1]):
                    return ops.matmul(a, b)
            return None
        if self.matmul and func in _BMM_FUNCS and not kwargs:
            a, b = args[0], args[1]
            if _tensors_ok(a, b) and _big_enough(a.shape[1], b.shape[2], a.shape[2]):
                return ops.bmm(a, b)
            return None
        if self.matmul and func in _ADDMM_FUNCS:
            inp, a, b = args[0], args[1], args[2]
            beta = kwargs.get("beta", 1)
            alpha = kwargs.get("alpha", 1)
            if (_tensors_ok(inp, a, b) and a.dim() == 2 and b.dim() == 2 and len(args) == 3
                    and _big_enough(a.shape[0], b.shape[1], a.shape[1])):
                y = ops.matmul2d(a, b)
                if alpha != 1:
                    y = y * alpha
                return y + (inp if beta == 1 else inp * beta)
            return None
        if self.conv and func is F.conv2d:
            x, w = args[0], args[1]
            rest = list(args[2:])
            names = ["bias", "stride", "padding", "dilation", "groups"]
            params = dict(zip(names, rest))
            params.update(kwargs)
            b = params.get("bias")
            if _tensors_ok(x, w) and (b is None or _tensors_ok(b)) and x.dim() == 4:
                O, Cg, kh, kw = w.shape
                return ops.conv2d(x, w, b, params.get("stride", 1), params.get("padding", 0),
                                  params.get("dilation", 1), params.get("groups", 1))
            return None
        return None


class _Fallback(Exception):
    pass


def npu_mode(**kw):
    """Context manager that routes matmul-type ops to the NPU."""
    return NPUMode(**kw)


_global_mode = None


def enable(**kw):
    """Route matmul-type ops to the NPU until disable() is called."""
    global _global_mode
    if _global_mode is None:
        _global_mode = NPUMode(**kw)
        _global_mode.__enter__()


def disable():
    global _global_mode
    if _global_mode is not None:
        _global_mode.__exit__(None, None, None)
        _global_mode = None
