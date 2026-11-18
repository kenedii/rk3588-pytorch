"""Transparent routing of PyTorch matmul-type ops to the NPU.

    with rk3588_torch.npu_mode():
        loss = model(x).sum()
        loss.backward()

Inside the context, F.linear, matmul / mm / bmm / addmm / baddbmm and
F.scaled_dot_product_attention on CPU float tensors run their GEMMs (forward and
backward) on the NPU; every other op stays on the CPU. conv2d can be routed too with
npu_mode(conv=True). If no NPU is available everything simply runs on the CPU.
"""
import torch
import torch.nn.functional as F
from torch.overrides import TorchFunctionMode

from . import ops

_MM_FUNCS = {torch.mm, torch.Tensor.mm, torch.matmul, torch.Tensor.matmul, torch.Tensor.__matmul__}
_BMM_FUNCS = {torch.bmm, torch.Tensor.bmm}
_ADDMM_FUNCS = {torch.addmm, torch.Tensor.addmm}
_BADDBMM_FUNCS = {torch.baddbmm, torch.Tensor.baddbmm}
_SDPA = getattr(F, "scaled_dot_product_attention", None)


def _tensors_ok(*ts):
    return all(isinstance(t, torch.Tensor) for t in ts) and ops._usable(*ts)


def _big_enough(macs):
    return macs >= ops.threshold()


class NPUMode(TorchFunctionMode):
    """See module docstring. Options: linear, matmul, attention, conv (bool), precision (str)."""

    def __init__(self, linear=True, matmul=True, attention=True, conv=False, precision=None, calibrate=True):
        super().__init__()
        self.precision = precision
        self._prev_precision = None
        self.linear = linear
        self.matmul = matmul
        self.attention = attention
        self.conv = conv
        self.calibrate = calibrate

    def __enter__(self):
        if self.precision is not None:
            self._prev_precision = ops.set_precision(self.precision)
        if self.calibrate and ops.MIN_MACS is None and ops.is_enabled():
            from .calibrate import ensure_calibrated
            ensure_calibrated(ops.get_precision())
        return super().__enter__()

    def __exit__(self, *exc):
        if self._prev_precision is not None:
            ops.set_precision(self._prev_precision)
            self._prev_precision = None
        return super().__exit__(*exc)

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if ops._enabled:
            out = self._route(func, args, kwargs)
            if out is not None:
                return out
        return func(*args, **kwargs)

    def _route(self, func, args, kwargs):
        if self.linear and func is F.linear:
            x, w = args[0], args[1]
            b = args[2] if len(args) > 2 else kwargs.get("bias")
            if _tensors_ok(x, w) and (b is None or _tensors_ok(b)) and x.dim() >= 1 and w.dim() == 2:
                if _big_enough(x.numel() // max(x.shape[-1], 1) * w.shape[0] * w.shape[1]):
                    return ops.linear(x, w, b)
            return None
        if self.matmul and func in _MM_FUNCS and not kwargs and len(args) == 2:
            a, b = args
            if _tensors_ok(a, b) and a.dim() >= 2 and b.dim() >= 2:
                if _big_enough(a.numel() // max(a.shape[-1], 1) * b.shape[-1] * a.shape[-1]):
                    return ops.matmul(a, b)
            return None
        if self.matmul and func in _BMM_FUNCS and not kwargs and len(args) == 2:
            a, b = args
            if _tensors_ok(a, b) and a.dim() == 3 and _big_enough(a.shape[0] * a.shape[1] * b.shape[2] * a.shape[2]):
                return ops.bmm(a, b)
            return None
        if self.matmul and func in _ADDMM_FUNCS and len(args) == 3:
            inp, a, b = args
            beta, alpha = kwargs.get("beta", 1), kwargs.get("alpha", 1)
            if (_tensors_ok(inp, a, b) and a.dim() == 2 and b.dim() == 2
                    and _big_enough(a.shape[0] * b.shape[1] * a.shape[1])):
                y = ops.matmul2d(a, b)
                if alpha != 1:
                    y = y * alpha
                return y if beta == 0 else y + (inp if beta == 1 else inp * beta)
            return None
        if self.matmul and func in _BADDBMM_FUNCS and len(args) == 3:
            inp, a, b = args
            if (_tensors_ok(inp, a, b) and a.dim() == 3
                    and _big_enough(a.shape[0] * a.shape[1] * b.shape[2] * a.shape[2])):
                return ops.baddbmm(inp, a, b, kwargs.get("beta", 1), kwargs.get("alpha", 1))
            return None
        if self.attention and _SDPA is not None and func is _SDPA:
            names = ["attn_mask", "dropout_p", "is_causal", "scale", "enable_gqa"]
            q, k, v = args[:3]
            params = dict(zip(names, args[3:]))
            params.update(kwargs)
            m = params.get("attn_mask")
            if _tensors_ok(q, k, v) and (m is None or isinstance(m, torch.Tensor)) and q.dim() >= 3:
                L, E = q.shape[-2:]
                S = k.shape[-2]
                if _big_enough(q.numel() // max(E, 1) * S * E):
                    return ops.scaled_dot_product_attention(q, k, v, **params)
            return None
        if self.conv and func is F.conv2d:
            x, w = args[0], args[1]
            names = ["bias", "stride", "padding", "dilation", "groups"]
            params = dict(zip(names, args[2:]))
            params.update(kwargs)
            b = params.get("bias")
            if _tensors_ok(x, w) and (b is None or _tensors_ok(b)) and x.dim() == 4:
                return ops.conv2d(x, w, b, params.get("stride", 1), params.get("padding", 0),
                                  params.get("dilation", 1), params.get("groups", 1))
            return None
        return None


def npu_mode(**kw):
    """Context manager that routes matmul-type ops to the NPU (see NPUMode)."""
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


class NPULinear(torch.nn.Linear):
    """nn.Linear whose matmuls run on the NPU (works without npu_mode, and with torch.compile)."""

    def forward(self, x):
        if (ops._enabled and ops._usable(x, self.weight)
                and x.numel() // max(x.shape[-1], 1) * self.weight.shape[0] * self.weight.shape[1] >= ops.threshold()):
            return ops.linear(x, self.weight, self.bias)
        return F.linear(x, self.weight, self.bias)


def convert(model):
    """Swap every nn.Linear in `model` for NPULinear, in place (weights are shared). Returns model."""
    for name, child in list(model.named_children()):
        if type(child) is torch.nn.Linear:
            child.__class__ = NPULinear
        else:
            convert(child)
    if type(model) is torch.nn.Linear:
        model.__class__ = NPULinear
    return model
