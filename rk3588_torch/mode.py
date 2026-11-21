"""Transparent routing of PyTorch matmul-type ops to the NPU.

    with rk3588_torch.npu_mode():
        loss = model(x).sum()
        loss.backward()

Inside the context, F.linear, matmul / mm / bmm / addmm / baddbmm, two-operand
torch.einsum, F.scaled_dot_product_attention, F.conv1d / F.conv2d and F.conv_transpose2d
on CPU float tensors run on the NPU (forward and backward); every other op stays on the
CPU. Convolutions use the NPU's native convolution engine (dilation 1; grouped convolutions
when the groups aren't too thin; depthwise ones stay on the CPU). Biases are added in the
NPU library's output stage. If no NPU is available everything simply runs on the CPU.
"""
import os
import threading

import torch
import torch.nn.functional as F
from torch.overrides import TorchFunctionMode

from . import ops

_MM_FUNCS = {torch.mm, torch.Tensor.mm, torch.matmul, torch.Tensor.matmul, torch.Tensor.__matmul__}
_BMM_FUNCS = {torch.bmm, torch.Tensor.bmm}
_ADDMM_FUNCS = {torch.addmm, torch.Tensor.addmm}
_BADDBMM_FUNCS = {torch.baddbmm, torch.Tensor.baddbmm}
_SDPA = getattr(F, "scaled_dot_product_attention", None)
_EINSUM = {torch.einsum, torch.functional.einsum} if hasattr(torch, "functional") else {torch.einsum}


_UPCAST_FUNCS = (_MM_FUNCS | _BMM_FUNCS | _ADDMM_FUNCS | _BADDBMM_FUNCS | _EINSUM | {F.linear}
                 | ({_SDPA} if _SDPA is not None else set()))


def _cpu_has_bf16():
    """True if the CPU has native bf16 arithmetic (aarch64 'bf16' feature flag). The RK3588's A76 cores don't."""
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.lower().startswith(("features", "flags")):
                    return "bf16" in line.split() or "avx512_bf16" in line.split()
    except OSError:
        pass
    return True          # unknown: leave PyTorch alone


# bf16/fp16 matmuls, attention and linear layers that stay on the CPU (below the NPU threshold, or not
# supported) run through slow software paths on a CPU without bf16 arithmetic: a 128-token attention call took
# 0.6 s, a small addmm 90 ms. Computing them in fp32 and casting the result back is many times faster
# and autograd follows the casts. RKNPU_CPU_UPCAST=0 turns it off.
_UPCAST = os.environ.get("RKNPU_CPU_UPCAST", "1") != "0" and not _cpu_has_bf16()
_UPCAST_DEFAULT = _UPCAST
_LOWP = (torch.bfloat16, torch.float16)


def _upcast_call(func, args, kwargs, force=False):
    """Run `func` in fp32 if its floating-point tensors are all bf16 (or all fp16) on the CPU and small enough
    to widen; returns the result in the original dtype, or None when this doesn't apply."""
    if not (_UPCAST or force) or kwargs.get("out") is not None:
        return None
    ts = [a for a in list(args) + list(kwargs.values()) if isinstance(a, torch.Tensor)]
    if any(isinstance(a, (list, tuple)) and any(isinstance(x, torch.Tensor) for x in a) for a in args):
        return None
    fl = {t.dtype for t in ts if t.is_floating_point()}
    if len(fl) != 1 or next(iter(fl)) not in _LOWP or any(t.device.type != "cpu" for t in ts):
        return None
    if ops.CONVERT_MB and sum(t.numel() for t in ts if t.is_floating_point()) * 4 > ops.CONVERT_MB << 20:
        return None
    dt = next(iter(fl))

    def wide(x):
        return x.float() if isinstance(x, torch.Tensor) and x.is_floating_point() else x
    out = func(*[wide(a) for a in args], **{k: wide(v) for k, v in kwargs.items()})
    return out.to(dt) if isinstance(out, torch.Tensor) and out.is_floating_point() else out


_BACKWARD_FUNCS = (torch.Tensor.backward, torch.autograd.backward)
_bw = threading.local()
_orig_engine_run = None


def _install_engine_wrapper():
    """Wrap torch.autograd's engine call once so it can run with an npu_mode() back on the mode stack."""
    global _orig_engine_run
    import torch.autograd as ag
    if _orig_engine_run is not None or not hasattr(ag, "_engine_run_backward"):
        return _orig_engine_run is not None
    _orig_engine_run = ag._engine_run_backward

    def wrapped(*a, **k):
        m = getattr(_bw, "mode", None)
        if m is None:
            return _orig_engine_run(*a, **k)
        TorchFunctionMode.__enter__(m)
        try:
            return _orig_engine_run(*a, **k)
        finally:
            TorchFunctionMode.__exit__(m, None, None, None)
    ag._engine_run_backward = wrapped
    return True


def _run_backward_in_mode(mode, func, args, kwargs):
    """loss.backward() is itself a torch function, so PyTorch takes the mode off the stack while handling
    it. Everything backward() re-runs (gradient checkpointing's recompute) would then miss the NPU routing
    the forward pass had, save different tensors, and fail the checkpoint's metadata check. Run the
    backward pass with the mode back on the stack. (Only if this PyTorch has the engine hook we wrap;
    otherwise backward runs as before and rk3588_torch.checkpoint covers checkpointing.)"""
    if not _install_engine_wrapper():
        return func(*args, **kwargs)
    prev = getattr(_bw, "mode", None)
    _bw.mode = mode
    try:
        return func(*args, **kwargs)
    finally:
        _bw.mode = prev


def _tensors_ok(*ts):
    return all(isinstance(t, torch.Tensor) for t in ts) and ops._usable(*ts)


def _big_enough(macs):
    return macs >= ops.threshold()


class NPUMode(TorchFunctionMode):
    """See module docstring. Options: linear, matmul, attention, conv (bool), precision (str)."""

    def __init__(self, linear=True, matmul=True, attention=True, conv=True, precision=None, calibrate=True):
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
        if func in _BACKWARD_FUNCS:
            return _run_backward_in_mode(self, func, args, kwargs)
        if ops._enabled:
            out = self._route(func, args, kwargs)
            if out is not None:
                return out
        if func in _UPCAST_FUNCS:
            out = _upcast_call(func, args, kwargs)
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
        if self.matmul and func in _ADDMM_FUNCS and len(args) == 3 and kwargs.get("out") is None:
            inp, a, b = args
            beta, alpha = kwargs.get("beta", 1), kwargs.get("alpha", 1)
            if (_tensors_ok(inp, a, b) and a.dim() == 2 and b.dim() == 2
                    and _big_enough(a.shape[0] * b.shape[1] * a.shape[1])):
                y = ops.matmul2d(a, b)
                if alpha != 1:
                    y = y * alpha
                return y if beta == 0 else y + (inp if beta == 1 else inp * beta)
            return None
        if self.matmul and func in _BADDBMM_FUNCS and len(args) == 3 and kwargs.get("out") is None:
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
        if self.conv and func in (F.conv2d, F.conv1d):
            x, w = args[0], args[1]
            names = ["bias", "stride", "padding", "dilation", "groups"]
            params = dict(zip(names, args[2:]))
            params.update(kwargs)
            b = params.get("bias")
            if _tensors_ok(x, w) and (b is None or _tensors_ok(b)):
                fn = ops.conv2d if func is F.conv2d else ops.conv1d
                return fn(x, w, b, params.get("stride", 1), params.get("padding", 0),
                          params.get("dilation", 1), params.get("groups", 1))
            return None
        if self.conv and func is F.conv_transpose2d:
            x, w = args[0], args[1]
            names = ["bias", "stride", "padding", "output_padding", "groups", "dilation"]
            params = dict(zip(names, args[2:]))
            params.update(kwargs)
            b = params.get("bias")
            if _tensors_ok(x, w) and (b is None or _tensors_ok(b)):
                return ops.conv_transpose2d(x, w, b, params.get("stride", 1), params.get("padding", 0),
                                            params.get("output_padding", 0), params.get("groups", 1),
                                            params.get("dilation", 1))
            return None
        if self.matmul and func in _EINSUM and not kwargs:
            eq, ops_ = (args[0], args[1:]) if args and isinstance(args[0], str) else (None, ())
            if len(ops_) == 1 and isinstance(ops_[0], (list, tuple)):
                ops_ = tuple(ops_[0])
            if eq is not None and len(ops_) == 2 and _tensors_ok(*ops_):
                if _big_enough(ops.einsum_macs(eq, *ops_)):
                    return ops.einsum(eq, *ops_)
            return None
        return None


def npu_mode(**kw):
    """Context manager that routes matmul-type ops to the NPU (see NPUMode)."""
    return NPUMode(**kw)


def _modes_on_stack():
    try:
        return list(torch.overrides._get_current_function_mode_stack())
    except AttributeError:        # very old PyTorch: no way to look
        return []


def checkpoint(function, *args, use_reentrant=False, **kwargs):
    """torch.utils.checkpoint.checkpoint that keeps working with the NPU.

    Gradient checkpointing re-runs `function` inside backward(). That re-run must take the same path as
    the forward pass, or non-reentrant checkpointing stops with "recomputed tensors have different
    metadata" (the NPU path saves the weight untransposed, the plain PyTorch path saves it transposed).
    npu_mode() itself now stays active through a backward() called inside its with-block; this wrapper
    additionally covers a backward() called outside the block (a Trainer loop, an accelerator): it
    remembers the active npu_mode() settings, including the precision, and re-enters them for the
    re-run. (Alternatively call rk3588_torch.enable() once and use any checkpoint function.) Without
    an active npu_mode() it is just torch.utils.checkpoint.checkpoint.

    It defaults to use_reentrant=False. The reentrant variant (use_reentrant=True) gives no gradients
    unless at least one input tensor requires grad, which is awkward when the first layer's input is
    an embedding lookup or a batch of token ids.
    """
    from torch.utils.checkpoint import checkpoint as _ckpt
    cur = next((m for m in reversed(_modes_on_stack()) if isinstance(m, NPUMode)), None)
    if cur is None:
        return _ckpt(function, *args, use_reentrant=use_reentrant, **kwargs)
    cfg = dict(linear=cur.linear, matmul=cur.matmul, attention=cur.attention, conv=cur.conv,
               precision=ops.get_precision(), calibrate=False)

    def run(*a, **k):
        if any(isinstance(m, NPUMode) for m in _modes_on_stack()):     # same routing as the forward pass
            return function(*a, **k)
        with NPUMode(**cfg):         # backward() outside the block: recompute the way the forward ran
            return function(*a, **k)
    return _ckpt(run, *args, use_reentrant=use_reentrant, **kwargs)


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
    """nn.Linear whose matmuls run on the NPU (works without npu_mode, and with torch.compile).
    `fused_relu` (set by convert() for Linear -> ReLU in a Sequential) applies the ReLU in
    the NPU output stage."""
    fused_relu = False

    def forward(self, x):
        act = "relu" if self.fused_relu else None
        if (ops._enabled and ops._usable(x, self.weight) and (self.bias is None or ops._usable(self.bias))
                and x.numel() // max(x.shape[-1], 1) * self.weight.shape[0] * self.weight.shape[1] >= ops.threshold()):
            return ops.linear(x, self.weight, self.bias, act)
        y = _upcast_call(F.linear, (x, self.weight, self.bias), {})
        if y is None:
            y = F.linear(x, self.weight, self.bias)
        return F.relu(y) if self.fused_relu else y


class NPUConv2d(torch.nn.Conv2d):
    """nn.Conv2d whose convolution runs on the NPU (works without npu_mode, and with torch.compile)."""
    fused_relu = False

    def forward(self, x):
        if (self.padding_mode == "zeros" and isinstance(self.padding, tuple) and ops._enabled
                and ops._usable(x, self.weight) and x.dim() == 4
                and ops.conv_supported(x, self.weight, self.stride, self.padding, self.dilation, self.groups)):
            y = ops.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups,
                           "relu" if self.fused_relu else None)
            if y is not None:
                return y
        y = super().forward(x)
        return F.relu(y) if self.fused_relu else y


class NPUConv1d(torch.nn.Conv1d):
    """nn.Conv1d on the NPU (as a height-1 2-D convolution)."""
    fused_relu = False

    def forward(self, x):
        if (self.padding_mode == "zeros" and isinstance(self.padding, tuple) and ops._enabled
                and ops._usable(x, self.weight)):
            y = ops.conv1d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups,
                           "relu" if self.fused_relu else None)
            if y is not None:
                return y
        y = super().forward(x)
        return F.relu(y) if self.fused_relu else y


class NPUConvTranspose2d(torch.nn.ConvTranspose2d):
    """nn.ConvTranspose2d on the NPU (one GEMM, plus a CPU scatter-add when kernel taps overlap)."""

    def forward(self, x, output_size=None):
        if self.padding_mode == "zeros" and ops._enabled and ops._usable(x, self.weight):
            op = self._output_padding(x, output_size, self.stride, self.padding, self.kernel_size, 2, self.dilation)
            y = ops.conv_transpose2d(x, self.weight, self.bias, self.stride, self.padding, op, self.groups,
                                     self.dilation)
            if y is not None:
                return y
        return super().forward(x, output_size)


class _FusedReLU(torch.nn.Module):
    """Placeholder for a ReLU that convert() folded into the preceding NPU layer."""

    def forward(self, x):
        return x

    def extra_repr(self):
        return "fused into the previous layer"


_SWAP = {torch.nn.Linear: NPULinear, torch.nn.Conv2d: NPUConv2d, torch.nn.Conv1d: NPUConv1d,
         torch.nn.ConvTranspose2d: NPUConvTranspose2d}
_FUSABLE = (NPULinear, NPUConv2d, NPUConv1d)


def convert(model, fuse_relu=True):
    """Swap every nn.Linear / nn.Conv1d / nn.Conv2d / nn.ConvTranspose2d in `model` for its NPU
    version, in place (weights are shared, state_dict keys unchanged). With fuse_relu, a
    Linear/Conv followed by nn.ReLU inside an nn.Sequential runs the ReLU in the NPU output
    stage (the ReLU module becomes a pass-through). Returns model."""
    if type(model) in _SWAP:
        model.__class__ = _SWAP[type(model)]
        return model
    for _, child in list(model.named_children()):
        convert(child, fuse_relu)
    if fuse_relu and isinstance(model, torch.nn.Sequential):
        mods = list(model._modules.items())
        for (_, a), (name, b) in zip(mods, mods[1:]):
            if isinstance(a, _FUSABLE) and type(b) is torch.nn.ReLU and not a.fused_relu \
                    and not a._forward_hooks and not b._forward_hooks and not b._forward_pre_hooks:
                a.fused_relu = True
                model._modules[name] = _FusedReLU()
    return model
