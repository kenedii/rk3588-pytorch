"""NPU-backed matmul primitives.

All NPU work goes through one custom op, ``torch.ops.rknpu.mm`` (2-D or batched 3-D
a @ b), registered with ``torch.library`` so it works with autograd and torch.compile.
Its backward is two more rknpu.mm calls (dA = dC @ B^T, dB = A^T @ dC); transposes are
passed to the NPU as strides, never copied.

Every call is guarded so that training can't be broken by the NPU:
  * small GEMMs (below a calibrated size) run on the CPU, where they are faster;
  * each new (shape, layout, precision) is checked against a CPU reference on a few
    rows the first time it is seen; a mismatch pins that shape to the CPU;
  * non-finite NPU output with finite inputs is recomputed on the CPU;
  * NPU errors fall back to the CPU, and after repeated errors the NPU is disabled.
"""
import math
import os
import threading
import time
import warnings

import torch

from ._lib import PRECISIONS, default_npu, npu_available

# ---------------------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------------------
# Below this many multiply-adds the CPU wins. None = use the calibrated value (see
# calibrate.py) or 8e6; an integer (or RKNPU_MIN_MACS) forces a fixed threshold.
MIN_MACS = int(os.environ["RKNPU_MIN_MACS"]) if os.environ.get("RKNPU_MIN_MACS") else None
DEFAULT_MIN_MACS = 8_000_000

# "first" (default): verify each new shape once; "always": every call; "off".
VERIFY = os.environ.get("RKNPU_VERIFY", "first")
# relative error allowed when checking an NPU result against the CPU, per precision
VERIFY_TOL = {"fp16": 5e-3, "fp16_out16": 5e-3, "bf16": 3e-2, "int8": 8e-2, "fp16x3": 1e-4}
MAX_FAILURES = int(os.environ.get("RKNPU_MAX_FAILURES", "3"))

_precision = os.environ.get("RKNPU_PRECISION", "fp16")
if _precision not in PRECISIONS:
    raise ValueError(f"RKNPU_PRECISION={_precision!r}; choose from {sorted(PRECISIONS)}")

_enabled = os.environ.get("RKNPU_DISABLE", "0") not in ("1", "true", "yes")
_verified = {}           # shape key -> True (ok) / False (pinned to CPU)
_failures = 0
_warned = set()
_lock = threading.Lock()

_stats = {"npu_calls": 0, "cpu_calls": 0, "npu_macs": 0, "cpu_macs": 0, "npu_time": 0.0, "cpu_time": 0.0,
          "fallbacks": 0, "verified_shapes": 0, "rejected_shapes": 0}


def set_precision(name):
    """Select the NPU GEMM precision for subsequent calls. Returns the previous one."""
    global _precision
    if name not in PRECISIONS:
        raise ValueError(f"unknown precision {name!r}; choose from {sorted(PRECISIONS)}")
    prev, _precision = _precision, name
    return prev


def get_precision():
    return _precision


def set_enabled(flag):
    """Turn NPU offload on/off for this process (off = everything on the CPU)."""
    global _enabled
    _enabled = bool(flag)


def is_enabled():
    return _enabled and npu_available()


def stats(reset=False):
    out = dict(_stats)
    if reset:
        for k in _stats:
            _stats[k] = 0.0 if isinstance(_stats[k], float) else 0
    return out


def reset_verification():
    """Forget which shapes were verified (they will be checked again on next use)."""
    _verified.clear()


def _warn_once(key, msg):
    if key not in _warned:
        _warned.add(key)
        warnings.warn("rk3588_torch: " + msg, RuntimeWarning, stacklevel=3)


def threshold(precision=None):
    """Smallest GEMM (in multiply-adds) that is sent to the NPU."""
    if MIN_MACS is not None:
        return MIN_MACS
    from .calibrate import cached_threshold
    t = cached_threshold(precision or _precision)
    return DEFAULT_MIN_MACS if t is None else t


def _usable(*ts):
    for t in ts:
        if t.device.type != "cpu" or t.dtype not in (torch.float32, torch.float16, torch.bfloat16):
            return False
        if any(s < 0 for s in t.stride()):
            return False
    return True


def _as_f32(t):
    return t if t.dtype == torch.float32 else t.float()


# ---------------------------------------------------------------------------------------
# The guarded GEMM
# ---------------------------------------------------------------------------------------
def _cpu_gemm(a, b):
    t0 = time.perf_counter()
    out = _as_f32(a) @ _as_f32(b)
    _stats["cpu_time"] += time.perf_counter() - t0
    _stats["cpu_calls"] += 1
    _stats["cpu_macs"] += out.numel() * a.shape[-1]
    return out


def _npu_gemm(a, b, precision):
    """a: (M,K) or (B,M,K); b: (K,N) or (B,K,N); returns fp32 (M,N) / (B,M,N)."""
    a = _as_f32(a)
    b = _as_f32(b)
    batched = a.dim() == 3
    if not batched:
        a, b = a.unsqueeze(0), b.unsqueeze(0)
    Bt, M, K = a.shape
    N = b.shape[2]
    # broadcast batch (stride 0) is supported; broadcast inside a matrix is not
    if a.stride(1) == 0 or a.stride(2) == 0:
        a = a.contiguous()
    if b.stride(1) == 0 or b.stride(2) == 0:
        b = b.contiguous()
    out = torch.empty((Bt, M, N), dtype=torch.float32)
    t0 = time.perf_counter()
    with torch.profiler.record_function("rknpu::gemm"):
        default_npu().gemm(Bt, M, N, K, a.data_ptr(), a.stride(0) if Bt > 1 else 0, a.stride(1), a.stride(2),
                           b.data_ptr(), b.stride(0) if Bt > 1 else 0, b.stride(1), b.stride(2),
                           out.data_ptr(), M * N, N, False, precision)
    _stats["npu_time"] += time.perf_counter() - t0
    _stats["npu_calls"] += 1
    _stats["npu_macs"] += Bt * M * N * K
    return out if batched else out[0]


def _all_finite(t):
    """One pass instead of isfinite().all()'s four: any NaN/inf makes the sum non-finite. Only
    if the sum is non-finite (which a huge-but-finite sum could also cause) do the exact check."""
    if math.isfinite(t.sum().item()):
        return True
    return bool(torch.isfinite(t).all())


def _check_sample(a, b, out, precision):
    """Compare a few rows of the NPU result with an fp64 CPU reference."""
    if a.dim() == 3:
        a, b, out = a[0], b[0] if b.dim() == 3 else b, out[0]
    M = a.shape[0]
    idx = torch.unique(torch.linspace(0, M - 1, min(8, M)).round().long())
    ref = a[idx].double() @ b.double()
    den = ref.norm().item()
    if den == 0.0 or not math.isfinite(den):
        return True
    err = (out[idx].double() - ref).norm().item() / den
    return err <= VERIFY_TOL[precision]


def gemm(a, b, precision=None):
    """Guarded a @ b (2-D, or 3-D batched with equal batch), fp32 result."""
    global _failures, _enabled
    precision = precision or _precision
    macs = a.numel() // a.shape[-1] * b.shape[-1] * a.shape[-1] if a.dim() == 2 else \
        a.shape[0] * a.shape[1] * b.shape[-1] * a.shape[2]
    if not _enabled or macs < threshold(precision) or not _usable(a, b) or not npu_available():
        return _cpu_gemm(a, b)
    key = (a.dim(), tuple(a.shape), tuple(b.shape), precision, a.stride(-1) == 1, b.stride(-1) == 1)
    status = _verified.get(key)
    if status is False:
        return _cpu_gemm(a, b)
    try:
        out = _npu_gemm(a, b, precision)
    except (RuntimeError, OSError) as e:
        with _lock:
            _failures += 1
            _stats["fallbacks"] += 1
            if _failures >= MAX_FAILURES:
                _enabled = False
                _warn_once("disabled", f"NPU failed {_failures} times (last: {e}); continuing on the CPU only")
            else:
                _warn_once("fail", f"NPU GEMM failed ({e}); this call ran on the CPU")
        return _cpu_gemm(a, b)
    if not _all_finite(out):
        if _all_finite(a) and _all_finite(b):
            _verified[key] = False
            _stats["rejected_shapes"] += 1
            _stats["fallbacks"] += 1
            _warn_once(("nonfinite", key), f"non-finite NPU output for shape {key[1]}x{key[2]} ({precision}); "
                                           "using the CPU for it")
            return _cpu_gemm(a, b)
        return out   # garbage in, garbage out: same as the CPU would give
    if VERIFY == "always" or (VERIFY == "first" and status is None):
        ok = _check_sample(a, b, out, precision)
        _verified[key] = ok
        if ok:
            _stats["verified_shapes"] += 1
        else:
            _stats["rejected_shapes"] += 1
            _stats["fallbacks"] += 1
            _warn_once(("mismatch", key), f"NPU result for shape {key[1]}x{key[2]} ({precision}) didn't match "
                                          "the CPU check; using the CPU for that shape")
            return _cpu_gemm(a, b)
    return out


# ---------------------------------------------------------------------------------------
# torch.library custom op: torch.ops.rknpu.mm(a, b, precision)
# ---------------------------------------------------------------------------------------
@torch.library.custom_op("rknpu::mm", mutates_args=())
def _rknpu_mm(a: torch.Tensor, b: torch.Tensor, precision: str) -> torch.Tensor:
    return gemm(a, b, precision)


@_rknpu_mm.register_fake
def _(a, b, precision):
    return a.new_empty((*a.shape[:-1], b.shape[-1]), dtype=torch.float32)


def _mm_setup(ctx, inputs, output):
    a, b, precision = inputs
    ctx.save_for_backward(a, b)
    ctx.precision = precision


def _mm_backward(ctx, g):
    a, b = ctx.saved_tensors
    ga = gb = None
    if ctx.needs_input_grad[0]:
        ga = torch.ops.rknpu.mm(g, b.transpose(-1, -2), ctx.precision).to(a.dtype)   # dA = dC @ B^T
    if ctx.needs_input_grad[1]:
        gb = torch.ops.rknpu.mm(a.transpose(-1, -2), g, ctx.precision).to(b.dtype)   # dB = A^T @ dC
    return ga, gb, None


_rknpu_mm.register_autograd(_mm_backward, setup_context=_mm_setup)


# torch.ops.rknpu.linear(x, w, precision) = x @ w.T with dW computed directly in w's layout
# (going through w.t() would hand autograd a transposed gradient, and PyTorch then makes a
# full strided copy of every weight gradient to match the parameter's layout)
@torch.library.custom_op("rknpu::linear", mutates_args=())
def _rknpu_linear(x: torch.Tensor, w: torch.Tensor, precision: str) -> torch.Tensor:
    return gemm(x, w.t(), precision)


@_rknpu_linear.register_fake
def _(x, w, precision):
    return x.new_empty((x.shape[0], w.shape[0]), dtype=torch.float32)


def _linear_setup(ctx, inputs, output):
    x, w, precision = inputs
    ctx.save_for_backward(x, w)
    ctx.precision = precision


def _linear_backward(ctx, g):
    x, w = ctx.saved_tensors
    gx = gw = None
    if ctx.needs_input_grad[0]:
        gx = torch.ops.rknpu.mm(g, w, ctx.precision).to(x.dtype)          # dX = dY @ W
    if ctx.needs_input_grad[1]:
        gw = torch.ops.rknpu.mm(g.t(), x, ctx.precision).to(w.dtype)      # dW = dY^T @ X, W's layout
    return gx, gw, None


_rknpu_linear.register_autograd(_linear_backward, setup_context=_linear_setup)


def _out_dtype(*ts):
    dt = ts[0].dtype
    for t in ts[1:]:
        dt = torch.promote_types(dt, t.dtype)
    return dt


# ---------------------------------------------------------------------------------------
# PyTorch-shaped wrappers
# ---------------------------------------------------------------------------------------
def mm_raw(a, b, out=None, accumulate=False, precision=None):
    """C = a @ b for 2-D CPU tensors, guarded, no autograd (kept for tests/benchmarks)."""
    r = gemm(a.detach(), b.detach(), precision)
    if out is None:
        return r
    if accumulate:
        out += r
    else:
        out.copy_(r)
    return out


def matmul2d(a, b):
    return torch.ops.rknpu.mm(a, b, _precision).to(_out_dtype(a, b))


def linear(x, weight, bias=None):
    shape = x.shape
    y = torch.ops.rknpu.linear(x.reshape(-1, shape[-1]), weight, _precision).to(_out_dtype(x, weight))
    if bias is not None:
        y = y + bias
    return y.reshape(*shape[:-1], weight.shape[0])


def bmm(a, b):
    """Batched matmul: all batch elements go to the NPU in one submission."""
    return torch.ops.rknpu.mm(a, b, _precision).to(_out_dtype(a, b))


def matmul(a, b):
    """torch.matmul semantics for the common cases; returns None if unsupported."""
    if a.dim() == 2 and b.dim() == 2:
        return matmul2d(a, b)
    if a.dim() >= 3 and b.dim() == 2:
        lead = a.shape[:-1]
        return matmul2d(a.reshape(-1, a.shape[-1]), b).reshape(*lead, b.shape[-1])
    if a.dim() >= 3 and b.dim() >= 3 and a.shape[:-2] == b.shape[:-2]:
        lead = a.shape[:-2]
        a3 = a.reshape(-1, *a.shape[-2:])
        b3 = b.reshape(-1, *b.shape[-2:])
        return bmm(a3, b3).reshape(*lead, a.shape[-2], b.shape[-1])
    return None


def baddbmm(inp, b1, b2, beta=1, alpha=1):
    y = bmm(b1, b2)
    if alpha != 1:
        y = y * alpha
    if beta == 0:
        return y
    return y + (inp if beta == 1 else inp * beta)


def scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None,
                                 enable_gqa=False):
    """SDPA as two batched NPU matmuls around a CPU softmax. Returns None if unsupported."""
    if enable_gqa or q.dim() < 3 or q.shape[:-2] != k.shape[:-2] or k.shape[:-2] != v.shape[:-2]:
        return None
    lead = q.shape[:-2]
    L, E = q.shape[-2:]
    S, Ev = v.shape[-2:]
    q3 = q.reshape(-1, L, E)
    k3 = k.reshape(-1, S, E)
    v3 = v.reshape(-1, S, Ev)
    sc = scale if scale is not None else 1.0 / math.sqrt(E)
    scores = bmm(q3, k3.transpose(1, 2)) * sc                                       # (B, L, S)
    if is_causal:
        causal = torch.ones(L, S, dtype=torch.bool).tril(diagonal=S - L)
        scores = scores.masked_fill(~causal, float("-inf"))
    if attn_mask is not None:
        m = attn_mask
        if m.dim() > 2:
            m = m.expand(*lead, L, S).reshape(-1, L, S)
        scores = scores.masked_fill(~m, float("-inf")) if m.dtype == torch.bool else scores + m
    attn = torch.softmax(scores, dim=-1)
    if dropout_p > 0.0:
        attn = torch.nn.functional.dropout(attn, p=dropout_p)
    out = bmm(attn.to(v.dtype), v3)                                                 # (B, L, Ev)
    return out.reshape(*lead, L, Ev)


# ---------------------------------------------------------------------------------------
# Convolution: native NPU conv (NC1HWC2), guarded like gemm()
# ---------------------------------------------------------------------------------------
def _pair(v):
    return (v, v) if isinstance(v, int) else tuple(v)


def conv_supported(x, weight, stride, padding, dilation, groups):
    st, pd, dl = _pair(stride), _pair(padding), _pair(dilation)
    if isinstance(padding, str) or groups != 1 or dl != (1, 1) or x.dim() != 4 or weight.dim() != 4:
        return False
    kh, kw = weight.shape[2:]
    return (1 <= st[0] <= 7 and 1 <= st[1] <= 7 and 0 <= pd[0] <= 15 and 0 <= pd[1] <= 15 and kh <= 31 and kw <= 31
            and x.shape[2] + 2 * pd[0] >= kh and x.shape[3] + 2 * pd[1] >= kw and x.shape[3] <= 2047)


def _cpu_conv(x, w, st, pd):
    t0 = time.perf_counter()
    out = torch.nn.functional.conv2d(_as_f32(x), _as_f32(w), stride=st, padding=pd)
    _stats["cpu_time"] += time.perf_counter() - t0
    _stats["cpu_calls"] += 1
    _stats["cpu_macs"] += out.numel() * w.shape[1] * w.shape[2] * w.shape[3]
    return out


def _npu_conv(x, w, st, pd, precision):
    x = _as_f32(x).contiguous()
    w = _as_f32(w).contiguous()
    N, C, H, W = x.shape
    O, _, kh, kw = w.shape
    oh = (H + 2 * pd[0] - kh) // st[0] + 1
    ow = (W + 2 * pd[1] - kw) // st[1] + 1
    y = torch.empty((N, O, oh, ow), dtype=torch.float32)
    t0 = time.perf_counter()
    with torch.profiler.record_function("rknpu::conv2d"):
        default_npu().conv2d(N, C, H, W, x.data_ptr(), O, kh, kw, w.data_ptr(), st[0], st[1], pd[0], pd[1], oh, ow,
                             y.data_ptr(), False, precision)
    _stats["npu_time"] += time.perf_counter() - t0
    _stats["npu_calls"] += 1
    _stats["npu_macs"] += y.numel() * C * kh * kw
    return y


def conv(x, w, stride=1, padding=0, precision=None):
    """Guarded conv2d (no bias, groups=1, dilation=1), fp32 result."""
    global _failures, _enabled
    precision = precision or _precision
    if precision not in ("fp16", "bf16"):
        precision = "fp16"                 # the conv path implements fp16 and bf16
    st, pd = _pair(stride), _pair(padding)
    macs = x.shape[0] * w.shape[0] * w.shape[1] * w.shape[2] * w.shape[3] * \
        ((x.shape[2] + 2 * pd[0] - w.shape[2]) // st[0] + 1) * ((x.shape[3] + 2 * pd[1] - w.shape[3]) // st[1] + 1)
    if (not _enabled or macs < threshold(precision) or not _usable(x, w) or not npu_available()
            or not conv_supported(x, w, st, pd, 1, 1)):
        return _cpu_conv(x, w, st, pd)
    key = ("conv", tuple(x.shape), tuple(w.shape), st, pd, precision)
    status = _verified.get(key)
    if status is False:
        return _cpu_conv(x, w, st, pd)
    try:
        out = _npu_conv(x, w, st, pd, precision)
    except (RuntimeError, OSError) as e:
        with _lock:
            _failures += 1
            _stats["fallbacks"] += 1
            if _failures >= MAX_FAILURES:
                _enabled = False
                _warn_once("disabled", f"NPU failed {_failures} times (last: {e}); continuing on the CPU only")
            else:
                _warn_once("fail", f"NPU conv failed ({e}); this call ran on the CPU")
        return _cpu_conv(x, w, st, pd)
    bad = not _all_finite(out)
    if bad or VERIFY == "always" or (VERIFY == "first" and status is None):
        ok = not bad
        if ok:   # check a few output channels of the first image against fp64
            o_idx = torch.unique(torch.linspace(0, w.shape[0] - 1, min(4, w.shape[0])).round().long())
            ref = torch.nn.functional.conv2d(x[:1].double(), w[o_idx].double(), stride=st, padding=pd)
            den = ref.norm().item()
            ok = den == 0.0 or (out[:1, o_idx].double() - ref).norm().item() / den <= VERIFY_TOL[precision]
        if bad and not (_all_finite(x) and _all_finite(w)):
            return out
        _verified[key] = ok
        if not ok:
            _stats["rejected_shapes"] += 1
            _stats["fallbacks"] += 1
            _warn_once(("conv", key), f"NPU conv for {tuple(x.shape)} * {tuple(w.shape)} didn't match the CPU check; "
                                      "using the CPU for it")
            return _cpu_conv(x, w, st, pd)
        _stats["verified_shapes"] += 1
    return out


@torch.library.custom_op("rknpu::conv2d", mutates_args=())
def _rknpu_conv2d(x: torch.Tensor, w: torch.Tensor, stride: list[int], padding: list[int],
                  precision: str) -> torch.Tensor:
    return conv(x, w, stride, padding, precision)


@_rknpu_conv2d.register_fake
def _(x, w, stride, padding, precision):
    oh = (x.shape[2] + 2 * padding[0] - w.shape[2]) // stride[0] + 1
    ow = (x.shape[3] + 2 * padding[1] - w.shape[3]) // stride[1] + 1
    return x.new_empty((x.shape[0], w.shape[0], oh, ow), dtype=torch.float32)


def _conv_setup(ctx, inputs, output):
    x, w, stride, padding, precision = inputs
    ctx.save_for_backward(x, w)
    ctx.stride, ctx.padding, ctx.precision = list(stride), list(padding), precision


def _conv_backward(ctx, g):
    x, w = ctx.saved_tensors
    (sy, sx), (pt, pl) = ctx.stride, ctx.padding
    O, C, kh, kw = w.shape
    N, _, H, W = x.shape
    gx = gw = None
    g = g.contiguous()
    if ctx.needs_input_grad[0]:
        if pt <= kh - 1 and pl <= kw - 1:
            # dX = conv(dY dilated by the stride, W rotated 180 deg with in/out swapped, pad k-1-p).
            # When the stride doesn't divide the input evenly, the leftover rows/columns
            # (output padding) are zero rows of dY at the bottom/right, not zeros of dX.
            oph = H + 2 * pt - kh - (g.shape[2] - 1) * sy
            opw = W + 2 * pl - kw - (g.shape[3] - 1) * sx
            if sy > 1 or sx > 1 or oph or opw:
                gd = g.new_zeros((N, O, (g.shape[2] - 1) * sy + 1 + oph, (g.shape[3] - 1) * sx + 1 + opw))
                gd[:, :, :(g.shape[2] - 1) * sy + 1:sy, :(g.shape[3] - 1) * sx + 1:sx] = g
            else:
                gd = g
            wr = w.flip(2, 3).transpose(0, 1).contiguous()
            gx = torch.ops.rknpu.conv2d(gd, wr, [1, 1], [kh - 1 - pt, kw - 1 - pl], ctx.precision)
        else:
            gx = torch.nn.grad.conv2d_input(x.shape, w.float(), g, stride=(sy, sx), padding=(pt, pl))
        gx = gx.to(x.dtype)
    if ctx.needs_input_grad[1]:
        # dW = dY (O x N*L) @ unfold(X)^T (N*L x C*kh*kw), one NPU GEMM
        cols = torch.nn.functional.unfold(x.float(), (kh, kw), padding=(pt, pl), stride=(sy, sx))   # N, C*kh*kw, L
        L = cols.shape[2]
        gm = g.reshape(N, O, L).permute(1, 0, 2).reshape(O, N * L)
        cm = cols.permute(0, 2, 1).reshape(N * L, C * kh * kw)
        gw = torch.ops.rknpu.mm(gm, cm, ctx.precision).reshape(O, C, kh, kw).to(w.dtype)
    return gx, gw, None, None, None


_rknpu_conv2d.register_autograd(_conv_backward, setup_context=_conv_setup)


def conv2d(x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    """F.conv2d with the convolution (forward and backward) on the NPU. Returns None if unsupported."""
    if not conv_supported(x, weight, stride, padding, dilation, groups):
        return None
    y = torch.ops.rknpu.conv2d(x, weight, list(_pair(stride)), list(_pair(padding)), _precision)
    y = y.to(_out_dtype(x, weight))
    if bias is not None:
        y = y + bias.view(1, -1, 1, 1)
    return y


def release():
    """Free the NPU's DMA buffers (they are re-created on demand)."""
    if npu_available():
        default_npu().release()
