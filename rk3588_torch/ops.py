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
from typing import Optional

import torch

from . import wcache
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
# The NPU does not propagate inf/nan: a non-finite input element yields *finite* garbage in its
# row (every precision, and convolutions), which would hide a diverging run or a NaN gradient.
# So inputs are checked first (one fast reduction each) and non-finite ones go to the CPU.
CHECK_FINITE = os.environ.get("RKNPU_CHECK_FINITE", "1") not in ("0", "false", "no")

# bf16/fp16 operands are converted to fp32 for the NPU library. An operand pair whose fp32 copy would be
# bigger than this many MB (an LLM's 150k-row embedding / lm_head, a huge logits gradient) is processed in
# pieces instead, so one op never makes a gigabyte-sized temporary on a 4 GB board. 0 disables the split.
CONVERT_MB = int(os.environ.get("RKNPU_CONVERT_MB", "128"))
# First-use check of a new shape: a weight with more than VERIFY_MAX_ELEMS elements is checked on a strided
# subset of VERIFY_COLS output columns (covering every tile) instead of converting all of it to fp64.
VERIFY_MAX_ELEMS = 8_000_000
VERIFY_COLS = 4096
VERIFY_CHUNK_ELEMS = 4_000_000

_CACHEABLE = (torch.float32, torch.bfloat16, torch.float16)

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
def _epilogue(out, bias, relu):
    if bias is not None:
        out = out + _as_f32(bias)
    return torch.relu(out) if relu else out


def _cpu_gemm(a, b, bias=None, relu=False):
    t0 = time.perf_counter()
    out = _epilogue(_as_f32(a) @ _as_f32(b), bias, relu)
    _stats["cpu_time"] += time.perf_counter() - t0
    _stats["cpu_calls"] += 1
    _stats["cpu_macs"] += out.numel() * a.shape[-1]
    return out


def _npu_gemm(a, b, precision, bias=None, relu=False, wkey=0):
    """a: (M,K) or (B,M,K); b: (K,N) or (B,K,N); returns fp32 (M,N) / (B,M,N), with the
    epilogue relu?(. + bias[n]) applied by the library."""
    a = _as_f32(a)
    b = _as_f32(b)
    if bias is not None:
        bias = _as_f32(bias).contiguous()
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
                           out.data_ptr(), M * N, N, False, precision,
                           bias_ptr=bias.data_ptr() if bias is not None else 0, relu=relu, wkey=wkey)
    _stats["npu_time"] += time.perf_counter() - t0
    _stats["npu_calls"] += 1
    _stats["npu_macs"] += Bt * M * N * K
    return out if batched else out[0]


def _all_finite(t):
    """One pass instead of isfinite().all()'s four: any NaN/inf makes the sum non-finite (float16/bfloat16
    are summed in fp32, without a copy). Only if the sum is non-finite (which a huge-but-finite sum could
    also cause) do the exact check."""
    s = t.sum() if t.dtype == torch.float32 else t.sum(dtype=torch.float32)
    if math.isfinite(s.item()):
        return True
    return bool(torch.isfinite(t).all())


def _check_sample(a, b, out, precision, bias=None, relu=False):
    """Compare a few rows of the NPU result with an fp64 CPU reference. The reference is built in column
    chunks (never an fp64 copy of the whole weight: a 150k x 896 lm_head would need over 1 GB), and for a
    very large weight only on a strided subset of the output columns."""
    if a.dim() == 3:
        a, b, out = a[0], b[0] if b.dim() == 3 else b, out[0]
    M, (K, N) = a.shape[0], b.shape
    idx = torch.unique(torch.linspace(0, M - 1, min(8, M)).round().long())
    ai = a[idx].double()
    if K * N > VERIFY_MAX_ELEMS and N > VERIFY_COLS:
        cols = torch.unique(torch.linspace(0, N - 1, VERIFY_COLS).round().long())
        ref = ai @ b[:, cols].double()
        got = out[idx][:, cols].double()
        bias_d = None if bias is None else bias[cols].double()
    else:
        step = max(VERIFY_CHUNK_ELEMS // max(K, 1), 1)
        ref = torch.cat([ai @ b[:, n0:n0 + step].double() for n0 in range(0, N, step)], 1)
        got = out[idx].double()
        bias_d = None if bias is None else bias.double()
    if bias_d is not None:
        ref = ref + bias_d
    if relu:
        ref = ref.clamp_min(0)
    den = ref.norm().item()
    if den == 0.0 or not math.isfinite(den):
        return True
    err = (got - ref).norm().item() / den
    return err <= VERIFY_TOL[precision]


def _convert_bytes(a, b):
    """Bytes of fp32 temporaries _npu_gemm would make for bf16/fp16 operands."""
    return 4 * ((a.numel() if a.dtype != torch.float32 else 0) + (b.numel() if b.dtype != torch.float32 else 0))


def _gemm_split(a, b, precision, bias, relu):
    """Split a GEMM whose low-precision operands would need a huge fp32 copy, along the dimension that
    shrinks the biggest one (N or M, else K, which accumulates). Each half goes back through gemm(), so
    it splits again if still too big. The pieces are not weight-cached (they are views of a weight that
    is too big to cache anyway) and are verified by their own shapes. Returns None if it can't split."""
    if a.dim() == 3:
        if a.shape[0] > 1:
            h = a.shape[0] // 2
            return torch.cat([gemm(a[:h], b[:h], precision, bias, relu), gemm(a[h:], b[h:], precision, bias, relu)])
        return gemm(a[0], b[0], precision, bias, relu).unsqueeze(0)
    M, K = a.shape
    N = b.shape[1]
    ca = 4 * a.numel() if a.dtype != torch.float32 else 0
    cb = 4 * b.numel() if b.dtype != torch.float32 else 0
    dims = [("n", N), ("k", K)] if cb >= ca else [("m", M), ("k", K)]
    dims.sort(key=lambda d: -d[1])
    for name, size in dims:
        if size < 2:
            continue
        h = size // 2
        if name == "m":
            return torch.cat([gemm(a[:h], b, precision, bias, relu), gemm(a[h:], b, precision, bias, relu)], 0)
        if name == "n":
            bl = None if bias is None else bias[:h]
            br = None if bias is None else bias[h:]
            return torch.cat([gemm(a, b[:, :h], precision, bl, relu), gemm(a, b[:, h:], precision, br, relu)], 1)
        part = gemm(a[:, :h], b[:h], precision) + gemm(a[:, h:], b[h:], precision)
        return _epilogue(part, bias, relu)
    return None


def gemm(a, b, precision=None, bias=None, relu=False, cache_b=None):
    """Guarded relu?(a @ b + bias) (2-D, or 3-D batched with equal batch), fp32 result.
    bias: optional (N,) added to every row; cache_b: a tag naming b as a weight operand,
    which lets the packed-weight cache (wcache.py) keep its converted form between calls."""
    global _failures, _enabled
    precision = precision or _precision
    if CONVERT_MB and a.dim() in (2, 3) and _convert_bytes(a, b) > CONVERT_MB << 20:
        out = _gemm_split(a, b, precision, bias, relu)
        if out is not None:
            return out
    macs = a.numel() // a.shape[-1] * b.shape[-1] * a.shape[-1] if a.dim() == 2 else \
        a.shape[0] * a.shape[1] * b.shape[-1] * a.shape[2]
    if (not _enabled or macs < threshold(precision) or not _usable(a, b) or not npu_available()
            or (bias is not None and not _usable(bias))):
        return _cpu_gemm(a, b, bias, relu)
    key = (a.dim(), tuple(a.shape), tuple(b.shape), precision, a.stride(-1) == 1, b.stride(-1) == 1)
    status = _verified.get(key)
    if status is False:
        return _cpu_gemm(a, b, bias, relu)
    if CHECK_FINITE and not (_all_finite(a) and (bias is None or _all_finite(bias))):
        return _cpu_gemm(a, b, bias, relu)
    # a weight that is already in the cache was checked when it went in and hasn't changed since (the
    # cache compares a fingerprint on every hit), so it isn't scanned again; fp32, bf16 and fp16 weights
    # are all cacheable
    wkey, b_ok = 0, False
    if cache_b and b.dim() == 2 and b.dtype in _CACHEABLE and precision != "fp16x3":
        wkey, b_ok = wcache.lookup(b, cache_b, check=_all_finite if CHECK_FINITE else None)
        b_ok = b_ok or wkey != 0
    if CHECK_FINITE and not b_ok and not _all_finite(b):
        return _cpu_gemm(a, b, bias, relu)
    try:
        out = _npu_gemm(a, b, precision, bias, relu, wkey)
    except (RuntimeError, OSError) as e:
        with _lock:
            _failures += 1
            _stats["fallbacks"] += 1
            if _failures >= MAX_FAILURES:
                _enabled = False
                _warn_once("disabled", f"NPU failed {_failures} times (last: {e}); continuing on the CPU only")
            else:
                _warn_once("fail", f"NPU GEMM failed ({e}); this call ran on the CPU")
        return _cpu_gemm(a, b, bias, relu)
    if not _all_finite(out):
        if _all_finite(a) and _all_finite(b) and (bias is None or _all_finite(bias)):
            _verified[key] = False
            _stats["rejected_shapes"] += 1
            _stats["fallbacks"] += 1
            _warn_once(("nonfinite", key), f"non-finite NPU output for shape {key[1]}x{key[2]} ({precision}); "
                                           "using the CPU for it")
            return _cpu_gemm(a, b, bias, relu)
        # non-finite inputs (only reachable with RKNPU_CHECK_FINITE=0): the NPU mangles inf/nan,
        # so let the CPU produce PyTorch's exact result
        return _cpu_gemm(a, b, bias, relu)
    if VERIFY == "always" or (VERIFY == "first" and status is None):
        ok = _check_sample(a, b, out, precision, bias, relu)
        _verified[key] = ok
        if ok:
            _stats["verified_shapes"] += 1
        else:
            _stats["rejected_shapes"] += 1
            _stats["fallbacks"] += 1
            _warn_once(("mismatch", key), f"NPU result for shape {key[1]}x{key[2]} ({precision}) didn't match "
                                          "the CPU check; using the CPU for that shape")
            return _cpu_gemm(a, b, bias, relu)
    return out


# ---------------------------------------------------------------------------------------
# torch.library custom ops (autograd + torch.compile)
# ---------------------------------------------------------------------------------------
# torch.ops.rknpu.mm(a, b, precision, cache_b): a @ b. cache_b names b as a weight operand
# ("" = not a weight), which allows the packed-weight cache to keep its NPU form.
@torch.library.custom_op("rknpu::mm", mutates_args=())
def _rknpu_mm(a: torch.Tensor, b: torch.Tensor, precision: str, cache_b: str = "") -> torch.Tensor:
    return gemm(a, b, precision, cache_b=cache_b or None)


@_rknpu_mm.register_fake
def _(a, b, precision, cache_b=""):
    return a.new_empty((*a.shape[:-1], b.shape[-1]), dtype=torch.float32)


def _mm_setup(ctx, inputs, output):
    a, b, precision = inputs[:3]
    ctx.save_for_backward(a, b)
    ctx.precision = precision


def _mm_backward(ctx, g):
    a, b = ctx.saved_tensors
    ga = gb = None
    if ctx.needs_input_grad[0]:
        ga = torch.ops.rknpu.mm(g, b.transpose(-1, -2), ctx.precision).to(a.dtype)   # dA = dC @ B^T
    if ctx.needs_input_grad[1]:
        gb = torch.ops.rknpu.mm(a.transpose(-1, -2), g, ctx.precision).to(b.dtype)   # dB = A^T @ dC
    return ga, gb, None, None


_rknpu_mm.register_autograd(_mm_backward, setup_context=_mm_setup)


# torch.ops.rknpu.linear(x, w, bias, act, precision) = act(x @ w.T + bias), with the bias and
# ReLU applied while the NPU output is unpacked, and dW computed directly in w's layout (going
# through w.t() would hand autograd a transposed gradient, which PyTorch then copies).
@torch.library.custom_op("rknpu::linear", mutates_args=())
def _rknpu_linear(x: torch.Tensor, w: torch.Tensor, bias: Optional[torch.Tensor], act: str,
                  precision: str) -> torch.Tensor:
    return gemm(x, w.t(), precision, bias=bias, relu=act == "relu", cache_b="linear.wT")


@_rknpu_linear.register_fake
def _(x, w, bias, act, precision):
    return x.new_empty((x.shape[0], w.shape[0]), dtype=torch.float32)


def _linear_setup(ctx, inputs, output):
    x, w, bias, act, precision = inputs
    ctx.relu = act == "relu"
    ctx.save_for_backward(x, w, output if ctx.relu else None)
    ctx.precision = precision
    ctx.has_bias = bias is not None
    ctx.bias_dtype = bias.dtype if bias is not None else None


def _linear_backward(ctx, g):
    x, w, y = ctx.saved_tensors
    if ctx.relu:
        g = g * (y > 0)
    gx = gw = gb = None
    if ctx.needs_input_grad[0]:
        gx = torch.ops.rknpu.mm(g, w, ctx.precision, "linear.w").to(x.dtype)       # dX = dY @ W
    if ctx.needs_input_grad[1]:
        gw = torch.ops.rknpu.mm(g.t(), x, ctx.precision).to(w.dtype)                # dW = dY^T @ X, W's layout
    if ctx.has_bias and ctx.needs_input_grad[2]:
        gb = g.sum(0).to(ctx.bias_dtype)
    return gx, gw, gb, None, None


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


def linear(x, weight, bias=None, act=None):
    """F.linear (optionally followed by ReLU: act="relu") with the matmuls on the NPU and the
    bias / ReLU fused into the NPU output stage."""
    shape = x.shape
    if bias is not None and (bias.dim() != 1 or bias.shape[0] != weight.shape[0]):
        y = torch.ops.rknpu.linear(x.reshape(-1, shape[-1]), weight, None, "none", _precision) + bias
        y = torch.relu(y) if act == "relu" else y
    else:
        y = torch.ops.rknpu.linear(x.reshape(-1, shape[-1]), weight, bias, act or "none", _precision)
    dt = _out_dtype(x, weight) if bias is None else _out_dtype(x, weight, bias)
    return y.to(dt).reshape(*shape[:-1], weight.shape[0])


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
    """SDPA as two batched NPU matmuls around a CPU softmax. Returns None if unsupported.
    enable_gqa (grouped-query attention: fewer key/value heads than query heads, as in Llama, Qwen,
    Mistral ...) repeats each key/value head for its group of query heads."""
    if enable_gqa:
        hq, hk = (q.shape[-3], k.shape[-3]) if q.dim() >= 4 and k.dim() >= 4 else (0, 0)
        if hk == 0 or hq % hk or v.shape[-3] != hk:
            return None
        if hq != hk:
            k = k.repeat_interleave(hq // hk, dim=-3)
            v = v.repeat_interleave(hq // hk, dim=-3)
    if q.dim() < 3 or q.shape[:-2] != k.shape[:-2] or k.shape[:-2] != v.shape[:-2]:
        return None
    lead = q.shape[:-2]
    L, E = q.shape[-2:]
    S, Ev = v.shape[-2:]
    q3 = q.reshape(-1, L, E)
    k3 = k.reshape(-1, S, E)
    v3 = v.reshape(-1, S, Ev)
    sc = scale if scale is not None else 1.0 / math.sqrt(E)
    scores = bmm(q3, k3.transpose(1, 2)).float() * sc                               # (B, L, S), fp32 softmax
    if is_causal:
        # same convention as PyTorch: top-left aligned, i.e. tril(diagonal=0) even when L != S
        causal = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
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
# einsum: two-operand contractions as one (batched) NPU matmul
# ---------------------------------------------------------------------------------------
def _parse_einsum(eq, nd_a, nd_b):
    eq = eq.replace(" ", "")
    if "..." in eq or "->" not in eq:
        return None
    lhs, out = eq.split("->")
    ins = lhs.split(",")
    if len(ins) != 2:
        return None
    ia, ib = ins
    if len(ia) != nd_a or len(ib) != nd_b or len(set(ia)) != len(ia) or len(set(ib)) != len(ib) \
            or len(set(out)) != len(out) or not set(out) <= set(ia) | set(ib) or not (ia + ib + out).isalpha():
        return None
    return ia, ib, out


def einsum(eq, a, b):
    """torch.einsum(eq, a, b) as permute/reshape + one NPU (batched) matmul + permute.
    Returns None for what it doesn't handle (ellipsis, repeated indices, broadcasting, other
    operand counts); those stay on the CPU."""
    p = _parse_einsum(eq, a.dim(), b.dim())
    if p is None:
        return None
    ia, ib, out = p
    size = {}
    for idx, t in ((ia, a), (ib, b)):
        for ch, n in zip(idx, t.shape):
            if size.setdefault(ch, n) != n:
                return None                       # broadcasting (size-1 vs n): leave to the CPU
    batch = [c for c in out if c in ia and c in ib]
    fa = [c for c in out if c in ia and c not in ib]
    fb = [c for c in out if c in ib and c not in ia]
    contr = [c for c in ia if c in ib and c not in out]
    # indices in one operand only and not in the output are summed out first
    sa = [c for c in ia if c not in ib and c not in out]
    sb = [c for c in ib if c not in ia and c not in out]
    if sa:
        a = a.sum(dim=[ia.index(c) for c in sa])
        ia = "".join(c for c in ia if c not in sa)
    if sb:
        b = b.sum(dim=[ib.index(c) for c in sb])
        ib = "".join(c for c in ib if c not in sb)
    prod = lambda cs: math.prod(size[c] for c in cs)  # noqa: E731
    B, M, N, K = prod(batch), prod(fa), prod(fb), prod(contr)
    a3 = a.permute([ia.index(c) for c in batch + fa + contr]).reshape(B, M, K)
    b3 = b.permute([ib.index(c) for c in batch + contr + fb]).reshape(B, K, N)
    y = bmm(a3, b3) if B > 1 else matmul2d(a3[0], b3[0]).unsqueeze(0)
    order = batch + fa + fb
    y = y.reshape([size[c] for c in order]) if order else y.reshape(())
    return y.permute([order.index(c) for c in out])


def einsum_macs(eq, a, b):
    p = _parse_einsum(eq, a.dim(), b.dim())
    if p is None:
        return 0
    ia, ib, _ = p
    size = dict(zip(ia, a.shape))
    size.update(zip(ib, b.shape))
    return math.prod(size.values())


# ---------------------------------------------------------------------------------------
# Convolution: native NPU conv (NC1HWC2), guarded like gemm()
# ---------------------------------------------------------------------------------------
def _pair(v):
    return (v, v) if isinstance(v, int) else tuple(v)


def _groups_ok(C, O, groups):
    """Grouped convolutions run as one NPU task chain with every group padded to 32 input /
    16 output channels; worth it only if that padding wastes less than ~8x. Depthwise
    (1 channel per group) stays on the CPU: the NPU's depthwise mode isn't reverse-engineered."""
    if groups == 1:
        return True
    if C % groups or O % groups:
        return False
    cg, og = C // groups, O // groups
    if cg < 2:
        return False
    eff = (cg / (-(-cg // 32) * 32)) * (og / (-(-og // 16) * 16))
    return eff >= 1 / 8


def conv_supported(x, weight, stride, padding, dilation, groups):
    st, pd, dl = _pair(stride), _pair(padding), _pair(dilation)
    if isinstance(padding, str) or dl != (1, 1) or x.dim() != 4 or weight.dim() != 4:
        return False
    if not _groups_ok(x.shape[1], weight.shape[0], groups) or weight.shape[1] * groups != x.shape[1]:
        return False
    kh, kw = weight.shape[2:]
    # heights below 5 give the NPU a non-positive input surface stride (untested territory)
    return (1 <= st[0] <= 7 and 1 <= st[1] <= 7 and 0 <= pd[0] <= 15 and 0 <= pd[1] <= 15 and kh <= 31 and kw <= 31
            and x.shape[2] + 2 * pd[0] >= kh and x.shape[3] + 2 * pd[1] >= kw and x.shape[3] <= 2047
            and 5 <= x.shape[2] <= 2047)


def _cpu_conv(x, w, st, pd, groups=1, bias=None, relu=False):
    t0 = time.perf_counter()
    out = torch.nn.functional.conv2d(_as_f32(x), _as_f32(w), None if bias is None else _as_f32(bias),
                                     stride=st, padding=pd, groups=groups)
    if relu:
        out = torch.relu(out)
    _stats["cpu_time"] += time.perf_counter() - t0
    _stats["cpu_calls"] += 1
    _stats["cpu_macs"] += out.numel() * w.shape[1] * w.shape[2] * w.shape[3]
    return out


def _npu_conv(x, w, st, pd, precision, groups=1, bias=None, relu=False, wkey=0):
    x = _as_f32(x).contiguous()
    w = _as_f32(w).contiguous()
    if bias is not None:
        bias = _as_f32(bias).contiguous()
    N, C, H, W = x.shape
    O, _, kh, kw = w.shape
    oh = (H + 2 * pd[0] - kh) // st[0] + 1
    ow = (W + 2 * pd[1] - kw) // st[1] + 1
    y = torch.empty((N, O, oh, ow), dtype=torch.float32)
    t0 = time.perf_counter()
    with torch.profiler.record_function("rknpu::conv2d"):
        default_npu().conv2d(N, C, H, W, x.data_ptr(), O, kh, kw, w.data_ptr(), st[0], st[1], pd[0], pd[1], oh, ow,
                             y.data_ptr(), False, precision, groups=groups,
                             bias_ptr=bias.data_ptr() if bias is not None else 0, relu=relu, wkey=wkey)
    _stats["npu_time"] += time.perf_counter() - t0
    _stats["npu_calls"] += 1
    _stats["npu_macs"] += y.numel() * w.shape[1] * kh * kw
    return y


def conv(x, w, stride=1, padding=0, precision=None, groups=1, bias=None, relu=False, cache_w=None):
    """Guarded relu?(conv2d(x, w) + bias) (dilation 1), fp32 result. cache_w: tag naming w as a
    weight, letting the packed-weight cache keep its converted form."""
    global _failures, _enabled
    precision = precision or _precision
    if precision not in ("fp16", "bf16"):
        if precision == "fp16x3":
            _warn_once("conv_prec", "convolutions have no fp16x3 path; they run in fp16 (use the CPU for "
                                    "fp32-accurate convolutions)")
        precision = "fp16"                 # the conv path implements fp16 and bf16
    st, pd = _pair(stride), _pair(padding)
    macs = x.shape[0] * w.shape[0] * w.shape[1] * w.shape[2] * w.shape[3] * \
        ((x.shape[2] + 2 * pd[0] - w.shape[2]) // st[0] + 1) * ((x.shape[3] + 2 * pd[1] - w.shape[3]) // st[1] + 1)
    if (not _enabled or macs < threshold(precision) or not _usable(x, w) or not npu_available()
            or not conv_supported(x, w, st, pd, 1, groups) or (bias is not None and not _usable(bias))):
        return _cpu_conv(x, w, st, pd, groups, bias, relu)
    key = ("conv", tuple(x.shape), tuple(w.shape), st, pd, groups, precision)
    status = _verified.get(key)
    if status is False:
        return _cpu_conv(x, w, st, pd, groups, bias, relu)
    if CHECK_FINITE and not (_all_finite(x) and (bias is None or _all_finite(bias))):
        return _cpu_conv(x, w, st, pd, groups, bias, relu)
    wkey, w_ok = 0, False
    if cache_w and w.dtype in _CACHEABLE:
        wkey, w_ok = wcache.lookup(w, cache_w, check=_all_finite if CHECK_FINITE else None)
        w_ok = w_ok or wkey != 0
    if CHECK_FINITE and not w_ok and not _all_finite(w):
        return _cpu_conv(x, w, st, pd, groups, bias, relu)
    try:
        out = _npu_conv(x, w, st, pd, precision, groups, bias, relu, wkey)
    except (RuntimeError, OSError) as e:
        with _lock:
            _failures += 1
            _stats["fallbacks"] += 1
            if _failures >= MAX_FAILURES:
                _enabled = False
                _warn_once("disabled", f"NPU failed {_failures} times (last: {e}); continuing on the CPU only")
            else:
                _warn_once("fail", f"NPU conv failed ({e}); this call ran on the CPU")
        return _cpu_conv(x, w, st, pd, groups, bias, relu)
    bad = not _all_finite(out)
    if bad or VERIFY == "always" or (VERIFY == "first" and status is None):
        ok = not bad
        if ok:   # check a few output channels of the first image against fp64
            O = w.shape[0]
            og = O // groups
            o_idx = torch.unique(torch.linspace(0, O - 1, min(4, O)).round().long())
            ref = torch.cat([torch.nn.functional.conv2d(
                x[:1, (o // og) * w.shape[1]:(o // og + 1) * w.shape[1]].double(), w[o:o + 1].double(),
                None if bias is None else bias[o:o + 1].double(), stride=st, padding=pd) for o in o_idx.tolist()], 1)
            if relu:
                ref = ref.clamp_min(0)
            den = ref.norm().item()
            ok = den == 0.0 or (out[:1, o_idx].double() - ref).norm().item() / den <= VERIFY_TOL[precision]
        if bad and not (_all_finite(x) and _all_finite(w)):
            return _cpu_conv(x, w, st, pd, groups, bias, relu)   # non-finite input: PyTorch's exact result
        _verified[key] = ok
        if not ok:
            _stats["rejected_shapes"] += 1
            _stats["fallbacks"] += 1
            _warn_once(("conv", key), f"NPU conv for {tuple(x.shape)} * {tuple(w.shape)} didn't match the CPU check; "
                                      "using the CPU for it")
            return _cpu_conv(x, w, st, pd, groups, bias, relu)
        _stats["verified_shapes"] += 1
    return out


@torch.library.custom_op("rknpu::conv2d", mutates_args=())
def _rknpu_conv2d(x: torch.Tensor, w: torch.Tensor, bias: Optional[torch.Tensor], stride: list[int],
                  padding: list[int], groups: int, act: str, precision: str, cache_w: str = "") -> torch.Tensor:
    return conv(x, w, stride, padding, precision, groups, bias, act == "relu", cache_w or None)


@_rknpu_conv2d.register_fake
def _(x, w, bias, stride, padding, groups, act, precision, cache_w=""):
    oh = (x.shape[2] + 2 * padding[0] - w.shape[2]) // stride[0] + 1
    ow = (x.shape[3] + 2 * padding[1] - w.shape[3]) // stride[1] + 1
    return x.new_empty((x.shape[0], w.shape[0], oh, ow), dtype=torch.float32)


def _conv_setup(ctx, inputs, output):
    x, w, bias, stride, padding, groups, act, precision = inputs[:8]
    ctx.relu = act == "relu"
    ctx.save_for_backward(x, w, output if ctx.relu else None)
    ctx.stride, ctx.padding, ctx.groups, ctx.precision = list(stride), list(padding), groups, precision
    ctx.has_bias = bias is not None
    ctx.bias_dtype = bias.dtype if bias is not None else None


def _conv_backward(ctx, g):
    x, w, y = ctx.saved_tensors
    (sy, sx), (pt, pl), G = ctx.stride, ctx.padding, ctx.groups
    O, Cg, kh, kw = w.shape
    N, C, H, W = x.shape
    Og = O // G
    if ctx.relu:
        g = g * (y > 0)
    gx = gw = gb = None
    g = g.contiguous()
    if ctx.needs_input_grad[0]:
        if pt <= kh - 1 and pl <= kw - 1:
            # dX = conv(dY dilated by the stride, W rotated 180 deg with in/out swapped per group,
            # pad k-1-p). When the stride doesn't divide the input evenly, the leftover rows/columns
            # (output padding) are zero rows of dY at the bottom/right, not zeros of dX.
            oph = H + 2 * pt - kh - (g.shape[2] - 1) * sy
            opw = W + 2 * pl - kw - (g.shape[3] - 1) * sx
            if sy > 1 or sx > 1 or oph or opw:
                gd = g.new_zeros((N, O, (g.shape[2] - 1) * sy + 1 + oph, (g.shape[3] - 1) * sx + 1 + opw))
                gd[:, :, :(g.shape[2] - 1) * sy + 1:sy, :(g.shape[3] - 1) * sx + 1:sx] = g
            else:
                gd = g
            wr = w.reshape(G, Og, Cg, kh, kw).transpose(1, 2).reshape(G * Cg, Og, kh, kw).flip(2, 3).contiguous()
            gx = torch.ops.rknpu.conv2d(gd, wr, None, [1, 1], [kh - 1 - pt, kw - 1 - pl], G, "none", ctx.precision)
        else:
            gx = torch.nn.grad.conv2d_input(x.shape, w.float(), g, stride=(sy, sx), padding=(pt, pl), groups=G)
        gx = gx.to(x.dtype)
    if ctx.needs_input_grad[1]:
        # dW = dY (O x N*L) @ unfold(X)^T (N*L x C*kh*kw): one NPU GEMM (one batched GEMM per group)
        cols = torch.nn.functional.unfold(x.float(), (kh, kw), padding=(pt, pl), stride=(sy, sx))   # N, C*kh*kw, L
        L = cols.shape[2]
        if G == 1:
            gm = g.reshape(N, O, L).permute(1, 0, 2).reshape(O, N * L)
            cm = cols.permute(0, 2, 1).reshape(N * L, C * kh * kw)
            gw = torch.ops.rknpu.mm(gm, cm, ctx.precision)
        else:
            gm = g.reshape(N, G, Og, L).permute(1, 2, 0, 3).reshape(G, Og, N * L)
            cm = cols.reshape(N, G, Cg * kh * kw, L).permute(1, 0, 3, 2).reshape(G, N * L, Cg * kh * kw)
            gw = torch.ops.rknpu.mm(gm, cm, ctx.precision)
        gw = gw.reshape(O, Cg, kh, kw).to(w.dtype)
    if ctx.has_bias and ctx.needs_input_grad[2]:
        gb = g.sum((0, 2, 3)).to(ctx.bias_dtype)
    return gx, gw, gb, None, None, None, None, None, None


_rknpu_conv2d.register_autograd(_conv_backward, setup_context=_conv_setup)


def conv2d(x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, act=None):
    """F.conv2d (optionally followed by ReLU: act="relu") with the convolution (forward and
    backward) on the NPU and bias / ReLU fused into its output stage. Returns None if
    unsupported."""
    if not conv_supported(x, weight, stride, padding, dilation, groups):
        return None
    if bias is not None and (bias.dim() != 1 or bias.shape[0] != weight.shape[0]):
        return None
    y = torch.ops.rknpu.conv2d(x, weight, bias, list(_pair(stride)), list(_pair(padding)), groups, act or "none",
                               _precision, "conv.w")
    dt = _out_dtype(x, weight) if bias is None else _out_dtype(x, weight, bias)
    return y.to(dt)


def conv1d(x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, act=None):
    """F.conv1d through the 2-D NPU convolution (height 1). Returns None if unsupported."""
    def one(v):
        return v[0] if isinstance(v, (tuple, list)) else v
    if isinstance(padding, str) or x.dim() not in (2, 3) or weight.dim() != 3:
        return None
    unbatched = x.dim() == 2
    x3 = x.unsqueeze(0) if unbatched else x
    # the sequence runs along the image height (width 1): the NPU's input surface stride is
    # derived from the height, and a height-1 image would make it negative
    y = conv2d(x3.unsqueeze(3), weight.unsqueeze(3), bias, (one(stride), 1), (one(padding), 0), (one(dilation), 1),
               groups, act)
    if y is None:
        return None
    y = y.squeeze(3)
    return y.squeeze(0) if unbatched else y


def conv_transpose2d(x, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1):
    """F.conv_transpose2d on the NPU, as one GEMM per group producing every kernel tap's
    contribution (N*H*W x C) @ (C x O*kh*kw), then a CPU scatter-add of the taps into the
    output (F.fold, i.e. col2im). Unlike convolving the zero-dilated input, this does no
    wasted multiply-adds. kernel == stride without padding needs no scatter at all (a pixel
    shuffle). Returns None if unsupported (dilation, or too small for the NPU)."""
    st, pd, op, dl = _pair(stride), _pair(padding), _pair(output_padding), _pair(dilation)
    if x.dim() != 4 or weight.dim() != 4 or dl != (1, 1) or isinstance(padding, str):
        return None
    N, C, H, W = x.shape
    Cin, Og, kh, kw = weight.shape
    if Cin != C or C % groups or op[0] >= max(st[0], 1) or op[1] >= max(st[1], 1):
        return None
    G, Cg = groups, C // groups
    O = Og * G
    if N * H * W * C * Og * kh * kw < threshold():
        return None
    xm = x.permute(0, 2, 3, 1).reshape(N * H * W, C)
    if G == 1:
        cols = torch.ops.rknpu.mm(xm, weight.reshape(C, O * kh * kw), _precision, "convT.w")   # NHW, O*kh*kw
    else:
        xg = xm.reshape(N * H * W, G, Cg).transpose(0, 1)                                     # G, NHW, Cg
        cols = torch.ops.rknpu.mm(xg, weight.reshape(G, Cg, Og * kh * kw), _precision)        # G, NHW, Og*kh*kw
        cols = cols.transpose(0, 1).reshape(N * H * W, O * kh * kw)
    Ho = (H - 1) * st[0] - 2 * pd[0] + kh + op[0]
    Wo = (W - 1) * st[1] - 2 * pd[1] + kw + op[1]
    if (kh, kw) == st and pd == (0, 0):
        y = cols.reshape(N, H, W, O, kh, kw).permute(0, 3, 1, 4, 2, 5).reshape(N, O, H * kh, W * kw)
        if op != (0, 0):
            y = torch.nn.functional.pad(y, (0, op[1], 0, op[0]))
    else:
        cols = cols.reshape(N, H * W, O * kh * kw).transpose(1, 2)                           # N, O*kh*kw, HW
        y = torch.nn.functional.fold(cols, (Ho, Wo), (kh, kw), padding=pd, stride=st)
    y = y.to(_out_dtype(x, weight))
    if bias is not None:
        y = y + bias.view(1, -1, 1, 1)
    return y


def release():
    """Free the NPU's DMA buffers and the packed-weight cache (re-created on demand)."""
    wcache.clear()
    if npu_available():
        default_npu().release()
