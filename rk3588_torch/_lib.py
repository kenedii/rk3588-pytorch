"""ctypes binding for librknpu_gemm.so, plus NPU availability detection."""
import ctypes
import os
import threading
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_CANDIDATES = [
    os.environ.get("RKNPU_GEMM_LIB", ""),
    os.path.join(_HERE, "librknpu_gemm.so"),
    os.path.join(_HERE, "..", "csrc", "librknpu_gemm.so"),
]

# Fraction of wall time the NPU path may be busy (NPU job plus the CPU packing around it), 0 < d <= 1.
# Below 1, every NPU call is followed by a pause of busy * (1/d - 1): the board draws less on average and
# gets breathing room between bursts, for supplies that brown out under sustained all-out load
# (RKNPU_DUTY, set_duty()). Pair it with fewer threads (RKNPU_THREADS, torch.set_num_threads).
_duty = min(max(float(os.environ.get("RKNPU_DUTY", "1.0")), 0.05), 1.0)


def set_duty(d):
    """Limit the NPU duty cycle to d (0.05 .. 1.0); returns the previous value."""
    global _duty
    prev, _duty = _duty, min(max(float(d), 0.05), 1.0)
    return prev


def get_duty():
    return _duty


def _throttle(busy_s):
    if _duty < 1.0 and busy_s > 0:
        time.sleep(busy_s * (1.0 / _duty - 1.0))


# precision names -> rk_gemm_ex codes (see csrc/rknpu_gemm.h)
PRECISIONS = {"fp16": 0, "fp16_out16": 1, "int8": 2, "fp16x3": 3, "bf16": 4}


class RkStats(ctypes.Structure):
    _fields_ = [
        ("pack_us", ctypes.c_double),
        ("submit_us", ctypes.c_double),
        ("hw_us", ctypes.c_double),
        ("unpack_us", ctypes.c_double),
        ("total_us", ctypes.c_double),
        ("tasks", ctypes.c_int),
        ("submits", ctypes.c_int),
        ("m_tile", ctypes.c_int),
        ("k_tile", ctypes.c_int),
        ("n_tile", ctypes.c_int),
        ("sync_us", ctypes.c_double),
        ("cache_hits", ctypes.c_int),
        ("waves", ctypes.c_int),
    ]

    def as_dict(self):
        return {f: getattr(self, f) for f, _ in self._fields_}


class RkInfo(ctypes.Structure):
    _fields_ = [
        ("driver_version", ctypes.c_int),
        ("hw_version", ctypes.c_uint32),
        ("iommu", ctypes.c_int),
        ("freq_hz", ctypes.c_uint32),
    ]


def library_path():
    for path in _CANDIDATES:
        if path and os.path.exists(path):
            return os.path.abspath(path)
    return None


def _try_build():
    """In a source checkout on an aarch64 board, build the library on first use."""
    import platform
    import subprocess
    csrc = os.path.join(_HERE, "..", "csrc")
    if platform.machine() not in ("aarch64", "arm64") or not os.path.exists(os.path.join(csrc, "Makefile")):
        return
    try:
        subprocess.run(["make", "-C", csrc, "librknpu_gemm.so"], check=True, capture_output=True, timeout=300)
    except Exception:  # noqa: BLE001
        pass


def _load():
    path = library_path()
    if path is None:
        _try_build()
        path = library_path()
    if path is None:
        raise OSError("librknpu_gemm.so not found (build it with `pip install .` or `make -C csrc`); "
                      "searched: %s" % [p for p in _CANDIDATES if p])
    lib = ctypes.CDLL(path)
    vp, i, i64, fp = ctypes.c_void_p, ctypes.c_int, ctypes.c_int64, ctypes.c_void_p
    lib.rk_open.argtypes = [ctypes.c_char_p]
    lib.rk_open.restype = vp
    lib.rk_close.argtypes = [vp]
    lib.rk_last_error.argtypes = [vp]
    lib.rk_last_error.restype = ctypes.c_char_p
    lib.rk_set_mode.argtypes = [vp, i]
    lib.rk_set_threads.argtypes = [vp, i]
    lib.rk_set_max_m_tile.argtypes = [vp, i]
    lib.rk_get_stats.argtypes = [vp, ctypes.POINTER(RkStats)]
    lib.rk_eltwise.argtypes = [vp, i, i, vp, vp, vp, vp, vp, vp, ctypes.c_long, ctypes.c_long, ctypes.c_float]
    lib.rk_eltwise.restype = i
    lib.rk_dwconv.argtypes = [vp, i, vp, vp, vp, vp] + [i] * 12
    lib.rk_dwconv.restype = i
    lib.rk_batchnorm.argtypes = [vp, i] + [vp] * 12 + [i, i, i, ctypes.c_float]
    lib.rk_batchnorm.restype = i
    lib.rk_gemm_batched.argtypes = [vp, i, i, i, i, i, fp, i64, i64, i64, fp, i64, i64, i64, fp, i64, i64, i]
    lib.rk_gemm_batched.restype = i
    lib.rk_release.argtypes = [vp]
    lib.rk_set_max_buf_mb.argtypes = [vp, i]
    lib.rk_conv2d.argtypes = [vp, i, i, i, i, i, fp, i, i, i, fp, i, i, i, i, i, i, fp, i]
    lib.rk_conv2d.restype = i
    lib.rk_conv2d_grouped.argtypes = [vp, i, i, i, i, i, fp, i, i, i, fp, i, i, i, i, i, i, fp, i, i]
    lib.rk_conv2d_grouped.restype = i
    lib.rk_set_epilogue.argtypes = [vp, fp, i]
    lib.rk_set_bgather.argtypes = [vp, ctypes.POINTER(ctypes.c_int)]
    lib.rk_set_wcache_key.argtypes = [vp, ctypes.c_uint64]
    lib.rk_wcache_drop.argtypes = [vp, ctypes.c_uint64]
    lib.rk_wcache_clear.argtypes = [vp]
    lib.rk_wcache_bytes.argtypes = [vp]
    lib.rk_wcache_bytes.restype = ctypes.c_size_t
    lib.rk_wcache_entries.argtypes = [vp]
    lib.rk_wcache_entries.restype = i
    lib.rk_set_wcache_mb.argtypes = [vp, i]
    lib.rk_buffer_bytes.argtypes = [vp]
    lib.rk_buffer_bytes.restype = ctypes.c_size_t
    lib.rk_query.argtypes = [vp, ctypes.POINTER(RkInfo)]
    return lib


class NPU:
    """A handle on the NPU. Calls are serialized: the context owns one set of DMA buffers."""

    def __init__(self, device=None):
        self.lib = _load()
        self.ctx = self.lib.rk_open(device.encode() if device else None)
        if not self.ctx:
            raise OSError("could not open the rknpu DRM device (is the rknpu module loaded? "
                          "is this user in the 'video' and 'render' groups?) "
                          "Run `python -m rk3588_torch doctor` for details.")
        self.lock = threading.Lock()

    def close(self):
        if self.ctx:
            self.lib.rk_close(self.ctx)
            self.ctx = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def eltwise(self, op, bf16, x, g, w, y, aux, dw, rows, D, eps=0.0):
        """Fused CPU element-wise kernel (see rk_eltwise); pointer arguments are ints (0 = NULL)."""
        with self.lock:
            return self.lib.rk_eltwise(self.ctx, int(op), int(bf16), x or None, g or None, w or None, y or None,
                                       aux or None, dw or None, int(rows), int(D), float(eps))

    def batchnorm(self, op, x, g, gamma, beta, mean_in, rstd_in, y, mean, rstd, var, dgamma, dbeta, N, C, HW, eps):
        """Training-mode BatchNorm kernel (see rk_batchnorm); pointer arguments are ints (0 = NULL)."""
        with self.lock:
            return self.lib.rk_batchnorm(self.ctx, int(op), *(v or None for v in (
                x, g, gamma, beta, mean_in, rstd_in, y, mean, rstd, var, dgamma, dbeta)),
                int(N), int(C), int(HW), float(eps))

    def dwconv(self, op, x, w, dy, out, N, C, H, W, KH, KW, sy, sx, pt, pl, OH, OW):
        """Depthwise conv kernel (see rk_dwconv); pointer arguments are ints (0 = NULL)."""
        with self.lock:
            return self.lib.rk_dwconv(self.ctx, int(op), x or None, w or None, dy or None, out,
                                      *(int(v) for v in (N, C, H, W, KH, KW, sy, sx, pt, pl, OH, OW)))

    def set_mode(self, mode):
        self.lib.rk_set_mode(self.ctx, int(mode))

    def set_threads(self, n):
        self.lib.rk_set_threads(self.ctx, int(n))

    def set_max_m_tile(self, m):
        self.lib.rk_set_max_m_tile(self.ctx, int(m))

    def _call(self, fn, args, bias_ptr, relu, wkey, bgather=None):
        with self.lock:
            if bgather is not None:
                self._bg = (ctypes.c_int * 11)(*bgather)
                self.lib.rk_set_bgather(self.ctx, self._bg)
            if bias_ptr or relu:
                self.lib.rk_set_epilogue(self.ctx, bias_ptr or None, 1 if relu else 0)
            if wkey:
                self.lib.rk_set_wcache_key(self.ctx, wkey)
            t0 = time.perf_counter()
            try:
                r = fn(self.ctx, *args)
            finally:
                if bgather is not None:
                    self.lib.rk_set_bgather(self.ctx, None)
                if bias_ptr or relu:
                    self.lib.rk_set_epilogue(self.ctx, None, 0)
                if wkey:
                    self.lib.rk_set_wcache_key(self.ctx, 0)
            if r != 0:
                raise RuntimeError(f"rknpu {fn.__name__[3:]} failed: " + self.lib.rk_last_error(self.ctx).decode())
            _throttle(time.perf_counter() - t0)

    def gemm(self, batch, M, N, K, a_ptr, sa_b, sa_m, sa_k, b_ptr, sb_b, sb_k, sb_n, c_ptr, sc_b, ldc,
             accumulate=False, precision="fp16", bias_ptr=0, relu=False, wkey=0, bgather=None):
        """C = A @ B (+ bias[n]) (ReLU); wkey != 0 caches B's packed form under that key. bgather =
        (C,H,W,kh,kw,sy,sx,pt,pl,oh,ow): b_ptr is a fp32 NCHW tensor x (sb_b = C*H*W) and B[b] is the im2col of
        image b, (K = oh*ow) x (N = C*kh*kw), gathered while packing (fp16 / bf16)."""
        self._call(self.lib.rk_gemm_batched, (PRECISIONS[precision], batch, M, N, K, a_ptr, sa_b, sa_m, sa_k,
                                              b_ptr, sb_b, sb_k, sb_n, c_ptr, sc_b, ldc, 1 if accumulate else 0),
                   bias_ptr, relu, wkey, bgather)

    def conv2d(self, N, C, H, W, x_ptr, O, kh, kw, w_ptr, sy, sx, pt, pl, oh, ow, y_ptr,
               accumulate=False, precision="fp16", groups=1, bias_ptr=0, relu=False, wkey=0):
        code = PRECISIONS["bf16" if precision == "bf16" else "fp16"]
        self._call(self.lib.rk_conv2d_grouped, (code, N, C, H, W, x_ptr, O, kh, kw, w_ptr, sy, sx, pt, pl, oh, ow,
                                                y_ptr, 1 if accumulate else 0, groups), bias_ptr, relu, wkey)

    def wcache_drop(self, key):
        with self.lock:
            self.lib.rk_wcache_drop(self.ctx, key)

    def wcache_clear(self):
        with self.lock:
            self.lib.rk_wcache_clear(self.ctx)

    def wcache_info(self):
        return {"bytes": int(self.lib.rk_wcache_bytes(self.ctx)), "entries": int(self.lib.rk_wcache_entries(self.ctx))}

    def set_wcache_mb(self, mb):
        from . import wcache
        wcache.CAP_MB = int(mb)
        with self.lock:
            self.lib.rk_set_wcache_mb(self.ctx, int(mb))

    # backwards compatible single-GEMM entry point
    def gemm_ptr(self, M, N, K, a_ptr, sa_m, sa_k, b_ptr, sb_k, sb_n, c_ptr, ldc, accumulate=False,
                 precision="fp16"):
        self.gemm(1, M, N, K, a_ptr, 0, sa_m, sa_k, b_ptr, 0, sb_k, sb_n, c_ptr, 0, ldc, accumulate, precision)

    def stats(self):
        s = RkStats()
        self.lib.rk_get_stats(self.ctx, ctypes.byref(s))
        return s.as_dict()

    def info(self):
        s = RkInfo()
        self.lib.rk_query(self.ctx, ctypes.byref(s))
        return {"driver_version": s.driver_version, "hw_version": s.hw_version, "iommu": s.iommu,
                "freq_hz": s.freq_hz}

    def set_max_buf_mb(self, mb):
        self.lib.rk_set_max_buf_mb(self.ctx, int(mb))

    def release(self):
        with self.lock:
            self.lib.rk_release(self.ctx)

    def buffer_bytes(self):
        return int(self.lib.rk_buffer_bytes(self.ctx))


_default = None
_default_error = None
_default_lock = threading.Lock()


def default_npu():
    """The shared NPU handle (raises if the NPU is unavailable)."""
    global _default, _default_error
    with _default_lock:
        if _default is None:
            if _default_error is not None:
                raise _default_error
            try:
                _default = NPU()
            except OSError as e:
                _default_error = e
                raise
        return _default


def npu_available():
    """True if the NPU can be opened (never raises)."""
    try:
        default_npu()
        return True
    except OSError:
        return False


def npu_error():
    """Why the NPU is unavailable, or None."""
    npu_available()
    return _default_error
