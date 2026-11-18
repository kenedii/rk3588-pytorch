"""ctypes binding for librknpu_gemm.so, plus NPU availability detection."""
import ctypes
import os
import threading

_HERE = os.path.dirname(os.path.abspath(__file__))
_CANDIDATES = [
    os.environ.get("RKNPU_GEMM_LIB", ""),
    os.path.join(_HERE, "librknpu_gemm.so"),
    os.path.join(_HERE, "..", "csrc", "librknpu_gemm.so"),
]

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
    lib.rk_gemm_batched.argtypes = [vp, i, i, i, i, i, fp, i64, i64, i64, fp, i64, i64, i64, fp, i64, i64, i]
    lib.rk_gemm_batched.restype = i
    lib.rk_release.argtypes = [vp]
    lib.rk_set_max_buf_mb.argtypes = [vp, i]
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

    def set_mode(self, mode):
        self.lib.rk_set_mode(self.ctx, int(mode))

    def set_threads(self, n):
        self.lib.rk_set_threads(self.ctx, int(n))

    def set_max_m_tile(self, m):
        self.lib.rk_set_max_m_tile(self.ctx, int(m))

    def gemm(self, batch, M, N, K, a_ptr, sa_b, sa_m, sa_k, b_ptr, sb_b, sb_k, sb_n, c_ptr, sc_b, ldc,
             accumulate=False, precision="fp16"):
        code = PRECISIONS[precision]
        with self.lock:
            r = self.lib.rk_gemm_batched(self.ctx, code, batch, M, N, K, a_ptr, sa_b, sa_m, sa_k,
                                         b_ptr, sb_b, sb_k, sb_n, c_ptr, sc_b, ldc, 1 if accumulate else 0)
            if r != 0:
                raise RuntimeError("rknpu gemm failed: " + self.lib.rk_last_error(self.ctx).decode())

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
