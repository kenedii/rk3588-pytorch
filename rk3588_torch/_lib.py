"""ctypes binding for csrc/librknpu_gemm.so."""
import ctypes
import os
import threading

_HERE = os.path.dirname(os.path.abspath(__file__))
_CANDIDATES = [
    os.environ.get("RKNPU_GEMM_LIB", ""),
    os.path.join(_HERE, "librknpu_gemm.so"),
    os.path.join(_HERE, "..", "csrc", "librknpu_gemm.so"),
]


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


def _load():
    for path in _CANDIDATES:
        if path and os.path.exists(path):
            lib = ctypes.CDLL(os.path.abspath(path))
            break
    else:
        raise OSError("librknpu_gemm.so not found; run `make -C csrc` (searched: %s)" % _CANDIDATES)
    vp, i, i64, fp = ctypes.c_void_p, ctypes.c_int, ctypes.c_int64, ctypes.c_void_p
    lib.rk_open.argtypes = [ctypes.c_char_p]
    lib.rk_open.restype = vp
    lib.rk_close.argtypes = [vp]
    lib.rk_last_error.argtypes = [vp]
    lib.rk_last_error.restype = ctypes.c_char_p
    lib.rk_set_mode.argtypes = [vp, i]
    lib.rk_set_threads.argtypes = [vp, i]
    lib.rk_set_max_m_tile.argtypes = [vp, i]
    lib.rk_gemm_f32.argtypes = [vp, i, i, i, fp, i64, i64, fp, i64, i64, fp, i64, i]
    lib.rk_gemm_f32.restype = i
    lib.rk_get_stats.argtypes = [vp, ctypes.POINTER(RkStats)]
    return lib


class NPU:
    """A handle on the NPU. Calls are serialized: the context owns one set of DMA buffers."""

    def __init__(self, device=None):
        self.lib = _load()
        self.ctx = self.lib.rk_open(device.encode() if device else None)
        if not self.ctx:
            raise OSError("could not open the rknpu DRM device (is the rknpu module loaded? "
                          "is the user in the 'video' and 'render' groups?)")
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

    def gemm_ptr(self, M, N, K, a_ptr, sa_m, sa_k, b_ptr, sb_k, sb_n, c_ptr, ldc, accumulate=False):
        with self.lock:
            r = self.lib.rk_gemm_f32(self.ctx, M, N, K, a_ptr, sa_m, sa_k, b_ptr, sb_k, sb_n,
                                     c_ptr, ldc, 1 if accumulate else 0)
            if r != 0:
                raise RuntimeError("rknpu gemm failed: " + self.lib.rk_last_error(self.ctx).decode())

    def stats(self):
        s = RkStats()
        self.lib.rk_get_stats(self.ctx, ctypes.byref(s))
        return s.as_dict()


_default = None
_default_lock = threading.Lock()


def default_npu():
    global _default
    with _default_lock:
        if _default is None:
            _default = NPU()
        return _default
