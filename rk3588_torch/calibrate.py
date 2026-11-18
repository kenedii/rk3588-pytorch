"""Find the GEMM size above which the NPU beats the CPU on this machine.

The result is cached in ~/.cache/rk3588_torch/calibration.json, keyed by precision and
PyTorch thread count, and used automatically by npu_mode(). Run it explicitly with
`python -m rk3588_torch calibrate`.
"""
import json
import os
import sys
import time

import torch

_CACHE = os.path.join(os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")), "rk3588_torch",
                      "calibration.json")
_mem = None
NEVER = 1 << 62   # the NPU never won


def _key(precision):
    return f"{precision}|threads={torch.get_num_threads()}"


def _load():
    global _mem
    if _mem is None:
        try:
            with open(_CACHE) as f:
                _mem = json.load(f)
        except (OSError, ValueError):
            _mem = {}
    return _mem


def _save():
    try:
        os.makedirs(os.path.dirname(_CACHE), exist_ok=True)
        tmp = _CACHE + ".tmp"
        with open(tmp, "w") as f:
            json.dump(_mem, f, indent=1)
        os.replace(tmp, _CACHE)
    except OSError:
        pass


def cached_threshold(precision):
    entry = _load().get(_key(precision))
    return None if entry is None else int(entry["min_macs"])


def _best(fn, reps=5):
    fn()
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def calibrate(precision="fp16", verbose=False):
    """Measure CPU vs NPU on growing square GEMMs; store and return the crossover in MACs."""
    from .ops import _npu_gemm
    sizes = [32, 48, 64, 96, 128, 160, 192, 256, 320, 384, 512]
    w = torch.randn(256, 256)
    t_end = time.perf_counter() + 0.5          # wake the CPU governor up
    while time.perf_counter() < t_end:
        w @ w
    rows = []
    for n in sizes:
        a, b = torch.randn(n, n), torch.randn(n, n)
        tc = _best(lambda: a @ b)
        tn = _best(lambda: _npu_gemm(a, b, precision))
        rows.append((n, tc, tn))
        if verbose:
            print(f"  {n:4d}^3  cpu {tc * 1e6:8.0f} us   npu {tn * 1e6:8.0f} us   {'NPU' if tn < tc else 'CPU'}")
    # smallest size from which the NPU wins at every larger size tested
    cross = None
    for i, (n, tc, tn) in enumerate(rows):
        if all(r[2] < r[1] for r in rows[i:]):
            cross = n
            break
    min_macs = cross ** 3 if cross is not None else NEVER
    _load()[_key(precision)] = {"min_macs": min_macs, "crossover_n": cross, "measured": time.time(),
                                "torch_threads": torch.get_num_threads()}
    _save()
    return min_macs


def ensure_calibrated(precision):
    if cached_threshold(precision) is not None:
        return
    from ._lib import npu_available
    if not npu_available():
        return
    print(f"rk3588_torch: calibrating CPU/NPU crossover for {precision} (one-time, a few seconds)...",
          file=sys.stderr, flush=True)
    m = calibrate(precision)
    print(f"rk3588_torch: GEMMs of {m:,} multiply-adds and up will use the NPU "
          f"(cached in {_CACHE})" if m < NEVER else "rk3588_torch: the NPU never beat the CPU here; "
          "GEMMs stay on the CPU", file=sys.stderr, flush=True)
