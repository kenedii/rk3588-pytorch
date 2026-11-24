"""Conv weight gradient: copy path vs copy-free batched path, per layer shape (the im2col matrix copy is what
the batched path removes).

    python3 bench/bench_wgrad.py
"""
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rk3588_torch import ops  # noqa: E402

torch.manual_seed(0)
LAYERS = [  # (N, C, H, O, k, stride, pad)
    (8, 96, 64, 32, 3, 1, 1), (8, 32, 64, 32, 3, 1, 1), (8, 64, 32, 64, 3, 1, 1), (8, 128, 32, 64, 3, 1, 1),
    (32, 64, 56, 64, 3, 1, 1), (32, 128, 28, 128, 3, 1, 1), (32, 256, 14, 256, 3, 1, 1), (32, 512, 7, 512, 3, 1, 1),
    (16, 64, 16, 64, 3, 1, 1), (16, 128, 8, 128, 3, 1, 1), (32, 64, 56, 128, 1, 1, 0), (32, 3, 64, 64, 7, 2, 3),
]
print(f"{'layer':30} {'L':>6} {'copy ms':>9} {'batched ms':>11} {'speedup':>8}")
for (N, C, H, O, k, st, pd) in LAYERS:
    x = torch.randn(N, C, H, H)
    OH = (H + 2 * pd - k) // st + 1
    g = torch.randn(N, O, OH, OH)
    t = {}
    for mode in ("copy", "batched"):
        ops.DW_MODE = mode
        f = lambda: ops._conv_weight_grad(x, g, (O, C, k, k), (st, st), (pd, pd), 1, "fp16")  # noqa: E731
        f()
        t0 = time.perf_counter()
        for _ in range(3):
            f()
        t[mode] = (time.perf_counter() - t0) / 3 * 1e3
    print(f"N{N} {C}->{O} {H}x{H} k{k} s{st}".ljust(30) + f" {OH * OH:6d} {t['copy']:9.1f} {t['batched']:11.1f} "
          f"{t['copy'] / t['batched']:7.2f}x", flush=True)
