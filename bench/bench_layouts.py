"""Where the time of one NPU GEMM goes, for the three GEMMs of a Linear layer's training step.

    python3 bench/bench_layouts.py [--precision fp16] [--reps 5]

For x (M x K) and W (N x K):  fwd  Y = x @ W^T,  dX = dY @ W,  dW = dY^T @ x  (A read through a transpose).
Prints total time, GFLOPS and the library's own breakdown (pack = fp32 -> fp16 conversion and layout,
sync = cache maintenance, submit = NPU run incl. ioctl, unpack = fp32 output copy / reduction).
"""
import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rk3588_torch._lib import default_npu  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--precision", default="fp16")
ap.add_argument("--reps", type=int, default=5)
ap.add_argument("--shapes", default="2048x384x1152,2048x384x384,2048x384x1536,2048x1536x384,2048x384x4096,"
                                   "256x784x2048,256x2048x2048,1024x1024x1024,2048x2048x2048")
args = ap.parse_args()
torch.set_num_threads(3)
npu = default_npu()


def run(M, N, K, A, sa_m, sa_k, B, sb_k, sb_n):
    C = torch.empty(M, N)
    best, st = 1e9, None
    for _ in range(args.reps + 1):
        t0 = time.perf_counter()
        npu.gemm(1, M, N, K, A.data_ptr(), 0, sa_m, sa_k, B.data_ptr(), 0, sb_k, sb_n, C.data_ptr(), 0, N, False,
                 args.precision)
        dt = time.perf_counter() - t0
        if dt < best:
            best, st = dt, npu.stats()
    return best, st


print(f"{'shape (MxKxN)':<16}{'gemm':<5}{'ms':>8}{'GFLOPS':>8}{'pack':>8}{'sync':>7}{'submit':>8}{'hw':>7}{'unpack':>8}"
      f"{'tasks':>6}{'waves':>6}")
for s in args.shapes.split(","):
    M, K, N = map(int, s.split("x"))
    x, w, g = torch.randn(M, K), torch.randn(N, K), torch.randn(M, N)
    cases = [("fwd", M, N, K, x, K, 1, w, 1, K),          # x @ W^T : B(k, n) = w[n, k]
             ("dX", M, K, N, g, N, 1, w, K, 1),           # g @ W  : B(k, n) = w[k, n]
             ("dW", N, K, M, g, 1, N, x, K, 1)]           # g^T @ x: A(m, k) = g[k, m]
    for name, m, n, k, A, sam, sak, B, sbk, sbn in cases:
        t, st = run(m, n, k, A, sam, sak, B, sbk, sbn)
        print(f"{s:<16}{name:<5}{t * 1e3:>8.2f}{2 * m * n * k / t / 1e9:>8.0f}{st['pack_us'] / 1e3:>8.2f}"
              f"{st['sync_us'] / 1e3:>7.2f}{st['submit_us'] / 1e3:>8.2f}{st['hw_us'] / 1e3:>7.2f}"
              f"{st['unpack_us'] / 1e3:>8.2f}{st['tasks']:>6}{st['waves']:>6}", flush=True)
