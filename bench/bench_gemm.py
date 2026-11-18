"""NPU vs CPU GEMM throughput on the board.

    python3 bench/bench_gemm.py [--threads 3]
CPU numbers are torch fp32 matmul with the given thread count (the board resets under
4+ busy big cores on a weak charger, so 3 is the default).
"""
import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rk3588_torch import ops  # noqa: E402
from rk3588_torch._lib import default_npu  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--threads", type=int, default=3)
p.add_argument("--reps", type=int, default=5)
p.add_argument("--modes", default="1,2")
args = p.parse_args()
torch.set_num_threads(args.threads)
ops.MIN_MACS = 0


def best(fn, reps):
    fn()
    t = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        t.append(time.perf_counter() - t0)
    return min(t)


shapes = [(64, 64, 64), (128, 256, 128), (256, 256, 256), (512, 512, 512), (64, 784, 512), (256, 784, 512),
          (1024, 1024, 1024), (2048, 512, 2048), (512, 4096, 512), (2048, 2048, 2048)]
# CPU first, before the NPU library starts any threads, so the baseline is not disturbed.
# Warm up for ~2 s so the ondemand governor has the cores at full clock.
w = torch.randn(512, 512)
t_end = time.perf_counter() + 2.0
while time.perf_counter() < t_end:
    w @ w
cpu = {}
for M, K, N in shapes:
    a, b = torch.randn(M, K), torch.randn(K, N)
    cpu[(M, K, N)] = best(lambda: a @ b, args.reps)
npu = default_npu()
print(f"{'M':>5} {'K':>5} {'N':>5} | {'CPU fp32':>10} | " + " | ".join(f"NPU mode{m:>2} (pack/npu/unpack us)" for m in args.modes.split(",")))
for M, K, N in shapes:
    a, b = torch.randn(M, K), torch.randn(K, N)
    flop = 2 * M * N * K
    tc = cpu[(M, K, N)]
    row = f"{M:5} {K:5} {N:5} | {flop / tc / 1e9:7.1f} GF |"
    for mode in args.modes.split(","):
        npu.set_mode(int(mode))
        tn = best(lambda: ops.mm_raw(a, b), args.reps)
        s = npu.stats()
        row += f" {flop / tn / 1e9:7.1f} GF ({s['pack_us'] + s['sync_us']:.0f}/{s['submit_us']:.0f}/{s['unpack_us']:.0f}) x{tc / tn:4.1f} |"
    print(row, flush=True)
