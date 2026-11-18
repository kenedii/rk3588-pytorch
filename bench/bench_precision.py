"""Speed and accuracy of each NPU precision mode vs CPU fp32.

    python3 bench/bench_precision.py [--threads 3]
Throughput counts 2*M*N*K "useful" FLOPs for every mode (fp16x3 does ~3x the NPU work).
Error is relative Frobenius error against an fp64 reference.
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
p.add_argument("--modes", default="fp16,fp16_out16,bf16,int8,fp16x3")
args = p.parse_args()
torch.set_num_threads(args.threads)
ops.MIN_MACS = 0
modes = args.modes.split(",")


def best(fn, reps):
    fn()
    t = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        t.append(time.perf_counter() - t0)
    return min(t)


def rel(a, ref):
    return ((a.double() - ref).norm() / ref.norm()).item()


shapes = [(256, 784, 512), (512, 512, 512), (1024, 1024, 1024), (2048, 512, 2048), (2048, 2048, 2048)]
torch.manual_seed(0)
data = {s: (torch.randn(s[0], s[1]), torch.randn(s[1], s[2])) for s in shapes}
refs = {s: data[s][0].double() @ data[s][1].double() for s in shapes}

# CPU first (warm-up so the governor is at full clock), before NPU threads exist
w = torch.randn(512, 512)
t_end = time.perf_counter() + 2.0
while time.perf_counter() < t_end:
    w @ w
cpu = {}
for s in shapes:
    a, b = data[s]
    cpu[s] = (best(lambda: a @ b, args.reps), rel(a @ b, refs[s]))

npu = default_npu()
hdr = f"{'M x K x N':>18} | {'CPU fp32':>18} | " + " | ".join(f"{m:>18}" for m in modes)
print(hdr)
print("-" * len(hdr))
for s in shapes:
    a, b = data[s]
    flop = 2 * s[0] * s[1] * s[2]
    tc, ec = cpu[s]
    row = f"{'%d x %d x %d' % s:>18} | {flop / tc / 1e9:6.0f} GF  {ec:7.1e} |"
    for m in modes:
        try:
            tn = best(lambda: ops.mm_raw(a, b, precision=m), args.reps)
            e = rel(ops.mm_raw(a, b, precision=m), refs[s])
            row += f" {flop / tn / 1e9:6.0f} GF  {e:7.1e} |"
        except Exception as ex:  # noqa: BLE001
            row += f" {'error':>18} |"
            print("   ", m, ex, file=sys.stderr)
    print(row, flush=True)
print("\ncells: effective GFLOPS (2MNK / time), relative error vs fp64")
