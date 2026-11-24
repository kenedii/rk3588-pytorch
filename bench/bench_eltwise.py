"""CPU cost of the non-GEMM ops of a training step (what stays on the A76 cores under npu_mode()).

    python3 bench/bench_eltwise.py [--n 3145728] [--threads 3]

Forward + backward of each op on a (rows x 1536) fp32 tensor, PyTorch's own kernels vs rk3588-pytorch's
fused NEON kernels (rk3588_torch.eltwise) where those exist.
"""
import argparse
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

ap = argparse.ArgumentParser()
ap.add_argument("--rows", type=int, default=2048)
ap.add_argument("--cols", type=int, default=1536)
ap.add_argument("--threads", type=int, default=3)
args = ap.parse_args()
torch.set_num_threads(args.threads)
try:
    from rk3588_torch import eltwise
except Exception:  # noqa: BLE001
    eltwise = None


def t(fn, reps=5):
    fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return sorted(ts)[len(ts) // 2] * 1e3


x = torch.randn(args.rows, args.cols, requires_grad=True)
g = torch.randn(args.rows, args.cols)
w = torch.randn(args.cols, requires_grad=True)
b = torch.randn(args.cols, requires_grad=True)


def fb(f):
    def run():
        y = f(x)
        y.backward(g)
    return run


ops = {
    "gelu (erf)": lambda v: F.gelu(v),
    "gelu (tanh)": lambda v: F.gelu(v, approximate="tanh"),
    "silu": F.silu,
    "relu": F.relu,
    "layer_norm": lambda v: F.layer_norm(v, (args.cols,), w, b),
    "rms_norm": lambda v: F.rms_norm(v, (args.cols,), w),
    "softmax": lambda v: F.softmax(v, -1),
    "swiglu x*silu(x)": lambda v: v * F.silu(v),
    "add + mul": lambda v: (v + 1.0) * 0.5,
}
print(f"{args.rows}x{args.cols} fp32, {args.threads} threads, forward+backward ms")
print(f"{'op':<20}{'torch':>9}{'rk3588':>9}")
for name, f in ops.items():
    tt = t(fb(f))
    tr = ""
    if eltwise is not None and eltwise.supports(name):
        tr = f"{t(fb(eltwise.get(name))):9.2f}"
    print(f"{name:<20}{tt:9.2f}{tr:>9}", flush=True)
