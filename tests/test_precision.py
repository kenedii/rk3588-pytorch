"""Accuracy of every NPU precision mode against fp64 references, plus fp16 range tests.

    python3 tests/test_precision.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import rk3588_torch as rt  # noqa: E402
from rk3588_torch import ops  # noqa: E402

ops.MIN_MACS = 0
torch.manual_seed(0)
failures = 0

# expected relative error bounds per mode (fp16 inputs ~2^-11, bf16 ~2^-8, int8 ~1/127)
TOL = {"fp16": 2e-3, "fp16_out16": 3e-3, "bf16": 1e-2, "int8": 3e-2, "fp16x3": 2e-5}
MODES = [m for m in os.environ.get("MODES", "fp16,fp16_out16,bf16,int8,fp16x3").split(",") if m]


def rel_err(a, b):
    return ((a.double() - b.double()).norm() / (b.double().norm() + 1e-300)).item()


def check(name, got, ref, tol):
    global failures
    e = rel_err(got, ref)
    ok = e < tol
    failures += not ok
    print(f"{'PASS' if ok else 'FAIL'}  {name:<52} rel_err={e:.2e} (tol {tol:.0e})")


for mode in MODES:
    tol = TOL[mode]
    for (M, K, N) in [(7, 33, 17), (64, 64, 64), (300, 257, 129), (512, 1024, 512)]:
        a, b = torch.randn(M, K), torch.randn(K, N)
        ref = a.double() @ b.double()
        check(f"[{mode}] mm {M}x{K}x{N}", ops.mm_raw(a, b, precision=mode), ref, tol)
        check(f"[{mode}] mm {M}x{K}x{N} transposed A,B",
              ops.mm_raw(a.t().contiguous().t(), b.t().contiguous().t(), precision=mode), ref, tol)
    # gradients through a linear layer
    x = torch.randn(96, 300, requires_grad=True)
    w = torch.randn(200, 300, requires_grad=True)
    prev = rt.set_precision(mode)
    y = ops.linear(x, w)
    g = torch.randn_like(y)
    y.backward(g)
    rt.set_precision(prev)
    xd, wd = x.detach().double().requires_grad_(), w.detach().double().requires_grad_()
    torch.nn.functional.linear(xd, wd).backward(g.double())
    check(f"[{mode}] linear dX", x.grad, xd.grad, tol)
    check(f"[{mode}] linear dW", w.grad, wd.grad, tol)

# range: values far outside fp16 ([6e-8, 65504]) must still work in fp16 mode (auto-scaling)
for scale, label in [(1e-7, "tiny (1e-7)"), (1e5, "huge (1e5)"), (1e-30, "denormal-ish (1e-30)")]:
    a, b = torch.randn(128, 256) * scale, torch.randn(256, 64)
    for mode in [m for m in ("fp16", "bf16", "fp16x3") if m in MODES]:
        check(f"[{mode}] range {label}", ops.mm_raw(a, b, precision=mode), a.double() @ b.double(), TOL[mode])
# mixed magnitudes inside one matrix: per-row int8 scaling handles row-to-row spread
if "int8" in MODES:
    a = torch.randn(128, 256) * torch.logspace(-3, 3, 128).unsqueeze(1)
    b = torch.randn(256, 64)
    check("[int8] rows spanning 6 decades", ops.mm_raw(a, b, precision="int8"), a.double() @ b.double(), TOL["int8"])

print("ALL PASS" if failures == 0 else f"{failures} FAILURES")
sys.exit(1 if failures else 0)
