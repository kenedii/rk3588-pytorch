"""Depthwise convolution kernels (rk3588_torch/dwconv.py, csrc/rknpu_dwconv.c) vs PyTorch: forward, input and
weight gradients over kernel sizes, strides, paddings ("same", asymmetric), bias, bf16, conv1d, and routing.

    python3 tests/test_dwconv.py
"""
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import rk3588_torch as rt  # noqa: E402
from rk3588_torch import dwconv, ops  # noqa: E402

torch.manual_seed(0)
dwconv.MIN_ELEMS = 0
fails = 0


def rel(a, b):
    return ((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-30)).item()


def check(name, N, C, H, W, k, stride, padding, bias=True, dtype=torch.float32, tol=2e-5):
    global fails
    x = torch.randn(N, C, H, W, dtype=dtype)
    w = torch.randn(C, 1, *k, dtype=dtype)
    b = torch.randn(C, dtype=dtype) if bias else None
    xr, wr = x.clone().requires_grad_(), w.clone().requires_grad_()
    br = b.clone().requires_grad_() if bias else None
    xm, wm = x.clone().requires_grad_(), w.clone().requires_grad_()
    bm = b.clone().requires_grad_() if bias else None
    yr = F.conv2d(xr, wr, br, stride, padding, 1, C)
    pads = ops._conv_pads(padding, k[0], k[1], (1, 1), ops._pair(stride))
    ym = dwconv.conv2d(xm, wm, bm, ops._pair(stride), pads)
    g = torch.randn_like(yr)
    yr.backward(g)
    ym.backward(g)
    errs = [rel(ym, yr), rel(xm.grad, xr.grad), rel(wm.grad, wr.grad)] + ([rel(bm.grad, br.grad)] if bias else [])
    ok = ym.shape == yr.shape and max(errs) < tol
    fails += not ok
    print(f"{'PASS' if ok else 'FAIL'}  {name:34s} max rel {max(errs):.1e}")


check("3x3 s1 p1", 4, 32, 17, 19, (3, 3), 1, 1)
check("3x3 s2 p1", 4, 32, 17, 19, (3, 3), 2, 1)
check("5x5 s1 p2 no bias", 2, 16, 23, 23, (5, 5), 1, 2, bias=False)
check("5x5 s2 p2", 2, 16, 24, 24, (5, 5), 2, 2)
check("7x7 s1 'same'", 2, 8, 20, 21, (7, 7), 1, "same")
check("3x5 s(1,2) p(1,0)", 3, 12, 15, 30, (3, 5), (1, 2), (1, 0))
check("3x3 s1 p0 'valid'", 2, 8, 9, 9, (3, 3), 1, "valid")
check("1x1 s1", 2, 8, 9, 9, (1, 1), 1, 0)
check("3x3 s1 p1 tiny 2x2 map", 4, 8, 2, 2, (3, 3), 1, 1)
check("3x3 s2 p1 bf16", 4, 32, 16, 16, (3, 3), 2, 1, dtype=torch.bfloat16, tol=2e-2)

# routing: npu_mode F.conv2d, convert()ed nn.Conv2d, and depthwise conv1d
m = nn.Sequential(nn.Conv2d(16, 16, 3, padding=1, groups=16), nn.ReLU())
x = torch.randn(4, 16, 32, 32)
ref = m(x)
calls = []
orig = dwconv._call
dwconv._call = lambda *a: (calls.append(a[0]), orig(*a))[1]
with rt.npu_mode():
    out = m(x)
mc = rt.convert(m)
out2 = mc(x)
c1 = nn.Conv1d(24, 24, 5, padding=2, groups=24)
x1 = torch.randn(4, 24, 400)
with rt.npu_mode():
    o1 = c1(x1)
dwconv._call = orig
ok = rel(out, ref) < 2e-5 and rel(out2, ref) < 2e-5 and rel(o1, c1(x1)) < 2e-5 and len(calls) == 3
fails += not ok
print(f"{'PASS' if ok else 'FAIL'}  routing (npu_mode, convert, conv1d): {len(calls)} kernel calls")

# speed
for (N, C, H) in ((32, 256, 32), (32, 512, 16)):
    x = torch.randn(N, C, H, H, requires_grad=True)
    w = torch.randn(C, 1, 3, 3, requires_grad=True)
    g = torch.randn(N, C, H, H)
    def ref_step():
        F.conv2d(x, w, None, 1, 1, 1, C).backward(g)
    def our_step():
        dwconv.conv2d(x, w, None, (1, 1), (1, 1, 1, 1)).backward(g)
    ts = []
    for f in (ref_step, our_step):
        f()
        t = time.perf_counter()
        for _ in range(3):
            f()
        ts.append((time.perf_counter() - t) / 3 * 1e3)
    print(f"      {N}x{C}x{H}x{H} 3x3 fwd+bwd: PyTorch {ts[0]:.1f} ms, rk_dwconv {ts[1]:.1f} ms ({ts[0] / ts[1]:.1f}x)")
print("ALL PASS" if not fails else f"{fails} FAILED")
sys.exit(1 if fails else 0)
