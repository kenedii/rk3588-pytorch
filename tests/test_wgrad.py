"""Convolution weight gradient: the copy-free batched path (RKNPU_DW_MODE=batched), the old copy path and the auto
dispatch against PyTorch's own, over strides, paddings, kernel shapes, bias, groups, bf16 and forced batch chunking.

    python3 tests/test_wgrad.py
"""
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import rk3588_torch as rt  # noqa: E402
from rk3588_torch import ops  # noqa: E402

torch.manual_seed(0)
ops.MIN_MACS = 0
fails = 0


def rel(a, b):
    return ((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-30)).item()


def case(name, N, C, H, O, k, st, pd, groups=1, prec="fp16", bias=False, chunk=None, tol=3e-3):
    global fails
    x = torch.randn(N, C, H, H)
    w = torch.randn(O, C // groups, k, k) * 0.2
    b = torch.randn(O) if bias else None
    res = {}
    for mode in ("copy", "batched", "auto"):
        ops.DW_MODE = mode
        ops.DW_CHUNK_ELEMS = chunk if chunk else 32 * (1 << 20) // 4
        xr, wr = x.clone().requires_grad_(), w.clone().requires_grad_()
        br = b.clone().requires_grad_() if bias else None
        with rt.npu_mode(precision=prec):
            y = F.conv2d(xr, wr, br, st, pd, 1, groups)
            gy = torch.ones_like(y) * torch.linspace(-1, 1, y.numel()).reshape(y.shape)
            y.backward(gy)
        res[mode] = (wr.grad.clone(), xr.grad.clone(), br.grad.clone() if bias else None)
    xc, wc = x.clone().requires_grad_(), w.clone().requires_grad_()
    bc = b.clone().requires_grad_() if bias else None
    y = F.conv2d(xc, wc, bc, st, pd, 1, groups)
    y.backward(gy)
    errs = {m: rel(r[0], wc.grad) for m, r in res.items()}
    ok = max(errs.values()) < tol
    # the two layouts must agree with each other much more tightly than with the CPU (same NPU arithmetic, only
    # the partial-sum grouping differs)
    ok = ok and rel(res["batched"][0], res["copy"][0]) < 2e-3
    ok = ok and all(rel(r[1], xc.grad) < tol for r in res.values())
    if bias:
        ok = ok and all(rel(r[2], bc.grad) < 1e-5 for r in res.values())
    fails += not ok
    print(f"{'PASS' if ok else 'FAIL'}  {name:42s} " + "  ".join(f"{m} {e:.1e}" for m, e in errs.items()))


case("3x3 s1 p1 N8 C32 32x32 (L=1024)", 8, 32, 32, 32, 3, 1, 1)
case("3x3 s1 p1 bias", 4, 32, 24, 48, 3, 1, 1, bias=True)
case("3x3 s2 p1", 4, 32, 32, 64, 3, 2, 1)
case("5x5 s1 p2", 4, 16, 24, 32, 5, 1, 2)
case("1x1 s1 p0", 4, 64, 32, 32, 1, 1, 0)
case("7x7 s2 p3 C3", 4, 3, 64, 64, 7, 2, 3)
case("3x3 chunked batch (nb=1 of 6)", 6, 32, 32, 32, 3, 1, 1, chunk=32 * 288)
case("3x3 chunked batch (nb=2 of 5, odd tail)", 5, 32, 32, 32, 3, 1, 1, chunk=2 * 32 * 288)
case("3x3 bf16", 4, 32, 32, 32, 3, 1, 1, prec="bf16", tol=2e-2)
case("small maps L=64 (auto -> copy path)", 8, 64, 8, 64, 3, 1, 1)
case("L=196 < O=256 (auto -> copy path)", 4, 128, 14, 256, 3, 1, 1)
case("grouped g=4", 4, 64, 24, 64, 3, 1, 1, groups=4)
ops.DW_MODE = "auto"
print("ALL PASS" if not fails else f"{fails} FAILED")
sys.exit(1 if fails else 0)
