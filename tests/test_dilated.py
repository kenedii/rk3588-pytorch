"""Dilated and "same"/asymmetric-padded convolutions on the NPU (space-to-batch), forward and backward.

    python3 tests/test_dilated.py
"""
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import rk3588_torch as rt  # noqa: E402
from rk3588_torch import ops  # noqa: E402

ops.MIN_MACS = 0
torch.manual_seed(0)
fails = 0


def rel(a, b):
    return ((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-30)).item()


def check(name, fn_npu, fn_ref, shapes, tol=4e-3):
    global fails
    ts = [torch.randn(*s, requires_grad=True) for s in shapes]
    ops.stats(reset=True)
    y = fn_npu(*ts)
    calls = ops.stats()["npu_calls"]
    if y is None:
        fails += 1
        print(f"FAIL  {name:<58} not handled (None)")
        return
    g = torch.randn_like(y)
    gs = torch.autograd.grad(y, ts, g)
    rs = [t.detach().double().requires_grad_() for t in ts]
    yr = fn_ref(*rs)
    grs = torch.autograd.grad(yr, rs, g.double())
    errs = [rel(y, yr)] + [rel(a, b) for a, b in zip(gs, grs)]
    ok = tuple(y.shape) == tuple(yr.shape) and max(errs) < tol and calls > 0
    fails += not ok
    print(f"{'PASS' if ok else 'FAIL'}  {name:<58} " + " ".join(f"{e:.1e}" for e in errs) + f"  npu calls {calls}")


with rt.npu_mode():
    for (L, k, d, pad) in [(500, 3, 2, 2), (777, 3, 4, 4), (1000, 5, 8, 16), (300, 3, 3, "same"), (256, 2, 2, "same"),
                           (400, 3, 16, 0), (999, 7, 3, "valid")]:
        check(f"conv1d L={L} k={k} dilation={d} padding={pad}",
              lambda x, w, b: F.conv1d(x, w, b, padding=pad, dilation=d),
              lambda x, w, b: F.conv1d(x, w, b, padding=pad, dilation=d), [(4, 48, L), (64, 48, k), (64,)])
    for (H, W, k, d, pad, s) in [(33, 40, 3, 2, 2, 1), (64, 64, 3, 4, 4, 1), (31, 29, 3, (2, 3), (2, 3), 1),
                                 (40, 40, 3, 2, 0, 2), (48, 48, 3, 2, 2, 4), (28, 28, 4, 1, "same", 1),
                                 (30, 30, 3, 6, "same", 1)]:
        check(f"conv2d {H}x{W} k={k} dilation={d} padding={pad} stride={s}",
              lambda x, w, b: F.conv2d(x, w, b, stride=s, padding=pad, dilation=d),
              lambda x, w, b: F.conv2d(x, w, b, stride=s, padding=pad, dilation=d),
              [(2, 64, H, W), (96, 64, k, k), (96,)])
    check("grouped dilated conv2d (groups=4)", lambda x, w: F.conv2d(x, w, padding=2, dilation=2, groups=4),
          lambda x, w: F.conv2d(x, w, padding=2, dilation=2, groups=4), [(2, 128, 32, 32), (128, 32, 3, 3)])
    m = rt.convert(torch.nn.Sequential(torch.nn.Conv1d(32, 64, 3, dilation=4, padding="same"), torch.nn.ReLU()))
    x = torch.randn(2, 32, 600)
    ops.stats(reset=True)
    y = m(x)
    ref = F.relu(F.conv1d(x.double(), m[0].weight.double(), m[0].bias.double(), padding="same", dilation=4))
    ok = rel(y, ref) < 4e-3 and ops.stats()["npu_calls"] > 0
    fails += not ok
    print(f"{'PASS' if ok else 'FAIL'}  {'convert(): Conv1d(dilation=4, padding=same) + fused ReLU':<58} {rel(y, ref):.1e}")

print("ALL PASS" if not fails else f"{fails} FAILED")
sys.exit(1 if fails else 0)
