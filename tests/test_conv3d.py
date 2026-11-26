"""Conv3d on the NPU (depth taps stacked along channels of one 2-D convolution), and non-zero padding modes in
converted conv modules, forward and backward against a float64 CPU reference.

    python3 tests/test_conv3d.py
"""
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import rk3588_torch as rt  # noqa: E402
from rk3588_torch import ops  # noqa: E402

ops.MIN_MACS = 0
torch.manual_seed(0)
fails = 0


def rel(a, b):
    return ((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-30)).item()


def check(name, fn, shapes, tol=4e-3, expect_npu=True):
    """fn(*tensors) is run under npu_mode() in fp32 and in float64 outside it (CPU)."""
    global fails
    ts = [torch.randn(*s, requires_grad=True) for s in shapes]
    ops.stats(reset=True)
    with rt.npu_mode():
        y = fn(*ts)
    calls = ops.stats()["npu_calls"]
    g = torch.randn_like(y)
    gs = torch.autograd.grad(y, ts, g)
    rs = [t.detach().double().requires_grad_() for t in ts]
    yr = fn(*rs)
    grs = torch.autograd.grad(yr, rs, g.double())
    errs = [rel(y, yr)] + [rel(a, b) for a, b in zip(gs, grs)]
    ok = tuple(y.shape) == tuple(yr.shape) and max(errs) < tol and (calls > 0 or not expect_npu)
    fails += not ok
    print(f"{'PASS' if ok else 'FAIL'}  {name:<60} " + " ".join(f"{e:.1e}" for e in errs) + f"  npu calls {calls}")


for (D, H, W, k, s, p, d) in [(8, 16, 16, 3, 1, 1, 1), (9, 15, 17, (3, 3, 3), 2, 1, 1), (6, 20, 20, (1, 3, 3), 1, (0, 1, 1), 1),
                              (10, 12, 12, (3, 1, 1), 1, (1, 0, 0), 1), (12, 16, 16, 3, 1, 2, 2), (7, 16, 16, 3, 1, 0, 1),
                              (8, 14, 14, (5, 3, 3), (2, 1, 1), (2, 1, 1), 1)]:
    check(f"F.conv3d {D}x{H}x{W} k={k} stride={s} padding={p} dilation={d}",
          lambda x, w, b: F.conv3d(x, w, b, stride=s, padding=p, dilation=d),
          [(2, 16, D, H, W), (32, 16) + ((k,) * 3 if isinstance(k, int) else k), (32,)])
check("F.conv3d, no bias, unbatched input", lambda x, w: F.conv3d(x, w, padding=1), [(16, 6, 12, 12), (24, 16, 3, 3, 3)])
check("F.conv3d groups=2 stays on the CPU", lambda x, w: F.conv3d(x, w, padding=1, groups=2),
      [(2, 16, 6, 12, 12), (24, 8, 3, 3, 3)], expect_npu=False)

# modules: npu_mode() alone, and convert()
m = nn.Sequential(nn.Conv3d(8, 16, 3, padding=1), nn.ReLU(), nn.Conv3d(16, 16, 3, stride=2, padding=1))
x = torch.randn(2, 8, 6, 12, 12)
ref = m(x)
with rt.npu_mode():
    y = m(x)
ok = rel(y, ref) < 4e-3
fails += not ok
print(f"{'PASS' if ok else 'FAIL'}  nn.Conv3d under npu_mode()                                      {rel(y, ref):.1e}")
mc = rt.convert(m)
ops.stats(reset=True)
yc = mc(x)
ok = rel(yc, ref) < 4e-3 and ops.stats()["npu_calls"] > 0 and isinstance(mc[0], rt.NPUConv3d)
fails += not ok
print(f"{'PASS' if ok else 'FAIL'}  convert(): NPUConv3d without npu_mode()                         {rel(yc, ref):.1e}"
      f"  npu calls {ops.stats()['npu_calls']}")

# padding modes through convert()
for mode in ("reflect", "replicate", "circular"):
    for mk in (lambda: nn.Conv2d(16, 32, 3, padding=1, padding_mode=mode),
               lambda: nn.Conv1d(16, 32, 5, padding=2, padding_mode=mode),
               lambda: nn.Conv3d(8, 16, 3, padding=1, padding_mode=mode)):
        c = mk()
        xin = torch.randn(2, c.in_channels, *([12] * len(c.kernel_size)))
        ref = c(xin)
        cc = rt.convert(c)
        ops.stats(reset=True)
        y = cc(xin)
        ok = rel(y, ref) < 4e-3 and ops.stats()["npu_calls"] > 0
        fails += not ok
        print(f"{'PASS' if ok else 'FAIL'}  convert(): {type(c).__name__} padding_mode={mode:<10}                      "
              f"{rel(y, ref):.1e}  npu calls {ops.stats()['npu_calls']}")

print("\nALL PASS" if not fails else f"\nFAILURES: {fails}")
sys.exit(1 if fails else 0)
