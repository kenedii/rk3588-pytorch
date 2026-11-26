"""Fused CPU BatchNorm (training mode): outputs, running statistics and gradients against PyTorch's own, plus timing.

    python3 tests/test_bnorm.py
"""
import copy
import os
import sys
import time

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import rk3588_torch as rt  # noqa: E402
from rk3588_torch import bnorm  # noqa: E402

assert rt.npu_available(), rt.npu_error()
bnorm.MIN_ELEMS = 0
torch.manual_seed(0)
fails = 0


def rel(a, b):
    return ((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-30)).item()


def run(name, mk, shape, mode, tol=2e-5):
    """Same module and input through PyTorch and the fused kernel (convert() or npu_mode()); compare everything."""
    global fails
    ref = mk()
    x = (torch.randn(*shape) * 2.5 + 1.0).requires_grad_()
    xr = x.detach().clone().requires_grad_()
    mod = copy.deepcopy(ref)
    bnorm.ENABLED = False
    yr = ref(xr)
    g = torch.randn_like(yr)
    yr.backward(g)
    bnorm.ENABLED = True
    if mode == "convert":
        mod = rt.convert(mod)
        y = mod(x)
    else:
        with rt.npu_mode():
            y = mod(x)
    y.backward(g)
    errs = {"y": rel(y, yr), "dx": rel(x.grad, xr.grad)}
    if ref.weight is not None:
        errs["dw"], errs["db"] = rel(mod.weight.grad, ref.weight.grad), rel(mod.bias.grad, ref.bias.grad)
    if ref.running_mean is not None:
        errs["rm"], errs["rv"] = rel(mod.running_mean, ref.running_mean), rel(mod.running_var, ref.running_var)
        errs["nbt"] = float(mod.num_batches_tracked != ref.num_batches_tracked)
    ok = max(errs.values()) < tol
    fails += not ok
    print(f"{'PASS' if ok else 'FAIL'}  {name:<46} {mode:<8} max err {max(errs.values()):.1e}")


for mode in ("convert", "npu_mode"):
    run("BatchNorm2d 16x64x32x32", lambda: nn.BatchNorm2d(64), (16, 64, 32, 32), mode)
    run("BatchNorm2d odd 8x33x17x13", lambda: nn.BatchNorm2d(33), (8, 33, 17, 13), mode)
    run("BatchNorm2d affine=False", lambda: nn.BatchNorm2d(48, affine=False), (8, 48, 16, 16), mode)
    run("BatchNorm2d track_running_stats=False", lambda: nn.BatchNorm2d(32, track_running_stats=False), (8, 32, 16, 16), mode)
    run("BatchNorm2d momentum=None (cumulative)", lambda: nn.BatchNorm2d(32, momentum=None), (8, 32, 16, 16), mode)
    run("BatchNorm1d 32x128x100 (N,C,L)", lambda: nn.BatchNorm1d(128), (32, 128, 100), mode)
    run("BatchNorm3d 4x16x6x16x16", lambda: nn.BatchNorm3d(16), (4, 16, 6, 16, 16), mode)

# eval mode and channels_last stay on PyTorch and still agree
m = nn.BatchNorm2d(32).eval()
x = torch.randn(4, 32, 16, 16)
with rt.npu_mode():
    ok = torch.allclose(m(x), nn.functional.batch_norm(x, m.running_mean, m.running_var, m.weight, m.bias, False, 0.1, 1e-5))
fails += not ok
print(f"{'PASS' if ok else 'FAIL'}  eval mode untouched")
xl = torch.randn(8, 32, 16, 16).contiguous(memory_format=torch.channels_last)
m = nn.BatchNorm2d(32)
with rt.npu_mode():
    y = m(xl)
bnorm.ENABLED = False
yr = nn.BatchNorm2d(32)(xl)
bnorm.ENABLED = True
ok = rel(y, yr) < 1e-5 and y.is_contiguous(memory_format=torch.channels_last)
fails += not ok
print(f"{'PASS' if ok else 'FAIL'}  channels_last input falls back to PyTorch")

# timing (forward + backward)
for shape in [(32, 256, 32, 32), (32, 512, 16, 16), (32, 64, 64, 64), (16, 128, 28, 28)]:
    x = torch.randn(*shape, requires_grad=True)
    m = nn.BatchNorm2d(shape[1])
    g = torch.randn(*shape)
    out = {}
    for label, on in (("PyTorch", False), ("fused", True)):
        bnorm.ENABLED = on
        mm = rt.convert(copy.deepcopy(m)) if on else copy.deepcopy(m)
        ts = []
        for i in range(4):
            t0 = time.perf_counter()
            y = mm(x)
            y.backward(g)
            ts.append(time.perf_counter() - t0)
        out[label] = sorted(ts[1:])[1] * 1e3
    bnorm.ENABLED = True
    print(f"      BatchNorm2d fwd+bwd {str(shape):<20} PyTorch {out['PyTorch']:7.1f} ms   fused {out['fused']:7.1f} ms   "
          f"{out['PyTorch'] / out['fused']:.2f}x")

print("\nALL PASS" if not fails else f"\nFAILURES: {fails}")
sys.exit(1 if fails else 0)
