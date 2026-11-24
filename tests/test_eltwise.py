"""Fused element-wise kernels (rk3588_torch/eltwise.py) vs PyTorch: GELU (erf/tanh), SiLU, RMSNorm, forward and
backward, fp32 and bf16, odd sizes (vector tails), routing under npu_mode() and convert()'s RMSNorm swap.

    python3 tests/test_eltwise.py
"""
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import rk3588_torch as rt  # noqa: E402
from rk3588_torch import eltwise  # noqa: E402

eltwise.MIN_ELEMS = 0
torch.manual_seed(0)
fails = 0


def rel(a, b):
    return ((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-30)).item()


def check(name, ours, ref, shape, dtype, tol, extra=()):
    global fails
    x = (torch.randn(*shape) * 3).to(dtype).requires_grad_()
    xs = [t.detach().clone().requires_grad_() for t in extra]
    g = torch.randn(*shape).to(dtype)
    y = ours(x, *xs)
    if y is None:
        fails += 1
        print(f"FAIL  {name}: not handled")
        return
    gr = torch.autograd.grad(y, [x] + xs, g)
    x2 = x.detach().float().requires_grad_()
    xs2 = [t.detach().float().requires_grad_() for t in xs]
    y2 = ref(x2, *xs2)
    gr2 = torch.autograd.grad(y2, [x2] + xs2, g.float())
    errs = [rel(y, y2)] + [rel(a, b) for a, b in zip(gr, gr2)]
    ok = max(errs) < tol and y.dtype == dtype
    fails += not ok
    print(f"{'PASS' if ok else 'FAIL'}  {name:<34} {str(dtype)[6:]:<9} " + " ".join(f"{e:.1e}" for e in errs))


for dt, tol in ((torch.float32, 2e-6), (torch.bfloat16, 1.5e-2)):
    for shape in ((513, 771), (4, 1031)):
        check(f"gelu erf {shape}", eltwise.gelu, F.gelu, shape, dt, tol)
        check(f"gelu tanh {shape}", lambda v: eltwise.gelu(v, "tanh"), lambda v: F.gelu(v, approximate="tanh"),
              shape, dt, tol)
        check(f"silu {shape}", eltwise.silu, F.silu, shape, dt, tol)
        D = shape[1]
        check(f"rms_norm {shape}", lambda v, w: eltwise.rms_norm(v, (D,), w, 1e-6),
              lambda v, w: F.rms_norm(v, (D,), w, 1e-6), shape, dt, tol * 3, extra=(torch.randn(D).to(dt),))
        check(f"rms_norm no weight {shape}", lambda v: eltwise.rms_norm(v, (D,), None, 1e-6),
              lambda v: F.rms_norm(v, (D,), None, 1e-6), shape, dt, tol * 3)

# extreme inputs: saturation, no NaN
x = torch.tensor([-1e4, -100., -20., -1e-30, 0., 1e-30, 20., 100., 1e4] * 4)
for name, f, r in (("gelu", eltwise.gelu, F.gelu), ("silu", eltwise.silu, F.silu),
                   ("gelu tanh", lambda v: eltwise.gelu(v, "tanh"), lambda v: F.gelu(v, approximate="tanh"))):
    y, yr = f(x.clone()), r(x.clone())
    ok = torch.isfinite(y).all() and (y - yr).abs().max() < 1e-4
    fails += not ok
    print(f"{'PASS' if ok else 'FAIL'}  {name} at +-1e4 / 1e-30: max |diff| {(y - yr).abs().max().item():.1e}")


class HFRMSNorm(nn.Module):     # what transformers' Qwen2RMSNorm / LlamaRMSNorm look like
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d) + 0.1 * torch.randn(d))
        self.variance_epsilon = eps

    def forward(self, h):
        dt = h.dtype
        h = h.float()
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)
        return self.weight * h.to(dt)


m = nn.Sequential(nn.Linear(256, 512), nn.GELU(), HFRMSNorm(512), nn.SiLU(), nn.RMSNorm(512), nn.Linear(512, 64))
ref = __import__("copy").deepcopy(m)
x = torch.randn(64, 256)
rt.convert(m)
swapped = isinstance(m[2], eltwise.RMSNorm)
with rt.npu_mode():
    y = m(x)
    y.square().mean().backward()
yr = ref(x)
yr.square().mean().backward()
e = max(rel(y, yr), rel(m[2].weight.grad, ref[2].weight.grad), rel(m[0].weight.grad, ref[0].weight.grad))
ok = swapped and e < 5e-3
fails += not ok
print(f"{'PASS' if ok else 'FAIL'}  convert() + npu_mode(): HF RMSNorm swapped {swapped}, max rel {e:.1e}")

print("ALL PASS" if not fails else f"{fails} FAILED")
sys.exit(1 if fails else 0)
