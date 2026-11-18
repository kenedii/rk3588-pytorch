"""Correctness of NPU GEMMs and their gradients against fp64 CPU references.

    python3 tests/test_ops.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import rk3588_torch as rt  # noqa: E402
from rk3588_torch import ops  # noqa: E402

ops.MIN_MACS = 0  # force everything onto the NPU
torch.manual_seed(0)
failures = 0


def rel_err(a, b):
    return ((a.double() - b.double()).norm() / (b.double().norm() + 1e-30)).item()


def check(name, got, ref, tol=2e-3):
    global failures
    e = rel_err(got, ref)
    ok = e < tol
    failures += not ok
    print(f"{'PASS' if ok else 'FAIL'}  {name:<46} rel_err={e:.2e}")


# fp16 rounding of inputs bounds the error at ~1e-3 relative; fp32 accumulate keeps it there
for (M, K, N) in [(1, 64, 64), (7, 33, 17), (64, 64, 64), (128, 784, 256), (300, 257, 129), (512, 1024, 512)]:
    a, b = torch.randn(M, K), torch.randn(K, N)
    check(f"mm {M}x{K} @ {K}x{N}", ops.mm_raw(a, b), a.double() @ b.double())
    check(f"mm transposed A ({M}x{K})", ops.mm_raw(a.t().contiguous().t(), b), a.double() @ b.double())
    check(f"mm transposed B ({K}x{N})", ops.mm_raw(a, b.t().contiguous().t()), a.double() @ b.double())

# linear forward + backward
x = torch.randn(96, 300, requires_grad=True)
w = torch.randn(200, 300, requires_grad=True)
bias = torch.randn(200, requires_grad=True)
y = ops.linear(x, w, bias)
g = torch.randn_like(y)
y.backward(g)
xd, wd, bd = x.detach().double().requires_grad_(), w.detach().double().requires_grad_(), bias.detach().double().requires_grad_()
yd = torch.nn.functional.linear(xd, wd, bd)
yd.backward(g.double())
check("linear forward", y, yd)
check("linear dX", x.grad, xd.grad)
check("linear dW", w.grad, wd.grad)
check("linear dbias", bias.grad, bd.grad)

# npu_mode routing: MLP + conv net, compare grads with CPU
def grads(model, inp, use_npu):
    model.zero_grad()
    if use_npu:
        with rt.npu_mode(conv=True):
            out = model(inp)
    else:
        out = model(inp)
    out.square().mean().backward()
    return out.detach(), [p.grad.detach().clone() for p in model.parameters()]


mlp = torch.nn.Sequential(torch.nn.Linear(784, 256), torch.nn.ReLU(), torch.nn.Linear(256, 10))
xi = torch.randn(64, 784)
o1, g1 = grads(mlp, xi, True)
o0, g0 = grads(mlp, xi, False)
check("npu_mode MLP output", o1, o0)
for i, (a1, a0) in enumerate(zip(g1, g0)):
    check(f"npu_mode MLP grad[{i}] {tuple(a1.shape)}", a1, a0, tol=5e-3)

cnn = torch.nn.Sequential(torch.nn.Conv2d(1, 16, 3, padding=1), torch.nn.ReLU(), torch.nn.Conv2d(16, 32, 3, stride=2),
                          torch.nn.Flatten(), torch.nn.LazyLinear(10))
xc = torch.randn(8, 1, 28, 28)
cnn(xc)
o1, g1 = grads(cnn, xc, True)
o0, g0 = grads(cnn, xc, False)
check("npu_mode CNN output", o1, o0)
for i, (a1, a0) in enumerate(zip(g1, g0)):
    check(f"npu_mode CNN grad[{i}] {tuple(a1.shape)}", a1, a0, tol=5e-3)

# batched matmul (attention-style)
q, k = torch.randn(4, 64, 32, requires_grad=True), torch.randn(4, 32, 64, requires_grad=True)
with rt.npu_mode():
    s = q @ k
s.sum().backward()
check("bmm via npu_mode", s, q.detach().double() @ k.detach().double())

print("stats:", ops.stats())
print("ALL PASS" if failures == 0 else f"{failures} FAILURES")
sys.exit(1 if failures else 0)
