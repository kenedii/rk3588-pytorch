"""inf / NaN operands must give PyTorch's exact (CPU) result on every NPU path now that the engine, not a Python
scan, reports them; huge-but-finite operands must stay on the NPU and be right.

    python3 tests/test_finite.py
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
fail = 0


def same(a, b):
    """same inf/nan pattern, finite entries close"""
    if a.shape != b.shape or not torch.equal(torch.isnan(a), torch.isnan(b)) or not torch.equal(torch.isinf(a), torch.isinf(b)):
        return False
    m = torch.isfinite(b)
    return m.sum() == 0 or ((a[m].double() - b[m].double()).norm() / (b[m].double().norm() + 1e-30)).item() < 3e-3


def check(name, ok):
    global fail
    fail += not ok
    print(f"{'PASS' if ok else 'FAIL'}  {name}", flush=True)


def calls():
    return ops._stats["npu_calls"]


for bad in (float("inf"), float("-inf"), float("nan")):
    tag = str(bad)
    a = torch.randn(64, 128); b = torch.randn(128, 96)
    a[3, 5] = bad
    check(f"gemm A {tag}", same(ops.gemm(a, b), a @ b))
    a = torch.randn(64, 128); b = torch.randn(128, 96)
    b[7, 9] = bad
    check(f"gemm B {tag}", same(ops.gemm(a, b), a @ b))
    w = torch.randn(96, 128)
    w[5, 5] = bad
    x = torch.randn(64, 128)
    check(f"linear (cached weight) {tag}", same(F.linear(x, w) if False else ops.gemm(x, w.t(), cache_b="w1"), x @ w.t()))
    xc = torch.randn(2, 32, 12, 12); wc = torch.randn(32, 32, 3, 3) * 0.1
    xc[1, 3, 4, 4] = bad
    check(f"conv x {tag}", same(ops.conv2d(xc, wc, None, 1, 1, 1, 1, None), F.conv2d(xc, wc, None, 1, 1)))
    xc = torch.randn(2, 32, 12, 12); wc = torch.randn(32, 32, 3, 3) * 0.1
    wc[2, 2, 1, 1] = bad
    check(f"conv w {tag}", same(ops.conv2d(xc, wc, None, 1, 1, 1, 1, None), F.conv2d(xc, wc, None, 1, 1)))
    # weight gradient with a non-finite activation: no NPU garbage, no permanently rejected shape
    x = torch.randn(4, 16, 12, 12, requires_grad=True); w = torch.randn(32, 16, 3, 3, requires_grad=True)
    xb = x.detach().clone(); xb[0, 1, 2, 2] = bad
    with rt.npu_mode():
        y = F.conv2d(xb, w, padding=1)
        y.sum().backward()
    ref = torch.nn.grad.conv2d_weight(xb, w.shape, torch.ones(4, 32, 12, 12), padding=1)
    ok = same(w.grad, ref)
    if not ok:
        print("   grad nan/inf:", int(torch.isnan(w.grad).sum()), int(torch.isinf(w.grad).sum()), " ref:",
              int(torch.isnan(ref).sum()), int(torch.isinf(ref).sum()), " max|d| finite:",
              float((w.grad - ref)[torch.isfinite(ref) & torch.isfinite(w.grad)].abs().max()))
    check(f"conv weight grad x {tag}", ok)
    w.grad = None

# finite but far outside fp16 range: scaled, stays on the NPU, correct
n0 = calls()
a = torch.randn(64, 128) * 1e6; b = torch.randn(128, 96) * 1e-6
check("huge finite operands correct", same(ops.gemm(a, b), (a.double() @ b.double()).float()))
check("huge finite operands ran on the NPU", calls() > n0)
a = torch.randn(64, 128) * 1e-9; b = torch.randn(128, 96) * 1e-9
check("tiny finite operands correct", ((ops.gemm(a, b).double() - (a.double() @ b.double())).norm() / (a.double() @ b.double()).norm()).item() < 3e-3)
# the flag does not stick: a clean call after a bad one is on the NPU
n0 = calls()
a = torch.randn(64, 128); b = torch.randn(128, 96)
ops.gemm(a, b)
check("clean call after bad ones uses the NPU", calls() > n0)
print("FAILURES:", fail)
sys.exit(1 if fail else 0)
