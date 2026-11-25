"""fp16 NPU convolution output (RKNPU_CONV_OUT16=1): correctness vs fp32 output and PyTorch, including
bias/ReLU epilogue, groups, strides, and the fp32 redo when an fp16 output would overflow.

    python3 tests/test_conv_out16.py
"""
import os
import sys

os.environ.setdefault("RKNPU_CONV_OUT16", "1")
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rk3588_torch import ops  # noqa: E402

torch.manual_seed(0)
ops.VERIFY = "off"
ops.MIN_MACS = 0
fail = 0


def check(name, got, ref, tol=3e-3):
    global fail
    e = ((got.double() - ref.double()).norm() / (ref.double().norm() + 1e-30)).item()
    ok = e < tol
    fail += not ok
    print(f"{'PASS' if ok else 'FAIL'}  {name:<56} rel_err={e:.2e}", flush=True)


for (N, C, H, O, k, s, p, g, bias, act) in [
    (2, 32, 16, 32, 3, 1, 1, 1, False, None), (3, 64, 15, 48, 3, 2, 1, 1, True, None),
    (2, 32, 14, 64, 1, 1, 0, 1, True, "relu"), (1, 64, 9, 80, 3, 1, 1, 1, True, "relu"),
    (2, 64, 12, 64, 3, 1, 1, 4, True, None), (4, 16, 20, 24, 5, 1, 2, 1, False, "relu"),
    (2, 128, 8, 256, 3, 1, 1, 1, True, None),
    (8, 256, 32, 64, 1, 1, 0, 1, False, None), (8, 64, 32, 128, 3, 1, 1, 1, False, None),   # multi-slice (row-sliced) maps
    (4, 512, 16, 128, 1, 1, 0, 1, True, "relu"), (2, 300, 8, 40, 3, 1, 1, 1, False, None),  # nk > 1 -> fp32 path
]:
    x = torch.randn(N, C, H, H)
    w = torch.randn(O, C // g, k, k) * 0.1
    b = torch.randn(O) if bias else None
    got = ops.conv2d(x, w, b, s, p, 1, g, act)
    ref = F.conv2d(x, w, b, s, p, 1, g)
    if act == "relu":
        ref = ref.clamp_min(0)
    check(f"N{N} {C}->{O} {H}x{H} k{k} s{s} p{p} g{g} b{int(bias)} {act}", got, ref)

# overflow: huge activations x huge weights -> fp16 output would be inf; must come back finite and right
x = torch.randn(2, 64, 8, 8) * 3e3
w = torch.randn(32, 64, 3, 3) * 3e3
got = ops.conv2d(x, w, None, 1, 1, 1, 1, None)
ref = F.conv2d(x.double(), w.double(), None, 1, 1)
print("finite:", bool(torch.isfinite(got).all()))
fail += not bool(torch.isfinite(got).all())
check("overflow redo (|y| ~ 1e9)", got, ref, 1e-2)
print("FAILURES:", fail)
sys.exit(1 if fail else 0)
