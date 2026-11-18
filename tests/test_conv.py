"""Native NPU convolution: forward/backward correctness over many geometries, plus timing.

    python3 tests/test_conv.py [--bench]
"""
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import rk3588_torch as rt  # noqa: E402
from rk3588_torch import ops  # noqa: E402

torch.manual_seed(0)
ops.VERIFY = "off"      # test the raw NPU path, not the safety net
ops.MIN_MACS = 0
failures = 0
TOL = {"fp16": 2e-3, "bf16": 1e-2}


def rel(a, b):
    return ((a.double() - b.double()).norm() / (b.double().norm() + 1e-300)).item()


def check(name, got, ref, tol):
    global failures
    e = rel(got, ref)
    ok = e < tol
    failures += not ok
    print(f"{'PASS' if ok else 'FAIL'}  {name:<60} rel_err={e:.2e}", flush=True)


def fwd_case(N, C, H, W, O, k, s, p, prec="fp16"):
    x, w = torch.randn(N, C, H, W), torch.randn(O, C, k, k)
    stats0 = ops.stats()["npu_calls"]
    got = ops.conv(x, w, s, p, prec)
    on_npu = ops.stats()["npu_calls"] > stats0
    ref = F.conv2d(x.double(), w.double(), stride=s, padding=p)
    check(f"[{prec}] fwd N{N} C{C} {H}x{W} -> O{O} k{k} s{s} p{p}{'' if on_npu else ' (CPU!)'}", got, ref, TOL[prec])


cases = [
    (1, 32, 8, 8, 16, 1, 1, 0), (1, 32, 8, 8, 16, 3, 1, 1), (2, 3, 28, 28, 16, 3, 1, 1),     # RGB-like input
    (4, 16, 14, 14, 32, 3, 1, 1), (2, 64, 16, 12, 32, 3, 1, 1), (2, 32, 9, 9, 24, 3, 2, 1),   # odd O
    (1, 64, 56, 56, 64, 3, 1, 1),                                                             # multi-slice
    (1, 512, 14, 14, 64, 3, 1, 1),                                                            # channel chunks
    (2, 8, 32, 32, 8, 5, 1, 2), (2, 16, 31, 17, 16, 7, 2, 3), (1, 3, 224, 224, 64, 7, 2, 3),  # big kernels, stem
    (8, 32, 7, 7, 64, 1, 1, 0), (2, 48, 15, 15, 40, 3, 2, 0),
]
for c in cases:
    fwd_case(*c)
for c in [(2, 16, 14, 14, 32, 3, 1, 1), (1, 3, 64, 64, 16, 7, 2, 3)]:
    fwd_case(*c, prec="bf16")

# gradients through the custom op vs fp64 autograd
for (N, C, H, W, O, k, s, p) in [(2, 16, 12, 12, 32, 3, 1, 1), (2, 8, 15, 15, 16, 3, 2, 1), (1, 3, 32, 32, 16, 5, 2, 2)]:
    x = torch.randn(N, C, H, W, requires_grad=True)
    w = torch.randn(O, C, k, k, requires_grad=True)
    y = torch.ops.rknpu.conv2d(x, w, [s, s], [p, p], "fp16")
    g = torch.randn_like(y)
    y.backward(g)
    xd, wd = x.detach().double().requires_grad_(), w.detach().double().requires_grad_()
    F.conv2d(xd, wd, stride=s, padding=p).backward(g.double())
    check(f"grad dX  N{N} C{C} {H}x{W} k{k} s{s} p{p}", x.grad, xd.grad, TOL["fp16"])
    check(f"grad dW  N{N} C{C} {H}x{W} k{k} s{s} p{p}", w.grad, wd.grad, TOL["fp16"])

# a whole CNN through npu_mode vs fp64 CPU autograd. Without nonlinearities every gradient
# must match to fp16 accuracy. With ReLU/MaxPool, fp16 rounding flips a handful of
# near-tie activations / pooling choices (as mixed-precision training does on a GPU), which
# reroutes a little gradient in early layers, so that variant gets a looser bound.
def cnn(nonlin):
    act = torch.nn.ReLU if nonlin else torch.nn.Identity
    pool = torch.nn.MaxPool2d(2) if nonlin else torch.nn.AvgPool2d(2)
    return torch.nn.Sequential(torch.nn.Conv2d(3, 32, 3, padding=1), act(), pool,
                               torch.nn.Conv2d(32, 64, 3, padding=1), act(), torch.nn.Conv2d(64, 64, 3, stride=2),
                               torch.nn.Flatten(), torch.nn.LazyLinear(10))


xi = torch.randn(8, 3, 32, 32)
for nonlin, tol in ((False, 2e-3), (True, 5e-2)):
    torch.manual_seed(1)
    net = cnn(nonlin)
    net(xi)
    net.zero_grad()
    with rt.npu_mode(conv=True):
        o1 = net(xi)
    o1.square().mean().backward()
    g1 = [p.grad.clone() for p in net.parameters()]
    net64 = net.double()
    net64.zero_grad()
    o0 = net64(xi.double())
    o0.square().mean().backward()
    tag = "ReLU+MaxPool" if nonlin else "linear"
    check(f"CNN ({tag}) forward", o1, o0, 2e-3)
    for i, (a, b) in enumerate(zip(g1, [p.grad for p in net64.parameters()])):
        check(f"CNN ({tag}) grad[{i}] {tuple(a.shape)}", a, b, tol)

# convert(): nn.Conv2d -> NPUConv2d, and torch.compile over it
torch.manual_seed(2)
net = torch.nn.Sequential(torch.nn.Conv2d(3, 16, 3, padding=1), torch.nn.ReLU(), torch.nn.Conv2d(16, 16, 3, stride=2))
xc = torch.randn(4, 3, 16, 16)
ref = net(xc).detach()
rt.convert(net)
ok = type(net[0]).__name__ == "NPUConv2d"
failures += not ok
print(f"{'PASS' if ok else 'FAIL'}  convert() swaps nn.Conv2d for NPUConv2d")
n0 = ops.stats()["npu_calls"]
check("converted conv net (no npu_mode)", net(xc), ref, 2e-3)
ok = ops.stats()["npu_calls"] > n0
failures += not ok
print(f"{'PASS' if ok else 'FAIL'}  ...and it ran on the NPU")
try:
    cnet = torch.compile(net, backend="aot_eager")
    y = cnet(xc)
    y.sum().backward()
    check("torch.compile(converted conv net) fwd", y, ref, 2e-3)
except Exception as e:  # noqa: BLE001
    failures += 1
    print("FAIL  torch.compile(converted conv net):", repr(e)[:200])

if "--bench" in sys.argv:
    torch.set_num_threads(3)
    print("\nconv timing (fp16, forward), CPU = torch fp32 3 threads")
    for (N, C, H, W, O, k, s, p) in [(32, 32, 28, 28, 64, 3, 1, 1), (8, 64, 56, 56, 64, 3, 1, 1),
                                      (16, 128, 28, 28, 128, 3, 1, 1), (8, 3, 224, 224, 64, 7, 2, 3)]:
        x, w = torch.randn(N, C, H, W), torch.randn(O, C, k, k)
        def best(fn):
            fn()
            t = []
            for _ in range(3):
                t0 = time.perf_counter()
                fn()
                t.append(time.perf_counter() - t0)
            return min(t)
        tc = best(lambda: F.conv2d(x, w, stride=s, padding=p))
        tn = best(lambda: ops.conv(x, w, s, p, "fp16"))
        oh = (H + 2 * p - k) // s + 1
        gf = 2 * N * O * oh * oh * C * k * k / 1e9
        print(f"  N{N} C{C} {H}x{W} O{O} k{k}: CPU {tc * 1e3:7.1f} ms ({gf / tc:5.0f} GF)  NPU {tn * 1e3:7.1f} ms "
              f"({gf / tn:5.0f} GF)  x{tc / tn:.2f}", flush=True)

print("ALL PASS" if failures == 0 else f"{failures} FAILURES")
sys.exit(1 if failures else 0)
