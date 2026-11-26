"""conv1d, conv_transpose2d, grouped conv, einsum, fused bias/ReLU and the packed-weight cache:
forward and gradients against fp64 CPU references.

    python3 tests/test_new_ops.py           # on the board (real NPU)
    python3 tests/test_new_ops.py --fake    # anywhere: NPU calls emulated on the CPU (tests the
                                            # decompositions and autograd, not the hardware)
"""
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import rk3588_torch as rt  # noqa: E402
from rk3588_torch import ops, wcache  # noqa: E402

FAKE = "--fake" in sys.argv
if FAKE:
    def _fake_gemm(a, b, precision, bias=None, relu=False, wkey=0):
        out = a.float().half().float() @ b.float().half().float()
        ops._stats["npu_calls"] += 1
        return ops._epilogue(out, bias, relu)

    def _fake_conv(x, w, st, pd, precision, groups=1, bias=None, relu=False, wkey=0):
        out = F.conv2d(x.float().half().float(), w.float().half().float(), None if bias is None else bias.float(),
                       stride=st, padding=pd, groups=groups)
        ops._stats["npu_calls"] += 1
        return torch.relu(out) if relu else out

    ops._npu_gemm, ops._npu_conv = _fake_gemm, _fake_conv
    ops.npu_available = lambda: True

torch.manual_seed(0)
ops.VERIFY = "off"      # test the raw NPU path, not the safety net
ops.MIN_MACS = 0
failures = 0
TOL = 3e-3


def rel(a, b):
    return ((a.double() - b.double()).norm() / (b.double().norm() + 1e-300)).item()


def check(name, got, ref, tol=TOL):
    global failures
    e = rel(got, ref)
    ok = e < tol
    failures += not ok
    print(f"{'PASS' if ok else 'FAIL'}  {name:<66} rel_err={e:.2e}", flush=True)


def flag(name, ok):
    global failures
    failures += not ok
    print(f"{'PASS' if ok else 'FAIL'}  {name}", flush=True)


def grads(name, fn_npu, fn_ref, *shapes, tol=TOL):
    """Compare outputs and gradients of fn_npu (fp32 inputs) and fn_ref (fp64 copies)."""
    xs = [torch.randn(*s, requires_grad=True) for s in shapes]
    n0 = ops.stats()["npu_calls"]
    y = fn_npu(*xs)
    on_npu = ops.stats()["npu_calls"] > n0
    g = torch.randn_like(y)
    y.backward(g)
    xd = [x.detach().double().requires_grad_() for x in xs]
    yd = fn_ref(*xd)
    yd.backward(g.double())
    check(f"{name} fwd{'' if on_npu else ' (CPU!)'}", y, yd, tol)
    if not on_npu:
        flag(f"{name} ran on the NPU", False)
    for i, (a, b) in enumerate(zip(xs, xd)):
        check(f"{name} grad[{i}]", a.grad, b.grad, tol)


# ---- fused bias / ReLU -------------------------------------------------------------------
grads("linear+bias", lambda x, w, b: ops.linear(x, w, b), lambda x, w, b: F.linear(x, w, b), (64, 96), (80, 96), (80,))
# ReLU tolerance: fp16 rounding flips near-zero pre-activations, rerouting a little gradient
grads("linear+bias+relu (fused)", lambda x, w, b: ops.linear(x, w, b, "relu"),
      lambda x, w, b: F.relu(F.linear(x, w, b)), (64, 96), (80, 96), (80,), tol=2e-2)
grads("linear+relu, no bias (ReLU in the DPU)", lambda x, w: ops.linear(x, w, None, "relu"),
      lambda x, w: F.relu(F.linear(x, w)), (64, 96), (80, 96), tol=2e-2)
grads("linear 3-D input + bias", lambda x, w, b: ops.linear(x, w, b), lambda x, w, b: F.linear(x, w, b),
      (4, 16, 96), (48, 96), (48,))
grads("conv2d+bias", lambda x, w, b: ops.conv2d(x, w, b, 1, 1), lambda x, w, b: F.conv2d(x, w, b, 1, 1),
      (2, 16, 12, 12), (32, 16, 3, 3), (32,))
grads("conv2d+bias+relu (fused)", lambda x, w, b: ops.conv2d(x, w, b, 2, 1, act="relu"),
      lambda x, w, b: F.relu(F.conv2d(x, w, b, 2, 1)), (2, 16, 13, 13), (32, 16, 3, 3), (32,), tol=2e-2)

# ---- grouped convolution -------------------------------------------------------------------
for (N, C, H, O, G, k, s, p) in [(2, 32, 10, 32, 2, 3, 1, 1), (2, 64, 9, 64, 4, 3, 2, 1), (1, 128, 8, 128, 8, 3, 1, 1),
                                 (2, 48, 8, 24, 3, 1, 1, 0)]:
    grads(f"grouped conv C{C} O{O} G{G} k{k} s{s}", lambda x, w, G=G, s=s, p=p: ops.conv2d(x, w, None, s, p, 1, G),
          lambda x, w, G=G, s=s, p=p: F.conv2d(x, w, None, s, p, 1, G), (N, C, H, H), (O, C // G, k, k))
flag("depthwise conv is left to the CPU", ops.conv2d(torch.randn(1, 32, 8, 8), torch.randn(32, 1, 3, 3),
                                                     padding=1, groups=32) is None)

# ---- conv1d ---------------------------------------------------------------------------------
grads("conv1d k5 s1 p2 + bias", lambda x, w, b: ops.conv1d(x, w, b, 1, 2), lambda x, w, b: F.conv1d(x, w, b, 1, 2),
      (4, 32, 100), (64, 32, 5), (64,))
grads("conv1d k3 s2 grouped", lambda x, w: ops.conv1d(x, w, None, 2, 1, 1, 2),
      lambda x, w: F.conv1d(x, w, None, 2, 1, 1, 2), (2, 64, 77), (64, 32, 3))
grads("conv1d k1 (pointwise), long sequence", lambda x, w: ops.conv1d(x, w), lambda x, w: F.conv1d(x, w),
      (2, 64, 2000), (128, 64, 1))
grads("conv1d unbatched", lambda x, w: ops.conv1d(x, w, None, 1, 1), lambda x, w: F.conv1d(x, w, None, 1, 1),
      (32, 50), (48, 32, 3))

# ---- conv_transpose2d ---------------------------------------------------------------------
for (C, O, H, k, s, p, op) in [(32, 16, 8, 2, 2, 0, 0), (32, 16, 7, 3, 2, 1, 1), (16, 32, 6, 4, 2, 1, 0),
                               (16, 16, 9, 3, 1, 1, 0), (24, 12, 5, 3, 3, 0, 2)]:
    grads(f"conv_transpose2d C{C}->O{O} k{k} s{s} p{p} op{op}",
          lambda x, w, b, s=s, p=p, op=op: ops.conv_transpose2d(x, w, b, s, p, op),
          lambda x, w, b, s=s, p=p, op=op: F.conv_transpose2d(x, w, b, s, p, op), (2, C, H, H), (C, O, k, k), (O,))
grads("conv_transpose2d grouped", lambda x, w: ops.conv_transpose2d(x, w, None, 2, 1, 1, 2),
      lambda x, w: F.conv_transpose2d(x, w, None, 2, 1, 1, 2), (2, 32, 6, 6), (32, 16, 3, 3))

# ---- einsum ----------------------------------------------------------------------------------
for eq, sa, sb in [("ij,jk->ik", (64, 48), (48, 80)), ("bij,bjk->bik", (4, 32, 48), (4, 48, 40)),
                   ("bhqd,bhkd->bhqk", (2, 4, 32, 16), (2, 4, 24, 16)), ("bnd,de->bne", (3, 40, 64), (64, 32)),
                   ("ik,jk->ij", (50, 64), (70, 64)), ("abc,cd->dba", (6, 10, 32), (32, 12)),
                   ("ijk,jl->il", (20, 30, 4), (30, 40)), ("bi,bj->bij", (8, 32), (8, 24)),
                   ("i j , j k -> k i", (40, 50), (50, 60))]:
    grads(f"einsum '{eq}'", lambda a, b, eq=eq: ops.einsum(eq, a, b), lambda a, b, eq=eq: torch.einsum(eq, a, b),
          sa, sb)
grads("einsum with '...'", lambda a, b: ops.einsum("...ij,...jk->...ik", a, b),
      lambda a, b: torch.einsum("...ij,...jk->...ik", a, b), (2, 3, 4), (2, 4, 5))
flag("einsum with a repeated index is left to torch", ops.einsum("ii,ij->j", torch.randn(4, 4), torch.randn(4, 5)) is None)

# ---- routing through npu_mode -------------------------------------------------------------
x = torch.randn(8, 16, 64)
n0 = ops.stats()["npu_calls"]
with rt.npu_mode():
    y1 = F.conv1d(x, torch.randn(32, 16, 3), padding=1)
    y2 = torch.einsum("bci,bdi->bcd", x, x)
    y3 = F.conv_transpose2d(torch.randn(2, 32, 8, 8), torch.randn(32, 16, 2, 2), stride=2)
    y4 = F.conv2d(torch.randn(2, 64, 8, 8), torch.randn(64, 16, 3, 3), groups=4, padding=1)
flag(f"npu_mode routes conv1d / einsum / conv_transpose2d / grouped conv ({ops.stats()['npu_calls'] - n0} NPU calls)",
     ops.stats()["npu_calls"] - n0 >= 4)
flag("shapes", (y1.shape, y2.shape, y3.shape, y4.shape) ==
     ((8, 32, 64), (8, 16, 16), (2, 16, 16, 16), (2, 64, 8, 8)))

# ---- convert(): new module types and ReLU fusion --------------------------------------------
torch.manual_seed(3)
net = nn.Sequential(nn.Conv1d(8, 32, 3, padding=1), nn.ReLU(), nn.Flatten(), nn.Linear(32 * 20, 64), nn.ReLU(),
                    nn.Linear(64, 10))
xin = torch.randn(16, 8, 20)
ref = net(xin).detach()
rt.convert(net)
flag("convert(): Conv1d/Linear swapped, ReLUs fused",
     type(net[0]).__name__ == "NPUConv1d" and net[0].fused_relu and net[3].fused_relu
     and type(net[1]).__name__ == "_FusedReLU" and not net[5].fused_relu)
check("converted + fused net forward", net(xin), ref)
flag("state_dict keys unchanged", list(net.state_dict()) == ["0.weight", "0.bias", "3.weight", "3.bias", "5.weight",
                                                             "5.bias"])
up = nn.Sequential(nn.ConvTranspose2d(16, 8, 2, stride=2), nn.Conv2d(8, 8, 3, padding=1))
xu = torch.randn(2, 16, 8, 8)
ref = up(xu).detach()
rt.convert(up)
check("converted ConvTranspose2d", up(xu), ref)
try:
    ref_c = net(xin).detach()
    cnet = torch.compile(net, backend="aot_eager")
    yc = cnet(xin)
    yc.sum().backward()
    check("torch.compile(converted fused net)", yc, ref_c)
except Exception as e:  # noqa: BLE001
    flag(f"torch.compile(converted fused net): {e!r}"[:200], False)

# ---- packed-weight cache -------------------------------------------------------------------
if not FAKE:
    wcache.clear()
    torch.manual_seed(4)
    frozen = nn.Linear(512, 512)
    frozen.requires_grad_(False)
    xs = [torch.randn(64, 512) for _ in range(4)]
    with torch.no_grad():
        refs = [F.linear(x, frozen.weight, frozen.bias) for x in xs]
        with rt.npu_mode():
            outs = [frozen(x) for x in xs]
    info = wcache.info()
    for i, (o, r) in enumerate(zip(outs, refs)):
        check(f"cached frozen linear, call {i}", o, r)
    flag(f"weight cache hits on repeated frozen weights ({info})", info["hits"] >= 2 and info["npu_bytes"] > 0)
    # an in-place update (optimizer-style) must invalidate
    with torch.no_grad():
        frozen.weight.mul_(-1.0)
        with rt.npu_mode():
            o = frozen(xs[0])
    check("after in-place update (version bump)", o, F.linear(xs[0], frozen.weight, frozen.bias))
    with torch.no_grad(), rt.npu_mode():
        frozen(xs[1])
        frozen(xs[1])
        frozen.weight.data.add_(1.0)     # bypasses the version counter
        o = frozen(xs[2])
    check("after .data update (fingerprint catches it)", o, F.linear(xs[2], frozen.weight, frozen.bias))
    # LoRA: frozen base + trainable low-rank adapter, a few SGD steps vs CPU
    torch.manual_seed(5)

    class LoRA(nn.Module):
        def __init__(self, d=512, r=8):
            super().__init__()
            self.base = nn.Linear(d, d)
            self.base.requires_grad_(False)
            self.a = nn.Parameter(torch.randn(r, d) * 0.01)
            self.b = nn.Parameter(torch.zeros(d, r))

        def forward(self, x):
            return self.base(x) + F.linear(F.linear(x, self.a), self.b)

    m1 = LoRA()
    m2 = LoRA()
    m2.load_state_dict(m1.state_dict())
    o1, o2 = torch.optim.SGD(m1.parameters(), lr=0.1), torch.optim.SGD(m2.parameters(), lr=0.1)
    xb, yb = torch.randn(128, 512), torch.randn(128, 512)
    h0 = wcache.info()["hits"]
    for _ in range(4):
        with rt.npu_mode():
            l1 = F.mse_loss(m1(xb), yb)
            o1.zero_grad()
            l1.backward()
        o1.step()
        l2 = F.mse_loss(m2(xb), yb)
        o2.zero_grad()
        l2.backward()
        o2.step()
    check("LoRA: adapter after 4 SGD steps (NPU vs CPU)", m1.b.detach(), m2.b.detach(), 1e-2)
    flag(f"LoRA: frozen base weight served from the cache ({wcache.info()['hits'] - h0} hits)",
         wcache.info()["hits"] - h0 >= 2)

print(f"\n{'ALL PASSED' if failures == 0 else f'{failures} FAILED'}")
sys.exit(1 if failures else 0)
