"""Safety nets and new op coverage: CPU fallback, shape verification, NaN guard,
batched GEMM, attention, baddbmm, memory cap, convert(), torch.compile, tools.

    python3 tests/test_robust.py
"""
import io
import os
import sys
import warnings

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import rk3588_torch as rt  # noqa: E402
from rk3588_torch import ops  # noqa: E402

torch.manual_seed(0)
warnings.simplefilter("ignore", RuntimeWarning)
failures = 0


def rel(a, b):
    return ((a.double() - b.double()).norm() / (b.double().norm() + 1e-300)).item()


def check(name, ok, detail=""):
    global failures
    failures += not ok
    print(f"{'PASS' if ok else 'FAIL'}  {name:<56} {detail}")


ops.MIN_MACS = 0
assert rt.npu_available(), rt.npu_error()

# ---- batched GEMM: one submission for all batch elements ----
a, b = torch.randn(12, 96, 80), torch.randn(12, 80, 112)
out = ops.gemm(a, b)
st = rt.default_npu().stats()
check("batched gemm correct", rel(out, a.double() @ b.double()) < 2e-3, f"rel {rel(out, a.double() @ b.double()):.1e}")
check("batched gemm is one submit per core-split", st["submits"] == 1, f"submits={st['submits']} tasks={st['tasks']}")
# shared operand (batch stride 0) and transposed views
w = torch.randn(80, 112)
out = ops.gemm(a, w.expand(12, 80, 112))
check("batched gemm, broadcast B", rel(out, a.double() @ w.double()) < 2e-3)
out = ops.gemm(a.transpose(1, 2).contiguous().transpose(1, 2), b)
check("batched gemm, transposed A", rel(out, a.double() @ b.double()) < 2e-3)
for p in ("bf16", "int8", "fp16x3"):
    out = ops.gemm(a, b, p)
    check(f"batched gemm {p}", rel(out, a.double() @ b.double()) < ops.VERIFY_TOL[p] * 2)

# ---- memory cap: a GEMM bigger than the cap is split and still correct ----
npu = rt.default_npu()
npu.set_max_buf_mb(2)
a2, b2 = torch.randn(1024, 512), torch.randn(512, 640)
out = ops.gemm(a2, b2)
check("GEMM split under a 2 MB buffer cap", rel(out, a2.double() @ b2.double()) < 2e-3,
      f"buffers held {npu.buffer_bytes() / 2**20:.1f} MB")
npu.set_max_buf_mb(512)
rt.release()
check("release() frees NPU buffers", npu.buffer_bytes() == 0)

# ---- safety nets (simulated NPU faults) ----
real = ops._npu_gemm
x, y = torch.randn(64, 128), torch.randn(128, 96)
ref = x.double() @ y.double()

ops.reset_verification()
ops._npu_gemm = lambda a, b, p, *r: real(a, b, p, *r) * 1.5         # silently wrong
out = ops.gemm(x, y)
check("wrong NPU result caught by first-use check", rel(out, ref) < 1e-6)
check("...and that shape is pinned to the CPU", ops._verified.get(next(iter(ops._verified))) is False)

ops.reset_verification()
ops._npu_gemm = lambda a, b, p, *r: torch.full((a.shape[0], b.shape[1]), float("nan"))
out = ops.gemm(x, y)
check("NaN NPU output recomputed on the CPU", torch.isfinite(out).all().item() and rel(out, ref) < 1e-6)

ops.reset_verification()
def boom(a, b, p, *r):
    raise RuntimeError("simulated NPU timeout")
ops._npu_gemm = boom
fails0 = ops._failures
out = ops.gemm(x, y)
check("NPU exception falls back to the CPU", rel(out, ref) < 1e-6)
for _ in range(ops.MAX_FAILURES):
    ops.gemm(x, y)
check("repeated failures disable the NPU (circuit breaker)", ops._enabled is False)
ops._npu_gemm = real
ops._failures = 0
rt.set_enabled(True)
ops.reset_verification()

# training still works with NPU disabled
rt.set_enabled(False)
lin = torch.nn.Linear(128, 64)
with rt.npu_mode():
    o = lin(x)
check("npu_mode with NPU disabled = plain CPU", rel(o, F.linear(x, lin.weight, lin.bias)) < 1e-6)
rt.set_enabled(True)

# ---- attention + baddbmm through npu_mode ----
q, k, v = (torch.randn(2, 4, 64, 32, requires_grad=True) for _ in range(3))
mask = torch.rand(64, 64) > 0.2
for kwargs, label in [({}, "plain"), ({"is_causal": True}, "causal"), ({"attn_mask": mask}, "bool mask")]:
    with rt.npu_mode():
        o = F.scaled_dot_product_attention(q, k, v, **kwargs)
    o.sum().backward()
    g_npu = [t.grad.clone() for t in (q, k, v)]
    for t in (q, k, v):
        t.grad = None
    o_ref = F.scaled_dot_product_attention(q, k, v, **kwargs)
    o_ref.sum().backward()
    g_ref = [t.grad.clone() for t in (q, k, v)]
    for t in (q, k, v):
        t.grad = None
    e = max(rel(o, o_ref), *(rel(g1, g0) for g1, g0 in zip(g_npu, g_ref)))
    check(f"scaled_dot_product_attention ({label}) fwd+bwd", e < 5e-3, f"rel {e:.1e}")
inp = torch.randn(6, 48, 40)
b1, b2 = torch.randn(6, 48, 64), torch.randn(6, 64, 40)
with rt.npu_mode():
    o = torch.baddbmm(inp, b1, b2, beta=0.5, alpha=2.0)
check("baddbmm", rel(o, torch.baddbmm(inp, b1, b2, beta=0.5, alpha=2.0)) < 2e-3)
mha = torch.nn.MultiheadAttention(64, 4, batch_first=True)
xs = torch.randn(3, 50, 64)
with rt.npu_mode():
    o, _ = mha(xs, xs, xs)
o_ref, _ = mha(xs, xs, xs)
check("nn.MultiheadAttention under npu_mode", rel(o, o_ref) < 5e-3)

# ---- convert() and torch.compile ----
model = torch.nn.Sequential(torch.nn.Linear(256, 512), torch.nn.GELU(), torch.nn.Linear(512, 10))
xin = torch.randn(64, 256)
ref_out = model(xin)
npu_model = rt.convert(model)
check("convert() swaps nn.Linear for NPULinear", isinstance(npu_model[0], rt.NPULinear))
ops.stats(reset=True)
o = npu_model(xin)
check("converted model runs on the NPU", rel(o, ref_out) < 5e-3 and ops.stats()["npu_calls"] >= 2)
try:
    compiled = torch.compile(npu_model)
    ops.stats(reset=True)
    o = compiled(xin)
    o.sum().backward()
    check("torch.compile(converted model) fwd+bwd", rel(o, ref_out) < 5e-3 and ops.stats()["npu_calls"] >= 2,
          f"npu calls {ops.stats()['npu_calls']}")
except Exception as e:  # noqa: BLE001
    check("torch.compile(converted model) fwd+bwd", False, repr(e)[:120])

# ---- tools ----
buf = io.StringIO()
res = rt.compare(torch.nn.Sequential(torch.nn.Linear(784, 1024), torch.nn.ReLU(), torch.nn.Linear(1024, 10)),
                 torch.randn(256, 784), torch.randint(0, 10, (256,)), precisions=("fp16", "bf16"), steps=3,
                 warmup=1, file=buf)
check("compare() runs and recommends", res["recommendation"] in ("cpu", "fp16", "bf16"),
      f"-> {res['recommendation']}")
buf = io.StringIO()
with rt.profile(file=buf):
    with rt.npu_mode():
        npu_model(xin).sum().backward()
check("profile() reports NPU ops", "NPU ops" in buf.getvalue() and "rknpu::gemm" in buf.getvalue())

print("stats:", ops.stats())
print("ALL PASS" if failures == 0 else f"{failures} FAILURES")
sys.exit(1 if failures else 0)
