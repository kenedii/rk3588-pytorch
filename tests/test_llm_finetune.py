"""What LLM fine-tuning needs: frozen bf16/fp16 weights (cached), bounded conversion memory for huge
operands, cheap first-use checks on a huge lm_head, gradient checkpointing, weight-cache sizing, the
duty-cycle throttle.

    python3 tests/test_llm_finetune.py           # on the board (real NPU)
    python3 tests/test_llm_finetune.py --fake    # anywhere: NPU calls emulated on the CPU (tests the
                                                 # logic and autograd, not the hardware)
"""
import os
import sys
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as tuc

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import rk3588_torch as rt  # noqa: E402
from rk3588_torch import _lib, ops, wcache  # noqa: E402

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

# record the biggest fp32 temporary every NPU call would make for bf16/fp16 operands
_real_npu_gemm = ops._npu_gemm
seen = {"convert": 0, "calls": 0}


def _recording(a, b, precision, bias=None, relu=False, wkey=0):
    seen["convert"] = max(seen["convert"], ops._convert_bytes(a, b))
    seen["calls"] += 1
    return _real_npu_gemm(a, b, precision, bias, relu, wkey)


ops._npu_gemm = _recording

torch.manual_seed(0)
ops.VERIFY = "off"
ops.MIN_MACS = 0
failures = 0


def rel(a, b):
    return ((a.double() - b.double()).norm() / (b.double().norm() + 1e-300)).item()


def check(name, ok, detail=""):
    global failures
    failures += not ok
    print(f"{'PASS' if ok else 'FAIL'}  {name:<62} {detail}", flush=True)


# ---------------------------------------------------------------------------------------------------
# frozen bf16 weights: cached after the second use, forward and backward correct
# ---------------------------------------------------------------------------------------------------
wcache.clear()
lin = nn.Linear(1024, 2048).to(torch.bfloat16)
x = torch.randn(64, 1024).to(torch.bfloat16)
ref = F.linear(x.float(), lin.weight.float(), lin.bias.float())
h0 = wcache.info()["hits"]
with torch.no_grad(), rt.npu_mode():
    for _ in range(4):
        out = lin(x)
hits = wcache.info()["hits"] - h0
check("bf16 weight: output correct", rel(out.float(), ref) < 1.5e-2, f"rel {rel(out.float(), ref):.1e}")
check("bf16 weight: reused weight goes through the cache", hits >= 2, f"{hits} hits in 4 calls")

# LoRA-style step: frozen bf16 base, trainable fp32 adapters, bf16 activations
base = nn.Linear(512, 1024).to(torch.bfloat16)
base.requires_grad_(False)
A = nn.Parameter(torch.randn(16, 512) * 0.05)
B = nn.Parameter(torch.randn(1024, 16) * 0.05)
xx = torch.randn(32, 512).to(torch.bfloat16)


def lora_loss():
    y = base(xx) + F.linear(F.linear(xx.float(), A), B).to(torch.bfloat16)
    return y.float().pow(2).mean()


for p in (A, B):
    p.grad = None
lora_loss().backward()
ga_ref, gb_ref = A.grad.clone(), B.grad.clone()
for p in (A, B):
    p.grad = None
with rt.npu_mode():
    for _ in range(3):                      # 3 steps: the frozen weight is cached from the 2nd on
        for p in (A, B):
            p.grad = None
        lora_loss().backward()
check("bf16 frozen base + LoRA: adapter gradients", rel(A.grad, ga_ref) < 3e-2 and rel(B.grad, gb_ref) < 3e-2,
      f"A {rel(A.grad, ga_ref):.1e}  B {rel(B.grad, gb_ref):.1e}")

# ---------------------------------------------------------------------------------------------------
# huge low-precision operands are processed in pieces (bounded fp32 temporaries)
# ---------------------------------------------------------------------------------------------------
ops.CONVERT_MB = 1
BUDGET = 1 << 20
cases = {
    "N split (big bf16 weight)": (torch.randn(64, 512).bfloat16(), torch.randn(512, 4096).bfloat16()),
    "M split (big bf16 activations, fp32 weight)": (torch.randn(8192, 256).bfloat16(), torch.randn(256, 64)),
    "K split (long contraction, e.g. lm_head backward)": (torch.randn(16, 65536).bfloat16(),
                                                           torch.randn(65536, 16).bfloat16()),
    "batched": (torch.randn(4, 128, 1024).bfloat16(), torch.randn(4, 1024, 256).bfloat16()),
}
for name, (a, b) in cases.items():
    seen["convert"] = 0
    if a.dim() == 2:
        bias = torch.randn(b.shape[1])
        got = ops.gemm(a, b, "fp16", bias=bias, relu=True)
        want = torch.relu(a.float() @ b.float() + bias)
    else:
        got = ops.gemm(a, b, "fp16")
        want = a.float() @ b.float()
    check(f"split GEMM, {name}", rel(got, want) < 3e-3 and seen["convert"] <= BUDGET,
          f"rel {rel(got, want):.1e}, largest fp32 temp {seen['convert'] / 2**20:.2f} MB (budget 1)")
ops.CONVERT_MB = int(os.environ.get("RKNPU_CONVERT_MB", "128"))

# the same through autograd, for a big bf16 embedding-sized lm_head (forward + both gradients)
ops.CONVERT_MB = 1
W = nn.Parameter(torch.randn(8192, 128).bfloat16(), requires_grad=False)
hs = torch.randn(32, 128, requires_grad=True)
with rt.npu_mode():
    logits = F.linear(hs, W)
    logits.float().pow(2).mean().backward()
ref_h = torch.randn(1)  # placeholder to keep names unique
hs2 = hs.detach().clone().requires_grad_()
F.linear(hs2, W.float()).pow(2).mean().backward()
check("big bf16 lm_head: logits and input gradient", rel(hs.grad, hs2.grad) < 2e-2,
      f"grad rel {rel(hs.grad, hs2.grad):.1e}")
ops.CONVERT_MB = int(os.environ.get("RKNPU_CONVERT_MB", "128"))

# ---------------------------------------------------------------------------------------------------
# first-use check of a new shape: no fp64 copy of a huge weight, still catches wrong results
# ---------------------------------------------------------------------------------------------------
real = ops._npu_gemm
ops.VERIFY = "first"
for label, (K, N, chunk) in {"chunked reference": (96, 4000, 1000),
                              "sampled columns (huge weight)": (128, 100_000, None)}.items():
    a, b = torch.randn(40, K), torch.randn(K, N)
    if chunk:
        b = b.bfloat16()                 # low-precision weight through the chunked reference
    old_chunk = ops.VERIFY_CHUNK_ELEMS
    if chunk:
        ops.VERIFY_CHUNK_ELEMS = chunk
    ops.reset_verification()
    out = ops.gemm(a, b)
    ok_good = rel(out, a.double() @ b.double()) < 3e-3 and all(v is True for v in ops._verified.values())
    ops.reset_verification()
    ops._npu_gemm = lambda a, b, p, *r: real(a, b, p, *r) * 1.5          # silently wrong
    out = ops.gemm(a, b)
    ok_bad = rel(out, a.double() @ b.double()) < 1e-6 and all(v is False for v in ops._verified.values())
    ops._npu_gemm = real
    ops.VERIFY_CHUNK_ELEMS = old_chunk
    check(f"first-use check, {label}", ok_good and ok_bad, f"good passes: {ok_good}, wrong caught: {ok_bad}")
ops.VERIFY = "off"
ops.reset_verification()

# a weight with inf is never trusted, cached or not
wcache.clear()
a = torch.randn(256, 512)
w = torch.randn(512, 256)
w[7, 9] = float("inf")
ok = True
for _ in range(4):
    got = ops.gemm(a, w, "fp16", cache_b="test.w")
    ok &= torch.equal(torch.isfinite(got), torch.isfinite(a @ w))
check("weight with inf takes the CPU path on every call (cached or not)", ok and wcache.info()["entries"] == 0)

# ---------------------------------------------------------------------------------------------------
# gradient checkpointing: backward outside the npu_mode() block
# ---------------------------------------------------------------------------------------------------
class Block(nn.Module):
    def __init__(self, d=256):
        super().__init__()
        self.up, self.down = nn.Linear(d, 4 * d), nn.Linear(4 * d, d)

    def forward(self, x):
        return x + self.down(F.gelu(self.up(x)))


torch.manual_seed(1)
blocks = nn.ModuleList([Block() for _ in range(3)])
ids_emb = torch.randn(4, 64, 256)          # stand-in for embeddings of token ids: no requires_grad


def run(use_ckpt, ckpt_fn=None, inside=False):
    for p in blocks.parameters():
        p.grad = None
    with rt.npu_mode():
        h = ids_emb
        for blk in blocks:
            h = (ckpt_fn or rt.checkpoint)(blk, h) if use_ckpt else blk(h)
        loss = h.pow(2).mean()
        if inside:
            loss.backward()
    if not inside:
        loss.backward()
    return torch.cat([p.grad.flatten() for p in blocks.parameters()]), loss.item()


g_ref, l_ref = run(False)
g_in, _ = run(True, inside=True)
g_out, l_out = run(True, inside=False)
check("rt.checkpoint: backward inside npu_mode()", rel(g_in, g_ref) < 1e-3, f"rel {rel(g_in, g_ref):.1e}")
check("rt.checkpoint: backward OUTSIDE npu_mode()", rel(g_out, g_ref) < 1e-3, f"rel {rel(g_out, g_ref):.1e}")
plain = lambda f, *a: tuc.checkpoint(f, *a, use_reentrant=False)  # noqa: E731
g_pin, _ = run(True, plain, inside=True)
check("plain torch.utils.checkpoint, backward inside npu_mode()", rel(g_pin, g_ref) < 1e-3,
      f"rel {rel(g_pin, g_ref):.1e}")
rt.enable()                                  # global mode: any checkpoint function, backward anywhere
try:
    for p_ in blocks.parameters():
        p_.grad = None
    h = ids_emb
    for blk in blocks:
        h = tuc.checkpoint(blk, h, use_reentrant=False)
    h.pow(2).mean().backward()
    g_en = torch.cat([p_.grad.flatten() for p_ in blocks.parameters()])
finally:
    rt.disable()
check("plain torch.utils.checkpoint with rk3588_torch.enable()", rel(g_en, g_ref) < 1e-3, f"rel {rel(g_en, g_ref):.1e}")
try:
    run(True, plain, inside=False)
    print("info  plain torch checkpoint, backward outside npu_mode(): worked on this PyTorch")
except Exception as e:  # noqa: BLE001
    print(f"info  plain torch checkpoint, backward outside npu_mode(): {type(e).__name__}: {str(e)[:90]!r}")
try:
    ids = torch.randn(4, 64, 256)
    with rt.npu_mode():
        y = ids
        for blk in blocks:
            y = rt.checkpoint(blk, y, use_reentrant=True)
        loss = y.pow(2).mean()
    loss.backward()
    got = [p.grad is not None and bool(p.grad.abs().sum() > 0) for p in blocks.parameters()]
    print(f"info  use_reentrant=True with an input that doesn't require grad: grads on {sum(got)}/{len(got)} params")
except Exception as e:  # noqa: BLE001
    print(f"info  use_reentrant=True, no-grad input: {type(e).__name__}: {str(e)[:90]!r}")

# ---------------------------------------------------------------------------------------------------
# weight cache sizing warning, duty throttle
# ---------------------------------------------------------------------------------------------------
wcache.clear()
old_cap = wcache.CAP_MB
wcache.CAP_MB = 1
big = torch.randn(1024, 1024)
with warnings.catch_warnings(record=True) as wl:
    warnings.simplefilter("always")
    for _ in range(2):
        wcache.lookup(big, "t")
    for _ in range(2):
        wcache.lookup(torch.randn(1024, 1024), "t2")
msgs = [str(w.message) for w in wl if "weight cache" in str(w.message)]
check("weight cache over its cap: warns once with the fix", len(msgs) == 1 and "RKNPU_WCACHE_MB" in msgs[0],
      f"{len(msgs)} warning(s)")
wcache.CAP_MB = old_cap
wcache.clear()

slept = []
real_sleep = _lib.time.sleep
_lib.time.sleep = slept.append
prev = _lib.set_duty(0.5)
_lib._throttle(0.2)
_lib.set_duty(1.0)
_lib._throttle(0.2)
_lib.set_duty(0.25)
_lib._throttle(0.1)
_lib.set_duty(prev)
_lib.time.sleep = real_sleep
check("duty cycle: pause = busy * (1/duty - 1), none at duty 1", [round(s, 3) for s in slept] == [0.2, 0.3],
      f"slept {slept}")

# ---- grouped-query attention on the NPU path, and bf16 ops that stay on the CPU --------------------
from rk3588_torch import mode as _mode  # noqa: E402

torch.manual_seed(5)
q = torch.randn(1, 8, 64, 32, requires_grad=True)
k = torch.randn(1, 2, 64, 32, requires_grad=True)
v = torch.randn(1, 2, 64, 32, requires_grad=True)
ref_o = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
(ref_o ** 2).sum().backward()
ref_g = [t.grad.clone() for t in (q, k, v)]
for t in (q, k, v):
    t.grad = None
old_min = ops.MIN_MACS
ops.MIN_MACS = 1                              # force the NPU route for this small shape
n0 = ops.stats()["npu_calls"]
with rt.npu_mode(calibrate=False):
    o = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
    (o ** 2).sum().backward()
ops.MIN_MACS = old_min
check("GQA attention (enable_gqa=True) runs on the NPU", ops.stats()["npu_calls"] > n0,
      f"{ops.stats()['npu_calls'] - n0} NPU calls")
check("GQA attention: output", rel(o, ref_o) < 5e-3, f"rel {rel(o, ref_o):.1e}")
check("GQA attention: q/k/v gradients", all(rel(t.grad, g) < 1e-2 for t, g in zip((q, k, v), ref_g)),
      " ".join(f"{rel(t.grad, g):.1e}" for t, g in zip((q, k, v), ref_g)))

# small bf16 ops that stay on the CPU are computed in fp32 (the A76 has no bf16 arithmetic)
x = torch.randn(4, 16, 32, dtype=torch.bfloat16, requires_grad=True)
w = torch.randn(24, 32, dtype=torch.bfloat16)
bb = torch.randn(24, dtype=torch.bfloat16)
qb = torch.randn(1, 4, 16, 8, dtype=torch.bfloat16, requires_grad=True)
seen_dtypes = []
real_linear = F.linear
_mode._UPCAST = True
try:
    with rt.npu_mode(calibrate=False):
        # (below the NPU threshold -> CPU) the call must come back bf16 and match the fp32 reference
        y = F.linear(x, w, bb)
        ya = F.scaled_dot_product_attention(qb, qb.detach(), qb.detach(), is_causal=True)
        (y.float().sum() + ya.float().sum()).backward()
    yref = real_linear(x.detach().float(), w.float(), bb.float())
    check("bf16 linear on the CPU path: bf16 result, fp32-accurate", y.dtype == torch.bfloat16
          and rel(y.float(), yref) < 1e-2, f"rel {rel(y.float(), yref):.1e}")
    check("bf16 attention on the CPU path: bf16 result", ya.dtype == torch.bfloat16 and ya.shape == qb.shape)
    check("bf16 CPU path: gradients flow in bf16", x.grad is not None and x.grad.dtype == torch.bfloat16
          and qb.grad is not None and torch.isfinite(x.grad.float()).all().item())
    check("fp32 op is left alone", _mode._upcast_call(F.linear, (x.detach().float(), w.float(), None), {}) is None)
    check("mixed dtypes are left alone", _mode._upcast_call(F.linear, (x.detach(), w.float(), None), {}) is None)
    _mode._UPCAST = False
    check("RKNPU_CPU_UPCAST=0 switches it off", _mode._upcast_call(F.linear, (x.detach(), w, None), {}) is None)
    check("...and force=True overrides", _mode._upcast_call(F.linear, (x.detach(), w, None), {}, force=True)
          is not None)
finally:
    _mode._UPCAST = _mode._UPCAST_DEFAULT

print("ALL PASS" if failures == 0 else f"{failures} FAILED")
sys.exit(1 if failures else 0)
