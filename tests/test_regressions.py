"""Regression tests for bugs found by training many architectures on the board:
SDPA causal mask with fewer queries than keys, addmm/baddbmm out=, the weight cache missing
.data edits and crashing on inference tensors, and non-finite inputs.

    python3 tests/test_regressions.py
"""
import os
import sys
import traceback
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import rk3588_torch as rt  # noqa: E402
from rk3588_torch import ops, wcache  # noqa: E402

assert rt.npu_available(), rt.npu_error()
ops.MIN_MACS = 0
ops.ATTENTION = "npu"   # exercise the NPU attention path (the default "auto" leaves small heads to the CPU)
warnings.simplefilter("ignore")
torch.manual_seed(0)
failures = 0


def rel(a, b):
    return ((a.double() - b.double()).norm() / (b.double().norm() + 1e-300)).item()


def case(fn):
    global failures
    try:
        ok, detail = fn()
    except Exception as e:  # noqa: BLE001
        ok, detail = False, repr(e)
        traceback.print_exc(limit=3)
    failures += not ok
    print(f"{'PASS' if ok else 'FAIL'}  {fn.__name__:<44} {detail}", flush=True)
    return fn


@case
def sdpa_causal_L_ne_S():
    """is_causal with fewer queries than keys (PyTorch masks top-left aligned)."""
    q, k, v = torch.randn(2, 4, 16, 64), torch.randn(2, 4, 48, 64), torch.randn(2, 4, 48, 64)
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    with rt.npu_mode():
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    return rel(out, ref) < 5e-3, f"rel {rel(out, ref):.2e}"


@case
def sdpa_float_mask_and_scale():
    q, k, v = torch.randn(2, 4, 64, 32), torch.randn(2, 4, 64, 32), torch.randn(2, 4, 64, 32)
    m = torch.randn(64, 64)
    ref = F.scaled_dot_product_attention(q, k, v, attn_mask=m, scale=0.3)
    with rt.npu_mode():
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=m, scale=0.3)
    return rel(out, ref) < 5e-3, f"rel {rel(out, ref):.2e}"


@case
def addmm_out_kwarg():
    inp, a, b = torch.randn(256), torch.randn(128, 256), torch.randn(256, 256)
    buf = torch.zeros(128, 256)
    with rt.npu_mode():
        r = torch.addmm(inp, a, b, out=buf)
    ref = torch.addmm(inp, a, b)
    return rel(buf, ref) < 5e-3, f"out buffer rel {rel(buf, ref):.2e} (returned rel {rel(r, ref):.2e})"


@case
def matmul_out_kwarg():
    a, b = torch.randn(128, 256), torch.randn(256, 256)
    buf = torch.zeros(128, 256)
    with rt.npu_mode():
        torch.matmul(a, b, out=buf)
    return rel(buf, a @ b) < 5e-3, f"rel {rel(buf, a @ b):.2e}"


@case
def wcache_stale_after_data_edit():
    """Eval twice (weight gets cached), edit a few rows via .data (no version bump), eval again."""
    lin = nn.Linear(1024, 1024)
    x = torch.randn(64, 1024)
    with torch.no_grad(), rt.npu_mode():
        for _ in range(3):
            lin(x)
        lin.weight.data[1:3] += 1.0         # rows the 16-row fingerprint sample doesn't look at
        out = lin(x)
    ref = F.linear(x, lin.weight, lin.bias)
    return rel(out, ref) < 5e-3, f"rel {rel(out, ref):.2e}  wcache {wcache.info()}"


@case
def wcache_load_state_dict():
    lin = nn.Linear(1024, 1024)
    x = torch.randn(64, 1024)
    with torch.no_grad(), rt.npu_mode():
        for _ in range(3):
            lin(x)
        lin.load_state_dict(nn.Linear(1024, 1024).state_dict())
        out = lin(x)
    ref = F.linear(x, lin.weight, lin.bias)
    return rel(out, ref) < 5e-3, f"rel {rel(out, ref):.2e}"


@case
def inference_mode_model():
    """Weights created inside torch.inference_mode() are inference tensors (no version counter)."""
    with torch.inference_mode():
        lin = nn.Linear(1024, 1024)
        x = torch.randn(64, 1024)
        with rt.npu_mode():
            outs = [lin(x) for _ in range(3)]
        ref = F.linear(x, lin.weight, lin.bias)
    return rel(outs[-1], ref) < 5e-3, f"rel {rel(outs[-1], ref):.2e}"


@case
def inference_mode_conv():
    with torch.inference_mode():
        c = nn.Conv2d(64, 64, 3, padding=1)
        x = torch.randn(4, 64, 32, 32)
        with rt.npu_mode():
            outs = [c(x) for _ in range(3)]
        ref = F.conv2d(x, c.weight, c.bias, padding=1)
    return rel(outs[-1], ref) < 5e-3, f"rel {rel(outs[-1], ref):.2e}"


@case
def convert_inference_mode():
    with torch.inference_mode():
        m = rt.convert(nn.Sequential(nn.Linear(512, 1024), nn.ReLU(), nn.Linear(1024, 512)))
        x = torch.randn(64, 512)
        outs = [m(x) for _ in range(3)]
        ref = F.linear(F.relu(F.linear(x, m[0].weight, m[0].bias)), m[2].weight, m[2].bias)
    return rel(outs[-1], ref) < 5e-3, f"rel {rel(outs[-1], ref):.2e}"


@case
def nonfinite_inputs_propagate():
    """The NPU turns inf/nan inputs into finite garbage; they must reach the CPU instead."""
    bad_cases = []
    for val in (float("inf"), float("nan")):
        for p in ("fp16", "bf16", "fp16_out16", "int8", "fp16x3"):
            a, b = torch.randn(256, 256), torch.randn(256, 256)
            a[3, 4] = val
            out = ops.gemm(a, b, p)
            if not torch.equal(torch.isfinite(out), torch.isfinite(a @ b)):
                bad_cases.append(f"{p}/{val}")
        x, w = torch.randn(2, 32, 16, 16), torch.randn(32, 32, 3, 3)
        x[0, 0, 5, 5] = val
        with rt.npu_mode():
            y = F.conv2d(x, w, padding=1)
        if not torch.equal(torch.isfinite(y), torch.isfinite(F.conv2d(x, w, padding=1))):
            bad_cases.append(f"conv/{val}")
    return not bad_cases, "mismatched: " + (", ".join(bad_cases) or "none")


@case
def baddbmm_out_kwarg():
    inp, a, b = torch.randn(8, 128, 128), torch.randn(8, 128, 256), torch.randn(8, 256, 128)
    buf = torch.zeros(8, 128, 128)
    with rt.npu_mode():
        torch.baddbmm(inp, a, b, out=buf)
    ref = torch.baddbmm(inp, a, b)
    return rel(buf, ref) < 5e-3, f"rel {rel(buf, ref):.2e}"


@case
def conv_large_padding_every_image():
    """Padding >= 2 made the NPU compute some images of a batch wrong (first output rows of the first kernel
    group, run-to-run different; plus every second image when CBUF weights were reused). Every image is
    checked separately, several times, since the error was intermittent."""
    ops.VERIFY = "off"
    bad = []
    try:
        for (C, H, k, p) in [(96, 5, 3, 2), (96, 13, 3, 2), (128, 9, 3, 2), (160, 6, 5, 3), (192, 7, 5, 4),
                             (64, 13, 3, 2), (96, 13, 3, 3)]:
            for rep in range(3):
                x, w = torch.randn(6, C, H, H), torch.randn(32, C, k, k)
                y = ops.conv(x, w, 1, p)
                r = F.conv2d(x.double(), w.double(), padding=p)
                errs = [rel(y[n], r[n]) for n in range(6)]
                if max(errs) > 5e-3:
                    bad.append(f"C{C} {H}x{H} k{k} p{p}: {max(errs):.1e}")
                    break
    finally:
        ops.VERIFY = "first"
    return not bad, "; ".join(bad) or "7 geometries x 3 runs, every image exact to fp16"


@case
def conv_small_feature_maps_on_npu():
    """Feature maps under 5 rows (deep ResNet layers at small resolution) used to fall back to the CPU, forward
    and backward. They now run as unfold + NPU GEMM."""
    x = torch.randn(16, 256, 4, 4, requires_grad=True)
    w = torch.randn(512, 256, 3, 3, requires_grad=True)
    ops.stats(reset=True)
    with rt.npu_mode():
        y = F.conv2d(x, w, padding=1, stride=2)
        y.square().sum().backward()
    xr, wr = x.detach().double().requires_grad_(), w.detach().double().requires_grad_()
    yr = F.conv2d(xr, wr, padding=1, stride=2)
    yr.square().sum().backward()
    e = max(rel(y, yr), rel(x.grad, xr.grad), rel(w.grad, wr.grad))
    st = ops.stats()
    return e < 5e-3 and st["npu_calls"] >= 3, f"max rel {e:.1e}, NPU calls {st['npu_calls']}"


@case
def multihead_attention_on_npu():
    """nn.MultiheadAttention / nn.Transformer* layers: F.multi_head_attention_forward used to run entirely on
    the CPU (npu_mode can't see inside it). Self- and cross-attention, bool/float masks, padding mask, causal."""
    torch.manual_seed(1)
    E, H = 256, 8
    errs, calls = [], []
    for kind in ("self+pad", "self+causal", "cross", "encoder_layer", "decoder_layer"):
        if kind == "encoder_layer":
            mod = nn.TransformerEncoderLayer(E, H, 512, dropout=0.0, batch_first=True)
        elif kind == "decoder_layer":
            mod = nn.TransformerDecoderLayer(E, H, 512, dropout=0.0, batch_first=True)
        else:
            mod = nn.MultiheadAttention(E, H, batch_first=True)
        x = torch.randn(8, 96, E, requires_grad=True)
        mem = torch.randn(8, 64, E)
        pad = torch.arange(96).expand(8, 96) >= torch.randint(48, 96, (8, 1))
        causal = nn.Transformer.generate_square_subsequent_mask(96)

        def run(m):
            if kind == "self+pad":
                return m(x, x, x, key_padding_mask=pad, need_weights=False)[0]
            if kind == "self+causal":
                return m(x, x, x, attn_mask=causal, need_weights=False)[0]
            if kind == "cross":
                return m(x, mem, mem, need_weights=False)[0]
            if kind == "encoder_layer":
                return m(x, src_key_padding_mask=pad)
            return m(x, mem, tgt_mask=causal, tgt_is_causal=True)
        proj = torch.randn(96, E)      # a random projection: sum() of a LayerNorm output has ~zero gradients
        ref = run(mod)
        gref = torch.autograd.grad((ref * proj).sum(), [x] + list(mod.parameters()))
        ops.stats(reset=True)
        with rt.npu_mode():
            out = run(mod)
            g = torch.autograd.grad((out * proj).sum(), [x] + list(mod.parameters()))
        calls.append(ops.stats()["npu_calls"])
        errs.append(max([rel(out, ref)] + [rel(a, b) for a, b in zip(g, gref)]))
    # whole layers (LayerNorm, FFN, every parameter's gradient) accumulate more fp16 rounding than one MHA
    tols = [5e-3, 5e-3, 5e-3, 2.5e-2, 2.5e-2]
    ok = all(e < t for e, t in zip(errs, tols)) and min(calls) >= 4
    return ok, "max rel " + ", ".join(f"{e:.1e}" for e in errs) + f"; NPU calls {calls}"


print("ALL PASS" if failures == 0 else f"{failures} FAILED")
sys.exit(1 if failures else 0)
