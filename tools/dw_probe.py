"""Probe the NPU's depthwise convolution mode (CNA/DPU CONV_MODE 3 + CORE DW_EN) and the DPU
ReLU bit, using rk_conv_raw. Usage: python3 tools/dw_probe.py [dw|relu|all]"""
import ctypes
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rk3588_torch._lib import default_npu  # noqa: E402


class Cfg(ctypes.Structure):
    _fields_ = [(n, ctypes.c_int) for n in ("h", "w", "c", "n", "kh", "kw", "sy", "sx", "pt", "pl", "oh", "ow",
                                            "grains", "dbanks", "layout", "dw", "relu")]


npu = default_npu()
npu.lib.rk_conv_raw.argtypes = [ctypes.c_void_p, ctypes.POINTER(Cfg), ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
npu.lib.rk_conv_raw.restype = ctypes.c_int


def nc1hwc2(x, Cp):
    C, H, W = x.shape
    xp = torch.zeros(Cp, H, W)
    xp[:C] = x
    return xp.reshape(Cp // 8, 8, H, W).permute(0, 2, 3, 1).contiguous().half()


def dw_weights(w, layout, Cp):
    """w: (C, 1, kh, kw) -> fp16 in a candidate layout"""
    C, _, kh, kw = w.shape
    wp = torch.zeros(Cp, kh, kw)
    wp[:C] = w[:, 0]
    a = int(layout[1:3]) if layout[0] == "c" else 32
    if layout.startswith("c"):     # [Cp/a][ky][kx][a]
        t = wp.reshape(Cp // a, a, kh, kw).permute(0, 2, 3, 1)
    else:                          # "x32": [Cp/32][kx][ky][32]
        t = wp.reshape(Cp // 32, 32, kh, kw).permute(0, 3, 2, 1)
    return t.contiguous().half()


def unpack(out, n, oh, ow):
    return out.reshape(n // 4, oh, ow, 4).permute(1, 2, 0, 3).reshape(oh, ow, n)


def run_dw(H, W, C, k, s=1, p=0, layouts=("c32", "c16", "c08", "x32"), nmul=(1,)):
    torch.manual_seed(0)
    Cp = (C + 31) // 32 * 32
    x = torch.randn(C, H, W).half().float()
    w = torch.randn(C, 1, k, k).half().float()
    ref = F.conv2d(x[None].double(), w.double(), stride=s, padding=p, groups=C)[0].permute(1, 2, 0)
    oh, ow = ref.shape[:2]
    res = {}
    for lay in layouts:
        for m in nmul:
            n = Cp * m
            wb = dw_weights(w, lay, Cp)
            out = torch.empty(n * oh * ow, dtype=torch.float32)
            cfg = Cfg(H, W, Cp, n, k, k, s, s, p, p, oh, ow, 0, 0, 1, 1, 0)
            r = npu.lib.rk_conv_raw(npu.ctx, ctypes.byref(cfg), nc1hwc2(x, Cp).data_ptr(), wb.data_ptr(), out.data_ptr())
            key = f"{lay}/n{m}"
            if r != 0:
                res[key] = "FAIL " + npu.lib.rk_last_error(npu.ctx).decode()[:50]
                continue
            unw = (out.view(torch.int32) == 0x7f7f7f7f).float().mean().item()
            got = unpack(out, n, oh, ow)[:, :, :C].double()
            e = ((got - ref).norm() / ref.norm()).item()
            res[key] = f"{e:.1e}(unw {unw * 100:.0f}%)"
            if os.environ.get("DUMP"):
                print("  got", got[0, 0, :8].tolist(), "\n  ref", ref[0, 0, :8].tolist())
    print(f"DW H{H} W{W} C{C} k{k} s{s} p{p}: " + "  ".join(f"{a}={b}" for a, b in res.items()), flush=True)


def run_relu(H=8, W=8, C=32, N=16, k=3, p=1):
    torch.manual_seed(0)
    x = torch.randn(C, H, W).half().float()
    w = torch.randn(N, C, k, k).half().float()
    ref = F.conv2d(x[None].double(), w.double(), padding=p)[0].permute(1, 2, 0)
    oh, ow = ref.shape[:2]
    Cp = (C + 31) // 32 * 32
    wp = torch.zeros(N, Cp, k, k)
    wp[:, :C] = w
    wb = wp.reshape(N // 16, 16, Cp // 32, 32, k, k).permute(0, 2, 4, 5, 1, 3).contiguous().half()
    for relu in (0, 1):
        out = torch.empty(N * oh * ow, dtype=torch.float32)
        cfg = Cfg(H, W, Cp, N, k, k, 1, 1, p, p, oh, ow, 0, 0, 1, 0, relu)
        r = npu.lib.rk_conv_raw(npu.ctx, ctypes.byref(cfg), nc1hwc2(x, Cp).data_ptr(), wb.data_ptr(), out.data_ptr())
        if r != 0:
            print("relu", relu, "FAIL", npu.lib.rk_last_error(npu.ctx).decode())
            continue
        got = unpack(out, N, oh, ow).double()
        tgt = ref.clamp_min(0) if relu else ref
        print(f"relu={relu}: err vs {'relu(ref)' if relu else 'ref'} {((got - tgt).norm() / tgt.norm()).item():.1e}, "
              f"negatives {(got < 0).sum().item()}", flush=True)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("relu", "all"):
        run_relu()
    if which in ("dw", "all"):
        run_dw(8, 8, 32, 3, p=1, nmul=(1, 2))
        run_dw(8, 8, 64, 3, p=1)
        run_dw(10, 12, 32, 3)
