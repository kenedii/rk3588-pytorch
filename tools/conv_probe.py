"""Probe native NPU convolution: run rk_conv_raw with several candidate weight layouts and
compare against F.conv2d. Usage: python3 tools/conv_probe.py"""
import ctypes
import itertools
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rk3588_torch._lib import default_npu  # noqa: E402


class Cfg(ctypes.Structure):
    _fields_ = [(n, ctypes.c_int) for n in ("h", "w", "c", "n", "kh", "kw", "sy", "sx", "pt", "pl", "oh", "ow",
                                            "grains", "dbanks", "layout")]


npu = default_npu()
npu.lib.rk_conv_raw.argtypes = [ctypes.c_void_p, ctypes.POINTER(Cfg), ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
npu.lib.rk_conv_raw.restype = ctypes.c_int


def pack_weights(w, layout, G=16):
    """w: (N, C, kh, kw) -> fp16 bytes in a candidate NPU layout."""
    N, C, kh, kw = w.shape
    Np = (N + G - 1) // G * G
    Cp = (C + 31) // 32 * 32 if layout != "flat" else C
    wp = torch.zeros(Np, Cp, kh, kw)
    wp[:N, :C] = w
    if layout == "yx_c":      # [ng][ky][kx][cb][16][32]
        t = wp.reshape(Np // G, G, Cp // 32, 32, kh, kw).permute(0, 4, 5, 2, 1, 3)
    elif layout == "xy_c":    # [ng][kx][ky][cb][16][32]
        t = wp.reshape(Np // G, G, Cp // 32, 32, kh, kw).permute(0, 5, 4, 2, 1, 3)
    elif layout == "c_yx":    # [ng][cb][ky][kx][16][32]
        t = wp.reshape(Np // G, G, Cp // 32, 32, kh, kw).permute(0, 2, 4, 5, 1, 3)
    elif layout == "flat":    # per kernel flatten (ky, kx, c), blocks of 32: [ng][f/32][16][32]
        f = wp.permute(0, 2, 3, 1).reshape(Np, kh * kw * C)
        F_ = f.shape[1]
        Fp = (F_ + 31) // 32 * 32
        f2 = torch.zeros(Np, Fp)
        f2[:, :F_] = f
        t = f2.reshape(Np // G, G, Fp // 32, 32).permute(0, 2, 1, 3)
    return t.contiguous().to(torch.float16), Np


def run(H, W, C, N, k, s=1, p=0, layouts=("c_yx", "yx_c", "xy_c"), grains=0, nc=1):
    torch.manual_seed(0)
    x = torch.randn(1, C, H, W).half().float()
    w = torch.randn(N, C, k, k).half().float()
    ref = F.conv2d(x.double(), w.double(), stride=s, padding=p)[0].permute(1, 2, 0)   # oh, ow, N
    oh, ow = ref.shape[:2]
    Cp = (C + 31) // 32 * 32
    if nc:   # NC1HWC2, C2 = 8 fp16 channels per 16-byte atom
        xp = torch.zeros(Cp, H, W)
        xp[:C] = x[0]
        xin = xp.reshape(Cp // 8, 8, H, W).permute(0, 2, 3, 1).contiguous().half()
    else:
        xin = x[0].permute(1, 2, 0).contiguous().half()                              # H, W, C
    res = {}
    for lay in layouts:
        wb, Np = pack_weights(w, lay)
        out = torch.empty(oh, ow, Np, dtype=torch.float32)
        cfg = Cfg(H, W, Cp if nc else C, Np, k, k, s, s, p, p, oh, ow, grains, 0, nc)
        r = npu.lib.rk_conv_raw(npu.ctx, ctypes.byref(cfg), xin.data_ptr(), wb.data_ptr(), out.data_ptr())
        if r != 0:
            res[lay] = "FAIL " + npu.lib.rk_last_error(npu.ctx).decode()[:60]
            continue
        poison = (out.view(torch.int32) == 0x7f7f7f7f)
        if nc:   # output NC1HWC2 fp32: [Np/4][oh][ow][4]
            out = out.reshape(-1).reshape(Np // 4, oh, ow, 4).permute(1, 2, 0, 3).reshape(oh, ow, Np)
        got = out[:, :, :N].double()
        e = ((got - ref).norm() / ref.norm()).item()
        res[lay] = f"{e:.1e}(unwritten {poison.float().mean().item() * 100:.0f}%)"
        if os.environ.get("DUMP"):
            print("   out[0,:4,:4]", out[0, :4, :4].tolist(), "\n   ref[0,:4,:4]", ref[0, :4, :4].tolist())
    print(f"H{H} W{W} C{C} N{N} k{k} s{s} p{p} -> oh{oh} ow{ow}: " + "  ".join(f"{k_}={v}" for k_, v in res.items()),
          flush=True)
    return res


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "basic"
    if which == "w1":
        run(64, 1, 32, 16, 1, layouts=("yx_c",))      # identical to a 64x32x16 matmul
        run(8, 2, 32, 16, 1, layouts=("yx_c",))
        run(4, 4, 32, 16, 1, layouts=("yx_c",))
    if which == "basic":
        run(8, 8, 32, 16, 1, layouts=("yx_c",))       # 1x1 conv with width > 1
        run(8, 8, 32, 16, 3)                         # 3x3, no pad
        run(8, 8, 32, 16, 3, p=1)                    # 3x3, pad 1
        run(9, 9, 32, 16, 3, s=2, p=1)               # stride 2
        run(16, 12, 64, 32, 3, p=1)
    elif which == "channels":
        for C in (8, 16, 24, 48):
            run(8, 8, C, 16, 3, p=1, layouts=("yx_c", "flat"))
