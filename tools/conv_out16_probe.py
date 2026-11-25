"""Can the NPU write a convolution's output as fp16 in NC1HWC2 (8-channel atoms), the layout the next convolution
reads? Runs rk_conv_raw with an fp16 output type and decodes the bytes under candidate layouts / surface-add
multipliers, against a CPU reference.

    python3 tools/conv_out16_probe.py
"""
import ctypes
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rk3588_torch._lib import default_npu  # noqa: E402


class Cfg(ctypes.Structure):
    _fields_ = [(n, ctypes.c_int) for n in ("h", "w", "c", "n", "kh", "kw", "sy", "sx", "pt", "pl", "oh", "ow",
                                            "grains", "dbanks", "layout", "dw", "relu", "out16", "surf_add")]


npu = default_npu()
npu.lib.rk_conv_raw.argtypes = [ctypes.c_void_p, ctypes.POINTER(Cfg), ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
npu.lib.rk_conv_raw.restype = ctypes.c_int


def nc1hwc2(x, Cp):
    C, H, W = x.shape
    xp = torch.zeros(Cp, H, W)
    xp[:C] = x
    return xp.reshape(Cp // 8, 8, H, W).permute(0, 2, 3, 1).contiguous().half()


def weights(w, Cp):
    N, C, kh, kw = w.shape
    wp = torch.zeros(N, Cp, kh, kw)
    wp[:, :C] = w
    return wp.reshape(N // 16, 16, Cp // 32, 32, kh, kw).permute(0, 2, 4, 5, 1, 3).contiguous().half()


def run(H=8, W=8, C=32, N=32, k=3, p=1, relu=0):
    torch.manual_seed(0)
    x = (torch.randn(C, H, W) * 0.5).half().float()
    w = (torch.randn(N, C, k, k) * 0.2).half().float()
    ref = F.conv2d(x[None].double(), w.double(), padding=p)[0]                     # N, oh, ow
    if relu:
        ref = ref.clamp_min(0)
    oh, ow = ref.shape[1:]
    Cp = (C + 31) // 32 * 32
    xin, wb = nc1hwc2(x, Cp), weights(w, Cp)
    print(f"conv {C}->{N} {H}x{W} k{k} p{p} relu{relu}  (ref range {ref.min():.2f}..{ref.max():.2f})")
    for surf_add in (0, 2, 4, 8, 16):
        out = torch.zeros(N * oh * ow, dtype=torch.float16)
        cfg = Cfg(H, W, Cp, N, k, k, 1, 1, p, p, oh, ow, 0, 0, 1, 0, relu, 1, surf_add)
        r = npu.lib.rk_conv_raw(npu.ctx, ctypes.byref(cfg), xin.data_ptr(), wb.data_ptr(), out.data_ptr())
        if r != 0:
            print(f"  surf_add={surf_add}: FAIL {npu.lib.rk_last_error(npu.ctx).decode()[:70]}")
            continue
        raw = out.view(torch.int16)
        unwritten = (raw == 0x7f7f).float().mean().item()
        f = out.float()
        cand = {
            "[N/8][oh][ow][8]": f.reshape(N // 8, oh, ow, 8).permute(0, 3, 1, 2).reshape(N, oh, ow),
            "[oh][ow][N]": f.reshape(oh, ow, N).permute(2, 0, 1),
            "[N/4][oh][ow][4] (fp32 atom order)": f.reshape(N // 4, oh, ow, 4).permute(0, 3, 1, 2).reshape(N, oh, ow),
        }
        errs = {name: ((v.double() - ref).norm() / ref.norm()).item() for name, v in cand.items()}
        best = min(errs, key=errs.get)
        print(f"  surf_add={surf_add:2d}: unwritten {unwritten * 100:3.0f}%  best {best} err {errs[best]:.2e}   "
              + "  ".join(f"{n.split(' ')[0]} {e:.1e}" for n, e in errs.items()))


run()
run(relu=1)
run(H=14, W=14, C=64, N=64, k=3, p=1)
run(H=12, W=12, C=32, N=48, k=1, p=0)
