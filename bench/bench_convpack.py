"""Where does a native NPU convolution spend its time? pack (CPU: NCHW fp32 -> NC1HWC2 fp16), sync + NPU
hardware, and unpack (CPU: NC1HWC2 fp32 -> NCHW fp32 + bias/ReLU), for typical CNN layers, forward and as the
dX convolution of the backward pass.

    python3 bench/bench_convpack.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rk3588_torch import ops  # noqa: E402
from rk3588_torch._lib import default_npu  # noqa: E402

torch.manual_seed(0)
npu = default_npu()
LAYERS = [  # (N, C, H, O, k, stride, pad)
    (32, 64, 56, 64, 3, 1, 1), (32, 128, 28, 128, 3, 1, 1), (32, 256, 14, 256, 3, 1, 1), (32, 512, 7, 512, 3, 1, 1),
    (32, 64, 56, 128, 1, 1, 0), (32, 3, 64, 64, 7, 2, 3), (16, 32, 64, 32, 3, 1, 1), (8, 128, 64, 128, 3, 1, 1),
]
print(f"{'layer':32} {'total':>7} {'pack':>7} {'npu':>7} {'unpack':>7} {'other':>7}  {'pack+unpack %':>13}  (ms)")
tot = {"total_us": 0, "pack_us": 0, "unpack_us": 0, "submit_us": 0}
for (N, C, H, O, k, st, pd) in LAYERS:
    x = torch.randn(N, C, H, H)
    w = torch.randn(O, C, k, k) * 0.05
    if max(pd, 0) > ops.HW_PAD_MAX:
        x = torch.nn.functional.pad(x, (pd,) * 4)
        pd = 0
    ops._npu_conv(x, w, (st, st), (pd, pd), "fp16", 1, None, True, 0)   # warm
    S = []
    for _ in range(3):
        ops._npu_conv(x, w, (st, st), (pd, pd), "fp16", 1, None, True, 0)
        S.append(npu.stats())
    s = {k_: sorted(v[k_] for v in S)[1] / 1e3 for k_ in ("total_us", "pack_us", "submit_us", "unpack_us")}
    other = s["total_us"] - s["pack_us"] - s["submit_us"] - s["unpack_us"]
    for k_ in tot:
        tot[k_] += s[k_]
    print(f"N{N} {C}->{O} {H}x{H} k{k} s{st}".ljust(32) +
          f" {s['total_us']:7.1f} {s['pack_us']:7.1f} {s['submit_us']:7.1f} {s['unpack_us']:7.1f} {other:7.1f}"
          f"  {100 * (s['pack_us'] + s['unpack_us']) / s['total_us']:12.0f}%")
t = dict(tot)
print(f"{'sum':32} {t['total_us']:7.1f} {t['pack_us']:7.1f} {t['submit_us']:7.1f} {t['unpack_us']:7.1f}")
