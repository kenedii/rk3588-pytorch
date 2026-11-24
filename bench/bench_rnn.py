"""LSTM/GRU training step: PyTorch vs rk3588_torch (NPU input projections + hand-written recurrence).

    python3 bench/bench_rnn.py
"""
import copy
import os
import sys
import time

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import rk3588_torch as rt  # noqa: E402
from rk3588_torch import rnn  # noqa: E402


def step(m, x):
    o, _ = m(x)
    o.square().mean().backward()


def bench(m, x, n=3):
    step(m, x)
    t = time.perf_counter()
    for _ in range(n):
        step(m, x)
    return (time.perf_counter() - t) / n * 1e3


print(f"{'cell':5} {'T':>4} {'B':>4} {'in':>5} {'H':>5} {'L':>2}  {'torch ms':>9} {'npu ms':>9} {'npu+rec ms':>10}")
for cell in (nn.LSTM, nn.GRU):
    for (T, B, I, H, L) in ((64, 32, 256, 512, 2), (128, 16, 512, 512, 1), (32, 64, 1024, 1024, 1)):
        ref = cell(I, H, L)
        x = torch.randn(T, B, I)
        t0 = bench(ref, x)
        m = rt.convert(copy.deepcopy(ref))
        with rt.npu_mode():
            rnn.REC_NPU = "0"
            t1 = bench(m, x)
            rnn.REC_NPU = "1"
            t2 = bench(m, x)
            rnn.REC_NPU = "auto"
        print(f"{cell.__name__:5} {T:4} {B:4} {I:5} {H:5} {L:2}  {t0:9.1f} {t1:9.1f} {t2:10.1f}", flush=True)
