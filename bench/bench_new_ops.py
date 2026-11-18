"""Timing for the newer NPU paths: fused bias/ReLU, the packed-weight cache (inference and
LoRA fine-tuning), conv_transpose2d, grouped conv, conv1d, einsum and a MobileNet-style block.

    python3 bench/bench_new_ops.py [--threads 3]

Each line: median wall time per call (or per training step) on the CPU (PyTorch fp32,
same thread count) vs through rk3588_torch (fp16 unless noted).
"""
import argparse
import copy
import os
import statistics
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import rk3588_torch as rt  # noqa: E402
from rk3588_torch import ops, wcache  # noqa: E402


def timeit(fn, reps=7, warmup=2):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


def row(name, t_cpu, t_npu, note=""):
    print(f"{name:<52} CPU {t_cpu * 1e3:8.1f} ms   NPU {t_npu * 1e3:8.1f} ms   {t_cpu / t_npu:5.2f}x  {note}",
          flush=True)


def train_step(model, x, y, npu, opt):
    def step():
        if npu:
            with rt.npu_mode():
                loss = F.mse_loss(model(x), y)
                opt.zero_grad()
                loss.backward()
        else:
            loss = F.mse_loss(model(x), y)
            opt.zero_grad()
            loss.backward()
        opt.step()
    return step


def bench_train(name, make, x, reps=5):
    torch.manual_seed(0)
    m_cpu = make()
    with torch.no_grad():
        y = torch.randn_like(m_cpu(x))
    m_npu = rt.convert(copy.deepcopy(m_cpu))
    t_cpu = timeit(train_step(m_cpu, x, y, False, torch.optim.SGD(m_cpu.parameters(), lr=1e-3)), reps)
    t_npu = timeit(train_step(m_npu, x, y, True, torch.optim.SGD(m_npu.parameters(), lr=1e-3)), reps)
    row(name + " (train step)", t_cpu, t_npu)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=3)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    print(f"threads={args.threads}  precision={ops.get_precision()}  threshold={ops.threshold() / 1e6:.0f} MMAC\n")

    # --- fused bias + ReLU ---------------------------------------------------------------
    x = torch.randn(256, 2048)
    lin = nn.Linear(2048, 2048)
    def unfused():
        with rt.npu_mode():
            return F.relu(ops.linear(x, lin.weight) + lin.bias)
    def fused():
        with rt.npu_mode():
            return ops.linear(x, lin.weight, lin.bias, "relu")
    with torch.no_grad():
        t_cpu = timeit(lambda: F.relu(lin(x)))
        wcache.set_enabled(False)
        t_un, t_fu = timeit(unfused), timeit(fused)
        wcache.set_enabled(True)
    row("linear 256x2048x2048 + bias + ReLU, unfused", t_cpu, t_un)
    row("linear 256x2048x2048 + bias + ReLU, fused", t_cpu, t_fu, f"(fusion saves {(t_un - t_fu) * 1e3:.1f} ms)")

    # --- packed-weight cache: inference -----------------------------------------------------
    torch.manual_seed(0)
    mlp = nn.Sequential(nn.Linear(1024, 4096), nn.ReLU(), nn.Linear(4096, 4096), nn.ReLU(), nn.Linear(4096, 1024))
    mlp.eval()
    xb = torch.randn(32, 1024)
    npu_mlp = rt.convert(copy.deepcopy(mlp))
    with torch.no_grad():
        t_cpu = timeit(lambda: mlp(xb))
        wcache.set_enabled(False)
        t_nc = timeit(lambda: npu_mlp(xb))
        wcache.set_enabled(True)
        t_c = timeit(lambda: npu_mlp(xb))
    row("MLP inference batch 32 (42M params), no weight cache", t_cpu, t_nc)
    row("MLP inference batch 32 (42M params), weight cache", t_cpu, t_c, f"({t_nc / t_c:.2f}x from caching)")

    # --- packed-weight cache: LoRA fine-tuning ------------------------------------------------
    class LoRALinear(nn.Module):
        def __init__(self, d_in, d_out, r=16):
            super().__init__()
            self.base = nn.Linear(d_in, d_out)
            self.base.requires_grad_(False)
            self.a = nn.Parameter(torch.randn(r, d_in) * 0.01)
            self.b = nn.Parameter(torch.zeros(d_out, r))

        def forward(self, x):
            return self.base(x) + F.linear(F.linear(x, self.a), self.b)

    def lora_net():
        return nn.Sequential(LoRALinear(1024, 4096), nn.GELU(), LoRALinear(4096, 4096), nn.GELU(),
                             LoRALinear(4096, 1024))
    xl = torch.randn(128, 1024)
    torch.manual_seed(0)
    m_cpu = lora_net()
    y = torch.randn(128, 1024)
    m_npu = rt.convert(copy.deepcopy(m_cpu))
    t_cpu = timeit(train_step(m_cpu, xl, y, False, torch.optim.SGD([p for p in m_cpu.parameters() if p.requires_grad],
                                                                   lr=1e-3)), 5)
    opt = torch.optim.SGD([p for p in m_npu.parameters() if p.requires_grad], lr=1e-3)
    wcache.set_enabled(False)
    t_nc = timeit(train_step(m_npu, xl, y, True, opt), 5)
    wcache.set_enabled(True)
    t_c = timeit(train_step(m_npu, xl, y, True, opt), 5)
    row("LoRA r=16 on a 42M frozen MLP, batch 128, no cache", t_cpu, t_nc)
    row("LoRA r=16 on a 42M frozen MLP, batch 128, weight cache", t_cpu, t_c, f"({t_nc / t_c:.2f}x from caching)")

    # --- conv_transpose2d (UNet decoder) ---------------------------------------------------------
    bench_train("UNet up-block: ConvT 2x2 s2 256->128 + 2 conv3x3, 32x32",
                lambda: nn.Sequential(nn.ConvTranspose2d(256, 128, 2, stride=2), nn.Conv2d(128, 128, 3, padding=1),
                                      nn.ReLU(), nn.Conv2d(128, 128, 3, padding=1), nn.ReLU()),
                torch.randn(8, 256, 32, 32))
    bench_train("ConvTranspose2d 4x4 s2 p1 128->64, 32x32",
                lambda: nn.Sequential(nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)),
                torch.randn(8, 128, 32, 32))

    # --- grouped conv (ResNeXt-style) -------------------------------------------------------------
    bench_train("grouped conv3x3 256->256 groups=8, 28x28",
                lambda: nn.Sequential(nn.Conv2d(256, 256, 3, padding=1, groups=8)), torch.randn(8, 256, 28, 28))
    bench_train("grouped conv3x3 256->256 groups=4, 28x28",
                lambda: nn.Sequential(nn.Conv2d(256, 256, 3, padding=1, groups=4)), torch.randn(8, 256, 28, 28))

    # --- conv1d (audio / sequence models) ---------------------------------------------------------
    bench_train("conv1d stack 3x (256 ch, k=5), length 2000",
                lambda: nn.Sequential(nn.Conv1d(80, 256, 5, padding=2), nn.ReLU(), nn.Conv1d(256, 256, 5, padding=2),
                                      nn.ReLU(), nn.Conv1d(256, 256, 5, padding=2)), torch.randn(8, 80, 2000))

    # --- einsum ------------------------------------------------------------------------------------
    q, k = torch.randn(16, 8, 256, 64), torch.randn(16, 8, 256, 64)
    t_cpu = timeit(lambda: torch.einsum("bhqd,bhkd->bhqk", q, k))
    def e_npu():
        with rt.npu_mode():
            return torch.einsum("bhqd,bhkd->bhqk", q, k)
    row("einsum bhqd,bhkd->bhqk (16x8 heads, 256x256x64)", t_cpu, timeit(e_npu))
    a, b = torch.randn(64, 512, 1024), torch.randn(1024, 1024)
    t_cpu = timeit(lambda: torch.einsum("bnd,de->bne", a, b))
    def e2():
        with rt.npu_mode():
            return torch.einsum("bnd,de->bne", a, b)
    row("einsum bnd,de->bne (64x512x1024 @ 1024x1024)", t_cpu, timeit(e2))

    # --- MobileNet-style block (depthwise stays on the CPU) -----------------------------------------
    def mbconv():
        return nn.Sequential(nn.Conv2d(32, 192, 1), nn.BatchNorm2d(192), nn.ReLU6(),
                             nn.Conv2d(192, 192, 3, padding=1, groups=192), nn.BatchNorm2d(192), nn.ReLU6(),
                             nn.Conv2d(192, 64, 1), nn.BatchNorm2d(64))
    bench_train("MobileNetV2 block 32->192->64 (dw on CPU), 56x56", mbconv, torch.randn(16, 32, 56, 56))
    print(f"\nweight cache: {wcache.info()}")


if __name__ == "__main__":
    main()
