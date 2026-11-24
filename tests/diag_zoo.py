"""Diagnose one zoo model: per-parameter gradient error (NPU vs CPU fp32, and CPU with fp16-rounded operands)
and a profile of the NPU step.

    python3 tests/diag_zoo.py resnet18 [--precision fp16] [--no-profile]
"""
import argparse
import copy
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import rk3588_torch as rt  # noqa: E402
from rk3588_torch import tools  # noqa: E402
import test_zoo as Z  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("name")
ap.add_argument("--precision", default="fp16")
ap.add_argument("--no-profile", action="store_true")
ap.add_argument("--threads", type=int, default=3)
args = ap.parse_args()
torch.set_num_threads(args.threads)
mk, mkin, lf = Z.cases()[args.name]
n_in = Z.N_IN.get(args.name, 1)
torch.manual_seed(0)
base = mk()
inputs = mkin()


def grads(m):
    return {n: p.grad.detach().double().clone() for n, p in m.named_parameters() if p.grad is not None}


m = copy.deepcopy(base)
Z.step(m, inputs, lf, n_in)
g_cpu = grads(m)

m = copy.deepcopy(base)
with rt.npu_mode(precision=args.precision):
    Z.step(m, inputs, lf, n_in)
g_npu = grads(m)

# CPU with operands rounded like the NPU rounds them (same idea as compare()'s baseline)
dt = torch.float16 if args.precision.startswith("fp16") else torch.bfloat16
m = copy.deepcopy(base)
rnd = lambda t: t.to(dt).to(t.dtype) if torch.is_tensor(t) and t.is_floating_point() else t  # noqa: E731
for mod in m.modules():
    if isinstance(mod, (torch.nn.Linear, torch.nn.Conv2d, torch.nn.Conv1d, torch.nn.ConvTranspose2d)):
        with torch.no_grad():
            mod.weight.copy_(rnd(mod.weight))
        mod.register_forward_pre_hook(lambda _m, a: tuple(rnd(x) for x in a))
        mod.register_full_backward_pre_hook(lambda _m, go: tuple(rnd(x) for x in go))
Z.step(m, inputs, lf, n_in)
g_emu = grads(m)


def rel(a, b):
    return ((a - b).norm() / b.norm().clamp_min(1e-30)).item()


tot = lambda g: torch.cat([v.flatten() for v in g.values()])  # noqa: E731
print(f"total grad rel err: npu {rel(tot(g_npu), tot(g_cpu)):.2e}   cpu-{args.precision}-emulated "
      f"{rel(tot(g_emu), tot(g_cpu)):.2e}   npu vs emulated {rel(tot(g_npu), tot(g_emu)):.2e}")
print(f"{'param':<34}{'shape':>22}{'npu err':>10}{'emu err':>10}{'|g|':>10}")
for n in g_cpu:
    print(f"{n:<34}{str(tuple(g_cpu[n].shape)):>22}{rel(g_npu[n], g_cpu[n]):>10.1e}{rel(g_emu[n], g_cpu[n]):>10.1e}"
          f"{g_cpu[n].norm().item():>10.2e}")

if not args.no_profile:
    m = copy.deepcopy(base)
    with rt.npu_mode(precision=args.precision):
        Z.step(m, inputs, lf, n_in)
    with tools.profile(top=25):
        with rt.npu_mode(precision=args.precision):
            Z.step(m, inputs, lf, n_in)
