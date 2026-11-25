"""Where a training step's time goes inside the NPU calls: pack / NPU (hw) / unpack, summed over every conv and GEMM
call of one step, against the step's wall time. Bounds what keeping activations on the NPU between layers can save.

    python3 bench/prof_phases.py resnet18 unet ...
"""
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import rk3588_torch as rt  # noqa: E402
import test_zoo as Z  # noqa: E402
from rk3588_torch._lib import default_npu  # noqa: E402

npu = default_npu()
acc = {}


def wrap(name):
    f = getattr(type(npu), name)

    def g(self, *a, **k):
        r = f(self, *a, **k)
        s = self.stats()
        d = acc.setdefault(name, dict(n=0, pack=0.0, hw=0.0, unpack=0.0, sub=0.0, total=0.0))
        d["n"] += 1
        d["pack"] += s["pack_us"]; d["hw"] += s["hw_us"]; d["unpack"] += s["unpack_us"]
        d["sub"] += s["submit_us"]; d["total"] += s["total_us"]
        return r
    setattr(type(npu), name, g)


for n in ("conv2d", "gemm"):
    if hasattr(type(npu), n):
        wrap(n)

for name in sys.argv[1:] or ["resnet18"]:
    mk, mkin, lf = Z.cases()[name]
    torch.manual_seed(0)
    m = rt.convert(mk())
    inputs = mkin()
    n_in = Z.N_IN.get(name, 1)
    with rt.npu_mode():
        Z.step(m, inputs, lf, n_in)
        acc.clear()
        t0 = time.perf_counter()
        Z.step(m, inputs, lf, n_in)
        wall = (time.perf_counter() - t0) * 1e3
    print(f"=== {name}: step {wall:.0f} ms")
    for k, d in acc.items():
        print(f"  {k:7} {d['n']:4d} calls  pack {d['pack'] / 1e3:7.1f}  submit {d['sub'] / 1e3:6.1f}  npu {d['hw'] / 1e3:7.1f}  "
              f"unpack {d['unpack'] / 1e3:7.1f}  total {d['total'] / 1e3:7.1f} ms  ({100 * d['total'] / 1e3 / wall:.0f}% of step)")
