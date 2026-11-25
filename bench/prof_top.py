"""Top CPU ops of one NPU training step per model (self time), to see what is left after the NPU work.

    python3 bench/prof_top.py resnet18 unet bert ...
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import rk3588_torch as rt  # noqa: E402
import test_zoo as Z  # noqa: E402

for name in sys.argv[1:] or ["resnet18"]:
    mk, mkin, lf = Z.cases()[name]
    torch.manual_seed(0)
    m = rt.convert(mk())
    inputs = mkin()
    n_in = Z.N_IN.get(name, 1)
    with rt.npu_mode():
        Z.step(m, inputs, lf, n_in)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
        with rt.npu_mode():
            Z.step(m, inputs, lf, n_in)
    ev = prof.key_averages()
    tot = sum(e.self_cpu_time_total for e in ev)
    print(f"=== {name}  (self CPU total {tot / 1e3:.0f} ms)")
    for e in sorted(ev, key=lambda e: e.self_cpu_time_total, reverse=True)[:11]:
        print(f"  {e.key[:44]:44} {e.count:4d}x {e.self_cpu_time_total / 1e3:8.1f} ms {100 * e.self_cpu_time_total / tot:5.1f}%")
