"""Which aten::copy_ / im2col calls dominate a training step? Groups them by input shape.

    python3 bench/prof_copies.py unet [resnet18 ...]
"""
import copy
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import rk3588_torch as rt  # noqa: E402
import test_zoo as Z  # noqa: E402

for name in sys.argv[1:] or ["unet"]:
    mk, mkin, lf = Z.cases()[name]
    torch.manual_seed(0)
    m = rt.convert(mk())
    inputs = mkin()
    n_in = Z.N_IN.get(name, 1)
    with rt.npu_mode():
        Z.step(m, inputs, lf, n_in)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], record_shapes=True) as prof:
        with rt.npu_mode():
            Z.step(m, inputs, lf, n_in)
    print(f"=== {name}")
    rows = [e for e in prof.key_averages(group_by_input_shape=True)
            if e.key in ("aten::copy_", "aten::im2col", "aten::col2im", "aten::sum", "aten::_to_copy")]
    rows.sort(key=lambda e: e.self_cpu_time_total, reverse=True)
    for e in rows[:14]:
        print(f"  {e.key:14} {e.count:3d}x {e.self_cpu_time_total / 1e3:8.1f} ms  {str(e.input_shapes)[:110]}")
