"""Profile one NPU training step of a zoo model (tests/test_zoo.py or tests/test_zoo2.py): where the time goes.

    python3 bench/prof_zoo.py unet [--precision bf16] [--threads 3] [--top 14]
"""
import argparse
import copy
import os
import sys

import torch

here = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(here, "..", "tests"))
sys.path.insert(0, os.path.join(here, ".."))
import rk3588_torch as rt  # noqa: E402
import test_zoo as Z1  # noqa: E402
import test_zoo2 as Z2  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("name")
ap.add_argument("--precision", default="fp16")
ap.add_argument("--threads", type=int, default=3)
ap.add_argument("--top", type=int, default=14)
args = ap.parse_args()
torch.set_num_threads(args.threads)
Z1.N_IN.update(Z2.N_IN)
cases = {**Z1.cases(), **Z2.cases()}
mk, mkin, lf = cases[args.name]
n_in = Z1.N_IN.get(args.name, 1)
torch.manual_seed(0)
model = rt.convert(mk())
inputs = mkin()
with rt.npu_mode(precision=args.precision):
    Z1.step(model, inputs, lf, n_in)          # warm-up: calibration, shape checks, weight packing
    with rt.profile(top=args.top):
        Z1.step(model, inputs, lf, n_in)
