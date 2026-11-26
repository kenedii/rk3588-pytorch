"""einsum on the NPU for any operand count, with `...` and implicit outputs, forward and backward.

    python3 tests/test_einsum_nary.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import rk3588_torch as rt  # noqa: E402
from rk3588_torch import ops  # noqa: E402

ops.MIN_MACS = 0
torch.manual_seed(0)
fails = 0


def rel(a, b):
    return ((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-30)).item()


def check(eq, shapes, tol=4e-3, expect_npu=True):
    global fails
    ts = [torch.randn(*s, requires_grad=True) for s in shapes]
    ops.stats(reset=True)
    with rt.npu_mode():
        y = torch.einsum(eq, *ts)
    calls = ops.stats()["npu_calls"]
    g = torch.randn_like(y)
    gs = torch.autograd.grad(y, ts, g)
    rs = [t.detach().double().requires_grad_() for t in ts]
    yr = torch.einsum(eq, *rs)
    grs = torch.autograd.grad(yr, rs, g.double())
    errs = [rel(y, yr)] + [rel(a, b) for a, b in zip(gs, grs)]
    ok = tuple(y.shape) == tuple(yr.shape) and max(errs) < tol and (calls > 0) == expect_npu
    fails += not ok
    print(f"{'PASS' if ok else 'FAIL'}  {eq:<30} {str(shapes):<52} {max(errs):.1e}  npu calls {calls}")


check("bij,bjk,bkl->bil", [(4, 64, 96), (4, 96, 80), (4, 80, 72)])
check("ij,jk,kl->il", [(128, 96), (96, 160), (160, 64)])
check("ij,jk,kl,lm->im", [(64, 96), (96, 128), (128, 96), (96, 64)])
check("...ij,...jk->...ik", [(3, 5, 64, 48), (3, 5, 48, 80)])
check("...ij,jk->...ik", [(2, 3, 64, 48), (48, 80)])
check("b...d,de->b...e", [(4, 7, 9, 64), (64, 96)])
check("ij,jk", [(96, 64), (64, 128)])                     # implicit output
check("bij,bjk", [(3, 64, 32), (3, 32, 48)])              # implicit: i, k and b kept (b alphabetical first)
check("ik,jk,j->i", [(64, 48), (80, 48), (80,)])          # summed-out index, vector operand
check("ab,bc,cd->", [(64, 48), (48, 32), (32, 16)])       # full contraction to a scalar
check("bhqd,bhkd,bhkv->bhqv", [(2, 4, 32, 16), (2, 4, 40, 16), (2, 4, 40, 24)])
check("ij,ij->i", [(64, 48), (64, 48)])
# not handled: stays on the CPU and is still right
check("ii->i", [(32, 32)], expect_npu=False)
check("ij->ji", [(32, 48)], expect_npu=False)
check("bij,bjk->bik", [(1, 8, 4), (5, 4, 6)], expect_npu=False)       # broadcasting of b
print("\nALL PASS" if not fails else f"\nFAILURES: {fails}")
sys.exit(1 if fails else 0)
