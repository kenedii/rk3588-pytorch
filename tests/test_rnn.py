"""NPU-backed nn.LSTM / nn.GRU (rk3588_torch/rnn.py) vs PyTorch: outputs, final states and all gradients.

    python3 tests/test_rnn.py
"""
import copy
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import rk3588_torch as rt  # noqa: E402
from rk3588_torch import ops, rnn  # noqa: E402

ops.MIN_MACS = 0
torch.manual_seed(0)
fails = 0


def rel(a, b):
    return ((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-30)).item()


for cell in (nn.LSTM, nn.GRU):
    for (layers, bidir, bf, with_h) in ((1, False, True, False), (2, False, True, True), (2, True, False, False),
                                        (3, True, True, True)):
        ref = cell(96, 128, layers, batch_first=bf, bidirectional=bidir)
        m = rt.convert(copy.deepcopy(ref))
        assert type(m) in (rnn.NPULSTM, rnn.NPUGRU)
        x = torch.randn(8, 40, 96) if bf else torch.randn(40, 8, 96)
        D = 2 if bidir else 1
        h0 = torch.randn(layers * D, 8, 128) if with_h else None
        hx = (h0, torch.randn_like(h0)) if (with_h and cell is nn.LSTM) else h0
        xr, xm = x.clone().requires_grad_(), x.clone().requires_grad_()
        o_r, s_r = ref(xr, hx)
        ops.stats(reset=True)
        with rt.npu_mode():
            o_m, s_m = m(xm, hx)
        calls = ops.stats()["npu_calls"]
        st_r = s_r if isinstance(s_r, tuple) else (s_r,)
        st_m = s_m if isinstance(s_m, tuple) else (s_m,)
        loss_r = o_r.square().sum() + sum(t.sum() for t in st_r)
        loss_m = o_m.square().sum() + sum(t.sum() for t in st_m)
        g_r = torch.autograd.grad(loss_r, [xr] + list(ref.parameters()))
        with rt.npu_mode():
            g_m = torch.autograd.grad(loss_m, [xm] + list(m.parameters()))
        errs = [rel(o_m, o_r)] + [rel(a, b) for a, b in zip(st_m, st_r)] + [rel(a, b) for a, b in zip(g_m, g_r)]
        ok = max(errs) < 3e-3 and calls >= layers * D
        fails += not ok
        print(f"{'PASS' if ok else 'FAIL'}  {cell.__name__} layers={layers} bidir={bidir} batch_first={bf} hx={with_h}: "
              f"max rel {max(errs):.1e}, NPU calls {calls}")

# PackedSequence falls back to PyTorch's own implementation
m = rt.convert(nn.LSTM(16, 32))
pk = nn.utils.rnn.pack_padded_sequence(torch.randn(10, 3, 16), [10, 7, 4])
out, _ = m(pk)
ok = isinstance(out, nn.utils.rnn.PackedSequence)
fails += not ok
print(f"{'PASS' if ok else 'FAIL'}  PackedSequence input falls back to PyTorch")
print("ALL PASS" if not fails else f"{fails} FAILED")
sys.exit(1 if fails else 0)
