"""nn.LSTM / nn.GRU with the input projections on the NPU.

PyTorch's CPU LSTM/GRU is a fused kernel that npu_mode() can't look into, so recurrent models ran entirely on the
CPU. Half or more of an RNN's multiply-adds are the input projections W_ih x_t, which don't depend on the hidden
state: for every layer and direction they are one (T*B x in) @ (in x gates) GEMM on the NPU here. Only the
recurrence h_{t-1} W_hh and the gate math stay on the CPU, step by step.

convert() swaps nn.LSTM / nn.GRU for these classes (same parameters, same state_dict). They fall back to
PyTorch's own implementation for what they don't cover: PackedSequence inputs, LSTM proj_size, unbatched inputs,
non-CPU tensors, or when the NPU isn't usable.
"""
import os

import torch
import torch.nn.functional as F

from . import ops


def _usable(x, mod):
    return (isinstance(x, torch.Tensor) and x.dim() == 3 and x.device.type == "cpu"
            and x.dtype in (torch.float32, torch.bfloat16) and ops._enabled and ops.npu_available()
            and getattr(mod, "proj_size", 0) == 0)


# Per-step h @ W_hh: on the NPU when a step has at least REC_MACS multiply-adds (measured crossover ~30M on the
# Rock 5B: B=32,H=512 LSTM 781 vs 924 ms/step on the NPU; B=32,H=512 GRU, 25M, is faster on the CPU).
# RKNPU_RNN_REC_NPU = auto (default) | 1 (always) | 0 (never).
REC_NPU = os.environ.get("RKNPU_RNN_REC_NPU", "auto")
REC_MACS = float(os.environ.get("RKNPU_RNN_REC_MACS", 30e6))


def _rec_on_npu(h, w_t):
    if REC_NPU in ("auto", None):
        return h.shape[0] * w_t.shape[0] * w_t.shape[1] >= REC_MACS
    return REC_NPU in (True, "1")


def _rec_mm(h, w_t, out):
    """h @ w_t for one timestep (small M = batch)."""
    if _rec_on_npu(h, w_t):
        out.copy_(ops.gemm(h, w_t, cache_b="rnn_whh"))
    else:
        torch.mm(h, w_t, out=out)
    return out


class _LSTMRec(torch.autograd.Function):
    """LSTM recurrence given precomputed input projections xp (T, B, 4H). No per-step autograd graph: the
    backward is a hand-written loop, and dW_hh / db_hh are single GEMMs/reductions over all timesteps."""

    @staticmethod
    def forward(ctx, xp, h0, c0, w_hh, b_hh, reverse):
        with torch._C.DisableTorchFunction():
            T, B, G = xp.shape
            H = G // 4
            gates = torch.empty_like(xp)                       # activated i, f, g, o
            hs = xp.new_empty(T + 1, B, H)
            cs = xp.new_empty(T + 1, B, H)
            w_t = w_hh.t()
            hp = xp.new_empty(B, G)
            steps = range(T - 1, -1, -1) if reverse else range(T)
            k = 0
            hs[0], cs[0] = h0, c0
            for t in steps:
                _rec_mm(hs[k], w_t, hp)
                g = gates[t]
                torch.add(xp[t], hp, out=g)
                if b_hh is not None:
                    g += b_hh
                torch.sigmoid(g[:, :2 * H], out=g[:, :2 * H])
                torch.tanh(g[:, 2 * H:3 * H], out=g[:, 2 * H:3 * H])
                torch.sigmoid(g[:, 3 * H:], out=g[:, 3 * H:])
                i, f, gg, o = g[:, :H], g[:, H:2 * H], g[:, 2 * H:3 * H], g[:, 3 * H:]
                c = cs[k + 1]
                torch.mul(f, cs[k], out=c)
                c.addcmul_(i, gg)
                torch.mul(o, torch.tanh(c), out=hs[k + 1])
                k += 1
            y = hs[1:].clone() if not reverse else hs[1:].flip(0)
        ctx.save_for_backward(gates, hs, cs, w_hh)
        ctx.reverse, ctx.has_b = reverse, b_hh is not None
        return y.contiguous(), hs[T].clone(), cs[T].clone()

    @staticmethod
    def backward(ctx, dy, dhn, dcn):
        gates, hs, cs, w_hh = ctx.saved_tensors
        reverse = ctx.reverse
        with torch._C.DisableTorchFunction():
            T, B, G = gates.shape
            H = G // 4
            dG = torch.empty_like(gates)
            dh = dhn.clone() if dhn is not None else gates.new_zeros(B, H)
            dc = dcn.clone() if dcn is not None else gates.new_zeros(B, H)
            steps = list(range(T - 1, -1, -1) if reverse else range(T))
            for k in range(T - 1, -1, -1):
                t = steps[k]
                if dy is not None:
                    dh += dy[t]
                g = gates[t]
                i, f, gg, o = g[:, :H], g[:, H:2 * H], g[:, 2 * H:3 * H], g[:, 3 * H:]
                tc = torch.tanh(cs[k + 1])
                d = dG[t]
                torch.mul(dh, tc, out=d[:, 3 * H:])
                d[:, 3 * H:] *= o * (1 - o)
                dc.addcmul_(dh * o, 1 - tc * tc)
                torch.mul(dc, gg, out=d[:, :H])
                d[:, :H] *= i * (1 - i)
                torch.mul(dc, cs[k], out=d[:, H:2 * H])
                d[:, H:2 * H] *= f * (1 - f)
                torch.mul(dc, i, out=d[:, 2 * H:3 * H])
                d[:, 2 * H:3 * H] *= 1 - gg * gg
                dc.mul_(f)
                torch.mm(d, w_hh, out=dh)
            # h_{t-1} for every gate row, in the same (T, B) order as dG
            hprev = hs[:-1] if not reverse else hs[:-1].flip(0)
            dGf = dG.reshape(T * B, G)
            dW = ops.gemm(dGf.t(), hprev.reshape(T * B, H)).to(w_hh.dtype)    # one NPU GEMM over all steps
            db = dGf.sum(0) if ctx.has_b else None
            dh0 = dh if ctx.needs_input_grad[1] else None
            dc0 = dc if ctx.needs_input_grad[2] else None
        return dG, dh0, dc0, dW, db, None


class _GRURec(torch.autograd.Function):
    """GRU recurrence given xp (T, B, 3H) = x W_ih^T + b_ih."""

    @staticmethod
    def forward(ctx, xp, h0, w_hh, b_hh, reverse):
        with torch._C.DisableTorchFunction():
            T, B, G = xp.shape
            H = G // 3
            acts = torch.empty_like(xp)              # r, z, n
            hns = xp.new_empty(T, B, H)              # h W_hn^T + b_hn
            hs = xp.new_empty(T + 1, B, H)
            w_t = w_hh.t()
            hp = xp.new_empty(B, G)
            steps = range(T - 1, -1, -1) if reverse else range(T)
            hs[0] = h0
            for k, t in enumerate(steps):
                _rec_mm(hs[k], w_t, hp)
                if b_hh is not None:
                    hp += b_hh
                a = acts[t]
                torch.add(xp[t, :, :2 * H], hp[:, :2 * H], out=a[:, :2 * H])
                torch.sigmoid(a[:, :2 * H], out=a[:, :2 * H])
                hns[t] = hp[:, 2 * H:]
                n = a[:, 2 * H:]
                torch.addcmul(xp[t, :, 2 * H:], a[:, :H], hp[:, 2 * H:], out=n)
                torch.tanh(n, out=n)
                z = a[:, H:2 * H]
                h = hs[k + 1]
                torch.sub(hs[k], n, out=h)
                h.mul_(z).add_(n)
            y = hs[1:].clone() if not reverse else hs[1:].flip(0)
        ctx.save_for_backward(acts, hns, hs, w_hh)
        ctx.reverse, ctx.has_b = reverse, b_hh is not None
        return y.contiguous(), hs[T].clone()

    @staticmethod
    def backward(ctx, dy, dhn):
        acts, hns, hs, w_hh = ctx.saved_tensors
        reverse = ctx.reverse
        with torch._C.DisableTorchFunction():
            T, B, G = acts.shape
            H = G // 3
            dX = torch.empty_like(acts)
            dHp = torch.empty_like(acts)
            dh = dhn.clone() if dhn is not None else acts.new_zeros(B, H)
            steps = list(range(T - 1, -1, -1) if reverse else range(T))
            for k in range(T - 1, -1, -1):
                t = steps[k]
                if dy is not None:
                    dh += dy[t]
                a = acts[t]
                r, z, n = a[:, :H], a[:, H:2 * H], a[:, 2 * H:]
                dx, dp = dX[t], dHp[t]
                dn = dx[:, 2 * H:]
                torch.mul(dh, 1 - z, out=dn)
                dn *= 1 - n * n                                  # d(pre-tanh)
                torch.mul(dn, r, out=dp[:, 2 * H:])              # d(h W_hn + b_hn)
                torch.mul(dn, hns[t], out=dx[:, :H])
                dx[:, :H] *= r * (1 - r)                         # d(pre-sigmoid r)
                torch.sub(hs[k], n, out=dx[:, H:2 * H])
                dx[:, H:2 * H] *= dh
                dx[:, H:2 * H] *= z * (1 - z)                    # d(pre-sigmoid z)
                dp[:, :2 * H] = dx[:, :2 * H]
                dh.mul_(z)
                dh.addmm_(dp, w_hh)
            hprev = hs[:-1] if not reverse else hs[:-1].flip(0)
            dpf = dHp.reshape(T * B, G)
            dW = ops.gemm(dpf.t(), hprev.reshape(T * B, H)).to(w_hh.dtype)
            db = dpf.sum(0) if ctx.has_b else None
            dh0 = dh if ctx.needs_input_grad[1] else None
        return dX, dh0, dW, db, None


def _run(mod, x, hx, cell):
    """Shared driver: x (B, T, in) if batch_first else (T, B, in). cell = 'lstm' | 'gru'."""
    if mod.batch_first:
        x = x.transpose(0, 1)
    T, B, _ = x.shape
    H, L = mod.hidden_size, mod.num_layers
    D = 2 if mod.bidirectional else 1
    if hx is None:
        z = torch.zeros(L * D, B, H, dtype=x.dtype)
        hx = (z, z.clone()) if cell == "lstm" else z
    h_n, c_n = [], []
    inp = x
    for layer in range(L):
        outs = []
        for d in range(D):
            sfx = f"_l{layer}" + ("_reverse" if d else "")
            w_ih, w_hh = getattr(mod, "weight_ih" + sfx), getattr(mod, "weight_hh" + sfx)
            b_ih = getattr(mod, "bias_ih" + sfx) if mod.bias else None
            b_hh = getattr(mod, "bias_hh" + sfx) if mod.bias else None
            # all timesteps' input projections in one NPU GEMM
            xp = ops.linear(inp.reshape(T * B, -1), w_ih, b_ih).reshape(T, B, -1)
            idx = layer * D + d
            if cell == "lstm":
                y, h, c = _LSTMRec.apply(xp, hx[0][idx], hx[1][idx], w_hh, b_hh, bool(d))
                c_n.append(c)
            else:
                y, h = _GRURec.apply(xp, hx[idx], w_hh, b_hh, bool(d))
            outs.append(y)
            h_n.append(h)
        inp = outs[0] if D == 1 else torch.cat(outs, -1)
        if mod.dropout and mod.training and layer < L - 1:
            inp = F.dropout(inp, mod.dropout, True)
    out = inp.transpose(0, 1) if mod.batch_first else inp
    hn = torch.stack(h_n)
    return (out, (hn, torch.stack(c_n))) if cell == "lstm" else (out, hn)


class NPULSTM(torch.nn.LSTM):
    def forward(self, input, hx=None):  # noqa: A002
        if not _usable(input, self) or (hx is not None and not isinstance(hx, tuple)):
            return super().forward(input, hx)
        return _run(self, input, hx, "lstm")


class NPUGRU(torch.nn.GRU):
    def forward(self, input, hx=None):  # noqa: A002
        if not _usable(input, self):
            return super().forward(input, hx)
        return _run(self, input, hx, "gru")
