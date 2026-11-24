"""Model zoo: one training step of many architectures, CPU fp32 vs npu_mode(), from identical weights.

For each model it checks that the NPU step runs without errors, that the loss and gradients match the CPU
within the precision's rounding, and reports the step time, the share of multiply-adds that ran on the NPU
and the number of CPU fallbacks. Models are sized to fit next to other work on a 4 GB board.

    python3 tests/test_zoo.py [--only gpt,vit] [--precision fp16|bf16] [--reps 3] [--threads 3]
    python3 tests/test_zoo.py --fake        # no NPU needed: checks routing and autograd plumbing only
"""
import argparse
import copy
import gc
import math
import os
import statistics
import sys
import time
import traceback

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import rk3588_torch as rt  # noqa: E402
from rk3588_torch import ops  # noqa: E402


# ----------------------------------------------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------------------------------------------
class GPT(nn.Module):
    """Decoder-only LM: pre-norm blocks, causal SDPA, GELU MLP, tied embeddings."""

    def __init__(self, vocab=4096, d=384, heads=6, layers=4, ctx=256):
        super().__init__()
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(ctx, d)
        self.blocks = nn.ModuleList()
        for _ in range(layers):
            self.blocks.append(nn.ModuleDict(dict(
                ln1=nn.LayerNorm(d), qkv=nn.Linear(d, 3 * d), proj=nn.Linear(d, d),
                ln2=nn.LayerNorm(d), fc=nn.Linear(d, 4 * d), out=nn.Linear(4 * d, d))))
        self.ln = nn.LayerNorm(d)
        self.heads = heads

    def forward(self, idx):
        B, T = idx.shape
        x = self.tok(idx) + self.pos(torch.arange(T))
        for b in self.blocks:
            q, k, v = b["qkv"](b["ln1"](x)).split(x.shape[-1], -1)
            sh = lambda t: t.view(B, T, self.heads, -1).transpose(1, 2)  # noqa: E731
            y = F.scaled_dot_product_attention(sh(q), sh(k), sh(v), is_causal=True)
            x = x + b["proj"](y.transpose(1, 2).reshape(B, T, -1))
            x = x + b["out"](F.gelu(b["fc"](b["ln2"](x))))
        return self.ln(x) @ self.tok.weight.t()


class LlamaBlockLM(nn.Module):
    """Llama-style: RMSNorm, rotary embeddings, grouped-query attention, SwiGLU, bias-free linears."""

    def __init__(self, vocab=4096, d=512, heads=8, kv_heads=2, layers=2, ff=1408):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.h, self.kv, self.hd = heads, kv_heads, d // heads
        self.layers = nn.ModuleList(nn.ModuleDict(dict(
            n1=nn.RMSNorm(d), q=nn.Linear(d, d, bias=False), k=nn.Linear(d, kv_heads * self.hd, bias=False),
            v=nn.Linear(d, kv_heads * self.hd, bias=False), o=nn.Linear(d, d, bias=False), n2=nn.RMSNorm(d),
            g=nn.Linear(d, ff, bias=False), u=nn.Linear(d, ff, bias=False), dn=nn.Linear(ff, d, bias=False)))
            for _ in range(layers))
        self.norm = nn.RMSNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)

    def rope(self, x):
        T, hd = x.shape[-2], x.shape[-1]
        f = 1.0 / (10000 ** (torch.arange(0, hd, 2).float() / hd))
        ang = torch.outer(torch.arange(T).float(), f)
        c, s = ang.cos(), ang.sin()
        x1, x2 = x[..., 0::2], x[..., 1::2]
        return torch.stack((x1 * c - x2 * s, x1 * s + x2 * c), -1).flatten(-2)

    def forward(self, idx):
        B, T = idx.shape
        x = self.emb(idx)
        for L in self.layers:
            h = L["n1"](x)
            q = self.rope(L["q"](h).view(B, T, self.h, self.hd).transpose(1, 2))
            k = self.rope(L["k"](h).view(B, T, self.kv, self.hd).transpose(1, 2))
            v = L["v"](h).view(B, T, self.kv, self.hd).transpose(1, 2)
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
            x = x + L["o"](y.transpose(1, 2).reshape(B, T, -1))
            h = L["n2"](x)
            x = x + L["dn"](F.silu(L["g"](h)) * L["u"](h))
        return self.head(self.norm(x))


class ViT(nn.Module):
    """Vision transformer: stride-16 patch conv, class token, nn.TransformerEncoder (MHA path)."""

    def __init__(self, img=64, patch=8, d=256, heads=8, layers=4, classes=10):
        super().__init__()
        self.patch = nn.Conv2d(3, d, patch, patch)
        n = (img // patch) ** 2
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.pos = nn.Parameter(torch.randn(1, n + 1, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, activation="gelu", batch_first=True,
                                           norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.head = nn.Linear(d, classes)

    def forward(self, x):
        x = self.patch(x).flatten(2).transpose(1, 2)
        x = torch.cat([self.cls.expand(x.shape[0], -1, -1), x], 1) + self.pos
        return self.head(self.enc(x)[:, 0])


class BertCls(nn.Module):
    """Encoder with a key padding mask (bool), post-norm, MHA path with need_weights=False."""

    def __init__(self, vocab=8000, d=256, heads=4, layers=3):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        layer = nn.TransformerEncoderLayer(d, heads, 1024, dropout=0.0, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.cls = nn.Linear(d, 2)

    def forward(self, tok, pad):
        return self.cls(self.enc(self.emb(tok), src_key_padding_mask=pad).mean(1))


class Seq2Seq(nn.Module):
    """Full nn.Transformer (encoder + decoder, cross-attention, causal target mask)."""

    def __init__(self, vocab=2000, d=256):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.tf = nn.Transformer(d, 4, 2, 2, 512, dropout=0.0, batch_first=True)
        self.out = nn.Linear(d, vocab)

    def forward(self, src, tgt):
        m = self.tf.generate_square_subsequent_mask(tgt.shape[1])
        return self.out(self.tf(self.emb(src), self.emb(tgt), tgt_mask=m, tgt_is_causal=True))


class MBConv(nn.Module):
    def __init__(self, c, e=4):
        super().__init__()
        self.pw1 = nn.Conv2d(c, c * e, 1, bias=False)
        self.b1 = nn.BatchNorm2d(c * e)
        self.dw = nn.Conv2d(c * e, c * e, 3, 1, 1, groups=c * e, bias=False)
        self.b2 = nn.BatchNorm2d(c * e)
        self.pw2 = nn.Conv2d(c * e, c, 1, bias=False)
        self.b3 = nn.BatchNorm2d(c)

    def forward(self, x):
        y = F.relu6(self.b1(self.pw1(x)))
        y = F.relu6(self.b2(self.dw(y)))
        return x + self.b3(self.pw2(y))


def mobilenet():
    return nn.Sequential(nn.Conv2d(3, 32, 3, 2, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU6(),
                         nn.Conv2d(32, 64, 1), MBConv(64), MBConv(64), nn.Conv2d(64, 128, 3, 2, 1),
                         MBConv(128), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(128, 10))


class UNet(nn.Module):
    def __init__(self, c=32):
        super().__init__()
        def blk(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.GroupNorm(8, o), nn.SiLU(),
                                 nn.Conv2d(o, o, 3, padding=1), nn.GroupNorm(8, o), nn.SiLU())
        self.d1, self.d2, self.d3 = blk(3, c), blk(c, 2 * c), blk(2 * c, 4 * c)
        self.u2 = nn.ConvTranspose2d(4 * c, 2 * c, 2, 2)
        self.c2 = blk(4 * c, 2 * c)
        self.u1 = nn.Upsample(scale_factor=2, mode="nearest")
        self.c1 = blk(2 * c + c, c)
        self.out = nn.Conv2d(c, 3, 1)

    def forward(self, x):
        a = self.d1(x)
        b = self.d2(F.max_pool2d(a, 2))
        c = self.d3(F.max_pool2d(b, 2))
        y = self.c2(torch.cat([self.u2(c), b], 1))
        y = self.c1(torch.cat([self.u1(y), a], 1))
        return self.out(y)


class DCGANGen(nn.Module):
    def __init__(self, z=100, c=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(z, c * 8, 4, 1, 0, bias=False), nn.BatchNorm2d(c * 8), nn.ReLU(),
            nn.ConvTranspose2d(c * 8, c * 4, 4, 2, 1, bias=False), nn.BatchNorm2d(c * 4), nn.ReLU(),
            nn.ConvTranspose2d(c * 4, c * 2, 4, 2, 1, bias=False), nn.BatchNorm2d(c * 2), nn.ReLU(),
            nn.ConvTranspose2d(c * 2, 3, 4, 2, 1, bias=False), nn.Tanh())

    def forward(self, z):
        return self.net(z)


class TCN(nn.Module):
    """Temporal conv net for audio-like sequences: dilated causal conv1d (dilation 1, 2, 4, 8)."""

    def __init__(self, c=128, levels=4):
        super().__init__()
        self.inp = nn.Conv1d(1, c, 1)
        self.layers = nn.ModuleList(nn.Conv1d(c, c, 3, dilation=2 ** i, padding=2 * 2 ** i) for i in range(levels))
        self.out = nn.Linear(c, 10)

    def forward(self, x):
        x = self.inp(x)
        for conv in self.layers:
            x = x + F.relu(conv(x)[..., :x.shape[-1]])
        return self.out(x.mean(-1))


class LSTMLM(nn.Module):
    def __init__(self, vocab=4000, d=256, h=512, layers=2, cell=nn.LSTM):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.rnn = cell(d, h, layers, batch_first=True)
        self.out = nn.Linear(h, vocab)

    def forward(self, tok):
        return self.out(self.rnn(self.emb(tok))[0])


class VAE(nn.Module):
    def __init__(self, d=784, h=1024, z=64):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 2 * z))
        self.dec = nn.Sequential(nn.Linear(z, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, d))

    def forward(self, x):
        mu, logv = self.enc(x).chunk(2, -1)
        torch.manual_seed(1)
        zz = mu + torch.randn_like(mu) * (0.5 * logv).exp()
        rec = self.dec(zz)
        return F.mse_loss(rec, x) + 1e-3 * (-0.5 * (1 + logv - mu ** 2 - logv.exp()).sum(-1).mean())


class DLRM(nn.Module):
    """Recommender: embedding bags for sparse features + bottom/top MLPs + pairwise dot interaction."""

    def __init__(self, tables=8, rows=20000, d=64):
        super().__init__()
        self.embs = nn.ModuleList(nn.EmbeddingBag(rows, d, mode="sum") for _ in range(tables))
        self.bot = nn.Sequential(nn.Linear(13, 512), nn.ReLU(), nn.Linear(512, d), nn.ReLU())
        n = tables + 1
        self.top = nn.Sequential(nn.Linear(d + n * (n - 1) // 2, 1024), nn.ReLU(), nn.Linear(1024, 1024), nn.ReLU(),
                                 nn.Linear(1024, 1))

    def forward(self, dense, sparse):
        x = self.bot(dense)
        feats = torch.stack([x] + [e(s) for e, s in zip(self.embs, sparse.unbind(1))], 1)
        inter = torch.bmm(feats, feats.transpose(1, 2))
        i, j = torch.triu_indices(feats.shape[1], feats.shape[1], 1)
        return self.top(torch.cat([x, inter[:, i, j]], 1))


class GNN(nn.Module):
    """Graph conv net on a dense normalized adjacency (A_hat @ X @ W), two layers."""

    def __init__(self, f=256, h=512, c=16):
        super().__init__()
        self.l1, self.l2 = nn.Linear(f, h), nn.Linear(h, c)

    def forward(self, adj, x):
        return self.l2(adj @ F.relu(self.l1(adj @ x)))


class SiameseEmb(nn.Module):
    """Contrastive (CLIP-style) loss: two towers and a batch x batch similarity matmul."""

    def __init__(self, d=512):
        super().__init__()
        self.a = nn.Sequential(nn.Linear(768, 1024), nn.GELU(), nn.Linear(1024, d))
        self.b = nn.Sequential(nn.Linear(512, 1024), nn.GELU(), nn.Linear(1024, d))

    def forward(self, x, y):
        za, zb = F.normalize(self.a(x), dim=-1), F.normalize(self.b(y), dim=-1)
        logits = za @ zb.t() / 0.07
        lab = torch.arange(x.shape[0])
        return (F.cross_entropy(logits, lab) + F.cross_entropy(logits.t(), lab)) / 2


def resnet18ish():
    class Blk(nn.Module):
        def __init__(self, i, o, s):
            super().__init__()
            self.c1, self.b1 = nn.Conv2d(i, o, 3, s, 1, bias=False), nn.BatchNorm2d(o)
            self.c2, self.b2 = nn.Conv2d(o, o, 3, 1, 1, bias=False), nn.BatchNorm2d(o)
            self.sc = nn.Sequential(nn.Conv2d(i, o, 1, s, bias=False), nn.BatchNorm2d(o)) if s != 1 or i != o \
                else nn.Identity()

        def forward(self, x):
            return F.relu(self.b2(self.c2(F.relu(self.b1(self.c1(x))))) + self.sc(x))
    return nn.Sequential(nn.Conv2d(3, 64, 7, 2, 3, bias=False), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(3, 2, 1),
                         Blk(64, 64, 1), Blk(64, 64, 1), Blk(64, 128, 2), Blk(128, 128, 1), Blk(128, 256, 2),
                         Blk(256, 256, 1), Blk(256, 512, 2), Blk(512, 512, 1), nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                         nn.Linear(512, 100))


class ChannelsLastCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 128, 3, 2, 1), nn.ReLU(),
                                 nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                 nn.Linear(128, 10))

    def forward(self, x):
        return self.net(x.contiguous(memory_format=torch.channels_last))


# ----------------------------------------------------------------------------------------------------------
# Cases: name -> (model factory, inputs factory, loss(out, *inputs) or None for "out is the loss")
# ----------------------------------------------------------------------------------------------------------
def ce(o, *_):
    return F.cross_entropy(o.reshape(-1, o.shape[-1]).float(), _[-1].reshape(-1))


def lm_loss(o, tok):
    return F.cross_entropy(o[:, :-1].reshape(-1, o.shape[-1]).float(), tok[:, 1:].reshape(-1))


def cases():
    g = torch.Generator().manual_seed(0)
    ri = lambda hi, *s: torch.randint(0, hi, s, generator=g)  # noqa: E731
    rn = lambda *s: torch.randn(*s, generator=g)  # noqa: E731
    C = {}
    C["gpt"] = (lambda: GPT(), lambda: (ri(4096, 8, 256),), lm_loss)
    C["llama_gqa"] = (lambda: LlamaBlockLM(), lambda: (ri(4096, 4, 256),), lm_loss)
    C["vit"] = (lambda: ViT(), lambda: (rn(32, 3, 64, 64), ri(10, 32)), lambda o, x, y: F.cross_entropy(o, y))
    C["bert_padmask"] = (lambda: BertCls(),
                         lambda: (ri(8000, 16, 128), torch.arange(128).expand(16, 128) >= ri(64, 16, 1) + 64,
                                  ri(2, 16)),
                         lambda o, t, p, y: F.cross_entropy(o, y))
    C["seq2seq"] = (lambda: Seq2Seq(), lambda: (ri(2000, 16, 64), ri(2000, 16, 64)),
                    lambda o, s, t: F.cross_entropy(o.reshape(-1, o.shape[-1]), t.reshape(-1)))
    C["resnet18"] = (lambda: resnet18ish(), lambda: (rn(16, 3, 64, 64), ri(100, 16)),
                     lambda o, x, y: F.cross_entropy(o, y))
    C["mobilenet"] = (lambda: mobilenet(), lambda: (rn(32, 3, 64, 64), ri(10, 32)),
                      lambda o, x, y: F.cross_entropy(o, y))
    C["unet"] = (lambda: UNet(), lambda: (rn(8, 3, 64, 64),), lambda o, x: F.mse_loss(o, x))
    C["dcgan_gen"] = (lambda: DCGANGen(), lambda: (rn(32, 100, 1, 1),), lambda o, z: o.pow(2).mean())
    C["tcn_dilated"] = (lambda: TCN(), lambda: (rn(16, 1, 2000), ri(10, 16)), lambda o, x, y: F.cross_entropy(o, y))
    C["lstm_lm"] = (lambda: LSTMLM(), lambda: (ri(4000, 32, 64),), lm_loss)
    C["gru_lm"] = (lambda: LSTMLM(cell=nn.GRU), lambda: (ri(4000, 32, 64),), lm_loss)
    C["vae"] = (lambda: VAE(), lambda: (torch.rand(256, 784, generator=g),), None)
    C["dlrm"] = (lambda: DLRM(), lambda: (rn(512, 13), ri(20000, 512, 8, 4), torch.rand(512, 1, generator=g)),
                 lambda o, d, s, y: F.binary_cross_entropy_with_logits(o, y))
    C["gnn_dense"] = (lambda: GNN(),
                      lambda: (torch.softmax(rn(1024, 1024), -1), rn(1024, 256), ri(16, 1024)),
                      lambda o, a, x, y: F.cross_entropy(o, y))
    C["clip_contrastive"] = (lambda: SiameseEmb(), lambda: (rn(256, 768), rn(256, 512)), None)
    C["channels_last"] = (lambda: ChannelsLastCNN(), lambda: (rn(32, 3, 64, 64), ri(10, 32)),
                          lambda o, x, y: F.cross_entropy(o, y))
    return C


# number of leading inputs that go to the model (the rest are labels for the loss)
N_IN = {'vit': 1, 'bert_padmask': 2, 'seq2seq': 2, 'resnet18': 1, 'mobilenet': 1, 'unet': 1, 'dcgan_gen': 1, 'tcn_dilated': 1, 'lstm_lm': 1, 'gru_lm': 1, 'vae': 1, 'dlrm': 2, 'gnn_dense': 2, 'clip_contrastive': 2, 'channels_last': 1, 'gpt': 1, 'llama_gqa': 1}


# ----------------------------------------------------------------------------------------------------------
# Runner
# ----------------------------------------------------------------------------------------------------------
def flat_grads(m):
    gs = [p.grad.detach().double().flatten() for p in m.parameters() if p.grad is not None]
    return torch.cat(gs) if gs else torch.zeros(0, dtype=torch.float64)


def step(model, inputs, loss_fn, n_in):
    out = model(*inputs[:n_in])
    loss = out if loss_fn is None else loss_fn(out, *inputs)
    model.zero_grad(set_to_none=True)
    loss.backward()
    return loss


def run_case(name, mk, mkin, loss_fn, precision, reps, tol):
    n_in = N_IN.get(name, 1)
    torch.manual_seed(0)
    base = mk()
    inputs = mkin()
    res = {"name": name}
    # CPU reference
    m = copy.deepcopy(base)
    times = []
    from torch.utils.flop_counter import FlopCounterMode
    with FlopCounterMode(display=False) as fc:        # every matmul/conv/attention MAC, routed or not
        loss = step(m, inputs, loss_fn, n_in)
    total_macs = fc.get_total_flops() / 2
    ref_loss, ref_g = float(loss), flat_grads(m)
    m = copy.deepcopy(base)
    for i in range(reps + 1):
        t0 = time.perf_counter()
        loss = step(m, inputs, loss_fn, n_in)
        times.append(time.perf_counter() - t0)
    res["cpu_ms"] = statistics.median(times[1:]) * 1e3
    del m
    gc.collect()
    # NPU
    m = rt.convert(copy.deepcopy(base))     # the recommended usage: convert() + npu_mode()
    ops.stats(reset=True)
    times = []
    with rt.npu_mode(precision=precision):
        for i in range(reps + 1):
            t0 = time.perf_counter()
            loss = step(m, inputs, loss_fn, n_in)
            times.append(time.perf_counter() - t0)
            if i == 0:
                npu_loss, npu_g = float(loss), flat_grads(m)
    st = ops.stats()
    res["npu_ms"] = statistics.median(times[1:]) * 1e3
    res["speedup"] = res["cpu_ms"] / res["npu_ms"]
    res["loss_diff"] = abs(npu_loss - ref_loss) / max(abs(ref_loss), 1e-12)
    res["grad_err"] = ((npu_g - ref_g).norm() / ref_g.norm().clamp_min(1e-30)).item()
    # share of the step's multiply-adds that ran on the NPU (stats count reps + 1 steps)
    res["npu_share"] = min(st["npu_macs"] / (reps + 1) / max(total_macs, 1), 1.0)
    res["fallbacks"] = st["fallbacks"]
    # what the precision itself costs: the CPU step with operands rounded where the NPU rounds them
    # (BatchNorm nets at small batch amplify rounding: ResNet-18 at batch 16 is ~1e-1 in fp16 on the CPU too)
    res["base_err"] = emulated_err(base, inputs, loss_fn, n_in, precision, ref_g)
    lim = max(tol, 2.0 * res["base_err"]) if res["base_err"] is not None else tol
    res["ok"] = math.isfinite(npu_loss) and res["grad_err"] <= lim and res["loss_diff"] <= tol
    return res


def emulated_err(base, inputs, loss_fn, n_in, precision, ref_g):
    dt = {"fp16": torch.float16, "fp16_out16": torch.float16, "bf16": torch.bfloat16}.get(precision)
    if dt is None:
        return None
    m = copy.deepcopy(base)
    rnd = lambda t: t.to(dt).to(t.dtype) if torch.is_tensor(t) and t.is_floating_point() else t  # noqa: E731
    for mod in m.modules():
        if isinstance(mod, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.ConvTranspose2d)):
            with torch.no_grad():
                mod.weight.copy_(rnd(mod.weight))
            mod.register_forward_pre_hook(lambda _m, a: tuple(rnd(x) for x in a))
            mod.register_full_backward_pre_hook(lambda _m, go: tuple(rnd(x) for x in go))
    step(m, inputs, loss_fn, n_in)
    g = flat_grads(m)
    return ((g - ref_g).norm() / ref_g.norm().clamp_min(1e-30)).item()


def install_fake():
    """Emulate the NPU on the CPU (fp16-rounded operands, fp32 accumulation), like tests/test_new_ops.py --fake."""
    def _fake_gemm(a, b, precision, bias=None, relu=False, wkey=0):
        out = a.float().half().float() @ b.float().half().float()
        ops._stats["npu_calls"] += 1
        ops._stats["npu_macs"] += out.numel() * a.shape[-1]
        return ops._epilogue(out, bias, relu)

    def _fake_conv(x, w, st, pd, precision, groups=1, bias=None, relu=False, wkey=0):
        out = F.conv2d(x.float().half().float(), w.float().half().float(), None if bias is None else bias.float(),
                       stride=st, padding=pd, groups=groups)
        ops._stats["npu_calls"] += 1
        ops._stats["npu_macs"] += out.numel() * w.shape[1] * w.shape[2] * w.shape[3]
        return torch.relu(out) if relu else out
    ops._npu_gemm, ops._npu_conv = _fake_gemm, _fake_conv
    ops.npu_available = lambda: True
    ops.MIN_MACS = 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--skip", default="")
    ap.add_argument("--precision", default="fp16")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--fake", action="store_true", help="no NPU: route through the custom ops but compute on CPU")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    if args.fake:
        install_fake()
    tol = {"fp16": 2e-2, "bf16": 6e-2, "fp16_out16": 2e-2, "fp16x3": 1e-3, "int8": 0.25}[args.precision]
    only = [s for s in args.only.split(",") if s]
    skip = [s for s in args.skip.split(",") if s]
    print(f"{'model':<18}{'cpu ms':>9}{'npu ms':>9}{'speedup':>9}{'grad err':>10}{'cpu lowp':>10}{'loss d':>9}"
          f"{'on NPU':>8}{'fallbk':>7}  ok", flush=True)
    bad = 0
    for name, (mk, mkin, lf) in cases().items():
        if (only and name not in only) or name in skip:
            continue
        try:
            r = run_case(name, mk, mkin, lf, args.precision, args.reps, tol)
            bad += not r["ok"]
            be = f"{r['base_err']:.1e}" if r["base_err"] is not None else "-"
            print(f"{name:<18}{r['cpu_ms']:>9.1f}{r['npu_ms']:>9.1f}{r['speedup']:>8.2f}x{r['grad_err']:>10.1e}"
                  f"{be:>10}{r['loss_diff']:>9.1e}{r['npu_share'] * 100:>7.0f}%{r['fallbacks']:>7}  "
                  f"{'yes' if r['ok'] else 'NO'}", flush=True)
        except Exception as e:  # noqa: BLE001
            bad += 1
            print(f"{name:<18} ERROR {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
        gc.collect()
    print(f"\n{'all ok' if not bad else f'{bad} problem(s)'}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
