"""Second model zoo: architectures that tests/test_zoo.py does not cover.

Same method as test_zoo.py (one training step, CPU fp32 vs convert() + npu_mode(), identical weights, gradient
error against the precision's own rounding bound). Sized to fit next to other work on a 4 GB board.

    python3 tests/test_zoo2.py [--only convnext,moe] [--precision fp16|bf16] [--reps 2] [--threads 3]
    python3 tests/test_zoo2.py --fake        # no NPU: checks routing and autograd plumbing only
"""
import argparse
import gc
import math
import os
import sys
import traceback

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
import test_zoo as z  # noqa: E402


# ----------------------------------------------------------------------------------------------------------
class LayerNorm2d(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.ln = nn.LayerNorm(c)

    def forward(self, x):
        return self.ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class ConvNeXtBlock(nn.Module):
    """7x7 depthwise + LayerNorm + 1x1 expand (as Linear) + GELU + 1x1 project + layer scale."""

    def __init__(self, c):
        super().__init__()
        self.dw = nn.Conv2d(c, c, 7, padding=3, groups=c)
        self.ln = nn.LayerNorm(c)
        self.fc1, self.fc2 = nn.Linear(c, 4 * c), nn.Linear(4 * c, c)
        self.gamma = nn.Parameter(1e-2 * torch.ones(c))

    def forward(self, x):
        y = self.dw(x).permute(0, 2, 3, 1)
        y = self.fc2(F.gelu(self.fc1(self.ln(y)))) * self.gamma
        return x + y.permute(0, 3, 1, 2)


class ConvNeXt(nn.Module):
    def __init__(self, classes=10):
        super().__init__()
        self.stem = nn.Conv2d(3, 48, 4, 4)
        self.b1 = nn.Sequential(LayerNorm2d(48), ConvNeXtBlock(48), ConvNeXtBlock(48))
        self.down = nn.Sequential(LayerNorm2d(48), nn.Conv2d(48, 96, 2, 2))
        self.b2 = nn.Sequential(ConvNeXtBlock(96), ConvNeXtBlock(96))
        self.head = nn.Linear(96, classes)

    def forward(self, x):
        x = self.b2(self.down(self.b1(self.stem(x))))
        return self.head(x.mean((2, 3)))


class SE(nn.Module):
    def __init__(self, c, r=4):
        super().__init__()
        self.f1, self.f2 = nn.Conv2d(c, c // r, 1), nn.Conv2d(c // r, c, 1)

    def forward(self, x):
        s = x.mean((2, 3), keepdim=True)
        return x * torch.sigmoid(self.f2(F.silu(self.f1(s))))


class MBConv(nn.Module):
    """EfficientNet block: expand 1x1, depthwise 3x3, squeeze-excite, project 1x1, SiLU, residual."""

    def __init__(self, c, e=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c, c * e, 1, bias=False), nn.BatchNorm2d(c * e), nn.SiLU(),
            nn.Conv2d(c * e, c * e, 3, padding=1, groups=c * e, bias=False), nn.BatchNorm2d(c * e), nn.SiLU(),
            SE(c * e), nn.Conv2d(c * e, c, 1, bias=False), nn.BatchNorm2d(c))

    def forward(self, x):
        return x + self.net(x)


class EfficientNetish(nn.Module):
    def __init__(self, classes=10):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 32, 3, 2, 1, bias=False), nn.BatchNorm2d(32), nn.SiLU())
        self.blocks = nn.Sequential(MBConv(32), MBConv(32), nn.Conv2d(32, 64, 3, 2, 1), MBConv(64))
        self.head = nn.Linear(64, classes)

    def forward(self, x):
        return self.head(self.blocks(self.stem(x)).mean((2, 3)))


class ResNeXtSE(nn.Module):
    """Grouped 3x3 bottlenecks (cardinality 8) with squeeze-excite."""

    def __init__(self, classes=10, c=128):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, c, 3, 2, 1), nn.BatchNorm2d(c), nn.ReLU())
        blk = lambda: nn.Sequential(  # noqa: E731
            nn.Conv2d(c, c, 1), nn.BatchNorm2d(c), nn.ReLU(),
            nn.Conv2d(c, c, 3, padding=1, groups=8), nn.BatchNorm2d(c), nn.ReLU(), nn.Conv2d(c, c, 1),
            nn.BatchNorm2d(c), SE(c))
        self.b1, self.b2 = blk(), blk()
        self.head = nn.Linear(c, classes)

    def forward(self, x):
        x = self.stem(x)
        x = F.relu(x + self.b1(x))
        x = F.relu(x + self.b2(x))
        return self.head(x.mean((2, 3)))


class SwinWindowBlock(nn.Module):
    """Windowed multi-head attention with a relative-position bias and a shifted window (roll), as in Swin."""

    def __init__(self, d=128, heads=4, win=8, hw=32):
        super().__init__()
        self.h, self.win, self.hw = heads, win, hw
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.bias = nn.Parameter(torch.zeros(heads, win * win, win * win))

    def forward(self, x, shift=0):
        B, H, W, C = x.shape
        y = self.n1(x)
        if shift:
            y = torch.roll(y, (-shift, -shift), (1, 2))
        w = self.win
        y = y.view(B, H // w, w, W // w, w, C).permute(0, 1, 3, 2, 4, 5).reshape(-1, w * w, C)
        q, k, v = self.qkv(y).view(-1, w * w, 3, self.h, C // self.h).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=self.bias.unsqueeze(0))
        y = self.proj(a.transpose(1, 2).reshape(-1, w * w, C))
        y = y.view(B, H // w, W // w, w, w, C).permute(0, 1, 3, 2, 4, 5).reshape(B, H, W, C)
        if shift:
            y = torch.roll(y, (shift, shift), (1, 2))
        x = x + y
        return x + self.mlp(self.n2(x))


class SwinTiny(nn.Module):
    def __init__(self, classes=10):
        super().__init__()
        self.embed = nn.Conv2d(3, 128, 4, 4)
        self.b1, self.b2 = SwinWindowBlock(), SwinWindowBlock()
        self.head = nn.Linear(128, classes)

    def forward(self, x):
        x = self.embed(x).permute(0, 2, 3, 1)
        x = self.b2(self.b1(x), shift=4)
        return self.head(x.mean((1, 2)))


class MoE(nn.Module):
    """Top-2 mixture-of-experts feed-forward (indexed dispatch, ragged per-expert batches) in a transformer block."""

    def __init__(self, d=256, experts=4, ff=512, vocab=3000):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.attn = nn.MultiheadAttention(d, 4, batch_first=True)
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.gate = nn.Linear(d, experts)
        self.exp = nn.ModuleList(nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Linear(ff, d)) for _ in range(experts))
        self.head = nn.Linear(d, vocab)

    def forward(self, idx):
        x = self.emb(idx)
        h = self.n1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        h = self.n2(x).reshape(-1, x.shape[-1])
        w, sel = torch.topk(torch.softmax(self.gate(h), -1), 2, -1)
        w = w / w.sum(-1, keepdim=True)
        out = torch.zeros_like(h)
        for e, ex in enumerate(self.exp):
            tok, slot = torch.where(sel == e)
            if tok.numel():
                out.index_add_(0, tok, ex(h[tok]) * w[tok, slot].unsqueeze(-1))
        return self.head(x + out.view_as(x))


class MLPMixer(nn.Module):
    def __init__(self, patches=64, d=256, layers=3, classes=10):
        super().__init__()
        self.patch = nn.Conv2d(3, d, 8, 8)
        self.tok = nn.ModuleList(nn.Sequential(nn.LayerNorm(d), nn.Linear(patches, 2 * patches), nn.GELU(),
                                               nn.Linear(2 * patches, patches)) for _ in range(layers))
        self.chn = nn.ModuleList(nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 4 * d), nn.GELU(),
                                               nn.Linear(4 * d, d)) for _ in range(layers))
        self.head = nn.Linear(d, classes)

    def forward(self, x):
        x = self.patch(x).flatten(2).transpose(1, 2)           # B, P, D
        for t, c in zip(self.tok, self.chn):
            ln, f1, act, f2 = t
            x = x + f2(act(f1(ln(x).transpose(1, 2)))).transpose(1, 2)
            x = x + c(x)
        return self.head(x.mean(1))


class PointNet(nn.Module):
    """Shared per-point MLP as Conv1d(k=1), max-pool over points, classifier."""

    def __init__(self, classes=10):
        super().__init__()
        ch = [3, 64, 128, 512]
        self.convs = nn.ModuleList(nn.Conv1d(a, b, 1) for a, b in zip(ch[:-1], ch[1:]))
        self.bns = nn.ModuleList(nn.BatchNorm1d(b) for b in ch[1:])
        self.fc = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, classes))

    def forward(self, x):
        for c, b in zip(self.convs, self.bns):
            x = F.relu(b(c(x)))
        return self.fc(x.max(-1)[0])


class AudioConvEncoder(nn.Module):
    """wav2vec / Whisper-style front end: strided Conv1d stack on a raw waveform, then a transformer layer."""

    def __init__(self, d=192, classes=10):
        super().__init__()
        self.convs = nn.Sequential(nn.Conv1d(1, 64, 10, 5), nn.GELU(), nn.Conv1d(64, 128, 3, 2), nn.GELU(),
                                   nn.Conv1d(128, d, 3, 2), nn.GELU())
        layer = nn.TransformerEncoderLayer(d, 4, 4 * d, dropout=0.0, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.head = nn.Linear(d, classes)

    def forward(self, w):
        return self.head(self.enc(self.convs(w).transpose(1, 2)).mean(1))


class Perceiver(nn.Module):
    """Latent array cross-attending to a long input (Q from 64 latents, K/V from 1024 inputs), then self-attention."""

    def __init__(self, d=192, classes=10):
        super().__init__()
        self.inp = nn.Linear(16, d)
        self.lat = nn.Parameter(torch.randn(64, d) * 0.02)
        self.cross = nn.MultiheadAttention(d, 4, batch_first=True)
        self.self_ = nn.TransformerEncoderLayer(d, 4, 2 * d, dropout=0.0, batch_first=True)
        self.head = nn.Linear(d, classes)

    def forward(self, x):
        k = self.inp(x)
        q = self.lat.expand(x.shape[0], -1, -1)
        h = self.cross(q, k, k, need_weights=False)[0]
        return self.head(self.self_(h).mean(1))


class VideoCNN3D(nn.Module):
    """Conv3d (no NPU path: must stay correct on the CPU) between NPU linears."""

    def __init__(self, classes=10):
        super().__init__()
        self.c1 = nn.Conv3d(3, 16, 3, padding=1)
        self.c2 = nn.Conv3d(16, 32, 3, stride=2, padding=1)
        self.fc = nn.Sequential(nn.Linear(32, 256), nn.ReLU(), nn.Linear(256, classes))

    def forward(self, x):
        x = F.relu(self.c2(F.relu(self.c1(x))))
        return self.fc(x.mean((2, 3, 4)))


class PaddingModesCNN(nn.Module):
    """reflect / replicate / circular padding (the NPU does zero padding only)."""

    def __init__(self, classes=10):
        super().__init__()
        self.c1 = nn.Conv2d(3, 32, 3, padding=1, padding_mode="reflect")
        self.c2 = nn.Conv2d(32, 64, 3, padding=1, padding_mode="replicate")
        self.c3 = nn.Conv2d(64, 64, 3, padding=2, padding_mode="circular", dilation=2)
        self.head = nn.Linear(64, classes)

    def forward(self, x):
        return self.head(F.relu(self.c3(F.relu(self.c2(F.relu(self.c1(x)))))).mean((2, 3)))


class DeepLabASPP(nn.Module):
    """Segmentation head: atrous spatial pyramid (dilations 1, 6, 12, 18), bilinear upsampling."""

    def __init__(self, classes=5):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),
                                  nn.Conv2d(64, 96, 3, 2, 1), nn.BatchNorm2d(96), nn.ReLU())
        self.aspp = nn.ModuleList(nn.Conv2d(96, 64, 3, padding=d, dilation=d) if d > 1 else nn.Conv2d(96, 64, 1)
                                  for d in (1, 6, 12, 18))
        self.fuse = nn.Conv2d(256, 64, 1)
        self.cls = nn.Conv2d(64, classes, 1)

    def forward(self, x):
        f = self.stem(x)
        y = F.relu(self.fuse(torch.cat([F.relu(a(f)) for a in self.aspp], 1)))
        return F.interpolate(self.cls(y), size=x.shape[-2:], mode="bilinear", align_corners=False)


class ConvAE(nn.Module):
    """Convolutional autoencoder: strided convs down, 2x2 stride-2 transposed convs up."""

    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(nn.Conv2d(3, 32, 3, 2, 1), nn.BatchNorm2d(32), nn.ReLU(),
                                 nn.Conv2d(32, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.ReLU())
        self.dec = nn.Sequential(nn.ConvTranspose2d(64, 32, 2, 2), nn.BatchNorm2d(32), nn.ReLU(),
                                 nn.ConvTranspose2d(32, 3, 2, 2))

    def forward(self, x):
        return self.dec(self.enc(x))


class Bf16LoRA(nn.Module):
    """Frozen bf16 base weights + fp32 LoRA adapters: the usual fine-tuning layout."""

    def __init__(self, d=512, layers=3):
        super().__init__()
        self.base = nn.ModuleList(nn.Linear(d, d) for _ in range(layers))
        for b in self.base:
            b.requires_grad_(False)
            b.to(torch.bfloat16)
        self.A = nn.ParameterList(nn.Parameter(torch.randn(d, 16) * 0.02) for _ in range(layers))
        self.B = nn.ParameterList(nn.Parameter(torch.zeros(16, d)) for _ in range(layers))
        self.head = nn.Linear(d, 10)

    def forward(self, x):
        for b, A, B in zip(self.base, self.A, self.B):
            x = x + F.gelu(b(x.to(torch.bfloat16)).float() + (x @ A) @ B)
        return self.head(x)


def cases():
    g = torch.Generator().manual_seed(1)
    rn = lambda *s: torch.randn(*s, generator=g)  # noqa: E731
    ri = lambda hi, *s: torch.randint(0, hi, s, generator=g)  # noqa: E731
    ce = lambda o, x, y: F.cross_entropy(o, y)  # noqa: E731
    C = {}
    C["convnext"] = (lambda: ConvNeXt(), lambda: (rn(16, 3, 64, 64), ri(10, 16)), ce)
    C["efficientnet"] = (lambda: EfficientNetish(), lambda: (rn(16, 3, 64, 64), ri(10, 16)), ce)
    C["resnext_se"] = (lambda: ResNeXtSE(), lambda: (rn(16, 3, 64, 64), ri(10, 16)), ce)
    C["swin_window"] = (lambda: SwinTiny(), lambda: (rn(8, 3, 128, 128), ri(10, 8)), ce)
    C["moe_top2"] = (lambda: MoE(), lambda: (ri(3000, 8, 64),),
                     lambda o, i: F.cross_entropy(o[:, :-1].reshape(-1, o.shape[-1]), i[:, 1:].reshape(-1)))
    C["mlp_mixer"] = (lambda: MLPMixer(), lambda: (rn(16, 3, 64, 64), ri(10, 16)), ce)
    C["pointnet"] = (lambda: PointNet(), lambda: (rn(32, 3, 1024), ri(10, 32)), ce)
    C["audio_conv_tf"] = (lambda: AudioConvEncoder(), lambda: (rn(8, 1, 16000), ri(10, 8)), ce)
    C["perceiver"] = (lambda: Perceiver(), lambda: (rn(8, 1024, 16), ri(10, 8)), ce)
    C["conv3d_video"] = (lambda: VideoCNN3D(), lambda: (rn(4, 3, 8, 24, 24), ri(10, 4)), ce)
    C["padding_modes"] = (lambda: PaddingModesCNN(), lambda: (rn(8, 3, 32, 32), ri(10, 8)), ce)
    C["deeplab_aspp"] = (lambda: DeepLabASPP(), lambda: (rn(4, 3, 96, 96), ri(5, 4, 96, 96)),
                         lambda o, x, y: F.cross_entropy(o, y))
    C["conv_autoencoder"] = (lambda: ConvAE(), lambda: (rn(16, 3, 64, 64),), lambda o, x: F.mse_loss(o, x))
    C["lora_bf16_base"] = (lambda: Bf16LoRA(), lambda: (rn(128, 512), ri(10, 128)), ce)
    return C


N_IN = {"convnext": 1, "efficientnet": 1, "resnext_se": 1, "swin_window": 1, "moe_top2": 1, "mlp_mixer": 1,
        "pointnet": 1, "audio_conv_tf": 1, "perceiver": 1, "conv3d_video": 1, "padding_modes": 1,
        "deeplab_aspp": 1, "conv_autoencoder": 1, "lora_bf16_base": 1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--skip", default="")
    ap.add_argument("--precision", default="fp16")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--fake", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    if args.fake:
        z.install_fake()
    z.N_IN.update(N_IN)
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
            r = z.run_case(name, mk, mkin, lf, args.precision, args.reps, tol)
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
