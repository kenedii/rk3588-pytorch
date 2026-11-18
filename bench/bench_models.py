"""What to expect for different model types: one training step on the CPU vs the NPU in
each precision, via rk3588_torch.compare() (same weights, same batch).

    python3 bench/bench_models.py [--threads 3] [--models mlp,cnn,resnet,transformer]
"""
import argparse
import io
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import rk3588_torch as rt  # noqa: E402


class Block(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.b1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.b2 = nn.BatchNorm2d(cout)
        self.sc = (nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))
                   if stride != 1 or cin != cout else nn.Identity())

    def forward(self, x):
        y = torch.relu(self.b1(self.c1(x)))
        return torch.relu(self.b2(self.c2(y)) + self.sc(x))


def resnet(width=64):
    w = width
    return nn.Sequential(nn.Conv2d(3, w, 3, 1, 1, bias=False), nn.BatchNorm2d(w), nn.ReLU(),
                         Block(w, w, 1), Block(w, 2 * w, 2), Block(2 * w, 4 * w, 2), Block(4 * w, 8 * w, 2),
                         nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(8 * w, 10))


class Transformer(nn.Module):
    def __init__(self, vocab=1000, d=256, heads=8, layers=4, ff=1024):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        layer = nn.TransformerEncoderLayer(d, heads, ff, dropout=0.0, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.head = nn.Linear(d, vocab)

    def forward(self, tok):
        return self.head(self.enc(self.emb(tok)))


def models():
    torch.manual_seed(0)
    yield ("MLP 784-2048-2048-1024-10", "b256",
           nn.Sequential(nn.Flatten(), nn.Linear(784, 2048), nn.ReLU(), nn.Linear(2048, 2048), nn.ReLU(),
                         nn.Linear(2048, 1024), nn.ReLU(), nn.Linear(1024, 10)),
           torch.randn(256, 1, 28, 28), torch.randint(0, 10, (256,)), "mlp")
    yield ("CNN 2 conv + 2 fc (MNIST)", "b64",
           nn.Sequential(nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                         nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                         nn.Flatten(), nn.Linear(64 * 7 * 7, 128), nn.ReLU(), nn.Linear(128, 10)),
           torch.randn(64, 1, 28, 28), torch.randint(0, 10, (64,)), "cnn")
    yield ("ResNet-10 (64-512 ch), CIFAR 32x32", "b32", resnet(64), torch.randn(32, 3, 32, 32),
           torch.randint(0, 10, (32,)), "resnet")
    yield ("Transformer 4L d256 h8, seq 128", "b16", Transformer(), torch.randint(0, 1000, (16, 128)),
           torch.randint(0, 1000, (16, 128)), "transformer")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--threads", type=int, default=3)
    p.add_argument("--models", default="mlp,cnn,resnet,transformer")
    p.add_argument("--precisions", default="fp16,bf16")
    p.add_argument("--steps", type=int, default=5)
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    want = args.models.split(",")
    rows = []
    for name, batch, model, x, y, key in models():
        if key not in want:
            continue
        print(f"### {name} ({batch})", flush=True)
        r = rt.compare(model, x, y, precisions=tuple(args.precisions.split(",")), steps=args.steps, warmup=2,
                       file=io.StringIO())
        res = r["results"]
        print({k: {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items() if kk != "losses"}
               for k, v in res.items()}, flush=True)
        rows.append((name, batch, res, r["recommendation"]))
    print(f"\n| Model | Batch | CPU step | " + " | ".join(f"NPU {p}" for p in args.precisions.split(","))
          + " | Grad error | Recommendation |")
    print("|---|---|---:|" + "---:|" * len(args.precisions.split(",")) + "---:|---|")
    for name, batch, res, rec in rows:
        cells = []
        for prec in args.precisions.split(","):
            r = res.get(prec, {})
            cells.append("error" if "error" in r else f"{r['step_ms']:.0f} ms ({r['speedup']:.2f}x)")
        ge = res.get(rec if rec != "cpu" else args.precisions.split(",")[0], {}).get("grad_err", float("nan"))
        print(f"| {name} | {batch[1:]} | {res['cpu']['step_ms']:.0f} ms | " + " | ".join(cells)
              + f" | {ge:.1e} | `{rec}` |", flush=True)


if __name__ == "__main__":
    main()
