"""Train an MLP (or a small CNN) on MNIST with matmuls on the RK3588 NPU, and compare
loss curves and step time against plain CPU training from the same initial weights.

    python3 examples/train_mnist.py --model mlp --steps 300
    python3 examples/train_mnist.py --model cnn --steps 200 --batch 64

MNIST is downloaded once to ~/.cache/rk3588_torch/mnist (falls back to a synthetic
dataset if offline).
"""
import argparse
import copy
import gzip
import os
import sys
import time
import urllib.request

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import rk3588_torch as rt  # noqa: E402

MIRROR = "https://ossci-datasets.s3.amazonaws.com/mnist/"
FILES = {"train_x": "train-images-idx3-ubyte.gz", "train_y": "train-labels-idx1-ubyte.gz",
         "test_x": "t10k-images-idx3-ubyte.gz", "test_y": "t10k-labels-idx1-ubyte.gz"}


def load_mnist():
    root = os.path.expanduser("~/.cache/rk3588_torch/mnist")
    os.makedirs(root, exist_ok=True)
    out = {}
    try:
        for key, name in FILES.items():
            path = os.path.join(root, name)
            if not os.path.exists(path):
                urllib.request.urlretrieve(MIRROR + name, path + ".tmp")
                os.replace(path + ".tmp", path)
            with gzip.open(path, "rb") as f:
                data = np.frombuffer(f.read(), dtype=np.uint8)
            if key.endswith("_x"):
                out[key] = torch.from_numpy(data[16:].reshape(-1, 1, 28, 28).astype(np.float32) / 255.0)
            else:
                out[key] = torch.from_numpy(data[8:].astype(np.int64))
        mean, std = 0.1307, 0.3081
        out["train_x"] = (out["train_x"] - mean) / std
        out["test_x"] = (out["test_x"] - mean) / std
        return out, "MNIST"
    except Exception as e:  # offline: separable synthetic data with the same shapes
        print(f"MNIST download failed ({e}); using synthetic data")
        g = torch.Generator().manual_seed(0)
        centers = torch.randn(10, 1, 28, 28, generator=g)
        def make(n):
            y = torch.randint(0, 10, (n,), generator=g)
            return centers[y] + 0.8 * torch.randn(n, 1, 28, 28, generator=g), y
        tx, ty = make(20000)
        vx, vy = make(2000)
        return {"train_x": tx, "train_y": ty, "test_x": vx, "test_y": vy}, "synthetic"


def make_model(kind):
    if kind == "mlp-wide":
        return nn.Sequential(nn.Flatten(), nn.Linear(784, 2048), nn.ReLU(), nn.Linear(2048, 2048), nn.ReLU(),
                             nn.Linear(2048, 1024), nn.ReLU(), nn.Linear(1024, 10))
    if kind == "mlp":
        return nn.Sequential(nn.Flatten(), nn.Linear(784, 512), nn.ReLU(), nn.Linear(512, 256), nn.ReLU(),
                             nn.Linear(256, 10))
    return nn.Sequential(nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                         nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                         nn.Flatten(), nn.Linear(64 * 7 * 7, 128), nn.ReLU(), nn.Linear(128, 10))


def accuracy(model, x, y, use_npu):
    model.eval()
    correct = 0
    with torch.no_grad():
        for i in range(0, len(x), 1000):
            if use_npu:
                with rt.npu_mode():
                    logits = model(x[i:i + 1000])
            else:
                logits = model(x[i:i + 1000])
            correct += (logits.argmax(1) == y[i:i + 1000]).sum().item()
    model.train()
    return correct / len(x)


def npu_ctx(args):
    return rt.npu_mode(conv=not getattr(args, "no_conv", False), precision=getattr(args, "precision", "fp16"))


def train(model, data, args, use_npu):
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    g = torch.Generator().manual_seed(1)
    n = len(data["train_x"])
    losses, times = [], []
    for step in range(args.steps):
        idx = torch.randint(0, n, (args.batch,), generator=g)
        x, y = data["train_x"][idx], data["train_y"][idx]
        t0 = time.perf_counter()
        if use_npu:
            with npu_ctx(args):
                loss = F.cross_entropy(model(x), y)
                opt.zero_grad()
                loss.backward()
        else:
            loss = F.cross_entropy(model(x), y)
            opt.zero_grad()
            loss.backward()
        opt.step()
        times.append(time.perf_counter() - t0)
        losses.append(loss.item())
        if step % args.log_every == 0 or step == args.steps - 1:
            tag = "NPU" if use_npu else "CPU"
            print(f"  [{tag}] step {step:4d}  loss {loss.item():.4f}  step {times[-1] * 1000:6.1f} ms", flush=True)
    return losses, times


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["mlp", "mlp-wide", "cnn"], default="mlp")
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--threads", type=int, default=3)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--only", choices=["npu", "cpu", "both"], default="both")
    p.add_argument("--no-conv", action="store_true", help="keep conv2d on the CPU")
    p.add_argument("--precision", default="fp16", help="fp16 | fp16_out16 | bf16 | int8 | fp16x3")
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)

    data, name = load_mnist()
    base = make_model(args.model)
    base(data["train_x"][:2])  # materialize lazy shapes
    print(f"{name}: {len(data['train_x'])} train / {len(data['test_x'])} test, model={args.model}, "
          f"batch={args.batch}, steps={args.steps}, cpu threads={args.threads}")

    results = {}
    # CPU first: once the NPU library has started its worker threads, they share the cores
    for mode in (["cpu", "npu"] if args.only == "both" else [args.only]):
        model = copy.deepcopy(base)
        use_npu = mode == "npu"
        rt.stats(reset=True)
        losses, times = train(model, data, args, use_npu)
        acc = accuracy(model, data["test_x"], data["test_y"], use_npu)
        steady = sorted(times[5:])[len(times[5:]) // 2] if len(times) > 10 else sum(times) / len(times)
        results[mode] = (losses, steady, acc)
        st = rt.stats()
        print(f"{mode.upper()}: final loss (avg last 20) {np.mean(losses[-20:]):.4f}  test acc {acc * 100:.2f}%  "
              f"median step {steady * 1000:.1f} ms" + (f"  | NPU GEMMs {st['npu_calls']} ({st['npu_macs'] / 1e9:.2f} GMAC), "
              f"CPU fallback {st['cpu_calls']}" if use_npu else ""))
    if len(results) == 2:
        ln, lc = np.array(results["npu"][0]), np.array(results["cpu"][0])
        print(f"\nstep time: NPU {results['npu'][1] * 1000:.1f} ms vs CPU {results['cpu'][1] * 1000:.1f} ms "
              f"-> {results['cpu'][1] / results['npu'][1]:.2f}x")
        print(f"loss curves: mean |NPU - CPU| = {np.abs(ln - lc).mean():.4f} "
              f"(first 50 steps {np.abs(ln[:50] - lc[:50]).mean():.4f})")
        print(f"test accuracy: NPU {results['npu'][2] * 100:.2f}%  CPU {results['cpu'][2] * 100:.2f}%")


if __name__ == "__main__":
    main()
