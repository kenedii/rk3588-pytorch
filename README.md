<div align="center">

# rk3588-pytorch

### Train PyTorch models on the Rockchip RK3588 NPU

**Up to 9× faster matrix multiplies and 3.5× faster training steps than the CPU,
on a single-board computer, with one line of code.**

![platform](https://img.shields.io/badge/platform-RK3588-blue)
![pytorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c)
![npu](https://img.shields.io/badge/NPU-731%20GFLOPS-brightgreen)
![tests](https://img.shields.io/badge/tests-all%20passing-success)
![license](https://img.shields.io/badge/license-GPL--3.0-lightgrey)

</div>

---

The RK3588's NPU (rated at 6 TOPS) is sold as an inference-only accelerator: you convert a finished
model with Rockchip's toolkit and run it. **rk3588-pytorch turns it into a training
device.** Every matrix multiply in the forward *and* backward pass runs on the NPU, while
PyTorch keeps doing everything else on the CPU. Your training loop doesn't change.

```python
import rk3588_torch

with rk3588_torch.npu_mode():          # matmuls, convolutions, attention -> NPU
    loss = F.cross_entropy(model(x), y)
    loss.backward()                    # gradients -> NPU too
optimizer.step()                       # everything else stays on the CPU
```

## Highlights

| | |
|---|---|
| ⚡ **731 GFLOPS** | 2048³ matmul end to end, **9.1×** the CPU. The NPU alone hits ~1.2 TFLOPS |
| 🏋️ **Up to 3.5× faster training** | MLP 3.5×, ResNet 2.2×, Transformer 1.6× per training step, same accuracy; LoRA on frozen layers 3.9× |
| 🎯 **Matches CPU training** | fp32 accumulation; same accuracy, mean loss difference 0.0006 on the wide MLP |
| 🔌 **Drop-in** | no model conversion, no RKNN runtime, no changes to your model code |
| 🎚️ **Five precisions** | fp16, bf16, int8, fp16-output, and an fp32-accurate split mode (~1e-7 error) |
| 🧠 **All three NPU cores** | work is split across cores automatically |
| 🧱 **Linear, conv, attention** | `Linear`, native NPU `Conv1d`/`Conv2d`/`ConvTranspose2d` (incl. grouped and dilated), `einsum`, attention, `nn.MultiheadAttention` / `nn.Transformer*`, `bmm`, `torch.compile` |
| 🧪 **17 architectures tested** | GPT, Llama-style (RoPE, GQA, SwiGLU), ViT, BERT, seq2seq Transformer, ResNet-18, MobileNet, UNet, DCGAN, TCN, LSTM/GRU, VAE, DLRM, GNN, CLIP-style: all train with CPU-matching gradients (`tests/test_zoo.py`) |
| 🪶 **Depthwise conv** | NEON CPU kernels replace PyTorch's slow depthwise backward: MobileNet 1.50x → 2.23x |
| 🔁 **LSTM / GRU** | after `convert()`: input projections, W_hh gradients and large recurrent steps on the NPU, 1.4-5.8x per layer |
| 🔗 **Fused bias + ReLU** | added while the NPU output is unpacked; a ReLU with no bias runs in the NPU's own DPU stage |
| 🧊 **Weight cache** | frozen or unchanged weights (LoRA, eval, inference) are converted for the NPU once: 5.5× faster MLP inference |
| 🛟 **Can't make things worse** | NPU errors, wrong results and NaNs fall back to the CPU automatically; small ops stay on the CPU |
| 🩺 **Tells you if it's worth it** | `doctor` checks your setup, `compare()` measures your model and recommends a precision |
| 🔬 **Built from scratch** | writes the NPU's register command streams directly; reverse-engineering tools included |

## Getting started

**1. Install** (on the board: an RK3588 with the vendor `rknpu` driver)

```sh
git clone https://github.com/kenedii/rk3588-pytorch && cd rk3588-pytorch
python3 -m venv venv && . venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install .                          # builds the NPU library and installs rk3588_torch
```

**2. Check the setup.** `doctor` verifies the driver, permissions and library, runs a
self-test in every precision, and prints the fix for anything that's wrong:

```sh
python -m rk3588_torch doctor
```

**3. Wrap your training step.** Nothing else in your code changes:

```python
import torch, rk3588_torch
torch.set_num_threads(3)               # leave a core for the NPU's packing threads

for x, y in loader:
    with rk3588_torch.npu_mode():      # or npu_mode(precision="bf16")
        loss = loss_fn(model(x), y)
        optimizer.zero_grad()
        loss.backward()
    optimizer.step()
```

**4. Find out if it's worth it for your model**, and which precision to use. `compare()`
trains a few steps on the CPU and on the NPU in each precision, from identical copies of
your model, and recommends one:

```python
rk3588_torch.compare(model, x, y)      # y: class labels (cross-entropy) or targets (MSE)
```
```
mode           step ms  speedup   grad err cpu same prec  loss diff  on NPU  fallbk  ok
cpu              209.6    1.00x          -             -          -       -       -
fp16              63.4    3.30x    2.2e-02       2.1e-02   1.61e-06    100%       0  yes
bf16              61.8    3.39x    5.9e-02       5.9e-02   6.05e-06    100%       0  yes

recommendation: npu_mode(precision='bf16')  (3.39x faster, gradient error 5.9e-02)
```
(wide MLP 784-2048-2048-1024-10, batch 256.) *grad err* is the first-step gradient's
difference from CPU fp32; *cpu same prec* is the same step on the CPU with inputs, weights
and gradients rounded to that precision. They match: the NPU adds no error beyond what
the precision itself implies.

**5. See where the time goes:**

```python
with rk3588_torch.profile():
    with rk3588_torch.npu_mode():
        model(x).sum().backward()
```

Other entry points: `rk3588_torch.convert(model)` swaps `nn.Linear`, `nn.Conv1d`,
`nn.Conv2d` and `nn.ConvTranspose2d` for NPU versions so the model uses the NPU without
`npu_mode()` (and works with `torch.compile`). Inside an `nn.Sequential` it also folds each
`Linear`/`Conv` -> `ReLU` pair into one fused NPU call (state_dict keys don't change).
`rk3588_torch.wcache.info()` shows the weight cache, and the CLI has
`python -m rk3588_torch bench` and `python -m rk3588_torch calibrate`.

## Benchmarks

Radxa Rock 5B (4 GB), NPU at 800 MHz, CPU baseline is fp32 PyTorch on 3 threads.

**Matrix multiply**, fp32 in and out, including all CPU-side conversion:

| Size (M × K × N) | CPU | NPU | Speedup |
|---|---:|---:|---:|
| 512 × 512 × 512 | 71 GFLOPS | 235 GFLOPS | 3.3× |
| 1024 × 1024 × 1024 | 79 GFLOPS | 469 GFLOPS | 5.9× |
| 2048 × 512 × 2048 | 78 GFLOPS | 452 GFLOPS | 5.8× |
| 2048 × 2048 × 2048 | 80 GFLOPS | **731 GFLOPS** | **9.1×** |

These were measured in fp16 without auto-scaling (`RKNPU_AUTOSCALE=0`). The default,
auto-scaled fp16 is ~15% slower on large GEMMs; bf16 matches these numbers (731 GFLOPS
at 2048³). See [Precisions](#precisions) for every mode.

**Training steps by model type** (forward + backward + SGD step, from identical weights;
`bench/bench_models.py`, which uses `compare()`):

| Model | Batch | CPU fp32 | NPU fp16 | NPU bf16 | Recommended |
|---|---:|---:|---:|---:|---|
| MLP 784-2048-2048-1024-10 | 256 | 208 ms | 66 ms (3.15×) | **60 ms (3.46×)** | `bf16` |
| ResNet-10 (64-512 channels, BatchNorm), 32×32 images | 32 | 1988 ms | 955 ms (2.08×) | **897 ms (2.22×)** | `bf16` |
| Transformer encoder, 4 layers, d=256, 8 heads, seq 128 | 16 | 1026 ms | 664 ms (1.54×) | **643 ms (1.60×)** | `bf16` |
| Small CNN (2 conv + 2 fc), 28×28 | 64 | 77 ms | **63 ms (1.21×)** | 65 ms (1.19×) | `fp16` |

Every NPU run's gradients match the CPU to within the rounding of its precision (that's
what `compare()` checks before recommending). In full MNIST training runs every
precision reaches the same test accuracy as the CPU (see [Precisions](#precisions)).

The bigger the layers and batches, the bigger the win. What's left on the CPU (norms,
activations, pooling, softmax, embeddings, the optimizer) caps the speedup; `profile()`
shows that split for your own model.

**Newer ops, the weight cache and fusion** (`bench/bench_new_ops.py`, fp16, median of
5-7 runs; "train step" is forward + backward + SGD):

| Workload | CPU fp32 | NPU | Speedup |
|---|---:|---:|---:|
| MLP inference, batch 32, 42M params, no weight cache | 70.3 ms | 27.5 ms | 2.6× |
| MLP inference, batch 32, 42M params, **weight cache** | 70.3 ms | **12.8 ms** | **5.5×** |
| LoRA r=16 on a frozen 42M MLP, batch 128, no weight cache | 260 ms | 148 ms | 1.8× |
| LoRA r=16 on a frozen 42M MLP, batch 128, **weight cache** | 260 ms | **66 ms** | **3.9×** |
| `einsum('bnd,de->bne')`, 32×512×1024 @ 1024×1024 | 444 ms | 68 ms | 6.6× |
| `einsum('bhqd,bhkd->bhqk')`, 16×8 heads, 256×256×64 | 58 ms | 21 ms | 2.8× |
| Grouped conv 3×3, 256→256, groups=4, 28×28 (train step) | 362 ms | 80 ms | 4.5× |
| Grouped conv 3×3, 256→256, groups=8, 28×28 (train step) | 87 ms | 70 ms | 1.2× |
| Conv1d stack, 3 × 256 channels, k=5, length 2000 (train step) | 1455 ms | 663 ms | 2.2× |
| UNet up-block: ConvTranspose 2×2 s2 256→128 + two 3×3 convs (train step) | 857 ms | 546 ms | 1.6× |
| MobileNetV2 block, depthwise on the CPU (train step, before the NEON depthwise kernels) | 704 ms | 497 ms | 1.4× |
| ConvTranspose2d 4×4 s2 p1, 128→64, 32×32 (train step) | 174 ms | 188 ms | 0.9× |
| Linear 256×2048×2048 + bias + ReLU, unfused / fused | 55.6 ms | 6.5 / 6.3 ms | 8.6× / 8.8× |

The weight cache is what makes inference and LoRA fast: converting fp32 weights for the NPU
is a big part of each call, and frozen weights now only pay for it once. Fusing bias and ReLU
saves an extra pass over the output; it's small next to the matmul itself. Transposed
convolutions with overlapping kernels (4×4, stride 2) spend their time in the CPU-side
scatter-add, so they only break even; the common 2×2 stride-2 upsampling layer is a single
GEMM.

<details>
<summary>Full GEMM table (1 core vs 3 cores) and timing breakdown</summary>

| M × K × N | CPU fp32 | NPU, 1 core | NPU, 3 cores | speedup |
|---|---:|---:|---:|---:|
| 64 × 64 × 64 | 12 GFLOPS | 5 | 6 | 0.5× (CPU fallback) |
| 512 × 512 × 512 | 71 | 191 | 235 | 3.3× |
| 256 × 784 × 512 | 72 | 147 | 152 | 2.1× |
| 1024 × 1024 × 1024 | 79 | 343 | 469 | 5.9× |
| 2048 × 512 × 2048 | 78 | 276 | 452 | 5.8× |
| 512 × 4096 × 512 | 76 | 141 | 264 | 3.5× |
| 2048 × 2048 × 2048 | 80 | 434 | 731 | 9.1× |

The NPU alone runs a 2048³ GEMM in 14.4 ms (~1.2 TFLOPS). The rest of the 23.5 ms is
fp32 to fp16 packing (7.4 ms) and copying C out (1.8 ms) on the CPU. CPU numbers are
measured before the NPU library starts any threads, after a governor warm-up; more than
3 busy cores resets this board on its charger. Training step times come from separate
CPU-only and NPU processes.

</details>

**A linear layer's three training GEMMs** (`bench/bench_layouts.py`, fp16, end-to-end ms including all
CPU-side conversion; "before" is the previous release). The weight gradient reads its first operand through a
transpose, which used to be packed one element at a time:

| x (M x K) @ W^T (K -> N) | fwd | dX = dY W | dW = dY^T x | step total |
|---|---:|---:|---:|---:|
| 2048 x 384 -> 4096 (LM head) | 19.9 -> 21.1 | 22.2 -> 21.5 | **38.7 -> 18.9** | 80.8 -> 61.5 (1.31x) |
| 2048 x 2048 -> 2048 | 23.2 -> 22.2 | 28.2 -> 25.3 | **39.3 -> 27.4** | 90.6 -> 74.9 (1.21x) |
| 2048 x 384 -> 1536 (MLP up) | 7.0 -> 7.7 | 5.8 -> 6.2 | **12.0 -> 8.0** | 24.8 -> 21.9 (1.13x) |
| 256 x 2048 -> 2048 | 6.5 -> 5.9 | **11.5 -> 8.2** | 11.2 -> 10.5 | 29.2 -> 24.5 (1.19x) |

## Tested architectures

One training step (forward + backward) of 17 model families, CPU fp32 vs `npu_mode()` from identical weights
(`tests/test_zoo.py`, fp16, 3 threads, median of 3; NPU at 600 MHz). *grad err* is the gradient's distance from
the CPU fp32 step; *cpu lowp* is the same step on the CPU with operands rounded to fp16 where the NPU rounds them,
i.e. the error the precision itself causes. Every model passes (NPU error within 2x of that bound). With `precision="bf16"` the transformers gain
more (Llama-style 1.99x, GPT 1.82x) and DCGAN reaches 4.2x.

| Model | Batch / input | CPU fp32 | NPU fp16 | Speedup | grad err | cpu lowp |
|---|---|---:|---:|---:|---:|---:|
| ResNet-18 (BatchNorm, 7x7 stem) | 16 x 3x64x64 | 1205 ms | 343 ms | **3.52x** | 1.0e-1 | 1.1e-1 |
| DCGAN generator (ConvTranspose 4x4 s2) | 32 x 100 | 768 ms | 279 ms | **2.75x** | 1.2e-2 | 1.6e-2 |
| UNet (GroupNorm, SiLU, skip concat) | 8 x 3x64x64 | 1188 ms | 467 ms | **2.54x** | 1.2e-3 | 1.9e-3 |
| MobileNet (depthwise in the library's NEON kernels) | 32 x 3x64x64 | 2428 ms | 980 ms | **2.48x** | 3.0e-2 | 3.7e-2 |
| Dense GCN (A·X·W) | 1024 nodes | 56 ms | 25 ms | **2.24x** | 4.5e-4 | 2.0e-3 |
| CNN on channels_last tensors | 32 x 3x64x64 | 1047 ms | 478 ms | **2.19x** | 6.7e-4 | 1.4e-2 |
| GPT (causal SDPA, tied embeddings) | 8 x 256 tokens | 2267 ms | 1040 ms | **2.18x** | 3.1e-4 | 3.0e-4 |
| LSTM language model (2 layers, H = 512) | 32 x 64 tokens | 2016 ms | 1026 ms | **1.97x** | 3.1e-4 | 2.7e-2 |
| TCN, dilated Conv1d (d = 1..8) | 16 x 2000 | 1222 ms | 638 ms | 1.91x | 2.5e-4 | 5.1e-3 |
| Llama-style LM (RMSNorm, RoPE, GQA, SwiGLU) | 4 x 256 tokens | 947 ms | 505 ms | 1.88x | 1.2e-3 | 2.0e-3 |
| ViT (patch conv + nn.TransformerEncoder) | 32 x 3x64x64 | 1042 ms | 574 ms | 1.81x | 8.0e-4 | 7.6e-4 |
| BERT-style encoder, key padding mask | 16 x 128 tokens | 687 ms | 417 ms | 1.65x | 5.6e-4 | 6.2e-4 |
| VAE (MLP encoder/decoder) | 256 x 784 | 106 ms | 65 ms | 1.63x | 1.0e-3 | 1.6e-3 |
| GRU language model (2 layers, H = 512) | 32 x 64 tokens | 1540 ms | 954 ms | 1.61x | 4.3e-4 | 1.7e-2 |
| CLIP-style contrastive towers | 256 pairs | 72 ms | 49 ms | 1.47x | 6.9e-4 | 6.5e-4 |
| nn.Transformer seq2seq (cross-attention) | 16 x 64 tokens | 467 ms | 384 ms | 1.22x | 9.6e-3 | 7.8e-3 |
| DLRM (EmbeddingBag + MLPs + dot interaction) | 512 | 123 ms | 105 ms | 1.17x | 1.9e-3 | 1.9e-3 |

ResNet-18's large gradient error is what fp16 itself does to it at batch 16 (the CPU run with fp16-rounded
operands differs from fp32 by the same 1e-1); use bf16 or fp16x3 if that matters. Depthwise convolutions run in
the library's NEON CPU kernels (PyTorch's own depthwise backward is several times slower). The LSTM/GRU rows use
`convert()`, which swaps in the NPU recurrent layers; with `npu_mode()` alone they were 1.17x and 1.22x.
`python tests/diag_zoo.py <model>` prints the per-parameter error and a profile for any of them.

## Precisions

The NPU's multipliers natively handle **fp16, bf16 and int8** (with fp32, fp16 or int32
outputs). bf16 isn't exposed by Rockchip's runtime on the RK3588, and was found by
probing the hardware. On top of these, the library provides an **fp32-accurate mode**
that splits each value into two fp16 halves. Every mode takes and returns ordinary fp32
tensors.

```python
rk3588_torch.set_precision("bf16")                  # global
with rk3588_torch.npu_mode(precision="fp16x3"):     # or per block
    ...
```

| Mode | What runs on the NPU | Rel. error vs fp64 | 1024³ | 2048³ | Use it for |
|---|---|---:|---:|---:|---|
| CPU fp32 (reference) | | 3e-7 | 77 GF | 79 GF | |
| `fp16` (default) | fp16 × fp16 → fp32, auto-scaled | 2.9e-4 | 375 GF | 608 GF | general training |
| `bf16` | bf16 × bf16 → fp32 | 2.4e-3 | 448 GF | **731 GF** | fastest float mode, fp32 range |
| `fp16_out16` | fp16 × fp16 → fp16 | 3.6e-4 | 415 GF | 660 GF | inference-style speed, outputs must fit fp16 |
| `int8` | int8 × int8 → int32, per-row/column scales | 1.1e-2 | 308 GF | 620 GF | approximate / experimental |
| `fp16x3` | fp16 hi/lo split, 3 products | **1.3e-7** | 126 GF | 148 GF | fp32-level accuracy (1.6-1.9× the CPU on big GEMMs) |

**Training with each mode** (wide MLP 784-2048-2048-1024-10 on MNIST, batch 256, 100
steps, same initial weights; each run in its own process):

| Mode | Step time | vs CPU | Test accuracy | Final loss |
|---|---:|---:|---:|---:|
| CPU fp32 | 218.0 ms | 1.00× | 94.18% | 0.1909 |
| `bf16` | 75.5 ms | **2.89×** | 94.16% | 0.1903 |
| `fp16_out16` | 76.0 ms | 2.87× | 94.17% | 0.1905 |
| `fp16` (default) | 80.9 ms | 2.69× | 94.13% | 0.1905 |
| `int8` | 95.2 ms | 2.29× | 94.23% | 0.1896 |
| `fp16x3` | 279.4 ms | 0.78× | 94.17% | 0.1906 |

All five modes train to the same accuracy as the CPU. `fp16x3` is about accuracy, not
speed: its CPU-side splitting outweighs the NPU gain at these layer sizes.

* **`fp16` is auto-scaled.** Each operand is multiplied by an exact power of two before
  the conversion, so values far outside fp16's range (tested from 1e-30 to 1e5) keep full
  fp16 precision. By default the scaling is *optimistic*: operands are packed at scale 1 and the packed
  fp16 data is checked (half the bytes of the fp32 source); only an operand whose largest value is outside
  [2^-10, 65504] is scanned and packed again with a scale. That removes a full pass over both operands in the
  common case. `RKNPU_AUTOSCALE=1` always scans first (the old behaviour), `0` disables scaling.
* **`fp16x3` matches CPU fp32.** A = A_hi + A_lo·2⁻¹¹ (both fp16), and likewise B; the NPU
  computes A_hi·B_hi plus [A_hi | A_lo]·[B_lo ; B_hi] with fp32 accumulation. The
  error equals native fp32's, at ~3× the NPU work.
* **`int8` is quantized per row of A and per column of B**, with exact int32
  accumulation. The int8 MAC array is twice as wide as fp16's, but the CPU-side
  quantization currently eats most of that gain.
* Not available: int16 (hangs the CNA), and there is no fp32 or tf32 MAC path (fp32
  input codes are read as pairs of 16-bit values). Found with `tools/prec_probe.c`.

`tests/test_precision.py` checks every mode (60 checks, all passing), and
`bench/bench_precision.py` produces the table above.

## Fine-tuning large models

LoRA and full fine-tuning of language models use a few things the small examples above don't:

* **Frozen weights can stay in bf16 or fp16.** They go through the packed-weight cache like fp32 ones
  (after the second use they are packed once and reused), so a 0.5B model needs half the RAM. Operands
  too big to convert in one go (an embedding / `lm_head` of 150k rows, a huge logits gradient) are split
  into pieces automatically so no single op makes a multi-gigabyte fp32 copy; `RKNPU_CONVERT_MB` (default
  128) sets the piece size.
* **bf16/fp16 ops that stay on the CPU are computed in fp32.** The RK3588's Cortex-A76 cores have no bf16
  arithmetic, so PyTorch runs bf16 matmuls and attention on them through a very slow software path. Measured
  on Qwen2.5-0.5B (bf16 weights, LoRA, 128 tokens): one attention call took 0.6 s and the small `k_proj`
  linears 90 ms each, 29 s of a 45 s step. Under `npu_mode()` (and in `convert()`ed `NPULinear`) small ops that
  don't reach the NPU are now widened to fp32 and the result is cast back, so autograd and dtypes are
  unchanged: that step takes 11.7 s (2.9 to 11 tokens/s) with the same loss curve. `RKNPU_CPU_UPCAST=0`
  turns it off; it is off automatically on a CPU that has native bf16.
* **Grouped-query attention runs on the NPU.** `F.scaled_dot_product_attention(..., enable_gqa=True)` (Llama,
  Qwen, Mistral and most recent LLMs) is routed like ordinary attention; each key/value head is repeated for its
  query group. The softmax is done in fp32.
* **Gradient checkpointing.** Re-running a layer inside `backward()` has to take the same path as the
  forward pass. `backward()` inside the `npu_mode()` block works with any checkpoint function. If your
  trainer calls `backward()` outside the block, either call `rk3588_torch.enable()` once at the start, or
  use `rk3588_torch.checkpoint(layer, x)` in place of `torch.utils.checkpoint.checkpoint` (it remembers
  the active `npu_mode()` and precision). Prefer non-reentrant checkpointing (the default of
  `rk3588_torch.checkpoint`): reentrant checkpointing returns no gradients unless an input requires
  grad, which an embedding lookup's input never does.
* **The weight cache has a size.** Every frozen weight you reuse needs its packed copy in NPU memory
  (`RKNPU_WCACHE_MB`, default 256). A 24-layer model has far more than that, so every step evicts and
  re-packs most of them; the library warns once when that happens. `rk3588_torch.wcache.info()` shows
  the estimate (`packed_estimate_mb` against `cap_mb`). Raise the cap if the board has memory to spare, but only to cover the whole estimate: with least-recently-used
  eviction a loop that cycles through every layer gets no hits from a cache that holds only part of them
  (Qwen2.5-0.5B needs about 1344 MB; at 640 MB the hit and miss counts and the step time were unchanged).
* **Every new input shape is checked once.** The first call with a new GEMM shape is verified against
  the CPU on a few rows (huge weights are checked on a strided sample of columns), so variable sequence
  lengths pay that cost per length. Padding to a few fixed lengths keeps the number of shapes small.
* **Weak power supplies.** A board that resets under sustained training is browning out. Lower the
  load: `RKNPU_DUTY=0.6` (or `rk3588_torch.set_duty(0.6)`) pauses after each NPU call so the average draw
  drops, `torch.set_num_threads(2)` with `RKNPU_THREADS=2` lowers the CPU side, and capping the clocks
  helps further (e.g. `echo 600000000 | sudo tee /sys/class/devfreq/fdab0000.npu/max_freq`).
  `python -m rk3588_torch doctor` prints the current limits. A 30 W+ USB-C PD supply is the real fix.

## How it works

```
PyTorch op (linear, matmul/mm/bmm/addmm/baddbmm, einsum, conv1d/conv2d/conv_transpose2d,
   │  scaled_dot_product_attention)
   │  rk3588_torch.npu_mode() is a TorchFunctionMode that intercepts these ops
   │  (or convert(model) swaps in NPU modules and fuses Linear/Conv -> ReLU)
   │  einsum -> permute + one batched GEMM; conv1d -> conv2d of width 1;
   │  conv_transpose2d -> one GEMM for every kernel tap, then a CPU scatter-add (col2im)
   ▼
torch.library custom ops rknpu::mm / rknpu::linear / rknpu::conv2d, with autograd formulas:
   │  forward GEMM + dX = dY·W + dW = dYᵀ·X; conv dX as a convolution, dW as a GEMM
   │  (one batched GEMM over the groups); ReLU masks the gradient with the saved output
   │  operands passed as (pointer, strides): transposes cost nothing
   ▼
weight cache: a weight seen twice unchanged (same storage, version counter and sampled
   │  values) gets a cache key; the library then keeps its packed NPU copy
   ▼
safety layer: below the calibrated size -> CPU; first use of each shape checked against
   │  a CPU sample; NPU errors / NaNs -> recompute on CPU; repeated failures -> CPU only
   ▼
librknpu_gemm.so
   │  A: fp32 → fp16 row-major      B: fp32 → fp16 16-kernel groups (or from the cache)
   │  tile M so each task's rows fit in CBUF; build 108-register task streams,
   │  chain them through the NPU program counter, split across the 3 cores;
   │  bias + ReLU applied while unpacking the output (ReLU in the DPU when there's no bias)
   ▼
RK3588 NPU: CNA (MACs) → CORE → DPU. Matmuls run as 1×1 convolutions on row-major
data; real convolutions use the native NC1HWC2 layout with the hardware's kernel,
stride and padding support
```

Things that had to be discovered to make this work on a current kernel/driver:

* **Row-major mode.** Earlier reverse engineering packed A into the NPU's NC1HWC2
  layout and unpacked C. Capturing librknnrt's `rknn_matmul` command streams (with
  `tools/ioctl_spy.c`) showed that `CNA_CONV_CON1` bit 29 plus line stride `K/8` and
  surface stride 0 make the CNA read plain row-major fp16, and that DPU surface stride
  1 / surface add 4 / notch `N/4-1` make it write plain row-major fp32. Only B needs
  repacking.
* **Per-core IOMMU mappings.** The patched 0.9.8 driver keeps a separate mapping per
  NPU core; buffers created without `core_mask` are invisible to the core and the job
  hangs with the PC stalled (`int_raw_status 0xc0010000`).
* **Cache maintenance on new buffers.** Freshly allocated cacheable buffers carry dirty
  lines from the kernel's page zeroing; without a clean+invalidate right after
  allocation the first GEMM in each process came back corrupted.
* **CBUF split.** Leaving a single bank for streamed weights hangs the CNA; like
  librknnrt, keep at least 4 weight banks (or all the weights if they fit in fewer).
* **Three cores.** With all three cores in `core_mask` the driver reads the per-core
  task ranges from `subcore_task[2..4]`; each core runs its own PC chain. Large GEMMs
  run ~2.4× faster than on one core.
* **Threads on the right cores, and asleep when idle.** An unpinned OpenMP pool often
  landed on the Cortex-A55 cluster (packing ~10× slower), and libgomp's busy-waiting
  workers then stole the big cores from PyTorch's own threads, slowing the CPU parts
  of training by 2-3×. The library now uses its own small pthread pool: workers are
  pinned to A76 cores, sleep on a condition variable between jobs, and the caller's
  affinity is never touched.
* **Native convolution.** Feature maps in NC1HWC2 (8 fp16 channels per 16-byte atom),
  weights as [16-kernel group][32-channel block][ky][kx][16][32], kernel size, stride and
  padding in the CNA registers. Tall images are split into row slices that share one CBUF
  split (the weight-reuse bit breaks if an edge slice moves the bank boundary); wide
  channel counts are split into chunks whose partial sums are added on the CPU. Mesa's
  open-source Rocket driver was a useful reference for the register semantics.
* **Other precisions.** Capturing librknnrt's int8 and fp16-output streams gave their
  register constants (precision codes, output converter, surface add, notch). Probing
  the unused precision code 3 with a layout-independent test (B = all ones, so every
  output is a row sum) revealed a working bf16 MAC path.
* **Streamed weights need 512-multiple K chunks.** When the weights don't fit beside the
  data banks, the CNA streams them; for K chunks such as 2016, 3008 or 4000 (with N = 64)
  that hung the NPU, while multiples of 512 always worked. The library pads the chunk with
  zeros in that case. The FEATURE_GRAINS field (input rows + 1 per task) is 10 bits, so
  long feature maps (e.g. conv1d sequences) are split into row slices of at most 1022 rows.
* **ReLU in the DPU, and what didn't work.** Clearing the bypass bits of the DPU's BS
  stage (`DPU_BS_CFG` = 0x12) applies a ReLU to the accumulator before it is written out;
  it's used when no bias has to be added first. The hardware depthwise mode
  (`CONV_MODE` 3 in CNA and DPU plus `CORE_MISC_CFG` `DW_EN`, as Mesa programs it for
  int8) timed out with every fp16 weight layout tried (`tools/dw_probe.py`), so depthwise
  convolutions stay on the CPU, in the library's own NEON kernels (`rk3588_torch/dwconv.py`: forward, input
  and weight gradients, 3-6x faster than PyTorch's CPU depthwise backward). Grouped convolutions run as one task chain with each
  group's weights and input channels in their own blocks.
* **`nn.MultiheadAttention` and the `nn.Transformer*` layers.** They call
  `F.multi_head_attention_forward`, which a `TorchFunctionMode` sees as one opaque function, and PyTorch
  runs it with the mode switched off: every projection and the attention inside ran on the CPU (half
  of a ViT's multiply-adds). The router now implements that function itself (packed or separate
  projection weights, bool/float attention and key-padding masks, causal, dropout) with the projections
  on the NPU.
* **Attention placement is measured, not assumed.** The NPU computes attention as two batched GEMMs around
  a CPU softmax, with a full score matrix and ~5 us per task (one task per head). PyTorch 2.x's fused CPU
  flash attention was faster for every head dim up to 64 (by 10-60%, lengths 65-1024); at head dim 128
  the NPU wins. So by default attention goes to the NPU only from head dim 128 (`RKNPU_ATTENTION`); this
  alone took GPT from 1.65x to 1.93x and ViT from 1.33x to 1.74x per training step.
* **Fused NEON kernels for the CPU's element-wise work.** With the matmuls on the NPU, a transformer step's
  GELU, SiLU and RMSNorm become a visible share of it, and PyTorch's aarch64 wheel runs them through its generic
  kernels (`CPU capability usage: DEFAULT`). `rk3588_torch.eltwise` does each forward and backward in one NEON
  pass (exp to ~2 ulp, erf to 1.5e-7) on the library's thread pool, for fp32 and bf16 tensors: on 2048 x 1536,
  GELU forward + backward 60 -> 15 ms, RMSNorm 26 -> 7.5 ms, SiLU 16 -> 12.5 ms. `npu_mode()` routes `F.gelu`,
  `F.silu` and `F.rms_norm` (so `nn.GELU`, `nn.SiLU`, `nn.RMSNorm`) to them, and `convert()` replaces Hugging
  Face-style `*RMSNorm` modules (Qwen2, Llama, ...). GPT in the zoo: 1.95x -> 2.07x.
* **The NPU is often DRAM-bound, so the K split is planned.** The CBUF (384 KB) holds a tile of A rows and
  either all of the weights or a streamed slice of them. With a long K (e.g. 8192) only a dozen rows of A fit,
  so the whole weight matrix is re-read from DRAM for every dozen rows: the DDR controller's counters
  (`rockchip_ddr` PMU) show the NPU pulling 16-20 GB/s of the LPDDR4X's ~34 GB/s. Splitting K into chunks
  makes the row tiles taller and the weight traffic proportionally smaller, at the price of fp32 partial sums
  that the CPU adds up. A small cost model (NPU time = max(MACs / 680 G/s, bytes / 20 GB/s) + the CPU
  reduction) picks the split per GEMM: 2048 x 8192 x 2048 went from 609 ms to 85 ms, 4096 x 4096 x 1024 from
  75 ms to 51 ms; compute-bound shapes such as 2048^3 keep a single chunk.
* **Overlapping the CPU and the NPU doesn't pay here.** `RKNPU_PIPELINE=n` splits a GEMM into up to n waves
  and packs wave i+1 while the NPU runs wave i. It works, but it hardly helps: the packing and the NPU share
  the same DRAM, and the NPU slows down as much as the CPU work is hidden (13 -> 20 ms NPU time on the LM
  head shape). Off by default.
* **Two convolution padding errata.** With 2+ rows of zero padding, (a) a task that reuses the
  weights already in CBUF from the previous task (`WEIGHT_REUSE`) computes wrong results for every
  second image, and (b) independently, the first output rows of the first 16-kernel group come out
  wrong now and then (a race at task start; run to run different). Both showed up in the backward
  pass (the input gradient of a "valid" 3x3 convolution is a convolution with padding 2) and only for
  images after the first, which the first-use check didn't look at. The library now pads on the CPU
  when the padding is 2 or more, never reuses weights in such chains, and checks the first, middle and
  last image of a batch. Details: the `raxda re` notes, `07_npu_errata.md`.
* **Dilated convolutions as space-to-batch.** A convolution with dilation d is d (d_h x d_w in 2-D)
  ordinary convolutions over the phases of the input (`x[r::d]`). The phases are stacked along the batch,
  run as one undilated NPU convolution and interleaved back: exact, and no wasted multiply-adds. With
  a stride that is a multiple of the dilation only phase 0 is read.
* **Small feature maps as unfold + GEMM.** The native convolution needs at least 5 input rows, so the
  deep, wide layers of a ResNet at small resolutions (4x4 and 2x2 maps with 256-512 channels) used to run
  on the CPU, forward and backward, which was 73% of a ResNet-18 step. They now run as `unfold` + one NPU
  GEMM through the linear op (weight cache, fused bias/ReLU); ResNet-18 went from 1.0x to 3.3x.
* **Transposing packers in NEON.** The weight gradient reads its first operand through a transpose
  (dW = dY^T X), `dX = dY W` reads W row-major, and convolutions convert NCHW to 8-channel atoms: all
  transposes. They are now packed as 8x8 blocks in registers (load 8 rows, convert to fp16/bf16, three
  `trn` stages, 8 stores), and the convolution output is de-interleaved with `ld4`. The dW GEMM of a
  2048 x 384 x 4096 linear layer dropped from 38.7 ms to 18.9 ms (with optimistic scaling).
* **`memset(0)` on uncached NPU memory faults.** glibc zeroes with `DC ZVA`, which is
  not allowed on Device memory (this is why the original mtx512 tests die with SIGBUS);
  the library uses cacheable buffers with explicit syncs instead.

## Layout

| Path | What |
|---|---|
| `csrc/rknpu_gemm.c`, `rknpu_gemm.h` | NPU GEMM library: tiling, packing, register streams, submit |
| `csrc/npu_regs.h` | Register offsets and command-word encoding |
| `csrc/rknpu_uapi.h` | ioctl structs matching the rknpu 0.9.8 driver |
| `rk3588_torch/ops.py` | guarded NPU ops, custom ops with autograd, conv / conv1d / conv_transpose2d, einsum, attention |
| `rk3588_torch/mode.py` | `npu_mode()` op router, `convert()` (with ReLU fusion), the `NPU*` modules |
| `rk3588_torch/wcache.py` | packed-weight cache policy (when a weight counts as unchanged) |
| `rk3588_torch/tools.py` | `compare()` and `profile()` |
| `rk3588_torch/doctor.py`, `calibrate.py`, `__main__.py` | `python -m rk3588_torch doctor / calibrate / bench / info` |
| `rk3588_torch/_lib.py` | ctypes binding to the C library |
| `tests/test_ops.py` | GEMM / gradient correctness vs fp64 CPU references |
| `tests/test_precision.py` | Accuracy and range of every precision mode |
| `tests/test_conv.py` | Native convolution forward/backward over many geometries (`--bench` for timing) |
| `tests/test_new_ops.py` | conv1d, conv_transpose2d, grouped conv, einsum, fusion, weight cache (`--fake` runs without an NPU) |
| `tests/test_llm_finetune.py` | bf16/fp16 frozen weights, bounded conversion memory, gradient checkpointing, weight-cache sizing, duty throttle (`--fake` runs without an NPU) |
| `tests/test_robust.py` | Fallbacks, verification, batching, attention, `convert()`, `torch.compile`, tools |
| `tests/test_zoo.py` | 17 architectures: one training step CPU vs NPU, gradient error vs the precision's own error, speed (`--fake` runs without an NPU) |
| `tests/diag_zoo.py` | per-parameter gradient error and a profile for one zoo model |
| `tests/test_rnn.py` | NPU LSTM/GRU after `convert()` vs PyTorch: outputs, final states, all gradients (multi-layer, bidirectional, initial states) |
| `tests/test_wgrad.py` | conv weight gradient: copy and copy-free batched paths vs PyTorch (strides, padding, bias, groups, bf16, chunking) |
| `tests/test_dwconv.py` | depthwise conv kernels vs PyTorch: kernel sizes, strides, "same"/asymmetric padding, bf16, conv1d, routing, speed |
| `tests/test_eltwise.py` | fused GELU / SiLU / RMSNorm kernels vs PyTorch (fp32, bf16, extremes), routing and the HF RMSNorm swap |
| `tests/test_dilated.py` | dilated, `"same"` and asymmetric-padded convolutions (space-to-batch), forward and backward |
| `bench/bench_gemm.py` | CPU vs NPU GEMM throughput |
| `bench/bench_precision.py` | Speed and error of every precision mode |
| `bench/bench_models.py` | MLP / CNN / ResNet / Transformer training steps, CPU vs NPU |
| `bench/bench_wgrad.py` | conv weight gradient, copy vs copy-free batched path, per layer shape |
| `bench/bench_convpack.py` | where a native NPU conv spends its time: pack / NPU / unpack (pack + unpack are 39-63% of a call) |
| `bench/prof_copies.py` | which `copy_` / `im2col` calls dominate a training step, grouped by shape |
| `bench/bench_rnn.py` | LSTM/GRU training step: PyTorch vs NPU input projections, with and without the per-step recurrent GEMM on the NPU |
| `bench/bench_eltwise.py` | CPU cost of the element-wise ops of a training step (GELU, norms, softmax, ...) |
| `bench/bench_layouts.py` | time breakdown (pack / sync / NPU / unpack) of a linear layer's three training GEMMs |
| `bench/bench_new_ops.py` | fusion, weight cache (inference, LoRA), conv_transpose2d, grouped conv, conv1d, einsum |
| `examples/train_mnist.py` | MLP / CNN training on NPU vs CPU |
| `tools/conv_probe.py` | Probe native convolution layouts |
| `tools/dw_probe.py` | Probe the depthwise mode and the DPU ReLU |
| `tools/gemm_test.c` | C-level correctness + timing |
| `tools/ioctl_spy.c` | `LD_PRELOAD` shim logging RKNPU ioctls and dumping command streams |
| `tools/rknn_mm_test.c` | Reference matmul through Rockchip's `rknn_matmul` API |
| `tools/prec_probe.c` | Probe NPU precision codes with a layout-independent test |
| `tools/rknn_mm_probe.c` | Run any `rknn_matmul` type (for capturing its command stream) |
| `tools/probe_mem.c` | Which allocation flags can be mapped and touched |

## Configuration

Everything works with no configuration. These environment variables are there for
experimenting:

| Variable | Default | What it does |
|---|---|---|
| `RKNPU_PRECISION` | `fp16` | default precision (`fp16`, `bf16`, `int8`, `fp16x3`, `fp16_out16`) |
| `RKNPU_MIN_MACS` | calibrated | smallest op (multiply-adds) sent to the NPU; unset = measured on first use and cached in `~/.cache/rk3588_torch/` |
| `RKNPU_VERIFY` | `first` | check each new shape against a CPU sample: `first`, `always`, or `off` |
| `RKNPU_CHECK_FINITE` | `1` | send ops whose inputs contain inf/nan to the CPU (the NPU turns them into finite values, which would hide a diverging run). The default lets the engine report non-finite fp16 operands from the scan it already does while scaling (no extra pass; GPT 2.19x -> 2.32x, BERT 1.61x -> 1.73x); `full` scans every operand and output in Python as before; `0` disables the check |
| `RKNPU_MAX_FAILURES` | `3` | NPU errors before it is disabled for the rest of the process |
| `RKNPU_DISABLE` | `0` | `1` = run everything on the CPU (same code path, for A/B tests) |
| `RKNPU_AUTOSCALE` | `2` | power-of-two scaling in `fp16` mode: `2` optimistic (pack at scale 1, rescale only operands outside [2^-10, 65504]), `1` always scan first, `0` off (values above 65504 overflow) |
| `RKNPU_RNN_REC_NPU` | `auto` | per-step LSTM/GRU `h @ W_hh` on the NPU: `auto` when a step has ≥ `RKNPU_RNN_REC_MACS` (30M) multiply-adds, `1` always, `0` never |
| `RKNPU_DW_MODE` | `auto` | conv weight gradient: `auto` runs a copy-free batched GEMM on the transposed im2col view when the image has at least `RKNPU_DW_RATIO` (2) x as many positions as output channels and `RKNPU_DW_MIN_L` (256) positions; `copy` / `batched` force one path (`RKNPU_DW_CHUNK_MB`: partial results per call, 32) |
| `RKNPU_DW_IM2COL` | `1` | the batched weight-gradient GEMM gathers its im2col operand straight from x while packing (no fp32 unfold matrix; fp16/bf16; spot-checked, falls back to unfold). 2-2.3x faster per GEMM; UNet step 2.57x -> 3.13x, channels_last 2.32x -> 2.86x. One GEMM over K = N*L (no per-image partials), faster than the copy path at every layer measured, so it ignores `RKNPU_DW_RATIO`. 0 = the unfold paths |
| `RKNPU_CONV_OUT16` | `1` | the NPU writes single-chunk fp16 convolutions' output as fp16 (half the readback traffic; an output that overflows fp16 redoes the call in fp32). 0 disables. ResNet-18 3.49x -> 3.74x, MobileNet 2.54x -> 2.64x |
| `RKNPU_DWCONV` | `1` | depthwise convolutions in the library's NEON CPU kernels (`RKNPU_DWCONV_MIN`: smallest input, 16384 elements) |
| `RKNPU_ELTWISE` | `1` | fused NEON GELU / SiLU / RMSNorm kernels under `npu_mode()` and in `convert()` (`RKNPU_ELTWISE_MIN`: smallest tensor, 32768 elements) |
| `RKNPU_KPLAN` | `1` | choose each GEMM's K split with the DRAM/compute cost model (`0`: one chunk up to `RKNPU_MAX_K`); `RKNPU_PLAN_GMACS` / `_NPU_GBS` / `_CPU_GBS` (680 / 20 / 3) are its constants |
| `RKNPU_PIPELINE` | `1` | waves to split a GEMM into to overlap CPU packing with the NPU (1 = off; see How it works) |
| `RKNPU_ATTENTION` | `auto` | where the attention GEMMs run: `auto` = NPU only for head dim >= 128 (PyTorch's fused CPU flash attention is faster below that), `npu` = always, `cpu` = never. Projections always use the NPU |
| `RKNPU_HW_PAD_MAX` | `1` | largest convolution padding left to the NPU; larger padding is applied on the CPU first (works around a hardware erratum, see How it works) |
| `RKNPU_IM2COL_MAX_MB` | `256` | largest unfolded input for convolutions run as unfold + GEMM (feature maps under 5 rows, padding > 15, ...) |
| `RKNPU_PINGPONG` | `1` | `0` submits without the PC ping-pong flag (debugging) |
| `RKNPU_THREADS` | `3` | CPU threads used for packing data for the NPU (pinned to the big cores) |
| `RKNPU_MODE` | `2` | `2` = all three NPU cores, `1` = one core, `0` = one task per submit (debugging) |
| `RKNPU_MAX_BUF_MB` | `512` | cap on NPU buffer memory; bigger ops are split |
| `RKNPU_CONVERT_MB` | `128` | an op whose bf16/fp16 operands would need more than this many MB of fp32 copies (a 150k-row `lm_head`, huge logits gradients) is split into pieces; `0` = never split |
| `RKNPU_CPU_UPCAST` | `1` | compute small bf16/fp16 matmuls, linears and attention that stay on the CPU in fp32 (the A76 has no bf16 arithmetic); `0` = leave them to PyTorch |
| `RKNPU_DUTY` | `1.0` | share of wall time the NPU path may be busy (0.05 to 1); below 1 every NPU call is followed by a proportional pause, for supplies that brown out under sustained load. Also `rk3588_torch.set_duty()` |
| `RKNPU_WCACHE` | `1` | `0` turns off the packed-weight cache |
| `RKNPU_WCACHE_MB` | `256` | NPU memory for cached weights (least recently used are evicted) |
| `RKNPU_LOCK` | `1` | serialize NPU use across processes that use this library (`/tmp/rk3588_torch.lock`) |
| `RKNPU_TIMEOUT_MS` | `3000` | per-job timeout before the driver resets the NPU |
| `RKNPU_DEV` | auto | NPU device node (default: the DRM card named `rknpu`) |

In Python: `set_precision()`, `set_enabled()`, `release()` (free NPU buffers), `stats()`,
`reset_verification()`, `calibrate()`.

## Compatibility

| | Tested | Expected to work | Not supported |
|---|---|---|---|
| Board | Radxa Rock 5B (RK3588, 4 GB) | other RK3588 / RK3588S boards (Orange Pi 5, Rock 5A/5C, ...) | RK3566/3568/3576 (different NPUs) |
| Kernel driver | vendor `rknpu` 0.9.8 (Armbian 6.12, patched build) | other 0.9.x vendor builds (`doctor` reports the version) | mainline `rocket` driver (different uAPI) |
| OS | Armbian 25.11 (Ubuntu 24.04) | any aarch64 Linux with the driver | |
| PyTorch | 2.15 CPU wheel | 2.4 or newer | |
| Ops on the NPU | `linear`, `matmul`/`mm`/`bmm`/`addmm`/`baddbmm`, two-operand `einsum`, `conv1d`/`conv2d`/`conv_transpose2d` (incl. grouped and dilated; zero padding incl. `"same"`), `scaled_dot_product_attention` | | depthwise conv (library NEON kernels), `conv3d`, LSTM/GRU gate math (run on the CPU) |
| Tensors | CPU `float32`, `float16`, `bfloat16` | | CUDA / other devices, integer tensors (left to PyTorch) |

On a non-RK3588 machine the package still installs and everything runs on the CPU, which
is handy for developing on a PC.

## Troubleshooting

`python -m rk3588_torch doctor` checks the setup-related ones and prints the fix. The common ones:

| Symptom | Fix |
|---|---|
| `could not open the rknpu DRM device` | load the driver: `sudo modprobe rknpu`, and check `dmesg \| grep -i rknpu` |
| permission denied on `/dev/dri/card*` | `sudo usermod -aG video,render $USER`, then log out and back in |
| `librknpu_gemm.so not found` | `pip install .` from the repo (or `make -C csrc` and run from the repo) |
| board resets or freezes under load | use a 30 W+ USB-C PD supply, and `torch.set_num_threads(3)`. On a weak supply, also `RKNPU_DUTY=0.6`, 2 threads, and optionally cap the NPU clock (see [Fine-tuning large models](#fine-tuning-large-models)) |
| `CheckpointError: Recomputed values ... different metadata` | gradient checkpointing re-ran a layer without the NPU routing. Use `rk3588_torch.checkpoint(...)` instead of `torch.utils.checkpoint.checkpoint`, or call `rk3588_torch.enable()` once; see [Fine-tuning large models](#fine-tuning-large-models) |
| warning: "weight cache ... capped at ... MB" | the frozen weights you reuse don't fit the cache, so each step re-packs them. Raise `RKNPU_WCACHE_MB` if memory allows (`rk3588_torch.wcache.info()` shows the numbers) |
| warning: "NPU failed ... running on the CPU" | an NPU job timed out or errored; that op ran on the CPU. After `RKNPU_MAX_FAILURES` the NPU is disabled for the process. Check `dmesg` for `RKNPU` lines |
| warning: "didn't match the CPU check" | that shape gave a wrong NPU result and is pinned to the CPU. Please open an issue with the shape |
| bf16 training is slow and `profile()` shows `aten::addmm` / `_scaled_dot_product_flash_attention` taking seconds | bf16 on the A76 CPU is a software path. Keep `RKNPU_CPU_UPCAST=1` (default), and make sure attention goes through `F.scaled_dot_product_attention` inside `npu_mode()` |
| a power loss leaves files of zero size | ext4 can lose recently written files on a reset. Run `sync` after copying code to the board |
| training is no faster | run `compare()` / `profile()`: small layers stay on the CPU by design, and CPU-side ops (norms, activations, pooling) may dominate |
| loss is NaN only on the NPU | try `precision="bf16"` (fp32 range) or `fp16x3` (fp32 accuracy) |
| another NPU program (e.g. an LLM server) misbehaves | a hung job makes the driver reset every NPU core; run experiments when it's idle |

## FAQ

**Will it speed up my model?** Run `rk3588_torch.compare(model, x, y)`. Rules of thumb: wide
`Linear` layers (≥ 512) with batches ≥ 64 and 3×3 convolutions with ≥ 32 channels gain the
most (2-3.5× per training step in the benchmarks above); Transformers gain ~1.4-1.9×. Tiny
models don't: below the calibrated size everything stays on the CPU.

**Which precision?** `bf16` is the fastest float mode and has fp32's range; `fp16` (default)
is ~8× more precise per value; `fp16x3` matches fp32 for numerically sensitive work;
`int8` is experimental. `compare()` picks one for you.

**Does it change my results?** Within fp16/bf16 rounding (relative 3e-4 / 2e-3 per op).
Training reaches the same accuracy as CPU training in every test here; individual
steps differ slightly, like mixed-precision training on a GPU.

**Does it work with `torch.compile`?** Yes, through `convert(model)`: the NPU ops are
registered as `torch.library` custom ops with fake (meta) implementations.

**Can I use it for inference?** Yes, it's just faster matmuls and convolutions; use
`torch.no_grad()` as usual. The weight cache keeps each layer's converted weights on the
NPU after the second call, so repeated inference skips the weight conversion.

**Does it help LoRA / fine-tuning with frozen layers?** Yes: frozen weights hit the weight
cache, so each step only converts activations and the small trainable adapters. Ordinary
training doesn't use the cache (weights change every step), and doesn't pay for it. For maximum inference speed Rockchip's RKNN toolkit, which
compiles the whole graph, will usually be faster.

**What if the NPU gets something wrong?** The first time a shape is seen, part of its
result is compared with the CPU; mismatches, NaNs and errors are recomputed on the CPU
and that shape stays on the CPU. The check samples a few rows of the first call per
shape; `RKNPU_VERIFY=always` checks every call if you want to be certain.

## Limitations

* The default `fp16` mode rounds inputs to 11 significant bits (auto-scaling handles
  range, not precision). Use `fp16x3` where fp32 accuracy matters, and treat `int8`
  as experimental for training.
* Only GEMM-shaped work runs on the NPU; elementwise ops, norms, softmax, pooling and
  the optimizer stay on the CPU, so small models gain less.
* Convolutions: zero padding only (`padding_mode="zeros"`). The native NPU convolution takes
  feature maps from 5 to 2047 rows/columns; anything else with `groups=1` (smaller maps, longer
  sequences, padding over 15, dilation with an odd stride) runs as unfold + one NPU GEMM if the
  unfolded input fits `RKNPU_IM2COL_MAX_MB`. Grouped convolutions run on the NPU only
  when each group has enough channels (the padding to 32 input / 16 output channels
  wastes less than 8×); `conv3d` runs on the CPU, and depthwise convolutions (dilation 1) run in the
  library's NEON CPU kernels (`RKNPU_DWCONV=0`: PyTorch's). 7×7 stems on
  3-channel images gain little (the NPU works on 32-channel blocks).
* `nn.LSTM` / `nn.GRU` use the NPU only after `convert()` (under `npu_mode()` alone they are PyTorch's fused
  CPU kernels). `convert()` puts the input projections of all timesteps in one NPU GEMM per layer and
  direction, the W_hh weight gradient in another, and the per-step `h @ W_hh` on the NPU when a step has
  ≥ 30M multiply-adds (`RKNPU_RNN_REC_NPU=auto|1|0`, `RKNPU_RNN_REC_MACS`); the gate math stays on the CPU.
  `PackedSequence` inputs, `proj_size` and unbatched inputs fall back to PyTorch.
* `einsum`: two operands, no `...` and no repeated indices within an operand; other forms
  are left to PyTorch.
* Each NPU call costs ~0.3 ms of CPU-side packing and submission, so many small ops
  (e.g. attention with small heads) stay on the CPU.
* A job that hangs makes the driver soft-reset all NPU cores, which would also abort
  another process's NPU job (e.g. an LLM server) that is running at that moment.

## Credits

* Register map, kernel layout and the original matmul-as-convolution work: Jasbir
  Matharu, [mtx512/rk3588-npu](https://github.com/mtx512/rk3588-npu) (GPL-3.0).
* Background: [SHARD: optimizing ViTs on the RK3588 NPU](https://amohan.dev/blog/2025/shard-optimizing-vision-transformers-edge-npu/).

License: GPL-3.0 (see `LICENSE`).
