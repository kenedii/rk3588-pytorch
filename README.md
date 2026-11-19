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
| 🧱 **Linear, conv, attention** | `Linear`, native NPU `Conv1d`/`Conv2d`/`ConvTranspose2d` (incl. grouped), `einsum`, attention, `bmm`, `torch.compile` |
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
| MobileNetV2 block, depthwise on the CPU (train step) | 704 ms | 497 ms | 1.4× |
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
  fp16 precision. Disable with `RKNPU_AUTOSCALE=0` to save the extra scan (~15%).
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
  convolutions stay on the CPU. Grouped convolutions run as one task chain with each
  group's weights and input channels in their own blocks.
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
| `tests/test_robust.py` | Fallbacks, verification, batching, attention, `convert()`, `torch.compile`, tools |
| `bench/bench_gemm.py` | CPU vs NPU GEMM throughput |
| `bench/bench_precision.py` | Speed and error of every precision mode |
| `bench/bench_models.py` | MLP / CNN / ResNet / Transformer training steps, CPU vs NPU |
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
| `RKNPU_CHECK_FINITE` | `1` | send ops whose inputs contain inf/nan to the CPU (the NPU turns them into finite values, which would hide a diverging run); `0` saves one reduction per input (~5-10% on large GEMMs) |
| `RKNPU_MAX_FAILURES` | `3` | NPU errors before it is disabled for the rest of the process |
| `RKNPU_DISABLE` | `0` | `1` = run everything on the CPU (same code path, for A/B tests) |
| `RKNPU_AUTOSCALE` | `1` | power-of-two scaling in `fp16` mode (`0` is ~15% faster, but values above 65504 overflow) |
| `RKNPU_THREADS` | `3` | CPU threads used for packing data for the NPU (pinned to the big cores) |
| `RKNPU_MODE` | `2` | `2` = all three NPU cores, `1` = one core, `0` = one task per submit (debugging) |
| `RKNPU_MAX_BUF_MB` | `512` | cap on NPU buffer memory; bigger ops are split |
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
| Ops on the NPU | `linear`, `matmul`/`mm`/`bmm`/`addmm`/`baddbmm`, two-operand `einsum`, `conv1d`/`conv2d`/`conv_transpose2d` (incl. grouped; dilation 1, zero padding), `scaled_dot_product_attention` | | depthwise / dilated conv, `conv3d`, fused RNNs (all run on the CPU) |
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
| board resets or freezes under load | use a 30 W+ USB-C PD supply, and `torch.set_num_threads(3)` |
| warning: "NPU failed ... running on the CPU" | an NPU job timed out or errored; that op ran on the CPU. After `RKNPU_MAX_FAILURES` the NPU is disabled for the process. Check `dmesg` for `RKNPU` lines |
| warning: "didn't match the CPU check" | that shape gave a wrong NPU result and is pinned to the CPU. Please open an issue with the shape |
| training is no faster | run `compare()` / `profile()`: small layers stay on the CPU by design, and CPU-side ops (norms, activations, pooling) may dominate |
| loss is NaN only on the NPU | try `precision="bf16"` (fp32 range) or `fp16x3` (fp32 accuracy) |
| another NPU program (e.g. an LLM server) misbehaves | a hung job makes the driver reset every NPU core; run experiments when it's idle |

## FAQ

**Will it speed up my model?** Run `rk3588_torch.compare(model, x, y)`. Rules of thumb: wide
`Linear` layers (≥ 512) with batches ≥ 64 and 3×3 convolutions with ≥ 32 channels gain the
most (2-3.5× per training step in the benchmarks above); Transformers gain ~1.6×. Tiny
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
* Convolutions: no dilation, zero padding only, images at least 5 rows tall and at most
  2047 wide/tall (conv1d sequences up to 2047). Grouped convolutions run on the NPU only
  when each group has enough channels (the padding to 32 input / 16 output channels
  wastes less than 8×); depthwise convolutions and `conv3d` run on the CPU. 7×7 stems on
  3-channel images gain little (the NPU works on 32-channel blocks).
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
