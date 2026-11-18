<div align="center">

# rk3588-pytorch

### Train PyTorch models on the Rockchip RK3588 NPU

**Up to 9× faster matrix multiplies and 2.9× faster training steps than the CPU,
on a single-board computer, with one line of code.**

![platform](https://img.shields.io/badge/platform-RK3588-blue)
![pytorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c)
![npu](https://img.shields.io/badge/NPU-731%20GFLOPS-brightgreen)
![tests](https://img.shields.io/badge/tests-35%2F35%20passing-success)
![license](https://img.shields.io/badge/license-GPL--3.0-lightgrey)

</div>

---

The RK3588's NPU (rated at 6 TOPS) is sold as an inference-only accelerator: you convert a finished
model with Rockchip's toolkit and run it. **rk3588-pytorch turns it into a training
device.** Every matrix multiply in the forward *and* backward pass runs on the NPU, while
PyTorch keeps doing everything else on the CPU. Your training loop doesn't change.

```python
import rk3588_torch

with rk3588_torch.npu_mode():          # matmuls -> NPU, everything else -> CPU
    loss = F.cross_entropy(model(x), y)
    loss.backward()                    # gradient matmuls -> NPU too
optimizer.step()
```

## Highlights

| | |
|---|---|
| ⚡ **731 GFLOPS** | 2048³ matmul end to end, **9.1×** the CPU. The NPU alone hits ~1.2 TFLOPS |
| 🏋️ **2.9× faster training** | wide MLP on MNIST: 75 ms (bf16) vs 218 ms per step, same accuracy |
| 🎯 **Matches CPU training** | fp32 accumulation; same accuracy, mean loss difference 0.0006 on the wide MLP |
| 🔌 **Drop-in** | no model conversion, no RKNN runtime, no changes to your model code |
| 🎚️ **Five precisions** | fp16, bf16, int8, fp16-output, and an fp32-accurate split mode (~1e-7 error) |
| 🧠 **All three NPU cores** | work is split across cores automatically |
| 🛟 **Safe fallback** | ops the NPU can't do, and matmuls too small to benefit, run on the CPU |
| 🔬 **Built from scratch** | writes the NPU's register command streams directly; reverse-engineering tools included |

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

**Training on MNIST**, same initial weights and batches:

| Model | CPU step | NPU step | Speedup | Test accuracy (CPU / NPU) |
|---|---:|---:|---:|---|
| MLP 784-2048-2048-1024-10, batch 256 | 219.6 ms | 77.5 ms | **2.8×** | 94.18% / 94.21% |
| MLP 784-512-256-10, batch 128 | 9.9 ms | 8.5 ms | 1.16× | 95.89% / 95.90% |
| Small CNN, batch 64 (linear layers on NPU) | 71.0 ms | 72.2 ms | 0.98× | 96.73% / 96.22% |

(fp16 without auto-scaling; per-mode training numbers are under [Precisions](#precisions).)
The bigger the layers and batches, the bigger the win. Small layers are dominated by
fixed per-call overhead, and convolutions, norms and activations stay on the CPU (see
[Limitations](#limitations)).

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
CPU-only and NPU processes; the mean loss difference between the two runs is 0.0006
(wide MLP), 0.004 (small MLP) and 0.019 (CNN).

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
PyTorch op (F.linear, matmul, mm, bmm, addmm, conv2d)
   │  rk3588_torch.npu_mode() is a TorchFunctionMode that intercepts these ops
   ▼
autograd.Function: forward GEMM + both backward GEMMs (dX = dY·W, dW = dYᵀ·X)
   │  operands passed as (pointer, strides): transposes cost nothing
   │  small GEMMs (< 8M MACs) stay on the CPU; conv2d optional (im2col + GEMM)
   ▼
librknpu_gemm.so
   │  A: fp32 → fp16 row-major      B: fp32 → fp16 16-kernel groups
   │  tile M so each task's rows fit in CBUF; build 108-register task streams,
   │  chain them through the NPU program counter, split across the 3 cores
   ▼
RK3588 NPU: 1×1 convolution, CNA (MACs) → CORE → DPU (writes row-major fp32 C)
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
* **Other precisions.** Capturing librknnrt's int8 and fp16-output streams gave their
  register constants (precision codes, output converter, surface add, notch). Probing
  the unused precision code 3 with a layout-independent test (B = all ones, so every
  output is a row sum) revealed a working bf16 MAC path.
* **`memset(0)` on uncached NPU memory faults.** glibc zeroes with `DC ZVA`, which is
  not allowed on Device memory (this is why the original mtx512 tests die with SIGBUS);
  the library uses cacheable buffers with explicit syncs instead.

## Layout

| Path | What |
|---|---|
| `csrc/rknpu_gemm.c`, `rknpu_gemm.h` | NPU GEMM library: tiling, packing, register streams, submit |
| `csrc/npu_regs.h` | Register offsets and command-word encoding |
| `csrc/rknpu_uapi.h` | ioctl structs matching the rknpu 0.9.8 driver |
| `rk3588_torch/` | ctypes binding, autograd functions, `npu_mode()` op router |
| `tests/test_ops.py` | GEMM / gradient correctness vs fp64 CPU references |
| `tests/test_precision.py` | Accuracy and range of every precision mode |
| `bench/bench_gemm.py` | CPU vs NPU GEMM throughput |
| `bench/bench_precision.py` | Speed and error of every precision mode |
| `examples/train_mnist.py` | MLP / CNN training on NPU vs CPU |
| `tools/gemm_test.c` | C-level correctness + timing |
| `tools/ioctl_spy.c` | `LD_PRELOAD` shim logging RKNPU ioctls and dumping command streams |
| `tools/rknn_mm_test.c` | Reference matmul through Rockchip's `rknn_matmul` API |
| `tools/prec_probe.c` | Probe NPU precision codes with a layout-independent test |
| `tools/rknn_mm_probe.c` | Run any `rknn_matmul` type (for capturing its command stream) |
| `tools/probe_mem.c` | Which allocation flags can be mapped and touched |

## Setup

Requirements: RK3588 board with the vendor `rknpu` driver loaded (`/dev/dri/cardN`
named `rknpu`; tested with 0.9.8 on Armbian 6.12), user in `video` and `render`.

```sh
sudo modprobe rknpu                     # if it isn't loaded at boot
make -C csrc                            # librknpu_gemm.so + gemm_test
python3 -m venv venv && venv/bin/pip install torch numpy --index-url https://download.pytorch.org/whl/cpu
venv/bin/python tests/test_ops.py
venv/bin/python bench/bench_gemm.py
venv/bin/python examples/train_mnist.py --model mlp
```

Knobs (environment): `RKNPU_MODE` (0 = one submit per task, 1 = one core chained,
2 = three cores, default), `RKNPU_THREADS` (packing threads, default 3),
`RKNPU_MIN_MACS` (CPU below this, default 8e6), `RKNPU_MAX_M`, `RKNPU_MAX_K`,
`RKNPU_MAX_N`, `RKNPU_TIMEOUT_MS`, `RKNPU_NO_PIN`, `RKNPU_PRECISION` (default `fp16`),
`RKNPU_AUTOSCALE` (default 1).

## Limitations

* The default `fp16` mode rounds inputs to 11 significant bits (auto-scaling handles
  range, not precision). Use `fp16x3` where fp32 accuracy matters, and treat `int8`
  as experimental for training.
* Only GEMM-shaped work runs on the NPU; elementwise ops, norms, softmax, pooling and
  the optimizer stay on the CPU, so small models gain less.
* `conv2d` is only routed with `npu_mode(conv=True)` and goes through `unfold` (im2col):
  that costs memory and, for small CNNs, is slower than PyTorch's native CPU conv.
  A native NPU convolution path (the hardware is a convolution engine) is the obvious
  next step.
* A job that hangs makes the driver soft-reset all NPU cores, which would also abort
  another process's NPU job (e.g. an LLM server) that is running at that moment.

## Credits

* Register map, kernel layout and the original matmul-as-convolution work: Jasbir
  Matharu, [mtx512/rk3588-npu](https://github.com/mtx512/rk3588-npu) (GPL-3.0).
* Background: [SHARD: optimizing ViTs on the RK3588 NPU](https://amohan.dev/blog/2025/shard-optimizing-vision-transformers-edge-npu/).

License: GPL-3.0 (see `LICENSE`).
