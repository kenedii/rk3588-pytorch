# rk3588-pytorch

Train PyTorch models with the **RK3588 NPU** (Radxa Rock 5B, Orange Pi 5, ...) doing
the matrix multiplies, and the CPU doing everything else.

```python
import rk3588_torch

with rk3588_torch.npu_mode():          # linear / matmul / mm / bmm / addmm -> NPU
    loss = F.cross_entropy(model(x), y)
    loss.backward()                    # dX and dW GEMMs -> NPU too
optimizer.step()                       # CPU
```

No RKNN model conversion and no RKNN runtime: the library writes NPU register command
streams itself and submits them through the vendor `rknpu` DRM driver.

## Results (Rock 5B, 4 GB, CPU capped at 1.8 GHz, NPU at 800 MHz)

**GEMM throughput**: fp32 in, fp32 out, end to end including all CPU-side conversion
(`bench/bench_gemm.py`). CPU is `torch.matmul` fp32 with 3 threads (more busy cores
reset this board on its charger), measured before the NPU library starts any threads.

| M × K × N | CPU fp32 | NPU, 1 core | NPU, 3 cores | speedup |
|---|---:|---:|---:|---:|
| 64 × 64 × 64 | 12 GFLOPS | 5 | 6 | 0.5× (CPU fallback) |
| 512 × 512 × 512 | 71 | 191 | 235 | 3.3× |
| 256 × 784 × 512 | 72 | 147 | 152 | 2.1× |
| 1024 × 1024 × 1024 | 79 | 343 | 469 | 5.9× |
| 2048 × 512 × 2048 | 78 | 276 | 452 | 5.8× |
| 512 × 4096 × 512 | 76 | 141 | 264 | 3.5× |
| 2048 × 2048 × 2048 | 80 | 434 | **731** | **9.1×** |

The NPU alone runs a 2048³ GEMM in 14.4 ms (~1.2 TFLOPS); the rest of the 23.5 ms is
fp32→fp16 packing (7.4 ms) and copying C out (1.8 ms) on the CPU.

**Training (MNIST, same initial weights and batches; step times from separate CPU-only and NPU processes)**

| model | batch | CPU step | NPU step | speedup | test acc CPU / NPU | mean \|Δloss\| |
|---|---:|---:|---:|---:|---|---:|
| MLP 784-2048-2048-1024-10 | 256 | 219.6 ms | 77.5 ms | **2.8×** | 94.18% / 94.21% | 0.0006 |
| MLP 784-512-256-10 | 128 | 9.9 ms | 8.5 ms | 1.16× | 95.89% / 95.90% | 0.004 |
| CNN 2×conv + 2×fc (fc on NPU) | 64 | 71.0 ms | 72.2 ms | 0.98× | 96.73% / 96.22% | 0.019 |

Training speedup tracks GEMM size. Small layers are dominated by per-call overhead
(packing, cache maintenance, ~0.1-0.3 ms submit latency) and by the ops that stay on the
CPU, so the bigger the layers and batches, the more the NPU helps. In the CNN almost all
the work is in convolutions and pooling, which run on the CPU by default, so routing only
its linear layers to the NPU just breaks even.

Numerics: inputs are rounded to fp16, products are accumulated in fp32 and returned as
fp32, so results match fp64 references to ~3e-4 relative (the fp16 input rounding) and
gradients match CPU autograd (`tests/test_ops.py`, 35 checks).

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
| `bench/bench_gemm.py` | CPU vs NPU GEMM throughput |
| `examples/train_mnist.py` | MLP / CNN training on NPU vs CPU |
| `tools/gemm_test.c` | C-level correctness + timing |
| `tools/ioctl_spy.c` | `LD_PRELOAD` shim logging RKNPU ioctls and dumping command streams |
| `tools/rknn_mm_test.c` | Reference matmul through Rockchip's `rknn_matmul` API |
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
`RKNPU_MAX_N`, `RKNPU_TIMEOUT_MS`, `RKNPU_NO_PIN`.

## Limitations

* fp16 inputs: values beyond ±65504 overflow and tiny gradients lose precision. Use
  loss scaling for models with very small gradients.
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
