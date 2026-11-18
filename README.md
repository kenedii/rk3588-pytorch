# rk3588-pytorch

Use the NPU in the Rockchip RK3588 (Radxa Rock 5B, Orange Pi 5, ...) as a matrix-multiply
accelerator for **PyTorch training**, with everything the NPU can't do left on the CPU.

> **Status: work in progress.** The register-level GEMM backend submits jobs to the
> NPU, but results are not reliable yet on the test board (see *Current state*). Do not
> use this for real training yet.

## How it works

Training a network is mostly GEMMs: for each linear layer `y = x Wᵀ`, the backward
pass needs `dX = dY W` and `dW = dYᵀ X`. All three are routed to the NPU:

```
PyTorch op (F.linear, matmul, mm, bmm, addmm, conv2d)
   │  TorchFunctionMode intercepts it (rk3588_torch.npu_mode)
   ▼
autograd.Function (NPULinearFn / NPUMatmul): forward + both backward GEMMs
   │  operands passed as (pointer, strides), so transposes are free
   ▼
librknnpu_gemm.so (C): fp32 → fp16 pack into NPU layouts, tile, build register
   │  command streams, submit through the rknpu DRM driver, unpack fp32 results
   ▼
RK3588 NPU: matmul expressed as a 1x1 convolution (CNA → CORE → DPU)
```

* **Matmul as convolution.** `A (M×K)` becomes a feature map of height M and K channels
  (NC1HWC2 layout, C2 = 8 for fp16). `B (K×N)` becomes N kernels (grouped 16 kernels ×
  32 channels). The output is fp32, in NC1HWC2 layout with C2 = 4.
* **Tiling.** The convolution buffer (CBUF) holds 12 × 32 KB banks per core. A is split
  into row tiles that fit in the feature-data banks, and K is split into chunks whose
  partial sums are reduced on the CPU in fp32.
* **No RKNN runtime.** Buffers, register commands and task descriptors are built in
  userspace and submitted with the vendor driver's `RKNPU_SUBMIT` ioctl in PC mode.
* **Cacheable buffers.** Packing writes to cacheable memory, followed by explicit
  `MEM_SYNC` calls. Uncached (Device-type) mappings are slow, and `memset(0)` on them
  faults on arm64 because of `DC ZVA`.
* **Everything else stays on the CPU.** Activations, norms, softmax, losses and
  optimizers run as normal PyTorch CPU ops.

## Layout

| Path | What |
|---|---|
| `csrc/rknpu_gemm.c`, `rknpu_gemm.h` | NPU GEMM library (tiling, packing, register streams, submit) |
| `csrc/npu_regs.h` | Register offsets and command-word encoding |
| `csrc/rknpu_uapi.h` | Minimal ioctl structs matching the rknpu 0.9.8 driver |
| `rk3588_torch/` | ctypes binding, autograd functions, `npu_mode()` op router |
| `tools/gemm_test.c` | Correctness and timing check against a CPU reference |
| `tools/probe_mem.c` | Checks which allocation flags can be mapped and touched |
| `tools/ioctl_spy.c` | `LD_PRELOAD` shim that logs RKNPU ioctls and dumps command streams (e.g. from librknnrt) |
| `tools/rknn_mm_test.c` | Reference fp16 matmul through Rockchip's `rknn_matmul` API |

## Build

On the board (needs the vendor `rknpu` driver loaded and the user in the `video`/`render` groups):

```sh
make -C csrc            # builds librknpu_gemm.so and gemm_test
./csrc/gemm_test 256 512 256 3
```

```python
import torch, rk3588_torch
model = torch.nn.Sequential(torch.nn.Linear(784, 512), torch.nn.ReLU(), torch.nn.Linear(512, 10))
with rk3588_torch.npu_mode():
    loss = torch.nn.functional.cross_entropy(model(x), y)
    loss.backward()
```

Environment knobs: `RKNPU_MODE` (0 = one submit per task, 1 = chained tasks, 2 = chained
across 3 cores), `RKNPU_THREADS` (CPU packing threads, default 3), `RKNPU_MAX_M`,
`RKNPU_MAX_K`, `RKNPU_CACHED`, `RKNPU_TIMEOUT_MS`, `RKNPU_MIN_MACS` (below this size
the CPU is used).

## Current state (test board: Rock 5B, Armbian 6.12, rknpu 0.9.8 with patches)

* The NPU device is found automatically (DRM node named `rknpu`), and buffers allocate and map.
* This driver keeps **per-core IOMMU mappings**. Buffers must be created with `core_mask`
  set, or the NPU's program counter can't fetch the command stream and the job times out.
  The original mtx512 tests hit this too.
* Jobs are now accepted and some complete with partially correct output, but most still
  time out (`int_raw_status 0xc0010000`). The next step is diffing our command stream
  against the one librknnrt generates for the same shape, captured with `tools/ioctl_spy.c`.
* A timed-out job makes the driver soft-reset **all** NPU cores. If another NPU
  workload is running, only experiment while it is idle.

## Credits

* Register sequence and data layouts: Jasbir Matharu's reverse engineering,
  [mtx512/rk3588-npu](https://github.com/mtx512/rk3588-npu) (GPL-3.0).
* Background: [SHARD: optimizing ViTs on the RK3588 NPU](https://amohan.dev/blog/2025/shard-optimizing-vision-transformers-edge-npu/).

License: GPL-3.0 (see `LICENSE`).
