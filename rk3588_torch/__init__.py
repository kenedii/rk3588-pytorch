"""rk3588_torch: use the Rockchip RK3588 NPU as a matmul accelerator for PyTorch training.

GEMMs (forward and both backward products) run on the NPU in fp16 with fp32 output;
everything else stays on the CPU.

    import rk3588_torch
    with rk3588_torch.npu_mode():
        loss = model(x).sum()
        loss.backward()
"""
from ._lib import NPU, default_npu
from .mode import NPUMode, npu_mode, enable, disable
from .ops import linear, matmul, bmm, conv2d, mm_raw, stats

__all__ = ["NPU", "default_npu", "NPUMode", "npu_mode", "enable", "disable",
           "linear", "matmul", "bmm", "conv2d", "mm_raw", "stats"]
