"""rk3588_torch: use the Rockchip RK3588 NPU as a matmul accelerator for PyTorch training.

GEMMs (forward and both backward products) run on the NPU; everything else stays on the
CPU. Anything the NPU can't do, or does too slowly, falls back to the CPU automatically.

    import rk3588_torch
    with rk3588_torch.npu_mode():
        loss = model(x).sum()
        loss.backward()

    rk3588_torch.compare(model, x, y)       # is the NPU worth it for this model? which precision?
    with rk3588_torch.profile(): ...        # where does the time go?
    python -m rk3588_torch doctor           # check the setup
"""
__version__ = "0.3.0"

from ._lib import NPU, PRECISIONS, default_npu, npu_available, npu_error
from .mode import NPULinear, NPUMode, convert, disable, enable, npu_mode
from .ops import (bmm, conv2d, get_precision, gemm, is_enabled, linear, matmul, mm_raw, release,
                  reset_verification, scaled_dot_product_attention, set_enabled, set_precision, stats,
                  threshold)
from .tools import compare, profile
from .calibrate import calibrate

__all__ = ["NPU", "PRECISIONS", "default_npu", "npu_available", "npu_error",
           "NPULinear", "NPUMode", "convert", "disable", "enable", "npu_mode",
           "bmm", "conv2d", "get_precision", "gemm", "is_enabled", "linear", "matmul", "mm_raw", "release",
           "reset_verification", "scaled_dot_product_attention", "set_enabled", "set_precision", "stats",
           "threshold", "compare", "profile", "calibrate", "__version__"]
