"""`python -m rk3588_torch doctor`: check that everything needed is in place, with fixes."""
import glob
import grp
import os
import platform
import sys
import time

TESTED_DRIVERS = {908}

_results = []


def _line(status, what, detail="", fix=""):
    _results.append(status)
    mark = {"ok": "\033[32m OK \033[0m", "warn": "\033[33mWARN\033[0m", "fail": "\033[31mFAIL\033[0m",
            "info": "\033[36minfo\033[0m"}[status]
    if not sys.stdout.isatty():
        mark = {"ok": " OK ", "warn": "WARN", "fail": "FAIL", "info": "info"}[status]
    print(f"[{mark}] {what}" + (f": {detail}" if detail else ""))
    if fix:
        print(f"       fix: {fix}")


def _read(path):
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return None


def _npu_nodes():
    nodes = []
    for card in sorted(glob.glob("/sys/class/drm/card[0-9]*")):
        if "-" in os.path.basename(card):
            continue
        dev = os.path.realpath(os.path.join(card, "device"))
        if "npu" in dev.lower():
            nodes.append("/dev/dri/" + os.path.basename(card))
    return nodes


def _npu_users(node):
    users = []
    real = os.path.realpath(node)
    for pid in os.listdir("/proc"):
        if not pid.isdigit() or int(pid) == os.getpid():
            continue
        try:
            for fd in os.listdir(f"/proc/{pid}/fd"):
                if os.path.realpath(f"/proc/{pid}/fd/{fd}") == real:
                    comm = _read(f"/proc/{pid}/comm") or "?"
                    users.append(f"{comm}[{pid}]")
                    break
        except OSError:
            continue
    return users


def run():
    _results.clear()
    print("rk3588_torch doctor\n")
    # --- platform ---
    mach = platform.machine()
    compat = (_read("/proc/device-tree/compatible") or "").replace("\x00", " ")
    if mach not in ("aarch64", "arm64"):
        _line("fail", "CPU architecture", mach, "this library needs an RK3588 (aarch64) board; "
              "on other machines everything runs on the CPU")
    elif "rk3588" not in compat:
        _line("warn", "SoC", f"device tree says '{compat[:60]}'", "only the RK3588/RK3588S NPU is supported")
    else:
        _line("ok", "SoC", "RK3588")
    _line("info", "kernel", platform.release())

    # --- driver ---
    if os.path.isdir("/sys/module/rknpu"):
        ver = _read("/sys/module/rknpu/version")
        _line("ok", "rknpu kernel module loaded", ver or "")
    else:
        rocket = os.path.isdir("/sys/module/rocket")
        _line("fail", "rknpu kernel module", "not loaded" + (" (the mainline 'rocket' driver is loaded instead, "
              "which this library does not support yet)" if rocket else ""),
              "sudo modprobe rknpu   (needs Rockchip's vendor rknpu driver for your kernel)")
    nodes = _npu_nodes()
    if not nodes:
        _line("fail", "NPU device node", "no /dev/dri/card* belongs to the NPU",
              "load the rknpu driver; on mainline kernels the NPU also needs its device-tree overlay")
    else:
        node = nodes[0]
        if os.access(node, os.R_OK | os.W_OK):
            _line("ok", "NPU device node", f"{node} is readable/writable")
        else:
            try:
                gname = grp.getgrgid(os.stat(node).st_gid).gr_name
            except (KeyError, OSError):
                gname = "video"
            _line("fail", "NPU device node", f"{node} is not accessible by this user",
                  f"sudo usermod -aG {gname},render $USER   then log out and back in")
        users = _npu_users(node)
        if users:
            _line("warn", "other NPU users", ", ".join(users[:5]),
                  "fine to share, but if an NPU job ever hangs the driver resets all cores and their "
                  "in-flight jobs fail too")

    # --- library ---
    from ._lib import library_path, npu_available, npu_error, default_npu
    lp = library_path()
    if lp:
        _line("ok", "librknpu_gemm.so", lp)
    else:
        _line("fail", "librknpu_gemm.so", "not found", "pip install .   (or: make -C csrc)")

    try:
        import torch
        _line("ok", "PyTorch", f"{torch.__version__}, {torch.get_num_threads()} intra-op threads")
        if torch.get_num_threads() > 4:
            _line("warn", "PyTorch threads", f"{torch.get_num_threads()}",
                  "torch.set_num_threads(3) leaves room for the NPU packing threads; boards on weak "
                  "power supplies can also reset under all-core load")
    except ImportError:
        _line("fail", "PyTorch", "not installed", "pip install torch")
        return _finish()

    if not npu_available():
        _line("fail", "open the NPU", str(npu_error()))
        return _finish()
    npu = default_npu()
    info = npu.info()
    dv = info["driver_version"]
    dvs = f"{dv // 10000}.{dv // 100 % 100}.{dv % 100}" if dv else "unknown"
    _line("ok" if dv in TESTED_DRIVERS else "warn", "driver version", dvs,
          "" if dv in TESTED_DRIVERS else "untested driver version; the self-test below decides")
    _line("info", "NPU", f"hw {info['hw_version']:#x}, IOMMU {'on' if info['iommu'] else 'off'}, "
          f"{info['freq_hz'] / 1e6:.0f} MHz")
    gov = _read("/sys/class/devfreq/fdab0000.npu/governor")
    if gov and gov != "performance":
        _line("info", "NPU frequency governor", gov,
              "for steady benchmarks: echo performance | sudo tee /sys/class/devfreq/fdab0000.npu/governor")

    # --- power ---
    npu_cap = _read("/sys/class/devfreq/fdab0000.npu/max_freq")
    big = _read("/sys/devices/system/cpu/cpu4/cpufreq/scaling_max_freq")
    from ._lib import get_duty
    parts = []
    if npu_cap:
        parts.append(f"NPU max {int(npu_cap) // 10**6} MHz")
    if big:
        parts.append(f"big cores max {int(big) // 1000} MHz")
    parts += [f"RKNPU_DUTY={get_duty():g}", f"{torch.get_num_threads()} torch threads"]
    _line("info", "power limits", ", ".join(parts))
    _line("info", "weak charger?", "if the board resets or reboots under sustained training, it is browning out",
          "lower the load: RKNPU_DUTY=0.6 (or rk3588_torch.set_duty(0.6)), torch.set_num_threads(2) with "
          "RKNPU_THREADS=2, and optionally cap the clocks, e.g. "
          "echo 600000000 | sudo tee /sys/class/devfreq/fdab0000.npu/max_freq; a 30 W+ USB-C PD supply fixes it")

    # --- self test ---
    from . import ops
    from ._lib import PRECISIONS
    a, b = torch.randn(128, 256), torch.randn(256, 96)
    ref = a.double() @ b.double()
    for p in PRECISIONS:
        try:
            t0 = time.perf_counter()
            out = ops._npu_gemm(a, b, p)
            dt = (time.perf_counter() - t0) * 1e3
            err = ((out.double() - ref).norm() / ref.norm()).item()
            ok = err <= ops.VERIFY_TOL[p]
            _line("ok" if ok else "fail", f"self-test {p:<10}", f"rel err {err:.1e}, {dt:.1f} ms",
                  "" if ok else "this precision gives wrong results here; avoid it")
        except Exception as e:  # noqa: BLE001
            _line("fail", f"self-test {p:<10}", repr(e))
    try:
        a3, b3 = torch.randn(4, 64, 96), torch.randn(4, 96, 64)
        o3 = ops._npu_gemm(a3, b3, "fp16")
        err = ((o3.double() - a3.double() @ b3.double()).norm() / (a3.double() @ b3.double()).norm()).item()
        _line("ok" if err < 5e-3 else "fail", "self-test batched", f"rel err {err:.1e}")
    except Exception as e:  # noqa: BLE001
        _line("fail", "self-test batched", repr(e))
    try:
        x, w = torch.randn(2, 16, 20, 20), torch.randn(32, 16, 3, 3)
        oc = ops._npu_conv(x, w, (2, 2), (1, 1), "fp16")
        rc = torch.nn.functional.conv2d(x.double(), w.double(), stride=2, padding=1)
        err = ((oc.double() - rc).norm() / rc.norm()).item()
        _line("ok" if err < 5e-3 else "fail", "self-test conv2d", f"rel err {err:.1e}",
              "" if err < 5e-3 else "convolutions will run on the CPU (npu_mode(conv=False) to skip them)")
    except Exception as e:  # noqa: BLE001
        _line("fail", "self-test conv2d", repr(e))
    try:
        x, w, b = torch.randn(2, 64, 12, 12), torch.randn(64, 16, 3, 3), torch.randn(64)
        oc = ops._npu_conv(x, w, (1, 1), (1, 1), "fp16", groups=4, bias=b, relu=True)
        rc = torch.relu(torch.nn.functional.conv2d(x.double(), w.double(), b.double(), padding=1, groups=4))
        err = ((oc.double() - rc).norm() / rc.norm()).item()
        _line("ok" if err < 5e-3 else "fail", "self-test grouped conv + bias + ReLU", f"rel err {err:.1e}",
              "" if err < 5e-3 else "grouped convolutions / fused epilogues are unreliable here")
    except Exception as e:  # noqa: BLE001
        _line("fail", "self-test grouped conv + bias + ReLU", repr(e))

    # --- calibration ---
    from .calibrate import cached_threshold, NEVER
    t = cached_threshold(ops.get_precision())
    if t is None:
        _line("info", "calibration", "not yet run (happens automatically on first npu_mode(), or run "
              "`python -m rk3588_torch calibrate`)")
    else:
        _line("ok", "calibration", "NPU never faster here" if t >= NEVER else f"NPU used from {t:,} MACs")
    return _finish()


def _finish():
    fails = _results.count("fail")
    warns = _results.count("warn")
    print()
    if fails:
        print(f"{fails} problem(s) found; matmuls will run on the CPU until they are fixed.")
    else:
        print("All good" + (f" ({warns} warning(s))" if warns else "") + ": wrap training in "
              "rk3588_torch.npu_mode() to use the NPU.")
    return 1 if fails else 0
