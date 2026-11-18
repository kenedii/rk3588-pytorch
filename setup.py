"""Builds csrc/librknpu_gemm.so and ships it inside the rk3588_torch package.

On machines that aren't aarch64 the library is skipped: the package still installs and
everything runs on the CPU (useful for developing on a PC)."""
import os
import platform
import shutil
import subprocess

from setuptools import Distribution, setup
from setuptools.command.build_py import build_py

HERE = os.path.dirname(os.path.abspath(__file__))


class BuildWithLib(build_py):
    def run(self):
        super().run()
        if platform.machine() not in ("aarch64", "arm64"):
            print("rk3588-torch: not an aarch64 machine; skipping the NPU library (CPU fallback only)")
            return
        subprocess.check_call(["make", "-C", os.path.join(HERE, "csrc"), "librknpu_gemm.so"])
        dst = os.path.join(self.build_lib, "rk3588_torch")
        os.makedirs(dst, exist_ok=True)
        shutil.copy2(os.path.join(HERE, "csrc", "librknpu_gemm.so"), dst)


class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return platform.machine() in ("aarch64", "arm64")


setup(cmdclass={"build_py": BuildWithLib}, distclass=BinaryDistribution)
