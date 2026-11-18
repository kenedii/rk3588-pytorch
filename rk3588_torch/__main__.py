"""Command line: python -m rk3588_torch {doctor,calibrate,bench,info}"""
import argparse
import sys


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m rk3588_torch", description=__doc__)
    sub = p.add_subparsers(dest="cmd")
    sub.add_parser("doctor", help="check driver, permissions and library; run an NPU self-test")
    c = sub.add_parser("calibrate", help="measure the CPU/NPU crossover size and cache it")
    c.add_argument("--precision", default="all", help="fp16, bf16, int8, fp16x3, fp16_out16 or all")
    b = sub.add_parser("bench", help="quick CPU vs NPU GEMM benchmark in every precision")
    b.add_argument("--size", type=int, default=1024)
    sub.add_parser("info", help="library path, driver and NPU information")
    args = p.parse_args(argv)

    if args.cmd in (None, "doctor"):
        from .doctor import run
        return run()
    import torch
    from ._lib import PRECISIONS, default_npu, library_path, npu_available, npu_error
    if args.cmd == "info":
        print("library:", library_path())
        if npu_available():
            print("npu:", default_npu().info())
        else:
            print("npu: unavailable:", npu_error())
        return 0
    if not npu_available():
        print("NPU unavailable:", npu_error(), "\nrun `python -m rk3588_torch doctor`")
        return 1
    if args.cmd == "calibrate":
        from .calibrate import calibrate, NEVER
        precs = list(PRECISIONS) if args.precision == "all" else [args.precision]
        for prec in precs:
            print(f"{prec}:")
            m = calibrate(prec, verbose=True)
            print(f"  -> NPU from {m:,} MACs" if m < NEVER else "  -> NPU never faster")
        return 0
    if args.cmd == "bench":
        import time
        from .ops import _npu_gemm
        n = args.size
        a, b = torch.randn(n, n), torch.randn(n, n)
        ref = a.double() @ b.double()

        def best(fn):
            fn()
            t = float("inf")
            for _ in range(5):
                t0 = time.perf_counter()
                fn()
                t = min(t, time.perf_counter() - t0)
            return t
        flop = 2 * n ** 3
        tc = best(lambda: a @ b)
        print(f"{n}^3 GEMM, CPU fp32 ({torch.get_num_threads()} threads): {flop / tc / 1e9:7.1f} GFLOPS")
        for prec in PRECISIONS:
            tn = best(lambda: _npu_gemm(a, b, prec))
            err = ((_npu_gemm(a, b, prec).double() - ref).norm() / ref.norm()).item()
            print(f"  NPU {prec:<10}: {flop / tn / 1e9:7.1f} GFLOPS  ({tc / tn:4.1f}x CPU)  rel err {err:.1e}")
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
