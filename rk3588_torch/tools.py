"""compare(): should this model use the NPU, and in which precision?
profile(): where does a training step's time go (NPU GEMMs vs CPU ops)?"""
import contextlib
import copy
import statistics
import sys
import time

import torch
import torch.nn.functional as F

from . import ops
from .mode import npu_mode

# Gradient error (relative to CPU fp32) always accepted for a precision. On top of this, a
# precision is accepted if its NPU error is at most BASELINE_FACTOR x the error of the
# same model run on the CPU in that precision: in ReLU/MaxPool networks a handful of
# rounding-induced flips can inflate the first-step gradient error without hurting
# training, and the CPU low-precision run measures exactly that inherent effect.
GRAD_TOL = {"fp16": 5e-3, "fp16_out16": 5e-3, "bf16": 2e-2, "int8": 1.5e-1, "fp16x3": 1e-3}
BASELINE_FACTOR = 2.0
_EMULATE = {"fp16": torch.float16, "fp16_out16": torch.float16, "bf16": torch.bfloat16}


def _lowprec_baseline(model, inputs, loss_fn, precision, ref_grads, seed):
    """Relative gradient error of an fp32 CPU step that rounds exactly where the NPU does:
    every Linear/Conv2d input, weight and incoming gradient is rounded to `precision`
    (fp16 or bf16) and the products accumulate in fp32. Fast (it runs at fp32 speed) and
    measures the error inherent to the precision, independent of the NPU."""
    dt = _EMULATE.get(precision)
    if dt is None or ref_grads is None:
        return None
    try:
        m = copy.deepcopy(model)
        m.train()
        rnd = lambda t: t.to(dt).to(t.dtype) if torch.is_tensor(t) and t.is_floating_point() else t  # noqa: E731
        hooks = []
        for mod in m.modules():
            if isinstance(mod, (torch.nn.Linear, torch.nn.Conv2d)):
                with torch.no_grad():
                    mod.weight.copy_(rnd(mod.weight))
                hooks.append(mod.register_forward_pre_hook(lambda _m, args: tuple(rnd(a) for a in args)))
                hooks.append(mod.register_full_backward_pre_hook(lambda _m, go: tuple(rnd(g) for g in go)))
        torch.manual_seed(seed)
        loss = loss_fn(m(*inputs))
        loss.backward()
        for h in hooks:
            h.remove()
        g = _flat_grads(m)
        den = ref_grads.norm().item()
        return (g - ref_grads).norm().item() / den if den > 0 else 0.0
    except Exception:  # noqa: BLE001
        return None


def _flat_grads(model):
    gs = [p.grad.detach().double().flatten() for p in model.parameters() if p.grad is not None]
    return torch.cat(gs) if gs else torch.zeros(0, dtype=torch.float64)


def _default_loss(target):
    def loss_fn(out):
        o = out[0] if isinstance(out, (tuple, list)) else out
        if hasattr(o, "logits"):
            o = o.logits
        if target is None:
            return o.float().pow(2).mean()
        if target.dtype in (torch.long, torch.int32, torch.int64):
            return F.cross_entropy(o.reshape(-1, o.shape[-1]), target.reshape(-1))
        return F.mse_loss(o.float(), target.float())
    return loss_fn


def compare(model, inputs, target=None, loss_fn=None, precisions=("fp16", "bf16", "int8", "fp16x3"),
            steps=6, warmup=2, lr=1e-3, conv=True, seed=0, verbose=True, file=None):
    """Train `steps` steps on the CPU and with the NPU in each precision, from identical copies
    of `model`, and report step time, speedup, gradient error and loss drift.

    inputs: a tensor or tuple of tensors passed to model(*inputs)
    target: optional labels (long -> cross-entropy) or regression targets (float -> MSE)
    loss_fn: optional callable(output) -> scalar loss (overrides target)
    Returns a dict with per-mode results and a recommendation.
    """
    file = file or sys.stdout
    inputs = tuple(inputs) if isinstance(inputs, (tuple, list)) else (inputs,)
    loss_fn = loss_fn or _default_loss(target)
    modes = ["cpu"] + [p for p in precisions]
    if not ops.is_enabled():
        modes = ["cpu"]
        print("rk3588_torch.compare: NPU unavailable; only the CPU can be measured.", file=file)

    results = {}
    ref_grads = None
    ref_losses = None
    for mode in modes:
        m = copy.deepcopy(model)
        m.train()
        opt = torch.optim.SGD(m.parameters(), lr=lr)
        torch.manual_seed(seed)
        ops.stats(reset=True)
        times, losses, grads = [], [], None
        try:
            for step in range(warmup + steps):
                t0 = time.perf_counter()
                ctx = contextlib.nullcontext() if mode == "cpu" else npu_mode(precision=mode, conv=conv)
                with ctx:
                    loss = loss_fn(m(*inputs))
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                if step == 0:
                    grads = _flat_grads(m)
                opt.step()
                times.append(time.perf_counter() - t0)
                losses.append(float(loss.detach()))
        except Exception as e:  # noqa: BLE001
            results[mode] = {"error": repr(e)}
            continue
        st = ops.stats()
        r = {"step_ms": statistics.median(times[warmup:]) * 1e3 if steps else float("nan"),
             "losses": losses,
             "npu_share": st["npu_macs"] / max(st["npu_macs"] + st["cpu_macs"], 1),
             "fallbacks": st["fallbacks"]}
        if mode == "cpu":
            ref_grads, ref_losses = grads, losses
        else:
            den = ref_grads.norm().item()
            r["grad_err"] = (grads - ref_grads).norm().item() / den if den > 0 else 0.0
            r["loss_diff"] = sum(abs(a - b) for a, b in zip(losses, ref_losses)) / len(losses)
            r["rel_loss_diff"] = r["loss_diff"] / max(sum(abs(v) for v in ref_losses) / len(ref_losses), 1e-12)
            r["baseline_err"] = _lowprec_baseline(model, inputs, loss_fn, mode, ref_grads, seed)
        results[mode] = r

    cpu_ms = results["cpu"]["step_ms"]
    best = None
    for mode in modes[1:]:
        r = results.get(mode, {})
        if "error" in r:
            continue
        r["speedup"] = cpu_ms / r["step_ms"]
        tol = GRAD_TOL.get(mode, 5e-2)
        if r.get("baseline_err") is not None:
            tol = max(tol, BASELINE_FACTOR * r["baseline_err"])
        r["accurate"] = r["grad_err"] <= tol and r["rel_loss_diff"] <= 0.05
        if r["accurate"] and r["speedup"] >= 1.1:
            if best is None or r["step_ms"] < results[best]["step_ms"]:
                best = mode
    rec = best or "cpu"
    if verbose:
        print(f"\n{'mode':<12}{'step ms':>10}{'speedup':>9}{'grad err':>11}{'cpu same prec':>14}{'loss diff':>11}"
              f"{'on NPU':>8}{'fallbk':>8}  ok", file=file)
        for mode in modes:
            r = results[mode]
            if "error" in r:
                print(f"{mode:<12}  error: {r['error']}", file=file)
                continue
            if mode == "cpu":
                print(f"{mode:<12}{r['step_ms']:>10.1f}{1.0:>8.2f}x{'-':>11}{'-':>14}{'-':>11}{'-':>8}{'-':>8}",
                      file=file)
            else:
                base = r.get("baseline_err")
                print(f"{mode:<12}{r['step_ms']:>10.1f}{r['speedup']:>8.2f}x{r['grad_err']:>11.1e}"
                      f"{('%.1e' % base) if base is not None else '-':>14}{r['loss_diff']:>11.2e}"
                      f"{r['npu_share'] * 100:>7.0f}%{r['fallbacks']:>8}  {'yes' if r.get('accurate') else 'NO'}",
                      file=file)
        print("  grad err: first-step gradient vs CPU fp32.  cpu same prec: the same step on the CPU with "
              "inputs, weights and gradients rounded to that precision (the error the precision itself causes)",
              file=file)
        if rec == "cpu":
            print("\nrecommendation: stay on the CPU (no precision was both accurate and >=1.1x faster "
                  "for this model and batch size; larger layers or batches help)", file=file)
        else:
            print(f"\nrecommendation: npu_mode(precision={rec!r})  "
                  f"({results[rec]['speedup']:.2f}x faster, gradient error {results[rec]['grad_err']:.1e})", file=file)
    return {"results": results, "recommendation": rec}


@contextlib.contextmanager
def profile(top=12, file=None):
    """Profile the enclosed code and print where the time went.

        with rk3588_torch.profile():
            with rk3588_torch.npu_mode():
                loss = model(x).sum(); loss.backward()
    """
    file = file or sys.stdout
    ops.stats(reset=True)
    act = [torch.profiler.ProfilerActivity.CPU]
    t0 = time.perf_counter()
    with torch.profiler.profile(activities=act) as prof:
        yield prof
    wall = time.perf_counter() - t0
    st = ops.stats()
    print(f"\nrk3588_torch profile: {wall * 1e3:.1f} ms wall", file=file)
    print(f"  NPU ops   : {st['npu_calls']:6d} calls  {st['npu_time'] * 1e3:9.1f} ms  "
          f"({st['npu_time'] / wall * 100:4.1f}% of wall)  {st['npu_macs'] / 1e9:.2f} GMAC", file=file)
    print(f"  CPU ops   : {st['cpu_calls']:6d} calls  {st['cpu_time'] * 1e3:9.1f} ms  "
          f"({st['cpu_time'] / wall * 100:4.1f}% of wall)  {st['cpu_macs'] / 1e9:.2f} GMAC "
          "(below the NPU threshold, outside npu_mode, or fallbacks)", file=file)
    if st["fallbacks"] or st["rejected_shapes"]:
        print(f"  fallbacks : {st['fallbacks']}  shapes pinned to CPU: {st['rejected_shapes']}", file=file)
    rows = [e for e in prof.key_averages() if e.self_cpu_time_total > 0]
    rows.sort(key=lambda e: e.self_cpu_time_total, reverse=True)
    total = sum(e.self_cpu_time_total for e in rows) or 1
    print(f"  top ops by self CPU time (profiler overhead included):", file=file)
    for e in rows[:top]:
        tag = "  <- NPU" if e.key.startswith("rknpu::") else ""
        print(f"    {e.key[:44]:<44} {e.count:6d}x {e.self_cpu_time_total / 1e3:9.1f} ms "
              f"{e.self_cpu_time_total / total * 100:5.1f}%{tag}", file=file)
