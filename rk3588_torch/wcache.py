"""Packed-weight cache: skip re-converting weights that haven't changed.

Every NPU call converts its fp32 weight operand to fp16/bf16/int8 and re-lays it out for the
NPU. For weights that don't change between calls (inference, evaluation, frozen layers in
LoRA-style fine-tuning, gradient-accumulation micro-batches) that work can be done once:
the library keeps the packed copy in its own NPU buffer and reuses it.

A weight is cached once it has been used twice with the same values, so ordinary training
(weights change every step) never pays for caching. "Same values" is tracked by
(storage address, PyTorch's in-place version counter, shape, strides, use) and confirmed on
every hit by comparing a small sample of the tensor, which also catches updates that bypass
the version counter (``p.data.add_(...)``; one element of every row and column is compared). A strong reference to each cached tensor keeps its
memory from being reused by another tensor while the entry exists.

Environment: RKNPU_WCACHE=0 disables it; RKNPU_WCACHE_MB caps the NPU memory (default 256).
"""
import itertools
import os
import threading
import warnings
from collections import OrderedDict

import torch

ENABLED = os.environ.get("RKNPU_WCACHE", "1") not in ("0", "false", "no")
MAX_ENTRIES = int(os.environ.get("RKNPU_WCACHE_ENTRIES", "512"))
# NPU memory the library keeps for packed weights (it evicts the least recently used beyond this);
# mirrors RKNPU_WCACHE_MB / NPU.set_wcache_mb so the module can tell when the weights in use don't fit
CAP_MB = int(os.environ.get("RKNPU_WCACHE_MB", "256"))
SAMPLE = 4096                    # fingerprint: up to this many rows and this many columns

_ids = itertools.count(1)
_lock = threading.Lock()
_seen = OrderedDict()            # signature -> times seen (not cached yet)
_entries = OrderedDict()         # signature -> (key, tensor, fingerprint)
_by_slot = {}                    # signature without the version -> signature (to drop old versions)
_stats = {"hits": 0, "misses": 0, "stale": 0}
_warned_cap = False


def _fingerprint(t):
    """One element from every row and one from every column (up to SAMPLE of each), at
    scattered positions. An edit to whole rows or columns (p.data[i] = ..., p.data[:, j] *= ...)
    therefore always changes the fingerprint, also on a transposed view; the old 16 x 16 grid
    saw only 16 rows and 16 columns. At most 2 x SAMPLE elements: negligible next to a repack."""
    m = t if t.dim() == 2 else t.reshape(t.shape[0], -1)
    rows, cols = m.shape
    r = torch.linspace(0, rows - 1, min(SAMPLE, rows)).long()
    c = torch.linspace(0, cols - 1, min(SAMPLE, cols)).long()
    # (i * odd) % n is a permutation of 0..n-1 when gcd(odd, n) == 1, so the partner indices
    # are spread out rather than all on the diagonal or in column 0
    rc = (r * 40503 + 7) % cols
    cr = (c * 2654435761 + 3) % rows
    return torch.cat([m[r, rc], m[cr, c]]).clone()


def _packed_estimate(t):
    """Rough NPU bytes of t's packed (fp16/bf16, padded to 32) form."""
    if t.dim() == 2:
        return 2 * (-(-t.shape[0] // 32) * 32) * (-(-t.shape[1] // 32) * 32)
    return 2 * t.numel()


def _warn_if_over_cap():
    global _warned_cap
    if _warned_cap:
        return
    total = sum(_packed_estimate(e[1]) for e in _entries.values())
    if total > CAP_MB << 20:
        _warned_cap = True
        warnings.warn(
            f"rk3588_torch: the {len(_entries)} weights reused across NPU calls need about {total >> 20} MB "
            f"packed, but the weight cache is capped at {CAP_MB} MB (RKNPU_WCACHE_MB). When a loop cycles "
            "through more weights than fit, least-recently-used eviction makes every weight miss on every "
            "step, so the cache costs time instead of saving it. Raise RKNPU_WCACHE_MB if the board has "
            "memory to spare, or set RKNPU_WCACHE=0.", RuntimeWarning, stacklevel=4)


def lookup(t, tag, check=None):
    """Cache key (int) for weight tensor `t` used as `tag`, or 0 if it shouldn't be cached (yet), and
    whether that key was already in the cache (True: `t` is unchanged since it went in, so whatever was
    verified about it then still holds). `check(t)` (e.g. a finite-values test) runs once, just before a
    weight is first entered; if it returns False the weight is not cached. Call with the NPU about to
    consume `t` as its weight operand. float32, bfloat16 and float16 weights are cacheable."""
    if not ENABLED or t.dim() < 2 or t.numel() == 0:
        return 0, False
    if t.dim() > 2 and not t.is_contiguous():
        return 0, False
    slot = (t.data_ptr(), tuple(t.shape), tuple(t.stride()), t.dtype, tag)
    # tensors made under torch.inference_mode() have no version counter (t._version raises);
    # for them the fingerprint alone detects changes
    sig = slot + (-1 if t.is_inference() else t._version,)
    with _lock:
        ent = _entries.get(sig)
        if ent is not None:
            key, ref, fp = ent
            if torch.equal(_fingerprint(t), fp):
                _entries.move_to_end(sig)
                _stats["hits"] += 1
                return key, True
            _stats["stale"] += 1           # changed without a version bump: repack
            _drop(sig)
        old = _by_slot.get(slot)
        if old is not None and old != sig:
            _drop(old)                     # an older version of the same tensor
        n = _seen.pop(sig, 0) + 1
        if n < 2:
            _seen[sig] = n
            while len(_seen) > 4 * MAX_ENTRIES:
                _seen.popitem(last=False)
            _stats["misses"] += 1
            return 0, False
        if check is not None and not check(t):
            _seen[sig] = n
            return 0, False
        key = next(_ids)
        _entries[sig] = (key, t, _fingerprint(t))
        _by_slot[slot] = sig
        while len(_entries) > MAX_ENTRIES:
            _drop(next(iter(_entries)))
        _stats["misses"] += 1
        _warn_if_over_cap()
        return key, False


def key_for(t, tag):
    """lookup() without the hit flag: the cache key for `t`, or 0."""
    return lookup(t, tag)[0]


def _drop(sig):
    ent = _entries.pop(sig, None)
    if ent is None:
        return
    if _by_slot.get(sig[:-1]) == sig:
        del _by_slot[sig[:-1]]
    try:
        from ._lib import default_npu
        default_npu().wcache_drop(ent[0])
    except OSError:
        pass


def clear():
    """Forget every cached weight (frees their NPU memory)."""
    global _warned_cap
    _warned_cap = False
    with _lock:
        for sig in list(_entries):
            _drop(sig)
        _seen.clear()


def info():
    out = dict(_stats, entries=len(_entries), packed_estimate_mb=sum(_packed_estimate(e[1]) for e in _entries.values()) >> 20,
               cap_mb=CAP_MB)
    try:
        from ._lib import default_npu
        out.update({"npu_" + k: v for k, v in default_npu().wcache_info().items()})
    except OSError:
        pass
    return out


def set_enabled(flag):
    global ENABLED
    ENABLED = bool(flag)
    if not ENABLED:
        clear()
