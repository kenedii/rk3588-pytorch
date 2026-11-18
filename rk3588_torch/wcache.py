"""Packed-weight cache: skip re-converting weights that haven't changed.

Every NPU call converts its fp32 weight operand to fp16/bf16/int8 and re-lays it out for the
NPU. For weights that don't change between calls (inference, evaluation, frozen layers in
LoRA-style fine-tuning, gradient-accumulation micro-batches) that work can be done once:
the library keeps the packed copy in its own NPU buffer and reuses it.

A weight is cached once it has been used twice with the same values, so ordinary training
(weights change every step) never pays for caching. "Same values" is tracked by
(storage address, PyTorch's in-place version counter, shape, strides, use) and confirmed on
every hit by comparing a small sample of the tensor, which also catches updates that bypass
the version counter (``p.data.add_(...)``). A strong reference to each cached tensor keeps its
memory from being reused by another tensor while the entry exists.

Environment: RKNPU_WCACHE=0 disables it; RKNPU_WCACHE_MB caps the NPU memory (default 256).
"""
import itertools
import os
import threading
from collections import OrderedDict

import torch

ENABLED = os.environ.get("RKNPU_WCACHE", "1") not in ("0", "false", "no")
MAX_ENTRIES = int(os.environ.get("RKNPU_WCACHE_ENTRIES", "512"))
SAMPLE = 16                      # fingerprint = SAMPLE x SAMPLE elements

_ids = itertools.count(1)
_lock = threading.Lock()
_seen = OrderedDict()            # signature -> times seen (not cached yet)
_entries = OrderedDict()         # signature -> (key, tensor, fingerprint)
_by_slot = {}                    # signature without the version -> signature (to drop old versions)
_stats = {"hits": 0, "misses": 0, "stale": 0}


def _fingerprint(t):
    m = t if t.dim() == 2 else t.reshape(t.shape[0], -1)
    r = torch.linspace(0, m.shape[0] - 1, min(SAMPLE, m.shape[0])).long()
    c = torch.linspace(0, m.shape[1] - 1, min(SAMPLE, m.shape[1])).long()
    return m.index_select(0, r).index_select(1, c).clone()


def key_for(t, tag):
    """Cache key (int) for weight tensor `t` used as `tag`, or 0 if it shouldn't be cached
    (yet). Call with the NPU about to consume `t` as its weight operand."""
    if not ENABLED or t.dim() < 2 or t.numel() == 0:
        return 0
    if t.dim() > 2 and not t.is_contiguous():
        return 0
    slot = (t.data_ptr(), tuple(t.shape), tuple(t.stride()), t.dtype, tag)
    sig = slot + (t._version,)
    with _lock:
        ent = _entries.get(sig)
        if ent is not None:
            key, ref, fp = ent
            if torch.equal(_fingerprint(t), fp):
                _entries.move_to_end(sig)
                _stats["hits"] += 1
                return key
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
            return 0
        key = next(_ids)
        _entries[sig] = (key, t, _fingerprint(t))
        _by_slot[slot] = sig
        while len(_entries) > MAX_ENTRIES:
            _drop(next(iter(_entries)))
        _stats["misses"] += 1
        return key


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
    with _lock:
        for sig in list(_entries):
            _drop(sig)
        _seen.clear()


def info():
    out = dict(_stats, entries=len(_entries))
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
