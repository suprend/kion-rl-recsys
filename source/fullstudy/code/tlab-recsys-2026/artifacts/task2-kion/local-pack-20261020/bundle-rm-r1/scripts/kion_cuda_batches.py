"""Opt-in, bounded, single-copy transport for validated KION numpy batches.

``with PackedBatcher(data, device, max_batch_size=256) as batches:``
``    x, lengths, y, user = batches.batch(indices)``

Each returned tensor owns its batch storage independently of subsequent calls.
Indices, duplicate rows, dynamic ``max(lengths)`` width and native dtypes match
the conditioned runner. User-field precedence is bundle_user, sex, then age.
This transport does not validate vocabulary, one-hot semantics or provenance;
the ordinary data/sidecar loaders must still run first. Source arrays must not
be modified concurrently. No random state is read or changed.

CPU uses owned pageable storage and calls no CUDA API. CUDA normally transfers
one packed byte tensor synchronously. Explicit ``pin_memory=True`` lazily
allocates at most two reusable pinned CPU slabs; their aggregate capacity is
checked against a hard 32 MiB ceiling before allocation. CUDA outputs are fresh
allocations, never reusable device buffers. A slot's completion event protects
its host source until the copy completes. Calls use the caller's current CUDA
stream; callers using a different consumer stream must order it themselves.

Use a context manager or call ``close()``: it drains copies before releasing
sources. The object is single-threaded. Pin-allocation failures fall back once
to synchronous pageable transport and expose ``fallback_reason``. Transfer and
event errors propagate; an unsuccessful safety drain retains sources and blocks
further batches until close succeeds. There is no automatic OOM retry, data
reordering, whole-dataset pinning or GPU availability probe.

CPU tests cover byte/layout parity and mocked lifecycle only. Real CUDA parity,
memory behaviour and performance require a separate admitted measurement.
"""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import numpy as np
import torch


MAX_STAGING_BYTES = 32 << 20
_USER_WIDTHS = {"bundle_user": 10, "sex": 3, "age": 7}


@dataclass(frozen=True)
class _Layout:
    rows: int
    width: int
    user_width: int

    @property
    def lengths_offset(self):
        return self.rows * self.width * 8

    @property
    def y_offset(self):
        return self.lengths_offset + self.rows * 8

    @property
    def user_offset(self):
        return self.y_offset + self.rows * 8

    @property
    def size(self):
        return self.user_offset + self.rows * self.user_width * 4


@dataclass
class _Slot:
    host: torch.Tensor
    event: object = None
    pending: bool = False


def _allocate_pinned(size):
    return torch.empty(size, dtype=torch.uint8, device="cpu", pin_memory=True)


def _copy_to_device(host, device, *, non_blocking):
    return host.to(device=device, non_blocking=non_blocking)


def _record_cuda_event(device):
    event = torch.cuda.Event()
    event.record(torch.cuda.current_stream(device))
    return event


def _synchronize_cuda(device):
    torch.cuda.synchronize(device)


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


class PackedBatcher:
    def __init__(self, data, device, *, max_batch_size,
                 max_staging_bytes=MAX_STAGING_BYTES, pin_memory=False, buffers=2):
        self.device = torch.device(device)
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError("PackedBatcher supports CPU and CUDA only")
        self.max_batch_size = _positive_integer(max_batch_size, "max_batch_size")
        budget = _positive_integer(max_staging_bytes, "max_staging_bytes")
        if budget > MAX_STAGING_BYTES:
            raise ValueError("max_staging_bytes exceeds the hard 32 MiB limit")
        self._buffer_count = _positive_integer(buffers, "buffers")
        if self._buffer_count > 2:
            raise ValueError("buffers must be one or two")
        if not isinstance(pin_memory, bool):
            raise TypeError("pin_memory must be bool")
        try:
            self._data = {key: data[key] for key in ("x", "lengths", "y")}
        except KeyError as error:
            raise ValueError(f"Missing batch array: {error.args[0]}") from error
        self.user_field = next((key for key in _USER_WIDTHS if key in data), None)
        if self.user_field:
            self._data[self.user_field] = data[self.user_field]
        if any(not isinstance(value, np.ndarray) for value in self._data.values()):
            raise TypeError("Batch arrays must be numpy arrays")
        x = self._data["x"]
        if x.ndim != 2 or x.shape[1] < 1:
            raise ValueError("x must be a matrix with positive context width")
        self._rows, self._context = x.shape
        for key in ("x", "lengths", "y"):
            value = self._data[key]
            if value.dtype != np.dtype(np.int64) or (key != "x" and value.shape != (self._rows,)):
                raise ValueError("Expected native int64 x, lengths and y with matching rows")
        self._user_width = _USER_WIDTHS.get(self.user_field, 0)
        if self.user_field:
            user = self._data[self.user_field]
            if user.dtype != np.dtype(np.float32) or user.shape != (self._rows, self._user_width):
                raise ValueError(f"Expected native float32 {self.user_field} with matching rows/width")
        self._pinned = pin_memory and self.device.type == "cuda"
        self._capacity = _Layout(self.max_batch_size, self._context, self._user_width).size
        required = self._capacity * (self._buffer_count if self._pinned else 1)
        if required > budget:
            raise ValueError(f"Configured batch staging needs {required} bytes, exceeding cap {budget}")
        self.fallback_reason = "CPU uses owned pageable batches" if pin_memory and not self._pinned else None
        self._slots = []
        self._next_slot = 0
        self._unsafe = False
        self._closing = False
        self._closed = False

    @property
    def mode(self):
        return "cpu" if self.device.type == "cpu" else ("pinned-packed" if self._pinned else "packed")

    @property
    def staging_reserved_bytes(self):
        return sum(slot.host.numel() for slot in self._slots)

    def _ensure_pool(self):
        if self._slots:
            return
        slots = []
        try:
            for _ in range(self._buffer_count):
                slots.append(_Slot(_allocate_pinned(self._capacity)))
        except (RuntimeError, MemoryError, OSError) as error:
            # No transfers exist during allocation: partial slabs are safe to release.
            slots.clear()
            self._pinned = False
            self.fallback_reason = f"Pinned staging allocation failed: {error}"
            return
        self._slots = slots

    def _indices_and_layout(self, indices):
        rows = np.asarray(indices)
        if rows.ndim != 1 or rows.dtype.kind not in "iu":
            raise TypeError("indices must be a one-dimensional integer array")
        if not len(rows) or len(rows) > self.max_batch_size:
            raise ValueError("indices must contain one to max_batch_size rows")
        if np.any(rows < 0) or np.any(rows >= self._rows):
            raise IndexError("indices outside source rows")
        rows = rows.astype(np.intp, copy=False)
        lengths = self._data["lengths"][rows]
        if np.any(lengths < 1) or np.any(lengths > self._context):
            raise ValueError("Batch contains invalid lengths")
        return rows, lengths, _Layout(len(rows), int(lengths.max()), self._user_width)

    def _pack(self, host, indices, lengths, layout):
        # Each indexed gather is bounded by this batch, including noncontiguous sources.
        storage = host.numpy()
        np.copyto(np.ndarray((layout.rows, layout.width), np.int64, storage, 0),
                  self._data["x"][indices, :layout.width])
        np.copyto(np.ndarray((layout.rows,), np.int64, storage, layout.lengths_offset), lengths)
        np.copyto(np.ndarray((layout.rows,), np.int64, storage, layout.y_offset), self._data["y"][indices])
        if self.user_field:
            np.copyto(np.ndarray((layout.rows, layout.user_width), np.float32, storage, layout.user_offset),
                      self._data[self.user_field][indices])

    @staticmethod
    def _unpack(storage, layout):
        x = storage[:layout.lengths_offset].view(torch.int64).reshape(layout.rows, layout.width)
        lengths = storage[layout.lengths_offset:layout.y_offset].view(torch.int64)
        y = storage[layout.y_offset:layout.user_offset].view(torch.int64)
        user = (storage[layout.user_offset:layout.size].view(torch.float32).reshape(layout.rows, layout.user_width)
                if layout.user_width else None)
        return x, lengths, y, user

    def batch(self, indices):
        if self._closed or self._closing:
            raise RuntimeError("PackedBatcher is closed")
        if self._unsafe:
            raise RuntimeError("PackedBatcher has a failed transfer; close must drain retained sources")
        rows, lengths, layout = self._indices_and_layout(indices)
        if self._pinned:
            self._ensure_pool()
        if not self._pinned:
            host = torch.empty(layout.size, dtype=torch.uint8, device="cpu")
            self._pack(host, rows, lengths, layout)
            result = host if self.device.type == "cpu" else _copy_to_device(host, self.device, non_blocking=False)
            return self._unpack(result, layout)

        slot = self._slots[self._next_slot]
        if slot.event is not None:
            slot.event.synchronize()
            slot.event = None
        host = slot.host[:layout.size]
        self._pack(host, rows, lengths, layout)
        slot.pending = True
        try:
            result = _copy_to_device(host, self.device, non_blocking=True)
            slot.event = _record_cuda_event(self.device)
        except BaseException:
            # A failed enqueue/event may still leave an asynchronous reader of host.
            self._unsafe = True
            _synchronize_cuda(self.device)
            slot.pending = False
            self._unsafe = False
            raise
        slot.pending = False
        self._next_slot = (self._next_slot + 1) % len(self._slots)
        return self._unpack(result, layout)

    def close(self):
        if self._closed:
            return
        self._closing = True
        if self._unsafe or any(slot.pending for slot in self._slots):
            _synchronize_cuda(self.device)
            self._unsafe = False
            for slot in self._slots:
                slot.event, slot.pending = None, False
        for slot in self._slots:
            if slot.event is not None:
                slot.event.synchronize()
                slot.event = None
        self._slots.clear()
        self._closed = True

    def __enter__(self):
        if self._closed or self._closing:
            raise RuntimeError("PackedBatcher is closed")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
