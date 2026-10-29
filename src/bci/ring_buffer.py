"""CPU-side ring buffer: a pre-allocated array with O(1) pointer updates.

Producer calls :meth:`RingBuffer.write` with each chunk; the consumer reads the most
recent window with :meth:`latest` and marks chunks as processed with :meth:`consume`
(or reads FIFO with :meth:`read`). If the producer outruns the consumer, the oldest
chunk is dropped and a warning is logged.
"""

from __future__ import annotations

import logging
import threading
import time

import numpy as np

from . import CHUNK_SAMPLES, N_CHANNELS, RING_CAPACITY

log = logging.getLogger(__name__)


class RingBuffer:
    """Fixed-capacity ``(capacity, n_channels)`` buffer of the most recent samples.

    Positions are tracked with monotonically increasing counters (``_written``,
    ``_read``); array indices are derived from them, so a write never moves existing
    data. A write costs one or two slice copies (two when it wraps) plus pointer
    arithmetic.
    """

    def __init__(
        self,
        capacity: int = RING_CAPACITY,
        n_channels: int = N_CHANNELS,
        chunk_samples: int = CHUNK_SAMPLES,
        dtype=np.float32,
    ) -> None:
        if capacity <= 0 or chunk_samples <= 0 or chunk_samples > capacity:
            raise ValueError("need 0 < chunk_samples <= capacity")
        self.capacity = capacity
        self.n_channels = n_channels
        self.chunk_samples = chunk_samples
        self._buf = np.zeros((capacity, n_channels), dtype=dtype)  # allocated once
        self._head = 0       # next write index
        self._written = 0    # total samples ever written
        self._read = 0       # total samples consumed (FIFO cursor)
        self._lock = threading.Lock()
        self._data_ready = threading.Condition(self._lock)
        # stats
        self.dropped_samples = 0
        self.overrun_events = 0
        self._write_calls = 0
        self._write_ns_total = 0
        self._write_ns_max = 0

    # ------------------------------------------------------------------ producer
    def write(self, chunk: np.ndarray) -> None:
        """Append ``chunk`` of shape ``(n, n_channels)``; drop oldest chunk(s) on overrun."""
        n = chunk.shape[0]
        if chunk.ndim != 2 or chunk.shape[1] != self.n_channels:
            raise ValueError(f"expected (n, {self.n_channels}), got {chunk.shape}")
        if n > self.capacity:
            raise ValueError(f"chunk of {n} samples exceeds capacity {self.capacity}")
        t0 = time.perf_counter_ns()
        with self._lock:
            overflow = (self._written - self._read) + n - self.capacity
            if overflow > 0:
                # Drop whole oldest chunks until the new data fits.
                unread = self._written - self._read
                drop = min(unread, -(-overflow // self.chunk_samples) * self.chunk_samples)
                self._read += drop
                self.dropped_samples += drop
                self.overrun_events += 1
                log.warning(
                    "ring buffer overrun: producer outran consumer, dropped oldest %d samples "
                    "(total dropped: %d)", drop, self.dropped_samples,
                )
            first = min(n, self.capacity - self._head)
            self._buf[self._head:self._head + first] = chunk[:first]
            if first < n:
                self._buf[:n - first] = chunk[first:]
            self._head = (self._head + n) % self.capacity
            self._written += n
            self._data_ready.notify_all()
        dt = time.perf_counter_ns() - t0
        self._write_calls += 1
        self._write_ns_total += dt
        if dt > self._write_ns_max:
            self._write_ns_max = dt

    # ------------------------------------------------------------------ consumer
    def latest(self, n: int, out: np.ndarray | None = None) -> np.ndarray:
        """Copy the most recent ``n`` samples (oldest first) without consuming them.

        If ``out`` (shape ``(n, n_channels)``) is given, the copy goes straight into it
        — this is how the pipeline fills its pinned staging buffer.
        """
        if not 0 < n <= self.capacity:
            raise ValueError(f"n must be in 1..{self.capacity}")
        if out is None:
            out = np.empty((n, self.n_channels), dtype=self._buf.dtype)
        elif out.shape != (n, self.n_channels):
            raise ValueError(f"out must have shape {(n, self.n_channels)}, got {out.shape}")
        with self._lock:
            if self._written < n:
                raise ValueError(f"only {self._written} samples written, need {n}")
            start = (self._head - n) % self.capacity
            first = min(n, self.capacity - start)
            out[:first] = self._buf[start:start + first]
            if first < n:
                out[first:] = self._buf[:n - first]
        return out

    def consume(self, n: int) -> None:
        """Mark the oldest ``n`` unread samples as processed (no copy)."""
        with self._lock:
            self._read = min(self._written, self._read + n)

    def read(self, n: int, timeout: float | None = None) -> np.ndarray | None:
        """FIFO-read the oldest ``n`` unread samples, blocking until available.

        Returns ``None`` on timeout.
        """
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._data_ready:
            while self._written - self._read < n:
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return None
                self._data_ready.wait(remaining)
            out = np.empty((n, self.n_channels), dtype=self._buf.dtype)
            start = self._read % self.capacity
            first = min(n, self.capacity - start)
            out[:first] = self._buf[start:start + first]
            if first < n:
                out[first:] = self._buf[:n - first]
            self._read += n
        return out

    # ------------------------------------------------------------------ introspection
    @property
    def samples_written(self) -> int:
        return self._written

    @property
    def unread(self) -> int:
        return self._written - self._read

    @property
    def wrap_count(self) -> int:
        """How many times the write pointer has passed the end of the array."""
        return self._written // self.capacity

    def stats(self) -> dict:
        calls = max(self._write_calls, 1)
        return {
            "samples_written": self._written,
            "wrap_count": self.wrap_count,
            "dropped_samples": self.dropped_samples,
            "overrun_events": self.overrun_events,
            "write_calls": self._write_calls,
            "write_us_mean": self._write_ns_total / calls / 1e3,
            "write_us_max": self._write_ns_max / 1e3,
            "write_ms_total": self._write_ns_total / 1e6,
        }
