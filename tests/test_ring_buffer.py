"""Ring buffer: wrap-around, no drops under normal load, drop-oldest + warning on overrun."""

from __future__ import annotations

import logging
import threading

import numpy as np

from bci import CHUNK_SAMPLES, N_CHANNELS, RING_CAPACITY
from bci.ring_buffer import RingBuffer
from bci.stream import SyntheticEEGStream


def _ramp(start: int, n: int) -> np.ndarray:
    """Samples whose value in every channel is the global sample index (easy to verify)."""
    return np.repeat(np.arange(start, start + n, dtype=np.float32)[:, None], N_CHANNELS, axis=1)


def test_wrap_around_preserves_order_and_data():
    ring = RingBuffer(capacity=RING_CAPACITY, chunk_samples=CHUNK_SAMPLES)
    buf_before = ring._buf
    n_chunks = 40                                     # 2560 samples > 2 x capacity; chunks straddle the wrap
    out = []
    for i in range(n_chunks):
        ring.write(_ramp(i * CHUNK_SAMPLES, CHUNK_SAMPLES))
        out.append(ring.read(CHUNK_SAMPLES, timeout=0.1))
    got = np.concatenate(out)
    np.testing.assert_array_equal(got, _ramp(0, n_chunks * CHUNK_SAMPLES))
    assert ring.dropped_samples == 0
    assert ring.wrap_count == n_chunks * CHUNK_SAMPLES // RING_CAPACITY == 2
    assert ring._buf is buf_before and ring._buf.shape == (RING_CAPACITY, N_CHANNELS)   # pre-allocated, never replaced


def test_latest_window_after_wrap():
    ring = RingBuffer()
    total = 30 * CHUNK_SAMPLES                        # 1920 samples: wrapped once, head mid-array
    for i in range(30):
        ring.write(_ramp(i * CHUNK_SAMPLES, CHUNK_SAMPLES))
        ring.consume(CHUNK_SAMPLES)
    np.testing.assert_array_equal(ring.latest(500), _ramp(total - 500, 500))
    out = np.empty((500, N_CHANNELS), dtype=np.float32)
    assert ring.latest(500, out=out) is out
    np.testing.assert_array_equal(out, _ramp(total - 500, 500))


def test_ten_seconds_real_time_no_drops_and_wraps():
    """10.24 s of data (40 chunks) at 1x real time: producer thread -> ring -> consumer thread."""
    n_chunks = 40
    ring = RingBuffer()
    consumed: list[np.ndarray] = []

    def produce():
        for chunk in SyntheticEEGStream(seed=5, n_chunks=n_chunks, realtime=True):
            ring.write(chunk.data)

    def consume():
        while len(consumed) < n_chunks:
            data = ring.read(CHUNK_SAMPLES, timeout=2.0)
            if data is None:
                return                                # timed out: test fails on the length assert
            consumed.append(data)

    threads = [threading.Thread(target=produce, daemon=True), threading.Thread(target=consume, daemon=True)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
        assert not t.is_alive()

    expected = np.concatenate([c.data for c in SyntheticEEGStream(seed=5, n_chunks=n_chunks)])
    got = np.concatenate(consumed)
    assert ring.dropped_samples == 0, f"unexpected drops: {ring.stats()}"
    assert ring.overrun_events == 0
    assert got.shape == expected.shape
    np.testing.assert_array_equal(got, expected)       # nothing lost, nothing reordered
    assert ring.wrap_count >= 2
    print(f"\nring buffer stats after 10.24 s at 1x: {ring.stats()}")


def test_overrun_drops_oldest_chunk_and_warns(caplog):
    ring = RingBuffer()
    with caplog.at_level(logging.WARNING, logger="bci.ring_buffer"):
        for i in range(20):                            # never read: 1280 samples into a 1000-sample buffer
            ring.write(_ramp(i * CHUNK_SAMPLES, CHUNK_SAMPLES))
    assert ring.overrun_events == 5 and ring.dropped_samples == 5 * CHUNK_SAMPLES == 320
    assert any("overrun" in r.message for r in caplog.records)
    remaining = ring.read(ring.unread, timeout=0.1)    # the *newest* contiguous data survives
    np.testing.assert_array_equal(remaining, _ramp(320, 960))
