"""LSL transport: producer process -> inlet -> ring buffer. Jitter must be below 1 ms (SPEC.md section 11)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pylsl", reason="pip install pylsl (the 'real' extra)")

from bci.gpu_pipeline import GPUPipeline, drive
from bci.lsl_stream import (LSLChunkSource, TimedRingBuffer, finish_producer, jitter_stats, measure_jitter,
                            new_source_id, spawn_producer)
from bci.ring_buffer import RingBuffer
from bci.stream import SyntheticEEGStream

SECONDS = 13.0                      # 50 chunks of 256 ms at 1x real time


def test_lsl_jitter_below_one_millisecond_and_data_intact():
    stats, data = measure_jitter(seconds=SECONDS, source="synthetic", channels=64, seed=0)
    print("\nLSL producer-timestamp -> ring buffer delay: " + ", ".join(
        f"{k}={v:.4f}" for k, v in stats.items() if isinstance(v, float)))
    assert stats["jitter_std_ms"] < 1.0, f"LSL jitter {stats['jitter_std_ms']:.3f} ms >= 1 ms: {stats}"
    assert stats["ring_dropped_samples"] == 0 and stats["n_chunks"] >= 40
    assert 0 < stats["delay_mean_ms"] < 5                                # a sane transport delay, not a clock error
    expected = np.concatenate([c.data for c in SyntheticEEGStream(seed=0, n_chunks=len(data) // 64)])
    np.testing.assert_array_equal(data, expected)                       # float32 survives the transport bit for bit


def test_jitter_stats_arithmetic():
    ts = [(0.256 * k, 0.0) for k in range(20)]
    delays = np.array([0.001 + 0.0005 * ((-1) ** k) for k in range(20)])    # 1.5 ms / 0.5 ms alternating
    t_ring = [ts[k][0] + delays[k] for k in range(20)]
    s = jitter_stats(ts, t_ring, skip=2)
    assert s["delay_mean_ms"] == pytest.approx(1.0, abs=1e-6) and s["jitter_std_ms"] == pytest.approx(0.5, rel=0.05)
    assert s["producer_schedule_std_ms"] == pytest.approx(0.0, abs=1e-6) and s["n_chunks"] == 18
    with pytest.raises(ValueError):
        jitter_stats(ts[:5], t_ring[:5], skip=5)


def test_pipeline_over_lsl_gives_the_same_scores_as_the_direct_path(params):
    """Same chunks, once straight from the generator and once through LSL (producer process, no pacing)."""
    n = 30
    direct = list(drive(GPUPipeline(params, "cpu", compile=False), RingBuffer(), SyntheticEEGStream(seed=3, n_chunks=n), lookahead=False))
    source_id = new_source_id()
    proc = spawn_producer(source="synthetic", source_id=source_id, n_chunks=n, realtime=False, channels=64, seed=3, startup_delay=1.0)
    try:
        src = LSLChunkSource(source_id, n_chunks=n)
        via_lsl = list(drive(GPUPipeline(params, "cpu", compile=False), TimedRingBuffer(), src, lookahead=False))
    finally:
        assert finish_producer(proc) == 0
    assert len(via_lsl) == len(direct) == n - 7
    np.testing.assert_allclose(np.stack([r.scores for r in via_lsl]), np.stack([r.scores for r in direct]), atol=1e-9)
    assert [r.label for r in via_lsl] == [r.label for r in direct]
