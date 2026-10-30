"""Lab Streaming Layer (LSL) transport: a producer process streams chunks, an inlet feeds the ring buffer.

    python -m bci.lsl_stream produce --source synthetic --n-chunks 50     # the producer process
    python -m bci.lsl_stream jitter  --seconds 30                         # spawn a producer, measure jitter

Jitter definition (SPEC.md section 11): for every chunk the producer stamps ``local_clock()`` at push; the
consumer records ``t_ring = local_clock()`` right after ``RingBuffer.write``. The transport delay is
``d = t_ring - (timestamp + time_correction)`` and **jitter is the standard deviation of d**. Also reported:
p99 - p50 of d, the max, and the producer's schedule jitter (std of timestamp interval - chunk period).
Inlet post-processing is off (``processing_flags = 0``), so liblsl's de-jitter smoothing does not flatter it.
Loopback on one machine only: network jitter is not measured.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

from . import CHUNK_SAMPLES, FS_HZ
from .ring_buffer import RingBuffer
from .stream import Chunk

STREAM_TYPE = "EEG"
DEFAULT_SOURCE_ID = "realtime-bci-producer"
SPIN_S = 0.002                      # busy-wait the last 2 ms before each deadline to cut sleep overshoot


def _lsl():
    try:
        import pylsl
    except ImportError as exc:       # pragma: no cover
        raise ImportError("pylsl is required for LSL streaming: pip install pylsl (or pip install -e '.[real]')") from exc
    return pylsl


def new_source_id() -> str:
    """Unique per run so a stale stream from an earlier run is never picked up."""
    return f"{DEFAULT_SOURCE_ID}-{uuid.uuid4().hex[:8]}"


# ------------------------------------------------------------------------------------------ producer
def produce(chunks: Iterable[Chunk], *, n_channels: int, source_id: str, name: str = "bci-eeg", fs: int = FS_HZ,
            realtime: bool = True, consumer_timeout: float = 30.0, startup_delay: float = 2.0) -> int:
    """Push ``chunks`` to an LSL outlet; returns the number pushed.

    Waits for a consumer, then ``startup_delay`` seconds so the consumer can finish its clock-offset
    estimation before data flows (otherwise the first chunks sit in the buffer and look like jitter).
    With ``realtime`` chunk k is pushed at ``t0 + (k + 1) * period`` (when its last sample would exist).
    """
    pylsl = _lsl()
    info = pylsl.StreamInfo(name, STREAM_TYPE, n_channels, fs, pylsl.cf_float32, source_id)
    outlet = pylsl.StreamOutlet(info, chunk_size=0, max_buffered=60)
    if not outlet.wait_for_consumers(consumer_timeout):
        raise TimeoutError(f"no LSL consumer connected within {consumer_timeout} s")
    time.sleep(startup_delay)
    t0, pushed = pylsl.local_clock(), 0
    for chunk in chunks:
        if realtime:
            deadline = t0 + (pushed + 1) * chunk.data.shape[0] / fs
            while True:
                remaining = deadline - pylsl.local_clock()
                if remaining <= 0:
                    break
                if remaining > SPIN_S:
                    time.sleep(remaining - SPIN_S)
        outlet.push_chunk(np.ascontiguousarray(chunk.data, dtype=np.float32), pylsl.local_clock())
        pushed += 1
    time.sleep(0.5)                  # let the consumer drain before the outlet is destroyed
    return pushed


def spawn_producer(*, source: str = "synthetic", source_id: str, n_chunks: int, realtime: bool = True,
                   channels: int = 64, seed: int = 0, subject: int = 1, session: str = "1test",
                   startup_delay: float = 2.0) -> subprocess.Popen:
    """Start ``python -m bci.lsl_stream produce ...`` as a separate process (no GIL sharing with the consumer).

    Its stderr (liblsl logging) goes to a temp file kept in ``proc.stderr_path`` for diagnosing failures.
    """
    cmd = [sys.executable, "-m", "bci.lsl_stream", "produce", "--source", source, "--source-id", source_id,
           "--n-chunks", str(n_chunks), "--channels", str(channels), "--seed", str(seed),
           "--subject", str(subject), "--session", session, "--startup-delay", str(startup_delay)]
    if not realtime:
        cmd.append("--no-realtime")
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [str(Path(__file__).resolve().parents[1]),
                                                                      os.environ.get("PYTHONPATH")]))}
    err = tempfile.NamedTemporaryFile("w+", suffix=".lsl-producer.log", delete=False)
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL, stderr=err)
    proc.stderr_path = err.name                                   # type: ignore[attr-defined]
    return proc


def finish_producer(proc: subprocess.Popen, timeout: float = 15.0) -> int | None:
    """Wait for the producer process (kill it on timeout) and delete its stderr log if it exited cleanly."""
    try:
        code = proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        code = proc.wait()
    path = getattr(proc, "stderr_path", None)
    if path and code == 0:
        try:
            os.remove(path)
        except OSError:
            pass
    return code


# ------------------------------------------------------------------------------------------ consumer
class LSLChunkSource:
    """Iterator of :class:`Chunk` pulled from an LSL inlet, ``chunk_samples`` samples at a time.

    ``log`` gets one record per chunk: the producer timestamp of its last sample (clock-corrected) and the
    local time the pull completed. Chunks carry ``info = {"lsl_ts", "t_pull"}``; ``label`` is -1 (labels are
    not transmitted).
    """

    def __init__(self, source_id: str, *, n_chunks: int | None = None, chunk_samples: int = CHUNK_SAMPLES,
                 resolve_timeout: float = 30.0, pull_timeout: float = 5.0) -> None:
        pylsl = _lsl()
        found = pylsl.resolve_byprop("source_id", source_id, timeout=resolve_timeout)
        if not found:
            raise TimeoutError(f"LSL stream with source_id={source_id!r} not found within {resolve_timeout} s")
        self._pylsl = pylsl
        self.inlet = pylsl.StreamInlet(found[0], max_buflen=60, processing_flags=0)
        self.inlet.open_stream(timeout=resolve_timeout)
        self.time_correction = self.inlet.time_correction(10.0)
        self.n_channels = found[0].channel_count()
        self.fs = found[0].nominal_srate()
        self.chunk_samples, self.n_chunks, self.pull_timeout = chunk_samples, n_chunks, pull_timeout
        self.log: list[tuple[float, float]] = []                  # (timestamp_corrected, t_pull)
        self._index = 0

    def __iter__(self) -> Iterator[Chunk]:
        return self

    def __next__(self) -> Chunk:
        if self.n_chunks is not None and self._index >= self.n_chunks:
            raise StopIteration
        parts, stamps, have = [], [], 0
        while have < self.chunk_samples:
            samples, ts = self.inlet.pull_chunk(timeout=self.pull_timeout, max_samples=self.chunk_samples - have)
            if not ts:
                raise TimeoutError(f"no LSL data for {self.pull_timeout} s after {self._index} chunks")
            parts.append(np.asarray(samples, dtype=np.float32))
            stamps.append(ts[-1])
            have += len(ts)
        t_pull = self._pylsl.local_clock()
        ts_last = stamps[-1] + self.time_correction
        self.log.append((ts_last, t_pull))
        chunk = Chunk(self._index, np.concatenate(parts), -1, time.perf_counter_ns(),
                      {"lsl_ts": ts_last, "t_pull": t_pull})
        self._index += 1
        return chunk


class TimedRingBuffer(RingBuffer):
    """RingBuffer that records ``local_clock()`` right after every completed write (``t_ring``)."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._clock = _lsl().local_clock
        self.write_clock: list[float] = []

    def write(self, chunk: np.ndarray) -> None:
        super().write(chunk)
        self.write_clock.append(self._clock())


# ------------------------------------------------------------------------------------------ statistics
def jitter_stats(log: list[tuple[float, float]], t_ring: list[float], *, fs: int = FS_HZ,
                 chunk_samples: int = CHUNK_SAMPLES, skip: int = 5) -> dict:
    """Transport-delay and schedule statistics in milliseconds; the first ``skip`` chunks are excluded
    (connection set-up)."""
    ts = np.array([r[0] for r in log])
    t_pull = np.array([r[1] for r in log])
    t_ring = np.asarray(t_ring)
    n = min(len(ts), len(t_ring))
    ts, t_pull, t_ring = ts[:n], t_pull[:n], t_ring[:n]
    if n <= skip + 2:
        raise ValueError(f"need more than {skip + 2} chunks, got {n}")
    d_ring = (t_ring - ts)[skip:] * 1e3
    d_pull = (t_pull - ts)[skip:] * 1e3
    sched = (np.diff(ts)[skip:] - chunk_samples / fs) * 1e3
    p50, p99 = np.percentile(d_ring, [50, 99])
    return {"n_chunks": int(d_ring.size), "skipped": skip,
            "delay_mean_ms": float(d_ring.mean()), "jitter_std_ms": float(d_ring.std(ddof=1)),
            "delay_p50_ms": float(p50), "delay_p99_ms": float(p99), "p99_minus_p50_ms": float(p99 - p50),
            "delay_max_ms": float(d_ring.max()), "pull_jitter_std_ms": float(d_pull.std(ddof=1)),
            "producer_schedule_std_ms": float(sched.std(ddof=1)), "producer_schedule_max_abs_ms": float(np.abs(sched).max())}


def measure_jitter(*, seconds: float = 30.0, source: str = "synthetic", channels: int = 64, seed: int = 0,
                   subject: int = 1, session: str = "1test") -> tuple[dict, np.ndarray]:
    """Spawn a producer process, consume ``seconds`` of 1x real-time stream into a ring buffer, return
    (stats, consumed data). Cleans up the producer."""
    n_chunks = max(int(round(seconds * FS_HZ / CHUNK_SAMPLES)), 8)
    source_id = new_source_id()
    proc = spawn_producer(source=source, source_id=source_id, n_chunks=n_chunks, channels=channels, seed=seed,
                          subject=subject, session=session)
    try:
        src = LSLChunkSource(source_id, n_chunks=n_chunks)
        ring = TimedRingBuffer(n_channels=src.n_channels)
        data = []
        for chunk in src:
            ring.write(chunk.data)
            data.append(chunk.data)
            ring.consume(chunk.data.shape[0])
        stats = jitter_stats(src.log, ring.write_clock)
        stats["ring_dropped_samples"] = ring.dropped_samples
        return stats, np.concatenate(data)
    finally:
        finish_producer(proc)


# ------------------------------------------------------------------------------------------ CLI
def _cli_produce(args: argparse.Namespace) -> int:
    if args.source == "real":
        from .stream import RealEEGStream
        stream = RealEEGStream(args.subject, args.session, n_chunks=args.n_chunks)
        n_channels = stream.n_channels
    else:
        from .stream import SyntheticEEGStream
        stream = SyntheticEEGStream(seed=args.seed, n_chunks=args.n_chunks, n_channels=args.channels)
        n_channels = args.channels
    produce(stream, n_channels=n_channels, source_id=args.source_id, realtime=not args.no_realtime,
            startup_delay=args.startup_delay)
    return 0


def _cli_jitter(args: argparse.Namespace) -> int:
    stats, _ = measure_jitter(seconds=args.seconds, source=args.source, channels=args.channels)
    for k, v in stats.items():
        print(f"{k:30s} {v:.4f}" if isinstance(v, float) else f"{k:30s} {v}")
    ok = stats["jitter_std_ms"] < 1.0
    print(f"\njitter (std of producer-timestamp -> ring-buffer delay) = {stats['jitter_std_ms']:.4f} ms -> "
          f"{'PASS' if ok else 'FAIL'} (< 1 ms)")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m bci.lsl_stream")
    sub = p.add_subparsers(dest="cmd", required=True)
    pr = sub.add_parser("produce", help="producer process: stream chunks over LSL")
    pr.add_argument("--source", choices=["synthetic", "real"], default="synthetic")
    pr.add_argument("--source-id", default=DEFAULT_SOURCE_ID)
    pr.add_argument("--n-chunks", type=int, default=50)
    pr.add_argument("--channels", type=int, default=64)
    pr.add_argument("--seed", type=int, default=0)
    pr.add_argument("--subject", type=int, default=1)
    pr.add_argument("--session", default="1test")
    pr.add_argument("--no-realtime", action="store_true")
    pr.add_argument("--startup-delay", type=float, default=2.0)
    pr.set_defaults(func=_cli_produce)
    jt = sub.add_parser("jitter", help="measure producer -> ring buffer jitter over LSL")
    jt.add_argument("--seconds", type=float, default=30.0)
    jt.add_argument("--source", choices=["synthetic", "real"], default="synthetic")
    jt.add_argument("--channels", type=int, default=64)
    jt.set_defaults(func=_cli_jitter)
    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
