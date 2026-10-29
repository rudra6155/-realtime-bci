"""GPU pipeline: pinned staging -> async H2D (copy stream) -> compute stream -> async D2H.

Hot-path rules (enforced by ``tests/test_latency.py``):
  * no ``.item()`` / ``.cpu()`` / ``.cuda()`` / ``.tolist()`` / ``print`` in ``stage``,
    ``launch``, ``collect``, ``process``, ``drive`` or the model/filter forward passes;
  * H2D/D2H are ``copy_(..., non_blocking=True)`` between pre-allocated pinned and device
    buffers; stream-to-stream dependencies use CUDA events (``wait_event``), not host syncs;
  * the only host waits are: ``h2d_end.synchronize()`` (is this pinned slot free to
    overwrite?) and ``d2h_end.synchronize()`` (is the label ready?) — plus the
    ``torch.linalg.eigh`` sync PyTorch documents (see SPEC.md §4).

Per-chunk life cycle (three calls so the host can enqueue chunk N's H2D *before* it
blocks inside chunk N-1's ``eigh``):  ``stage(N)`` -> ``launch(N-1)`` -> ``collect(N-1)``.
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import csv
import logging
import math
import platform
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import torch

from . import (CHUNK_SAMPLES, FS_HZ, LATENCY_TARGET_MS, N_CHANNELS, WINDOW_SAMPLES)
from .classifier import RiemannianLDA, RiemannianLDAParams, train_default
from .filters import configure_precision, make_bandpass
from .ring_buffer import RingBuffer
from .stream import Chunk, SyntheticEEGStream

log = logging.getLogger(__name__)

_MARKS = ("h2d_start", "h2d_end", "c_start", "filter_end", "cov_end", "cls_end", "d2h_end")
_STAGE_SPANS = {
    "h2d": ("h2d_start", "h2d_end"),
    "filter": ("c_start", "filter_end"),
    "cov": ("filter_end", "cov_end"),
    "classifier": ("cov_end", "cls_end"),      # tangent space (eigh) + LDA head
    "d2h": ("cls_end", "d2h_end"),
}
STAGES = tuple(_STAGE_SPANS)


@dataclass
class Result:
    index: int
    label: int                       # argmax of scores
    scores: np.ndarray               # (n_classes,) float64
    latency_ms: float                # host clock: t_ready - t_recv
    stage_ms: dict[str, float]       # CUDA-event GPU times (host timestamps on CPU)
    t_ready_ns: int


@dataclass
class Frame:
    chunk: Chunk
    result: Result | None            # None while the decode window is still filling


def resolve_device(device: str | torch.device = "auto") -> torch.device:
    if isinstance(device, torch.device):
        return device
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


class GPUPipeline:
    """band-pass -> covariance -> tangent space -> LDA for one decode window per chunk.

    ``overlap=True`` (CUDA only): two streams, pinned buffers, async copies.
    ``overlap=False``: **benchmark baseline only** — one stream, pageable staging buffers,
    blocking copies. It exists to measure what the overlap design buys.
    On a CPU device the same code runs without streams (used for tests/dashboards).
    """

    def __init__(self, params: RiemannianLDAParams, device: str | torch.device = "auto",
                 overlap: bool = True, n_slots: int = 3, compile: bool | str = "auto",
                 window: int = WINDOW_SAMPLES, n_channels: int = N_CHANNELS,
                 profile_stages: bool = True) -> None:
        configure_precision()
        self.device = resolve_device(device)
        self.on_cuda = self.device.type == "cuda"
        self.overlap = bool(overlap) and self.on_cuda
        self.window, self.n_channels, self.n_slots = window, n_channels, n_slots
        self.profile_stages = profile_stages
        self._async = self.overlap

        self.model = RiemannianLDA(params, self.device).requires_grad_(False)
        self.filter, self.filter_info = make_bandpass(
            params.taps, self.device, compile, example_shape=(1, window, n_channels))
        n_cls = params.lda_w.shape[0]

        pin = self.overlap
        self._host_in = [torch.empty((window, n_channels), dtype=torch.float32, pin_memory=pin)
                         for _ in range(n_slots)]
        self._host_scores = [torch.empty(n_cls, dtype=torch.float64, pin_memory=pin)
                             for _ in range(n_slots)]
        self._dev_in = [torch.empty((window, n_channels), dtype=torch.float32, device=self.device)
                        for _ in range(n_slots)]
        self._dev_scores = [torch.empty(n_cls, dtype=torch.float64, device=self.device)
                            for _ in range(n_slots)]
        self._host_in_np = [t.numpy() for t in self._host_in]        # views, for ring.latest(out=)
        self._host_scores_np = [t.numpy() for t in self._host_scores]

        if self.on_cuda:
            if self.overlap:
                self._copy_stream = torch.cuda.Stream(self.device)
                self._compute_stream = torch.cuda.Stream(self.device)
            else:
                self._copy_stream = self._compute_stream = torch.cuda.current_stream(self.device)
            self._events = [{m: torch.cuda.Event(enable_timing=True) for m in _MARKS}
                            for _ in range(n_slots)]
        else:
            self._copy_stream = self._compute_stream = None
            self._cpu_marks = [dict.fromkeys(_MARKS, 0) for _ in range(n_slots)]

        self._free = collections.deque(range(n_slots))
        self._meta: list[tuple[int, int]] = [(0, 0)] * n_slots

    # ------------------------------------------------------------------ helpers
    def _ctx(self, stream):
        return torch.cuda.stream(stream) if self.on_cuda else contextlib.nullcontext()

    def _mark(self, slot: int, name: str) -> None:
        if self.on_cuda:
            self._events[slot][name].record()
        else:
            self._cpu_marks[slot][name] = time.perf_counter_ns()

    def _span_ms(self, slot: int, a: str, b: str) -> float:
        if self.on_cuda:
            return self._events[slot][a].elapsed_time(self._events[slot][b])
        return (self._cpu_marks[slot][b] - self._cpu_marks[slot][a]) / 1e6

    @property
    def description(self) -> str:
        if not self.on_cuda:
            return "no CUDA streams"
        return "cuda: 2 streams, pinned, async" if self.overlap else "cuda: 1 stream, pageable, blocking (baseline)"

    # ------------------------------------------------------------------ hot path
    def stage(self, source: RingBuffer | np.ndarray, index: int, t_recv_ns: int) -> int:
        """Copy the newest window into a pinned slot and enqueue its async H2D copy."""
        if not self._free:
            raise RuntimeError("all pipeline slots are in flight; collect() one first")
        slot = self._free.popleft()
        if self.overlap:
            self._events[slot]["h2d_end"].synchronize()      # previous H2D finished reading this pinned slot
        if isinstance(source, RingBuffer):
            source.latest(self.window, out=self._host_in_np[slot])
        else:
            np.copyto(self._host_in_np[slot], source)
        with self._ctx(self._copy_stream):
            if self.overlap:
                self._copy_stream.wait_event(self._events[slot]["cls_end"])   # device buffer no longer read
            self._mark(slot, "h2d_start")
            self._dev_in[slot].copy_(self._host_in[slot], non_blocking=self._async)
            self._mark(slot, "h2d_end")
        self._meta[slot] = (index, t_recv_ns)
        return slot

    def launch(self, slot: int) -> None:
        """Enqueue filter -> covariance -> tangent -> LDA on the compute stream, then D2H."""
        with self._ctx(self._compute_stream), torch.no_grad():
            if self.overlap:
                self._compute_stream.wait_event(self._events[slot]["h2d_end"])
            self._mark(slot, "c_start")
            y = self.filter(self._dev_in[slot].unsqueeze(0))
            self._mark(slot, "filter_end")
            cov = self.model.covariance(y)
            self._mark(slot, "cov_end")
            scores = self.model.head(self.model.tangent(cov))
            self._dev_scores[slot].copy_(scores[0])
            self._mark(slot, "cls_end")
        with self._ctx(self._copy_stream):
            if self.overlap:
                self._copy_stream.wait_event(self._events[slot]["cls_end"])
            self._host_scores[slot].copy_(self._dev_scores[slot], non_blocking=self._async)
            self._mark(slot, "d2h_end")

    def collect(self, slot: int) -> Result:
        """Wait for the slot's output, build the :class:`Result`, free the slot."""
        index, t_recv_ns = self._meta[slot]
        if self.on_cuda:
            self._events[slot]["d2h_end"].synchronize()
        t_ready_ns = time.perf_counter_ns()
        scores = self._host_scores_np[slot].copy()
        stage_ms = ({k: self._span_ms(slot, a, b) for k, (a, b) in _STAGE_SPANS.items()}
                    if self.profile_stages else {})
        self._free.append(slot)
        return Result(index, int(np.argmax(scores)), scores,
                      (t_ready_ns - t_recv_ns) / 1e6, stage_ms, t_ready_ns)

    def process(self, source: RingBuffer | np.ndarray, index: int, t_recv_ns: int) -> Result:
        """stage + launch + collect, no look-ahead (the real-time, one-chunk-at-a-time path)."""
        slot = self.stage(source, index, t_recv_ns)
        self.launch(slot)
        return self.collect(slot)

    def warmup(self, n: int = 5) -> None:
        """Run a few throw-away windows (cuBLAS/cuSOLVER/cuDNN init, allocator warm-up)."""
        window = np.random.default_rng(0).standard_normal((self.window, self.n_channels)).astype(np.float32)
        for _ in range(n):
            self.process(window, -1, time.perf_counter_ns())


def drive(pipe: GPUPipeline, ring: RingBuffer, chunks: Iterable[Chunk],
          lookahead: bool = True) -> Iterator[Result]:
    """Feed chunks through ring buffer + pipeline, yielding a :class:`Result` per decoded chunk.

    ``lookahead=True`` stages chunk N (pinned copy + async H2D) *before* launching chunk N-1,
    which is what lets the H2D copy overlap the previous chunk's GPU work.
    """
    pending: int | None = None
    for chunk in chunks:
        t_recv_ns = time.perf_counter_ns()                 # chunk handed to the consumer
        ring.write(chunk.data)
        n = chunk.data.shape[0]
        if ring.samples_written < pipe.window:
            ring.consume(n)                                # window still filling
            continue
        if lookahead:
            slot = pipe.stage(ring, chunk.index, t_recv_ns)
            ring.consume(n)
            if pending is not None:
                pipe.launch(pending)
                yield pipe.collect(pending)
            pending = slot
        else:
            result = pipe.process(ring, chunk.index, t_recv_ns)
            ring.consume(n)
            yield result
    if pending is not None:
        pipe.launch(pending)
        yield pipe.collect(pending)


# ============================================================================ live session
class LatencyTracker:
    """Thread-safe rolling latency window for the dashboard gauge."""

    def __init__(self, maxlen: int = 500) -> None:
        self._d: collections.deque[float] = collections.deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def add(self, ms: float) -> None:
        with self._lock:
            self._d.append(ms)

    def percentiles(self) -> dict[str, float]:
        with self._lock:
            a = np.fromiter(self._d, dtype=np.float64)
        if a.size == 0:
            return {"n": 0, "p50": float("nan"), "p90": float("nan"), "p99": float("nan")}
        p50, p90, p99 = np.percentile(a, [50, 90, 99])
        return {"n": int(a.size), "p50": float(p50), "p90": float(p90), "p99": float(p99)}


class StreamingSession:
    """stream -> ring buffer -> pipeline, one chunk per :meth:`step` (paced at 1x by default)."""

    def __init__(self, params: RiemannianLDAParams | None = None, device: str = "auto",
                 realtime: bool = True, seed: int = 0, overlap: bool = True,
                 compile: bool | str = "auto") -> None:
        self.params = params if params is not None else train_default()
        self.stream = SyntheticEEGStream(seed=seed, realtime=realtime)
        self.ring = RingBuffer()
        self.pipeline = GPUPipeline(self.params, device, overlap=overlap, compile=compile)
        self.pipeline.warmup()

    def step(self) -> Frame:
        chunk = next(self.stream)
        t_recv_ns = time.perf_counter_ns()
        self.ring.write(chunk.data)
        n = chunk.data.shape[0]
        if self.ring.samples_written < self.pipeline.window:
            self.ring.consume(n)
            return Frame(chunk, None)
        result = self.pipeline.process(self.ring, chunk.index, t_recv_ns)
        self.ring.consume(n)
        return Frame(chunk, result)


class LiveEngine:
    """Runs a :class:`StreamingSession` on a background thread; the UI polls :meth:`snapshot`."""

    def __init__(self, params: RiemannianLDAParams | None = None, device: str = "auto",
                 seed: int = 0, history_s: float = 5.0, latency_window: int = 500,
                 compile: bool | str = "auto") -> None:
        self.session = StreamingSession(params, device, realtime=True, seed=seed, compile=compile)
        self.tracker = LatencyTracker(latency_window)
        self._lock = threading.Lock()
        self._raw: collections.deque[np.ndarray] = collections.deque(
            maxlen=max(1, int(history_s * FS_HZ / CHUNK_SAMPLES)))
        self._latest: Frame | None = None
        self._frames = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.error: BaseException | None = None

    def start(self) -> "LiveEngine":
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="bci-engine", daemon=True)
            self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                frame = self.session.step()
                if frame.result is not None:
                    self.tracker.add(frame.result.latency_ms)
                with self._lock:
                    self._raw.append(frame.chunk.data)
                    self._latest = frame
                    self._frames += 1
        except BaseException as exc:  # noqa: BLE001 - surfaced to the UI via snapshot()
            self.error = exc
            log.exception("live engine stopped")

    def snapshot(self) -> dict:
        with self._lock:
            raw = np.concatenate(list(self._raw)) if self._raw else np.zeros((0, N_CHANNELS), np.float32)
            frame, frames = self._latest, self._frames
        res = frame.result if frame else None
        pipe = self.session.pipeline
        return {
            "raw": raw,
            "frames": frames,
            "true_label": frame.chunk.label if frame else None,
            "label": res.label if res else None,
            "scores": res.scores if res else None,
            "latency_ms": res.latency_ms if res else None,
            "stage_ms": res.stage_ms if res else {},
            "percentiles": self.tracker.percentiles(),
            "warming_up": res is None,
            "starting": False,
            "running": bool(self._thread and self._thread.is_alive()),
            "error": None if self.error is None else f"{type(self.error).__name__}: {self.error}",
            "device": device_label(pipe.device),
            "pipeline": pipe.description,
            "filter": pipe.filter_info.note,
            "ring": self.session.ring.stats(),
        }


def _empty_snapshot(error: str | None = None, running: bool = True) -> dict:
    return {"raw": np.zeros((0, N_CHANNELS), np.float32), "frames": 0, "true_label": None, "label": None,
            "scores": None, "latency_ms": None, "stage_ms": {}, "warming_up": True, "starting": True,
            "percentiles": {"n": 0, "p50": float("nan"), "p90": float("nan"), "p99": float("nan")},
            "running": running, "error": error, "device": "starting...", "pipeline": "", "filter": "",
            "ring": {"dropped_samples": 0, "wrap_count": 0}}


def _publish_loop(conn, engine: LiveEngine, publish_s: float) -> None:
    """Push a snapshot to the parent every ``publish_s`` until it says stop or disconnects."""
    while True:
        if conn.poll(publish_s) and conn.recv() == "stop":
            return
        snap = engine.snapshot()
        snap["starting"] = False
        conn.send(("snap", snap))


def _worker_main(args: argparse.Namespace) -> int:
    """Entry point of the engine child process (``python -m bci.gpu_pipeline engine-worker``)."""
    import os
    from multiprocessing.connection import Client
    host, port = args.connect.rsplit(":", 1)
    conn = Client((host, int(port)), authkey=bytes.fromhex(os.environ["BCI_ENGINE_AUTHKEY"]))
    engine = None
    try:
        compile_arg: bool | str = args.compile if args.compile == "auto" else args.compile == "true"
        engine = LiveEngine(device=args.device, seed=args.seed, compile=compile_arg).start()
        _publish_loop(conn, engine, args.publish_s)
    except (EOFError, OSError):
        pass                                          # parent went away
    except BaseException as exc:  # noqa: BLE001 - report to the parent, then exit
        try:
            conn.send(("error", f"{type(exc).__name__}: {exc}"))
        except OSError:
            pass
        return 1
    finally:
        if engine is not None:
            engine.stop()
    return 0


class ProcessEngine:
    """A :class:`LiveEngine` in its own *process*, with the same ``snapshot()`` interface.

    Why a process: a UI that builds figures in the same process holds the GIL for tens of
    milliseconds at a time and stalls the pipeline thread (measured on CPU: p99 9 ms alone vs
    174 ms with a figure-building thread in the same process; see FINDINGS.md). A separate
    process keeps the UI from inflating the latency being displayed. The child is started with
    ``subprocess`` (not ``multiprocessing.spawn``, which would re-import Streamlit's script).
    """

    def __init__(self, publish_s: float = 0.25, device: str = "auto", seed: int = 0,
                 compile: bool | str = "auto") -> None:
        self._argv_tail = ["--publish-s", str(publish_s), "--device", device, "--seed", str(seed),
                           "--compile", compile if isinstance(compile, str) else str(compile).lower()]
        self._latest = _empty_snapshot()
        self._lock = threading.Lock()
        self._proc = None
        self._conn = None
        self._reader: threading.Thread | None = None

    def start(self) -> "ProcessEngine":
        import os
        import subprocess
        from multiprocessing.connection import Listener
        if self._reader is not None:
            return self
        authkey = os.urandom(16)
        self._listener = Listener(("127.0.0.1", 0), authkey=authkey)
        host, port = self._listener.address
        pkg_root = str(Path(__file__).resolve().parents[1])         # directory that contains the `bci` package
        env = {**os.environ, "BCI_ENGINE_AUTHKEY": authkey.hex(),
               "PYTHONPATH": os.pathsep.join(filter(None, [pkg_root, os.environ.get("PYTHONPATH")]))}
        self._proc = subprocess.Popen(
            [sys.executable, "-m", "bci.gpu_pipeline", "engine-worker", "--connect", f"{host}:{port}",
             *self._argv_tail], env=env)
        self._reader = threading.Thread(target=self._read_loop, name="bci-engine-reader", daemon=True)
        self._reader.start()
        return self

    def _set(self, snap: dict) -> None:
        with self._lock:
            self._latest = snap

    def _read_loop(self) -> None:
        accepted: dict = {}
        acceptor = threading.Thread(target=lambda: accepted.setdefault("conn", self._listener.accept()),
                                    daemon=True)
        acceptor.start()
        while acceptor.is_alive():
            if self._proc.poll() is not None:
                self._set(_empty_snapshot(error=f"engine process exited (code {self._proc.returncode}) "
                                                "before connecting", running=False))
                return
            acceptor.join(0.1)
        self._conn = accepted["conn"]
        while True:
            try:
                kind, payload = self._conn.recv()
            except (EOFError, OSError):
                break
            self._set(payload if kind == "snap" else _empty_snapshot(error=payload, running=False))
        with self._lock:
            if self._latest.get("error") is None:
                self._latest = {**self._latest, "running": False, "error": "engine process exited"}

    def snapshot(self) -> dict:
        with self._lock:
            return self._latest

    def stop(self) -> None:
        try:
            if self._conn is not None:
                self._conn.send("stop")
        except (OSError, ValueError):
            pass
        if self._proc is not None:
            try:
                self._proc.wait(timeout=10)
            except Exception:  # noqa: BLE001
                self._proc.terminate()


# ============================================================================ benchmark
def device_label(device: torch.device) -> str:
    return torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"


def environment_info(device: torch.device) -> dict:
    info = {"device": device.type, "device_name": device_label(device),
            "torch": torch.__version__, "cuda": torch.version.cuda or "none",
            "python": platform.python_version(), "platform": platform.platform()}
    return info


@dataclass
class BenchmarkRun:
    name: str                        # overlap | baseline | paced
    mode: str                        # saturated | paced
    pipeline: str
    filter_note: str
    filter_compiled: bool
    warmup: int
    e2e_ms: np.ndarray
    stage_ms: dict[str, np.ndarray]
    wall_ms_per_chunk: float
    ring_stats: dict
    env: dict = field(default_factory=dict)

    @property
    def device_type(self) -> str:
        return self.env.get("device", "?")

    def percentiles(self, a: np.ndarray) -> dict[str, float]:
        p50, p90, p99 = np.percentile(a, [50, 90, 99])
        return {"n": int(a.size), "mean": float(a.mean()), "p50": float(p50), "p90": float(p90),
                "p99": float(p99), "max": float(a.max())}

    def p99(self) -> float:
        return self.percentiles(self.e2e_ms)["p99"]


RUN_SPECS = {                         # name -> (mode, overlap)
    "overlap": ("saturated", True),
    "baseline": ("saturated", False),
    "paced": ("paced", True),
}


def run_benchmark(params: RiemannianLDAParams, name: str = "overlap", *, n_chunks: int = 1000,
                  warmup: int = 50, device: str = "auto", compile: bool | str = "auto",
                  seed: int = 0) -> BenchmarkRun:
    """Measure ``n_chunks`` decoded chunks after discarding ``warmup`` warm-up chunks."""
    mode, overlap = RUN_SPECS[name]
    pipe = GPUPipeline(params, device, overlap=overlap, compile=compile)
    ring = RingBuffer()
    fill = math.ceil(WINDOW_SAMPLES / CHUNK_SAMPLES) - 1          # chunks before the first full window
    total = n_chunks + warmup
    if mode == "paced":
        chunks: Iterable[Chunk] = SyntheticEEGStream(seed=seed, n_chunks=total + fill, realtime=True)
    else:                                                          # pre-generate: synthesis is not pipeline cost
        chunks = list(SyntheticEEGStream(seed=seed, n_chunks=total + fill))
    results = list(drive(pipe, ring, chunks, lookahead=(mode == "saturated" and pipe.overlap)))
    if len(results) != total:
        raise RuntimeError(f"expected {total} results, got {len(results)}")
    meas = results[warmup:]
    wall = (meas[-1].t_ready_ns - meas[0].t_ready_ns) / max(len(meas) - 1, 1) / 1e6
    return BenchmarkRun(
        name=name, mode=mode, pipeline=pipe.description, filter_note=pipe.filter_info.note,
        filter_compiled=pipe.filter_info.compiled, warmup=warmup,
        e2e_ms=np.array([r.latency_ms for r in meas]),
        stage_ms={k: np.array([r.stage_ms[k] for r in meas]) for k in STAGES},
        wall_ms_per_chunk=wall, ring_stats=ring.stats(), env=environment_info(pipe.device))


def write_chunk_csv(run: BenchmarkRun, path: Path) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["chunk", "e2e_ms", *[f"{s}_ms" for s in STAGES]])
        for i in range(run.e2e_ms.size):
            w.writerow([i, f"{run.e2e_ms[i]:.6f}", *[f"{run.stage_ms[s][i]:.6f}" for s in STAGES]])


def write_summary_csv(runs: list[BenchmarkRun], path: Path) -> None:
    cols = ["run", "metric", "n", "mean_ms", "p50_ms", "p90_ms", "p99_ms", "max_ms", "mode", "pipeline",
            "filter_compiled", "warmup_discarded", "wall_ms_per_chunk", "device", "device_name",
            "torch", "cuda", "python"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for run in runs:
            series = {"e2e": run.e2e_ms, **run.stage_ms}
            for metric, a in series.items():
                p = run.percentiles(a)
                w.writerow([run.name, metric, p["n"], *[f"{p[k]:.6f}" for k in ("mean", "p50", "p90", "p99", "max")],
                            run.mode, run.pipeline, run.filter_compiled, run.warmup,
                            f"{run.wall_ms_per_chunk:.6f}", run.env["device"], run.env["device_name"],
                            run.env["torch"], run.env["cuda"], run.env["python"]])


# chart tokens (light surface) from the validated reference palette
_SURFACE, _INK, _INK2, _GRID, _SERIES1 = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6"


def plot_histogram(run: BenchmarkRun, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    a = run.e2e_ms
    p = run.percentiles(a)
    hi = max(a.max() * 1.15, p["p99"] * 1.3)
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150, facecolor=_SURFACE)
    ax.set_facecolor(_SURFACE)
    ax.hist(a, bins=60, range=(0, hi), color=_SERIES1, edgecolor=_SURFACE, linewidth=0.6)
    ymax = ax.get_ylim()[1]
    for key, label, dy in (("p50", "p50", 0.93), ("p99", "p99", 0.80)):
        ax.axvline(p[key], color=_INK2, linewidth=1.2, linestyle=(0, (4, 3)))
        ax.text(p[key], ymax * dy, f" {label} {p[key]:.2f} ms", color=_INK, fontsize=9, va="top")
    if LATENCY_TARGET_MS <= hi:                        # off-scale target is named in the title instead
        ax.axvline(LATENCY_TARGET_MS, color=_INK, linewidth=1.4)
        ax.text(LATENCY_TARGET_MS, ymax * 0.67, f" target {LATENCY_TARGET_MS:.0f} ms", color=_INK, fontsize=9, va="top")
    ax.set_xlim(0, hi)
    ax.set_xlabel("end-to-end latency, chunk received -> label ready (ms)", color=_INK2, fontsize=9)
    ax.set_ylabel("chunks", color=_INK2, fontsize=9)
    ax.set_title(f"{run.env['device_name']} | {run.name} ({run.mode}) | n={p['n']} | p99 {p['p99']:.2f} ms "
                 f"(target < {LATENCY_TARGET_MS:.0f} ms)", color=_INK, fontsize=10, loc="left")
    ax.grid(axis="y", color=_GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(_GRID)
    ax.tick_params(colors=_INK2, labelsize=8)
    fig.tight_layout()
    fig.savefig(path, facecolor=_SURFACE)
    plt.close(fig)


def _format_table(runs: list[BenchmarkRun]) -> str:
    lines = [f"{'run':9s} {'metric':11s} {'p50':>9s} {'p90':>9s} {'p99':>9s} {'max':>9s}  (ms)"]
    for run in runs:
        for metric, a in {"e2e": run.e2e_ms, **run.stage_ms}.items():
            p = run.percentiles(a)
            lines.append(f"{run.name:9s} {metric:11s} {p['p50']:9.3f} {p['p90']:9.3f} {p['p99']:9.3f} {p['max']:9.3f}")
        lines.append(f"{run.name:9s} wall/chunk   {run.wall_ms_per_chunk:9.3f} ms   [{run.pipeline}; filter: {run.filter_note}]")
    return "\n".join(lines)


def cmd_benchmark(args: argparse.Namespace) -> int:
    device = resolve_device(args.device)
    if device.type != "cuda" and not args.allow_cpu:
        print("ERROR: no CUDA device. Refusing to write GPU benchmark artifacts from a CPU run.\n"
              "       (use --allow-cpu for a harness smoke test; outputs are prefixed 'cpu_smoke_')",
              file=sys.stderr)
        return 2
    prefix = "" if device.type == "cuda" else "cpu_smoke_"
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    params = train_default()
    names = [n.strip() for n in args.runs.split(",") if n.strip()]
    runs = []
    for n in names:
        print(f"[benchmark] run '{n}': {args.chunks} chunks + {args.warmup} warm-up ...", flush=True)
        run = run_benchmark(params, n, n_chunks=args.chunks, warmup=args.warmup, device=str(device),
                            compile=False if args.no_compile else "auto")
        runs.append(run)
        write_chunk_csv(run, out / (f"{prefix}benchmark_latency.csv" if n == names[0]
                                    else f"{prefix}benchmark_latency_{n}.csv"))
    write_summary_csv(runs, out / f"{prefix}benchmark_summary.csv")
    plot_histogram(runs[0], out / f"{prefix}latency_histogram.png")
    print(_format_table(runs))
    head = runs[0]
    ok = head.p99() < LATENCY_TARGET_MS
    print(f"\nheadline run '{head.name}' on {head.env['device_name']}: p99 = {head.p99():.3f} ms "
          f"-> {'PASS' if ok else 'FAIL'} (target < {LATENCY_TARGET_MS:.0f} ms)")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(prog="python -m bci.gpu_pipeline")
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("benchmark", help="measure end-to-end and per-stage latency")
    b.add_argument("--chunks", type=int, default=1000)
    b.add_argument("--warmup", type=int, default=50)
    b.add_argument("--runs", default="overlap,baseline", help="comma list of: overlap, baseline, paced")
    b.add_argument("--out-dir", default="results")
    b.add_argument("--device", default="auto")
    b.add_argument("--allow-cpu", action="store_true")
    b.add_argument("--no-compile", action="store_true")
    b.set_defaults(func=cmd_benchmark)
    w = sub.add_parser("engine-worker", help=argparse.SUPPRESS)       # used by ProcessEngine
    w.add_argument("--connect", required=True)
    w.add_argument("--publish-s", type=float, default=0.25)
    w.add_argument("--device", default="auto")
    w.add_argument("--seed", type=int, default=0)
    w.add_argument("--compile", default="auto", choices=["auto", "true", "false"])
    w.set_defaults(func=_worker_main)
    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
