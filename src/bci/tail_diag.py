"""Tail-latency diagnostic for the paced real-data run.

Why: in some runs about 1-2% of chunks take 3-4x longer in the covariance and classifier stages while the copy
and filter stages do not move (FINDINGS.md section 7.1). The cause was not isolated. This tool records, for every
chunk of the same paced run, what could tell the candidates apart:

* **host time inside each pipeline call** (``stage``, ``launch_pre``, ``launch_post``, ``collect``). ``launch_pre`` only
  enqueues kernels and never waits, so a long ``launch_pre`` means the Python thread itself was stalled (descheduled,
  waiting for the GIL, garbage collection); a normal ``launch_pre`` with slow GPU stage times points at the GPU;
* **Python garbage-collection pauses** that overlap the chunk (``gc.callbacks``);
* **GPU telemetry** from ``nvidia-smi`` at 1 Hz (SM clock, utilisation, temperature, power, throttle reasons): too
  coarse to hit a single chunk, but it shows clock drops and thermal/power throttling around the slow ones.

A/B switches test the explanations directly: ``--no-gc`` disables the collector during the run, ``--priority high``
raises the process priority (Windows ``SetPriorityClass``; elsewhere ``os.nice``), ``--no-telemetry`` removes the
``nvidia-smi`` polling (it spawns a process once a second and could itself disturb a laptop).

    python -m bci.tail_diag --chunks 1000 --warmup 50 --out-dir results/real/tail_diag_baseline
    python -m bci.tail_diag --chunks 1000 --warmup 50 --no-gc --out-dir results/real/tail_diag_nogc

The output is a per-chunk CSV and a text summary; the verdict lines are hints from simple comparisons, not proofs.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

SLOW_FACTOR = 2.0            # a chunk is "slow" when its end-to-end latency exceeds this many medians
HOST_CALLS = ("stage", "launch_pre", "launch_post", "collect")
GPU_STAGES = ("h2d_ms", "filter_ms", "cov_ms", "classifier_ms", "d2h_ms")
_QUERY = "clocks.sm,clocks.max.sm,utilization.gpu,temperature.gpu,power.draw,{throttle}"
_THROTTLE_FIELDS = ("clocks_throttle_reasons.active", "clocks_event_reasons.active")


# ----------------------------------------------------------------------------------- telemetry
class GpuTelemetry:
    """Polls ``nvidia-smi`` once per ``period_s`` in a thread. Silent no-op when it is not available."""

    def __init__(self, period_s: float = 1.0) -> None:
        self.period_s, self.samples, self._stop = period_s, [], threading.Event()
        self._thread: threading.Thread | None = None
        self._query = self._find_query()

    @staticmethod
    def _run(query: str) -> list[str] | None:
        try:
            out = subprocess.run(["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=5)
        except (OSError, subprocess.SubprocessError):
            return None
        if out.returncode != 0 or not out.stdout.strip():
            return None
        return [v.strip() for v in out.stdout.strip().splitlines()[0].split(",")]

    def _find_query(self) -> str | None:
        for field in _THROTTLE_FIELDS:
            q = _QUERY.format(throttle=field)
            if self._run(q) is not None:
                return q
        return None

    @property
    def available(self) -> bool:
        return self._query is not None

    def start(self) -> "GpuTelemetry":
        if self.available:
            self._thread = threading.Thread(target=self._loop, name="bci-gpu-telemetry", daemon=True)
            self._thread.start()
        return self

    def _loop(self) -> None:
        while not self._stop.wait(self.period_s):
            t = time.perf_counter_ns()
            vals = self._run(self._query)
            if vals is not None and len(vals) == 6:
                self.samples.append((t, vals))

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)


def _num(v: str) -> float:
    try:
        return float(v)
    except ValueError:
        return float("nan")


def nearest_sample(samples: list, t_ns: int):
    if not samples:
        return None, None
    i = int(np.argmin([abs(t - t_ns) for t, _ in samples]))
    return samples[i][1], abs(samples[i][0] - t_ns) / 1e9


# ----------------------------------------------------------------------------------- garbage collection
class GcLog:
    """Durations of Python garbage-collection passes, as (start_ns, stop_ns, generation)."""

    def __init__(self) -> None:
        self.events: list[tuple[int, int, int]] = []
        self._t0 = 0

    def _cb(self, phase: str, info: dict) -> None:
        if phase == "start":
            self._t0 = time.perf_counter_ns()
        else:
            self.events.append((self._t0, time.perf_counter_ns(), info.get("generation", -1)))

    def __enter__(self) -> "GcLog":
        gc.callbacks.append(self._cb)
        return self

    def __exit__(self, *exc) -> None:
        gc.callbacks.remove(self._cb)

    def overlap_ms(self, a_ns: int, b_ns: int) -> float:
        return sum(max(0, min(b_ns, e) - max(a_ns, s)) for s, e, _ in self.events) / 1e6


# ----------------------------------------------------------------------------------- analysis
def analyse(rows: list[dict]) -> tuple[dict, str]:
    """Compare slow chunks (end-to-end > SLOW_FACTOR x median) with normal ones. Returns (numbers, text)."""
    e2e = np.array([r["e2e_ms"] for r in rows])
    med = float(np.median(e2e))
    slow = e2e > SLOW_FACTOR * med
    cols = [c for c in (*GPU_STAGES, "host_stage_ms", "host_launch_pre_ms", "host_launch_post_ms", "host_collect_ms",
                        "gc_ms", "gpu_sm_mhz", "gpu_util_pct", "gpu_temp_c", "gpu_power_w") if c in rows[0]]
    res = {"n": len(rows), "median_e2e_ms": med, "n_slow": int(slow.sum()), "slow_threshold_ms": SLOW_FACTOR * med,
           "columns": {}}
    for c in cols:
        a = np.array([r[c] for r in rows], dtype=float)
        res["columns"][c] = {"median_normal": float(np.nanmedian(a[~slow])),
                             "median_slow": float(np.nanmedian(a[slow])) if slow.any() else float("nan"),
                             "max_slow": float(np.nanmax(a[slow])) if slow.any() else float("nan")}
    gc_hit = np.array([r["gc_ms"] > 0.2 for r in rows])
    res["slow_with_gc_pause"] = int((slow & gc_hit).sum())
    lines = [f"chunks {res['n']}, median e2e {med:.3f} ms, slow = e2e > {res['slow_threshold_ms']:.2f} ms: {res['n_slow']} chunks"]
    if not slow.any():
        lines.append("no slow chunks in this run: nothing to compare (the tail is intermittent; repeat the run)")
        return res, "\n".join(lines)
    lines.append(f"{'column':22s} {'median normal':>14s} {'median slow':>12s} {'ratio':>7s} {'max slow':>10s}")
    for c, d in res["columns"].items():
        ratio = d["median_slow"] / d["median_normal"] if d["median_normal"] else float("nan")
        lines.append(f"{c:22s} {d['median_normal']:14.3f} {d['median_slow']:12.3f} {ratio:7.2f} {d['max_slow']:10.3f}")
    lines.append(f"slow chunks overlapping a Python GC pause (> 0.2 ms): {res['slow_with_gc_pause']} of {res['n_slow']}")
    hints = []
    pre = res["columns"].get("host_launch_pre_ms")
    if pre and pre["median_slow"] > 3 * max(pre["median_normal"], 0.05):
        hints.append("launch_pre (enqueue only, never waits) is much slower on slow chunks: the Python thread was "
                     "stalled while enqueuing -> host-side cause (scheduling, GIL, GC), not GPU compute")
    elif pre:
        hints.append("launch_pre is normal on slow chunks while the GPU stages are slow -> the extra time is on the GPU "
                     "side (kernel execution/queueing), not in the enqueue")
    if res["slow_with_gc_pause"] >= max(2, 0.5 * res["n_slow"]):
        hints.append("most slow chunks overlap a Python GC pause -> try --no-gc")
    sm = res["columns"].get("gpu_sm_mhz")
    if sm and sm["median_normal"] and sm["median_slow"] < 0.85 * sm["median_normal"]:
        hints.append("SM clock is lower around slow chunks -> GPU clock/thermal/power throttling")
    if not hints:
        hints.append("no single signal stands out")
    lines += ["hints (simple comparisons, not proofs):"] + [f"  - {h}" for h in hints]
    return res, "\n".join(lines)


# ----------------------------------------------------------------------------------- run
def set_high_priority() -> str:
    try:
        if sys.platform == "win32":
            import ctypes
            ok = ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x80)  # HIGH
            return "HIGH_PRIORITY_CLASS set" if ok else "SetPriorityClass failed"
        os.nice(-5)
        return "nice -5 set"
    except Exception as exc:  # noqa: BLE001 - report, do not abort the measurement
        return f"priority not changed: {exc}"


def run(args: argparse.Namespace) -> dict:
    from . import real_benchmark as rb
    from .gpu_pipeline import GPUPipeline, device_label, drive, resolve_device
    from .ring_buffer import RingBuffer
    from .stream import RealEEGStream
    from . import CHUNK_SAMPLES, WINDOW_SAMPLES

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    dev = resolve_device(args.device)
    params = rb.train_subject(args.subject, decoder=args.decoder)
    pipe = GPUPipeline(params, dev, n_channels=22, compile=args.compile)
    pipe.warmup()
    last: dict[str, float] = {}

    def timed(name):
        orig = getattr(pipe, name)

        def wrapper(*a, **k):
            t0 = time.perf_counter_ns()
            try:
                return orig(*a, **k)
            finally:
                last[name] = (time.perf_counter_ns() - t0) / 1e6
        setattr(pipe, name, wrapper)

    for name in HOST_CALLS:
        timed(name)
    notes = []
    if args.priority == "high":
        notes.append(set_high_priority())
    gc.collect()
    if args.no_gc:
        gc.disable()
        notes.append("garbage collector disabled during the run")
    tele = GpuTelemetry() if (not args.no_telemetry and dev.type == "cuda") else None
    if tele is not None:
        notes.append("nvidia-smi telemetry on" if tele.available else "nvidia-smi not available: no telemetry")
        tele.start()
    fill = -(-WINDOW_SAMPLES // CHUNK_SAMPLES) - 1
    n_total = args.chunks + args.warmup + fill
    rows = []
    try:
        with GcLog() as gclog:
            stream = RealEEGStream(args.subject, "1test", realtime=not args.unpaced, n_chunks=n_total)
            for res in drive(pipe, RingBuffer(n_channels=22), stream, lookahead=False):
                t_ready = res.t_ready_ns
                row = {"chunk": res.index, "e2e_ms": res.latency_ms, **{f"{k}_ms" if not k.endswith("_ms") else k: v
                                                                          for k, v in res.stage_ms.items()},
                       "host_stage_ms": last.get("stage", float("nan")),
                       "host_launch_pre_ms": last.get("launch_pre", float("nan")),
                       "host_launch_post_ms": last.get("launch_post", float("nan")),
                       "host_collect_ms": last.get("collect", float("nan")),
                       "gc_ms": gclog.overlap_ms(t_ready - int(res.latency_ms * 1e6), t_ready)}
                if tele is not None and tele.samples:
                    vals, age = nearest_sample(tele.samples, t_ready)
                    row.update(gpu_sm_mhz=_num(vals[0]), gpu_util_pct=_num(vals[2]), gpu_temp_c=_num(vals[3]),
                               gpu_power_w=_num(vals[4]), gpu_throttle=vals[5], gpu_sample_age_s=age)
                rows.append(row)
    finally:
        if tele is not None:
            tele.stop()
        if args.no_gc:
            gc.enable()
    meas = rows[args.warmup:]
    if len(meas) != args.chunks:
        raise RuntimeError(f"expected {args.chunks} measured chunks, got {len(meas)}")
    prefix = "" if dev.type == "cuda" else "cpu_smoke_"
    keys = list(dict.fromkeys(k for r in meas for k in r))
    with open(out / f"{prefix}tail_diag_chunks.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in meas:
            w.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in r.items()})
    res, text = analyse([{k: (float(v) if k != "gpu_throttle" else 0.0) for k, v in r.items()} for r in meas])
    header = f"device: {device_label(dev)} | {pipe.description} | filter: {pipe.filter_info.note}\nflags: " + \
             json.dumps({"decoder": args.decoder, "no_gc": args.no_gc, "priority": args.priority, "telemetry": tele is not None, "paced": not args.unpaced,
                         "notes": notes})
    (out / f"{prefix}tail_diag_summary.txt").write_text(header + "\n" + text + "\n")
    print(header + "\n" + text)
    return res


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m bci.tail_diag", description=__doc__.split("\n\n")[0])
    p.add_argument("--subject", type=int, default=1)
    p.add_argument("--device", default="auto")
    p.add_argument("--decoder", choices=("riemann", "fbcsp"), default="riemann",
                   help="riemann reproduces the runs in which the tail was seen; fbcsp has no eigh (compare the tails)")
    p.add_argument("--chunks", type=int, default=1000)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--compile", default="auto", help="auto | true | false (the torch.compile'd FIR filter)")
    p.add_argument("--no-gc", action="store_true", help="disable the Python garbage collector during the run")
    p.add_argument("--priority", choices=("normal", "high"), default="normal")
    p.add_argument("--no-telemetry", action="store_true", help="do not poll nvidia-smi")
    p.add_argument("--unpaced", action="store_true", help="do not pace chunks at 1x real time (tests only)")
    p.add_argument("--out-dir", default="results/real/tail_diag")
    args = p.parse_args(argv)
    args.compile = {"true": True, "false": False}.get(str(args.compile).lower(), args.compile)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
