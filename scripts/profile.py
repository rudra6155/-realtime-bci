#!/usr/bin/env python
"""Profile the overlapped GPU pipeline and show H2D copies running concurrently with kernels.

    python scripts/profile.py [--chunks 40] [--warmup 30] [--out-dir results]

Needs a CUDA GPU. Writes:
  results/profiler_trace.json   Chrome trace from torch.profiler (open in https://ui.perfetto.dev
                                or chrome://tracing: one row per CUDA stream)
  results/profiler_overlap.png  timeline of a few consecutive chunks, one row per stream
  results/profiler_overlap.csv  numbers: how much H2D copy time ran concurrently with kernels

The overlap numbers are computed from the exported trace (GPU events: category kernel /
gpu_memcpy, with their `stream` id). If torch/Kineto changes those names the script stops with
an error instead of reporting zeros.
"""

from __future__ import annotations

import sys
from pathlib import Path

# This file is called profile.py: keep its directory off sys.path so it cannot shadow the
# stdlib `profile` module (cProfile imports it).
_HERE = Path(__file__).resolve().parent
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != _HERE]

import argparse
import csv
import json
import math

import numpy as np

GPU_CATS = {"kernel", "gpu_memcpy", "gpu_memset"}
_SURFACE, _INK, _INK2, _GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
_COLORS = {"kernel": "#2a78d6", "h2d": "#eb6834", "d2h": "#1baf7a"}      # validated categorical slots 1-3
_LABELS = {"kernel": "kernel (compute)", "h2d": "memcpy host->device", "d2h": "memcpy device->host"}


# ------------------------------------------------------------------ pure functions (unit-tested)
def classify(name: str, cat: str) -> str:
    if cat == "kernel":
        return "kernel"
    if cat == "gpu_memcpy":
        if "HtoD" in name:
            return "h2d"
        if "DtoH" in name:
            return "d2h"
    return "other"


def load_gpu_events(trace_path: Path) -> list[dict]:
    data = json.loads(Path(trace_path).read_text())
    events = []
    for e in data.get("traceEvents", []):
        if e.get("ph") == "X" and e.get("cat") in GPU_CATS and "dur" in e:
            kind = classify(e.get("name", ""), e["cat"])
            if kind != "other":
                events.append({"kind": kind, "name": e["name"], "start": float(e["ts"]),
                               "end": float(e["ts"]) + float(e["dur"]),
                               "stream": (e.get("args") or {}).get("stream", -1)})
    return sorted(events, key=lambda ev: ev["start"])


def merge(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for s, e in sorted(intervals):
        if out and s <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((s, e))
    return out


def intersection_length(a: list[tuple[float, float]], b: list[tuple[float, float]]) -> float:
    """Total length of (union of a) intersected with (union of b)."""
    a, b = merge(a), merge(b)
    i = j = 0
    total = 0.0
    while i < len(a) and j < len(b):
        lo, hi = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        total += max(0.0, hi - lo)
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return total


def analyse(events: list[dict]) -> dict:
    iv = {k: [(e["start"], e["end"]) for e in events if e["kind"] == k] for k in ("kernel", "h2d", "d2h")}
    out: dict = {"n_kernels": len(iv["kernel"]), "n_h2d": len(iv["h2d"]), "n_d2h": len(iv["d2h"]),
                 "streams_with_kernels": sorted({e["stream"] for e in events if e["kind"] == "kernel"}),
                 "streams_with_memcpy": sorted({e["stream"] for e in events if e["kind"] in ("h2d", "d2h")})}
    for k in ("h2d", "d2h"):
        total = sum(e - s for s, e in iv[k])
        over = intersection_length(iv[k], iv["kernel"])
        out[f"{k}_total_us"] = total
        out[f"{k}_overlapped_with_kernels_us"] = over
        out[f"{k}_overlap_fraction"] = over / total if total else float("nan")
    out["kernel_total_us"] = sum(e - s for s, e in iv["kernel"])
    return out


# ------------------------------------------------------------------ plotting
def plot_timeline(events: list[dict], summary: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    h2d = [e for e in events if e["kind"] == "h2d"]
    # window: three consecutive H2D copies, starting at one that overlaps a kernel if there is one
    kern = [(e["start"], e["end"]) for e in events if e["kind"] == "kernel"]
    start_i = len(h2d) // 2
    for i in range(len(h2d) // 2, len(h2d) - 3):
        if intersection_length([(h2d[i]["start"], h2d[i]["end"])], kern) > 0:
            start_i = i
            break
    start_i = max(0, min(start_i, len(h2d) - 3))
    span = (h2d[min(start_i + 1, len(h2d) - 1)]["start"] - h2d[start_i]["start"]) or 1000.0
    t0, t1 = h2d[start_i]["start"] - 0.3 * span, h2d[min(start_i + 2, len(h2d) - 1)]["start"] + 0.7 * span
    view = [e for e in events if e["end"] >= t0 and e["start"] <= t1]
    has_kernel = {s: any(e["kind"] == "kernel" for e in view if e["stream"] == s) for s in {e["stream"] for e in view}}
    streams = sorted(has_kernel, key=lambda s: (has_kernel[s], s))            # copy stream(s) first, then compute

    fig, ax = plt.subplots(figsize=(10, 1.1 * len(streams) + 2.2), dpi=150, facecolor=_SURFACE)
    ax.set_facecolor(_SURFACE)
    for row, s in enumerate(streams):
        for e in (x for x in view if x["stream"] == s):
            ax.broken_barh([(e["start"] - t0, e["end"] - e["start"])], (row - 0.3, 0.6), facecolors=_COLORS[e["kind"]],
                           edgecolor=_SURFACE, linewidth=0.8)
    k_union = merge(kern)
    for h in (e for e in view if e["kind"] == "h2d"):                          # shade where a copy runs during a kernel
        for ks, ke in k_union:
            lo, hi = max(h["start"], ks), min(h["end"], ke)
            if hi > lo:
                ax.axvspan(lo - t0, hi - t0, color=_INK, alpha=0.12, linewidth=0)
    labels = [f"stream {s}\n" + ("compute" if has_kernel[s] else "copy") for s in streams]
    ax.set_yticks(range(len(streams)), labels, color=_INK2, fontsize=8)
    ax.set_ylim(-0.7, len(streams) - 0.3)
    ax.invert_yaxis()
    ax.set_xlim(0, t1 - t0)
    ax.set_xlabel("time (µs)", color=_INK2, fontsize=9)
    frac = summary["h2d_overlap_fraction"]
    ax.set_title(f"CUDA streams, consecutive chunks. H2D copy time overlapped with kernels over the whole profile: "
                 f"{summary['h2d_overlapped_with_kernels_us']:.0f} of {summary['h2d_total_us']:.0f} µs "
                 f"({100 * frac:.0f}%)" if not math.isnan(frac) else "CUDA streams", color=_INK, fontsize=9, loc="left")
    ax.grid(axis="x", color=_GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(_GRID)
    ax.tick_params(colors=_INK2, labelsize=8)
    ax.legend(handles=[Patch(facecolor=_COLORS[k], label=_LABELS[k]) for k in ("h2d", "kernel", "d2h")],
              loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=3, frameon=False, fontsize=8, labelcolor=_INK2)
    fig.tight_layout()
    fig.savefig(path, facecolor=_SURFACE)
    plt.close(fig)


# ------------------------------------------------------------------ main
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--chunks", type=int, default=40)
    p.add_argument("--warmup", type=int, default=30)
    p.add_argument("--out-dir", default="results")
    args = p.parse_args(argv)

    import torch
    from torch.profiler import ProfilerActivity, profile

    from bci import CHUNK_SAMPLES, WINDOW_SAMPLES
    from bci.classifier import train_default
    from bci.gpu_pipeline import GPUPipeline, drive
    from bci.ring_buffer import RingBuffer
    from bci.stream import SyntheticEEGStream

    if not torch.cuda.is_available():
        print("ERROR: profile.py needs a CUDA GPU (torch.cuda.is_available() is False).", file=sys.stderr)
        return 2
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    pipe = GPUPipeline(train_default(), "cuda", overlap=True)
    ring = RingBuffer()
    fill = math.ceil(WINDOW_SAMPLES / CHUNK_SAMPLES) - 1
    chunks = list(SyntheticEEGStream(seed=0, n_chunks=fill + args.warmup + args.chunks))
    list(drive(pipe, ring, chunks[:fill + args.warmup]))                    # warm-up, not profiled
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        results = list(drive(pipe, ring, chunks[fill + args.warmup:]))
        torch.cuda.synchronize()
    trace = out / "profiler_trace.json"
    prof.export_chrome_trace(str(trace))

    events = load_gpu_events(trace)
    if not any(e["kind"] == "kernel" for e in events) or not any(e["kind"] == "h2d" for e in events):
        print(f"ERROR: no GPU kernel / HtoD memcpy events found in {trace}; the trace schema may have changed "
              "(expected cat in {kernel, gpu_memcpy}, name containing 'HtoD', args.stream).", file=sys.stderr)
        return 3
    summary = analyse(events)
    summary.update(chunks_profiled=len(results), device=torch.cuda.get_device_name(0), torch=torch.__version__)
    with open(out / "profiler_overlap.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        for k, v in summary.items():
            w.writerow([k, json.dumps(v) if isinstance(v, list) else v])
    plot_timeline(events, summary, out / "profiler_overlap.png")
    for k, v in summary.items():
        print(f"{k:34s} {v}")
    print(f"\nwrote {trace}, {out / 'profiler_overlap.csv'}, {out / 'profiler_overlap.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
