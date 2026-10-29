#!/usr/bin/env python
"""Why the dashboard runs the pipeline in a separate process (see FINDINGS.md).

Measures paced (1x real-time) pipeline latency twice on the same machine:
  A) pipeline thread alone,
  B) pipeline thread + a thread that, every 0.5 s, builds and JSON-serializes a 64-trace Plotly
     figure (what a Streamlit refresh does) in the SAME process.
The figure thread holds the GIL for tens of milliseconds at a time and stalls the pipeline thread.

    python scripts/ui_contention.py [--device auto] [--seconds 15]
"""

from __future__ import annotations

import argparse
import threading
import time

import numpy as np
import plotly.graph_objects as go

from bci.gpu_pipeline import LiveEngine


def build_and_serialize(raw: np.ndarray) -> int:
    fig = go.Figure()
    y = raw[::2].T.astype(np.float64)
    for ch in range(y.shape[0]):
        fig.add_trace(go.Scatter(x0=0, dx=0.008, y=np.round(y[ch] + ch * 60.0, 1), mode="lines", showlegend=False))
    return len(fig.to_json())


def measure(engine: LiveEngine, label: str, seconds: float) -> None:
    engine.tracker._d.clear()
    time.sleep(seconds)
    p = engine.tracker.percentiles()
    print(f"{label:42s} n={p['n']:3d}  p50={p['p50']:8.2f}  p90={p['p90']:8.2f}  p99={p['p99']:8.2f} ms")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seconds", type=float, default=15.0)
    args = ap.parse_args()
    engine = LiveEngine(device=args.device).start()
    time.sleep(3)                                                   # let the first frames fill the window
    measure(engine, "A) pipeline thread alone", args.seconds)

    stop = threading.Event()

    def ui_like_load() -> None:
        while not stop.is_set():
            build_and_serialize(engine.snapshot()["raw"])
            time.sleep(0.5)

    t = threading.Thread(target=ui_like_load, daemon=True)
    t.start()
    measure(engine, "B) + figure-building thread, same process", args.seconds)
    stop.set()
    t.join()
    engine.stop()


if __name__ == "__main__":
    main()
