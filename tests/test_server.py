"""FastAPI/WebSocket server (stub engine for endpoint tests) and the real process-isolated engine."""

from __future__ import annotations

import os
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from bci.gpu_pipeline import ProcessEngine, _empty_snapshot
from bci.server import create_app


class StubEngine:
    def snapshot(self) -> dict:
        snap = _empty_snapshot()
        snap.update(starting=False, warming_up=False, frames=7, label=1, true_label=0,
                    scores=np.array([-3.5, 3.5]), latency_ms=4.2, device="CPU", pipeline="stub", filter="eager",
                    percentiles={"n": 7, "p50": 4.0, "p90": 5.0, "p99": 6.0},
                    raw=np.zeros((64, 64), np.float32))
        return snap

    def stop(self) -> None:
        pass


def test_http_and_websocket_endpoints():
    with TestClient(create_app(StubEngine())) as client:
        assert client.get("/health").json()["status"] == "ok"
        stats = client.get("/stats").json()
        assert stats["label_name"] == "Right Hand" and stats["true_label_name"] == "Left Hand"
        assert stats["percentiles"]["p99"] == 6.0
        with client.websocket_connect("/ws?raw=1") as ws:
            msg = ws.receive_json()
        assert msg["frames"] == 7 and msg["scores"] == [-3.5, 3.5]
        assert np.array(msg["raw_last_chunk"]).shape == (64, 64)


def test_process_engine_runs_pipeline_in_child_process():
    """Real engine in its own process: trains, warms up, then publishes live paced snapshots."""
    engine = ProcessEngine(device="cpu").start()
    try:
        deadline = time.time() + 120
        while time.time() < deadline and engine.snapshot()["frames"] < 15:
            time.sleep(0.5)
        snap = engine.snapshot()
        assert snap["error"] is None, snap["error"]
        assert snap["frames"] >= 15 and not snap["starting"]
        assert snap["percentiles"]["n"] > 0 and snap["label"] in (0, 1)
        assert snap["ring"]["dropped_samples"] == 0
        assert snap["raw"].shape[1] == 64 and snap["device"] == "CPU"
        # closed loop: the cursor is driven by the decisions and sample-to-action latency is tracked
        assert snap["cursor"] is not None and -1.0 <= snap["cursor"]["x"] <= 1.0
        assert snap["s2a"]["n"] > 0 and 0 < snap["s2a"]["p50"] < 1000 and snap["s2a_last_ms"] > 0
        assert snap["s2a"]["p50"] >= snap["percentiles"]["p50"] * 0.5      # it contains the pipeline latency, not less
        assert snap["class_names"] == ("Left Hand", "Right Hand") and snap["source"] == "synthetic"
    finally:
        engine.stop()


@pytest.mark.skipif(os.environ.get("BCI_RUN_REAL") != "1", reason="set BCI_RUN_REAL=1 (downloads the dataset via MOABB)")
def test_process_engine_replays_real_eeg_with_four_classes():
    from bci.stream import BCICIV2A_CLASSES
    engine = ProcessEngine(device="cpu", source="real", subject=1, smooth_k=4).start()
    try:
        deadline = time.time() + 240
        while time.time() < deadline and engine.snapshot()["frames"] < 25:
            time.sleep(1)
        snap = engine.snapshot()
        assert snap["error"] is None, snap["error"]
        assert snap["class_names"] == BCICIV2A_CLASSES and snap["n_channels"] == 22 and snap["raw"].shape[1] == 22
        assert snap["source"] == "real" and snap["subject"] == 1 and snap["smooth_k"] == 4
        assert snap["cursor"] is not None and snap["s2a"]["n"] > 0 and snap["label"] in range(4)
    finally:
        engine.stop()
