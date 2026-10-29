"""FastAPI/WebSocket server exposing the live pipeline.

    python -m bci.server --host 0.0.0.0 --port 8000

    GET  /health   liveness + device + frames processed
    GET  /stats    latest latency percentiles and per-stage times
    WS   /ws       JSON frame every ~250 ms: label, scores, latency, percentiles
                   (add ``?raw=1`` to include the newest 64x64 chunk of raw EEG)

The Streamlit dashboard (app/dashboard.py) is self-contained and does not use this server;
this is for clients that want the decoded stream over a socket.
"""

from __future__ import annotations

import argparse
import asyncio
from contextlib import asynccontextmanager
from typing import Protocol

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from . import CHUNK_SAMPLES, CLASS_NAMES
from .gpu_pipeline import ProcessEngine

PUSH_PERIOD_S = 0.25


class _Engine(Protocol):
    def snapshot(self) -> dict: ...
    def stop(self) -> None: ...


def _public(snap: dict, include_raw: bool = False) -> dict:
    """JSON-safe subset of an engine snapshot."""
    label, true_label = snap["label"], snap["true_label"]
    out = {
        "frames": snap["frames"], "starting": snap["starting"], "warming_up": snap["warming_up"],
        "label": label, "label_name": None if label is None else CLASS_NAMES[label],
        "true_label": true_label, "true_label_name": None if true_label is None else CLASS_NAMES[true_label],
        "scores": None if snap["scores"] is None else [float(v) for v in snap["scores"]],
        "latency_ms": snap["latency_ms"],
        "percentiles": snap["percentiles"],
        "stage_ms": snap["stage_ms"],
        "device": snap["device"], "pipeline": snap["pipeline"], "filter": snap["filter"],
        "ring": snap["ring"], "running": snap["running"], "error": snap["error"],
    }
    if include_raw and snap["raw"].shape[0] >= CHUNK_SAMPLES:
        out["raw_last_chunk"] = np.round(snap["raw"][-CHUNK_SAMPLES:], 2).tolist()
    return out


def create_app(engine: _Engine | None = None) -> FastAPI:
    """Build the app. Pass ``engine`` to inject one (tests); otherwise a :class:`ProcessEngine`
    is started on startup and stopped on shutdown."""
    owned = engine is None

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.engine = ProcessEngine().start() if owned else engine
        yield
        if owned:
            app.state.engine.stop()

    app = FastAPI(title="realtime-bci", lifespan=lifespan)

    @app.get("/health")
    def health() -> dict:
        snap = app.state.engine.snapshot()
        return {"status": "error" if snap["error"] else "ok", "device": snap["device"],
                "frames": snap["frames"], "starting": snap["starting"], "error": snap["error"]}

    @app.get("/stats")
    def stats() -> dict:
        return _public(app.state.engine.snapshot())

    @app.websocket("/ws")
    async def ws(websocket: WebSocket, raw: int = 0) -> None:
        await websocket.accept()
        try:
            while True:
                await websocket.send_json(_public(app.state.engine.snapshot(), include_raw=bool(raw)))
                await asyncio.sleep(PUSH_PERIOD_S)
        except WebSocketDisconnect:
            return

    return app


def main(argv: list[str] | None = None) -> int:
    import uvicorn
    p = argparse.ArgumentParser(prog="python -m bci.server")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args(argv)
    uvicorn.run(create_app(), host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
