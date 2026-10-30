"""Tail-latency diagnostic: analysis logic, GC log, and a CPU run on a synthetic stand-in recording (no download)."""

from __future__ import annotations

import gc
import time
from types import SimpleNamespace

import numpy as np

from bci import tail_diag
from tests.test_real_stream import fake_recording


def _rows(n=300, slow_every=50, **slow_changes):
    rows = []
    for i in range(n):
        slow = bool(slow_every) and i % slow_every == 7
        r = {"e2e_ms": 7.0, "h2d_ms": 0.08, "filter_ms": 0.11, "cov_ms": 0.8, "classifier_ms": 5.0, "d2h_ms": 0.09,
             "host_stage_ms": 0.2, "host_launch_pre_ms": 1.0, "host_launch_post_ms": 5.5, "host_collect_ms": 0.1, "gc_ms": 0.0}
        if slow:
            r.update({"e2e_ms": 25.0, "cov_ms": 3.5, "classifier_ms": 18.0, **slow_changes})
        rows.append(r)
    return rows


def test_analyse_flags_a_stalled_enqueue_as_host_side():
    res, text = tail_diag.analyse(_rows(host_launch_pre_ms=8.0))
    assert res["n_slow"] == 6 and res["columns"]["cov_ms"]["median_slow"] == 3.5
    assert "host-side cause" in text


def test_analyse_points_to_the_gpu_when_the_enqueue_is_normal():
    _, text = tail_diag.analyse(_rows())
    assert "GPU side" in text and "host-side cause" not in text


def test_analyse_reports_gc_overlap_and_no_slow_chunks():
    res, text = tail_diag.analyse(_rows(gc_ms=3.0))
    assert res["slow_with_gc_pause"] == res["n_slow"] and "--no-gc" in text
    res, text = tail_diag.analyse(_rows(slow_every=0))
    assert res["n_slow"] == 0 and "no slow chunks" in text


def test_gc_log_records_a_collection_and_measures_overlap():
    with tail_diag.GcLog() as log:
        t0 = time.perf_counter_ns()
        gc.collect()
        t1 = time.perf_counter_ns()
    assert log.events and all(s >= t0 and e <= t1 for s, e, _ in log.events)
    assert 0 < log.overlap_ms(t0, t1) <= (t1 - t0) / 1e6
    assert log.overlap_ms(t1 + 10, t1 + 20) == 0.0
    assert tail_diag.GcLog().events == []


def test_telemetry_is_a_noop_without_nvidia_smi(monkeypatch):
    monkeypatch.setattr(tail_diag.GpuTelemetry, "_run", staticmethod(lambda q: None))
    tele = tail_diag.GpuTelemetry()
    assert not tele.available and tele.start().samples == []
    tele.stop()
    assert tail_diag.nearest_sample([], 5) == (None, None)
    assert tail_diag.nearest_sample([(10, ["a"]), (100, ["b"])], 90)[0] == ["b"]


def test_cpu_run_writes_per_chunk_csv_and_summary(tmp_path, monkeypatch):
    from bci import real_benchmark as rb
    from bci import stream as stream_mod
    from bci.classifier import fit
    rng = np.random.default_rng(0)
    params = fit(rng.standard_normal((40, 500, 22)) * 10, np.arange(40) % 4, class_names=("a", "b", "c", "d"))
    rec = fake_recording()
    monkeypatch.setattr(rb, "train_subject", lambda subject, **kw: params)
    Real = stream_mod.RealEEGStream
    monkeypatch.setattr(stream_mod, "RealEEGStream",
                        lambda subject, session, realtime, n_chunks: Real(recording=rec, n_chunks=n_chunks, realtime=False))
    args = SimpleNamespace(subject=1, device="cpu", chunks=30, warmup=5, compile=False, no_gc=True, priority="normal",
                           no_telemetry=True, unpaced=True, decoder="riemann", out_dir=str(tmp_path))
    res = tail_diag.run(args)
    import csv
    rows = list(csv.DictReader(open(tmp_path / "cpu_smoke_tail_diag_chunks.csv")))
    assert len(rows) == 30 and {"e2e_ms", "host_launch_pre_ms", "host_launch_post_ms", "gc_ms", "cov_ms"} <= set(rows[0])
    assert all(float(r["host_launch_pre_ms"]) > 0 for r in rows)
    assert (tmp_path / "cpu_smoke_tail_diag_summary.txt").read_text().startswith("device:")
    assert res["n"] == 30 and gc.isenabled()

