"""Unit tests for the interval math / plotting in scripts/profile.py using a hand-built trace.

This checks OUR arithmetic. It does not check that a real torch.profiler trace uses these
field names (cat == kernel / gpu_memcpy, name contains HtoD, args.stream); that is only
confirmed by running scripts/profile.py on a GPU.
"""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "profile.py"


@pytest.fixture(scope="module")
def prof():
    spec = importlib.util.spec_from_file_location("bci_profile_script", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _ev(cat, name, ts, dur, stream):
    return {"ph": "X", "cat": cat, "name": name, "ts": ts, "dur": dur, "args": {"stream": stream}}


def _trace(tmp_path, h2d_starts=(150, 450, 750, 1050, 1350)):
    events = [_ev("cpu_op", "aten::conv1d", 0, 50, 0), _ev("kernel", "k_filter", 100, 100, 7),
              _ev("kernel", "k_eigh", 300, 100, 7), _ev("gpu_memcpy", "Memcpy DtoH (Device -> Pinned)", 410, 10, 13),
              _ev("gpu_memcpy", "Memcpy DtoD (Device -> Device)", 500, 10, 7)]
    events += [_ev("gpu_memcpy", "Memcpy HtoD (Pinned -> Device)", t, 30 if i == 0 else 20, 13)
               for i, t in enumerate(h2d_starts)]
    path = tmp_path / "trace.json"
    path.write_text(json.dumps({"traceEvents": events}))
    return path


def test_intersection_length(prof):
    assert prof.intersection_length([(0, 10)], [(5, 15)]) == 5
    assert prof.intersection_length([(0, 10), (8, 20)], [(5, 12), (30, 40)]) == 7      # unions: [0,20] & [5,12]
    assert prof.intersection_length([(0, 1)], [(2, 3)]) == 0
    assert prof.intersection_length([], [(0, 5)]) == 0


def test_load_and_analyse(prof, tmp_path):
    events = prof.load_gpu_events(_trace(tmp_path))
    kinds = [e["kind"] for e in events]
    assert kinds.count("kernel") == 2 and kinds.count("h2d") == 5 and kinds.count("d2h") == 1
    assert "cpu_op" not in kinds and all(e["name"] != "Memcpy DtoD (Device -> Device)" for e in events)
    s = prof.analyse(events)
    # first copy [150,180] lies inside kernel [100,200] (30 us); the other copies are outside kernels
    assert s["h2d_total_us"] == 30 + 4 * 20
    assert s["h2d_overlapped_with_kernels_us"] == 30
    assert s["h2d_overlap_fraction"] == pytest.approx(30 / 110)
    assert s["d2h_overlapped_with_kernels_us"] == 0
    assert s["streams_with_kernels"] == [7] and s["streams_with_memcpy"] == [13]


def test_no_overlap_gives_zero_fraction(prof, tmp_path):
    events = [e for e in prof.load_gpu_events(_trace(tmp_path, h2d_starts=(210, 450, 750, 1050)))]
    assert prof.analyse(events)["h2d_overlap_fraction"] == 0


def test_timeline_png_is_written(prof, tmp_path):
    events = prof.load_gpu_events(_trace(tmp_path))
    out = tmp_path / "t.png"
    prof.plot_timeline(events, prof.analyse(events), out)
    assert out.stat().st_size > 5_000


def test_profile_script_does_not_shadow_stdlib_profile(prof):
    import profile as stdlib_profile            # noqa: F401 - would be our script if the guard failed
    assert not hasattr(stdlib_profile, "load_gpu_events")


def test_from_trace_mode_redraws_without_a_gpu(prof, tmp_path, capsys):
    trace = _trace(tmp_path)
    rc = prof.main(["--from-trace", str(trace), "--out-dir", str(tmp_path / "out")])
    assert rc == 0 and (tmp_path / "out" / "profiler_overlap.png").stat().st_size > 5_000
    assert "h2d_overlapped_with_kernels_us" in capsys.readouterr().out


def test_zoom_focuses_on_the_overlapping_copy(prof, tmp_path):
    events = prof.load_gpu_events(_trace(tmp_path))
    focus = prof._focus_h2d(events)
    assert (focus["start"], focus["end"]) == (150.0, 180.0)          # the copy inside kernel [100, 200]
    no_overlap = prof.load_gpu_events(_trace(tmp_path, h2d_starts=(210, 450, 750, 1050)))
    assert prof._focus_h2d(no_overlap)["start"] in (450.0, 750.0)    # falls back to a middle copy
