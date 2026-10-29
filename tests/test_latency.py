"""Latency contract + anti-pattern guards.

* GPU: 1000 chunks, assert p99 < 50 ms, two streams, pinned buffers   (skip w/o CUDA)
* CPU-runnable: AST checks that the hot path has no `.item()/.cuda()/.cpu()/.tolist()/print`
  and no Python loop over channels; benchmark harness smoke test; the CLI refuses to write
  GPU artifacts from a CPU run.
"""

from __future__ import annotations

import ast
import inspect
import textwrap

import numpy as np
import pytest

from bci import LATENCY_TARGET_MS
from bci import classifier, filters, gpu_pipeline
from bci.gpu_pipeline import GPUPipeline, main, run_benchmark, write_chunk_csv

from conftest import require_cuda

FORBIDDEN_ATTRS = {"item", "cuda", "cpu", "tolist"}
HOT_PATH = {                         # module -> functions that run per chunk
    gpu_pipeline: ["GPUPipeline.stage", "GPUPipeline.launch", "GPUPipeline.collect",
                   "GPUPipeline.process", "drive"],
    filters: ["FIRBandpass.forward"],
    classifier: ["RiemannianLDA.covariance", "RiemannianLDA.tangent", "RiemannianLDA.head",
                 "RiemannianLDA.forward"],
}
NO_LOOPS = [filters.FIRBandpass.forward, classifier.RiemannianLDA.covariance,
            classifier.RiemannianLDA.tangent, classifier.RiemannianLDA.head,
            classifier.RiemannianLDA.forward]        # vectorized over channels: no loops at all


def _hot_functions():
    for module, names in HOT_PATH.items():
        src = inspect.getsource(module)
        tree = ast.parse(src)
        for dotted in names:
            parts = dotted.split(".")
            scope = tree.body
            node = None
            for part in parts:
                node = next(n for n in scope if getattr(n, "name", None) == part)
                scope = node.body
            yield dotted, node


@pytest.mark.parametrize("dotted,node", list(_hot_functions()), ids=lambda v: v if isinstance(v, str) else "")
def test_hot_path_has_no_sync_or_print_calls(dotted, node):
    bad = []
    for n in ast.walk(node):
        if isinstance(n, ast.Attribute) and n.attr in FORBIDDEN_ATTRS:
            bad.append(f".{n.attr} (line {n.lineno})")
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "print":
            bad.append(f"print() (line {n.lineno})")
    assert not bad, f"{dotted} contains forbidden hot-path calls: {bad}"


def test_async_copies_are_used_in_hot_path():
    for name in ("stage", "launch"):
        src = inspect.getsource(getattr(GPUPipeline, name))
        assert "non_blocking=" in src, f"GPUPipeline.{name} must issue non_blocking copies"


@pytest.mark.parametrize("fn", NO_LOOPS, ids=lambda f: f.__qualname__)
def test_no_python_loop_over_channels(fn):
    tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    loops = [n for n in ast.walk(tree) if isinstance(n, (ast.For, ast.While, ast.ListComp, ast.GeneratorExp, ast.DictComp))]
    assert not loops, f"{fn.__qualname__} has Python loops/comprehensions; use vectorized ops"


# ----------------------------------------------------------------------------- harness (CPU-runnable)
def test_benchmark_harness_runs_on_cpu(params, tmp_path):
    run = run_benchmark(params, "overlap", n_chunks=20, warmup=5, device="cpu")
    assert run.e2e_ms.shape == (20,) and np.all(run.e2e_ms > 0)
    assert set(run.stage_ms) == {"h2d", "filter", "cov", "classifier", "d2h"}
    assert run.ring_stats["dropped_samples"] == 0
    write_chunk_csv(run, tmp_path / "x.csv")
    assert len((tmp_path / "x.csv").read_text().splitlines()) == 21


def test_cli_refuses_to_write_gpu_artifacts_from_cpu(tmp_path, capsys):
    rc = main(["benchmark", "--chunks", "10", "--warmup", "2", "--out-dir", str(tmp_path), "--device", "cpu"])
    assert rc == 2 and list(tmp_path.iterdir()) == []
    assert "Refusing" in capsys.readouterr().err


def test_cli_cpu_smoke_files_are_prefixed(tmp_path):
    rc = main(["benchmark", "--chunks", "10", "--warmup", "2", "--out-dir", str(tmp_path), "--device", "cpu",
               "--allow-cpu", "--runs", "overlap,baseline"])
    assert rc == 0
    names = sorted(p.name for p in tmp_path.iterdir())
    assert names == ["cpu_smoke_benchmark_latency.csv", "cpu_smoke_benchmark_latency_baseline.csv",
                     "cpu_smoke_benchmark_summary.csv", "cpu_smoke_latency_histogram.png"]


# ----------------------------------------------------------------------------- GPU contract
@pytest.mark.gpu
def test_two_streams_and_pinned_buffers(params):
    require_cuda()
    pipe = GPUPipeline(params, "cuda")
    assert pipe.overlap
    assert pipe._copy_stream != pipe._compute_stream, "need separate H2D and compute streams"
    assert all(t.is_pinned() for t in pipe._host_in + pipe._host_scores), "staging buffers must be pinned"
    assert all(t.is_cuda for t in pipe._dev_in + pipe._dev_scores)


@pytest.mark.gpu
def test_p99_latency_under_50ms(params):
    """1000 chunks, back-to-back with copy/compute overlap. Artifacts are written by `make benchmark`
    (single writer, so CSV/summary/histogram always come from the same run)."""
    require_cuda()
    run = run_benchmark(params, "overlap", n_chunks=1000, warmup=50)
    p = run.percentiles(run.e2e_ms)
    print(f"\n{run.env['device_name']} | filter: {run.filter_note} | {run.pipeline}\n"
          f"e2e latency over {p['n']} chunks: p50={p['p50']:.3f} p90={p['p90']:.3f} p99={p['p99']:.3f} max={p['max']:.3f} ms")
    assert run.e2e_ms.size == 1000
    assert run.ring_stats["dropped_samples"] == 0
    assert p["p99"] < LATENCY_TARGET_MS, (
        f"p99 {p['p99']:.2f} ms >= {LATENCY_TARGET_MS} ms. Follow the failure protocol: run "
        "`python scripts/profile.py`, find the bottleneck, log before/after in FINDINGS.md.")
