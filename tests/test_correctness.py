"""GPU/torch output must match the CPU (scipy/numpy/sklearn, float64) reference.

Each test is parametrized over devices; the ``cuda`` cases skip without CUDA (or fail with
BCI_REQUIRE_CUDA=1). On a machine without CUDA this therefore only checks torch-on-CPU
against scipy reference implementation.
"""

from __future__ import annotations

import shutil

import numpy as np
import pytest
import torch

from bci import CHUNK_SAMPLES, WINDOW_SAMPLES
from bci.classifier import (OnnxLDAHead, RiemannianLDA, covariance_cpu, export_lda_head_onnx,
                            features_cpu, tangent_cpu)
from bci.filters import bandpass_cpu, make_bandpass
from bci.gpu_pipeline import GPUPipeline, drive
from bci.ring_buffer import RingBuffer
from bci.stream import SyntheticEEGStream, make_trials

ATOL = 1e-4


@pytest.fixture(scope="module")
def windows() -> np.ndarray:
    """10 labelled windows (5 per class) that were not used for training."""
    return make_trials(5, seed=99)[0]


def assert_matches(name: str, cpu_out: np.ndarray, gpu_out: torch.Tensor) -> None:
    """torch.allclose(cpu_out, gpu_out, atol=1e-4); on failure report max |diff| and where."""
    cpu_t = torch.from_numpy(np.asarray(cpu_out, dtype=np.float64))
    gpu_t = gpu_out.detach().cpu().to(torch.float64)
    diff = (cpu_t - gpu_t).abs()
    idx = tuple(int(i) for i in np.unravel_index(int(diff.argmax()), diff.shape))
    print(f"\n[{name}] max |cpu-gpu| = {diff.max().item():.3e} at index {idx}  (|cpu|max = {cpu_t.abs().max().item():.3g})")
    assert torch.allclose(cpu_t, gpu_t, atol=ATOL), (
        f"{name}: max abs diff {diff.max().item():.3e} at index {idx} exceeds atol={ATOL}")


def _filtered(params, windows, device):
    filt, _ = make_bandpass(params.taps, device, compile=False)
    with torch.no_grad():
        return filt(torch.from_numpy(windows).to(device))


def test_bandpass_matches_scipy(params, windows, device):
    assert_matches("bandpass", bandpass_cpu(windows, params.taps), _filtered(params, windows, device))


def test_covariance_matches_cpu(params, windows, device):
    model = RiemannianLDA(params, device)
    with torch.no_grad():
        cov = model.covariance(_filtered(params, windows, device))
    assert_matches("covariance", covariance_cpu(bandpass_cpu(windows, params.taps), params.shrinkage), cov)


def test_tangent_features_match_cpu(params, windows, device):
    model = RiemannianLDA(params, device)
    ref = features_cpu(params, windows)
    with torch.no_grad():
        feat = model.tangent(model.covariance(_filtered(params, windows, device)))
    assert_matches("tangent features", ref["feat"], feat)


def test_classification_matches_cpu(params, windows, device):
    model = RiemannianLDA(params, device)
    ref = features_cpu(params, windows)
    with torch.no_grad():
        scores = model(_filtered(params, windows, device))
    assert_matches("LDA scores", ref["scores"], scores)
    np.testing.assert_array_equal(scores.cpu().numpy().argmax(1), ref["scores"].argmax(1))


def test_streaming_pipeline_matches_cpu_reference(params, device):
    """Whole path (ring buffer -> pinned staging -> async H2D -> kernels -> D2H), both orderings."""
    chunks = list(SyntheticEEGStream(seed=3, n_chunks=30))
    full = np.concatenate([c.data for c in chunks])
    for lookahead in (False, True):
        pipe = GPUPipeline(params, device, compile=False)
        ring = RingBuffer()
        results = list(drive(pipe, ring, chunks, lookahead=lookahead))
        assert len(results) == 30 - (-(-WINDOW_SAMPLES // CHUNK_SAMPLES) - 1)
        assert ring.dropped_samples == 0 and ring.unread == 0
        ref = np.stack([
            features_cpu(params, full[(r.index + 1) * CHUNK_SAMPLES - WINDOW_SAMPLES:(r.index + 1) * CHUNK_SAMPLES][None])["scores"][0]
            for r in results])
        assert_matches(f"pipeline scores, {len(results)} chunks (lookahead={lookahead})", ref,
                       torch.from_numpy(np.stack([r.scores for r in results])))
        assert [r.label for r in results] == ref.argmax(1).tolist()


def test_onnx_head_matches_pytorch(params, windows, device, tmp_path):
    """Export the LDA head, run it in onnxruntime, compare with the torch head on the same features."""
    model = RiemannianLDA(params, device)
    path = export_lda_head_onnx(model, tmp_path / "lda_head.onnx")
    head = OnnxLDAHead(path)                                  # CUDAExecutionProvider when available
    print(f"\nonnxruntime providers: {head.providers}")
    with torch.no_grad():
        feats = model.tangent(model.covariance(_filtered(params, windows, device)))
        torch_scores = model.head(feats)
    onnx_scores = head(feats.cpu().numpy())
    assert_matches("ONNX vs PyTorch head", onnx_scores, torch_scores)
    if device == "cuda" and "CUDAExecutionProvider" not in head.providers:
        pytest.fail("device is cuda but onnxruntime has no CUDAExecutionProvider (install onnxruntime-gpu, not onnxruntime)")


def test_heldout_accuracy_on_synthetic_data(params):
    """Accuracy of the CPU reference classifier on synthetic held-out trials (not a real-EEG number)."""
    x, y = make_trials(100, seed=1234)                         # seeds 0 (train), 7/8/21/22 (tuning) are not reused
    pred = features_cpu(params, x)["scores"].argmax(1)
    acc = float((pred == y).mean())
    print(f"\nHELD-OUT ACCURACY (synthetic, n={len(y)}, seed=1234): {acc:.3f}")
    assert acc >= 0.75


def test_compile_failure_falls_back_to_eager_and_says_so(params, windows, monkeypatch):
    def broken(*a, **k):
        raise RuntimeError("no triton here")
    monkeypatch.setattr(torch, "compile", broken)
    filt, info = make_bandpass(params.taps, "cpu", compile=True)
    assert info.compiled is False and "compile failed" in info.note
    with torch.no_grad():
        out = filt(torch.from_numpy(windows))
    assert_matches("eager fallback bandpass", bandpass_cpu(windows, params.taps), out)


def test_torch_compile_filter_matches_eager(params, windows, device):
    if device == "cpu" and not (shutil.which("g++") or shutil.which("cc") or shutil.which("cl")):
        pytest.skip("no C++ compiler for Inductor on CPU")
    filt, info = make_bandpass(params.taps, device, compile=True)
    if not info.compiled:
        pytest.skip(f"torch.compile unavailable on this machine: {info.note}")
    with torch.no_grad():
        out = filt(torch.from_numpy(windows[:1]).to(device))
    assert_matches("torch.compile'd bandpass", bandpass_cpu(windows[:1], params.taps), out)
