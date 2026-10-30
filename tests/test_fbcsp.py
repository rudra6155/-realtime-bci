"""Filter-bank CSP reference (used by the session-T model selection)."""

from __future__ import annotations

import numpy as np

from bci import fbcsp


def _two_class_windows(n=60, c=6, t=400, seed=0):
    rng = np.random.default_rng(seed)
    y = np.arange(n) % 2
    x = rng.standard_normal((n, c, t))
    x[y == 0, 0] *= 3.0                                         # class 0: extra variance on channel 0
    x[y == 1, 1] *= 3.0                                         # class 1: extra variance on channel 1
    x -= x.mean(axis=-1, keepdims=True)
    return x, y


def test_trace_normalised_cov_has_unit_trace():
    x, _ = _two_class_windows()
    cov = fbcsp.trace_normalised_cov(x)
    np.testing.assert_allclose(np.trace(cov, axis1=1, axis2=2), 1.0, atol=1e-12)


def test_csp_filters_separate_the_variance_classes_and_have_the_right_shape():
    x, y = _two_class_windows()
    cov = fbcsp.trace_normalised_cov(x)
    w = fbcsp.csp_ovr(cov, y, n_classes=2, m=2)
    assert w.shape == (2 * 2 * 2, 6)
    f = fbcsp.log_variance(cov, w)
    assert f.shape == (60, 8)
    # the first filter of class 0 (largest eigenvalue) has more variance in class-0 windows than in class-1 windows
    assert f[y == 0, 3].mean() > f[y == 1, 3].mean() + 0.5
    # and the first filter of class 1 the opposite way round
    assert f[y == 1, 7].mean() > f[y == 0, 7].mean() + 0.5


def test_fbcsp_features_stack_all_bands_and_use_train_filters_only():
    x, y = _two_class_windows()
    covs = [fbcsp.trace_normalised_cov(x)] * 3
    model = fbcsp.FBCSP(n_classes=2, m=2).fit(covs, y)
    assert model.transform(covs).shape == (60, 3 * 8)
    x2, _ = _two_class_windows(n=10, seed=5)
    assert model.transform([fbcsp.trace_normalised_cov(x2)] * 3).shape == (10, 24)


def test_filter_bank_has_nine_four_hz_bands_and_matching_taps():
    assert len(fbcsp.FB_BANDS) == 9 and fbcsp.FB_BANDS[0] == (4, 8) and fbcsp.FB_BANDS[-1] == (36, 40)
    taps = fbcsp.fb_taps()
    assert len(taps) == 9 and all(len(t) == fbcsp.FB_TAPS for t in taps)


# ---------------------------------------------------------------------------------- deployable decoder
def _four_class_windows(n=96, c=22, t=500, seed=1):
    rng = np.random.default_rng(seed)
    y = np.arange(n) % 4
    x = rng.standard_normal((n, t, c)).astype(np.float64) * 8
    for k in range(4):
        x[y == k, :, k] *= 2.5                                   # class k: more variance on channel k
    return x.astype(np.float32), y


def test_fit_fbcsp_separates_synthetic_classes_and_selects_features():
    x, y = _four_class_windows()
    params = fbcsp.fit_fbcsp(x, y, d=24, class_names=("a", "b", "c", "d"))
    assert params.taps.shape == (9, fbcsp.FB_TAPS) and params.filters.shape == (9, 16, 22)
    assert params.select.shape == (24,) and params.n_features == 24 and params.lda_w.shape == (4, 24)
    scores = fbcsp.features_cpu(params, x)["scores"]
    assert (scores.argmax(1) == y).mean() > 0.9                  # training accuracy on easily separable data


def test_banked_filter_equals_one_band_at_a_time():
    import torch
    from bci.filters import FIRBandpass
    x = torch.from_numpy(_four_class_windows(n=2)[0])
    taps = np.stack(fbcsp.fb_taps())
    bank = FIRBandpass(taps)(x)
    assert bank.shape == (2, 9, 22, 500 - fbcsp.FB_TAPS + 1)
    for b in (0, 4, 8):
        np.testing.assert_allclose(FIRBandpass(taps[b])(x).numpy(), bank[:, b].numpy(), atol=1e-5)


def test_fbcsp_pipeline_matches_cpu_reference(device):
    """Whole streaming path (ring buffer -> staging -> H2D -> banked FIR -> covariance -> CSP features -> LDA -> D2H)."""
    from bci import CHUNK_SAMPLES, WINDOW_SAMPLES
    from bci.gpu_pipeline import GPUPipeline, drive
    from bci.ring_buffer import RingBuffer
    from bci.stream import Chunk
    x, y = _four_class_windows()
    params = fbcsp.fit_fbcsp(x, y, d=24, class_names=("a", "b", "c", "d"))
    rng = np.random.default_rng(7)
    full = (rng.standard_normal((20 * CHUNK_SAMPLES, 22)) * 8).astype(np.float32)
    chunks = [Chunk(i, full[i * CHUNK_SAMPLES:(i + 1) * CHUNK_SAMPLES], -1, 0) for i in range(20)]
    for lookahead in (False, True):
        pipe = GPUPipeline(params, device, n_channels=22, compile=False)
        results = list(drive(pipe, RingBuffer(n_channels=22), chunks, lookahead=lookahead))
        assert len(results) == 20 - (-(-WINDOW_SAMPLES // CHUNK_SAMPLES) - 1)
        ref = np.stack([fbcsp.features_cpu(params, full[(r.index + 1) * CHUNK_SAMPLES - WINDOW_SAMPLES:
                                                         (r.index + 1) * CHUNK_SAMPLES][None])["scores"][0] for r in results])
        got = np.stack([r.scores for r in results])
        np.testing.assert_allclose(got, ref, atol=2e-3, rtol=1e-3)          # float32 filter, float64 everything after
        assert [r.label for r in results] == ref.argmax(1).tolist()


def test_both_param_types_report_their_channel_count():
    from bci.classifier import fit
    x, y = _four_class_windows(n=40)
    assert fbcsp.fit_fbcsp(x, y, d=8).n_channels == 22
    assert fit(x, y).n_channels == 22
