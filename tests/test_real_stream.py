"""RealEEGStream, noise log and the evaluation maths, on a synthetic stand-in Recording (no download).

Tests that need the real BCI Competition IV 2a files are opt-in: ``BCI_RUN_REAL=1 pytest tests/test_real_stream.py``
(downloads ~80 MB per subject through MOABB on first use).
"""

from __future__ import annotations

import os
import time

import numpy as np
import pytest

from bci import CHUNK_SAMPLES
from bci import real_benchmark as rb
from bci.stream import RealEEGStream, Recording, RunRecording, noise_level

T = 7000
CUES = np.array([600, 2100, 3600, 5100])


def fake_recording(seed: int = 0) -> Recording:
    rng = np.random.default_rng(seed)
    runs = []
    for labels, flagged in (([0, 1, 2, 3], []), ([3, 2, 1, 0], [2])):
        art = np.zeros(4, dtype=bool)
        art[flagged] = True
        runs.append(RunRecording(
            eeg=(10 * rng.standard_normal((T, 22))).astype(np.float32),
            eog=(15 * rng.standard_normal((T, 3))).astype(np.float32),
            cue_samples=CUES.copy(), labels=np.array(labels), artifact=art))
    return Recording(1, "1test", tuple(runs), tuple(f"ch{i}" for i in range(22)))


def test_chunking_never_straddles_runs_and_matches_source():
    rec = fake_recording()
    chunks = list(RealEEGStream(recording=rec))
    assert len(chunks) == 2 * (T // CHUNK_SAMPLES)                       # remainder of each run is dropped
    for c in chunks:
        assert c.data.shape == (CHUNK_SAMPLES, 22) and c.data.dtype == np.float32
        run = rec.runs[c.info["run"]]
        np.testing.assert_array_equal(c.data, run.eeg[c.info["end_sample"] - CHUNK_SAMPLES:c.info["end_sample"]])
    assert [c.info["run"] for c in chunks][:1] == [0] and chunks[-1].info["run"] == 1
    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_labels_follow_the_imagery_period():
    rec = fake_recording()
    for c in RealEEGStream(recording=rec, runs=[0]):
        last = c.info["end_sample"] - 1
        inside = [(cue, lab) for cue, lab in zip(CUES, rec.runs[0].labels) if cue <= last < cue + 1000]
        assert c.label == (inside[0][1] if inside else -1)
    labels = {c.label for c in RealEEGStream(recording=rec, runs=[0])}
    assert labels == {-1, 0, 1, 2, 3}


def test_runs_and_n_chunks_selection():
    rec = fake_recording()
    assert len(RealEEGStream(recording=rec, runs=[1])) == T // CHUNK_SAMPLES
    chunks = list(RealEEGStream(recording=rec, n_chunks=5))
    assert len(chunks) == 5 and {c.info["run"] for c in chunks} == {0}


def test_realtime_pacing_is_one_chunk_per_256_ms():
    stream = RealEEGStream(recording=fake_recording(), n_chunks=4, realtime=True)
    t0 = time.perf_counter()
    list(stream)
    assert 0.95 <= time.perf_counter() - t0 <= 1.5                       # 4 x 0.256 s = 1.024 s


def test_noise_level_metrics():
    t = np.arange(500) / 250
    sine_10 = np.tile(np.sin(2 * np.pi * 10 * t)[:, None], (1, 4)) * 20
    sine_60 = np.tile(np.sin(2 * np.pi * 60 * t)[:, None], (1, 4)) * 20
    a, b = noise_level(sine_10), noise_level(sine_60, np.full((500, 3), 50.0))
    assert a["hf_power_frac"] < 0.01 and b["hf_power_frac"] > 0.99      # >30 Hz share separates EMG-like noise
    assert a["eeg_p2p_p95_uv"] == pytest.approx(40, rel=0.01) and "eog_rms_uv" not in a
    assert b["eog_rms_uv"] == pytest.approx(50.0)
    clipped = np.full((500, 2), 100.0)
    assert noise_level(clipped)["frac_clipped"] == 1.0 and a["frac_clipped"] == 0.0


def test_training_windows_cover_the_imagery_period():
    rec = fake_recording()
    x, y = rb.training_windows(rec)
    n_starts = len(rb.TRAIN_START_S)
    assert x.shape == (8 * n_starts, 500, 22) and y.shape == (8 * n_starts,)
    assert np.bincount(y).tolist() == [2 * n_starts] * 4
    np.testing.assert_array_equal(x[0], rec.runs[0].eeg[600:1100])       # first window starts at the cue


def _perfect_preds(rec: Recording, wrong_for_flagged: bool = True) -> dict:
    preds = {}
    for r, run in enumerate(rec.runs):
        ends = np.arange(CHUNK_SAMPLES, T + 1, CHUNK_SAMPLES)[7:]
        scores = np.zeros((len(ends), 4))
        for i, e in enumerate(ends):
            cls = 0
            for cue, lab, flag in zip(run.cue_samples, run.labels, run.artifact):
                if cue <= e < cue + 1500:
                    cls = (int(lab) + 1) % 4 if (flag and wrong_for_flagged) else int(lab)
            scores[i, cls] = 1.0
        preds[r] = (ends, scores)
    return preds


def test_trial_mapping_and_metrics_exclude_artifact_trials():
    rec = fake_recording()
    scores, labels, art, run_of = rb.trial_bin_scores(rec, _perfect_preds(rec))
    assert scores.shape == (8, rb.N_BINS, 4) and not np.isnan(scores).any()
    assert labels.tolist() == [0, 1, 2, 3, 3, 2, 1, 0] and art.sum() == 1 and run_of.tolist() == [0] * 4 + [1] * 4
    m = rb.evaluate(scores, labels, art)
    assert m["kappa_fixed_clean"] == pytest.approx(1.0) and m["acc_fixed_clean"] == pytest.approx(1.0)
    assert m["kappa_fixed_all"] < 0.9 and m["n_artifact"] == 1           # the flagged trial is predicted wrong
    assert m["kappa_peak_clean"] == pytest.approx(1.0) and m["acc_lr_fixed_clean"] == pytest.approx(1.0)


def test_stream_session_smoothing_is_a_causal_mean_restarted_at_each_run():
    from bci.classifier import fit
    rng = np.random.default_rng(1)
    params = fit(rng.standard_normal((40, 500, 22)) * 10, np.arange(40) % 4, class_names=("a", "b", "c", "d"))
    rec = fake_recording()
    raw = rb.stream_session(params, rec, "cpu", smooth_k=1)
    smooth = rb.stream_session(params, rec, "cpu", smooth_k=3)
    for r in raw:
        ends1, s1 = raw[r]
        ends3, s3 = smooth[r]
        np.testing.assert_array_equal(ends1, ends3)
        expected = np.stack([s1[max(0, i - 2):i + 1].mean(axis=0) for i in range(len(s1))])
        np.testing.assert_allclose(s3, expected, atol=1e-9)
    first_of_run1 = smooth[1][1][0]                                   # the average restarts: nothing carries over from run 0
    np.testing.assert_allclose(first_of_run1, raw[1][1][0], atol=1e-9)


def test_kappa_is_zero_for_chance_level_constant_predictions():
    rec = fake_recording()
    scores, labels, art, _ = rb.trial_bin_scores(rec, _perfect_preds(rec))
    scores[:] = 0
    scores[:, :, 0] = 1                                                  # always predict class 0
    m = rb.evaluate(scores, labels, art)
    assert m["kappa_fixed_clean"] == pytest.approx(0.0, abs=1e-9) and m["acc_fixed_clean"] == pytest.approx(0.25, abs=0.13)


def test_trial_noise_rows_one_per_trial():
    rec = fake_recording()
    rows = rb.trial_noise_rows(rec, np.ones(8, dtype=int))
    assert len(rows) == 8 and sum(r["expert_artifact_flag"] for r in rows) == 1
    assert {"eeg_p2p_p95_uv", "frac_clipped", "hf_power_frac", "eog_rms_uv", "correct_fixed"} <= set(rows[0])


def test_parse_subjects():
    assert rb.parse_subjects("1-3,5") == [1, 2, 3, 5] and rb.parse_subjects("9") == [9]


def test_competition_numbers_average_to_the_published_mean():
    assert len(rb.COMPETITION_KAPPA) == 9 and round(float(np.mean(rb.COMPETITION_KAPPA)), 2) == rb.COMPETITION_MEAN_KAPPA


# ------------------------------------------------------------------------------------------ opt-in: real data
real = pytest.mark.skipif(os.environ.get("BCI_RUN_REAL") != "1", reason="set BCI_RUN_REAL=1 to run tests on the real dataset (downloads via MOABB)")


@real
def test_real_recording_structure():
    from bci.stream import load_bciciv2a
    rec = load_bciciv2a(1, "1test")
    assert len(rec.runs) == 6 and rec.n_trials == 288 and rec.fs == 250
    assert all(r.eeg.shape[1] == 22 and r.eog.shape[1] == 3 and len(r.labels) == 48 for r in rec.runs)
    assert np.bincount(np.concatenate([r.labels for r in rec.runs])).tolist() == [72] * 4
    assert int(sum(r.artifact.sum() for r in rec.runs)) == 7             # subject 1, session E (verified against MOABB annotations)
    eeg = np.concatenate([r.eeg for r in rec.runs])
    assert 5 < np.median(eeg.std(axis=0)) < 20 and np.abs(eeg).max() <= 100.5     # microvolts, amplifier limit 100 uV
    assert all(int(r.cue_samples[-1]) + 1000 <= r.eeg.shape[0] for r in rec.runs)


@real
def test_real_subject1_decodes_far_above_chance_through_the_pipeline():
    m = _subject_metrics(1)
    assert m["kappa_fixed_clean"] > 0.4 and m["acc_fixed_clean"] > 0.55      # chance: kappa 0, accuracy 0.25
    print(f"\nS1 fixed kappa={m['kappa_fixed_clean']:.3f} acc={m['acc_fixed_clean']:.3f} peak={m['kappa_peak_clean']:.3f}")


def _subject_metrics(s: int) -> dict:
    from bci.stream import load_bciciv2a
    params = rb.train_subject(s)
    rec = load_bciciv2a(s, "1test")
    scores, labels, art, _ = rb.trial_bin_scores(rec, rb.stream_session(params, rec, "cpu"))
    return rb.evaluate(scores, labels, art)
