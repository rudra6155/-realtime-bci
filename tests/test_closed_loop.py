"""Cursor controller, sample-to-action latency definition, score smoothing."""

from __future__ import annotations

import time

import numpy as np
import pytest

from bci.closed_loop import CursorController, RollingAgreement, ScoreSmoother

NAMES4 = ("Left Hand", "Right Hand", "Feet", "Tongue")


def test_cursor_moves_with_the_decoded_class_and_clips():
    c = CursorController(NAMES4, gain=0.25)
    t = time.perf_counter_ns()
    xs = [c.update(1, t).x for _ in range(6)]                    # right x6
    assert xs[:4] == [0.25, 0.5, 0.75, 1.0] and xs[4:] == [1.0, 1.0]       # clipped at the right edge
    assert [c.update(0, t).x for _ in range(2)] == [0.75, 0.5]             # left moves back
    assert c.update(2, t).x == 0.5 and c.update(3, t).x == 0.5             # feet / tongue: hold
    assert c.update(2, t).direction == 0 and c.update(0, t).direction == -1 and c.update(1, t).direction == 1


def test_leaky_cursor_returns_to_zero_and_settles_at_the_decision_balance():
    t = time.perf_counter_ns()
    c = CursorController(NAMES4, gain=0.1, decay=0.9)
    for _ in range(100):
        c.update(1, t)                                            # constant right: x -> gain / (1 - decay) = 1.0
    assert c.x == pytest.approx(1.0, abs=1e-3)
    xs = [c.update(2, t).x for _ in range(40)]                    # feet: no push, the position leaks away
    assert all(a > b > 0 for a, b in zip([1.0] + xs[:-1], xs)) and xs[-1] < 0.02
    d = CursorController(NAMES4, gain=0.1, decay=0.9)
    for i in range(400):
        d.update(1 if i % 4 else 0, t)                            # 75% right / 25% left: x -> 0.1 * 0.5 / 0.1 = 0.5
    assert d.x == pytest.approx(0.5, abs=0.15)
    with pytest.raises(ValueError):
        CursorController(NAMES4, decay=1.2)
    with pytest.raises(ValueError):
        CursorController(NAMES4, decay=0.0)


def test_two_class_names_work_and_unknown_names_never_move():
    c2 = CursorController(("Left Hand", "Right Hand"), gain=0.1)
    assert c2.update(0, 0).x == pytest.approx(-0.1)
    other = CursorController(("a", "b"))
    assert other.update(0, 0).direction == 0 and other.update(1, 0).direction == 0


def test_sample_to_action_is_action_time_minus_sample_time():
    c = CursorController(NAMES4)
    t_sample = time.perf_counter_ns() - 7_000_000                 # the newest sample existed 7 ms ago
    u = c.update(1, t_sample)
    assert 7.0 <= u.sample_to_action_ms < 30.0
    assert u.sample_to_action_ms == pytest.approx((u.t_action_ns - u.t_sample_ns) / 1e6)
    assert u.t_action_wall - u.t_sample_wall == pytest.approx(u.sample_to_action_ms / 1e3, abs=1e-6)
    assert abs(u.t_action_wall - time.time()) < 0.5                # wall clock fields are real wall times
    assert [c.update(1, t_sample).decision_index for _ in range(2)] == [1, 2]


def test_reset():
    c = CursorController(NAMES4, gain=0.5)
    c.update(1, 0); c.reset()
    assert c.x == 0.0 and c.decisions == 0


def test_score_smoother_is_a_causal_moving_average():
    s = ScoreSmoother(3)
    out = [s.update(np.array([v, -v])) for v in (3.0, 6.0, 9.0, 12.0)]
    np.testing.assert_allclose([o[0] for o in out], [3.0, 4.5, 6.0, 9.0])    # mean of the last <= 3 vectors
    assert ScoreSmoother(1).update(np.array([1.0, 2.0])).tolist() == [1.0, 2.0]
    s.reset()
    assert s.update(np.array([5.0, 5.0])).tolist() == [5.0, 5.0]
    with pytest.raises(ValueError):
        ScoreSmoother(0)


def test_rolling_agreement_counts_only_left_right_intents():
    c = CursorController(NAMES4)
    a = RollingAgreement(c, n=4)
    for intent, decided in ((0, 0), (1, 0), (2, 2), (None, 1), (3, 3), (1, 1)):   # feet/tongue/None are not counted
        a.update(intent, decided)
    s = a.summary()
    assert s["n"] == 3 and s["rate"] == pytest.approx(2 / 3)
    assert np.isnan(RollingAgreement(c).summary()["rate"])


# ------------------------------------------------------------------------------------------ session / engine
def _four_class_params():
    from bci.classifier import fit
    from bci.stream import BCICIV2A_CLASSES
    rng = np.random.default_rng(0)
    x = (10 * rng.standard_normal((80, 500, 22))).astype(np.float32)
    return fit(x, np.tile([0, 1, 2, 3], 20), class_names=BCICIV2A_CLASSES)


def test_session_resets_the_ring_at_run_boundaries_loops_and_smooths():
    """Real-source logic on a synthetic stand-in recording: windows never span two runs, a finished session replays."""
    from bci.gpu_pipeline import StreamingSession
    from bci.stream import RealEEGStream
    from test_real_stream import fake_recording
    rec = fake_recording()
    per_run = 7000 // 64
    session = StreamingSession(_four_class_params(), "cpu", realtime=False, source="real", smooth_k=3, compile=False,
                               stream_factory=lambda: RealEEGStream(recording=rec, realtime=False))
    frames = [session.step() for _ in range(2 * per_run + 20)]            # past the end of both runs -> loops
    decoded = [f.result is not None for f in frames]
    first = 7                                                              # 8th chunk completes the first 500-sample window
    for start in (0, per_run, 2 * per_run):                                # run 0, run 1, replayed run 0
        assert not any(decoded[start:start + first]) and all(decoded[start + first:start + first + 5])
    assert frames[per_run].chunk.info["run"] == 1 and frames[2 * per_run].chunk.info["run"] == 0
    f = frames[first + 4]
    assert f.decision_label == int(np.argmax(f.decision_scores)) and f.decision_scores.shape == (4,)
    np.testing.assert_allclose(f.decision_scores, np.mean([frames[i].result.scores for i in (first + 2, first + 3, first + 4)], axis=0))


def test_stream_factory_requires_params():
    from bci.gpu_pipeline import StreamingSession
    with pytest.raises(ValueError):
        StreamingSession(None, "cpu", stream_factory=lambda: iter(()))
