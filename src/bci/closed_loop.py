"""Closed-loop control: a 1-D cursor driven by the decoded class, with sample-to-action latency.

Every decision (one per 256 ms chunk) moves the cursor one step: ``Left Hand`` -> left, ``Right Hand`` ->
right, any other class (feet, tongue, ...) -> no push. The position is a leaky integrator,
``x <- clip(decay * x + gain * direction, -1, 1)``: with ``decay < 1`` it drifts back to 0 when the decoder
stops pushing, so it does not stay pinned at an edge after a burst of one-sided decisions. ``decay = 1`` is a
plain integrator. In steady state ``x`` is ``gain * (share of right - share of left decisions) / (1 - decay)``
(clipped), i.e. a window of about ``1 / (1 - decay)`` decisions.

**Sample-to-action latency** = time from the moment the *newest sample used by the decision existed* (the
chunk's completion time, ``Chunk.t_emit_ns``) to the moment the new cursor position has been computed (the
action). It covers stream hand-over, ring-buffer write, host-to-device copy, filter, covariance, the feature and
classifier stages (``eigh`` for the Riemannian decoder), device-to-host copy, optional score smoothing and the
controller update. It does **not** cover
the window's own length (the decoder needs ~2 s of data: a label trails the user's intent by about that
long), the browser, or the screen. The dashboard adds a server-side *sample-to-render* estimate on top.
"""

from __future__ import annotations

import collections
import time
from dataclasses import dataclass

import numpy as np

# Decision smoothing (chunks averaged) per data source. The real-data value is chosen by cross-validation on
# session T only (FINDINGS.md sections 8 and 9: k = 4 in both rounds); synthetic data is not smoothed.
DEFAULT_SMOOTH_K = {"synthetic": 1, "real": 4}
# Cursor leak used by the dashboard engine: a design choice made before looking at any data (time constant
# 1 / (1 - 0.95) = 20 decisions = 5.1 s), not a tuned value.
DEFAULT_CURSOR_DECAY = 0.95


@dataclass(frozen=True)
class CursorUpdate:
    x: float                    # new cursor position in [-1, 1]
    direction: int              # -1 left, 0 hold, +1 right
    decision_index: int         # 0, 1, 2, ... (one per decoded chunk)
    t_sample_ns: int            # perf_counter_ns() when the newest sample existed
    t_action_ns: int            # perf_counter_ns() when the new position was computed
    t_sample_wall: float        # the same two instants as time.time() (comparable across processes on one machine)
    t_action_wall: float

    @property
    def sample_to_action_ms(self) -> float:
        return (self.t_action_ns - self.t_sample_ns) / 1e6


class CursorController:
    """Maps decoded class indices to cursor steps. ``gain`` is the step size per decision, ``decay`` the factor
    applied to the previous position (1.0 = no leak)."""

    def __init__(self, class_names: tuple[str, ...], gain: float = 0.1,
                 left: str = "Left Hand", right: str = "Right Hand", decay: float = 1.0) -> None:
        if not 0.0 < decay <= 1.0:
            raise ValueError("decay must be in (0, 1]")
        self.decay = decay
        self.class_names = tuple(class_names)
        self.left_idx = self.class_names.index(left) if left in self.class_names else None
        self.right_idx = self.class_names.index(right) if right in self.class_names else None
        self.gain = gain
        self.x = 0.0
        self.decisions = 0

    def direction_of(self, label: int) -> int:
        return -1 if label == self.left_idx else (1 if label == self.right_idx else 0)

    def update(self, label: int, t_sample_ns: int) -> CursorUpdate:
        direction = self.direction_of(label)
        self.x = float(np.clip(self.decay * self.x + self.gain * direction, -1.0, 1.0))
        t_action_ns = time.perf_counter_ns()
        wall = time.time()
        update = CursorUpdate(self.x, direction, self.decisions, t_sample_ns, t_action_ns,
                              wall - (t_action_ns - t_sample_ns) / 1e9, wall)
        self.decisions += 1
        return update

    def reset(self) -> None:
        self.x, self.decisions = 0.0, 0


class ScoreSmoother:
    """Mean of the last ``k`` per-chunk score vectors (causal). ``k = 1`` is a pass-through."""

    def __init__(self, k: int = 1) -> None:
        if k < 1:
            raise ValueError("k must be >= 1")
        self.k = k
        self._buf: collections.deque[np.ndarray] = collections.deque(maxlen=k)

    def update(self, scores: np.ndarray) -> np.ndarray:
        self._buf.append(np.asarray(scores, dtype=np.float64))
        return np.mean(self._buf, axis=0)

    def reset(self) -> None:
        self._buf.clear()


class RollingAgreement:
    """Fraction of the last ``n`` decisions whose class matched the simulated/recorded intent, counting only
    chunks whose intent is a class the cursor responds to (left or right)."""

    def __init__(self, controller: CursorController, n: int = 100) -> None:
        self.controller = controller
        self._hits: collections.deque[bool] = collections.deque(maxlen=n)

    def update(self, intent_label: int | None, decided_label: int) -> None:
        if intent_label is not None and intent_label in (self.controller.left_idx, self.controller.right_idx):
            self._hits.append(intent_label == decided_label)

    def summary(self) -> dict:
        return {"n": len(self._hits), "rate": float(np.mean(self._hits)) if self._hits else float("nan")}
