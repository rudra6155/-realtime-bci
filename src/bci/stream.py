"""Simulated 64-channel EEG stream (synthetic motor-imagery-like data).

Model (see SPEC.md §5): four narrow-band latent sources (mu ~10 Hz and beta ~20 Hz over
the left and right sensorimotor areas of an 8x8 electrode grid), mixed through fixed
spatial patterns, plus 1/f-like background and white sensor noise. Imagining a right-hand
movement attenuates the left-hemisphere sources (event-related desynchronization) and
vice versa. This is a toy generator; accuracy on it says nothing about real EEG.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Iterator

import numpy as np
from scipy.signal import lfilter

from . import CHUNK_SAMPLES, CHUNK_PERIOD_S, FS_HZ, N_CHANNELS, WINDOW_SAMPLES

# 3rd-order IIR approximation of a 1/f ("pink") spectrum.
_PINK_B = np.array([0.049922035, -0.095993537, 0.050612699, -0.004408786])
_PINK_A = np.array([1.0, -2.494956002, 2.017265875, -0.522189400])

_SOURCE_HZ = (10.0, 20.0, 10.0, 20.0)   # mu_L, beta_L, mu_R, beta_R
_SOURCE_POLE_RADIUS = 0.97              # ~2.4 Hz bandwidth at fs=250
TRIAL_CHUNKS = 16                       # class switches every 16 chunks (~4.1 s)


@dataclass(frozen=True)
class Chunk:
    index: int
    data: np.ndarray        # (chunk_samples, n_channels) float32
    label: int              # ground-truth class of this chunk (-1: none / unknown)
    t_emit_ns: int          # time.perf_counter_ns() when the chunk was completed
    info: dict | None = None  # source-specific extras (RealEEGStream: run, end_sample, noise)


@dataclass(frozen=True)
class _Static:
    patterns: np.ndarray    # (n_channels, 4) spatial pattern of each source
    smooth: np.ndarray      # (n_channels, n_channels) background spatial smoothing
    ar_a: tuple             # AR(2) denominators, one per source
    ar_gain: np.ndarray     # (4,) 1/std so each source has unit variance
    hemisphere: np.ndarray  # (4,) 0 = left hemisphere source, 1 = right


@lru_cache(maxsize=4)
def _static(n_channels: int, fs: int) -> _Static:
    side = int(round(np.sqrt(n_channels)))
    if side * side != n_channels:
        raise ValueError("n_channels must be a perfect square (electrode grid)")
    rows, cols = np.divmod(np.arange(n_channels), side)
    centers = [(0.5 * (side - 1), 0.22 * (side - 1)), (0.5 * (side - 1), 0.30 * (side - 1)),
               (0.5 * (side - 1), 0.78 * (side - 1)), (0.5 * (side - 1), 0.70 * (side - 1))]
    patterns = np.empty((n_channels, 4))
    for k, (cr, cc) in enumerate(centers):
        d2 = (rows - cr) ** 2 + (cols - cc) ** 2
        p = np.exp(-d2 / (2 * (0.22 * side) ** 2))
        patterns[:, k] = p / np.linalg.norm(p) * np.sqrt(n_channels) * 0.35
    d2 = (rows[:, None] - rows[None, :]) ** 2 + (cols[:, None] - cols[None, :]) ** 2
    smooth = np.exp(-d2 / (2 * 1.2 ** 2))
    smooth /= smooth.sum(axis=1, keepdims=True)
    ar_a, ar_gain = [], []
    for f0 in _SOURCE_HZ:
        r, w = _SOURCE_POLE_RADIUS, 2 * np.pi * f0 / fs
        a = np.array([1.0, -2 * r * np.cos(w), r * r])
        h = lfilter([1.0], a, np.r_[1.0, np.zeros(4095)])
        ar_a.append(a)
        ar_gain.append(1.0 / np.sqrt(np.sum(h * h)))
    return _Static(patterns, smooth, tuple(ar_a), np.array(ar_gain), np.array([0, 0, 1, 1]))


class EEGSynth:
    """Stateful generator: filter states persist across :meth:`generate` calls."""

    def __init__(self, seed: int = 0, n_channels: int = N_CHANNELS, fs: int = FS_HZ,
                 erd: float = 0.45, source_amp: float = 0.3, trial_jitter: float = 0.4,
                 scale_uv: float = 60.0) -> None:
        # erd / source_amp / trial_jitter were chosen on tuning seeds (7, 21) so held-out
        # accuracy is well below 100% (an easily-separable toy would be a weak demo);
        # tests evaluate on different seeds.
        self.n_channels, self.fs = n_channels, fs
        self.erd, self.source_amp, self.scale_uv = erd, source_amp, scale_uv
        self.trial_jitter = trial_jitter
        self._st = _static(n_channels, fs)
        self._rng = np.random.default_rng(seed)
        self._src_zi = [np.zeros(2) for _ in range(4)]
        self._pink_zi = np.zeros((3, n_channels))
        self._trial_gain = np.ones(4)
        self.new_trial()

    def new_trial(self) -> None:
        """Draw per-trial amplitude jitter (trial-to-trial variability)."""
        self._trial_gain = np.exp(self.trial_jitter * self._rng.standard_normal(4))

    def class_gains(self, label: int) -> np.ndarray:
        """Per-source gain: the hemisphere contralateral to the imagined hand is attenuated."""
        gains = np.ones(4)
        suppressed = 1 if label == 0 else 0   # Left Hand -> right-hemisphere sources drop
        gains[self._st.hemisphere == suppressed] = 1.0 - self.erd
        return gains * self._trial_gain

    def generate(self, n: int, label: int) -> np.ndarray:
        """Next ``n`` samples for class ``label`` as ``(n, n_channels)`` float32."""
        st, rng = self._st, self._rng
        src = np.empty((n, 4))
        for k in range(4):
            w = rng.standard_normal(n)
            y, self._src_zi[k] = lfilter([1.0], st.ar_a[k], w, zi=self._src_zi[k])
            src[:, k] = y * st.ar_gain[k]
        sig = (src * self.class_gains(label) * self.source_amp) @ st.patterns.T
        pink, self._pink_zi = lfilter(_PINK_B, _PINK_A, rng.standard_normal((n, self.n_channels)),
                                      axis=0, zi=self._pink_zi)
        bg = (pink * 4.0) @ st.smooth.T
        noise = 0.5 * rng.standard_normal((n, self.n_channels))
        return ((sig + bg + noise) * self.scale_uv / 4.0).astype(np.float32)


class SyntheticEEGStream:
    """Iterator of :class:`Chunk` objects; optionally paced at 1x real time.

    With ``realtime=True`` chunk *k* is yielded at ``t0 + (k+1) * chunk_period`` — i.e.
    when its last sample would have arrived from the amplifier.
    """

    def __init__(self, seed: int = 0, n_chunks: int | None = None, realtime: bool = False,
                 chunk_samples: int = CHUNK_SAMPLES, n_channels: int = N_CHANNELS,
                 fs: int = FS_HZ, trial_chunks: int = TRIAL_CHUNKS, **synth_kwargs) -> None:
        self.n_chunks, self.realtime = n_chunks, realtime
        self.chunk_samples, self.fs, self.trial_chunks = chunk_samples, fs, trial_chunks
        self._synth = EEGSynth(seed, n_channels, fs, **synth_kwargs)
        self._label_rng = np.random.default_rng(seed + 10_000)
        self._label = int(self._label_rng.integers(2))
        self._index = 0
        self._t0: float | None = None

    @property
    def chunk_period_s(self) -> float:
        return self.chunk_samples / self.fs

    def __iter__(self) -> Iterator[Chunk]:
        return self

    def __next__(self) -> Chunk:
        if self.n_chunks is not None and self._index >= self.n_chunks:
            raise StopIteration
        if self._index > 0 and self._index % self.trial_chunks == 0:
            self._label = 1 - self._label
            self._synth.new_trial()
        data = self._synth.generate(self.chunk_samples, self._label)
        if self.realtime:
            now = time.perf_counter()
            if self._t0 is None:
                self._t0 = now
            deadline = self._t0 + (self._index + 1) * self.chunk_period_s
            if deadline > now:
                time.sleep(deadline - now)
        chunk = Chunk(self._index, data, self._label, time.perf_counter_ns())
        self._index += 1
        return chunk


def make_trials(n_per_class: int, seed: int = 0, n_samples: int = WINDOW_SAMPLES,
                burn_in: int = 600, **synth_kwargs) -> tuple[np.ndarray, np.ndarray]:
    """Independent labelled windows ``X (N, n_samples, 64)``, ``y (N,)`` for training/eval.

    Each trial gets a fresh generator (new filter states, new jitter) that is run for
    ``burn_in`` samples first so the resonators and pink filter are in steady state.
    """
    rng = np.random.default_rng(seed)
    X = np.empty((2 * n_per_class, n_samples, N_CHANNELS), dtype=np.float32)
    y = np.empty(2 * n_per_class, dtype=np.int64)
    labels = np.tile([0, 1], n_per_class)
    for i, label in enumerate(labels):
        synth = EEGSynth(int(rng.integers(2**31)), **synth_kwargs)
        synth.generate(burn_in, int(label))
        X[i] = synth.generate(n_samples, int(label))
        y[i] = label
    return X, y


# =============================================================================== real EEG (BCI Competition IV 2a)
BCICIV2A_CLASSES = ("Left Hand", "Right Hand", "Feet", "Tongue")
BCICIV2A_FS = 250
BCICIV2A_N_EEG = 22
MI_SECONDS = 4.0              # imagery period after the cue (cue at t=2 s, imagery until t=6 s)
CLIP_UV = 99.9                # the recording amplifier saturates at +-100 uV
_MI_LABEL = {"left_hand": 0, "right_hand": 1, "feet": 2, "tongue": 3}


@dataclass(frozen=True)
class RunRecording:
    """One 48-trial run. Arrays are in microvolts, sample index 0 = start of the run."""
    eeg: np.ndarray              # (T, 22) float32
    eog: np.ndarray              # (T, 3) float32  (never used for decoding; noise log only)
    cue_samples: np.ndarray      # (n_trials,) int64, sample index of the cue onset
    labels: np.ndarray           # (n_trials,) int64, 0..3 = left, right, feet, tongue
    artifact: np.ndarray         # (n_trials,) bool, the experts' artifact flag


@dataclass(frozen=True)
class Recording:
    subject: int
    session: str                 # "0train" (session T) or "1test" (session E)
    runs: tuple[RunRecording, ...]
    ch_names: tuple[str, ...]
    fs: int = BCICIV2A_FS

    @property
    def n_trials(self) -> int:
        return sum(len(r.labels) for r in self.runs)


@lru_cache(maxsize=2)
def _load_subject(subject: int, data_path: str | None):
    """Both sessions of one subject as MOABB raws (downloads ~80 MB on first use)."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")                         # MOABB/MNE FutureWarnings
        from moabb.datasets import BNCI2014_001
        return BNCI2014_001(artifact_handling="annotate").get_data(subjects=[subject])[subject]


def load_bciciv2a(subject: int, session: str = "1test", data_path: str | None = None) -> Recording:
    """BCI Competition IV 2a through MOABB. ``session``: ``"0train"`` (T) or ``"1test"`` (E).

    Each class annotation starts at the **cue** and lasts 4 s (checked empirically: decoding is at
    chance before it and peaks for windows starting 1 s after it). Artifact flags are matched to
    trials by cue onset. Requires the ``real`` extra (moabb).
    """
    raws = _load_subject(subject, data_path)[session]
    runs = []
    for key in sorted(raws, key=int):
        raw = raws[key]
        fs = raw.info["sfreq"]
        cues, labels = [], []
        flagged = [round(a["onset"] * fs) for a in raw.annotations if a["description"] == "bnci_artifact"]
        for a in raw.annotations:
            if a["description"] in _MI_LABEL:
                cues.append(round(a["onset"] * fs))
                labels.append(_MI_LABEL[a["description"]])
        cues = np.array(cues, dtype=np.int64)
        runs.append(RunRecording(
            eeg=(raw.get_data(picks="eeg") * 1e6).T.astype(np.float32),
            eog=(raw.get_data(picks="eog") * 1e6).T.astype(np.float32),
            cue_samples=cues, labels=np.array(labels, dtype=np.int64),
            artifact=np.isin(cues, np.array(flagged, dtype=np.int64))))
    names = tuple(raws[sorted(raws, key=int)[0]].copy().pick("eeg").ch_names)
    return Recording(subject, session, tuple(runs), names)


def noise_level(eeg_uv: np.ndarray, eog_uv: np.ndarray | None = None, fs: int = BCICIV2A_FS) -> dict:
    """Noise metrics for a ``(T, C)`` EEG window in microvolts (artifacts are logged, never removed).

    * ``eeg_p2p_p95_uv``: 95th percentile over channels of the peak-to-peak amplitude;
    * ``frac_clipped``: fraction of samples at the amplifier limit (|x| >= 99.9 uV);
    * ``hf_power_frac``: share of spectral power above 30 Hz (muscle/EMG indicator);
    * ``eog_rms_uv``: RMS of the EOG channels, if given.
    """
    x = np.asarray(eeg_uv, dtype=np.float64)
    spec = np.abs(np.fft.rfft(x - x.mean(axis=0), axis=0)) ** 2
    freqs = np.fft.rfftfreq(x.shape[0], 1.0 / fs)
    total = spec[freqs >= 1.0].sum()
    out = {"eeg_p2p_p95_uv": float(np.percentile(np.ptp(x, axis=0), 95)),
           "frac_clipped": float((np.abs(x) >= CLIP_UV).mean()),
           "hf_power_frac": float(spec[freqs > 30.0].sum() / total) if total > 0 else 0.0}
    if eog_uv is not None:
        out["eog_rms_uv"] = float(np.sqrt(np.mean(np.asarray(eog_uv, dtype=np.float64) ** 2)))
    return out


class RealEEGStream:
    """Plays a BCI Competition IV 2a session as a stream of 64-sample chunks (one run after another).

    Same interface as :class:`SyntheticEEGStream` (iterator of :class:`Chunk`, optional 1x real-time
    pacing) but with 22 EEG channels in microvolts. ``Chunk.label`` is the class of the trial whose
    imagery period (cue .. cue + 4 s) contains the chunk's last sample, else -1. ``Chunk.info`` carries
    ``run``, ``end_sample`` (within the run) and ``noise_p2p_uv`` (95th-percentile peak-to-peak of the
    chunk). Chunks never straddle two runs; the few samples left over at the end of a run are dropped.
    Pass ``recording=`` to stream an already loaded (or synthetic stand-in) :class:`Recording`.
    """

    def __init__(self, subject: int = 1, session: str = "1test", *, recording: Recording | None = None,
                 runs: list[int] | None = None, n_chunks: int | None = None, realtime: bool = False,
                 chunk_samples: int = CHUNK_SAMPLES) -> None:
        self.recording = recording if recording is not None else load_bciciv2a(subject, session)
        self.n_channels = self.recording.runs[0].eeg.shape[1]
        self.fs, self.chunk_samples, self.realtime = self.recording.fs, chunk_samples, realtime
        run_ids = list(range(len(self.recording.runs))) if runs is None else list(runs)
        self._plan = [(r, k) for r in run_ids for k in range(self.recording.runs[r].eeg.shape[0] // chunk_samples)]
        if n_chunks is not None:
            self._plan = self._plan[:n_chunks]
        self._index = 0
        self._t0: float | None = None

    @property
    def chunk_period_s(self) -> float:
        return self.chunk_samples / self.fs

    def __len__(self) -> int:
        return len(self._plan)

    def __iter__(self) -> Iterator[Chunk]:
        return self

    def _label_at(self, run: RunRecording, last_sample: int) -> int:
        k = int(np.searchsorted(run.cue_samples, last_sample, side="right")) - 1
        if k >= 0 and last_sample < run.cue_samples[k] + MI_SECONDS * self.fs:
            return int(run.labels[k])
        return -1

    def __next__(self) -> Chunk:
        if self._index >= len(self._plan):
            raise StopIteration
        run_i, k = self._plan[self._index]
        run = self.recording.runs[run_i]
        start = k * self.chunk_samples
        data = run.eeg[start:start + self.chunk_samples]
        if self.realtime:
            now = time.perf_counter()
            if self._t0 is None:
                self._t0 = now
            deadline = self._t0 + (self._index + 1) * self.chunk_period_s
            if deadline > now:
                time.sleep(deadline - now)
        chunk = Chunk(self._index, data, self._label_at(run, start + self.chunk_samples - 1), time.perf_counter_ns(),
                      {"run": run_i, "end_sample": start + self.chunk_samples,
                       "noise_p2p_uv": float(np.percentile(np.ptp(data, axis=0), 95))})
        self._index += 1
        return chunk
