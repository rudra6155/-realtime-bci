"""Benchmark on real EEG: BCI Competition IV dataset 2a (4-class motor imagery, 9 subjects).

    python -m bci.real_benchmark accuracy --subjects 1-9 --out-dir results/real
    python -m bci.real_benchmark latency  --subject 1 --out-dir results/real      # needs LSL; GPU for real numbers
    python -m bci.real_benchmark all

Protocol (SPEC.md section 11): per subject, train the Riemannian + LDA decoder on session T (all 288 trials,
9 windows per trial), then stream session E run by run through the *same* ring buffer + pipeline used for the
latency benchmark and score the per-chunk predictions against the cue-aligned trial labels.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.metrics import cohen_kappa_score

from . import CHUNK_SAMPLES, LATENCY_TARGET_MS, WINDOW_SAMPLES
from .classifier import RiemannianLDAParams, fit
from .closed_loop import DEFAULT_SMOOTH_K, ScoreSmoother
from .gpu_pipeline import GPUPipeline, device_label, drive, environment_info, resolve_device, STAGES
from .ring_buffer import RingBuffer
from .stream import (BCICIV2A_CLASSES, Chunk, RealEEGStream, Recording, load_bciciv2a, noise_level)

log = logging.getLogger(__name__)

# Winner of BCI Competition IV dataset 2a: Kai Keng Ang et al., filter-bank CSP; kappa per subject S1..S9
# (bbci.de/competition/iv/results). Mean 0.57.
COMPETITION_KAPPA = (0.68, 0.42, 0.75, 0.48, 0.40, 0.27, 0.77, 0.75, 0.61)
COMPETITION_MEAN_KAPPA = 0.57

TRAIN_START_S = tuple(np.arange(0.0, 2.001, 0.25))        # window starts after the cue (all inside the imagery)
BIN_FIRST_END_S = 0.5                                      # first scored window ends 0.5 s after the cue ...
BIN_S = CHUNK_SAMPLES / 250                                # ... then every chunk (0.256 s)
N_BINS = 16                                                # ... up to 4.3 s after the cue
FIXED_BIN = 10                                             # nominal window end 3.06 s after the cue (window 1.06-3.06 s)
NOISE_WINDOW_S = (1.0, 3.0)


# ----------------------------------------------------------------------------------- training
def training_windows(rec: Recording, starts_s=TRAIN_START_S) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for run in rec.runs:
        for cue, label in zip(run.cue_samples, run.labels):
            for st in starts_s:
                a = int(cue) + int(round(st * rec.fs))
                if a + WINDOW_SAMPLES <= run.eeg.shape[0]:
                    xs.append(run.eeg[a:a + WINDOW_SAMPLES])
                    ys.append(int(label))
    return np.stack(xs), np.array(ys)


def train_subject(subject: int, rec_train: Recording | None = None) -> RiemannianLDAParams:
    rec = rec_train if rec_train is not None else load_bciciv2a(subject, "0train")
    x, y = training_windows(rec)
    return fit(x, y, class_names=BCICIV2A_CLASSES)


# ----------------------------------------------------------------------------------- streaming evaluation
def stream_session(params: RiemannianLDAParams, rec: Recording, device: str = "auto",
                   compile: bool | str = False, smooth_k: int = 1) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Run every run of ``rec`` through ring buffer + pipeline. Returns ``{run: (end_samples, scores)}`` where
    ``end_samples[i]`` is the (exclusive) sample index at which decoded window i ends within the run and ``scores``
    are the decision scores: the mean of the last ``smooth_k`` chunk scores (causal; reset at every run start)."""
    n_ch = rec.runs[0].eeg.shape[1]
    pipe = GPUPipeline(params, device, n_channels=n_ch, compile=compile)
    out = {}
    for r in range(len(rec.runs)):
        ring = RingBuffer(n_channels=n_ch)                 # fresh buffer per run: no window crosses a run boundary
        smoother = ScoreSmoother(smooth_k)
        results = list(drive(pipe, ring, RealEEGStream(recording=rec, runs=[r]), lookahead=False))
        out[r] = (np.array([(res.index + 1) * CHUNK_SAMPLES for res in results]),
                  np.stack([smoother.update(res.scores) for res in results]))
    return out


def trial_bin_scores(rec: Recording, preds: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """For each trial and each time bin j the scores of the chunk whose window ends closest to
    ``cue + 0.5 s + j * 0.256 s``. Returns scores ``(n_trials, N_BINS, n_classes)`` (NaN where unavailable),
    true labels, artifact flags, and run index per trial."""
    n_cls = next(iter(preds.values()))[1].shape[1]
    rows, labels, art, run_of = [], [], [], []
    for r, run in enumerate(rec.runs):
        ends, scores = preds[r]
        by_end = {int(e): scores[i] for i, e in enumerate(ends)}
        for cue, label, flag in zip(run.cue_samples, run.labels, run.artifact):
            row = np.full((N_BINS, n_cls), np.nan)
            for j in range(N_BINS):
                target = int(cue) + int(round(rec.fs * (BIN_FIRST_END_S + j * BIN_S)))
                end = int(round(target / CHUNK_SAMPLES)) * CHUNK_SAMPLES
                if end in by_end:
                    row[j] = by_end[end]
            rows.append(row); labels.append(int(label)); art.append(bool(flag)); run_of.append(r)
    return np.stack(rows), np.array(labels), np.array(art), np.array(run_of)


def _acc_kappa(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    if len(y_true) == 0:
        return float("nan"), float("nan")
    return float((y_true == y_pred).mean()), float(cohen_kappa_score(y_true, y_pred))


def evaluate(scores: np.ndarray, labels: np.ndarray, artifact: np.ndarray) -> dict:
    """Metrics from ``trial_bin_scores`` output. 'clean' = artifact-free trials (the competition's evaluation set)."""
    pred = scores.argmax(axis=2)                                       # (n_trials, N_BINS)
    valid = ~np.isnan(scores).any(axis=2)
    out: dict = {}
    for tag, sel in (("clean", ~artifact), ("all", np.ones_like(artifact))):
        kap = np.full(N_BINS, np.nan); acc = np.full(N_BINS, np.nan)
        for j in range(N_BINS):
            m = sel & valid[:, j]
            acc[j], kap[j] = _acc_kappa(labels[m], pred[m, j])
        out[f"acc_bins_{tag}"], out[f"kappa_bins_{tag}"] = acc, kap
        out[f"acc_fixed_{tag}"], out[f"kappa_fixed_{tag}"] = acc[FIXED_BIN], kap[FIXED_BIN]
        jp = int(np.nanargmax(kap))
        out[f"kappa_peak_{tag}"], out[f"t_peak_{tag}"] = float(kap[jp]), BIN_FIRST_END_S + jp * BIN_S
    lr = (labels < 2) & ~artifact & valid[:, FIXED_BIN]                 # left vs right from the same 4-class scores
    p2 = scores[lr, FIXED_BIN, :2].argmax(axis=1)
    out["acc_lr_fixed_clean"], out["kappa_lr_fixed_clean"] = _acc_kappa(labels[lr], p2)
    out["n_trials"], out["n_artifact"] = int(len(labels)), int(artifact.sum())
    return out


# ----------------------------------------------------------------------------------- noise log
def trial_noise_rows(rec: Recording, correct_fixed: np.ndarray | None = None) -> list[dict]:
    rows, i = [], 0
    lo, hi = (int(round(t * rec.fs)) for t in NOISE_WINDOW_S)
    for r, run in enumerate(rec.runs):
        for k, (cue, label, flag) in enumerate(zip(run.cue_samples, run.labels, run.artifact)):
            a, b = int(cue) + lo, int(cue) + hi
            row = {"subject": rec.subject, "session": rec.session, "run": r, "trial_in_run": k,
                   "label": int(label), "expert_artifact_flag": int(flag),
                   **noise_level(run.eeg[a:b], run.eog[a:b], rec.fs)}
            if correct_fixed is not None:
                row["correct_fixed"] = int(correct_fixed[i])
            rows.append(row); i += 1
    return rows


# ----------------------------------------------------------------------------------- accuracy benchmark
def parse_subjects(text: str) -> list[int]:
    out: list[int] = []
    for part in text.split(","):
        lo, _, hi = part.partition("-")
        out += list(range(int(lo), int(hi or lo) + 1))
    return out


def run_accuracy(subjects: list[int], device: str, out_dir: Path, smooth_k: int = 1) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    per_subject, tc_rows, noise_rows = [], [], []
    for s in subjects:
        t0 = time.time()
        params = train_subject(s)
        rec_e = load_bciciv2a(s, "1test")
        preds = stream_session(params, rec_e, device, smooth_k=smooth_k)
        scores, labels, art, _ = trial_bin_scores(rec_e, preds)
        m = evaluate(scores, labels, art)
        correct = scores[:, FIXED_BIN].argmax(axis=1) == labels
        noise_rows += trial_noise_rows(rec_e, correct)
        row = {"subject": s, "n_trials": m["n_trials"], "n_artifact_trials": m["n_artifact"],
               **{k: m[k] for k in m if not k.endswith("_bins_clean") and not k.endswith("_bins_all")
                  and k not in ("n_trials", "n_artifact")},
               "competition_kappa": COMPETITION_KAPPA[s - 1] if 1 <= s <= 9 else float("nan")}
        per_subject.append(row)
        for j in range(N_BINS):
            tc_rows.append({"subject": s, "bin": j, "t_window_end_s": round(BIN_FIRST_END_S + j * BIN_S, 3),
                            "kappa_clean": m["kappa_bins_clean"][j], "acc_clean": m["acc_bins_clean"][j],
                            "kappa_all": m["kappa_bins_all"][j], "acc_all": m["acc_bins_all"][j]})
        print(f"S{s}: fixed kappa(clean)={m['kappa_fixed_clean']:.3f} acc={m['acc_fixed_clean']:.3f} | "
              f"peak kappa(clean)={m['kappa_peak_clean']:.3f} @ {m['t_peak_clean']:.2f}s | "
              f"L/R acc={m['acc_lr_fixed_clean']:.3f} | artifact trials {m['n_artifact']}/{m['n_trials']} | {time.time() - t0:.0f}s",
              flush=True)
    _write_csv(out_dir / "accuracy_per_subject.csv", per_subject)
    _write_csv(out_dir / "kappa_timecourse.csv", tc_rows)
    _write_csv(out_dir / "noise_log.csv", noise_rows)
    plot_kappa(per_subject, out_dir / "kappa_vs_competition.png")
    return per_subject


def summarize(per_subject: list[dict]) -> dict:
    keys = ["acc_fixed_clean", "kappa_fixed_clean", "acc_fixed_all", "kappa_fixed_all", "kappa_peak_clean",
            "kappa_peak_all", "acc_lr_fixed_clean", "kappa_lr_fixed_clean"]
    return {k: float(np.nanmean([r[k] for r in per_subject])) for k in keys}


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in r.items()})


_SURFACE, _INK, _INK2, _GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
_C = {"fixed": "#2a78d6", "peak": "#eb6834", "winner": "#1baf7a"}          # validated categorical slots 1-3


def plot_kappa(per_subject: list[dict], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    subs = [r["subject"] for r in per_subject]
    x = np.arange(len(subs)); w = 0.27
    series = (("fixed", "this pipeline, fixed window (deployable)", [r["kappa_fixed_clean"] for r in per_subject]),
              ("peak", "this pipeline, best time point (competition-style)", [r["kappa_peak_clean"] for r in per_subject]),
              ("winner", "2008 winner (Ang et al., FBCSP)", [r["competition_kappa"] for r in per_subject]))
    fig, ax = plt.subplots(figsize=(9, 4.6), dpi=150, facecolor=_SURFACE)
    ax.set_facecolor(_SURFACE)
    for i, (key, label, vals) in enumerate(series):
        ax.bar(x + (i - 1) * w, vals, w, color=_C[key], edgecolor=_SURFACE, linewidth=0.8, label=label)
    means = summarize(per_subject)
    ax.axhline(COMPETITION_MEAN_KAPPA, color=_INK2, linewidth=1.0, linestyle=(0, (4, 3)),
               label=f"winner mean {COMPETITION_MEAN_KAPPA:.2f}")
    ax.set_xticks(x, [f"S{s}" for s in subs], color=_INK2, fontsize=9)
    ax.set_ylabel("Cohen's kappa, 4 classes, artifact-free trials of session E", color=_INK2, fontsize=8)
    ax.set_ylim(0, 1.0)
    ax.set_title(f"BCI Competition IV 2a: mean kappa {means['kappa_fixed_clean']:.2f} (fixed window), "
                 f"{means['kappa_peak_clean']:.2f} (best time point) vs 0.57", color=_INK, fontsize=10, loc="left")
    ax.grid(axis="y", color=_GRID, linewidth=0.6); ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(_GRID)
    ax.tick_params(colors=_INK2, labelsize=8)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=False, fontsize=7.5, labelcolor=_INK2)
    fig.tight_layout()
    fig.savefig(path, facecolor=_SURFACE)
    plt.close(fig)


# ----------------------------------------------------------------------------------- latency + jitter through LSL
def run_latency(subject: int, device: str, chunks: int, warmup: int, out_dir: Path, compile: bool | str = "auto") -> dict:
    """Real recordings arrive at 1x real time through LSL (producer in another process) into the ring buffer and
    the pipeline; measures label latency and LSL jitter in the same run."""
    from .lsl_stream import LSLChunkSource, TimedRingBuffer, finish_producer, jitter_stats, new_source_id, spawn_producer
    out_dir.mkdir(parents=True, exist_ok=True)
    import pylsl
    params = train_subject(subject)
    dev = resolve_device(device)
    pipe = GPUPipeline(params, dev, n_channels=22, compile=compile)
    pipe.warmup()
    fill = -(-WINDOW_SAMPLES // CHUNK_SAMPLES) - 1
    n_total = chunks + warmup + fill
    source_id = new_source_id()
    proc = spawn_producer(source="real", source_id=source_id, n_chunks=n_total, subject=subject, session="1test")
    try:
        src = LSLChunkSource(source_id, n_chunks=n_total)
        ring = TimedRingBuffer(n_channels=src.n_channels)
        rows = []
        for res in drive(pipe, ring, src, lookahead=False):
            t_label = pylsl.local_clock()
            ts, t_pull = src.log[res.index]
            rows.append({"chunk": res.index, "e2e_ms": res.latency_ms, "lsl_ts_to_label_ms": (t_label - ts) * 1e3,
                         **{f"{k}_ms": res.stage_ms[k] for k in STAGES}})
        stats = jitter_stats(src.log, ring.write_clock, skip=warmup + fill)
        stats["ring_dropped_samples"] = ring.dropped_samples
    finally:
        finish_producer(proc)
    meas = rows[warmup:]
    if len(meas) != chunks:
        raise RuntimeError(f"expected {chunks} measured chunks, got {len(meas)}")
    prefix = "" if dev.type == "cuda" else "cpu_smoke_"
    _write_csv(out_dir / f"{prefix}real_latency_paced.csv", meas)
    summ = {"device": device_label(dev), "n": chunks, "warmup_discarded": warmup, "pipeline": pipe.description,
            "filter": pipe.filter_info.note}
    for col in ("e2e_ms", "lsl_ts_to_label_ms"):
        a = np.array([r[col] for r in meas])
        p50, p90, p99 = np.percentile(a, [50, 90, 99])
        summ[col] = {"p50": float(p50), "p90": float(p90), "p99": float(p99), "max": float(a.max())}
    summ["jitter"] = stats
    _write_csv(out_dir / f"{prefix}real_latency_summary.csv",
               [{"metric": "e2e_ms", **summ["e2e_ms"]}, {"metric": "lsl_ts_to_label_ms", **summ["lsl_ts_to_label_ms"]}]
               + [{"metric": f"jitter_{k}", "p50": v} for k, v in stats.items()])
    return summ


# ----------------------------------------------------------------------------------- model selection (session T only)
SELECT_BANDS = {"b8_30": (8, 30, 101), "b6_10": (6, 10, 201), "b10_14": (10, 14, 201),
                "b14_22": (14, 22, 151), "b22_32": (22, 32, 151)}        # (low Hz, high Hz, FIR taps)
SELECT_CONFIGS = {"base 8-30": ("b8_30",), "4 bands": ("b6_10", "b10_14", "b14_22", "b22_32")}
SELECT_WINDOWS = (500, 750)                                               # samples (2 s, 3 s)
SELECT_K = (1, 2, 3, 4, 6, 8)                                             # chunks averaged


def _select_subject(subject: int) -> dict:
    """Leave-one-run-out CV on session T: decision scores at all 16 time bins for every trial, per config/window."""
    from scipy.signal import firwin
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from . import classifier as C
    from .filters import bandpass_cpu
    rec = load_bciciv2a(subject, "0train")
    fs, out = rec.fs, {}
    for win in SELECT_WINDOWS:
        starts = np.arange(0.0, (4.0 - win / fs) + 1e-9, 0.25)           # training windows lie inside the 4 s imagery period
        tr_w, tr_y, tr_run, ev_w = [], [], [], []
        for r, run in enumerate(rec.runs):
            for cue, lab in zip(run.cue_samples, run.labels):
                for st in starts:
                    a = int(cue) + int(round(st * fs)); tr_w.append(run.eeg[a:a + win])
                tr_y += [int(lab)] * len(starts); tr_run += [r] * len(starts)
                for j in range(N_BINS):
                    end = int(round((int(cue) + int(round(fs * (BIN_FIRST_END_S + j * BIN_S)))) / CHUNK_SAMPLES)) * CHUNK_SAMPLES
                    ev_w.append(run.eeg[end - win:end])
        tr_w, ev_w, tr_y, tr_run = np.stack(tr_w), np.stack(ev_w), np.array(tr_y), np.array(tr_run)
        ns = len(starts); n_tr = len(tr_y) // ns
        trial_y, trial_run = tr_y[::ns], tr_run[::ns]
        feats = {}
        for bk, (lo, hi, nt) in SELECT_BANDS.items():
            taps = firwin(nt, [lo, hi], pass_zero=False, fs=fs, window="hamming")
            ctr = C.covariance_cpu(bandpass_cpu(tr_w, taps), 0.05); cev = C.covariance_cpu(bandpass_cpu(ev_w, taps), 0.05)
            ris = C._sym_fn(C.riemann_mean(ctr[::ns]), lambda w: w ** -0.5)
            feats[bk] = (C.tangent_cpu(ctr, ris), C.tangent_cpu(cev, ris))
        for name, keys in SELECT_CONFIGS.items():
            ftr = np.hstack([feats[k][0] for k in keys]); fev = np.hstack([feats[k][1] for k in keys])
            scores = np.zeros((n_tr, N_BINS, 4))
            for held in sorted(set(trial_run.tolist())):
                tr = tr_run != held; idx = np.where(trial_run == held)[0]
                lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(ftr[tr], tr_y[tr])
                ev_idx = (idx[:, None] * N_BINS + np.arange(N_BINS)[None]).ravel()
                scores[idx] = lda.decision_function(fev[ev_idx]).reshape(len(idx), N_BINS, 4)
            out[(name, win)] = scores
    art = np.array([bool(a) for run in rec.runs for a in run.artifact])
    return {"scores": out, "labels": trial_y, "artifact": art}


def select_config(subjects: list[int], workers: int = 4) -> list[dict]:
    """Choose the decoder configuration by cross-validation on session T only (session E is never loaded).

    Decision rule, fixed before looking at any result: pick the configuration with the highest mean kappa at the
    fixed decision bin; if simpler configurations are statistically indistinguishable from it (mean paired
    per-subject difference <= 1 paired standard error), pick the simplest, where simplicity is the ascending order
    of (number of bands, window length, smoothing k)."""
    from multiprocessing import Pool
    with Pool(min(workers, len(subjects))) as pool:
        data = pool.map(_select_subject, subjects)
    cands = [(n, w, k) for n in SELECT_CONFIGS for w in SELECT_WINDOWS for k in SELECT_K]

    def kappa(d: dict, c: tuple) -> float:
        sm = d["scores"][(c[0], c[1])][:, max(0, FIXED_BIN - c[2] + 1):FIXED_BIN + 1].mean(axis=1)
        m = ~d["artifact"]
        return float(cohen_kappa_score(d["labels"][m], sm[m].argmax(1)))

    per = {c: np.array([kappa(d, c) for d in data]) for c in cands}
    best = max(cands, key=lambda c: per[c].mean())
    rows = []
    for c in cands:
        diff = per[best] - per[c]
        se = float(diff.std(ddof=1) / np.sqrt(len(diff))) if len(diff) > 1 else 0.0
        rows.append({"config": c[0], "n_bands": len(SELECT_CONFIGS[c[0]]), "window_samples": c[1], "smooth_k": c[2],
                     "mean_kappa_T_cv": float(per[c].mean()), "diff_to_best": float(diff.mean()), "paired_se": se,
                     "indistinguishable": bool(diff.mean() <= se)})
    ok = [r for r in rows if r["indistinguishable"]]
    chosen = min(ok, key=lambda r: (r["n_bands"], r["window_samples"], r["smooth_k"]))
    for r in rows:
        r["chosen"] = r is chosen
    return rows


# ----------------------------------------------------------------------------------- alignment check
def offset_check(subject: int = 1, starts_s=tuple(np.arange(-2.0, 6.01, 1.0))) -> list[tuple[float, float]]:
    """Cross-validated (5-fold, session T) 4-class accuracy of a 2 s window starting ``start`` seconds after the
    class annotation. Decoding must sit at chance before the cue and peak during the imagery: this is how the
    annotation onset was identified as the cue (SPEC.md section 11)."""
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from . import classifier as C
    from .filters import bandpass_cpu, design_bandpass_taps
    rec = load_bciciv2a(subject, "0train")
    taps, out = design_bandpass_taps(), []
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    for st in starts_s:
        xs, ys = [], []
        for run in rec.runs:
            for cue, label in zip(run.cue_samples, run.labels):
                a = int(cue) + int(round(st * rec.fs))
                if 0 <= a and a + WINDOW_SAMPLES <= run.eeg.shape[0]:
                    xs.append(run.eeg[a:a + WINDOW_SAMPLES]); ys.append(int(label))
        cov = C.covariance_cpu(bandpass_cpu(np.stack(xs), taps))
        feat = C.tangent_cpu(cov, C._sym_fn(C.riemann_mean(cov), lambda w: w ** -0.5))
        acc = float(cross_val_score(LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"), feat, np.array(ys), cv=cv).mean())
        out.append((float(st), acc))
    return out


# ----------------------------------------------------------------------------------- CLI
def _print_accuracy(per_subject: list[dict]) -> None:
    m = summarize(per_subject)
    print("\nmean over subjects (session T -> E, artifact-free trials unless noted):")
    print(f"  4-class accuracy, fixed window       : {m['acc_fixed_clean']:.3f}   (all trials: {m['acc_fixed_all']:.3f})")
    print(f"  4-class Cohen's kappa, fixed window  : {m['kappa_fixed_clean']:.3f}   (all trials: {m['kappa_fixed_all']:.3f})")
    print(f"  4-class kappa, best time point       : {m['kappa_peak_clean']:.3f}   (all trials: {m['kappa_peak_all']:.3f})")
    print(f"  left-vs-right accuracy / kappa       : {m['acc_lr_fixed_clean']:.3f} / {m['kappa_lr_fixed_clean']:.3f}")
    print(f"  competition winner (Ang et al.) kappa: {COMPETITION_MEAN_KAPPA:.2f}")


def _print_latency(s: dict) -> None:
    print(f"\nreal-data latency over LSL, {s['n']} chunks after {s['warmup_discarded']} warm-up, {s['device']} ({s['pipeline']}; filter: {s['filter']})")
    for col in ("e2e_ms", "lsl_ts_to_label_ms"):
        v = s[col]
        print(f"  {col:20s} p50={v['p50']:.3f} p90={v['p90']:.3f} p99={v['p99']:.3f} max={v['max']:.3f} ms")
    j = s["jitter"]
    print(f"  LSL producer-timestamp -> ring buffer: jitter(std)={j['jitter_std_ms']:.4f} ms, p99-p50={j['p99_minus_p50_ms']:.4f} ms, "
          f"mean delay={j['delay_mean_ms']:.4f} ms, max={j['delay_max_ms']:.4f} ms, producer schedule std={j['producer_schedule_std_ms']:.4f} ms")


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(prog="python -m bci.real_benchmark")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("accuracy", "latency", "all"):
        sp = sub.add_parser(name)
        sp.add_argument("--out-dir", default="results/real")
        sp.add_argument("--device", default="auto")
        sp.add_argument("--subjects", default="1-9", help="accuracy: e.g. 1-9 or 1,3,5")
        sp.add_argument("--subject", type=int, default=1, help="latency: which subject's session E is streamed")
        sp.add_argument("--chunks", type=int, default=1000)
        sp.add_argument("--warmup", type=int, default=50)
        sp.add_argument("--allow-cpu", action="store_true", help="latency on a CPU device (smoke test; files get a cpu_smoke_ prefix)")
        sp.add_argument("--smooth-k", type=int, default=DEFAULT_SMOOTH_K["real"],
                        help="accuracy: decision = mean of the last k chunk scores (default: the value chosen by 'select' on session T)")
    se = sub.add_parser("select", help="choose the decoder configuration by CV on session T only (never loads session E)")
    se.add_argument("--subjects", default="1-9")
    se.add_argument("--workers", type=int, default=4)
    se.add_argument("--out-dir", default="results/real")
    oc = sub.add_parser("offset-check", help="CV accuracy vs window offset after the annotation (cue alignment check)")
    oc.add_argument("--subject", type=int, default=1)
    args = p.parse_args(argv)
    if args.cmd == "select":
        rows = select_config(parse_subjects(args.subjects), args.workers)
        out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
        _write_csv(out / "model_selection_T.csv", rows)
        print("session-T leave-one-run-out kappa at the fixed bin (mean over subjects); rows: config/window, columns: k")
        for n in SELECT_CONFIGS:
            for w in SELECT_WINDOWS:
                print(f"  {n:10s} {w:4d} " + "".join(f"  k={r['smooth_k']}: {r['mean_kappa_T_cv']:.3f}" for r in rows
                                                     if r["config"] == n and r["window_samples"] == w))
        c = next(r for r in rows if r["chosen"])
        print(f"\nchosen by the pre-declared rule: {c['config']}, window {c['window_samples']} samples, smooth k={c['smooth_k']} "
              f"(T-CV kappa {c['mean_kappa_T_cv']:.3f})")
        return 0
    if args.cmd == "offset-check":
        print(f"subject {args.subject}, session T: 5-fold CV 4-class accuracy of a 2 s window starting at offset (s) after the annotation (chance 0.25)")
        for st, acc in offset_check(args.subject):
            print(f"  start {st:+4.1f} s  acc={acc:.3f} " + "#" * int(max(acc - 0.25, 0) * 80))
        return 0
    out = Path(args.out_dir)
    status = 0
    if args.cmd in ("accuracy", "all"):
        per_subject = run_accuracy(parse_subjects(args.subjects), args.device, out, args.smooth_k)
        _print_accuracy(per_subject)
    if args.cmd in ("latency", "all"):
        if resolve_device(args.device).type != "cuda" and not args.allow_cpu:
            print("\nlatency: no CUDA device; refusing to write GPU latency files (use --allow-cpu for a smoke test).", file=sys.stderr)
            return 2
        summ = run_latency(args.subject, args.device, args.chunks, args.warmup, out)
        _print_latency(summ)
        status = 0 if (summ["e2e_ms"]["p99"] < LATENCY_TARGET_MS and summ["jitter"]["jitter_std_ms"] < 1.0) else 1
    return status


if __name__ == "__main__":
    raise SystemExit(main())
