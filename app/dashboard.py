"""Live closed-loop dashboard: simulated or real 64/22-channel EEG -> ring buffer -> GPU/CPU pipeline -> label ->
cursor.

Run locally:  make dashboard        (streamlit run app/dashboard.py)
Hugging Face Spaces: Streamlit SDK, ``app_file: app/dashboard.py`` (see README front-matter).
The pipeline runs in its own process (so rendering cannot stall it); this page only renders snapshots.

Closed loop: every decision (one per 256 ms chunk) moves the cursor one step (Left Hand -> left, Right Hand ->
right, other classes -> hold). *Sample-to-action* latency = newest sample of the decision window existed -> new
cursor position computed (measured in the engine process). *Sample-to-render* = the same start, until this page
renders the decision (server-side clock; includes the up-to-250 ms refresh poll; browser paint is not measured).
"""

from __future__ import annotations

import collections
import importlib.util
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))   # works without `pip install`

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from bci import FS_HZ, LATENCY_TARGET_MS
from bci.closed_loop import DEFAULT_SMOOTH_K
from bci.gpu_pipeline import ProcessEngine

REFRESH_S = 0.25           # closed-loop refresh (the pipeline decides once per 256 ms chunk)
EEG_REFRESH_S = 0.5        # the EEG figure is heavier: refresh it half as often
DISPLAY_DECIM = 2          # plot every 2nd sample (125 Hz display of the 250 Hz stream)
CH_SPACING_UV = 60.0       # vertical spacing between stacked channels
LINE = "#2a78d6"
HAS_REAL = importlib.util.find_spec("moabb") is not None
SOURCES = {"Synthetic stream": "synthetic", "Real EEG (BCI Competition IV 2a)": "real"}

st.set_page_config(page_title="Real-time BCI decoder", layout="wide")


@st.cache_resource
def _registry() -> dict:
    return {"key": None, "engine": None}


def get_engine(source: str, subject: int) -> ProcessEngine:
    """One engine process at a time (it trains/warms up before publishing). Changing the source replaces it."""
    reg = _registry()
    key = (source, subject)
    if reg["key"] != key:
        if reg["engine"] is not None:
            reg["engine"].stop(wait=False)
        reg["engine"] = ProcessEngine(source=source, subject=subject, smooth_k=DEFAULT_SMOOTH_K[source]).start()
        reg["key"] = key
    return reg["engine"]


def eeg_figure(raw: np.ndarray) -> go.Figure:
    """Stacked, scrolling view of all channels (x = seconds before now)."""
    n_ch = raw.shape[1]
    fig = go.Figure()
    if raw.shape[0] > 0:
        y = raw[::-1][::DISPLAY_DECIM][::-1].T.astype(np.float64)      # keep the newest sample, decimate backwards
        y = y - y.mean(axis=1, keepdims=True)
        n = y.shape[1]
        dx = DISPLAY_DECIM / FS_HZ
        x0 = -(n - 1) * dx
        for ch in range(n_ch):
            fig.add_trace(go.Scatter(x0=x0, dx=dx, y=np.round(y[ch] + ch * CH_SPACING_UV, 1),
                                     mode="lines", line=dict(width=0.8, color=LINE),
                                     hoverinfo="skip", showlegend=False))
    ticks = list(range(0, n_ch, 8)) + [n_ch - 1]
    fig.update_layout(
        height=520, margin=dict(l=50, r=10, t=10, b=40), showlegend=False, uirevision="eeg",
        xaxis=dict(title="seconds before now", range=[-5, 0], showgrid=True, gridwidth=0.5, zeroline=False),
        yaxis=dict(title="channel (stacked, 60 µV spacing)", tickmode="array",
                   tickvals=[c * CH_SPACING_UV for c in ticks], ticktext=[f"Ch {c + 1}" for c in ticks],
                   range=[-CH_SPACING_UV, n_ch * CH_SPACING_UV], showgrid=False, zeroline=False),
    )
    return fig


def gauge_figure(p99: float | None) -> go.Figure:
    value = 0.0 if p99 is None or np.isnan(p99) else p99
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=value, number=dict(suffix=" ms", valueformat=".2f"),
        title=dict(text="p99 latency (rolling)"),
        gauge=dict(
            axis=dict(range=[0, 2 * LATENCY_TARGET_MS]),
            bar=dict(color=LINE, thickness=0.35),
            steps=[dict(range=[0, LATENCY_TARGET_MS], color="rgba(12,163,12,0.18)"),
                   dict(range=[LATENCY_TARGET_MS, 2 * LATENCY_TARGET_MS], color="rgba(208,59,59,0.18)")],
            threshold=dict(line=dict(width=3), thickness=0.8, value=LATENCY_TARGET_MS)),
    ))
    fig.update_layout(height=260, margin=dict(l=30, r=50, t=50, b=10))
    return fig


def cursor_html(x: float, names: tuple[str, ...], intent: int | None) -> str:
    """A track with the left/right classes at its ends and the cursor at x in [-1, 1]; the end matching the
    current intent (simulated or recorded) is outlined."""
    left = names.index("Left Hand") if "Left Hand" in names else -1
    right = names.index("Right Hand") if "Right Hand" in names else -1
    outline = "outline:2px solid currentColor;"
    return f"""
<div style="position:relative;height:64px;border-radius:32px;background:rgba(128,128,128,0.16);margin:6px 0 2px 0;">
  <div style="position:absolute;left:10px;top:14px;height:36px;padding:0 14px;border-radius:18px;line-height:36px;
              font-size:0.85rem;{outline if intent == left else ''}">&#9664; Left Hand</div>
  <div style="position:absolute;right:10px;top:14px;height:36px;padding:0 14px;border-radius:18px;line-height:36px;
              font-size:0.85rem;{outline if intent == right else ''}">Right Hand &#9654;</div>
  <div id="bci-cursor" data-x="{x:+.2f}" style="position:absolute;top:12px;left:calc(50% + {x * 36:.1f}%);width:40px;
       height:40px;margin-left:-20px;border-radius:50%;background:{LINE};box-shadow:0 0 0 3px rgba(42,120,214,0.25);"></div>
</div>"""


def _pct(values, q) -> float:
    return float(np.percentile(values, q)) if len(values) else float("nan")


@st.fragment(run_every=REFRESH_S)
def live_view(engine: ProcessEngine) -> None:
    snap = engine.snapshot()
    if snap["error"]:
        st.error(f"Pipeline stopped: {snap['error']}")
        return
    if snap["starting"]:
        st.info("Starting the engine process: training the classifier and warming up the pipeline "
                "(real EEG downloads the dataset on first use)...")
        return
    names = tuple(snap["class_names"])
    pct, s2a, cur = snap["percentiles"], snap["s2a"], snap["cursor"]

    # sample -> render: first time this page sees a new decision
    render = st.session_state.setdefault("render_ms", collections.deque(maxlen=300))
    if cur is not None and cur["decision_index"] != st.session_state.get("seen_decision", -1):
        st.session_state["seen_decision"] = cur["decision_index"]
        render.append((time.time() - cur["t_sample_wall"]) * 1e3)

    c1, c2, c3 = st.columns([1.1, 1, 1.1])
    with c1:
        st.markdown("##### Predicted class")
        if snap["warming_up"]:
            st.markdown("## ...")
            st.caption("filling the 2 s decode window")
        else:
            st.markdown(f"## {names[snap['label']]}")
            truth = snap["true_label"]
            st.caption("intent of the newest chunk: " + (names[truth] if truth is not None and truth >= 0 else "none")
                       + f" (the decision averages the last {snap['smooth_k']} chunk(s) of a 2 s window, "
                         "so it trails an intent change by about 2 s)")
            s = snap["scores"]
            st.caption("scores: " + " | ".join(f"{n} {v:+.1f}" for n, v in zip(names, s)))
    with c2:
        st.plotly_chart(gauge_figure(pct["p99"]), key="gauge")
        if pct["n"]:
            ok = pct["p99"] < LATENCY_TARGET_MS
            st.markdown(("✅ within" if ok else "⚠️ above") + f" the {LATENCY_TARGET_MS:.0f} ms target "
                        f"(p99 over the last {pct['n']} chunks)")
    with c3:
        st.markdown("##### Latency, chunk received → label ready")
        last = snap["latency_ms"]
        st.markdown(f"last **{last:.2f} ms** | p50 **{pct['p50']:.2f} ms** | p90 **{pct['p90']:.2f} ms** | "
                    f"p99 **{pct['p99']:.2f} ms**" if last is not None else "waiting for first decode")
        src = ("Real EEG, subject %d, session E replayed" % snap["subject"]) if snap["source"] == "real" else "Synthetic stream"
        st.caption(f"Source: {src} | Device: {snap['device']} ({snap['pipeline']}) | filter: {snap['filter']}")
        st.caption(f"Chunks processed: {snap['frames']} | ring buffer dropped samples: "
                   f"{snap['ring']['dropped_samples']} | ring wraps: {snap['ring']['wrap_count']}")

    st.markdown("##### Closed loop: cursor")
    if cur is None:
        st.caption("waiting for the first decision")
        return
    truth = snap["true_label"]
    st.markdown(cursor_html(cur["x"], names, truth if truth is not None and truth >= 0 else None), unsafe_allow_html=True)
    move = {-1: "pushed left", 0: "no push", 1: "pushed right"}[cur["direction"]]
    agree = snap["agreement"]
    st.caption(f"Cursor x = {cur['x']:+.2f} ({move}); each Left/Right Hand decision pushes by 0.1, feet/tongue do not push, and the position leaks 5% per decision "
               "(time constant ~5 s) so it returns to 0 when the decoder stops pushing. "
               + (f"Decision matched the intent on {100 * agree['rate']:.0f}% of the last {agree['n']} left/right chunks "
                  "(lower right after an intent change: the window lags)." if agree["n"] else ""))
    m1, m2, m3 = st.columns(3)
    m1.metric("Sample-to-action, last", f"{snap['s2a_last_ms']:.2f} ms")
    m2.metric("Sample-to-action p50 / p99", f"{s2a['p50']:.2f} / {s2a['p99']:.2f} ms")
    m3.metric("Sample-to-render p50 / p99 (this page)", f"{_pct(render, 50):.0f} / {_pct(render, 99):.0f} ms" if render else "...")
    st.caption("Sample-to-action: the newest sample of the decision existed → new cursor position computed "
               "(stream hand-over, ring buffer, H2D copy, filter, covariance, feature and classifier stages, D2H, smoothing, controller). "
               "Sample-to-render adds this page's refresh poll (up to 250 ms); browser paint is not measured.")


@st.fragment(run_every=EEG_REFRESH_S)
def eeg_view(engine: ProcessEngine) -> None:
    snap = engine.snapshot()
    if snap["starting"] or snap["error"]:
        return
    n_ch = snap["raw"].shape[1]
    kind = "synthetic" if snap["source"] == "synthetic" else "BCI Competition IV 2a, session E"
    st.markdown(f"##### Raw EEG, {n_ch} channels ({kind})")
    st.plotly_chart(eeg_figure(snap["raw"]), key="eeg")


with st.sidebar:
    st.header("Data source")
    labels = [k for k, v in SOURCES.items() if v == "synthetic" or HAS_REAL]
    choice = st.radio("Source", labels, key="source_choice")
    source = SOURCES[choice]
    subject = st.selectbox("Subject (session E is replayed in a loop)", list(range(1, 10)), key="subject_choice") if source == "real" else 1
    if not HAS_REAL:
        st.caption("Real EEG needs `pip install moabb pylsl` (not installed here); the synthetic stream is used.")

st.title("Real-time BCI decoder")
_decoder = "covariance -> Riemannian tangent space -> LDA" if source == "synthetic" else "filter bank -> CSP log-variance -> LDA"
st.caption(f"64-sample chunks every 256 ms -> ring buffer -> band-pass -> {_decoder} -> cursor. "
           "Not a medical device; not validated on people.")
_engine = get_engine(source, subject)
live_view(_engine)
eeg_view(_engine)
