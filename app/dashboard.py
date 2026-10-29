"""Live dashboard: simulated 64-channel EEG -> ring buffer -> GPU/CPU pipeline -> label.

Run locally:  make dashboard        (streamlit run app/dashboard.py)
Hugging Face Spaces: Streamlit SDK, ``app_file: app/dashboard.py`` (see README front-matter).
The pipeline runs in its own process (so rendering cannot stall it); this page only renders snapshots.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))   # works without `pip install`

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from bci import CLASS_NAMES, FS_HZ, LATENCY_TARGET_MS, N_CHANNELS
from bci.gpu_pipeline import ProcessEngine

REFRESH_S = 0.5            # UI refresh period (the pipeline itself runs at the 256 ms chunk rate)
DISPLAY_DECIM = 2          # plot every 2nd sample (125 Hz display of the 250 Hz stream)
CH_SPACING_UV = 60.0       # vertical spacing between stacked channels
LINE = "#2a78d6"

st.set_page_config(page_title="Real-time BCI decoder", layout="wide")


@st.cache_resource
def get_engine() -> ProcessEngine:
    """One engine process per server; it trains the classifier and warms up before publishing."""
    return ProcessEngine().start()


def eeg_figure(raw: np.ndarray) -> go.Figure:
    """Stacked, scrolling view of all 64 channels (x = seconds before now)."""
    fig = go.Figure()
    if raw.shape[0] > 0:
        y = raw[::-1][::DISPLAY_DECIM][::-1].T.astype(np.float64)      # keep the newest sample, decimate backwards
        y = y - y.mean(axis=1, keepdims=True)
        n = y.shape[1]
        dx = DISPLAY_DECIM / FS_HZ
        x0 = -(n - 1) * dx
        for ch in range(N_CHANNELS):
            fig.add_trace(go.Scatter(x0=x0, dx=dx, y=np.round(y[ch] + ch * CH_SPACING_UV, 1),
                                     mode="lines", line=dict(width=0.8, color=LINE),
                                     hoverinfo="skip", showlegend=False))
    ticks = list(range(0, N_CHANNELS, 8)) + [N_CHANNELS - 1]
    fig.update_layout(
        height=520, margin=dict(l=50, r=10, t=10, b=40), showlegend=False, uirevision="eeg",
        xaxis=dict(title="seconds before now", range=[-5, 0], showgrid=True, gridwidth=0.5, zeroline=False),
        yaxis=dict(title="channel (stacked, 60 µV spacing)", tickmode="array",
                   tickvals=[c * CH_SPACING_UV for c in ticks], ticktext=[f"Ch {c + 1}" for c in ticks],
                   range=[-CH_SPACING_UV, N_CHANNELS * CH_SPACING_UV], showgrid=False, zeroline=False),
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


@st.fragment(run_every=REFRESH_S)
def live_view(engine: ProcessEngine) -> None:
    snap = engine.snapshot()
    if snap["error"]:
        st.error(f"Pipeline stopped: {snap['error']}")
        return
    if snap["starting"]:
        st.info("Starting the engine process: training the classifier on synthetic trials and warming up the pipeline...")
        return
    pct = snap["percentiles"]

    c1, c2, c3 = st.columns([1.1, 1, 1.1])
    with c1:
        st.markdown("##### Predicted class")
        if snap["warming_up"]:
            st.markdown("## ...")
            st.caption("filling the 2 s decode window")
        else:
            st.markdown(f"## {CLASS_NAMES[snap['label']]}")
            truth = CLASS_NAMES[snap["true_label"]]
            st.caption(f"simulated intent of the newest chunk: {truth} (the decoder uses the latest 2 s, "
                       "so its label trails an intent change by up to ~2 s)")
            s = snap["scores"]
            st.caption(f"LDA scores: {CLASS_NAMES[0]} {s[0]:+.1f} | {CLASS_NAMES[1]} {s[1]:+.1f}")
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
        st.caption(f"Device: {snap['device']} ({snap['pipeline']}) | filter: {snap['filter']}")
        st.caption(f"Chunks processed: {snap['frames']} | ring buffer dropped samples: "
                   f"{snap['ring']['dropped_samples']} | ring wraps: {snap['ring']['wrap_count']}")

    st.markdown("##### Raw EEG, 64 channels (synthetic)")
    st.plotly_chart(eeg_figure(snap["raw"]), key="eeg")


st.title("Real-time BCI decoder")
st.caption("Synthetic 64-channel EEG at 250 Hz, 64-sample chunks -> ring buffer -> band-pass -> covariance -> "
           "Riemannian tangent space -> LDA. Not a medical device; not validated on people.")
live_view(get_engine())
