---
title: Realtime BCI Decoder
sdk: streamlit
app_file: app/dashboard.py
pinned: false
---

# Real-Time GPU-Accelerated BCI Decoding Pipeline

A high-performance, real-time Brain-Computer Interface (BCI) decoding pipeline optimized for low-latency streaming inference on consumer GPUs. Built with PyTorch, CUDA streams, Lab Streaming Layer (LSL), and Streamlit.

The pipeline ingests multi-channel EEG into a lock-free CPU circular buffer, stages windows into pinned host memory, and dispatches across asynchronous dual CUDA streams for vectorized FIR bandpass filtering, spatial covariance estimation, Riemannian tangent-space projection / Filter Bank Common Spatial Patterns (FBCSP), and LDA classification—delivering end-to-end decisions in **sub-25 ms p99 latency**.

---

## Key Highlights

- **Sub-25 ms Latency SLA**: Achieves **p99 end-to-end latency of 23.4 ms** on an NVIDIA RTX 4060 Laptop GPU under strict real-time pacing (256 ms chunks @ 250 Hz), comfortably exceeding the <50 ms target.
- **Asynchronous Dual CUDA Streams**: Concurrent memory transfers (copy stream) and kernel execution (compute stream) using pre-allocated pinned memory slots and CUDA events for synchronization.
- **Vectorized Signal Processing**: 101-tap FIR filter bank accelerated via `torch.compile` (channel axis folded into batch dimension; zero Python loops on hot path).
- **Competitive Decoding Accuracy**: Evaluated on the gold-standard **BCI Competition IV 2a** dataset (9 subjects, 4-class motor imagery via MOABB), achieving **0.587 fixed-window kappa (0.598 peak kappa)**, matching the 2008 world champion benchmark (Ang et al., 0.57 kappa) under streaming conditions.
- **Clinical Transport & Low Jitter**: Native **Lab Streaming Layer (LSL)** support with inter-process streaming jitter < 0.08 ms.
- **Closed-Loop Feedback**: Live Streamlit biofeedback interface featuring real-time intent classification, rolling p99 latency telemetry, and a leaky-integrator cursor control loop.

---

## System Architecture

```
[ LSL Inlet / Synthetic Stream ]
            │  (64 channels, 250 Hz, 64-sample chunks = 256 ms)
            ▼
┌────────────────────────────────────────────────────────┐
│ CPU Circular Ring Buffer (O(1) pointer writes, 4s buf) │
└────────────────────────────────────────────────────────┘
            │  latest 500 samples (2s window) copied into pinned memory
            ▼
┌────────────────────────────────────────────────────────────────────────┐
│ GPU Streaming Pipeline (Double-Buffered Async CUDA Streams)            │
│                                                                        │
│  Copy Stream   : Pinned H2D ─── CUDA Event ──┐                         │
│                                              ▼                         │
│  Compute Stream: FIR Bandpass (torch.compile) ─▶ Covariance Shrinkage  │
│                  └▶ Tangent Space Projection / FBCSP ─▶ LDA Classifier │
│                                              │                         │
│  Output        : Async D2H ◀── CUDA Event ───┘                         │
└────────────────────────────────────────────────────────────────────────┘
            │  decisions + probabilities
            ▼
┌────────────────────────────────────────────────────────┐
│ Closed-Loop Engine & Live Dashboard (Streamlit UI)     │
│ - Leaky Integrator Cursor Control (decay λ = 0.95)     │
│ - Rolling p99 Latency & Decision Telemetry             │
└────────────────────────────────────────────────────────┘
```

---

## Measured Performance & Benchmarks

### 1. End-to-End Latency (NVIDIA GeForce RTX 4060 Laptop GPU)
Evaluated over 1,000 continuous chunks (after 50 warm-up iterations) on Windows 11 with PyTorch 2.10.0+cu128:

| Benchmark Run (1,000 chunks) | Mode | p50 | p90 | **p99** | Max |
|---|---|---|---|---|---|
| **`paced` (Real-Time Rate)** | 1 chunk / 256 ms, 2 CUDA streams, pinned memory | 18.61 ms | 20.65 ms | **23.42 ms** | 26.97 ms |
| **`overlap` (Pipelined)** | Back-to-back, look-ahead stage/compute | 7.84 ms | 8.76 ms | **11.22 ms** | 14.18 ms |
| **`baseline` (Synchronous)** | Back-to-back, single stream, pageable memory | 3.86 ms | 4.34 ms | **5.19 ms** | 6.63 ms |

*All raw per-chunk timings, summary percentiles, and latency distributions are recorded in `results/benchmark_summary.csv` and `results/latency_histogram.png`.*

### 2. Real EEG Decoding (BCI Competition IV 2a)
Tested across all 9 subjects on 4-class motor imagery (Left Hand, Right Hand, Feet, Tongue), streaming Session E recordings through LSL:

| Decoder | Accuracy | Cohen's Kappa (Fixed Window) | Cohen's Kappa (Peak Time) |
|---|---|---|---|
| **Filter-Bank CSP + LDA (4-chunk smoothed)** | **0.690** | **0.587** | **0.598** |
| Riemannian Tangent Space + LDA (4-chunk smoothed) | 0.629 | 0.505 | 0.523 |
| Riemannian + LDA (Unsmoothed baseline) | 0.592 | 0.456 | 0.510 |
| **2008 Competition Champion (Ang et al., FBCSP)** | — | **0.57** | **0.57** |

- **Selection Protocol**: Candidate decoders were evaluated strictly using leave-one-run-out cross-validation on Session T (`select2`), without ever peeking at Session E.
- **Reliability & Noise**: Evaluated on official artifact-free evaluation trials as well as full raw recordings (0.585 / 0.594 kappa with noisy trials included).
- **Transport Stability**: Inter-process LSL stream measured a standard deviation jitter of **0.05–0.08 ms**, well within the 1.0 ms real-time transmission budget.

---

## Live Biofeedback Dashboard

The Streamlit dashboard runs in an isolated process to ensure UI rendering never introduces pipeline contention or frame stalls.

![Dashboard Demo](results/dashboard_demo.gif)

- **Closed-Loop Cursor Dynamics**: Decisions drive a 1D continuous cursor (Left Hand: −0.1, Right Hand: +0.1, neutral classes hold). A leaky integrator (`decay = 0.95`, ~5s time constant) prevents edge pinning.
- **Sample-to-Action Latency**: Measures the delay from the newest sample arrival to cursor position update (**p50: 7.6 ms, p99: 19.2 ms** on RTX 4060).
- **Multi-Source Switching**: Seamless toggle between real-time synthetic test streams and replayed BCI Competition IV 2a patient recordings.

---

## Quick Start

### Prerequisites & CUDA Setup
For GPU acceleration on Windows, ensure a CUDA build of PyTorch is installed first:

```powershell
# 1. Install CUDA PyTorch (CUDA 12.8 wheel)
pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128

# 2. Install repository in editable mode
pip install -e ".[dev]"

# 3. Optional: Windows Triton support for torch.compile
pip install "triton-windows<3.7"

# 4. Install browser binaries for dashboard integration tests
python -m playwright install chromium
```

### Running the System

```bash
# Run test suite (validates ring buffer, numeric equivalence with scipy, and latency bounds)
make test

# Run multi-run latency benchmarks (writes summary and distribution plots to results/)
make benchmark

# Run BCI Competition IV 2a real-data benchmark (downloads dataset via MOABB)
make benchmark-real

# Launch live Streamlit dashboard
make dashboard

# Generate GPU profiler trace (Kineto / PyTorch Profiler)
make profile
```

*On Windows PowerShell, benchmark and test commands can also be run directly:*
```powershell
$env:BCI_REQUIRE_CUDA = "1"; python -m pytest -v -rs
python -m bci.gpu_pipeline benchmark --chunks 1000 --warmup 50 --runs overlap,baseline,paced --out-dir results
python -m streamlit run app\dashboard.py
```

---

## Systems & Engineering Insights

1. **Zero Allocations & Synchronizations on the Hot Path**:
   - The critical path forbids dynamic host-device transfers (`.cuda()`, `.cpu()`), scalar unwraps (`.item()`), Python list conversions (`.tolist()`), or logging statements. This invariant is continuously checked via AST static analysis tests.
2. **Double-Buffered Overlapped Transfers**:
   - By issuing chunk *N*’s asynchronous host-to-device transfer concurrently with chunk *N-1*’s early kernel pipeline, memory latency is hidden behind compute.
3. **Hardware & Eigensolver Synchronization**:
   - Riemannian tangent space mapping relies on eigendecomposition (`torch.linalg.eigh`). On CUDA, `eigh` incurs an internal CPU-GPU synchronization barrier. The Filter-Bank CSP pipeline eliminates eigendecomposition at inference time, avoiding host syncs.
4. **Precision Management**:
   - cuDNN and matrix-multiplication TF32 modes are explicitly disabled to maintain strict numerical parity with double-precision SciPy references (`atol ≤ 1.2e-5`).

Detailed profiling traces, architectural design decisions, and diagnostic analyses are documented in [SPEC.md](SPEC.md) and [FINDINGS.md](FINDINGS.md).

---

## Repository Structure

```
├── app/
│   └── dashboard.py          # Streamlit live biofeedback dashboard
├── src/bci/
│   ├── gpu_pipeline.py       # Core double-buffered CUDA streaming engine
│   ├── ring_buffer.py        # O(1) lock-free circular host buffer
│   ├── filters.py            # Vectorized FIR bandpass filters (torch.compile)
│   ├── classifier.py         # Riemannian tangent-space and LDA classification
│   ├── fbcsp.py              # Filter-Bank Common Spatial Pattern implementation
│   ├── lsl_stream.py         # Lab Streaming Layer (LSL) transport & jitter harness
│   ├── closed_loop.py        # Leaky-integrator cursor control dynamics
│   ├── real_benchmark.py     # BCI Competition IV 2a benchmark & model selection
│   ├── server.py             # FastAPI WebSocket ingestion engine
│   └── tail_diag.py          # Kernel execution vs host scheduling diagnostic
├── tests/                    # Correctness, numerical parity, and latency tests
├── scripts/                  # Profiling, GIF recording, and contention benchmarks
└── results/                  # Benchmark summaries, traces, and validation logs
```

---

## License

MIT License.
