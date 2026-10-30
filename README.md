---
title: Realtime BCI Decoder
sdk: streamlit
app_file: app/dashboard.py
pinned: false
---

# realtime-bci

A real-time, GPU-accelerated EEG decoding pipeline on **synthetic** data: a simulated 64-channel
stream goes through a CPU ring buffer into pinned memory, then onto the GPU (two CUDA streams)
for band-pass filtering, covariance, a Riemannian tangent-space map and an LDA classifier. A
Streamlit dashboard shows the raw EEG, the predicted class and a live p99-latency gauge.

## What this is / what this is NOT

**This is** a streaming-systems demonstration: ring buffer, pinned-memory transfers, CUDA
streams, a `torch.compile`d vectorized filter, CPU/GPU correctness tests, an ONNX parity check,
and a latency benchmark that measures from "chunk received" to "label ready".

**This is not** a hardware-validated BCI. It has never seen a real EEG amplifier or a person.
All data are synthetic, there is no artifact removal, and accuracy on this generator (0.865
held-out) says nothing about accuracy on real EEG.

## Latency (measured)

**p99 = 23.4 ms** end to end, chunk received → label ready, at real-time rate (one 256 ms chunk
every 256 ms), target < 50 ms. NVIDIA GeForce RTX 4060 Laptop GPU, Windows, torch 2.10.0+cu128,
1000 measured chunks after 50 discarded warm-up chunks.

| run (1000 chunks) | p50 | p90 | **p99** | max |
|---|---|---|---|---|
| `paced`: real-time rate, 2 streams, pinned | 18.61 ms | 20.65 ms | **23.42 ms** | 26.97 ms |
| `overlap`: back-to-back, look-ahead | 7.84 ms | 8.76 ms | 11.22 ms | 14.18 ms |
| `baseline`: back-to-back, 1 stream, blocking | 3.86 ms | 4.34 ms | 5.19 ms | 6.63 ms |

Sources: `results/benchmark_summary.csv`, per-chunk raw data `results/benchmark_latency*.csv`,
histogram `results/latency_histogram.png`. Stage breakdown, the profiler trace and everything that
did not go as hoped are in [FINDINGS.md](FINDINGS.md); every claim and how it was verified is in
[CLAIMS.md](CLAIMS.md).

Two things to know before quoting these numbers:
* The classifier stage (a float64 `eigh` plus its documented host sync) is ~75% of the compute
  time. The CUDA stream overlap is real but tiny (7 µs of an 11 µs copy overlapped a kernel) and
  **did not reduce latency**: the look-ahead variant is slower than the plain baseline.
* Back-to-back runs flatter the pipeline; the paced run is the realistic one. Numbers come from one
  laptop with other applications open and vary ~10% run to run.

## Dashboard

![dashboard demo](results/dashboard_demo.gif)

*Recorded on the RTX 4060 laptop: the page shows the device and a live rolling p99.* The decoder
uses the latest 2 s of data, so its label trails a change of simulated intent by up to ~2 s.

## Real EEG (BCI Competition IV 2a) and LSL

The same pipeline decodes the real **BCI Competition IV 2a** motor-imagery dataset (9 subjects, 4 classes,
22 EEG channels, loaded with MOABB) streamed through **Lab Streaming Layer** from a separate producer process.
Per subject: train on session T, stream session E, score on the experts' artifact-free trials.

| | accuracy | Cohen's kappa |
|---|---|---|
| this pipeline, fixed window (deployable) | 0.592 | **0.456** |
| this pipeline, best time point (competition-style) | | 0.510 |
| 2008 competition winner (Ang et al., FBCSP) | | 0.57 |

**It does not beat the winner.** It matches or beats it on S1 and S6, is far below on S2, S3, S5, S7, S8, and the
two sets of per-subject kappas correlate at r = 0.88. The comparison is approximate (the competition scored a
sample-by-sample causal output and took the best time point). BCI Competition IV was held in **2008**.
Artifacts are logged, not removed (no ICA): 8.6% of evaluation trials are expert-flagged and decode at 0.52
accuracy vs 0.60 for clean trials. Detail, per-subject table and caveats: [FINDINGS.md](FINDINGS.md) §6,
figure `results/real/kappa_vs_competition.png`.

**LSL jitter** (std of producer timestamp → ring buffer delay, loopback, separate producer process) was
**0.24 ms** in the test (limit 1 ms), 0.47 ms with the pipeline running on the same machine. These were measured
on Linux; the GPU-machine run of the real-data latency and jitter is still to do (FINDINGS §7.1).

```bash
pip install -e ".[dev]"                       # includes moabb + pylsl
make benchmark-real                           # 9-subject accuracy/kappa, then real-data latency + LSL jitter (GPU) -> results/real/
make test-real                                # also runs the tests that download the dataset (~80 MB per subject)
```

The closed-loop dashboard (cursor, sample-to-action latency) is **not implemented yet**.

## Quick start

On a machine with an NVIDIA GPU, **first** install a CUDA build of PyTorch (the default PyPI wheel
on Windows is CPU-only; GPU tests then fail by design instead of silently skipping). The stack this
repo was measured and tested on (Windows, Python 3.13):

```powershell
pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
pip install -e ".[dev]"
pip install "triton-windows<3.7"     # lets torch.compile run on Windows; without it the filter falls back to eager
python -m playwright install chromium
```

Keep CUDA builds matched: `onnxruntime-gpu` 1.27+ is built for CUDA 13 and cannot load next to a
CUDA 12.x torch (`pyproject.toml` pins the 1.21–1.26 series, built for CUDA 12.8). `triton-windows`
3.6 is the series the triton-windows docs pair with torch 2.10; I found no mapping for newer torch.
On Linux, Triton ships with the CUDA torch wheels.

```bash
make install     # pip install -e ".[dev]" + playwright chromium
make test        # pytest; GPU tests FAIL (not skip) without CUDA. Use `make test REQUIRE_CUDA=0` on CPU.
make benchmark   # overlap + baseline + paced runs, 1000 measured chunks each -> results/benchmark_*.csv, latency_histogram.png (~6 min; SKIP_PACED=1 omits the real-time run)
make dashboard   # streamlit run app/dashboard.py
make profile     # torch.profiler -> results/profiler_trace.json, profiler_overlap.png, profiler_overlap.csv
```

`make` and `scripts/run_benchmark.sh` need a POSIX shell. On Windows PowerShell the same things are:

```powershell
$env:BCI_REQUIRE_CUDA = "1"; python -m pytest -v -rs
python -m bci.gpu_pipeline benchmark --chunks 1000 --warmup 50 --runs overlap,baseline,paced --out-dir results
python scripts\profile.py --out-dir results
python -m streamlit run app\dashboard.py
```

The `paced` run takes about 4.5 minutes. Before benchmarking: plug in AC power and close other
GPU-heavy applications.

## How it works

```
SyntheticEEGStream ──64×64 chunk──▶ RingBuffer (1000×64, O(1) writes)
                                        │ latest 500 samples, copied straight into a pinned slot
                                        ▼
             copy stream:   H2D (non_blocking) ─ event ─┐
             compute stream:                            └▶ FIR band-pass → covariance → eigh/log → LDA ─ event ─▶ D2H
                                                                                                                  │
                                                                              label + scores in pinned memory ◀──┘
```

* **Look-ahead:** the host enqueues chunk *N-1*'s filter/covariance kernels, then stages chunk *N*
  (pinned copy + async H2D), then runs *N-1*'s `eigh` + classifier. Slot reuse is guarded by CUDA
  events, not host syncs. Measured effect: see above (none on latency).
* **Hot path has no `.item()`, `.cuda()`, `.cpu()`, `.tolist()` or `print`**, enforced by an AST
  test. One host synchronization is unavoidable: `torch.linalg.eigh` on CUDA synchronizes with the
  CPU (PyTorch documents it). It follows from choosing a Riemannian classifier.
* **Precision:** TF32 is disabled (PyTorch leaves cuDNN TF32 on by default); the filter runs in
  float32, covariance/`eigh`/LDA in float64. CUDA output matches scipy float64 to ≤ 1.2e-5
  (requirement 1e-4).
* **Filter:** one `conv1d` over all 64 channels (channels folded into the batch axis).
  `torch.compile` is attempted on CUDA; if Triton or a compiler is missing it falls back to eager
  and the result files say `filter_compiled=False`. On the measured machine compiling it made no
  measurable difference.
* **Dashboard:** the pipeline runs in its own process so Streamlit's rendering cannot stall it.
* **ONNX:** the LDA head is exported and run in onnxruntime (CUDA provider). The eigendecomposition
  step is not exportable (`aten::linalg_eigh` unsupported), so the full model is not.

Details and every decision: [SPEC.md](SPEC.md).

## Layout

```
src/bci/    stream.py ring_buffer.py filters.py classifier.py gpu_pipeline.py server.py lsl_stream.py real_benchmark.py
app/        dashboard.py            Streamlit UI (Hugging Face Spaces entry point)
tests/      test_ring_buffer.py test_correctness.py test_latency.py test_dashboard.py test_server.py test_profile_analysis.py test_real_stream.py test_lsl.py
scripts/    run_benchmark.sh profile.py record_gif.py ui_contention.py
results/    benchmark_*.csv, latency_histogram.png, profiler_*, dashboard_demo.gif   (from the GPU run)
            real/  real-data accuracy, kappa time course, noise log, kappa chart (container CPU run)
gpu_run/    extra GPU runs (eager-filter A/B);  gpu_*.txt in the root: console logs of the GPU runs (UTF-16)
```

## Deploy the dashboard to Hugging Face Spaces

The front-matter at the top of this file declares a Streamlit Space with `app/dashboard.py` as the
entry point; `requirements.txt` lists the runtime dependencies (CPU torch). Push the repo to a
Space; no code changes are needed. A free CPU Space runs the same pipeline on CPU and says
"Device: CPU" on the page. For a GPU Space delete the first line of `requirements.txt` (the CPU
index). **Not deployed or tested on Spaces** (see CLAIMS.md C4).

## Limitations

Synthetic data only; no artifact removal; the dashboard assumes an ideal network; one laptop and
one software stack were measured; Docker, CI and Hugging Face deployment were never run.
