---
title: Realtime BCI Decoder
sdk: streamlit
app_file: app/dashboard.py
pinned: false
---

# realtime-bci

A real-time, GPU-oriented EEG decoding pipeline on **synthetic** data: a simulated 64-channel
stream goes through a CPU ring buffer into pinned memory, then onto the GPU (two CUDA streams)
for band-pass filtering, covariance, a Riemannian tangent-space map and an LDA classifier. A
Streamlit dashboard shows the raw EEG, the predicted class and a live p99-latency gauge.

## What this is / what this is NOT

**This is** a streaming-systems demonstration: ring buffer, pinned-memory transfers, CUDA
stream overlap, a vectorized `torch.compile`-able filter, CPU/GPU correctness tests, an ONNX
parity check, and a latency benchmark that measures from "chunk received" to "label ready".

**This is not** a hardware-validated BCI. It has never seen a real EEG amplifier or a person.
All data are synthetic, there is no artifact removal, and accuracy on this generator says
nothing about accuracy on real EEG.

## Latency

| | |
|---|---|
| Target | p99 < 50 ms, chunk received → label ready, on an RTX 4060 Laptop GPU |
| **Measured p99 on the RTX 4060** | **Not measured yet.** This repo was written in a container with no GPU. Run `make benchmark`; the number is then in `results/benchmark_summary.csv` (headline run `overlap`, metric `e2e`) and every chunk is in `results/benchmark_latency.csv`. |

Do not read any other number in this repo as the GPU latency. Numbers that exist today were
measured on a CPU and are labelled as such in [FINDINGS.md](FINDINGS.md). Each claim and its
verification status is listed in [CLAIMS.md](CLAIMS.md).

## Dashboard

![dashboard demo](results/dashboard_demo.gif)

*Recorded on a CPU-only machine (the page shows "Device: CPU"). Re-record on the GPU with
`make gif`.* The decoder uses the latest 2 s of data, so its label trails a change of simulated
intent by up to ~2 s.

## Quick start

On a machine with an NVIDIA GPU, **first** install a CUDA build of PyTorch that matches your
driver (selector: <https://pytorch.org/get-started/locally/>). The default PyPI wheel on Windows
is CPU-only, in which case the GPU tests fail (by design) instead of silently skipping.

```bash
make install     # pip install -e ".[dev]" + playwright chromium
make test        # pytest; GPU tests FAIL (not skip) without CUDA. Use `make test REQUIRE_CUDA=0` on CPU.
make benchmark   # 1000 measured chunks -> results/benchmark_latency.csv, benchmark_summary.csv, latency_histogram.png
make dashboard   # streamlit run app/dashboard.py
make profile     # torch.profiler -> results/profiler_trace.json, profiler_overlap.png, profiler_overlap.csv
```

`make` and `bash scripts/run_benchmark.sh` need a POSIX shell (Git Bash or WSL on Windows).
`PACED=1 make benchmark` adds a 1× real-time run (about 4.3 minutes).

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

* **Look-ahead overlap:** the host stages chunk *N* (pinned copy + async H2D) before it launches
  the compute for chunk *N-1*, so the copy for *N* runs while the GPU works on *N-1*. Slot reuse
  is guarded by CUDA events, not host syncs.
* **Hot path has no `.item()`, `.cuda()`, `.cpu()`, `.tolist()` or `print`** — enforced by an AST
  test. One host synchronization is unavoidable: `torch.linalg.eigh` on CUDA synchronizes with
  the CPU (PyTorch documents it). It follows from choosing a Riemannian classifier.
* **Precision:** TF32 is disabled (PyTorch leaves cuDNN TF32 on by default); the filter runs in
  float32, covariance/`eigh`/LDA in float64, to meet `atol=1e-4` against scipy.
* **Filter:** one `conv1d` over all 64 channels (channels folded into the batch axis).
  `torch.compile` is attempted on CUDA; if Triton or a compiler is missing it falls back to eager
  and the result files say `filter_compiled=False`.
* **Dashboard:** the pipeline runs in its own process so Streamlit's rendering cannot stall it
  (see FINDINGS.md).
* **ONNX:** the LDA head is exported and run in onnxruntime. The eigendecomposition step is not
  exportable (`aten::linalg_eigh` unsupported), so the full model is not.

Details and every decision: [SPEC.md](SPEC.md).

## Layout

```
src/bci/    stream.py ring_buffer.py filters.py classifier.py gpu_pipeline.py server.py
app/        dashboard.py            Streamlit UI (Hugging Face Spaces entry point)
tests/      test_ring_buffer.py test_correctness.py test_latency.py test_dashboard.py test_server.py test_profile_analysis.py
scripts/    run_benchmark.sh profile.py record_gif.py ui_contention.py
results/    benchmark_latency.csv, benchmark_summary.csv, latency_histogram.png, profiler_* (written by make benchmark / make profile)
```

## Deploy the dashboard to Hugging Face Spaces

The front-matter at the top of this file declares a Streamlit Space with `app/dashboard.py` as
the entry point; `requirements.txt` lists the runtime dependencies (CPU torch). Push the repo to
a Space; no code changes are needed. A free CPU Space runs the same pipeline on CPU and says
"Device: CPU" on the page. For a GPU Space delete the first line of `requirements.txt` (the CPU
index). Not deployed from here; see CLAIMS.md.

## Limitations

Synthetic data only; no artifact removal; the dashboard assumes an ideal network; the latency
target is defined for an RTX 4060 Laptop GPU and is only a claim once `make benchmark` has run
there. `overlap`/`baseline` benchmark runs keep the GPU busy (optimistic for clock state);
the `paced` run is the realistic one.
