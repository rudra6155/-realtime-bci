# SPEC — realtime-bci

Scope statement, written before any code. Anything not listed here is out of scope.

## 1. Goal

Show that an offline EEG-benchmark-style decoder (band-pass → covariance →
Riemannian tangent space → LDA) can run as a **low-latency streaming system**:
simulated 64-channel stream → CPU ring buffer → pinned memory → GPU pipeline →
predicted label → live dashboard.

Target: **p99 latency < 50 ms**, measured from "chunk fully received" to
"classification output ready", on an NVIDIA RTX 4060 Laptop GPU (8 GB,
CUDA 12.x, PyTorch 2.x).

This repo is **not** a hardware-validated BCI and has **not** been tested on
people. All data is synthetic.

## 2. Scope

| Item | Decision |
|---|---|
| Input | Simulated 64-channel EEG at 250 Hz (`stream.py`) |
| Chunk | **64 samples = 256 ms** (the brief says "250 ms, approx 64 samples"; 62.5 samples is not an integer, so 64 is used) |
| Ring buffer | Pre-allocated `numpy` array, shape `(1000, 64)`, `float32` (1000 samples = 4 s) |
| Decode window | Latest **500 samples (2 s)** read from the ring buffer on every chunk |
| Classes | 2: `Left Hand`, `Right Hand` (simulated motor-imagery ERD, see §5) |
| Pipeline | Ring buffer → pinned buffer → H2D (copy stream) → FIR band-pass → covariance → Riemannian tangent space → LDA → D2H → label |
| Classifier | Tangent space at the affine-invariant mean + LDA. Fitted offline on CPU (scipy/sklearn, float64) on synthetic trials |
| Filter | `torch.compile`d vectorized FIR (one `conv1d` over all 64 channels, no Python channel loop) |
| ONNX | The trained LDA **head** is exported and run with onnxruntime; see §6 for why only the head |
| Dashboard | Streamlit (`app/dashboard.py`), self-contained (pipeline runs in-process), Hugging Face Spaces-ready |
| Out of scope | Physical EEG hardware, artifact removal (stated as a limitation), multi-GPU, real subjects |

## 3. Pipeline definition (what "correct" means)

All shapes are for a single window; every stage is also batched over a leading `B` axis.

1. **Band-pass**: 101-tap Hamming-window FIR, 8–30 Hz, `scipy.signal.firwin`, `fs=250`.
   "Valid" convolution of the 500-sample window → 400 samples. Per-channel mean removed.
   CPU reference: `scipy.signal.lfilter` (float64), first 100 outputs dropped. GPU: float32.
2. **Covariance** `C (64×64)`: `Y Yᵀ / (T-1)`, scaled so `trace(C) = 64`, then shrinkage
   `C ← (1-α)C + αI` with `α = 0.05`. Computed in **float64**.
3. **Tangent space**: `M = W C W` with `W = C_ref^{-1/2}` fixed at training time
   (`C_ref` = affine-invariant mean of the training covariances); `L = logm(M)` via
   `eigh`; upper triangle vectorized with off-diagonals × √2 → 2080 features. float64.
4. **LDA head**: `scores = X wᵀ + b`; label = `argmax(scores)`.

**Correctness contract** (`tests/test_correctness.py`): on 10 windows, CPU
(scipy/numpy, float64) vs torch on the available device(s), `torch.allclose(cpu, gpu, atol=1e-4)`
for the filtered signal, the covariance, the tangent features and the LDA scores.
Tests are parametrized over `cpu` and `cuda`; `cuda` cases are skipped when CUDA is absent
(see §8).

**Precision decisions** (each exists to meet `atol=1e-4`, not for speed):
* TF32 is disabled explicitly (`torch.backends.cudnn.allow_tf32 = False`,
  `torch.backends.cuda.matmul.allow_tf32 = False`). PyTorch leaves cuDNN TF32 **on** by
  default, which would degrade the `conv1d` filter.
* Filter in float32; covariance, `eigh`, tangent map and LDA head in float64.
  Whether float64 `eigh` is fast enough on the RTX 4060 is a **measurement still to be
  taken** (`make benchmark`).

## 4. Concurrency and latency design

* **Two CUDA streams**: `copy_stream` (H2D and D2H) and `compute_stream`
  (filter, covariance, classifier). Dependencies between them use CUDA events
  (`wait_event`), not host synchronization.
* **Pinned memory**: `n_slots = 3` pre-allocated pinned input buffers and pinned output
  buffers. Device buffers are pre-allocated. H2D is `copy_(..., non_blocking=True)`.
  No `.cuda()` / `.to(device)` calls in the hot path.
* **Overlap**: per chunk the host runs `launch_pre(N-1)` (enqueue filter → covariance →
  whitening; non-blocking), then `stage(N)` (pinned memcpy + async H2D on the copy stream),
  then `launch_post(N-1)` (`eigh`, which blocks the host, then log → LDA → async D2H). The H2D
  copy for *N* is therefore issued while the GPU is executing *N-1*'s first kernels. A slot is
  reused only after the events guarding it have completed. (A first version staged *N* before
  launching *N-1*; the profiler trace showed 0 µs of overlap because the GPU was idle at that
  moment, see FINDINGS.md.)
* **No `.item()` / `print()` / tensor-to-Python reads in the hot path.** Results are read
  from pinned host buffers after the per-slot "done" event. A test inspects the hot-path
  source (AST) for `.item`, `.cuda`, `.cpu`, `.tolist` and `print`.
* **Known synchronization point**: `torch.linalg.eigh` on CUDA "synchronizes that device
  with the CPU" (PyTorch docs; there is no `eigh_ex`). The Riemannian tangent-space map
  needs a symmetric eigendecomposition, so the compute launch for a chunk blocks the host
  once, inside `eigh`. This is a direct consequence of choosing Riemannian + LDA.
  The launch is split at `eigh` so the copy for the next chunk is enqueued before the host
  blocks there. The profiler trace shows this sync (device-to-host copies on the compute stream).
* **`torch.compile`**: Inductor generates GPU code with Triton. If compile or Triton is
  unavailable (for example, as far as I know, native Windows has no official Triton), the
  filter **falls back to eager** with a logged warning, and the result files record
  `compiled=False`. Neither the compiled nor the eager GPU path can be exercised in the
  authoring environment (no GPU).

### Latency definition

`e2e_ms = t_ready − t_recv`, host `time.perf_counter_ns()`:
* `t_recv`: taken the moment the consumer obtains the completed chunk, **before** the ring
  write (so the ring write is inside the measurement).
* `t_ready`: the per-chunk "done" event has completed and the label is readable from the
  pinned output buffer.

Includes: ring write, window read into pinned memory, launch overhead, H2D, all kernels,
D2H, event wait.
Excludes: synthetic data generation. Per-stage numbers (`h2d`, `filter`, `cov`,
`classifier`, `d2h`) are CUDA-event GPU times from the same run.

Percentiles: `numpy.percentile`, linear interpolation. Warm-up chunks (default 50: compile,
cuDNN/cuBLAS initialization) are run first and **discarded**; the discard count is written
to the summary.

### Benchmark modes

| Run | What it is | Used for |
|---|---|---|
| `overlap` | 1000 chunks back-to-back, copy/compute overlap on | p99 assertion, `benchmark_latency.csv`, histogram |
| `baseline` | 1000 chunks back-to-back, single stream, synchronous pageable copy | the "overlap saved X ms" comparison |
| `paced` | chunks arrive at 1× real time (256 ms apart), no look-ahead | the realistic streaming case (≈ 4.3 min for 1000 chunks) |

`overlap` and `baseline` keep the GPU busy and the CPU never idles, which is optimistic
for GPU clock state compared with a real stream. `paced` removes that optimism. Which run
is quoted in the README is stated next to the number.

## 5. Synthetic data model (`stream.py`)

64 channels laid out on an 8×8 grid. Left- and right-sensorimotor sources each carry mu
(~10 Hz) and beta (~20 Hz) narrow-band activity (AR(2) resonators, state carried across
chunks), mixed through fixed spatial patterns, plus 1/f-like background and white sensor
noise. "Right Hand" imagery attenuates the left-hemisphere sources (and vice versa) —
event-related desynchronization. Trials switch class every 16 chunks. Ground-truth labels
are exposed per chunk. Real EEG is not modeled beyond this; accuracy on this data says
nothing about accuracy on people.

## 6. ONNX

`torch.onnx.export` of the LDA head (`Linear(2080 → 2)`), loaded with onnxruntime
(`CUDAExecutionProvider` when available, else CPU). The tangent-space step uses
`torch.linalg.eigh`; whether it can be exported is checked and the exact outcome recorded
in FINDINGS.md rather than assumed. The hot path uses PyTorch; ONNX is the parity check.

## 7. Dashboard

Streamlit. The engine (stream → ring buffer → pipeline at 1× real time) runs in its **own
process** (`ProcessEngine`, started with `subprocess`, snapshots over an authenticated
localhost connection); a periodic Streamlit fragment polls the latest snapshot and renders:
a) scrolling 64-channel raw EEG, b) predicted class, c) rolling p99-latency gauge.
A separate process is used because building Plotly figures in the same process holds the GIL
and stalls the pipeline thread, which would inflate the very latency the gauge displays
(`scripts/ui_contention.py`; numbers in FINDINGS.md). It shows the device it is running on. On a CPU-only host (e.g. a free Hugging Face Space)
it runs the same code on CPU and says so. Hugging Face metadata lives in the README
front-matter; `requirements.txt` lists runtime dependencies.

## 8. Status vocabulary and test strictness

Every claim in `CLAIMS.md` has one of:
* **VERIFIED** — a command was run in this repo and its real output is cited.
* **PENDING-GPU** — requires the RTX 4060; not run. No number is given.
* **UNVERIFIED** — written but not executed anywhere.

`BCI_REQUIRE_CUDA=1` turns every GPU-only skip into a **failure**. `make test` sets it by
default (`REQUIRE_CUDA ?= 1`) so a green `make test` on your machine means the GPU tests
ran. Where CUDA is absent: `make test REQUIRE_CUDA=0` (CI does this).

## 9. Environments (read before trusting any result)

The code was written in a Linux container with **no NVIDIA GPU**, CPU-only torch 2.15.0; CPU-side
checks (ring buffer, CPU-vs-torch correctness, ONNX on CPU, the pipeline's CPU path, dashboard
rendering) ran there. Everything GPU-side was then run on the target machine: **RTX 4060 Laptop
GPU, Windows, Python 3.13.5, torch 2.10.0+cu128 (CUDA 12.8), triton-windows 3.6.0,
onnxruntime-gpu 1.26.0**. Results, evidence and what is still unverified are in CLAIMS.md; the
GPU numbers are in FINDINGS.md. The README quotes the `paced` (real-time rate) run as the latency
headline; the `overlap` run is what the pytest assertion and `benchmark_latency.csv` use.

Changes forced by the first GPU runs: the launch was split at `eigh` (the first look-ahead ordering
measured 0 µs of overlap); `onnxruntime-gpu` is pinned `<1.27` (its default build targets CUDA 13).

## 10. Interpretations of the brief (flag if wrong)

1. `results/benchmark_latency.csv` holds the per-chunk raw samples of the headline run;
   percentiles go to `results/benchmark_summary.csv` (the brief says "p50/p99 CSV").
2. `make install` runs `pip install -e ".[dev]"` so `make test` has pytest/playwright.
3. Extra files beyond the listed structure: `tests/test_dashboard.py` (test-matrix row 4),
   `tests/test_server.py`, `tests/test_profile_analysis.py`, `tests/conftest.py`,
   `requirements.txt` (HF Spaces), `scripts/record_gif.py` (README GIF),
   `scripts/ui_contention.py` (reproduces the process-isolation finding), `.gitignore`,
   `results/profiler_trace.json`, `results/profiler_overlap.png`,
   `results/profiler_overlap.csv`, `results/benchmark_summary.csv`,
   `results/benchmark_latency_baseline.csv` (and `_paced.csv`).
4. The brief's latency *test* "writes results/benchmark_latency.csv". Here `test_latency.py`
   asserts p99 and `make benchmark` is the single writer of `results/`, so the CSV, summary
   and histogram always come from the same run.
5. The synthetic generator's difficulty (`erd=0.45, source_amp=0.3, trial_jitter=0.4`) was
   tuned on seeds 7/21 so held-out accuracy is not ~100%; tests evaluate on other seeds.
6. The README GIF is recorded on whatever device runs `scripts/record_gif.py`. The committed
   one was recorded on a CPU-only container and says so on screen; re-record on the 4060.
7. Torch on Windows: the default PyPI wheel is CPU-only (as far as I know); install a CUDA
   build first (README says how). With a CPU-only torch the GPU tests fail under
   `make test` rather than silently skipping.
