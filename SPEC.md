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
a) scrolling raw EEG, b) predicted class, c) rolling p99-latency gauge. Since phase 3 it also shows the
closed-loop cursor and sample-to-action latency and can replay the real recordings (§12).
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

## 11. Phase 2: real EEG (BCI Competition IV 2a) and LSL

Stop point set by the requester for this phase: **LSL integration and the real-data benchmark only**; the
closed-loop dashboard and the decoder selection that followed are §12. No ICA.

**Dataset.** BCI Competition IV dataset 2a via MOABB 1.7.2 (`BNCI2014_001`, `artifact_handling="annotate"`):
9 subjects, 2 sessions each (`0train` = T, `1test` = E, different days), 6 runs of 48 trials (12 per
class), 22 EEG + 3 EOG channels at 250 Hz, 0.5–100 Hz band-pass plus 50 Hz notch, experts' artifact flag
per trial. Trial timing (official description): fixation at t = 0, cue at t = 2 s, motor imagery until
t = 6 s. In MOABB's raw data each class annotation starts at the **cue** and lasts 4 s (verified
empirically: decoding is at chance for windows before the annotation and peaks for windows starting
1 s after it). Classes: left hand, right hand, feet, tongue.

**Protocol (subject-specific, session T → session E, as in the competition).**
* Decoder: the existing pipeline (FIR 8–30 Hz → covariance → tangent space → LDA), 22 channels, 4 classes,
  trained per subject on all session-T trials (artifact-flagged trials included; no artifact removal).
  Training windows: 500 samples (2 s) starting 0.0, 0.25 … 2.0 s after the cue (9 per trial, all inside the
  imagery period). This is the **baseline** decoder (no temporal smoothing); §12.2 selects the smoothing used
  afterwards. The EOG channels are **not** used for decoding (the competition forbids it); they are
  used only for the noise log.
* Evaluation: session E recordings are streamed run by run as 64-sample chunks through a fresh ring buffer
  and the same `GPUPipeline` used for the latency work (CPU device when no GPU); every chunk gets a label
  from its latest 500-sample window.
* **Fixed-latency metric (the deployable number):** the prediction of the chunk whose window ends 3.06 s
  after the cue (bin 10 of the 256 ms grid that starts 0.5 s after the cue; window = 1.06–3.06 s after the
  cue). Chosen from session-T cross-validation of subject 1 (decoding peaks for windows starting ~1 s after
  the cue) and fixed for every subject; no per-subject tuning on E.
* **Peak metric (competition-style):** kappa as a function of time after the cue (one point per 256 ms
  chunk, over artifact-free trials); the maximum over time. The competition ranked entries by "the largest
  kappa value" of this time course, so this is the like-for-like comparison with 0.57, and it is
  optimistic by construction (the best time point is chosen on the evaluation data).
* Both metrics are reported on artifact-free trials (the competition's evaluation set) and on all trials.
  Also reported: 2-class (left vs right) accuracy/kappa from the same 4-class scores.
* Kappa = unweighted Cohen's kappa (`sklearn.metrics.cohen_kappa_score`).

**Comparison target.** Kai Keng Ang et al. won dataset 2a with filter-bank CSP: mean kappa **0.57**
(per subject 0.68, 0.42, 0.75, 0.48, 0.40, 0.27, 0.77, 0.75, 0.61; bbci.de results page). Correction to
the brief: BCI Competition IV ran in **2008** (results announced 12 Dec 2008); 2012 is the year of the
review article and of the journal version of the winning method. Differences that make the comparison
approximate: the competition scored a causal sample-by-sample output, we score 256 ms chunk decisions;
their classifier saw the whole recording history causally, ours a 2 s window.

**Noise log.** For each trial, over the decision window: EEG peak-to-peak (95th percentile over channels,
µV), fraction of EEG samples with |x| > 100 µV, EOG RMS (µV), fraction of power above 30 Hz, and the experts'
artifact flag. Written to CSV; accuracy is also split by flag. Artifacts are logged, not removed.

**LSL.** `pylsl` outlet in a **separate process** streaming 64-sample float32 chunks at 1× real time; an
inlet in the consumer process pulls chunks into the ring buffer. Producer timestamp = `local_clock()` at
push. Per chunk the consumer records `t_pull` and `t_ring` (after `ring.write`). Definitions:
* **Transport delay** `d = t_ring − timestamp (+ time_correction)`; **jitter = standard deviation of d**.
  The test asserts jitter < 1 ms. Also reported: p99 − p50 of d, max, and the producer's schedule jitter
  (std of timestamp interval − 256 ms).
* Same machine (loopback) only: this says nothing about network jitter. Inlet post-processing is off
  (`processing_flags = 0`), so liblsl's de-jitter smoothing does not flatter the measurement.

**Real-data latency.** A paced (1× real time) run of 1000 chunks of real recordings arriving **through LSL**
into the ring buffer and the GPU pipeline, so the p99 label latency and the LSL jitter are measured in the
same run. Run on the GPU machine.

**Dependencies.** `moabb`, `mne` (pulled by moabb) and `pylsl` as the `real` extra. The synthetic path does
not import them.

## 12. Phase 3: closed-loop dashboard and decoder selection

Requested after phase 2: (1) the closed-loop dashboard, (2) close the kappa gap using only changes selected
on **session T**. Checking the Docker / Hugging Face deployment was deferred by the requester.

### 12.1 Closed loop

* **Sources.** `synthetic` (2 classes, 64 channels, as before) or `real`: BCI Competition IV 2a session E of a
  chosen subject, replayed at 1× real time through the same ring buffer and `GPUPipeline`, with the decoder
  trained on that subject's session T. The ring buffer is reset at every run start (a window never spans two
  runs) and the replay restarts after the last run, so the loop is a replay of recordings, not a continuous
  recording.
* **Decision.** One per 256 ms chunk once the ring holds the 500-sample window. With smoothing `k` the score
  vector is the mean of the last `k` chunk score vectors (causal; reset at run starts); the decision is its argmax.
  Default `k`: 1 for synthetic, 4 for real (§12.2; `DEFAULT_SMOOTH_K`).
* **Cursor.** Position `x` in [−1, 1], starts at 0, and is a pure integrator: each decision of *Left Hand*
  subtracts 0.1, *Right Hand* adds 0.1, any other class (feet, tongue) leaves it unchanged; clipped at ±1.
* **Sample-to-action latency.** From the instant the newest sample used by the decision existed (the chunk's
  completion time, `Chunk.t_emit_ns`) to the instant the new cursor position has been computed. It contains
  stream hand-over, ring-buffer write, H2D copy, filter, covariance, `eigh`, classifier, D2H copy, smoothing and
  the controller update. It does **not** contain the window length (a label trails the user's intent by about
  the 2 s window plus the smoothing delay), the browser, or the screen.
* **Sample-to-render** is estimated on the server (action time → the moment the page's poll reads it); it adds
  the poll period (≤ 250 ms). Browser paint is **not** measured.
* **Intent agreement** (live indicator, not an evaluation metric): share of the last up-to-100 chunks whose true
  intent is left or right hand where the decision equals the intent. Chunks just after an intent change count
  too, so it understates steady-state accuracy.

### 12.2 Decoder selection on session T only

`python -m bci.real_benchmark select` never loads session E.
* **Candidates (24):** bands = {one band 8–30 Hz; four bands 6–10, 10–14, 14–22, 22–32 Hz with the tangent
  features concatenated} × window = {500, 750 samples} × smoothing `k` = {1, 2, 3, 4, 6, 8} chunks. Training
  windows start every 0.25 s from the cue and end inside the 4 s imagery period (9 starts for 500 samples,
  5 for 750).
* **Score:** leave-one-run-out cross-validation over the 6 runs of session T (LDA, automatic shrinkage, fitted
  on 5 runs); Cohen's kappa of the decision at the fixed bin (3.06 s after the cue) on the artifact-free
  trials; mean over the 9 subjects. The tangent-space reference mean is computed once from one window per
  trial of **all** session-T runs (no labels, but it includes the held-out run's windows); the deployed `fit`
  uses every training window.
* **Rule** (written into `select_config` before the 24-candidate table was computed): take the candidate with the
  highest mean kappa; every candidate whose mean paired per-subject difference to it is ≤ 1 paired standard
  error is "indistinguishable"; among those choose the simplest in the order (number of bands, window
  length, `k`).
* An earlier exploratory pass (scratch scripts, not in the repo) screened more band layouts and covariance
  shrinkage values on session T only; the list above is what was kept after looking at those session-T
  numbers, so the session-T figure of the chosen candidate is optimistically biased by that screening.
  Session E played no part.

### 12.3 Session-E evaluation of the selected decoder

One run over the 9 subjects with the protocol of §11 and the selected smoothing; the fixed bin is unchanged.
The phase-2 baseline (no smoothing) is kept in `results/real/baseline_k1/` for comparison. No other
configuration was evaluated on session E.
