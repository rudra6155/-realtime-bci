# FINDINGS

Every number below is traceable to a committed file; where a number comes from a console log
instead of a CSV, the log is named. Measurements were taken by running the commands in this
repo on the target machine:

* **GPU machine**: NVIDIA GeForce RTX 4060 Laptop GPU (8 GB), Windows, Python 3.13.5, driver
  reporting CUDA 13.4 (`nvidia-smi`), **torch 2.10.0+cu128 (CUDA 12.8), triton-windows 3.6.0,
  onnxruntime-gpu 1.26.0**. The filter ran through `torch.compile` unless a row says eager.
* **CPU-only container** used to author the code: 4 vCPU, torch 2.15.0+cpu, Python 3.13 — only
  sections explicitly labelled CPU.

Per-chunk raw latencies: `results/benchmark_latency.csv` (run `overlap`),
`benchmark_latency_baseline.csv`, `benchmark_latency_paced.csv`. Percentiles:
`results/benchmark_summary.csv` (recomputed independently from the raw files: identical to
< 1e-6 ms). 1000 measured chunks per run after 50 discarded warm-up chunks. Console logs are in
the repo root and `gpu_run/` (PowerShell writes them as UTF-16).

## 1. System architecture

```
SyntheticEEGStream (64 ch, 250 Hz, 64-sample chunks = 256 ms)
   │  Chunk
   ▼
RingBuffer  (pre-allocated float32 (1000, 64); O(1) pointer writes; drop-oldest + warning on overrun)
   │  ring.latest(500, out=pinned_slot)        one memcpy, ring → pinned host memory
   ▼
GPUPipeline  (src/bci/gpu_pipeline.py)
   copy stream   : H2D copy_(non_blocking) → event h2d_end                    … D2H copy_(non_blocking) → event d2h_end
   compute stream: wait h2d_end → FIR band-pass → covariance → whitening → eigh/log → LDA → event cls_end
   │  label + scores in pinned memory after d2h_end
   ▼
LiveEngine (thread) ── ProcessEngine (child process) ──▶ Streamlit dashboard  /  FastAPI WebSocket server
```

Per chunk the host runs `launch_pre(N-1)` → `stage(N)` → `launch_post(N-1)` → `collect(N-1)`:
the filter/covariance/whitening kernels for N-1 are enqueued first (non-blocking), then N's
H2D copy is issued while they run, then the host blocks inside N-1's `torch.linalg.eigh` (a
documented CPU/GPU sync). `n_slots = 3` pinned input slots and events guard reuse in both
directions.

## 2. Latency breakdown (RTX 4060 Laptop GPU)

**Headline: the real-time (`paced`, one chunk every 256 ms) run has p99 = 23.42 ms, under the
50 ms target. The back-to-back runs are 11.22 ms (`overlap`) and 5.19 ms (`baseline`).**
Source: `results/benchmark_summary.csv`. All three runs pass; the test
`test_p99_latency_under_50ms` (back-to-back `overlap` run) passes on the machine
(`gpu_pytest_full2.txt`: 54 passed, 1 skipped).

End-to-end latency, chunk received → label ready (ms):

| run | what it is | p50 | p90 | p99 | max |
|---|---|---|---|---|---|
| `overlap` | back-to-back, 2 streams, pinned, look-ahead | 7.838 | 8.756 | **11.215** | 14.178 |
| `baseline` | back-to-back, 1 stream, pageable, blocking, no look-ahead | 3.857 | 4.341 | **5.186** | 6.625 |
| `paced` | 1× real time, 2 streams, pinned, no look-ahead | 18.611 | 20.653 | **23.417** | 26.967 |

Per stage, GPU-event times (ms), p50 / p90 / p99:

| stage | `overlap` | `baseline` | `paced` |
|---|---|---|---|
| h2d | 0.036 / 0.057 / 0.095 | 0.059 / 0.075 / 0.110 | 0.106 / 0.129 / 0.217 |
| filter (FIR + mean removal) | 0.312 / 0.418 / 0.595 | 0.325 / 0.429 / 0.597 | 0.176 / 0.180 / 0.431 |
| cov (covariance **+ whitening**) | 0.490 / 0.586 / 0.875 | 0.503 / 0.612 / 0.842 | 0.936 / 1.085 / 2.390 |
| classifier (`eigh` + log + LDA) | 2.951 / 3.312 / 4.150 | 2.765 / 3.059 / 3.437 | 16.194 / 16.677 / 20.831 |
| d2h | 0.065 / 0.086 / 0.163 | 0.077 / 0.104 / 0.157 | 0.086 / 0.114 / 0.173 |
| per-chunk wall time | 4.004 | 3.953 | 256.004 (the pacing) |

How to read it:
* In `baseline` and `paced` the stage p50s add up to the end-to-end p50 within ~1 ms
  (3.73 vs 3.86; 17.5 vs 18.6). In `overlap` they add up to only about half of it (3.85 vs
  7.84): look-ahead makes each chunk wait one extra loop iteration (see §3).
* The **classifier stage dominates**: ~2.8–3.0 ms back-to-back (about 75% of the stage total),
  ~16 ms in the paced run.
* The `cov` stage now includes the two whitening matmuls, because the launch was split at
  `eigh` (§3). It is not comparable to the pre-split log `gpu_benchmark.txt`.
* The **paced run is the realistic one**. Its classifier stage is ~5.5× the back-to-back one (16.19 vs 2.95 ms p50)
  even though the profiled kernels take only ~1.8 ms of GPU time per chunk (§3.3), so most of the
  paced classifier time is not GPU execution. The cause is **not isolated**; candidates (not
  tested) are CPU/GPU clock ramp-up after the 256 ms idle gap and Windows (WDDM) scheduling.
  Paced-run kernel durations were not profiled.

### 2.1 Eager vs `torch.compile` filter

`gpu_run/nocompile_after/benchmark_summary.csv` (same stack, `--no-compile`) against the table
above:

| | `overlap` e2e p50 / p99 | `overlap` filter p50 | `baseline` e2e p50 / p99 | wall/chunk (overlap, baseline) |
|---|---|---|---|---|
| compiled | 7.838 / 11.215 | 0.312 | 3.857 / 5.186 | 4.004, 3.953 |
| eager | 8.543 / 12.575 | 0.282 | 4.192 / 5.872 | 4.458, 4.350 |

No difference you can rely on: the filter stage is 0.28–0.31 ms either way, and the other stages
moved by about as much between the two runs (run-to-run noise on this laptop is ~10%). `torch.compile`
does run on this machine (the trace shows two Triton kernels per chunk) but buys nothing measurable
for a 0.3 ms stage.

### 2.2 Variance worth knowing about

* The latency test inside the `pytest` process printed p50 13.79 / p99 17.05 ms once on this exact
  stack (`gpu_pytest2.txt`, `-s` output), against 7.7 / 10.0 ms from the CLI benchmark on the same
  stack (`gpu_benchmark.txt`, same pre-split code). About 1.8×; the cause is **not isolated** (the pytest process also loads onnxruntime's CUDA
  provider; background GPU apps were running). The assertion has ample margin either way, but quote
  CLI-benchmark numbers, not pytest printouts.
* Back-to-back numbers flatter the pipeline: the paced p99 (23.4 ms) is about 2× the `overlap` p99 (11.2 ms) and 4.5× the `baseline` p99 (5.2 ms).

## 3. CUDA stream overlap

Files: `results/profiler_trace.json` (open at <https://ui.perfetto.dev>),
`results/profiler_overlap.png`, `results/profiler_overlap.csv`; 40 chunks profiled, torch.profiler.
`python scripts/profile.py --from-trace results/profiler_trace.json` regenerates the figure and
the numbers without a GPU.

### 3.1 The answer to "how many ms did overlapping save": none

| measure | with overlap design | single-stream blocking baseline |
|---|---|---|
| e2e p50 / p99 (ms) | 7.838 / 11.215 | **3.857 / 5.186** |
| wall time per chunk (ms) | 4.004 | 3.953 |

The look-ahead variant is **slower per chunk (+3.98 ms p50)** and equal in throughput (+0.05 ms).
Reasons, from the data:
* A 500×64 float32 window is 128,000 bytes; the H2D copy takes 11.45 µs on average
  (458.0 µs over 40 chunks) against ~4,000 µs of work per chunk. There is almost nothing to hide.
* The pipeline is launch/sync-bound, not copy-bound (§3.3).
* Look-ahead delays each chunk by one loop iteration to make room for the next chunk's copy, which
  costs far more latency than the copy could ever save.

### 3.2 Before and after the first fix: 0 µs → 7.3 µs of overlap

* First version (`stage(N)` issued before `launch(N-1)`): `h2d_overlapped_with_kernels_us =
  0.0` (`gpu_profile.txt`). The GPU was idle when the copy was issued, because the previous chunk
  had already been fully collected.
* Second version (launch split at `eigh`: `launch_pre(N-1)` → `stage(N)` → `launch_post(N-1)`):
  **7.33 µs of 458 µs H2D time (1.6%) ran concurrently with kernels** (`profiler_overlap.csv`;
  `gpu_profile2.txt`). Exactly one of the 40 copies overlapped anything: its 11.9 µs span
  overlapped the whitening GEMM kernels (4.74 µs a `cutlass` double-precision GEMM + 2.59 µs a
  `cublasLt` split-K reduction). D2H copies overlapped nothing. The zoom panel of
  `results/profiler_overlap.png` shows this.

So the contract "H2D copy runs concurrently with the previous chunk's kernels" is met, but only
barely and with a negligible effect, for the reasons in §3.1. The code and docs say so rather than
claim a speed-up.

### 3.3 What the trace says about where the time goes

Per chunk (40 chunks, `profiler_trace.json`):
* **183 kernels, ~1,794 µs of kernel time.** 90% of that kernel time is in kernels named for
  cuSOLVER's symmetric eigensolver (`sytrd`, `ormtr_*`, `steqr`, `laed*`): the float64
  `torch.linalg.eigh` of the 64×64 whitened covariance. One `sytrd` kernel alone is 782 µs (44%);
  126 small `ormtr` kernels (~3 µs each) follow.
* Extra copies on the **compute stream** every chunk: one device-to-host into pageable memory
  (~1.2 µs), one into pinned memory (~1.9 µs) and three device-to-device (~1.1 µs each). They are
  issued from the classifier stage and are consistent with the documented host sync inside
  `torch.linalg.eigh`; I did not trace them to a specific line. My own D2H of the 16-byte score
  vector (on the copy stream) takes ~1.1 µs.
* The GPU is busy roughly 1.8 ms of a ~4 ms chunk (profiled run is slower than the benchmark
  because of profiler overhead): the rest is host launch/sync time.
* Two Triton (Inductor) kernels per chunk: `torch.compile` really is running.

**Not tried** (candidate ways to cut the classifier stage, each would need its own accuracy and
latency check): `eigh` in float32 (FP64 is a small fraction of FP32 throughput on GeForce GPUs, from
general knowledge, not measured here), a matmul-only iterative matrix logarithm that needs no host
sync, `torch.cuda.CUDAGraph` for the non-`eigh` kernels. The 50 ms target is met without them.

## 4. Other findings

### 4.1 Running the pipeline in the Streamlit process distorts the latency it displays (CPU)

Measured on the CPU container with `scripts/ui_contention.py --device cpu`, paced 1× pipeline
thread, ~58 chunks per condition (p99 ≈ max at this n). A = pipeline alone; B = plus a thread that
builds and serializes a 64-trace Plotly figure every 0.5 s in the same process. Four repetitions:

| rep | A p50 / p99 (ms) | B p50 / p99 (ms) |
|---|---|---|
| 1 | 5.35 / 10.32 | 5.11 / 26.94 |
| 2 | 5.17 / 13.77 | 5.30 / 45.61 |
| 3 | 5.46 / 10.48 | 5.28 / 27.24 |
| 4 | 5.48 / 38.52 | 6.12 / 39.60 |

p50 unaffected; p99 rose 2–3× in three of four repetitions (rep 4's pipeline-alone run had its own
outlier): real but noisy. An earlier ad-hoc run with the dashboard's actual figure builders saw
9 ms vs 174 ms (script not committed: anecdotal). Cause: figure construction (~35 ms) and
serialization (~16 ms) hold the GIL. **Decision:** the dashboard runs the pipeline in a child
process (`ProcessEngine`). The GPU recording of the dashboard (`results/dashboard_demo.gif`) shows a
live paced p99 of 23.87 ms with the UI rendering, consistent with the benchmark's paced 23.42 ms.
Whether GIL contention would matter on the GPU path was not tested.

### 4.2 `torch.linalg.eigh` synchronizes the host on CUDA

From the docstring of torch 2.15.0: "When inputs are on a CUDA device, this function synchronizes
that device with the CPU." There is no `eigh_ex`. The trace in §3.3 shows eigensolver kernels
dominating the GPU time and extra D2H copies on the compute stream. The AST guard (no `.item()`
etc.) cannot see this sync.

### 4.3 TF32 defaults would break the correctness contract

In torch 2.15.0 (CPU build; flag defaults), `torch.backends.cudnn.allow_tf32` is **True** and
`torch.backends.cuda.matmul.allow_tf32` False. TF32 keeps ~10 mantissa bits. `configure_precision()`
disables both. On the 4060 the CUDA-vs-CPU differences are ≤ 1.2e-5 (§4.5).

### 4.4 ONNX

* The full model cannot be exported with the legacy exporter:
  `UnsupportedOperatorError: Exporting the operator 'aten::linalg_eigh' to ONNX opset version 17 is not supported`.
  The `torch.export`-based exporter was not evaluated (it needs `onnxscript`).
* The LDA head exports and runs in onnxruntime. With `onnxruntime-gpu 1.26.0` the session reports
  `['CUDAExecutionProvider', 'CPUExecutionProvider']` and matches PyTorch to 2.8e-14
  (`gpu_pytest2.txt`). I did not verify that the float64 Gemm node itself is placed on the CUDA
  provider rather than falling back to CPU.
* Environment lesson: `onnxruntime-gpu` ≥ 1.27 is built for **CUDA 13** by default; with a CUDA 12.x
  torch the first GPU run failed to load the provider (`cublasLt64_13.dll` missing; the head still
  matched, on the CPU fallback). Fixed by pairing CUDA 12.8 builds: torch 2.10.0+cu128 and
  onnxruntime-gpu 1.26.0 (`pyproject.toml` now pins `<1.27`).

### 4.5 Correctness margins, CUDA vs scipy float64 (RTX 4060)

Max |difference| over 10 windows, `atol = 1e-4` required (`gpu_pytest2.txt`): band-pass 1.17e-5
(values up to 30), covariance 1.32e-7, tangent features 1.41e-7, LDA scores 7.24e-7, whole streaming
path over 23 chunks 1.06e-6 (with and without look-ahead), `torch.compile`'d band-pass 1.03e-5.
The CPU-vs-scipy numbers from the container are essentially identical (the band-pass figure matches
to four digits), which says float32 rounding of the filter output dominates that error.

### 4.6 `torch.compile` and Triton on Windows

First GPU run (torch 2.14.1+cu126): `TritonMissing`, filter fell back to eager and said so
(`filter: eager (compile failed: TritonMissing)`). After installing `triton-windows 3.6.0`
(the version the triton-windows docs pair with torch 2.10; I found no mapping for newer torch) the
compiled filter works and matches scipy to 1.03e-5. The CPU-side compile test is skipped on the
Windows machine because no C++ compiler is on `PATH` (it passes in the Linux container).

### 4.7 Synthetic data

The first generator settings gave 100% held-out accuracy, a trivially separable toy. Difficulty was
raised (`erd=0.45, source_amp=0.3, trial_jitter=0.4`, chosen on seeds 7 and 21); the classifier is
evaluated on different seeds: **0.865** on 200 held-out trials (seed 1234). This characterizes the
generator, not EEG decoding.

### 4.8 Back-to-back vs paced

GPU: `baseline` e2e p50 3.86 ms vs `paced` 18.61 ms on identical code paths (§2). CPU container, same
effect in smaller form (paced ~2× p50). Quote the paced number as the real-time latency.

## 5. Limitations

* **Synthetic data.** Class structure is a two-hemisphere mu/beta amplitude drop; real EEG has
  non-stationarity, subject variability and artifacts.
* **No artifact removal** (eye blinks, muscle, line noise).
* **Ideal network for the dashboard.** The browser is on the same machine; remote latency is not
  modelled. Dashboard latency numbers describe the pipeline, not the network.
* **Single machine, single stack.** Everything GPU-side was measured on one laptop with one
  torch/Triton/onnxruntime combination, with other applications open on the GPU (`nvidia-smi` showed
  browsers, Docker Desktop and others). Run-to-run noise is ~10%, and one pytest printout was 1.8×.
* **Not run anywhere:** Docker build, GitHub Actions, a Hugging Face Space (CLAIMS.md rows C1, C2, C5).
* **The overlap design buys nothing here** (§3.1): it is implemented and measured, not beneficial.
* **Single-chunk window.** One decode window per chunk; no batching across chunks or users.

## 6. Phase 2: real EEG, BCI Competition IV dataset 2a

Scope set by the requester: real-data stream, LSL with a jitter test, and the real-data benchmark. The
closed-loop dashboard is **not** part of this phase. Everything in §6–7 was run in the **CPU-only
container** (torch 2.15.0+cpu, MOABB 1.7.2, MNE 1.13.2, pylsl with liblsl library version 118); the
**GPU re-run of the real-data latency and of the LSL jitter on the RTX 4060 has not been done yet**
(commands at the end).

**Data and protocol** (details in SPEC.md §11). 9 subjects, 2 sessions on different days, 288 trials per
session, 4 classes (left hand, right hand, feet, tongue), 22 EEG + 3 EOG channels, 250 Hz. Per subject:
train on session T (all trials, artifact-flagged ones included, 9 windows per trial), stream session E run
by run as 64-sample chunks through ring buffer + pipeline, score on the experts' artifact-free trials (the
competition's evaluation set). EOG is used only for the noise log. No ICA, no artifact removal.

Cue alignment was **checked, not assumed**: `python -m bci.real_benchmark offset-check --subject 1` gives
5-fold CV accuracy of a 2 s window by start offset after MOABB's class annotation: 0.239 (window −2..0 s,
chance 0.25), 0.292, **0.657** (0..2 s), **0.722** (1..3 s), 0.642, then ~0.5. So the annotation is the cue
(trial t = 2 s) and imagery follows it, as the official description says (cue at 2 s, imagery until 6 s).

### 6.1 Accuracy and Cohen's kappa (4 classes, session T → E, artifact-free trials)

`results/real/accuracy_per_subject.csv`; per-time-bin kappas `results/real/kappa_timecourse.csv`; figure
`results/real/kappa_vs_competition.png`. Fixed window = window ending 3.06 s after the cue (chosen a priori
from subject 1's session-T cross-validation; not tuned on E). Best time point = maximum over the 16-bin
kappa time course (the competition's criterion was "the largest kappa value"; optimistic by construction).

| subject | artifact trials (E) | accuracy (fixed) | kappa (fixed) | kappa (best time point) | at (s after cue) | 2008 winner kappa | left-vs-right accuracy |
|---|---|---|---|---|---|---|---|
| S1 | 7/288 | 0.765 | 0.687 | 0.701 | 2.55 | 0.68 | 0.801 |
| S2 | 5/288 | 0.463 | 0.284 | 0.284 | 3.06 | 0.42 | 0.535 |
| S3 | 15/288 | 0.703 | 0.604 | 0.707 | 2.55 | 0.75 | 0.876 |
| S4 | 60/288 | 0.469 | 0.295 | 0.423 | 2.29 | 0.48 | 0.724 |
| S5 | 12/288 | 0.388 | 0.188 | 0.240 | 2.55 | 0.40 | 0.556 |
| S6 | 73/288 | 0.474 | 0.299 | 0.311 | 2.04 | 0.27 | 0.713 |
| S7 | 11/288 | 0.708 | 0.611 | 0.663 | 2.55 | 0.77 | 0.793 |
| S8 | 17/288 | 0.679 | 0.571 | 0.625 | 2.80 | 0.75 | 0.948 |
| S9 | 24/288 | 0.678 | 0.570 | 0.635 | 2.29 | 0.61 | 0.823 |
| **mean** | 224/2592 | **0.592** | **0.456** | **0.510** | | **0.57** | **0.752** |

All trials (flagged ones included): accuracy 0.590, kappa 0.454 (fixed) / 0.507 (best). Left-vs-right from
the same 4-class scores: accuracy 0.752, kappa 0.506. Chance is accuracy 0.25 and kappa 0.

### 6.2 Comparison with the BCI Competition IV winner (kappa 0.57)

* **We do not beat it.** Mean kappa 0.456 at the fixed window and 0.510 at the best time point, against
  0.57 (Kai Keng Ang et al., filter-bank CSP + Naive Bayes Parzen window). That is 0.11 / 0.06 below. The
  best-time-point figure is close to the competition's second place (0.52; third was 0.31).
* Per subject we match or exceed the winner on S1 and S6 (both metrics) and S9 (best time point only); we are
  well below it on S2, S3, S5, S7, S8. Across subjects the two sets of kappas correlate at r = 0.88: the
  same subjects are easy and hard for both methods.
* **Correction to the brief:** BCI Competition IV was held in **2008** (launched 3 Jul, submissions closed
  21 Nov, results 12 Dec 2008). 2012 is the year of the review article (Tangermann et al., Front. Neurosci.)
  and of the journal version of the winning method. The 0.57 is right (bbci.de results page).
* **The comparison is approximate.** The competition scored a causal, sample-by-sample class output and took
  the best point of the kappa time course; here the output is one decision per 256 ms chunk from a 2 s
  window, one 8–30 Hz band, no filter bank, no temporal smoothing. Both numbers choose the best time point on
  the evaluation data. Both use artifact-free trials, session T → E, subject-specific models.
* The fixed window was not re-selected after seeing E. The mean time course peaks earlier: kappa(t) of
  −0.00 (0.50 s), 0.07 (1.01), 0.25 (1.52), 0.44 (2.04), **0.49 (2.29)**, 0.49 (2.55), 0.46 (3.06), 0.43 (3.57),
  0.38 (4.34). Using 2.29 s would have given 0.49 instead of 0.46, but that is a choice made on E.
* Levers not tried (any would have to be chosen by session-T cross-validation, never on E): a filter bank of
  several bands, voting/smoothing over consecutive chunks, window length, regularization.

### 6.3 Noise log (artifacts logged, not removed)

`results/real/noise_log.csv`: one row per session-E trial (2592), measured over the decision window
(1–3 s after the cue): EEG peak-to-peak (95th percentile over channels), fraction of samples at the
amplifier limit (|x| ≥ 99.9 µV; the recording clips at ±100 µV), share of power above 30 Hz, EOG RMS, the
experts' flag and whether the fixed-window prediction was right.
* Experts flagged 224 of 2592 trials (8.6%); S4 (60) and S6 (73) account for 133 of them.
* Fixed-window accuracy (pooled): **0.522 on flagged vs 0.597 on clean** trials. By peak-to-peak tercile:
  0.615 (37–63 µV), 0.595 (63–78 µV), 0.561 (78–200 µV): noisier trials decode worse, but not dramatically.
* Median EEG peak-to-peak 69.8 µV (95th percentile 112.4); median EOG RMS 9.1 µV (95th 17.6); only 1.4% of
  trials contain clipped samples, so clipping is not the main artifact. Flagged trials have a *lower* share of
  >30 Hz power (median 0.114 vs 0.181 for clean): the flagged artifacts look low-frequency (eye/drift), not
  muscle.
* The pipeline handled all of it without special cases: no NaNs in the data (checked for subjects 1 and 4),
  no numerical failures in the float64 covariance/`eigh` path.

### 6.4 What was run where

Accuracy/kappa/noise: container, CPU device, `python -m bci.real_benchmark accuracy --subjects 1-9 --device cpu`
(about 25 s per subject; the 22-channel pipeline takes ~2 ms per chunk on CPU). The GPU machine should reproduce
these numbers up to float rounding (the correctness tests bound the difference at ≤ 1e-5 on scores).
The latency path (real chunks → LSL → ring buffer → pipeline) ran in the container as a CPU smoke test only
(150 chunks: label latency p50 1.91 / p99 3.72 ms, LSL-timestamp-to-label p99 6.6 ms): **not a GPU number**.

## 7. Phase 2: LSL transport and jitter

`src/bci/lsl_stream.py`: a producer in its own process pushes 64-sample float32 chunks at 1× real time
through a `pylsl` outlet; an inlet pulls them into the ring buffer. **Jitter = standard deviation of
(`t_ring` − producer timestamp)**, with `t_ring` = `local_clock()` right after `RingBuffer.write`, clock offset
corrected by `time_correction()`, inlet post-processing off. The test `tests/test_lsl.py` asserts **< 1 ms**.

Container, loopback, 64-channel synthetic chunks, producer process separate, first 5 chunks excluded:

| measurement | chunks | jitter (std) | mean delay | p50 | p99 | p99−p50 | max | producer schedule std |
|---|---|---|---|---|---|---|---|---|
| `tests/test_lsl.py` (idle CPU) | 50 | **0.240 ms** | 0.976 | 0.885 | 1.611 | 0.726 | 1.704 | 0.062 ms |
| `python -m bci.lsl_stream jitter --seconds 15` | 54 | 0.177 ms | 0.927 | 0.892 | 1.658 | 0.766 | 1.808 | 0.232 ms |
| real recordings (S1 E) with the pipeline running (`real_benchmark latency`, CPU smoke) | 150 | 0.472 ms | 0.832 | 0.768 | 1.581 | 0.813 | 6.279 | 1.335 ms |

* Jitter stays below the 1 ms bar in all three, but the third shows what contention does: with the pipeline
  busy on the same 4-vCPU machine the maximum delay hit 6.3 ms, the producer's own schedule jitter rose to
  1.3 ms and its worst single deviation from the 256 ms schedule was 11.3 ms. A real amplifier on a quiet machine, or the 4060 laptop, may differ in either direction.
* Float32 data crosses LSL **bit-identical** (test), and the pipeline gives the same scores whether chunks come
  straight from the generator or through LSL (`atol = 1e-9`).
* **Loopback on one machine only**: this says nothing about network (Wi-Fi/Ethernet) jitter. Windows LSL
  behaviour and the RTX 4060 laptop were **not** measured yet.

### 7.1 GPU-machine run still to do

```powershell
pip install -e ".[dev]"                       # now includes moabb + pylsl
$env:BCI_RUN_REAL = "1"; python -m pytest tests/test_lsl.py tests/test_real_stream.py -v -s
python -m bci.real_benchmark all --out-dir results\real      # 9-subject accuracy on the GPU + real-data latency/jitter over LSL
```
