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

Scope of this phase: real-data stream, LSL with a jitter test, and the real-data benchmark (the closed-loop
dashboard came afterwards). Everything was run twice: in the **CPU-only container** (Linux, torch 2.15.0+cpu,
MOABB 1.7.2, MNE 1.13.2, pylsl with liblsl library version 118) and on the **RTX 4060 laptop** (Windows,
torch 2.10.0+cu128, same MOABB/pylsl; liblsl 1.18.0). Each number below says which.

**Data and protocol** (details in SPEC.md §11). 9 subjects, 2 sessions on different days, 288 trials per
session, 4 classes (left hand, right hand, feet, tongue), 22 EEG + 3 EOG channels, 250 Hz. Per subject:
train on session T (all trials, artifact-flagged ones included, 9 windows per trial), stream session E run
by run as 64-sample chunks through ring buffer + pipeline, score on the experts' artifact-free trials (the
competition's evaluation set). EOG is used only for the noise log. No ICA, no artifact removal.

Cue alignment was **checked, not assumed**: `python -m bci.real_benchmark offset-check --subject 1` gives
5-fold CV accuracy of a 2 s window by start offset after MOABB's class annotation: 0.239 (window −2..0 s,
chance 0.25), 0.292, **0.657** (0..2 s), **0.722** (1..3 s), 0.642, then ~0.5. So the annotation is the cue
(trial t = 2 s) and imagery follows it, as the official description says (cue at 2 s, imagery until 6 s).

### 6.1 Accuracy and Cohen's kappa (4 classes, session T → E, artifact-free trials) — baseline decoder, no smoothing

`results/real/baseline_k1/accuracy_per_subject.csv`; per-time-bin kappas `results/real/baseline_k1/kappa_timecourse.csv`;
figure `results/real/baseline_k1/kappa_vs_competition.png`. The Riemannian decoder with smoothing (§8) is in `results/real/riemann_k4/`, the current default decoder (§9) in `results/real/`. Fixed window = window ending 3.06 s after the cue (chosen a priori
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

* **We do not beat it** (baseline decoder; the one selected afterwards is §8 and is still below). Mean kappa 0.456 at the fixed window and 0.510 at the best time point, against
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
* Levers named here as untried were tried afterwards on session T only, §8: smoothing over consecutive chunks was
  selected; a four-band filter bank and a 3 s window were evaluated but not selected.

### 6.3 Noise log (artifacts logged, not removed)

`results/real/baseline_k1/noise_log.csv` (baseline decoder; `results/real/riemann_k4/noise_log.csv` and `results/real/noise_log.csv` are the
same log for the later decoders, only the `correct_fixed` column differs): one row per session-E trial (2592), measured over the decision window
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

* Accuracy / kappa / noise log: run on both machines with `python -m bci.real_benchmark accuracy|all`
  (container: CPU device; RTX 4060 laptop: CUDA device, filter not compiled). For the baseline decoder the three
  CSVs (`accuracy_per_subject.csv`, `kappa_timecourse.csv`, `noise_log.csv`, now in `results/real/baseline_k1/`) came out **identical**: git recorded
  no change when the GPU-run files were pushed over the CPU-run ones, and the nine per-subject console lines
  match to the printed precision. The GPU machine took 33–79 s per subject (it includes a ~80 MB MOABB
  download per subject), the container ~25 s.
* Real-data label latency and LSL jitter: GPU machine, five runs with the Riemannian decoder, §7.1.

## 7. Phase 2: LSL transport and jitter

`src/bci/lsl_stream.py`: a producer in its own process pushes 64-sample float32 chunks at 1× real time
through a `pylsl` outlet; an inlet pulls them into the ring buffer. **Jitter = standard deviation of
(`t_ring` − producer timestamp)**, with `t_ring` = `local_clock()` right after `RingBuffer.write`, clock offset
corrected by `time_correction()`, inlet post-processing off. The test `tests/test_lsl.py` asserts **< 1 ms**.

Container, loopback, 64-channel synthetic chunks, producer process separate, first 5 chunks excluded:

| measurement | chunks | jitter (std) | mean delay | p50 | p99 | p99−p50 | max | producer schedule std |
|---|---|---|---|---|---|---|---|---|
| container: `tests/test_lsl.py` (idle CPU) | 50 | **0.240 ms** | 0.976 | 0.885 | 1.611 | 0.726 | 1.704 | 0.062 ms |
| container: `python -m bci.lsl_stream jitter --seconds 15` | 54 | 0.177 ms | 0.927 | 0.892 | 1.658 | 0.766 | 1.808 | 0.232 ms |
| container: real recordings (S1 E) with the pipeline running (`real_benchmark latency`, CPU smoke) | 150 | 0.472 ms | 0.832 | 0.768 | 1.581 | 0.813 | 6.279 | 1.335 ms |
| **RTX 4060 laptop, Windows**: `tests/test_lsl.py` | 50 | **0.097 ms** | 0.601 | 0.592 | 0.791 | 0.199 | 0.801 | 0.014 ms |
| **RTX 4060 laptop, Windows**: real recordings (S1 E) with the GPU pipeline running, five runs (`results/real/latency_run0..4/real_latency_summary.csv`; range over runs) | 1000 each | **0.051–0.067 ms** | 0.436–0.477 | 0.429–0.470 | 0.614–0.666 | 0.151–0.233 | 0.863–1.469 | 0.010–0.021 ms |

* Jitter stays below the 1 ms bar in every measurement, but the third shows what contention does: with the pipeline
  busy on the same 4-vCPU machine the maximum delay hit 6.3 ms, the producer's own schedule jitter rose to
  1.3 ms and its worst single deviation from the 256 ms schedule was 11.3 ms. The Windows laptop, with more CPU headroom than the 4-vCPU container, was tighter: 0.05–0.10 ms (std) in the test and all five 1000-chunk runs, with 0 ring-buffer drops (maximum delay 0.86–1.47 ms).
* Float32 data crosses LSL **bit-identical** (test), and the pipeline gives the same scores whether chunks come
  straight from the generator or through LSL (`atol = 1e-9`).
* **Loopback on one machine only**: this says nothing about network (Wi-Fi/Ethernet) jitter or about a real
  amplifier's own timestamp noise. Windows LSL discovery worked without any configuration (the `Stream
  transmission broke off` lines in the log are the producer exiting at the end of the stream).

### 7.1 Real-data label latency over LSL on the GPU (RTX 4060 laptop): five runs

Each run: subject 1's session E, 1000 chunks after 50 warm-up chunks (and the 7 chunks that fill the first
window), arriving at 1× real time through LSL from a separate process into the ring buffer and the pipeline
(2 CUDA streams, pinned buffers, `torch.compile`d filter, 22 channels, 4 classes, **Riemannian decoder** (the only one at the time; the
filter-bank decoder of §9 has not been measured on the GPU)). Raw per-chunk files:
`results/real/latency_run0..4/real_latency_paced.csv`; the percentiles recomputed from all five raw files are
identical to the five `real_latency_summary.csv` (within 1e-5 ms). Run 0 is the first GPU run (before phase 3);
run 1 is the `real_benchmark all` run after phase 3; runs 2, 3 and 4 are `real_benchmark latency`, run 3 and 4
back to back (about 6 min apart by the console clock, which is not stored in the repo). The code that is measured
(`GPUPipeline`, `drive`, `LatencyTracker`, `run_latency`) has no changed line between run 0 and runs 1–4 (git diff).

| run | e2e p50 (ms) | p90 | **p99** | max | chunks > 20 ms | LSL timestamp → label p99 | LSL jitter std (ms) |
|---|---|---|---|---|---|---|---|
| 0 | 7.064 | 8.546 | **10.289** | 12.986 | 0 | 10.709 | 0.053 |
| 1 | 7.142 | 9.877 | **20.133** | 87.658 | 11 | 20.621 | 0.067 |
| 2 | 7.321 | 9.864 | **20.424** | 25.490 | 13 | 20.967 | 0.062 |
| 3 | 7.205 | 9.605 | **25.970** | 49.953 | 19 | 26.543 | 0.051 |
| 4 | 7.261 | 8.163 | **10.227** | 11.518 | 0 | 10.729 | 0.057 |

* **Median stable, tail not.** p50 is 7.06–7.32 ms in all five runs; p99 ranges from 10.2 to 26.0 ms and the maximum
  from 11.5 to 87.7 ms. Run 1 has one chunk above 50 ms (87.7 ms) and run 3's slowest chunk is 49.95 ms; in the
  other three runs no chunk exceeds 26 ms. The p99 stays under the 50 ms target in every run (margin 1.9× to 4.9×).
  Runs 3 and 4 were taken minutes apart and gave 26.0 and 10.2 ms, so this is not a slow drift. The number
  of runs is too small to say whether p99 has two typical values or a spread.
* **Where the slow chunks are** (runs 1–3, 11 / 13 / 19 chunks above 20 ms, i.e. 1.1–1.9% of chunks; runs 0 and 4
  have none): scattered over the run (positions 18–997, at most two neighbouring pairs), not one stall. On those
  chunks the covariance+whitening stage is 4.1–4.8× its normal median (3.4–3.9 ms against 0.79–0.86) and the
  classifier stage 3.4–4.3× (17.1–21.7 ms against 5.0–5.1); the host-to-device copy, filter and device-to-host
  times barely move (h2d +0.01 ms, filter +0.00–0.01 ms median), and the part of the end-to-end time outside the
  five GPU-event stages is 0.9 ms on normal chunks and 1.1–1.8 ms on slow ones. So the extra time is inside the two
  float64 stages, not in the LSL transport or the copies. **The cause was not isolated.** Two candidates, both
  untested: something else intermittently competing for the GPU, or the Python thread being descheduled while it
  enqueues the many small float64 kernels of those two stages (the stage times are CUDA-event times, which also
  include a delayed enqueue).
* **LSL jitter** (std of producer timestamp → ring-buffer delay) is 0.051–0.067 ms in all five runs, far under
  1 ms; the maximum delay was 0.863–1.469 ms (above 1 ms in runs 1 and 4: the bar is on the std); 0 ring drops.
* Stage breakdown of run 0 (GPU-event times), p50 / p90 / p99: h2d 0.066 / 0.125 / 0.202; filter 0.108 / 0.111 / 0.360;
  cov (+whitening) 0.771 / 0.926 / 1.553; **classifier 4.975 / 5.260 / 6.975**; d2h 0.070 / 0.093 / 0.158.
* Compared with the synthetic 64-channel paced run in §2 (e2e p50 18.6 ms, p99 23.4 ms; classifier stage 16.2 ms
  p50) the 22-channel real-data runs have a ~2.6× lower median and a ~3.2× lower classifier median; at p99 the
  real-data runs range from 2.3× lower to slightly higher (26.0 ms). The likely reason for the lower median is the
  smaller eigenproblem (22×22 instead of 64×64 per chunk), but this was not isolated: the runs also differ in
  producer process, subject data and machine state.
* This is **compute latency** (chunk complete → label). The decoder's 2 s window means a label trails the
  user's intent by about two seconds regardless of it.

## 8. Phase 3: closing the kappa gap, and the closed-loop dashboard

*Status: §8.1–8.2 describe the Riemannian decoder, superseded as the default by §9 (filter-bank CSP). The cursor described in §8.3 was a pure integrator; §9.4 replaces it.*

### 8.1 What was tried, and how it was chosen (session T only)

The decoder was changed only through a selection that never touches session E (protocol: SPEC.md §12.2;
`python -m bci.real_benchmark select`, output `results/real/riemann_k4/model_selection_T.csv` and `.log`, 24 candidates, about
4 minutes on the 4-vCPU container with `--workers 4`). Mean kappa over the 9 subjects, leave-one-run-out on session T, at the fixed bin:

| bands / window (samples) | k=1 | k=2 | k=3 | k=4 | k=6 | k=8 |
|---|---|---|---|---|---|---|
| 8–30 Hz, 500 (baseline row) | 0.484 | 0.495 | 0.506 | **0.517** | 0.514 | 0.474 |
| 8–30 Hz, 750 | 0.493 | 0.481 | 0.463 | 0.438 | 0.402 | 0.350 |
| 4 bands, 500 | 0.450 | 0.472 | 0.492 | 0.511 | **0.536** | 0.513 |
| 4 bands, 750 | 0.533 | 0.520 | 0.495 | 0.475 | 0.432 | 0.355 |

The highest mean is 4 bands / 500 / k=6 (0.536), but three other candidates are within one paired standard error of
it (it versus 8–30 Hz/500/k=4: difference 0.019, paired SE 0.030). The pre-declared rule takes the simplest of
those: **one band 8–30 Hz, 500-sample window, causal mean of the last 4 chunk scores** (= the same decoder as
before plus smoothing). So the session-T evidence for the filter bank and for the 750-sample window is that they
are not distinguishable from the simpler decoder, not that they are worse. The causal mean over k=4
chunks adds a group delay of (k−1)/2 = 1.5 chunks = 0.38 s: the label trails the intent longer than before.

### 8.2 Session E, scored once (container, CPU device)

`python -m bci.real_benchmark accuracy --subjects 1-9 --device cpu --smooth-k 4 --decoder riemann` → `results/real/riemann_k4/accuracy_per_subject.csv`,
`kappa_timecourse.csv`, `noise_log.csv`, `kappa_vs_competition.png`, console log `accuracy_k4.log` (these ran before `--decoder` existed; the
default then was the Riemannian decoder). Artifact-free
trials; the baseline columns are `results/real/baseline_k1/`.

| subject | artifact trials (E) | accuracy (fixed) | kappa (fixed) | kappa (best time point) | at (s after cue) | kappa fixed, baseline | kappa best, baseline | 2008 winner kappa | left-vs-right accuracy |
|---|---|---|---|---|---|---|---|---|---|
| S1 | 7/288 | 0.790 | 0.720 | 0.725 | 3.32 | 0.687 | 0.701 | 0.68 | 0.844 |
| S2 | 5/288 | 0.445 | 0.260 | 0.326 | 3.57 | 0.284 | 0.284 | 0.42 | 0.549 |
| S3 | 15/288 | 0.795 | 0.726 | 0.726 | 3.06 | 0.604 | 0.707 | 0.75 | 0.949 |
| S4 | 60/288 | 0.548 | 0.401 | 0.430 | 2.80 | 0.295 | 0.423 | 0.48 | 0.733 |
| S5 | 12/288 | 0.409 | 0.216 | 0.227 | 2.80 | 0.188 | 0.240 | 0.40 | 0.578 |
| S6 | 73/288 | 0.479 | 0.305 | 0.336 | 2.80 | 0.299 | 0.311 | 0.27 | 0.648 |
| S7 | 11/288 | 0.769 | 0.692 | 0.692 | 3.06 | 0.611 | 0.663 | 0.77 | 0.829 |
| S8 | 17/288 | 0.705 | 0.606 | 0.616 | 3.32 | 0.571 | 0.625 | 0.75 | 0.948 |
| S9 | 24/288 | 0.716 | 0.620 | 0.625 | 3.57 | 0.570 | 0.635 | 0.61 | 0.854 |
| **mean** | 224/2592 | **0.629** | **0.505** | **0.523** | | 0.456 | 0.510 | **0.57** | **0.770** |

All trials: accuracy 0.629, kappa 0.506 (fixed) / 0.517 (best). Left-vs-right kappa 0.542.

* **The gap to the 2008 winner is smaller but not closed:** 0.57 − 0.505 = **0.065** at the fixed window (was 0.114)
  and 0.57 − 0.523 = **0.047** at the best time point (was 0.060). At or above the winner on S1, S6 and S9 (both
  metrics); per-subject correlation with the winner's kappas r = 0.91 (was 0.88).
* **Gain over the baseline:** fixed-window kappa +0.049 (paired SE 0.016, better on 8 of 9 subjects; session-T
  cross-validation had predicted +0.033), best-time-point kappa +0.012 (SE 0.007, better on 6 of 9), accuracy
  +0.037. Individual subjects went down (fixed: S2 −0.024; best point: S5 −0.014, S8 −0.010, S9 −0.010). The
  best-time-point gain is small because that metric already picks each subject's best moment, which smoothing
  partly does by itself.
* The mean kappa time course now peaks later and flatter: 0.505 at 3.06 s (the fixed bin) and 0.503 at 2.80 and
  3.32 s, against a baseline peak of 0.494 at 2.29 s; the early part is lower (0.31 at 2.04 s against 0.44). A
  plausible reason (not tested): the mean of the last 4 windows includes windows ending as early as 1.28 s, which
  hold more pre-cue data. The fixed bin was not re-selected on E.
* Flagged trials still decode worse: accuracy 0.589 (224 flagged) vs 0.633 clean (baseline 0.522 vs 0.597).
* **Not done, on purpose:** no further candidate was scored on E, no per-subject tuning, no artifact removal, no
  filter-bank CSP as in the winning method. Nothing here supports claiming the winner is beaten.
* Same caveats as §6.2 about what the 0.57 measures (sample-by-sample causal output, best time point chosen on E).
* **Reproduced on the GPU** (`python -m bci.real_benchmark all` on the RTX 4060 laptop): the nine per-subject console
  lines and the mean lines (accuracy 0.629, kappa 0.505 / 0.523, left-vs-right 0.770 / 0.542) are identical to the
  container's, and git reports no change in `accuracy_per_subject.csv`, `kappa_timecourse.csv` and `noise_log.csv`
  after that run. The PNG was re-rendered with different bytes (not committed).

### 8.3 Closed-loop dashboard

`app/dashboard.py`, engine in `src/bci/closed_loop.py` and `gpu_pipeline.py` (definitions: SPEC.md §12.1). The page
shows the decoded class, a cursor that moves left/right with *Left Hand* / *Right Hand* decisions, the live
agreement with the replayed intent, sample-to-action latency (last, p50, p99) and an estimated sample-to-render
latency, next to the p99 gauge and the raw EEG; a sidebar switches between the synthetic stream and the real
recordings (subject 1–9, session E replayed in a loop).

Checked in the container (CPU device, headless Chromium, `results/closed_loop_synthetic_cpu.png`,
`results/closed_loop_real_cpu.png`): both sources render with no Streamlit exception; `tests/test_dashboard.py` asserts
the cursor takes at least two positions within 12 s and that sample-to-action is displayed and positive;
`tests/test_server.py` runs the process engine on both sources (the real source needs `BCI_RUN_REAL=1`). Full suite:
74 passed, 9 skipped (CUDA-only) in the container; **82 passed, 1 skipped in 150 s on the RTX 4060 laptop** with
`BCI_RUN_REAL=1` and `BCI_REQUIRE_CUDA=1` (the skip is the CPU `torch.compile` test: no C++ compiler on Windows).

Read off the two screenshots (one readout each, CPU, **not a benchmark**): sample-to-action p50 / p99 =
4.66 / 16.40 ms (synthetic, 94 decisions) and 3.05 / 8.90 ms (real subject 1, 153 decisions), sample-to-render
p50 / p99 = 283 / 567–571 ms.

**On the RTX 4060 laptop** (`ProcessEngine(source='real', subject=1, smooth_k=4)`, device line `NVIDIA GeForce RTX 4060
Laptop GPU`, `cuda: 2 streams, pinned, async torch.compile`; 180 s run, statistics over the newest 500 decisions;
one run, console output, no raw file saved):

| (ms) | p50 | p90 | p99 |
|---|---|---|---|
| sample-to-action | 7.580 | 9.369 | 19.173 |
| chunk received → label | 7.218 | 8.910 | 18.635 |

The action step (smoothing, cursor update, hand-over) adds 0.36 ms at p50 and 0.54 ms at p99. An earlier 90 s run
(279 decisions, device not printed) read 7.686 / 11.952 / 20.297 ms. The p99 here is on the high side of the §7.1
range for the same reasons (a few slow chunks decide a p99 over 500 samples); the cause was not isolated.

Two things visible in those screenshots that should not be read as results:
* The cursor sits at +1.00 in both: it is an integrator with step 0.1 that holds on feet/tongue, so after ten
  net right-hand decisions it is pinned until left-hand decisions walk it back. It shows the decoder's
  output direction, not a calibrated control signal.
* The real-data panel shows 52% agreement with the intent over 79 left/right chunks. That window is short, spans
  the lag after each intent change and counts feet/tongue decisions as misses; the evaluated figure is §8.2.

## 9. Phase 4: round-2 decoder selection, cursor leak, tail diagnostic, deployment check

Requested after phase 3: close the four remaining gaps. What could be done in the container was done; what needs the
GPU laptop or the user's accounts is listed in §9.7.

### 9.1 Round 2 of the selection, session T only

Candidates and rule were written into SPEC.md §13.1 before they were computed (`python -m bci.real_benchmark select2`,
output `results/real/model_selection2_T.csv` and `.log`, 36 candidates, 9 subjects, about 7 min on the 4-vCPU container).
Mean kappa over the 9 subjects, leave-one-run-out on session T, at the fixed bin, artifact-free trials:

| family, parameter | k=1 | k=2 | k=3 | k=4 | k=6 | k=8 |
|---|---|---|---|---|---|---|
| R: Riemannian, shrinkage 0.05 | 0.484 | 0.495 | 0.506 | 0.517 | 0.514 | 0.474 |
| R: Riemannian, shrinkage 0.2 | 0.470 | 0.496 | 0.509 | 0.520 | 0.512 | 0.478 |
| R: Riemannian, shrinkage 0.01 | 0.489 | 0.504 | 0.502 | 0.519 | 0.521 | 0.481 |
| F: filter-bank CSP, features 24 | 0.495 | 0.511 | 0.538 | 0.554 | 0.561 | 0.526 |
| F: filter-bank CSP, features 48 | 0.556 | 0.594 | 0.614 | **0.628** | 0.631 | 0.592 |
| H: Riemannian + CSP, CSP features 24 | 0.537 | 0.548 | 0.575 | 0.587 | 0.602 | 0.558 |

* **Sanity check:** the R / 0.05 row reproduces round 1's 8–30 Hz / 500-sample row to three decimals (0.484 … 0.474),
  although the code path is new.
* The best raw candidate is F / 48 features / k = 6 (0.631). Only F / 48 / k = 4 (0.628) is within one
  paired standard error of it, and k = 4 is the simpler, so the declared rule selects **filter-bank CSP, 48
  mutual-information-selected features, mean of the last 4 chunk scores**. The incumbent (R / 0.05 / k = 4, 0.517) is
  not indistinguishable from the best: its difference is 0.114 with a paired SE of 0.048.
* Chosen against incumbent, per subject, in session-T CV: +0.111 mean kappa (paired SE 0.048), better on 7 of 9 subjects.
  The gain is concentrated in S5 (0.15 → 0.59 in CV) and S7 (0.64 → 0.82); S2 is lower (0.36 → 0.32).
* **Caveats.** 36 candidates make the best raw session-T figure optimistic (the winner's family was among them on
  purpose: it is the method of the 2008 winner). The choice was made by the declared rule, not by hand, and session E
  played no part in it. R and F also differ in more than the feature type (nine 4 Hz bands against one 8–30 Hz band,
  151 against 101 taps); this was not separated.

### 9.2 Session E, scored once

`python -m bci.real_benchmark accuracy --subjects 1-9 --device cpu --decoder fbcsp --smooth-k 4` (container, CPU device)
→ `results/real/accuracy_per_subject.csv`, `kappa_timecourse.csv`, `noise_log.csv`, `kappa_vs_competition.png`, console
log `accuracy_fbcsp_k4.log`. Artifact-free trials; the Riemannian columns are `results/real/riemann_k4/` and `baseline_k1/`.

| subject | artifact trials (E) | accuracy (fixed) | kappa (fixed) | kappa (best time point) | at (s after cue) | kappa fixed: Riemannian k=4 | kappa fixed: Riemannian k=1 | 2008 winner kappa | left-vs-right accuracy |
|---|---|---|---|---|---|---|---|---|---|
| S1 | 7/288 | 0.843 | 0.791 | 0.805 | 2.80 | 0.720 | 0.687 | 0.68 | 0.858 |
| S2 | 5/288 | 0.527 | 0.369 | 0.397 | 3.32 | 0.260 | 0.284 | 0.42 | 0.592 |
| S3 | 15/288 | 0.846 | 0.795 | 0.795 | 3.06 | 0.726 | 0.604 | 0.75 | 0.964 |
| S4 | 60/288 | 0.570 | 0.427 | 0.444 | 3.32 | 0.401 | 0.295 | 0.48 | 0.621 |
| S5 | 12/288 | 0.540 | 0.387 | 0.392 | 2.80 | 0.216 | 0.188 | 0.40 | 0.859 |
| S6 | 73/288 | 0.498 | 0.330 | 0.330 | 3.06 | 0.305 | 0.299 | 0.27 | 0.750 |
| S7 | 11/288 | 0.830 | 0.774 | 0.774 | 3.06 | 0.692 | 0.611 | 0.77 | 0.907 |
| S8 | 17/288 | 0.838 | 0.784 | 0.808 | 2.80 | 0.606 | 0.571 | 0.75 | 0.933 |
| S9 | 24/288 | 0.720 | 0.626 | 0.636 | 2.80 | 0.620 | 0.570 | 0.61 | 0.869 |
| **mean** | 224/2592 | **0.690** | **0.587** | **0.598** | | 0.505 | 0.456 | **0.57** | **0.817** |

All trials: accuracy 0.689, kappa 0.585 (fixed) / 0.594 (best). Left-vs-right kappa 0.634.

* **Against the 2008 winner (0.57):** fixed-window kappa 0.587 is +0.017 above it (paired SE over subjects 0.018), best-time-point
  0.598 is +0.028 above (SE 0.017). **That is a match, not a demonstrated win:** both differences are within about
  one to two standard errors, and the winner's figure is one point estimate with no spread. At or above the winner
  on S1, S3, S6, S7, S8, S9 (6 of 9) at both metrics; below on S2, S4, S5. Per-subject correlation with the winner's kappas r = 0.97.
* **Against the previous decoders:** +0.082 over Riemannian k = 4 at the fixed window (SE 0.021, better on 9 of 9),
  +0.075 at the best time point (SE 0.023, better on 8 of 9); +0.130 over the unsmoothed Riemannian decoder (better on 9 of 9).
  Session-T CV had predicted +0.111.
* The mean kappa time course now peaks at the fixed bin: 0.587 at 3.06 s, 0.579 at 3.32 s, 0.573 at 2.80 s (the fixed
  bin was set a priori in phase 2 and not changed).
* Flagged trials still decode worse: accuracy 0.603 (224 flagged) vs 0.697 clean.

### 9.3 What this does and does not show

* **Disclosure.** Round 2 was started because round 1's session-E score (0.505) was below 0.57. Session E therefore
  influenced *whether to try again*, though never *which candidate*: every choice among candidates was made on session T
  with a rule fixed beforehand. Across the project session E has been scored three times, once per decoder
  (Riemannian k = 1, Riemannian k = 4, filter-bank CSP k = 4), and each decoder exactly once.
* The comparison with 0.57 stays approximate (FINDINGS §6.2): the competition scored a causal sample-by-sample output;
  here a decision per 256 ms chunk from a 2 s window plus a 4-chunk mean, and both numbers use a best time point chosen on E.
* One dataset, 9 subjects, replayed recordings, no artifact removal, no person or amplifier.
* The 0.587 / 0.598 come from the container's CPU and **the GPU has not confirmed them**: the accuracy files that were pushed as
  the filter-bank GPU results are the Riemannian decoder's (§10.2). The streaming code is checked against a CPU reference by a test
  that also exists for CUDA; it has not been seen passing on the GPU. Latency on the RTX 4060 was measured once with the filter-bank
  decoder (§10.1: p50 4.08, p99 6.48 ms); on the container's CPU a 200-chunk smoke gave e2e p50 / p99 of 2.16 / 3.75 ms
  (filter-bank) against 2.18 / 16.74 ms (Riemannian), one run each.
* The filter-bank decoder has no eigendecomposition and no host synchronisation. Whether that removes the slow-chunk tail
  of §7.1 (which sat in the float64 covariance and classifier stages) is untested.

### 9.4 Cursor control law

The pinned cursor of §8.3 came from the pure integrator. It now leaks, `x ← clip(0.95·x + push)` (SPEC.md §13.3; the 0.95
was fixed before measuring). `scripts/cursor_sim.py` replays each decoder's session-E decisions, 9 subjects, through both laws
(`results/real/cursor_sim_riemann.csv`, `cursor_sim_fbcsp.csv`):

| decoder | share of decisions at an edge: pure integrator (per-subject range) | leaky | cursor on the intended side during left/right imagery: pure | leaky |
|---|---|---|---|---|
| Riemannian, k = 4 | 43.5% (29–76) | 8.2% (1–23) | 53.0% | 57.0% |
| filter-bank CSP, k = 4 | 40.3% (32–56) | 7.4% (1–12) | 52.3% | 59.7% |

* The leak removes most of the pinning. The cursor is still a **weak control signal**: during left/right imagery it is on the
  intended side about 60% of the time (chance ≈ 50%), because feet and tongue decisions do not push, the 4-class decoder
  spreads its decisions, and the window lags every intent change. It shows the decoder's direction, not calibrated control.

### 9.5 Tail-latency diagnostic

`python -m bci.tail_diag` (SPEC.md §13.4) was built for the GPU laptop. Tested here: the analysis logic, the GC log, the telemetry
no-op without `nvidia-smi`, and a CPU run on a stand-in recording and on real data. Not run on a GPU, so **the cause stays open**
(CLAIMS P24). Runs to do on the RTX 4060, each about 5 minutes:

```
python -m bci.tail_diag --out-dir results\real\tail_diag_baseline
python -m bci.tail_diag --no-gc --out-dir results\real\tail_diag_nogc
python -m bci.tail_diag --priority high --out-dir results\real\tail_diag_highprio
python -m bci.tail_diag --no-telemetry --out-dir results\real\tail_diag_notele
python -m bci.tail_diag --decoder fbcsp --out-dir results\real\tail_diag_fbcsp
```
Reading the summary: a long `host_launch_pre_ms` on slow chunks means the Python thread stalled while enqueuing (scheduling, GIL, GC);
a normal one with slow GPU stages points at the GPU; a drop in `gpu_sm_mhz` points at clocks or throttling; fewer slow chunks with
`--no-gc` or `--priority high` points at those causes. Runs can be tail-free by chance (runs 0 and 4 of §7.1 were), so repeat before concluding.

### 9.6 Deployment checks

* **Docker.** The repo's `Dockerfile` was built **unmodified** and run on CPU. The sandbox needed three workarounds, none in the repo:
  a Docker daemon started inside it (`dockerd`); `python:3.11-slim` pulled from Google's public mirror (`mirror.gcr.io/library/python:3.11-slim`,
  digest `sha256:673719e0…`) because Docker Hub answered 429; and the sandbox proxy's CA added in a local base layer so `pip` could verify TLS. The
  build took about 21 minutes and the image is 10.8 GB (3.6 GB compressed content), mostly the CUDA libraries that PyPI's default `torch` wheel
  pulls in. Run with `--network=host`: `/_stcore/health` answered 200, `torch 2.15.0+cu132` imports with CUDA unavailable, and headless
  Chromium showed the dashboard (synthetic source only, `Device: CPU`, cursor and sample-to-action metrics, no exception). The image held the
  phase-3 code; the final code was checked in the same image with `src` and `app` mounted over the installed copies.
* **Hugging Face path.** `requirements.txt` alone was installed into a clean Python 3.13.17 virtual environment (torch 2.15.0+cpu, numpy, scipy,
  scikit-learn, streamlit, plotly: no fastapi, onnx or MOABB) and `streamlit run app/dashboard.py` served the final dashboard with no exception.
* **Not done:** a real Space (needs the user's Hugging Face account), `docker run --gpus all`, `docker-compose.yml`, and the GitHub Actions workflow
  (never executed). The real-data source is absent from both CPU deployments because MOABB is not in their dependencies; the page says so.

### 9.7 Open after phase 4

1. Still to run on the RTX 4060: the phase-4 test suite (the CUDA variant of the filter-bank equivalence test), the 9-subject accuracy run with the
   filter-bank decoder (`real_benchmark accuracy --decoder fbcsp --device cuda`, see `results/real_gpu_fbcsp/README.md`), and its sample-to-action in the
   dashboard engine. Done once: its real-data latency over LSL (§10.1).
2. The tail diagnostic runs of §9.5 (cause of the slow chunks).
3. A real Hugging Face Space and `docker run --gpus all`.
4. Beyond this project's reach from the container: any real amplifier or participant.

## 10. Intermittent Tail Latency on Windows

### 10.1 Evidence so far (RTX 4060 laptop, real data, 1000 paced chunks per run, chunk received → label)

| run | tool / path | decoder | p50 ms | p99 ms | max ms | chunks > 20 ms |
|---|---|---|---|---|---|---|
| 0 | `real_benchmark` over LSL (`latency_run0`) | Riemannian | 7.064 | 10.289 | 12.986 | 0 |
| 1 | same (`latency_run1`) | Riemannian | 7.142 | 20.133 | 87.658 | 11 |
| 2 | same (`latency_run2`) | Riemannian | 7.321 | 20.424 | 25.490 | 13 |
| 3 | same (`latency_run3`) | Riemannian | 7.204 | 25.970 | 49.953 | 19 |
| 4 | same (`latency_run4`) | Riemannian | 7.261 | 10.227 | 11.518 | 0 |
| 5 | same (`results/real_gpu_fbcsp/`) | filter-bank (by stage profile; not recorded in the files) | 4.078 | 6.477 | 13.962 | 0 |
| 6 | `bci.tail_diag`, in-process stream, no LSL (console paste only) | not recorded (tool default is Riemannian) | 3.450 | not in the paste | not in the paste | 0 slow (threshold 6.90 ms = 2 × median) |

Run 5 in detail (recomputed from its 1000 raw rows, identical to its summary file): e2e p50 / p90 / p99 / max 4.078 / 5.183 / 6.477 / 13.962 ms,
one chunk above 10 ms, none above 20 ms; producer timestamp → label p50 / p90 / p99 / max 4.625 / 5.748 / 7.085 / 15.682 ms; stage medians h2d 0.099, filter 0.585
(banked 9-band FIR), cov 0.961, classifier 1.656 (no `eigh`), d2h 0.083 ms. **LSL jitter (std) 0.0837 ms**, mean delay 0.507 ms, maximum delay 1.780 ms
(above 1 ms; the bar is on the std), 0 ring-buffer drops. The producer's own schedule was less steady than in runs 0–4: std 0.348 ms (earlier 0.011–0.021) and a worst deviation
of 6.0 ms; that moves arrival times, not the compute time measured here.

### 10.2 Two things in the pushed files that must not be misread

* **The median of run 5 is 4.078 ms, not 3.45 ms.** The 3.450 ms median belongs to run 6, a different tool (in-process stream, instrumented calls, nvidia-smi polling) whose decoder was not
  recorded. The two cannot be quoted as one result set. If run 6 used the default (Riemannian) decoder, its median is half of runs 0–4 (7.1–7.3 ms) on the same machine, which would need
  its own explanation (no LSL producer process or inlet thread, the instrumentation, the nvidia-smi polling keeping the GPU awake between paced chunks — none of these tested).
  `bci.tail_diag` now writes the decoder into its summary header.
* **`results/real_gpu_fbcsp/` holds the Riemannian decoder's accuracy files.** Its `accuracy_per_subject.csv`, `kappa_timecourse.csv` and `noise_log.csv` are identical to `results/real/riemann_k4/` apart
  from line endings (kappa 0.505 / 0.523, accuracy 0.629), whereas the filter-bank decoder scored 0.587 / 0.598 in the container. So the filter-bank decoder's accuracy has **not** been checked on the GPU;
  `results/real_gpu_fbcsp/README.md` says which files are which and gives the command that produces the missing result.

### 10.3 What can and cannot be said

* The tail was present in 3 of 5 Riemannian runs (p99 20.1–26.0 ms, 1–2% of chunks 4× slower in the covariance and classifier stages, §7.1) and absent in 2 of 5. It was absent in both later
  runs (5 and 6), which used different tools and, for run 5, a different decoder. Two tail-free runs cannot separate "the decoder changed", "the tool changed" and "luck": runs 0 and 4 were tail-free too.
* **One explanation on the table is transient interference from the operating system** (the Python garbage collector, antivirus real-time scanning, scheduling of other processes). It fits an intermittent,
  scattered tail, but nothing measured so far supports it over the others: the other candidates are another GPU consumer and the host thread being descheduled while it enqueues the float64 kernels
  (§7.1). Run 6 had no slow chunks, so it could neither show nor exclude garbage-collection pauses (the `gc_ms` column only helps when slow chunks occur), and the tool cannot see antivirus activity.
  The statement that these were "not pipeline bugs" is likewise **not established**: no defect was found, and the evidence does not exclude one.
* What would settle it: several `bci.tail_diag` runs with the Riemannian decoder (the one that showed the tail), with `--no-gc`, `--priority high` and `--no-telemetry` as A/B switches, until slow chunks occur
  (SPEC.md §13.4, §9.5). `--no-telemetry` is the one to run first: it tests whether the polling in run 6 changed the median.
* For the project's latency target none of this changes the outcome: p99 is under 50 ms in all six benchmark runs (10.2–26.0 ms Riemannian; 6.5 ms in the one filter-bank run), so the margin is 1.9× to
  7.7×, and the number to quote is the range, not one run.
