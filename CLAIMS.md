# CLAIMS — anti-hallucination ledger

One row per claim. **VERIFIED** = a command was run and the observed output is cited.
**UNVERIFIED** = written but not executed anywhere. **OPEN** = a question the data raised that
nobody has answered. Every latency number points to `results/benchmark_latency*.csv` and
`results/benchmark_summary.csv`; the summary was independently recomputed from the raw per-chunk
files (identical to < 1e-6 ms).

Two environments, named in every row where it matters:
* **GPU machine**: RTX 4060 Laptop GPU, Windows, Python 3.13.5, torch 2.10.0+cu128, triton-windows
  3.6.0, onnxruntime-gpu 1.26.0 (console logs `gpu_*.txt` in the repo root are UTF-16).
* **Container**: Linux, 4 vCPU, no GPU, torch 2.15.0+cpu.

All data are synthetic.

## A. GPU machine (RTX 4060)

| # | Claim | Status | Evidence |
|---|---|---|---|
| G1 | **p99 end-to-end latency < 50 ms.** Real-time paced run (1 chunk / 256 ms, 1000 chunks after 50 warm-up): **p99 23.417 ms** (p50 18.611, p90 20.653, max 26.967). Back-to-back `overlap`: p99 11.215 ms. `baseline`: p99 5.186 ms | VERIFIED | `results/benchmark_summary.csv` (rows `paced`/`overlap`/`baseline`, metric `e2e`); raw `results/benchmark_latency_paced.csv`, `benchmark_latency.csv`, `benchmark_latency_baseline.csv` (1000 rows each); console `gpu_benchmark2.txt` |
| G2 | Per-stage p50/p90/p99 (h2d, filter, cov, classifier, d2h) for all three runs, e.g. `overlap` classifier 2.951 / 3.312 / 4.150 ms; `paced` classifier 16.194 / 16.677 / 20.831 ms | VERIFIED | same summary CSV; table in FINDINGS §2. `cov` includes the whitening matmuls (changed when the launch was split) |
| G3 | `results/latency_histogram.png` exists and shows the `overlap` run (n = 1000, p50 7.84, p99 11.22 ms) | VERIFIED | file viewed |
| G4 | Two CUDA streams and pinned staging buffers are used | VERIFIED | `test_two_streams_and_pinned_buffers` passed (`gpu_pytest_full2.txt`); trace shows streams 29 (copy: H2D + score D2H) and 33 (compute) |
| G5 | **Copy/compute overlap exists but is tiny.** 7.328 µs of 457.996 µs of H2D time (1.6%) ran concurrently with kernels; D2H 0 µs. Before the launch was split at `eigh`: 0.0 µs | VERIFIED | `results/profiler_overlap.csv`; `gpu_profile.txt` (before) vs `gpu_profile2.txt` (after); figure `results/profiler_overlap.png`; trace `results/profiler_trace.json` |
| G6 | **The overlap design did not reduce latency.** `overlap` e2e p50 7.838 vs `baseline` 3.857 ms (+3.98 ms); wall per chunk 4.004 vs 3.953 ms (+0.05 ms). There is no "X ms saved" | VERIFIED | `results/benchmark_summary.csv` (`wall_ms_per_chunk` column) |
| G7 | CUDA output matches the scipy float64 reference within `atol=1e-4`: band-pass 1.17e-5, covariance 1.32e-7, tangent features 1.41e-7, LDA scores 7.24e-7, whole streaming path 1.06e-6 (with and without look-ahead) | VERIFIED | `gpu_pytest2.txt` (`-s` output of the `[cuda]` tests); `gpu_pytest_full2.txt` 54 passed |
| G8 | ONNX: with `onnxruntime-gpu 1.26.0` the session lists `CUDAExecutionProvider` and the exported LDA head matches the PyTorch head (max diff 2.84e-14) | VERIFIED for provider availability and numeric match. Whether the float64 Gemm node actually runs on the CUDA provider (vs CPU fallback) was **not checked** | `gpu_pytest2.txt`: `onnxruntime providers: ['CUDAExecutionProvider', 'CPUExecutionProvider']` |
| G9 | `torch.compile` works for the filter on the GPU (Triton present): `filter_compiled=True`, matches scipy to 1.03e-5, two Triton kernels per chunk in the trace. It gives **no measurable latency benefit** (filter p50 0.312 compiled vs 0.282 ms eager; noise ~10%) | VERIFIED | `results/benchmark_summary.csv` (`filter_compiled`), `gpu_run/nocompile_after/benchmark_summary.csv`, `gpu_pytest2.txt`, trace |
| G10 | float64 `eigh` per chunk fits the budget: classifier stage p99 4.150 ms back-to-back, 20.831 ms paced (total paced p99 23.417 ms) | VERIFIED | summary CSV |
| G11 | Where the time goes: per chunk 183 kernels / ~1,794 µs kernel time, 90% of it in cuSOLVER eigensolver kernels (one `sytrd` kernel = 782 µs); extra device-to-host copies on the compute stream consistent with the documented `eigh` host sync | VERIFIED (kernel/copy counts from the trace). That the extra copies come from `eigh` is an inference, not traced to a call | `results/profiler_trace.json`; `python scripts/profile.py --from-trace results/profiler_trace.json` |
| G12 | Dashboard on the GPU shows the GPU device and a live paced p99 of 23.87 ms (114 chunks) | VERIFIED | `results/dashboard_demo.gif` (last frame: "Device: NVIDIA GeForce RTX 4060 Laptop GPU (cuda: 2 streams, pinned, async) \| filter: torch.compile"); `test_dashboard_renders_and_updates` passed on this machine |
| G13 | Full test suite on the GPU machine: **54 passed, 1 skipped** (CPU `torch.compile` test: no C++ compiler on `PATH`) with `BCI_REQUIRE_CUDA=1` | VERIFIED | `gpu_pytest_full2.txt` |
| G14 | The look-ahead stage/launch/collect ordering and event-based slot reuse work on real CUDA streams: streaming test matches the CPU reference over 23 chunks; the 1000-chunk `overlap` run in `test_p99_latency_under_50ms` asserts 0 ring-buffer drops; all three 1000-chunk benchmark runs completed without error | VERIFIED functionally. **No race detector** (e.g. `compute-sanitizer`) was run | `gpu_pytest_full2.txt`, benchmark logs |
| G15 | `pip install -e ".[dev]"` and `playwright install chromium` work on Windows / Python 3.13.5 | VERIFIED (console output pasted by the user in chat; not a committed file) | — |
| G16 | Dependency pairing: ONNX Runtime ≥ 1.27 ships a CUDA 13 build; with a CUDA 12.x torch the provider fails to load (`cublasLt64_13.dll` missing). Pairing torch 2.10.0+cu128, triton-windows 3.6.0 and onnxruntime-gpu 1.26.0 works | VERIFIED | failure in `gpu_pytest.txt`, pass in `gpu_pytest2.txt`; onnxruntime CUDA docs (version table) |
| G17 | `scripts/profile.py` works end to end on a real Kineto trace (event categories `kernel`/`gpu_memcpy`, `args.stream`, "HtoD" names matched) | VERIFIED | `gpu_profile2.txt`, `results/profiler_overlap.csv` |

## B. Container (no GPU)

| # | Claim | Status | Evidence |
|---|---|---|---|
| A1 | Ring buffer: 10.24 s (40 chunks) at 1× real time → 0 dropped samples, 0 overrun events, ≥ 2 wraps, data bit-identical to an unpaced regeneration; wrap-around and `latest()` correct; overrun drops the oldest chunk(s) and logs a warning | VERIFIED (also passes on the GPU machine) | `pytest tests/test_ring_buffer.py -s` → `{'samples_written': 2560, 'wrap_count': 2, 'dropped_samples': 0, 'overrun_events': 0, ...}` (timing fields vary per run) |
| A2 | torch-on-CPU matches scipy float64 within `atol=1e-4` (max: band-pass 1.17e-5, covariance 1.33e-7, tangent 1.40e-7, scores 7.77e-7) and the whole streaming path matches (1.08e-6) | VERIFIED (CPU device) | `pytest tests/test_correctness.py -s -k cpu` |
| A3 | Held-out accuracy of the classifier on **synthetic** data: 0.865 (n = 200, seed 1234; trained on seed 0) | VERIFIED (synthetic only; says nothing about real EEG) | `pytest tests/test_correctness.py -s -k accuracy` → `HELD-OUT ACCURACY (synthetic, n=200, seed=1234): 0.865` |
| A4 | The full model cannot be exported with the legacy ONNX exporter: `aten::linalg_eigh` unsupported. The `torch.export`-based exporter was not evaluated (needs `onnxscript`) | VERIFIED | `UnsupportedOperatorError: Exporting the operator 'aten::linalg_eigh' to ONNX opset version 17 is not supported` (container, torch 2.15.0+cpu) |
| A5 | `torch.compile` fallback: when compilation raises, the filter falls back to eager and reports `compiled=False` | VERIFIED | `test_compile_failure_falls_back_to_eager_and_says_so` |
| A6 | Hot-path source has no `.item/.cuda/.cpu/.tolist/print`; vectorized stages have no Python loops/comprehensions; `stage`/`launch_post` use `non_blocking` copies | VERIFIED (static check; cannot see implicit syncs inside library calls, see G11) | `pytest tests/test_latency.py -k "hot_path or async or loop"` (also passed on the GPU machine) |
| A7 | `torch.linalg.eigh` synchronizes the host on CUDA (documented); no `eigh_ex` | VERIFIED (documentation text of torch 2.15.0; consistent with G11) | `torch.linalg.eigh.__doc__` |
| A8 | PyTorch 2.15.0 flag defaults: `cudnn.allow_tf32 = True`, `cuda.matmul.allow_tf32 = False`; the code disables both | VERIFIED (defaults read from the CPU build) | `python -c "import torch;print(torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32)"` → `True False` |
| A9 | Running a figure-building thread in the pipeline's process raised paced p99 from 10.3–13.8 ms to 26.9–45.6 ms in 3 of 4 CPU repetitions (one alone-run had its own 38.5 ms outlier); p50 unchanged. ~58 chunks per condition | VERIFIED (CPU, noisy, small n) | `python scripts/ui_contention.py --device cpu` ×4; FINDINGS §4.1 |
| A10 | The benchmark CLI **refuses** to write GPU artifacts on a CPU machine (exit 2) and prefixes `--allow-cpu` output with `cpu_smoke_` | VERIFIED | `make benchmark` in the container; `pytest tests/test_latency.py -k cli` |
| A11 | Profiler-analysis arithmetic and the two-panel figure are correct on a hand-built trace; `--from-trace` redraws from a saved trace without a GPU; the script does not shadow stdlib `profile` | VERIFIED | `pytest tests/test_profile_analysis.py` |
| A12 | FastAPI endpoints work with a stub engine; the real `ProcessEngine` runs the pipeline in a child process on CPU (and on the GPU machine) | VERIFIED | `pytest tests/test_server.py` |
| A13 | The dashboard renders headlessly in Chromium with no Streamlit exceptions or page errors and is live (chunk counter increases) | VERIFIED (container on CPU; GPU machine: G12/G13) | `pytest tests/test_dashboard.py` |
| A14 | `make test REQUIRE_CUDA=0` in the container on the final code: **48 passed, 9 skipped** (all 9 skips are CUDA-only cases, each printing the reason; with `BCI_REQUIRE_CUDA=1` they fail instead) | VERIFIED | `make test PYTHON=... REQUIRE_CUDA=0` → `48 passed, 9 skipped, 1 warning in 81.29s` |

## C. Still unverified or open

| # | Item | Status | Note |
|---|---|---|---|
| C1 | `Dockerfile` / `docker-compose.yml` build and run (GPU reservation) | UNVERIFIED | no Docker daemon in the container; not run on the GPU machine either |
| C2 | `.github/workflows/ci.yml` passes on GitHub | UNVERIFIED | never run |
| C3 | The `Makefile` targets and `scripts/run_benchmark.sh` on Windows | UNVERIFIED | the GPU run used the underlying `python -m ...` commands directly (PowerShell); `make test`/`make benchmark` themselves were run only in the container |
| C4 | Hugging Face Spaces deployment (front-matter + `requirements.txt`) | UNVERIFIED | not deployed |
| C5 | A Linux + NVIDIA GPU environment, or a CUDA 13 stack | UNVERIFIED | everything GPU-side ran on one Windows laptop with one stack |
| C6 | **OPEN:** why the paced classifier stage is ~5.5× the back-to-back one (16.19 vs 2.95 ms p50) although the profiled kernels need ~1.8 ms | OPEN | candidates (not tested): clock ramp-up after idle gaps, WDDM scheduling; paced-run kernel durations were not profiled |
| C7 | **OPEN:** why one pytest printout was 1.8× the CLI benchmark on the same stack (13.79 vs 7.7 ms p50, `gpu_pytest2.txt` vs `gpu_benchmark.txt`) | OPEN | candidates: other CUDA libraries loaded in the pytest process, background GPU load |
| C8 | Alternatives to shrink the classifier stage (float32 `eigh`, a matmul-only matrix log, CUDA graphs for the non-`eigh` kernels) | UNVERIFIED | not tried; the 50 ms target is met without them |

## D. Phase 2: real EEG (BCI Competition IV 2a) and LSL

Run in the container (CPU device) unless a row says otherwise. The closed-loop dashboard is **not done**
(stop point set by the requester).

| # | Claim | Status | Evidence |
|---|---|---|---|
| P1 | MOABB 1.7.2 loads BCICIV 2a as expected: subject 1 session E has 6 runs, 288 trials (72 per class), 22 EEG + 3 EOG channels at 250 Hz, EEG std ≈ 10 µV, amplifier limit 100 µV, 7 expert-flagged trials | VERIFIED (S1; S1 and S4 also checked for NaNs: none) | `BCI_RUN_REAL=1 pytest tests/test_real_stream.py::test_real_recording_structure` |
| P2 | The class annotation onset is the **cue**: CV accuracy of a 2 s window is 0.239 (window −2..0 s), 0.292 (−1..1), 0.657 (0..2), 0.722 (1..3), 0.642 (2..4) | VERIFIED (subject 1, session T) | `python -m bci.real_benchmark offset-check --subject 1` |
| P3 | `RealEEGStream`: 64-sample chunks that never straddle a run, labels = class during the imagery period else −1, 1× pacing (4 chunks ≈ 1.02 s) | VERIFIED on a synthetic stand-in recording (no download) | `pytest tests/test_real_stream.py` |
| P4 | **Real-data results**, 9 subjects, session T → E, streamed through the pipeline, artifact-free trials: 4-class accuracy **0.592**, Cohen's kappa **0.456** (fixed window) and **0.510** (best time point); left-vs-right accuracy 0.752 | VERIFIED (container, CPU device; GPU re-run pending) | `results/real/accuracy_per_subject.csv`, `kappa_timecourse.csv`; `python -m bci.real_benchmark accuracy --subjects 1-9 --device cpu` |
| P5 | **Below the 2008 winner's mean kappa of 0.57** (0.11 below at the fixed window, 0.06 at the best time point; ≥ winner on S1 and S6, and S9 at the best time point; r = 0.88 with the winner's per-subject kappas). The comparison is approximate (see FINDINGS §6.2) | VERIFIED | same files; winner's numbers from bbci.de/competition/iv/results (fetched 2026-10-30) |
| P6 | BCI Competition IV took place in **2008** (results 12 Dec 2008); the brief's "2012" is the review/journal-paper year. The competition scored the largest kappa of a time course on artifact-free trials; trials: fixation 0 s, cue 2 s, imagery until 6 s; artifact trials flagged by experts | VERIFIED from sources (not from this repo) | bbci.de/competition/iv/ (dates); bbci.de/competition/iv/desc_2a.pdf (timing, evaluation, artifacts); results page (0.57 and the per-subject table) |
| P7 | Noise log: 2592 trials, 224 (8.6%) expert-flagged; accuracy 0.522 flagged vs 0.597 clean; 0.615 / 0.595 / 0.561 across peak-to-peak terciles; 1.4% of trials contain clipped samples | VERIFIED | `results/real/noise_log.csv` (analysis snippet in FINDINGS §6.3 was run ad hoc, not a repo command) |
| P8 | **LSL jitter < 1 ms** (std of producer-timestamp → ring-buffer delay): 0.240 ms over 50 chunks with an idle CPU (test), 0.177 ms over 54 chunks (CLI), 0.472 ms over 150 chunks while the pipeline ran on the same machine. Loopback only; separate producer process; first 5 chunks excluded | VERIFIED (container, Linux). **Not measured on the Windows GPU machine yet** | `pytest tests/test_lsl.py -s`; `python -m bci.lsl_stream jitter --seconds 15`; `cpu_smoke` run of `real_benchmark latency` |
| P9 | Float32 data crosses LSL bit-identical; the pipeline gives identical scores (atol 1e-9) whether chunks come from the generator or through LSL | VERIFIED | `tests/test_lsl.py` |
| P10 | Real-data **label latency on the GPU** (chunks arriving through LSL) | **NOT YET RUN**. Only a CPU smoke test exists (150 chunks: p50 1.91 / p99 3.72 ms, not a GPU number) | `python -m bci.real_benchmark latency` on the GPU machine |
| P11 | Windows behaviour of `pylsl`/liblsl (discovery, firewall prompts) and of the MOABB download on the GPU machine | UNVERIFIED | |
| P12 | Full suite in the container with the real-data tests enabled: **64 passed, 9 skipped** (all CUDA-only) | VERIFIED | `make test-real REQUIRE_CUDA=0` → `64 passed, 9 skipped in 136.45s` |

## Definition of done

| Item | Status |
|---|---|
| `make test` green (latency < 50 ms and CPU/GPU correctness) | **Met on the GPU machine** with the equivalent `python -m pytest` and `BCI_REQUIRE_CUDA=1`: 54 passed, 1 skipped (G13). The `make` wrapper itself: C3 |
| `benchmark_latency.csv` and `latency_histogram.png` committed | **Met** (G1–G3) |
| Profiler trace showing stream overlap committed | **Met, with the honest finding that the overlap is 1.6% of a ~11 µs copy** (G5, G6) |
| Dashboard runs and is ready for Hugging Face Spaces | Runs locally on CPU and GPU (A13, G12); the Spaces deployment itself is untested (C4) |
| CLAIMS.md all VERIFIED | **Not literally**: C1–C5 are UNVERIFIED and C6–C8 are open, each stated above |

## GPU-machine re-run of the final code

Commit `7c23d61` (pulled and run on the GPU machine): **56 passed, 1 skipped** in 83.6 s with
`BCI_REQUIRE_CUDA=1` (`gpu_pytest_full3.txt`). The skip is the CPU `torch.compile` test (no C++
compiler on `PATH` on Windows). This closes the gap that earlier rows listed (two new profiler tests,
the `profile.py` figure change, the paced-by-default `run_benchmark.sh` and the `onnxruntime-gpu`
pin): all of it ran on the GPU machine and passed.
