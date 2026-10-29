# CLAIMS — anti-hallucination ledger

One row per claim. **VERIFIED** = a command was run in this repo and the observed output is
quoted. **PENDING-GPU** = needs the RTX 4060; not run; no number is given. **UNVERIFIED** = written
but not executed anywhere.

Authoring environment: Linux container, 4 vCPU, **no NVIDIA GPU**, torch 2.15.0+cpu, Python 3.13.
"CPU" below means torch on the CPU device unless stated. All data are synthetic.

**Every latency number must point to `results/benchmark_latency.csv`. That file does not exist
yet** (it is written by `make benchmark` on a CUDA machine), so there is **no VERIFIED GPU latency
claim in this ledger**.

## A. Verified here (CPU / no GPU needed)

| # | Claim | Status | Evidence |
|---|---|---|---|
| A1 | Ring buffer: 10.24 s of data (40 chunks) at 1× real time → 0 dropped samples, 0 overrun events, ≥ 2 wraps, data bit-identical to an unpaced regeneration of the same stream | VERIFIED | `pytest tests/test_ring_buffer.py -s` → `{'samples_written': 2560, 'wrap_count': 2, 'dropped_samples': 0, 'overrun_events': 0, 'write_calls': 40, 'write_us_mean': 85.26, 'write_us_max': 197.75, ...}`, 4 passed |
| A2 | Ring buffer wrap-around keeps order/data; `latest()` correct after wrap; buffer array is pre-allocated and never replaced | VERIFIED | same command (`test_wrap_around_preserves_order_and_data`, `test_latest_window_after_wrap`) |
| A3 | Overrun drops the oldest chunk(s), counts them (320 of 1280 samples in the test) and logs a warning; the newest data survive | VERIFIED | same command (`test_overrun_drops_oldest_chunk_and_warns`) |
| A4 | torch-on-CPU matches scipy/numpy float64 within `atol=1e-4` on 10 windows (max abs diff: band-pass 1.17e-5, covariance 1.33e-7, tangent features 1.40e-7, LDA scores 7.77e-7) | VERIFIED (CPU device only) | `pytest tests/test_correctness.py -s -k "cpu"` prints `[bandpass] max |cpu-gpu| = 1.174e-05 …` etc. |
| A5 | Whole streaming path (ring buffer → staging → pipeline → scores) on the CPU device matches the scipy reference: max abs diff 1.08e-6 over 23 chunks, with and without look-ahead; 0 dropped samples | VERIFIED (CPU device only) | `test_streaming_pipeline_matches_cpu_reference[cpu]` |
| A6 | ONNX: the exported LDA head run in onnxruntime (`CPUExecutionProvider`) matches the PyTorch head: max abs diff 8.66e-15 | VERIFIED (CPU provider only) | `pytest tests/test_correctness.py -s -k onnx` → `onnxruntime providers: ['CPUExecutionProvider']`, `[ONNX vs PyTorch head] max |cpu-gpu| = 8.660e-15` |
| A7 | Held-out accuracy of the classifier on **synthetic** data: 0.865 (n = 200, seed 1234; trained on seed 0) | VERIFIED (synthetic only; says nothing about real EEG) | `pytest tests/test_correctness.py -s -k accuracy` → `HELD-OUT ACCURACY (synthetic, n=200, seed=1234): 0.865` |
| A8 | The full model cannot be exported with the legacy ONNX exporter: `aten::linalg_eigh` unsupported. The `torch.export`-based exporter was not evaluated (needs `onnxscript`) | VERIFIED | `UnsupportedOperatorError: Exporting the operator 'aten::linalg_eigh' to ONNX opset version 17 is not supported` (command in FINDINGS §4.4) |
| A9 | `torch.compile` works for the filter on this CPU (compile ≈ 3 s, output within 1.03e-5 of scipy); when compilation raises, the filter falls back to eager and reports `compiled=False` | VERIFIED (CPU only) | `test_torch_compile_filter_matches_eager[cpu]`, `test_compile_failure_falls_back_to_eager_and_says_so` |
| A10 | Hot-path source contains no `.item`, `.cuda`, `.cpu`, `.tolist` or `print`; stages and model code contain no Python loops/comprehensions (no per-channel loop); `stage`/`launch` use `non_blocking` copies | VERIFIED (static source check; the guard demonstrably fires: it flagged my own attribute `self.cuda`, which I renamed). It cannot see implicit syncs inside library calls (see A11) | `pytest tests/test_latency.py -k "hot_path or async or loop"` |
| A11 | `torch.linalg.eigh` synchronizes the host on CUDA (documented); there is no `eigh_ex` | VERIFIED (documentation text of the installed torch; the behaviour itself was not measured on a GPU) | `torch.linalg.eigh.__doc__` → "When inputs are on a CUDA device, this function synchronizes that device with the CPU."; `hasattr(torch.linalg,'eigh_ex')` → False |
| A12 | PyTorch 2.15.0 defaults: `torch.backends.cudnn.allow_tf32 = True`, `torch.backends.cuda.matmul.allow_tf32 = False`; the code disables both | VERIFIED (default values read from the installed CPU build; the code path `configure_precision()` is exercised by every pipeline test) | `python -c "import torch;print(torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32)"` → `True False` |
| A13 | Dashboard renders headlessly in Chromium without Streamlit exceptions or browser page errors, shows a class heading, a Plotly gauge and EEG plot, and is live (chunk counter increases over 3 s). Ran on CPU | VERIFIED (CPU) | `pytest tests/test_dashboard.py` → passed |
| A14 | FastAPI endpoints (`/health`, `/stats`, `/ws`) work with a stub engine; the real `ProcessEngine` runs the pipeline in a child process on CPU and publishes ≥ 15 chunks with 0 dropped samples | VERIFIED (CPU) | `pytest tests/test_server.py` → 2 passed |
| A15 | Running a figure-building thread in the same process as the pipeline raised paced p99 from 10.3–13.8 ms to 26.9–45.6 ms in 3 of 4 repetitions (one alone-run had its own 38.5 ms outlier); p50 unchanged (~5.2–6.1 ms). ~58 chunks per condition | VERIFIED (CPU, noisy, small n) | `python scripts/ui_contention.py --device cpu` ×4; table in FINDINGS §4.1 |
| A16 | The benchmark harness runs all three run types on CPU; `paced` wall time per chunk 256.1 ms (pacing works); `make benchmark` and the CLI **refuse** to write GPU artifacts on a CPU machine (exit 2, nothing written to `results/`); `--allow-cpu` output is prefixed `cpu_smoke_` | VERIFIED | `python -m bci.gpu_pipeline benchmark --device cpu --allow-cpu --chunks 100 --warmup 10 --runs overlap,baseline,paced`; `make benchmark` → `ERROR: no CUDA device. Refusing …`, exit 2; `pytest tests/test_latency.py -k cli` |
| A17 | Profiler-analysis arithmetic (interval overlap, classification of memcpy/kernel events, PNG generation) is correct on a hand-built trace | VERIFIED for the **arithmetic only**; that a real torch.profiler trace uses the assumed field names is UNVERIFIED (see C3) | `pytest tests/test_profile_analysis.py` → 5 passed |
| A18 | Full suite with CUDA absent: `make test REQUIRE_CUDA=0` → **40 passed, 9 skipped** (all 9 skips are CUDA-only cases, each printing the reason). Under `REQUIRE_CUDA=1` those 9 **fail** instead of skipping (2 failed + 7 errors in the deselect-run: dashboard/server excluded) | VERIFIED | `make test PYTHON=… REQUIRE_CUDA=0` → `40 passed, 9 skipped, 1 warning in 68.76s`; strict run → `2 failed, 37 passed, 3 deselected, 1 warning, 7 errors` |
| A19 | README GIF exists and shows the dashboard decoding in real time **on a CPU** (page says `Device: CPU`, latency in the GIF is CPU latency) | VERIFIED as a CPU recording | `results/dashboard_demo.gif` (38 frames, 4.3 MB), `python scripts/record_gif.py` |

## B. Pending — requires the RTX 4060 (no number is claimed)

| # | Claim | Status | How it gets verified |
|---|---|---|---|
| B1 | p99 end-to-end latency < 50 ms (chunk received → label ready) | PENDING-GPU | `make benchmark` → `results/benchmark_latency.csv`, `results/benchmark_summary.csv` (run `overlap`, metric `e2e`); `make test` (`test_p99_latency_under_50ms`) |
| B2 | p50 / p90 / p99 per stage (h2d, filter, cov, classifier, d2h) | PENDING-GPU | `results/benchmark_summary.csv`; fills the table in FINDINGS §2 |
| B3 | `latency_histogram.png` | PENDING-GPU | written by `make benchmark` |
| B4 | Two CUDA streams + pinned buffers are actually used | PENDING-GPU | `test_two_streams_and_pinned_buffers` |
| B5 | Copy/compute overlap, and X ms saved vs the single-stream baseline | PENDING-GPU | `make profile` (`results/profiler_overlap.csv`, `.png`, `profiler_trace.json`) and `overlap` vs `baseline` rows of the summary |
| B6 | GPU output matches CPU within `atol=1e-4` (filter, covariance, tangent features, scores, whole pipeline) | PENDING-GPU | the `[cuda]` cases of `tests/test_correctness.py` (skipped here) |
| B7 | ONNX Runtime with `CUDAExecutionProvider` matches the PyTorch GPU path | PENDING-GPU | `test_onnx_head_matches_pytorch[cuda]` (fails if the CUDA provider is missing) |
| B8 | `torch.compile` works on CUDA on the target machine (needs Triton; fallback to eager is recorded in the CSV as `filter_compiled`) | PENDING-GPU | `filter_compiled` column of the summary; `test_torch_compile_filter_matches_eager[cuda]` |
| B9 | float64 `eigh` per chunk fits the latency budget on this GPU | PENDING-GPU (open risk) | `classifier` rows of the summary |
| B10 | Dashboard on the GPU shows GPU latency | PENDING-GPU | `make dashboard`, then `make gif` |

## C. Written but never executed

| # | Claim | Status | Note |
|---|---|---|---|
| C1 | `Dockerfile` / `docker-compose.yml` build and run (GPU reservation) | UNVERIFIED | no Docker daemon in the authoring container |
| C2 | `.github/workflows/ci.yml` passes on GitHub | UNVERIFIED | never run |
| C3 | `scripts/profile.py` works end to end and its parser matches real Kineto traces (`cat` = `kernel`/`gpu_memcpy`, `args.stream`, "HtoD" in the name) | UNVERIFIED | needs CUDA; it exits with an error rather than reporting zeros if no events match |
| C4 | `make install` from a fresh clone | UNVERIFIED | not run (it downloads packages); dependencies were installed manually into a venv and `pip install -e . --no-deps` ran. `make test`, `make benchmark` (refusal) were run through the Makefile |
| C5 | Hugging Face Spaces deployment (front-matter + `requirements.txt`) | UNVERIFIED | not deployed; the dashboard was run locally with `streamlit run app/dashboard.py` only |
| C6 | Behaviour on native Windows (path with spaces, `make`/`bash` availability, Triton availability, CUDA wheel selection) | UNVERIFIED | authored and tested on Linux only |
| C7 | The look-ahead stage/launch/collect ordering and event-based slot reuse are correct under real CUDA streams | UNVERIFIED on CUDA | the same code path was exercised on CPU only; stream/event calls (`wait_event`, `record`, `synchronize`, `elapsed_time`) were written from PyTorch API knowledge and never executed |
