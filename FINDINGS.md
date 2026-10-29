# FINDINGS

Status legend: **PENDING-GPU** = needs the RTX 4060, not run, no number given. Everything
else below was observed in the CPU-only authoring container (4 vCPU, torch 2.15.0+cpu,
Python 3.13) and is labelled as CPU. No GPU number appears in this file.

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
   compute stream: wait h2d_end → FIR band-pass → covariance → eigh/log → LDA → event cls_end
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

**PENDING-GPU.** Produced by `make benchmark`; source of truth is
`results/benchmark_summary.csv` (per-chunk raw data: `results/benchmark_latency.csv`). Stage
times are CUDA-event GPU times from the same run; `e2e` is the host-clock latency defined in
SPEC.md §4. 1000 measured chunks after 50 discarded warm-up chunks.

| run `overlap` (ms) | p50 | p90 | p99 |
|---|---|---|---|
| e2e (chunk received → label ready) | PENDING-GPU | PENDING-GPU | PENDING-GPU |
| h2d | PENDING-GPU | PENDING-GPU | PENDING-GPU |
| filter (FIR + mean removal) | PENDING-GPU | PENDING-GPU | PENDING-GPU |
| cov | PENDING-GPU | PENDING-GPU | PENDING-GPU |
| classifier (tangent space incl. `eigh` + LDA) | PENDING-GPU | PENDING-GPU | PENDING-GPU |
| d2h | PENDING-GPU | PENDING-GPU | PENDING-GPU |

Open risk worth checking first: the tangent-space step runs `torch.linalg.eigh` in float64 on a
64×64 matrix per chunk and syncs the host. Whether that fits the 50 ms budget on this GPU is
exactly what the `classifier` row will say. If p99 ≥ 50 ms, follow the failure protocol:
`make profile`, identify the stage, optimize, and log before/after here.

### CPU harness validation (not a GPU result)

The same benchmark harness was run on the CPU in `--allow-cpu` smoke mode
(`python -m bci.gpu_pipeline benchmark --device cpu --allow-cpu --chunks 100 --warmup 10 --runs overlap,baseline,paced`),
100 measured chunks per run, output not committed:

| CPU, e2e (ms) | p50 | p90 | p99 | max |
|---|---|---|---|---|
| `overlap` (back-to-back) | 1.588 | 1.966 | 3.479 | 3.532 |
| `baseline` (back-to-back) | 1.577 | 1.724 | 3.675 | 5.749 |
| `paced` (1× real time, 256 ms apart) | 3.255 | 5.080 | 14.837 | 25.730 |

On CPU there are no streams, so `overlap` and `baseline` are the same code and differ only by
noise. The point of the table is the third row: with the same pipeline, pacing at real time
roughly doubled p50 and quadrupled p99 compared with running chunks back-to-back (n = 100, so
p99 is close to the max). Back-to-back runs flatter a pipeline (hot caches, no idle periods);
on a laptop GPU, idle periods can also change clock state. That is why SPEC.md defines a
`paced` run and says which run each quoted number comes from. The paced run's measured wall
time per chunk was 256.1 ms, i.e. the 1× pacing works.

## 3. CUDA stream overlap

**PENDING-GPU.** `make profile` writes `results/profiler_trace.json` (open in
<https://ui.perfetto.dev>), `results/profiler_overlap.png` (timeline, one row per CUDA stream)
and `results/profiler_overlap.csv` (microseconds of H2D copy time that ran concurrently with
kernels, from the exported trace). `make benchmark` also runs the `baseline` (one stream,
pageable memory, blocking copies) so that

> overlapping transfers with compute changed per-chunk latency by **X = PENDING-GPU ms**
> (`overlap` vs `baseline` rows of `results/benchmark_summary.csv`; compare `p50_ms`, `p99_ms`
> and `wall_ms_per_chunk`).

What can be said without the GPU: each window is 500 × 64 float32 = 128,000 bytes, so the H2D
copy is small. How small relative to the kernels is a measurement; do not expect X to be large
and report it as measured. The `baseline` differs from `overlap` in three ways at once (pinned
vs pageable, async vs blocking, look-ahead staging), so X is the effect of the whole design,
not of one ingredient.

Things the trace should show, and what would mean something is wrong:
* H2D memcpys on one stream, kernels on another — if everything is on one stream the
  `torch.cuda.Stream` context is not being applied.
* Memcpys named `Pinned -> Device`; `Pageable -> Device` would mean the staging buffer is not pinned.
* A host-side wait around the `eigh` kernels (the documented sync). H2D for the next chunk
  should already be on the copy stream when it happens.

## 4. Findings made before GPU access

### 4.1 Running the pipeline in the Streamlit process distorts the latency it displays (CPU)

Measured with `scripts/ui_contention.py --device cpu`, paced 1× pipeline thread, ~58 chunks per
condition (p99 ≈ max at this n). A = pipeline alone; B = plus a thread that builds and
serializes a 64-trace Plotly figure every 0.5 s in the same process. Four repetitions:

| rep | A p50 / p99 (ms) | B p50 / p99 (ms) |
|---|---|---|
| 1 | 5.35 / 10.32 | 5.11 / 26.94 |
| 2 | 5.17 / 13.77 | 5.30 / 45.61 |
| 3 | 5.46 / 10.48 | 5.28 / 27.24 |
| 4 | 5.48 / 38.52 | 6.12 / 39.60 |

p50 is unaffected; p99 rose 2–3× in three of the four repetitions, and rep 4's pipeline-alone
run had its own 38 ms outlier, so the effect is real but noisy. An earlier ad-hoc run using the
dashboard's actual figure builders (script not committed) saw 9 ms alone vs 174 ms with the UI
thread; treat that single number as anecdotal. Cause: figure construction (~35 ms) and
serialization (~16 ms) hold the GIL. **Decision:** the dashboard runs the pipeline in a child
process (`ProcessEngine`). With that, the live dashboard on CPU showed p50 5.5 ms / p99 15 ms
while Chromium was rendering it. Whether GIL contention matters on the GPU path is untested.

### 4.2 `torch.linalg.eigh` synchronizes the host on CUDA

From the docstring of the installed torch 2.15.0: "When inputs are on a CUDA device, this
function synchronizes that device with the CPU." There is no `eigh_ex`. A Riemannian
tangent-space map needs it, so the pipeline has one documented host sync per chunk. The AST
guard (no `.item()` etc.) cannot see this; it is stated in SPEC.md §4 and shows in the trace.
Alternative designs that avoid the sync (matmul-only iterative matrix logarithm) were not tried.

### 4.3 TF32 defaults would break the correctness contract

In torch 2.15.0, `torch.backends.cudnn.allow_tf32` defaults to **True** and
`torch.backends.cuda.matmul.allow_tf32` to False. TF32 keeps ~10 mantissa bits, far from the
1e-4 tolerance on a 100-tap float32 convolution. `configure_precision()` disables both.

### 4.4 ONNX covers the LDA head only

`torch.onnx.export` of the full model fails with
`UnsupportedOperatorError: Exporting the operator 'aten::linalg_eigh' to ONNX opset version 17 is not supported`
(legacy exporter). The `torch.export`-based exporter was not evaluated: it needs `onnxscript`,
which is not a dependency here. The head exports with the legacy exporter (deprecated in this
torch version) and runs in onnxruntime; CPU provider agrees with PyTorch to 8.7e-15. The CUDA
provider was not run.

### 4.5 Correctness margins on CPU (torch-on-CPU vs scipy float64)

Max |difference| over 10 windows, `atol = 1e-4` required: band-pass 1.17e-5 (values up to 30),
covariance 1.33e-7, tangent features 1.40e-7, LDA scores 7.77e-7; whole streaming path over 23
chunks 1.08e-6. The band-pass has the thinnest margin (~8×) because it is the float32 stage.
The CUDA-vs-CPU comparison was not run; if it fails the protocol is: read the printed max diff
and index, then try moving the filter to float64.

### 4.6 `torch.compile`

On this CPU (g++ present) the compiled filter built in ~3 s and matched scipy to 1.03e-5.
On CUDA, Inductor needs Triton; as far as I know, native Windows has no official Triton, in
which case the code falls back to eager and says so (`filter_compiled=False`). PENDING-GPU.

### 4.7 Synthetic data

The first generator settings gave 100% held-out accuracy, i.e. a trivially separable toy.
Difficulty was raised (`erd=0.45, source_amp=0.3, trial_jitter=0.4`, chosen on seeds 7 and 21)
and the classifier is evaluated on different seeds: **0.865** on 200 held-out trials (seed
1234). This number characterizes the generator, not EEG decoding.

## 5. Limitations

* **Synthetic data.** Class structure is a two-hemisphere mu/beta amplitude drop; real EEG has
  non-stationarity, subject variability and artifacts.
* **No artifact removal** (eye blinks, muscle, line noise).
* **Ideal network for the dashboard.** The browser is on the same machine; remote latency is
  not modelled. Dashboard latency numbers describe the pipeline, not the network.
* **Hardware unverified.** Nothing was run on an NVIDIA GPU, CUDA streams, `torch.compile` on
  CUDA, the onnxruntime CUDA provider, the profiler trace, Docker, GitHub Actions or a
  Hugging Face Space. See CLAIMS.md.
* **Benchmark optimism.** `overlap`/`baseline` runs are back-to-back; use `PACED=1` for the
  realistic run.
* **Single-chunk window.** One decode window per chunk; no batching across chunks or users.
