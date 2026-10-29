#!/usr/bin/env bash
# Runs the pipeline 1000 times (after 50 discarded warm-up chunks) and writes, under results/:
#   benchmark_latency.csv   per-chunk raw latencies of the headline run (overlap)
#   benchmark_summary.csv   p50 / p90 / p99 / max per metric and run, with device + versions
#   latency_histogram.png   histogram of the headline run
# Runs: overlap (back-to-back, look-ahead; the first run, so it names benchmark_latency.csv and the
# histogram), baseline (single stream, blocking copies) and paced (1x real time, the realistic
# latency; about 4.5 minutes for 1000 chunks). SKIP_PACED=1 omits the paced run.
# Exit status is non-zero if the first run's p99 >= 50 ms.
# Needs a CUDA GPU; on a CPU-only machine it refuses to write results (see --allow-cpu).
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PYTHON:-python}"
RUNS="overlap,baseline,paced"
if [[ "${SKIP_PACED:-0}" == "1" ]]; then RUNS="overlap,baseline"; fi

exec "$PY" -m bci.gpu_pipeline benchmark \
    --chunks "${CHUNKS:-1000}" --warmup "${WARMUP:-50}" --runs "$RUNS" --out-dir results "$@"
