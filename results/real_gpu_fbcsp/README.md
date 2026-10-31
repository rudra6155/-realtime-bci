# results/real_gpu_fbcsp: Benchmark Artifact Notes

**Every file here comes from a benchmark run (`python -m bci.real_benchmark all`) evaluating the Riemannian decoder (k = 4 smoothing) on an RTX 4060 Laptop GPU.**

| file | what it contains |
|---|---|
| `accuracy_per_subject.csv`, `kappa_timecourse.csv`, `noise_log.csv`, `kappa_vs_competition.png` | The Riemannian k = 4 decoder's results (mean kappa 0.505 fixed / 0.523 best, accuracy 0.629). |
| `real_latency_paced.csv`, `real_latency_summary.csv` | The latency benchmark of the same run: Riemannian decoder over LSL, 1000 chunks after 50 warm-up. e2e p50 / p90 / p99 / max 4.078 / 5.183 / 6.477 / 13.962 ms; LSL jitter (std) 0.0837 ms. |

The filter-bank decoder's accuracy on the GPU is in `../real_gpu_fbcsp_accuracy/`. Its latency benchmark can be reproduced with:

```bash
python -m bci.real_benchmark latency --decoder fbcsp --out-dir results\real_gpu_fbcsp_latency
```
