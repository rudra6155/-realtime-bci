# results/real_gpu_fbcsp: what the files in this folder are

Pushed from the RTX 4060 laptop (commit `edf6541`). **Every file here comes from one `python -m bci.real_benchmark all` run made on the phase-3 checkout, i.e. with the
Riemannian decoder, not the filter-bank one** (despite the folder name). Evidence from the pasted terminal log: the `git pull` that would have brought the filter-bank code had aborted,
pytest right after it collected 83 tests (the phase-4 suite has 109), the accuracy output of the same command is the Riemannian k = 4 result (0.629 / 0.505 / 0.523), and `bci.tail_diag` did not exist yet.

| file | what it contains |
|---|---|
| `accuracy_per_subject.csv`, `kappa_timecourse.csv`, `noise_log.csv`, `kappa_vs_competition.png` | The Riemannian k = 4 decoder's results (identical to `../real/riemann_k4/` apart from line endings; mean kappa 0.505 fixed / 0.523 best, accuracy 0.629). |
| `real_latency_paced.csv`, `real_latency_summary.csv` | The latency half of the same run: Riemannian decoder over LSL, 1000 chunks after 50 warm-up. e2e p50 / p90 / p99 / max 4.078 / 5.183 / 6.477 / 13.962 ms; LSL jitter (std) 0.0837 ms. Its stage times (filter 0.585, cov 0.961, classifier 1.656 ms) differ a lot from the other Riemannian runs (about 0.11 / 0.8 / 5.0 ms); that is unexplained. |

The filter-bank decoder's accuracy on the GPU is in `../real_gpu_fbcsp_accuracy/` (raw files, byte-identical to the container's). **Its latency on the GPU has not been measured**:

```
python -m bci.real_benchmark latency --decoder fbcsp --out-dir results\real_gpu_fbcsp_latency
```
