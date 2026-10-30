# results/real_gpu_fbcsp: what the files in this folder are

Pushed from the RTX 4060 laptop (commit `edf6541`). Checked after the push:

| file | what it contains | evidence |
|---|---|---|
| `real_latency_paced.csv`, `real_latency_summary.csv` | One real-data run over LSL, 1000 chunks after 50 warm-up. **Stage profile is that of the filter-bank decoder** (filter stage median 0.585 ms for the banked 9-band FIR, classifier stage 1.656 ms with no `eigh`; the Riemannian runs had 0.108 and about 5.0 ms). The files do not record the decoder. Percentiles recomputed from the 1000 raw rows equal the summary: e2e p50 / p90 / p99 / max 4.078 / 5.183 / 6.477 / 13.962 ms; LSL jitter (std) 0.0837 ms. | `real_latency_summary.csv`, recomputation in FINDINGS §10.1 |
| `accuracy_per_subject.csv`, `kappa_timecourse.csv`, `noise_log.csv`, `kappa_vs_competition.png` | **The Riemannian k = 4 decoder's results, not the filter-bank decoder's.** The three CSVs are identical to `../riemann_k4/` apart from line endings (mean kappa 0.505 fixed / 0.523 best, accuracy 0.629), and the PNG has the size of the earlier Riemannian re-render from the GPU machine (51,680 bytes). The filter-bank decoder's container result is 0.587 / 0.598 (`../accuracy_per_subject.csv`). | `cmp` / `diff` against `../riemann_k4/`, FINDINGS §10.2 |

So this folder does **not** show the filter-bank decoder's accuracy on the GPU. That run was done later: its console output is in `../real_gpu_fbcsp_accuracy/console_paste.txt` (it equals the container's filter-bank result at 3 decimals; its CSV files were not pushed). The command that produces the files:

```
python -m bci.real_benchmark accuracy --subjects 1-9 --device cuda --decoder fbcsp --out-dir results\real_gpu_fbcsp_accuracy
```
