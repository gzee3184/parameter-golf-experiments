# figures

These PNG files show results of H100 runs and local sweeps. The scripts that made them are in [../analysis/](../analysis/README.md). Text inside the figures, such as leaderboard positions, shows the state at the time of the plot and is out of date.

## H100 run figures

| File | Content |
|---|---|
| `h100_run_deepdive.png` | Deep dive of the early SP4096 run with 11 layers and 40.6% pruning (Run 2). It shows the loss curve, the step time before and after recurrence, where BPB is lost in the pipeline, the pruning damage curve, the artifact budget and the reason for the compression failure. |
| `h100_run3_results.png` | Run 3 (int5 attention, 0% pruning, 1.1131 sliding BPB): training loss, comparison with Run 2, budget use and leaderboard position. |
| `h100_run5_results.png` | Run 5 (competitor hyperparameters): TTT adaptation curve, comparison with Run 3, budget use. It shows a pre-quantization regression and a smaller quantization gap. |
| `h100_run6_results.png` | Run 6 (late QAT): pre-quantization BPB, all metrics for Runs 3, 5 and 6, and the quantization gap. |
| `h100_run_analysis.png` | Forensic comparison of Runs 3 to 7: loss curves, configuration differences, validation BPB at step 4000, quantization gap, compile time and step time, and a list of causes of the regression. |

## Local sweep figures

All four show local runs of about 3 hours, which match the H100 step count. Each has panels for training loss, step time (the rise at step 2000 is the start of recurrence), validation BPB and a summary table.

| File | Content |
|---|---|
| `training_analysis.png` | 14 runs from the v5 to v8.5 sweeps. |
| `top10_master_comparison.png` | The 10 best of those runs on the same hardware. |
| `v8_focused_analysis.png` | The v8 rounds 3 to 5 and the first v8.5 test (6 runs). |
| `v8_sweep_r3_analysis.png` | Two runs of v8 round 3: softcap 20 and data ramp with warmdown. |

In these tables the post-EMA BPB values lie between 1.0959 and 1.1110 for runs that finish, and the quantized BPB between 1.1119 and 1.1540.
