# analysis

This folder holds scripts that parse logs, make plots and measure properties of trained models. The plots are in [../figures/](../figures/README.md).

## Log parsing and plots

| File | Content |
|---|---|
| `parse_logs.py` | Reads training logs and writes `experiment_summary.csv`, `step_curves.csv` and `ttt_curves.csv`. |
| `experiment_summary.csv` | One row per experiment, with configuration and final metrics. |
| `step_curves.csv` | Per-step training curves. |
| `ttt_curves.csv` | Per-chunk curves of the TTT evaluation. |
| `visualize_training.py` | Makes multi-panel plots of the training logs. It produces plots of the style of `training_analysis.png` in the figures folder. |
| `make_run3_viz.py`, `make_run5_plot.py`, `make_run6_plot.py` | Plots of H100 Runs 3, 5 and 6. |
| `run_analysis.py` | Comparison of H100 Runs 3 to 7. It looks for the reason that Run 3 stayed best. |

## Cross-hardware estimates

| File | Content |
|---|---|
| `analyze.py` | Compares local and H100 training curves. |
| `predict_h100.py` | Estimates H100 BPB from local results. The project later found that such estimates are unreliable for architecture changes. See [../docs/infrastructure.md](../docs/infrastructure.md). |
| `h100_time_guess.py` | Rough time estimates from GPU peak throughput. |

## Model measurements

| File | Content |
|---|---|
| `gradient_analysis.py` | Gradient norm by layer on a trained model. Described in [../docs/research_notes/gradient_analysis.md](../docs/research_notes/gradient_analysis.md). |
| `analyze_rope_layers012.py`, `analyze_rope_v2.py` | How much layers 0 to 2 depend on position information. |
| `layer2_corruption_experiment.py` | How sensitive layers 3 to 10 are to corruption of the output of layer 2. |

## Requirements

Run the scripts from the repository root.

- `gradient_analysis.py`, `analyze_rope_*.py` and `layer2_corruption_experiment.py` load the model code from `src/train_gpt_readable.py`. They also load a trained checkpoint from `parameter-golf/final_model.pt`. The repository does not include checkpoints.
- The scripts use SP4096 data paths under `parameter-golf/data/`.

Edit the paths before use.
