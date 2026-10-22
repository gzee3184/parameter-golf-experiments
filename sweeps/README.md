# sweeps

This folder holds the logs and launch scripts of screening runs. Almost all runs used the local workstation with one GPU per job. Treat the results as screening results. They show relative effects for one fixed step count and do not predict H100 results. See [../docs/infrastructure.md](../docs/infrastructure.md).

## local/

Screening sweeps on the local GPUs, from 2026-04-01 to 2026-04-09. The folder has about 440 files, mostly logs. Launch scripts (`*.sh`) sit beside the logs.

### Subfolders

| Folder | Content |
|---|---|
| `v3_sweep/`, `v3_mlp_share/` | First sweeps of the v3 code (MLP sharing, recurrence pairs). |
| `v4_maxparams/` | The v4 model with the largest parameter count that fits. A 30-minute and a 3-hour test. |
| `v5_long/`, `v6_long/` | Three-hour tests of the v5 and v6 code. |
| `v6_sweep/` | One-hour tests: control, triple loop and attention residuals. |
| `attnres_sweep/` | Four short configurations of attention residuals. Not meaningful for quality, because the tests were too short. See [../docs/research_notes/strategy.md](../docs/research_notes/strategy.md). |
| `v7_sweep/` | The LoRA and improvements bundle. |
| `v8_sweep_r1/` to `v8_sweep_r5/` | Rounds of the v8 sweep: RoPE base and layer-norm scale, skip topology and parallel start, softcap and data ramp, comparison with PR #1326, combined fixes. |
| `v8.5_sweep/` | Architecture tests (attention LoRA, disabling the layer 0 MLP, skipping attention on repeat passes, a gradient scale for parallel scalars, layer-norm scale off with softcap 20). |
| `v8.5_sweep_INVALID_warmdown/` | The first attempt of the v8.5 sweep. Its warmdown setting was wrong, so the results are invalid. |
| `v9_compression/` | Late QAT and SWA tests. |
| `v10_int5attn/` | Int5 against int6 attention. |
| `v11_scale/` | 12 layers and untied down-projection tests. |
| `ttt_sweep/` | TTT settings (SGD and AdamW, learning rates, epochs) on a fixed quantized model. |
| `scripts/` | Launch scripts of the versioned sweeps (`run_v3_*.sh` to `run_v11_*.sh`) and two RoPE analysis scripts. |

### Log names at the top level

| Pattern | Meaning |
|---|---|
| `screen_*` | Short screening runs of one factor: MLP width and quantization bits, Muon weight decay, warmdown, GPTQ calibration size, compression type. |
| `rel_*` | Repeats of the MLP-width and quantization tests, run one at a time on one GPU, to remove contention noise. |
| `budget_*`, `long_*` | Tests of how model size and quantization bits fit the 16 MB budget. |
| `s2_*` | Sweep 2: layers, EMA, QK gain with softcap, batch size. |
| `s3_*` | Sweep 3: TTT only (learning rate, epochs, chunk size, weight decay) on a saved model. |
| `v4*`, `vocab4096_*`, `v4096_smoke.log` | Tests of the SP4096 vocabulary: EMA decay, layer count, weight decay, batch size, QK gain, quantization bits. |
| `slm_*`, `shared_*` | Tests of mixed attention and MLP-only layer stacks, and shared attention. |
| `overnight_*`, `leaky05_muon_plus.log`, `fast_mish2_muon_plus.log` | The overnight comparison of activations and Muon variants, and the follow-up on step time. |
| `grid_*`, `queue_*`, `ttt_*`, `eval_*` | Grid searches, job queues, TTT variants and evaluation-context tests. |

The names carry the setting. For example `v4_12L_wd0.15.log` is the v4 code with 12 layers and weight decay 0.15, and `s3_lr0.002_ep5.log` is a TTT test with learning rate 0.002 and 5 epochs.

Folder `local` also holds output files named `*_output.log` of the launch scripts, which collect the output of several runs.

## early_sp1024/

The first sweeps, from March 2026, on the SP1024 baseline of the competition. They tested the activation function, XSA depth, multi-token prediction, RoPE dimensions, TTT, sliding-window stride and combinations. The folders are named by date and topic (`2026-03-27_ActivationSearch`, `2026-03-27_XSA_Sweep` and others) and by result type (`combined_test_results`, `pairwise_ablation_results`, `tier1_test_results`, `overnight_results`). The `run_*.sh` scripts launch the tests. The `*_master.log` files collect the output. See [../docs/research_notes/activation_functions.md](../docs/research_notes/activation_functions.md) and [../docs/research_notes/stacking_failure_analysis.md](../docs/research_notes/stacking_failure_analysis.md).

## kevclark_base_logs/

Runs on the SP4096 baseline of the `kevclark` setup, which has the training script `train_gpt_modified.py`. The names give the variant and the wallclock in seconds:

- `baseline_*` are the unchanged baseline (600 s and 1800 s).
- `muoneqr_only_600s.log` adds the MuonEq-R change.
- `modified_*`, `v2_*`, `novel_asymm_recscale_*`, `no_shared_attn_*` and `warmdown_*` are variants with recurrence, lanes, a recurrence scale, no shared attention and warmdown fractions of 0.40 and 0.50.

Three logs are empty: `modified_full_1800s.log`, `modified_v2_730s.log` and `no_shared_attn_730s.log`. The two `.txt` files are large copies of the `v2_lanes_recmix` logs.
