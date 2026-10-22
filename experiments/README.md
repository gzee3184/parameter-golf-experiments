# experiments

This folder holds two groups of follow-up experiments. One group tests ideas that apply at evaluation time. The other group tests quantization changes. Sweeps of training settings are in [../sweeps/](../sweeps/README.md).

## eval_time/

These experiments started on 2026-04-19, after Run #10c. They use the saved EMA checkpoint of Run #10 (pre-quantization BPB 1.1033) and do not retrain. All results are local. The checkpoint itself is not in this repository.

| Folder | Idea | Result |
|---|---|---|
| `baseline_r10/` | Frozen copy of the code of Run #10c. It is the reference for the other folders. Do not change it. | Not applicable. |
| `expT_temp_scaling/` | Divide the logits by a temperature T before the softmax. | No gain. The best T was 1.00. BPB at T = 0.95 was 1.1060 and at T = 1.05 was 1.1048. |
| `expTq_postquant_temp/` | The same test after quantization. | No gain. The best T was again 1.00. |
| `expDS_deep_supervision/` | Average the logits of the last N decoder states. | Strongly worse. All variants lost between 0.02 and 0.24 BPB. Intermediate states are not aimed at the vocabulary. |
| `expZ_zloss/` | Add a z-loss term to the training loss (short test from scratch). | Worse in a 500-step test. The test is too short to judge convergence. |
| `expA_kernel_fusion/` | Compile the model used to collect GPTQ Hessians. | Plan only. The README has the plan and no results. |
| `expB_lora_ensemble/` | Average three LoRA adapters trained on the frozen backbone. | Plan only. |
| `expE_ovb/` | Upsample hard sequences in a short fine-tune. | Plan only. |

Each experiment folder has a `README.md` with the hypothesis. Folders with results also have `results.md`, `run_sweep.sh` (or a similar script), `logs/` and a modified copy of `train_gpt.py`.

Other files in this folder:

- `verify_r10_checkpoint.py`, `verify_r10_canonical.sh`, `verify_r10_v2.sh` and the matching `.log` files check that the harness reproduces the pre-quantization BPB of 1.1033 from the checkpoint. The logs of some checks end with errors (a state-dict shape mismatch, an NCCL shutdown message). The experiment folders use the canonical path instead.
- `download_sp8192.py`, `download_sp8192.sh` and `download_sp8192.log` download the SP8192 data.

## quantization/

These experiments test GPTQ and quantization-aware training changes with local runs on the SP4096 model. The logs have the suffix `_test.log` or `_long_test.log`.

| Group | Scripts and logs | Question |
|---|---|---|
| CAGE-lite | `test_cage_lite.sh`, `test_cage_lite_long.sh`, `cage_lite_test.log`, `cage_lite_long_test.log` | Does a gentle push of the weights toward the quantization grid in the last 15% of training help? It hurt in the 30-minute test. The long test checked a well-trained model. |
| Late QAT | `test_late_qat_long.sh`, `late_qat_long_test.log`, `test_qat_realdata_vol.sh`, `qat_realdata_vol_test.log` | Does noise injection in the last 10% of training reduce the quantization loss? |
| GPTQ calibration | `test_ffix_control.sh`, `test_ffix_realdata_vol.sh`, `test_gptq_fix_long.sh`, `test_gptq_realdata_damped.sh` and the matching logs | Effect of real-data Hessians, calibration volume and pre-damping. A test without pre-damping gave 43.3% pruning and a quantized BPB of 1.1756. |
| GPTQ ablation | `test_gptq_ablation.py`, `ablation_results.log` | Find which GPTQ change raised the compressed size from 14.47 MB to 16.25 MB, by testing each change alone on a saved checkpoint. |
| Reference run | `test_pr1326_local.sh`, `pr1326_test.log` | Run the code of [PR #1326](https://github.com/openai/parameter-golf/pull/1326) locally for reference. |
| TTT | `test_ttt_local.sh`, `test_ttt_int6.sh`, `ttt_test.log`, `ttt_int6_test.log` | Local TTT tests with int5 and int6 attention. |
| Diagnostics | `diag_statedict.py`, `diag_quantized.py`, `diag_recompress.py` | Inspect the keys and sizes of a quantized file, and re-compress it with a different type for one tensor. |

The local compression numbers in these logs, such as 39% to 93% pruning, are not valid for the H100. See [../docs/infrastructure.md](../docs/infrastructure.md). The Late QAT and CAGE-lite results are discussed in [../docs/findings.md](../docs/findings.md).
