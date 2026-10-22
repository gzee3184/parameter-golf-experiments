# runs

This folder holds the logs of the runs on 8 x H100 nodes. Some folders also hold a frozen copy of the code of the run. [../docs/h100_runs.md](../docs/h100_runs.md) describes each run and its result. [../docs/infrastructure.md](../docs/infrastructure.md) explains why the code of each run is kept in its own folder.

## Folders

| Folder | Run | Content |
|---|---|---|
| `run00_sp1024_baseline/` | Run 0 | One log of the first full pipeline (SP1024, 13 layers). |
| `run01_sp4096/` | Run 1 | Logs of the SP4096 attempts, including the detailed log of an earlier attempt that crashed in evaluation. |
| `run02/` | Run 2 | Resume logs only (`resume_full.log`, `resume_seed1337.log`). |
| `run03/` | Run 3 | `train.log`. Best SP4096 run. |
| `run04/` | Run 4 | `resume_quant.log`. |
| `run05/` | Run 5 | `resume_quant.log`. |
| `run06/`, `run07/` | Runs 6 and 7 | `train.log`. |
| `run08/`, `run09/` | Runs 8 and 9 | `train.log`, plus the code and launch script of the run. |
| `run10/` | Runs 10, 10a and 10b | Logs of the first SP8192 run and its resume attempts, plus the code and launch script. |
| `run10c/` | Run 10c | `train.log`, code, wrapper and launch script. Best result (TTT 1.1009, sliding 1.1056). |

Folders `run08` to `run10c` hold `train_gpt.py`, `train_gpt_lzma.py` and `run_h100.sh`. Earlier folders hold logs only.

## Log names

- `train.log` is the console output of the launch script.
- `resume_*.log` are logs of sessions that re-quantized a saved checkpoint and did not train.
- `run_*_full.log` and `run10*_seed1337.log` are longer logs of the same sessions.
- `run_YYYYMMDD_HHMMSS.log` is a log named by the start time.

Important lines in the logs: `post_ema`, `prune:`, `Total submission size`, `final_int6_roundtrip`, `final_int6_sliding_window` and `final_int6_ttt`.

The checkpoints (`final_model.pt`) and the compressed models are not in this repository.
