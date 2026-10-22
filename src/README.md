# src

This folder holds the training code and the launch tools. The code trains the model, quantizes it, compresses it and evaluates it in one script.

## Files

| File | Content |
|---|---|
| `train_gpt.py` | The Run #10c deployed script. 67,781 bytes, lightly minified (short names, no comments). |
| `train_gpt_lzma.py` | Self-extracting wrapper of `train_gpt.py`. 23,524 bytes. This file counts as the code of the artifact. |
| `train_gpt_readable.py` | Unminified development version from 2026-04-09 (Run #3 era, SP4096). 111,814 bytes. Not submitted. |
| `build_lzma_wrapper.py` | Builds the wrapper from a script. |
| `run_h100.sh` | Launch script with all hyperparameters of Run #10c. |
| `flash_attn_interface_sdpa_shim.py` | Local replacement for FlashAttention 3, built on PyTorch SDPA. |
| `requirements.txt` | Python packages. |
| `baselines/` | The `kevclark` SP4096 baseline script and my modified copy of it. The kernel and perf benchmarks import the modified copy. |

[../docs/infrastructure.md](../docs/infrastructure.md) explains the wrapper, the budget and the shim. [../docs/research_notes/architecture.md](../docs/research_notes/architecture.md) explains the model.

## Differences between train_gpt_readable.py and train_gpt.py

The model structure is the same in both files. The differences are in features, defaults and the quantization and evaluation code.

Only in `train_gpt.py`:

- Score-first TTT (`TTT_*` variables) and the sliding-window evaluation (`SLIDING_WINDOW_EVAL`, `EVAL_STRIDE`).
- `RESUME_QUANT`, which loads `final_model.pt` and skips training.
- `WARMDOWN_FRAC`, which sets the warmdown as a fraction of the wallclock.
- GPTQ calibration on real training data (`GPTQ_REAL_DATA`, `GPTQ_CALIB_BATCHES`), capped at 256 sequences.
- A Hessian for the tied embedding, collected from the output of the final norm, and a separate clip factor for it (`SDCLIP_K_EMBED`).
- An NCCL timeout of 1800 s, and quantization, pruning and saving only on rank 0.

Only in `train_gpt_readable.py`:

- Switches for ideas that were tested and dropped: attention residuals (`ATTN_RES`), a per-layer lane merge (`PER_LAYER_LANE_MERGE`), a LoRA on attention in the repeat pass (`REPEAT_ATTN_LORA`), skipping attention in the repeat pass (`SKIP_REPEAT_ATTN`), disabling the layer 0 MLP (`DISABLE_LAYER0_MLP`), a gradient scale on the parallel scalars, and a variant of the recurrence that skips the physical layers (`RECUR_SKIP_PHYSICAL`).
- Stochastic weight averaging (`SWA_ENABLED`, `USE_SWA_WEIGHTS`).
- Calibration for GPTQ only on sequences that the model generates.
- An import of `DistributedDataParallel` that the script does not use, and an `nvidia-smi` call at start.

Different defaults:

| Setting | `train_gpt_readable.py` | `train_gpt.py` |
|---|---|---|
| `LN_SCALE` | 1 | 0 |
| `QUANT_BITS_ATTN` | 6 | 5 |
| `BYTE_SHUFFLE` | 1 | 0 |
| `RECUR_START_STEP` | 3000 | 3000 (`run_h100.sh` sets 2000) |
| `WARMDOWN_ITERS` | 4000 | 3500 |
| `REPEAT_PASS_SCALE` | 0 | 0 (`run_h100.sh` sets 1) |
| Log label of the EMA evaluation | `DIAGNOSTIC post_ema` | `post_ema` |
| Log label of the pruning | `selective_prune:` | `prune:` |

The launch scripts set most values, so the defaults matter only when a variable is missing.

The structure of the code also differs. The readable file has full names, comments and about 2200 lines. The deployed file uses short names, joins lines and has about 1250 lines. Both files use the same data loader and the same Muon and AdamW split.

## Local smoke test

1. Install the shim as `flash_attn_interface.py` in the `site-packages` folder of the virtual environment (see the infrastructure document).
2. Get the data. The SP4096 and SP8192 data come from the Hugging Face dataset `kevclark/parameter-golf`. The upstream repository https://github.com/openai/parameter-golf has the SP1024 data and the download script `data/cached_challenge_fineweb.py`.
3. Run the script on one GPU:

```bash
nice -n 10 torchrun --standalone --nproc_per_node=1 \
  src/train_gpt.py RUN_ID=local_smoke ITERATIONS=200 VAL_LOSS_EVERY=0
```

The defaults point to the SP1024 data and tokenizer. For SP8192, also set `VOCAB_SIZE=8192`, `DATA_PATH` and `TOKENIZER_PATH` as in `run_h100.sh`. The script has no flag to turn off quantization and evaluation after training, so a smoke test also runs GPTQ and the evaluations. A 200-step test takes several minutes. A full local test that matches the H100 step count uses `ITERATIONS=5100` and takes about two hours or more.

Local results are valid for pre-quantization BPB only. Compression and pruning numbers from a local run are not valid.

## H100 launch

1. Copy the code of the run to a new folder `runs/run{N}/` before the launch.
2. Copy `train_gpt.py` and `run_h100.sh` to the pod, in the folder that holds the `data` folder.
3. Install `brotli` if the image lacks it.
4. Set `RESUME_QUANT=1` only to re-quantize an existing `final_model.pt`. The default (0) trains from the start. The frozen copy in `runs/run10c/` keeps the original value (1).
5. Run `bash run_h100.sh`. The log goes to `run10c_seed${SEED}.log`.
6. Stop the pod after the run.

To run the wrapper, build it and run `train_gpt_lzma.py` in place of `train_gpt.py`. Set `CODE_SIZE_OVERRIDE` only when you run the raw script.
