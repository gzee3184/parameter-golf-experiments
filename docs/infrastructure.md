# Infrastructure

This document describes how the training runs were done. It covers the local workstation, the rented H100 pods, the code-size and artifact-size budget, and the fixes that made the H100 runs finish.

Abbreviations used in this document:

- BPB: bits per byte, the competition score. Lower is better.
- GPTQ: a post-training weight quantization method that uses a Hessian estimate from calibration data.
- TTT: test-time training.
- SDPA: scaled dot-product attention, the PyTorch attention function.
- GQA: grouped-query attention.
- NCCL: the NVIDIA multi-GPU communication library.
- EMA: exponential moving average of the weights.

## Two environments

The work used two kinds of hardware.

| Item | Local workstation | Rented H100 pod |
|---|---|---|
| GPUs | 2 x RTX PRO 6000 Blackwell Max-Q, 96 GB each | 8 x H100 SXM, 80 GB each |
| Compute capability | sm_120 | sm_90 |
| Launch | `torchrun --nproc_per_node=1`, one GPU per job | `torchrun --nproc_per_node=8` |
| Attention | PyTorch SDPA through a shim | FlashAttention 3 (FA3) |
| Step time | about 1600 to 1850 ms | about 93 ms before recurrence, about 112 to 118 ms after |
| Use | Screening of ideas, sweeps, analysis | Final runs that count as results |
| Cost | None | About $23 per run |

The competition limit is 600 s of training on 8 x H100. A local run is about 15 to 17 times slower per step. A local run of 10,800 s gives about the same step count (5300 to 5800 steps) as one H100 run.

## Local workstation

### One GPU per job

The machine had two GPUs and was shared with other users. Each job used one GPU with `nproc_per_node=1`, so the code ran without multi-GPU communication. Two jobs could run in parallel, one per GPU. Long jobs ran with `nice -n 10`.

Parallel jobs on both GPUs slowed each other. One overnight comparison of two activation functions showed a gap of about 100 ms per step. A repeat on a single GPU showed that most of this gap came from contention and not from the activation function. See [research_notes/activation_functions.md](research_notes/activation_functions.md). The H100 pod has separate memory for each GPU and does not show this effect.

### FlashAttention shim

FlashAttention 3 and 4 do not run on the consumer Blackwell GPU (sm_120). The training script imports `flash_attn_func` from a module named `flash_attn_interface`. On the workstation, a small shim provides that module.

The file [`src/flash_attn_interface_sdpa_shim.py`](../src/flash_attn_interface_sdpa_shim.py) implements the FA3 function signature with `torch.nn.functional.scaled_dot_product_attention`:

- The function takes tensors in the layout (batch, sequence, heads, head dimension) and moves heads to dimension 1 for SDPA.
- It sets `enable_gqa=True` when the query has more heads than the key.
- It returns the result in the original layout.

To use the shim, copy it into the virtual environment as `site-packages/flash_attn_interface.py`. On the H100 pod, the `kernels` package (listed in `src/requirements.txt`) supplies the real `flash_attn_interface`.

### What transfers to the H100 pod

Only some local results carry over.

- Pre-quantization BPB (after the EMA weights are applied) is comparable between local and H100 runs for the same architecture and step count.
- Architecture changes that change the compute per step do not transfer. A 12-layer model was better locally by 0.0064 BPB. On the H100 it was worse, because it had fewer steps in 600 s and needed 12.5% pruning to fit the artifact.
- Compression and pruning results do not transfer. See the section "Local compression results" below.

Because of these limits, the local workstation was used to screen same-architecture changes. Each H100 run tested a change that had a local signal.

## Rented H100 pods

### Provider and cost

The H100 runs recorded in this repository used a RunPod pod with 8 x H100 SXM GPUs. The pod billed about $21.52 per hour. Planning used about $23 for one full session. A three-seed validation therefore costs about $69. A session that only re-quantizes a checkpoint is shorter and costs less.

A full session has these phases:

1. Training for 600 s.
2. EMA evaluation, about 10 s.
3. GPTQ and quantization, about 3 to 5 min.
4. Pruning and compression.
5. Round-trip evaluation, sliding-window evaluation (stride 64) and TTT.

The pod had a persistent volume. The tokenized dataset stayed on the volume between sessions. The pod was stopped after each session to stop billing.

A guide for a Google Cloud setup (`a3-highgpu-8g`, 8 x H100) was also written. It describes three safety layers: a maximum run duration set on the VM, a watchdog timer in the run script, and an automatic shutdown after training. The recorded runs in this repository do not depend on it.

### Launch script

The script [`src/run_h100.sh`](../src/run_h100.sh) sets all hyperparameters as environment variables and starts `torchrun --standalone --nproc_per_node=8 train_gpt.py`. The log goes through `tee` to a file named by the seed.

The copy in this repository is the Run #10c script. It sets `RESUME_QUANT=1`, so it re-quantizes a saved checkpoint and does not train. To train from the start, set `RESUME_QUANT=0`.

The data must be on the pod before the run. For the SP4096 and SP8192 vocabularies, the data comes from the Hugging Face dataset `kevclark/parameter-golf`, not from the official repository. The script header shows the download command:

```bash
cd parameter-golf && rm -f data/manifest.json
MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf \
  python3 data/cached_challenge_fineweb.py --variant sp8192 --train-shards 143
```

The only extra Python package that the pod image lacked was `brotli`. Install it with `pip install brotli`.

### Freeze each run's code

Every H100 run has its own folder under `runs/`. The folder holds a copy of `train_gpt.py`, `train_gpt_lzma.py`, `run_h100.sh` and the log. The top-level files in `src/` are the working copy. The run folders are not changed after the run.

This rule exists because of Run #3. Run #3 had the best result for a long time, but later runs with the same hyperparameters did not reproduce it. The code had been overwritten between runs. A forensic comparison of Runs #3 to #7 found that small code edits changed the `torch.compile` graph. The edits were a bug fix, the added resume path and the removal of dead features. The comparison measured about +0.004 BPB at step 4000 from this effect alone, even with matched hyperparameters. A change of the RoPE base from 10,000 to 22,000 explained about +0.0006. The frozen folders let a later reader compare the exact code of each run.

Folders `run08` to `run10c` hold the code. Folders `run00` to `run07` hold logs only. See [../runs/README.md](../runs/README.md).

## Code-size budget

### Limit

The artifact is the code file plus the compressed model. The limit is 16,000,000 bytes (decimal, not MiB):

```
model budget = 16,000,000 - code bytes
```

Every byte of code reduces the model budget by one byte.

### LZMA self-extracting wrapper

The script [`src/build_lzma_wrapper.py`](../src/build_lzma_wrapper.py) reduces the code bytes:

1. It reads the training script.
2. It compresses the script with LZMA at preset 9 with the extreme flag.
3. It encodes the result with base85.
4. It writes one line of Python that decodes, decompresses and runs the script with `exec`.

Usage:

```bash
python src/build_lzma_wrapper.py src/train_gpt.py src/train_gpt_lzma.py --verify
```

The `--verify` option checks that the decompressed code equals the input.

| File | Size in bytes |
|---|---|
| `src/train_gpt.py` (Run #10c, lightly minified) | 67,781 |
| `src/train_gpt_lzma.py` (wrapper that counts as code) | 23,524 |
| `src/train_gpt_readable.py` (unminified development version, not submitted) | 111,814 |

The wrapper saves about 44,000 bytes for the Run #10c script. An earlier note measured a saving of about 38,000 bytes for a smaller script.

The training script also reads the environment variable `CODE_SIZE_OVERRIDE`. When a run uses the raw `train_gpt.py`, this variable makes the budget calculation use the wrapper size. Run #10c used `CODE_SIZE_OVERRIDE=23524`.

### Budget arithmetic for Run #10c

| Item | Bytes |
|---|---|
| Limit | 16,000,000 |
| Code (wrapper) | 23,524 |
| Model budget | 15,976,476 |
| Compressed model | 15,967,716 |
| Total artifact | 15,991,240 |
| Margin | 8,760 |

The unpruned model was larger than the budget. The script then zeroed the least costly plus and minus one quantized values until the model fitted. It zeroed 226,187 of 10,680,192 candidates, which is 2.1%.

### Levers that kept the model inside the budget

- Attention weights use 5 bits and MLP weights use 6 bits. Earlier tests showed that 5-bit attention saved about 0.68 MB at no measurable BPB cost.
- With the SP8192 vocabulary, the tied embedding is large. Run #10b stored it at 8 bits. The result needed 41.1% pruning. Run #10c stored it at 6 bits, which reduced pruning to 2.1%.
- Compression uses Brotli quality 11. The Run #10c script also applies a byte shuffle (`BYTE_SHUFFLE=1`) before compression.
- Quantization clips each row at a multiple of its standard deviation (`SDCLIP`, k = 12.85, and k = 20 for the embedding). This reduced the quantization gap. See [findings.md](findings.md).

## Fixes that made the H100 runs finish

### GPTQ timing and NCCL timeout

Run #10a failed. GPTQ ran on all eight ranks with 24,000 calibration sequences. Ranks 1 to 7 waited for rank 0 and the watchdog stopped them. Run #10c has these fixes in `train_gpt.py`:

- The NCCL process-group timeout is 1800 s.
- Quantization, pruning and saving run only on rank 0. Other ranks wait at a barrier.
- Calibration uses real training data (`GPTQ_REAL_DATA=1`) and caps the count at 256 sequences.
- The decompression helpers are defined before the rank-0 gate, so every rank can load the result.

The training loop stops at the wallclock limit (`MAX_WALLCLOCK_SECONDS=600`). GPTQ, pruning and evaluation run after the loop. The script has no separate time-reserve setting for GPTQ.

### RESUME_QUANT

A normal run saves the EMA weights as `final_model.pt` on rank 0. With `RESUME_QUANT=1`, the script does this:

1. It loads `final_model.pt`.
2. It activates recurrence if the model uses it.
3. It evaluates the pre-quantization BPB.
4. It runs GPTQ, pruning, compression, the round-trip check, the sliding-window evaluation and TTT.

No training happens. This mode allowed three uses:

- Re-quantize a good checkpoint after a quantization failure (Run #10 to Run #10c).
- Test quantization and TTT changes on a fixed set of weights.
- Run evaluation-only experiments. See [../experiments/README.md](../experiments/README.md).

A resume session is shorter than a full session. The run logs record $4.50 and $11 for two resume sessions. Planning used the full $23 as the cost of any session.

## Local compression results

Local runs always showed heavy pruning. One local test needed 43% pruning for a model of 16.25 MB. The same architecture on the H100 (Run #3) compressed to 14.47 MB with 0% pruning. Plain round-to-nearest quantization and every GPTQ variant produced the same local size.

The likely cause is that local training gives weights with higher entropy. The local run uses the SDPA shim, one GPU and a different GPU type. This cause is not proven. The practical rule is:

- Do not use a local run to judge compression or pruning.
- Do not tune GPTQ to improve a local compression number.
- Test compression on the H100.

## Related documents

- [research_notes/architecture.md](research_notes/architecture.md) describes the model that the code implements.
- [../src/README.md](../src/README.md) lists the source files and how to run them.
- [../perf/README.md](../perf/README.md) and [../kernels/README.md](../kernels/README.md) describe the speed experiments.
