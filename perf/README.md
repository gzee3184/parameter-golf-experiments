# perf

This folder holds experiments on compiler settings and training-loop changes. The aim was more training steps in 600 s. The result was no gain. See `RESULTS.md` for the tables.

## Files

| File | Content |
|---|---|
| `RESULTS.md` | Results of the experiments, with a list of reasons why nothing helped. |
| `bench_harness.py` | Shared harness. It builds the model, runs a fixed number of steps on synthetic data and reports tokens per second and memory. |
| `inductor_sweep.py` | Tests 13 `torch._inductor.config` settings against the default. |
| `training_loop_opts.py` | Tests changes of the training loop. |
| `best_config.py` | The best known settings. The default settings were best. |
| `__init__.py` | Package marker. |

## Result

The tests ran on the local Blackwell GPU, with batch 48 and sequence length 2048.

- No inductor setting was faster than the default. The default was 199.7 ms per step. Settings were between 0.975 and 1.001 times the default speed. CUDA graphs failed, because the RoPE cache tensor is overwritten between steps.
- No training-loop change was faster. The tests covered TF32 matrix multiplication, cuDNN benchmark mode, a prefetch stream, setting gradients to none and gradient accumulation. Gradient accumulation of two halved peak memory (26,513 MiB to 13,743 MiB) at no speed cost.
- `RESULTS.md` gives these reasons: `torch.compile` already fuses all profitable operations, the model is compute-bound and not launch-bound, TF32 was already on, and the matrix shapes are standard.

`RESULTS.md` also lists ideas for the H100 that were researched and not tested. Communication cost and gradient compression were judged not worth it.

## Limits

- The scripts import `train_gpt_modified.py` from `src/baselines/`. Run them from the repository root. The folder had the name `optim_experiments` during the work, and some log lines still use that name.
- Local tests of GPU-specific settings (autotune) do not predict H100 behavior.
