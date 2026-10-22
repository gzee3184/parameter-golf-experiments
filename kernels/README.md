# kernels

This folder holds an experiment with custom Triton kernels for the MLP block. The result was negative. The experiment is kept for the record.

## Files

| File | Content |
|---|---|
| `fused_mlp.py` | Triton kernels, forward and backward. Level 1 fuses `leaky_relu(x, 0.5)` squared. Level 2 adds a fused RMS norm with scale, and a fused residual add with the MLP scale. |
| `test_kernels.py` | Checks the forward and backward results against PyTorch and times 100 iterations. |
| `benchmark.py` | Puts the kernels in the model and compares training steps with and without them. |
| `profile_compiled.py` | Profiles the compiled model to show where GPU time goes. |
| `__init__.py` | Package marker. |

## Result

- Each kernel is about twice as fast as the PyTorch version when it runs alone.
- `torch.compile` with `fullgraph=True` already fuses these element-wise operations. With compile on, the custom kernels were 0.4% to 3.1% slower, because they add dispatch overhead.
- The profile of the compiled model showed this split of GPU time: compiled fused graph about 55%, attention about 15%, matrix multiplications about 14%, reductions about 10% and element-wise operations about 7%.

The conclusion was that custom kernels for element-wise fusion do not help when the model is compiled. The follow-up experiments on compiler settings are in [../perf/README.md](../perf/README.md).

## Limits

The scripts import `train_gpt_modified.py` from `src/baselines/`. That file is my modified copy of the `kevclark` SP4096 baseline. Run the scripts from the repository root.
