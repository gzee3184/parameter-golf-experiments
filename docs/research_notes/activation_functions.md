# Activation functions

This note records what was learned about the MLP activation function. It combines a literature-based analysis written in March 2026 with local screening results.

## Setting

The MLP has no gate. It has one up-projection, one activation and one down-projection. The activation output is squared, so the activation also acts as an implicit gate. A separate gate matrix would add parameters, and the artifact limit makes that costly.

The code supports two activations, selected by `MLP_ACTIVATION`:

| Name | Function |
|---|---|
| `leaky05` | `leaky_relu(x, 0.5)` squared |
| `mish2` | `mish(x)` squared |

The Run #10c script uses `leaky05`. The names `relu2` and `gelu2` were tested in the early sweeps only.

## Literature analysis

An external research report was written for the project. It discussed squared activations from these sources:

- The Primer architecture search found that squaring ReLU reduces training compute.
- Squared ReLU is used in some production models.
- The Muon optimizer orthogonalizes the update with a Newton-Schulz iteration.

The report made four main points.

1. Smooth activations such as GELU squared and Mish squared have a continuous derivative. The report argued that this suits the aggressive learning-rate schedule of a 10-minute run.
2. Squared ReLU gives exact zeros for negative inputs and high activation sparsity. The report argued that sparsity could give a throughput gain on H100 hardware.
3. The Muon update may work better with smooth activations. Exact zeros give rank-deficient Jacobians, and Newton-Schulz orthogonalization of the update may degrade.
4. The report criticized the experiment design. The baseline `leaky05` with slope 0.5 is far from the usual slopes of 0.01 to 0.1. The report asked for a plain ReLU squared baseline and for equal seeds and schedules.

The report predicted that ReLU squared would finally beat GELU squared on throughput. The local results did not support this prediction for short runs.

## Local results

All local runs used one GPU and the SDPA shim. They are screening results and not H100 results.

### Activation sweep (SP1024, 2400 s wallclock, 11 layers)

| Activation | Sliding BPB | Steps | ms per step |
|---|---|---|---|
| `mish2` | 1.3652 | 1673 | 1435 |
| `gelu2` | 1.3658 | 1683 | 1427 |
| `leaky05` | 1.3757 | 1697 | 1415 |
| `leaky07` | 1.3788 | 1694 | 1418 |
| `silu2` | 1.3905 | 1608 | 1493 |
| `leaky03` | 1.4030 | 1613 | 1488 |

In short runs on a Colab A100 (390 steps), the order was `mish2`, `gelu2`, `relu2`, `leaky05`. The two smooth activations were ahead of `relu2` and `leaky05`. These runs are too short to show convergence behavior.

### Speed

The activation changes the step time only a little. In a solo-GPU test, `mish2` cost 12 ms per step more than `leaky05`, which is +0.8%. At about 84 ms per step on an H100, this is about 60 fewer steps in 600 s.

A rewrite of Mish with a single exponential did not help. `torch.compile` already optimizes `F.mish`, and the rewrite was 15 ms per step slower.

### Contention artifact

An overnight comparison with both GPUs busy gave `leaky05` 1.1070 and `mish2` with Muon+ 1.1181. That difference was mostly GPU contention. A solo run showed 1471 ms per step for `leaky05` and 1484 ms for `mish2`. In the parallel run the values were 1478 ms and 1577 ms. The result of the parallel comparison was not valid.

### SP4096, 4x MLP

On the SP4096 vocabulary with a 4x MLP and weight decay 0.085, a 60 s screen gave pre-quantization values of 3.423 for `leaky05` and 3.428 for `mish2`, with 36 and 35 steps. The advantage of `mish2` from the SP1024 and 3x MLP setting did not appear. The project then used `leaky05`.

## Reading these results

- The two activations are close. The measured differences are about 0.01 BPB at best in short local runs, and they depend on step count.
- Step count differs between runs because step time differs. A change of about 80 steps moves the final BPB by a similar amount. See [stacking_failure_analysis.md](stacking_failure_analysis.md) for an example of this problem in the sweep tables.
- The final H100 recipe uses `leaky05`. The recipe shared by the top PRs, for example [PR #1541](https://github.com/openai/parameter-golf/pull/1541), uses the same squared leaky ReLU with slope 0.5.

## Related files

- `sweeps/early_sp1024/` holds the early activation sweeps.
- `sweeps/local/` holds the SP4096 sweeps, including `leaky05_muon_plus.log` and `fast_mish2_muon_plus.log`.
- [architecture.md](architecture.md) describes where the MLP sits in the model.
