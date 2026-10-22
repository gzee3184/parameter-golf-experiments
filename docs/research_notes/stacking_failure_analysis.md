# Stacking failure analysis

In March 2026, three changes each improved the local baseline when tested alone. The combination of the three was no better than the best single change. This note records the data, the external analysis and what the project took from it.

## The three changes

The tests used the SP1024 model (11 layers, dimension 512) on one local GPU with a 2400 s wallclock and about 1600 to 1700 steps.

| Change | Setting | Sliding BPB gain when tested alone |
|---|---|---|
| Activation | `MLP_ACTIVATION=mish2` | about 0.011 |
| Exclusive self-attention (XSA) depth | `XSA_LAST_N=6` | about 0.034 |
| Multi-token prediction (MTP) | one extra head, loss weight 0.05 | about 0.040 |

XSA removes the component of the attention output that lies along the token's own value vector. MTP adds an auxiliary loss for the token after next. The MTP head is zero-initialized and is removed at export, so it costs no artifact bytes.

## The observation

The combined model (`mish2` with XSA depth 6 and MTP) never beat the plain baseline in the first 1050 steps. The combined model matched the `mish2`-only model at every logged step.

| Step | Combined | Baseline | `mish2` only |
|---|---|---|---|
| 100 | 1.9676 | 1.9467 | 1.9623 |
| 400 | 1.4577 | 1.4498 | 1.4554 |
| 800 | 1.3393 | 1.3340 | 1.3391 |
| 1050 | 1.3051 | 1.3008 | 1.3053 |

The values are validation loss values from the logs. Lower is better.

## Hypotheses

The project listed five explanations before asking for outside analysis:

1. The three changes address the same bottleneck, which is the quality of the representation in the decoder layers.
2. The gradients of the three changes interfere.
3. The local learning-rate schedule hides the effect. Local runs start in warmdown at the first step, because the warmdown length is set in iterations and a local step is slow. An H100 run has full learning rate for the first 3730 steps.
4. The activation and XSA both change how a layer gates its own state.
5. The step count differs between variants, and faster variants get more steps.

## External analysis

A research report analyzed the problem. Its conclusions were:

- The stacking failure is consistent with representational saturation. A model of 27 million parameters trained for 10 minutes has limited capacity for new useful features, and the three changes compete for the same capacity.
- Muon normalizes the update matrix. The report argued that Muon already enforces diversity at the weight level, which makes XSA partly redundant. It also argued that an auxiliary MTP gradient is projected into the same space as the main gradient and loses its separate effect.
- A smooth activation makes the hidden state more informative. This leaves less work for XSA and MTP.
- The learning-rate schedule may hide the effect of fine-grained changes such as XSA and MTP, because the model spends the whole run at high learning rate.

The report ranked the hypotheses. It rated representational saturation and structural redundancy as best supported. It rated feature erasure as least supported.

The report recommended changes that address other bottlenecks:

- Faster optimization steps, for example a cheaper Newton-Schulz iteration, so the model takes more steps in 600 s.
- Better quantization, for example weight rotation before quantization.
- Lighter versions of the existing changes, for example XSA on the last two layers and a smaller MTP weight.

The report also proposed a hybrid with an n-gram cache. The project did not pursue that proposal.

## Problem in the data

The sweep tables hold a second explanation. The same nominal baseline (`leaky05`, XSA depth 4, no MTP) appears with different results in different sweeps:

| Sweep | Sliding BPB | Steps |
|---|---|---|
| Activation sweep | 1.3757 | 1697 |
| XSA depth sweep | 1.4094 | 1613 |
| MTP sweep | 1.4283 | 1612 |

The baseline moves by 0.05 BPB between sweeps. The step count differs by 85 steps. The sliding BPB in these short runs depends strongly on the number of steps and on the learning-rate position in warmdown. A gain of 0.011 to 0.040 BPB measured against a different baseline in each sweep is not comparable across sweeps. Each sweep compared its own arms, but the arms were also affected by contention between the two GPUs. A reader should treat the "gain alone" column above with caution.

The XSA result at full depth was later used anyway. The final code applies XSA to all 11 layers (`XSA_LAST_N=11`). XSA depth 11 was tested on the same sweep and ranked second to depth 6 there (1.3927 against 1.3757).

## Outcome

- The project kept the smooth squared activation choice discussed in [activation_functions.md](activation_functions.md), and it used `leaky05` in the final recipe.
- The project did not keep MTP in the final model. The Run #10c script has no MTP option.
- The project kept XSA on all layers.
- Later work used full-length local runs of about 3 hours and H100 runs for decisions. It treated short sweeps as screening only. See [strategy.md](strategy.md).

## Related files

- `sweeps/early_sp1024/` holds the logs of the activation, XSA, MTP and RoPE sweeps and the combined and pairwise tests.
- The RoPE sweep chose 16 rotated dimensions out of 64 (value at step 400: 1.4501, against 1.5284 for 24 and 1.5576 for 8).
