# Gradient analysis

This note records a per-layer gradient measurement on a trained local model, made on 2026-04-07. The aim was to find layers that receive too little learning signal.

## Method

The script is [`analysis/gradient_analysis.py`](../../analysis/gradient_analysis.py). It does these steps:

1. It builds the model and loads a trained checkpoint (EMA weights).
2. It switches recurrence on. This step is required. Without it, the model runs without the repeated layers and the results are wrong.
3. It draws 8 batches of 16 sequences of 2048 tokens from the validation file.
4. It runs forward and backward on each batch and records the gradient norm of every weight bank and every scalar parameter.
5. It averages the norms over the batches.

The measured model had these settings:

- 11 layers, dimension 512, SP4096 vocabulary, MLP 4x.
- Recurrence on layers 3, 4 and 5 with two passes, which gives 14 virtual layers.
- Parallel residuals from layer 7, value embeddings on layers 9 and 10, a LoRA adapter of rank 16 on the repeated MLP, and layer 0 attention off.
- Logit softcap 30 and layer-norm scaling on (`LN_SCALE=1`).

The checkpoint came from a local development model (v7 or v8). The script loads the model class from a development file that is not in this repository. See the note in [../../analysis/README.md](../../analysis/README.md).

## Results

### MLP up-projection gradient norm by layer

| Layer | Norm | Role |
|---|---|---|
| 0 | 0.084 | attention off |
| 1 | 0.268 | serial |
| 2 | 0.384 | serial |
| 3 | 0.370 | recurrence |
| 4 | 0.429 | recurrence |
| 5 | 0.447 | recurrence, largest |
| 6 | 0.252 | last serial layer |
| 7 | 0.177 | first parallel layer |
| 8 | 0.123 | parallel |
| 9 | 0.125 | parallel |
| 10 | 0.245 | output |

### Group averages

| Group | Layers | Average norm | Ratio to parallel |
|---|---|---|---|
| Recurrence | 3, 4, 5 | 0.415 | 2.9 |
| Serial | 1, 2, 6 | 0.301 | 2.1 |
| Parallel | 7, 8, 9 | 0.142 | 1.0 |
| Output | 10 | 0.245 | 1.7 |

The recurrence layers use their weights twice, so their gradients add up. The parallel layers split their signal over two lanes. The model also had layer-norm scaling on, which reduces the signal in deep layers.

### Other findings

- **Skip connection.** The fifth skip connection (index 4) received a non-zero gradient (0.053) after a code fix. Before the fix it received exactly zero.
- **Lane merge.** The scalar that merges the two parallel lanes had the largest gradient of all scalars (0.0875). This suggested a separate merge value for each layer.
- **Repeat pass scale.** The per-pass scales had gradients of 0.006, 0.023 and 0.076 for layers 3, 4 and 5. The scale for layer 5 was 0.921 and still changing.
- **Layer 5 attention.** The output projection and the value projection of layer 5 had exactly zero gradient in the repeat pass. Only its MLP carried signal.
- **Layer 7.** The first parallel layer had the lowest values of several metrics: attention scale gradient 0.007, residual-mix gradient 0.037 and QK gain gradient 0.002.
- **Layer 0 MLP scale.** This had the largest block-scalar gradient (0.247). The model pushed the scale of this MLP toward zero.

### Skip connection gradients

| Skip | Weight gradient | Gate gradient |
|---|---|---|
| 0 | 0.020 | 0.003 |
| 1 | 0.026 | 0.012 |
| 2 | 0.052 | 0.036 |
| 3 | 0.063 | 0.048 |
| 4 | 0.053 | 0.020 |

## Ideas that followed

The analysis gave these ideas:

1. A learning-rate multiplier of two to three for the parallel layers.
2. A separate lane-merge value per layer. The development code has this as `PER_LAYER_LANE_MERGE`. The Run #10c script has one scalar.
3. Removal of the layer 0 MLP.
4. Skipping attention in the repeat pass of layer 5.
5. A study of the gradient drop at layer 7.

Tests of these ideas ran in the v8 and v8.5 local sweeps (`sweeps/local/v8_sweep_r1` to `r5` and `sweeps/local/v8.5_sweep`). The final deployed model kept one lane-merge scalar and kept the layer 0 MLP. The sweep logs hold the details.

## Limits of this measurement

- The sanity-check loss of the script was 5.05, higher than expected for a model at about 1.10 BPB. The likely cause is that the batches were raw consecutive windows and not the sliding-window evaluation. Relative gradients stay valid, because all layers see the same loss.
- The measured model differs from the final model. The final Run #10c model uses the SP8192 vocabulary, softcap 20, layer-norm scaling off and no value embeddings. The imbalance was not measured again on that model.
- The measurement used eight batches from one file.
