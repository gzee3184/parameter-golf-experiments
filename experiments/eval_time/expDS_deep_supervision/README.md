# expDS — Deep Supervision at Eval (Multi-Layer Logit Ensemble)

## Hypothesis
In tied-embedding transformers, intermediate-layer hidden states are projections onto the same "semantic basis" as the final hidden state. Applying the tied embedding matrix to an intermediate state yields a noisier version of the final logits. **Averaging logits from the last N decoder outputs** may reduce noise → lower CE loss → lower val_bpb.

No retraining needed. The r10 model has already learned intermediate states that support next-token prediction (because of the residual stream and skip connections). We're just harvesting that signal.

## Design

### What we extract
During the decoder phase of `_fb`, collect a list of "final-shape" hidden states from the last N layers. For our architecture:
- Parallel-lane layers (>= psl=7): after each parallel update, capture combined `m*l0 + (1-m)*l1` (same as final output formula)
- Non-parallel layers (< psl): capture `x`

### What we compute at eval
For each captured hidden state `h_i`:
```
logits_i = softcap * tanh(final_norm(h_i) @ tok_emb.T / softcap)
```
Then ensemble:
```
final_logits = sum(w_i * logits_i) / sum(w_i)
```

Where `w_i` are fixed weights (e.g., uniform, or geometric decay).

### Control check
With N=1 (only last state), the ensemble reduces to the baseline — should reproduce **1.1033**. If not, bug.

### Experiment arms
1. N=1 uniform (control, must equal 1.1033)
2. N=2 uniform (layers 9, 10)
3. N=3 uniform (layers 8, 9, 10)
4. N=3 weighted [0.2, 0.3, 0.5] (more weight on later layers)
5. N=3 weighted [0.1, 0.2, 0.7]

### Success criterion
Any arm ≥ 0.001 BPB better than 1.1033. Weighted arms if uniform hurts.

## Files
- `train_gpt.py` — modified: `_fb` captures intermediate states, new `fwl_ensemble` path
- `run_sweep.sh` — runs all arms via canonical path, greps post_ema BPB
- `results.md` — fills in as results come

## Novelty claim
Searched competitor code, our memory, EXPERIMENT_LOG. No prior art for multi-layer logit averaging at eval time in this competition.

## Revert
`cp ../baseline_r10/train_gpt.py ../../train_gpt.py` (no change to main code — this experiment is self-contained)
