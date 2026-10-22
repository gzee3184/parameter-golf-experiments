# expZ — Z-Loss Regularization

## Hypothesis
Adding `Z_LOSS_WEIGHT * logsumexp(logits)²` to training CE (PaLM/Gemma technique) keeps logits tight. Two plausible mechanisms:
1. **Quant tolerance**: tighter logit distribution → smaller dynamic range per layer → less clipping damage under int5/int6
2. **Gradient stability**: bounded logits → softmax probabilities don't saturate → healthier gradients through the tanh softcap

Competitor Ternary uses this with `1e-4` weight. Our codebase doesn't have it.

## Two-phase test plan

### Phase 1: Fine-tune from r10 with Z-Loss (cheap, ~30-60 min)
- Load r10 checkpoint
- Continue training for ~200-500 steps with Z-Loss added
- Compare post-EMA val_bpb:
  - Control: same fine-tune, Z-Loss weight = 0
  - Treatment: same fine-tune, Z-Loss weight = 1e-4
- If treatment ≤ control by ≥ 0.001 BPB, proceed to Phase 2

### Phase 2: From-scratch comparison (expensive, ~4h)
- Fresh 3000-step training, both arms same seed
- Compare post-EMA val_bpb

## Implementation
Modify `GPT.forward` to compute raw-logit `lp` pre-softcap, add Z-Loss term to CE:

```python
z_w = float(os.environ.get("Z_LOSS_WEIGHT", "0"))
if z_w > 0:
    # logsumexp over vocab dim of raw pre-softcap logits
    zloss = (torch.logsumexp(lp.float(), dim=-1) ** 2).mean()
    loss = loss + z_w * zloss
```

## Files
- `train_gpt.py` — copy of baseline with Z_LOSS_WEIGHT added
- `run_finetune.sh` — Phase 1: fine-tune from r10
- `run_scratch.sh` — Phase 2: from-scratch (only if Phase 1 shows signal)

## Success criterion
Phase 1: ≥ 0.001 BPB improvement. Phase 2 (if reached): ≥ 0.003 BPB improvement for H100 deploy.
