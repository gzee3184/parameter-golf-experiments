# expA — Kernel Fusion + Compile Hessian

**Status**: Downgraded from TIER-S to TIER-C after discovery that `_zpns` (NS5) already has `@torch.compile` at baseline line 64. The 12-day-old memory claiming it was missing is stale.

## What's actually testable
| Target | Status | Est. savings |
|--------|--------|--------------|
| NS5 (`_zpns`) | Already compiled | 0 (already done) |
| Training fwd/bwd | Already compiled (line 986) | 0 |
| Eval fwl | Already compiled (line 1229) | 0 |
| **Hessian model `_HG` forward** | **NOT compiled** — runs eager during `_cht` | ~10s on GPTQ → ~80 extra training steps → ~0.001 BPB |
| **`_gac` calibration generation** | Eager autoregressive loop | Skipped when `gcrd=1` (r10c default) |

## Hypothesis
Compiling `_HG.forward` used by `_cht` will reduce Hessian collection from ~10s to ~3s.

## Risk
`_cht` uses forward hooks that write to a Python dict. `fullgraph=True` will break this. Must use `fullgraph=False` or restructure to capture activations without hooks.

## Test plan (local)
1. Baseline: time `_cht` with 32 calibration sequences on local GPU
2. Modified: `torch.compile(hgm, fullgraph=False)` before `_cht` call
3. Compare wall-clock

## Success criterion
- Hessian collection ≥ 2× faster AND
- Produces numerically equivalent Hessians (cosine similarity > 0.999 vs eager)

## Files
- `train_gpt.py` — modified baseline (diff at bottom of this README)
- `run_local.sh` — launches local benchmark
- `bench_hessian.py` — standalone Hessian timing script (no full training needed)

## Revert
```bash
cp ../baseline_r10/train_gpt.py ../../train_gpt.py
```
