#!/usr/bin/env python3
"""
H100 Performance Predictor — Parameter Golf

Models cross-hardware scaling to predict H100 BPB from RTX PRO 6000 results.

Key insight: the relationship between hardware isn't a simple ratio on final BPB.
We decompose it into:
  1. Training effect: H100 gets more steps (better-trained model)
  2. TTT effect: AdamW TTT adapts to the specific val distribution
  3. Quantization effect: quantized model quality depends on training quality

We use per-step training curves to model (1) and TTT curves to model (2).
"""
import csv
import math
from collections import defaultdict
from pathlib import Path

ANALYSIS_DIR = Path(__file__).parent


def load_csv(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            for k in r:
                if r[k] == '' or r[k] is None:
                    r[k] = None
                    continue
                try:
                    if '.' in str(r[k]):
                        r[k] = float(r[k])
                    else:
                        r[k] = int(r[k])
                except (ValueError, TypeError):
                    pass
            rows.append(r)
    return rows


def main():
    summary = load_csv(ANALYSIS_DIR / "experiment_summary.csv")
    steps_raw = load_csv(ANALYSIS_DIR / "step_curves.csv")
    ttt_raw = load_csv(ANALYSIS_DIR / "ttt_curves.csv")

    # Organize
    steps = defaultdict(list)
    for r in steps_raw:
        steps[r['experiment']].append(r)
    ttt = defaultdict(list)
    for r in ttt_raw:
        ttt[r['experiment']].append(r)

    print("=" * 80)
    print("H100 PERFORMANCE PREDICTION MODEL")
    print("=" * 80)

    # ─── 1. Training Step Scaling ───
    print("\n--- 1. TRAINING STEP SCALING ---")
    print("How many more steps does H100 get per wallclock second?")

    rtx_runs = {s['name']: s for s in summary if s.get('hardware') == 'rtx_pro_6000'}
    h100_runs = {s['name']: s for s in summary if s.get('hardware') == 'h100'}

    # RTX: ~1550ms/step with world_size=1, grad_accum=8
    # H100: ~760ms/step with world_size=1, grad_accum=8 (from our single-GPU H100 runs)
    # H100 8-GPU competition: world_size=8, grad_accum=1 → much faster per-step
    # But each step processes same total batch (786432 tokens)

    rtx_step_ms = []
    for s in summary:
        if s.get('hardware') == 'rtx_pro_6000' and s.get('step_avg_ms'):
            rtx_step_ms.append(s['step_avg_ms'])
    h100_step_ms = []
    for s in summary:
        if s.get('hardware') == 'h100' and s.get('step_avg_ms'):
            h100_step_ms.append(s['step_avg_ms'])

    rtx_avg = sum(rtx_step_ms) / len(rtx_step_ms) if rtx_step_ms else 1550.0
    h100_1gpu_avg = sum(h100_step_ms) / len(h100_step_ms) if h100_step_ms else 760.0

    # With 8 GPUs: each GPU processes 1/8 of the batch
    # grad_accum goes from 8 to 1, so each step is ~1 micro-batch
    # Step time drops roughly proportional: ~760/8 ≈ 95ms, but communication overhead
    # Real-world 8xH100: ~100-120ms/step based on competition leaderboard times
    # Maio's blog (Mar 27 2026): measured 104.7ms/step on 8×H100 SXM Iceland pod
    # Our earlier estimate was 110ms. Using measured data.
    h100_8gpu_est = 104.7  # ms/step, from Maio's CoDA ablation (standard attention)

    rtx_steps_600s = int(600000 / rtx_avg)
    h100_1gpu_steps_600s = int(600000 / h100_1gpu_avg)
    h100_8gpu_steps_600s = int(600000 / h100_8gpu_est)

    print(f"  RTX PRO 6000 (1 GPU):  {rtx_avg:.0f}ms/step → {rtx_steps_600s} steps in 600s")
    print(f"  H100 (1 GPU):          {h100_1gpu_avg:.0f}ms/step → {h100_1gpu_steps_600s} steps in 600s")
    print(f"  H100 (8 GPU, est):     {h100_8gpu_est:.0f}ms/step → {h100_8gpu_steps_600s} steps in 600s")
    print(f"  Step ratio (8xH100/RTX): {h100_8gpu_steps_600s/rtx_steps_600s:.1f}x")

    # ─── 2. Val BPB vs Training Steps Power Law ───
    print("\n--- 2. VAL BPB vs TRAINING STEPS ---")
    print("Fitting power law: val_bpb = a * step^(-b) + c")

    # Collect (steps, val_bpb) from RTX runs
    step_bpb_pairs = []
    for name, s in rtx_runs.items():
        if s.get('final_step') and s.get('final_val_bpb'):
            step_bpb_pairs.append((s['final_step'], s['final_val_bpb']))

    if step_bpb_pairs:
        step_bpb_pairs.sort()
        print(f"  Data points: {len(step_bpb_pairs)}")
        print(f"  {'Steps':>8} {'Val BPB':>10}")
        seen_steps = set()
        for steps_n, bpb in step_bpb_pairs:
            if steps_n not in seen_steps:
                print(f"  {steps_n:>8} {bpb:>10.4f}")
                seen_steps.add(steps_n)

        # Simple log-linear fit: log(bpb) = a - b*log(steps)
        # Using only runs with >= 100 steps
        filtered = [(s, b) for s, b in step_bpb_pairs if s >= 100]
        if len(filtered) >= 3:
            log_steps = [math.log(s) for s, _ in filtered]
            log_bpb = [math.log(b) for _, b in filtered]
            n = len(filtered)
            sx = sum(log_steps)
            sy = sum(log_bpb)
            sxx = sum(x*x for x in log_steps)
            sxy = sum(x*y for x, y in zip(log_steps, log_bpb))

            b = (n * sxy - sx * sy) / (n * sxx - sx * sx)
            a = (sy - b * sx) / n

            print(f"\n  Power law fit: log(bpb) = {a:.4f} + {b:.4f} * log(steps)")
            print(f"  → bpb ≈ {math.exp(a):.2f} * steps^({b:.3f})")

            # Predict for H100 8-GPU step counts
            for h100_steps in [h100_8gpu_steps_600s, 6000, 7000]:
                predicted_bpb = math.exp(a + b * math.log(h100_steps))
                print(f"  At {h100_steps} steps: predicted val_bpb = {predicted_bpb:.4f}")

    # ─── 3. TTT Improvement Model ───
    print("\n--- 3. TTT IMPROVEMENT MODEL ---")
    print("How much does AdamW TTT improve BPB as a function of pre-quant quality?")

    ttt_data = []
    for name, s in rtx_runs.items():
        if (s.get('sliding_bpb') and s.get('ttt_bpb')
                and s['ttt_bpb'] < 2.0  # AdamW TTT only
                and s.get('final_val_bpb')):
            ttt_data.append({
                'name': name,
                'val_bpb': s['final_val_bpb'],
                'sliding_bpb': s['sliding_bpb'],
                'ttt_bpb': s['ttt_bpb'],
                'ttt_improvement': s['sliding_bpb'] - s['ttt_bpb'],
                'ttt_pct': (s['sliding_bpb'] - s['ttt_bpb']) / s['sliding_bpb'] * 100,
            })

    if ttt_data:
        ttt_data.sort(key=lambda x: x['ttt_bpb'])
        print(f"\n  {'Name':<35} {'ValBPB':>8} {'SlidBPB':>8} {'TTTBPB':>8} {'Δ':>8} {'%':>6}")
        for t in ttt_data[:10]:
            print(f"  {t['name'][:34]:<35} {t['val_bpb']:>8.4f} {t['sliding_bpb']:>8.4f} {t['ttt_bpb']:>8.4f} {t['ttt_improvement']:>8.4f} {t['ttt_pct']:>5.1f}%")

        # Key observation: TTT BPB clusters around 1.63-1.69 regardless of pre-quant BPB
        # This suggests TTT converges to a fixed point determined by the model + val data
        mean_ttt = sum(t['ttt_bpb'] for t in ttt_data) / len(ttt_data)
        std_ttt = (sum((t['ttt_bpb'] - mean_ttt)**2 for t in ttt_data) / len(ttt_data)) ** 0.5
        print(f"\n  AdamW TTT BPB: mean={mean_ttt:.4f} std={std_ttt:.4f}")
        print(f"  Range: {min(t['ttt_bpb'] for t in ttt_data):.4f} - {max(t['ttt_bpb'] for t in ttt_data):.4f}")

    # ─── 4. H100 Prediction ───
    print("\n" + "=" * 80)
    print("FINAL H100 PREDICTIONS")
    print("=" * 80)

    # Known data points:
    # H100 1-GPU, 600s, SGD TTT, mish2: 0.8203 BPB (770 training steps)
    # H100 1-GPU, 600s, no TTT (sliding): 1.1689 BPB
    # RTX 1-GPU, 600s, AdamW TTT, mish2 int8/int6: 1.635 BPB (384 steps)
    # RTX 1-GPU, 300s, AdamW TTT, mish2 int8/int5: 1.651 BPB (185 steps)

    # SGD TTT improvement on H100: 1.1689 → 0.8203 = -0.3486 (29.8%)
    # AdamW TTT improvement on RTX: ~3.37 → ~1.65 = -1.72 (51.0%)
    # AdamW TTT is ~1.7x better than SGD TTT on RTX

    h100_sgd_sliding = 1.1689
    h100_sgd_ttt = 0.8203
    sgd_improvement = h100_sgd_sliding - h100_sgd_ttt  # 0.3486

    print(f"\n  Known H100 results (1-GPU, 600s, mish2, SGD TTT):")
    print(f"    Sliding BPB:   {h100_sgd_sliding}")
    print(f"    SGD TTT BPB:   {h100_sgd_ttt}")
    print(f"    Improvement:   {sgd_improvement:.4f} ({sgd_improvement/h100_sgd_sliding*100:.1f}%)")

    # Prediction Method A: Scale SGD improvement by AdamW/SGD ratio
    # On RTX: AdamW gets ~51% improvement, SGD gets ~17% improvement
    # Ratio: ~3x more improvement with AdamW
    # On H100: SGD gets 29.8% improvement → AdamW should get ~3x = ~60-90%?
    # But there are diminishing returns at lower BPB...

    # Method B: Model TTT as converging to a hardware-dependent fixed point
    # On RTX: TTT converges to ~1.65 regardless of pre-quant quality
    # On H100 with SGD: converges to ~0.82
    # H100 TTT fixed point with AdamW: should be LOWER than 0.82

    # Method C: Use absolute ratio
    # RTX AdamW TTT / RTX SGD TTT = 1.635 / 2.825 = 0.579
    # Apply to H100: 0.820 * 0.579 = 0.475 (very optimistic)

    # Method D: Use the fact that H100 8-GPU gets ~14x more training steps
    # Better training → lower pre-quant BPB → lower sliding BPB → TTT starts from better point
    # With 5400 steps (8xH100), pre-quant BPB would be much lower (~1.1-1.2)
    # Quantization degradation is smaller for better-trained models

    rtx_adamw_sgd_ratio = 1.635 / 2.825
    h100_adamw_est_c = h100_sgd_ttt * rtx_adamw_sgd_ratio

    # More conservative: additive model
    # AdamW vs SGD advantage on RTX: 2.825 - 1.635 = 1.19
    # Scale down for H100 (diminishing returns at lower BPB)
    rtx_adamw_advantage = 2.825 - 1.635  # 1.19
    # At H100's lower BPB, the advantage shrinks (can't go below 0)
    # Scale by (h100_sgd / rtx_sgd) to account for absolute BPB level
    scale = h100_sgd_ttt / 2.825
    h100_adamw_est_d = h100_sgd_ttt - rtx_adamw_advantage * scale

    print(f"\n  Prediction methods:")
    print(f"    A. Multiplicative ratio ({rtx_adamw_sgd_ratio:.3f}):  {h100_adamw_est_c:.4f}")
    print(f"    B. Additive (scaled advantage):     {h100_adamw_est_d:.4f}")

    # Method E: Direct interpolation from our data
    # The key missing piece: what does AdamW TTT achieve on a well-trained model?
    # Our comparison_leaderboard run (9742 steps, 4hrs) got 1.107 BPB with AdamW TTT
    # That was with a VERY well-trained model (val_bpb=1.127 pre-quant)
    # H100 8-GPU in 600s would get ~5400 steps → val_bpb ~ 1.2-1.3 (interpolating)

    best_rtx = min(
        (s for s in summary
         if s.get('hardware') == 'rtx_pro_6000'
         and s.get('ttt_bpb')
         and s['ttt_bpb'] < 2.0
         and s.get('max_wallclock_s', 0) <= 700),
        key=lambda x: x['ttt_bpb'],
        default=None
    )

    if best_rtx:
        print(f"\n  Best RTX result (≤600s wallclock):")
        print(f"    {best_rtx['name']}: TTT={best_rtx['ttt_bpb']:.4f}, "
              f"steps={best_rtx.get('final_step')}, "
              f"artifact={best_rtx.get('artifact_bytes', 0)/1024/1024:.1f}MB")

    # ─── 5. TTT Saturation-Aware Prediction ───
    print(f"\n--- 5. TTT SATURATION-AWARE MODEL ---")
    print("Key insight: TTT improvement SATURATES as base model quality improves.")
    print("Evidence: comparison runs (9742 steps) show 0% TTT improvement.")

    # We observe TTT improvement depends on the gap between sliding_bpb and val_bpb
    # i.e. how much quantization hurts. When quant degradation is large, TTT helps a lot.
    # When the model is well-trained (low val_bpb), quant degradation is smaller,
    # and TTT has less to recover.

    # Collect (val_bpb, sliding_bpb, ttt_bpb) triples for AdamW TTT runs
    if ttt_data:
        # Sort by val_bpb (proxy for training quality)
        ttt_data.sort(key=lambda x: x['val_bpb'])
        print(f"\n  TTT improvement vs training quality:")
        print(f"  {'ValBPB':>8} {'SlidBPB':>8} {'TTTBPB':>8} {'QuantDeg':>9} {'TTTRecov':>9} {'RecovPct':>9}")
        for t in ttt_data:
            quant_deg = t['sliding_bpb'] - t['val_bpb']  # how much quant hurts
            ttt_recov = t['sliding_bpb'] - t['ttt_bpb']  # how much TTT recovers
            recov_pct = (ttt_recov / quant_deg * 100) if quant_deg > 0.01 else 0
            t['quant_deg'] = quant_deg
            t['ttt_recov'] = ttt_recov
            t['recov_pct'] = recov_pct
            print(f"  {t['val_bpb']:>8.4f} {t['sliding_bpb']:>8.4f} {t['ttt_bpb']:>8.4f} "
                  f"{quant_deg:>+9.4f} {ttt_recov:>9.4f} {recov_pct:>8.1f}%")

    # H100 8-GPU prediction chain:
    # 1. Power law: ~5454 steps → val_bpb ≈ 1.105 (pre-quant)
    # 2. Quantization: at val_bpb=1.105, quant degradation is small
    #    Our comparison runs: val_bpb=1.127 → sliding=1.107 (quant IMPROVED! negative degradation)
    #    This means at high training quality, int6 quant barely hurts or even helps (regularization)
    # 3. TTT: with near-zero quant degradation, TTT has nothing to recover
    #    → H100 TTT BPB ≈ sliding BPB ≈ val_bpb (or slightly better)

    print(f"\n  === REVISED H100 8×GPU Prediction (Saturation-Aware) ===")

    # Power law predictions for H100 step counts
    if len(filtered) >= 3:
        for h100_steps, label in [(5454, "8×H100 600s"), (4000, "8×H100 440s"), (3000, "8×H100 330s")]:
            pred_val = math.exp(a + b * math.log(h100_steps))

            # At high step counts, quant degradation is near-zero or negative
            # Interpolate: at 384 steps, quant_deg ≈ +2.7; at 9742 steps, quant_deg ≈ -0.02
            # Linear interpolation in log-space
            log_384 = math.log(384)
            log_9742 = math.log(9742)
            log_h = math.log(h100_steps)
            # Known points: (384 steps, ~2.7 quant_deg) and (9742 steps, ~-0.02 quant_deg)
            frac = (log_h - log_384) / (log_9742 - log_384)
            est_quant_deg = 2.7 * (1 - frac) + (-0.02) * frac
            est_sliding = pred_val + est_quant_deg

            # TTT recovery: at high quality, recovery approaches 0
            # At 384 steps: recovery ≈ 2.9 (64%), at 9742 steps: recovery ≈ 0 (0%)
            est_ttt_recov = max(0, 2.9 * (1 - frac))
            est_ttt = est_sliding - est_ttt_recov

            # But TTT BPB can't be worse than val_bpb (pre-quant)
            est_ttt = min(est_ttt, pred_val)
            # And can't be better than the absolute best we've seen
            est_ttt = max(est_ttt, 0.5)

            print(f"\n  {label} (~{h100_steps} steps):")
            print(f"    Predicted val_bpb (pre-quant): {pred_val:.4f}")
            print(f"    Est. quant degradation:        {est_quant_deg:+.4f}")
            print(f"    Est. sliding_bpb:              {est_sliding:.4f}")
            print(f"    Est. TTT recovery:             {est_ttt_recov:.4f}")
            print(f"    → Predicted final BPB:         {est_ttt:.4f}")

    # Also show the simple SGD-ratio prediction for comparison
    print(f"\n  Naive ratio prediction (for comparison): {h100_adamw_est_c:.4f}")
    print(f"  (This is unrealistic — ignores saturation)")

    print(f"\n  Current #1 (Abay): 1.1086 BPB")
    print(f"\n  Key finding: With enough training steps, the bottleneck shifts from")
    print(f"  'recover quant damage via TTT' to 'train a better base model'.")
    print(f"  H100 8-GPU advantage is primarily MORE TRAINING STEPS, not better TTT.")
    print(f"  AdamW TTT is still our secret weapon for SHORT training (RTX-like scenarios).")

    print(f"\n  ⚠ REMAINING UNCERTAINTY:")
    print(f"    1. Power law extrapolation beyond our data range")
    print(f"    2. 8-GPU batch parallelism may change training dynamics")
    print(f"    3. Mixed quant (int8/int6) behavior at very low val_bpb unknown")
    print(f"    4. TTT may still help even at low quant degradation (fine-grained adaptation)")
    print(f"\n  → GCP test run remains essential for calibration")


if __name__ == '__main__':
    main()
