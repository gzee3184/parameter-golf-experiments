#!/usr/bin/env python3
"""
Cross-hardware analysis and prediction framework.

Compares RTX PRO 6000 vs H100 training curves, models scaling factors,
and builds a simple predictor for H100 performance from local runs.
"""
import csv
import os
from collections import defaultdict
from pathlib import Path

ANALYSIS_DIR = Path(__file__).parent
SUMMARY_CSV = ANALYSIS_DIR / "experiment_summary.csv"
STEPS_CSV = ANALYSIS_DIR / "step_curves.csv"
TTT_CSV = ANALYSIS_DIR / "ttt_curves.csv"


def load_summary():
    rows = []
    with open(SUMMARY_CSV) as f:
        for r in csv.DictReader(f):
            # Convert numeric fields
            for k in r:
                if r[k] == '':
                    r[k] = None
                    continue
                try:
                    if '.' in r[k]:
                        r[k] = float(r[k])
                    else:
                        r[k] = int(r[k])
                except (ValueError, TypeError):
                    pass
            rows.append(r)
    return rows


def load_steps():
    data = defaultdict(list)
    with open(STEPS_CSV) as f:
        for r in csv.DictReader(f):
            exp = r['experiment']
            for k in r:
                if k == 'experiment':
                    continue
                if r[k]:
                    try:
                        r[k] = float(r[k])
                    except ValueError:
                        pass
            data[exp].append(r)
    return dict(data)


def load_ttt():
    data = defaultdict(list)
    with open(TTT_CSV) as f:
        for r in csv.DictReader(f):
            exp = r['experiment']
            for k in r:
                if k == 'experiment':
                    continue
                if r[k]:
                    try:
                        r[k] = float(r[k])
                    except ValueError:
                        pass
            data[exp].append(r)
    return dict(data)


def cross_hardware_analysis(summary, steps):
    """Compare RTX vs H100 at matched configs."""
    print("\n" + "=" * 80)
    print("CROSS-HARDWARE ANALYSIS: RTX PRO 6000 vs H100")
    print("=" * 80)

    rtx = [s for s in summary if s.get('hardware') == 'rtx_pro_6000']
    h100 = [s for s in summary if s.get('hardware') == 'h100']

    print(f"\nRTX runs: {len(rtx)}, H100 runs: {len(h100)}")

    # Step timing comparison
    rtx_avgs = [s['step_avg_ms'] for s in rtx if s.get('step_avg_ms')]
    h100_avgs = [s['step_avg_ms'] for s in h100 if s.get('step_avg_ms')]

    if rtx_avgs and h100_avgs:
        rtx_mean = sum(rtx_avgs) / len(rtx_avgs)
        h100_mean = sum(h100_avgs) / len(h100_avgs)
        speedup = rtx_mean / h100_mean
        print(f"\nStep timing:")
        print(f"  RTX PRO 6000:  {rtx_mean:.1f}ms/step (mean)")
        print(f"  H100:          {h100_mean:.1f}ms/step (mean)")
        print(f"  H100 speedup:  {speedup:.2f}x")
        print(f"  In 600s: RTX gets ~{int(600000/rtx_mean)} steps, H100 gets ~{int(600000/h100_mean)} steps")

    # Per-step val_bpb comparison (matched step counts)
    print("\n--- Val BPB at matched steps ---")
    # Find H100 experiments with per-step data
    h100_with_steps = {name: s for name, s in steps.items()
                       if any(r.get('hardware') == 'h100' for r in summary if r.get('name') == name)}
    rtx_with_steps = {name: s for name, s in steps.items()
                      if any(r.get('hardware') == 'rtx_pro_6000' for r in summary if r.get('name') == name)}

    # Compare at common step milestones
    milestones = [50, 100, 150, 200, 300, 400, 500, 600, 700, 800]

    if h100_with_steps and rtx_with_steps:
        # Pick one representative H100 and RTX run
        h100_name = next(iter(h100_with_steps))
        h100_steps = h100_with_steps[h100_name]

        # Find a comparable RTX run (600s, similar config)
        rtx_600s = [name for name, s_list in rtx_with_steps.items()
                    if any(r.get('max_wallclock_s') == 600.0 for r in summary if r.get('name') == name)]

        if rtx_600s:
            rtx_name = rtx_600s[0]
            rtx_steps_data = rtx_with_steps[rtx_name]

            print(f"\n  H100: {h100_name}")
            print(f"  RTX:  {rtx_name}")
            print(f"  {'Step':>6} {'H100 BPB':>10} {'RTX BPB':>10} {'Ratio':>8}")
            print(f"  {'-'*36}")

            h100_by_step = {int(s['step']): s for s in h100_steps if s.get('val_bpb')}
            rtx_by_step = {int(s['step']): s for s in rtx_steps_data if s.get('val_bpb')}

            ratios = []
            for ms in milestones:
                h_val = h100_by_step.get(ms, {}).get('val_bpb')
                r_val = rtx_by_step.get(ms, {}).get('val_bpb')
                if h_val and r_val:
                    ratio = r_val / h_val
                    ratios.append(ratio)
                    print(f"  {ms:>6} {h_val:>10.4f} {r_val:>10.4f} {ratio:>8.4f}")

            if ratios:
                mean_ratio = sum(ratios) / len(ratios)
                print(f"\n  Mean BPB ratio (RTX/H100): {mean_ratio:.4f}")
                print(f"  => To predict H100 BPB: multiply RTX BPB by ~{1/mean_ratio:.4f}")

    # TTT analysis
    print("\n--- TTT Performance Comparison ---")
    h100_ttt = [(s['name'], s['ttt_bpb']) for s in h100 if s.get('ttt_bpb')]
    rtx_ttt = [(s['name'], s['ttt_bpb']) for s in rtx if s.get('ttt_bpb')]

    if h100_ttt:
        print(f"\n  H100 TTT results:")
        for name, bpb in sorted(h100_ttt, key=lambda x: x[1]):
            print(f"    {name:<40} {bpb:.6f}")

    if rtx_ttt:
        print(f"\n  RTX TTT results (top 10):")
        for name, bpb in sorted(rtx_ttt, key=lambda x: x[1])[:10]:
            print(f"    {name:<40} {bpb:.6f}")

    # Cross-hardware TTT prediction
    if h100_ttt and rtx_ttt:
        best_h100_ttt = min(bpb for _, bpb in h100_ttt)
        best_rtx_ttt = min(bpb for _, bpb in rtx_ttt)

        # The H100 runs used SGD TTT, our RTX best uses AdamW TTT
        # SGD TTT on RTX: ~2.825 BPB -> SGD TTT on H100: ~0.820
        # AdamW TTT on RTX: ~1.635 BPB -> AdamW TTT on H100: ???

        # Method 1: Same ratio
        sgd_rtx = None
        for name, bpb in rtx_ttt:
            if 'standard_sgd' in name:
                sgd_rtx = bpb
                break

        if sgd_rtx:
            sgd_ratio = best_h100_ttt / sgd_rtx
            adamw_predicted = best_rtx_ttt * sgd_ratio
            print(f"\n  --- Cross-Hardware TTT Prediction ---")
            print(f"  SGD TTT:   RTX={sgd_rtx:.4f}  H100={best_h100_ttt:.4f}  ratio={sgd_ratio:.4f}")
            print(f"  AdamW TTT: RTX={best_rtx_ttt:.4f}  H100=???")
            print(f"  Prediction (same ratio):     {adamw_predicted:.4f}")
            print(f"  Prediction (conservative):   {adamw_predicted * 1.05:.4f}")
            print(f"  Current #1 (Abay):           1.1086")


def training_efficiency_analysis(summary):
    """Analyze how training duration affects outcomes."""
    print("\n" + "=" * 80)
    print("TRAINING DURATION vs OUTCOME")
    print("=" * 80)

    rtx = [s for s in summary if s.get('hardware') == 'rtx_pro_6000' and s.get('ttt_bpb')]

    # Group by approximate wallclock
    groups = defaultdict(list)
    for s in rtx:
        wc = s.get('max_wallclock_s', 300)
        if wc <= 100:
            groups['short(≤100s)'].append(s)
        elif wc <= 350:
            groups['medium(300s)'].append(s)
        else:
            groups['long(600s)'].append(s)

    for group_name in ['short(≤100s)', 'medium(300s)', 'long(600s)']:
        exps = groups.get(group_name, [])
        if not exps:
            continue
        best = min(exps, key=lambda x: x['ttt_bpb'])
        mean_bpb = sum(x['ttt_bpb'] for x in exps) / len(exps)
        print(f"\n  {group_name}: {len(exps)} runs")
        print(f"    Best TTT BPB: {best['ttt_bpb']:.6f} ({best['name']})")
        print(f"    Mean TTT BPB: {mean_bpb:.6f}")
        print(f"    Steps range:  {min(x.get('final_step', 0) for x in exps)}-{max(x.get('final_step', 0) for x in exps)}")


def quantization_analysis(summary):
    """Analyze quantization impact."""
    print("\n" + "=" * 80)
    print("QUANTIZATION ANALYSIS")
    print("=" * 80)

    rtx_quant = [s for s in summary
                 if s.get('hardware') == 'rtx_pro_6000'
                 and s.get('quant_bits_mlp') is not None
                 and s.get('ttt_bpb')]

    if not rtx_quant:
        print("  No quantization data available")
        return

    print(f"\n  {'Config':<30} {'MLP bits':>8} {'Attn bits':>9} {'TTT BPB':>8} {'Artifact':>10}")
    print(f"  {'-'*70}")
    for s in sorted(rtx_quant, key=lambda x: x['ttt_bpb']):
        name = s['name'][:29]
        mlp = s.get('quant_bits_mlp', '?')
        attn = s.get('quant_bits_attn', '?')
        ttt = s['ttt_bpb']
        art = s.get('artifact_bytes')
        art_str = f"{art/1024/1024:.2f}MB" if art else "?"
        print(f"  {name:<30} {mlp:>8} {attn:>9} {ttt:>8.4f} {art_str:>10}")

    # Impact analysis
    configs = defaultdict(list)
    for s in rtx_quant:
        key = f"int{s.get('quant_bits_mlp', '?')}/int{s.get('quant_bits_attn', '?')}"
        configs[key].append(s['ttt_bpb'])

    print(f"\n  Quantization summary:")
    for key, bpbs in sorted(configs.items(), key=lambda x: min(x[1])):
        print(f"    {key}: best={min(bpbs):.4f} mean={sum(bpbs)/len(bpbs):.4f} n={len(bpbs)}")


def ttt_optimizer_analysis(summary):
    """Analyze TTT optimizer impact."""
    print("\n" + "=" * 80)
    print("TTT OPTIMIZER ANALYSIS")
    print("=" * 80)

    rtx_ttt = [s for s in summary
               if s.get('hardware') == 'rtx_pro_6000'
               and s.get('ttt_bpb') is not None]

    # Categorize by TTT BPB ranges (proxy for optimizer since we know the configs)
    # AdamW TTT: < 2.0, SGD TTT: 2.0-3.0, No TTT or broken: > 3.0
    adamw = [s for s in rtx_ttt if s['ttt_bpb'] < 2.0]
    sgd = [s for s in rtx_ttt if 2.0 <= s['ttt_bpb'] < 3.0]
    broken = [s for s in rtx_ttt if s['ttt_bpb'] >= 3.0]

    print(f"\n  AdamW TTT (<2.0 BPB): {len(adamw)} runs")
    if adamw:
        best = min(adamw, key=lambda x: x['ttt_bpb'])
        print(f"    Best: {best['ttt_bpb']:.6f} ({best['name']})")
        print(f"    Mean: {sum(x['ttt_bpb'] for x in adamw)/len(adamw):.6f}")

    print(f"\n  SGD TTT (2.0-3.0 BPB): {len(sgd)} runs")
    if sgd:
        best = min(sgd, key=lambda x: x['ttt_bpb'])
        print(f"    Best: {best['ttt_bpb']:.6f} ({best['name']})")

    print(f"\n  Broken/No TTT (>3.0 BPB): {len(broken)} runs")

    # AdamW TTT improvement over sliding window
    if adamw:
        print(f"\n  AdamW TTT improvement over sliding window:")
        for s in sorted(adamw, key=lambda x: x['ttt_bpb'])[:5]:
            slid = s.get('sliding_bpb')
            ttt = s['ttt_bpb']
            if slid:
                delta = slid - ttt
                pct = delta / slid * 100
                print(f"    {s['name'][:35]:<35} sliding={slid:.4f} ttt={ttt:.4f} Δ={delta:.4f} ({pct:.1f}%)")


def main():
    summary = load_summary()
    steps = load_steps()

    cross_hardware_analysis(summary, steps)
    training_efficiency_analysis(summary)
    quantization_analysis(summary)
    ttt_optimizer_analysis(summary)

    # Final prediction
    print("\n" + "=" * 80)
    print("H100 PERFORMANCE PREDICTIONS (Best Config)")
    print("=" * 80)

    best_rtx = min(
        (s for s in summary if s.get('hardware') == 'rtx_pro_6000' and s.get('ttt_bpb')),
        key=lambda x: x['ttt_bpb']
    )
    print(f"\n  Best local result: {best_rtx['name']}")
    print(f"    TTT BPB:    {best_rtx['ttt_bpb']:.6f}")
    print(f"    Artifact:   {best_rtx.get('artifact_bytes', 0)/1024/1024:.2f}MB")
    print(f"    Steps:      {best_rtx.get('final_step', '?')}")
    print(f"    Wallclock:  {best_rtx.get('max_wallclock_s', '?')}s")

    # On H100 with 8 GPUs, 600s → ~6000+ steps (vs 384 on RTX)
    # The model would be MUCH better trained
    # Conservative: multiply by SGD ratio
    h100_sgd_ttt = 0.8203  # known from our H100 run
    rtx_sgd_ttt = 2.8247   # from ttt_standard_sgd
    ratio = h100_sgd_ttt / rtx_sgd_ttt

    predicted = best_rtx['ttt_bpb'] * ratio
    print(f"\n  Prediction methods:")
    print(f"    Method 1 (SGD ratio {ratio:.4f}):       {predicted:.4f}")
    print(f"    Method 2 (conservative +5%):    {predicted * 1.05:.4f}")
    print(f"    Method 3 (pessimistic +10%):    {predicted * 1.10:.4f}")
    print(f"\n  Current #1 (Abay):               1.1086")
    print(f"  Would beat #1?  {predicted < 1.1086} (optimistic), {predicted*1.10 < 1.1086} (pessimistic)")


if __name__ == '__main__':
    main()
