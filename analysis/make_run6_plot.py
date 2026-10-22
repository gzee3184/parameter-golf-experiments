#!/usr/bin/env python3
"""Generate Run #6 results visualization — 3-run comparison."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('dark_background')
fig, axes = plt.subplots(2, 2, figsize=(16, 11))
fig.suptitle('H100 Run #6: Best of Both Strategy (REGRESSION)',
             fontsize=16, fontweight='bold', color='#ff6b6b', y=0.98)
fig.text(0.5, 0.945, 'Worst of 3 runs — Late QAT destroyed pre-quant quality. Run #3 (1.1131) remains best.',
         ha='center', fontsize=11, color='#aaaaaa')

C_R3 = '#00d4ff'   # cyan — Run #3 (best)
C_R5 = '#ff9944'   # orange — Run #5
C_R6 = '#ff4444'   # red — Run #6 (worst)
C_LB = '#ffd700'   # gold — leaderboard
C_TTT = '#80ff80'  # green — TTT

# ── Panel 1: Pre-quant BPB progression ──
ax1 = axes[0, 0]
runs = ['Run #3\n(baseline)', 'Run #5\n(competitor hp)', 'Run #6\n(+Late QAT)']
prequant = [1.0986, 1.1113, 1.1157]
colors_pq = [C_R3, C_R5, C_R6]

bars = ax1.bar(runs, prequant, color=colors_pq, alpha=0.85, width=0.55)
for b, v in zip(bars, prequant):
    ax1.text(b.get_x() + b.get_width()/2, v + 0.0005, f'{v:.4f}',
            ha='center', fontsize=11, fontweight='bold', color='white')

# Delta annotations
ax1.annotate('', xy=(1, 1.1113), xytext=(0, 1.0986),
            arrowprops=dict(arrowstyle='->', color='#ff9999', lw=2))
ax1.text(0.5, 1.105, '+0.0127', ha='center', fontsize=9, color='#ff9999')
ax1.annotate('', xy=(2, 1.1157), xytext=(0, 1.0986),
            arrowprops=dict(arrowstyle='->', color='#ff6666', lw=2))
ax1.text(1.3, 1.108, '+0.0171\n(Late QAT)', ha='center', fontsize=9, color='#ff6666')

ax1.set_ylabel('Pre-quant BPB (lower is better)', fontsize=10)
ax1.set_title('Panel 1 — Pre-quant Training Quality', fontsize=12, color='white')
ax1.set_ylim(1.090, 1.125)
ax1.grid(True, alpha=0.15, axis='y')

# ── Panel 2: Quant gap comparison ──
ax2 = axes[0, 1]
metrics = ['Pre-quant\nBPB', 'Quant gap\n(roundtrip-prequant)', 'Sliding\nBPB', 'TTT\nBPB']
r3 = [1.0986, 0.0131, 1.1131, None]
r5 = [1.1113, 0.0035, 1.1148, 1.1109]
r6 = [1.1157, 0.0041, 1.1198, 1.1151]

x = np.arange(len(metrics))
w = 0.25

# For quant gap panel, use different scale
bars3 = ax2.bar(x - w, [r3[0], r3[1]*10+1, r3[2], 0], w, label='Run #3', color=C_R3, alpha=0.85)
bars5 = ax2.bar(x,     [r5[0], r5[1]*10+1, r5[2], r5[3]], w, label='Run #5', color=C_R5, alpha=0.85)
bars6 = ax2.bar(x + w, [r6[0], r6[1]*10+1, r6[2], r6[3]], w, label='Run #6', color=C_R6, alpha=0.85)

# This doesn't work well with mixed scales. Let me redo as grouped bars with same scale.
ax2.clear()

metrics2 = ['Pre-quant', 'Roundtrip', 'Sliding', 'TTT']
r3v = [1.0986, 1.1117, 1.1131, 0]
r5v = [1.1113, 1.1333, 1.1148, 1.1109]
r6v = [1.1157, 1.1386, 1.1198, 1.1151]

x = np.arange(len(metrics2))
w = 0.25

b3 = ax2.bar(x - w, r3v, w, label='Run #3 (best)', color=C_R3, alpha=0.85)
b5 = ax2.bar(x, r5v, w, label='Run #5', color=C_R5, alpha=0.85)
b6 = ax2.bar(x + w, r6v, w, label='Run #6 (worst)', color=C_R6, alpha=0.85)

# Mark Run #3 TTT as N/A
ax2.text(3 - w, 0.001, 'N/A', ha='center', fontsize=9, color=C_R3)

ax2.set_xticks(x)
ax2.set_xticklabels(metrics2, fontsize=9)
ax2.set_ylabel('BPB (lower is better)', fontsize=10)
ax2.set_title('Panel 2 — All Metrics Comparison', fontsize=12, color='white')
ax2.legend(fontsize=9, loc='upper left')
ax2.set_ylim(1.085, 1.145)
ax2.grid(True, alpha=0.15, axis='y')

# ── Panel 3: Quant gap isolation ──
ax3 = axes[1, 0]
runs_qg = ['Run #3\n(no SDClip)', 'Run #5\n(SDClip+QAT)', 'Run #6\n(SDClip+QAT)']
gaps = [0.0131, 0.0035, 0.0041]
gap_colors = ['#ff6666', '#44ff44', '#88ff44']

bars_g = ax3.bar(runs_qg, gaps, color=gap_colors, alpha=0.85, width=0.55)
for b, v in zip(bars_g, gaps):
    ax3.text(b.get_x() + b.get_width()/2, v + 0.0003, f'{v:.4f}',
            ha='center', fontsize=12, fontweight='bold', color='white')

ax3.annotate('SDClip cuts\nquant damage\n69-73%', xy=(1.5, 0.008),
            fontsize=11, color='#44ff44', ha='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#003300', alpha=0.6))

ax3.set_ylabel('Quant Gap (roundtrip - pre-quant)', fontsize=10)
ax3.set_title('Panel 3 — Quantization Damage (lower is better)', fontsize=12, color='white')
ax3.set_ylim(0, 0.018)
ax3.grid(True, alpha=0.15, axis='y')

# ── Panel 4: Leaderboard Position ──
ax4 = axes[1, 1]
entries = [
    ('#1 aryanbhosale', 1.0896, C_LB),
    ('#2 dexhunter', 1.0900, C_LB),
    ('Run #3 sliding', 1.1131, C_R3),
    ('Run #5 TTT', 1.1109, C_R5),
    ('Run #6 TTT', 1.1151, C_R6),
    ('Run #7 projected', 1.098, '#80ff80'),
]

y_pos = np.arange(len(entries))
colors = [e[2] for e in entries]
values = [e[1] for e in entries]
labels = [e[0] for e in entries]

bars_lb = ax4.barh(y_pos, values, color=colors, alpha=0.8, height=0.55)
ax4.set_yticks(y_pos)
ax4.set_yticklabels(labels, fontsize=9)
ax4.set_xlabel('BPB (lower = better)', fontsize=10)
ax4.set_title('Panel 4 — Leaderboard + Projected Run #7', fontsize=12, color='white')
ax4.set_xlim(1.08, 1.125)
ax4.invert_yaxis()

for i, v in enumerate(values):
    ax4.text(min(v + 0.0003, 1.123), i, f'{v:.4f}', va='center', fontsize=10,
            fontweight='bold', color=colors[i])

# Run #7 dashed outline
ax4.barh([5], [1.098], color='none', edgecolor='#80ff80', linewidth=2,
         linestyle='--', height=0.55, alpha=0.8)

gap = 1.098 - 1.0896
ax4.annotate(f'Run #7 gap: {gap:.4f}',
            xy=(1.095, -0.4), fontsize=10, color='#80ff80',
            ha='center', style='italic')

ax4.grid(True, alpha=0.15, axis='x')

plt.subplots_adjust(left=0.07, right=0.97, top=0.91, bottom=0.06, hspace=0.35, wspace=0.25)
plt.savefig('./h100_run6_results.png', dpi=150,
            facecolor='#1a1a2e')
print("Saved to ./h100_run6_results.png")
