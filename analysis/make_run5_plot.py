#!/usr/bin/env python3
"""Generate Run #5 results visualization matching Run #3 style."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('dark_background')
fig, axes = plt.subplots(2, 2, figsize=(16, 11))
fig.suptitle('H100 Run #5: Match+Layer Strategy (REGRESSION)',
             fontsize=16, fontweight='bold', color='#ff6b6b', y=0.98)
fig.text(0.5, 0.945, '0% pruning, 1.1148 sliding BPB (+0.0017 vs Run #3) — BUT quant damage cut 75%',
         ha='center', fontsize=11, color='#aaaaaa')

# Colors
C_R3 = '#00d4ff'   # cyan for Run #3 (best)
C_R5 = '#ff6b6b'   # red for Run #5 (regression)
C_LB = '#ffd700'    # gold for leaderboard
C_TTT = '#80ff80'   # green for TTT

# ── Panel 1: TTT BPB curve (we have this data) ──
ax1 = axes[0, 0]
# Parse TTT chunks from resume log
chunks, bpbs = [], []
with open('resume_run5.log') as f:
    for line in f:
        if 'ttt_chunk' in line and 'bpb=' in line:
            parts = line.strip().split()
            for p in parts:
                if p.startswith('['):
                    c = int(p.split('/')[0].lstrip('['))
                    chunks.append(c)
                if p.startswith('bpb='):
                    bpbs.append(float(p.split('=')[1]))

ax1.plot(chunks, bpbs, color=C_TTT, linewidth=1.5, alpha=0.6, label='Per-chunk BPB')
# Running average
window = 20
if len(bpbs) > window:
    avg = np.convolve(bpbs, np.ones(window)/window, mode='valid')
    ax1.plot(chunks[window-1:], avg, color=C_TTT, linewidth=2.5, label=f'Smoothed (w={window})')

ax1.axhline(y=1.1148, color=C_R5, linestyle='--', alpha=0.7, linewidth=1)
ax1.text(50, 1.1148+0.0005, 'sliding BPB 1.1148', color=C_R5, fontsize=9)
ax1.axhline(y=1.1109, color=C_TTT, linestyle='--', alpha=0.7, linewidth=1)
ax1.text(50, 1.1109-0.0015, 'final TTT BPB 1.1109', color=C_TTT, fontsize=9)

ax1.set_xlabel('TTT Chunk', fontsize=10)
ax1.set_ylabel('BPB (lower is better)', fontsize=10)
ax1.set_title('Panel 1 — TTT Adaptation Curve (Run #5)', fontsize=12, color='white')
ax1.legend(fontsize=9, loc='upper right')
ax1.set_ylim(1.095, 1.155)
ax1.grid(True, alpha=0.15)

# ── Panel 2: BPB Comparison bars (Run #3 vs Run #5) ──
ax2 = axes[0, 1]
metrics = ['Pre-EMA BPB', 'Quant BPB\n(roundtrip)', 'Sliding BPB', 'TTT BPB']
r3_vals = [1.0986, 1.1117, 1.1131, None]  # Run #3 roundtrip from run3 log
r5_vals = [1.1113, 1.1333, 1.1148, 1.1109]

x = np.arange(len(metrics))
width = 0.35

bars3 = ax2.bar(x - width/2, [v if v else 0 for v in r3_vals], width,
                label='Run #3 (best)', color=C_R3, alpha=0.85)
bars5 = ax2.bar(x + width/2, r5_vals, width,
                label='Run #5 (regression)', color=C_R5, alpha=0.85)

# Add delta annotations
deltas = [0.0127, 0.0216, 0.0017, None]
for i, d in enumerate(deltas):
    if d is not None and r3_vals[i] is not None:
        y_top = max(r3_vals[i] or 0, r5_vals[i]) + 0.001
        ax2.annotate(f'Δ+{d:.4f}', xy=(x[i], y_top),
                    ha='center', fontsize=8, color='#ff9999', fontweight='bold')

# TTT label
ax2.annotate('NEW', xy=(x[3] + width/2, r5_vals[3] + 0.001),
            ha='center', fontsize=8, color=C_TTT, fontweight='bold')

# Quant gap annotations
ax2.annotate('gap: 0.0145', xy=(x[1] - width/2, 1.095), ha='center',
            fontsize=7, color=C_R3, alpha=0.8)
ax2.annotate('gap: 0.0035', xy=(x[1] + width/2 + 0.01, 1.095), ha='center',
            fontsize=7, color='#80ff80', alpha=0.8,
            bbox=dict(boxstyle='round,pad=0.2', facecolor='#004400', alpha=0.5))

ax2.set_xticks(x)
ax2.set_xticklabels(metrics, fontsize=9)
ax2.set_ylabel('BPB (lower is better)', fontsize=10)
ax2.set_title('Panel 2 — Run #3 vs Run #5 BPB', fontsize=12, color='white')
ax2.legend(fontsize=9)
ax2.set_ylim(1.085, 1.145)
ax2.grid(True, alpha=0.15, axis='y')

# ── Panel 3: Budget Utilization ──
ax3 = axes[1, 0]

runs = ['Run #3\n(0% pruned)', 'Run #5\n(0% pruned)']
code_sizes = [0.060, 0.023229]  # MB  (Run #3 raw code, Run #5 LZMA)
model_sizes = [14.47, 15.93]  # MB
headroom = [16.0 - 0.060 - 14.47, 16.0 - 0.023229 - 15.93]

bars_code = ax3.bar(runs, code_sizes, color='#ff4488', label='Code', width=0.5)
bars_model = ax3.bar(runs, model_sizes, bottom=code_sizes, color='#4488ff', label='Model weights', width=0.5)
bars_head = ax3.bar(runs, headroom, bottom=[c+m for c,m in zip(code_sizes, model_sizes)],
                    color='#ffdd44', label='Headroom', width=0.5, alpha=0.6)

ax3.axhline(y=16.0, color='red', linestyle='--', linewidth=2, alpha=0.7)
ax3.text(1.3, 16.05, '16MB limit', color='red', fontsize=9, alpha=0.8)

# Size labels
for i, (c, m) in enumerate(zip(code_sizes, model_sizes)):
    ax3.text(i, c + m/2, f'{m:.2f}MB', ha='center', va='center', fontsize=11, fontweight='bold')
    ax3.text(i, c/2, f'{c*1000:.0f}KB' if c > 0.05 else f'{c*1000:.1f}KB',
            ha='center', va='center', fontsize=8)

ax3.set_ylabel('Size (MB)', fontsize=10)
ax3.set_title('Panel 3 — Budget Utilization', fontsize=12, color='white')
ax3.legend(fontsize=9, loc='upper left')
ax3.set_ylim(0, 17)
ax3.grid(True, alpha=0.15, axis='y')

# ── Panel 4: Leaderboard Position ──
ax4 = axes[1, 1]

entries = [
    ('#1 aryanbhosale', 1.0896, C_LB),
    ('#2 dexhunter', 1.0900, C_LB),
    ('Us — Run #3\n(sliding)', 1.1131, C_R3),
    ('Us — Run #5\n(sliding)', 1.1148, C_R5),
    ('Us — Run #5\n(TTT)', 1.1109, C_TTT),
]

y_pos = np.arange(len(entries))
colors = [e[2] for e in entries]
values = [e[1] for e in entries]
labels = [e[0] for e in entries]

bars = ax4.barh(y_pos, values, color=colors, alpha=0.8, height=0.6)
ax4.set_yticks(y_pos)
ax4.set_yticklabels(labels, fontsize=9)
ax4.set_xlabel('BPB (lower = better, bars extend right)', fontsize=10)
ax4.set_title('Panel 4 — Leaderboard Position', fontsize=12, color='white')
ax4.set_xlim(1.08, 1.12)
ax4.invert_yaxis()

# Value labels on bars
for i, v in enumerate(values):
    ax4.text(v + 0.0003, i, f'{v:.4f}', va='center', fontsize=10, fontweight='bold',
            color=colors[i])

# Gap annotation
gap = 1.1109 - 1.0896
ax4.annotate(f'gap = {gap:.4f} BPB to close',
            xy=(1.10, -0.4), fontsize=10, color='#ffaa00',
            ha='center', style='italic')

ax4.grid(True, alpha=0.15, axis='x')

plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig('./h100_run5_results.png', dpi=150,
            bbox_inches='tight', facecolor='#1a1a2e')
print("Saved to ./h100_run5_results.png")
