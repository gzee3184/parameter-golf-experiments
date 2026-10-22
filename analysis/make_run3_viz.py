#!/usr/bin/env python3
"""H100 Run #3 results visualization — dark theme, 4-panel 2×2."""

import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# ── Parse training log ────────────────────────────────────────────────────────
log_path = './h100_deploy/run3_output.log'
steps, losses = [], []
with open(log_path) as f:
    for line in f:
        m = re.match(r'step:(\d+)/20000 tnl:([\d.]+)', line)
        if m:
            steps.append(int(m.group(1)))
            losses.append(float(m.group(2)))

steps  = np.array(steps)
losses = np.array(losses)

# ── Dark style ────────────────────────────────────────────────────────────────
BG       = '#0d1117'
PANEL_BG = '#161b22'
GRID_C   = '#21262d'
TEXT     = '#e6edf3'
MUTED    = '#8b949e'
BLUE     = '#58a6ff'
CYAN     = '#39d0d8'
GREEN    = '#3fb950'
RED      = '#f85149'
ORANGE   = '#d29922'
PURPLE   = '#bc8cff'
YELLOW   = '#e3b341'

plt.rcParams.update({
    'figure.facecolor':  BG,
    'axes.facecolor':    PANEL_BG,
    'axes.edgecolor':    GRID_C,
    'axes.labelcolor':   TEXT,
    'axes.titlecolor':   TEXT,
    'xtick.color':       MUTED,
    'ytick.color':       MUTED,
    'text.color':        TEXT,
    'grid.color':        GRID_C,
    'grid.linewidth':    0.6,
    'legend.facecolor':  PANEL_BG,
    'legend.edgecolor':  GRID_C,
    'legend.labelcolor': TEXT,
    'font.family':       'monospace',
    'font.size':         9,
    'axes.titlesize':    11,
    'axes.labelsize':    9,
})

fig = plt.figure(figsize=(16, 12), dpi=100)
fig.patch.set_facecolor(BG)

# Suptitle
fig.text(0.5, 0.97, 'H100 Run #3: int5 Attention + ROPE_BASE Fix',
         ha='center', va='top', fontsize=16, color=TEXT, fontweight='bold',
         fontfamily='monospace')
fig.text(0.5, 0.945, '0% pruning, 1.1131 sliding BPB',
         ha='center', va='top', fontsize=11, color=MUTED, fontfamily='monospace')

gs = GridSpec(2, 2, figure=fig, top=0.92, bottom=0.07,
              left=0.07, right=0.97, hspace=0.45, wspace=0.35)

# ─── Panel 1: Training Loss Curve ────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])

def smooth(y, w=20):
    if len(y) < w:
        return y
    kernel = np.ones(w) / w
    pad = np.pad(y, (w//2, w//2), mode='edge')
    return np.convolve(pad, kernel, mode='valid')[:len(y)]

raw_color  = '#1f6feb'
smth_color = CYAN

# Y limits first so annotations land correctly
YMIN, YMAX = 2.2, 5.0
ax1.set_ylim(YMIN, YMAX)
ax1.set_xlim(0, 5400)

mask = steps >= 1
ax1.plot(steps[mask], losses[mask], color=raw_color, lw=0.7, alpha=0.45, label='Raw loss')
ax1.plot(steps[mask], smooth(losses[mask], 15), color=smth_color, lw=1.8, label='Smoothed (w=15)')

# Recurrence marker
ax1.axvline(x=2000, color=ORANGE, lw=1.2, ls='--', alpha=0.85)
ax1.text(2060, YMAX - 0.12, 'recurrence\nactivated',
         color=ORANGE, fontsize=7.5, va='top', ha='left')

# Wallclock cap marker
ax1.axvline(x=5209, color=RED, lw=1.2, ls='--', alpha=0.85)
ax1.text(5140, 2.65, 'wallclock\ncap',
         color=RED, fontsize=7.5, ha='right', va='bottom')

# Val BPB annotations
ax1.annotate('val BPB 1.1100\n(post-EMA)',
             xy=(5209, 2.5567), xytext=(4200, 2.32),
             arrowprops=dict(arrowstyle='->', color=GREEN, lw=1.2),
             color=GREEN, fontsize=7.5, ha='center')
ax1.annotate('val BPB 1.1550\n(step 4000)',
             xy=(4000, 2.6576), xytext=(2800, 2.37),
             arrowprops=dict(arrowstyle='->', color=YELLOW, lw=1.2),
             color=YELLOW, fontsize=7.5, ha='center')

ax1.set_xlabel('Training Step')
ax1.set_ylabel('Training Loss')
ax1.set_title('Panel 1 — Training Loss Curve (Run #3)', pad=8)
ax1.legend(loc='upper right', fontsize=7.5)
ax1.grid(True, alpha=0.4)

# ─── Panel 2: Run #2 vs Run #3 BPB Comparison ────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])

metrics  = ['Post-EMA BPB', 'Quant BPB', 'Sliding BPB']
run2_bpb = [1.1113, 1.1634, 1.1443]
run3_bpb = [1.1100, 1.1316, 1.1131]

x      = np.arange(len(metrics))
width  = 0.35
r2_col = '#cf6679'
r3_col = '#58a6ff'

bars2 = ax2.bar(x - width/2, run2_bpb, width, label='Run #2 (int6 attn, 40.6% pruned)',
                color=r2_col, edgecolor=GRID_C, linewidth=0.6)
bars3 = ax2.bar(x + width/2, run3_bpb, width, label='Run #3 (int5 attn, 0% pruned)',
                color=r3_col, edgecolor=GRID_C, linewidth=0.6)

for bar in bars2:
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0008,
             f'{bar.get_height():.4f}', ha='center', va='bottom', fontsize=7.5, color=r2_col)
for bar in bars3:
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0008,
             f'{bar.get_height():.4f}', ha='center', va='bottom', fontsize=7.5, color=BLUE)

# Delta annotations — placed clearly above both bars
for i, (r2, r3) in enumerate(zip(run2_bpb, run3_bpb)):
    delta = r3 - r2
    ax2.text(x[i], max(r2, r3) + 0.009,
             f'Δ{delta:+.4f}', ha='center', fontsize=8,
             color=GREEN if delta < 0 else RED, fontweight='bold')

ax2.set_xticks(x)
ax2.set_xticklabels(metrics, fontsize=8.5)
ax2.set_ylabel('BPB (lower is better)')
ax2.set_title('Panel 2 — Run #2 vs Run #3 BPB', pad=8)
ax2.legend(fontsize=7.5, loc='upper left')
ax2.grid(True, axis='y', alpha=0.4)
ax2.set_ylim(1.08, 1.195)
ax2.yaxis.set_major_formatter(plt.FormatStrFormatter('%.4f'))

# ─── Panel 3: Budget Utilization ─────────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 0])

total_mb = 16.0

# Run #2 segments (stacked bottom-up)
r2_code_mb   = 60006  / 1e6   # 0.060 MB
r2_model_fit = 15.20          # squeezed model after pruning
r2_model_raw = 16.21          # what model actually was before pruning
r2_prune_dmg = r2_model_raw - r2_model_fit  # 1.01 MB pruned away

# Run #3: 15232121 bytes total brotli
r3_total_brotli_mb = 15232121 / 1e6   # 15.232 MB
r3_model_mb        = 14.47
r3_code_mb         = 60006 / 1e6
r3_headroom        = total_mb - r3_total_brotli_mb  # ~0.768 MB

# Run #2 brotli total ~15.23 MB (same code, squeezed model)
r2_total_brotli_mb = 15.23
r2_headroom        = total_mb - r2_total_brotli_mb   # ~0.77 MB

bar_w = 0.52

seg_colors = {
    'Code':     PURPLE,
    'Model':    BLUE,
    'Pruning':  RED,
    'Headroom': GREEN,
}

def draw_stacked(ax, x, segments, colors, width, min_label_mb=0.12):
    """segments: list of (label, value_mb) in bottom-up order."""
    bottom = 0.0
    for label, val in segments:
        if val <= 0:
            continue
        ax.bar(x, val, width, bottom=bottom,
               color=colors[label], edgecolor=BG, linewidth=0.9)
        mid = bottom + val / 2
        if val >= min_label_mb:
            ax.text(x, mid, f'{val:.2f}MB', ha='center', va='center',
                    fontsize=8, color='white', fontweight='bold')
        bottom += val

r2_segments = [
    ('Code',     r2_code_mb),
    ('Model',    r2_model_fit),
    ('Pruning',  r2_prune_dmg),
    ('Headroom', r2_headroom),
]
r3_segments = [
    ('Code',     r3_code_mb),
    ('Model',    r3_model_mb),
    ('Pruning',  0.0),
    ('Headroom', r3_headroom),
]

draw_stacked(ax3, 0.5, r2_segments, seg_colors, bar_w)
draw_stacked(ax3, 1.5, r3_segments, seg_colors, bar_w)

ax3.axhline(y=total_mb, color=ORANGE, lw=1.5, ls='--', alpha=0.85)

ax3.set_xlim(0.1, 1.9)
ax3.set_ylim(0, 17.8)
ax3.set_xticks([0.5, 1.5])
ax3.set_xticklabels(['Run #2\n(40.6% pruned)', 'Run #3\n(0% pruned)'], fontsize=9.5)
ax3.set_ylabel('Size (MB)')
ax3.set_title('Panel 3 — Budget Utilization', pad=8)
ax3.grid(True, axis='y', alpha=0.3)

legend_patches = [
    mpatches.Patch(color=PURPLE, label='Code (~60KB)'),
    mpatches.Patch(color=BLUE,   label='Model weights'),
    mpatches.Patch(color=RED,    label='Pruning damage (Run #2 only)'),
    mpatches.Patch(color=GREEN,  label='Headroom'),
]
ax3.legend(handles=legend_patches, fontsize=7.5, loc='lower right', framealpha=0.85)
ax3.text(0.5, total_mb + 0.2, '16MB limit', ha='center', fontsize=8, color=ORANGE)

# Pruning annotation on Run #2
r2_prune_y = r2_code_mb + r2_model_fit + r2_prune_dmg / 2
ax3.annotate('Pruning\ncatastrophe\n+0.031 BPB', xy=(0.5, r2_prune_y),
             xytext=(0.15, 16.8),
             arrowprops=dict(arrowstyle='->', color=RED, lw=1.2,
                             connectionstyle='arc3,rad=0.1'),
             color=RED, fontsize=7.5, ha='left', va='top')

# Headroom annotation on Run #3
r3_headroom_y = r3_code_mb + r3_model_mb + r3_headroom / 2
ax3.annotate(f'{r3_headroom:.2f}MB\nheadroom', xy=(1.5, r3_headroom_y),
             xytext=(1.65, 16.2),
             arrowprops=dict(arrowstyle='->', color=GREEN, lw=1.2,
                             connectionstyle='arc3,rad=-0.15'),
             color=GREEN, fontsize=8, ha='left', va='top')

# ─── Panel 4: Leaderboard Position ───────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 1])

lb_names  = ['#1 aryanbhosale', '#2 dexhunter', 'Us — Run #3', 'Us — Run #2']
lb_scores = [1.0896,             1.0900,          1.1131,         1.1443]
lb_colors = [YELLOW,             ORANGE,           r3_col,         r2_col]

# Worst at bottom (y=0), best at top
order    = np.argsort(lb_scores)[::-1]   # descending
names_s  = [lb_names[i]  for i in order]
scores_s = [lb_scores[i] for i in order]
colors_s = [lb_colors[i] for i in order]

y_pos = np.arange(len(names_s))
ax4.barh(y_pos, scores_s, color=colors_s, edgecolor=GRID_C, linewidth=0.6, height=0.55)

for yi, score in zip(y_pos, scores_s):
    ax4.text(score + 0.0006, yi, f'{score:.4f}',
             va='center', fontsize=9.5, color=TEXT, fontweight='bold')

ax4.set_yticks(y_pos)
ax4.set_yticklabels(names_s, fontsize=9.5)
ax4.set_xlabel('BPB (lower = better, bars extend right)')
ax4.set_title('Panel 4 — Leaderboard Position', pad=8)
ax4.grid(True, axis='x', alpha=0.4)
ax4.set_xlim(1.065, 1.165)

# Gap bracket between Run #3 and #1
idx_r3   = names_s.index('Us — Run #3')
idx_best = names_s.index('#1 aryanbhosale')
gap = scores_s[idx_r3] - scores_s[idx_best]

gap_y = idx_best + 0.05
ax4.annotate('', xy=(scores_s[idx_best], gap_y),
             xytext=(scores_s[idx_r3], gap_y),
             arrowprops=dict(arrowstyle='<->', color=CYAN, lw=1.5, mutation_scale=12))
ax4.text((scores_s[idx_best] + scores_s[idx_r3]) / 2, gap_y + 0.28,
         f'gap = {gap:.4f} BPB to close', ha='center', fontsize=8.5, color=CYAN)

# ─── Final save ──────────────────────────────────────────────────────────────
out = './h100_run3_results.png'
fig.savefig(out, dpi=200, bbox_inches='tight', facecolor=BG)
print(f'Saved: {out}')
plt.close(fig)
