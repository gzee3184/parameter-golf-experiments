#!/usr/bin/env python3
"""Training Log Visualizer for Parameter Golf experiments.

Parses training logs, extracts metrics, annotates events, and generates
a comprehensive multi-panel visualization + summary report.

Usage:
    python visualize_training.py                          # All logs
    python visualize_training.py log1.log log2.log        # Specific logs
    python visualize_training.py --dir sweeps/local/v8_sweep_r3  # Directory
    python visualize_training.py --latest 6               # Most recent 6 logs
    python visualize_training.py --output my_plot.png     # Custom output path
"""

import argparse
import glob
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
import numpy as np

# ─── Data structures ───────────────────────────────────────────────────────

@dataclass
class TrainStep:
    step: int
    train_loss: float
    train_time_ms: float
    step_avg_ms: float

@dataclass
class ValStep:
    step: int
    val_loss: float
    val_bpb: float
    train_time_ms: float
    step_avg_ms: float

@dataclass
class Event:
    step: int
    time_ms: float
    event_type: str  # 'recurrence', 'warmdown', 'swa', 'val_eval'
    description: str

@dataclass
class FinalResult:
    post_ema_bpb: Optional[float] = None
    quant_bpb: Optional[float] = None
    sliding_bpb: Optional[float] = None
    prune_pct: Optional[float] = None
    prune_count: Optional[int] = None
    prune_total: Optional[int] = None
    model_params: Optional[int] = None
    total_steps: Optional[int] = None
    code_size: Optional[int] = None

@dataclass
class RunLog:
    filepath: str
    name: str
    train_steps: list = field(default_factory=list)
    val_steps: list = field(default_factory=list)
    events: list = field(default_factory=list)
    result: FinalResult = field(default_factory=FinalResult)
    config: dict = field(default_factory=dict)


# ─── Parser ────────────────────────────────────────────────────────────────

def _make_run_name(filepath: str) -> str:
    """Generate a meaningful display name from filepath."""
    name = Path(filepath).stem
    name = re.sub(r'_\d{8}_\d{6}$', '', name)  # remove timestamp suffix
    parent = Path(filepath).parent.name

    # If name is generic ("run"), prefix with parent dir
    if name in ('run', 'output', 'log', 'train'):
        name = parent
    elif parent.startswith('v8_sweep_r') or parent.startswith('v8.5'):
        round_label = parent.replace('v8_sweep_', '').upper().replace('V8.5_SWEEP', 'v8.5')
        name = f'{round_label}:{name}'
    elif parent.startswith('v7_sweep'):
        name = f'v7:{name}'

    # Shorten common prefixes for legend readability
    name = name.replace('test_a_', '').replace('test_b_', 'B:')
    name = name.replace('_all_improvements', '_improv')

    return name


def parse_log(filepath: str) -> RunLog:
    """Parse a single training log file into structured data."""
    name = _make_run_name(filepath)
    run = RunLog(filepath=filepath, name=name)

    with open(filepath, 'r') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()

        # ── Config lines ──
        if line.startswith('model_params:'):
            m = re.match(r'model_params:(\d+)', line)
            if m:
                run.result.model_params = int(m.group(1))

        elif line.startswith('arch_config:'):
            for kv in re.findall(r'(\w+)=([^\s,\]]+|\[[^\]]*\])', line):
                run.config[kv[0]] = kv[1]

        elif line.startswith('tie_embeddings:'):
            for kv in re.findall(r'(\w+):([^\s]+)', line):
                run.config[kv[0]] = kv[1]

        elif line.startswith('Code size:'):
            m = re.match(r'Code size:\s*(\d+)', line)
            if m:
                run.result.code_size = int(m.group(1))

        # ── PR#1326 style config: "  key: value" (YAML-like) ──
        elif line.startswith('  ') and ': ' in line and not line.startswith('  #'):
            kv = line.strip().split(': ', 1)
            if len(kv) == 2 and kv[0].replace('_','').isalpha():
                run.config[kv[0]] = kv[1]

        # ── Our format: Training step with val ──
        elif line.startswith('step:') and 'val_loss:' in line:
            m = re.match(
                r'step:(\d+)/\d+\s+val_loss:([\d.]+)\s+val_bpb:([\d.]+)\s+'
                r'train_time:(\d+)ms\s+step_avg:([\d.]+)ms', line)
            if m:
                vs = ValStep(
                    step=int(m.group(1)),
                    val_loss=float(m.group(2)),
                    val_bpb=float(m.group(3)),
                    train_time_ms=float(m.group(4)),
                    step_avg_ms=float(m.group(5)),
                )
                run.val_steps.append(vs)
                run.events.append(Event(
                    step=vs.step, time_ms=vs.train_time_ms,
                    event_type='val_eval',
                    description=f'val_bpb={vs.val_bpb:.4f}'))

        # ── Our format: Training step (train only) ──
        elif line.startswith('step:') and 'train_loss:' in line:
            m = re.match(
                r'step:(\d+)/\d+\s+train_loss:([\d.]+)\s+'
                r'train_time:(\d+)ms\s+step_avg:([\d.]+)ms', line)
            if m:
                ts = TrainStep(
                    step=int(m.group(1)),
                    train_loss=float(m.group(2)),
                    train_time_ms=float(m.group(3)),
                    step_avg_ms=float(m.group(4)),
                )
                run.train_steps.append(ts)

        # ── PR#1326 format: "N/20000 val_loss: X val_bpb: Y" ──
        elif re.match(r'^\d+/\d+\s+val_loss:', line):
            m = re.match(r'^(\d+)/\d+\s+val_loss:\s*([\d.]+)\s+val_bpb:\s*([\d.]+)', line)
            if m:
                vs = ValStep(
                    step=int(m.group(1)),
                    val_loss=float(m.group(2)),
                    val_bpb=float(m.group(3)),
                    train_time_ms=0, step_avg_ms=0,
                )
                run.val_steps.append(vs)
                run.events.append(Event(
                    step=vs.step, time_ms=0,
                    event_type='val_eval',
                    description=f'val_bpb={vs.val_bpb:.4f}'))

        # ── PR#1326 format: "N/20000 train_loss: X train_time: Y.Ym tok/s: Z" ──
        elif re.match(r'^\d+/\d+\s+train_loss:', line):
            m = re.match(r'^(\d+)/\d+\s+train_loss:\s*([\d.]+)\s+train_time:\s*([\d.]+)m\s+tok/s:\s*(\d+)', line)
            if m:
                step = int(m.group(1))
                time_min = float(m.group(3))
                time_ms = time_min * 60000
                tok_s = int(m.group(4))
                # Estimate ms/step from tok/s (batch=786432 tokens, but might differ)
                avg_ms = time_ms / max(step, 1)
                ts = TrainStep(
                    step=step,
                    train_loss=float(m.group(2)),
                    train_time_ms=time_ms,
                    step_avg_ms=avg_ms,
                )
                run.train_steps.append(ts)

        # ── Recurrence activation ──
        elif line.startswith('recurrence:activated'):
            m = re.search(r'at step (\d+)', line)
            if m:
                step = int(m.group(1))
                # Find closest train_time
                time_ms = _interpolate_time(run.train_steps, step)
                run.events.append(Event(
                    step=step, time_ms=time_ms,
                    event_type='recurrence',
                    description=f'Recurrence ON (step {step})'))

        # ── DIAGNOSTIC (post-EMA) — our format ──
        elif line.startswith('DIAGNOSTIC post_ema'):
            m = re.search(r'val_bpb:([\d.]+)', line)
            if m:
                run.result.post_ema_bpb = float(m.group(1))

        # ── PR#1326 format: "pre-quantization post-ema val_loss:X val_bpb:Y" ──
        elif line.startswith('pre-quantization post-ema'):
            m = re.search(r'val_bpb:([\d.]+)', line)
            if m:
                run.result.post_ema_bpb = float(m.group(1))

        # ── Pruning ──
        elif 'selective_prune: pruning' in line:
            # Our format: "(14.8%)"
            m = re.search(r'pruning (\d+)/(\d+).*\(([\d.]+)%\)', line)
            if m:
                run.result.prune_count = int(m.group(1))
                run.result.prune_total = int(m.group(2))
                run.result.prune_pct = float(m.group(3))
            else:
                # PR#1326 format: "pruning N/M lowest-error ±1 values (excess=XB)"
                m = re.search(r'pruning (\d+)/(\d+)', line)
                if m:
                    run.result.prune_count = int(m.group(1))
                    run.result.prune_total = int(m.group(2))
                    run.result.prune_pct = run.result.prune_count / run.result.prune_total * 100
        elif 'selective_prune:' in line and 'prune=' in line and run.result.prune_pct is None:
            pass

        # ── Final quant result ──
        elif line.startswith('final_int6_roundtrip_exact') or \
             (line.startswith('final_int6_roundtrip') and 'exact' not in line):
            m = re.search(r'val_bpb:([\d.]+)', line)
            if m:
                run.result.quant_bpb = float(m.group(1))

        # ── Sliding window ──
        elif 'sliding_window' in line or 'sliding' in line.lower():
            m = re.search(r'val_bpb:([\d.]+)', line)
            if m:
                run.result.sliding_bpb = float(m.group(1))

    # Set total steps from last train step
    if run.train_steps:
        run.result.total_steps = run.train_steps[-1].step
    elif run.val_steps:
        run.result.total_steps = run.val_steps[-1].step

    # Set prune_pct to 0 if we have quant but no pruning line
    if run.result.quant_bpb is not None and run.result.prune_pct is None:
        run.result.prune_pct = 0.0

    return run


def _interpolate_time(train_steps: list, step: int) -> float:
    """Estimate wall time for a given step from training data."""
    if not train_steps:
        return 0.0
    # Find bracketing steps
    for i, ts in enumerate(train_steps):
        if ts.step >= step:
            if i == 0:
                return ts.train_time_ms
            prev = train_steps[i - 1]
            frac = (step - prev.step) / max(ts.step - prev.step, 1)
            return prev.train_time_ms + frac * (ts.train_time_ms - prev.train_time_ms)
    # Extrapolate from last
    last = train_steps[-1]
    return last.train_time_ms


# ─── Visualization ─────────────────────────────────────────────────────────

# Distinct colors for runs
COLORS = [
    '#2196F3', '#FF5722', '#4CAF50', '#9C27B0', '#FF9800',
    '#00BCD4', '#E91E63', '#8BC34A', '#673AB7', '#FFC107',
    '#009688', '#F44336', '#3F51B5', '#CDDC39', '#795548',
]

EVENT_MARKERS = {
    'recurrence': ('v', 'red', 10),      # down triangle
    'val_eval': ('D', 'gold', 7),         # diamond
    'warmdown': ('s', 'orange', 8),       # square
    'swa': ('^', 'purple', 8),            # up triangle
}


def create_visualization(runs: list, output_path: str, title: str = "Training Analysis"):
    """Generate comprehensive multi-panel visualization."""
    if not runs:
        print("No runs to visualize.")
        return

    # Sort runs by final quant BPB (best first), with incomplete runs last
    def sort_key(r):
        if r.result.quant_bpb is not None:
            return (0, r.result.quant_bpb)
        if r.result.post_ema_bpb is not None:
            return (1, r.result.post_ema_bpb)
        return (2, 999)
    runs.sort(key=sort_key)

    n_runs = len(runs)
    has_final = any(r.result.quant_bpb is not None for r in runs)

    # Layout: 3 rows, 2 cols + summary bar at bottom
    fig = plt.figure(figsize=(22, 26))
    gs = fig.add_gridspec(4, 2, height_ratios=[3, 3, 3, 2.5], hspace=0.35, wspace=0.30,
                          left=0.06, right=0.94, top=0.95, bottom=0.03)

    ax_loss_step = fig.add_subplot(gs[0, 0])
    ax_loss_time = fig.add_subplot(gs[0, 1])
    ax_step_time = fig.add_subplot(gs[1, 0])
    ax_val_bpb = fig.add_subplot(gs[1, 1])
    ax_loss_zoom = fig.add_subplot(gs[2, 0])
    ax_delta = fig.add_subplot(gs[2, 1])
    ax_summary = fig.add_subplot(gs[3, :])

    # ── Compute smart Y limits for loss panels ──
    # Collect all losses after step 200 (skip initial spike)
    all_late_losses = []
    for run in runs:
        for s in run.train_steps:
            if s.step >= 200:
                all_late_losses.append(s.train_loss)
    if all_late_losses:
        loss_y_min = min(all_late_losses) - 0.05
        loss_y_max = max(all_late_losses) + 0.1
    else:
        loss_y_min, loss_y_max = 2.4, 3.5

    # ── Panel 1: Train Loss vs Step (cropped Y) ──
    for i, run in enumerate(runs):
        if not run.train_steps:
            continue
        steps = [s.step for s in run.train_steps]
        losses = [s.train_loss for s in run.train_steps]
        color = COLORS[i % len(COLORS)]
        ax_loss_step.plot(steps, losses, color=color, alpha=0.85, linewidth=1.2, label=run.name)
        for ev in run.events:
            if ev.event_type == 'recurrence':
                ax_loss_step.axvline(ev.step, color=color, linestyle='--', alpha=0.4, linewidth=0.8)

    ax_loss_step.set_ylim(loss_y_min, loss_y_max)
    ax_loss_step.set_xlabel('Step')
    ax_loss_step.set_ylabel('Train Loss')
    ax_loss_step.set_title('Train Loss vs Step (Y cropped to convergence region)')
    ax_loss_step.grid(True, alpha=0.3)
    ax_loss_step.legend(fontsize=7, loc='upper right', ncol=max(1, n_runs // 5))

    # ── Panel 2: Final BPB Comparison (bar chart) ──
    # Much more readable than a second loss curve for comparing results
    bar_runs = [r for r in runs if r.result.post_ema_bpb is not None]
    if bar_runs:
        bar_names = [r.name for r in bar_runs]
        bar_ema = [r.result.post_ema_bpb for r in bar_runs]
        bar_quant = [r.result.quant_bpb if r.result.quant_bpb else 0 for r in bar_runs]
        bar_sliding = [r.result.sliding_bpb if r.result.sliding_bpb else 0 for r in bar_runs]

        x = np.arange(len(bar_names))
        width = 0.28
        bars_ema = ax_loss_time.barh(x + width, bar_ema, width, label='Post-EMA', color='#2196F3', alpha=0.8)
        bars_quant = ax_loss_time.barh(x, bar_quant, width, label='Quant', color='#FF5722', alpha=0.8)
        if any(bar_sliding):
            bars_slide = ax_loss_time.barh(x - width, bar_sliding, width, label='Sliding', color='#4CAF50', alpha=0.8)

        # Annotate with values
        for i_bar, (ema, quant) in enumerate(zip(bar_ema, bar_quant)):
            ax_loss_time.text(ema + 0.0003, i_bar + width, f'{ema:.4f}', va='center', fontsize=6, fontweight='bold')
            if quant > 0:
                ax_loss_time.text(quant + 0.0003, i_bar, f'{quant:.4f}', va='center', fontsize=6)

        # Zoom x-axis to meaningful range
        all_bpb_vals = [v for v in bar_ema + bar_quant + bar_sliding if v > 0]
        bpb_min = min(all_bpb_vals) - 0.005
        bpb_max = max(all_bpb_vals) + 0.008
        ax_loss_time.set_xlim(bpb_min, bpb_max)
        ax_loss_time.set_yticks(x)
        ax_loss_time.set_yticklabels(bar_names, fontsize=7)
        ax_loss_time.set_xlabel('BPB (lower is better)')
        ax_loss_time.set_title('Final BPB Comparison')
        ax_loss_time.legend(fontsize=8, loc='upper right')
        ax_loss_time.grid(True, alpha=0.3, axis='x')
        ax_loss_time.invert_yaxis()

    # ── Panel 3: Step Time (ms/step) vs Step ──
    # Filter out early steps where avg is unstable
    for i, run in enumerate(runs):
        if not run.train_steps:
            continue
        filtered = [(s.step, s.step_avg_ms) for s in run.train_steps if s.step >= 100 and s.step_avg_ms > 0]
        if not filtered:
            continue
        steps, avgs = zip(*filtered)
        color = COLORS[i % len(COLORS)]
        ax_step_time.plot(steps, avgs, color=color, alpha=0.85, linewidth=1.2, label=run.name)
        for ev in run.events:
            if ev.event_type == 'recurrence':
                ax_step_time.axvline(ev.step, color=color, linestyle='--', alpha=0.4, linewidth=0.8)

    # Tighten y-axis around actual range
    all_step_times = [s.step_avg_ms for r in runs for s in r.train_steps if s.step >= 100 and s.step_avg_ms > 0]
    if all_step_times:
        st_min = min(all_step_times) * 0.95
        st_max = max(all_step_times) * 1.05
        ax_step_time.set_ylim(st_min, st_max)
    ax_step_time.set_xlabel('Step')
    ax_step_time.set_ylabel('Avg ms/step')
    ax_step_time.set_title('Step Time vs Step (shows recurrence overhead)')
    ax_step_time.grid(True, alpha=0.3)
    ax_step_time.legend(fontsize=7, loc='lower right')

    # ── Panel 4: Val BPB Zoomed (skip initial eval, tight Y) ──
    all_final_bpbs = []
    for run in runs:
        # Skip initial eval (step 0), only show mid-training + final
        late_vals = [v for v in run.val_steps if v.step > 0]
        if late_vals:
            for v in late_vals:
                all_final_bpbs.append(v.val_bpb)
        if run.result.post_ema_bpb:
            all_final_bpbs.append(run.result.post_ema_bpb)

    for i, run in enumerate(runs):
        color = COLORS[i % len(COLORS)]
        late_vals = [v for v in run.val_steps if v.step > 0]
        if late_vals:
            vsteps = [v.step for v in late_vals]
            vbpbs = [v.val_bpb for v in late_vals]
            ax_val_bpb.plot(vsteps, vbpbs, 'o-', color=color, markersize=6, linewidth=1.5, label=run.name)
            for v in late_vals:
                ax_val_bpb.annotate(f'{v.val_bpb:.4f}', xy=(v.step, v.val_bpb),
                                   fontsize=6, color=color, textcoords='offset points',
                                   xytext=(5, 5))
        if run.result.post_ema_bpb is not None and run.result.total_steps:
            ax_val_bpb.plot(run.result.total_steps, run.result.post_ema_bpb,
                           '*', color=color, markersize=14, zorder=5)
            ax_val_bpb.annotate(f'EMA {run.result.post_ema_bpb:.4f}',
                               xy=(run.result.total_steps, run.result.post_ema_bpb),
                               fontsize=6, color=color, fontweight='bold',
                               textcoords='offset points', xytext=(5, -10))

    # Tight Y around the actual BPB values (not the initial 3.6)
    if all_final_bpbs:
        vbpb_min = min(all_final_bpbs) - 0.005
        vbpb_max = max(all_final_bpbs) + 0.005
        ax_val_bpb.set_ylim(vbpb_min, vbpb_max)
    ax_val_bpb.set_xlabel('Step')
    ax_val_bpb.set_ylabel('Val BPB')
    ax_val_bpb.set_title('Validation BPB (zoomed, stars = post-EMA)')
    ax_val_bpb.grid(True, alpha=0.3)
    ax_val_bpb.legend(fontsize=6, loc='upper right', ncol=max(1, n_runs // 6))

    # ── Panel 5: Loss Zoom (last 30% of steps, tight Y) ──
    max_step = max((run.train_steps[-1].step for run in runs if run.train_steps), default=5000)
    zoom_start = int(max_step * 0.7)
    zoom_losses = []
    for i, run in enumerate(runs):
        if not run.train_steps:
            continue
        zoomed = [(s.step, s.train_loss) for s in run.train_steps if s.step >= zoom_start]
        if zoomed:
            steps, losses = zip(*zoomed)
            color = COLORS[i % len(COLORS)]
            ax_loss_zoom.plot(steps, losses, color=color, alpha=0.85, linewidth=1.2, label=run.name)
            zoom_losses.extend(losses)

    if zoom_losses:
        zm_min = min(zoom_losses) - 0.02
        zm_max = max(zoom_losses) + 0.02
        ax_loss_zoom.set_ylim(zm_min, zm_max)
    ax_loss_zoom.set_xlabel('Step')
    ax_loss_zoom.set_ylabel('Train Loss')
    ax_loss_zoom.set_title(f'Train Loss Zoom (step {zoom_start}+, warmdown region)')
    ax_loss_zoom.grid(True, alpha=0.3)
    ax_loss_zoom.legend(fontsize=7, loc='upper right')

    # ── Panel 6: Delta from best (step-aligned, skip initial spike) ──
    if len(runs) >= 2:
        ref = runs[0]
        if ref.train_steps:
            ref_dict = {s.step: s.train_loss for s in ref.train_steps}
            delta_skip = 500  # skip noisy early steps
            all_deltas = []
            for i, run in enumerate(runs[1:], 1):
                if not run.train_steps:
                    continue
                color = COLORS[i % len(COLORS)]
                deltas = []
                d_steps = []
                for s in run.train_steps:
                    if s.step in ref_dict and s.step >= delta_skip:
                        d = s.train_loss - ref_dict[s.step]
                        deltas.append(d)
                        d_steps.append(s.step)
                        all_deltas.append(d)
                if deltas:
                    ax_delta.plot(d_steps, deltas, color=color, alpha=0.85, linewidth=1.2, label=f'{run.name}')
            ax_delta.axhline(0, color='black', linewidth=1.0, linestyle='-')
            # Tight Y around actual delta range
            if all_deltas:
                d_min = min(all_deltas)
                d_max = max(all_deltas)
                d_pad = max((d_max - d_min) * 0.15, 0.005)
                ax_delta.set_ylim(d_min - d_pad, d_max + d_pad)
            ax_delta.set_xlabel('Step')
            ax_delta.set_ylabel(f'Δ Train Loss (vs {ref.name})')
            ax_delta.set_title(f'Train Loss Delta vs Best (step {delta_skip}+)')
            ax_delta.grid(True, alpha=0.3)
            ax_delta.legend(fontsize=6, loc='upper right', ncol=max(1, (n_runs - 1) // 6))
    else:
        ax_delta.text(0.5, 0.5, 'Need ≥2 runs for delta plot', ha='center', va='center',
                     transform=ax_delta.transAxes, fontsize=12, color='gray')
        ax_delta.set_title('Train Loss Delta (need ≥2 runs)')

    # ── Summary Table ──
    ax_summary.axis('off')
    headers = ['Run', 'Steps', 'Params', 'Post-EMA', 'Quant BPB', 'Δ Quant', 'Prune%', 'Sliding', 'ms/step', 'Code KB']
    cell_data = []
    best_quant = min((r.result.quant_bpb for r in runs if r.result.quant_bpb), default=None)

    for run in runs:
        r = run.result
        steps = str(r.total_steps) if r.total_steps else '—'
        params = f'{r.model_params/1e6:.1f}M' if r.model_params else '—'
        ema = f'{r.post_ema_bpb:.4f}' if r.post_ema_bpb else '—'
        quant = f'{r.quant_bpb:.4f}' if r.quant_bpb else '—'
        delta_q = ''
        if r.quant_bpb and best_quant:
            d = r.quant_bpb - best_quant
            delta_q = f'{d:+.4f}' if abs(d) > 0.00005 else 'BEST'
        prune = f'{r.prune_pct:.1f}%' if r.prune_pct is not None else '—'
        sliding = f'{r.sliding_bpb:.4f}' if r.sliding_bpb else '—'
        avg_ms = f'{run.train_steps[-1].step_avg_ms:.0f}' if run.train_steps else '—'
        code = f'{r.code_size/1024:.1f}' if r.code_size else '—'
        cell_data.append([run.name, steps, params, ema, quant, delta_q, prune, sliding, avg_ms, code])

    if cell_data:
        table = ax_summary.table(
            cellText=cell_data, colLabels=headers,
            loc='center', cellLoc='center',
            colColours=['#E3F2FD'] * len(headers),
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.5)

        # Highlight best row
        for j in range(len(headers)):
            table[1, j].set_facecolor('#C8E6C9')  # light green for best

        # Color code quant BPB cells
        for row_idx, run in enumerate(runs):
            if run.result.quant_bpb and best_quant:
                d = run.result.quant_bpb - best_quant
                if d <= 0.0001:
                    table[row_idx + 1, 4].set_facecolor('#C8E6C9')  # green
                elif d < 0.005:
                    table[row_idx + 1, 4].set_facecolor('#FFF9C4')  # yellow
                else:
                    table[row_idx + 1, 4].set_facecolor('#FFCDD2')  # red

        # Color prune cells
        for row_idx, run in enumerate(runs):
            if run.result.prune_pct is not None:
                if run.result.prune_pct == 0:
                    table[row_idx + 1, 6].set_facecolor('#C8E6C9')
                elif run.result.prune_pct < 5:
                    table[row_idx + 1, 6].set_facecolor('#FFF9C4')
                else:
                    table[row_idx + 1, 6].set_facecolor('#FFCDD2')

    ax_summary.set_title('Results Summary (sorted by quant BPB, best first)', fontsize=12, fontweight='bold', pad=20)

    # ── Title + metadata ──
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)

    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved visualization to: {output_path}")
    plt.close()


def print_chronological_report(runs: list):
    """Print a text-based chronological breakdown of each run."""
    for run in runs:
        print(f"\n{'='*70}")
        print(f"  {run.name}")
        print(f"  File: {run.filepath}")
        print(f"{'='*70}")

        r = run.result
        if r.model_params:
            print(f"  Model: {r.model_params:,} params")
        if run.config:
            key_config = {k: v for k, v in run.config.items()
                         if k in ('recur_layers', 'parallel_start_layer', 'sigmoid_skip',
                                  've_enabled', 'qk_gain_init', 'muon_wd', 'matrix_lr')}
            if key_config:
                print(f"  Config: {key_config}")

        # Chronological timeline
        print(f"\n  Timeline:")
        print(f"  {'─'*64}")

        # Merge train steps and events, sorted by step
        timeline = []
        for ts in run.train_steps:
            timeline.append(('train', ts.step, ts.train_time_ms, ts))
        for ev in run.events:
            if ev.event_type != 'val_eval':  # val_eval is noisy
                timeline.append(('event', ev.step, ev.time_ms, ev))
        for vs in run.val_steps:
            timeline.append(('val', vs.step, vs.train_time_ms, vs))
        timeline.sort(key=lambda x: (x[1], 0 if x[0] == 'event' else 1))

        # Phase detection
        phases = []
        recur_step = None
        for ev in run.events:
            if ev.event_type == 'recurrence':
                recur_step = ev.step

        # Print key moments
        prev_loss = None
        for kind, step, time_ms, obj in timeline:
            time_min = time_ms / 60000
            if kind == 'event':
                print(f"  ▸ {time_min:7.1f}min  step {step:5d}  *** {obj.description} ***")
            elif kind == 'val':
                marker = ''
                if run.val_steps.index(obj) == 0:
                    marker = ' (initial)'
                print(f"  ◆ {time_min:7.1f}min  step {step:5d}  val_bpb={obj.val_bpb:.4f}{marker}")
            elif kind == 'train':
                # Only print select training steps for readability
                show = (step <= 10 or step % 500 == 0 or step == run.train_steps[-1].step)
                if not show:
                    continue

                # Phase label
                phase = 'warmup' if step < 20 else 'pre-recur' if (recur_step and step < recur_step) else 'post-recur' if recur_step else 'training'
                delta = ''
                if prev_loss is not None and step > 10:
                    d = obj.train_loss - prev_loss
                    delta = f'  Δ={d:+.4f}'
                prev_loss = obj.train_loss

                # Step time change detection
                speed_note = ''
                if len(run.train_steps) > 1:
                    idx = run.train_steps.index(obj)
                    if idx > 0:
                        prev_avg = run.train_steps[idx-1].step_avg_ms
                        pct_change = (obj.step_avg_ms - prev_avg) / prev_avg * 100 if prev_avg > 0 else 0
                        if abs(pct_change) > 2:
                            speed_note = f'  [{pct_change:+.1f}% speed]'

                print(f"    {time_min:7.1f}min  step {step:5d}  loss={obj.train_loss:.4f}  "
                      f"avg={obj.step_avg_ms:.0f}ms  [{phase}]{delta}{speed_note}")

        # Final results
        print(f"\n  Results:")
        print(f"  {'─'*64}")
        if r.total_steps:
            print(f"    Total steps:  {r.total_steps}")
        if r.post_ema_bpb:
            print(f"    Post-EMA BPB: {r.post_ema_bpb:.4f}")
        if r.quant_bpb:
            gap = r.quant_bpb - r.post_ema_bpb if r.post_ema_bpb else 0
            print(f"    Quant BPB:    {r.quant_bpb:.4f}  (quant gap: +{gap:.4f})")
        if r.prune_pct is not None:
            print(f"    Pruning:      {r.prune_pct:.1f}%", end='')
            if r.prune_count:
                print(f'  ({r.prune_count:,}/{r.prune_total:,} values)')
            else:
                print()
        if r.sliding_bpb:
            print(f"    Sliding BPB:  {r.sliding_bpb:.4f}")
        if run.train_steps:
            # Compute pre/post recurrence speed
            pre_recur = [s for s in run.train_steps if recur_step and s.step < recur_step and s.step > 100]
            post_recur = [s for s in run.train_steps if recur_step and s.step > recur_step + 200]
            if pre_recur and post_recur:
                pre_avg = pre_recur[-1].step_avg_ms
                post_avg = post_recur[-1].step_avg_ms
                overhead = (post_avg - pre_avg) / pre_avg * 100
                print(f"    Speed:        {pre_avg:.0f}ms pre-recur → {post_avg:.0f}ms post-recur (+{overhead:.1f}% overhead)")


# ─── Log discovery ─────────────────────────────────────────────────────────

def find_logs(args) -> list:
    """Find log files based on arguments."""
    base = './sweeps/local'
    logs = []

    if args.files:
        logs = args.files
    elif args.dir:
        dirpath = args.dir if os.path.isabs(args.dir) else os.path.join(base, args.dir)
        logs = sorted(glob.glob(os.path.join(dirpath, '*.log')))
        # Exclude launcher/wrapper logs
        logs = [l for l in logs if not any(x in os.path.basename(l) for x in ['launcher', 'wrapper', 'gpu0_wrapper', 'gpu1_wrapper'])]
    else:
        # Default: find all experiment logs across all subdirectories
        for pattern in ['v8.5_sweep/*.log', 'v8_sweep_r*/*.log', 'v7_sweep/test_*.log',
                        'v6_long/*.log', 'v5_long/*.log', 'v4_long/*.log']:
            found = sorted(glob.glob(os.path.join(base, pattern)))
            found = [l for l in found if not any(x in os.path.basename(l)
                     for x in ['launcher', 'wrapper', 'gpu0_wrapper', 'gpu1_wrapper', 'INVALID'])]
            logs.extend(found)

    if args.latest and logs:
        # Sort by modification time, take most recent N
        logs.sort(key=os.path.getmtime, reverse=True)
        logs = logs[:args.latest]

    return logs


# ─── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Visualize Parameter Golf training logs')
    parser.add_argument('files', nargs='*', help='Specific log files to analyze')
    parser.add_argument('--dir', '-d', help='Directory containing log files')
    parser.add_argument('--latest', '-n', type=int, help='Only show N most recent logs')
    parser.add_argument('--output', '-o', default=None, help='Output image path')
    parser.add_argument('--no-plot', action='store_true', help='Text report only, no plot')
    parser.add_argument('--title', '-t', default=None, help='Plot title')
    args = parser.parse_args()

    logs = find_logs(args)
    if not logs:
        print("No log files found. Usage:")
        print("  python visualize_training.py                    # all logs")
        print("  python visualize_training.py --dir v8_sweep_r3  # specific dir")
        print("  python visualize_training.py --latest 4         # most recent 4")
        sys.exit(1)

    print(f"Found {len(logs)} log files:")
    for l in logs:
        print(f"  {l}")

    # Parse all logs
    runs = []
    for logpath in logs:
        try:
            run = parse_log(logpath)
            if run.train_steps or run.val_steps:  # skip empty logs
                runs.append(run)
            else:
                print(f"  [skip] {logpath} — no training data")
        except Exception as e:
            print(f"  [error] {logpath}: {e}")

    if not runs:
        print("No valid runs found.")
        sys.exit(1)

    print(f"\nParsed {len(runs)} runs successfully.")

    # Text report
    print_chronological_report(runs)

    # Visualization
    if not args.no_plot:
        if args.output:
            output = args.output
        elif args.dir:
            dirname = os.path.basename(args.dir.rstrip('/'))
            output = f'./{dirname}_analysis.png'
        else:
            output = './training_analysis.png'

        title = args.title or f"Parameter Golf Training Analysis ({len(runs)} runs)"
        create_visualization(runs, output, title=title)

    # Quick comparison summary
    print(f"\n{'='*70}")
    print("  QUICK COMPARISON")
    print(f"{'='*70}")
    runs_with_quant = [r for r in runs if r.result.quant_bpb is not None]
    runs_with_quant.sort(key=lambda r: r.result.quant_bpb)
    for i, run in enumerate(runs_with_quant):
        r = run.result
        marker = ' ◀ BEST' if i == 0 else ''
        prune_str = f'prune={r.prune_pct:.1f}%' if r.prune_pct else ''
        print(f"  #{i+1}  {run.name:40s}  EMA={r.post_ema_bpb:.4f}  Quant={r.quant_bpb:.4f}  {prune_str}{marker}")


if __name__ == '__main__':
    main()
