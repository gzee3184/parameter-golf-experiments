#!/usr/bin/env python3
"""
Experiment Log Parser — Parameter Golf Data Framework

Parses training logs into structured data:
  1. experiment_summary.csv — one row per experiment (config + final metrics)
  2. step_curves.csv — per-step training curves (step, loss, bpb, time)
  3. ttt_curves.csv — per-chunk TTT curves (chunk, bpb, time)

Usage:
  python parse_logs.py                    # parse all known log dirs
  python parse_logs.py /path/to/logs/     # parse specific directory
  python parse_logs.py --add new_run.log  # add a single log
"""
import csv
import glob
import os
import re
import sys
from pathlib import Path

OUT_DIR = Path(__file__).parent
SUMMARY_CSV = OUT_DIR / "experiment_summary.csv"
STEPS_CSV = OUT_DIR / "step_curves.csv"
TTT_CSV = OUT_DIR / "ttt_curves.csv"

# ─── Log directories ───
DEFAULT_DIRS = [
    "./sweeps/local",
    "./Results_colab_package/Results_h100_package",
]

def parse_header(text):
    """Extract config from log header."""
    cfg = {}
    # model params
    m = re.search(r'model_params:(\d+)', text)
    if m: cfg['model_params'] = int(m.group(1))

    m = re.search(r'world_size:(\d+)', text)
    if m: cfg['world_size'] = int(m.group(1))

    m = re.search(r'grad_accum_steps:(\d+)', text)
    if m: cfg['grad_accum_steps'] = int(m.group(1))

    m = re.search(r'train_batch_tokens:(\d+)', text)
    if m: cfg['train_batch_tokens'] = int(m.group(1))

    m = re.search(r'train_seq_len:(\d+)', text)
    if m: cfg['train_seq_len'] = int(m.group(1))

    m = re.search(r'iterations:(\d+)', text)
    if m: cfg['iterations'] = int(m.group(1))

    m = re.search(r'max_wallclock_seconds:([\d.]+)', text)
    if m: cfg['max_wallclock_s'] = float(m.group(1))

    m = re.search(r'seed:(\d+)', text)
    if m: cfg['seed'] = int(m.group(1))

    m = re.search(r'num_heads:(\d+)', text)
    if m: cfg['num_heads'] = int(m.group(1))

    m = re.search(r'num_kv_heads:(\d+)', text)
    if m: cfg['num_kv_heads'] = int(m.group(1))

    m = re.search(r'XSA:last_(\d+)', text)
    if m: cfg['xsa_last_n'] = int(m.group(1))

    m = re.search(r'mtp_num_heads:(\d+)', text)
    if m: cfg['mtp_num_heads'] = int(m.group(1))

    # Architecture config — try log fields first, fall back to name parsing
    for field, pattern in [
        ('num_layers', r'num_layers:(\d+)'),
        ('model_dim', r'model_dim:(\d+)'),
        ('mlp_mult', r'mlp_mult:([\d.]+)'),
        ('qk_gain_init', r'qk_gain_init:([\d.]+)'),
        ('logit_softcap', r'logit_softcap:([\d.]+)'),
        ('warmdown_iters', r'warmdown_iters:(\d+)'),
        ('warmup_steps', r'warmup_steps:(\d+)'),
        ('muon_wd', r'muon_wd:([\d.]+)'),
        ('bigram_vocab_size', r'bigram_vocab_size:(\d+)'),
        ('bigram_dim', r'bigram_dim:(\d+)'),
        ('ema_decay', r'ema_decay[=:]([\d.]+)'),
    ]:
        m = re.search(pattern, text)
        if m:
            val = m.group(1)
            cfg[field] = float(val) if '.' in val else int(val)

    # Pre-quant val_bpb (post-EMA diagnostic)
    m = re.search(r'DIAGNOSTIC post_ema.*val_bpb:([\d.]+)', text)
    if m: cfg['prequant_bpb'] = float(m.group(1))

    # quant config
    m = re.search(r'quant_config: mode=(\S+) bits_mlp=(\d+) bits_attn=(\d+)', text)
    if m:
        cfg['quant_mode'] = m.group(1)
        cfg['quant_bits_mlp'] = int(m.group(2))
        cfg['quant_bits_attn'] = int(m.group(3))

    # TTT config
    m = re.search(r'ttt_sliding:start.*ttt_lr=([\d.]+).*ttt_epochs=(\d+)', text)
    if m:
        cfg['ttt_lr'] = float(m.group(1))
        cfg['ttt_epochs'] = int(m.group(2))
        cfg['ttt_enabled'] = True
    else:
        cfg['ttt_enabled'] = False

    m = re.search(r'ttt_sliding:start.*freeze_blocks=(\d+)', text)
    if m: cfg['ttt_freeze_blocks'] = int(m.group(1))

    # Detect TTT optimizer from log (if logged)
    m = re.search(r'ttt_optimizer:(\w+)', text)
    if m: cfg['ttt_optimizer'] = m.group(1)

    # Detect hardware from step timing
    step_avgs = re.findall(r'step_avg:([\d.]+)ms', text)
    if step_avgs:
        avg_ms = float(step_avgs[-1])
        cfg['step_avg_ms'] = avg_ms
        # RTX PRO 6000: ~1500-1600ms/step, H100: ~760-780ms/step
        cfg['hardware'] = 'h100' if avg_ms < 1000 else 'rtx_pro_6000'

    # Submission size
    m = re.search(r'Total submission size \S+: (\d+) bytes', text)
    if m: cfg['artifact_bytes'] = int(m.group(1))

    # Compressor
    m = re.search(r'Serialized model (\w+):', text)
    if m: cfg['compressor'] = m.group(1)

    return cfg


def parse_steps(text):
    """Extract per-step training data."""
    steps = []
    for m in re.finditer(
        r'step:(\d+)/\d+ (?:train_loss:([\d.]+) )?(?:val_loss:([\d.]+) val_bpb:([\d.]+) )?'
        r'train_time:(\d+)ms step_avg:([\d.]+)ms', text
    ):
        row = {
            'step': int(m.group(1)),
            'train_loss': float(m.group(2)) if m.group(2) else None,
            'val_loss': float(m.group(3)) if m.group(3) else None,
            'val_bpb': float(m.group(4)) if m.group(4) else None,
            'train_time_ms': int(m.group(5)),
            'step_avg_ms': float(m.group(6)),
        }
        steps.append(row)
    return steps


def parse_ttt_chunks(text):
    """Extract per-chunk TTT data."""
    chunks = []
    for m in re.finditer(r'ttt_chunk \[(\d+)/(\d+)\] bpb=([\d.]+) time=([\d.]+)s', text):
        chunks.append({
            'chunk': int(m.group(1)),
            'total_chunks': int(m.group(2)),
            'bpb': float(m.group(3)),
            'time_s': float(m.group(4)),
        })
    return chunks


def parse_final_metrics(text):
    """Extract final evaluation metrics."""
    metrics = {}

    m = re.search(r'final_int6_roundtrip_exact.*val_bpb:([\d.]+)', text)
    if m: metrics['quant_bpb'] = float(m.group(1))

    m = re.search(r'final_int6_sliding_window_exact.*val_bpb:([\d.]+)', text)
    if m: metrics['sliding_bpb'] = float(m.group(1))

    m = re.search(r'legal_ttt_exact.*val_bpb:([\d.]+)', text)
    if m: metrics['ttt_bpb'] = float(m.group(1))

    # Last val_bpb during training (pre-quant)
    vals = re.findall(r'step:\d+/\d+.*val_bpb:([\d.]+)', text)
    if vals: metrics['final_val_bpb'] = float(vals[-1])

    # Training steps completed
    m = re.search(r'stopping_early:.*step:(\d+)', text)
    if m: metrics['final_step'] = int(m.group(1))
    else:
        steps = re.findall(r'step:(\d+)/\d+', text)
        if steps: metrics['final_step'] = int(steps[-1])

    # Training time
    m = re.search(r'stopping_early:.*train_time:(\d+)ms', text)
    if m: metrics['train_time_ms'] = int(m.group(1))

    # GPTQ
    if 'gptq:collecting' in text:
        metrics['gptq'] = True
    else:
        metrics['gptq'] = False

    # N-gram cache
    m = re.search(r'ngram_cache:(\w+)', text)
    if m: metrics['ngram_cache'] = m.group(1) == 'enabled'

    return metrics


def parse_log(filepath):
    """Parse a single log file into structured data."""
    text = Path(filepath).read_text(errors='replace')

    # Skip empty or non-training logs
    if 'model_params' not in text and 'step:' not in text:
        return None

    name = Path(filepath).stem
    cfg = parse_header(text)
    cfg['name'] = name
    cfg['logfile'] = str(filepath)

    # Parse config from experiment name (for screening runs)
    # e.g., screen_muon_wd0.04, screen_mlp3.5_q86, s2_layers12, s2_qk1.5_sc30.0
    if name.startswith('screen_') or name.startswith('s2_') or name.startswith('s3_'):
        cfg['sweep_type'] = 'screening'
        # MLP mult from name
        m_name = re.search(r'mlp([\d.]+)', name)
        if m_name and 'mlp_mult' not in cfg:
            cfg['mlp_mult'] = float(m_name.group(1))
        # Quant bits from name (e.g., q86 = mlp8/attn6)
        m_name = re.search(r'q(\d)(\d)', name)
        if m_name and 'quant_bits_mlp' not in cfg:
            cfg['quant_bits_mlp'] = int(m_name.group(1))
            cfg['quant_bits_attn'] = int(m_name.group(2))
        # Muon WD from name
        m_name = re.search(r'muon_wd([\d.]+)', name)
        if m_name and 'muon_wd' not in cfg:
            cfg['muon_wd'] = float(m_name.group(1))
        # Warmdown from name
        m_name = re.search(r'warmdown(\d+)', name)
        if m_name and 'warmdown_iters' not in cfg:
            cfg['warmdown_iters'] = int(m_name.group(1))
        # Bigram from name
        m_name = re.search(r'bigram_v(\d+)_d(\d+)', name)
        if m_name:
            cfg.setdefault('bigram_vocab_size', int(m_name.group(1)))
            cfg.setdefault('bigram_dim', int(m_name.group(2)))
        # Num layers from name
        m_name = re.search(r'layers(\d+)|(\d+)L', name)
        if m_name and 'num_layers' not in cfg:
            cfg['num_layers'] = int(m_name.group(1) or m_name.group(2))
        # QK gain from name
        m_name = re.search(r'qk([\d.]+)', name)
        if m_name and 'qk_gain_init' not in cfg:
            cfg['qk_gain_init'] = float(m_name.group(1))
        # Softcap from name
        m_name = re.search(r'sc([\d.]+)', name)
        if m_name and 'logit_softcap' not in cfg:
            cfg['logit_softcap'] = float(m_name.group(1))
        # EMA from name
        m_name = re.search(r'ema([\d.]+)', name)
        if m_name and 'ema_decay' not in cfg:
            cfg['ema_decay'] = float(m_name.group(1))
        # Batch from name
        m_name = re.search(r'batch(\d+)', name)
        if m_name and 'train_batch_tokens' not in cfg:
            cfg['train_batch_tokens'] = int(m_name.group(1))
        # Compressor from name
        if 'brotli' in name:
            cfg.setdefault('compressor', 'brotli')
        elif 'lzma' in name:
            cfg.setdefault('compressor', 'lzma')
        # Spectral K from name
        m_name = re.search(r'spectral(\d+)|spec(\d+)', name)
        if m_name:
            cfg['ttt_spectral_k'] = int(m_name.group(1) or m_name.group(2))
        # TTT chunk from name
        m_name = re.search(r'chunk(\d+)', name)
        if m_name:
            cfg['ttt_chunk_tokens'] = int(m_name.group(1))

    steps = parse_steps(text)
    ttt = parse_ttt_chunks(text)
    finals = parse_final_metrics(text)

    cfg.update(finals)

    return cfg, steps, ttt


def collect_all_logs(dirs):
    """Collect all parseable log files from directories."""
    logs = []
    for d in dirs:
        for ext in ['*.log', '*.txt']:
            logs.extend(glob.glob(os.path.join(d, ext)))
    # Also grab the (1), (2), (3) files
    for d in dirs:
        for f in glob.glob(os.path.join(d, '*')):
            if os.path.isfile(f) and f not in logs:
                # Check if it looks like a log
                try:
                    head = open(f, 'r', errors='replace').read(500)
                    if 'model_params' in head or 'step:' in head:
                        logs.append(f)
                except:
                    pass
    return sorted(set(logs))


def write_csvs(all_summaries, all_steps, all_ttt):
    """Write structured CSVs."""
    # Summary CSV
    if all_summaries:
        all_keys = set()
        for s in all_summaries:
            all_keys.update(s.keys())
        all_keys = sorted(all_keys)

        with open(SUMMARY_CSV, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=all_keys)
            w.writeheader()
            for s in all_summaries:
                w.writerow(s)
        print(f"  Summary: {len(all_summaries)} experiments -> {SUMMARY_CSV}")

    # Step curves CSV
    if all_steps:
        step_keys = ['experiment', 'step', 'train_loss', 'val_loss', 'val_bpb',
                      'train_time_ms', 'step_avg_ms']
        with open(STEPS_CSV, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=step_keys)
            w.writeheader()
            for name, steps in all_steps:
                for s in steps:
                    row = {'experiment': name, **s}
                    w.writerow(row)
        total_rows = sum(len(s) for _, s in all_steps)
        print(f"  Steps:   {total_rows} rows from {len(all_steps)} experiments -> {STEPS_CSV}")

    # TTT curves CSV
    if all_ttt:
        ttt_keys = ['experiment', 'chunk', 'total_chunks', 'bpb', 'time_s']
        with open(TTT_CSV, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=ttt_keys)
            w.writeheader()
            for name, chunks in all_ttt:
                for c in chunks:
                    row = {'experiment': name, **c}
                    w.writerow(row)
        total_rows = sum(len(c) for _, c in all_ttt)
        print(f"  TTT:     {total_rows} rows from {len(all_ttt)} experiments -> {TTT_CSV}")


def main():
    dirs = DEFAULT_DIRS
    extra_logs = []

    for arg in sys.argv[1:]:
        if arg == '--add':
            continue
        elif os.path.isdir(arg):
            dirs.append(arg)
        elif os.path.isfile(arg):
            extra_logs.append(arg)

    print(f"Scanning {len(dirs)} directories...")
    logs = collect_all_logs(dirs) + extra_logs
    print(f"Found {len(logs)} log files")

    all_summaries = []
    all_steps = []
    all_ttt = []
    skipped = 0

    for lf in logs:
        try:
            result = parse_log(lf)
            if result is None:
                skipped += 1
                continue
            cfg, steps, ttt = result
            all_summaries.append(cfg)
            if steps:
                all_steps.append((cfg['name'], steps))
            if ttt:
                all_ttt.append((cfg['name'], ttt))
        except Exception as e:
            print(f"  ERROR parsing {lf}: {e}")
            skipped += 1

    print(f"Parsed {len(all_summaries)} experiments, skipped {skipped}")
    write_csvs(all_summaries, all_steps, all_ttt)

    # Print quick summary table
    print("\n=== EXPERIMENT SUMMARY ===")
    print(f"{'Name':<45} {'HW':<15} {'Steps':>6} {'ValBPB':>8} {'SlidBPB':>8} {'TTTBPB':>8} {'Artifact':>10}")
    print("-" * 110)
    for s in sorted(all_summaries, key=lambda x: x.get('ttt_bpb', x.get('sliding_bpb', x.get('final_val_bpb', 99)))):
        name = s.get('name', '?')[:44]
        hw = s.get('hardware', '?')[:14]
        steps = s.get('final_step', '')
        val = s.get('final_val_bpb', '')
        slid = s.get('sliding_bpb', '')
        ttt = s.get('ttt_bpb', '')
        art = s.get('artifact_bytes', '')
        if art:
            art = f"{art/1024/1024:.2f}MB"
        print(f"{name:<45} {hw:<15} {steps:>6} {val:>8} {slid:>8} {ttt:>8} {art:>10}")


if __name__ == '__main__':
    main()
