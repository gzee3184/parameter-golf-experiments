#!/usr/bin/env python3
"""
Napkin math for Parameter Golf / 8×H100 / 10min thinking.

We pretend peak TFLOPS tell you wall time. They don't — bandwidth, kernels, NCCL
all matter — but this gives you a ballpark before/after a run.

  python3 scripts/h100_time_guess.py
      → list GPUs, ask: "8×H100 @ 10min ≈ how long on this box?"

  python3 scripts/h100_time_guess.py check run.log
      → read train_gpt.py log; estimate train time if scaled to 8×H100 peaks; vs 600s cap

  python3 scripts/h100_time_guess.py check run.log 210
      → same, but say your machine was 210 TFLOPS total (e.g. log from another PC)

Optional env for `check` (only if you need them):
  LOCAL_TFLOPS=210     same as the extra number arg
  H100_MS_PER_STEP=42  if you measured avg ms/step on real 8×H100, we use steps*ms instead
  H100_TRAIN_MS=480000 if this log is *already* from 8×H100, compare train ms directly

Edit GPU_TFLOPS below when nvidia-smi shows a name we don't know.
"""

# --- knobs you might actually touch ------------------------------------------
REF_GPUS = 8
REF_MINUTES = 10.0
H100_PEAK_TFLOPS = 1979.0  # per GPU, SXM FP16/BF16 tensor peak (datasheet round number)
OFFICIAL_TRAIN_SECONDS = 600.0

# longer substring first (order matters)
GPU_TFLOPS = [
    ("RTX PRO 6000 Blackwell Max-Q", 209.0),  # BF16 tensor peak, Blackwell sm_120
    ("RTX PRO 6000", 209.0),
    ("H100 SXM", 1979.0),
    ("H100 NVL", 1671.0),
    ("H100 PCIe", 756.0),
    ("A100-SXM4-80GB", 624.0),
    ("A100 SXM4", 624.0),
    ("A100 80GB PCIe", 312.0),
    ("A100 80GB", 624.0),
    ("A100 40GB", 312.0),
    ("A100", 312.0),
    ("L40S", 733.0),
    ("L40", 181.05),
    ("GeForce RTX 5090", 105.0),
    ("RTX 5090", 105.0),
    ("GeForce RTX 5080", 56.0),
    ("RTX 5080", 56.0),
    ("GeForce RTX 4090", 82.6),
    ("RTX 4090", 82.6),
    ("GeForce RTX 4080", 48.7),
    ("RTX 4080", 48.7),
    ("GeForce RTX 4070 Ti", 23.2),
    ("RTX 4070 Ti", 23.2),
    ("GeForce RTX 4070", 14.3),
    ("RTX 4070", 14.3),
    ("H100", 1979.0),
]

import os
import re
import subprocess
import sys
from pathlib import Path


def nvidia_gpu_names():
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        text=True,
        timeout=30,
    )
    return [ln.strip() for ln in out.splitlines() if ln.strip()]


def tflops_one(name):
    for key, t in GPU_TFLOPS:
        if key in name:
            return t, key
    return None, None


def machine_total_tflops(names):
    total = 0.0
    unknown = False
    for n in names:
        t, _ = tflops_one(n)
        if t is None:
            unknown = True
        else:
            total += t
    return total, unknown


def pretty_duration(sec):
    if sec < 90:
        return f"{sec:.0f} s"
    if sec < 3600:
        return f"{sec / 60:.1f} min"
    return f"{sec / 3600:.1f} h"


def show_equiv():
    try:
        names = nvidia_gpu_names()
    except Exception as e:
        print("nvidia-smi problem:", e)
        sys.exit(1)

    print(f"visible GPUs ({len(names)}):")
    total = 0.0
    for i, n in enumerate(names):
        t, key = tflops_one(n)
        if t:
            print(f"  [{i}] {n}  →  ~{t:.0f} TFLOPS ({key})")
            total += t
        else:
            print(f"  [{i}] {n}  →  ??? add a line to GPU_TFLOPS in h100_time_guess.py")

    ref_total = REF_GPUS * H100_PEAK_TFLOPS
    ref_sec = REF_MINUTES * 60.0
    # "how much work" in TFLOP-seconds (meaningless absolute number; ratio is what we use)
    ref_work = ref_total * ref_sec

    print()
    if total <= 0:
        print("no TFLOPS sum — can't convert.")
        return

    here_sec = ref_work / total
    print(
        f"ballpark: {REF_GPUS}× ~{H100_PEAK_TFLOPS:.0f} TFLOPS for {REF_MINUTES:g} min "
        f"≈ {pretty_duration(here_sec)} on this rack ({total:.0f} TFLOPS peak total)"
    )
    print("grain of salt: use log step_avg when you have it.")


def scrub_log(text):
    """pull numbers from train_gpt.py stdout"""
    times = [float(m.group(1)) for m in re.finditer(r"train_time:(\d+(?:\.\d+)?)ms", text)]
    steps = [int(m.group(1)) for m in re.finditer(r"step:(\d+)/(\d+)", text)]
    iters = None
    m = re.search(r"\biterations:(\d+)\b", text)
    if m:
        iters = int(m.group(1))
    wallcap = "stopping_early: wallclock_cap" in text
    return max(times) if times else None, max(steps) if steps else None, iters, wallcap


def check_log(path, local_tflops_arg=None):
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    train_ms, max_step, iterations, early = scrub_log(text)
    if train_ms is None:
        print("no train_time:..ms lines — is this a train_gpt.py log?")
        sys.exit(1)

    ref_total = REF_GPUS * H100_PEAK_TFLOPS
    train_s = train_ms / 1000.0

    ms_direct = os.environ.get("H100_TRAIN_MS")
    ms_per_step_h100 = os.environ.get("H100_MS_PER_STEP")
    local_env = os.environ.get("LOCAL_TFLOPS")
    local_total = None
    if local_tflops_arg is not None:
        local_total = float(local_tflops_arg)
    elif local_env:
        local_total = float(local_env)
    if ms_direct:
        est_s = float(ms_direct) / 1000.0
        how = "direct H100_TRAIN_MS"
    elif ms_per_step_h100 and max_step:
        est_s = max_step * float(ms_per_step_h100) / 1000.0
        how = f"steps × H100_MS_PER_STEP ({max_step} × {ms_per_step_h100} ms)"
    else:
        if local_total is None:
            try:
                names = nvidia_gpu_names()
            except Exception as e:
                print("need LOCAL_TFLOPS=... or pass number after log path. nvidia-smi:", e)
                sys.exit(1)
            local_total, bad = machine_total_tflops(names)
            if bad or local_total <= 0:
                print("unknown GPU in table — set LOCAL_TFLOPS=sum_peak_tflobs or extend GPU_TFLOPS")
                sys.exit(1)
        est_s = train_s * local_total / ref_total
        how = f"FLOPS scale (local {local_total:.0f} TFLOPS vs ref {ref_total:.0f})"

    ok = est_s <= OFFICIAL_TRAIN_SECONDS
    print(path)
    print(f"  last train_time in log: {train_ms:.0f} ms ({train_s:.1f} s)  |  max step: {max_step}")
    if iterations is not None:
        print(f"  iterations in header: {iterations}")
        if early and max_step is not None and max_step < iterations:
            print("  note: log shows wallclock early-stop before full iterations")
    print(f"  {how}")
    print(
        f"  → guess at 8×H100 train time: {est_s:.1f} s   official cap: {OFFICIAL_TRAIN_SECONDS:.0f} s   "
        + ("PASS" if ok else "FAIL (napkin math)")
    )
    if not ok:
        sys.exit(1)


def main():
    if len(sys.argv) == 1:
        show_equiv()
    elif sys.argv[1] == "check":
        if len(sys.argv) < 3:
            print("usage: h100_time_guess.py check run.log [local_total_tflops]")
            sys.exit(1)
        extra = None
        if len(sys.argv) > 3:
            try:
                extra = float(sys.argv[3])
            except ValueError:
                print("third argument, if present, must be local total TFLOPS (a number)")
                sys.exit(1)
        check_log(sys.argv[2], extra)
    else:
        print("usage: h100_time_guess.py   OR   h100_time_guess.py check run.log [local_total_tflops]")
        sys.exit(1)


if __name__ == "__main__":
    main()
