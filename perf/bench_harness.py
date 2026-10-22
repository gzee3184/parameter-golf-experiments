"""
Shared benchmark harness for optimization experiments.

Provides a standardized way to:
- Create the GPT model with competition hyperparameters
- Run N training steps with synthetic data
- Measure tok/s, ms/step, peak memory
- Compare configurations A/B with statistical rigor
"""

import copy
import sys
import time
import torch
import torch.nn.functional as F
from torch import Tensor

sys.path.insert(0, "./src/baselines")
from train_gpt_modified import GPT, Hyperparameters, restore_fp32_params

DEVICE = "cuda"


def create_model(h: Hyperparameters | None = None, seed: int = 42) -> tuple[GPT, Hyperparameters, dict]:
    """Create a fresh GPT model with competition hyperparameters.
    Returns (model, hparams, init_state_dict)."""
    if h is None:
        h = Hyperparameters()
    torch.manual_seed(seed)
    model = GPT(h).to(DEVICE).bfloat16()
    restore_fp32_params(model)
    init_state = copy.deepcopy(model.state_dict())
    return model, h, init_state


def run_steps(
    model: GPT,
    h: Hyperparameters,
    num_steps: int = 50,
    batch_size: int = 48,
    seq_len: int = 2048,
    warmup_steps: int = 3,
    compile_model: bool = True,
    compile_kwargs: dict | None = None,
    pre_step_hook=None,
    use_amp: bool = True,
) -> dict:
    """Run training steps and collect metrics.

    Args:
        model: GPT model (uncompiled)
        h: hyperparameters
        num_steps: steps to measure
        batch_size: sequences per step (no grad accum here)
        seq_len: sequence length
        warmup_steps: untimed warmup steps
        compile_model: whether to torch.compile
        compile_kwargs: extra kwargs for torch.compile
        pre_step_hook: callable(step) called before each measured step
        use_amp: use autocast

    Returns dict with keys: losses, elapsed_s, tok_per_s, ms_per_step, peak_mib, compile_time_s
    """
    compile_kwargs = compile_kwargs or {}

    compile_start = time.perf_counter()
    if compile_model:
        compiled = torch.compile(model, dynamic=False, fullgraph=True, **compile_kwargs)
    else:
        compiled = model
    compile_time = time.perf_counter() - compile_start

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    vocab_size = h.vocab_size

    # Warmup (includes compile + CUDA warmup)
    for i in range(warmup_steps):
        x = torch.randint(0, vocab_size, (batch_size, seq_len), device=DEVICE)
        y = torch.randint(0, vocab_size, (batch_size, seq_len), device=DEVICE)
        optimizer.zero_grad()
        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = compiled(x, y)
        else:
            loss = compiled(x, y)
        loss.backward()
        optimizer.step()

    # Measured steps
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    losses = []
    t0 = time.perf_counter()
    tokens_total = 0

    for step in range(num_steps):
        if pre_step_hook:
            pre_step_hook(step)
        x = torch.randint(0, vocab_size, (batch_size, seq_len), device=DEVICE)
        y = torch.randint(0, vocab_size, (batch_size, seq_len), device=DEVICE)
        optimizer.zero_grad()
        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = compiled(x, y)
        else:
            loss = compiled(x, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        tokens_total += batch_size * seq_len

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    peak_mib = torch.cuda.max_memory_allocated() / 1024**2

    return {
        "losses": losses,
        "elapsed_s": elapsed,
        "tok_per_s": tokens_total / elapsed,
        "ms_per_step": (elapsed / num_steps) * 1000,
        "peak_mib": peak_mib,
        "compile_time_s": compile_time,
    }


def print_comparison(results: dict[str, dict], baseline_key: str = "baseline"):
    """Print a comparison table of multiple configurations."""
    base = results.get(baseline_key)
    if base is None:
        base = next(iter(results.values()))

    print(f"\n{'Config':<40} {'ms/step':<10} {'tok/s':<14} {'Speedup':<10} {'Peak MiB':<10} {'Compile s':<10}")
    print("-" * 94)
    for name, r in results.items():
        speedup = base["ms_per_step"] / r["ms_per_step"]
        marker = " *" if name == baseline_key else ""
        print(
            f"{name:<40} {r['ms_per_step']:<10.1f} {r['tok_per_s']:<14,.0f} "
            f"{speedup:<10.2f}x {r['peak_mib']:<10.0f} {r['compile_time_s']:<10.1f}{marker}"
        )

    # Loss divergence
    if len(results) > 1:
        base_losses = base["losses"]
        print(f"\n{'Config':<40} {'Max loss diff':<15} {'Final loss':<12}")
        print("-" * 67)
        for name, r in results.items():
            max_diff = max(abs(a - b) for a, b in zip(r["losses"], base_losses))
            print(f"{name:<40} {max_diff:<15.6f} {r['losses'][-1]:<12.4f}")
