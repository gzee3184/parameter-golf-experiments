#!/usr/bin/env python3
"""
Inductor config sweep: test different torch._inductor.config settings
to find the fastest configuration for the Parameter Golf GPT model.

Tests configs in isolation first, then combines the winners.

Usage:
    python perf/inductor_sweep.py [--steps 30] [--batch-size 48] [--seq-len 2048]
    python perf/inductor_sweep.py --quick  # 10 steps, batch 16, seq 512
"""

import argparse
import gc
import os
import sys
import time
import torch

sys.path.insert(0, ".")
from perf.bench_harness import create_model, run_steps, print_comparison


def reset_inductor():
    """Reset inductor state between runs to avoid cache contamination."""
    torch._dynamo.reset()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def set_inductor_config(overrides: dict):
    """Apply inductor config overrides. Reset first."""
    import torch._inductor.config as cfg
    # Save defaults so we can restore
    for key, val in overrides.items():
        parts = key.split(".")
        obj = cfg
        for part in parts[:-1]:
            obj = getattr(obj, part)
        setattr(obj, parts[-1], val)


def restore_inductor_defaults(defaults: dict):
    """Restore inductor config to saved defaults."""
    set_inductor_config(defaults)


def get_current_config(keys: list[str]) -> dict:
    """Get current values for config keys."""
    import torch._inductor.config as cfg
    result = {}
    for key in keys:
        parts = key.split(".")
        obj = cfg
        for part in parts:
            obj = getattr(obj, part)
        result[key] = obj
    return result


# ============================================================================
# Configuration presets to test
# ============================================================================

# Each preset is a dict of inductor config overrides.
# We test them individually first, then combine winners.

PRESETS = {
    # --- Fusion configs (universal, not GPU-specific) ---
    "aggressive_fusion": {
        "aggressive_fusion": True,
    },
    "epilogue_fusion_off": {
        # Test: is the default epilogue fusion actually helping or hurting?
        "epilogue_fusion": False,
    },
    "cutlass_epilogue": {
        # GEMM epilogue fusion: fuse pointwise ops into CUTLASS GEMM writeback
        "cuda.cutlass_epilogue_fusion_enabled": True,
    },
    "b2b_gemm": {
        # Back-to-back GEMM fusion (fc -> act -> proj could become one kernel)
        "b2b_gemm_pass": True,
    },
    "combo_kernels": {
        # Fuse multiple independent pointwise kernels into one launch
        "combo_kernels": True,
    },
    "permute_fusion": {
        # Fuse permute/transpose into adjacent ops
        "permute_fusion": True,
    },
    "max_fusion_128": {
        # Increase max fusion size from 64 to 128
        "max_fusion_size": 128,
    },

    # --- Autotune configs (GPU-specific — will need re-tuning on H100) ---
    "max_autotune_gemm": {
        # Benchmark ATEN/Triton/CUTLASS GEMMs and pick fastest
        "max_autotune_gemm": True,
        "max_autotune_gemm_backends": "ATEN,TRITON",
    },
    "coord_descent": {
        # Coordinate descent tuning of Triton kernel tile sizes
        "coordinate_descent_tuning": True,
    },
    "max_autotune_pointwise": {
        # Also autotune pointwise Triton kernels
        "max_autotune_pointwise": True,
    },

    # --- CUDA graphs ---
    "cudagraphs": {
        "triton.cudagraphs": True,
    },

    # --- Reduction optimization ---
    "cooperative_reductions": {
        "triton.cooperative_reductions": True,
    },

    # --- Combined: promising fusion flags ---
    "fusion_combo": {
        "aggressive_fusion": True,
        "combo_kernels": True,
        "permute_fusion": True,
        "max_fusion_size": 128,
    },
}


def run_preset(name: str, overrides: dict, h, init_state, args, defaults: dict) -> dict:
    """Run a single preset and return results."""
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"  Config: {overrides}")
    print(f"{'='*60}")

    reset_inductor()
    set_inductor_config(overrides)

    try:
        model, _, _ = create_model(h)
        model.load_state_dict(init_state)

        result = run_steps(
            model, h,
            num_steps=args.steps,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            compile_model=True,
        )
        result["status"] = "OK"
        print(f"  -> {result['ms_per_step']:.1f} ms/step, {result['tok_per_s']:,.0f} tok/s, "
              f"compile: {result['compile_time_s']:.1f}s, peak: {result['peak_mib']:.0f} MiB")

    except Exception as e:
        print(f"  -> FAILED: {e}")
        result = {
            "losses": [],
            "elapsed_s": 0,
            "tok_per_s": 0,
            "ms_per_step": float("inf"),
            "peak_mib": 0,
            "compile_time_s": 0,
            "status": f"FAILED: {e}",
        }

    finally:
        restore_inductor_defaults(defaults)
        del model
        gc.collect()
        torch.cuda.empty_cache()

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--quick", action="store_true", help="Quick mode: 10 steps, batch 16, seq 512")
    parser.add_argument("--only", type=str, default=None, help="Comma-separated preset names to test")
    parser.add_argument("--skip", type=str, default=None, help="Comma-separated preset names to skip")
    args = parser.parse_args()

    if args.quick:
        args.steps = 10
        args.batch_size = 16
        args.seq_len = 512

    print("=" * 70)
    print("Inductor Config Sweep — Parameter Golf GPT")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Steps: {args.steps}, Batch: {args.batch_size}, Seq: {args.seq_len}")
    print(f"Tokens/step: {args.batch_size * args.seq_len:,}")
    print("=" * 70)

    # Save default config values for all keys we'll touch
    all_keys = set()
    for overrides in PRESETS.values():
        all_keys.update(overrides.keys())
    defaults = get_current_config(list(all_keys))
    print(f"\nDefaults: {defaults}")

    # Create model and save initial state
    model, h, init_state = create_model()

    # Determine which presets to run
    preset_names = list(PRESETS.keys())
    if args.only:
        preset_names = [n for n in args.only.split(",") if n in PRESETS]
    if args.skip:
        skip = set(args.skip.split(","))
        preset_names = [n for n in preset_names if n not in skip]

    # Run baseline first
    print(f"\n{'='*60}")
    print("Testing: baseline (default inductor config)")
    print(f"{'='*60}")
    reset_inductor()
    model.load_state_dict(init_state)
    baseline = run_steps(
        model, h,
        num_steps=args.steps,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        compile_model=True,
    )
    baseline["status"] = "OK"
    print(f"  -> {baseline['ms_per_step']:.1f} ms/step, {baseline['tok_per_s']:,.0f} tok/s, "
          f"compile: {baseline['compile_time_s']:.1f}s, peak: {baseline['peak_mib']:.0f} MiB")

    del model
    gc.collect()
    torch.cuda.empty_cache()

    # Run each preset
    results = {"baseline": baseline}
    for name in preset_names:
        results[name] = run_preset(name, PRESETS[name], h, init_state, args, defaults)

    # Summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    # Filter to successful runs only for comparison
    ok_results = {k: v for k, v in results.items() if v.get("status") == "OK"}
    print_comparison(ok_results, baseline_key="baseline")

    # Rank by speedup
    base_ms = baseline["ms_per_step"]
    ranked = sorted(ok_results.items(), key=lambda x: x[1]["ms_per_step"])
    print(f"\n{'Rank':<6} {'Config':<40} {'Speedup':<10}")
    print("-" * 56)
    for i, (name, r) in enumerate(ranked, 1):
        speedup = base_ms / r["ms_per_step"]
        marker = " <-- WINNER" if i == 1 and name != "baseline" else ""
        print(f"{i:<6} {name:<40} {speedup:.3f}x{marker}")

    # Failed configs
    failed = {k: v for k, v in results.items() if v.get("status", "").startswith("FAILED")}
    if failed:
        print(f"\nFAILED configs:")
        for name, r in failed.items():
            print(f"  {name}: {r['status']}")

    # Suggest combined config
    winners = [(name, base_ms / r["ms_per_step"]) for name, r in ranked if name != "baseline" and base_ms / r["ms_per_step"] > 1.005]
    if winners:
        print(f"\nConfigs faster than baseline:")
        for name, speedup in winners:
            print(f"  {name}: {speedup:.3f}x — {PRESETS[name]}")
        print("\nNext step: combine these winners and test on H100.")


if __name__ == "__main__":
    main()
