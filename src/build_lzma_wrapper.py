#!/usr/bin/env python3
"""Build an LZMA-compressed code wrapper for submission.

Takes train_gpt.py (or any Python file), LZMA-compresses it, and produces a
small bootstrap script that decompresses + exec's it. This shrinks ~60KB code
to ~18KB, freeing ~42KB for model weights in the 16MB artifact budget.

Usage:
    python build_lzma_wrapper.py train_gpt.py train_gpt_lzma.py
    python build_lzma_wrapper.py train_gpt.py train_gpt_lzma.py --verify
"""
import argparse
import base64
import lzma


def build_wrapper(input_path: str, output_path: str, verify: bool = False):
    with open(input_path, "rb") as f:
        raw = f.read()
    raw_size = len(raw)

    # LZMA compress with maximum compression
    compressed = lzma.compress(raw, preset=9 | lzma.PRESET_EXTREME)
    comp_size = len(compressed)

    # Use base85 encoding (most compact ASCII-safe encoding in stdlib)
    encoded = base64.b85encode(compressed)

    # Build the bootstrap wrapper
    wrapper = f'import base64,lzma;exec(lzma.decompress(base64.b85decode({encoded!r})))\n'
    wrapper_bytes = wrapper.encode("utf-8")
    wrapper_size = len(wrapper_bytes)

    with open(output_path, "wb") as f:
        f.write(wrapper_bytes)

    savings = raw_size - wrapper_size
    print(f"Raw code:    {raw_size:>8,} bytes")
    print(f"Compressed:  {comp_size:>8,} bytes (LZMA ratio: {comp_size/raw_size:.1%})")
    print(f"Wrapper:     {wrapper_size:>8,} bytes (with base85 overhead)")
    print(f"Savings:     {savings:>8,} bytes ({savings/raw_size:.1%} smaller)")
    print(f"Written to:  {output_path}")

    if verify:
        print("\nVerifying wrapper produces identical code...")
        decompressed = lzma.decompress(base64.b85decode(encoded))
        if decompressed == raw:
            print("OK: decompressed code matches original")
        else:
            print(f"VERIFY FAILED: decompressed {len(decompressed)} bytes != original {len(raw)} bytes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build LZMA code wrapper")
    parser.add_argument("input", help="Input Python file")
    parser.add_argument("output", help="Output wrapper file")
    parser.add_argument("--verify", action="store_true", help="Verify decompression matches original")
    args = parser.parse_args()
    build_wrapper(args.input, args.output, args.verify)
