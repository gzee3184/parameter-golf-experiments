"""Download minimal SP8192 data to scratch.

Val (1 file) + tokenizer + N train shards. Files placed at
parameter-golf/data/datasets/fineweb10B_sp8192/ and parameter-golf/data/tokenizers/
so the current train_gpt.py paths work without changes.

Source: kevclark/parameter-golf (files live under datasets/datasets/fineweb10B_sp8192/).

IMPORTANT: run via the wrapper download_sp8192.sh so HF_HOME is set BEFORE Python starts.
"""
import os, sys, time, shutil
from pathlib import Path

# Set HF_HOME before you run this script if your home directory has a quota.

from huggingface_hub import hf_hub_download

REPO = "kevclark/parameter-golf"
PROJ = Path(".")
DATA_ROOT = PROJ / "parameter-golf" / "data"
DS_DIR = DATA_ROOT / "datasets" / "fineweb10B_sp8192"
TOK_DIR = DATA_ROOT / "tokenizers"
DS_DIR.mkdir(parents=True, exist_ok=True)
TOK_DIR.mkdir(parents=True, exist_ok=True)

N_SHARDS = int(os.environ.get("N_SHARDS", "10"))

def get(remote_rel: str, local_dest: Path):
    # Remove stale symlinks or zero-byte files from prior failed attempts.
    if local_dest.is_symlink() or (local_dest.exists() and local_dest.stat().st_size == 0):
        local_dest.unlink(missing_ok=True)
    if local_dest.exists() and local_dest.stat().st_size > 0:
        print(f"SKIP {local_dest.name} (exists, {local_dest.stat().st_size/1e6:.1f}MB)")
        return
    t0 = time.perf_counter()
    print(f"  fetching {remote_rel} ...", flush=True)
    # Files in repo are nested at datasets/datasets/fineweb10B_sp8192/ or datasets/tokenizers/
    path = Path(remote_rel)
    cached = hf_hub_download(
        repo_id=REPO,
        filename=path.name,
        subfolder=path.parent.as_posix(),
        repo_type="dataset",
    )
    # Hardlink if same FS, otherwise copy (home HF cache -> scratch is cross-FS)
    cached_real = Path(cached).resolve(strict=True)
    try:
        os.link(cached_real, local_dest)
    except OSError:
        shutil.copy2(cached_real, local_dest)
    dt = time.perf_counter() - t0
    print(f"  -> {local_dest} ({local_dest.stat().st_size/1e6:.1f}MB, {dt:.1f}s)")

print(f"Downloading SP8192 to {DS_DIR}")
print(f"HF cache: {os.environ['HUGGINGFACE_HUB_CACHE']}")
print()

print("[1/3] Tokenizer:")
get("datasets/tokenizers/fineweb_8192_bpe.model", TOK_DIR / "fineweb_8192_bpe.model")
get("datasets/tokenizers/fineweb_8192_bpe.vocab", TOK_DIR / "fineweb_8192_bpe.vocab")

print("\n[2/3] Validation:")
get("datasets/datasets/fineweb10B_sp8192/fineweb_val_000000.bin",
    DS_DIR / "fineweb_val_000000.bin")

print(f"\n[3/3] Train shards (first {N_SHARDS}):")
for i in range(N_SHARDS):
    name = f"fineweb_train_{i:06d}.bin"
    get(f"datasets/datasets/fineweb10B_sp8192/{name}", DS_DIR / name)

print("\nDONE")
total_sz = sum(f.stat().st_size for f in DS_DIR.glob("*.bin"))
print(f"Total: {total_sz/1e9:.2f}GB across {len(list(DS_DIR.glob('*.bin')))} files")
