"""Verify the r10 EMA checkpoint loads cleanly against the current train_gpt.py model.

Success: post_ema val_bpb approximately 1.1033 (within ~0.01 of reported).
Signal: any key mismatches, shape mismatches, or wildly wrong BPB.

Uses the current h100_deploy/train_gpt.py (which is baseline_r10/). Runs locally on 1 GPU.
"""
import os, sys, glob, time, math
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
H100_DEPLOY = os.path.abspath(os.path.join(HERE, ".."))
PROJ = os.path.abspath(os.path.join(H100_DEPLOY, ".."))
CHECKPOINT = os.path.join(H100_DEPLOY, "run10", "final_model_r10.pt")
DATA_PATH = os.path.join(PROJ, "parameter-golf", "data", "datasets", "fineweb10B_sp8192")
TOKENIZER_PATH = os.path.join(PROJ, "parameter-golf", "data", "tokenizers", "fineweb_8192_bpe.model")

os.environ.setdefault("VOCAB_SIZE", "8192")
os.environ.setdefault("NUM_LAYERS", "11")
os.environ.setdefault("MODEL_DIM", "512")
os.environ.setdefault("NUM_HEADS", "8")
os.environ.setdefault("NUM_KV_HEADS", "4")
os.environ.setdefault("MLP_MULT", "4")
os.environ.setdefault("ROPE_BASE", "10000")
os.environ.setdefault("LOGIT_SOFTCAP", "20")
os.environ.setdefault("TIE_EMBEDDINGS", "1")
os.environ.setdefault("SIGMOID_SKIP", "1")
os.environ.setdefault("RECUR_LAYERS", "3,4,5")
os.environ.setdefault("RECUR_NUM_LOOPS", "2")
os.environ.setdefault("VE_ENABLED", "0")
os.environ.setdefault("PARALLEL_START_LAYER", "7")
os.environ.setdefault("DISABLE_LAYER0_ATTN", "1")
os.environ.setdefault("XSA_LAST_N", "11")
os.environ.setdefault("REPEAT_UNTIE_MLP", "lora")
os.environ.setdefault("REPEAT_LORA_RANK", "16")
os.environ.setdefault("REPEAT_PASS_SCALE", "1")
os.environ.setdefault("QK_GAIN_INIT", "5.0")
os.environ.setdefault("DATA_PATH", DATA_PATH)
os.environ.setdefault("TOKENIZER_PATH", TOKENIZER_PATH)
os.environ.setdefault("VAL_BATCH_SIZE", str(524_288))

sys.path.insert(0, H100_DEPLOY)
# Choose which train_gpt.py to use: current (default) or r10-frozen (VERIFY_WITH_R10_CODE=1)
TG_SOURCE = os.path.join(H100_DEPLOY, "run10", "train_gpt.py") if os.environ.get("VERIFY_WITH_R10_CODE") else os.path.join(H100_DEPLOY, "train_gpt.py")
print(f"Using train_gpt.py: {TG_SOURCE}")
import importlib.util
spec = importlib.util.spec_from_file_location("train_gpt", TG_SOURCE)
tg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tg)

def main():
    device = torch.device("cuda:0")
    print(f"Loading checkpoint: {CHECKPOINT}")
    sd = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    print(f"Checkpoint has {len(sd)} keys")

    # Build model using current train_gpt.py factory
    a = tg.Hyperparameters()
    rll = [int(x) for x in a.rcl.split(",")] if a.rcl else None
    vll = None
    print(f"Building model: vs={a.vs} nl={a.nl} md={a.md} recur={rll}")
    model = tg._mk_gpt(a, rll, vll).to(device).bfloat16()
    # CL layers are kept fp32 for numerical stability at norm/quant boundaries.
    for m in model.modules():
        if isinstance(m, tg.CL):
            m.float()
    tg._sgd(model)

    model_sd = model.state_dict()
    print(f"Model has {len(model_sd)} keys")

    ck_keys = set(sd.keys())
    mk_keys = set(model_sd.keys())
    only_ck = ck_keys - mk_keys
    only_mk = mk_keys - ck_keys
    print(f"Only in checkpoint: {len(only_ck)}  {sorted(only_ck)[:5]}")
    print(f"Only in model:      {len(only_mk)}  {sorted(only_mk)[:5]}")

    # Shape check on common keys
    mismatches = []
    for k in ck_keys & mk_keys:
        if sd[k].shape != model_sd[k].shape:
            mismatches.append((k, tuple(sd[k].shape), tuple(model_sd[k].shape)))
    print(f"Shape mismatches: {len(mismatches)}")
    for k, a_sh, b_sh in mismatches[:5]:
        print(f"  {k}: ckpt {a_sh} vs model {b_sh}")

    if mismatches:
        print("ABORT: shape mismatches present, cannot proceed to eval.")
        return 2

    # Load (strict=False tolerates extra/missing scalars)
    load_out = model.load_state_dict({k: v.to(device) for k, v in sd.items() if k in model_sd}, strict=False)
    print(f"Missing after load: {load_out.missing_keys[:5]}  (total {len(load_out.missing_keys)})")
    print(f"Unexpected: {load_out.unexpected_keys[:5]}  (total {len(load_out.unexpected_keys)})")

    # Enable recurrence (r10 trained with it active at checkpoint time)
    model.set_ra(True)
    print(f"Virtual layer plan (recur active): {model._gvl()}")

    # Run eval on validation set — single GPU, so no dist
    print("Running eval on val...")
    sp = None
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(model_file=TOKENIZER_PATH)
    bbl, hll, ibl = tg._bsl(sp, a.vs, device)

    vt = tg._lvt(a.vaf, a.esl or a.tsl)
    print(f"Val tokens: {vt.numel():,}")

    t0 = time.perf_counter()
    vlo, vbp = tg.ev(a, model, rank=0, wz=1, device=device, ga=1, vt=vt,
                      bbl=bbl, hll=hll, ibl=ibl)
    dt = time.perf_counter() - t0
    print(f"\nRESULT: val_loss={vlo:.4f}  val_bpb={vbp:.4f}  time={dt:.1f}s")
    print(f"Expected: val_bpb approximately 1.1033 (Run #10c post-EMA)")
    delta = vbp - 1.1033
    print(f"Delta vs reported: {delta:+.4f}")
    if abs(delta) < 0.02:
        print("PASS: checkpoint loads and eval matches report within tolerance.")
        return 0
    else:
        print("WARN: BPB differs from report by more than 0.02 — investigate.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
