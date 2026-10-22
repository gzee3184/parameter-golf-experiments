#!/usr/bin/env python3
"""GPTQ ablation: find which change caused 14.47MB -> 16.25MB compression regression.

Uses saved checkpoint from final_model.pt to avoid retraining.
Tests each GPTQ change independently:
  A: Original (double damping, stale perm, no recurrence)
  B: Remove pre-damping only
  C: Fix stale perm only
  D: Add recurrence only
  E: All three changes
  F: Original + real-data calibration (instead of gen seqs)
"""
import os, sys, io, math, time, glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import brotli

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

device = torch.device("cuda", 0)
torch.cuda.set_device(device)

# ── Shortcuts ──
_f32 = torch.float32
_f16 = torch.float16
_bf16 = torch.bfloat16
_i8 = torch.int8
_S = None
_P = nn.Parameter
_PL = nn.ParameterList
_ta = torch.arange
_tt = torch.tensor
_to = torch.ones
_tz = torch.zeros
_te = torch.empty
_tc = torch.clamp
_tr = torch.round
_ng = torch.no_grad
_im = torch.inference_mode

def _leaky05(x): return F.leaky_relu(x, 0.05)
_AF = _leaky05

class RN(nn.Module):
    def forward(self, x): return F.rms_norm(x, (x.size(-1),))

class CL(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

def _fa3(q, k, v, causal=True):
    q2 = q.transpose(1, 2); k2 = k.transpose(1, 2); v2 = v.transpose(1, 2)
    o = F.scaled_dot_product_attention(q2, k2, v2, is_causal=causal, enable_gqa=True)
    return o.transpose(1, 2)

class Rotary(nn.Module):
    def __init__(self, dim, base=10000.0, tsl=1024, rd=0):
        super().__init__()
        self.dim = dim; self.base = base; self.tsl = tsl
        self.rd = rd if rd > 0 else dim
        ief = 1.0 / (base ** (_ta(0, self.rd, 2, dtype=_f32) / self.rd))
        self.register_buffer("ief", ief, persistent=False)
        self.register_buffer("_cos", _te(0), persistent=False)
        self.register_buffer("_sin", _te(0), persistent=False)
    def forward(self, sl, device, dtype):
        if sl > self._cos.size(0):
            t = _ta(sl, device=device, dtype=_f32)
            f = torch.outer(t, self.ief.to(device))
            fc = torch.polar(_to(f.shape, device=device), f)
            self._cos = fc.real.to(dtype); self._sin = fc.imag.to(dtype)
        return self._cos[:sl], self._sin[:sl]

def _are(x, cos, sin, rd):
    d = x.shape[-1]
    if rd < d:
        xr, xp = x[..., :rd], x[..., rd:]
        c = cos[:x.shape[1]].unsqueeze(0).unsqueeze(2)
        s = sin[:x.shape[1]].unsqueeze(0).unsqueeze(2)
        x1, x2 = xr[..., :rd//2], xr[..., rd//2:]
        rot = torch.cat((-x2, x1), dim=-1)
        return torch.cat((xr * c + rot * s, xp), dim=-1)
    c = cos[:x.shape[1]].unsqueeze(0).unsqueeze(2)
    s = sin[:x.shape[1]].unsqueeze(0).unsqueeze(2)
    x1, x2 = x[..., :d//2], x[..., d//2:]
    return torch.cat((x1 * c - x2 * s, x2 * c + x1 * s), dim=-1)

class VE(nn.Module):
    def __init__(self, vs, vd, kvd):
        super().__init__()
        self.embed = nn.Embedding(vs, vd)
        self.proj = CL(vd, kvd, bias=False)
        self.scale = _P(_tz((), dtype=_f32))
    def forward(self, ids):
        return self.proj(_AF(self.embed(ids))) * self.scale.to(dtype=self.proj.weight.dtype)

class _HA(nn.Module):
    def __init__(self, dim, nh, nkv, rb, qgi, rd=0, tsl=2048):
        super().__init__()
        self.nh, self.nkv = nh, nkv
        self.hdm = dim // nh; kvd = nkv * self.hdm
        self.c_q = CL(dim, dim, bias=False)
        self.c_k = CL(dim, kvd, bias=False)
        self.c_v = CL(dim, kvd, bias=False)
        self.proj = CL(dim, dim, bias=False)
        self.q_gain = _P(torch.full((nh,), qgi, dtype=_f32))
        self.rd = rd
        self.rotary = Rotary(self.hdm, base=rb, tsl=tsl, rd=rd)
        self.uxs = False
    def forward(self, x, ve=None):
        bsz, sq, dim = x.shape
        q = self.c_q(x).reshape(bsz, sq, self.nh, self.hdm)
        k = self.c_k(x).reshape(bsz, sq, self.nkv, self.hdm)
        v = self.c_v(x)
        if ve is not None: v = v + ve
        v = v.reshape(bsz, sq, self.nkv, self.hdm)
        q = F.rms_norm(q, (q.size(-1),)); k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(sq, x.device, q.dtype)
        q = _are(q, cos, sin, self.rd); k = _are(k, cos, sin, self.rd)
        q = q * self.q_gain.to(dtype=q.dtype)[None, None, :, None]
        y = _fa3(q, k, v, causal=True)
        return self.proj(y.reshape(bsz, sq, dim))

class _HM(nn.Module):
    def __init__(self, dim, mlm):
        super().__init__()
        self.fc = CL(dim, int(mlm * dim), bias=False)
        self.proj = CL(int(mlm * dim), dim, bias=False)
        self.act_fn = _AF
    def forward(self, x): return self.proj(self.act_fn(self.fc(x)))

class _HB(nn.Module):
    def __init__(self, dim, nh, nkv, mlm, rb, qgi, lxi=0, lns=False, rd=0, tsl=2048):
        super().__init__()
        self.attn_norm = RN(); self.mlp_norm = RN()
        self.attn = _HA(dim, nh, nkv, rb, qgi, rd=rd, tsl=tsl)
        self.mlp = _HM(dim, mlm)
        self.attn_scale = _P(_to(dim, dtype=_f32))
        self.mlp_scale = _P(_to(dim, dtype=_f32))
        self.resid_mix = _P(torch.stack((_to(dim), _tz(dim))).float())
        self.lsf = 1.0 / math.sqrt(lxi + 1) if lns else 1.0
        self.dsa = False
    def forward(self, x, x0, ve=None):
        mix = self.resid_mix.to(dtype=x.dtype)
        xi = mix[0][_S] * x + mix[1][_S] * x0; xo = xi
        if not self.dsa:
            aout = self.attn(self.attn_norm(xi) * self.lsf, ve=ve)
            xo = xo + self.attn_scale.to(dtype=xi.dtype)[_S] * aout
        return xo + self.mlp_scale.to(dtype=xo.dtype)[_S] * self.mlp(self.mlp_norm(xo) * self.lsf)

class _HG(nn.Module):
    def __init__(self, vs, nl, md, nh, nkv, mlm, te, lsc, rb, qgi,
                 xln=0, rd=0, lns=False, sgsk=False, ven=False, vd=128,
                 vel=None, psl=-1, dl0a=False, tsl=2048, pllm=False,
                 rcl=None, rnl=2, use_recurrence=False):
        super().__init__()
        self.te = te; self.lsc = lsc; self.sgsk = sgsk
        self.nl = nl; self.pllm = pllm
        self.tok_emb = nn.Embedding(vs, md)
        self.rcl = sorted(rcl) if rcl else []; self.rnl = rnl
        vl = [(i, False) for i in range(nl)]
        if use_recurrence and self.rcl:
            ins = False; vl_r = []
            for i in range(nl):
                vl_r.append((i, False))
                if not ins and i == self.rcl[-1]:
                    for _ in range(rnl - 1):
                        for rl in self.rcl: vl_r.append((rl, True))
                    ins = True
            vl = vl_r
        self._vl = vl; nv = len(vl)
        self.nel = nv // 2; self.ndl = nv - self.nel
        pnel = nl // 2; pndl = nl - pnel
        self.nsw = min(pnel, pndl)
        self.skip_weights = _P(_to(self.nsw, md, dtype=_f32))
        self.skip_gates = _P(_tz(self.nsw, md, dtype=_f32)) if sgsk else None
        self.blocks = nn.ModuleList([_HB(md, nh, nkv, mlm, rb, qgi, lxi=i, lns=lns, rd=rd, tsl=tsl) for i in range(nl)])
        if xln > 0:
            for i in range(max(0, nl - xln), nl): self.blocks[i].attn.uxs = True
        self.final_norm = RN()
        self.lm_head = None if te else CL(md, vs, bias=False)
        kvd = nkv * (md // nh)
        self.veli = sorted(vel) if ven and vel else []
        if self.veli:
            self.ves = VE(vs, vd, kvd)
            self.vels = _PL([_P(_to(1, dtype=_f32)) for _ in self.veli])
        else: self.ves = None; self.vels = _PL()
        self.psl = psl; self.pllm = pllm
        if 0 < psl < nl:
            np_ = nl - psl
            self.lm = _P(torch.full((np_,), 0.5, dtype=_f32)) if pllm else _P(_tt(0.5, dtype=_f32))
        else: self.lm = None
        if dl0a and len(self.blocks) > 0: self.blocks[0].dsa = True

    def _get_ve(self, pi, ids, vc):
        if self.ves is None or pi not in self.veli: return None
        if 've' not in vc: vc['ve'] = self.ves(ids)
        vb = vc['ve']; vi = self.veli.index(pi)
        return vb * self.vels[vi].to(dtype=vb.dtype)

    def forward(self, ii, tgi):
        x = self.tok_emb(ii); x = F.rms_norm(x, (x.size(-1),)); x0 = x
        skips = []; vc = {}; pt = self.psl if self.lm is not None else self.nl + 1
        ip = False; l0 = None; l1 = None; si = 0
        for vi in range(self.nel):
            pi, ir = self._vl[vi]
            x = self.blocks[pi](x, x0, ve=self._get_ve(pi, ii, vc))
            if not ir: skips.append(x)
        for vi in range(self.ndl):
            pi, ir = self._vl[self.nel + vi]
            if skips and si < self.nsw and not ir:
                t = l0 if ip else x
                ss = self.skip_weights[si].to(dtype=t.dtype)[_S] * skips.pop()
                if self.skip_gates is not None:
                    g = torch.sigmoid(self.skip_gates[si].to(dtype=t.dtype))[_S]
                    t = torch.lerp(ss, t, g)
                else: t = t + ss
                if ip: l0 = t
                else: x = t
                si += 1
            if pi >= pt and not ip: l0 = x; l1 = x; ip = True
            if ip:
                blk = self.blocks[pi]; ve = self._get_ve(pi, ii, vc)
                if self.pllm:
                    mix = blk.resid_mix.to(dtype=x.dtype)
                    ai = mix[0][_S] * x + mix[1][_S] * x0
                    ao = blk.attn(blk.attn_norm(ai) * blk.lsf, ve=ve)
                    ac = blk.attn_scale.to(dtype=x.dtype)[_S] * ao
                    mc = blk.mlp_scale.to(dtype=x.dtype)[_S] * blk.mlp(blk.mlp_norm(x) * blk.lsf)
                    m = torch.sigmoid(self.lm[pi - pt]).to(dtype=x.dtype)
                    x = x + m * ac + (1 - m) * mc
                else:
                    mix = blk.resid_mix.to(dtype=l0.dtype)
                    ai = mix[0][_S] * l0 + mix[1][_S] * x0
                    ao = blk.attn(blk.attn_norm(ai) * blk.lsf, ve=ve)
                    l0 = ai + blk.attn_scale.to(dtype=ai.dtype)[_S] * ao
                    l1 = l1 + blk.mlp_scale.to(dtype=l1.dtype)[_S] * blk.mlp(blk.mlp_norm(l1) * blk.lsf)
            else:
                x = self.blocks[pi](x, x0, ve=self._get_ve(pi, ii, vc))
        if ip and not self.pllm:
            m = self.lm.to(dtype=l0.dtype); x = m * l0 + (1 - m) * l1
        x = self.final_norm(x); xf = x.reshape(-1, x.size(-1)); tgt = tgi.reshape(-1)
        lp = F.linear(xf, self.tok_emb.weight) if self.te else self.lm_head(xf)
        return F.cross_entropy((self.lsc * torch.tanh(lp / self.lsc)).float(), tgt, reduction="mean")

    def fwl(self, ii):
        x = self.tok_emb(ii); x = F.rms_norm(x, (x.size(-1),)); x0 = x
        skips = []; vc = {}; pt = self.psl if self.lm is not None else self.nl + 1
        ip = False; l0 = None; l1 = None; si = 0
        for vi in range(self.nel):
            pi, ir = self._vl[vi]
            x = self.blocks[pi](x, x0, ve=self._get_ve(pi, ii, vc))
            if not ir: skips.append(x)
        for vi in range(self.ndl):
            pi, ir = self._vl[self.nel + vi]
            if skips and si < self.nsw and not ir:
                t = l0 if ip else x
                ss = self.skip_weights[si].to(dtype=t.dtype)[_S] * skips.pop()
                if self.skip_gates is not None:
                    g = torch.sigmoid(self.skip_gates[si].to(dtype=t.dtype))[_S]
                    t = torch.lerp(ss, t, g)
                else: t = t + ss
                if ip: l0 = t
                else: x = t
                si += 1
            if pi >= pt and not ip: l0 = x; l1 = x; ip = True
            if ip:
                blk = self.blocks[pi]; ve = self._get_ve(pi, ii, vc)
                if self.pllm:
                    mix = blk.resid_mix.to(dtype=x.dtype)
                    ai = mix[0][_S] * x + mix[1][_S] * x0
                    ao = blk.attn(blk.attn_norm(ai) * blk.lsf, ve=ve)
                    ac = blk.attn_scale.to(dtype=x.dtype)[_S] * ao
                    mc = blk.mlp_scale.to(dtype=x.dtype)[_S] * blk.mlp(blk.mlp_norm(x) * blk.lsf)
                    m = torch.sigmoid(self.lm[pi - pt]).to(dtype=x.dtype)
                    x = x + m * ac + (1 - m) * mc
                else:
                    mix = blk.resid_mix.to(dtype=l0.dtype)
                    ai = mix[0][_S] * l0 + mix[1][_S] * x0
                    ao = blk.attn(blk.attn_norm(ai) * blk.lsf, ve=ve)
                    l0 = ai + blk.attn_scale.to(dtype=ai.dtype)[_S] * ao
                    l1 = l1 + blk.mlp_scale.to(dtype=l1.dtype)[_S] * blk.mlp(blk.mlp_norm(l1) * blk.lsf)
            else:
                x = self.blocks[pi](x, x0, ve=self._get_ve(pi, ii, vc))
        if ip and not self.pllm:
            m = self.lm.to(dtype=l0.dtype); x = m * l0 + (1 - m) * l1
        x = self.final_norm(x); xf = x.reshape(-1, x.size(-1))
        lp = F.linear(xf, self.tok_emb.weight) if self.te else self.lm_head(xf)
        return self.lsc * torch.tanh(lp / self.lsc)


# ── Quantization functions with toggleable flags ──

def _qpr(t, cr=31):
    t32 = t.float()
    if t32.ndim == 2:
        be = float('inf'); bq = bs = None
        for pct in [0.9990, 0.9995, 0.9999, 0.99999, 1.0]:
            rc = torch.quantile(t32.abs(), pct, dim=1) if pct < 1.0 else t32.abs().amax(dim=1)
            s = (rc / cr).clamp_min(1.0 / cr).to(_f16); sf = s.float()
            q = _tc(_tr(t32 / sf[:, None]), -cr, cr).to(_i8)
            err = (t32 - q.float() * sf[:, None]).pow(2).mean().item()
            if err < be: bq, bs, be = q, s, err
        return bq, bs
    amax = t32.abs().max().item()
    scale = _tt(amax / cr if amax > 0 else 1.0, dtype=_f16)
    return _tc(_tr(t32 / scale.float()), -cr, cr).to(_i8), scale


def _qgq(weight, hessian, cr=31, bsz=128, use_stale_perm=True):
    t32 = weight.float()
    if t32.ndim != 2: return _qpr(t32, cr)
    rows, cols = t32.shape
    H = hessian.float().clone()
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    dm = torch.mean(torch.diag(H))
    Hinv = None; perm = None

    for attempt, df in enumerate([0.01, 0.05, 0.1, 0.5, 1.0]):
        try:
            Htry = H.clone()
            damp = df * dm
            Htry[_ta(cols), _ta(cols)] += damp
            if use_stale_perm:
                if perm is None:
                    perm = torch.argsort(torch.diag(Htry), descending=True)
            else:
                perm = torch.argsort(torch.diag(Htry), descending=True)
            L = torch.linalg.cholesky(Htry[perm][:, perm])
            Hinv = torch.linalg.cholesky(torch.cholesky_inverse(L), upper=True)
            break
        except torch._C._LinAlgError:
            if attempt == 4: return _qpr(t32, cr)

    ip = torch.argsort(perm)
    W = t32[:, perm].clone(); W[:, dead[perm]] = 0
    bq, bsc, be = None, None, float('inf')
    for pct in [0.9990, 0.9995, 0.9999, 0.99999, 1.0]:
        rc = torch.quantile(t32.abs(), pct, dim=1) if pct < 1.0 else t32.abs().amax(dim=1)
        s = (rc / cr).clamp_min(1.0 / cr).to(_f16); sf = s.float()
        Q = torch.zeros_like(W, dtype=_i8); W_work = W.clone()
        for i1 in range(0, cols, bsz):
            i2 = min(i1 + bsz, cols); count = i2 - i1
            W1 = W_work[:, i1:i2].clone(); Q1 = _tz(rows, count, dtype=_i8)
            Err1 = _tz(rows, count); Hinv1 = Hinv[i1:i2, i1:i2]
            for i in range(count):
                w = W1[:, i]; d = Hinv1[i, i]
                q = _tc(_tr(w / sf), -cr, cr).to(_i8)
                Q1[:, i] = q; err = (w - q.float() * sf) / d
                W1[:, i:] -= err.unsqueeze(1) * Hinv1[i, i:].unsqueeze(0)
                Err1[:, i] = err
            Q[:, i1:i2] = Q1
            if i2 < cols: W_work[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]
        mse = (W - Q.float() * sf[:, None]).pow(2).mean().item()
        if mse < be: bq, bsc, be = Q, s, mse
    return bq[:, ip], bsc


def collect_hessians(hm, seqs, device, use_pre_damping=True):
    H = {}; hooks = []
    for nm, mod in hm.named_modules():
        if isinstance(mod, CL):
            pn = nm + ".weight"; c = mod.weight.shape[1]
            H[pn] = _tz(c, c, dtype=_f32, device='cpu')
            def mk(p):
                def fn(m, inp, out):
                    x = inp[0].detach().float()
                    if x.ndim == 3: x = x.reshape(-1, x.shape[-1])
                    H[p] += (x.T @ x).cpu()
                return fn
            hooks.append(mod.register_forward_hook(mk(pn)))
    hm.eval()
    with _im(), torch.autocast(device_type="cuda", dtype=_bf16):
        for s in seqs:
            hm(s[:, :-1].to(device), s[:, 1:].to(device))
    for h in hooks: h.remove()
    for n in H:
        H[n] /= len(seqs)
        if use_pre_damping:
            d = 0.01 * torch.diag(H[n]).mean().clamp_min(1e-6)
            H[n] += d * torch.eye(H[n].shape[0], device=H[n].device, dtype=H[n].dtype)
    return H


def _ubs(sd, n):
    out = {}
    for nm, t in sd.items():
        if nm == "qo_bank":
            for i in range(n): out[f"blocks.{i}.attn.c_q.weight"] = t[i]; out[f"blocks.{i}.attn.proj.weight"] = t[n + i]
        elif nm == "kv_bank":
            for i in range(n): out[f"blocks.{i}.attn.c_k.weight"] = t[i]; out[f"blocks.{i}.attn.c_v.weight"] = t[n + i]
        elif nm == "mlp_up_bank":
            for i in range(n): out[f"blocks.{i}.mlp.fc.weight"] = t[i]
        elif nm == "mlp_down_bank":
            for i in range(n): out[f"blocks.{i}.mlp.proj.weight"] = t[i]
        else: out[nm] = t
    return out

_CTNP = tuple("attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,q_gain,skip_weight,skip_weights,skip_gates,vels,ves.scale,lm,rps".split(","))

def _cp(name):
    if "tok_emb" in name or "lm_head" in name: return "embed"
    if ".mlp." in name: return "mlp"
    if ".attn." in name or (".proj." in name and ".mlp." not in name): return "attn"
    return "other"

def _qft(t):
    t32 = t.float()
    if t32.ndim == 2:
        ca = torch.quantile(t32.abs(), 99.99984 / 100.0, dim=1)
        s = (ca / 127).clamp_min(1.0 / 127).to(_f16); sf = s.float()
        q = _tc(_tr(t32 / sf[:, None]), -127, 127).to(_i8)
        return q, s
    amax = t32.abs().max().item()
    scale = _tt(amax / 127 if amax > 0 else 1.0, dtype=_f16)
    return _tc(_tr(t32 / scale.float()), -127, 127).to(_i8), scale


def quantize_model(sdi, hs=None, bm=6, ba=5, be=8, use_stale_perm=True):
    i6c = {"mlp", "attn", "embed"}
    result = {}; meta = {}
    for name, tensor in sdi.items():
        t = tensor.detach().cpu().contiguous()
        cat = _cp(name)
        if not t.is_floating_point() or t.numel() <= 65536:
            result[name] = t.to(_f16) if t.is_floating_point() else t
            meta[name] = "pt"; continue
        if any(p in name for p in _CTNP):
            result[name] = t.float(); meta[name] = "ptc"; continue
        if cat in i6c and t.ndim >= 1:
            bits = bm if cat == "mlp" else (be if cat == "embed" else ba)
            cr_map = {5: 15, 6: 31, 8: 127}
            cr = cr_map[bits]
            H = hs.get(name) if hs else None
            if H is not None and t.ndim == 2:
                q, s = _qgq(t, hessian=H, cr=cr, use_stale_perm=use_stale_perm)
            else:
                q, s = _qpr(t, cr=cr)
            result[name + ".q"] = q; result[name + ".scale"] = s
            meta[name] = {"type": "int" + str(bits)}
        else:
            q, s = _qft(t)
            result[name + ".q"] = q; result[name + ".scale"] = s
            meta[name] = {"type": "int8"}
    return result, meta


def compute_pruning_and_size(qr, qm, code_bytes=67924, budget_limit=16000000):
    oi = []
    for name in qm:
        info = qm[name]
        if not isinstance(info, dict): continue
        qk = name + ".q"; sk = name + ".scale"
        if qk not in qr or sk not in qr: continue
        q, s = qr[qk], qr[sk]
        if s.ndim > 0 and (q.abs() == 1).any():
            om = (q.abs() == 1)
            ri = _ta(q.shape[0]).unsqueeze(1).expand_as(q)[om]
            fi = _ta(q.numel()).reshape(q.shape)[om]
            for f, e in zip(fi.tolist(), (s.float()[ri].pow(2)).tolist()):
                oi.append((qk, f, e))

    def _bsh(data):
        arr = np.frombuffer(data, dtype=np.uint8)
        pad = (4 - len(arr) % 4) % 4
        if pad: arr = np.concatenate([arr, np.zeros(pad, dtype=np.uint8)])
        return arr.reshape(-1, 4).T.tobytes() + pad.to_bytes(1, 'little')

    def _compress(data, q=11):
        data = _bsh(data)
        return brotli.compress(data, quality=q)

    tbyt = budget_limit - code_bytes

    def _tp(n, q=11):
        tmp = {k: v.clone() for k, v in qr.items()}
        for i in range(min(n, len(oi))):
            tmp[oi[i][0]].view(-1)[oi[i][1]] = 0
        buf = io.BytesIO(); torch.save({"w": tmp, "m": qm}, buf)
        return len(_compress(buf.getvalue(), q=q)), tmp

    oi.sort(key=lambda x: x[2])
    no_sz, _ = _tp(0, q=6)

    prune_count = 0
    if no_sz <= tbyt:
        prune_pct = 0.0
        final_sz, _ = _tp(0)
    else:
        full_sz, _ = _tp(len(oi), q=6)
        lo, hi = 0, len(oi)
        while lo < hi:
            mid = (lo + hi) // 2
            sz, _ = _tp(mid, q=6)
            if sz <= tbyt: hi = mid
            else: lo = mid + 1
        prune_pct = 100.0 * lo / len(oi) if len(oi) > 0 else 0.0
        prune_count = lo
        final_sz, _ = _tp(lo)

    return {
        "unpruned_mb": no_sz / 1048576,
        "budget_mb": tbyt / 1048576,
        "prune_pct": prune_pct,
        "prune_count": prune_count,
        "final_compressed_bytes": final_sz,
        "n_candidates": len(oi),
    }


def generate_calib_seqs(model, device, nseq=64, sln=2048, vs=4096, tpr=0.8, seed=42):
    model_dev = next(model.parameters()).device
    rng = torch.Generator(device=model_dev); rng.manual_seed(seed)
    out = []
    with _im(), torch.autocast(device_type="cuda", dtype=_bf16):
        for bs0 in range(0, nseq, 8):
            bs = min(8, nseq - bs0)
            toks = torch.randint(0, vs, (bs, 1), device=model_dev, generator=rng)
            for _ in range(sln - 1):
                logits = model.fwl(toks)[:, -1, :] / tpr
                next_tok = torch.multinomial(torch.softmax(logits, dim=-1), 1, generator=rng)
                toks = torch.cat([toks, next_tok], dim=1)
            out.extend(toks[i:i+1] for i in range(bs))
    return out


def load_real_data_seqs(n_seqs, sln=2048, seed=42):
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "../parameter-golf/data/datasets/fineweb10B_sp4096")
    cfiles = sorted(glob.glob(data_path + "/*"))
    if not cfiles:
        data_path = "./data/datasets/fineweb10B_sp4096"
        cfiles = sorted(glob.glob(data_path + "/*"))
    if not cfiles:
        raise RuntimeError(f"No data files found")
    print(f"  Loading from {cfiles[0]} ({len(cfiles)} shards)")
    cmm = np.memmap(cfiles[0], dtype=np.uint16, mode='r')
    cn = len(cmm)
    rng_c = torch.Generator(); rng_c.manual_seed(seed)
    seqs = []
    for _ in range(n_seqs):
        ci = torch.randint(0, cn - sln - 1, (1,), generator=rng_c).item()
        seqs.append(torch.as_tensor(np.array(cmm[ci:ci + sln + 1], dtype=np.int64)).unsqueeze(0))
    return seqs


# ── Also test: NO GPTQ at all (pure RTN) ──
def quantize_model_rtn(sdi, bm=6, ba=5, be=8):
    """Quantize with RTN only (no Hessians)."""
    return quantize_model(sdi, hs=None, bm=bm, ba=ba, be=be)


# ── Config ──
NL = 11; MD = 512; NH = 8; NKV = 4; MLM = 4; VS = 4096
RB = 22000; QGI = 5.0; LSC = 20.0
XLN = 11; RD = 0; LNS = False; SGSK = True
VEN = True; VD = 128; VEL = [9, 10]
PSL = 7; DL0A = True; TSL = 2048; PLLM = False
RCL = [3, 4, 5]; RNL = 2

CKPT = "./parameter-golf/final_model.pt"


def run_test(name, use_recurrence=False, use_pre_damping=True, use_stale_perm=True,
             use_real_data=False, n_calib_seqs=64, rtn_only=False):
    print(f"\n{'='*60}")
    print(f"Test {name}")
    if rtn_only:
        print(f"  RTN only (no GPTQ)")
    else:
        print(f"  recurrence={use_recurrence}, pre_damping={use_pre_damping}, stale_perm={use_stale_perm}")
        print(f"  real_data={use_real_data}, n_calib_seqs={n_calib_seqs}")
    print(f"{'='*60}")

    sd = torch.load(CKPT, map_location="cpu", weights_only=False)
    us = _ubs(sd, NL)

    if rtn_only:
        t1 = time.time()
        qr, qm = quantize_model_rtn(us, bm=6, ba=5, be=8)
        print(f"  RTN quantized in {time.time()-t1:.1f}s")
    else:
        t0 = time.time()
        hgm = _HG(
            vs=VS, nl=NL, md=MD, nh=NH, nkv=NKV, mlm=MLM, te=True, lsc=LSC,
            rb=RB, qgi=QGI, xln=XLN, rd=RD, lns=LNS, sgsk=SGSK,
            ven=VEN, vd=VD, vel=VEL, psl=PSL, dl0a=DL0A, tsl=TSL, pllm=PLLM,
            rcl=RCL, rnl=RNL, use_recurrence=use_recurrence
        ).to(device).bfloat16()

        for m in hgm.modules():
            if isinstance(m, CL): m.float()
        with _ng():
            for nm, p in hgm.named_parameters():
                if (p.ndim < 2 or any(s in nm for s in _CTNP)) and p.dtype != _f32:
                    p.data = p.data.float()

        hgm.load_state_dict(us, strict=False)
        print(f"  VL schedule: {len(hgm._vl)} steps")

        if use_real_data:
            seqs = load_real_data_seqs(n_calib_seqs, sln=TSL, seed=42)
            print(f"  Loaded {len(seqs)} real-data seqs")
        else:
            seqs = generate_calib_seqs(hgm, device, nseq=n_calib_seqs, sln=TSL, vs=VS, seed=42)
            print(f"  Generated {len(seqs)} seqs")

        hs = collect_hessians(hgm, seqs, device, use_pre_damping=use_pre_damping)
        print(f"  Collected {len(hs)} Hessians in {time.time()-t0:.1f}s")
        del hgm; torch.cuda.empty_cache()

        t1 = time.time()
        qr, qm = quantize_model(us, hs=hs, bm=6, ba=5, be=8, use_stale_perm=use_stale_perm)
        print(f"  Quantized in {time.time()-t1:.1f}s")

    t2 = time.time()
    stats = compute_pruning_and_size(qr, qm)
    print(f"  Compression in {time.time()-t2:.1f}s")
    print(f"  >> unpruned={stats['unpruned_mb']:.2f}MB  budget={stats['budget_mb']:.2f}MB  "
          f"prune={stats['prune_pct']:.1f}%  cands={stats['n_candidates']}")
    return stats


if __name__ == "__main__":
    print("GPTQ Ablation Test")
    print(f"Checkpoint: {CKPT}")
    print(f"Device: {device}")

    results = {}

    # 0: RTN baseline (no GPTQ at all)
    results['0'] = run_test("0: RTN only (no GPTQ)", rtn_only=True)

    # A: Original (double damping, stale perm, no recurrence, gen seqs)
    results['A'] = run_test("A: Original (baseline)",
                            use_recurrence=False, use_pre_damping=True,
                            use_stale_perm=True, use_real_data=False, n_calib_seqs=64)

    # B: Remove pre-damping only
    results['B'] = run_test("B: No pre-damping only",
                            use_recurrence=False, use_pre_damping=False,
                            use_stale_perm=True, use_real_data=False, n_calib_seqs=64)

    # C: Fix stale perm only
    results['C'] = run_test("C: Fresh perm only",
                            use_recurrence=False, use_pre_damping=True,
                            use_stale_perm=False, use_real_data=False, n_calib_seqs=64)

    # D: Add recurrence only
    results['D'] = run_test("D: Recurrence only",
                            use_recurrence=True, use_pre_damping=True,
                            use_stale_perm=True, use_real_data=False, n_calib_seqs=64)

    # E: All three changes + gen seqs
    results['E'] = run_test("E: All 3 fixes + gen seqs",
                            use_recurrence=True, use_pre_damping=False,
                            use_stale_perm=False, use_real_data=False, n_calib_seqs=64)

    # F: Original + real data (64 seqs)
    results['F'] = run_test("F: Original + real data 64",
                            use_recurrence=False, use_pre_damping=True,
                            use_stale_perm=True, use_real_data=True, n_calib_seqs=64)

    # G: Original + real data (24K seqs)
    results['G'] = run_test("G: Original + real data 24K",
                            use_recurrence=False, use_pre_damping=True,
                            use_stale_perm=True, use_real_data=True, n_calib_seqs=24512)

    # H: All fixes + real data (24K seqs)
    results['H'] = run_test("H: All fixes + real data 24K",
                            use_recurrence=True, use_pre_damping=False,
                            use_stale_perm=False, use_real_data=True, n_calib_seqs=24512)

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    header = f"{'Test':<42} {'Unpruned':>10} {'Budget':>10} {'Prune%':>8} {'Cands':>8}"
    print(header)
    print("-" * 80)
    labels = {
        '0': 'RTN only (no GPTQ)',
        'A': 'Original (baseline)',
        'B': 'No pre-damping only',
        'C': 'Fresh perm only',
        'D': 'Recurrence only',
        'E': 'All 3 fixes + gen seqs',
        'F': 'Orig + real data 64',
        'G': 'Orig + real data 24K',
        'H': 'All fixes + real 24K',
    }
    for k in ['0', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']:
        if k not in results: continue
        v = results[k]
        print(f"{k}: {labels[k]:<39} {v['unpruned_mb']:>8.2f}MB {v['budget_mb']:>8.2f}MB "
              f"{v['prune_pct']:>7.1f}% {v['n_candidates']:>7d}")
