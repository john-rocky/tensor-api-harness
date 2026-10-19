#!/usr/bin/env python3
"""Decoder block 0 of Qwen3-0.6B (prefill) computed again in numpy, to show
which arithmetic the CPU int8 forms of `xnnpack_int8_fc.md` ran.

usage: python3 xnnpack_int8_fc_block0.py <model.safetensors> <ref_dir> \
           <reference/prompts.json> <dump_root> <out.txt>

<ref_dir>/<prompt>/emb.f32 and block_00.f32: the PyTorch reference dumps
  (`reference/dump_reference.py`).
<dump_root>/<form>_blocks/<prompt>/block_00.f32: `qwen3_main --dump_blocks`
  for the forms cpu_int8 (A1), cpu_int8_dequant_fp32 (B1) and
  cpu_int8_builtin_fc (A1-off); a missing form is skipped.

Block 0 is computed in float32 from the reference emb rows (the runner's emb
dump is byte-identical to them) in four ways:
  fp32      checkpoint weights, fp32 FC inputs (the reference arithmetic)
  w8        int8 per-channel weights (code * scale, the loader's nearest
            integer codes), fp32 FC inputs (the B forms)
  w8_qp8    w8 + every FC input quantized per row the way KleidiAI's
            kai_run_lhs_quant_pack_qai8dxp_f32 does it (asymmetric int8,
            scale 255 / (max(0, max) - min(0, min)), zero point by rintf,
            values to the nearest integer with halves away from zero),
            integer sums, the scales applied after
  w8_sym8   w8 + every FC input quantized per row symmetrically (scale
            max|x| / 127, halves away from zero; TFLite's
            SymmetricQuantizeFloats), integer sums
and each is compared (max|d|, cosine) with the reference and the dumps.
"""
import json
import math
import struct
import sys

import numpy as np

F32 = np.float32
EMB, NH, NKV, HD, HID = 1024, 16, 8, 128, 3072
EPS = F32(1e-6)
BASE = 1000000.0
MAX_SEQ = 64
MASK_FLOOR = F32(-30000.0)


def read_header(path):
    f = open(path, "rb")
    n = struct.unpack("<Q", f.read(8))[0]
    return f, json.loads(f.read(n)), 8 + n


def tensor(f, header, base, name):
    meta = header[name]
    assert meta["dtype"] == "BF16", meta["dtype"]
    a, b = meta["data_offsets"]
    f.seek(base + a)
    raw = np.frombuffer(f.read(b - a), dtype="<u2").astype(np.uint32) << 16
    return raw.view(F32).reshape(meta["shape"])


def norm_weight(w):
    # The loader adds 1.0 to every norm weight and the harness subtracts it
    # again, both in fp32.
    return (w + F32(1.0)) - F32(1.0)


def round_half_away(x):
    x = np.asarray(x, dtype=np.float64)
    return np.where(x >= 0, np.floor(x + 0.5), -np.floor(-x + 0.5))


def quantize_weight(w):
    """[out, in] fp32 -> (codes int, scales fp32), as qwen3_weights.cc."""
    amax = np.max(np.abs(w), axis=1)
    scale = np.where(amax > 0, amax / F32(127.0), F32(1.0)).astype(F32)
    v = (w / scale[:, None]).astype(F32)  # fp32 division
    q = np.clip(round_half_away(v), -127, 127).astype(np.int64)
    return q, scale


def lhs_qp8(x):
    """Per-row asymmetric int8 as kai_lhs_quant_pack_qai8dxp_f32.c:77-259."""
    rows = []
    for r in x.astype(F32):
        mn, mx = F32(r.min()), F32(r.max())
        rmin, rmax = min(F32(0.0), mn), max(F32(0.0), mx)
        scale = F32(1.0) if rmin == rmax else F32(F32(255.0) / F32(rmax - rmin))
        recip = F32(F32(1.0) / scale) if scale else F32(0.0)
        dmin, dmax = F32(rmin * scale), F32(rmax * scale)
        zp = F32(-128.0) - dmin if F32(-128.0 + dmin) + F32(127.0 + dmax) > 0 else F32(127.0) - dmax
        zp = min(max(zp, F32(-128.0)), F32(127.0))
        nzp = int(np.rint(zp))
        v = round_half_away((r * scale).astype(F32))
        v = np.clip(np.clip(v, -32768, 32767) + nzp, -128, 127).astype(np.int64)
        rows.append((v - nzp, recip))
    return rows


def lhs_sym8(x):
    """Per-row symmetric int8 (TFLite SymmetricQuantizeFloats)."""
    rows = []
    for r in x.astype(F32):
        amax = F32(np.max(np.abs(r)))
        if amax == 0:
            rows.append((np.zeros(r.shape, np.int64), F32(1.0)))
            continue
        scale = F32(amax / F32(127.0))
        v = np.clip(round_half_away((r * (F32(127.0) / amax)).astype(F32)), -127, 127).astype(np.int64)
        rows.append((v, scale))
    return rows


class Weights:
    def __init__(self, f, header, base):
        p = "model.layers.0."
        g = lambda n: tensor(f, header, base, p + n)
        self.in_norm = norm_weight(g("input_layernorm.weight"))
        self.post_norm = norm_weight(g("post_attention_layernorm.weight"))
        self.q_norm = norm_weight(g("self_attn.q_norm.weight"))
        self.k_norm = norm_weight(g("self_attn.k_norm.weight"))
        self.fc = {n: g(n + ".weight") for n in (
            "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
            "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")}
        self.q8 = {n: quantize_weight(w) for n, w in self.fc.items()}
        # Codes as float64: the int8 x int8 sums below stay exact (|sum| is
        # at most 3072 * 127 * 255 < 2**53).
        self.q8f = {n: q.astype(np.float64) for n, (q, _) in self.q8.items()}


def fc(x, W, name, mode):
    """x [rows, in] fp32 -> [rows, out] fp32."""
    if mode == "fp32":
        return (x.astype(np.float64) @ W.fc[name].astype(np.float64).T).astype(F32)
    q, s = W.q8[name]
    if mode == "w8":
        wd = (q.astype(F32) * s[:, None]).astype(F32)
        return (x.astype(np.float64) @ wd.astype(np.float64).T).astype(F32)
    rows = lhs_qp8(x) if mode == "w8_qp8" else lhs_sym8(x)
    qf = W.q8f[name]
    out = []
    for v, a_scale in rows:
        acc = qf @ v.astype(np.float64)  # exact sums of the int8 products
        out.append((acc * float(a_scale) * s.astype(np.float64)).astype(F32))
    return np.stack(out)


def rms(x, w):
    x = x.astype(F32)
    ms = np.mean((x * x).astype(F32), axis=-1, keepdims=True, dtype=F32)
    return ((x * (F32(1.0) / np.sqrt(ms + EPS)).astype(F32)) * w).astype(F32)


def rope_tables(seq):
    half = HD // 2
    cos = np.zeros((seq, HD), F32)
    sin = np.zeros((seq, HD), F32)
    for s in range(seq):
        for i in range(half):
            inv = F32(1.0) / F32(math.pow(F32(BASE), F32(2 * i) / F32(HD)))
            ang = F32(F32(s) * inv)
            cos[s, i] = cos[s, half + i] = F32(math.cos(ang))
            sin[s, i] = sin[s, half + i] = F32(math.sin(ang))
    return cos, sin


def rope(x, cos, sin):  # x [heads, seq, hd]
    half = HD // 2
    rot = np.concatenate([-x[..., half:], x[..., :half]], axis=-1)
    return (x * cos + rot * sin).astype(F32)


def block0(emb, W, mode):
    seq = emb.shape[0]
    a_in = rms(emb, W.in_norm)
    q = fc(a_in, W, "self_attn.q_proj", mode).reshape(seq, NH, HD).transpose(1, 0, 2)
    k = fc(a_in, W, "self_attn.k_proj", mode).reshape(seq, NKV, HD).transpose(1, 0, 2)
    v = fc(a_in, W, "self_attn.v_proj", mode).reshape(seq, NKV, HD).transpose(1, 0, 2)
    q, k = rms(q, W.q_norm), rms(k, W.k_norm)
    cos, sin = rope_tables(seq)
    q, k = rope(q, cos, sin), rope(k, cos, sin)
    kc = np.zeros((NKV, MAX_SEQ, HD), F32)
    vc = np.zeros((NKV, MAX_SEQ, HD), F32)
    kc[:, :seq], vc[:, :seq] = k, v
    mask = np.full((seq, MAX_SEQ), MASK_FLOOR, F32)
    for s in range(seq):
        mask[s, :s + 1] = 0
    out = np.zeros((NH, seq, HD), F32)
    scale = F32(1.0) / F32(math.sqrt(HD))
    for h in range(NH):
        g = h // (NH // NKV)
        sc = ((q[h].astype(np.float64) @ kc[g].astype(np.float64).T).astype(F32) * scale + mask).astype(F32)
        sc = sc - sc.max(axis=-1, keepdims=True)
        e = np.exp(sc.astype(np.float64))
        p = (e / e.sum(axis=-1, keepdims=True)).astype(F32)
        out[h] = (p.astype(np.float64) @ vc[g].astype(np.float64)).astype(F32)
    attn = out.transpose(1, 0, 2).reshape(seq, NH * HD)
    h1 = (emb + fc(attn, W, "self_attn.o_proj", mode)).astype(F32)
    f_in = rms(h1, W.post_norm)
    gate = fc(f_in, W, "mlp.gate_proj", mode)
    up = fc(f_in, W, "mlp.up_proj", mode)
    act = ((gate * (F32(1.0) / (F32(1.0) + np.exp(-gate)))).astype(F32) * up).astype(F32)
    return (h1 + fc(act, W, "mlp.down_proj", mode)).astype(F32)


def cmp(a, b):
    a, b = a.astype(np.float64).ravel(), b.astype(np.float64).ravel()
    return float(np.max(np.abs(a - b))), float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def main():
    st, ref_root, prompts_json, got_root, out_path = sys.argv[1:6]
    f, header, base = read_header(st)
    W = Weights(f, header, base)
    prompts = [p["id"] for p in json.load(open(prompts_json))["prompts"]]
    forms = {"A1": "cpu_int8", "B1": "cpu_int8_dequant_fp32", "A1-off": "cpu_int8_builtin_fc"}
    lines = []
    for p in prompts:
        emb = np.fromfile(f"{ref_root}/{p}/emb.f32", dtype="<f4").reshape(-1, EMB)
        ref = np.fromfile(f"{ref_root}/{p}/block_00.f32", dtype="<f4").reshape(-1, EMB)
        dumps = {}
        for label, key in forms.items():
            try:
                dumps[label] = np.fromfile(f"{got_root}/{key}_blocks/{p}/block_00.f32", dtype="<f4").reshape(-1, EMB)
            except OSError:
                pass
        sims = {m: block0(emb, W, m) for m in ("fp32", "w8", "w8_qp8", "w8_sym8")}
        for m, x in sims.items():
            row = [f"{p} sim {m:<8}", "vs ref %.4e %.8f" % cmp(x, ref)]
            for label, d in dumps.items():
                row.append(f"vs {label} " + "%.4e %.8f" % cmp(x, d))
            lines.append("  ".join(row))
        for label, d in dumps.items():
            lines.append(f"{p} dump {label:<6} vs ref " + "%.4e %.8f" % cmp(d, ref))
    open(out_path, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
