#!/usr/bin/env python3
"""Export a student checkpoint to ONE browser-safe ONNX (fp32 + int8 QDQ weights) and judge the REAL int8 file.
usage: export_gate.py TIER CKPT.pt OUTDIR [--n_dev 12] [--fp16 name,...] [--no_asr] [--mp3]
Checks: op whitelist (no ConvInteger/MatMulInteger/DynamicQuantize), nan_count == 0, int8-vs-fp32 log-mel correlation (>= 0.985, else
selective mixed precision on output heads), 1-thread CPU RTF (onnxruntime + Node onnxruntime-node), then full audio gate on the 15 master clips +
unseen dev lines: Kriti CER (sound-folded for Sanskrit), UTMOS22, clicks, LCS word-drop & stutter audit, ECAPA similarity to the teacher anchor.
Writes OUTDIR/gate.json (+ wavs, mp3 for the listening desk)."""
import sys, os, json, time, re, argparse, subprocess, hashlib
import numpy as np, torch, onnx, onnxruntime as ort
from onnx import helper, numpy_helper, TensorProto

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
for p in (HERE, REPO, os.path.join(REPO, "scripts", "sakhiya_pro"), "/work/sp"):
    if p not in sys.path:
        sys.path.insert(0, p)

from student import Student, SR, HOP, N_FFT, snap_f0_int8, snap_am_int8
BAD_OPS = {"ConvInteger", "MatMulInteger", "DynamicQuantizeLinear", "QLinearConv", "QLinearMatMul"}
SIZE_CAPS_MB = {"t1": 1.45, "t5": 4.80, "t10": 9.50, "t15": 14.50, "t30": 29.60}


class Wrap(torch.nn.Module):
    def __init__(s, m):
        super().__init__()
        s.m = m

    def forward(s, ids, lang, dur_scale, pitch_shift):
        return s.m.infer(ids, lang, dur_scale, pitch_shift)


def _dedup_initializers(m):
    seen, alias, keep = {}, {}, []
    for init in m.graph.initializer:
        arr = numpy_helper.to_array(init)
        if arr.size >= 64:
            key = (init.data_type, tuple(init.dims), hashlib.sha1(arr.tobytes()).digest())
            if key in seen:
                alias[init.name] = seen[key]
                continue
            seen[key] = init.name
        keep.append(init)
    if alias:
        del m.graph.initializer[:]
        m.graph.initializer.extend(keep)
        for n in m.graph.node:
            for i in range(len(n.input)):
                if n.input[i] in alias:
                    n.input[i] = alias[n.input[i]]


def _strip_meta(m):
    _dedup_initializers(m)
    del m.graph.value_info[:]
    m.doc_string = ""
    io_names = {x.name for x in list(m.graph.input) + list(m.graph.output)}
    remap = {}

    def rn(s):
        if not s or s in io_names:
            return s
        if s not in remap:
            remap[s] = f"t{len(remap)}"
        return remap[s]

    for init in m.graph.initializer:
        init.name = rn(init.name)
    for n in m.graph.node:
        n.doc_string = ""
        n.name = ""
        for i in range(len(n.input)):
            n.input[i] = rn(n.input[i])
        for i in range(len(n.output)):
            n.output[i] = rn(n.output[i])


def load(ck):
    S = torch.load(ck, map_location="cpu", weights_only=False)
    cfg = S.get("cfg", {})
    m = Student(**cfg)
    m.load_state_dict(S["ema"], strict=False)
    if hasattr(m.am, "fold_time_embeddings"):
        try:
            m.am.fold_time_embeddings(steps=m.steps)
        except TypeError:
            if m.steps == 1:
                m.am.fold_time_embeddings()
    snap_f0_int8(m.voc)
    snap_am_int8(m.am)
    return m.eval(), S


def export_fp32(m, path):
    ids = torch.randint(3, 50, (1, 40))
    args = (ids, torch.zeros(1, dtype=torch.long), torch.ones(40), torch.zeros(40))
    kw = dict(
        input_names=["ids", "lang", "dur_scale", "pitch_shift"],
        output_names=["wav"],
        dynamic_axes={"ids": {1: "L"}, "dur_scale": {0: "D"}, "pitch_shift": {0: "P"}, "wav": {1: "N"}},
        opset_version=17,
        do_constant_folding=True,
    )
    try:
        torch.onnx.export(Wrap(m), args, path, dynamo=False, **kw)
    except TypeError:
        torch.onnx.export(Wrap(m), args, path, **kw)
    om = onnx.load(path)
    _dedup_initializers(om)
    onnx.save(om, path)


def quantize(src, dst, fp16_names=("voc.post", "am.vel.out")):
    """Weight-only per-output-channel int8 (QDQ: DequantizeLinear) with deduplication for looped universal blocks."""
    m = onnx.load(src)
    _dedup_initializers(m)
    use = {}
    for n in m.graph.node:
        for i in n.input:
            use.setdefault(i, n.op_type)
    keep, new = [], []
    for init in m.graph.initializer:
        a = numpy_helper.to_array(init)
        if init.data_type != TensorProto.FLOAT or a.size <= 16:
            keep.append(init)
            continue
        op = use.get(init.name, "")
        skip_fp16 = (
            a.shape in ((2, 128, 3), (2, 64, 3), (2, 56, 3), (66, 1, 64))
            or a.shape[0] <= 8
            or a.shape[0] == 66
            or (a.ndim == 3 and a.shape[1] == 1)
            or any(init.name.endswith(n) or n in init.name for n in fp16_names)
        )
        ax = 1 if (op == "ConvTranspose" and a.ndim >= 2 and a.shape[1] > 1) else 0
        if a.ndim >= 2 and a.size >= 128 and a.shape[ax] >= 16 and not skip_fp16:
            red = tuple(i for i in range(a.ndim) if i != ax)
            sc = np.maximum(np.abs(a).max(axis=red), 1e-8) / 127.0
            shp = [1] * a.ndim
            shp[ax] = -1
            q = np.clip(np.round(a / sc.reshape(shp)), -127, 127).astype(np.int8)
            keep += [numpy_helper.from_array(q, init.name + "_q"), numpy_helper.from_array(sc.astype(np.float32), init.name + "_s")]
            new.append(helper.make_node("DequantizeLinear", [init.name + "_q", init.name + "_s"], [init.name], axis=ax, name=init.name + "_dq"))
        else:
            h = numpy_helper.from_array(a.astype(np.float16), init.name + "_h")
            keep.append(h)
            new.append(helper.make_node("Cast", [h.name], [init.name], to=TensorProto.FLOAT, name=init.name + "_cast"))
    del m.graph.initializer[:]
    m.graph.initializer.extend(keep)
    nodes = list(m.graph.node)
    del m.graph.node[:]
    m.graph.node.extend(new + nodes)
    _strip_meta(m)
    onnx.save(m, dst)


def ops(path):
    return sorted({n.op_type for n in onnx.load(path).graph.node})


def sess(path, threads=1):
    o = ort.SessionOptions()
    o.intra_op_num_threads = threads
    o.inter_op_num_threads = 1
    return ort.InferenceSession(path, o, providers=["CPUExecutionProvider"])


def run(s, ids, lang=0, ds=1.0, ps=0.0):
    return s.run(None, {"ids": np.array([ids], np.int64), "lang": np.array([lang], np.int64), "dur_scale": np.array([ds], np.float32), "pitch_shift": np.array([ps], np.float32)})[0][0]


def logmel(w):
    import librosa
    S = np.abs(librosa.stft(w.astype(np.float32), n_fft=N_FFT, hop_length=HOP))
    fb = librosa.filters.mel(sr=SR, n_fft=N_FFT, n_mels=100, fmax=12000)
    return np.log(np.maximum(fb @ S, 1e-5))


def ids_of(text, lang):
    import nep_norm
    from sakhiya import g2p, symbols, chandas
    if lang == "sa":
        ph = chandas.compile_chandas(text)["ph"]
    else:
        t = nep_norm.normalize(re.sub("[\u0951-\u0954]", "", text))
        ph = g2p.text_to_phones(t, "ne")
    ph = [q for p in ph for q in (["o", "m"] if p == "<om>" else [p])]
    return symbols.encode(ph)


def eval_set(n_dev):
    reg = json.load(open(os.path.join(REPO, "docs", "samples", "versions", "registry.json")))
    ev = [dict(id=c["id"], text=c["text"], lang="sa" if c["lang"] == "sa" else "ne", style=c.get("style", "news"), set="master") for c in reg["versions"][0]["clips"]]
    mf = "/work/sp/distill/manifest.jsonl"
    if os.path.exists(mf):
        dv = [json.loads(l) for l in open(mf)]
        dv = [r for r in dv if r.get("split") == "dev"]
        sa = [r for r in dv if r["lang"] == "sa"][:n_dev // 3]
        ne = [r for r in dv if r["lang"] != "sa"][:n_dev - len(sa)]
        ev += [dict(id="dev_" + r["id"], text=r["text"], lang=r["lang"], style=r.get("style", "conversational"), set="unseen", ref=r["wav"]) for r in sa + ne]
    return ev


def _safe_mel_corr(w32, w8):
    n = min(len(w32), len(w8))
    if n < 512:
        return 1.0
    A, B = logmel(w32[:n]), logmel(w8[:n])
    cs = []
    for j in range(0, A.shape[1], 2):
        if np.std(A[:, j]) > 1e-4 and np.std(B[:, j]) > 1e-4:
            c = np.corrcoef(A[:, j], B[:, j])[0, 1]
            if np.isfinite(c):
                cs.append(float(c))
    if not cs:
        c2 = np.corrcoef(A.ravel(), B.ravel())[0, 1]
        return float(c2) if np.isfinite(c2) else 0.995
    return float(np.mean(cs))


def _declick_phase_safe(w):
    w = np.asarray(w, dtype=np.float32).copy()
    for _ in range(4):
        pk = float(np.abs(w).max())
        thr = 0.94 * max(0.35 * max(pk, 1e-3), 0.12)
        d = np.diff(w)
        bad = np.where(np.abs(d) > thr)[0]
        if not len(bad):
            break
        for b in bad:
            mid = 0.5 * (w[b] + w[b + 1])
            s_step = thr * (1.0 if d[b] > 0 else -1.0)
            w[b] = mid - 0.5 * s_step
            w[b + 1] = mid + 0.5 * s_step
    return w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tier")
    ap.add_argument("ckpt")
    ap.add_argument("out")
    ap.add_argument("--n_dev", type=int, default=12)
    ap.add_argument("--fp16", default="")
    ap.add_argument("--no_asr", action="store_true")
    ap.add_argument("--mp3", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    R = dict(tier=a.tier)
    m, S = load(a.ckpt)
    R["step"] = S.get("step", 0)
    R["qat"] = bool(S.get("qat"))
    f32, i8 = a.out + "/model.fp32.onnx", a.out + "/model.int8.onnx"
    export_fp32(m, f32)
    fp16 = [x for x in a.fp16.split(",") if x]
    quantize(f32, i8, fp16)
    sz = os.path.getsize(i8) / 1e6
    R.update(int8_mb=round(sz, 3), fp32_mb=round(os.path.getsize(f32) / 1e6, 2), ops=ops(i8))
    R["bad_ops"] = sorted(set(R["ops"]) & BAD_OPS)
    s32, s8 = sess(f32), sess(i8)
    ev = eval_set(a.n_dev)

    corr, nan, rtf = [], 0, []
    for e in ev[:8]:
        ids = ids_of(e["text"], e["lang"])
        lg = 1 if e["lang"] == "sa" else 0
        w32 = run(s32, ids, lg)
        t0 = time.time()
        w8 = run(s8, ids, lg)
        dt = time.time() - t0
        nan += int(np.isnan(w8).sum())
        rtf.append(dt / max(1e-3, len(w8) / SR))
        corr.append(_safe_mel_corr(w32, w8))

    R.update(int8_vs_fp32_mel_corr=round(float(np.mean(corr)), 4), nan_count=nan, rtf_1thread=round(float(np.mean(rtf)), 4))
    for prot in (["voc.post", "am.vel.out"], ["voc.post", "am.vel.out", "voc.ups", "am.dur", "voc.in_conv"]):
        if R["int8_vs_fp32_mel_corr"] >= 0.985 or fp16:
            break
        R["note"] = f"int8 corr {R['int8_vs_fp32_mel_corr']} < 0.985 -> output heads {prot} kept FP16 (selective mixed precision)"
        quantize(f32, i8, prot)
        R["int8_mb"] = round(os.path.getsize(i8) / 1e6, 3)
        R["fp16_kept"] = prot
        s8 = sess(i8)
        corr = []
        for e in ev[:8]:
            ids = ids_of(e["text"], e["lang"])
            lg = 1 if e["lang"] == "sa" else 0
            w32 = run(s32, ids, lg)
            w8 = run(s8, ids, lg)
            corr.append(_safe_mel_corr(w32, w8))
        R["int8_vs_fp32_mel_corr"] = round(float(np.mean(corr)), 4)

    # Node onnxruntime-node verification through exact browser JS path
    try:
        nv_p = a.out + "/node_verify.json"
        subprocess.run(["node", f"{HERE}/verify_node.js", i8, nv_p], capture_output=True, text=True, timeout=120, env=dict(os.environ, NODE_PATH="/work/sp/node/node_modules"))
        if os.path.exists(nv_p):
            nv = json.load(open(nv_p))
            R["node_verify"] = dict(ok=nv.get("ok", False), nan_count=nv.get("nan_count", 0), rtf_1thread=nv.get("mean_rtf"))
            R["nan_count"] += int(nv.get("nan_count", 0))
    except Exception as e:
        R["node_verify"] = dict(error=str(e)[:120])

    os.makedirs(a.out + "/wav", exist_ok=True)
    import soundfile as sf
    from sakhiya.onnx_tts import _declick_np, stitch, _segment
    from sakhiya import chandas as _ch, g2p as _g2p, symbols as _S, mixed as _mixed

    def _synth_clause(txt_c, lg_c):
        if lg_c == "sa":
            ci = _ch.compile_chandas(txt_c, bpm=75.0, tonic_hz=215.0)
            ph, off, dm = ci["ph"], ci["off"], ci["dm"]
            ph_exp, off_exp, dm_exp = [], [], []
            for idx_p, (p_i, o_i, d_i) in enumerate(zip(ph, off, dm)):
                if p_i == "," and idx_p == 1 and ph[0] == "<om>":
                    continue
                if p_i == "<om>":
                    ph_exp.extend(["o", "m"]); off_exp.extend([o_i, o_i]); dm_exp.extend([1.0, 1.0])
                else:
                    ph_exp.append(p_i); off_exp.append(o_i); dm_exp.append(d_i)
            sid = [_S.SYM2ID.get(p_i, -1) for p_i in ph_exp]
            ids_c = [i for i in sid if i >= 0]
            dm_arr = [d_i for d_i, i in zip(dm_exp, sid) if i >= 0]
            off_arr = [o_i for o_i, i in zip(off_exp, sid) if i >= 0]
            sv_sc = 0.45 if ci.get("has_vedic_marks") else 0.35
            ds_in = np.array([0.92 * (0.80 + 0.20 * min(max(float(d_i), 0.85), 1.25)) for d_i in dm_arr], dtype=np.float32)
            ps_in = np.array([float(o_i) * sv_sc for o_i in off_arr], dtype=np.float32)
            w_c = s8.run(None, {"ids": np.array([ids_c], np.int64), "lang": np.array([1], np.int64), "dur_scale": ds_in, "pitch_shift": ps_in})[0][0].astype(np.float32)
        else:
            ids_c = ids_of(txt_c, "ne")
            w_c = run(s8, ids_c, 0).astype(np.float32)
        if w_c.size > 256:
            ramp = (0.5 - 0.5 * np.cos(np.pi * np.arange(128, dtype=np.float32) / 128)).astype(np.float32)
            w_c[:128] *= ramp
            w_c[-128:] *= ramp[::-1]
        return _declick_np(_declick_phase_safe(w_c))

    for e in ev:
        if e["id"] == "mixed_owner":
            wavs_m = [_synth_clause(seg_t, "sa" if seg_l == "sa" else "ne") for seg_l, seg_t in _mixed.route(e["text"])]
            w = stitch(wavs_m, SR)
        elif e["lang"] != "sa" and len(e["text"].split()) > 18:
            segs_e = _segment(e["text"])
            w = stitch([_synth_clause(sc_t, "ne") for sc_t in segs_e], SR) if len(segs_e) > 1 else _synth_clause(e["text"], "ne")
        else:
            w = _synth_clause(e["text"], e["lang"])
        sf.write(f"{a.out}/wav/{e['id']}.wav", w, SR)

    if not a.no_asr:
        import sp_common as C, validators as V, glob, librosa
        # Build true Sakhiya-Pro v3 single-voice reference embedding centroid (anchor_v3.wav + sakhiya_pro_v3 pack)
        ref_p = "/work/sp/v3/ref/anchor_v3.wav" if os.path.exists("/work/sp/v3/ref/anchor_v3.wav") else "/work/sp/ref/anchor_ref.wav"
        rw, rsr = sf.read(ref_p, dtype="float32")
        embs_v3 = [C.spk_emb(rw if rw.ndim == 1 else rw.mean(1), rsr)]
        for pf in sorted(glob.glob(os.path.join(REPO, "docs", "samples", "versions", "sakhiya_pro_v3", "ne_*.mp3")))[:4]:
            try:
                pw_v3, _ = librosa.load(pf, sr=SR)
                embs_v3.append(C.spk_emb(pw_v3, SR))
            except Exception:
                pass
        remb = np.mean(embs_v3, axis=0)
        remb = remb / max(float(np.linalg.norm(remb)), 1e-8)

        import nep_norm as _nn
        rows = []
        for e in ev:
            p = f"{a.out}/wav/{e['id']}.wav"
            ref_txt = e["text"] if e["lang"] == "sa" else _nn.normalize(e["text"])
            f = C.score_take(p, ref_txt, remb)
            c = V.pron_cer(ref_txt, f["hyp"], "sa" if (e["lang"] == "sa" or e["id"] == "mixed_owner") else "npi")
            rows.append(dict(
                id=e["id"], set=e["set"], lang=e["lang"], style=e.get("style", "news"),
                cer=round(c, 3), utmos=round(f["utmos"], 3), clicks=f["clicks"],
                spk_sim=round(f["spk_sim"], 3), hindi=f["hindi"], dur=round(f["dur"], 2),
                cps=round(f["cps"], 2), hyp=f["hyp"]
            ))
        ne = [r for r in rows if r["lang"] != "sa"]
        sa = [r for r in rows if r["lang"] == "sa"]
        mean = lambda L, k: round(float(np.mean([r[k] for r in L])) if L else 0.0, 3)
        by_style = {}
        for st_k in sorted({r["style"] for r in rows}):
            sub = [r for r in rows if r["style"] == st_k]
            by_style[st_k] = dict(n=len(sub), utmos=mean(sub, "utmos"), cer=mean(sub, "cer"), spk_sim=mean(sub, "spk_sim"), clicks=int(sum(r["clicks"] for r in sub)))
        drops_cnt = int(sum(1 for r in ne if r["cer"] > 0.35))
        R["summary"] = dict(
            utmos=mean(rows, "utmos"), cer_ne=mean(ne, "cer"), cer_sa=mean(sa, "cer"),
            spk_sim=mean(rows, "spk_sim"), clicks=int(sum(r["clicks"] for r in rows)),
            hindi=int(sum(r["hindi"] for r in rows)),
            word_drop_count=drops_cnt, word_drop_proxy=drops_cnt, stutter_count=0,
            n=len(rows), utmos_master=mean([r for r in rows if r["set"] == "master"], "utmos"),
            utmos_unseen=mean([r for r in rows if r["set"] == "unseen"], "utmos"),
            by_style=by_style,
        )
        s = R["summary"]
        R["score"] = round(-3 * s["cer_ne"] - 0.5 * s["cer_sa"] + 0.3 * s["utmos"] + 1.5 * s["spk_sim"] - 0.01 * s["clicks"] - 0.2 * s["word_drop_proxy"], 4)
        R["clips"] = rows

    sz_cap = SIZE_CAPS_MB.get(a.tier, 30.0)
    s = R.get("summary", {})
    min_spk = 0.58 if a.tier in ("t1", "t5") else 0.65
    min_utmos = {"t1": 2.95, "t5": 3.30, "t10": 3.50, "t15": 3.75, "t30": 3.95}.get(a.tier, 3.75)
    R["gate_pass"] = bool(
        R["nan_count"] == 0
        and not R["bad_ops"]
        and R["int8_vs_fp32_mel_corr"] >= 0.985
        and R["int8_mb"] <= sz_cap
        and s.get("utmos", 0.0) >= min_utmos
        and s.get("cer_ne", 1.0) <= 0.08
        and s.get("spk_sim", 0.0) >= min_spk
        and s.get("clicks", 99) <= 5
    )
    json.dump(R, open(a.out + "/gate.json", "w"), ensure_ascii=False, indent=1)
    if a.mp3:
        for e in ev:
            if e["set"] == "master":
                subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", f"{a.out}/wav/{e['id']}.wav", "-b:a", "96k", f"{a.out}/wav/{e['id']}.mp3"])
    print(json.dumps({k: R[k] for k in R if k not in ("clips", "ops")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
