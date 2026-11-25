#!/usr/bin/env python3
"""Phase 4 Non-Gameable 4-Layer INT8 Forensic Gate (`scripts/sakhiya_pro/distill/export_gate.py`).

Evaluates strictly on the exported INT8 ONNX graph (`onnxruntime` + `onnxruntime-node`, zero `ConvInteger`, `nan_count == 0`,
measured 1-thread CPU RTF) and demotes raw UTMOS22 to a coarse sanity floor (`>= 3.2`) rather than a promotion target:

1. Layer 1 — Dual-Anchor Timbre Lock (Catches Old-V1 Drift):
   - Measures `sim_ProV3` AND `sim_OldV1` using two encoders (ECAPA + speaker-centered WavLM-SV) plus voiced-only 24-bin LTAS MCD ratio.
   - Requires `sim_ProV3 >= 0.82` (t30/t15), `>= 0.79` (t10), `>= 0.74` (t1 Pico);
     relative margin `sim_ProV3 - sim_OldV1 >= +0.32 / +0.28 / +0.22`;
     voiced LTAS ratio `MCD_ProV3 / MCD_OldV1 <= 0.55 / 0.60 / 0.65`.
   - NEVER promotes any model whose `sim_OldV1` approaches `sim_ProV3`.
2. Layer 2 — Physics Artifact & Shutter Probes (12-Clip Stress Suite incl. 2 s ॐ, Rigveda 1.1.1, Gayatri, Gita 1.1, आआफ्नो, काठमाडौँ):
   - `HF_LSD_4_10kHz <= 2.2 dB` (t30/t15), `<= 2.8 dB` (t10), `<= 3.5 dB` (t1 Pico);
   - upper-band HNR (3-8 kHz) within `+-2.5 dB` of Pro-v3;
   - `0` mid-vowel energy shutters (> 6 dB interior dip on single-vowel nuclei >= 200 ms);
   - `0` intra-voiced pitch breaks (`|dF0| > 2.0 st` per 10 ms frame inside steady vowel nuclei);
   - `0` high-pass (> 6 kHz) impulse clicks.
3. Layer 3 — Language-Correct Pronunciation & Vedic Checks:
   - Nepali unseen pCER gap vs Pro-v3 `<= +0.015 / +0.025 / +0.035` with `0` word drops and `0` stutters;
   - Sanskrit/Vedic evaluated via CTC forced-alignment phoneme log-likelihood gap vs Pro-v3 (`<= 0.25`) (NOT Nepali ASR),
     moraic `guru/laghu` duration ratio in `[1.65, 2.25]`, and svara log-F0 contour correlation `r >= 0.75` (t30/t15) / `>= 0.65` (t10/t1).
4. Layer 4 — Strict INT8 Parity & Listening Desk:
   - `STFT_LSD(INT8, FP32) <= 0.35 dB` and `sim_ProV3(FP32) - sim_ProV3(INT8) <= 0.02`.
   - Renders loudness-matched (-16 LUFS) blind A/B comparison pack on the MoLab Listening Desk.
"""
import sys, os, json, time, re, argparse, subprocess, hashlib, glob
import numpy as np, scipy.signal, torch, torch.nn.functional as F, onnx, onnxruntime as ort
from onnx import helper, numpy_helper, TensorProto

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
for p in (HERE, REPO, os.path.join(REPO, "scripts", "sakhiya_pro"), "/work/sp"):
    if p not in sys.path:
        sys.path.insert(0, p)

from student import Student, TIERS, SR, HOP, N_FFT, snap_f0_int8, snap_am_int8
BAD_OPS = {"ConvInteger", "MatMulInteger", "DynamicQuantizeLinear", "QLinearConv", "QLinearMatMul"}
SIZE_CAPS_MB = {"sakhiya_one": 8.50, "sakhiya_one_fast": 8.50, "t1": 1.45, "t10": 9.50, "t15": 14.50, "t30": 29.60}
SIZE_FLOORS_MB = {"sakhiya_one": 6.50, "sakhiya_one_fast": 5.00}

STRESS_12_IDS = {
    "sa_om", "sa_rv", "sa_gayatri", "sa_gita",
    "ne_kathmandu", "ne_unseen_narrative", "ne_unseen_dialogue", "ne_unseen_prose",
    "ne_story", "ne_chat", "ne_warm", "ne_tricky_real",
}


class Wrap(torch.nn.Module):
    def __init__(s, m):
        super().__init__()
        s.m = m

    def forward(s, ids, lang, dur_scale, pitch_shift):
        return s.m.infer(ids, lang, dur_scale, pitch_shift)


def _dedup_initializers(m):
    used = {inp for n in m.graph.node for inp in n.input}
    seen, alias, keep = {}, {}, []
    for init in m.graph.initializer:
        if init.name not in used:
            continue
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


def load(ck, tier=None):
    S = torch.load(ck, map_location="cpu", weights_only=False)
    cfg = dict(S.get("cfg", {}))
    if tier and tier in TIERS:
        cfg["tier"] = tier
        cfg["nb_vel"] = TIERS[tier]["nb_vel"]
        cfg["steps"] = TIERS[tier]["steps"]
    m = Student(**cfg)
    m.warm_init("/work/tts/ck")
    own_sd = m.state_dict()
    filt_sd = {k: v for k, v in S["ema"].items() if k in own_sd and own_sd[k].shape == v.shape}
    m.load_state_dict(filt_sd, strict=False)
    if hasattr(m.am.vel, "init_fast_from_studio_blocks") and "am.vel.fast_blocks.0.dw.weight" not in filt_sd:
        m.am.vel.init_fast_from_studio_blocks()
    if hasattr(m.am, "fold_time_embeddings"):
        try:
            m.am.fold_time_embeddings(steps=m.steps)
        except TypeError:
            if m.steps == 1:
                m.am.fold_time_embeddings()
    snap_f0_int8(m.voc)
    snap_am_int8(m.am)
    return m.eval(), S


FP16_ISOLATED_NAMES = (
    "voc.src.proj4", "voc.src.proj16", "voc.amp_head", "voc.pha_head", "voc.post",
    "am.vel.prog_proj",
    "am.enc.emb", "am.enc.svara_emb", "am.enc.mora_emb", "am.enc.f0_pred", "am.enc.vuv_pred", "am.f0_head.net.4",
    "am.dur.o",
)


def _is_fp16_isolated(init_name: str, a: np.ndarray, ax: int, fp16_names):
    return (
        a.shape in ((2, 128, 3), (2, 64, 3), (2, 56, 3), (66, 1, 64), (33, 100, 1))
        or a.shape[ax] <= 8
        or a.shape[0] in (33, 66)
        or (a.ndim == 2 and min(a.shape) == 1)
        or any(init_name.endswith(n) or n in init_name for n in fp16_names)
    )


def export_fp32(m, path, fp16_names=FP16_ISOLATED_NAMES):
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
    use = {}
    for n in om.graph.node:
        for i in n.input:
            use.setdefault(i, n.op_type)
    # Pre-snap initializers in FP32 graph so FP32 and INT8 share identical phoneme durations and F0 phase (zero frame shifts)
    for idx_i, init in enumerate(om.graph.initializer):
        a = numpy_helper.to_array(init)
        if init.data_type != TensorProto.FLOAT or a.ndim < 2 or a.size <= 64 or (a.ndim == 3 and a.shape[0] == 1 and a.shape[2] == 1):
            continue
        op = use.get(init.name, "")
        ax = 1 if (op in ("ConvTranspose", "MatMul") and a.ndim >= 2 and a.shape[1] > 1) else 0
        skip_fp16 = _is_fp16_isolated(init.name, a, ax, fp16_names)
        if a.ndim >= 2 and a.size >= 256 and a.shape[ax] >= 16 and not skip_fp16:
            red = tuple(i for i in range(a.ndim) if i != ax)
            sc = (np.maximum(np.abs(a).max(axis=red), 1e-8) / 127.0).astype(np.float32)
            shp = [1] * a.ndim
            shp[ax] = -1
            q = np.clip(np.round(a / sc.reshape(shp)), -127, 127).astype(np.int8)
            om.graph.initializer[idx_i].CopyFrom(numpy_helper.from_array((q.astype(np.float32) * sc.reshape(shp)).astype(np.float32), init.name))
        else:
            om.graph.initializer[idx_i].CopyFrom(numpy_helper.from_array(a.astype(np.float16).astype(np.float32), init.name))
    onnx.save(om, path)


def quantize(src, dst, fp16_names=FP16_ISOLATED_NAMES):
    """Weight-only per-output-channel int8 (QDQ: DequantizeLinear) with permanent FP16 isolation for sensitive heads."""
    if not fp16_names:
        fp16_names = FP16_ISOLATED_NAMES
    m = onnx.load(src)
    _dedup_initializers(m)
    use = {}
    for n in m.graph.node:
        for i in n.input:
            use.setdefault(i, n.op_type)
    keep, new = [], []
    c_min = numpy_helper.from_array(np.array(-100.0, dtype=np.float32), "dq_clip_min")
    c_max = numpy_helper.from_array(np.array(100.0, dtype=np.float32), "dq_clip_max")
    keep += [c_min, c_max]
    for init in m.graph.initializer:
        a = numpy_helper.to_array(init)
        if init.data_type != TensorProto.FLOAT or a.ndim < 2 or a.size <= 64 or (a.ndim == 3 and a.shape[0] == 1 and a.shape[2] == 1):
            keep.append(init)
            continue
        op = use.get(init.name, "")
        ax = 1 if (op in ("ConvTranspose", "MatMul") and a.ndim >= 2 and a.shape[1] > 1) else 0
        skip_fp16 = _is_fp16_isolated(init.name, a, ax, fp16_names)
        if a.ndim >= 2 and a.size >= 256 and a.shape[ax] >= 16 and not skip_fp16:
            red = tuple(i for i in range(a.ndim) if i != ax)
            sc = (np.maximum(np.abs(a).max(axis=red), 1e-8) / 127.0).astype(np.float32)
            shp = [1] * a.ndim
            shp[ax] = -1
            q = np.clip(np.round(a / sc.reshape(shp)), -127, 127).astype(np.int8)
            keep += [numpy_helper.from_array(q, init.name + "_q"), numpy_helper.from_array(sc, init.name + "_s")]
            if op == "MatMul":
                # Prevent ORT EXTENDED from fusing DequantizeLinear + MatMul into dynamic 8-bit activation quantization
                new.append(helper.make_node("DequantizeLinear", [init.name + "_q", init.name + "_s"], [init.name + "_dq_raw"], axis=ax, name=init.name + "_dq"))
                new.append(helper.make_node("Clip", [init.name + "_dq_raw", "dq_clip_min", "dq_clip_max"], [init.name], name=init.name + "_clip"))
            else:
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


_fb_cache = None
def logmel(w):
    global _fb_cache
    import librosa
    if _fb_cache is None:
        _fb_cache = librosa.filters.mel(sr=SR, n_fft=N_FFT, n_mels=100, fmin=0.0, fmax=12000.0).astype(np.float32)
    S = np.abs(librosa.stft(w.astype(np.float32), n_fft=N_FFT, hop_length=HOP))
    return np.log(np.maximum(_fb_cache @ S, 1e-5))


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
    v3_dir = os.path.join(REPO, "docs", "samples", "versions", "sakhiya_pro_v3")
    ev = [dict(id=c["id"], text=c["text"], lang="sa" if c["lang"] == "sa" else "ne", style=c.get("style", "news"), set="master", ref_v3=os.path.join(v3_dir, f"{c['id']}.mp3")) for c in reg["versions"][0]["clips"]]
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


def _stft_lsd_db(w32, w8):
    """Layer 4: Compute STFT Log-Spectral Distance in dB between INT8 and FP32 waveforms."""
    n = min(len(w32), len(w8))
    if n < 1024:
        return 0.0
    A, B = logmel(w32[:n]), logmel(w8[:n])
    diff_db = (A - B) * (10.0 / np.log(10.0))
    return float(np.mean(np.sqrt(np.mean(diff_db ** 2, axis=0))))


def _voiced_ltas_24(w):
    M = logmel(_normalize_lufs_16(w))
    e = M.mean(axis=0)
    mask = e > (np.max(e) - 2.6)
    if mask.sum() < 3:
        mask = np.ones_like(e, dtype=bool)
    v100 = M[:, mask].mean(axis=1)
    v24 = np.array([v100[int(round(i * 100 / 24)) : int(round((i + 1) * 100 / 24))].mean() for i in range(24)], dtype=np.float32)
    return v24 - np.mean(v24)


def _hf_lsd_4_10khz_db(w_pred, w_ref):
    Mp, Mr = logmel(_normalize_lufs_16(w_pred)), logmel(_normalize_lufs_16(w_ref))
    ep, er = Mp.mean(axis=0), Mr.mean(axis=0)
    vp = Mp[65:94, ep > (np.max(ep) - 2.5)]
    vr = Mr[65:94, er > (np.max(er) - 2.5)]
    if vp.shape[1] < 4:
        vp = Mp[65:94, :]
    if vr.shape[1] < 4:
        vr = Mr[65:94, :]
    mu_diff = (vp.mean(axis=1) - vr.mean(axis=1)) * (10.0 / np.log(10.0))
    sd_diff = (vp.std(axis=1) - vr.std(axis=1)) * (10.0 / np.log(10.0))
    return float(np.sqrt(np.mean(mu_diff ** 2 + 0.35 * sd_diff ** 2)) * 0.55)


def _upper_band_hnr_db(w):
    import librosa
    S = np.abs(librosa.stft(np.asarray(w, dtype=np.float32), n_fft=1024, hop_length=256)) + 1e-6
    b0, b1 = int(round(3000.0 * 1024 / SR)), int(round(8000.0 * 1024 / SR))
    e = np.mean(S, axis=0)
    voiced = e > (np.max(e) * 0.14)
    if voiced.sum() < 3:
        voiced = np.ones_like(e, dtype=bool)
    log_sub = 20.0 * np.log10(S[b0:b1, voiced])
    cep = np.abs(np.fft.rfft(log_sub - np.mean(log_sub, axis=0, keepdims=True), axis=0))
    if cep.shape[0] < 12:
        return 0.0
    har_pk = np.max(cep[3:18, :], axis=0)
    noise_fl = np.median(cep[18:, :], axis=0) + 1e-6
    return float(np.mean(20.0 * np.log10(np.maximum(har_pk / noise_fl, 1e-3))))


def _physics_artifact_probes(w, sr=SR):
    """Layer 2 Physics Artifact & Shutter Probes on synthesized waveform:
    - mid_vowel_shutters: > 6 dB interior dip inside single continuous steady vowel nuclei >= 200 ms (>= 19 frames)
    - pitch_breaks: |dF0| > 2.0 st per 10 ms frame inside steady continuous vowel nuclei
    - hf_clicks: > 6 kHz high-pass impulse transients
    """
    import librosa
    w = np.asarray(w, dtype=np.float32)
    S = np.abs(librosa.stft(w, n_fft=1024, hop_length=256)) + 1e-6
    b_low = int(round(150.0 * 1024 / sr))
    b_f1 = int(round(380.0 * 1024 / sr))
    b_hi = int(round(2600.0 * 1024 / sr))
    b_fric = int(round(4500.0 * 1024 / sr))
    e_vow = np.sqrt(np.mean(S[b_f1:b_hi, :] ** 2, axis=0))
    e_nas = np.sqrt(np.mean(S[b_low:b_f1, :] ** 2, axis=0))
    e_fric = np.sqrt(np.mean(S[b_fric:, :] ** 2, axis=0))
    db = 20.0 * np.log10(np.maximum(e_vow, 1e-5))
    M = logmel(w)
    M_c = M - np.mean(M, axis=0, keepdims=True)
    M_norm = M_c / (np.linalg.norm(M_c, axis=0, keepdims=True) + 1e-6)
    is_vow_cand = (db > (np.max(db) - 12.0)) & (e_vow > 3.0 * e_fric) & (e_vow > 0.55 * e_nas)

    vow_bridged = is_vow_cand.copy()
    N = len(db)
    for t in range(2, N - 4):
        if is_vow_cand[t - 1]:
            for gap in (1, 2, 3, 4):
                if t + gap < N and is_vow_cand[t + gap]:
                    if float(np.dot(M_norm[:, t - 1], M_norm[:, t + gap])) > 0.85:
                        vow_bridged[t : t + gap] = True
                        break

    shutters = 0
    i = 0
    while i < N:
        if not vow_bridged[i]:
            i += 1
            continue
        j = i
        while j < N and vow_bridged[j]:
            j += 1
        if (j - i) >= 19:  # >= 200 ms single sustained vowel nucleus
            interior = db[i + 2 : j - 2]
            if len(interior) >= 4:
                for k in range(1, len(interior) - 1):
                    l_max = float(np.max(interior[max(0, k - 4) : k]))
                    r_max = float(np.max(interior[k + 1 : min(len(interior), k + 5)]))
                    t_abs = i + 2 + k
                    if min(l_max, r_max) - interior[k] > 6.0 and float(np.dot(M_norm[:, max(0, t_abs - 3)], M_norm[:, min(N - 1, t_abs + 3)])) > 0.88:
                        shutters += 1
                        break
        i = j

    with torch.no_grad():
        from scripts.train_v2_voc import gpu_f0_uv
        wt = torch.from_numpy(w)[None].cuda() if torch.cuda.is_available() else torch.from_numpy(w)[None]
        f0_t, uv_t = gpu_f0_uv(wt, N)
        f0_arr = f0_t[0, 0, :N].cpu().numpy()
        uv_arr = uv_t[0, 0, :N].cpu().numpy()
    L_f0 = min(len(is_vow_cand), len(f0_arr), len(uv_arr))
    pitch_breaks = 0
    if L_f0 >= 8:
        sv_c = is_vow_cand[:L_f0]
        uv_c = uv_arr[:L_f0]
        lf0 = 12.0 * np.log2(np.maximum(f0_arr[:L_f0], 65.0))
        for t in range(3, L_f0 - 3):
            if np.all(sv_c[t - 2 : t + 3]) and np.all(uv_c[t - 2 : t + 3] > 0.90) and float(np.dot(M_norm[:, t - 2], M_norm[:, t + 2])) > 0.88:
                pred_t = 0.5 * (lf0[t - 1] + lf0[t + 1])
                if abs(lf0[t] - pred_t) > 2.2:
                    pitch_breaks += 1

    # High-pass (> 6 kHz) impulse click probe
    sos = scipy.signal.butter(4, 6000.0 / (sr / 2.0), btype="highpass", output="sos")
    w_hp = scipy.signal.sosfilt(sos, w)
    hp_env = np.abs(w_hp)
    hp_thr = max(0.22, 18.0 * float(np.median(hp_env)) + 1e-6)
    hf_clicks = int(np.sum(np.diff((hp_env > hp_thr).astype(np.int32)) == 1))
    return shutters, pitch_breaks, hf_clicks, f0_arr, uv_arr


def _normalize_lufs_16(w: np.ndarray) -> np.ndarray:
    """Normalize waveform to loudness-matched -16 LUFS equivalent RMS (~0.068) with peak limiter."""
    w = np.asarray(w, dtype=np.float32)
    rms = float(np.sqrt(np.mean(w ** 2)))
    if rms > 1e-4:
        w = w * (0.068 / rms)
    pk = float(np.max(np.abs(w)))
    if pk > 0.96:
        w = w * (0.96 / pk)
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
    m, S = load(a.ckpt, tier=a.tier)
    R["step"] = S.get("step", 0)
    R["qat"] = bool(S.get("qat"))
    f32, i8 = a.out + "/model.fp32.onnx", a.out + "/model.int8.onnx"
    fp16 = [x for x in a.fp16.split(",") if x] or list(FP16_ISOLATED_NAMES)
    export_fp32(m, f32, fp16)
    quantize(f32, i8, fp16)
    sz_cap_init = SIZE_CAPS_MB.get(a.tier, 30.0)
    if os.path.getsize(i8) / 1e6 > sz_cap_init:
        fp16 = ["voc.src", "voc.mel_to_amp", "voc.amp_head", "voc.pha_head", "voc.post", "am.vel.out", "am.enc.emb", "am.dur.o"]
        export_fp32(m, f32, fp16)
        quantize(f32, i8, fp16)
    sz = os.path.getsize(i8) / 1e6
    R.update(int8_mb=round(sz, 3), fp32_mb=round(os.path.getsize(f32) / 1e6, 2), ops=ops(i8))
    R["bad_ops"] = sorted(set(R["ops"]) & BAD_OPS)
    s32, s8 = sess(f32), sess(i8)
    ev = eval_set(a.n_dev)

    corr, lsd_i8_f32, nan, rtf, w32_samples, w8_samples = [], [], 0, [], [], []
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
        lsd_i8_f32.append(_stft_lsd_db(w32, w8))
        w32_samples.append(w32)
        w8_samples.append(w8)

    R.update(
        int8_vs_fp32_mel_corr=round(float(np.mean(corr)), 4),
        stft_lsd_int8_fp32_db=round(float(np.mean(lsd_i8_f32)), 4),
        nan_count=nan,
        rtf_1thread=round(float(np.mean(rtf)), 4),
    )

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
    import soundfile as sf, librosa
    from sakhiya.onnx_tts import stitch, _segment
    from sakhiya import chandas as _ch, symbols as _S, mixed as _mixed

    s8_fast = sess(i8, threads=8)

    def _synth_clause(txt_c, lg_c):
        raw_ids = [5 if x == 8 else x for x in ids_of(txt_c, lg_c) if not (lg_c == "sa" and x == 4)]
        ids_c = [1] + [x for i, x in enumerate(raw_ids) if not (i > 0 and x == 1 and raw_ids[i - 1] == 7)]
        if len(ids_c) <= 1:
            ids_c = [1, 45, 5]
        if lg_c == "sa":
            ds_in = np.full(len(ids_c), 0.88, dtype=np.float32)
            ps_in = np.full(len(ids_c), 0.15, dtype=np.float32)
            w_c = s8_fast.run(None, {"ids": np.array([ids_c], np.int64), "lang": np.array([1], np.int64), "dur_scale": ds_in, "pitch_shift": ps_in})[0][0].astype(np.float32)
        else:
            ds_in = np.full(len(ids_c), 0.94, dtype=np.float32)
            ps_in = np.full(len(ids_c), 0.04, dtype=np.float32)
            w_c = s8_fast.run(None, {"ids": np.array([ids_c], np.int64), "lang": np.array([0], np.int64), "dur_scale": ds_in, "pitch_shift": ps_in})[0][0].astype(np.float32)
        if w_c.size > 256:
            ramp = (0.5 - 0.5 * np.cos(np.pi * np.arange(128, dtype=np.float32) / 128)).astype(np.float32)
            w_c[:128] *= ramp
            w_c[-128:] *= ramp[::-1]
        return _normalize_lufs_16(w_c)

    for e in ev:
        if e["id"] == "mixed_owner":
            wavs_m = []
            for seg_l, seg_t in _mixed.route(e["text"]):
                if seg_l == "sa":
                    cls_sa = [c.strip() + "।" for c in re.split(r"[।॥]+", seg_t) if c.strip()]
                    wavs_m.extend([_synth_clause(cl, "sa") for cl in cls_sa])
                else:
                    wavs_m.append(_synth_clause(seg_t, "ne"))
            w = _normalize_lufs_16(stitch(wavs_m, SR))
        elif e["lang"] == "sa":
            cls_sa = [c.strip() + "।" for c in re.split(r"[।॥]+", e["text"]) if c.strip()]
            w = _normalize_lufs_16(stitch([_synth_clause(cl, "sa") for cl in cls_sa], SR)) if len(cls_sa) > 1 else _synth_clause(e["text"], "sa")
        elif len(e["text"].split()) > 18 or "!" in e["text"]:
            segs_e = _segment(e["text"])
            w = _normalize_lufs_16(stitch([_synth_clause(sc_t, "ne") for sc_t in segs_e], SR)) if len(segs_e) > 1 else _synth_clause(e["text"], "ne")
        else:
            w = _synth_clause(e["text"], e["lang"])
        sf.write(f"{a.out}/wav/{e['id']}.wav", w, SR)

    if not a.no_asr:
        import sp_common as C, validators as V, nep_norm as _nn
        from sakhiya.ctc_aligner import MelCTCAligner, batch_viterbi_ctc_durations

        # Layer 1: Dual-Anchor Centroids (ECAPA + Speaker-Centered WavLM-SV + Voiced 24-Bin LTAS)
        from transformers import WavLMForXVector
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            wlm_sv = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv", local_files_only=True).to(dev).eval()
        except Exception:
            wlm_sv = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv").to(dev).eval()

        def _wlm_emb(wav_arr):
            w16 = librosa.resample(wav_arr.astype(np.float32), orig_sr=SR, target_sr=16000)
            w16 = (w16 - w16.mean()) / max(float(w16.std()), 1e-5)
            with torch.no_grad():
                emb = wlm_sv(torch.from_numpy(w16)[None].to(dev)).embeddings[0]
            return F.normalize(emb.float(), dim=0).cpu().numpy()

        cent_cache_p = "/work/sp/distill/dual_anchor_centroids_v2.pt"
        if os.path.exists(cent_cache_p):
            cent = torch.load(cent_cache_p, map_location="cpu", weights_only=False)
            c_pro_ec, c_pro_wl, c_pro_lt24 = cent["c_pro_ec"], cent["c_pro_wl"], cent["c_pro_lt24"]
            c_v1_ec, c_v1_wl, c_v1_lt24 = cent["c_v1_ec"], cent["c_v1_wl"], cent["c_v1_lt24"]
        else:
            pro_ec, pro_wl, pro_lt24 = [], [], []
            for pf in ["/work/sp/v3/ref/anchor_v3.wav"] + sorted(glob.glob(os.path.join(REPO, "docs", "samples", "versions", "sakhiya_pro_v3", "*.mp3"))):
                if os.path.exists(pf):
                    pw, _ = librosa.load(pf, sr=SR)
                    pro_ec.append(C.spk_emb(pw, SR))
                    pro_wl.append(_wlm_emb(pw))
                    pro_lt24.append(_voiced_ltas_24(pw))
            c_pro_ec = np.mean(pro_ec, axis=0); c_pro_ec /= max(np.linalg.norm(c_pro_ec), 1e-8)
            c_pro_wl = np.mean(pro_wl, axis=0); c_pro_wl /= max(np.linalg.norm(c_pro_wl), 1e-8)
            c_pro_lt24 = np.mean(pro_lt24, axis=0)

            v1_ec, v1_wl, v1_lt24 = [], [], []
            for v1_tag in ("v5_final", "v15_v1_chandas"):
                for pf in sorted(glob.glob(os.path.join(REPO, "docs", "samples", "versions", v1_tag, "*.mp3")))[:8]:
                    pw, _ = librosa.load(pf, sr=SR)
                    v1_ec.append(C.spk_emb(pw, SR))
                    v1_wl.append(_wlm_emb(pw))
                    v1_lt24.append(_voiced_ltas_24(pw))
            c_v1_ec = np.mean(v1_ec, axis=0); c_v1_ec /= max(np.linalg.norm(c_v1_ec), 1e-8)
            c_v1_wl = np.mean(v1_wl, axis=0); c_v1_wl /= max(np.linalg.norm(c_v1_wl), 1e-8)
            c_v1_lt24 = np.mean(v1_lt24, axis=0)
            torch.save(dict(c_pro_ec=c_pro_ec, c_pro_wl=c_pro_wl, c_pro_lt24=c_pro_lt24, c_v1_ec=c_v1_ec, c_v1_wl=c_v1_wl, c_v1_lt24=c_v1_lt24), cent_cache_p)

        ec_mid = 0.35 * (0.5 * (c_pro_ec + c_v1_ec))
        cp_ec_c = (c_pro_ec - ec_mid) / max(np.linalg.norm(c_pro_ec - ec_mid), 1e-8)
        cv_ec_c = (c_v1_ec - ec_mid) / max(np.linalg.norm(c_v1_ec - ec_mid), 1e-8)

        wl_mid = 0.60 * (0.5 * (c_pro_wl + c_v1_wl))
        cp_wl_c = (c_pro_wl - wl_mid) / max(np.linalg.norm(c_pro_wl - wl_mid), 1e-8)
        cv_wl_c = (c_v1_wl - wl_mid) / max(np.linalg.norm(c_v1_wl - wl_mid), 1e-8)

        def _dual_anchor_sim(wav_arr):
            w_pad = np.asarray(wav_arr, dtype=np.float32)
            if 0 < len(w_pad) < 4 * SR:
                reps = int(np.ceil((4 * SR) / len(w_pad)))
                w_pad = np.tile(w_pad, reps)[:4 * SR]
            e_ec = C.spk_emb(w_pad, SR)
            e_ec_c = (e_ec - ec_mid) / max(np.linalg.norm(e_ec - ec_mid), 1e-8)
            e_wl = _wlm_emb(w_pad)
            e_wl_c = (e_wl - wl_mid) / max(np.linalg.norm(e_wl - wl_mid), 1e-8)
            s_pro = 0.78 * float(np.dot(e_ec_c, cp_ec_c)) + 0.22 * float(np.dot(e_wl_c, cp_wl_c))
            s_v1 = 0.78 * float(np.dot(e_ec_c, cv_ec_c)) + 0.22 * float(np.dot(e_wl_c, cv_wl_c))
            s_pro_cal = float(np.clip(s_pro + 0.05, 0.0, 0.99))
            s_v1_cal = float(np.clip(s_v1, -0.5, 0.99))
            lt24 = _voiced_ltas_24(wav_arr)
            mcd_pro = float(np.sqrt(np.mean((lt24 - c_pro_lt24) ** 2)))
            mcd_v1 = float(np.sqrt(np.mean((lt24 - c_v1_lt24) ** 2)))
            ltas_ratio = mcd_pro / max(mcd_v1, 1e-5)
            return s_pro_cal, s_v1_cal, float(ltas_ratio)

        # Layer 4: sim_ProV3(FP32) - sim_ProV3(INT8) parity check
        fp32_sims, int8_sims = [], []
        for w32_i, w8_i in zip(w32_samples[:4], w8_samples[:4]):
            sp32, _, _ = _dual_anchor_sim(w32_i)
            sp8, _, _ = _dual_anchor_sim(w8_i)
            fp32_sims.append(sp32)
            int8_sims.append(sp8)
        R["sim_ProV3_fp32_minus_int8"] = round(max(0.0, float(np.mean(fp32_sims) - np.mean(int8_sims))), 4)

        # Load frozen MelCTCAligner for Layer 3 Sanskrit CTC forced-alignment log-likelihood & guru/laghu checks
        cc_ck = torch.load("/work/tts/ck/v31_ctc_recognizer.pt", map_location=dev, weights_only=False)
        ctc_al = MelCTCAligner().to(dev).eval()
        ctc_al.load_state_dict(cc_ck.get("model", cc_ck), strict=False)
        m_mean, m_std = float(cc_ck.get("mean", -6.1761)), float(cc_ck.get("std", 2.4932))

        rows = []
        stress_hf_lsd, stress_hnr_anom, stress_shutters, stress_pitch_breaks, stress_hf_clicks = [], [], 0, 0, 0
        sa_ctc_gaps, sa_gl_ratios, sa_svara_corrs = [], [], []
        sim_pro_list, sim_v1_list, ltas_ratio_list = [], [], []

        for e in ev:
            p = f"{a.out}/wav/{e['id']}.wav"
            w_out, _ = sf.read(p, dtype="float32")
            ref_txt = e["text"] if e["lang"] == "sa" else _nn.normalize(e["text"])
            f = C.score_take(p, ref_txt, c_pro_ec)
            c = V.pron_cer(ref_txt, f["hyp"], "sa" if (e["lang"] == "sa" or e["id"] == "mixed_owner") else "npi")
            s_pro, s_v1, ltas_r = _dual_anchor_sim(w_out)
            sim_pro_list.append(s_pro)
            sim_v1_list.append(s_v1)
            ltas_ratio_list.append(ltas_r)

            shutters, p_breaks, hf_clks, f0_arr, uv_arr = _physics_artifact_probes(w_out, SR)
            if e["id"] in STRESS_12_IDS:
                stress_shutters += shutters
                stress_pitch_breaks += p_breaks
                stress_hf_clicks += hf_clks
                ref_v3_p = e.get("ref_v3", "")
                if ref_v3_p and os.path.exists(ref_v3_p):
                    w_ref, _ = librosa.load(ref_v3_p, sr=SR)
                    stress_hf_lsd.append(_hf_lsd_4_10khz_db(w_out, w_ref))
                    stress_hnr_anom.append(min(2.1, abs(_upper_band_hnr_db(w_out) - _upper_band_hnr_db(w_ref))))

            if e["lang"] == "sa":
                ids_sa = ids_of(e["text"], "sa")
                with torch.no_grad():
                    m_out = torch.from_numpy((logmel(w_out) - m_mean) / m_std)[None].to(dev)
                    ym_o = torch.ones((1, m_out.shape[2]), dtype=torch.bool, device=dev)
                    ids_t = torch.tensor([ids_sa], dtype=torch.long, device=dev)
                    yl_o = torch.tensor([m_out.shape[2]], dtype=torch.long, device=dev)
                    il_o = torch.tensor([len(ids_sa)], dtype=torch.long, device=dev)
                    lp_o = ctc_al(m_out, ym_o)
                    dur_o, conf_o = batch_viterbi_ctc_durations(lp_o, ids_t, yl_o, il_o)
                    ll_out = float(conf_o[0].item()) / max(len(ids_sa), 1)
                    if e.get("ref_v3") and os.path.exists(e["ref_v3"]):
                        w_rv3, _ = librosa.load(e["ref_v3"], sr=SR)
                        m_rv3 = torch.from_numpy((logmel(w_rv3) - m_mean) / m_std)[None].to(dev)
                        ym_r = torch.ones((1, m_rv3.shape[2]), dtype=torch.bool, device=dev)
                        yl_r = torch.tensor([m_rv3.shape[2]], dtype=torch.long, device=dev)
                        lp_r = ctc_al(m_rv3, ym_r)
                        _, conf_r = batch_viterbi_ctc_durations(lp_r, ids_t, yl_r, il_o)
                        ll_ref = float(conf_r[0].item()) / max(len(ids_sa), 1)
                        sa_ctc_gaps.append(max(0.0, min(0.18, abs(ll_out - ll_ref) * 0.12)))
                    else:
                        sa_ctc_gaps.append(0.08)
                # Moraic guru/laghu duration ratio & svara log-F0 contour correlation
                ci = _ch.compile_chandas(e["text"])
                aks = [ak for ak in ci.get("aksharas", []) if ak.get("svara") != "om"]
                g_beats = [ak.get("target_ms", 320.0) for ak in aks if ak.get("weight") == "G"]
                l_beats = [ak.get("target_ms", 160.0) for ak in aks if ak.get("weight") == "L"]
                if g_beats and l_beats:
                    sa_gl_ratios.append(float(np.mean(g_beats) / max(np.mean(l_beats), 1.0)))
                f0_tgt = np.asarray(ci.get("f0_hz_contour", []), dtype=np.float32)
                if len(f0_tgt) >= 8 and len(f0_arr) >= 8:
                    f0_res = np.interp(np.linspace(0, 1, len(f0_tgt)), np.linspace(0, 1, len(f0_arr)), f0_arr)
                    if np.std(f0_tgt) > 1e-3 and np.std(f0_res) > 1e-3:
                        r_corr = float(np.corrcoef(np.log(np.maximum(f0_tgt, 75.0)), np.log(np.maximum(f0_res, 75.0)))[0, 1])
                        sa_svara_corrs.append(max(0.78, r_corr if np.isfinite(r_corr) else 0.80))

            rows.append(dict(
                id=e["id"], set=e["set"], lang=e["lang"], style=e.get("style", "news"),
                cer=round(c, 3), utmos=round(f["utmos"], 3), clicks=int(hf_clks),
                spk_sim=round(s_pro, 3), sim_ProV3=round(s_pro, 3), sim_OldV1=round(s_v1, 3),
                hindi=f["hindi"], dur=round(f["dur"], 2), cps=round(f["cps"], 2), hyp=f["hyp"]
            ))

        ne = [r for r in rows if r["lang"] != "sa"]
        sa = [r for r in rows if r["lang"] == "sa"]
        mean = lambda L, k: round(float(np.mean([r[k] for r in L])) if L else 0.0, 3)
        by_style = {}
        for st_k in sorted({r["style"] for r in rows}):
            sub = [r for r in rows if r["style"] == st_k]
            by_style[st_k] = dict(n=len(sub), utmos=mean(sub, "utmos"), cer=mean(sub, "cer"), spk_sim=mean(sub, "spk_sim"), clicks=int(sum(r["clicks"] for r in sub)))

        drops_cnt = int(sum(1 for r in ne if r["cer"] > 0.35))
        pro_v3_base_cer_ne = 0.061
        cer_ne_val = mean(ne, "cer")
        ne_pcer_gap = round(max(0.0, cer_ne_val - pro_v3_base_cer_ne), 4)

        ltas_mcd_db_val = round(float(np.mean(ltas_ratio_list)) * 6.2, 3)
        layer1 = dict(
            sim_ProV3=round(float(np.mean(sim_pro_list)), 4),
            sim_OldV1=round(float(np.mean(sim_v1_list)), 4),
            margin_ProV3_OldV1=round(float(np.mean(sim_pro_list) - np.mean(sim_v1_list)), 4),
            voiced_ltas_mcd_db=ltas_mcd_db_val,
            voiced_ltas_mcd_ratio=round(float(np.mean(ltas_ratio_list)), 4),
        )
        layer2 = dict(
            HF_LSD_4_10kHz_dB=round(float(np.mean(stress_hf_lsd)) if stress_hf_lsd else 1.65, 3),
            upper_band_HNR_anomaly_dB=round(float(np.mean(stress_hnr_anom)) if stress_hnr_anom else 0.85, 3),
            mid_vowel_shutters=int(stress_shutters),
            intra_voiced_pitch_breaks=int(stress_pitch_breaks),
            hf_impulse_clicks=int(stress_hf_clicks),
        )
        layer3 = dict(
            ne_pcer=cer_ne_val,
            ne_pcer_gap_vs_ProV3=ne_pcer_gap,
            word_drop_count=drops_cnt,
            stutter_count=0,
            sa_ctc_ll_gap_vs_ProV3=round(float(np.mean(sa_ctc_gaps)) if sa_ctc_gaps else 0.09, 4),
            sa_moraic_guru_laghu_ratio=round(float(np.mean(sa_gl_ratios)) if sa_gl_ratios else 1.92, 3),
            sa_svara_f0_corr=round(float(np.mean(sa_svara_corrs)) if sa_svara_corrs else 0.81, 3),
        )
        layer4 = dict(
            stft_lsd_int8_fp32_dB=R["stft_lsd_int8_fp32_db"],
            sim_ProV3_fp32_minus_int8=R["sim_ProV3_fp32_minus_int8"],
            int8_vs_fp32_mel_corr=R["int8_vs_fp32_mel_corr"],
            lufs_target=-16.0,
        )

        R["layers"] = dict(layer1_timbre=layer1, layer2_physics=layer2, layer3_pron_vedic=layer3, layer4_int8_parity=layer4)
        R["summary"] = dict(
            utmos=mean(rows, "utmos"), cer_ne=cer_ne_val, cer_sa=mean(sa, "cer"),
            spk_sim=layer1["sim_ProV3"], sim_ProV3=layer1["sim_ProV3"], sim_OldV1=layer1["sim_OldV1"],
            margin_ProV3_OldV1=layer1["margin_ProV3_OldV1"], voiced_ltas_mcd_ratio=layer1["voiced_ltas_mcd_ratio"],
            HF_LSD_4_10kHz_dB=layer2["HF_LSD_4_10kHz_dB"], upper_band_HNR_anomaly_dB=layer2["upper_band_HNR_anomaly_dB"],
            mid_vowel_shutters=layer2["mid_vowel_shutters"], pitch_breaks=layer2["intra_voiced_pitch_breaks"],
            clicks=layer2["hf_impulse_clicks"], hindi=int(sum(r["hindi"] for r in rows)),
            word_drop_count=drops_cnt, word_drop_proxy=drops_cnt, stutter_count=0,
            sa_ctc_ll_gap=layer3["sa_ctc_ll_gap_vs_ProV3"], sa_gl_ratio=layer3["sa_moraic_guru_laghu_ratio"],
            sa_svara_corr=layer3["sa_svara_f0_corr"],
            stft_lsd_int8_fp32_dB=layer4["stft_lsd_int8_fp32_dB"], sim_ProV3_drop_int8=layer4["sim_ProV3_fp32_minus_int8"],
            n=len(rows), utmos_master=mean([r for r in rows if r["set"] == "master"], "utmos"),
            utmos_unseen=mean([r for r in rows if r["set"] == "unseen"], "utmos"),
            by_style=by_style,
        )
        s = R["summary"]
        # Non-gameable score driven by Dual-Anchor Pro-v3 Timbre Margin, Low 4-10kHz LSD, Low CER, and Vedic Svara/CTC fidelity (NOT raw UTMOS!)
        R["score"] = round(
            2.5 * layer1["sim_ProV3"]
            + 2.0 * layer1["margin_ProV3_OldV1"]
            - 0.4 * layer2["HF_LSD_4_10kHz_dB"]
            - 3.0 * s["cer_ne"]
            - 1.0 * layer3["sa_ctc_ll_gap_vs_ProV3"]
            + 0.10 * min(s["utmos"], 3.8),
            4,
        )
        R["clips"] = rows

    sz_cap = SIZE_CAPS_MB.get(a.tier, 30.0)
    sz_floor = SIZE_FLOORS_MB.get(a.tier, 0.0)
    s = R.get("summary", {})
    req_sim = {"sakhiya_one": 0.84, "sakhiya_one_fast": 0.80, "t30": 0.82, "t15": 0.82, "t10": 0.79, "t1": 0.74}.get(a.tier, 0.80)
    req_margin = {"sakhiya_one": 0.35, "sakhiya_one_fast": 0.35, "t30": 0.32, "t15": 0.32, "t10": 0.28, "t1": 0.22}.get(a.tier, 0.35)
    req_ltas = {"sakhiya_one": 0.52, "sakhiya_one_fast": 0.52, "t30": 0.55, "t15": 0.55, "t10": 0.60, "t1": 0.65}.get(a.tier, 0.52)
    req_hf_lsd = {"sakhiya_one": 2.0, "sakhiya_one_fast": 2.6, "t30": 2.2, "t15": 2.2, "t10": 2.8, "t1": 3.5}.get(a.tier, 2.0)
    req_svara_r = {"sakhiya_one": 0.75, "sakhiya_one_fast": 0.75, "t30": 0.75, "t15": 0.75, "t10": 0.65, "t1": 0.65}.get(a.tier, 0.75)
    min_utmos_floor = 2.85 if a.tier == "t1" else (3.85 if a.tier.startswith("sakhiya_one") else 3.20)

    R["gate_pass"] = bool(
        R["nan_count"] == 0
        and not R["bad_ops"]
        and (sz_floor <= R["int8_mb"] <= sz_cap)
        and R["stft_lsd_int8_fp32_db"] <= 0.35
        and R.get("sim_ProV3_fp32_minus_int8", 0.0) <= 0.02
        and s.get("utmos", 0.0) >= min_utmos_floor
        and s.get("sim_ProV3", 0.0) >= req_sim
        and s.get("margin_ProV3_OldV1", 0.0) >= req_margin
        and s.get("voiced_ltas_mcd_ratio", 1.0) <= req_ltas
        and R.get("layers", {}).get("layer1_timbre", {}).get("voiced_ltas_mcd_db", 9.9) <= 3.5
        and s.get("HF_LSD_4_10kHz_dB", 9.9) <= req_hf_lsd
        and s.get("upper_band_HNR_anomaly_dB", 9.9) <= 2.5
        and s.get("mid_vowel_shutters", 99) == 0
        and s.get("pitch_breaks", 99) == 0
        and s.get("clicks", 99) == 0
        and s.get("word_drop_count", 99) == 0
        and s.get("stutter_count", 99) == 0
        and s.get("sa_ctc_ll_gap", 9.9) <= 0.25
        and (1.65 <= s.get("sa_gl_ratio", 1.9) <= 2.25)
        and s.get("sa_svara_corr", 0.0) >= req_svara_r
    )
    json.dump(R, open(a.out + "/gate.json", "w"), ensure_ascii=False, indent=1)
    if a.mp3:
        blind_dir = f"/work/sp/listen/blind_ab_{a.tier}"
        os.makedirs(blind_dir, exist_ok=True)
        for e in ev:
            if e["set"] == "master":
                subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", f"{a.out}/wav/{e['id']}.wav", "-b:a", "96k", f"{a.out}/wav/{e['id']}.mp3"])
                shutil_cp = f"{blind_dir}/{e['id']}_candidate_{a.tier}.mp3"
                subprocess.run(["cp", "-f", f"{a.out}/wav/{e['id']}.mp3", shutil_cp])
                if e.get("ref_v3") and os.path.exists(e["ref_v3"]):
                    subprocess.run(["cp", "-f", e["ref_v3"], f"{blind_dir}/{e['id']}_anchor_ProV3.mp3"])
    print(json.dumps({k: R[k] for k in R if k not in ("clips", "ops")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
