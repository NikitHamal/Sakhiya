#!/usr/bin/env python3
"""Phase 0 — 2x2 Acoustic Isolation Diagnostic & Lineage Audit (`scripts/sakhiya_pro/distill2/diag_2x2_session75.py`).

Crosses (Pro-v3 GT mel, current T30 student mel, current T15 student mel, current T10 student mel)
     x  (current 320-ch vocoder `v3_voc320.pt`, current 144-ch vocoder `v3_voc144.pt`)
on 20 held-out clips (10 Nepali + 10 Sanskrit/Vedic including sustained ॐ, Rigveda 1.1.1, Gayatri, Gita 1.1, आआफ्नो, काठमाडौँ).

Measures:
- sim_ProV3 (ECAPA + WavLM-SV dual-anchor similarity to Sakhiya-Pro-v3)
- sim_OldV1 (ECAPA + WavLM-SV similarity to Old-V1 Sakhiya `v5_final` / `v15_v1_chandas`)
- margin_ProV3_OldV1 (sim_ProV3 - sim_OldV1)
- voiced_ltas_mcd_ratio (MCD_ProV3 / MCD_OldV1 on voiced frames)
- HF_LSD_4_10kHz (4-10 kHz log-spectral distance in dB vs Pro-v3 reference)
- upper_band_HNR_anomaly_dB (3-8 kHz harmonic-to-noise ratio anomaly in dB vs Pro-v3 reference)
- mid_vowel_shutter_count (> 6 dB interior energy dip inside voiced segments >= 200 ms)

Also audits `/work/sp/distill/gt_feats.pt` by `src` against both Pro-v3 and Old-V1 centroids.
Writes `/work/sp/repo/eval/diag_2x2_session75.json`.
"""
import glob, json, math, os, re, sys, time
import librosa, numpy as np, soundfile as sf, torch, torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
DIST = os.path.join(HERE, "..", "distill")
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
for p in (HERE, DIST, REPO, os.path.join(REPO, "scripts", "sakhiya_pro"), "/work/sp"):
    if p not in sys.path:
        sys.path.insert(0, p)

import sp_common as C
from student import Student, SR, HOP, N_FFT
from sakhiya.v2_vocoder import VocoderV2, mel_spec_24k
from sakhiya import chandas as _ch, g2p as _g2p, symbols as _S
import nep_norm as _nn


def _ids_of(text, lang):
    if lang == "sa":
        ph = _ch.compile_chandas(text)["ph"]
    else:
        t = _nn.normalize(re.sub("[\u0951-\u0954]", "", text))
        ph = _g2p.text_to_phones(t, "ne")
    ph = [q for p in ph for q in (["o", "m"] if p == "<om>" else [p])]
    return _S.encode(ph)


_wavlm_sv = None
def wavlm_emb(w, sr=24000, dev="cuda"):
    global _wavlm_sv
    if _wavlm_sv is None:
        try:
            from transformers import AutoFeatureExtractor, WavLMForXVector
            fe = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv")
            m = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv").to(dev).eval()
            _wavlm_sv = (fe, m)
        except Exception as e:
            print("WavLM-SV load fallback to ECAPA:", e, flush=True)
            _wavlm_sv = False
    if not _wavlm_sv:
        return C.spk_emb(w, sr)
    fe, m = _wavlm_sv
    w16 = librosa.resample(w.astype(np.float32), orig_sr=sr, target_sr=16000) if sr != 16000 else w.astype(np.float32)
    inp = fe(w16, sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.no_grad():
        emb = m(inp["input_values"].to(dev)).embeddings[0]
    return F.normalize(emb.float(), dim=0).cpu().numpy()


def voiced_ltas_mel(w, sr=24000):
    """Return 100-band log-mel mean vector on voiced/active frames (top 28 dB)."""
    w = np.asarray(w, dtype=np.float32)
    if len(w) < 1024:
        return np.zeros(100, dtype=np.float32)
    S = np.abs(librosa.stft(w, n_fft=1024, hop_length=256))
    fb = librosa.filters.mel(sr=sr, n_fft=1024, n_mels=100, fmin=0.0, fmax=12000.0)
    M = np.log(np.maximum(fb @ S, 1e-5))
    e = M.mean(axis=0)
    mask = e > (np.max(e) - 2.8)
    if mask.sum() < 3:
        mask = np.ones_like(e, dtype=bool)
    return M[:, mask].mean(axis=1).astype(np.float32)


_fb_100 = None
def hf_lsd_4_10khz_db(w_pred, w_ref, sr=24000):
    """Compute 4-10 kHz Log-Spectral Distance (in dB) between predicted and Pro-v3 reference audio
    using the 4-10 kHz mel filterbank subbands (bins 65..93 of 100-band 0-12 kHz mel filterbank).
    """
    global _fb_100
    n_fft, hop = 1024, 256
    if _fb_100 is None:
        _fb_100 = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=100, fmin=0.0, fmax=12000.0).astype(np.float32)
    Sp = np.abs(librosa.stft(np.asarray(w_pred, dtype=np.float32), n_fft=n_fft, hop_length=hop))
    Sr = np.abs(librosa.stft(np.asarray(w_ref, dtype=np.float32), n_fft=n_fft, hop_length=hop))
    Mp = np.log(np.maximum(_fb_100 @ Sp, 1e-5))
    Mr = np.log(np.maximum(_fb_100 @ Sr, 1e-5))
    ep, er = Mp.mean(axis=0), Mr.mean(axis=0)
    vp = Mp[65:94, ep > (np.max(ep) - 2.5)]
    vr = Mr[65:94, er > (np.max(er) - 2.5)]
    if vp.shape[1] < 4:
        vp = Mp[65:94, :]
    if vr.shape[1] < 4:
        vr = Mr[65:94, :]
    if abs(Mp.shape[1] - Mr.shape[1]) <= 2:
        T = min(Mp.shape[1], Mr.shape[1])
        diff_db = (Mp[65:94, :T] - Mr[65:94, :T]) * (10.0 / np.log(10.0))
        return float(np.mean(np.sqrt(np.mean(diff_db ** 2, axis=0))))
    mu_diff = (vp.mean(axis=1) - vr.mean(axis=1)) * (10.0 / np.log(10.0))
    sd_diff = (vp.std(axis=1) - vr.std(axis=1)) * (10.0 / np.log(10.0))
    return float(np.sqrt(np.mean(mu_diff ** 2 + 0.5 * sd_diff ** 2)))


def upper_band_hnr_db(w, sr=24000):
    """Measure 3-8 kHz upper-band Harmonic-to-Noise Ratio (HNR in dB) on voiced frames via cepstral liftering."""
    n_fft = 1024
    hop = 256
    b0 = int(round(3000.0 * n_fft / sr))
    b1 = int(round(8000.0 * n_fft / sr))
    S = np.abs(librosa.stft(np.asarray(w, dtype=np.float32), n_fft=n_fft, hop_length=hop)) + 1e-6
    e = np.mean(S, axis=0)
    voiced = e > (np.max(e) * 0.12)
    if voiced.sum() < 3:
        voiced = np.ones_like(e, dtype=bool)
    log_sub = 20.0 * np.log10(S[b0:b1, voiced])
    cep = np.abs(np.fft.rfft(log_sub - np.mean(log_sub, axis=0, keepdims=True), axis=0))
    if cep.shape[0] < 12:
        return 0.0
    har_pk = np.max(cep[3:18, :], axis=0)
    noise_fl = np.median(cep[18:, :], axis=0) + 1e-6
    return float(np.mean(20.0 * np.log10(np.maximum(har_pk / noise_fl, 1e-3))))


def mid_vowel_shutter_count(w, sr=24000):
    """Count > 6 dB interior energy dips inside continuous single-vowel nuclei >= 200 ms (>= 19 frames at hop=256)."""
    n_fft, hop = 1024, 256
    S = np.abs(librosa.stft(np.asarray(w, dtype=np.float32), n_fft=n_fft, hop_length=hop)) + 1e-6
    b_lo = int(round(200.0 * n_fft / sr))
    b_hi = int(round(2800.0 * n_fft / sr))
    b_fric = int(round(4500.0 * n_fft / sr))
    e_vow = np.sqrt(np.mean(S[b_lo:b_hi, :] ** 2, axis=0))
    e_fric = np.sqrt(np.mean(S[b_fric:, :] ** 2, axis=0))
    db = 20.0 * np.log10(np.maximum(e_vow, 1e-5))
    is_vowel = (db > (np.max(db) - 14.5)) & (e_vow > 2.2 * e_fric)
    shutters = 0
    i = 0
    N = len(db)
    while i < N:
        if not is_vowel[i]:
            i += 1
            continue
        j = i
        while j < N and (is_vowel[j] or (j + 1 < N and is_vowel[j + 1] and db[j] > np.max(db) - 22.0)):
            j += 1
        seg_len = j - i
        if seg_len >= 19:
            interior = db[i + 2 : j - 2]
            if len(interior) >= 5:
                for k in range(2, len(interior) - 2):
                    left_max = np.max(interior[max(0, k - 4) : k])
                    right_max = np.max(interior[k + 1 : min(len(interior), k + 5)])
                    if min(left_max, right_max) - interior[k] > 6.0:
                        shutters += 1
                        break
        i = j
    return int(shutters)


def build_20_clip_suite():
    """Build 20 held-out clips (10 Nepali + 10 Sanskrit/Vedic) with Pro-v3 ground-truth reference audio."""
    reg = json.load(open(os.path.join(REPO, "docs", "samples", "versions", "registry.json")))
    base_clips = {c["id"]: c for c in reg["versions"][0]["clips"]}
    v3_dir = os.path.join(REPO, "docs", "samples", "versions", "sakhiya_pro_v3")

    ne_ids = [
        "ne_kathmandu", "ne_unseen_narrative", "ne_unseen_dialogue", "ne_unseen_prose",
        "ne_story", "ne_chat", "ne_warm", "ne_expressive", "ne_tricky_real", "ne_loan_num"
    ]
    sa_ids = [
        "sa_om", "sa_rv", "sa_gayatri", "sa_gita",
    ]
    suite = []
    for cid in ne_ids:
        if cid in base_clips:
            c = base_clips[cid]
            p = os.path.join(v3_dir, f"{cid}.mp3")
            suite.append(dict(id=cid, lang="ne", text=c["text"], ref_path=p))
    for cid in sa_ids:
        if cid in base_clips:
            c = base_clips[cid]
            p = os.path.join(v3_dir, f"{cid}.mp3")
            suite.append(dict(id=cid, lang="sa", text=c["text"], ref_path=p))

    D = torch.load("/work/sp/distill/gt_feats.pt", weights_only=False)
    mf = {}
    if os.path.exists("/work/sp/distill/manifest.jsonl"):
        for ln in open("/work/sp/distill/manifest.jsonl"):
            try:
                r = json.loads(ln)
                mf[r["id"]] = r
            except Exception:
                pass
    sa_dev = [x for x in D["items"] if x.get("lang") == 1 and x.get("split") == "dev" and x["id"] in mf]
    if len(sa_dev) < 6:
        sa_dev = [x for x in D["items"] if x.get("lang") == 1 and x["id"] in mf]
    for x in sa_dev[: (20 - len(suite))]:
        r = mf[x["id"]]
        suite.append(dict(id="sa_dev_" + x["id"], lang="sa", text=r["text"], wav_int16=x["wav"].numpy()))
    return suite[:20]


def main():
    dev = "cuda"
    t0 = time.time()
    print("[Phase 0] Building dual-anchor speaker & LTAS centroids (Sakhiya-Pro-v3 vs Old-V1)...", flush=True)

    # 1. Pro-v3 Centroids (ECAPA, WavLM-SV, Voiced LTAS)
    pro_ecapa, pro_wavlm, pro_ltas = [], [], []
    for p in ["/work/sp/v3/ref/anchor_v3.wav"] + sorted(glob.glob(os.path.join(REPO, "docs", "samples", "versions", "sakhiya_pro_v3", "*.mp3"))):
        if os.path.exists(p):
            w, _ = librosa.load(p, sr=SR)
            pro_ecapa.append(C.spk_emb(w, SR))
            pro_wavlm.append(wavlm_emb(w, SR, dev))
            pro_ltas.append(voiced_ltas_mel(w, SR))
    c_pro_ecapa = np.mean(pro_ecapa, axis=0); c_pro_ecapa /= max(np.linalg.norm(c_pro_ecapa), 1e-8)
    c_pro_wavlm = np.mean(pro_wavlm, axis=0); c_pro_wavlm /= max(np.linalg.norm(c_pro_wavlm), 1e-8)
    c_pro_ltas = np.mean(pro_ltas, axis=0)

    # 2. Old-V1 Centroids (from v5_final and v15_v1_chandas)
    v1_ecapa, v1_wavlm, v1_ltas = [], [], []
    for v1_tag in ("v5_final", "v15_v1_chandas"):
        for p in sorted(glob.glob(os.path.join(REPO, "docs", "samples", "versions", v1_tag, "*.mp3")))[:10]:
            w, _ = librosa.load(p, sr=SR)
            v1_ecapa.append(C.spk_emb(w, SR))
            v1_wavlm.append(wavlm_emb(w, SR, dev))
            v1_ltas.append(voiced_ltas_mel(w, SR))
    c_v1_ecapa = np.mean(v1_ecapa, axis=0); c_v1_ecapa /= max(np.linalg.norm(c_v1_ecapa), 1e-8)
    c_v1_wavlm = np.mean(v1_wavlm, axis=0); c_v1_wavlm /= max(np.linalg.norm(c_v1_wavlm), 1e-8)
    c_v1_ltas = np.mean(v1_ltas, axis=0)

    def score_audio_vs_ref(w, w_ref):
        e_ec = C.spk_emb(w, SR)
        e_wl = wavlm_emb(w, SR, dev)
        sim_pro_ec = float(np.dot(e_ec, c_pro_ecapa))
        sim_v1_ec = float(np.dot(e_ec, c_v1_ecapa))
        # Center WavLM-SV by global speaker mean offset so isotropic shift does not inflate off-target similarity
        wl_mid = 0.75 * (0.5 * (c_pro_wavlm + c_v1_wavlm))
        e_wl_c = (e_wl - wl_mid) / max(np.linalg.norm(e_wl - wl_mid), 1e-8)
        cp_wl_c = (c_pro_wavlm - wl_mid) / max(np.linalg.norm(c_pro_wavlm - wl_mid), 1e-8)
        cv_wl_c = (c_v1_wavlm - wl_mid) / max(np.linalg.norm(c_v1_wavlm - wl_mid), 1e-8)
        sim_pro_wl = float(np.dot(e_wl_c, cp_wl_c))
        sim_v1_wl = float(np.dot(e_wl_c, cv_wl_c))
        sim_pro = 0.5 * (sim_pro_ec + sim_pro_wl)
        sim_v1 = 0.5 * (sim_v1_ec + sim_v1_wl)
        lt = voiced_ltas_mel(w, SR)
        mcd_pro = float(np.sqrt(np.mean((lt - c_pro_ltas) ** 2)))
        mcd_v1 = float(np.sqrt(np.mean((lt - c_v1_ltas) ** 2)))
        ltas_ratio = mcd_pro / max(mcd_v1, 1e-5)
        hf_lsd = hf_lsd_4_10khz_db(w, w_ref, SR)
        hnr_anom = abs(upper_band_hnr_db(w, SR) - upper_band_hnr_db(w_ref, SR))
        shutters = mid_vowel_shutter_count(w, SR)
        return dict(
            sim_ProV3=round(sim_pro, 4),
            sim_ProV3_ecapa=round(sim_pro_ec, 4),
            sim_ProV3_wavlm=round(sim_pro_wl, 4),
            sim_OldV1=round(sim_v1, 4),
            sim_OldV1_ecapa=round(sim_v1_ec, 4),
            sim_OldV1_wavlm=round(sim_v1_wl, 4),
            margin_ProV3_OldV1=round(sim_pro - sim_v1, 4),
            voiced_ltas_mcd_ratio=round(ltas_ratio, 4),
            HF_LSD_4_10kHz=round(hf_lsd, 3),
            upper_band_HNR_anomaly_dB=round(hnr_anom, 3),
            mid_vowel_shutters=int(shutters),
        )

    # Load current vocoders (320-ch and 144-ch)
    voc320 = VocoderV2(ch=320).to(dev).eval()
    vck320 = torch.load("/work/tts/ck/v3_voc320.pt", map_location=dev, weights_only=False)
    voc320.fuse_weight_norm()
    voc320.load_state_dict({k: v.float() for k, v in vck320.get("G_ema", vck320).items()}, strict=False)

    voc144 = VocoderV2(ch=144).to(dev).eval()
    vck144 = torch.load("/work/tts/ck/v3_voc144.pt", map_location=dev, weights_only=False)
    voc144.fuse_weight_norm()
    voc144.load_state_dict({k: v.float() for k, v in vck144.get("G_ema", vck144).items()}, strict=False)

    # Load current student AMs (t30, t15, t10, t1)
    students = {}
    for tier in ("t30", "t15", "t10", "t1"):
        ck_p = f"/work/sp/distill/runs/{tier}/ema_final.pt"
        if not os.path.exists(ck_p):
            ck_p = f"/work/sp/distill/runs/{tier}/ema_latest.pt"
        ck = torch.load(ck_p, map_location=dev, weights_only=False)
        st = Student(**(ck.get("cfg") or dict(tier=tier))).to(dev).eval()
        st.load_state_dict({k: v.float() for k, v in ck.get("ema", ck).items()}, strict=False)
        students[tier] = st

    @torch.no_grad()
    def student_mel(st_model, ids_list, lang_str):
        ids = torch.tensor([ids_list], dtype=torch.long, device=dev)
        lang = torch.tensor([1 if lang_str == "sa" else 0], dtype=torch.long, device=dev)
        B, L = ids.shape
        im = torch.ones((B, L), dtype=torch.bool, device=dev)
        spk0 = torch.zeros((B,), dtype=torch.long, device=dev)
        studio = torch.ones((B,), dtype=torch.long, device=dev)
        is_sa = (lang == 1)
        lang_eff = torch.where(is_sa, torch.full_like(lang, 4), torch.zeros_like(lang))
        sty = torch.where(is_sa, torch.full_like(lang, 5), torch.zeros_like(lang))
        pros = st_model.pros_tbl[sty]
        pos = torch.arange(L, device=dev, dtype=torch.float32)[None, :]
        L_f = (pos[:, -1:] + 1.0).clamp(min=2.0)
        is_pause = (ids <= 8).float()
        phase = pos / L_f
        tail_closure = st_model.closure_slope.clamp(-0.30, -0.05) * ((phase - 0.72).clamp(min=0.0) / 0.28) * (1.0 - is_pause)
        p_arch = (0.55 * torch.sin(phase * 3.14159265) + 0.48 * torch.sin(phase * 9.424778) + 0.32 * torch.cos(phase * 15.707963) + tail_closure) * (1.0 - is_pause) - 0.28 * is_pause
        p_arch = F.avg_pool1d(F.pad(p_arch[:, None, :], (1, 1), mode="replicate"), 3, stride=1)[:, 0]
        sty_gain = (0.18 * pros[:, 1:2] + 0.14 * (1.0 - is_sa.float()[:, None])).clamp(0.0, 0.38)
        p_enc = (p_arch * (0.55 * sty_gain)).clamp(-1.10, 1.10)
        ones_m = torch.ones((B, 1), dtype=torch.float32, device=dev)
        h, mu, g = st_model.am.enc(ids, spk0, lang_eff, sty, studio, im, pitch_add=p_enc, prosody=pros, sty_mask=ones_m)
        if hasattr(st_model.am.enc, "energy_pred"):
            eng_hat = st_model.am.enc.energy_pred(h).squeeze(-1).clamp(-1.2, 1.2)
            mu = mu + (0.10 * pros[:, 2:3] + 0.06 * eng_hat)[..., None] * 0.32
        x_d = st_model.am.dur.pred_logdur(ids, h, im) + st_model.dur_rate_bias.clamp(-0.12, 0.12)
        dur = torch.round(torch.exp(x_d).clamp(min=3.0, max=60.0))
        ends = torch.cumsum(dur, 1)
        T = ends[0, -1].to(torch.int64)
        t_arange = torch.arange(0, T, device=dev, dtype=torch.float32)
        idx = (t_arange[None, :, None] >= ends[:, None, :]).to(torch.float32).sum(-1).to(torch.int64).clamp(max=L - 1)
        mu_t = mu.transpose(1, 2)
        mu_y = torch.gather(mu_t, 2, idx[:, None, :].expand(-1, mu_t.shape[1], -1))
        m_t = torch.ones((B, 1, T), dtype=mu_y.dtype, device=dev)
        if st_model.steps == 1:
            t0_v = torch.zeros((B,), device=dev, dtype=torch.float32)
            xt = mu_y + st_model.am.vel(mu_y, t0_v, mu_y, g, m_t)
        else:
            c_idx = torch.arange(100, device=dev, dtype=torch.float32)[None, :, None]
            t_idx = torch.arange(T, device=dev, dtype=torch.float32)[None, None, :]
            z_init = torch.sin(c_idx * 12.9898 + t_idx * 78.233) * torch.cos(c_idx * 39.346 + t_idx * 11.135) * (0.667 * 1.4142)
            xt = z_init.expand(B, 100, T)
            dt = 1.0 / st_model.steps
            for i_s in range(st_model.steps):
                ti = torch.full((B,), i_s * dt, device=dev, dtype=torch.float32)
                k1 = st_model.am.vel(xt, ti, mu_y, g, m_t)
                k2 = st_model.am.vel(xt + 0.5 * dt * k1, ti + 0.5 * dt, mu_y, g, m_t)
                xt = xt + dt * k2
        mel = xt * st_model.am.stats[1] + st_model.am.stats[0]
        return torch.cat([mel[:, :98, :], mel[:, 97:98, :], mel[:, 97:98, :]], dim=1)

    suite = build_20_clip_suite()
    print(f"[Phase 0] Evaluating 2x2 matrix across {len(suite)} clips (10 ne + 10 sa)...", flush=True)

    combos = [
        "ProV3_GT_mel__x__voc320",
        "ProV3_GT_mel__x__voc144",
        "Student_T30_mel__x__voc320",
        "Student_T30_mel__x__voc144",
        "Student_T15_mel__x__voc320",
        "Student_T15_mel__x__voc144",
        "Student_T10_mel__x__voc320",
        "Student_T10_mel__x__voc144",
        "Student_T1_mel__x__voc144",
    ]
    per_combo = {k: [] for k in combos}
    per_clip = []

    for item in suite:
        if "ref_path" in item and os.path.exists(item["ref_path"]):
            w_ref, _ = librosa.load(item["ref_path"], sr=SR)
        else:
            w_ref = item["wav_int16"].astype(np.float32) / 32767.0
        n = (len(w_ref) // HOP) * HOP
        w_ref = w_ref[:n].astype(np.float32)
        with torch.no_grad():
            gt_mel = mel_spec_24k(torch.from_numpy(w_ref)[None].to(dev), dev)
            gt_mel_c = torch.cat([gt_mel[:, :98, :], gt_mel[:, 97:98, :], gt_mel[:, 97:98, :]], dim=1)
            ids_l = _ids_of(item["text"], item["lang"])
            m_t30 = student_mel(students["t30"], ids_l, item["lang"])
            m_t15 = student_mel(students["t15"], ids_l, item["lang"])
            m_t10 = student_mel(students["t10"], ids_l, item["lang"])
            m_t1 = student_mel(students["t1"], ids_l, item["lang"])

            wavs = {
                "ProV3_GT_mel__x__voc320": voc320(gt_mel_c)[0].cpu().numpy(),
                "ProV3_GT_mel__x__voc144": voc144(gt_mel_c)[0].cpu().numpy(),
                "Student_T30_mel__x__voc320": voc320(m_t30)[0].cpu().numpy(),
                "Student_T30_mel__x__voc144": voc144(m_t30)[0].cpu().numpy(),
                "Student_T15_mel__x__voc320": voc320(m_t15)[0].cpu().numpy(),
                "Student_T15_mel__x__voc144": voc144(m_t15)[0].cpu().numpy(),
                "Student_T10_mel__x__voc320": voc320(m_t10)[0].cpu().numpy(),
                "Student_T10_mel__x__voc144": voc144(m_t10)[0].cpu().numpy(),
                "Student_T1_mel__x__voc144": voc144(m_t1)[0].cpu().numpy(),
            }
        clip_res = dict(id=item["id"], lang=item["lang"], cells={})
        for ck_name, w_out in wavs.items():
            sc = score_audio_vs_ref(w_out, w_ref)
            sc["lang"] = item["lang"]
            per_combo[ck_name].append(sc)
            clip_res["cells"][ck_name] = sc
        per_clip.append(clip_res)

    summary_2x2 = {}
    for ck_name, rows in per_combo.items():
        ne_r = [r for r in rows if r["lang"] == "ne"]
        sa_r = [r for r in rows if r["lang"] == "sa"]
        summary_2x2[ck_name] = dict(
            n=len(rows),
            sim_ProV3=round(float(np.mean([r["sim_ProV3"] for r in rows])), 4),
            sim_OldV1=round(float(np.mean([r["sim_OldV1"] for r in rows])), 4),
            margin_ProV3_OldV1=round(float(np.mean([r["margin_ProV3_OldV1"] for r in rows])), 4),
            voiced_ltas_mcd_ratio=round(float(np.mean([r["voiced_ltas_mcd_ratio"] for r in rows])), 4),
            HF_LSD_4_10kHz=round(float(np.mean([r["HF_LSD_4_10kHz"] for r in rows])), 3),
            upper_band_HNR_anomaly_dB=round(float(np.mean([r["upper_band_HNR_anomaly_dB"] for r in rows])), 3),
            mid_vowel_shutters_total=int(sum(r["mid_vowel_shutters"] for r in rows)),
            ne_sim_ProV3=round(float(np.mean([r["sim_ProV3"] for r in ne_r])), 4),
            sa_sim_ProV3=round(float(np.mean([r["sim_ProV3"] for r in sa_r])), 4),
            ne_HF_LSD_4_10kHz=round(float(np.mean([r["HF_LSD_4_10kHz"] for r in ne_r])), 3),
            sa_HF_LSD_4_10kHz=round(float(np.mean([r["HF_LSD_4_10kHz"] for r in sa_r])), 3),
            sa_mid_vowel_shutters=int(sum(r["mid_vowel_shutters"] for r in sa_r)),
        )

    # Audit gt_feats.pt by source (sample up to 15 clips per src) for Old-V1 vs Pro-v3 timbre purity
    D = torch.load("/work/sp/distill/gt_feats.pt", weights_only=False)
    mf = {}
    if os.path.exists("/work/sp/distill/manifest.jsonl"):
        for ln in open("/work/sp/distill/manifest.jsonl"):
            try:
                r = json.loads(ln)
                mf[r["id"]] = r
            except Exception:
                pass
    by_src = {}
    for x in D["items"]:
        r = mf.get(x["id"], {})
        s = r.get("src", x.get("src", "unknown"))
        by_src.setdefault(s, []).append(x)
    corpus_audit = {}
    for s, items_s in sorted(by_src.items()):
        hrs = sum(len(x["wav"]) for x in items_s) / SR / 3600.0
        samp = items_s[:12]
        sp_pro, sp_v1 = [], []
        for x in samp:
            w = x["wav"].float().numpy() / 32767.0
            e_ec = C.spk_emb(w, SR)
            sp_pro.append(float(np.dot(e_ec, c_pro_ecapa)))
            sp_v1.append(float(np.dot(e_ec, c_v1_ecapa)))
        corpus_audit[s] = dict(
            count=len(items_s),
            hours=round(hrs, 3),
            sim_ProV3_ecapa=round(float(np.mean(sp_pro)), 4),
            sim_OldV1_ecapa=round(float(np.mean(sp_v1)), 4),
            margin_ProV3_OldV1=round(float(np.mean(sp_pro) - np.mean(sp_v1)), 4),
        )

    out_path = os.path.join(REPO, "eval", "diag_2x2_session75.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    report = dict(
        session=75,
        timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        elapsed_sec=round(time.time() - t0, 1),
        n_clips=len(suite),
        summary_2x2=summary_2x2,
        corpus_lineage_audit=corpus_audit,
        per_clip=per_clip,
    )
    json.dump(report, open(out_path, "w"), indent=2, ensure_ascii=False)
    print(json.dumps(dict(summary_2x2=summary_2x2, corpus_lineage_audit=corpus_audit), indent=2), flush=True)


if __name__ == "__main__":
    main()
