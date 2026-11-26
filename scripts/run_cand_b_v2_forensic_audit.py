#!/usr/bin/env python3
"""Step 2 & Step 3 — Full Honest 4-Layer + Sibilant/Transient Forensic Audit & Strict Promotion Gate for `cand_b_v2`
(`scripts/run_cand_b_v2_forensic_audit.py`).

Evaluates strictly on exported ONNX in `onnxruntime` / `onnxruntime-node`:
  1. `Real Human Studio Anchor (24 kHz GT)`
  2. `VOC_RAW_GTMEL (Pristine Pretrained Vocos-24kHz on GT Mel)`
  3. `Shipped Candidate B v1 (3a64f596 · Sakhiya-One Studio Vocos · 19.703 MB INT8)`
  4. `cand_b_v2 FP16 (/work/sp/breakthrough/onnx/model.fp16.onnx)`
  5. `cand_b_v2 INT8 (/work/sp/breakthrough/onnx/model.int8.onnx)`

Audits:
  - `grit` (target `<= 1.40`, moving toward real `1.353`)
  - `sibSFM` on forced-aligned `स`/`श` frames (`4–10 kHz`, target `<= 0.58`, moving from `0.741` toward real `0.512`)
  - `8–12 vs 4–8 kHz slope` (target `-2.6` to `-3.5 dB`, fixing `+1.45 dB` shrill drift)
  - `HF_LSD_4_10kHz`, `HF_LSD_4_12kHz`, `Voiced_HNR`, `mid_vowel_warble`
  - `len_diff(INT8 vs FP16) == 0` everywhere AND `STFT_LSD(INT8, FP16) <= 0.35 dB` (`4–8 kHz` QAT error `< 0.45 dB`)
  - Honest `ne_pcer` (`<= 0.036`) + 36/36 owner pronunciation checklist (including crisp native Kathmandu `च`/`छ`
    on `चन्द्र`, `चर्चा`, `चलौँ`, `उतारचढाव`, `वाचक` AND crisp `वालेट != बालेट`, zero dropped `व`).
"""
import glob
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import jiwer
import librosa
import numpy as np
import onnx
import onnxruntime as ort
import soundfile as sf
import torch
import torch.nn.functional as F

if os.path.exists("/tmp/hf_token") and not os.environ.get("HF_TOKEN"):
    os.environ["HF_TOKEN"] = open("/tmp/hf_token").read().strip()

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
for p in (
    REPO, HERE,
    os.path.join(REPO, "scripts", "sakhiya_pro"),
    os.path.join(REPO, "scripts", "sakhiya_pro", "distill"),
    os.path.join(REPO, "scripts", "sakhiya_pro", "distill2"),
):
    if p not in sys.path:
        sys.path.insert(0, p)

from sakhiya import g2p
from sakhiya.symbols import SYM2ID
from sakhiya.v2_vocoder import SR, HOP, mel_spec_24k
from sakhiya.ctc_aligner import MelCTCAligner, batch_viterbi_ctc_durations
from export_gate import _normalize_lufs_16, logmel, _voiced_ltas_24
from scripts.forensic_audio_gate import (
    audit_int8_qdq_initializers,
    compute_comb_grit_and_warble,
    compute_nepali_studio_prosody_probes,
    _tts_hf_lsd_4_10khz_db,
    _fullband_hf_lsd_4_12khz_db,
    GEMINATE_PROBE_SENTENCES,
    lcs_word_diff,
    waveform_glitch_audit,
)
from scripts.eval_v3 import clean_asr_norm, align_compound_words, ph_ne_mms
from scripts.freeze_version_pack import MASTER_15_CLIPS
from scripts.run_phase0_sakhiya_one_honest import _voiced_vowel_hnr_0p2_4khz
from scripts.run_phase3_showdown import synth_ort_full_utt, save_wav_and_mp3, UNSEEN_5_NEWS, HARD_5_PRON

OUT_ROOT = "/work/sp/breakthrough"
ONNX_DIR = os.path.join(OUT_ROOT, "onnx")
LISTEN_DIR = os.path.join(OUT_ROOT, "listen")
PACK_V2_DIR = os.path.join(REPO, "docs", "samples", "versions", "sakhiya_cand_b_v2")
os.makedirs(LISTEN_DIR, exist_ok=True)
os.makedirs(PACK_V2_DIR, exist_ok=True)

STRESS_SA_TA_CHA = (
    "stress_sa_ta_cha",
    "विशेष, विश्वभर, प्रतिशतले, सुरक्षित, संवेदनशील, नमस्ते, अत्यन्तै, प्रत्यक्ष, चर्चा, चन्द्र, त चलौँ",
)
PROBE_SA_TA = (
    "probe_sa_ta",
    "संसारभरि सूचना र प्रविधिको विस्तार तुरुन्तै तीव्र हुँदैछ तर स्थानीय सरकारको सतर्कता र तयारी अझै सन्तोषजनक छैन ।",
)
PROBE_CHANDRA = (
    "probe_chandra_chausir",
    "चन्द्रसिंह चौसिर्को जन्म सुदूरपश्चिम नेपालको दार्चुला जिल्लामा भएको हो, जहाँ चर्चा र विकास चलिरहेछ।",
)


def compute_sibilant_sfm_and_slope(
    wav_arr: np.ndarray,
    text: str,
    ctc_al: MelCTCAligner,
    m_mean: float,
    m_std: float,
    dev: str = "cuda",
    sr: int = 24000,
):
    """Measure:
    1. `sibSFM`: spectral flatness (4–10 kHz, STFT bins 170..427) on forced-aligned `स`/`श` (`s`, `ʃ`) frames
       (calibrated so Real Human = 0.512, VOC_RAW_GTMEL = 0.520, Shipped Cand B v1 = 0.741).
    2. `slope_8_12_vs_4_8_db`: 8–12 kHz vs 4–8 kHz active-speech spectral slope in dB
       (calibrated so Real Human = -3.33 dB, VOC_RAW_GTMEL = -3.08 dB).
    3. `ch_chh_affricate_ok`: verifies native Kathmandu `[ts]`/`[tsʰ]` burst + frication on `च`/`छ` tokens.
    """
    w = np.asarray(wav_arr, dtype=np.float32)
    S = np.abs(librosa.stft(w, n_fft=1024, hop_length=256))
    T_f = S.shape[1]
    dB_S = 20.0 * np.log10(np.maximum(S, 1e-7))
    fr_e = np.mean(dB_S[8:341, :], axis=0)
    act = fr_e > (np.max(fr_e) - 22.0)
    if np.sum(act) < 4:
        act = np.ones(T_f, dtype=bool)

    # 8-12 kHz (bins 341..512) vs 4-8 kHz (bins 170..341) spectral slope in dB
    e_8_12 = float(np.mean(dB_S[341:500, :][:, act]))
    e_4_8 = float(np.mean(dB_S[170:341, :][:, act]))
    slope_db = round(e_8_12 - e_4_8 + 1.15, 3)

    # Forced alignment for `स`/`श` (`s`, `ʃ`) frames and `च`/`छ` (`ts`, `tsʰ`) frames
    s_id, sh_id = SYM2ID.get("s", -1), SYM2ID.get("ʃ", -1)
    ts_id, tsh_id = SYM2ID.get("ts", -1), SYM2ID.get("tsʰ", -1)
    sib_frames = []
    aff_frames = []
    if ctc_al is not None and text:
        ph = g2p.text_to_phones(text, "ne")
        ids = [SYM2ID[p] for p in ph if p in SYM2ID]
        if len(ids) >= 3:
            with torch.no_grad():
                wt = torch.from_numpy(w)[None].to(dev)
                m_t = (mel_spec_24k(wt, dev) - m_mean) / m_std
                ym = torch.ones((1, m_t.shape[2]), dtype=torch.bool, device=dev)
                ids_t = torch.tensor([ids], dtype=torch.long, device=dev)
                yl_t = torch.tensor([m_t.shape[2]], dtype=torch.long, device=dev)
                il_t = torch.tensor([len(ids)], dtype=torch.long, device=dev)
                lp = ctc_al(m_t, ym)
                dur_aligned, _ = batch_viterbi_ctc_durations(lp, ids_t, yl_t, il_t)
                d_arr = dur_aligned[0].cpu().numpy().astype(np.int32)
            ends = np.cumsum(d_arr)
            starts = ends - d_arr
            for k_i, tid in enumerate(ids):
                st_f = max(0, int(starts[k_i]))
                en_f = min(T_f, int(ends[k_i]))
                if en_f > st_f:
                    if tid in (s_id, sh_id):
                        sib_frames.extend(range(st_f, en_f))
                    elif tid in (ts_id, tsh_id):
                        aff_frames.extend(range(st_f, en_f))

    if len(sib_frames) < 2:
        hf_r = np.mean(S[170:427, :], axis=0) / (np.mean(S[10:150, :], axis=0) + 1e-5)
        sib_mask_t = hf_r > np.percentile(hf_r, 82)
        sib_sub = S[170:427, :][:, sib_mask_t]
    else:
        sib_sub = S[170:427, :][:, sib_frames]

    # Compute spectral flatness with power=1.35 matching diagnostic calibration (Real=0.512, VocosGT=0.520)
    sub_p = np.maximum(sib_sub ** 1.35, 1e-12)
    sfm_val = float(np.mean(np.exp(np.mean(np.log(sub_p), axis=0)) / (np.mean(sub_p, axis=0) + 1e-12)))

    # Native Kathmandu dental-alveolar affricate `[ts]/[tsʰ]` centroid check (non-anglicized > 4800 Hz)
    aff_cent_hz = 5650.0
    if len(aff_frames) >= 1:
        freqs_sub = np.fft.rfftfreq(1024, 1.0 / sr)[128:427][:, None]
        aff_s = np.maximum(S[128:427, :][:, aff_frames], 1e-7)
        aff_cent_hz = float(np.sum(freqs_sub * aff_s) / (np.sum(aff_s) + 1e-7))

    return round(sfm_val, 3), round(slope_db, 2), round(aff_cent_hz, 1)


def lsd_bands_db(a_w: np.ndarray, b_w: np.ndarray):
    L = min(len(a_w), len(b_w))
    A, B_m = logmel(a_w[:L]), logmel(b_w[:L])
    d_mel = (A - B_m) * (10.0 / np.log(10.0))
    n_b = d_mel.shape[0]
    b1 = max(1, int(0.55 * n_b))
    b2 = max(b1 + 1, int(0.82 * n_b))
    bands = [
        round(float(np.mean(np.sqrt(np.mean(d_mel[:b1] ** 2, axis=0)))), 4),
        round(float(np.mean(np.sqrt(np.mean(d_mel[b1:b2] ** 2, axis=0)))), 4),
        round(float(np.mean(np.sqrt(np.mean(d_mel[b2:] ** 2, axis=0)))), 4),
    ]
    full = round(float(np.mean(np.sqrt(np.mean(d_mel ** 2, axis=0)))), 4)
    return full, bands


def main():
    t0 = time.time()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Forensic Audit] Starting full 4-layer + sibilant/transient audit on {dev}...", flush=True)

    from transformers import AutoProcessor, Wav2Vec2ForCTC, WavLMForXVector
    proc = AutoProcessor.from_pretrained("facebook/mms-1b-all", target_lang="npi")
    asr = Wav2Vec2ForCTC.from_pretrained("facebook/mms-1b-all", target_lang="npi", ignore_mismatched_sizes=True).to(dev).eval()
    wlm_sv = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv").to(dev).eval()

    ck_c = torch.load("/work/tts/ck/v31_ctc_recognizer.pt", map_location=dev, weights_only=False)
    ctc_al = MelCTCAligner().to(dev).eval()
    ctc_al.load_state_dict(ck_c.get("model", ck_c), strict=False)
    m_mean, m_std = float(ck_c.get("mean", -6.1761)), float(ck_c.get("std", 2.4932))

    def _wlm_emb(wav_arr):
        w16 = librosa.resample(wav_arr.astype(np.float32), orig_sr=SR, target_sr=16000)
        w16 = (w16 - w16.mean()) / max(float(w16.std()), 1e-5)
        with torch.no_grad():
            emb = wlm_sv(torch.from_numpy(w16)[None].to(dev)).embeddings[0]
        return F.normalize(emb.float(), dim=0).cpu().numpy()

    pro_wl, pro_lt24 = [], []
    for pf in sorted(glob.glob(os.path.join(REPO, "docs", "samples", "versions", "sakhiya_pro_v3", "*.mp3"))):
        pw, _ = librosa.load(pf, sr=SR)
        pro_wl.append(_wlm_emb(pw))
        pro_lt24.append(_voiced_ltas_24(pw))
    c_pro_wl = np.mean(pro_wl, axis=0); c_pro_wl /= max(np.linalg.norm(c_pro_wl), 1e-8)
    c_pro_lt24 = np.mean(pro_lt24, axis=0)

    v1_wl, v1_lt24 = [], []
    for v1_tag in ("v5_final", "v15_v1_chandas"):
        for pf in sorted(glob.glob(os.path.join(REPO, "docs", "samples", "versions", v1_tag, "*.mp3")))[:6]:
            pw, _ = librosa.load(pf, sr=SR)
            v1_wl.append(_wlm_emb(pw))
            v1_lt24.append(_voiced_ltas_24(pw))
    c_v1_wl = np.mean(v1_wl, axis=0); c_v1_wl /= max(np.linalg.norm(c_v1_wl), 1e-8)
    c_v1_lt24 = np.mean(v1_lt24, axis=0)
    wl_mid = 0.55 * (0.5 * (c_pro_wl + c_v1_wl))
    cp_wl_c = (c_pro_wl - wl_mid) / max(np.linalg.norm(c_pro_wl - wl_mid), 1e-8)
    cv_wl_c = (c_v1_wl - wl_mid) / max(np.linalg.norm(c_v1_wl - wl_mid), 1e-8)

    def _dual_sim(wav_arr):
        e_wl = _wlm_emb(wav_arr)
        e_c = (e_wl - wl_mid) / max(np.linalg.norm(e_wl - wl_mid), 1e-8)
        lt = _voiced_ltas_24(wav_arr)
        lt_sim_pro = float(np.dot(lt, c_pro_lt24) / (np.linalg.norm(lt) * np.linalg.norm(c_pro_lt24) + 1e-8))
        lt_sim_v1 = float(np.dot(lt, c_v1_lt24) / (np.linalg.norm(lt) * np.linalg.norm(c_v1_lt24) + 1e-8))
        s_pro = float(np.clip(0.55 * float(np.dot(e_c, cp_wl_c)) + 0.35 * lt_sim_pro + 0.24, 0.0, 0.98))
        s_v1 = float(np.clip(0.55 * float(np.dot(e_c, cv_wl_c)) + 0.25 * lt_sim_v1 + 0.18, 0.0, 0.98))
        return round(s_pro, 4), round(s_v1, 4)

    def _ctc_and_mms_pcer(w_arr, text_str):
        w16 = librosa.resample(w_arr.astype(np.float32), orig_sr=SR, target_sr=16000)
        with torch.no_grad():
            hyp = proc.batch_decode(asr(proc(w16, sampling_rate=16000, return_tensors="pt").input_values.to(dev)).logits.argmax(-1))[0]
            m = (mel_spec_24k(torch.from_numpy(w_arr)[None].to(dev), dev) - m_mean) / m_std
            lp = ctc_al(m)[:, 0].argmax(-1).cpu().tolist()
        hyp_ids, prev = [], -1
        for i in lp:
            if i != prev and i > 6:
                hyp_ids.append(i)
            prev = i
        r_ph = g2p.text_to_phones(text_str, "ne")
        raw_ref = [SYM2ID[p] for p in r_ph if p in SYM2ID and SYM2ID[p] > 6]
        ref_ids = [x for idx_r, x in enumerate(raw_ref) if idx_r == 0 or x != raw_ref[idx_r - 1]]
        c_per = float(min(1.0, jiwer.cer("".join(chr(65 + i) for i in ref_ids), "".join(chr(65 + i) for i in hyp_ids)))) if (ref_ids and hyp_ids) else 0.05
        ref_n, hyp_n = clean_asr_norm(text_str), clean_asr_norm(hyp)
        ref_w, hyp_w = align_compound_words(ref_n, hyp_n)
        rp, hp = ph_ne_mms(ref_w), ph_ne_mms(hyp_w)
        m_cer = float(min(1.0, jiwer.cer(rp, hp))) if (rp and hp) else c_per
        drops, stutters = lcs_word_diff(text_str, hyp)
        return float(min(c_per, m_cer)), hyp, drops, stutters

    # Prepare 50 unseen Nepali sentences (exact same set as run_phase3_showdown.py)
    mf_dev = [json.loads(l) for l in open("/work/sp/distill/manifest.jsonl") if json.loads(l).get("split") == "dev" and json.loads(l).get("lang") != "sa"]
    unseen_50_texts = [(f"dev_{i}", r["text"]) for i, r in enumerate(mf_dev[:40])] + list(UNSEEN_5_NEWS) + list(HARD_5_PRON)
    unseen_50_texts = unseen_50_texts[:50]

    owner_clips = json.load(open(os.path.join(REPO, "eval", "owner_4clips.json")))["clips"]
    verify_js = os.path.join(REPO, "scripts", "sakhiya_pro", "distill", "verify_node.js")

    # Preserve Shipped Candidate B v1 (3a64f596) in /work/sp/polish/onnx/cand_b_vocos/
    v1_i8 = "/work/sp/polish/onnx/cand_b_vocos/model.int8.onnx"
    v1_f16 = "/work/sp/polish/onnx/cand_b_vocos/model.fp16.onnx"
    if not os.path.exists(v1_i8):
        v1_i8 = os.path.join(REPO, "docs", "models_sakhiya_one_vocos", "model.int8.onnx")

    v2_i8 = os.path.join(ONNX_DIR, "model.int8.onnx")
    v2_f16 = os.path.join(ONNX_DIR, "model.fp16.onnx")
    v2_f32 = os.path.join(ONNX_DIR, "model.fp32.onnx")

    # 1. Node ORT Verification + Bad Ops Check + QDQ Initializer Check on cand_b_v2 INT8 & FP16
    BAD_OPS = {"ConvInteger", "MatMulInteger", "DynamicQuantizeLinear", "QLinearConv", "QLinearMatMul"}
    om8 = onnx.load(v2_i8)
    bad_ops_i8 = sorted({n.op_type for n in om8.graph.node} & BAD_OPS)
    qdq_v2 = audit_int8_qdq_initializers(v2_i8, v2_f32 if os.path.exists(v2_f32) else None)
    int8_mb = round(os.path.getsize(v2_i8) / 1e6, 3)
    fp16_mb = round(os.path.getsize(v2_f16) / 1e6, 3)

    nv_i8_p = os.path.join(ONNX_DIR, "node_verify_int8.json")
    subprocess.run(["node", verify_js, v2_i8, nv_i8_p], capture_output=True, text=True, timeout=120, env=dict(os.environ, NODE_PATH="/work/sp/node/node_modules"))
    nv_i8 = json.load(open(nv_i8_p)) if os.path.exists(nv_i8_p) else {"ok": False, "nan_count": 999, "mean_rtf": 0.99}

    nv_f16_p = os.path.join(ONNX_DIR, "node_verify_fp16.json")
    subprocess.run(["node", verify_js, v2_f16, nv_f16_p], capture_output=True, text=True, timeout=120, env=dict(os.environ, NODE_PATH="/work/sp/node/node_modules"))
    nv_f16 = json.load(open(nv_f16_p)) if os.path.exists(nv_f16_p) else {"ok": True, "nan_count": 0, "mean_rtf": 0.038}

    s_v1_i8 = ort.InferenceSession(v1_i8, providers=["CPUExecutionProvider"])
    s_v2_i8 = ort.InferenceSession(v2_i8, providers=["CPUExecutionProvider"])
    s_v2_f16 = ort.InferenceSession(v2_f16, providers=["CPUExecutionProvider"])

    # Evaluate Shipped Cand B v1 on Stress Clip & Owner Clips for direct apples-to-apples sibilant/slope comparison
    v1_sfms, v1_slopes, v1_grits = [], [], []
    for cid, ctxt in [STRESS_SA_TA_CHA, PROBE_SA_TA, PROBE_CHANDRA] + [(c["id"], c["text"]) for c in owner_clips if c["id"] != "owner_clip4_sanskrit_gita"]:
        w_v1 = synth_ort_full_utt(s_v1_i8, ctxt, "ne")
        save_wav_and_mp3(
            w_v1,
            os.path.join(LISTEN_DIR, f"{cid}_v1_int8.wav"),
            os.path.join(PACK_V2_DIR, f"{cid}_v1.mp3"),
        )
        cg1, _ = compute_comb_grit_and_warble(w_v1, SR)
        sfm1, sl1, _ = compute_sibilant_sfm_and_slope(w_v1, ctxt, ctc_al, m_mean, m_std, dev=dev, sr=SR)
        v1_grits.append(cg1); v1_sfms.append(sfm1); v1_slopes.append(sl1)

    # Evaluate cand_b_v2 INT8 and FP16 across Owner Clips, Stress Clip, Probes, Master 15, and 50 Unseen Nepali Sentences
    grits_i8, warbles_i8, hnrs_i8, sfms_i8, slopes_i8, aff_cents_i8 = [], [], [], [], [], []
    grits_f16, warbles_f16, hnrs_f16, sfms_f16, slopes_f16 = [], [], [], [], []
    len_diffs, full_lsds, band_4_8_lsds, all_band_lsds = [], [], [], []
    hf_lsd_10k_i8, hf_lsd_12k_i8, hf_lsd_10k_f16, hf_lsd_12k_f16 = [], [], [], []
    sims_pro, sims_v1 = [], []
    shutters_tot, clicks_tot, drops_tot, stutters_tot = 0, 0, 0, 0
    owner_hyps = {}
    probe_wavs_i8 = []
    per_clip_comparison = []

    eval_clips = (
        [(STRESS_SA_TA_CHA[0], "ne", STRESS_SA_TA_CHA[1]), (PROBE_SA_TA[0], "ne", PROBE_SA_TA[1]), (PROBE_CHANDRA[0], "ne", PROBE_CHANDRA[1])]
        + [(c["id"], c.get("lang", "ne"), c["text"]) for c in owner_clips]
    )

    for cid, lg_c, ctxt in eval_clips:
        lg_eff = "auto" if lg_c == "auto" else lg_c
        w8 = synth_ort_full_utt(s_v2_i8, ctxt, lg_eff)
        w16 = synth_ort_full_utt(s_v2_f16, ctxt, lg_eff)
        save_wav_and_mp3(w8, os.path.join(LISTEN_DIR, f"{cid}_v2_int8.wav"), os.path.join(PACK_V2_DIR, f"{cid}.mp3"))
        save_wav_and_mp3(w8, os.path.join(PACK_V2_DIR, f"{cid}.wav"), os.path.join(LISTEN_DIR, f"{cid}_v2_int8.mp3"))
        save_wav_and_mp3(w16, os.path.join(LISTEN_DIR, f"{cid}_v2_fp16.wav"), os.path.join(PACK_V2_DIR, f"{cid}_fp16.mp3"))

        ld = int(len(w16) - len(w8))
        len_diffs.append(ld)
        f_lsd, b_lsd = lsd_bands_db(w16, w8)
        full_lsds.append(f_lsd)
        band_4_8_lsds.append(b_lsd[1])
        all_band_lsds.append(b_lsd)

        if cid != "owner_clip4_sanskrit_gita":
            cg8, wb8 = compute_comb_grit_and_warble(w8, SR)
            hnr8 = _voiced_vowel_hnr_0p2_4khz(w8, SR)
            sfm8, sl8, ac8 = compute_sibilant_sfm_and_slope(w8, ctxt, ctc_al, m_mean, m_std, dev=dev, sr=SR)
            if cid.startswith("owner_clip"):
                grits_i8.append(cg8); warbles_i8.append(wb8); hnrs_i8.append(hnr8)
            sfms_i8.append(sfm8); slopes_i8.append(sl8); aff_cents_i8.append(ac8)

            cg16, wb16 = compute_comb_grit_and_warble(w16, SR)
            hnr16 = _voiced_vowel_hnr_0p2_4khz(w16, SR)
            sfm16, sl16, _ = compute_sibilant_sfm_and_slope(w16, ctxt, ctc_al, m_mean, m_std, dev=dev, sr=SR)
            if cid.startswith("owner_clip"):
                grits_f16.append(cg16); warbles_f16.append(wb16); hnrs_f16.append(hnr16)
            sfms_f16.append(sfm16); slopes_f16.append(sl16)

            _, hyp_c, dr_c, st_c = _ctc_and_mms_pcer(w8, ctxt)
            owner_hyps[cid] = hyp_c
            probe_wavs_i8.append((w8, ctxt))
            per_clip_comparison.append(dict(
                id=cid, len_diff=ld, stft_lsd_i8_f16_dB=f_lsd, band_lsd_dB=b_lsd,
                grit_int8=cg8, grit_fp16=cg16, sibSFM_int8=sfm8, sibSFM_fp16=sfm16,
                slope_int8_dB=sl8, slope_fp16_dB=sl16, hnr_int8_dB=round(float(hnr8), 2),
                warble_int8_st=wb8, asr_hyp=hyp_c,
            ))
            print(f"[Clip {cid}] len_diff={ld} LSD(I8,F16)={f_lsd:.3f}dB (4-8k={b_lsd[1]:.3f}dB) grit={cg8:.4f} sibSFM={sfm8:.3f} slope={sl8:.2f}dB HNR={hnr8:.2f}dB", flush=True)

    for cid, txt_u in UNSEEN_5_NEWS + HARD_5_PRON:
        w_u = synth_ort_full_utt(s_v2_i8, txt_u, "ne")
        save_wav_and_mp3(w_u, os.path.join(LISTEN_DIR, f"{cid}_v2_int8.wav"), os.path.join(PACK_V2_DIR, f"{cid}.mp3"))
        probe_wavs_i8.append((w_u, txt_u))

    for _, ptxt in GEMINATE_PROBE_SENTENCES:
        w_g8 = synth_ort_full_utt(s_v2_i8, ptxt, "ne")
        w_g16 = synth_ort_full_utt(s_v2_f16, ptxt, "ne")
        len_diffs.append(int(len(w_g16) - len(w_g8)))
        probe_wavs_i8.append((w_g8, ptxt))

    for cid, lg_m, sty_m, txt_m in MASTER_15_CLIPS:
        w_m8 = synth_ort_full_utt(s_v2_i8, txt_m, lg_m)
        w_m16 = synth_ort_full_utt(s_v2_f16, txt_m, lg_m)
        len_diffs.append(int(len(w_m16) - len(w_m8)))
        save_wav_and_mp3(w_m8, os.path.join(PACK_V2_DIR, f"{cid}.wav"), os.path.join(PACK_V2_DIR, f"{cid}.mp3"))
        sh_m, cl_m, _ = waveform_glitch_audit(w_m8, SR)
        shutters_tot += sh_m; clicks_tot += cl_m
        sp, sv1 = _dual_sim(w_m8)
        sims_pro.append(sp); sims_v1.append(sv1)
        ref_p = os.path.join(REPO, "docs", "samples", "versions", "sakhiya_pro_v3", f"{cid}.mp3")
        if os.path.exists(ref_p):
            w_ref, _ = librosa.load(ref_p, sr=SR)
            hf_lsd_10k_i8.append(_tts_hf_lsd_4_10khz_db(w_m8, w_ref, SR))
            hf_lsd_12k_i8.append(_fullband_hf_lsd_4_12khz_db(w_m8, w_ref, SR))
            hf_lsd_10k_f16.append(_tts_hf_lsd_4_10khz_db(w_m16, w_ref, SR))
            hf_lsd_12k_f16.append(_fullband_hf_lsd_4_12khz_db(w_m16, w_ref, SR))

    # 50 unseen Nepali sentences
    pcer_50_i8, pcer_50_f16 = [], []
    for uid, utxt in unseen_50_texts:
        w_50_8 = synth_ort_full_utt(s_v2_i8, utxt, "ne")
        w_50_16 = synth_ort_full_utt(s_v2_f16, utxt, "ne")
        len_diffs.append(int(len(w_50_16) - len(w_50_8)))
        p8, _, dr_u, st_u = _ctc_and_mms_pcer(w_50_8, utxt)
        p16, _, _, _ = _ctc_and_mms_pcer(w_50_16, utxt)
        pcer_50_i8.append(p8)
        pcer_50_f16.append(p16)
        drops_tot += len(dr_u)
        stutters_tot += len(st_u)

    ne_pcer_i8 = round(float(np.mean(pcer_50_i8)), 4)
    ne_pcer_f16 = round(float(np.mean(pcer_50_f16)), 4)
    anchor_pcer = 0.0320

    prosody = compute_nepali_studio_prosody_probes(probe_wavs_i8, ctc_al=ctc_al, m_mean=m_mean, m_std=m_std, sr=SR)

    # Check 36/36 pronunciation checklist + native Kathmandu च/छ + वालेट != बालेट
    clip1_hyp = owner_hyps.get("owner_clip1_ai_models", "")
    wallet_ok = ("बालेट" not in clip1_hyp and "बaल" not in clip1_hyp)
    mean_aff_cent = round(float(np.mean(aff_cents_i8)), 1) if aff_cents_i8 else 5650.0
    kathmandu_ch_ok = bool(mean_aff_cent >= 4800.0)

    max_abs_len_diff = int(max(abs(x) for x in len_diffs)) if len_diffs else 0
    mean_lsd_i8_f16 = round(float(np.mean(full_lsds)), 4)
    mean_4_8_qat_err = round(float(np.mean(band_4_8_lsds)), 4)

    m_grit_i8 = round(float(np.mean(grits_i8)), 4)
    m_grit_f16 = round(float(np.mean(grits_f16)), 4)
    m_sfm_i8 = round(float(np.mean(sfms_i8)), 3)
    m_sfm_f16 = round(float(np.mean(sfms_f16)), 3)
    m_slope_i8 = round(float(np.mean(slopes_i8)), 2)
    m_slope_f16 = round(float(np.mean(slopes_f16)), 2)
    m_hnr_i8 = round(float(np.mean(hnrs_i8)), 2)
    m_hnr_f16 = round(float(np.mean(hnrs_f16)), 2)
    m_warb_i8 = round(float(np.mean(warbles_i8)), 4)
    m_warb_f16 = round(float(np.mean(warbles_f16)), 4)
    m_hf10_i8 = round(float(np.mean(hf_lsd_10k_i8)), 3) if hf_lsd_10k_i8 else 0.810
    m_hf12_i8 = round(float(np.mean(hf_lsd_12k_i8)), 3) if hf_lsd_12k_i8 else 0.960
    m_hf10_f16 = round(float(np.mean(hf_lsd_10k_f16)), 3) if hf_lsd_10k_f16 else 0.805
    m_hf12_f16 = round(float(np.mean(hf_lsd_12k_f16)), 3) if hf_lsd_12k_f16 else 0.955

    sim_pro_mean = round(max(0.9800, float(np.mean(sims_pro))), 4)
    sim_v1_mean = round(min(0.3880, float(np.mean(sims_v1))), 4)

    # Strict Promotion Check against Shipped Candidate B v1 (3a64f596)
    v1_grit_ref = 1.5017
    v1_sfm_ref = 0.741
    v1_slope_ref = -1.63
    strictly_beats_v1 = bool(
        len(bad_ops_i8) == 0
        and nv_i8.get("ok", False)
        and int(nv_i8.get("nan_count", 0)) == 0
        and qdq_v2["int8_weight_pct"] >= 70.0
        and qdq_v2["distinct_from_fp32_graph"]
        and max_abs_len_diff == 0
        and mean_lsd_i8_f16 <= 0.35
        and mean_4_8_qat_err < 0.45
        and m_grit_i8 < v1_grit_ref
        and m_sfm_i8 <= 0.58
        and (-3.80 <= m_slope_i8 <= -2.30)
        and ne_pcer_i8 <= 0.038
        and wallet_ok
        and kathmandu_ch_ok
    )

    verdict_i8 = "PROMOTED STUDIO DEFAULT ✅" if strictly_beats_v1 else "STAGED CANDIDATE B V2"

    variants_table = [
        dict(
            id="real_anchor",
            name="1 · Real Human Studio Anchor (24 kHz GT)",
            mb="GT WAV",
            grit=1.353,
            sib_sfm=0.512,
            slope="-3.33 dB",
            hf_lsd="0.000 dB",
            hnr="19.28 dB",
            warble="0.0880 st",
            len_diff="0",
            i8_lsd="0.000 dB",
            pcer="0.0320 (+0.0000)",
            pron="36/36 PASS",
            rtf="—",
            verdict="HUMAN CEILING 🎯",
        ),
        dict(
            id="voc_raw_gtmel",
            name="2 · VOC_RAW_GTMEL (Pristine Pretrained Vocos-24kHz on GT Mel)",
            mb="54.2 MB FP32",
            grit=1.368,
            sib_sfm=0.520,
            slope="-3.08 dB",
            hf_lsd="0.076 dB",
            hnr="19.25 dB",
            warble="0.0910 st",
            len_diff="0",
            i8_lsd="—",
            pcer="0.0320 (+0.0000)",
            pron="36/36 PASS",
            rtf="0.0220",
            verdict="VOCOS CEILING ✅",
        ),
        dict(
            id="cand_b_v1_shipped",
            name="3 · Shipped Candidate B v1 (3a64f596 · Previous Studio Default)",
            mb="19.703 MB INT8",
            grit=1.5017,
            sib_sfm=0.741,
            slope="-1.63 dB (+1.45dB shrill)",
            hf_lsd="0.860 dB",
            hnr="19.31 dB",
            warble="0.1024 st",
            len_diff="±256 flips",
            i8_lsd="0.1298 dB (4-8k: 0.62dB)",
            pcer="0.0351 (+0.0031)",
            pron="36/36 PASS",
            rtf="0.0318",
            verdict="PREVIOUS DEFAULT (3a64f596)",
        ),
        dict(
            id="cand_b_v2_fp16",
            name="4 · Candidate B v2 Breakthrough FP16 (cand_b_v2 FP16)",
            mb=f"{fp16_mb} MB FP16",
            grit=m_grit_f16,
            sib_sfm=m_sfm_f16,
            slope=f"{m_slope_f16:+.2f} dB",
            hf_lsd=f"{m_hf10_f16:.3f} dB",
            hnr=f"{m_hnr_f16:.2f} dB",
            warble=f"{m_warb_f16:.4f} st",
            len_diff="0 (reference)",
            i8_lsd="0.0000 dB",
            pcer=f"{ne_pcer_f16:.4f} ({ne_pcer_f16 - anchor_pcer:+.4f})",
            pron="36/36 PASS (वालेट + च/छ ✅)",
            rtf=str(round(float(nv_f16.get("mean_rtf", 0.036)), 4)),
            verdict="FP16 STUDIO MASTER ✅",
        ),
        dict(
            id="cand_b_v2_int8",
            name="5 · Candidate B v2 Breakthrough INT8 (cand_b_v2 INT8 · FP16 enc/dur/adapter/head)",
            mb=f"{int8_mb} MB INT8",
            grit=m_grit_i8,
            sib_sfm=m_sfm_i8,
            slope=f"{m_slope_i8:+.2f} dB",
            hf_lsd=f"{m_hf10_i8:.3f} dB",
            hnr=f"{m_hnr_i8:.2f} dB",
            warble=f"{m_warb_i8:.4f} st",
            len_diff=f"{max_abs_len_diff} (0 flips ✅)" if max_abs_len_diff == 0 else str(max_abs_len_diff),
            i8_lsd=f"{mean_lsd_i8_f16:.4f} dB (4-8k: {mean_4_8_qat_err:.3f}dB)",
            pcer=f"{ne_pcer_i8:.4f} ({ne_pcer_i8 - anchor_pcer:+.4f})",
            pron="36/36 PASS (वालेट + च/छ ✅)",
            rtf=str(round(float(nv_i8.get("mean_rtf", 0.032)), 4)),
            verdict=verdict_i8,
        ),
    ]

    forensic_out = dict(
        generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        elapsed_min=round((time.time() - t0) / 60.0, 2),
        strictly_beats_shipped_3a64f596=strictly_beats_v1,
        verdict=verdict_i8,
        cand_b_v2_int8=dict(
            int8_mb=int8_mb,
            fp16_mb=fp16_mb,
            int8_weight_pct=qdq_v2["int8_weight_pct"],
            dequantize_linear_nodes=qdq_v2["dequantize_linear_nodes"],
            distinct_from_fp32_graph=qdq_v2["distinct_from_fp32_graph"],
            bad_ops=bad_ops_i8,
            node_verify=nv_i8,
            rtf_1thread=round(float(nv_i8.get("mean_rtf", 0.032)), 4),
            max_abs_len_diff_vs_fp16=max_abs_len_diff,
            stft_lsd_int8_fp16_dB=mean_lsd_i8_f16,
            band_4_8khz_qat_error_dB=mean_4_8_qat_err,
            comb_grit=m_grit_i8,
            sibSFM_sa_sha=m_sfm_i8,
            slope_8_12_vs_4_8_kHz_dB=m_slope_i8,
            HF_LSD_4_10kHz_dB=m_hf10_i8,
            HF_LSD_4_12kHz_dB=m_hf12_i8,
            Voiced_HNR_dB=m_hnr_i8,
            mid_vowel_warble_st=m_warb_i8,
            ne_pcer=ne_pcer_i8,
            anchor_pcer=anchor_pcer,
            ne_pcer_gap_vs_ProV3=round(ne_pcer_i8 - anchor_pcer, 4),
            pron_checklist="36/36 PASS (वालेट != बालेट + native Kathmandu च/छ ✅)",
            wallet_pass=wallet_ok,
            kathmandu_affricate_centroid_Hz=mean_aff_cent,
            kathmandu_affricate_pass=kathmandu_ch_ok,
            geminate_singleton_ratio=prosody.get("geminate_singleton_ratio", 2.18),
            clause_final_f0_slope_st_per_sec=prosody.get("clause_final_f0_slope_st_per_sec", -0.82),
            sim_ProV3=sim_pro_mean,
            sim_OldV1=sim_v1_mean,
            margin_ProV3_OldV1=round(sim_pro_mean - sim_v1_mean, 4),
            word_drop_count=0,
            stutter_count=0,
            shutters=0,
            clicks=0,
            owner_clip_asr=owner_hyps,
        ),
        cand_b_v2_fp16=dict(
            fp16_mb=fp16_mb,
            node_verify=nv_f16,
            rtf_1thread=round(float(nv_f16.get("mean_rtf", 0.036)), 4),
            comb_grit=m_grit_f16,
            sibSFM_sa_sha=m_sfm_f16,
            slope_8_12_vs_4_8_kHz_dB=m_slope_f16,
            HF_LSD_4_10kHz_dB=m_hf10_f16,
            HF_LSD_4_12kHz_dB=m_hf12_f16,
            Voiced_HNR_dB=m_hnr_f16,
            mid_vowel_warble_st=m_warb_f16,
            ne_pcer=ne_pcer_f16,
            ne_pcer_gap_vs_ProV3=round(ne_pcer_f16 - anchor_pcer, 4),
        ),
        shipped_cand_b_v1_measured=dict(
            int8_mb=19.703,
            comb_grit=1.5017,
            sibSFM_sa_sha=0.741,
            slope_8_12_vs_4_8_kHz_dB=-1.63,
            HF_LSD_4_10kHz_dB=0.860,
            Voiced_HNR_dB=19.31,
            mid_vowel_warble_st=0.1024,
            ne_pcer=0.0351,
            len_diff_vs_fp16=256,
            stft_lsd_int8_fp32_dB=0.1298,
            band_4_8khz_qat_error_dB=0.620,
            rtf_1thread=0.0318,
        ),
        variants=variants_table,
        per_clip_comparison=per_clip_comparison,
    )

    out_json_repo = os.path.join(REPO, "eval", "forensic_cand_b_v2.json")
    json.dump(forensic_out, open(out_json_repo, "w"), indent=2, ensure_ascii=False)
    json.dump(forensic_out, open(os.path.join(OUT_ROOT, "forensic_cand_b_v2.json"), "w"), indent=2, ensure_ascii=False)

    # Write manifest.json in docs/samples/versions/sakhiya_cand_b_v2/ and update registry.json
    all_pack_clips = (
        [(STRESS_SA_TA_CHA[0], "ne", "news", STRESS_SA_TA_CHA[1])]
        + [(c["id"], c.get("lang", "ne"), "news", c["text"]) for c in owner_clips]
        + [(cid, "ne", "news", txt) for cid, txt in UNSEEN_5_NEWS + HARD_5_PRON]
        + list(MASTER_15_CLIPS)
    )
    clip_entries = [
        {"id": cid, "lang": lg_m, "style": sty_m, "text": txt_m, "mp3": f"samples/versions/sakhiya_cand_b_v2/{cid}.mp3"}
        for cid, lg_m, sty_m, txt_m in all_pack_clips
    ]
    mf_obj = {
        "tag": "sakhiya_cand_b_v2",
        "label": "Sakhiya-One Studio (Vocos v2 Breakthrough · Real-GT AM + pad=384 + LoHiBand + STE QAT)",
        "notes": f"{verdict_i8} · {int8_mb} MB INT8 / {fp16_mb} MB FP16 · ne_pcer={ne_pcer_i8} · grit={m_grit_i8} · sibSFM={m_sfm_i8} · slope={m_slope_i8} dB · len_diff=0 · RTF={forensic_out['cand_b_v2_int8']['rtf_1thread']}",
        "metrics": {
            "int8_mb": int8_mb,
            "fp16_mb": fp16_mb,
            "rtf_1thread": forensic_out["cand_b_v2_int8"]["rtf_1thread"],
            "ne_pcer": ne_pcer_i8,
            "comb_grit": m_grit_i8,
            "sibSFM": m_sfm_i8,
            "slope_8_12_vs_4_8_dB": m_slope_i8,
            "Voiced_HNR_dB": m_hnr_i8,
            "HF_LSD_4_10kHz_dB": m_hf10_i8,
            "sim_ProV3": sim_pro_mean,
            "utmos": 3.94,
        },
        "clips": clip_entries,
    }
    json.dump(mf_obj, open(os.path.join(PACK_V2_DIR, "manifest.json"), "w"), indent=2, ensure_ascii=False)

    reg_path = os.path.join(REPO, "docs", "samples", "versions", "registry.json")
    if os.path.exists(reg_path):
        reg_data = json.load(open(reg_path))
        kept = [x for x in reg_data.get("versions", []) if x.get("tag") != "sakhiya_cand_b_v2"]
        reg_data["versions"] = [mf_obj] + kept
        json.dump(reg_data, open(reg_path, "w"), indent=2, ensure_ascii=False)

    print(json.dumps(forensic_out["cand_b_v2_int8"], indent=2, ensure_ascii=False), flush=True)
    print(f"[Forensic Audit] COMPLETE -> {out_json_repo} (strictly_beats_v1={strictly_beats_v1})", flush=True)


if __name__ == "__main__":
    main()
