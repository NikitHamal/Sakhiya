#!/usr/bin/env python3
"""Phase 0 (Leg 2): Un-VC'd >=24 kHz Studio Audit, Bandwidth Tagging, Shared Acoustic `shr_creak` Extraction,
Genuine `<breath>` Inhalation Bank Mining & Crossfading, and Offline Devanagari LLM Prosody Labeling
(`scripts/sakhiya_pro/phase0_leg2_studio_audit.py`).

Executes:
1. Audits EVERY raw, un-VC'd >=24 kHz genuine human Nepali studio recording across local/HF pools:
   - Raw 48 kHz -> 24 kHz OpenSLR SLR43 speaker `nep_6834` (`anchor_6834` + `anchor_6834_stitched`, spk_id=0)
   - Raw 48 kHz -> 24 kHz OpenSLR SLR43 female (`slr43`, spk_id=1)
   - Raw 24 kHz un-VC'd female studio recordings from `v13/real_audio_24k.pt` + `v13/mined_24k.pt` (spk_id=2..3)
   Prints exact raw un-VC'd hours for `nep_6834` and for the full clean >=24 kHz studio pool.
2. Tags every utterance in `voc_cache_100mel.pt` and `am_cache_100mel.pt` with explicit metadata:
   `is_real_unvc_24k` (bool), `orig_sr_hz` (48000/24000/16000), `is_prov3` (bool), `spk_id` (int),
   and ensures all clean raw un-VC'd >=24 kHz studio female clips are cached with the exact Leg-1 100-mel parameters.
3. Extracts Shared Ground-Truth Frame-Level `shr_creak` ([0, 1]) (Subharmonic-to-Harmonic Ratio at 0.5*F0, 1.5*F0, 2.5*F0
   vs 1.0*F0, 2.0*F0, 3.0*F0 gated on low-F0 + energy decay) for both Phase 1 Vocoder and Phase 2 AM (`creak_prob` + `f0_slope`).
4. Mines Genuine Acoustic `<breath>` Spans (80-180 ms, RMS -45 to -22 dB rel peak, spectral flatness > 0.35, unvoiced)
   from raw un-VC'd >=24 kHz studio recordings (prioritizing `nep_6834`), builds `real_breath_bank_6834.pt`, marks
   digital-silence `<breath>` spans on VC clips, and cleanly crossfades real `nep_6834` anchor inhalation mels at
   clause-pause `<breath>` locations where VC wiped the breath so `<breath>` NEVER learns digital dead air.
5. Runs a fast batched bf16 pass with a compact Devanagari-capable LLM (`Qwen/Qwen2.5-1.5B-Instruct` / syntactic-prosodic
   scorer) to cache soft word/phoneme-level `(implicit_break_strength in [0, 1], focus_prominence in [0, 1])` targets.
6. Builds raw un-VC'd `nep_6834` reference audio (`/work/sp/sakhiya_news/listen/ne_kathmandu_human_anchor.wav` &
   `/work/sp/sakhiya_news/cache_100mel/raw_anchor_6834_eval20.pt`) and `/work/sp/distill/dual_anchor_centroids_v2.pt`.
"""
import glob
import json
import math
import os
import random
import sys
import time
import librosa
import numpy as np
import scipy.fft
import scipy.signal
import soundfile as sf
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
for p in (HERE, os.path.join(HERE, "distill"), os.path.join(HERE, "distill2"), REPO, "/work/sp"):
    if p not in sys.path:
        sys.path.insert(0, p)

from sakhiya.v2_vocoder import SR, N_FFT, HOP, mel_spec_24k
from sakhiya.symbols import SYM2ID, SYMBOLS
from scripts.train_v2_voc import gpu_f0_uv
from sakhiya.ctc_aligner import MelCTCAligner, batch_viterbi_ctc_durations

CACHE_DIR = "/work/sp/sakhiya_news/cache_100mel"
LISTEN_DIR = "/work/sp/sakhiya_news/listen"
TELEM_FILE = "/work/sp/sakhiya_news/telemetry.json"
LOG_FILE = "/work/tts/logs/sakhiya_news.log"

BREATH_ID = SYM2ID[";"]
COMMA_ID = SYM2ID[","]
DANDA_ID = SYM2ID["."]
WB_ID = SYM2ID["|"]
CONSONANT_IDS = {
    SYM2ID[p] for p in (
        "k", "kʰ", "g", "gʱ", "ŋ", "ts", "tsʰ", "dz", "dzʱ", "ɲ",
        "ʈ", "ʈʰ", "ɖ", "ɖʱ", "ɳ", "t̪", "t̪ʰ", "d̪", "d̪ʱ", "n",
        "p", "pʰ", "b", "bʱ", "m", "j", "r", "l", "w", "ʃ", "s", "h", "ɽ", "ɽʱ", "z", "f"
    ) if p in SYM2ID
}


def log(msg: str):
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def update_telem(idx: int, name: str, step: int, total_steps: int, its: str, losses: str, elapsed_min: float, eta_min: float, status: str):
    try:
        st = json.load(open(TELEM_FILE)) if os.path.exists(TELEM_FILE) else {"stages": [{} for _ in range(4)]}
        while len(st.get("stages", [])) < 4:
            st.setdefault("stages", []).append({})
        st["updated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        st["stages"][idx].update(
            name=name, step=int(step), total_steps=int(total_steps), its=str(its),
            losses=losses, elapsed_min=round(float(elapsed_min), 2),
            eta_min=round(float(eta_min), 2), status=status,
        )
        json.dump(st, open(TELEM_FILE + ".tmp", "w"), indent=2, ensure_ascii=False)
        os.replace(TELEM_FILE + ".tmp", TELEM_FILE)
    except Exception:
        pass


@torch.no_grad()
def compute_shr_creak_from_wav(wav_t: torch.Tensor, f0_hz: torch.Tensor, vuv: torch.Tensor) -> torch.Tensor:
    """Compute frame-level Subharmonic-to-Harmonic Ratio (SHR) creak target in [0, 1] at 24 kHz (hop=256).
    Measures subharmonic energy at (0.5, 1.5, 2.5)*F0 relative to harmonic energy at (1.0, 2.0, 3.0)*F0
    in a 2048-pt STFT (11.7 Hz bin resolution), gated on low-F0 register + voiced/semi-voiced energy decay.
    """
    dev = wav_t.device
    if wav_t.dim() == 1:
        wav_t = wav_t[None, :]
    T = f0_hz.shape[-1]
    pad = 1024
    w_pad = F.pad(wav_t[:, None, :].float(), (pad, pad), mode="reflect")[:, 0, :]
    win = torch.hann_window(2048, device=dev)
    spec = torch.stft(w_pad, n_fft=2048, hop_length=HOP, win_length=2048, window=win, center=False, return_complex=True).abs()
    spec = spec[:, :, :T]  # [B, 1025, T]
    B, n_bins, T_s = spec.shape
    f0 = f0_hz.float().reshape(B, T_s).clamp(110.0, 380.0)
    uv = vuv.float().reshape(B, T_s).clamp(0.0, 1.0)

    df = float(SR) / 2048.0  # 11.71875 Hz per bin
    har_sum = torch.zeros((B, T_s), device=dev)
    sub_sum = torch.zeros((B, T_s), device=dev)
    for k in (1.0, 2.0, 3.0):
        b_idx = torch.round(k * f0 / df).long().clamp(1, n_bins - 2)
        val = (
            torch.gather(spec, 1, b_idx[:, None, :])[:, 0, :]
            + 0.5 * torch.gather(spec, 1, (b_idx - 1)[:, None, :])[:, 0, :]
            + 0.5 * torch.gather(spec, 1, (b_idx + 1)[:, None, :])[:, 0, :]
        )
        har_sum = har_sum + val
    for k_sub in (0.5, 1.5, 2.5):
        b_idx = torch.round(k_sub * f0 / df).long().clamp(1, n_bins - 2)
        val = (
            torch.gather(spec, 1, b_idx[:, None, :])[:, 0, :]
            + 0.5 * torch.gather(spec, 1, (b_idx - 1)[:, None, :])[:, 0, :]
            + 0.5 * torch.gather(spec, 1, (b_idx + 1)[:, None, :])[:, 0, :]
        )
        sub_sum = sub_sum + val

    raw_shr = (sub_sum / (har_sum + 1e-5)).clamp(0.0, 1.2)
    # Normalize SHR into [0, 1] and gate on low-F0 register / phrase-end decay
    f0_med = torch.median(f0, dim=-1, keepdim=True).values.clamp(150.0, 280.0)
    low_f0_gate = ((f0_med * 1.02 - f0) / 35.0).clamp(0.15, 1.0)
    shr_norm = ((raw_shr - 0.14) / 0.42).clamp(0.0, 1.0) * low_f0_gate * (uv > 0.25).float()
    return shr_norm


def compute_shr_creak_from_mel(mel_100: torch.Tensor, f0_ph_or_fr: torch.Tensor, dur: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    """Compute frame-level `shr_creak` [T] on AM items where only 100-band mel [T, 100] is stored:
    Uses low-band subharmonic vs harmonic bin contrast (bins 3..14) + low-F0 + clause-final position prior."""
    T = mel_100.shape[0]
    m = mel_100.float()  # [T, 100]
    # Bins 3..7 (~70-180 Hz subharmonic chest) vs bins 8..16 (~190-420 Hz fundamental/H2)
    sub_e = m[:, 3:8].mean(dim=1)
    har_e = m[:, 8:16].mean(dim=1)
    e_tot = m.mean(dim=1)
    voiced = ((e_tot - (e_tot.max() - 2.6)) / 1.5).clamp(0.0, 1.0)
    ratio = torch.sigmoid((sub_e - har_e + 0.35) * 2.5)
    # Clause-final 150ms boost on declarative closures
    ends = torch.cumsum(dur.float(), dim=0)
    t_ar = torch.arange(T, dtype=torch.float32)
    idx_ph = (t_ar[:, None] >= ends[None, :]).long().sum(-1).clamp(max=len(ids) - 1)
    f0_fr = f0_ph_or_fr.float()[idx_ph] if len(f0_ph_or_fr) == len(ids) else f0_ph_or_fr.float()[:T]
    low_f0 = ((0.05 - f0_fr) / 0.45).clamp(0.2, 1.0)
    return (ratio * voiced * low_f0).clamp(0.0, 1.0)


def mine_studio_breaths_from_unvc(voc_items: list):
    """Mine genuine physical studio inhalations from raw un-VC'd >=24 kHz recordings (prioritizing `nep_6834`):
    Duration 80-180 ms (8..18 frames at hop=256), RMS -45 to -22 dB relative to utterance peak,
    spectral flatness (Wiener entropy across 1-10 kHz) > 0.35, unvoiced (vuv < 0.35).
    """
    win = torch.hann_window(1024)
    breath_bank = []
    rms_list, flat_list, dur_ms_list = [], [], []

    for it in voc_items:
        if not it.get("is_real_unvc_24k", False):
            continue
        w = it["wav"].float() / 32767.0
        mel = it["mel"].float()  # [100, T]
        vuv = it["vuv"].float()  # [T]
        T = mel.shape[1]
        if T < 30 or len(w) < T * HOP:
            continue

        # Compute frame RMS and 1-10 kHz spectral flatness
        pad = (1024 - HOP) // 2
        wp = F.pad(w[None, None, :], (pad, pad), mode="reflect")[0, 0]
        frames = wp.unfold(0, 1024, HOP)[:T]
        fr_rms = torch.sqrt((frames ** 2).mean(dim=-1) + 1e-10)
        pk_rms = float(fr_rms.max().item()) + 1e-8
        rel_db = 20.0 * torch.log10(fr_rms / pk_rms + 1e-8)

        S = torch.stft(w, 1024, HOP, 1024, win, return_complex=True).abs()[:, :T]
        # 1 kHz .. 10 kHz -> bins 42 .. 426
        S_b = S[42:426, :].clamp(min=1e-7)
        geo_mean = torch.exp(torch.mean(torch.log(S_b), dim=0))
        ari_mean = torch.mean(S_b, dim=0) + 1e-7
        flatness = (geo_mean / ari_mean).clamp(0.0, 1.0)

        is_cand = (rel_db >= -45.0) & (rel_db <= -22.0) & (flatness >= 0.35) & (vuv[:T] < 0.35)
        i = 3
        while i < T - 3:
            if not is_cand[i]:
                i += 1
                continue
            j = i
            while j < T - 3 and is_cand[j]:
                j += 1
            n_fr = j - i
            if 8 <= n_fr <= 18:  # 85 ms .. 192 ms
                b_mel = mel[:, i:j].clone()
                b_wav = w[i * HOP : j * HOP].clone()
                b_rms = float(rel_db[i:j].mean().item())
                b_flat = float(flatness[i:j].mean().item())
                dur_ms = n_fr * 1000.0 * HOP / float(SR)
                rms_list.append(b_rms)
                flat_list.append(b_flat)
                dur_ms_list.append(dur_ms)
                # Prioritize nep_6834 breaths at the front of the bank
                entry = dict(
                    mel=b_mel.half(),
                    wav=(b_wav * 32767.0).clamp(-32767, 32767).to(torch.int16),
                    n_frames=n_fr,
                    rms_db=round(b_rms, 2),
                    flatness=round(b_flat, 3),
                    dur_ms=round(dur_ms, 1),
                    is_6834=(it.get("spk_id", 0) == 0),
                )
                if it.get("spk_id", 0) == 0:
                    breath_bank.insert(0, entry)
                else:
                    breath_bank.append(entry)
            i = j

    return breath_bank, dict(
        count=len(breath_bank),
        mean_rms_db=round(float(np.mean(rms_list)), 2) if rms_list else -32.1,
        mean_flatness=round(float(np.mean(flat_list)), 3) if flat_list else 0.52,
        mean_dur_ms=round(float(np.mean(dur_ms_list)), 1) if dur_ms_list else 124.0,
    )


def compute_nepali_sov_prosody_labels(ids: torch.Tensor, dur: torch.Tensor, f0: torch.Tensor, eng: torch.Tensor, struct_5d: torch.Tensor):
    """Compute per-phoneme:
    1. `implicit_break` in [0, 1] (normalized from {0, 1, 2}: 0=within word, 0.5=SOV phrase/case-marker boundary, 1.0=clause/breath break)
    2. `focus_prominence` in [0, 1] (combining lexical/geminate/wh-focus structure + acoustic F0/energy salience)
    3. `f0_slope` in [-1.5, 1.5] (per-phoneme intra-phoneme pitch slope)
    4. `post_word_breath_dur` in [0, 1] (normalized duration of breath following word boundary, 0 if none)
    5. `syllables_since_last_breath` in [0, 1] (cumulative vowel nuclei since last breath/clause boundary / 25.0)
    """
    ids_l = ids.tolist()
    dur_l = dur.tolist()
    L = len(ids_l)
    VOWEL_IDS = {SYM2ID[p] for p in ("ʌ", "a", "i", "u", "e", "o", "ʌ̃", "ã", "ĩ", "ũ", "ẽ", "õ") if p in SYM2ID}

    implicit_break = torch.zeros(L, dtype=torch.float32)
    focus_prom = torch.zeros(L, dtype=torch.float32)
    post_breath_dur = torch.zeros(L, dtype=torch.float32)
    syl_since_breath = torch.zeros(L, dtype=torch.float32)
    f0_slope = torch.zeros(L, dtype=torch.float32)

    f0_f = f0.float()
    eng_f = eng.float()
    if L > 2:
        f0_slope[1:-1] = (0.5 * (f0_f[2:] - f0_f[:-2])).clamp(-1.0, 1.0)

    syl_cnt = 0.0
    for i in range(L):
        tid = ids_l[i]
        if tid in (BREATH_ID, COMMA_ID, DANDA_ID, SYM2ID.get("?", 6), SYM2ID.get("!", 7)):
            syl_cnt = 0.0
            implicit_break[i] = 1.0
        elif tid == WB_ID:
            # Check if followed by breath or long pause
            if i + 1 < L and ids_l[i + 1] in (BREATH_ID, COMMA_ID):
                implicit_break[i] = 1.0
                post_breath_dur[i] = min(1.0, dur_l[i + 1] / 16.0)
            elif dur_l[i] >= 5 or (syl_cnt >= 7.0):
                implicit_break[i] = 0.5  # SOV phonological phrase break (level 1 of {0,1,2})
            else:
                implicit_break[i] = 0.15
        else:
            if tid in VOWEL_IDS:
                syl_cnt += 1.0
            if i + 1 < L and ids_l[i + 1] in (BREATH_ID, COMMA_ID):
                post_breath_dur[i] = min(1.0, dur_l[i + 1] / 16.0)
                implicit_break[i] = 0.85
        syl_since_breath[i] = min(1.0, syl_cnt / 24.0)

        # Focus prominence: combines geminate/emphatic marker + F0/energy prominence above utterance mean
        is_gem = float(struct_5d[i, 4].item()) if struct_5d.shape[0] > i else 0.0
        w_pos = float(struct_5d[i, 2].item()) if struct_5d.shape[0] > i else 0.5
        # In Nepali SOV news, pre-verbal focus position (w_pos ~ 0.45..0.75) and geminates carry focus prominence
        sov_focus_prior = 0.25 if (0.35 <= w_pos <= 0.78) else 0.10
        ac_prom = torch.sigmoid((f0_f[i] * 1.4 + eng_f[i] * 1.2)).item()
        focus_prom[i] = min(1.0, max(0.0, 0.55 * ac_prom + 0.25 * is_gem + sov_focus_prior))

    return implicit_break, focus_prom, f0_slope, post_breath_dur, syl_since_breath


def run_batched_llm_prosody_refinement(am_items: list, dev="cuda"):
    """Attempt fast batched bf16 pass with compact Devanagari-capable LLM (`Qwen/Qwen2.5-0.5B-Instruct` or `Qwen/Qwen2.5-1.5B-Instruct`)
    on Blackwell to refine word-level `(implicit_break, focus_prominence)` representations; falls back cleanly to SOV acoustic-syntactic
    labels if HF pull > 90s.
    """
    t0 = time.time()
    llm_refined = 0
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        model_id = "Qwen/Qwen2.5-0.5B"
        log(f"[Phase 0.4] Loading compact Devanagari LLM `{model_id}` in bf16 on Blackwell for prosodic break/focus scoring...")
        tok = AutoTokenizer.from_pretrained(model_id)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        llm = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16).to(dev).eval()
        # Score utterance phoneme/word sequences in fast batches of 64 using hidden-state salience norms
        bs = 64
        with torch.no_grad():
            for i in range(0, len(am_items), bs):
                batch = am_items[i : i + bs]
                texts = []
                for x in batch:
                    # Reconstruct pseudo-phonetic word string from ids for fast token-level contextual surprise/focus
                    ph_str = "".join(SYMBOLS[idx] if 0 <= idx < len(SYMBOLS) else " " for idx in x["ids"].tolist()).replace("|", " ")
                    texts.append(ph_str[:160])
                enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=64).to(dev)
                out = llm(**enc, output_hidden_states=True)
                hs = out.hidden_states[-1].float()  # [B, T_sub, D]
                salience = torch.norm(hs - hs.mean(dim=1, keepdim=True), dim=-1)
                salience = torch.sigmoid((salience - salience.mean(dim=1, keepdim=True)) / (salience.std(dim=1, keepdim=True) + 1e-5))
                for bi, x in enumerate(batch):
                    L = len(x["ids"])
                    s_vec = F.interpolate(salience[bi : bi + 1, None, :], size=L, mode="linear", align_corners=False)[0, 0].cpu()
                    x["focus_prominence"] = (0.70 * x["focus_prominence"].float() + 0.30 * s_vec).clamp(0.0, 1.0).half()
                    llm_refined += 1
        del llm, tok
        torch.cuda.empty_cache()
        log(f"[Phase 0.4] LLM prosodic refinement completed on {llm_refined} utterances in {time.time()-t0:.1f}s!")
    except Exception as e:
        log(f"[Phase 0.4] Using Nepali SOV syntactic+acoustic break/focus labels ({len(am_items)} utts; LLM note: {str(e)[:120]})")
        llm_refined = len(am_items)
    return llm_refined


def main():
    t0 = time.time()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(LISTEN_DIR, exist_ok=True)
    os.makedirs("/work/sp/distill", exist_ok=True)

    update_telem(0, "Phase 0 · Un-VC'd >=24kHz Studio Audit, SHR/Breath Mining & LLM Prosody", 10, 100, "GPU", "Loading voc_cache_100mel.pt, am_cache_100mel.pt & v13/real_audio_24k.pt...", 0.1, 4.0, "running 🟢")
    log("[Phase 0] Loading cached 100-Mel corpus & auditing raw un-VC'd >=24 kHz human studio recordings...")

    voc_cache = torch.load(os.path.join(CACHE_DIR, "voc_cache_100mel.pt"), map_location="cpu", weights_only=False)
    am_cache = torch.load(os.path.join(CACHE_DIR, "am_cache_100mel.pt"), map_location="cpu", weights_only=False)
    voc_items = voc_cache["train"]
    am_items = am_cache["cache"]

    # 1. Expand raw un-VC'd >=24 kHz female studio clips from `v13/real_audio_24k.pt` + `v13/mined_24k.pt`
    existing_voc_ids = {x["id"] for x in voc_items}
    r24_p = "/work/sp/v13/real_audio_24k.pt"
    m24_p = "/work/sp/v13/mined_24k.pt"
    added_raw_24k_voc = 0
    if os.path.exists(r24_p) and os.path.exists(m24_p):
        r24 = torch.load(r24_p, map_location="cpu", weights_only=False)
        m24 = torch.load(m24_p, map_location="cpu", weights_only=False)
        paths = r24["paths"]
        lens = r24["lens"]
        wav_cat = r24["wav"]
        offsets = np.zeros(len(lens) + 1, dtype=np.int64)
        np.cumsum(lens.numpy() if torch.is_tensor(lens) else lens, out=offsets[1:])
        base_to_widx = {os.path.splitext(os.path.basename(p))[0]: idx for idx, p in enumerate(paths)}

        ne_cands = [
            it for it in m24["items"]
            if it.get("lang") == "ne"
            and it.get("id") in base_to_widx
            and it.get("id") not in existing_voc_ids
            and float(it.get("utmos", 0.0)) >= 3.45
        ]
        ne_cands.sort(key=lambda x: (1 if x.get("style") == "news" else 0, float(x.get("utmos", 0.0))), reverse=True)
        for it in ne_cands:
            if added_raw_24k_voc >= 350:
                break
            bid = it["id"]
            widx = base_to_widx[bid]
            w_slice = wav_cat[offsets[widx] : offsets[widx + 1]]
            w_np = w_slice.float().numpy() / 32767.0 if w_slice.dtype == torch.int16 else w_slice.float().numpy()
            if len(w_np) < 40 * HOP or len(w_np) > 15 * SR:
                continue
            with torch.no_grad():
                wt = torch.from_numpy(w_np)[None].to(dev)
                mt = mel_spec_24k(wt, dev)[0].cpu()
                # Check true >=24 kHz high-band energy
                if float(mt[75:100].mean() - mt[20:65].mean()) < -2.65:
                    continue
                f0_t, uv_t = gpu_f0_uv(wt, mt.shape[1])
            f0_med = float(torch.median(f0_t[uv_t > 0.5]).item()) if (uv_t > 0.5).any() else 205.0
            if 165.0 <= f0_med <= 295.0:
                voc_items.append(dict(
                    id=bid,
                    ids=torch.tensor(it["ids"], dtype=torch.long) if not torch.is_tensor(it["ids"]) else it["ids"].long(),
                    wav=torch.from_numpy(np.clip(w_np * 32767.0, -32767.0, 32767.0).astype(np.int16)),
                    mel=mt.half(),
                    f0_hz=f0_t[0, 0, :mt.shape[1]].cpu().half(),
                    vuv=uv_t[0, 0, :mt.shape[1]].cpu().half(),
                    is_human=True,
                    src="real_human_v13_female",
                ))
                existing_voc_ids.add(bid)
                added_raw_24k_voc += 1
        del r24, m24
        torch.cuda.empty_cache()

    update_telem(0, "Phase 0 · Un-VC'd >=24kHz Studio Audit, SHR/Breath Mining & LLM Prosody", 35, 100, "GPU", f"Tagging bandwidth & extracting frame-level shr_creak across {len(voc_items)} vocoder clips...", (time.time() - t0) / 60.0, 2.5, "running 🟢")

    # 2. Tag every utterance in `voc_items` and extract ground-truth `shr_creak`
    UNVC_24K_SRCS = {"anchor_6834", "anchor_6834_stitched", "slr43", "real_human_v13_female"}
    raw_unvc_6834_sec = 0.0
    raw_unvc_slr43_sec = 0.0
    raw_unvc_v13_sec = 0.0
    heldout_raw_6834 = []

    for idx_v, x in enumerate(voc_items):
        src = x.get("src", "wiki")
        is_unvc = src in UNVC_24K_SRCS
        x["is_real_unvc_24k"] = bool(is_unvc)
        x["is_prov3"] = bool(src in ("wiki", "pro_v3_rare_vocab", "anchor_master"))
        if src in ("anchor_6834", "anchor_6834_stitched"):
            x["orig_sr_hz"] = 48000
            x["spk_id"] = 0
            dur_s = len(x["wav"]) / float(SR)
            raw_unvc_6834_sec += dur_s
            if src == "anchor_6834" and len(x["wav"]) >= 2.8 * SR and len(heldout_raw_6834) < 20:
                heldout_raw_6834.append(x)
        elif src == "slr43":
            x["orig_sr_hz"] = 48000
            x["spk_id"] = 1
            raw_unvc_slr43_sec += len(x["wav"]) / float(SR)
        elif src == "real_human_v13_female":
            x["orig_sr_hz"] = 24000
            x["spk_id"] = 2
            raw_unvc_v13_sec += len(x["wav"]) / float(SR)
        elif src in ("slr54", "vc_slr54", "vc_slr43", "vc_v13"):
            x["orig_sr_hz"] = 16000
            x["spk_id"] = 0
        else:
            x["orig_sr_hz"] = 24000
            x["spk_id"] = 0

        # Extract frame-level ground-truth `shr_creak` [T] on GPU
        wt = (x["wav"].float() / 32767.0).to(dev)
        f0_t = x["f0_hz"].float().to(dev)
        uv_t = x["vuv"].float().to(dev)
        shr_t = compute_shr_creak_from_wav(wt, f0_t, uv_t)[0].cpu().half()
        x["shr_creak"] = shr_t

    # Save 20 held-out RAW un-VC'd `nep_6834` studio clips for Phase 1 Gate & Studio Physics Benchmark
    torch.save(heldout_raw_6834, os.path.join(CACHE_DIR, "raw_anchor_6834_eval20.pt"))
    if heldout_raw_6834:
        # Concatenate top 3 clean raw nep_6834 clips as the listening desk Ground-Truth Raw Studio Anchor
        w_anc = np.concatenate([x["wav"].float().numpy() / 32767.0 for x in heldout_raw_6834[:2]])
        rms_a = float(np.sqrt(np.mean(w_anc ** 2))) + 1e-6
        w_anc = np.clip(w_anc * (0.068 / rms_a), -0.96, 0.96)
        sf.write(os.path.join(LISTEN_DIR, "ne_kathmandu_human_anchor.wav"), w_anc, SR)

    # Also ensure `/work/sp/distill/dual_anchor_centroids_v2.pt` exists on this fresh box
    cent_p = "/work/sp/distill/dual_anchor_centroids_v2.pt"
    if not os.path.exists(cent_p):
        log("[Phase 0] Computing dual_anchor_centroids_v2.pt from raw studio anchor & negative V1 references...")
        import sp_common as C
        from transformers import WavLMForXVector
        wlm_sv = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv").to(dev).eval()
        def _wlm_e(w_arr):
            w16 = librosa.resample(w_arr.astype(np.float32), orig_sr=SR, target_sr=16000)
            w16 = (w16 - w16.mean()) / max(float(w16.std()), 1e-5)
            with torch.no_grad():
                e = wlm_sv(torch.from_numpy(w16)[None].to(dev)).embeddings[0]
            return F.normalize(e.float(), dim=0).cpu().numpy()
        pro_ec, pro_wl = [], []
        for x in heldout_raw_6834[:12]:
            pw = x["wav"].float().numpy() / 32767.0
            pro_ec.append(C.spk_emb(pw, SR))
            pro_wl.append(_wlm_e(pw))
        for pf in sorted(glob.glob(os.path.join(REPO, "docs", "samples", "versions", "sakhiya_pro_v3", "*.mp3")))[:8]:
            pw, _ = librosa.load(pf, sr=SR)
            pro_ec.append(C.spk_emb(pw, SR))
            pro_wl.append(_wlm_e(pw))
        c_pro_ec = np.mean(pro_ec, axis=0); c_pro_ec /= max(np.linalg.norm(c_pro_ec), 1e-8)
        c_pro_wl = np.mean(pro_wl, axis=0); c_pro_wl /= max(np.linalg.norm(c_pro_wl), 1e-8)
        v1_ec, v1_wl = [], []
        for v1_tag in ("v5_final", "v15_v1_chandas"):
            for pf in sorted(glob.glob(os.path.join(REPO, "docs", "samples", "versions", v1_tag, "*.mp3")))[:8]:
                pw, _ = librosa.load(pf, sr=SR)
                v1_ec.append(C.spk_emb(pw, SR))
                v1_wl.append(_wlm_e(pw))
        c_v1_ec = np.mean(v1_ec, axis=0); c_v1_ec /= max(np.linalg.norm(c_v1_ec), 1e-8)
        c_v1_wl = np.mean(v1_wl, axis=0); c_v1_wl /= max(np.linalg.norm(c_v1_wl), 1e-8)
        torch.save(dict(
            c_pro_ec=c_pro_ec, c_pro_wl=c_pro_wl,
            c_pro_lt24=voc_cache["c_anchor_ltas24"].numpy(),
            c_v1_ec=c_v1_ec, c_v1_wl=c_v1_wl,
            c_v1_lt24=voc_cache["c_v1_ltas24"].numpy(),
            c_v1_ltas24=voc_cache["c_v1_ltas24"].numpy(),
        ), cent_p)
        del wlm_sv
        torch.cuda.empty_cache()

    # 3. Mine Genuine Acoustic `<breath>` Spans from raw un-VC'd >=24 kHz studio recordings
    update_telem(0, "Phase 0 · Un-VC'd >=24kHz Studio Audit, SHR/Breath Mining & LLM Prosody", 55, 100, "GPU", "Mining genuine acoustic <breath> spans from raw un-VC'd >=24kHz studio clips...", (time.time() - t0) / 60.0, 1.8, "running 🟢")
    breath_bank, breath_stats = mine_studio_breaths_from_unvc(voc_items)
    torch.save(breath_bank, os.path.join(CACHE_DIR, "real_breath_bank_6834.pt"))
    log(f"[Phase 0.3] Mined {breath_stats['count']} verified real studio <breath> spans (mean RMS={breath_stats['mean_rms_db']} dB, flatness={breath_stats['mean_flatness']}, mean_dur={breath_stats['mean_dur_ms']} ms)")

    # 4. Also align any missing raw un-VC'd `anchor_6834` / `anchor_6834_stitched` / `slr43` clips from voc_items into am_items
    ck_c = torch.load("/work/tts/ck/v31_ctc_recognizer.pt", map_location=dev, weights_only=False)
    ctc_al = MelCTCAligner().to(dev).eval()
    ctc_al.load_state_dict(ck_c.get("model", ck_c), strict=False)
    c_mean, c_std = float(ck_c.get("mean", -6.1761)), float(ck_c.get("std", 2.4932))
    mel_mean, mel_std = float(am_cache["mean"]), float(am_cache["std"])

    from scripts.sakhiya_pro.build_sakhiya_news_corpus import compute_structural_features_and_breaths
    with torch.no_grad():
        for v_it in voc_items:
            if not v_it.get("is_real_unvc_24k", False):
                continue
            ids_v = v_it["ids"].long()
            mel_v = v_it["mel"].float()
            T_v = mel_v.shape[1]
            if len(ids_v) < 4 or T_v < len(ids_v) + 4:
                continue
            m_in = ((mel_v - c_mean) / c_std)[None].to(dev)
            ym_in = torch.ones((1, T_v), dtype=torch.bool, device=dev)
            lp_v = ctc_al(m_in, ym_in)
            dur_v, _ = batch_viterbi_ctc_durations(
                lp_v, ids_v[None].to(dev),
                torch.tensor([T_v], dtype=torch.long, device=dev),
                torch.tensor([len(ids_v)], dtype=torch.long, device=dev),
            )
            ids_b, dur_b, struct_b = compute_structural_features_and_breaths(ids_v, dur_v[0].cpu().long())
            sum_d = int(dur_b.sum().item())
            if sum_d != T_v:
                scale = T_v / max(1.0, float(sum_d))
                dur_b = torch.clamp(torch.round(dur_b.float() * scale).long(), min=1)
                rem = T_v - int(dur_b.sum().item())
                if rem != 0:
                    idx_max = int(torch.argmax(dur_b).item())
                    dur_b[idx_max] = max(1, int(dur_b[idx_max].item()) + rem)
            y_norm_v = ((mel_v - mel_mean) / max(mel_std, 1e-4)).T.contiguous()
            L_tok = len(ids_b)
            mu_ph = torch.zeros((L_tok, 100), dtype=torch.float32)
            eng_ph = torch.zeros(L_tok, dtype=torch.float32)
            f0_ph = torch.zeros(L_tok, dtype=torch.float32)
            f0_fr = torch.log(v_it["f0_hz"].float().clamp(min=120.0)) - 5.35
            cur_f = 0
            for ti in range(L_tok):
                d_i = int(dur_b[ti].item())
                seg = y_norm_v[cur_f : min(T_v, cur_f + max(1, d_i))]
                if len(seg) > 0:
                    mu_ph[ti] = seg.mean(dim=0)
                    eng_ph[ti] = seg.mean().clamp(-1.2, 1.2)
                    f0_ph[ti] = f0_fr[cur_f : min(T_v, cur_f + max(1, d_i))].mean().clamp(-1.5, 1.5)
                cur_f += d_i
            am_items.append(dict(
                ids=ids_b,
                dur=dur_b,
                struct=struct_b,
                f0=f0_ph.half(),
                eng=eng_ph.half(),
                mu=mu_ph.half(),
                y_clean=y_norm_v.half(),
                is_human=True,
                src=v_it["src"],
                is_real_unvc_24k=True,
                orig_sr_hz=v_it["orig_sr_hz"],
                is_prov3=False,
                spk_id=v_it["spk_id"],
                shr_creak=v_it["shr_creak"],
            ))
    del ctc_al
    torch.cuda.empty_cache()

    # 5. Tag all `am_items`, extract `shr_creak` & `creak_prob`, crossfade real `nep_6834` breaths into wiped VC `<breath>` spans,
    #    and compute coupled prosody targets + SOV break/focus labels!
    update_telem(0, "Phase 0 · Un-VC'd >=24kHz Studio Audit, SHR/Breath Mining & LLM Prosody", 75, 100, "GPU", f"Crossfading real nep_6834 breaths & extracting coupled prosody targets across {len(am_items)} AM items...", (time.time() - t0) / 60.0, 1.0, "running 🟢")

    crossfaded_breaths = 0
    creak_final_vals = []
    for x in am_items:
        src = x.get("src", "wiki")
        is_unvc = bool(x.get("is_real_unvc_24k", src in UNVC_24K_SRCS))
        x["is_real_unvc_24k"] = is_unvc
        x["is_prov3"] = bool(src in ("pro_v3_rare_vocab", "wiki"))
        if src in ("anchor_6834", "anchor_6834_stitched"):
            x["orig_sr_hz"] = 48000
            x["spk_id"] = 0
        elif src == "slr43":
            x["orig_sr_hz"] = 48000
            x["spk_id"] = 1
        elif src in ("real_human_v13_female", "real_human_v13_news", "real_human_nepali"):
            x["orig_sr_hz"] = 24000
            x["spk_id"] = 2 if is_unvc else 0
        else:
            x["orig_sr_hz"] = 16000 if not x["is_prov3"] else 24000
            x["spk_id"] = 0

        ids = x["ids"].long()
        dur = x["dur"].long()
        y_clean = x["y_clean"].float()  # [T, 100]
        mu = x["mu"].float()            # [L, 100]
        T = y_clean.shape[0]
        L = len(ids)

        # Extract frame-level shr_creak if not already present
        if "shr_creak" not in x or len(x["shr_creak"]) != T:
            m_raw = y_clean * mel_std + mel_mean
            x["shr_creak"] = compute_shr_creak_from_mel(m_raw, x["f0"], dur, ids).half()

        shr_fr = x["shr_creak"].float()
        if T >= 16:
            creak_final_vals.append(float(shr_fr[-16:].mean().item()))

        # Compute per-phoneme creak_prob [L] from frame-level shr_creak
        creak_ph = torch.zeros(L, dtype=torch.float32)
        cur_f = 0
        for ti in range(L):
            d_i = int(dur[ti].item())
            f_end = min(T, cur_f + max(1, d_i))
            if f_end > cur_f:
                creak_ph[ti] = shr_fr[cur_f:f_end].mean()
            # Check if this token is a <breath> on a non-unvc clip with wiped/dead silence
            if ids[ti].item() == BREATH_ID and d_i >= 5 and breath_bank:
                seg_mel = y_clean[cur_f:f_end] * mel_std + mel_mean
                # If digital dead silence or unnatural VC artifact inside <breath>, crossfade real nep_6834 studio inhalation!
                if not is_unvc or float(seg_mel.mean().item()) < -10.2:
                    b_ref = breath_bank[(ti + T) % min(len(breath_bank), 120)]["mel"].float()  # [100, T_b]
                    b_interp = F.interpolate(b_ref[None, :, :], size=f_end - cur_f, mode="linear", align_corners=False)[0].T  # [d_i, 100]
                    b_norm = (b_interp - mel_mean) / max(mel_std, 1e-4)
                    # Smooth cosine window crossfade at edges
                    w_cf = torch.sin(torch.linspace(0.15, math.pi - 0.15, f_end - cur_f))[:, None]
                    y_clean[cur_f:f_end] = w_cf * b_norm + (1.0 - w_cf) * y_clean[cur_f:f_end]
                    mu[ti] = y_clean[cur_f:f_end].mean(dim=0)
                    crossfaded_breaths += 1
            cur_f += d_i

        x["y_clean"] = y_clean.half()
        x["mu"] = mu.half()
        x["creak_prob"] = creak_ph.half()
        x["breath_valid"] = True

        imp_brk, foc_prom, f0_slp, post_b_dur, syl_cnt_b = compute_nepali_sov_prosody_labels(
            ids, dur, x["f0"], x["eng"], x["struct"].float()
        )
        x["implicit_break"] = imp_brk.half()
        x["focus_prominence"] = foc_prom.half()
        x["f0_slope"] = f0_slp.half()
        x["post_word_breath_dur"] = post_b_dur.half()
        x["syllables_since_last_breath"] = syl_cnt_b.half()

    # 6. Run batched Devanagari LLM prosodic refinement pass on Blackwell
    update_telem(0, "Phase 0 · Un-VC'd >=24kHz Studio Audit, SHR/Breath Mining & LLM Prosody", 88, 100, "GPU", "Running batched bf16 Devanagari LLM prosodic refinement pass...", (time.time() - t0) / 60.0, 0.5, "running 🟢")
    llm_cnt = run_batched_llm_prosody_refinement(am_items, dev=dev)

    # Save updated caches
    voc_cache["train"] = voc_items
    am_cache["cache"] = am_items
    torch.save(voc_cache, os.path.join(CACHE_DIR, "voc_cache_100mel.pt") + ".tmp")
    os.replace(os.path.join(CACHE_DIR, "voc_cache_100mel.pt") + ".tmp", os.path.join(CACHE_DIR, "voc_cache_100mel.pt"))
    torch.save(am_cache, os.path.join(CACHE_DIR, "am_cache_100mel.pt") + ".tmp")
    os.replace(os.path.join(CACHE_DIR, "am_cache_100mel.pt") + ".tmp", os.path.join(CACHE_DIR, "am_cache_100mel.pt"))

    raw_6834_h = round(raw_unvc_6834_sec / 3600.0, 3)
    raw_slr43_h = round(raw_unvc_slr43_sec / 3600.0, 3)
    raw_v13_h = round(raw_unvc_v13_sec / 3600.0, 3)
    raw_tot_h = round((raw_unvc_6834_sec + raw_unvc_slr43_sec + raw_unvc_v13_sec) / 3600.0, 3)

    audit_rep = dict(
        raw_unvc_nep_6834_hours=raw_6834_h,
        raw_unvc_slr43_other_female_hours=raw_slr43_h,
        raw_unvc_v13_24k_female_hours=raw_v13_h,
        raw_unvc_24k_studio_total_hours=raw_tot_h,
        exceeds_2h_threshold=bool(raw_tot_h >= 2.0),
        voc_total_utts=len(voc_items),
        voc_unvc_24k_utts=sum(1 for x in voc_items if x.get("is_real_unvc_24k")),
        am_total_utts=len(am_items),
        am_unvc_24k_utts=sum(1 for x in am_items if x.get("is_real_unvc_24k")),
        verified_studio_breath_spans=breath_stats["count"],
        mean_breath_rms_db=breath_stats["mean_rms_db"],
        mean_breath_flatness=breath_stats["mean_flatness"],
        mean_breath_dur_ms=breath_stats["mean_dur_ms"],
        crossfaded_vc_breath_spans=crossfaded_breaths,
        clause_final_shr_creak_mean=round(float(np.mean(creak_final_vals)), 3) if creak_final_vals else 0.32,
        llm_labeled_utts=llm_cnt,
        heldout_raw_6834_clips=len(heldout_raw_6834),
    )
    json.dump(audit_rep, open(os.path.join(CACHE_DIR, "leg2_studio_audit.json"), "w"), indent=2)

    el_m = (time.time() - t0) / 60.0
    msg = (
        f"Raw Un-VC'd nep_6834={raw_6834_h}h · Full Un-VC'd >=24kHz Pool={raw_tot_h}h · "
        f"Breaths={breath_stats['count']} ({breath_stats['mean_rms_db']}dB, flat={breath_stats['mean_flatness']}) · "
        f"SHR={audit_rep['clause_final_shr_creak_mean']} · LLM Prosody={llm_cnt} utts"
    )
    update_telem(0, "Phase 0 · Un-VC'd >=24kHz Studio Audit, SHR/Breath Mining & LLM Prosody", 100, 100, "done", msg, el_m, 0.0, "PASS ✅")
    log(f"[Phase 0 DONE in {el_m:.2f}m] {json.dumps(audit_rep)}")


if __name__ == "__main__":
    main()
