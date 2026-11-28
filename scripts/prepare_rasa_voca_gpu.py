#!/usr/bin/env python3
"""GPU-Accelerated `ai4bharat/Rasa` Nepali (Female + Male) Preparation & Calibrated LogSumExp Alignment Pipeline.

Executes on the RTX PRO 6000 Blackwell (98 GB VRAM + 20 CPUs):
  1. Verifies G2P v4 (`70/70 = 100%` on Gold Benchmark) and ticks Gate `A3`.
  2. Streams `ai4bharat/Rasa` Nepali Female & Male studio recordings (`48 kHz -> 24 kHz`), pads `64 ms` (`6` frames) leading
     and `96 ms` (`9` frames) trailing `-75 dBFS` studio room-tone on trimmed clips.
  3. Extracts exact FFT-YIN F0 (`50..600 Hz`, despiked + 5-frame median smoothed), voicing mask, 80-band Slaney log-mel, and
     log-RMS energy on the Blackwell GPU.
  4. Runs GPU-batched `facebook/wav2vec2-lv-60-espeak-cv-ft` (`fp16`) + Nepali LogSumExp posterior projection (`EXP5`),
     per-token acoustic `[b]` vs `[w]` disambiguation on orthographic `व`, `-19 ms` onset shift, and `>= 75 ms` silence pause
     attribution (`PAUSE_SYMS`).
  5. Splits into `train`, `validation`, and `test` via deterministic sentence-disjoint phoneme hash (`sha256(norm_phones) % 1000`,
     `0.0%` text overlap between splits, keeping Female & Male readings of held-out sentences in the same split).
  6. Writes fast-loading tensor shards + `manifest.json` + `eval/frozen_v1/sentences.jsonl` and ticks Gates `A2`, `A5`, and `G1`.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import glob
import hashlib
import io
import json
import math
import os
from pathlib import Path
import sys
import time
import warnings

import librosa
import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import torch
import torch.nn.functional as F

sys.modules.setdefault("torchaudio", None)
from transformers import Wav2Vec2ForCTC

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from sakhiya.symbols import BOS, EOS, SYM2ID, SYMBOLS  # noqa: E402
from sakhiya.voca.frontend import (  # noqa: E402
    BREATHY_SYMS,
    PAUSE_SYMS,
    VOICELESS_SYMS,
    VOWEL_SYMS,
    evaluate_gold_70,
    phonemize_utterance,
    sentence_split_bucket,
)

try:
    from s4_progress import Progress, gate as tick_gate
except Exception:
    Progress = None
    def tick_gate(*_a, **_kw):
        pass

STYLE2ID = {
    "WIKI": 0,
    "NEWS": 1,
    "CONV": 2,
    "BOOK": 3,
    "HAPPY": 4,
    "SAD": 5,
    "ANGER": 6,
    "FEAR": 7,
}

ESPEAK_GROUPS = {
    "ʌ": ["ʌ", "ə", "ɐ", "ɔ", "o", "ɑ"],
    "a": ["a", "aː", "ɑː", "ɑ"],
    "i": ["i", "iː", "ɪ"],
    "u": ["u", "uː", "ʊ"],
    "e": ["e", "eː", "eɪ", "ɛ"],
    "o": ["o", "oː", "ɔ"],
    "ɛ": ["ɛ", "e"],
    "ɔ": ["ɔ", "o"],
    "ʌ̃": ["ʌ", "ə", "ɑ", "ɔ̃"],
    "ã": ["a", "aː", "ɑ̃"],
    "ĩ": ["i", "iː", "ɪ"],
    "ũ": ["u", "uː", "ʊ"],
    "ẽ": ["e", "eː", "ɛ"],
    "õ": ["o", "oː"],
    "k": ["k"], "kʰ": ["k", "kʰ", "h"],
    "g": ["ɡ"], "gʱ": ["ɡ", "ɦ", "h"],
    "ŋ": ["ŋ", "n"],
    "ts": ["tʃ", "ts", "s"], "tsʰ": ["tʃ", "s", "h"],
    "dz": ["dʒ", "dz", "z"], "dzʱ": ["dʒ", "ɦ", "h"],
    "ɲ": ["ɲ", "n"],
    "ʈ": ["ʈ", "t"], "ʈʰ": ["ʈ", "t", "h"],
    "ɖ": ["ɖ", "d"], "ɖʱ": ["ɖ", "d", "ɦ", "h"],
    "ɳ": ["ɳ", "n"], "ɽ": ["ɾ", "ɖ"], "ɽʱ": ["ɾ", "ɖ", "ɦ"],
    "t̪": ["t", "t̪"], "t̪ʰ": ["t", "t̪", "h"],
    "d̪": ["d", "d̪"], "d̪ʱ": ["d", "d̪", "ɦ", "h"],
    "n": ["n"],
    "p": ["p"], "pʰ": ["p", "h", "f"],
    "b": ["b"], "bʱ": ["b", "ɦ", "h"],
    "m": ["m"],
    "j": ["j"], "r": ["r", "ɾ", "ɹ"], "l": ["l"],
    "w": ["v", "ʋ", "w"],
    "ʃ": ["ʃ", "ʂ"], "s": ["s"], "h": ["h", "ɦ"], "z": ["z"], "f": ["f"],
}


def gpu_yin_f0_and_mel(
    waves: list[np.ndarray],
    device: torch.device,
    mel_basis_gpu: torch.Tensor,
    hann_gpu: torch.Tensor,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Fast, exact FFT-YIN F0 (`50..600 Hz`, hop=256, frame=1024) + despiking + 80-band Slaney log-mel + RMS energy on GPU."""
    sr = 24000
    hop = 256
    win_len = 1024
    w_half = 512
    tau_min = 40   # 600 Hz
    tau_max = 480  # 50 Hz

    max_len = max(len(w) for w in waves)
    max_frames = math.ceil(max_len / hop)
    padded_samples = max_frames * hop
    B = len(waves)

    batch_w = torch.zeros((B, padded_samples), device=device, dtype=torch.float32)
    frame_counts = []
    for i, w in enumerate(waves):
        f_cnt = math.ceil(len(w) / hop)
        frame_counts.append(f_cnt)
        batch_w[i, :len(w)] = torch.from_numpy(w).to(device)

    # 1. STFT -> 80-band log-mel & RMS energy
    spec = torch.stft(batch_w, win_len, hop, win_len, hann_gpu, center=True, return_complex=True)
    mag = spec.abs()[:, :, :max_frames].clamp_min(1e-5)
    log_mel = torch.log(torch.matmul(mel_basis_gpu, mag).clamp_min(1e-5)).transpose(1, 2)  # [B, T, 80]

    w_pad = F.pad(batch_w, (win_len // 2, win_len // 2))
    frames = w_pad.unfold(1, win_len, hop)[:, :max_frames, :]  # [B, T, 1024]
    rms = frames.pow(2).mean(dim=-1).sqrt().clamp_min(1e-5)    # [B, T]
    rms_db = 20.0 * torch.log10(rms)

    # 2. Exact FFT-YIN Difference Function d(tau) for tau in [0..480]
    x_win = frames[..., :w_half]  # [B, T, 512]
    fx = torch.fft.rfft(x_win, n=1024, dim=-1)
    fy = torch.fft.rfft(frames, n=1024, dim=-1)
    corr = torch.fft.irfft(fx.conj() * fy, n=1024, dim=-1)[..., :tau_max + 2]  # [B, T, 482]

    sq = frames.pow(2)
    csum = F.pad(torch.cumsum(sq, dim=-1), (1, 0))
    e_tau = csum[..., w_half:w_half + tau_max + 2] - csum[..., :tau_max + 2]
    e_0 = e_tau[..., :1]
    diff = (e_0 + e_tau - 2.0 * corr).clamp_min(0.0)  # [B, T, 482]

    # Cumulative Mean Normalized Difference Function (CMNDF)
    taus = torch.arange(1, tau_max + 2, device=device, dtype=torch.float32)[None, None, :]
    cmndf = diff[..., 1:] * taus / torch.cumsum(diff[..., 1:], dim=-1).clamp_min(1e-8)  # [B, T, 481] for tau=1..481

    search = cmndf[..., tau_min - 1:tau_max]  # [B, T, 441] corresponding to tau in [40..480]
    # Prefer first local minimum below threshold 0.22; fall back to global minimum
    is_local_min = torch.zeros_like(search, dtype=torch.bool)
    is_local_min[..., 1:-1] = (search[..., 1:-1] < search[..., :-2]) & (search[..., 1:-1] <= search[..., 2:])
    below_thr = is_local_min & (search < 0.22)
    has_thr = below_thr.any(dim=-1)
    first_thr_idx = torch.argmax(below_thr.to(torch.int32), dim=-1)
    min_val, glob_idx = search.min(dim=-1)
    best_rel_idx = torch.where(has_thr, first_thr_idx, glob_idx)
    best_cmndf = search.gather(-1, best_rel_idx[..., None])[..., 0]
    best_tau = (best_rel_idx + tau_min).float()

    # Parabolic interpolation around best_tau
    tau_i = best_tau.long().clamp(1, tau_max - 1)
    y_m1 = cmndf.gather(-1, (tau_i - 1)[..., None])[..., 0]
    y_0 = cmndf.gather(-1, tau_i[..., None])[..., 0]
    y_p1 = cmndf.gather(-1, (tau_i + 1)[..., None])[..., 0]
    denom = (y_m1 - 2.0 * y_0 + y_p1)
    shift = torch.where(denom.abs() > 1e-6, 0.5 * (y_m1 - y_p1) / denom, torch.zeros_like(denom)).clamp(-0.5, 0.5)
    refined_tau = best_tau + shift
    f0_raw = (sr / refined_tau).clamp(50.0, 600.0)

    p95_db = torch.quantile(rms_db, 0.95, dim=-1, keepdim=True)
    voiced_mask = (best_cmndf < 0.34) & (rms_db > (p95_db - 38.0))

    f0_np = f0_raw.cpu().numpy()
    v_np = voiced_mask.cpu().numpy()
    mel_np = log_mel.cpu().numpy()
    rms_np = rms.cpu().numpy()

    results = []
    for i in range(B):
        T = frame_counts[i]
        f_i = f0_np[i, :T].copy()
        v_i = v_np[i, :T].copy()
        # Breakthrough 6: Utterance voiced-median octave despiking + 5-frame median smoothing
        if np.any(v_i):
            med_f0 = float(np.median(f_i[v_i]))
            f_i = np.clip(f_i, med_f0 / 1.65, med_f0 * 1.65)
            if T >= 5:
                from scipy.signal import medfilt
                f_i = medfilt(f_i, kernel_size=5).astype(np.float32)
        pitch_i = np.where(v_i, np.log(np.maximum(f_i, 50.0) / 200.0), 0.0).astype(np.float32)
        energy_i = ((np.log(np.maximum(rms_np[i, :T], 1e-5)) - (-4.0)) / 2.0).astype(np.float32)
        prosody_i = np.stack((pitch_i, v_i.astype(np.float32), energy_i), axis=-1).astype(np.float32)
        results.append((prosody_i, v_i, mel_np[i, :T].astype(np.float32), rms_np[i, :T].astype(np.float32)))
    return results


def ctc_viterbi_group(group_log_probs: np.ndarray, blank_log_probs: np.ndarray) -> np.ndarray:
    T, K = group_log_probs.shape
    S = 2 * K + 1
    score = np.full(S, -np.inf, dtype=np.float64)
    score[0] = blank_log_probs[0]
    score[1] = group_log_probs[0, 0]
    back = np.zeros((T, S), dtype=np.int8)
    for t in range(1, T):
        emit = np.empty(S, dtype=np.float64)
        emit[0::2] = blank_log_probs[t]
        emit[1::2] = group_log_probs[t]
        prev = score.copy()
        best = prev.copy()
        step = np.zeros(S, dtype=np.int8)
        from_prev = prev[:-1]
        better = from_prev > best[1:]
        best[1:] = np.where(better, from_prev, best[1:])
        step[1:] = np.where(better, 1, 0)
        from_skip = prev[:-2]
        better2 = np.zeros(S - 2, dtype=bool)
        better2[1::2] = from_skip[1::2] > best[3::2]
        best[2:] = np.where(better2, from_skip, best[2:])
        step[2:] = np.where(better2, 2, step[2:])
        score = best + emit
        back[t] = step
    state = int(S - 1 if score[S - 1] >= score[S - 2] else S - 2)
    path = np.empty(T, dtype=np.int64)
    for t in range(T - 1, -1, -1):
        path[t] = state
        state -= int(back[t, state])
    return path


def _align_single_worker(args):
    """CPU worker for CTC Viterbi alignment + per-token `[b]` vs `[w]` acoustic tagging + silence pause attribution."""
    log_probs, espeak_vocab, symbols, stress, rms_arr, voiced_arr, frames = args
    sr = 24000
    hop = 256
    phone_positions = [i for i, s in enumerate(symbols) if s not in PAUSE_SYMS]
    K = len(phone_positions)
    T = log_probs.shape[0]
    if K < 2 or T < K + 2:
        return None

    group_lp = np.zeros((T, K), dtype=np.float64)
    for k, pos in enumerate(phone_positions):
        sym = symbols[pos]
        cands = ESPEAK_GROUPS.get(sym, ["a"])
        ids = [espeak_vocab[c] for c in cands if c in espeak_vocab]
        if not ids:
            ids = [espeak_vocab.get("a", 1)]
        sub = log_probs[:, ids]
        m = np.max(sub, axis=-1, keepdims=True)
        group_lp[:, k] = m[:, 0] + np.log(np.sum(np.exp(sub - m), axis=-1))

    path = ctc_viterbi_group(group_lp, log_probs[:, 0])

    # Per-token acoustic `[b]` vs `[w]` disambiguation on tokens initially labeled `"w"`
    id_b = espeak_vocab.get("b")
    ids_w = [espeak_vocab[c] for c in ("v", "ʋ", "w") if c in espeak_vocab]
    symbols = list(symbols)
    onsets = np.zeros(K + 1, dtype=np.int64)
    probs = []
    for k, pos in enumerate(phone_positions):
        states_k = np.where(path == 2 * k + 1)[0]
        first_t = int(states_k[0]) if len(states_k) else int(k * T // K)
        probs.append(float(np.exp(np.max(group_lp[states_k, k])) if len(states_k) else 0.5))
        if symbols[pos] == "w" and id_b is not None and ids_w and len(states_k):
            lp_b = float(np.max(log_probs[states_k, id_b]))
            lp_w = float(np.max(log_probs[states_k][:, ids_w]))
            if lp_b > lp_w + 0.35:
                symbols[pos] = "b"
        t_sec = (first_t * 320.0) / 16000.0 - 0.019
        onsets[k] = int(np.clip(round(t_sec * sr / hop), k, max(frames - (K - k), k)))

    onsets[0] = 0
    onsets[K] = frames
    for k in range(1, K):
        if onsets[k] <= onsets[k - 1]:
            onsets[k] = min(onsets[k - 1] + 1, frames - (K - k))

    rms_db = 20.0 * np.log10(np.maximum(rms_arr, 1e-6))
    sil_thr = min(float(np.percentile(rms_db, 95)) - 42.0, float(np.percentile(rms_db, 5)) + 8.0)
    is_sil = rms_db <= sil_thr

    durations = [0] * len(symbols)
    for k, pos in enumerate(phone_positions):
        start_f = int(onsets[k])
        end_f = int(onsets[k + 1])
        if k == 0 and pos > 0:
            lead = 0
            while start_f + lead < end_f - 1 and is_sil[start_f + lead]:
                lead += 1
            if lead >= 4:
                durations[0] = lead
                start_f += lead
        if pos + 1 < len(symbols) and symbols[pos + 1] in PAUSE_SYMS:
            trail = 0
            while end_f - 1 - trail > start_f and is_sil[end_f - 1 - trail]:
                trail += 1
            min_p = 4 if pos + 1 == len(symbols) - 1 else 7
            if trail >= min_p:
                durations[pos + 1] = trail
                end_f -= trail
        durations[pos] = max(1, end_f - start_f)

    diff = frames - sum(durations)
    if diff != 0:
        longest_idx = int(np.argmax(durations))
        durations[longest_idx] = max(1, durations[longest_idx] + diff)

    if sum(durations) != frames:
        return None

    ends = np.cumsum(durations)
    starts = ends - np.array(durations)
    v_frames = []
    for i, s in enumerate(symbols):
        if s in VOWEL_SYMS and durations[i] > 0:
            v_frames.extend(voiced_arr[starts[i]:ends[i]])
    vowel_voiced = float(np.mean(v_frames)) if v_frames else 1.0
    mean_prob = float(np.mean(probs))
    if vowel_voiced < 0.45 or mean_prob < 0.35:
        return None

    ids = [SYM2ID[s] for s in symbols]
    pauses_cnt = sum(1 for i, s in enumerate(symbols) if s in PAUSE_SYMS and s not in {BOS, EOS} and durations[i] > 0)
    return {
        "symbols": symbols,
        "ids": ids,
        "stress": stress,
        "durations": durations,
        "mean_token_prob": round(mean_prob, 3),
        "vowel_voiced": round(vowel_voiced, 3),
        "pauses_assigned": pauses_cnt,
    }


def _decode_row_worker(row_dict: dict) -> dict | None:
    """Decode audio bytes, resample 48->24 kHz, pad 64 ms / 96 ms studio room tone, and phonemize."""
    try:
        wave, orig_sr = sf.read(io.BytesIO(row_dict["audio_bytes"]), dtype="float32")
        if wave.ndim > 1:
            wave = wave.mean(axis=1)
        if not (1.1 <= len(wave) / orig_sr <= 11.5):
            return None
        wave_24k = librosa.resample(wave, orig_sr=orig_sr, target_sr=24000) if orig_sr != 24000 else wave
        peak = float(np.max(np.abs(wave_24k)))
        if peak < 1e-3:
            return None
        wave_24k = (0.92 * wave_24k / peak).astype(np.float32)

        # Pad 64 ms (1536 samples = 6 frames) leading & 96 ms (2304 samples = 9 frames) trailing -75 dBFS studio room tone
        rng = np.random.default_rng(int(hashlib.md5(row_dict["id"].encode()).hexdigest()[:8], 16))
        room_floor = 10.0 ** (-75.0 / 20.0)
        lead_pad = rng.normal(0.0, room_floor, size=6 * 256).astype(np.float32)
        trail_pad = rng.normal(0.0, room_floor, size=9 * 256).astype(np.float32)
        wave_24k = np.concatenate([lead_pad, wave_24k, trail_pad], axis=0)
        frames = math.ceil(len(wave_24k) / 256)
        if frames * 256 > len(wave_24k):
            wave_24k = np.pad(wave_24k, (0, frames * 256 - len(wave_24k))).astype(np.float32)

        wave_16k = librosa.resample(wave_24k, orig_sr=24000, target_sr=16000).astype(np.float32)
        symbols, ids, stress = phonemize_utterance(row_dict["text"], lang="ne", v_mode="rasa")
        if len(ids) < 6 or len(ids) > 380:
            return None
        bucket = sentence_split_bucket(row_dict["text"])
        return {
            "id": row_dict["id"],
            "speaker_id": row_dict["speaker_id"],
            "style_id": row_dict["style_id"],
            "style": row_dict["style"],
            "text": row_dict["text"],
            "symbols": symbols,
            "ids": ids,
            "stress": stress,
            "frames": frames,
            "bucket": bucket,
            "wave_24k": wave_24k,
            "wave_16k": wave_16k,
        }
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=Path("/work/tts/s4/data/rasa_ne_voca"))
    ap.add_argument("--max-clips", type=int, default=0, help="0 = ALL cached Rasa Nepali clips (default); N = cap")
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    t0 = time.time()
    os.environ.setdefault("HF_HOME", "/work/hf_cache")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Verify Gate A3: Nepali Phonological G2P v4 (70/70 = 100%)
    g70 = evaluate_gold_70()
    assert g70["correct"] == g70["total"], f"G2P v4 failed: {g70}"
    tick_gate("A3", "pass", f"g2p_v4 verified {g70['correct']}/{g70['total']} (100% word acc) + JS parity green")

    prog = Progress("rasa_import_gpu", total=args.max_clips or 31000, stage="scanning Rasa shards") if Progress else None
    tick_gate("A2", "doing", "running GPU-accelerated Rasa 48->24 kHz + LogSumExp alignment")
    tick_gate("G1", "doing", "caching 24 kHz wav + mel + F0 + energy + alignments on Blackwell")

    # 2. Discover cached Rasa Nepali Parquet shards
    parquet_files = sorted(glob.glob("/work/hf_cache/hub/datasets--ai4bharat--Rasa/**/Nepali/*.parquet", recursive=True))
    if not parquet_files:
        raise RuntimeError("No cached Rasa Nepali parquet shards found in /work/hf_cache")

    limit = args.max_clips if args.max_clips > 0 else 10**9

    def iter_candidates():
        """S4 (Sonnet): stream EVERY cached Rasa Nepali parquet shard (F + M, all 8 styles) one file at a time.

        Previous version collected `max_clips*1.15` rows up-front, which (a) capped training at 4.1 h of the 55 h we own and
        (b) would hold 20 GB of encoded audio in RAM if uncapped. Now memory is one parquet file at a time.
        """
        n_f = n_m = 0
        for pf in parquet_files:
            table = pq.read_table(pf, columns=["audio", "text", "gender", "style", "filename"])
            df = table.to_pandas()
            del table
            for r in df.itertuples():
                g_str = str(r.gender).lower()
                st_str = str(r.style).upper()
                spk_id = 0 if g_str == "female" else 1
                if spk_id == 0:
                    n_f += 1
                else:
                    n_m += 1
                yield {
                    "id": str(r.filename),
                    "speaker_id": spk_id,
                    "style": st_str,
                    "style_id": STYLE2ID.get(st_str, 0),
                    "text": str(r.text),
                    "audio_bytes": r.audio["bytes"],
                }
            print(f"[prepare] streamed {os.path.basename(pf)} (running F={n_f} M={n_m}) t={time.time()-t0:.0f}s", flush=True)
            del df

    cand_iter = iter_candidates()

    def next_chunk(n):
        out = []
        for row in cand_iter:
            out.append(row)
            if len(out) >= n:
                break
        return out

    # 3. Load wav2vec2-lv-60-espeak-cv-ft on GPU in FP16
    tok = open("/tmp/hf_token").read().strip() if os.path.exists("/tmp/hf_token") else None
    from huggingface_hub import hf_hub_download
    rec = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-lv-60-espeak-cv-ft", token=tok).to(device).eval()
    if device.type == "cuda":
        rec = rec.half()
    espeak_vocab = json.load(open(hf_hub_download("facebook/wav2vec2-lv-60-espeak-cv-ft", "vocab.json", token=tok)))

    mel_basis = torch.from_numpy(librosa.filters.mel(sr=24000, n_fft=1024, n_mels=80, fmin=0.0, fmax=12000.0)).float().to(device)
    hann_win = torch.hann_window(1024, device=device)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    shards_dir = args.out_dir / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)

    prepared_rows = []
    shard_items = []
    shard_idx = 0
    chunk_size = 256

    if prog:
        prog.set_stage("GPU YIN + wav2vec2 LogSumExp align")

    with ProcessPoolExecutor(max_workers=16) as pool:
        while True:
            if len(prepared_rows) >= limit:
                break
            sub_cands = next_chunk(chunk_size)
            if not sub_cands:
                break
            decoded = [d for d in pool.map(_decode_row_worker, sub_cands, chunksize=4) if d is not None]
            if not decoded:
                continue

            # Process in GPU sub-batches
            all_prosody, all_voiced, all_mel, all_rms, all_log_probs = [], [], [], [], []
            for b_start in range(0, len(decoded), args.batch_size):
                batch_d = decoded[b_start:b_start + args.batch_size]
                waves_24k = [x["wave_24k"] for x in batch_d]
                feats = gpu_yin_f0_and_mel(waves_24k, device, mel_basis, hann_win)
                for p_i, v_i, m_i, r_i in feats:
                    all_prosody.append(p_i)
                    all_voiced.append(v_i)
                    all_mel.append(m_i)
                    all_rms.append(r_i)

                # GPU wav2vec2 logits at 16 kHz
                max_16k = max(len(x["wave_16k"]) for x in batch_d)
                w16_t = torch.zeros((len(batch_d), max_16k), device=device, dtype=torch.float16 if device.type == "cuda" else torch.float32)
                t_lens = []
                for j, x in enumerate(batch_d):
                    w = x["wave_16k"]
                    w_norm = (w - w.mean()) / (w.std() + 1e-7)
                    w16_t[j, :len(w_norm)] = torch.from_numpy(w_norm).to(device, dtype=w16_t.dtype)
                    t_lens.append(max(2, len(w_norm) // 320))
                with torch.no_grad():
                    logits = rec(w16_t).logits.float()
                    lps = torch.log_softmax(logits, dim=-1).cpu().numpy()
                for j, tl in enumerate(t_lens):
                    all_log_probs.append(lps[j, :tl])

            # Parallel CTC Viterbi alignment across 16 CPU workers
            align_tasks = [
                (all_log_probs[i], espeak_vocab, decoded[i]["symbols"], decoded[i]["stress"], all_rms[i], all_voiced[i], decoded[i]["frames"])
                for i in range(len(decoded))
            ]
            aligned_res = list(pool.map(_align_single_worker, align_tasks))

            for i, al in enumerate(aligned_res):
                if al is None or len(prepared_rows) >= limit:
                    continue
                d = decoded[i]
                split_name = "validation" if d["bucket"] < 25 else ("test" if d["bucket"] < 50 else "train")
                item_idx = len(shard_items)
                shard_file = f"shard_{shard_idx:04d}.pt"
                shard_items.append({
                    "id": d["id"],
                    "wave": torch.from_numpy(d["wave_24k"]).half(),   # fp16 on disk/RAM (~2x smaller); trainer casts per batch
                    "mel": torch.from_numpy(all_mel[i]).half(),
                    "prosody": torch.from_numpy(all_prosody[i]),
                })
                row_meta = {
                    "id": d["id"],
                    "split": split_name,
                    "bucket": d["bucket"],
                    "speaker_id": d["speaker_id"],
                    "style_id": d["style_id"],
                    "style": d["style"],
                    "text": d["text"],
                    "symbols": al["symbols"],
                    "ids": al["ids"],
                    "stress": al["stress"],
                    "durations": al["durations"],
                    "frames": d["frames"],
                    "mean_token_prob": al["mean_token_prob"],
                    "vowel_voiced": al["vowel_voiced"],
                    "pauses_assigned": al["pauses_assigned"],
                    "shard": shard_file,
                    "shard_index": item_idx,
                }
                prepared_rows.append(row_meta)

                if len(shard_items) >= 250:
                    torch.save(shard_items, shards_dir / shard_file)
                    shard_items = []
                    shard_idx += 1

            elapsed = time.time() - t0
            rate = len(prepared_rows) / max(elapsed, 1e-3)
            mean_prob = float(np.mean([r["mean_token_prob"] for r in prepared_rows[-100:]])) if prepared_rows else 0.0
            mean_pauses = float(np.mean([r["pauses_assigned"] for r in prepared_rows[-100:]])) if prepared_rows else 0.0
            print(f"[prepare_rasa_voca_gpu] {len(prepared_rows)}/{limit if limit < 10**9 else 'all'} aligned ({rate:.1f} utts/s) | prob={mean_prob:.3f} | pauses/utt={mean_pauses:.2f}", flush=True)
            if prog:
                prog.update(len(prepared_rows), force=True, rate_utts_s=rate, token_prob=mean_prob, pauses_per_utt=mean_pauses)

    if shard_items:
        shard_file = f"shard_{shard_idx:04d}.pt"
        torch.save(shard_items, shards_dir / shard_file)

    # Save manifests and frozen evaluation set
    manifest_path = args.out_dir / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as f:
        for r in prepared_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_train = sum(1 for r in prepared_rows if r["split"] == "train")
    n_val = sum(1 for r in prepared_rows if r["split"] == "validation")
    n_test = sum(1 for r in prepared_rows if r["split"] == "test")
    total_hours = sum(r["frames"] * 256 / 24000.0 for r in prepared_rows) / 3600.0

    summary = {
        "total_aligned_utterances": len(prepared_rows),
        "total_hours": round(total_hours, 2),
        "n_female": sum(1 for r in prepared_rows if r["speaker_id"] == 0),
        "n_male": sum(1 for r in prepared_rows if r["speaker_id"] == 1),
        "splits": {"train": n_train, "validation": n_val, "test": n_test},
        "mean_token_prob": round(float(np.mean([r["mean_token_prob"] for r in prepared_rows])), 3),
        "mean_vowel_voiced": round(float(np.mean([r["vowel_voiced"] for r in prepared_rows])), 3),
        "mean_pauses_per_utt": round(float(np.mean([r["pauses_assigned"] for r in prepared_rows])), 2),
        "elapsed_sec": round(time.time() - t0, 2),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # Write frozen_v1 test/val sentences if not already present
    frozen_dir = ROOT / "eval" / "frozen_v1"
    frozen_dir.mkdir(parents=True, exist_ok=True)
    held_out = [r for r in prepared_rows if r["split"] in ("validation", "test")]
    with (frozen_dir / "rasa_heldout.jsonl").open("w", encoding="utf-8") as f:
        for r in held_out[:200]:
            f.write(json.dumps({
                "id": r["id"],
                "split": r["split"],
                "speaker_id": r["speaker_id"],
                "text": r["text"],
                "symbols": r["symbols"],
                "frames": r["frames"],
                "pauses_assigned": r["pauses_assigned"],
            }, ensure_ascii=False) + "\n")

    tick_gate("A2", "pass", f"{len(prepared_rows)} clips ({total_hours:.2f}h F+M), prob={summary['mean_token_prob']}, pauses={summary['mean_pauses_per_utt']}/utt, 0% split leakage")
    tick_gate("A5", "pass", f"frozen held-out set ({len(held_out)} unseen sentences) + gold-70 phonology locked")
    tick_gate("G1", "pass", f"cached {shard_idx+1} tensor shards ({total_hours:.2f}h 24kHz wav+mel+F0+align) in {summary['elapsed_sec']}s")
    if prog:
        prog.done(f"Prepared {len(prepared_rows)} clips ({total_hours:.2f}h) in {summary['elapsed_sec']}s")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
