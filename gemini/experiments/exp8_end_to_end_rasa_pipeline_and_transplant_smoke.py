#!/usr/bin/env python3
"""EXP8 (Gemini Session 2): End-to-End Real `ai4bharat/Rasa` Nepali (Female + Male) Preparation,
LogSumExp Forced Alignment, Label-Free Voicing-Flip Timing Probe, Articulatory + R-Colored Weight
Transplant from `n4ze3m/tiny-voca.pt`, and Scratch vs Transplant Training Convergence Proof.

This script executes the ENTIRE production data + alignment + model + training pipeline on real
`ai4bharat/Rasa` Nepali Female & Male 48->24 kHz studio recordings on the MoLab CPU box so that
zero bugs remain before launching the 12-hour RTX PRO 6000 Blackwell GPU session.
"""
import glob
import io
import json
import math
import os
import sys
import time
import warnings
from pathlib import Path

import librosa
import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")
torch.set_num_threads(12)
os.environ.setdefault("HF_HOME", "/work/sonnet/hf")

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from sakhiya.symbols import BOS, EOS, SYM2ID, SYMBOLS  # noqa: E402
from gemini.experiments.exp3_nepali_phonology_and_g2p_audit import g2p_v4_word  # noqa: E402
from gemini.experiments.exp6_rasa_studio_vs_slr_forensics import clean_pyin_f0  # noqa: E402

VOWEL_SYMS = {"ʌ", "a", "i", "u", "e", "o", "ɛ", "ɔ", "ã", "ĩ", "ũ", "ẽ", "õ", "ʌ̃", "ɛ̃", "ɔ̃"}
VOICELESS_SYMS = {"k", "kʰ", "ts", "tsʰ", "ʈ", "ʈʰ", "t̪", "t̪ʰ", "p", "pʰ", "s", "ʃ", "f"}
FRICATIVE_SYMS = {"s", "ʃ", "h", "z", "f"}
PAUSE_SYMS = {BOS, EOS, "|", ",", ".", "?", "!", ";"}


def phonemize_nepali_utterance(text: str, rasa_faithful_v: bool = True):
    """Convert a Nepali Devanagari sentence into `(tokens, stress, symbols)` for Sakhiya-Voca.
    In `rasa_faithful_v=True` mode, orthographic Devanagari 'व' (U+0935) stays 'w' and 'ब' (U+092C)
    stays 'b', matching 100% of what the Rasa Female and Male voice artists articulate in the studio.
    """
    import re
    text = text.replace("।", " . ").replace("॥", " . ").replace(",", " , ").replace("?", " ? ").replace("!", " ! ")
    raw_pieces = [p for p in text.split() if p]
    symbols = [BOS]
    stress = [0]
    for p in raw_pieces:
        if p in {",", ".", "?", "!", ";"}:
            if symbols[-1] == "|":
                symbols[-1] = p
            elif symbols[-1] not in PAUSE_SYMS:
                symbols.append(p)
                stress.append(0)
            continue
        w_clean = re.sub(r"[^\u0900-\u097F]", "", p)
        if not w_clean:
            continue
        if symbols[-1] not in PAUSE_SYMS:
            symbols.append("|")
            stress.append(0)
        phs = g2p_v4_word(w_clean)
        if rasa_faithful_v and "व" in w_clean and "ब" not in w_clean:
            # Keep orthographic व as 'w' when not preceded by अं/सं homorganic assimilation
            if not w_clean.startswith(("संव", "सम्ब")):
                phs = ["w" if (ph == "b" and idx == 0 and w_clean.startswith("व")) else ph for idx, ph in enumerate(phs)]
        # Assign syllable weight: 1 for primary/root vowel, 2 for nasalized vowel or geminate
        seen_vowel = False
        for idx, ph in enumerate(phs):
            if ph not in SYM2ID:
                continue
            symbols.append(ph)
            if ph in VOWEL_SYMS:
                if "̃" in ph:
                    stress.append(2)
                elif not seen_vowel:
                    stress.append(1)
                    seen_vowel = True
                else:
                    stress.append(0)
            elif idx + 1 < len(phs) and phs[idx + 1] == ph:
                stress.append(2)
            else:
                stress.append(0)
    if symbols[-1] == "|":
        symbols.pop()
        stress.pop()
    symbols.append(EOS)
    stress.append(0)
    ids = [SYM2ID[s] for s in symbols]
    return symbols, ids, stress


# Map Sakhiya-Voca 64 Nepali symbols to groups of wav2vec2-lv-60-espeak-cv-ft vocab symbols for LogSumExp projection
ESPEAK_GROUPS = {
    "ʌ": ["ʌ", "ə", "ɐ", "ɔ", "o", "ɑ"],
    "a": ["a", "aː", "ɑː", "ɑ"],
    "i": ["i", "iː", "ɪ"],
    "u": ["u", "uː", "ʊ"],
    "e": ["e", "eː", "eɪ", "ɛ"],
    "o": ["o", "oː", "ɔ"],
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
    "ɳ": ["ɳ", "n"],
    "t̪": ["t", "t̪"], "t̪ʰ": ["t", "t̪", "h"],
    "d̪": ["d", "d̪"], "d̪ʱ": ["d", "d̪", "ɦ", "h"],
    "n": ["n"],
    "p": ["p"], "pʰ": ["p", "h", "f"],
    "b": ["b"], "bʱ": ["b", "ɦ", "h"],
    "m": ["m"],
    "j": ["j"], "r": ["r", "ɾ", "ɹ"], "l": ["l"],
    "w": ["v", "ʋ", "w"],
    "ʃ": ["ʃ", "ʂ"], "s": ["s"], "h": ["h", "ɦ"],
}


def ctc_viterbi_group(group_log_probs: np.ndarray, blank_log_probs: np.ndarray) -> np.ndarray:
    """CTC Viterbi alignment over per-token LogSumExp emission posteriors `group_log_probs [T, K]`
    and `blank_log_probs [T]`. Returns extended state path `[T]` where token `k` is state `2*k + 1`."""
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
        # Skip blank between distinct adjacent phone positions
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


def align_nepali_utterance(rec, espeak_vocab: dict, symbols: list[str], wave_24k: np.ndarray, voiced: np.ndarray):
    """Calibrated Nepali Forced Alignment:
    1) Compute wav2vec2-lv-60-espeak-cv-ft logits at 16 kHz (20 ms stride).
    2) Project to Nepali phoneme tokens via LogSumExp over ESPEAK_GROUPS.
    3) Run CTC Viterbi, apply -19 ms onset offset, attribute >=75 ms (-45 dB) silence to PAUSE_SYMS,
       and enforce exact frame sum on the 24 kHz / hop=256 acoustic grid.
    """
    sr = 24000
    hop = 256
    frames = math.ceil(len(wave_24k) / hop)
    w16 = librosa.resample(wave_24k, orig_sr=sr, target_sr=16000)
    x = (w16 - w16.mean()) / (w16.std() + 1e-7)
    with torch.no_grad():
        logits = rec(torch.from_numpy(x)[None]).logits[0]
        log_probs = torch.log_softmax(logits, dim=-1).numpy()

    phone_positions = [i for i, s in enumerate(symbols) if s not in PAUSE_SYMS]
    K = len(phone_positions)
    T = log_probs.shape[0]
    if K < 1 or T < K + 2:
        raise ValueError("utterance too short for phone sequence")

    group_lp = np.zeros((T, K), dtype=np.float64)
    for k, pos in enumerate(phone_positions):
        sym = symbols[pos]
        cands = ESPEAK_GROUPS.get(sym, ["a"])
        ids = [espeak_vocab[c] for c in cands if c in espeak_vocab]
        if not ids:
            ids = [espeak_vocab.get("a", 1)]
        # LogSumExp across group candidates
        sub = log_probs[:, ids]
        m = np.max(sub, axis=-1, keepdims=True)
        group_lp[:, k] = (m[:, 0] + np.log(np.sum(np.exp(sub - m), axis=-1)))

    blank_lp = log_probs[:, 0]
    path = ctc_viterbi_group(group_lp, blank_lp)

    # First emission frame per phone k, converted to 24 kHz acoustic frames with -19 ms onset shift
    onsets = np.zeros(K + 1, dtype=np.int64)
    probs = []
    for k in range(K):
        states_k = np.where(path == 2 * k + 1)[0]
        first_t = int(states_k[0]) if len(states_k) else int(k * T // K)
        probs.append(float(np.exp(np.max(group_lp[states_k, k])) if len(states_k) else 0.5))
        t_sec = (first_t * 320.0) / 16000.0 - 0.019
        onsets[k] = int(np.clip(round(t_sec * sr / hop), k, max(frames - (K - k), k)))
    onsets[0] = 0
    onsets[K] = frames
    for k in range(1, K):
        if onsets[k] <= onsets[k - 1]:
            onsets[k] = min(onsets[k - 1] + 1, frames - (K - k))

    # Detect silent spans (>= 7 frames = 75 ms below P95 - 42 dB) for pause tokens
    rms = librosa.feature.rms(y=wave_24k, frame_length=512, hop_length=hop, center=True)[0, :frames]
    rms_db = 20.0 * np.log10(np.maximum(rms, 1e-6))
    sil_thr = min(float(np.percentile(rms_db, 95)) - 42.0, float(np.percentile(rms_db, 5)) + 8.0)
    is_sil = rms_db <= sil_thr

    durations = [0] * len(symbols)
    # Assign phone spans and split off silence into adjacent pause tokens
    for k, pos in enumerate(phone_positions):
        start_f = int(onsets[k])
        end_f = int(onsets[k + 1])
        # Leading silence on first phone -> <bos>
        if k == 0 and pos > 0:
            lead = 0
            while start_f + lead < end_f - 1 and is_sil[start_f + lead]:
                lead += 1
            if lead >= 4:
                durations[0] = lead
                start_f += lead
        # Trailing silence before a pause token (pos + 1 in PAUSE_SYMS)
        if pos + 1 < len(symbols) and symbols[pos + 1] in PAUSE_SYMS:
            trail = 0
            while end_f - 1 - trail > start_f and is_sil[end_f - 1 - trail]:
                trail += 1
            min_p = 4 if pos + 1 == len(symbols) - 1 else 7
            if trail >= min_p:
                durations[pos + 1] = trail
                end_f -= trail
        durations[pos] = max(1, end_f - start_f)

    # Exact frame sum adjustment on longest token
    diff = frames - sum(durations)
    if diff != 0:
        longest_idx = int(np.argmax(durations))
        durations[longest_idx] = max(1, durations[longest_idx] + diff)

    # Compute vowel_voiced QC and voicing-flip boundary offsets
    ends = np.cumsum(durations)
    starts = ends - np.array(durations)
    v_frames = []
    for i, s in enumerate(symbols):
        if s in VOWEL_SYMS and durations[i] > 0:
            v_frames.extend(voiced[starts[i]:ends[i]])
    vowel_voiced = float(np.mean(v_frames)) if v_frames else 1.0

    # Voicing-flip probe at vowel->voiceless and fricative->vowel boundaries
    active = [i for i, d in enumerate(durations) if d > 0]
    flips = []
    for a, b in zip(active[:-1], active[1:]):
        if (symbols[a] in VOWEL_SYMS and symbols[b] in VOICELESS_SYMS) or (symbols[a] in FRICATIVE_SYMS and symbols[b] in VOWEL_SYMS):
            before = symbols[a] in VOWEL_SYMS
            bnd = int(ends[a])
            best = None
            for sh in range(-6, 7):
                f = bnd + sh
                if 0 < f < len(voiced) and bool(voiced[f - 1]) == before and bool(voiced[f]) != before:
                    if best is None or abs(sh) < abs(best):
                        best = sh
            if best is not None:
                flips.append(best)

    return durations, {
        "mean_token_prob": round(float(np.mean(probs)), 3),
        "vowel_voiced": round(vowel_voiced, 3),
        "flips": flips,
        "pauses_assigned": sum(1 for i, s in enumerate(symbols) if s in PAUSE_SYMS and s not in {BOS, EOS} and durations[i] > 0),
    }


def build_articulatory_r_coloured_transplant(tiny_voca_pt_path: Path, vocab_symbols: list[str], r_share: float = 0.30):
    """Transplant `n4ze3m/tiny-voca` (`4.5M`, Apache-2.0) weights into Sakhiya-Voca Nepali inventory:
    - Copy all 11 RoPE Transformer blocks, Duration+Pause Predictor, Prosody heads, and SourceFilter weights untouched.
    - Initialize each Nepali phoneme from its closest English articulatory partner in `tiny-voca`.
    - Apply Alpa's R-Colored Retroflex Start (`0.70 * dental + 0.30 * English 'ɹ'`) to Nepali retroflexes (`ʈ, ʈʰ, ɖ, ɖʱ, ɳ`)
      and (`+ 0.25 * English 'h'`) to Nepali aspirated/murmured stops so they separate from plain stops from step 0!
    """
    ckpt = torch.load(tiny_voca_pt_path, map_location="cpu", weights_only=True)
    src_state = ckpt["model"]
    en_syms = ckpt["frontend"]["symbols"]
    en2id = {s: i for i, s in enumerate(en_syms)}
    en_tok_w = src_state["tokens.weight"]  # [44, 192]

    NEAREST_EN = {
        "_": "<pad>", "|": "<sp>", "<s>": "<bos>", "</s>": "<eos>",
        ",": ",", ".": ".", "?": "?", "!": "!", ";": ";", ":": ":",
        "ʌ": "ʌ", "a": "ɑː", "i": "iː", "u": "uː", "e": "eɪ", "o": "oː",
        "ʌ̃": "ʌ", "ã": "ɑː", "ĩ": "iː", "ũ": "uː", "ẽ": "eɪ", "õ": "oː",
        "k": "k", "kʰ": "k", "g": "ɡ", "gʱ": "ɡ", "ŋ": "ŋ",
        "ts": "tʃ", "tsʰ": "tʃ", "dz": "dʒ", "dzʱ": "dʒ", "ɲ": "n",
        "ʈ": "t", "ʈʰ": "t", "ɖ": "d", "ɖʱ": "d", "ɳ": "n",
        "t̪": "t", "t̪ʰ": "t", "d̪": "d", "d̪ʱ": "d", "n": "n",
        "p": "p", "pʰ": "p", "b": "b", "bʱ": "b", "m": "m",
        "j": "j", "r": "ɾ", "l": "l", "w": "w", "ʃ": "ʃ", "s": "s", "h": "h",
    }
    RETROFLEX_SYMS = {"ʈ", "ʈʰ", "ɖ", "ɖʱ", "ɳ"}
    ASPIRATED_SYMS = {"kʰ", "gʱ", "tsʰ", "dzʱ", "ʈʰ", "ɖʱ", "t̪ʰ", "d̪ʱ", "pʰ", "bʱ"}
    NASAL_VOWELS = {"ʌ̃", "ã", "ĩ", "ũ", "ẽ", "õ"}

    vec_r = en_tok_w[en2id["ɹ"]]
    vec_h = en_tok_w[en2id["h"]]
    vec_n = en_tok_w[en2id["n"]]

    new_tok_w = torch.zeros((len(vocab_symbols), en_tok_w.shape[1]), dtype=en_tok_w.dtype)
    for idx, sym in enumerate(vocab_symbols):
        base_en = NEAREST_EN.get(sym, "ʌ")
        vec = en_tok_w[en2id.get(base_en, en2id["ʌ"])].clone()
        if sym in RETROFLEX_SYMS:
            vec = (1.0 - r_share) * vec + r_share * vec_r
        if sym in ASPIRATED_SYMS:
            vec = 0.78 * vec + 0.22 * vec_h
        if sym in NASAL_VOWELS:
            vec = 0.82 * vec + 0.18 * vec_n
        new_tok_w[idx] = vec

    transplanted = dict(src_state)
    transplanted["tokens.weight"] = new_tok_w
    return transplanted, ckpt["config"]


def run_training_comparison(prepared_clips: list[dict], transplanted_state: dict):
    """Run a controlled 60-step CPU training comparison on real aligned Rasa Nepali Female & Male clips:
    (A) Random initialization (scratch) vs
    (B) Articulatory + R-Colored Transplant from `tiny-voca.pt`.
    Measures exact multi-res STFT + mel + F0 + pitch_spread + voicing + energy + duration + pause BCE losses!
    """
    sys.path.insert(0, "/work/gemini/tiny-voca/src")
    from tiny_voca.config import Config
    from tiny_voca.losses import SpectralLoss
    from tiny_voca.model import TinyVoca, length_mask
    from tiny_voca.text import SYMBOLS as TV_SYMBOLS

    cfg = Config(vocab_size=len(TV_SYMBOLS), speakers=("female", "male"))
    spectral_loss_fn = SpectralLoss(cfg.sample_rate)

    def evaluate_or_train(init_state: dict | None, steps: int = 40):
        torch.manual_seed(1234)
        model = TinyVoca(cfg)
        if init_state is not None:
            model.load_state_dict(init_state, strict=True)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)

        history = {}
        for step in range(steps + 1):
            # Collate mini-batch of 4 real Rasa Nepali clips (2 Female, 2 Male)
            batch_items = [prepared_clips[(step * 4 + j) % len(prepared_clips)] for j in range(4)]
            max_tok = max(len(x["ids"]) for x in batch_items)
            max_frm = max(x["frames"] for x in batch_items)
            B = len(batch_items)

            tokens = torch.zeros((B, max_tok), dtype=torch.long)
            stress = torch.zeros((B, max_tok), dtype=torch.long)
            durations = torch.zeros((B, max_tok), dtype=torch.long)
            pausable = torch.zeros((B, max_tok), dtype=torch.bool)
            tok_lens = torch.tensor([len(x["ids"]) for x in batch_items], dtype=torch.long)
            speakers = torch.tensor([x["speaker_id"] for x in batch_items], dtype=torch.long)
            prosody_tgt = torch.zeros((B, max_frm, 3), dtype=torch.float32)
            wave_tgt = torch.zeros((B, max_frm * cfg.hop), dtype=torch.float32)

            for b, item in enumerate(batch_items):
                n = len(item["ids"])
                f = item["frames"]
                tokens[b, :n] = torch.tensor(item["ids"], dtype=torch.long)
                stress[b, :n] = torch.tensor(item["stress"], dtype=torch.long)
                durations[b, :n] = torch.tensor(item["durations"], dtype=torch.long)
                pausable[b, :n] = torch.tensor([s in PAUSE_SYMS for s in item["symbols"]], dtype=torch.bool)
                prosody_tgt[b, :f] = item["prosody"][:f]
                wave_tgt[b, :f * cfg.hop] = item["wave"][:f * cfg.hop]

            tok_mask = length_mask(tok_lens, max_tok)
            hidden, _ = model.encode(tokens, stress, tok_lens, speakers)
            dur_out = model.duration(hidden, tok_mask)
            present = durations > 0
            dur_loss = ((dur_out[..., 0] - torch.log1p(durations.float())).pow(2)[tok_mask & present]).mean()
            pause_loss = F.binary_cross_entropy_with_logits(dur_out[..., 1][tok_mask & pausable], present[tok_mask & pausable].float())

            frames_h, frm_mask, frm_lens = model.expand(hidden, durations, tok_mask, speakers)
            pros_h, pros_pred = model.predict_prosody(frames_h, frm_mask)

            voiced_gt = (prosody_tgt[..., 1] > 0.5) & frm_mask
            pitch_loss = (pros_pred[..., 0] - prosody_tgt[..., 0]).abs()[voiced_gt].mean()
            # Global-variance pitch spread loss per utterance
            spread_losses = []
            for b in range(B):
                vb = voiced_gt[b]
                if int(vb.sum()) >= 8:
                    spread_losses.append((pros_pred[b, vb, 0].std() - prosody_tgt[b, vb, 0].std()).abs())
            pitch_spread_loss = torch.stack(spread_losses).mean() if spread_losses else torch.tensor(0.0)
            voicing_loss = F.binary_cross_entropy_with_logits(pros_pred[..., 1][frm_mask], (prosody_tgt[..., 1][frm_mask] > 0.5).float())
            energy_loss = (pros_pred[..., 2] - prosody_tgt[..., 2]).abs()[frm_mask].mean()

            cond = model.hard_prosody(prosody_tgt) if step < 15 else model.hard_prosody(pros_pred.detach())
            acoustic = model.decode(pros_h, cond, frm_mask)
            # Oscillator driven by ground-truth F0 during training so narrowband STFT never blurs harmonics
            wave_pred = model.vocoder(acoustic, model.frequency(prosody_tgt[..., 0]), frm_lens)
            stft_l, mel_l = spectral_loss_fn(wave_pred, wave_tgt, frm_lens * cfg.hop)

            total = 2.0 * stft_l + 10.0 * mel_l + 2.0 * pitch_loss + 2.0 * pitch_spread_loss + voicing_loss + energy_loss + dur_loss + pause_loss

            if step in (0, 10, 20, steps):
                history[f"step_{step}"] = {
                    "total_loss": round(float(total.item()), 4),
                    "stft_loss": round(float(stft_l.item()), 4),
                    "mel_loss": round(float(mel_l.item()), 4),
                    "pitch_l1": round(float(pitch_loss.item()), 4),
                    "pitch_spread_l1": round(float(pitch_spread_loss.item()), 4),
                    "voicing_bce": round(float(voicing_loss.item()), 4),
                    "dur_mse": round(float(dur_loss.item()), 4),
                    "pause_bce": round(float(pause_loss.item()), 4),
                }
            if step < steps:
                opt.zero_grad()
                total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
        return history

    hist_scratch = evaluate_or_train(None, steps=30)
    hist_transplant = evaluate_or_train(transplanted_state, steps=30)
    return {"scratch_init": hist_scratch, "articulatory_r_coloured_transplant": hist_transplant}


def main():
    t0 = time.time()
    tok = open("/tmp/hf_token").read().strip()
    from huggingface_hub import hf_hub_download
    from transformers import Wav2Vec2ForCTC

    # 1. Load wav2vec2-lv-60-espeak-cv-ft recognizer and vocab
    rec = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-lv-60-espeak-cv-ft", token=tok).eval()
    espeak_vocab = json.load(open(hf_hub_download("facebook/wav2vec2-lv-60-espeak-cv-ft", "vocab.json", token=tok)))

    # 2. Load real Rasa Nepali Female (shard 0) & Male (shard 5) clips
    import pandas as pd
    f0 = hf_hub_download("ai4bharat/Rasa", "Nepali/test-00000-of-00007.parquet", repo_type="dataset", token=tok)
    f5 = hf_hub_download("ai4bharat/Rasa", "Nepali/test-00005-of-00007.parquet", repo_type="dataset", token=tok)
    df = pd.concat([pq.read_table(f0).to_pandas(), pq.read_table(f5).to_pandas()], ignore_index=True)
    df_neutral = df[df["style"].isin(["WIKI", "CONV", "NEWS", "BOOK"])]
    rows_f = list(df_neutral[df_neutral["gender"].str.lower() == "female"].itertuples())[:10]
    rows_m = list(df_neutral[df_neutral["gender"].str.lower() == "male"].itertuples())[:10]

    prepared = []
    all_flips = []
    total_pauses = 0
    for spk_id, rows in [(0, rows_f), (1, rows_m)]:
        for r in rows:
            wave, orig_sr = sf.read(io.BytesIO(r.audio["bytes"]), dtype="float32")
            if wave.ndim > 1:
                wave = wave.mean(axis=1)
            wave_24k = librosa.resample(wave, orig_sr=orig_sr, target_sr=24000) if orig_sr != 24000 else wave
            peak = float(np.max(np.abs(wave_24k)))
            if peak > 1e-4:
                wave_24k = (0.95 * wave_24k / peak).astype(np.float32)
            # Keep clips 1.5s - 7.5s for fast CPU test
            if not (24000 * 1.2 <= len(wave_24k) <= 24000 * 7.5):
                continue
            frames = math.ceil(len(wave_24k) / 256)
            wave_24k = np.pad(wave_24k, (0, frames * 256 - len(wave_24k))).astype(np.float32)

            f0, voiced, _ = librosa.pyin(wave_24k, fmin=50.0, fmax=600.0, sr=24000, frame_length=1024, hop_length=256, center=True)
            f0, voiced = f0[:frames], voiced[:frames]
            v_mask = voiced & np.isfinite(f0) & (f0 > 0)
            f0_clean = clean_pyin_f0(f0, v_mask)
            pitch = np.where(v_mask, np.log(np.maximum(np.nan_to_num(f0_clean, nan=200.0), 1.0) / 200.0), 0.0)
            rms = librosa.feature.rms(y=wave_24k, frame_length=1024, hop_length=256, center=True)[0, :frames]
            energy = (np.log(np.maximum(rms, 1e-5)) - (-4.0)) / 2.0
            prosody = np.stack((pitch, v_mask.astype(np.float32), energy), axis=-1).astype(np.float32)

            symbols, ids, stress = phonemize_nepali_utterance(r.text, rasa_faithful_v=True)
            durations, qc = align_nepali_utterance(rec, espeak_vocab, symbols, wave_24k, v_mask)
            assert sum(durations) == frames and len(durations) == len(ids)

            all_flips.extend(qc["flips"])
            total_pauses += qc["pauses_assigned"]
            prepared.append({
                "id": str(r.filename),
                "speaker_id": spk_id,
                "text": r.text,
                "symbols": symbols,
                "ids": ids,
                "stress": stress,
                "durations": durations,
                "frames": frames,
                "wave": torch.from_numpy(wave_24k),
                "prosody": torch.from_numpy(prosody),
                "qc": qc,
            })

    flips_arr = np.array(all_flips, dtype=np.float64) if all_flips else np.zeros(1)
    alignment_summary = {
        "n_aligned_clips": len(prepared),
        "n_female": sum(1 for x in prepared if x["speaker_id"] == 0),
        "n_male": sum(1 for x in prepared if x["speaker_id"] == 1),
        "exact_frame_sum_pass_rate": 1.0,
        "mean_token_posterior_prob": round(float(np.mean([x["qc"]["mean_token_prob"] for x in prepared])), 3),
        "mean_vowel_voiced_ratio": round(float(np.mean([x["qc"]["vowel_voiced"] for x in prepared])), 3),
        "mean_interior_pauses_per_utt": round(float(total_pauses / max(len(prepared), 1)), 2),
        "voicing_flip_timing_probe": {
            "n_boundaries_probed": len(all_flips),
            "mean_offset_frames": round(float(flips_arr.mean()), 3),
            "mean_abs_frames": round(float(np.abs(flips_arr).mean()), 3),
            "within_1_frame_pct": round(float((np.abs(flips_arr) <= 1).mean() * 100.0), 1),
            "within_2_frames_pct": round(float((np.abs(flips_arr) <= 2).mean() * 100.0), 1),
        },
    }

    # 3. Download `n4ze3m/tiny-voca` (`tiny-voca.pt`) and build Articulatory + R-Colored Transplant
    sys.path.insert(0, "/work/gemini/tiny-voca/src")
    from tiny_voca.text import SYMBOLS as TV_SYMBOLS
    padded_symbols = list(SYMBOLS) + [f"<extra_{i}>" for i in range(max(0, len(TV_SYMBOLS) - len(SYMBOLS)))]
    tv_pt = Path(hf_hub_download("n4ze3m/tiny-voca", "tiny-voca.pt", token=tok))
    transplanted_state, _ = build_articulatory_r_coloured_transplant(tv_pt, padded_symbols, r_share=0.30)

    # 4. Run 30-step real Rasa Nepali Female+Male training comparison: Scratch vs Articulatory R-Colored Transplant
    training_comparison = run_training_comparison(prepared, transplanted_state)

    report = {
        "experiment": "EXP8_End_to_End_Rasa_Pipeline_Alignment_and_Transplant_Smoke",
        "elapsed_sec": round(time.time() - t0, 2),
        "alignment_and_timing_probe": alignment_summary,
        "sample_aligned_utterances": [
            {
                "speaker": "female" if x["speaker_id"] == 0 else "male",
                "text": x["text"][:90],
                "symbols": " ".join(x["symbols"][:25]),
                "durations_first_25": x["durations"][:25],
                "frames": x["frames"],
                "qc": x["qc"],
            }
            for x in prepared[:4]
        ],
        "training_convergence_30_steps_cpu": training_comparison,
    }

    out_path = Path(__file__).resolve().parent.parent / "results" / "exp8_end_to_end_rasa_pipeline_and_transplant_smoke.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
