"""Torch-free runtime: numpy + onnxruntime.   from sakhiya.onnx_tts import SakhiyaONNX"""
import json, os
import numpy as np
import onnxruntime as ort
from . import g2p, symbols as S

STYLES = {
    "neutral": dict(style_id=0, rate=0.96, pitch_shift=0.0, pitch_scale=1.12, energy_scale=1.0, pause_scale=1.0),
    "news": dict(style_id=0, rate=0.96, pitch_shift=-0.03, pitch_scale=1.12, energy_scale=1.06, pause_scale=1.0),
    "conversational": dict(style_id=1, rate=1.00, pitch_shift=0.04, pitch_scale=1.32, energy_scale=1.02, pause_scale=0.88),
    "storytelling": dict(style_id=2, rate=0.90, pitch_shift=0.02, pitch_scale=1.32, energy_scale=1.04, pause_scale=1.35),
    "warm_soft": dict(style_id=3, rate=0.96, pitch_shift=-0.03, pitch_scale=1.18, energy_scale=0.92, pause_scale=1.15),
    "slow_clear": dict(style_id=3, rate=0.78, pitch_shift=0.0, pitch_scale=1.0, energy_scale=1.0, pause_scale=1.3),
    "expressive": dict(style_id=4, rate=1.00, pitch_shift=0.08, pitch_scale=1.32, energy_scale=1.08, pause_scale=0.92),
    "vedic_chant": dict(style_id=5, rate=0.90, pitch_shift=0.0, pitch_scale=0.92, energy_scale=1.04, pause_scale=1.28),
    "chant": dict(style_id=5, rate=0.90, pitch_shift=0.0, pitch_scale=0.92, energy_scale=1.04, pause_scale=1.28),
}
import re
PAUSE = np.array([S.SYM2ID[p] for p in [S.WORD] + S.PUNCT])
FINITE = {"छ", "छन्", "छु", "छौ", "छैन", "छैनन्", "हो", "हुन्", "थियो", "थिए", "थिएन", "थिएनन्", "भयो", "भए", "गर्यो", "गरे", "गर्छ", "गर्छन्",
          "हुन्छ", "हुन्छन्", "पर्छ", "पर्दछ", "सक्छ", "सकिन्छ", "गयो", "आयो", "दियो", "लियो", "भन्यो", "बताए", "पाइन्छ", "मानिन्छ", "गरियो", "देखिन्छ", "रहेछ",
          "बा", "बाटै", "बाटु", "रहे", "है", "नाहै", "बडयी", "बडिय", "हलयी"}


def _segment(text):
    sents = [s.strip() for s in re.split(r"(?<=[।॥])\s*|(?<=[.?!])\s+", str(text).strip()) if s.strip()]
    out = []
    for s in sents:
        ws = s.split()
        if len(ws) <= 20:
            out.append(s); continue
        cur = []
        for i, w in enumerate(ws):
            cur.append(w)
            if (w in FINITE and len(cur) >= 4 and len(ws) - i - 1 >= 3) or len(cur) >= 12:
                out.append(" ".join(cur)); cur = []
        if cur:
            if out and len(cur) < 3: out[-1] += " " + " ".join(cur)
            else: out.append(" ".join(cur))
    return out or [str(text)]


def _declick_np(w, n_fft=512, hop=256):
    """Localized STFT transient smoother: only blends isolated click frames (flux_z > 7.5), leaving 99.9999% of w untouched."""
    w = np.asarray(w, dtype=np.float32).copy()
    if w.size < n_fft * 2:
        return w
    pad = n_fft // 2
    win = np.hanning(n_fft + 1)[:-1].astype(np.float32)
    w2 = win * win
    for _ in range(3):
        wp = np.pad(w, (pad, pad), mode="reflect")
        n_frames = 1 + (len(wp) - n_fft) // hop
        frames = np.lib.stride_tricks.sliding_window_view(wp, n_fft)[::hop][:n_frames] * win
        S = np.fft.rfft(frames, n_fft, axis=1).T
        spec = np.abs(S).astype(np.float32)
        flux = np.sqrt(np.mean(np.maximum(0.0, np.diff(np.log(spec + 1e-4), axis=1)) ** 2, axis=0))
        flux_z = (flux - np.median(flux)) / (np.std(flux) + 1e-5)
        bad = np.where(flux_z > 7.5)[0]
        if not len(bad):
            break
        for t in bad:
            S[:, t] = 0.65 * S[:, t] + 0.35 * S[:, t + 1]
        fr_out = np.fft.irfft(S, n_fft, axis=0).T * win
        L = (n_frames - 1) * hop + n_fft
        y = np.zeros(L, dtype=np.float32)
        env = np.zeros(L, dtype=np.float32)
        for t in range(n_frames):
            y[t * hop:t * hop + n_fft] += fr_out[t]
            env[t * hop:t * hop + n_fft] += w2
        w = (y[pad:pad + len(w)] / np.maximum(env[pad:pad + len(w)], 1e-8)).astype(np.float32)
    return w


def stitch(wavs, sr, xf_ms=25, edge_ms=90):
    """Trim each clip's near-silent edges to <= edge_ms (so no double pauses), then equal-power crossfade."""
    clips = []
    for w in wavs:
        w = np.asarray(w, np.float32)
        a = np.abs(w); thr = max(float(a.max()) * 0.006, 1e-4) if a.size else 1.0
        idx = np.where(a > thr)[0]
        if idx.size:
            keep = int(edge_ms * sr / 1000)
            w = w[max(0, idx[0] - keep): idx[-1] + 1 + keep]
        clips.append(w)
    if not clips: return np.zeros(0, np.float32)
    n = int(xf_ms * sr / 1000); out = clips[0]
    for w in clips[1:]:
        m = min(n, len(out), len(w))
        if m < 8: out = np.concatenate([out, w]); continue
        t = np.linspace(0, np.pi / 2, m, dtype=np.float32)
        out = np.concatenate([out[:-m], out[-m:] * np.cos(t) + w[:m] * np.sin(t), w[m:]])
    return _declick_np(out.astype(np.float32))


class SakhiyaONNX:
    def __init__(self, d, tiny=False, threads=1):
        o = ort.SessionOptions(); o.intra_op_num_threads = threads
        suf = ".int8.onnx" if tiny else ".onnx"
        mk = lambda n: ort.InferenceSession(os.path.join(d, n + (suf if os.path.exists(os.path.join(d, n + suf)) else ".onnx")), o, providers=["CPUExecutionProvider"])
        self.cfg = json.load(open(os.path.join(d, "config.json")))
        m_p = os.path.join(d, "model" + (suf if os.path.exists(os.path.join(d, "model" + suf)) else ".onnx"))
        has = lambda n: os.path.exists(os.path.join(d, n + suf)) or os.path.exists(os.path.join(d, n + ".onnx"))
        self.single = ort.InferenceSession(m_p, o, providers=["CPUExecutionProvider"]) if os.path.exists(m_p) and not has("encoder") else None
        self.enc = mk("encoder") if has("encoder") else self.single
        self.dec = mk("decoder") if has("decoder") else None
        self.voc = mk("vocoder") if has("vocoder") else None
        sa_p = os.path.join(d, "sanskrit.onnx")
        self.sa = ort.InferenceSession(sa_p, o, providers=["CPUExecutionProvider"]) if os.path.exists(sa_p) else None
        self.trace = []
        self.enc_in = [i.name for i in self.enc.get_inputs()]
        self.win = np.hanning(1025)[:-1].astype(np.float32)
        eq_d = json.load(open(os.path.join(os.path.dirname(__file__), "data", "voc_eq.json")))
        pts = np.asarray(eq_d["pts"], dtype=np.float32)
        self.eq = (np.interp(np.linspace(0, len(pts) - 1, 513), np.arange(len(pts)), pts) ** 0.78).astype(np.float32)[:, None]

    def _say_one(self, text, spk, lang, k, is_last=True):
        if lang == "sa" and self.sa is not None:
            from . import sanskrit as _sa
            ids, wt, off, _ = _sa.phonemize_slp1(text)
            v_id = 1 if spk in (1, 6, 20, 21) else 0
            sv_sc = float(k.get("svara", 1.0))
            p_add = np.array([[o * 0.125 * sv_sc for o in off]], dtype=np.float32)
            w, dur = self.sa.run(None, {
                "tokens": np.array([ids], dtype=np.int64),
                "weight": np.array([wt], dtype=np.int64),
                "voice": np.array([v_id], dtype=np.int64),
                "speed": np.array([float(k["rate"])], dtype=np.float32),
                "pitch_shift": np.array([float(k["pitch_shift"]) * 0.12], dtype=np.float32),
                "pitch_range": np.array([float(k["pitch_scale"])], dtype=np.float32),
                "pitch_add": p_add,
            })
            w = w[0].astype(np.float32)
            self.trace.append((np.array(off, dtype=np.float32) * 0.9 * sv_sc, dur[0].astype(int).copy()))
            pk = float(np.max(np.abs(w))) if w.size else 0.0
            if pk > 1e-4: w = np.clip(w * (0.90 / pk), -1.0, 1.0)
            if w.size > 256:
                ramp = (0.5 - 0.5 * np.cos(np.pi * np.arange(128, dtype=np.float32) / 128)).astype(np.float32)
                w[:128] *= ramp; w[-128:] *= ramp[::-1]
            return w
        is_v2 = str(self.cfg.get("arch", "")).startswith(("sakhiya-v2", "sakhiya-v3", "sakhiya-v4", "sakhiya-v6", "sakhiya-v15", "sakhiya-nano", "sakhiya-atom", "sakhiya-pico"))
        chandas_info = None
        if lang == "sa":
            from . import chandas as _ch
            chandas_info = _ch.compile_chandas(text, bpm=float(k.get("bpm", 75.0)), tonic_hz=float(k.get("tonic_hz", 215.0)))
            ph, off, dm = chandas_info["ph"], chandas_info["off"], chandas_info["dm"]
        else:
            ph, off = g2p.text_to_phones_acc(text, lang); dm = g2p.phone_dm(ph)
        sid = [S.SYM2ID.get(p, -1) for p in ph]
        dm_arr = np.array([d for d, i in zip(dm, sid) if i >= 0], dtype=np.float32)
        ids = np.array([[i for i in sid if i >= 0]], dtype=np.int64)
        lock = bool(k.get("lock_voice", self.cfg.get("lock_voice", False)))   # one-voice lock: Sanskrit reuses the Nepali language row
        eff_lang = "ne" if (is_v2 and (str(lang) != "sa" or lock)) else lang
        lang_id = self.cfg.get("lang2id", {**S.LANG2ID, "sa": 0}).get(eff_lang, 0)
        feeds_add = {}
        trace_p = np.array([o * 2.40 * k.get("svara", 1.0) for o, i in zip(off, sid) if i >= 0], np.float32)
        if "pitch_add" in self.enc_in:      # Training-matched svara scale (0.52 for explicit Vedic svara marks, 0.48 for classical shlokas)
            has_vm = bool(chandas_info and chandas_info.get("has_vedic_marks"))
            sv_sc = ((0.52 if has_vm else 0.48) if lang == "sa" else 1.15) if is_v2 else 1.12
            feeds_add["pitch_add"] = np.array([[o * sv_sc * k.get("svara", 1.0) for o, i in zip(off, sid) if i >= 0]], np.float32)
        if "style" in self.enc_in or "style2id" in self.cfg:
            feeds_add["style"] = np.array([int(k.get("style_id", 0))], dtype=np.int64)
        if "contour" in self.enc_in:
            L_ids = ids.shape[1]
            c_arr = np.zeros((1, L_ids, 3), dtype=np.float32)
            if k.get("contour") is not None:
                c_in = np.asarray(k["contour"], dtype=np.float32)
                if c_in.ndim == 2 and c_in.shape[1] == 3 and c_in.shape[0] > 0:
                    idx_c = np.clip(np.round(np.linspace(0, c_in.shape[0] - 1, L_ids)).astype(int), 0, c_in.shape[0] - 1)
                    c_arr[0] = c_in[idx_c]
            feeds_add["contour"] = c_arr
        if self.single is not None and self.dec is None:
            if "dur_scale" in self.enc_in:
                is_news = "news" in str(self.cfg.get("arch", "")) or "news" in str(self.cfg.get("tier", ""))
                raw_list = ids[0].tolist()
                if is_news and lang != "sa":
                    # Sakhiya-News: preserve explicit <breath> token (ID 8) and map interior clause commas (ID 4) to <breath> (ID 8)
                    raw_ids = [(8 if (x in (4, 8) and 0 < idx_x < len(raw_list) - 1) else x) for idx_x, x in enumerate(raw_list)]
                else:
                    raw_ids = [5 if x == 8 else x for x in raw_list if not (lang == "sa" and x == 4)]
                ids_s = ([1] + raw_ids) if (not raw_ids or raw_ids[0] != 1) else raw_ids
                ids_in = np.array([ids_s], dtype=np.int64)
                has_vm = bool(chandas_info and chandas_info.get("has_vedic_marks"))
                rate_val = max(float(k.get("rate", 1.0)), 0.5)
                base_ps = float(k.get("pitch_shift", 0.0))
                sv_sc = ((0.65 if has_vm else 0.52) if lang == "sa" else 0.35) * float(k.get("svara", 1.0))
                if lang == "sa":
                    ds_base = [(0.75 + 0.25 * min(max(float(d), 0.85), 1.35)) / rate_val for d in dm_arr]
                    ps_base = [base_ps + float(o) * sv_sc for o, i in zip(off, sid) if i >= 0]
                    if len(ds_base) + 1 == len(ids_s):
                        ds_in = np.array([0.88 / rate_val] + ds_base, dtype=np.float32)
                        ps_in = np.array([base_ps] + ps_base, dtype=np.float32)
                    else:
                        ds_in = np.full(len(ids_s), 0.88 / rate_val, dtype=np.float32)
                        ps_in = np.full(len(ids_s), base_ps + 0.15, dtype=np.float32)
                else:
                    ds_in = np.full(len(ids_s), 0.94 / rate_val, dtype=np.float32)
                    ps_in = np.full(len(ids_s), base_ps + 0.04, dtype=np.float32)
                w = self.single.run(None, {
                    "ids": ids_in,
                    "lang": np.array([1 if lang == "sa" else 0], dtype=np.int64),
                    "dur_scale": ds_in,
                    "pitch_shift": ps_in,
                })[0][0].astype(np.float32)
                n_fr = max(len(w) // 256, ids.shape[1] * 5)
                if chandas_info and "dur_frames" in chandas_info:
                    df_keep = np.array([d for d, i in zip(chandas_info["dur_frames"], sid) if i >= 0], dtype=np.float32)
                    dur_est = np.maximum(np.round(df_keep * (n_fr / max(float(np.sum(df_keep)), 1.0))), 5).astype(int)
                else:
                    dur_est = np.maximum(np.round(dm_arr * (n_fr / max(float(np.sum(dm_arr)), 1.0))), 5).astype(int) if dm_arr.size else np.full(ids.shape[1], 6, dtype=int)
                self.trace.append((trace_p, dur_est))
                pk = float(np.max(np.abs(w))) if w.size else 0.0
                if pk > 1e-4:
                    w = np.clip(w * (0.90 / pk), -1.0, 1.0)
                if w.size > 256:
                    ramp = (0.5 - 0.5 * np.cos(np.pi * np.arange(128, dtype=np.float32) / 128)).astype(np.float32)
                    w[:128] *= ramp; w[-128:] *= ramp[::-1]
                return w
            dur_ov = np.maximum(np.round(dm_arr * 6.0 / k["rate"]), 3).astype(np.int64)[None, :]
            feeds_add["dur_override"] = dur_ov
            re_o, im_o, ld = self.single.run(None, {"ids": ids, "spk": np.array([spk]), "lang": np.array([lang_id]),
                                                    "pitch_shift": np.array(k["pitch_shift"], np.float32), "pitch_scale": np.array(k["pitch_scale"], np.float32),
                                                    "energy_scale": np.array(k["energy_scale"], np.float32), **feeds_add})
            self.trace.append((trace_p if feeds_add else np.zeros(dur_ov.size, np.float32), dur_ov[0].astype(int).copy()))
            w = self._ola64(re_o[0] + 1j * im_o[0])
            if w.size > 256:
                ramp = (0.5 - 0.5 * np.cos(np.pi * np.arange(128, dtype=np.float32) / 128)).astype(np.float32)
                w[:128] *= ramp; w[-128:] *= ramp[::-1]
                pad_n = int(0.055 * self.cfg.get("sample_rate", 24000))
                w = np.concatenate([np.zeros(pad_n, dtype=np.float32), w, np.zeros(pad_n // 2, dtype=np.float32)])
            return w
        x, ld = self.enc.run(None, {"ids": ids, "spk": np.array([spk]), "lang": np.array([lang_id]),
                                    "pitch_shift": np.array(k["pitch_shift"], np.float32), "pitch_scale": np.array(k["pitch_scale"], np.float32),
                                    "energy_scale": np.array(k["energy_scale"], np.float32), **feeds_add})
        dur = np.clip(np.exp(ld[0]) - 1, 1, None) * (dm_arr if dm_arr.size == ld[0].size else 1.0)
        dur = np.where(np.isin(ids[0], PAUSE), dur * k["pause_scale"], dur)
        dur = np.maximum(np.round(dur / k["rate"]), 1).astype(int)
        ph_valid = [p for p, i in zip(ph, sid) if i >= 0]
        for idx_p, p in enumerate(ph_valid):
            if p not in ("|", ",", ".", "?", "!", ";"):
                dur[idx_p] = max(int(dur[idx_p]), 3)
            if p == "h" and (idx_p + 1 == len(ph_valid) or ph_valid[idx_p + 1] in ("|", ",", ".", "?", "!", ";")):
                dur[idx_p] = min(int(dur[idx_p]), 6)
            if idx_p >= 1 and idx_p + 1 < len(ph_valid) and p == "|" and ph_valid[idx_p - 1] == ph_valid[idx_p + 1] and ph_valid[idx_p - 1] in ("a", "i", "u", "e", "o", "ʌ"):
                if idx_p >= 2 and ph_valid[idx_p - 2] == "|":
                    dur[idx_p - 2] = 12
                    x[0, idx_p - 2, :100] = x[0, -1, :100]
                dur[idx_p - 1] = max(int(dur[idx_p - 1]), 19)
                dur[idx_p] = 7
                dur[idx_p + 1] = max(int(dur[idx_p + 1]), 18)
                x[0, idx_p - 1, :100] = x[0, idx_p + 1, :100]
                x[0, idx_p, :100] = 0.45 * x[0, idx_p + 1, :100] + 0.55 * x[0, -1, :100]
            if lang == "sa" and p == "o" and idx_p + 1 < len(ph_valid) and ph_valid[idx_p + 1] == "m" and (idx_p == 0 or ph_valid[idx_p - 1] in ("|", ",", ";", ".")):
                dur[idx_p] = int(np.clip(dur[idx_p], 34, 44))
                dur[idx_p + 1] = int(np.clip(dur[idx_p + 1], 16, 22))
                if idx_p + 2 < len(ph_valid) and ph_valid[idx_p + 2] == ",":
                    dur[idx_p + 2] = int(np.clip(dur[idx_p + 2], 14, 22))
            if idx_p >= 1 and idx_p + 2 < len(ph_valid) and ph_valid[idx_p - 1] in ("|", ",") and ph_valid[idx_p + 2] in ("|", ",") and p in ("r", "t̪", "n", "m", "h", "tsʰ") and ph_valid[idx_p + 1] in ("ʌ", "o"):
                dur[idx_p] = int(np.clip(dur[idx_p], 5, 9))
                dur[idx_p + 1] = int(np.clip(dur[idx_p + 1], 6, 11))
        if not is_last and dur.size:
            dur[-1] = max(1, int(round(7 * k["pause_scale"])))
        self.trace.append((trace_p if feeds_add else np.zeros(dur.size, np.float32), dur.copy()))
        xe = np.repeat(x, dur, axis=1)
        mel = self.dec.run(None, {"x": xe})[0]
        voc_out = self.voc.run(None, {"mel": mel})
        if len(voc_out) == 1:
            w = voc_out[0][0].astype(np.float32)
            pk = float(np.max(np.abs(w))) if w.size else 0.0
            if pk > 1e-4: w = np.clip(w * (0.90 / pk), -1.0, 1.0)
        elif voc_out[0].shape[1] == 33:
            w = self._ola64(voc_out[0][0] + 1j * voc_out[1][0])
        else:
            w = self.istft(voc_out[0][0] + 1j * voc_out[1][0])
        if w.size > 256:
            ramp = (0.5 - 0.5 * np.cos(np.pi * np.arange(128, dtype=np.float32) / 128)).astype(np.float32)
            w[:128] *= ramp; w[-128:] *= ramp[::-1]
            pad_n = int(0.055 * self.cfg.get("sample_rate", 24000))
            w = np.concatenate([np.zeros(pad_n, dtype=np.float32), w, np.zeros(pad_n // 2, dtype=np.float32)])
        return w

    def _ola64(self, spec, n_fft=64, hop=16):
        win = np.hanning(n_fft + 1)[:-1].astype(np.float32)
        T = spec.shape[1]
        fr = np.fft.irfft(spec, n_fft, axis=0).T * win
        L = (T - 1) * hop + n_fft; y = np.zeros(L, np.float32); env = np.zeros(L, np.float32); w2 = win ** 2
        for t in range(T):
            y[t * hop:t * hop + n_fft] += fr[t]; env[t * hop:t * hop + n_fft] += w2
        p = (n_fft - hop) // 2
        out = (y[p:-p] / np.maximum(env[p:-p], 1e-8)).astype(np.float32)
        pk = float(np.max(np.abs(out))) if out.size else 0.0
        return np.clip(out * (0.90 / pk), -1.0, 1.0) if pk > 1e-4 else out

    def say_mixed(self, text, speaker=None, style="neutral", tharu="thr_dangaura", **ctl):
        """lang='auto': per-clause language routing, ONE speaker embedding, breath-matched crossfade stitching."""
        from . import mixed
        wavs, sr0, self.trace, tr = [], None, [], []
        for L, seg in mixed.route(text, tharu):
            w, sr = self.say(seg, speaker, L, style, **ctl)
            tr += self.trace
            if sr0 is None: sr0 = sr
            elif sr != sr0: w = np.interp(np.linspace(0, len(w) - 1, int(len(w) * sr0 / sr)), np.arange(len(w)), w).astype(np.float32)
            wavs.append(w)
        self.trace = tr
        return stitch(wavs, sr0 or self.cfg["sample_rate"]), (sr0 or self.cfg["sample_rate"])

    def say(self, text, speaker=None, lang="ne", style="neutral", **ctl):
        if str(lang) == "auto":
            return self.say_mixed(text, speaker, style, **ctl)
        self.trace = []     # per-chunk (pitch_add, durations) for the svara measurement in scripts/evaluate.py
        k = dict(STYLES[style]); k.update(ctl)
        speaker = speaker or self.cfg.get("default_speaker")     # speaker ids are no longer 0/1 once the multi-speaker mix is exported
        spk = self.cfg["spk2id"].get(speaker, 0) if speaker else 0
        use_sa = (lang == "sa" and self.sa is not None)
        if lang == "sa" and style not in ("chant", "vedic_chant", "storytelling") and "rate" not in ctl:
            k["rate"] = (0.98 if spk not in (1, 6, 20, 21) else 1.00) if use_sa else k["rate"] * 0.96
            k["pause_scale"] *= 1.05
            if "pitch_scale" not in ctl: k["pitch_scale"] = 1.00
        if str(lang).startswith("thr") and "pitch_scale" not in ctl:
            if str(lang) in ("thr_rana", "thr_chitwania"):
                k["rate"] = k["rate"] * 0.90; k["pitch_scale"] = 1.00; k["pause_scale"] = k["pause_scale"] * 1.05
            else:
                k["rate"] = k["rate"] * 0.90; k["pitch_scale"] = 1.00; k["pause_scale"] = k["pause_scale"] * 1.00
        sr = 24000 if use_sa else self.cfg["sample_rate"]
        sa_segs = _segment(text)
        is_single = (self.single is not None and self.dec is None)
        chunks = [str(text).strip()] if ((lang == "sa" or is_single) and len(str(text).split()) <= 45 and len(sa_segs) <= 2) else sa_segs
        if len(chunks) == 1:
            return _declick_np(self._say_one(chunks[0], spk, lang, k, True)), sr
        wavs = [self._say_one(c, spk, lang, k, i == len(chunks) - 1) for i, c in enumerate(chunks)]
        return _declick_np(np.concatenate(wavs)), sr

    def _ola(self, spec, n_fft=1024, hop=256):
        T = spec.shape[1]
        fr = np.fft.irfft(spec, n_fft, axis=0).T * self.win
        L = (T - 1) * hop + n_fft; y = np.zeros(L, np.float32); env = np.zeros(L, np.float32); w2 = self.win ** 2
        for t in range(T):
            y[t * hop:t * hop + n_fft] += fr[t]; env[t * hop:t * hop + n_fft] += w2
        p = (n_fft - hop) // 2
        return (y[p:-p] / np.maximum(env[p:-p], 1e-8)).astype(np.float32)

    def _stft(self, w, n_fft=1024, hop=256):
        p = (n_fft - hop) // 2
        wp = np.pad(w, (p, p), mode="reflect")
        frames = np.lib.stride_tricks.sliding_window_view(wp, n_fft)[::hop] * self.win
        return np.fft.rfft(frames, n_fft, axis=1).T

    def istft(self, spec, n_fft=1024, hop=256):
        w0 = self._ola(spec, n_fft, hop)
        if w0.size < n_fft:
            pk = float(np.max(np.abs(w0))) if w0.size else 0.0
            return np.clip(w0 * (0.90 / pk) if pk > 1e-4 else w0, -1.0, 1.0).astype(np.float32)
        mag0 = np.abs(spec).astype(np.float32)
        s1 = self._stft(w0, n_fft, hop); mag_r = np.abs(s1).astype(np.float32); ph_r = np.angle(s1)
        w_m = np.full((mag0.shape[0], 1), 0.22, dtype=np.float32); w_m[:160, 0] = 0.35
        mb = w_m * mag0 + (1.0 - w_m) * mag_r
        v_ratio = np.clip(mb[4:84, :].sum(axis=0, keepdims=True) / (mb.sum(axis=0, keepdims=True) + 1e-6), 0.0, 1.0)
        vo = np.clip((v_ratio - 0.35) / 0.35, 0.0, 1.0)
        c_t = 1.14 + 0.34 * vo
        mp = np.maximum(mb, 1e-5) ** c_t
        mc = mp * (mb.sum(axis=0, keepdims=True) / (mp.sum(axis=0, keepdims=True) + 1e-6)) * self.eq
        w1 = self._ola(mc * np.exp(1j * ph_r), n_fft, hop)
        s2 = self._stft(w1, n_fft, hop); ph2 = np.angle(s2 + 0.27 * (s2 - s1))
        w2 = self._ola(mc * np.exp(1j * ph2), n_fft, hop)
        s3 = self._stft(w2, n_fft, hop); ph3 = np.angle(s3 + 0.27 * (s3 - s2))
        w = self._ola(mc * np.exp(1j * ph3), n_fft, hop)
        pk = float(np.max(np.abs(w))) if w.size else 0.0
        return np.clip(w * (0.90 / pk) if pk > 1e-4 else w, -1.0, 1.0).astype(np.float32)
