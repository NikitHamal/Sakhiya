"""F0-contour accent proxies shared by the v3 audit / eval / renderer."""
import numpy as np, librosa


def accent_metrics(w, sr):
    """Pitch-contour accent proxies. Positive end_slope = clause ends float UP (English-like); native Nepali closes DOWN."""
    w16 = librosa.resample(np.asarray(w, np.float32), orig_sr=sr, target_sr=16000)
    hop = 160
    f0 = librosa.yin(w16, fmin=75, fmax=420, sr=16000, frame_length=1024, hop_length=hop)
    rms = librosa.feature.rms(y=w16, frame_length=1024, hop_length=hop)[0][:len(f0)]
    f0 = f0[:len(rms)]
    voiced = (rms > 0.12 * np.percentile(rms, 95)) & (f0 > 80) & (f0 < 400)
    if voiced.sum() < 30: return dict(end_slope=0.0, rise_frac=0.0, decl=0.0, f0_std=0.0, f0_med=0.0)
    st = 12 * np.log2(np.where(voiced, f0, np.nan) / 100.0)
    t = np.arange(len(st)) * hop / 16000.0
    # median-smooth the voiced contour to kill octave glitches
    sm = st.copy(); idx = np.where(voiced)[0]
    for k in idx: sm[k] = np.nanmedian(st[max(0, k - 3):k + 4])
    ok = ~np.isnan(sm)
    decl = float(np.polyfit(t[ok], sm[ok], 1)[0])
    # clause ends = voiced segments followed by >=150 ms of unvoiced/pause (or the final segment)
    segs, start = [], None
    for i, v in enumerate(ok):
        if v and start is None: start = i
        if (not v) and start is not None:
            segs.append([start, i]); start = None
    if start is not None: segs.append([start, len(ok)])
    merged = []
    for s in segs:
        if merged and (s[0] - merged[-1][1]) * hop / 16000 < 0.15: merged[-1][1] = s[1]
        else: merged.append(s)
    slopes = []
    for a, b in merged:
        if (b - a) * hop / 16000 < 0.8: continue
        tail = sm[max(a, b - 30):b]; pre = sm[max(a, b - 90):max(a, b - 30)]
        tail, pre = tail[~np.isnan(tail)], pre[~np.isnan(pre)]
        if len(tail) < 8 or len(pre) < 8: continue
        slopes.append(float(np.mean(tail) - np.mean(pre)))
    end_slope = float(slopes[-1]) if slopes else 0.0
    return dict(end_slope=end_slope, rise_frac=float(np.mean([s > 1.0 for s in slopes])) if slopes else 0.0,
                clause_slope=float(np.mean(slopes)) if slopes else 0.0, decl=decl, f0_std=float(np.nanstd(sm[ok])), f0_med=float(100 * 2 ** (np.nanmedian(sm[ok]) / 12)))
