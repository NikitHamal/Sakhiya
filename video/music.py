"""Synth sound design: pad + pluck arp + soft drums + whooshes/hits, ducked under the voice. Pure numpy."""
import numpy as np, wave
from scipy.signal import sosfilt, butter, fftconvolve

BPM = 100; SPB = 60 / BPM
CH = [[57, 60, 64], [53, 57, 60], [48, 52, 55], [55, 59, 62]]  # Am F C G
INTENS = dict(hello=.35, about=.4, series=.65, n1=.6, n2=.6, n3=.6, n4=.6, n5=.6, fast=.9, app=.7, outro=.4)

def f(m): return 440 * 2 ** ((m - 69) / 12)

def lp(x, fc, sr, o=2): return sosfilt(butter(o, fc, "low", fs=sr, output="sos"), x)
def hp(x, fc, sr, o=2): return sosfilt(butter(o, fc, "high", fs=sr, output="sos"), x)

def whoosh(dur, sr, up=True, seed=0):
    n = int(dur * sr); rng = np.random.default_rng(seed); nz = rng.standard_normal(n); t = np.linspace(0, 1, n)
    out = np.zeros(n)
    for k, fc in enumerate([300, 600, 1200, 2400, 4800, 8000]):
        b = sosfilt(butter(2, [fc * .7, fc * 1.3], "band", fs=sr, output="sos"), nz)
        c = (k + .5) / 6; c = c if up else 1 - c
        out += b * np.exp(-((t - c) ** 2) / 0.03)
    return out * np.sin(np.pi * t) ** 1.5 * 0.25

def hit(sr, dur=1.4):
    n = int(dur * sr); t = np.arange(n) / sr
    k = np.sin(2 * np.pi * (48 + 90 * np.exp(-t * 14)) * t) * np.exp(-t * 3.2)
    air = lp(np.random.default_rng(3).standard_normal(n), 3000, sr) * np.exp(-t * 7) * .15
    return (k + air) * .8

def mix(voice, tl, B, SR):
    total = tl["total"]; n = len(voice); t = np.arange(n) / SR
    music = np.zeros(n); sec = SPB * 4
    # per-scene intensity curve (smoothed)
    it = np.zeros(n)
    for sc in tl["scenes"]: it[int(sc["t0"] * SR):int(sc["t1"] * SR)] = INTENS.get(sc["id"], .5)
    it = lp(it, 0.8, SR, 1) if it.any() else it
    for bar in range(int(total / sec) + 2):
        notes = CH[bar % 4]; s0 = int(bar * sec * SR); L = int(sec * 1.08 * SR)
        if s0 >= n: break
        tt = np.arange(L) / SR; pad = np.zeros(L)
        for m in notes + [notes[0] - 12]:
            for d in (-0.15, 0.15):
                pad += np.sin(2 * np.pi * f(m) * (1 + d / 100) * tt) + .3 * np.sin(2 * np.pi * 2 * f(m) * tt)
        e = np.minimum(1, tt / 0.8) * np.minimum(1, (L / SR - tt) / 0.5)
        seg = lp(pad * e * .05, 1600, SR); music[s0:s0 + L] += seg[:max(0, min(L, n - s0))] if s0 + L <= n else seg[:n - s0]
        for i in range(8):  # pluck arp, 8th notes
            ps = s0 + int(i * SPB / 2 * SR)
            if ps >= n: break
            note = notes[[0, 1, 2, 1, 2, 1, 0, 2][i]] + 12 * (i % 3 == 2)
            d = int(0.5 * SR); tt2 = np.arange(d) / SR
            pl = (np.sin(2 * np.pi * f(note) * tt2) + .4 * np.sin(4 * np.pi * f(note) * tt2)) * np.exp(-tt2 * 7)
            sl = slice(ps, min(n, ps + d)); music[sl] += (pl * .05 * it[ps])[:sl.stop - sl.start]
        for b in range(4):  # drums
            ps = s0 + int(b * SPB * SR)
            if ps >= n: break
            d = int(0.22 * SR); tt3 = np.arange(d) / SR
            kick = np.sin(2 * np.pi * (45 + 80 * np.exp(-tt3 * 30)) * tt3) * np.exp(-tt3 * 12) * .5
            sl = slice(ps, min(n, ps + d)); music[sl] += (kick * max(0, it[ps] - .45) * 1.2)[:sl.stop - sl.start]
            hs = ps + int(SPB / 2 * SR); dh = int(0.05 * SR)
            if hs + dh < n:
                hh = hp(np.random.default_rng(b + bar).standard_normal(dh), 7000, SR) * np.exp(-np.arange(dh) / SR * 70)
                music[hs:hs + dh] += hh * .035 * it[hs]
    ir = np.random.default_rng(9).standard_normal(int(1.4 * SR)) * np.exp(-np.arange(int(1.4 * SR)) / SR * 3.2)
    music = music + .35 * fftconvolve(music, ir * .02)[:n]
    # voice envelope duck
    ev = lp(np.abs(voice), 6, SR, 1); ev = np.clip(ev / (np.percentile(ev, 98) + 1e-6), 0, 1)
    music *= (1 - .62 * ev)
    sfx = np.zeros(n)
    for i, sc in enumerate(tl["scenes"]):
        t0 = sc["t0"]; w = whoosh(1.1, SR, up=(i % 2 == 0), seed=i); s = max(0, int((t0 - .75) * SR))
        sfx[s:s + len(w)] += w[:max(0, min(len(w), n - s))]
        if sc["id"] in ("n1", "n2", "n3", "n4", "n5", "app", "series"):
            h = hit(SR); s = int((t0 + .55) * SR); sfx[s:s + len(h)] += h[:max(0, min(len(h), n - s))] * .5
    h = hit(SR, 2.0); sfx[int(1.2 * SR):int(1.2 * SR) + len(h)] += h * .9
    st = np.stack([voice + music + sfx * .8, voice + music + sfx * .8], 1)
    fade = np.minimum(1, (total - t) / 2.0).clip(0, 1)[:, None]; st = st * fade * np.minimum(1, t / .05)[:, None]
    pk = np.abs(st).max(); st = np.tanh(st * 1.1) / np.tanh(1.1) if pk > .95 else st
    st = st / max(1e-6, np.abs(st).max()) * .89
    st = st[: int(total * SR) + 1]
    w_ = wave.open(B + "/mix.wav", "wb"); w_.setnchannels(2); w_.setsampwidth(2); w_.setframerate(SR)
    w_.writeframes((st * 32767).astype(np.int16).tobytes()); w_.close()
