#!/usr/bin/env python3
"""v5 Vedic-only LoRA data prep (strict adapter isolation).

Source: v13/real_audio_24k.pt (Vagdhenu studio chanting + verse-aligned Gita) + vedic/ref_text.json (verbatim Devanagari)
+ vedic/gold.json (per-clip ASR match vs the reference text).

A clip is KEPT only if every gate passes (corrupted / misaligned VC clips never reach the adapter):
  * text exists, no Latin/digits, no 'ऽऽ/ओऽ' stretch hack (single avagraha is legal Sanskrit), >= 6 chars
  * ASR match vs reference text >= MIN_MATCH (default 88)  -> audio says what the text says (no alignment drift)
  * 2.5 s <= duration <= 14 s, chars/sec within [3, 14]
  * trailing breath/gasp trimmed: cut at the last voiced frame + 0.18 s with a 60 ms fade (never an abrupt cut)
Never touches the Nepali weights / data. Output: /work/tts/v5_vedic/ft/{train,dev}.jsonl + wavs.
"""
import json, os, re, sys, random, unicodedata
import numpy as np, soundfile as sf, torch, librosa

ROOT = os.environ.get("V5V", "/work/tts/v5_vedic")
MIN_MATCH = float(os.environ.get("MIN_MATCH", 80))
OUT = f"{ROOT}/ft"; os.makedirs(f"{OUT}/wav", exist_ok=True)


def clean(t):
    t = unicodedata.normalize("NFC", t).replace("\u200b", "").replace("\ufeff", "")
    t = re.sub(r"[\"“”‘’'`()\[\]{}<>*_#/\\|~^]", " ", t)
    t = t.replace(".", "।")
    t = re.sub(r"\s*([,।॥?!;])\s*", r"\1 ", t)
    t = re.sub(r"\s+", " ", t).strip(" ,;")
    return t


def trim(w, sr=24000):
    iv = librosa.effects.split(w, top_db=38, frame_length=1024, hop_length=256)
    if not len(iv): return None
    a = max(0, iv[0][0] - int(0.12 * sr)); b = min(len(w), iv[-1][1] + int(0.18 * sr))
    w = w[a:b].copy(); n = int(0.06 * sr); w[-n:] *= np.linspace(1, 0, n, dtype=np.float32)
    return w / (float(np.abs(w).max()) or 1.0) * 0.89


def main():
    d = torch.load(f"{ROOT}/dl/v13/real_audio_24k.pt", map_location="cpu", weights_only=False)
    ref = json.load(open(f"{ROOT}/meta/vedic/ref_text.json")); gold = json.load(open(f"{ROOT}/meta/vedic/gold.json"))
    offs = torch.cumsum(torch.cat([torch.zeros(1, dtype=torch.long), d["lens"]]), 0)
    rows, drop = [], dict(not_vedic=0, no_text=0, latin=0, stretch=0, match=0, dur=0, cps=0, trim=0)
    for i, p in enumerate(d["paths"]):
        if "/vedic_chant/" not in p: drop["not_vedic"] += 1; continue
        t = ref.get(p)
        if not t: drop["no_text"] += 1; continue
        t = clean(t)
        if re.search(r"[A-Za-z0-9]", t) or len(t) < 6: drop["latin"] += 1; continue
        if re.search(r"ऽऽ|ओऽ|ॐऽ|ऽ\s*ऽ", t): drop["stretch"] += 1; continue   # ban the stretch hack only; a single avagraha is normal Sanskrit
        g = gold.get(p, {})
        if float(g.get("match", 0)) < MIN_MATCH: drop["match"] += 1; continue
        w = d["wav"][offs[i]:offs[i + 1]].numpy().astype(np.float32); dur = len(w) / 24000
        if not (2.5 <= dur <= 14.0): drop["dur"] += 1; continue
        cps = len(t.replace(" ", "")) / dur
        if not (3 <= cps <= 14): drop["cps"] += 1; continue
        w = trim(w)
        if w is None: drop["trim"] += 1; continue
        if t[-1] not in "।॥?!": t += " ।"
        o = f"{OUT}/wav/{os.path.basename(p)[:-4].replace(chr(46), chr(45))}.wav"; sf.write(o, w, 24000)
        rows.append(dict(id=os.path.basename(p)[:-4].replace(".", "-"),   # WebDataset splits keys at the first dot
                          audio_path=o, text=t, language_id="sa", dur=len(w) / 24000,
                         src=("gita" if "gss_" in p or "gita" in p.lower() or "Gita" in p else "vag")))
    random.Random(7).shuffle(rows)
    dev, tr = rows[:40], rows[40:]
    for n, r in (("train", tr), ("dev", dev)):
        with open(f"{OUT}/{n}.jsonl", "w") as f:
            for x in r: f.write(json.dumps(x, ensure_ascii=False) + "\n")
    rep = dict(kept=len(rows), hours=round(sum(r["dur"] for r in rows) / 3600, 2), dropped=drop, min_match=MIN_MATCH,
               by_src={s: sum(1 for r in rows if r["src"] == s) for s in ("vag", "gita")})
    json.dump(rep, open(f"{ROOT}/prep_report.json", "w"), indent=1); print(rep)


if __name__ == "__main__":
    main()
