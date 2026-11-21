#!/usr/bin/env python3
"""Vedic purity gate (owner fix): keep ONLY real Sanskrit chanting, drop Tamil/Telugu/Malayalam/English talk.

Root cause found on the box: 9,277 'vedic_chant' segments came from Internet-Archive *lectures* (Nochur Venkataraman discourses,
a Gita *class*, a Rig-Veda *introduction*) = spoken commentary in Malayalam/Tamil/English.  This gate scores every candidate clip with
 1. Whisper-large-v3-turbo language-ID posterior (Sanskrit/Hindi/Nepali/Marathi vs Dravidian vs English),
 2. forced-Sanskrit transcript must be >= 90 % Devanagari,
 3. acoustic chant-ness (voiced ratio + sustained-note steadiness from WORLD DIO F0),
 4. a trust tier: studio `vag_*` (prathoshap/vagdhenu-data, reference text) is always trusted.
Outputs /work/tts/vedic/verdicts.jsonl, source_report.json, ok_paths.json (consumed by yt_asr_ensemble.py).
usage: python scripts/vedic_purity_gate.py [--limit 4000] [--per-src 60]"""
import argparse, glob, json, os, random, re, sys, time, collections
import numpy as np, soundfile as sf, torch, torchaudio.functional as AF
ROOT = os.environ.get("TTS_ROOT", "/work/tts"); os.environ.setdefault("HF_HOME", f"{ROOT}/hf"); V = f"{ROOT}/vedic"; os.makedirs(V, exist_ok=True)
SA_OK = ("sa", "hi", "ne", "mr"); DRA = ("ta", "te", "ml", "kn"); DEV = re.compile(r"[\u0900-\u097F]")
TRUSTED = ("vag_",)
def log(*a): print(time.strftime("%H:%M:%S"), *a, flush=True)

def chant_feats(x, sr=24000):
    import pyworld as pw
    f0, t = pw.dio(x.astype("float64"), sr, f0_floor=70, f0_ceil=500, frame_period=10.0, speed=3); v = f0 > 0; vr = float(v.mean())
    st = 12 * np.log2(np.where(v, f0, 1) / 100.0); both = v[1:] & v[:-1]
    steady = float((np.abs(np.diff(st))[both] < 0.6).mean()) if both.any() else 0.0
    rng = float(np.percentile(st[v], 90) - np.percentile(st[v], 10)) if v.sum() > 10 else 0.0
    return vr, steady, rng

def verdict(r):
    why = []
    if r["trusted"]: return "trusted_reference", []
    dra, en, ok = r["p_dravidian"], r["p_english"], r["p_sanskritic"]
    if dra > 0.20: why.append(f"sounds Dravidian (Tamil/Telugu/Malayalam/Kannada) p={dra:.2f}")
    if en > 0.20: why.append(f"English speech p={en:.2f}")
    if ok < 0.45: why.append(f"not Sanskrit/Hindi/Nepali-like p={ok:.2f}")
    if r["dev_ratio"] < 0.90 or r["n_dev"] < 8: why.append("no clean Devanagari Sanskrit text")
    if r["voiced_ratio"] < 0.35: why.append("too little voiced sound (not chanting)")
    return ("pure_sanskrit" if not why else "rejected"), why

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--limit", type=int, default=4000); ap.add_argument("--per-src", type=int, default=60); a = ap.parse_args()
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    wp = WhisperProcessor.from_pretrained("openai/whisper-large-v3-turbo")
    wm = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3-turbo", torch_dtype=torch.float16).to("cuda").eval()
    l2i = {k.strip("<|>"): v for k, v in wm.generation_config.lang_to_id.items()}; codes = list(l2i); ids = torch.tensor([l2i[c] for c in codes]).cuda()
    sot = wp.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
    files = sorted(glob.glob(f"{ROOT}/segments/vedic_chant/*.wav")); by = collections.defaultdict(list)
    for f in files: by[os.path.basename(f).rsplit("_", 1)[0]].append(f)
    random.seed(5); pick = []
    for s, fl in by.items(): random.shuffle(fl); pick += fl[: 150 if s.startswith(TRUSTED) else a.per_src]   # per-source sample => every source gets a verdict
    pick = pick[: a.limit]; out = f"{V}/verdicts.jsonl"; done = {json.loads(l)["path"] for l in open(out)} if os.path.exists(out) else set()
    pick = [p for p in pick if p not in done]; log(len(files), "vedic segments;", len(pick), "to judge from", len(by), "sources")
    B = 32
    for i in range(0, len(pick), B):
        bt = pick[i:i + B]; xs = [sf.read(p, dtype="float32")[0] for p in bt]; xs = [x.mean(1) if x.ndim > 1 else x for x in xs]
        ws = [AF.resample(torch.from_numpy(x), 24000, 16000).numpy() for x in xs]
        with torch.no_grad():
            f = wp(ws, sampling_rate=16000, return_tensors="pt").input_features.to("cuda").half()
            lg = wm(input_features=f, decoder_input_ids=torch.full((len(bt), 1), sot, device="cuda")).logits[:, -1].float()
            pr = torch.softmax(lg[:, ids], -1).cpu().numpy()
            g = wm.generate(f, language="sa", task="transcribe", num_beams=1, max_new_tokens=60)
            txt = wp.batch_decode(g, skip_special_tokens=True)
        with open(out, "a") as fo:
            for p, x, pp, t in zip(bt, xs, pr, txt):
                P = dict(zip(codes, pp.tolist())); vr, steady, rng = chant_feats(x); dv = [c for c in t if DEV.match(c)]; nz = [c for c in t if not c.isspace() and c not in ".,।॥!?-"]
                r = {"path": p, "src": os.path.basename(p).rsplit("_", 1)[0], "trusted": os.path.basename(p).startswith(TRUSTED), "dur": round(len(x) / 24000, 2),
                     "p_sanskritic": round(sum(P[c] for c in SA_OK), 3), "p_dravidian": round(sum(P[c] for c in DRA), 3), "p_english": round(P["en"], 3),
                     "top_lang": max(P, key=P.get), "dev_ratio": round(len(dv) / max(1, len(nz)), 3), "n_dev": len(dv), "text_sa": t.strip(),
                     "voiced_ratio": round(vr, 3), "steadiness": round(steady, 3), "f0_range_st": round(rng, 2)}
                r["verdict"], r["why"] = verdict(r); r["chant_like"] = bool(steady >= 0.55 and vr >= 0.45); fo.write(json.dumps(r, ensure_ascii=False) + "\n")
        if (i // B) % 10 == 0: log(f"{i + len(bt)}/{len(pick)}")
    rows = [json.loads(l) for l in open(out)]; rep = {}
    for s in sorted({r["src"] for r in rows}):
        rs = [r for r in rows if r["src"] == s]; n = sum(r["verdict"] in ("pure_sanskrit", "trusted_reference") for r in rs)
        rep[s] = {"judged": len(rs), "pure": n, "pure_pct": round(100 * n / len(rs), 1), "top_langs": collections.Counter(r["top_lang"] for r in rs).most_common(3), "trusted": rs[0]["trusted"]}
    json.dump(rep, open(f"{V}/source_report.json", "w"), ensure_ascii=False, indent=1)
    good_src = {s for s, d in rep.items() if d["trusted"] or d["pure_pct"] >= 60}   # a whole source is kept only if most sampled clips are pure
    ok = [r["path"] for r in rows if r["verdict"] in ("pure_sanskrit", "trusted_reference") and r["src"] in good_src]
    json.dump({"ok_paths": ok, "good_sources": sorted(good_src), "denied_sources": sorted(set(rep) - good_src)}, open(f"{V}/ok_paths.json", "w"), ensure_ascii=False)
    log("sources", {s: (d["pure_pct"], d["top_langs"][:1]) for s, d in rep.items()}); log("kept", len(ok), "of", len(rows))

if __name__ == "__main__": main()
