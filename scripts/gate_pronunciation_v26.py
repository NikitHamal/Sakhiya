#!/usr/bin/env python3
"""v2.6 HARD pronunciation gate. Synthesizes the 6 owner sentences on BOTH voices (sakhiya + sandesh) from an exported ONNX dir,
transcribes with an ASR, and FAILS (exit 1) if any known defect form is heard: उठेरर, एर, मामकाहा, पाण्डवास सैव, काख्वा, कोहम, टेक्सटु.
Also runs a deterministic phoneme check (no duplicate adjacent phones, all duration multipliers 1.0), WavLM-SV cos_sim(ne, sa) per voice,
and the News-vs-Storytelling F0-std separation. Writes eval/gate_v26.json (+ /work/tts/ev/gate_v26.json) which the MoLab gate card renders.
usage: gate_pronunciation_v26.py --model docs/models_v26 [--asr spktsagar/wav2vec2-large-xls-r-300m-nepali-openslr] [--promote]"""
import argparse, json, os, sys, time, unicodedata
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
BAD = ["उठेरर", "एर", "मामकाहा", "पाण्डवास सैव", "काख्वा", "कोहम", "टेक्सटु"]
norm = lambda s: unicodedata.normalize("NFC", s)


def heard_bad(asr, extra):
    toks, flat = norm(asr).split(), norm(asr).replace(" ", "")
    out = []
    for b in BAD + list(extra):
        b = norm(b)
        if (len(b) <= 3 and b in toks) or (len(b) > 3 and b.replace(" ", "") in flat): out.append(b)
    return sorted(set(out))


def phoneme_check(text, lang):
    from sakhiya import g2p, sanskrit as sa, mixed
    segs = mixed.route(text) if lang == "auto" else [(lang, text)]
    bad = []
    for L, seg in segs:
        ph, dm = (lambda r: (r[0], r[2]))(sa.laya_scales(seg, v2=True)) if L == "sa" else (g2p.text_to_phones(seg, L), g2p.phone_dm(g2p.text_to_phones(seg, L)))
        if any(a == b and a not in (".", ",", "|", ";") for a, b in zip(ph, ph[1:])): bad.append("dup-phone")
        if L != "sa" and set(dm) != {1.0}: bad.append("duration-hack")
    return sorted(set(bad))


def f0_std(w, sr):
    import librosa
    f0, vf, _ = librosa.pyin(w.astype(np.float32), fmin=70, fmax=400, sr=sr, frame_length=1024)
    f = f0[~np.isnan(f0)]
    return float(np.std(1200 * np.log2(f / np.median(f)))) if len(f) > 20 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(ROOT, "docs/models_v26"))
    ap.add_argument("--asr", default="spktsagar/wav2vec2-large-xls-r-300m-nepali-openslr")
    ap.add_argument("--style", default="neutral")
    ap.add_argument("--promote", action="store_true", help="copy --model over docs/models only if EVERY clip passes")
    ap.add_argument("--no-sv", action="store_true")
    a = ap.parse_args()
    from sakhiya.onnx_tts import SakhiyaONNX
    import torch
    from transformers import AutoModelForCTC, AutoProcessor
    spec = json.load(open(os.path.join(ROOT, "eval/owner_sentences_v26.json")))
    tts = SakhiyaONNX(a.model, threads=4)
    proc, am = AutoProcessor.from_pretrained(a.asr), AutoModelForCTC.from_pretrained(a.asr).eval()

    def asr(w, sr):
        import librosa
        x = librosa.resample(w.astype(np.float32), orig_sr=sr, target_sr=16000) if sr != 16000 else w
        with torch.no_grad(): lg = am(**proc(x, sampling_rate=16000, return_tensors="pt")).logits
        return proc.batch_decode(lg.argmax(-1))[0]

    res, clips = {}, {}
    for v in spec["voices"]:
        res[v] = {"sentences": {}}
        for s in spec["sentences"]:
            w, sr = tts.say(s["text"], v, s["lang"], a.style)
            clips[(v, s["id"])] = (w, sr)
            h = asr(w, sr); bad = heard_bad(h, s["must_not"]) + phoneme_check(s["text"], s["lang"])
            miss = [m for m in s["must_have"] if norm(m).replace(" ", "") not in norm(h).replace(" ", "")]
            res[v]["sentences"][s["id"]] = {"pass": not bad, "bad": bad, "asr": h, "weak_missing": miss}
            print(v, s["id"], "PASS" if not bad else "FAIL " + str(bad), "|", h[:80], flush=True)
        if not a.no_sv:
            try:
                from transformers import AutoFeatureExtractor, WavLMForXVector
                import librosa
                fe, sv = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv"), WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv").eval()
                emb = lambda k: sv(**fe(librosa.resample(clips[k][0], orig_sr=clips[k][1], target_sr=16000), sampling_rate=16000, return_tensors="pt")).embeddings
                ne, sa_ = emb((v, "story")), emb((v, "gita"))
                res[v]["cos_sim"] = float(torch.nn.functional.cosine_similarity(ne, sa_).item())
            except Exception as e:
                res[v]["cos_sim"] = None; res[v]["sv_error"] = type(e).__name__
    st = {}
    try:
        txt = next(s["text"] for s in spec["sentences"] if s["id"] == "story")
        for k, key in (("news", "news_f0_std"), ("storytelling", "storytelling_f0_std")):
            w, sr = tts.say(txt, "sakhiya", "ne", k); st[key] = f0_std(w, sr)
    except Exception as e:
        st["error"] = type(e).__name__
    allp = all(r["pass"] for v in res.values() for r in v["sentences"].values())
    sv_ok = all((res[v].get("cos_sim") or 0) >= 0.88 for v in res) if not a.no_sv else True
    out = {"updated": time.time(), "model": a.model, "voices": res, "style": st, "all_pass": bool(allp and sv_ok), "pronunciation_pass": bool(allp)}
    for p in (os.path.join(ROOT, "eval/gate_v26.json"), "/work/tts/ev/gate_v26.json"):
        try:
            os.makedirs(os.path.dirname(p), exist_ok=True); json.dump(out, open(p, "w"), ensure_ascii=False, indent=1)
        except Exception: pass
    print("GATE", "PASS" if out["all_pass"] else "FAIL", flush=True)
    if a.promote and out["all_pass"]:
        import shutil
        dst = os.path.join(ROOT, "docs/models"); shutil.rmtree(dst, ignore_errors=True); shutil.copytree(a.model, dst); print("promoted ->", dst)
    sys.exit(0 if out["all_pass"] else 1)


if __name__ == "__main__":
    main()
