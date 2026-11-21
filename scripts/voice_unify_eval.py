"""Cross-lingual voice-unification gate: cos_sim(speaker(ne clips), speaker(sa clips)) with WavLM-base-plus-sv x-vectors.
Usage: python3 scripts/voice_unify_eval.py --export DIR [--tiny] [--out eval/voice_unify.json] [--wavdir DIR]
Compares lock_voice=False (Sanskrit uses its own lang row) vs lock_voice=True (Sanskrit reuses the Nepali row)."""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sakhiya.onnx_tts import SakhiyaONNX, stitch
NE = ["म सखिया हुँ, निकित हमालज्यूले बनाउनुभएको सानो टेक्स्ट टु स्पीच मोडेल।", "हेलो साथीहरू, आज मौसम धेरै राम्रो छ।",
      "म संस्कृत पनि बोल्न सक्छु।", "नेपाल हिमालको देश हो, यहाँ धेरै भाषा बोलिन्छ।"]
SA = ["धर्मक्षेत्रे कुरुक्षेत्रे समवेता युयुत्सवः । मामकाः पाण्डवाश्चैव किमकुर्वत सञ्जय ॥",
      "ॐ पूर्णमदः पूर्णमिदं पूर्णात्पूर्णमुदच्यते। पूर्णस्य पूर्णमादाय पूर्णमेवावशिष्यते॥",
      "असतो मा सद्गमय। तमसो मा ज्योतिर्गमय। मृत्योर्मा अमृतं गमय॥", "सत्यमेव जयते नानृतं सत्येन पन्था विततो देवयानः॥"]
TH = ["हेलो संघरियाहुक्रे। मै सखिया हों, निकित हमाल जी बनाइल छोट टेक्स्ट टु स्पीच मोडेल।"]


def xvec(models, w, sr):
    import torch, librosa
    fe, m = models
    w16 = librosa.resample(np.asarray(w, np.float32), orig_sr=sr, target_sr=16000)
    with torch.no_grad():
        e = m(**fe(w16, sampling_rate=16000, return_tensors="pt")).embeddings
    return torch.nn.functional.normalize(e, dim=-1)[0].numpy()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--export", required=True); ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--out", default="eval/voice_unify.json"); ap.add_argument("--wavdir", default="")
    a = ap.parse_args()
    from transformers import AutoFeatureExtractor, WavLMForXVector
    tok = open("/tmp/hf_token").read().strip() if os.path.exists("/tmp/hf_token") else None
    fe = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv", token=tok)
    sv = (fe, WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv", token=tok).eval())
    tts = SakhiyaONNX(a.export, tiny=a.tiny, threads=4)
    res = {}
    import soundfile as sf
    for lock in (False, True):
        E = {"ne": [], "sa": [], "thr_dangaura": []}
        for lang, texts in (("ne", NE), ("sa", SA), ("thr_dangaura", TH)):
            for i, t in enumerate(texts):
                w, sr = tts.say(t, lang=lang, lock_voice=lock)
                E[lang].append(xvec(sv, w, sr))
                if a.wavdir:
                    os.makedirs(a.wavdir, exist_ok=True); sf.write(f"{a.wavdir}/{lang}{i}_lock{int(lock)}.wav", w, sr)
        cen = {k: np.mean(v, 0) / np.linalg.norm(np.mean(v, 0)) for k, v in E.items()}
        cs = lambda x, y: float(np.dot(x, y))
        res[f"lock_voice={lock}"] = {"cos_ne_sa": cs(cen["ne"], cen["sa"]), "cos_ne_thr": cs(cen["ne"], cen["thr_dangaura"]),
            "cos_sa_thr": cs(cen["sa"], cen["thr_dangaura"]), "mean_pairwise_ne_sa": float(np.mean([cs(x, y) for x in E["ne"] for y in E["sa"]])),
            "within_ne": float(np.mean([cs(x, y) for i, x in enumerate(E["ne"]) for j, y in enumerate(E["ne"]) if i < j]))}
        print(lock, json.dumps(res[f"lock_voice={lock}"]), flush=True)
    json.dump(res, open(a.out, "w"), indent=1, allow_nan=False)


if __name__ == "__main__":
    main()
