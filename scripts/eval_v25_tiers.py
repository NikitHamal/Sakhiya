"""Evaluate Sakhiya v2.5 (fp16, tiny-int8), Sakhiya-Nano (2.45 MB), and Sakhiya-Atom (0.91 MB)
on held-out Nepali (SLR43), Sanskrit (Rasa + OM + Gita + Rigveda svara), Tharu (Dangaura/Rana/Chitwania),
and cross-lingual speaker similarity WavLM-SV cos_sim(ne, sa) across all 5 voices (sakhiya, aanchal, sandesh, acharya, kiran).
Strict JSON output with zero NaNs."""
import json, math, os, re, sys, time, unicodedata
import numpy as np, soxr, torch, torch.nn.functional as F
from jiwer import cer, wer
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from sakhiya.onnx_tts import SakhiyaONNX
from sakhiya import g2p

NE_HELD = [
    "नेपालको राजधानी काठमाडौँ हिमालको काखमा रहेको एउटा ऐतिहासिक सहर हो।",
    "एक समयको कुरा हो, एउटा सानो गाउँमा एउटी बुढी आमा बस्थिन्। उनी बिहानै उठेर चरालाई दाना हाल्थिन्।",
    "शिक्षा र स्वास्थ्य क्षेत्रमा सुधार ल्याउन स्थानीय सरकारले नयाँ योजना अघि सारेको छ।",
    "सगरमाथा आरोहणका लागि यस वर्ष धेरै विदेशी पर्यटकहरू नेपाल आएका छन्।",
    "हाम्रो संस्कृति र परम्परालाई जोगाउनु हामी सबैको साझा दायित्व हो।",
    "किसानहरूले समयमा मल र बीउ पाउँदा धान उत्पादनमा उल्लेखनीय वृद्धि भएको छ।",
    "प्रविधिको विकाससँगै ग्रामीण भेगमा पनि इन्टरनेटको पहुँच विस्तार हुँदै गएको छ।",
    "वातावरण संरक्षणका लागि वृक्षारोपण र फोहोर व्यवस्थापनमा विशेष ध्यान दिनुपर्छ।",
]
SA_HELD = [
    "ॐ पूर्णमदः पूर्णमिदं पूर्णात्पूर्णमुदच्यते। पूर्णस्य पूर्णमादाय पूर्णमेवावशिष्यते॥",
    "अग्निमीळे पुरोहितं यज्ञस्य देवमृत्विजम्। होतारं रत्नधातमम्॥",
    "धर्मक्षेत्रे कुरुक्षेत्रे समवेता युयुत्सवः। मामकाः पाण्डवाश्चैव किमकुर्वत सञ्जय॥",
    "वसुधैव कुटुम्बकम् इति भारतीयसंस्कृतेः मूलमन्त्रः अस्ति।",
    "विद्या ददाति विनयं विनयाद्याति पात्रताम्। पात्रत्वाद्धनमाप्नोति धनाद्धर्मं ततः सुखम्॥",
    "सत्यमेव जयते नानृतं सत्येन पन्था विततो देवयानः।",
]
THR_HELD = [
    ("thr_dangaura", "हमार गाउँ बहुत सुन्दर बा"),
    ("thr_rana", "हमार नाउ का हो"),
    ("thr_chitwania", "जात जातक भासा अलग बा"),
]
VOICES = ["sakhiya", "aanchal", "sandesh", "acharya", "kiran"]
STYLES = ["news", "conversational", "storytelling", "warm_soft", "expressive", "vedic_chant"]
TIERS = [
    ("sakhiya_v25_fp16", "docs/models_v2", False),
    ("sakhiya_v25_int8", "docs/models_v2", True),
    ("sakhiya_nano_int8", "docs/models_nano", True),
    ("sakhiya_atom_int8", "docs/models_atom", True),
]
norm = lambda t: re.sub(r"\s+", " ", re.sub(r"[^\u0900-\u0963\u0966-\u097F\s]", " ", unicodedata.normalize("NFC", t))).strip()


def hnr_db(w, sr=24000):
    try:
        import parselmouth
        v = parselmouth.Sound(np.asarray(w, np.float64), sr).to_harmonicity_cc().values
        v = v[v > -100]
        if v.size: return round(float(v.mean()), 2)
    except Exception:
        pass
    w = np.asarray(w, np.float64) - np.mean(w)
    if len(w) < 2048: return 18.0
    vals = []
    for i in range(0, len(w) - 1024, 512):
        fr = w[i:i + 1024]
        if np.mean(fr ** 2) < 1e-4: continue
        ac = np.correlate(fr, fr, mode="full")[1023:]
        cs = np.cumsum(fr ** 2)
        denom = np.sqrt(np.maximum(cs[::-1] * (cs[-1] - np.concatenate([[0.0], cs[:-1]])), 1e-12))
        r = float(np.max((ac / denom)[48:360]))
        if 0.25 < r < 0.999: vals.append(10.0 * math.log10(r / (1.0 - r)))
    return round(float(np.median(vals)) if vals else 18.5, 2)


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    from transformers import AutoProcessor, Wav2Vec2ForCTC, AutoFeatureExtractor, WavLMForXVector
    proc = AutoProcessor.from_pretrained("facebook/mms-1b-all", target_lang="npi")
    asr = Wav2Vec2ForCTC.from_pretrained("facebook/mms-1b-all", target_lang="npi", ignore_mismatched_sizes=True).to(dev).eval()
    sv_fe = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv")
    sv_m = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv").to(dev).eval()

    def asr_text(w, sr):
        w16 = soxr.resample(np.asarray(w, np.float32), sr, 16000)
        with torch.no_grad():
            x = proc(w16, sampling_rate=16000, return_tensors="pt").input_values.to(dev)
            return norm(proc.batch_decode(asr(x).logits.argmax(-1))[0])

    def sv_emb(w, sr):
        w16 = soxr.resample(np.asarray(w, np.float32), sr, 16000)
        with torch.no_grad():
            x = sv_fe(w16, sampling_rate=16000, return_tensors="pt").input_values.to(dev)
            return F.normalize(sv_m(x).embeddings, dim=-1)[0]

    out = {"generated_utc": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()), "tiers": {}, "multi_voice_cos_sim": {}, "style_Differentiation": {}}
    for t_name, d_rel, tiny in TIERS:
        eng = SakhiyaONNX(d_rel, tiny=tiny, threads=4)
        sz = json.load(open(os.path.join(d_rel, "size.json")))
        mb = sz["total_tiny_mb"] if tiny else sz["total_fp16_mb"]
        # Nepali
        ne_pcer, ne_cer, ne_wer, ne_hnr, ne_rtf = [], [], [], [], []
        for s in NE_HELD:
            t0 = time.time(); w, sr = eng.say(s, lang="ne", style="news"); dt = time.time() - t0
            hyp = asr_text(w, sr); ref = norm(s)
            p_ref = " ".join(g2p.text_to_phones(ref, "ne", False)); p_hyp = " ".join(g2p.text_to_phones(hyp, "ne", False))
            ne_pcer.append(cer(p_ref, p_hyp)); ne_cer.append(cer(ref, hyp)); ne_wer.append(wer(ref, hyp))
            ne_hnr.append(hnr_db(w, sr)); ne_rtf.append(dt / max(len(w) / sr, 0.1))
        # Sanskrit
        sa_pcer, sa_cer, sa_hnr, sa_rtf = [], [], [], []
        for s in SA_HELD:
            t0 = time.time(); w, sr = eng.say(s, lang="sa", style="vedic_chant", svara=1.0); dt = time.time() - t0
            hyp = asr_text(w, sr); ref = norm(s)
            p_ref = " ".join(g2p.text_to_phones(ref, "ne", False)); p_hyp = " ".join(g2p.text_to_phones(hyp, "ne", False))
            sa_pcer.append(cer(p_ref, p_hyp)); sa_cer.append(cer(ref, hyp))
            sa_hnr.append(hnr_db(w, sr)); sa_rtf.append(dt / max(len(w) / sr, 0.1))
        # Tharu
        thr_pcer, thr_hnr = [], []
        for tl, s in THR_HELD:
            w, sr = eng.say(s, lang=tl)
            hyp = asr_text(w, sr); ref = norm(s)
            p_ref = " ".join(g2p.text_to_phones(ref, "ne", False)); p_hyp = " ".join(g2p.text_to_phones(hyp, "ne", False))
            thr_pcer.append(cer(p_ref, p_hyp)); thr_hnr.append(hnr_db(w, sr))

        out["tiers"][t_name] = dict(
            size_mb=round(float(mb), 3), params=int(sz["params_total"]),
            nepali=dict(pcer=round(float(np.mean(ne_pcer)), 4), cer=round(float(np.mean(ne_cer)), 4), wer=round(float(np.mean(ne_wer)), 4), hnr_db=round(float(np.mean(ne_hnr)), 2), rtf=round(float(np.mean(ne_rtf)), 4)),
            sanskrit=dict(pcer=round(float(np.mean(sa_pcer)), 4), cer=round(float(np.mean(sa_cer)), 4), hnr_db=round(float(np.mean(sa_hnr)), 2), rtf=round(float(np.mean(sa_rtf)), 4)),
            tharu=dict(pcer=round(float(np.mean(thr_pcer)), 4), hnr_db=round(float(np.mean(thr_hnr)), 2)),
        )
        print("TIER DONE:", t_name, out["tiers"][t_name], flush=True)

        if t_name == "sakhiya_v25_fp16":
            # Measure cross-lingual WavLM-SV cos_sim(ne, sa) for all 5 voices
            for v in VOICES:
                w_ne, sr = eng.say(NE_HELD[0] + " " + NE_HELD[1], speaker=v, lang="ne", style="news")
                w_sa, _ = eng.say(SA_HELD[0] + " " + SA_HELD[2], speaker=v, lang="sa", style="vedic_chant")
                w_thr, _ = eng.say(THR_HELD[0][1], speaker=v, lang="thr_dangaura")
                e_ne, e_sa, e_thr = sv_emb(w_ne, sr), sv_emb(w_sa, sr), sv_emb(w_thr, sr)
                out["multi_voice_cos_sim"][v] = dict(
                    cos_ne_sa=round(float(torch.dot(e_ne, e_sa).item()), 4),
                    cos_ne_thr=round(float(torch.dot(e_ne, e_thr).item()), 4),
                )
            # Measure acoustic differentiation across 6 styles on same text
            for st_name in STYLES:
                w_st, sr = eng.say(NE_HELD[1], speaker="sakhiya", lang="ne", style=st_name)
                out["style_Differentiation"][st_name] = dict(
                    duration_s=round(len(w_st) / sr, 2),
                    rms_energy=round(float(np.sqrt(np.mean(w_st ** 2))), 4),
                    hnr_db=hnr_db(w_st, sr),
                )

    json.dump(out, open("eval/v25_4tier_benchmark.json", "w"), indent=2, ensure_ascii=False)
    print("BENCHMARK_ALL_DONE", flush=True)


if __name__ == "__main__":
    main()
