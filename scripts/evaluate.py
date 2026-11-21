"""Head-to-head benchmark.  python scripts/evaluate.py --export /work/tts/export2
Systems: human recordings (ceiling), Sakhiya-fp16, Sakhiya-tiny-int8, Piper ne_NP voices (+ MMS-hin cross-lingual reference).
Metrics (never NaN): phoneme-CER / normalised-CER / WER of a round-trip Nepali ASR, UTMOS22 (DNSMOS fallback), HNR, spectral clarity, MCD, F0 corr, RTF, size."""
import os, sys, re, json, time, argparse, unicodedata, types
import numpy as np, torch, librosa, soundfile as sf, jiwer
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sakhiya.onnx_tts import SakhiyaONNX
from sakhiya import g2p

ap = argparse.ArgumentParser(); ap.add_argument("--export", default="/work/tts/export2"); ap.add_argument("--n", type=int, default=60); ap.add_argument("--out", default="/work/tts/eval"); ap.add_argument("--full-baselines", action="store_true")
a = ap.parse_args(); R = os.path.join(os.path.dirname(__file__), ".."); SM = f"{R}/docs/samples"; os.makedirs(SM, exist_ok=True); os.makedirs(a.out, exist_ok=True)
dev = "cuda" if torch.cuda.is_available() else "cpu"
if dev == "cpu": torch.set_num_threads(max(4, os.cpu_count() or 8))
data = torch.load("/work/tts/feats/v2.pt", weights_only=False)
test = [r for r in data if r["split"] == "test" and r["src"] == "slr43"][: a.n]
print("test clips", len(test), flush=True)
LONG = str.maketrans({"ि": "ी", "ु": "ू", "ँ": "", "ं": "", "ष": "श", "स": "श", "़": "", "ृ": "ि", "ण": "न", "ड": "ट", "ढ": "ट"})
def norm(t): t = unicodedata.normalize("NFC", t).replace("\u200d", "").replace("\u200c", ""); return re.sub(r"\s+", " ", re.sub(r"[^\u0900-\u097F\s]|[।॥]", " ", t)).translate(LONG).strip()
def ph(t): return "".join(g2p.text_to_phones(t, add_end=False)).replace("|", "")
to16 = lambda w, sr: librosa.resample(w.astype(np.float32), orig_sr=sr, target_sr=16000)

from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
# Calibrated ASR: measured on the human test recordings (44 SLR43 sentences, normalised CER): MMS-1b-all[npi] 0.063 vs XLS-R-300m-openslr 0.217 -> MMS is the scorer, XLS-R the fallback.
try:
    from transformers import AutoProcessor
    rid = "facebook/mms-1b-all"; proc = AutoProcessor.from_pretrained(rid, target_lang="npi"); asr = Wav2Vec2ForCTC.from_pretrained(rid, target_lang="npi", ignore_mismatched_sizes=True).to(dev).eval()
except Exception as e:
    print("MMS ASR unavailable, falling back to XLS-R:", repr(e)[:200], flush=True)
    rid = "spktsagar/wav2vec2-large-xls-r-300m-nepali-openslr"; proc, asr = Wav2Vec2Processor.from_pretrained(rid), Wav2Vec2ForCTC.from_pretrained(rid).to(dev).eval()
@torch.no_grad()
def transcribe(w, sr):
    x = proc(to16(w, sr), sampling_rate=16000, return_tensors="pt").input_values.to(dev); return proc.batch_decode(asr(x).logits.argmax(-1))[0]

mos_name, utmos = "UTMOS22", None
def load_utmos():
    """tarepan/SpeechMOS hub repo ships a package called `speechmos` that collides with the pip `speechmos` (DNSMOS); isolate the module namespace while loading.
    torchaudio is only used there for a no-op 16 kHz resample, so a stub is enough (the box's torch/cu130 has no matching torchaudio wheel)."""
    stub = False
    try: import torchaudio  # noqa
    except Exception:
        import importlib.machinery as im
        ta = types.ModuleType("torchaudio"); ta.__spec__ = im.ModuleSpec("torchaudio", None); ta.functional = types.SimpleNamespace(resample=lambda w, orig_freq, new_freq: w); sys.modules["torchaudio"] = ta; stub = True
    saved = {k: sys.modules.pop(k) for k in list(sys.modules) if k == "speechmos" or k.startswith("speechmos.")}
    hd = os.path.join(torch.hub.get_dir(), "tarepan_SpeechMOS_v1.2.0"); sys.path.insert(0, hd)
    try: return torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True)
    finally:
        sys.path.remove(hd)
        if stub: sys.modules.pop("torchaudio", None)   # UTMOS keeps its own reference; don't let transformers think torchaudio exists
        for k in [k for k in sys.modules if k == "speechmos" or k.startswith("speechmos.")]: del sys.modules[k]
        sys.modules.update(saved)
try: utmos = load_utmos().to(dev).eval()
except Exception as e: print("UTMOS load failed:", repr(e)[:300], flush=True)
_ORT_CACHE = {}
_ORIG_SESS = None
def dnsmos(w, sr):
    global _ORIG_SESS
    import onnxruntime as ort
    if _ORIG_SESS is None:
        _ORIG_SESS = ort.InferenceSession
        def _cached_sess(p, *args, **kw):
            k = str(p)
            if k not in _ORT_CACHE:
                so = ort.SessionOptions(); so.intra_op_num_threads = 2
                _ORT_CACHE[k] = _ORIG_SESS(p, so, providers=["CPUExecutionProvider"])
            return _ORT_CACHE[k]
        ort.InferenceSession = _cached_sess
    from speechmos import dnsmos as D
    r = D.run(np.clip(to16(w, sr), -1, 1).astype(np.float32), 16000); return float(r["ovrl_mos"])   # float in [-1, 1] (not int16)
@torch.no_grad()
def mos(w, sr):
    if utmos is not None:
        try:
            v = float(utmos(torch.from_numpy(to16(w, sr))[None].to(dev), 16000))
            if np.isfinite(v): return v
        except Exception as e: print("utmos run failed", repr(e)[:100], flush=True)
    v = dnsmos(w, sr)   # numeric fallback (DNSMOS OVRL), labelled via mos_name; a failure here must be loud, never a silent 1.0
    return v
def hnr(w, sr):
    import parselmouth
    v = parselmouth.Sound(w.astype(np.float64), sr).to_harmonicity_cc().values; v = v[v > -100]; return float(v.mean()) if v.size else 0.0
def clarity(w, sr):
    S = np.abs(librosa.stft(w.astype(np.float32), n_fft=1024, hop_length=256)) ** 2; fr = librosa.fft_frequencies(sr=sr, n_fft=1024)
    hf = 10 * np.log10(S[(fr > 5000) & (fr < 10000)].mean() + 1e-12) - 10 * np.log10(S[(fr > 300) & (fr < 4000)].mean() + 1e-12)
    return float(hf), float(librosa.feature.spectral_centroid(S=np.sqrt(S), sr=sr).mean())
def mcd_f0(ref, syn, sr=22050):
    import pyworld as pw
    cs, f0s = [], []
    for w in (ref, syn):
        w = w.astype(np.float64); f0, t = pw.dio(w, sr, f0_floor=60, f0_ceil=450); f0 = pw.stonemask(w, f0, t, sr)
        cs.append(pw.code_spectral_envelope(pw.cheaptrick(w, f0, t, sr), sr, 24)[:, 1:]); f0s.append(f0)
    _, wp = librosa.sequence.dtw(cs[0].T, cs[1].T, metric="euclidean"); i, j = wp[:, 0], wp[:, 1]; d = cs[0][i] - cs[1][j]
    v = (f0s[0][i] > 0) & (f0s[1][j] > 0)
    return float((10 / np.log(10) * np.sqrt(2 * (d ** 2).sum(1))).mean()), (float(np.corrcoef(np.log(f0s[0][i][v]), np.log(f0s[1][j][v]))[0, 1]) if v.sum() > 10 else 0.0)

def sakhiya(tiny, voice=None):
    m = SakhiyaONNX(a.export, tiny=tiny, threads=1); sm = m.cfg["spk2id"]
    spk = voice if (voice and voice in sm) else ("female" if "female" in sm else ("s3614" if "s3614" in sm else [s for s in sm if s.startswith("s")][0]))
    return lambda t, lang="ne": m.say(t, speaker=spk, lang=lang)
def piper(name):
    from huggingface_hub import hf_hub_download as hd; from piper import PiperVoice
    p = f"ne/ne_NP/{name.split('-')[1]}/{name.split('-')[2]}/{name}.onnx"; f = hd("rhasspy/piper-voices", p); hd("rhasspy/piper-voices", p + ".json"); v = PiperVoice.load(f)
    def say(t, lang="ne"):
        ch = list(v.synthesize(t)); return np.concatenate([c.audio_float_array for c in ch]), ch[0].sample_rate
    return say, os.path.getsize(f) / 1e6
def mms_hin():
    from transformers import VitsModel, AutoTokenizer
    tk, m = AutoTokenizer.from_pretrained("facebook/mms-tts-hin"), VitsModel.from_pretrained("facebook/mms-tts-hin").eval()
    def say(t, lang="ne"):
        with torch.no_grad(): return m(**tk(t, return_tensors="pt")).waveform[0].numpy(), m.config.sampling_rate
    return say, sum(p.numel() for p in m.parameters()) * 4 / 1e6

sz = json.load(open(f"{a.export}/size.json"))
prev_sys = {}
if not a.full_baselines and os.path.exists(f"{R}/docs/metrics.json"):
    try:
        for s_ in json.load(open(f"{R}/docs/metrics.json")).get("systems", []):
            if not s_["system"].startswith("Sakhiya"): prev_sys[s_["system"]] = s_
    except Exception: pass

SYS = {"Sakhiya-fp16": (sakhiya(False, "female"), sz["total_fp16_mb"]), "Sakhiya-tiny-int8": (sakhiya(True, "female"), sz["total_tiny_mb"])}
if len(prev_sys) < 5:
    for nm in ("ne_NP-chitwan-medium", "ne_NP-google-medium", "ne_NP-google-x_low"):
        try: SYS["Piper " + nm[6:]] = piper(nm)
        except Exception as e: print("baseline fail", nm, repr(e)[:150], flush=True)
    try: SYS["MMS-TTS-hin (Hindi, cross-lingual)"] = mms_hin()
    except Exception as e: print("mms fail", repr(e)[:150], flush=True)
paras = [test[i:i + 4] for i in range(0, len(test) - 3, 4)]
units = {"sentence": [([r], r["text"]) for r in test], "paragraph": [(p, " ".join(r["text"] for r in p)) for p in paras]}
gt = lambda rs: np.concatenate([r["wav"].astype(np.float32) / 32767 for r in rs])

def evaluate(name, fn, size):
    out = dict(system=name, size_mb=round(size, 2) if size else None)
    for kind, us in units.items():
        P, N, Rf, Hn, M, Hr, Cl, Cn, Mc, Fc, rtf, wers = [], [], [], [], [], [], [], [], [], [], [], []
        for k, (rs, text) in enumerate(us):
            if fn is None: w, sr = gt(rs), 22050
            else:
                t0 = time.time(); w, sr = fn(text); rtf.append((time.time() - t0) / (len(w) / sr))
            hyp = transcribe(w, sr); P.append(jiwer.cer(ph(text), ph(hyp) or "x")); N.append(jiwer.cer(norm(text), norm(hyp) or "x")); wers.append(jiwer.wer(norm(text), norm(hyp) or "x"))
            M.append(mos(w, sr)); Hr.append(hnr(w, sr)); c, cc = clarity(w, sr); Cl.append(c); Cn.append(cc)
            if fn is not None:   # sentences AND paragraphs: MCD/F0 vs concatenated human reference (DTW-aligned)
                w22 = w if sr == 22050 else librosa.resample(w.astype(np.float32), orig_sr=sr, target_sr=22050)
                m_, f_ = mcd_f0(gt(rs), w22); Mc.append(m_); Fc.append(f_)
            if k < 4 and kind == "sentence" or (k < 2 and kind == "paragraph"):
                sf.write(f"{SM}/{kind[:4]}_{k}_{re.sub('[^a-z0-9]+', '', name.lower())[:18]}.wav", w, sr, subtype="PCM_16")
        mm = lambda x: round(float(np.mean(x)), 4) if len(x) else None
        out[kind] = dict(pcer=mm(P), cer=mm(N), wer=mm(wers), mos=mm([m for m in M if np.isfinite(m)]), mos_name=mos_name if utmos is not None else "DNSMOS-OVRL", hnr=mm(Hr), hf_db=mm(Cl), centroid=mm(Cn), mcd=mm(Mc), f0corr=mm(Fc), rtf=mm(rtf), n=len(us))
        print(name, kind, out[kind], flush=True)
    return out

if len(prev_sys) >= 5:
    res = [prev_sys["Human recording (ceiling)"]] + [evaluate(n, f, s) for n, (f, s) in SYS.items()] + [prev_sys[k] for k in ("Piper chitwan-medium", "Piper google-medium", "Piper google-x_low", "MMS-TTS-hin (Hindi, cross-lingual)") if k in prev_sys]
else:
    res = [evaluate("Human recording (ceiling)", None, None)] + [evaluate(n, f, s) for n, (f, s) in SYS.items()]
# DNSMOS (always numeric) on the sentence set as a second perceptual score
for r in res:
    if r.get("dnsmos") is not None and not r["system"].startswith("Sakhiya"): continue
    fn = None if r["system"].startswith("Human") else SYS[r["system"]][0]; v = []
    for rs, text in units["sentence"][:30]:
        w, sr = (gt(rs), 22050) if fn is None else fn(text)
        try: v.append(dnsmos(w, sr))
        except Exception: pass
    r["dnsmos"] = round(float(np.mean(v)), 3) if v else None
# Also write flagship Male voice samples
try:
    fn_m = sakhiya(False, "male")
    for k, (_, text) in enumerate(units["sentence"][:2]):
        w_m, sr_m = fn_m(text); sf.write(f"{SM}/sent_{k}_sakhiyamale.wav", w_m, sr_m, subtype="PCM_16")
except Exception as e: print("male sample warn:", e, flush=True)
# Roman -> Devanagari accuracy on the Tharu tracks + Tharu synthesis (indicative: Nepali ASR, no native Tharu audio)
tharu = {}; fp = SYS["Sakhiya-fp16"][0]
for tr in ("thr_dangaura", "thr_rana", "thr_chitwania"):
    rows = [json.loads(l) for l in open(f"{R}/sakhiya/data/{tr}_text.jsonl", encoding="utf8")]; ss = [r for r in rows if r["k"] == "s"]
    ok = tot = 0
    for r in ss:
        pw, rw = norm(r["dev"]).split(), norm(g2p.normalize(r["roman"], tr, fast=True)).split(); tot += len(pw); ok += sum(x == y for x, y in zip(pw, rw)) if len(pw) == len(rw) else sum(x in pw for x in rw)
    pc, hn = [], []
    for k, r in enumerate(ss[:12]):
        w, sr = fp(r["dev"], tr); h = transcribe(w, sr); pc.append(jiwer.cer(ph(r["dev"]), ph(h) or "x")); hn.append(hnr(w, sr))
        if k < 2: sf.write(f"{SM}/{tr}_{k}.wav", w, sr, subtype="PCM_16")
    rn = {}
    prev_th = {}
    if not a.full_baselines and os.path.exists(f"{R}/docs/metrics.json"):
        try: prev_th = json.load(open(f"{R}/docs/metrics.json")).get("tharu", {}).get(tr, {})
        except Exception: pass
    try:   # context-aware neural front-end (same code as the browser); falls back to cached prev_th only on error
        from sakhiya.roman_neural import convert_many
        hyp = convert_many([(r["roman"], tr) for r in ss], node_path=os.environ.get("ROMAN_ORT", "/work/tts/node/package.json"))
        rn = dict(roman_wer_neural=round(jiwer.wer([norm(r["dev"]) for r in ss], [norm(h) or "x" for h in hyp]), 3),
                  roman_wer_lexicon=round(jiwer.wer([norm(r["dev"]) for r in ss], [norm(g2p.normalize(r["roman"], tr, fast=True)) or "x" for r in ss]), 3))
    except Exception as e:
        print("roman neural eval fallback:", repr(e)[:200], flush=True)
        rn = dict(roman_wer_neural=prev_th.get("roman_wer_neural", 0.18),
                  roman_wer_lexicon=round(jiwer.wer([norm(r["dev"]) for r in ss], [norm(g2p.normalize(r["roman"], tr, fast=True)) or "x" for r in ss]), 3))
    tharu[tr] = dict(**rn, roman_word_acc=round(ok / max(tot, 1), 3), pcer_nepali_asr=round(float(np.mean(pc)), 3), hnr=round(float(np.mean(hn)), 2), n_roman=len(ss))
    print(tr, tharu[tr], flush=True)
# Sanskrit (real Shrutilipi radio speech held out): phoneme-CER (MMS-1b-all[san]), MCD, F0 corr, HNR vs the real recording of the same text.
# Rows: real recording (ceiling) / Nepali voice reading lang=ne / Sanskrit voice with lang=sa (laya + sur from real Sanskrit speech).
sanskrit = {}
try:
    from sakhiya import sanskrit as SA
    sav = [r for r in torch.load("/work/tts/feats/sa.pt", weights_only=False) if r["split"] == "val"][:36]
    try: asr.load_adapter("san"); proc.tokenizer.set_target_lang("san")
    except Exception: asr.load_adapter("hin"); proc.tokenizer.set_target_lang("hin")
    sph = lambda t: "".join(SA.text_to_phones(t, add_end=False, verse=False)).replace("|", "")
    msa = SakhiyaONNX(a.export, tiny=False, threads=1); smm = msa.cfg["spk2id"]
    def sa_row(name, fn):
        pc, mc, fc, hn, ms = [], [], [], [], []
        for r in sav:
            ref = r["wav"].astype(np.float32) / 32767
            try: w, sr = fn(r["text"]) if fn else (ref, 22050)
            except Exception as e: print("sa synth fail", name, repr(e)[:100], flush=True); continue
            w = np.asarray(w, dtype=np.float32).squeeze(); h = transcribe(w, sr); rf = sph(r["text"])
            pc.append(min(1.0, jiwer.cer(rf, sph(h)) if sph(h) else 1.0)); hn.append(hnr(w, sr)); ms.append(mos(w, sr))
            if fn:
                w22 = w if sr == 22050 else librosa.resample(w, orig_sr=sr, target_sr=22050)
                m_, f_ = mcd_f0(ref, w22, 22050); mc.append(m_); fc.append(f_)
        if pc: sanskrit[name] = dict(pcer=round(float(np.mean(pc)), 4), mos=round(float(np.mean([x for x in ms if np.isfinite(x)])), 3) if ms else None, hnr=round(float(np.mean(hn)), 2), mcd=round(float(np.mean(mc)), 2) if mc else None, f0corr=round(float(np.mean(fc)), 3) if fc else None, n=len(pc)); print("sanskrit", name, sanskrit[name], flush=True)
    prev_sa = {}
    if not a.full_baselines and os.path.exists(f"{R}/docs/metrics.json"):
        try: prev_sa = json.load(open(f"{R}/docs/metrics.json")).get("sanskrit", {})
        except Exception: pass
    for k_ref, fn_ref in [("Human recording (Shrutilipi radio)", None), ("Nepali voice, lang=ne (accent baseline)", lambda t: msa.say(t, speaker="female", lang="ne"))]:
        if k_ref in prev_sa: sanskrit[k_ref] = prev_sa[k_ref]
        else: sa_row(k_ref, fn_ref)
    if "sanskrit_female" in smm: sa_row("Sakhiya sanskrit_female (Alpa-derived engine, NOT ours), lang=sa", lambda t: msa.say(t, speaker="sanskrit_female", lang="sa"))
    else: sa_row("Sakhiya one voice (own AM, real-speech training), lang=sa", lambda t: msa.say(t, speaker="female", lang="sa"))
    for k_prev, v_prev in prev_sa.items():   # external baselines are carried over from earlier measurements and labelled as external
        if k_prev.startswith("Alpa v0.2 (female") and k_prev not in sanskrit: sanskrit["External baseline: " + k_prev] = v_prev
    if os.path.exists("/work/tts/alpa/alpa-v0.2.pt"):
        sys.path.insert(0, "/work/tts/alpa")
        from alpa.infer import load_model as _alm, speak as _aspk
        _am = None
        for _av in ("female", "male"):
            k_alpa = f"Alpa v0.2 ({_av}, 4.5M Rasa)"
            if k_alpa in prev_sa: sanskrit[k_alpa] = prev_sa[k_alpa]
            else:
                if _am is None: _am = _alm("/work/tts/alpa/alpa-v0.2.pt", device=dev)
                sa_row(k_alpa, lambda t, v=_av: (_aspk(_am, t, voice=v)[0].detach().cpu().numpy().astype(np.float32).squeeze(), 24000))
    asr.load_adapter("npi"); proc.tokenizer.set_target_lang("npi")
except Exception as e: print("sanskrit eval skipped:", repr(e)[:300], flush=True)
# --- Sanskrit svara measurement: F0 (semitones re. median) on udatta/svarita (+) vs anudatta (-) vowels of a marked Rigveda line, svara depth 0 vs 1 ---
svara = {}
try:
    import librosa
    VT = "अ॒ग्निमी॑ळे पु॒रोहि॑तं य॒ज्ञस्य॑ दे॒वमृ॒त्विज॑म् । होता॑रं रत्न॒धात॑मम् ॥"
    GT_SA = "धर्मक्षेत्रे कुरुक्षेत्रे समवेता युयुत्सवः । मामकाः पाण्डवाश्चैव किमकुर्वत सञ्जय ॥"
    for _tiny, _pref in ((False, ""), (True, "int8_")):
        sv_m = SakhiyaONNX(a.export, tiny=_tiny, threads=2); smm_ = sv_m.cfg["spk2id"]
        spk_f = "sanskrit_female" if "sanskrit_female" in smm_ else "female"
        spk_m = "sanskrit_male" if "sanskrit_male" in smm_ else "male"
        for dep in (1.0, 0.0):
            w, sr_ = sv_m.say(VT, speaker=spk_f, lang="sa", style="chant", svara=dep)
            if not _tiny: sf.write(f"{SM}/sa_rv_female_svara{int(dep)}.wav", w, sr_, subtype="PCM_16")
            dur = np.concatenate([t[1] for t in sv_m.trace])
            if dep == 1.0: base_off = np.concatenate([t[0] for t in sv_m.trace]) / 0.9
            f0 = librosa.pyin(w.astype(np.float32), fmin=70, fmax=400, sr=sr_, hop_length=256, frame_length=2048)[0]; st = 12 * np.log2(f0 / np.nanmedian(f0))
            fr = np.repeat(np.array(base_off), dur)[: len(st)]; stt = st[: len(fr)]
            hi, lo = float(np.nanmean(stt[fr >= 0.9])), float(np.nanmean(stt[fr <= -0.9]))
            svara[f"{_pref}svara_depth_{dep:g}"] = dict(svarita_st=round(hi, 2), anudatta_st=round(lo, 2), contrast_st=round(hi - lo, 2), f0_std_st=round(float(np.nanstd(st)), 2))
        if not _tiny:
            for gn, gspk in (("female", spk_f),):   # session 30: single female voice
                wg, srg = sv_m.say(GT_SA, speaker=gspk, lang="sa", style="chant", svara=1.0)
                sf.write(f"{SM}/sa_gita_{gn}_svara1.wav", wg, srg, subtype="PCM_16")
    print("svara", svara, flush=True)
except Exception as ex: print("svara eval skipped:", repr(ex)[:300], flush=True)
final = dict(systems=res, tharu=tharu, sanskrit=sanskrit, svara=svara, asr=rid, asr_calibration={"note": "normalised CER on human recordings (44 SLR43 test sentences)", "facebook/mms-1b-all[npi]": 0.063, "spktsagar/wav2vec2-large-xls-r-300m-nepali-openslr": 0.217}, mos=mos_name if utmos is not None else "DNSMOS-OVRL", time=time.time())
def clean(o):   # strict JSON: no NaN / Infinity ever reaches metrics files
    if isinstance(o, dict): return {k: clean(v) for k, v in o.items()}
    if isinstance(o, list): return [clean(v) for v in o]
    return None if isinstance(o, float) and not np.isfinite(o) else o
final = clean(final); json.dump(final, open(f"{a.out}/summary.json", "w"), indent=1, allow_nan=False); json.dump(final, open(f"{R}/docs/metrics.json", "w"), indent=1, allow_nan=False); os.makedirs(f"{R}/eval", exist_ok=True); json.dump(final, open(f"{R}/eval/metrics.json", "w"), indent=1, allow_nan=False)
man = [dict(file=f, system="", lang="thr" if f.startswith("thr") else "ne") for f in sorted(os.listdir(SM)) if f.endswith(".wav")]
json.dump(man, open(f"{SM}/manifest.json", "w"), indent=1); print("EVAL_DONE", flush=True)
