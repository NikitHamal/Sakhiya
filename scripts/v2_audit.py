"""v2 audit (torch path, no ONNX): Tharu alias, Om / visarga, Rigveda svara-scale sweep, temp/steps sweep.  Writes wavs + ASR text + F0 contrast.
usage: python3 scripts/v2_audit.py --am runs/v2_am_B/last.pt --voc runs/v2_voc/G_best.pt --out /work/tts/audit1 [--sweep]"""
import os, sys, json, argparse, re
import numpy as np, torch, soundfile as sf, librosa, jiwer
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sakhiya.cfm import SakhiyaAM, V2_LANG2ID
from sakhiya.v2_vocoder import VocoderV2
from sakhiya import symbols as S, g2p, sanskrit as SA
import v2_full_eval as E
ap = argparse.ArgumentParser()
ap.add_argument("--am", default="/work/tts/runs/v2_am_B/last.pt"); ap.add_argument("--voc", default="/work/tts/runs/v2_voc/G_best.pt")
ap.add_argument("--out", default="/work/tts/audit1"); ap.add_argument("--sweep", action="store_true"); ap.add_argument("--rv_only", action="store_true"); ap.add_argument("--om_variants", action="store_true"); ap.add_argument("--n", type=int, default=12)
a = ap.parse_args(); dev = "cuda"; os.makedirs(a.out, exist_ok=True)
ck = torch.load(a.am, map_location=dev, weights_only=False); am = SakhiyaAM().to(dev).eval(); am.load_state_dict(ck["ema"])
G = VocoderV2().to(dev).eval(); sd = torch.load(a.voc, map_location=dev, weights_only=False); G.load_state_dict(sd.get("G_ema", sd)); 
from transformers import AutoProcessor, Wav2Vec2ForCTC
proc = AutoProcessor.from_pretrained("facebook/mms-1b-all", target_lang="npi"); asr = Wav2Vec2ForCTC.from_pretrained("facebook/mms-1b-all", target_lang="npi", ignore_mismatched_sizes=True).to(dev).eval()
utmos = E.load_utmos(dev)
def asr_txt(w): 
    with torch.no_grad(): x = proc(E.to16(w, 24000), sampling_rate=16000, return_tensors="pt").input_values.to(dev); return proc.batch_decode(asr(x).logits.argmax(-1))[0]
def mos(w):
    with torch.no_grad(): return float(utmos(torch.from_numpy(E.to16(w, 24000))[None].to(dev), 16000))

@torch.no_grad()
def say(text, lang="ne", svara=0.4, steps=8, temp=0.667, rate=None, seed=0, vs=0.0):
    sa = lang == "sa"
    if sa: ph, off, dm = SA.laya_scales(text, v2=True)
    else: ph, off = g2p.text_to_phones_acc(text, lang); dm = [1.0] * len(ph)
    keep = [i for i, p in enumerate(ph) if p in S.SYM2ID]
    ids = torch.tensor([[S.SYM2ID[ph[i]] for i in keep]], device=dev)
    pa = torch.tensor([[off[i] * svara for i in keep]], dtype=torch.float32, device=dev); dmt = torch.tensor([[dm[i] for i in keep]], device=dev)
    rate = rate or (0.90 if lang.startswith("thr") else (0.96 if sa else 1.0))
    torch.manual_seed(seed); z = torch.zeros(1, dtype=torch.long, device=dev)
    mel = am.synth(ids, z + (2 if sa else 0), z + V2_LANG2ID[lang], z + 4, z + 1, rate=rate, steps=steps, temp=temp, pitch_add=pa, dm=dmt)
    G.src.shift_st = None
    if vs:
        fo = np.convolve(np.repeat(np.array([off[i] for i in keep]), am._last_dur.astype(int)), np.ones(9) / 9, mode="same")[: mel.shape[-1]]
        G.src.shift_st = torch.tensor(fo * vs, dtype=torch.float32, device=dev)[None, None]
    w = G(mel)[0].float().cpu().numpy(); pk = np.abs(w).max(); w = w * (0.9 / pk) if pk > 1e-4 else w
    return w, [ph[i] for i in keep], [off[i] for i in keep]

def contrast(w, ph, off):
    f0 = librosa.pyin(w, fmin=70, fmax=400, sr=24000, hop_length=256, frame_length=2048)[0]; st = 12 * np.log2(f0 / np.nanmedian(f0))
    fr = np.repeat(np.array(off), am._last_dur.astype(int))[:len(st)]; s_ = st[:len(fr)]
    hi, lo = s_[fr >= 1.0], s_[fr <= -1.0]; hi, lo = hi[~np.isnan(hi)], lo[~np.isnan(lo)]
    return dict(std=float(np.nanstd(st)), hi=float(hi.mean()) if hi.size else None, lo=float(lo.mean()) if lo.size else None, contrast=float(hi.mean() - lo.mean()) if hi.size and lo.size else None)

rv = "अ॒ग्निमी॑ळे पु॒रोहि॑तं य॒ज्ञस्य॑ दे॒वमृ॒त्विज॑म् । होता॑रं रत्न॒धात॑मम् ॥"
om = "ॐ पूर्णमदः पूर्णमिदं पूर्णात्पूर्णमुदच्यते।पूर्णस्य पूर्णमादाय पूर्णमेवावशिष्यते॥"
R = {}
if a.om_variants:
    V = {"om_1.8_1.4": (("o", "m"), (1.8, 1.4)), "om_1.3_1.2": (("o", "m"), (1.3, 1.2)), "om_1.0_1.0": (("o", "m"), (1.0, 1.0)), "oom_1_1_1": (("o", "o", "m"), (1.0, 1.0, 1.0)),
         "oom_.7_.7_1.2": (("o", "o", "m"), (0.7, 0.7, 1.2)), "oom_.6_.6_1.0": (("o", "o", "m"), (0.6, 0.6, 1.0))}
    for k, (ph_, dm_) in V.items():
        SA.OM_PH2, SA.OM_DM2, SA.OM_OFF2 = ph_, dm_, (0.0,) * len(ph_); row = []
        for sd_ in range(3):
            w, _, _ = say(om, "sa", 0.0, seed=sd_); row.append(asr_txt(w).split(" ")[0])
        sf.write(f"{a.out}/om_{k}.wav", w, 24000); R[k] = row; print(k, row, flush=True)
    json.dump(R, open(f"{a.out}/om_variants.json", "w"), indent=1, ensure_ascii=False); sys.exit()
if not a.rv_only:
    w, _, _ = say("ॐ", "sa", 0.0); sf.write(f"{a.out}/om_only.wav", w, 24000); R["om_only"] = dict(asr=asr_txt(w), dur=len(w) / 24000)
    w, _, _ = say(om, "sa", 0.0); sf.write(f"{a.out}/om_full.wav", w, 24000); R["om_full"] = dict(asr=asr_txt(w), utmos=mos(w))
for sc in (0.0, 0.25, 0.4, 0.55, 0.8, 1.12, 1.6):
    w, ph, off = say(rv, "sa", sc); sf.write(f"{a.out}/rv_s{sc}.wav", w, 24000); h = asr_txt(w)
    R[f"rv_s{sc}"] = dict(asr=h, h_count=h.count("ह"), utmos=mos(w), f0=contrast(w, ph, off))
for vs in (1.0, 1.5, 2.0):
    w, ph, off = say(rv, "sa", 0.0, vs=vs); sf.write(f"{a.out}/rv_vs{vs}.wav", w, 24000); h = asr_txt(w)
    R[f"rv_vshift{vs}"] = dict(asr=h, h_count=h.count("ह"), utmos=mos(w), f0=contrast(w, ph, off))
if a.rv_only: json.dump(R, open(f"{a.out}/audit.json", "w"), indent=1, ensure_ascii=False); [print(k, v, flush=True) for k, v in R.items()]; sys.exit()
for tr, t in (("thr_chitwania", "जात जातक भासा अलग बा"), ("thr_rana", "हमार नाउँ का हो"), ("thr_dangaura", "का हाल बा")):
    w, _, _ = say(t, tr); sf.write(f"{a.out}/{tr}.wav", w, 24000); R[tr] = dict(asr=asr_txt(w), ref=t, utmos=mos(w))
if a.sweep:   # temp / steps sweep on held-out Rasa/native val clips (Nepali + Sanskrit)
    D = torch.load("/work/tts/feats/v2_studio_24k.pt", weights_only=False); vals = {"ne": [], "sa": []}
    for x in D["items"]:
        if x["split"] == "val" and x["spk"] in (0, 2) and len(vals[x["lang"]]) < a.n: vals[x["lang"]].append(x)
    for temp in (0.667, 1.0):
        for steps in (4, 6, 8, 10):
            row = {}
            for lg, xs in vals.items():
                pc, ms = [], []
                for x in xs:
                    w, _, _ = say(x["text"], lg, 0.0, steps, temp); hyp = asr_txt(w); f = E.ph_ne if lg == "ne" else E.ph_sa
                    pc.append(jiwer.cer(f(E.norm(x["text"]) and x["text"]), f(hyp) or "x")); ms.append(mos(w))
                row[lg] = dict(pcer=float(np.mean(pc)), utmos=float(np.mean(ms)))
            R[f"sweep_t{temp}_s{steps}"] = row; print("sweep", temp, steps, row, flush=True)
json.dump(R, open(f"{a.out}/audit.json", "w"), indent=1, ensure_ascii=False)
for k, v in R.items(): print(k, v, flush=True)
