"""Build clean studio Sanskrit + Native Nepali training set (/work/tts/feats/edge_sa.pt).
Re-phonemizes sa.pt with single-token sakhiya.sanskrit G2P (fixing doubled-vowel tokens) and distills
studio-clarity Female (sa_f0) and Male (sa_m0) Sanskrit utterances on GPU from Apache-2.0 Alpa v0.2 (77ethers/alpa)
while keeping held-out val clips strictly untouched."""
import os, sys, random, collections
import numpy as np, librosa, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, "/work/tts/alpa")
from sakhiya import audio, sanskrit as SA, g2p, symbols as S

F = "/work/tts/feats"
dev = "cuda" if torch.cuda.is_available() else "cpu"


def build():
    sa = torch.load(f"{F}/sa.pt", weights_only=False)
    val_clips = []
    train_texts = []
    for r in sa:
        ph = SA.text_to_phones(r["text"], add_end=True, verse=False)
        r["ph"] = S.encode(ph)
        r["phs"] = " ".join(ph)
        r["lang"] = "sa"
        if r.get("split") == "val":
            val_clips.append(r)
        else:
            train_texts.append(r["text"])

    # Classic Sanskrit shlokas & prose sentences (Vedic, Gita, Upanishads, Subhashitas)
    shlokas = [
        "धर्मक्षेत्रे कुरुक्षेत्रे समवेता युयुत्सवः । मामकाः पाण्डवाश्चैव किमकुर्वत सञ्जय ॥",
        "कर्मण्येवाधिकारस्ते मा फलेषु कदाचन । मा कर्मफलहेतुर्भूर्मा ते सङ्गोऽस्त्वकर्मणि ॥",
        "यदा यदा हि धर्मस्य ग्लानिर्भवति भारत । अभ्युत्थानमधर्मस्य तदात्मानं सृजाम्यहम् ॥",
        "परित्राणाय साधूनां विनाशाय च दुष्कृताम् । धर्मसंस्थापनार्थाय सम्भवामि युगे युगे ॥",
        "असतो मा सद्गमय तमसो मा ज्योतिर्गमय मृत्योर्मा अमृतं गमय ।",
        "सत्यमेव जयते नानृतं सत्येन पन्था विततो देवयानः ।",
        "विद्या ददाति विनयं विनयाद्याति पात्रताम् । पात्रत्वाद्धनमाप्नोति धनाद्धर्मं ततः सुखम् ॥",
        "अयं निजः परो वेति गणना लघुचेतसाम् । उदारचरितानां तु वसुधैव कुटुम्बकम् ॥",
        "सर्वे भवन्तु सुखिनः सर्वे सन्तु निरामयाः । सर्वे भद्राणि पश्यन्तु मा कश्चिद्दुःखभाग्भवेत् ॥",
        "वागर्थाविव सम्पृक्तौ वागर्थप्रतिपत्तये । जगतः पितरौ वन्दे पार्वतीपरमेश्वरौ ॥",
    ] * 8
    pool_texts = train_texts[:900] + shlokas
    random.seed(42)
    random.shuffle(pool_texts)

    from concurrent.futures import ThreadPoolExecutor
    from alpa.infer import load_model, speak
    alpa_m = load_model("/work/tts/alpa/alpa-v0.2.pt", device=dev)
    raw_items = []
    t_set = {r["text"] for r in val_clips}
    for idx, txt in enumerate(pool_texts[:600]):
        if txt in t_set:
            continue
        vname = "female" if idx % 2 == 0 else "male"
        spk = "sa_f0" if vname == "female" else "sa_m0"
        try:
            res = speak(alpa_m, txt, voice=vname)
            w24 = res[0].detach().cpu().numpy().astype(np.float32).squeeze()
            if w24.size < 24000 * 0.8 or w24.size > 24000 * 16:
                continue
            raw_items.append((idx, spk, txt, w24))
        except Exception as e:
            if idx < 5:
                print("skip", idx, e, flush=True)
        if (idx + 1) % 200 == 0:
            print("synthesized", idx + 1, "raw clips:", len(raw_items), flush=True)

    def feat_one(it):
        idx, spk, txt, w24 = it
        x = librosa.resample(w24, orig_sr=24000, target_sr=audio.SR)
        pk = float(np.max(np.abs(x)))
        if pk > 1e-4:
            x = (x * (0.90 / pk)).astype(np.float32)
        mel = audio.mel_spec(torch.from_numpy(x))[0].numpy().astype(np.float16)
        f0 = audio.f0_track(x, mel.shape[1]).astype(np.float16)
        ph = SA.text_to_phones(txt, add_end=True, verse=None)
        return dict(
            id=f"alpa_{idx:05d}", spk=spk, text=txt, src="rasa_alpa", lic="apache-2.0",
            lang="sa", split="train", wide=True, ph=S.encode(ph), phs=" ".join(ph),
            mel=mel, f0=f0, wav=(x * 32767).astype(np.int16)
        )

    with ThreadPoolExecutor(16) as ex:
        studio_sa = list(ex.map(feat_one, raw_items))
    print("distilled studio sa clips:", len(studio_sa), flush=True)

    # Also keep the cleanest high-SNR real Shrutilipi clips so real acoustic diversity stays present
    clean_real_sa = [r for r in sa if r.get("split") == "train" and r.get("snr", 0) >= 24.0 and r.get("cer", 1.0) <= 0.22][:300]
    sa_out = studio_sa + clean_real_sa + val_clips
    torch.save(sa_out, f"{F}/sa_clean.pt")
    print("saved sa_clean.pt:", len(sa_out), "studio:", len(studio_sa), "real_clean:", len(clean_real_sa), "val:", len(val_clips), flush=True)

    # Re-phonemize edge.pt and top real human speakers from v2.pt with updated native Nepali g2p
    ed = torch.load(f"{F}/edge.pt", weights_only=False)
    for r in ed:
        ph = g2p.text_to_phones(r["text"])
        r["ph"] = S.encode(ph); r["phs"] = " ".join(ph); r["lang"] = "ne"
    v2 = torch.load(f"{F}/v2.pt", weights_only=False)
    real_keep = [r for r in v2 if r["spk"] in ("s3614", "cv_b0487694", "cv_13b830fe") or r["split"] == "val"]
    for r in real_keep:
        ph = g2p.text_to_phones(r["text"])
        r["ph"] = S.encode(ph); r["phs"] = " ".join(ph); r["lang"] = "ne"

    joint = ed + real_keep + studio_sa + clean_real_sa
    hrs = collections.defaultdict(float)
    for r in joint:
        if r["split"] == "train":
            hrs[r["spk"]] += len(r["wav"]) / audio.SR / 3600
    print("edge_sa.pt hours per speaker:", {k: round(v, 2) for k, v in hrs.items()}, "total clips:", len(joint), flush=True)
    torch.save(joint, f"{F}/edge_sa.pt")
    print("EDGE_SA_DONE", flush=True)


if __name__ == "__main__":
    build()
