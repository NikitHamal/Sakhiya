"""Deterministically regenerate all 24 Sakhiya v3 gallery samples in docs/samples/v25/ & docs/samples/v3/ from the shipped ONNX tiers.

usage: python3 scripts/gen_all_samples.py [--tiers fp16,int8] [--out docs/samples/v25]

Covers 100% Nepali + Sanskrit + Auto (Mixed) across BOTH studio voices (`sakhiya` F + `sandesh` M) and ALL 5 real CFG styles
(`news`, `storytelling`, `conversational`, `warm_soft`, `vedic_chant`) with zero post-hoc DSP/EQ hacks."""
import os, sys, json, time, argparse
import numpy as np, soundfile as sf

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from sakhiya.onnx_tts import SakhiyaONNX

ap = argparse.ArgumentParser()
ap.add_argument("--tiers", default="fp16,int8")
ap.add_argument("--out", default=os.path.join(ROOT, "docs", "samples", "v25"))
ap.add_argument("--threads", type=int, default=4)
a = ap.parse_args()

OM = "ॐ पूर्णमदः पूर्णमिदं पूर्णात्पूर्णमुदच्यते। पूर्णस्य पूर्णमादाय पूर्णमेवावशिष्यते॥"
RV = "अ॒ग्निमी॑ळे पु॒रोहि॑तं य॒ज्ञस्य॑ दे॒वमृ॒त्विज॑म् । होता॑रं रत्न॒धात॑मम् ॥"
GITA = "धर्मक्षेत्रे कुरुक्षेत्रे समवेता युयुत्सवः । मामकाः पाण्डवाश्चैव किमकुर्वत सञ्जय ॥"
GITA_247 = "कर्मण्येवाधिकारस्ते मा फलेषु कदाचन । मा कर्मफलहेतुर्भूर्मा ते सङ्गोऽस्त्वकर्मणि ॥"
GAYATRI = "ॐ भूर्भुवः स्वः तत्सवितुर्वरेण्यं भर्गो देवस्य धीमहि धियो यो नः प्रचोदयात् ॥"
SUBHASHITA = "विद्या ददाति विनयं विनयाद्याति पात्रताम् । पात्रत्वाद्धनमाप्नोति धनाद्धर्मं ततः सुखम् ॥"
STORY = "एक समयको कुरा हो, एउटा सानो गाउँमा एउटी बुढी आमा बस्थिन्। उनी बिहानै उठेर चरालाई दाना हाल्थिन्।"
STORY_M = "हिमालको फेदीमा रहेको शान्त उपत्यकामा बिहानको कलिलो घाम पर्दा खोलाको कलकल आवाज अझै मीठो सुनिन्थ्यो।"
MIXED_F = ("हेलो साथीहरू। म सखिया हुँ, निकित हमालज्यूले बनाउनुभएको सानो टेक्स्ट टु स्पीच मोडेल। म संस्कृत पनि स्पष्ट बोल्न सक्छु। "
           + GITA + " आज मौसम निकै रमाइलो र शान्त छ।")
MIXED_M = ("नमस्ते साथीहरू। म सन्देश हुँ, नेपाली र संस्कृत दुवै भाषामा स्वाभाविक आवाज दिने मोडेल। "
           + GITA_247 + " प्रविधि र भाषाको यो संगमले नयाँ सम्भावना खोलेको छ।")

# 24 studio-clean Nepali & Sanskrit items across both voices (sakhiya F + sandesh M) & all 5 styles
ITEMS = [
    # 1. Owner's Pronunciation & Mixed-Language Proof Pack (6 items)
    ("proof1_kakhma", "Owner proof pack", "Nepal capital · काखमा (सखिया F)", "ne", "नेपालको राजधानी काठमाडौँ हिमालको काखमा रहेको एउटा ऐतिहासिक सहर हो।", dict(speaker="sakhiya", style="news")),
    ("proof2_om", "Owner proof pack", "ॐ पूर्णमदः · Isha Upanishad (सखिया F)", "sa", OM, dict(speaker="sakhiya", svara=1.0, style="vedic_chant")),
    ("proof3_rv", "Owner proof pack", "Rigveda 1.1.1 · अग्निमीळे Svara (सखिया F)", "sa", RV, dict(speaker="sakhiya", svara=1.0, style="vedic_chant")),
    ("proof4_gita", "Owner proof pack", "Gita 1.1 · समवेता युयुत्सवः पाण्डवाश्चैव (सखिया F)", "sa", GITA, dict(speaker="sakhiya", style="vedic_chant")),
    ("proof5_story", "Owner proof pack", "Storytelling · बिहानै उठेर (सखिया F)", "ne", STORY, dict(speaker="sakhiya", style="storytelling")),
    ("proof6_mixed", "Owner proof pack", "Auto (Mixed) Nepali + Sanskrit · सखिया (F)", "auto", MIXED_F, dict(speaker="sakhiya", style="news")),
    # 2. 5-Style Showcase — सखिया · Sakhiya (Studio Female) (5 items)
    ("ne_news", "5 Styles · Sakhiya (F)", "News · Kathmandu Broadcast (सखिया F)", "ne", "प्रधानमन्त्रीले आज राष्ट्रका नाममा सम्बोधन गर्दै विकास बजेटको घोषणा गरे।", dict(speaker="sakhiya", style="news")),
    ("ne_chat", "5 Styles · Sakhiya (F)", "Conversational · Everyday Chat (सखिया F)", "ne", "नमस्ते! आज मौसम धेरै राम्रो छ, चिया खान जाऔँ न?", dict(speaker="sakhiya", style="conversational")),
    ("ne_story_style", "5 Styles · Sakhiya (F)", "Storytelling · Warm Narrative (सखिया F)", "ne", STORY, dict(speaker="sakhiya", style="storytelling")),
    ("ne_warm_soft", "5 Styles · Sakhiya (F)", "Warm / Soft · Calm & Gentle (सखिया F)", "ne", "शान्त मन र मधुर मुस्कानले हरेक कठिन दिनलाई सहज बनाउँछ।", dict(speaker="sakhiya", style="warm_soft")),
    ("sa_vedic_style", "5 Styles · Sakhiya (F)", "Vedic Chant · Rigveda 1.1.1 + Svara (सखिया F)", "sa", RV, dict(speaker="sakhiya", style="vedic_chant", svara=1.0)),
    # 3. 5-Style Showcase — सन्देश · Sandesh (Studio Male) (5 items)
    ("voice_sandesh", "5 Styles · Sandesh (M)", "News · National Broadcast (सन्देश M)", "ne", "नेपालको राजधानी काठमाडौँ हिमालको काखमा रहेको एउटा ऐतिहासिक सहर हो।", dict(speaker="sandesh", style="news")),
    ("sandesh_chat", "5 Styles · Sandesh (M)", "Conversational · Lively Dialogue (सन्देश M)", "ne", "तपाईंको यात्रा कस्तो रह्यो? पोखराको फेवाताल र माछापुच्छ्रे हिमाल त अद्भुत देखिएका थिए!", dict(speaker="sandesh", style="conversational")),
    ("sandesh_story", "5 Styles · Sandesh (M)", "Storytelling · Himalayan Valley (सन्देश M)", "ne", STORY_M, dict(speaker="sandesh", style="storytelling")),
    ("sandesh_warm", "5 Styles · Sandesh (M)", "Warm / Soft · Evening Reflection (सन्देश M)", "ne", "गहिरो श्वास लिनुहोस्, मनलाई शान्त राख्नुहोस् र आजको पललाई हृदयदेखि अनुभव गर्नुहोस्।", dict(speaker="sandesh", style="warm_soft")),
    ("sandesh_vedic", "5 Styles · Sandesh (M)", "Vedic Chant · Rigveda 1.1.1 + Svara (सन्देश M)", "sa", RV, dict(speaker="sandesh", style="vedic_chant", svara=1.0)),
    # 4. Sanskrit Classical & Vedic Showcase — Both Studio Voices (4 items)
    ("sa_sandesh_om", "Sanskrit · Both Voices", "Isha Upanishad · ॐ पूर्णमदः (सन्देश M)", "sa", OM, dict(speaker="sandesh", style="vedic_chant", svara=1.0)),
    ("sa_sandesh_gita", "Sanskrit · Both Voices", "Bhagavad Gita 1.1 · धर्मक्षेत्रे कुरुक्षेत्रे (सन्देश M)", "sa", GITA, dict(speaker="sandesh", style="vedic_chant")),
    ("sa_sakhiya_gayatri", "Sanskrit · Both Voices", "Gayatri Mantra · तत्सवितुर्वरेण्यं (सखिया F)", "sa", GAYATRI, dict(speaker="sakhiya", style="vedic_chant", svara=1.0)),
    ("sa_sandesh_subhashita", "Sanskrit · Both Voices", "Subhashita · विद्या ददाति विनयं (सन्देश M)", "sa", SUBHASHITA, dict(speaker="sandesh", style="storytelling")),
    # 5. Cross-Lingual Unification, Numbers, Loanwords & Romanized Nepali (4 items)
    ("voice_sakhiya", "General Nepali & Mixed", "Sakhiya Studio Intro · नेपाली र संस्कृत (सखिया F)", "ne", "नमस्ते, म सखिया हुँ। म नेपाली र संस्कृत भाषा स्पष्ट र स्वाभाविक आवाजमा बोल्छु।", dict(speaker="sakhiya", style="news")),
    ("sandesh_mixed", "General Nepali & Mixed", "Auto (Mixed) Nepali + Gita 2.47 · सन्देश (M)", "auto", MIXED_M, dict(speaker="sandesh", style="news")),
    ("ne_numbers", "General Nepali & Mixed", "Numbers, Currency & Loanwords (सन्देश M)", "ne", "आज रु १२५० को सामान किन्नुभयो, कम्प्युटर र इन्टरनेटबाट डिजिटल भुक्तानी सहजै भयो।", dict(speaker="sandesh", style="news")),
    ("ne_roman", "General Nepali & Mixed", "Romanized Nepali Input (सखिया F)", "ne", "ma gaunma baschu", dict(speaker="sakhiya", style="conversational")),
    # 6. 5 Newly Mined General Real-World Nepali Sentences (v3.1 Pronunciation Breakthrough)
    ("v31_mined1_baata", "v3.1 Mined Real-World Nepali", "Postposition Schwa · पोखराबाट काठमाडौँ (सखिया F)", "ne", "पोखराबाट बसपार्क हुँदै काठमाडौँ आउँदा बाटोमा धेरै रमाइलो ठाउँ देखियो।", dict(speaker="sakhiya", style="news")),
    ("v31_mined2_diphthong", "v3.1 Mined Real-World Nepali", "Nasal Diphthongs · गाउँमा दशैँ र तिहार (सन्देश M)", "ne", "हामी सबैजना गाउँमा दशैँ र तिहार मनाउन सँगै जान्छौँ।", dict(speaker="sandesh", style="conversational")),
    ("v31_mined3_loanword", "v3.1 Mined Real-World Nepali", "Loanwords & Compounds · सुदूरपश्चिम र सगरमाथा (सखिया F)", "ne", "सुदूरपश्चिमदेखि सगरमाथासम्म इन्टरनेट र एम्बुलेन्स सेवा विस्तार भइरहेको छ।", dict(speaker="sakhiya", style="news")),
    ("v31_mined4_short_ho", "v3.1 Mined Real-World Nepali", "Sentence-Final हो/छैन · समय र निर्णय (सन्देश M)", "ne", "सय वर्षको इतिहास बोकेको यो सम्पदा हाम्रो गौरव हो, यसमा कुनै शंका छैन।", dict(speaker="sandesh", style="storytelling")),
    ("v31_mined5_compound", "v3.1 Mined Real-World Nepali", "Everyday Nepali · भैँसी, दूध र तरकारी (सखिया F)", "ne", "किसानले बिहानै उठेर भैँसी दुहेर बजारमा ताजा दूध र तरकारी पुर्याउँछ।", dict(speaker="sakhiya", style="warm_soft")),
]
TIERS = {"fp16": ("docs/models_v2", False), "int8": ("docs/models_v2", True),
         "nano": ("docs/models_nano", True), "atom": ("docs/models_atom", True)}

os.makedirs(a.out, exist_ok=True)
for f in os.listdir(a.out):
    if f.endswith((".mp3", ".json")): os.remove(os.path.join(a.out, f))
engines, missing = {}, []
for t in a.tiers.split(","):
    d, tiny = TIERS[t]; d = os.path.join(ROOT, d)
    if not os.path.exists(os.path.join(d, "config.json")) or (tiny and not any(n.endswith(".int8.onnx") for n in os.listdir(d))):
        missing.append(t); continue
    engines[t] = (SakhiyaONNX(d, tiny=tiny, threads=a.threads), d, tiny)

rows = []
for sid, grp, label, lang, text, kw in ITEMS:
    row = dict(id=sid, group=grp, label=label, lang=lang, text=text)
    for t, (eng, d, tiny) in engines.items():
        try:
            t0 = time.time(); w, sr = eng.say(text, lang=lang, **kw); dt = time.time() - t0
            w = np.asarray(w, np.float32); w = w / max(1.0, float(np.abs(w).max()) / 0.95)
            sf.write(os.path.join(a.out, f"{sid}_{t}.mp3"), w, sr, format="MP3")
            row[t] = dict(file=f"{sid}_{t}.mp3", sec=round(len(w) / sr, 2), rtf=round(dt / (len(w) / sr), 3))
            print(sid, t, "ok", row[t], flush=True)
        except Exception as e:
            print(sid, t, "FAILED", repr(e)[:200], flush=True)
    rows.append(row)
sizes = {t: sum(os.path.getsize(os.path.join(d, n)) for n in os.listdir(d) if n.endswith(".onnx") and (n.endswith(".int8.onnx") == tiny))
         for t, (e, d, tiny) in engines.items()}
json.dump(dict(generated_utc=time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()), tiers=sorted(engines), missing_tiers=missing,
               onnx_bytes=sizes, items=rows), open(os.path.join(a.out, "manifest.json"), "w"), ensure_ascii=False, indent=1)
print("done", len(rows), "items; tiers", sorted(engines), "missing", missing)
