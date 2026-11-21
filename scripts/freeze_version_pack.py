#!/usr/bin/env python3
"""Render and immutably freeze the 15-Clip Master Comparison Pack for a version milestone.
Once `docs/samples/versions/<tag>/` is created and locked with `.frozen`, it is NEVER overwritten."""
import argparse, datetime, json, os, subprocess, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from sakhiya.onnx_tts import SakhiyaONNX

MASTER_15_CLIPS = [
    ("ne_kathmandu", "ne", "news", "नेपालको राजधानी काठमाडौँ हिमालको काखमा रहेको एउटा ऐतिहासिक सहर हो।"),
    ("ne_story", "ne", "storytelling", "एक समयको कुरा हो, एउटा सानो गाउँमा एउटी बुढी आमा बस्थिन्। उनी बिहानै उठेर चरालाई दाना हाल्थिन्।"),
    ("ne_chat", "ne", "conversational", "नमस्ते! आज मौसम धेरै राम्रो छ, चिया खान जाऔँ न? तिमी कताबाट आउँदै छौ?"),
    ("ne_warm", "ne", "warm_soft", "शान्त मन र मधुर मुस्कानले हरेक कठिन दिनलाई सहज बनाउँछ। गहिरो श्वास लिनुहोस् र आजको पललाई हृदयदेखि अनुभव गर्नुहोस्।"),
    ("ne_expressive", "ne", "expressive", "वाह! के अद्भुत दृश्य हो यो! माछापुच्छ्रे हिमाल बिहानको घाममा सुनजस्तै चम्किरहेको छ!"),
    ("ne_tricky_real", "ne", "conversational", "बाटोबाट हिँड्दै गर्दा गाउँका साथीहरूसँग भेट भयो, अनि हामीले विद्यालयको कार्यक्रमबारे लामो कुरा गर्‍यौँ।"),
    ("ne_loan_num", "ne", "news", "सन् २०२६ मा नेपालको प्रविधि क्षेत्रमा १५ प्रतिशत वृद्धि भएको छ, र मोबाइल एप्लिकेसन चलाउने प्रयोगकर्ता झन् बढेका छन्।"),
    ("sa_om", "sa", "vedic_chant", "ॐ पूर्णमदः पूर्णमिदं पूर्णात्पूर्णमुदच्यते। पूर्णस्य पूर्णमादाय पूर्णमेवावशिष्यते॥"),
    ("sa_rv", "sa", "vedic_chant", "अ॒ग्निमी॑ळे पु॒रोहि॑तं य॒ज्ञस्य॑ दे॒वमृ॒त्विज॑म् । होता॑रं रत्न॒धात॑मम् ॥"),
    ("sa_gita", "sa", "storytelling", "धर्मक्षेत्रे कुरुक्षेत्रे समवेता युयुत्सवः। मामकाः पाण्डवाश्चैव किमकुर्वत सञ्जय॥"),
    ("sa_gayatri", "sa", "vedic_chant", "ॐ भूर्भुवः स्वः। तत्सवितुर्वरेण्यं भर्गो देवस्य धीमहि। धियो यो नः प्रचोदयात्॥"),
    ("mixed_owner", "auto", "news", "हेलो साथीहरू। म सखिया हुँ, निकित हमालज्यूले बनाउनुभएको सानो टेक्स्ट टु स्पीच मोडेल। म संस्कृत पनि स्पष्ट बोल्न सक्छु। धर्मक्षेत्रे कुरुक्षेत्रे समवेता युयुत्सवः। मामकाः पाण्डवाश्चैव किमकुर्वत सञ्जय॥ आज मौसम निकै रमाइलो र शान्त छ।"),
    ("ne_unseen_dialogue", "ne", "conversational", "दिदी, आज बजारमा नयाँ किताबहरू आएका रहेछन्। हामी साँझपख सँगै गएर हेरौँला नि, हुँदैन र?"),
    ("ne_unseen_narrative", "ne", "storytelling", "वर्षा थामिएपछि पहाडको टुप्पोबाट कलकल बग्ने झरनाको आवाज अझै मीठो सुनियो। गाउँलेहरू आआफ्नो खेतबारीमा काम गर्न निस्किए।"),
    ("ne_unseen_prose", "ne", "news", "स्वास्थ्य र शिक्षामा समान पहुँच पुर्‍याउन स्थानीय तहले डिजिटल पुस्तकालय र घुम्ती स्वास्थ्य शिविर सञ्चालन गरेको छ।"),
]


def save_mp3(wav, sr, mp3_path):
    import soundfile as sf
    wav_tmp = mp3_path + ".tmp.wav"
    sf.write(wav_tmp, np.asarray(wav, dtype=np.float32), sr, subtype="PCM_16")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", wav_tmp, "-codec:a", "libmp3lame", "-qscale:a", "3", mp3_path], check=True)
    os.remove(wav_tmp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="e.g. v3_1, v4_m1, v4_m2, v4_final")
    ap.add_argument("--label", required=True)
    ap.add_argument("--models", default="docs/models")
    ap.add_argument("--metrics-json", default="eval/bench200_v31.json")
    ap.add_argument("--notes", default="")
    ap.add_argument("--tiny", action="store_true", help="render with the int8 graphs the website actually serves")
    a = ap.parse_args()

    out_dir = os.path.join("docs", "samples", "versions", a.tag)
    lock_file = os.path.join(out_dir, ".frozen")
    if os.path.exists(lock_file):
        print(f"[IMMUTABLE GUARD] {out_dir} is already frozen! Refusing to overwrite.")
        return

    os.makedirs(out_dir, exist_ok=True)
    eng = SakhiyaONNX(a.models, tiny=a.tiny)
    clips_meta = []
    for cid, lang, style, text in MASTER_15_CLIPS:
        w, sr = eng.say(text, speaker="sakhiya", lang=lang, style=style)
        mp3_p = os.path.join(out_dir, f"{cid}.mp3")
        save_mp3(w, sr, mp3_p)
        dur_s = round(len(w) / float(sr), 2)
        clips_meta.append(dict(id=cid, lang=lang, style=style, text=text, mp3=f"samples/versions/{a.tag}/{cid}.mp3", duration_s=dur_s))
        print(f"  rendered {a.tag}/{cid}.mp3 ({dur_s}s)")

    sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    now_utc = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    m_data = json.load(open(a.metrics_json)) if os.path.exists(a.metrics_json) else {}
    ver_entry = dict(
        tag=a.tag,
        label=a.label,
        commit_sha=sha,
        timestamp_utc=now_utc,
        notes=a.notes,
        metrics=dict(
            # measured-only: a missing metric is None, never a hand-typed placeholder
            ne_pcer=m_data.get("ne_pcer"), ne_wer=m_data.get("ne_wer"), sa_pcer=m_data.get("sa_pcer"),
            utmos=m_data.get("utmos_sakhiya", m_data.get("utmos_15")),
            sv_cos_sim=m_data.get("wavlm_sv_sakhiya", m_data.get("sv_cos_ne_sa")),
            cross_style_sv=m_data.get("cross_style_sv"), rigveda_svara_st=m_data.get("rigveda_svara_st"),
            style_f0_ratio=m_data.get("style_f0_std_ratio"),
            glitch_frames=m_data.get("glitch_frames", m_data.get("f0_shutters")),
            word_drops=m_data.get("word_drop_count"),
        ),
        clips=clips_meta,
    )
    json.dump(ver_entry, open(os.path.join(out_dir, "manifest.json"), "w"), indent=2, ensure_ascii=False)
    open(lock_file, "w").write(f"Frozen at {now_utc} commit {sha}\n")

    reg_path = os.path.join("docs", "samples", "versions", "registry.json")
    reg = json.load(open(reg_path)) if os.path.exists(reg_path) else {"versions": [], "clips": [dict(id=c[0], lang=c[1], style=c[2], text=c[3]) for c in MASTER_15_CLIPS]}
    reg["versions"] = [v for v in reg["versions"] if v["tag"] != a.tag] + [ver_entry]
    json.dump(reg, open(reg_path, "w"), indent=2, ensure_ascii=False)
    print(f"FROZEN {a.tag} ({len(clips_meta)} clips) -> {out_dir}")


if __name__ == "__main__":
    main()
