#!/usr/bin/env python3
"""v11: decide whether the mined-data fine-tune of Sakhiya-V1 ships, on the real forensic gate scored on the INT8 graphs the
site serves, against V1 re-measured on the same gate on the same box (fair A/B).
Ships the better of best/last only if ALL hold:
  * overall gate score >= V1 + 0.03 (UTMOS - 0.5 drops - 0.3 stutters - clicks/500);
  * no more word drops or stutters than V1;
  * Sakhiya voice preserved: speaker-verification cosine (ne vs sa) >= V1 - 0.005;
  * storytelling and vedic_chant per-style UTMOS each >= V1 - 0.02 (the styles the owner hears problems in), and at least one
    of them improves by >= 0.05.
Always writes eval/v11_v1_decision.json + the candidate gate; renders A/B wavs for the MoLab Listen card."""
import json, os, shutil, subprocess, sys
import numpy as np, soundfile as sf
R = "/work/tts"; V = f"{R}/v11"; REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..")); sys.path.insert(0, REPO)
GIT = "GIT_ASKPASS=/tmp/gitenv.sh GIT_TERMINAL_PROMPT=0 git -c user.name=Sakhiya -c user.email=sakhiya@example.com"


def sh(c): return subprocess.run(c, shell=True, cwd=REPO, capture_output=True, text=True)
def score(f): return (f.get("utmos_15") or 0) - 0.5 * (f.get("word_drop_count") or 0) - 0.3 * (f.get("stutter_count") or 0) - 0.02 * (f.get("clicks") or 0) / 10.0


def by_style(f):
    d = {}
    for c in f.get("per_clip", []): d.setdefault(c["style"], []).append(c["utmos"])
    return {k: round(float(np.mean(v)), 3) for k, v in d.items()}


def summ(f): return dict(score=round(score(f), 3), utmos=f.get("utmos_15"), drops=f.get("word_drop_count"), stutters=f.get("stutter_count"), clicks=f.get("clicks"), sv_cos=f.get("sv_cos_ne_sa"), styles=by_style(f))


def main():
    g1 = json.load(open(f"{V}/gate_v1.json")); cands = {}
    for k in ("best", "last"):
        p = f"{V}/gate_{k}.json"
        if os.path.exists(p): cands[k] = json.load(open(p))
    if not cands: sys.exit("no candidate gates")
    pick = max(cands, key=lambda k: score(cands[k])); g = cands[pick]
    s1, sc = summ(g1), summ(g)
    st1, stc = s1["styles"], sc["styles"]
    keys = [k for k in ("storytelling", "vedic_chant") if k in st1 and k in stc]
    checks = dict(
        score=sc["score"] >= s1["score"] + 0.03,
        drops=(sc["drops"] or 0) <= (s1["drops"] or 0),
        stutters=(sc["stutters"] or 0) <= (s1["stutters"] or 0),
        voice=(sc["sv_cos"] or 0) >= (s1["sv_cos"] or 0) - 0.005,
        weak_styles_not_worse=all(stc[k] >= st1[k] - 0.02 for k in keys),
        weak_style_improved=any(stc[k] >= st1[k] + 0.05 for k in keys))
    ok = all(checks.values())
    rec = dict(pick=pick, ship=ok, checks=checks, v1=s1, candidate=sc, all_candidates={k: summ(v) for k, v in cands.items()})
    json.dump(rec, open(f"{REPO}/eval/v11_v1_decision.json", "w"), indent=1, ensure_ascii=False)
    shutil.copyfile(f"{V}/gate_{pick}.json", f"{REPO}/eval/forensic_v11_v1_candidate.json")
    print("DECISION", json.dumps(rec, ensure_ascii=False), flush=True)
    # A/B wavs (same sentences, int8 graphs) for the Listen card
    from sakhiya.onnx_tts import SakhiyaONNX
    out = f"{R}/v10/samples"; os.makedirs(out, exist_ok=True)
    SENT = [("v11_1_news", "ne", "news", "सरकारले आज नयाँ बजेट सार्वजनिक गर्दै शिक्षा र स्वास्थ्य क्षेत्रमा थप लगानी गर्ने घोषणा गरेको छ।"),
            ("v11_2_story", "ne", "storytelling", "धेरै पहिले एउटा सानो गाउँमा एउटी बुढी आमा बस्नुहुन्थ्यो, उहाँको एउटा प्यारो बाख्रो थियो।"),
            ("v11_3_expressive", "ne", "expressive", "वाह! के अद्भुत दृश्य हो यो! माछापुच्छ्रे हिमाल बिहानको घाममा चम्किरहेको छ!"),
            ("v11_4_conv", "ne", "conversational", "अनि तिमी भोलि कति बजे आउँछौ? म चाहिँ बिहानै तयार भएर बस्छु है।"),
            ("v11_5_vedic", "sa", "vedic_chant", "ॐ असतो मा सद्गमय। तमसो मा ज्योतिर्गमय। मृत्योर्मा अमृतं गमय॥")]
    for name, d in (("V1", f"{V}/stage_v1"), ("V1-v11", f"{V}/stage_{pick}")):
        eng = SakhiyaONNX(d, tiny=True, threads=4)
        for sid, lang, style, text in SENT:
            w, sr = eng.say(text, speaker="sakhiya", lang=lang, style=style)
            sf.write(f"{out}/{sid}__{name}.wav", np.asarray(w, dtype=np.float32), sr, subtype="PCM_16")
    if not ok:
        print(sh(f"{GIT} add eval/v11_v1_decision.json eval/forensic_v11_v1_candidate.json && {GIT} commit -qm 'v11: V1 mined-data fine-tune measured, NOT shipped (see eval/v11_v1_decision.json)' && {GIT} pull -q --rebase origin main && {GIT} push -q origin HEAD:main").stderr[-300:])
        sys.exit("NOT SHIPPED: " + ", ".join(k for k, v in checks.items() if not v))
    d = f"{REPO}/docs/models_v2"; stg = f"{V}/stage_{pick}"
    for f in os.listdir(stg):
        if f.endswith(".int8.onnx") or f in ("config.json", "manifest.json", "size.json"): shutil.copyfile(f"{stg}/{f}", f"{d}/{f}")
    shutil.copyfile(f"{V}/gate_{pick}.json", f"{REPO}/eval/forensic_v11_v1.json")
    r = sh(f"{sys.executable} scripts/freeze_version_pack.py --tag v11_v1 --label 'Sakhiya-V1 (v11 mined-data fine-tune)' --models docs/models_v2 --metrics-json eval/forensic_v11_v1.json --tiny "
           f"--notes 'v11: V1 acoustic model fine-tuned on 9 h mined Nepali (AssemblyAI/Gemini transcripts) + 2.7 h verse-matched Vedic via speaker slots; Sakhiya voice kept on slot 0; gated vs V1 on the int8 graphs'")
    print(r.stdout[-300:], r.stderr[-300:])
    tj = json.load(open(f"{REPO}/docs/tiers.json"))
    for t in tj["tiers"]:
        if t["id"] == "v1": t.update(utmos=g.get("utmos_15"), desc="Flagship · fine-tuned on new expressive Nepali + verified Vedic · unified Sakhiya voice")
    json.dump(tj, open(f"{REPO}/docs/tiers.json", "w"), indent=1, ensure_ascii=False)
    print(sh(f"{GIT} add docs eval && {GIT} commit -qm 'v11: ship V1 mined-data fine-tune (gate {s1['score']} -> {sc['score']})' && {GIT} pull -q --rebase origin main && {GIT} push -q origin HEAD:main").stderr[-300:])
    print("SHIPPED v11_v1", flush=True)


if __name__ == "__main__":
    main()
