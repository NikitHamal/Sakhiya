#!/usr/bin/env python3
"""v12: ship decision for the cleaned V1 vocoder. Picks the best chunk that passed its per-chunk gate and ships ONLY if, on the
forensic gate (shipped V1 int8 graphs, only vocoder.int8.onnx swapped) vs shipped V1 on the same clips:
  score >= V1 + 0.03, no more word drops / stutters / clicks, voice match >= V1 - 0.005, and no style worse than V1 - 0.03.
Always renders the A/B samples (same 5 sentences, int8) so the owner can listen whatever the decision."""
import json, os, shutil, subprocess, sys
import numpy as np, soundfile as sf
R = "/work/tts"; V = f"{R}/v12"; REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..")); sys.path.insert(0, REPO)
from scripts.v12_voc_clean import summ  # noqa: E402
GIT = "GIT_ASKPASS=/tmp/gitenv.sh GIT_TERMINAL_PROMPT=0 git"
SENT = [("1_news", "ne", "news", "सरकारले आज नयाँ बजेट सार्वजनिक गर्दै शिक्षा र स्वास्थ्य क्षेत्रमा थप लगानी गर्ने घोषणा गरेको छ।"),
        ("2_story", "ne", "storytelling", "धेरै पहिले एउटा सानो गाउँमा एउटी बुढी आमा बस्नुहुन्थ्यो, उहाँको एउटा प्यारो बाख्रो थियो।"),
        ("3_expressive", "ne", "expressive", "वाह! के अद्भुत दृश्य हो यो! माछापुच्छ्रे हिमाल बिहानको घाममा चम्किरहेको छ!"),
        ("4_conv", "ne", "conversational", "अनि तिमी भोलि कति बजे आउँछौ? म चाहिँ बिहानै तयार भएर बस्छु है।"),
        ("5_vedic", "sa", "vedic_chant", "ॐ असतो मा सद्गमय। तमसो मा ज्योतिर्गमय। मृत्योर्मा अमृतं गमय॥")]


def sh(c): return subprocess.run(c, shell=True, cwd=REPO, capture_output=True, text=True)


def main():
    rows = [json.loads(l) for l in open(f"{V}/chunks.jsonl")] if os.path.exists(f"{V}/chunks.jsonl") else []
    oks = [r for r in rows if r["ok"]]
    if not rows: sys.exit("no chunks")
    pick = max(oks, key=lambda r: r["cand"]["score"]) if oks else max(rows, key=lambda r: r["cand"]["score"])
    b, c = pick["base"], pick["cand"]
    checks = dict(score=c["score"] >= b["score"] + 0.03, drops=c["drops"] <= b["drops"], stutters=c["stutters"] <= b["stutters"], clicks=c["clicks"] <= b["clicks"],
                  voice=(c["sv_cos"] or 0) >= (b["sv_cos"] or 0) - 0.005,
                  no_style_worse=all(c["styles"].get(k, 0) >= v - 0.03 for k, v in b["styles"].items()), passed_chunk_gate=pick["ok"])
    ship = all(checks.values())
    rec = dict(ship=ship, chunk=pick["chunk"], step=pick["step"], checks=checks, v1=b, candidate=c, chunks=[dict(chunk=r["chunk"], score=r["cand"]["score"], ok=r["ok"]) for r in rows],
               summary=json.load(open(f"{V}/summary.json")) if os.path.exists(f"{V}/summary.json") else None)
    os.makedirs(f"{REPO}/eval", exist_ok=True)
    json.dump(rec, open(f"{REPO}/eval/v12_voc_decision.json", "w"), indent=1, ensure_ascii=False)
    print("DECISION", json.dumps(rec, ensure_ascii=False)[:1800], flush=True)
    from sakhiya.onnx_tts import SakhiyaONNX
    out = f"{V}/samples"; os.makedirs(out, exist_ok=True)
    for f in os.listdir(out): os.remove(f"{out}/{f}")
    for name, d in (("V1", f"{V}/stage_base"), ("V1-cleanvoc", f"{V}/stage_c{pick['chunk']}")):
        eng = SakhiyaONNX(d, tiny=True, threads=4)
        for sid, lang, style, text in SENT:
            w, sr = eng.say(text, speaker="sakhiya", lang=lang, style=style)
            sf.write(f"{out}/{sid}__{name}.wav", np.asarray(w, dtype=np.float32), sr, subtype="PCM_16")
    print("samples rendered", flush=True)
    shutil.copyfile(f"{V}/gate_c{pick['chunk']}.json", f"{REPO}/eval/forensic_v12_voc_candidate.json")
    if not ship:
        print(sh(f"{GIT} add eval/v12_voc_decision.json eval/forensic_v12_voc_candidate.json && {GIT} commit -qm 'v12: cleaned V1 vocoder measured, NOT shipped (eval/v12_voc_decision.json)' && {GIT} pull -q --rebase origin main && {GIT} push -q origin HEAD:main").stderr[-300:])
        sys.exit("NOT SHIPPED: " + ", ".join(k for k, v in checks.items() if not v))
    shutil.copyfile(f"{V}/stage_c{pick['chunk']}/vocoder.int8.onnx", f"{REPO}/docs/models_v2/vocoder.int8.onnx")
    shutil.copyfile(f"{V}/gate_c{pick['chunk']}.json", f"{REPO}/eval/forensic_v12_v1.json")
    r = sh(f"{sys.executable} scripts/freeze_version_pack.py --tag v12_v1 --label 'Sakhiya-V1 (v12 clean vocoder)' --models docs/models_v2 --metrics-json eval/forensic_v12_v1.json --tiny "
           f"--notes 'v12: same V1 acoustic model; vocoder fine-tuned on real speech in 10-min chunks, each chunk gated on the int8 graphs'")
    print(r.stdout[-300:], r.stderr[-300:])
    tj = json.load(open(f"{REPO}/docs/tiers.json"))
    for t in tj["tiers"]:
        if t["id"] == "v1": t.update(utmos=c["utmos"])
    json.dump(tj, open(f"{REPO}/docs/tiers.json", "w"), indent=1, ensure_ascii=False)
    print(sh(f"{GIT} add docs eval && {GIT} commit -qm 'v12: ship cleaned V1 vocoder (gate {b['score']} -> {c['score']})' && {GIT} pull -q --rebase origin main && {GIT} push -q origin HEAD:main").stderr[-300:])
    print("SHIPPED v12 clean vocoder", flush=True)


if __name__ == "__main__":
    main()
