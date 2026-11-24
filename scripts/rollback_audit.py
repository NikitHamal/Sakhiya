#!/usr/bin/env python3
"""Rollback audit: runs the Leak/Loop/Splice gate on the poisoned v4 Vedic clips (ref_text = the anchor text that was
actually fed to OmniVoice) and collects Nepali accent metrics for v3b_400/200/600 vs v4 from the stored eval.json files.
Writes state/rollback_audit.json for the Marimo Rollback Audit card. Runs on the GPU box."""
import json, os, sys, glob
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import leak_loop_gate as G

ROOT = os.path.abspath(os.path.join(HERE, ".."))
OUT = os.environ.get("V5_STATE", "/work/tts/v5_recovery/state")
os.makedirs(OUT, exist_ok=True)
reg = json.load(open(os.path.join(ROOT, "docs/samples/versions/registry.json")))
texts = {c["id"]: c["text"] for c in reg["versions"][0]["clips"] if c["lang"] == "sa"}
anchors = {a["key"]: a for a in json.load(open("/work/sp/v4/vedic/anchors/anchors.json"))} if os.path.exists("/work/sp/v4/vedic/anchors/anchors.json") else {}


def anchor_for(text):
    key = "rv" if "अग्निमीळे" in text else ("om" if text.strip().startswith("ॐ") else "gita")
    return anchors.get(key, {}).get("ref_text", ""), key


rows = []
for d in ("sakhiya_pro_v4", "sakhiya_pro_v4_vedic"):
    for p in sorted(glob.glob(os.path.join(ROOT, "docs/samples/versions", d, "sa_*.mp3"))):
        cid = os.path.basename(p)[:-4]
        if cid not in texts:
            continue
        rows.append(dict(pack=d, id=cid, wav=p, target_text=texts[cid], mode="vedic"))
try:
    hyps = G._asr_batch([r["wav"] for r in rows])
except Exception as e:
    print("ASR unavailable:", repr(e)[:200]); hyps = {}
v4 = []
for r in rows:
    ref, key = anchor_for(r["target_text"])
    w, sr = G.load_wav(r["wav"])
    res = G.check_clip(w, sr, r["target_text"], ref, hyps.get(r["wav"]), "vedic")
    v4.append(dict(pack=r["pack"], id=r["id"], anchor=key, ok=res["ok"], rejects=res["rejects"], leaks=res["leaks"],
                   loops=[(l["ngram"], l["repeats"]) for l in res["loops"]], hyp=res["hyp"], om_s=res["om_s"], clicks=res["clicks"]))
    print(("PASS " if res["ok"] else "REJECT "), r["pack"], r["id"], res["rejects"])

acc = {}
for name, pat in (("v3b_400", "v3b_checkpoint-400"), ("v3b_200", "v3b_checkpoint-200"), ("v3b_600", "v3b_checkpoint-600")):
    for base in ("/work/sp/hfpull/sakhiya_pro_v3/v3/eval", "/work/sp/v4/eval", "/work/tts/v5_recovery/hf/sakhiya_pro_v3/v3/eval"):
        f = os.path.join(base, pat, "eval.json")
        if os.path.exists(f):
            j = json.load(open(f)); s = j.get("summary") or j.get("agg") or {k: v for k, v in j.items() if not isinstance(v, (list, dict))}
            acc[name] = dict(src=f, **{k: (round(v, 3) if isinstance(v, float) else v) for k, v in (s.items() if isinstance(s, dict) else [])})
            break
for name, f in (("v4_poisoned", "/work/sp/v4/eval/v4b_checkpoint-100/eval.json"),):
    if os.path.exists(f):
        j = json.load(open(f)); s = j.get("summary") or j.get("agg") or {}
        acc[name] = dict(src=f, **{k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items()})

audit = dict(
    root_cause=["anchor = anchor_v4[:3.2s] (hard slice, mid-sentence Nepali) + chant[:5.8s] (hard slice) but ref_text = FULL verse -> "
                "model completes the missing verse tail (leaked 'प्रचोदयात्', '...सञ्जय')",
                "Gayatri anchor reused for Shanti (startswith ॐ) -> identical broken opening",
                "'ओऽऽऽम्' stretch hack + post-hoc Om splice (soften_visarga_and_shape_svara) -> नम् नम् loop / collapsed ॐ / splice",
                "Seed-VC converted timbre anchors: ref audio no longer matches ref_text phonetics"],
    v4_vedic=v4, v4_rejected=sum(1 for r in v4 if not r["ok"]), v4_leak_clips=sum(1 for r in v4 if r["leaks"]),
    v4_loop_clips=sum(1 for r in v4 if r["loops"]), nepali_accent=acc)
json.dump(audit, open(os.path.join(OUT, "rollback_audit.json"), "w"), ensure_ascii=False, indent=1)
print("AUDIT", audit["v4_rejected"], "/", len(v4), "rejected; leak clips", audit["v4_leak_clips"], "loop clips", audit["v4_loop_clips"])
