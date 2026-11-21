#!/usr/bin/env python3
"""Build the GitHub-Pages Clip Library (docs/clips/*.mp3 + docs/clips/library.json) from the live pipeline state on the GPU box.
Status per clip: kept (packed dataset) | heard (ASR+alignment) | cleaned (quality gate only) | vedic_ok / vedic_bad (purity gate) | sakhiya (voice-converted).
Caps keep the repo small (32 kbps mono mp3).  usage: python scripts/publish_clip_library.py [--push]"""
import argparse, collections, glob, json, os, random, re, subprocess, time
from concurrent.futures import ThreadPoolExecutor
ROOT = os.environ.get("TTS_ROOT", "/work/tts"); REPO = f"{ROOT}/repo"; OUT = f"{REPO}/docs/clips"
CAP = {"kept": 350, "heard": 500, "cleaned": 120, "vedic_gold": 250, "vedic_ok": 100, "vedic_bad": 60, "sakhiya": 120}
def jl(p):
    try: return [json.loads(l) for l in open(p)]
    except Exception: return []

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--push", action="store_true"); a = ap.parse_args(); random.seed(3); os.makedirs(OUT, exist_ok=True); items = {}
    def add(status, path, style, text="", **kw):
        if not path or not os.path.exists(path): return
        cid = os.path.basename(path)[:-4]
        if cid in items and items[cid]["status"] in ("kept", "vedic_gold", "vedic_ok", "sakhiya") or (status == "cleaned" and cid in items): return
        items[cid] = {"id": cid, "status": status, "style": style, "text": text, "src": os.path.basename(path).rsplit("_", 1)[0][:48], "_p": path, **kw}
    meta = jl(f"{ROOT}/dataset/meta.jsonl"); asr = {os.path.basename(r["path"])[:-4]: r for r in jl(f"{ROOT}/asr/asr.jsonl")}
    for m in meta:
        p = next(iter(glob.glob(f"{ROOT}/segments/*/{m['id']}.wav")), ""); add("kept", p, m.get("style", ""), m.get("text", ""), utmos=m.get("utmos"), agree=m.get("agree"))
    for cid, r in asr.items(): add("heard", r["path"], r["cat"], r.get("text_w", ""), utmos=r.get("utmos"), agree=r.get("agree"))
    for r in jl(f"{ROOT}/gate/passed.jsonl"): add("cleaned", r["path"], r["cat"], "", utmos=r.get("utmos"))
    # Vedic (owner fix): ONLY verified Sanskrit is shown as good, and its words are the REFERENCE verse text (vag / Gita-supersite / verse-matched class clips).
    try: ref = json.load(open(f"{ROOT}/vedic/ref_text.json"))
    except Exception: ref = {}
    ver = {r["path"]: r for r in jl(f"{ROOT}/vedic/verified.jsonl") if r.get("ok")}; good = set()
    try: gold = json.load(open(f"{ROOT}/vedic/gold.json"))   # independent sound-vs-verse audit (scripts/vedic_gold_audit.py): only audited clips can be shown as good
    except Exception: gold = {}
    for p, t in ref.items():
        b = os.path.basename(p); src = "vag" if b.startswith("vag_") else "gss" if b.startswith("gss_") else "rec" if p in ver else ""
        g = gold.get(p)
        if not src or not g: continue
        kw = dict(vsrc=src, lid="sa", score=ver.get(p, {}).get("score"), match=round(g["match"]), melodic=g.get("melodic", False))
        if g["tier"] == "gold": good.add(p); add("vedic_gold", p, "vedic_chant", t, **kw)
        elif g["tier"] == "ok": good.add(p); add("vedic_ok", p, "vedic_chant", t, **kw)
        else: good.add(p); add("vedic_bad", p, "vedic_chant", t, why=g.get("why", []), **kw)
    for r in jl(f"{ROOT}/vedic/verdicts.jsonl"):
        if r["path"] not in good and r["verdict"] == "rejected": add("vedic_bad", r["path"], "vedic_chant", "", why=r.get("why", []), lid=r.get("top_lang"))
    try:   # voice-converted (sakhiya) clips live inside dataset/vc/*.parquet -> unpack a preview set
        import pyarrow.parquet as pq; os.makedirs(f"{ROOT}/vc_preview", exist_ok=True)
        for pf in sorted(glob.glob(f"{ROOT}/dataset/vc/*.parquet")):
            t = pq.read_table(pf, columns=["id", "wav", "style"]).to_pylist()
            for r in t[:CAP["sakhiya"]]:
                o = f"{ROOT}/vc_preview/{r['style'][:5]}_{r['id']}.wav"
                if not os.path.exists(o): open(o, "wb").write(r["wav"])
    except Exception as e: print("vc preview:", e)
    for p in sorted(glob.glob(f"{ROOT}/vc_preview/*.wav"))[:CAP["sakhiya"]]: add("sakhiya", p, "storytelling" if "story" in p else "vedic_chant" if "vedic" in p else "news" if "news" in p else "conversational")
    ic = {r["id"]: r for r in jl(f"{ROOT}/asr/indic.jsonl") if r.get("text_ic")}   # IndicConformer-600M Nepali words (far more accurate than Whisper); old guess kept for comparison
    for cid, c in items.items():
        r = ic.get(cid)
        if r and c["status"] in ("kept", "heard", "cleaned"):
            old = c.get("text", "") or ""; m = re.search(r"[?!।]\s*$", old)
            c["text_w"] = old; c["text"] = r["text_ic"].strip() + (m.group(0).strip() if m else ""); c["engine"] = "ic"
            if r.get("agree_ic") is not None: c["ic_agree"] = r["agree_ic"]
    # Gemini (3.x) transcripts: far more accurate than Whisper/MMS; the language it hears is the Vedic purity gate (Tamil/Telugu/... = removed, real Sanskrit = shown)
    gm = {}
    for r in jl(f"{ROOT}/asr/gemini.jsonl"): gm[r["id"]] = r
    BADL = {"tamil", "telugu", "malayalam", "kannada", "english", "other", "none", ""}
    for cid, r in gm.items():
        t = r.get("text", "")
        if r["style"] == "vedic_chant" or "/vedic_chant/" in r.get("path", ""): continue   # owner rule: Gemini never touches Sanskrit/Vedic
        elif cid in items and t and items[cid]["status"] in ("kept", "heard", "cleaned"):
            c = items[cid]; c["text_w"] = c.get("text_w") or c.get("text", ""); c["text"] = t; c["engine"] = "gemini"; c["lid"] = r["lang"]
    try: fx = json.load(open(f"{ROOT}/corrections.json"))   # owner audit: corrected words override the machine's; garbage clips never appear as good
    except Exception: fx = {}
    for cid, f in fx.items():
        if cid in items:
            if f.get("verdict") == "bad" and items[cid]["status"] in ("kept", "heard", "vedic_gold", "vedic_ok", "cleaned"): items.pop(cid); continue
            if f.get("text"): items[cid]["text"] = f["text"]; items[cid]["fixed"] = f.get("verdict") or "fix"
    by = collections.defaultdict(list)
    for c in items.values(): by[(c["status"], c["style"])].append(c)
    sel = []
    for st, cap in CAP.items():
        pools = [v for (s, _), v in by.items() if s == st]; [random.shuffle(v) for v in pools]; n = 0
        if st in ("vedic_gold", "vedic_ok"): [v.sort(key=lambda c: (c.get("vsrc") == "rec", c.get("melodic", False), c.get("match", 0))) for v in pools]   # best-matching melodic chanting is popped first
        while n < cap and any(pools):   # round-robin across styles/sources so every style is represented
            for v in pools:
                if v and n < cap: sel.append(v.pop()); n += 1
    def enc(c):
        o = f"{OUT}/{c['id']}.mp3"; c["file"] = c["id"] + ".mp3"
        if not os.path.exists(o): subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", c["_p"], "-ac", "1", "-ar", "22050", "-b:a", "24k", o], timeout=60)
        c["dur"] = round(os.path.getsize(o) / 3000, 2) if os.path.exists(o) else 0; return c
    with ThreadPoolExecutor(12) as ex: sel = [c for c in ex.map(enc, sel) if c["dur"] > 0]
    for c in sel: c.pop("_p", None); [c.pop(k) for k in [k for k, v in c.items() if v is None]]
    keep = {c["file"] for c in sel}
    for f in glob.glob(f"{OUT}/*.mp3"):
        if os.path.basename(f) not in keep: os.remove(f)
    json.dump({"clips": sel, "meta": {"hours": sum(c["dur"] for c in sel) / 3600, "built": time.strftime("%Y-%m-%d %H:%M"), "counts": dict(collections.Counter(c["status"] for c in sel))}},
              open(f"{OUT}/library.json", "w"), ensure_ascii=False)
    print("LIBRARY", len(sel), dict(collections.Counter(c["status"] for c in sel)), "MB", round(sum(os.path.getsize(f) for f in glob.glob(f"{OUT}/*.mp3")) / 1e6, 1), flush=True)
    if a.push:
        env = dict(os.environ, GIT_ASKPASS="/tmp/gitenv.sh", GIT_TERMINAL_PROMPT="0"); sh = lambda c: subprocess.run(c, shell=True, cwd=REPO, env=env, capture_output=True, text=True).stdout[-300:]
        sh("git add docs/clips && git commit -qm 'docs: Clip Library refresh (mined clips, verified Sanskrit, removed clips)'"); print(sh("git pull --rebase origin main 2>&1 | tail -1; git push origin HEAD:main 2>&1 | tail -1"))

if __name__ == "__main__": main()
