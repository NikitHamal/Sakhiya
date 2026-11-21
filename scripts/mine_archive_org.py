#!/usr/bin/env python3
"""Internet-Archive mirror miner (fallback/primary source when YouTube bot-walls the datacenter IP).
Discovers items by category queries + curated seeds, downloads audio files in parallel, converts to
24 kHz mono WAV under /work/tts/youtube_raw/<cat>/ia_<item>__<n>.wav, records provenance in
/work/tts/ia_manifest.jsonl (item id, license url, title, file) for licence auditing.
usage: mine_archive_org.py [--max-hours 120] [--workers 8] [--per-item-mb 500]"""
import argparse, json, os, subprocess, sys, time, urllib.parse, urllib.request, threading
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = os.environ.get("TTS_ROOT", "/work/tts"); RAW = f"{ROOT}/youtube_raw"; MAN = f"{ROOT}/ia_manifest.jsonl"
UA = {"User-Agent": "curl/8.5.0"}
QUERIES = {
    "storytelling": ["language:(nep) AND mediatype:audio AND (novel OR katha OR story OR upanyas OR kahani OR drama)",
                     "(nepali OR nepal) AND mediatype:audio AND (novel OR katha OR audiobook OR kahani OR upanyas)",
                     "chisapani nepal novel", "dear kalyan katha", "radio nepal katha", "nepali kavita vachan", "nepali audio book"],
    "conversational": ["language:(nep) AND mediatype:audio AND (podcast OR interview OR kurakani OR bhet OR speech OR bhashan)",
                       "nepali interview mediatype:audio", "nepali podcast mediatype:audio", "nepali bhashan pravachan"],
    "news": ["nepali samachar mediatype:audio", "radio nepal news nepali", "nepali news bulletin mediatype:audio", "language:(nep) AND news AND mediatype:audio"],
    "vedic_chant": ["(rigveda OR yajurveda OR samaveda OR atharvaveda) AND mediatype:audio", "(upanishad OR upanishads) AND (sanskrit OR chant) AND mediatype:audio",
                    "(gayatri OR mantra) AND (chanting OR chant OR paath) AND mediatype:audio", "(bhagavad OR gita) AND sanskrit AND mediatype:audio",
                    "sanskrit AND (stotra OR stotram OR sukta OR sahasranama) AND mediatype:audio", "vedic chanting ghanapatha", "veda parayanam"],
}
SEEDS = {"vedic_chant": ["Ramayana-recitation-Sriram-harisItArAmamUrti-Ghanapaati-v2",
                         "VEDAVEDAMVEDICYAJURVEDAKRISHNAYAJURVEDACOMPLETEAUDIOSAMHITAARANYAKABRAHMANAMSriK", "mahAbhArata-mUla-paThanam-GP", "Shrimad-Bhagawat-Samhita-sanskrit",
                         "mantrachantingmeditation", "VishnuSahasranamaStotra", "RamaRakshaStotra", "GanapatiAtharvashirsha", "SrimadramayanamSundarakandaSanskritsarga1-10", "BhagavadGitaCh01"],
         "storytelling": ["SubinBhattaraiSummerLove", "AmarNeupaneSeroDharti", "DaminiBhir", "SetoBaghDiamondShumsherRana", "SaradPoudelLikhe", "KrishnaAbiralLalChudi",
                          "BijayMallaAnuradha", "JagdishGhimireySakas", "PagalBasti", "MadhavPyakurelGularKoPhool", "FullNovel-Saya-SubinBhattraiChisapaniNepal",
                          "FullNovel-KarodauKasturiByAmarNeupaneFMProgramParijat", "BabuAamaRaChhora-BpKoirala", "merokathadecMarch", "HermanHesseSiddhartha"]}
# v6.1 owner fix: lectures/classes/discourses are TALK (Malayalam/Tamil/Telugu/English commentary), never chanting.
LECTURE = ("discourse", "class ", "lecture", "pravachan", "pravachanam", "commentary", "introduction", "malayalam", "tamil", "telugu", "kannada",
           "english", "nochur", "upanyasam", "satsang", "talk", "explanation", "meaning", "translation", "interview", "bhashan")
BAD = ("song", "geet", "gazal", "remix", "karaoke", "music", "bhajan_", "lofi", "dohori", "movie", "film")
AUD = (".mp3", ".ogg", ".flac", ".m4a", ".wav", ".opus")
_l = threading.Lock()
def log(*a): print(time.strftime("%H:%M:%S"), *a, flush=True)
def jget(u, t=40):
    for i in range(3):
        try: return json.load(urllib.request.urlopen(urllib.request.Request(u, headers=UA), timeout=t))
        except Exception: time.sleep(2 + i * 3)
    return {}

def discover(cat, rows=100):
    rows = rows * int(os.environ.get("ROWS_SCALE", "1"))
    ids = list(SEEDS.get(cat, []))
    for q in QUERIES[cat]:
        u = "https://archive.org/advancedsearch.php?q=" + urllib.parse.quote(q) + f"&fl[]=identifier&fl[]=title&fl[]=item_size&rows={rows}&sort[]=downloads+desc&output=json"
        for d in jget(u).get("response", {}).get("docs", []):
            t = (d.get("title") or "").lower() if isinstance(d.get("title"), str) else ""
            if any(b in (d["identifier"] + t).lower() for b in BAD) or int(d.get("item_size", 0) or 0) > 20e9: continue
            ids.append(d["identifier"])
    return list(dict.fromkeys(ids))

def to_wav(src, dst):
    r = subprocess.run(["ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-i", src, "-ac", "1", "-ar", "24000", "-c:a", "pcm_s16le", dst])
    return r.returncode == 0 and os.path.exists(dst) and os.path.getsize(dst) > 200_000

CAPS = {"storytelling": 90, "vedic_chant": 70, "news": 30, "conversational": 45}   # hours per category
def cat_hours(cat):
    d = f"{RAW}/{cat}"; return sum(os.path.getsize(f"{d}/{x}") for x in os.listdir(d)) / 48000 / 3600 if os.path.isdir(d) else 0
CAP_SCALE = float(os.environ.get("CAP_SCALE", "1")); ROWS_SCALE = int(os.environ.get("ROWS_SCALE", "1"))

def fetch_file(cat, ident, f, n):
    if cat_hours(cat) >= CAPS.get(cat, 50) * CAP_SCALE: return None, 0
    out = f"{RAW}/{cat}/ia_{ident[:60]}__{n}.wav"
    if os.path.exists(out): return out, 0
    tmp = f"{ROOT}/dl_tmp/ia_{abs(hash(ident + f['name']))}{os.path.splitext(f['name'])[1]}"
    url = f"https://archive.org/download/{ident}/{urllib.parse.quote(f['name'])}"
    r = subprocess.run(["curl", "-sL", "--retry", "3", "--max-time", "1500", "-H", "User-Agent: curl/8.5.0", "-o", tmp, url])
    ok = r.returncode == 0 and os.path.exists(tmp) and to_wav(tmp, out)
    if os.path.exists(tmp): os.remove(tmp)
    return (out, os.path.getsize(out)) if ok else (None, 0)

def hours():
    t = 0
    for c in os.listdir(RAW):
        for f in os.listdir(f"{RAW}/{c}"): t += os.path.getsize(f"{RAW}/{c}/{f}")
    return t / 48000 / 3600

def run(max_hours, workers, per_item_mb, cats):
    done = set(json.loads(l)["item"] for l in open(MAN)) if os.path.exists(MAN) else set()
    jobs = []
    OKLANG = ("nep", "nepali", "sanskrit", "san", "hin", "hindi")
    def plan(cat, ident):
        meta = jget(f"https://archive.org/metadata/{ident}"); md = meta.get("metadata", {})
        lang = str(md.get("language", "")).lower(); txt = (ident + " " + str(md.get("title", "")) + " " + str(md.get("subject", ""))).lower()
        seed = ident in SEEDS.get(cat, [])
        if cat == "vedic_chant" and any(k in txt or k in lang for k in LECTURE) and not seed: return []
        if cat == "vedic_chant" and any(k in lang for k in ("mal", "tam", "tel", "kan", "eng")) and "san" not in lang: return []
        if not seed:
            if cat == "vedic_chant" and not any(k in txt or k in lang for k in ("sanskrit", "san", "veda", "gita", "upanish", "mantra", "stotra", "sukta", "gayatri", "ramayan")): return []
            if cat != "vedic_chant" and not (any(k in lang for k in ("nep",)) or any(k in txt for k in ("nepali", "chisapani", "nepal", "katha", "samachar"))): return []
        fl = [f for f in meta.get("files", []) if f.get("name", "").lower().endswith(AUD) and 3e6 < int(f.get("size", 0) or 0) < 800e6]
        seen, sel, tot = set(), [], 0
        for f in sorted(fl, key=lambda x: x["name"]):
            stem = os.path.splitext(f["name"])[0]
            if stem in seen: continue
            seen.add(stem)
            if tot + int(f["size"]) > per_item_mb * 1e6: break
            sel.append(f); tot += int(f["size"])
        log(f"plan [{cat}] {ident}: {len(sel)} files {tot/1e6:.0f}MB")
        return [(cat, ident, f, n, md.get("licenseurl", ""), md.get("title", "")) for n, f in enumerate(sel)]
    with ThreadPoolExecutor(16) as pex:
        futs = [pex.submit(plan, cat, ident) for cat in cats for ident in discover(cat) if ident not in done]
        for fu in futs: jobs += fu.result()
    # interleave categories
    by = {c: [j for j in jobs if j[0] == c] for c in cats}; jobs = []
    while any(by.values()):
        for c in cats:
            if by[c]: jobs.append(by[c].pop(0))
    with ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(fetch_file, j[0], j[1], j[2], j[3]): j for j in jobs}
        for fu in as_completed(futs):
            cat, ident, f, n, lic, title = futs[fu]
            try: out, sz = fu.result()
            except Exception as e: out, sz = None, 0
            if out:
                with _l, open(MAN, "a") as m: m.write(json.dumps({"item": ident, "cat": cat, "file": os.path.basename(out), "src_name": f["name"], "license": lic, "title": title}, ensure_ascii=False) + "\n")
            h = hours(); log(f"{'ok' if out else 'x'} {cat} {ident} #{n} total={h:.1f}h")
            if h >= max_hours: log("max hours"); ex.shutdown(wait=False, cancel_futures=True); break

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--max-hours", type=float, default=120); ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--per-item-mb", type=int, default=450); ap.add_argument("--cats", default=",".join(QUERIES))
    a = ap.parse_args(); [os.makedirs(f"{RAW}/{c}", exist_ok=True) for c in QUERIES]; os.makedirs(f"{ROOT}/dl_tmp", exist_ok=True)
    run(a.max_hours, a.workers, a.per_item_mb, a.cats.split(","))
