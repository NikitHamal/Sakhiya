#!/usr/bin/env python3
"""Resilient discovery + ingestion for expressive Nepali / Vedic speech.

Stage 1 of the v6 mining pipeline.  Resumable: state in /work/tts/mining_state.json.
  discover : yt-dlp flat search (many queries per category) -> candidate list (jsonl)
  download : best audio, multi-client fallback (android, ios, web_creator, tv_embedded, mweb, web),
             -> 24 kHz mono WAV in /work/tts/youtube_raw/<cat>/<id>.wav
  fallback : RSS podcasts / Internet Archive / HF mirrors when YouTube rate-limits the datacenter IP.
usage: mine_youtube_nepali.py [--per-query 25] [--max-hours 60] [--workers 6] [--cats a,b] [--loop]
"""
import argparse, json, os, re, subprocess, sys, time, glob, threading, urllib.request, urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = os.environ.get("TTS_ROOT", "/work/tts")
RAW = f"{ROOT}/youtube_raw"; STATE = f"{ROOT}/mining_state.json"; CAND = f"{ROOT}/candidates.jsonl"
CLIENTS = ["android", "ios", "web_creator", "tv_embedded", "mweb", "tv", "web"]
MIN_DUR, MAX_DUR = 60, 5400          # skip <1 min and >90 min videos (keeps shards balanced)
QUERIES = {
    "storytelling": ["नेपाली कथा", "नेपाली उपन्यास अडियो", "नेपाली audiobook उपन्यास", "नेपाली हाॅरर कथा", "नेपाली बालकथा",
                     "नेपाली कथा वाचन", "नेपाली लघुकथा सुन्नुहोस्", "nepali katha audio story", "नेपाली साहित्य श्रव्य", "नेपाली कविता वाचन",
                     "नेपाली रहस्य कथा", "नेपाली प्रेम कथा अडियो", "Nepali novel audiobook narration", "बीपी कोइराला कथा अडियो", "मुनामदन वाचन"],
    "news": ["नेपाली समाचार वाचन", "आजका मुख्य समाचार नेपाली", "नेपाली खबर बुलेटिन", "Nepali news bulletin", "रेडियो नेपाल समाचार",
             "कान्तिपुर समाचार", "नेपाल टेलिभिजन समाचार", "नेपाली समाचार सार"],
    "conversational": ["नेपाली पोडकास्ट", "Nepali podcast", "नेपाली कुराकानी", "नेपाली मोनोलग", "नेपाली प्रेरणादायी भाषण",
                       "नेपाली जीवन अनुभव कुराकानी", "नेपाली vlog कुरा", "नेपाली स्टोरीटेलिङ", "नेपाली मोटिभेसनल स्पिच", "रेडियो नेपाली कार्यक्रम"],
    "vedic_chant": ["गायत्री मन्त्र", "Gayatri mantra chanting", "ऋग्वेद पाठ", "Rigveda chanting", "उपनिषद् पाठ", "Upanishad chanting",
                    "भगवद्गीता पाठ संस्कृत", "Bhagavad Gita chanting Sanskrit", "विष्णु सहस्रनाम", "रुद्राष्टकम्", "शिव तांडव स्तोत्रम",
                    "महामृत्युञ्जय मन्त्र", "श्रीसूक्तम्", "पुरुष सूक्तम्", "वेद पाठ घनपाठ", "Vedic chanting swara", "यजुर्वेद पाठ",
                    "ॐ जप", "दुर्गा सप्तशती पाठ", "हनुमान चालीसा"],
}
RSS_FEEDS = {  # public Nepali podcast / audiobook feeds (best-effort fallback)
    "conversational": ["https://anchor.fm/s/ee8e6bf4/podcast/rss", "https://feeds.megaphone.fm/nepalipodcast"],
    "storytelling": ["https://librivox.org/rss/latest_releases"],
}
IA_QUERIES = {"storytelling": "language:(nep) AND mediatype:audio", "vedic_chant": "(rigveda OR upanishad OR gayatri) AND mediatype:audio",
              "news": "(nepali news) AND mediatype:audio"}
_lock = threading.Lock()

def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)

def load_state():
    if os.path.exists(STATE):
        try: return json.load(open(STATE))
        except Exception: pass
    return {"done": {}, "failed": {}, "queries_done": [], "hours": {}, "stage": "mine"}

def save_state(st):
    with _lock:
        tmp = STATE + ".tmp"; json.dump(st, open(tmp, "w"), ensure_ascii=False); os.replace(tmp, STATE)

def ytdlp(args, timeout=900):
    return subprocess.run([sys.executable, "-m", "yt_dlp", "--no-warnings", "--ignore-errors", "--no-cache-dir"] + args,
                          capture_output=True, text=True, timeout=timeout)

def discover(cats, per_query, st):
    seen = {json.loads(l)["id"] for l in open(CAND)} if os.path.exists(CAND) else set()
    with open(CAND, "a") as f:
        for cat in cats:
            for q in QUERIES[cat]:
                key = f"{cat}|{q}|{per_query}"
                if key in st["queries_done"]: continue
                for client in CLIENTS[:3] + ["web"]:
                    r = ytdlp(["--flat-playlist", "--dump-json", "--extractor-args", f"youtube:player_client={client}",
                               f"ytsearch{per_query}:{q}"], 240)
                    rows = [json.loads(l) for l in r.stdout.splitlines() if l.startswith("{")]
                    if rows: break
                n = 0
                for j in rows:
                    d = j.get("duration") or 0
                    if j["id"] in seen or not (MIN_DUR <= d <= MAX_DUR): continue
                    seen.add(j["id"]); n += 1
                    f.write(json.dumps({"id": j["id"], "cat": cat, "title": j.get("title"), "dur": d, "q": q,
                                        "ch": j.get("channel") or j.get("uploader"), "src": "youtube"}, ensure_ascii=False) + "\n")
                f.flush(); st["queries_done"].append(key); save_state(st)
                log(f"discover [{cat}] {q!r}: +{n} (rows {len(rows)})")

def to_wav24(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    r = subprocess.run(["ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-i", src, "-ac", "1", "-ar", "24000", "-c:a", "pcm_s16le", dst])
    return r.returncode == 0 and os.path.exists(dst)

def dl_youtube(c):
    out = f"{RAW}/{c['cat']}/{c['id']}.wav"
    if os.path.exists(out): return True, "cached"
    tmp = f"{ROOT}/dl_tmp/{c['id']}"; os.makedirs(os.path.dirname(tmp), exist_ok=True)
    last = ""
    for client in CLIENTS:
        for f in glob.glob(tmp + ".*"): os.remove(f)
        r = ytdlp(["-f", "bestaudio/best", "-o", tmp + ".%(ext)s", "--extractor-args", f"youtube:player_client={client}",
                   "--sleep-requests", "1", "--retries", "3", "--fragment-retries", "3", "--concurrent-fragments", "4",
                   "--socket-timeout", "20", f"https://www.youtube.com/watch?v={c['id']}"], 1800)
        fs = [f for f in glob.glob(tmp + ".*") if not f.endswith((".part", ".ytdl"))]
        if fs and to_wav24(fs[0], out):
            for f in glob.glob(tmp + ".*"): os.remove(f)
            return True, client
        last = (r.stderr or "")[-200:].replace("\n", " ")
        if "429" in last or "Sign in" in last or "bot" in last: time.sleep(3)
    return False, last

def fetch_rss(cat, st):
    import feedparser
    for url in RSS_FEEDS.get(cat, []):
        try: feed = feedparser.parse(url)
        except Exception: continue
        for e in feed.entries[:40]:
            enc = next((l.href for l in e.get("links", []) if l.get("type", "").startswith("audio")), None)
            vid = "rss_" + re.sub(r"\W+", "", e.get("id", enc or ""))[-24:]
            out = f"{RAW}/{cat}/{vid}.wav"
            if not enc or os.path.exists(out) or vid in st["done"]: continue
            tmp = f"{ROOT}/dl_tmp/{vid}.mp3"
            try:
                urllib.request.urlretrieve(enc, tmp)
                if to_wav24(tmp, out): st["done"][vid] = {"cat": cat, "src": "rss"}; log("rss ok", vid)
            except Exception as ex: log("rss fail", vid, ex)
            if os.path.exists(tmp): os.remove(tmp)

def fetch_ia(cat, st, limit=30):
    q = IA_QUERIES.get(cat)
    if not q: return
    url = "https://archive.org/advancedsearch.php?q=" + urllib.parse.quote(q) + f"&fl[]=identifier&rows={limit}&output=json"
    try: ids = [d["identifier"] for d in json.load(urllib.request.urlopen(url, timeout=30))["response"]["docs"]]
    except Exception as ex: log("ia search fail", ex); return
    for ident in ids:
        if ident in st["done"]: continue
        try:
            meta = json.load(urllib.request.urlopen(f"https://archive.org/metadata/{ident}", timeout=30))
            fl = [f for f in meta.get("files", []) if f["name"].lower().endswith((".mp3", ".ogg", ".flac")) and int(f.get("size", 0)) > 2e6][:3]
            for f in fl:
                tmp = f"{ROOT}/dl_tmp/ia_{ident}{os.path.splitext(f['name'])[1]}"
                urllib.request.urlretrieve(f"https://archive.org/download/{ident}/{urllib.parse.quote(f['name'])}", tmp)
                to_wav24(tmp, f"{RAW}/{cat}/ia_{ident}_{abs(hash(f['name'])) % 9999}.wav"); os.remove(tmp)
            st["done"][ident] = {"cat": cat, "src": "ia"}; log("ia ok", ident)
        except Exception as ex: log("ia fail", ident, str(ex)[:100])

def hours_by_cat():
    h = {}
    for c in os.listdir(RAW) if os.path.isdir(RAW) else []:
        h[c] = round(sum(os.path.getsize(f) for f in glob.glob(f"{RAW}/{c}/*.wav")) / 48000 / 3600, 2)
    return h

def download(cats, max_hours, workers, st):
    cands = [json.loads(l) for l in open(CAND)] if os.path.exists(CAND) else []
    todo = [c for c in cands if c["cat"] in cats and c["id"] not in st["done"] and st["failed"].get(c["id"], 0) < 2]
    # round-robin across categories so every style fills evenly
    by = {c: [x for x in todo if x["cat"] == c] for c in cats}; order = []
    while any(by.values()):
        for c in cats:
            if by[c]: order.append(by[c].pop(0))
    fails = 0
    with ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(dl_youtube, c): c for c in order}
        for fu in as_completed(futs):
            c = futs[fu]
            try: ok, info = fu.result()
            except Exception as e: ok, info = False, str(e)[:100]
            if ok: st["done"][c["id"]] = {"cat": c["cat"], "dur": c["dur"], "title": c["title"], "src": "youtube", "client": info}; fails = 0
            else:
                st["failed"][c["id"]] = st["failed"].get(c["id"], 0) + 1; fails += 1; log("FAIL", c["id"], info[:120])
            st["hours"] = hours_by_cat(); save_state(st); log(f"{'ok' if ok else 'x'} {c['cat']} {c['id']} hours={st['hours']}")
            if fails >= 12:   # datacenter rate-limit -> fall back to open mirrors
                log("YouTube throttled; running RSS/IA fallbacks")
                for cat in cats: fetch_rss(cat, st); fetch_ia(cat, st)
                fails = 0; save_state(st)
            if sum(st["hours"].values()) >= max_hours:
                log("max hours reached"); ex.shutdown(wait=False, cancel_futures=True); break

if __name__ == "__main__":
    import urllib.parse
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-query", type=int, default=25); ap.add_argument("--max-hours", type=float, default=60)
    ap.add_argument("--workers", type=int, default=6); ap.add_argument("--cats", default=",".join(QUERIES))
    ap.add_argument("--loop", action="store_true")
    a = ap.parse_args(); cats = a.cats.split(",")
    os.makedirs(RAW, exist_ok=True); os.makedirs(f"{ROOT}/dl_tmp", exist_ok=True)
    while True:
        st = load_state(); discover(cats, a.per_query, st); download(cats, a.max_hours, a.workers, st)
        for cat in cats: fetch_ia(cat, st)
        save_state(st); log("pass complete", hours_by_cat())
        if not a.loop: break
        a.per_query += 15; time.sleep(60)
