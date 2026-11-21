"""Gold re-transcription of EVERY mined clip with Gemini 3.5 Transcribe LIVE (Live API / WebSocket) on the owner's AI-Studio key.
Findings on this key (2026-11): the REST models are capped at ~20-25 requests/day each (gemini-3.5-transcribe 25/day, 3.5-flash 20/day, flash-lite ~13...), so a REST
call-per-clip cannot scale.  The Live API has no request cap: gemini-3.5-transcribe-live works when audio is streamed at REAL-TIME speed (0.2 s chunks every 0.2 s);
the quota is TOKENS/minute (transcribe-live ~20K, flash-live ~65K; audio ~31 tokens/s) so sessions-in-parallel are capped per model (4 + 8 by default; the key allows ~15 new sessions/min).  Fallback model gemini-3.1-flash-live-preview (also Live, also excellent Nepali).
Vedic purity gate = the script Gemini writes: Devanagari = Sanskrit/Hindi/Nepali recitation (kept), Tamil/Telugu/Malayalam/Kannada/Latin script = wrong language (thrown out).
Output (resumable, append-only): /work/tts/asr/gemini.jsonl  {id,path,style,text,lang,music,noise,model,t}   lang = nepali|devanagari|tamil|telugu|malayalam|kannada|english|other|none
Key from env GEMINI_API_KEY or /tmp/gemini_key (never committed).
run (N_TRANSCRIBE=8 N_FLASH=24): setsid nohup python scripts/gemini_transcribe.py > /work/tts/logs/gemini.log 2>&1 < /dev/null &"""
import argparse, asyncio, base64, glob, json, os, random, subprocess, time
import websockets
ROOT = os.environ.get("TTS_ROOT", "/work/tts"); OUT = os.environ.get("GEMINI_OUT", f"{ROOT}/asr/gemini.jsonl"); STATE = f"{ROOT}/asr/gemini_state.json"
KEY = os.environ.get("GEMINI_API_KEY") or open("/tmp/gemini_key").read().strip()
URL = f"wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent?key={KEY}"
MODELS = ["gemini-3.5-transcribe-live", "gemini-3.1-flash-live-preview"]
RANGES = {"devanagari": (0x900, 0x97f), "tamil": (0xb80, 0xbff), "telugu": (0xc00, 0xc7f), "kannada": (0xc80, 0xcff), "malayalam": (0xd00, 0xd7f), "bengali": (0x980, 0x9ff), "gujarati": (0xa80, 0xaff), "gurmukhi": (0xa00, 0xa7f)}
stats = {"ok": 0, "err": 0, "retry": 0}; used = {}; errs = {}
def log(*a): print(time.strftime("%H:%M:%S"), *a, flush=True)
def jl(p):
    try: return [json.loads(l) for l in open(p)]
    except Exception: return []
def pcm16(path):
    return subprocess.run(["ffmpeg", "-loglevel", "error", "-i", path, "-ar", "16000", "-ac", "1", "-f", "s16le", "-"], capture_output=True, timeout=60).stdout
def script(t):
    """-> (language label from the script Gemini wrote, Devanagari share)"""
    a = [ord(c) for c in t if c.isalpha()]
    if not a: return "none", 0.0
    cnt = {k: sum(lo <= c <= hi for c in a) for k, (lo, hi) in RANGES.items()}; lat = sum(c < 0x250 for c in a); n = len(a)
    best = max(cnt, key=cnt.get)
    lang = "english" if lat / n > .6 else best if cnt[best] / n > .6 else "other"
    return lang, cnt["devanagari"] / n
_gate = {"next": 0.0}; SPACING = float(os.environ.get("SESSION_SPACING", 4.4))      # seconds between new sessions, all models together (~13.6/min)
async def _slot():
    now = time.time(); t = max(now, _gate["next"]); _gate["next"] = t + SPACING
    if t > now: await asyncio.sleep(t - now)
class T(str):
    """transcript text that also carries the raw fragments the model emitted (kept in gemini.jsonl as `frags` so joining can be re-done without re-listening)"""
    frags = ()
def join_frags(fr):
    """Gemini streams sub-word pieces: a leading space marks a new word.  Join RAW (no strip, no extra spaces) then tidy; the old ' '.join(strip) cut words like 'गायनमा' into 'गा यन मा'."""
    import re
    return re.sub(r"\s+", " ", "".join(fr)).strip()
async def live(pcm, model, langs):
    await _slot()
    """stream at real-time speed, return the transcript (finals joined; last interim if there was no final)"""
    mod = "TEXT" if "transcribe" in model else "AUDIO"
    cfg = {"languageCodes": langs}
    async with websockets.connect(URL, max_size=2 ** 24, open_timeout=40, ping_interval=None) as ws:
        await ws.send(json.dumps({"setup": {"model": "models/" + model, "generationConfig": {"responseModalities": [mod]}, "inputAudioTranscription": cfg}}))
        m = json.loads(await asyncio.wait_for(ws.recv(), 40))
        if "setupComplete" not in m: raise RuntimeError(str(m)[:150])
        fin, interim, st = [], [""], {"t": time.time()}
        async def reader():
            try:
                async for raw in ws:
                    sc = (json.loads(raw).get("serverContent") or {}); a = sc.get("inputTranscription"); b = sc.get("interimInputTranscription")
                    if a and a.get("text"): fin.append(a["text"]); st["t"] = time.time()
                    if b and b.get("text"): interim[0] = b["text"]; st["t"] = time.time()
            except Exception: pass
        rt = asyncio.create_task(reader()); step = 6400
        for o in range(0, len(pcm), step):
            await ws.send(json.dumps({"realtimeInput": {"audio": {"data": base64.b64encode(pcm[o:o + step]).decode(), "mimeType": "audio/pcm;rate=16000"}}}))
            await asyncio.sleep(0.19)
        await ws.send(json.dumps({"realtimeInput": {"audioStreamEnd": True}})); st["t"] = time.time(); t0 = time.time()
        while time.time() - st["t"] < 3.0 and time.time() - t0 < 25 and not rt.done(): await asyncio.sleep(0.25)
        rt.cancel()
    out = T(join_frags(fin) or join_frags([interim[0]])); out.frags = tuple(fin) or (interim[0],); return out
POOLS = {"gemini-3.5-transcribe-live": int(os.environ.get("N_TRANSCRIBE", 4)), "gemini-3.1-flash-live-preview": int(os.environ.get("N_FLASH", 8))}
cool = {}          # model -> unix time until which it is resting (quota window: ~20K tokens/min transcribe-live, ~65K flash-live; real-time audio = ~31 tokens/s)
async def one(item, fo, primary):
    """try the primary model, then the other one; a model that answers 'quota' rests for 45 s so the minute-window refills"""
    cid, path, style = item; ved = style == "vedic_chant"; err = ""; other = [m for m in POOLS if m != primary][0]
    for k in range(10):
        model = primary if k % 2 == 0 else other
        if cool.get(model, 0) > time.time():
            model = other if cool.get(other, 0) <= time.time() else model
            if cool.get(model, 0) > time.time(): await asyncio.sleep(min(20, cool[model] - time.time()))
        try:
            pcm = pcm16(path)
            if len(pcm) < 3200: raise ValueError("empty audio")
            txt = await live(pcm, model, [] if ved else ["ne-NP"]); lang, d = script(txt)
            if not ved and lang == "devanagari": lang = "nepali"
            rec = {"id": cid, "path": path, "style": style, "text": txt, "lang": lang, "deva": round(d, 3), "frags": list(getattr(txt, "frags", ())), "model": model, "t": int(time.time())}
            fo.write(json.dumps(rec, ensure_ascii=False) + "\n"); fo.flush(); stats["ok"] += 1; used[model] = used.get(model, 0) + 1; return
        except Exception as e:
            stats["retry"] += 1; err = repr(e)[:150]; errs[err[-60:]] = errs.get(err[-60:], 0) + 1
            if "quota" in err: _gate["next"] = max(_gate["next"], time.time() + 20 + random.random() * 10)
            await asyncio.sleep(2 + random.random() * 3)
    stats["err"] += 1; log("ERR", cid, err)

# ---- v2 (session 60): ONE websocket serves many clips in a row -> the "new sessions per minute" limit stops being the bottleneck;
# only the tokens/minute window (audio ~31 tok/s, silence counts too) limits speed.  Each clip is followed by silence until the model stops writing.
PER_SESSION = int(os.environ.get("PER_SESSION", 30)); QUIET = float(os.environ.get("QUIET", 2.4))
async def session_worker(model, queue, fo, done_ids):
    ved_langs = lambda style: [] if style == "vedic_chant" else ["ne-NP"]
    mod = "TEXT" if "transcribe" in model else "AUDIO"
    while True:
        if cool.get(model, 0) > time.time(): await asyncio.sleep(min(10, cool[model] - time.time())); continue
        if queue.empty(): return
        item = None; langs = None; n = 0
        try:
            first = queue.get_nowait()
        except asyncio.QueueEmpty:
            return
        item = first; style0 = item[2]
        try:
            async with websockets.connect(URL, max_size=2 ** 24, open_timeout=40, ping_interval=None) as ws:
                await ws.send(json.dumps({"setup": {"model": "models/" + model, "generationConfig": {"responseModalities": [mod]}, "inputAudioTranscription": {"languageCodes": ["ne-NP"] if style0 != "vedic_chant" else []}}}))
                m = json.loads(await asyncio.wait_for(ws.recv(), 40))
                if "setupComplete" not in m: raise RuntimeError(str(m)[:150])
                buf = {"fin": [], "interim": "", "t": time.time()}
                async def reader():
                    async for raw in ws:
                        sc = (json.loads(raw).get("serverContent") or {}); a = sc.get("inputTranscription"); b = sc.get("interimInputTranscription")
                        if a and a.get("text"): buf["fin"].append(a["text"].strip()); buf["t"] = time.time()
                        if b and b.get("text"): buf["interim"] = b["text"].strip(); buf["t"] = time.time()
                rt = asyncio.create_task(reader())
                silence = base64.b64encode(b"\x00" * 6400).decode()
                while item is not None:
                    cid, path, style = item
                    try:
                        pcm = pcm16(path)
                        if len(pcm) < 3200: raise ValueError("empty audio")
                    except Exception as e:
                        stats["err"] += 1; log("BAD AUDIO", cid, repr(e)[:80]); pcm = None
                    if pcm is not None:
                        buf["fin"], buf["interim"] = [], ""
                        for o in range(0, len(pcm), 6400):
                            await ws.send(json.dumps({"realtimeInput": {"audio": {"data": base64.b64encode(pcm[o:o + 6400]).decode(), "mimeType": "audio/pcm;rate=16000"}}})); await asyncio.sleep(0.19)
                            if rt.done(): raise RuntimeError("reader ended: " + repr(rt.exception())[:120])
                        buf["t"] = time.time(); t0 = time.time(); sent = 0
                        while (time.time() - buf["t"] < QUIET or sent < 6 or not (buf["fin"] or buf["interim"])) and sent < 45:   # up to ~6 s of silence, at least 0.8 s
                            await ws.send(json.dumps({"realtimeInput": {"audio": {"data": silence, "mimeType": "audio/pcm;rate=16000"}}})); await asyncio.sleep(0.19); sent += 1
                            if rt.done(): raise RuntimeError("reader ended: " + repr(rt.exception())[:120])
                        txt = " ".join(x for x in buf["fin"] if x).strip() or buf["interim"]
                        lang, d = script(txt)
                        if style != "vedic_chant" and lang == "devanagari": lang = "nepali"
                        fo.write(json.dumps({"id": cid, "path": path, "style": style, "text": txt, "lang": lang, "deva": round(d, 3), "model": model, "t": int(time.time())}, ensure_ascii=False) + "\n"); fo.flush()
                        stats["ok"] += 1; used[model] = used.get(model, 0) + 1
                    n += 1; item = None
                    if n >= PER_SESSION: break
                    try: item = queue.get_nowait()
                    except asyncio.QueueEmpty: break
                rt.cancel()
        except Exception as e:
            err = repr(e)[:150]; stats["retry"] += 1; errs[err[-60:]] = errs.get(err[-60:], 0) + 1
            if item is not None: queue.put_nowait(item)       # clip not finished: give it back
            cool[model] = time.time() + (30 + random.random() * 20 if "quota" in err or "1011" in err else 4); await asyncio.sleep(2 + random.random() * 3)
def universe():
    """priority: kept > heard > cleaned > Vedic with reference text > everything else, styles interleaved."""
    seen, out = set(), []
    def add(path, style):
        if not path or not os.path.exists(path): return
        cid = os.path.basename(path)[:-4]
        if cid not in seen: seen.add(cid); out.append((cid, path, style))
    for m in jl(f"{ROOT}/dataset/meta.jsonl"): add(next(iter(glob.glob(f"{ROOT}/segments/*/{m['id']}.wav")), ""), m.get("style", ""))
    try:
        for p in json.load(open(f"{ROOT}/vedic/ref_text.json")): add(p, "vedic_chant")      # owner pain first: Vedic purity
    except Exception: pass
    for r in jl(f"{ROOT}/asr/asr.jsonl"): add(r["path"], r["cat"])
    for r in jl(f"{ROOT}/gate/passed.jsonl"): add(r["path"], r["cat"])
    try:
        for p in json.load(open(f"{ROOT}/vedic/ref_text.json")): add(p, "vedic_chant")
    except Exception: pass
    ds = ("vedic_chant", "storytelling", "news", "conversational"); pools = []
    for d in ds:
        rest = sorted(glob.glob(f"{ROOT}/segments/{d}/*.wav")); random.Random(7).shuffle(rest); pools.append(rest)
    for i in range(max(map(len, pools))):
        for pi, d in enumerate(ds):
            if i < len(pools[pi]): add(pools[pi][i], d)
    return out
async def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=100); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--style", default=""); a = ap.parse_args()
    # OWNER RULE (session 64): Gemini is used ONLY on Nepali voices.  Sanskrit/Vedic stays on the local stack (verse-matched reference text + local purity gate).
    done = {r["id"] for r in jl(OUT) if r.get("text") and r.get("style") != "vedic_chant"}
    todo = [u for u in universe() if u[0] not in done and u[2] != "vedic_chant" and "/vedic_chant/" not in u[1] and (not a.style or u[2] == a.style)]
    if a.limit: todo = todo[:a.limit]
    log(len(todo), "clips to transcribe,", len(done), "already done", POOLS); fo = open(OUT, "a", encoding="utf8"); t0 = time.time()
    def save(fin=False): json.dump({"ok": stats["ok"] + len(done), "err": stats["err"], "total": len(todo) + len(done), "ts": time.time(), "done": fin, "models": used, "parked": {}}, open(STATE, "w"))
    async def beat():
        while True: await asyncio.sleep(30); log("progress", stats, used, dict(sorted(errs.items(), key=lambda x: -x[1])[:3]), f"{stats['ok'] / max(1, time.time() - t0):.2f} clips/s"); save()
    bt = asyncio.create_task(beat()); queue = asyncio.Queue()
    if os.environ.get("SESSION_REUSE") == "1":
        # EXPERIMENTAL, measured worse (session 60): the model carries text across clips / hallucinates on long silence (mean similarity 0.78 vs per-clip sessions). Off by default.
        for x in todo: queue.put_nowait(x)
        ws_ = [session_worker(m, queue, fo, done) for m, n in POOLS.items() for _ in range(n)]
    else:
        it = iter(todo); lk = asyncio.Lock()
        async def worker(primary):
            while True:
                async with lk: x = next(it, None)
                if x is None: return
                await one(x, fo, primary)
        ws_ = [worker(m) for m, n in POOLS.items() for _ in range(n)]
    await asyncio.gather(*ws_); bt.cancel(); log("DONE", stats); save(True)
if __name__ == "__main__": asyncio.run(main())
