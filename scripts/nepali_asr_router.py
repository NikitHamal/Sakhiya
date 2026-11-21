#!/usr/bin/env python3
"""Parallel Nepali transcription router (owner rules, session 65):
  * Nepali ONLY (never vedic_chant: Sanskrit stays on the local stack)
  * AssemblyAI (language auto, $100 credit, budget-guarded) + Groq Whisper large-v3 then large-v3-turbo (until the free quota is used)
    run IN PARALLEL with the Gemini Live job, each clip belongs to exactly ONE engine (asr_claims.py).
  * Qwen3.8-27B on Groq is text+vision only (no audio) -> not used as a transcriber.
output (append-only): /work/tts/asr/cloud_asr.jsonl {id,path,style,text,lang,provider,extra,t};  state: /work/tts/asr/cloud_state.json
run: setsid nohup python3 scripts/nepali_asr_router.py > /work/tts/logs/cloud_asr.log 2>&1 < /dev/null &"""
import glob, json, os, random, sys, threading, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import asr_claims as C, asr_providers as P
ROOT = C.ROOT; OUT = f"{ROOT}/asr/cloud_asr.jsonl"; STATE = f"{ROOT}/asr/cloud_state.json"
AAI_THREADS = int(os.environ.get("AAI_THREADS", 12)); GROQ_THREADS = int(os.environ.get("GROQ_THREADS", 3))
AAI_RATE = float(os.environ.get("AAI_USD_PER_HOUR", 0.40)); AAI_BUDGET = float(os.environ.get("AAI_BUDGET_USD", 90))   # conservative: stop before the $100 credit is gone
lock = threading.Lock(); fo = None
S = {"ok": {}, "err": {}, "limit": {}, "cool": {}, "aai_usd": 0.0, "aai_sec": 0.0, "groq_sec": {}, "t0": time.time(), "last": {}}
def log(*a): print(time.strftime("%H:%M:%S"), *a, flush=True)
def jl(p):
    try: return [json.loads(l) for l in open(p)]
    except Exception: return []

def load_universe():
    import importlib.util
    sp = importlib.util.spec_from_file_location("gem", os.path.join(os.path.dirname(os.path.abspath(__file__)), "gemini_transcribe.py"))
    g = importlib.util.module_from_spec(sp); sp.loader.exec_module(g)          # reuse its priority order (kept > heard > cleaned > rest)
    return [u for u in g.universe() if u[2] != "vedic_chant" and "/vedic_chant/" not in u[1]]

def deva(t):
    a = [c for c in t if c.isalpha()]; return sum("\u0900" <= c <= "\u097f" for c in a) / max(1, len(a))

def write(item, prov, r):
    cid, path, style = item
    rec = {"id": cid, "path": path, "style": style, "text": r["text"], "lang": "nepali" if deva(r["text"]) > .6 else r["lang"], "deva": round(deva(r["text"]), 3),
           "provider": prov, "extra": r.get("extra", {}), "t": int(time.time())}
    with lock:
        fo.write(json.dumps(rec, ensure_ascii=False) + "\n"); fo.flush(); S["ok"][prov] = S["ok"].get(prov, 0) + 1; S["last"][prov] = int(time.time())
    C.finish(cid, prov)

def aai_worker(q):
    while True:
        if S["aai_usd"] >= AAI_BUDGET: log("AssemblyAI budget reached", S["aai_usd"]); return
        with lock: item = next(q, None)
        if item is None: return
        if not C.claim(item[0], "assemblyai"): continue
        for k in range(6):
            try:
                r = P.assembly(item[1]); dur = float((r["extra"] or {}).get("dur") or 8)
                with lock: S["aai_sec"] += dur; S["aai_usd"] = S["aai_sec"] / 3600 * AAI_RATE
                if r["text"]: write(item, "assemblyai", r)
                else: C.release(item[0], "assemblyai")
                break
            except P.RateLimited as e: S["limit"]["assemblyai"] = S["limit"].get("assemblyai", 0) + 1; time.sleep(e.wait + random.random() * 5)
            except Exception as e:
                S["err"]["assemblyai"] = S["err"].get("assemblyai", 0) + 1; log("aai err", repr(e)[:140]); time.sleep(3 + k * 3)
        else: C.release(item[0], "assemblyai")

GROQ_MODELS = ["whisper-large-v3", "whisper-large-v3-turbo"]       # one after another: v3 first, turbo when v3's quota is gone
def groq_worker(q):
    while True:
        now = time.time(); ready = [m for m in GROQ_MODELS if S["cool"].get(m, 0) <= now]
        if not ready:
            time.sleep(min(30, max(2, min(S["cool"].values()) - now))); continue
        with lock: item = next(q, None)
        if item is None: return
        model = ready[0]; prov = "groq-" + model.replace("whisper-large-", "whisper-")
        if not C.claim(item[0], prov): continue
        done = False
        for k in range(4):
            try:
                r = P.groq_whisper(item[1], model)
                with lock: S["groq_sec"][model] = S["groq_sec"].get(model, 0) + 8
                if r["text"]: write(item, prov, r)
                else: C.release(item[0], prov)
                done = True; break
            except P.RateLimited as e:
                S["limit"][model] = S["limit"].get(model, 0) + 1
                S["cool"][model] = time.time() + (e.wait + 5 if not e.daily else max(e.wait, 3600) + 5)
                log("groq", model, "limited, resting", int(S["cool"][model] - time.time()), "s", "(daily)" if e.daily else "")
                nxt = [m for m in GROQ_MODELS if S["cool"].get(m, 0) <= time.time()]
                if not nxt: break
                new = "groq-" + nxt[0].replace("whisper-large-", "whisper-"); C.retarget(item[0], prov, new); model, prov = nxt[0], new
            except Exception as e:
                S["err"][model] = S["err"].get(model, 0) + 1; log("groq err", repr(e)[:140]); time.sleep(2 + k * 2)
        if not done: C.release(item[0], prov)

def beat(total):
    while True:
        time.sleep(30); n = sum(S["ok"].values()); dt = time.time() - S["t0"]
        log("progress", S["ok"], "limits", S["limit"], "errs", S["err"], f"aai ${S['aai_usd']:.2f}", f"{n / dt:.2f} clips/s")
        json.dump({"ok": S["ok"], "err": S["err"], "limit": S["limit"], "aai_usd": round(S["aai_usd"], 2), "aai_budget": AAI_BUDGET, "aai_hours": round(S["aai_sec"] / 3600, 2),
                   "cool": {m: max(0, int(t - time.time())) for m, t in S["cool"].items()}, "last": S["last"], "claims": C.stats(), "total": total, "ts": time.time(),
                   "rate": round(n / dt, 3)}, open(STATE, "w"))
        C.reclaim_stale()

def main():
    global fo
    log("claims seeded from Gemini:", C.seed([("asr/gemini.jsonl", "gemini"), ("asr/cloud_asr.jsonl", "cloud")]))
    uni = load_universe(); taken = C.claimed_ids(); todo = [u for u in uni if u[0] not in taken]
    done = {r["id"] for r in jl(OUT) if r.get("text")}; todo = [u for u in todo if u[0] not in done]
    log(len(uni), "Nepali clips in universe,", len(todo), "to go"); fo = open(OUT, "a", encoding="utf8")
    q = iter(todo); threading.Thread(target=beat, args=(len(uni),), daemon=True).start()
    ts = [threading.Thread(target=aai_worker, args=(q,), daemon=True) for _ in range(AAI_THREADS)]
    ts += [threading.Thread(target=groq_worker, args=(q,), daemon=True) for _ in range(GROQ_THREADS)]
    for t in ts: t.start()
    for t in ts: t.join()
    log("DONE", S["ok"])
if __name__ == "__main__": main()
