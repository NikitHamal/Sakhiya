#!/usr/bin/env python3
"""Cloud ASR providers for NEPALI clips (owner rule: Sanskrit/Vedic never goes here - local stack only).
Keys come from files in /tmp (never committed): /tmp/assemblyai_key, /tmp/groq_key.
Every provider returns {"text","lang","extra"} or raises RateLimited(wait_s, daily) / Exception."""
import base64, json, os, subprocess, time
import requests

class RateLimited(Exception):
    def __init__(self, wait=60.0, daily=False, msg=""):
        super().__init__(msg or f"rate limited, wait {wait:.0f}s"); self.wait = wait; self.daily = daily

def key(name):
    return open(f"/tmp/{name}").read().strip()

def flac16(path):
    """16 kHz mono FLAC bytes (small upload, lossless)"""
    return subprocess.run(["ffmpeg", "-loglevel", "error", "-i", path, "-ar", "16000", "-ac", "1", "-f", "flac", "-"],
                          capture_output=True, timeout=60, check=True).stdout

def _parse_wait(h):
    """Groq: retry-after seconds (or '2m59.5s' style reset headers)"""
    try:
        if h.get("retry-after"): return float(h["retry-after"])
    except Exception: pass
    return 30.0

def groq_whisper(path, model="whisper-large-v3", language="ne"):
    r = requests.post("https://api.groq.com/openai/v1/audio/transcriptions",
                      headers={"Authorization": "Bearer " + key("groq_key")},
                      files={"file": ("clip.flac", flac16(path), "audio/flac")},
                      data={"model": model, "language": language, "response_format": "verbose_json", "temperature": "0"}, timeout=90)
    if r.status_code == 429:
        body = r.text.lower(); wait = _parse_wait(r.headers)
        daily = "per day" in body or "audio seconds per day" in body or wait > 900
        raise RateLimited(wait, daily, r.text[:200])
    if r.status_code >= 400: raise RuntimeError(f"groq {r.status_code}: {r.text[:200]}")
    j = r.json(); segs = j.get("segments") or []
    nsp = max([s.get("no_speech_prob", 0) for s in segs] or [0]); lp = min([s.get("avg_logprob", 0) for s in segs] or [0])
    return {"text": (j.get("text") or "").strip(), "lang": j.get("language", language), "extra": {"no_speech": round(nsp, 3), "avg_logprob": round(lp, 3)}}

def groq_qwen(path, model="qwen/qwen3.8-27b"):
    b64 = base64.b64encode(flac16(path)).decode()
    msg = [{"role": "user", "content": [
        {"type": "text", "text": "Transcribe this Nepali speech exactly as spoken in Devanagari script, with natural punctuation (, ? ! ।). Output ONLY the transcript."},
        {"type": "input_audio", "input_audio": {"data": b64, "format": "flac"}}]}]
    r = requests.post("https://api.groq.com/openai/v1/chat/completions", headers={"Authorization": "Bearer " + key("groq_key")},
                      json={"model": model, "messages": msg, "temperature": 0, "max_tokens": 400}, timeout=90)
    if r.status_code == 429:
        body = r.text.lower(); wait = _parse_wait(r.headers)
        raise RateLimited(wait, "per day" in body or wait > 900, r.text[:200])
    if r.status_code >= 400: raise RuntimeError(f"qwen {r.status_code}: {r.text[:200]}")
    return {"text": r.json()["choices"][0]["message"]["content"].strip(), "lang": "ne", "extra": {}}

AAI = "https://api.assemblyai.com/v2"
def assembly(path, poll=1.0, timeout=240):
    """AssemblyAI, language = AUTO (it also does Nepali). no speaker labels (extra cost, single-speaker clips)."""
    h = {"authorization": key("assemblyai_key")}
    up = requests.post(AAI + "/upload", headers=h, data=flac16(path), timeout=90)
    if up.status_code >= 400: raise RuntimeError(f"aai upload {up.status_code}: {up.text[:160]}")
    body = {"audio_url": up.json()["upload_url"], "speech_models": ["assemblyai-stt-3-7"], "language_detection": True,
            "punctuate": True, "format_text": True, "disfluencies": False}
    t = requests.post(AAI + "/transcript", headers=h, json=body, timeout=60)
    if t.status_code == 429: raise RateLimited(20, False, t.text[:200])
    if t.status_code >= 400: raise RuntimeError(f"aai submit {t.status_code}: {t.text[:200]}")
    tid = t.json()["id"]; t0 = time.time()
    while time.time() - t0 < timeout:
        j = requests.get(f"{AAI}/transcript/{tid}", headers=h, timeout=60).json()
        if j["status"] == "completed":
            return {"text": (j.get("text") or "").strip(), "lang": j.get("language_code", "auto"),
                    "extra": {"conf": j.get("confidence"), "lang_conf": j.get("language_confidence"), "dur": j.get("audio_duration"), "tid": tid}}
        if j["status"] == "error":
            e = str(j.get("error"))
            if "rate" in e.lower() or "concurren" in e.lower(): raise RateLimited(20, False, e)
            raise RuntimeError("aai: " + e[:200])
        time.sleep(poll)
    raise RuntimeError("aai timeout")

PROVIDERS = {
    "assemblyai": lambda p: assembly(p),
    "groq-whisper-v3": lambda p: groq_whisper(p, "whisper-large-v3"),
    "groq-whisper-v3-turbo": lambda p: groq_whisper(p, "whisper-large-v3-turbo"),
    "groq-qwen3.8-27b": lambda p: groq_qwen(p),
}

if __name__ == "__main__":
    import sys
    path = sys.argv[1]
    for n in (sys.argv[2:] or PROVIDERS):
        t0 = time.time()
        try: r = PROVIDERS[n](path); print(n, f"{time.time() - t0:.1f}s", json.dumps(r, ensure_ascii=False)[:400], flush=True)
        except Exception as e: print(n, "ERR", repr(e)[:300], flush=True)
