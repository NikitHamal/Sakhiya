#!/usr/bin/env python3
"""Stage 2: GPU vocal isolation (htdemucs vocals stem) + Silero-VAD breath-group segmentation.
raw 24k wav -> /work/tts/clean/<cat>/<name>.wav (dry vocals, 24k) -> /work/tts/segments/<cat>/<name>_<i>.wav (2.5-14 s)
Segments keep 150 ms real trailing audio and all internal comma/breath pauses (cuts only at VAD gaps >= GAP_CUT).
usage: yt_separate_segment.py [--shard 0/1] [--no-demucs-cats news] ; resumable (skips existing .done markers)."""
import argparse, glob, json, os, sys, time
import numpy as np, soundfile as sf, torch, torchaudio.functional as AF

ROOT = os.environ.get("TTS_ROOT", "/work/tts"); RAW = f"{ROOT}/youtube_raw"; CLEAN = f"{ROOT}/clean"; SEG = f"{ROOT}/segments"
MIN_S, MAX_S, TAIL, LEAD, GAP_CUT, SR = 2.5, 14.0, 0.15, 0.08, 0.9, 24000
def log(*a): print(time.strftime("%H:%M:%S"), *a, flush=True)

def load_demucs(dev):
    from demucs.pretrained import get_model
    m = get_model("htdemucs"); m.to(dev).eval(); return m

@torch.no_grad()
def isolate(model, w24, dev):
    """w24: 1-D float tensor @24k -> dry vocals 1-D @24k (processed in 10-min chunks, 5 s overlap crossfade-free via apply_model split)."""
    from demucs.apply import apply_model
    out = []; step = 600 * SR
    for s in range(0, len(w24), step):
        x = AF.resample(w24[s:s + step].to(dev), SR, 44100)[None].repeat(2, 1)[None]          # (1,2,T)
        ref = x.mean(1, keepdim=True); mu, sd = ref.mean(), ref.std() + 1e-8
        y = apply_model(model, (x - mu) / sd, device=dev, shifts=0, split=True, overlap=0.1, progress=False)[0]
        v = (y[model.sources.index("vocals")] * sd + mu).mean(0)
        out.append(AF.resample(v, 44100, SR).cpu())
    return torch.cat(out)

def segment(w24, vad, get_ts):
    w16 = AF.resample(w24, SR, 16000)
    ts = get_ts(w16, vad, sampling_rate=16000, min_silence_duration_ms=220, speech_pad_ms=50, min_speech_duration_ms=200,
                max_speech_duration_s=MAX_S - 1.0, return_seconds=True)
    if not ts: return []
    groups, cur = [], [ts[0]["start"], ts[0]["end"]]
    for t in ts[1:]:
        gap = t["start"] - cur[1]
        if gap < GAP_CUT and t["end"] - cur[0] + TAIL + LEAD <= MAX_S: cur[1] = t["end"]
        else: groups.append(cur); cur = [t["start"], t["end"]]
    groups.append(cur); dur = len(w24) / SR; res = []
    for i, (s, e) in enumerate(groups):
        nxt = groups[i + 1][0] if i + 1 < len(groups) else dur
        prv = groups[i - 1][1] if i else 0.0
        a = max(s - min(LEAD, max(0, (s - prv) / 2)), 0.0)
        b = min(e + min(TAIL, max(0, (nxt - e) * 0.9)), dur)
        if MIN_S <= b - a <= MAX_S and (nxt - e) >= 0.05: res.append((a, b, round(e - s, 3)))
    return res

def fade(x, n=72):
    x = x.copy(); r = np.linspace(0, 1, n, dtype=np.float32); x[:n] *= r; x[-n:] *= r[::-1]; return x

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--shard", default="0/1"); ap.add_argument("--skip-demucs-cats", default="")
    a = ap.parse_args(); si, sn = map(int, a.shard.split("/")); dev = "cuda"
    from silero_vad import load_silero_vad, get_speech_timestamps
    vad = load_silero_vad(); dm = load_demucs(dev); skip = a.skip_demucs_cats.split(",")
    PRI = {"vedic_chant": 0, "storytelling": 1, "news": 2, "conversational": 3}
    files = sorted(glob.glob(f"{RAW}/*/*.wav"), key=lambda f: (PRI.get(f.split("/")[-2], 9), os.path.getsize(f)))
    for f in files:
        cat, name = f.split("/")[-2], os.path.basename(f)[:-4]
        done = f"{SEG}/{cat}/{name}.done"
        if os.path.exists(done): continue
        os.makedirs(f"{SEG}/{cat}", exist_ok=True); os.makedirs(f"{CLEAN}/{cat}", exist_ok=True)
        try: os.close(os.open(f"{SEG}/{cat}/{name}.claim", os.O_CREAT | os.O_EXCL))   # atomic claim: workers never double-process
        except FileExistsError: continue
        try:
            t0 = time.time(); w, sr = sf.read(f, dtype="float32"); w = torch.from_numpy(w if w.ndim == 1 else w.mean(1))
            if len(w) < 5 * SR: open(done, "w").write("short"); continue
            v = w if cat in skip else isolate(dm, w, dev)
            pk = float(v.abs().max()) + 1e-6; v = v / pk * 0.95
            sf.write(f"{CLEAN}/{cat}/{name}.wav", v.numpy(), SR, subtype="PCM_16")
            segs = segment(v, vad, get_speech_timestamps); meta = []
            x = v.numpy()
            for i, (s, e, sp) in enumerate(segs):
                p = f"{SEG}/{cat}/{name}_{i:04d}.wav"; sf.write(p, fade(x[int(s * SR):int(e * SR)]), SR, subtype="PCM_16")
                meta.append({"path": p, "cat": cat, "src": name, "t0": round(s, 2), "dur": round(e - s, 2), "speech": sp})
            with open(f"{SEG}/segments.{os.getpid()}.jsonl", "a") as j:
                for m in meta: j.write(json.dumps(m, ensure_ascii=False) + "\n")
            open(done, "w").write(str(len(meta)))
            log(f"{cat}/{name}: {len(w)/SR/60:.1f}min -> {len(meta)} segs in {time.time()-t0:.0f}s")
        except Exception as ex:
            log("ERR", f, repr(ex)[:200]); torch.cuda.empty_cache()
    log("SHARD_DONE", a.shard)

if __name__ == "__main__": main()
