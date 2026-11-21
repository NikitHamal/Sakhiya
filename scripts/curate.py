"""Curate the v2 acoustic set (SLR43 CC-BY-SA + FLEURS ne_np CC-BY + CommonVoice ne-NP CC0).
Per-clip: silence trim, SNR, clipping, pitch stability, bandwidth flag; then ASR phoneme-CER gate;
then speaker-level pruning. Output: /work/tts/feats/v2.pt   (python scripts/curate.py)"""
import os, re, sys, glob, json, random, collections, unicodedata
from concurrent.futures import ProcessPoolExecutor
import numpy as np, torch, librosa
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sakhiya import audio, g2p, sanskrit, symbols as S

D, OUT = "/work/tts/data", "/work/tts/feats"
OK = re.compile(r"^[\u0900-\u097F\s,.?!।\-–;:'\"()]+$")
IDX = {}


def index(root):
    if root not in IDX:
        IDX[root] = {os.path.basename(p): p for p in glob.glob(root + "/**/*.*", recursive=True) if p.endswith((".wav", ".mp3", ".flac"))}
    return IDX[root]


def items():
    for ln in (open(f"{D}/slr43/ne_np_female/line_index.tsv", encoding="utf8") if os.path.exists(f"{D}/slr43/ne_np_female/line_index.tsv") else []):
        i, t = ln.rstrip("\n").split("\t")[:2]
        yield dict(id=i, spk="s" + i.split("_")[1], text=t.strip(), path=f"{D}/slr43/ne_np_female/wavs/{i}.wav", src="slr43", lic="cc-by-sa")
    import csv
    for sp in ("train", "dev", "test"):
        f = f"{D}/fleurs/data/ne_np/{sp}.tsv"
        if not os.path.exists(f): continue
        ix = index(f"{D}/fleurs/{sp}")
        for p in csv.reader(open(f, encoding="utf8"), delimiter="\t", quoting=csv.QUOTE_NONE):
            if len(p) >= 7 and p[1] in ix:
                yield dict(id="fl_" + p[1], spk="fl_" + p[6].lower()[:1], text=p[2].strip(), path=ix[p[1]], src="fleurs", lic="cc-by-4.0")
    ixc = index(f"{D}/cv")
    for sp in ("train", "dev", "test", "other", "validated"):
        f = f"{D}/cv/transcript/ne-NP/{sp}.tsv"
        if not os.path.exists(f): continue
        for p in csv.DictReader(open(f, encoding="utf8"), delimiter="\t", quoting=csv.QUOTE_NONE):
            b = os.path.basename(p.get("path", ""))
            if b in ixc and int(p.get("up_votes") or 0) >= 0 and sp != "other":
                yield dict(id="cv_" + b, spk="cv_" + p["client_id"][:8], text=p["sentence"].strip(), path=ixc[b], src="cv", lic="cc0")


def proc(it):
    try:
        torch.set_num_threads(1)
        if not OK.match(it["text"]): return None
        import soundfile as sf
        raw, sr = sf.read(it["path"], dtype="float32")
        raw = raw.mean(1) if raw.ndim > 1 else raw
        if sr != audio.SR: raw = librosa.resample(raw, orig_sr=sr, target_sr=audio.SR, res_type="soxr_hq")
        db = 20 * np.log10(librosa.feature.rms(y=raw, frame_length=1024, hop_length=256)[0] + 1e-6)
        snr = float(min(np.percentile(db, 92) - np.percentile(db, 8), 60))
        clip = float((np.abs(raw) > 0.99).mean())
        x = audio.load_wav(it["path"], trim_db=35 if it["src"] != "slr43" else 40)
        if not 0.8 < len(x) / audio.SR < it.get("maxdur", 15): return None
        S_ = np.abs(librosa.stft(x, n_fft=1024, hop_length=256)) ** 2
        fr = librosa.fft_frequencies(sr=audio.SR, n_fft=1024)
        hf = 10 * np.log10(S_[(fr > 8200) & (fr < 10500)].mean() + 1e-12) - 10 * np.log10(S_[(fr > 1000) & (fr < 4000)].mean() + 1e-12)
        mel = audio.mel_spec(torch.from_numpy(x))[0].numpy()
        f0 = audio.f0_track(x, mel.shape[1]); v = f0 > 0
        lf = np.log(f0[v] + 1e-6); jump = float((np.abs(np.diff(lf)) > 0.4).mean()) if v.sum() > 5 else 1.0
        ph = sanskrit.text_to_phones(it["text"]) if it.get("lang") == "sa" else g2p.text_to_phones(it["text"])
        return dict(it, ph=S.encode(ph), phs=" ".join(ph), mel=mel.astype(np.float16), f0=f0.astype(np.float16), wav=(x * 32767).astype(np.int16),
                    snr=snr, clip=clip, hf=float(hf), wide=bool(hf > -48), vr=float(v.mean()), jump=jump, f0m=float(np.median(lf)) if v.sum() > 5 else 0.0,
                    mmel=mel[:, v[:mel.shape[1]]].mean(1) if v.sum() > 5 else mel.mean(1))
    except Exception as e:
        return None


def asr_cer(res):
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
    import jiwer
    rid = "spktsagar/wav2vec2-large-xls-r-300m-nepali-openslr"
    pr, m = Wav2Vec2Processor.from_pretrained(rid), Wav2Vec2ForCTC.from_pretrained(rid).cuda().eval().half()
    order = sorted(range(len(res)), key=lambda i: len(res[i]["wav"]))
    for b in range(0, len(order), 32):
        rs = [res[i] for i in order[b:b + 32]]
        ws = [librosa.resample(r["wav"].astype(np.float32) / 32767, orig_sr=audio.SR, target_sr=16000) for r in rs]
        inp = pr(ws, sampling_rate=16000, return_tensors="pt", padding=True)
        with torch.no_grad(): lg = m(inp.input_values.cuda().half()).logits.argmax(-1)
        for r, h in zip(rs, pr.batch_decode(lg)):
            ref = r["phs"].replace("|", "").split(); hyp = g2p.text_to_phones(h)
            r["cer"] = jiwer.cer(r["phs"].replace(" ", ""), " ".join(hyp).replace(" ", "").replace("|", "")) if h.strip() else 1.0
        if b % 1600 == 0: print("asr", b, len(res), flush=True)


if __name__ == "__main__":
    its = list(items()); print("candidates", collections.Counter(i["src"] for i in its), flush=True)
    with ProcessPoolExecutor(18) as ex: res = [r for r in ex.map(proc, its, chunksize=8) if r]
    print("decoded", collections.Counter(r["src"] for r in res), flush=True)
    th = dict(slr43=(22, .55), fleurs=(26, .35), cv=(28, .35))
    res = [r for r in res if r["clip"] < 5e-4 and r["vr"] > 0.2 and r["jump"] < 0.07 and r["snr"] >= th[r["src"]][0]]
    print("after snr/pitch", collections.Counter(r["src"] for r in res), flush=True)
    asr_cer(res)
    res = [r for r in res if r["cer"] <= th[r["src"]][1]]
    print("after asr", collections.Counter(r["src"] for r in res), flush=True)
    # pseudo-speakers for FLEURS (no speaker ids): k-means on mean voiced mel + median f0 per gender
    from sklearn.cluster import KMeans
    for g in ("fl_f", "fl_m"):
        rs = [r for r in res if r["spk"] == g]
        if len(rs) > 60:
            X = np.stack([np.concatenate([r["mmel"], [r["f0m"] * 4]]) for r in rs]); X = (X - X.mean(0)) / (X.std(0) + 1e-6)
            for r, k in zip(rs, KMeans(8, n_init=4, random_state=0).fit_predict(X)): r["spk"] = f"{g}{k}"
    by = collections.defaultdict(list)
    for r in res: by[r["spk"]].append(r)
    keep = {}
    for s, rs in by.items():
        if len(rs) < 40: continue
        sc = float(np.median([r["snr"] for r in rs])) - 40 * float(np.median([r["cer"] for r in rs]))
        keep[s] = sc
    for src, cap in (("slr43", 99), ("fleurs", 12), ("cv", 30)):
        ss = sorted([s for s in keep if by[s][0]["src"] == src], key=lambda s: -keep[s])
        for s in ss[cap:]: keep.pop(s)
    if sum(1 for s in keep if by[s][0]["src"] == "slr43") > 8:   # prune worst 15% of SLR43 speakers
        ss = sorted([s for s in keep if by[s][0]["src"] == "slr43"], key=lambda s: -keep[s])
        for s in ss[int(len(ss) * .85):]: keep.pop(s)
    res = [r for r in res if r["spk"] in keep]
    random.seed(0)
    ts = sorted({i["text"] for i in its if i["src"] == "slr43"}); random.shuffle(ts); test_t, val_t = set(ts[:60]), set(ts[60:90])
    for r in res:
        r["split"] = "test" if r["text"] in test_t else "val" if r["text"] in val_t else "train"
        if r["src"] != "slr43" and r["split"] == "train" and random.random() < 0.004: r["split"] = "val"
        for k in ("mmel",): r.pop(k)
    per = collections.defaultdict(float)
    for r in res:
        if r["split"] == "train": per[r["src"]] += len(r["wav"]) / audio.SR / 3600
    print("FINAL hours", dict(per), "speakers", len(keep), "wide-band clips", sum(r["wide"] for r in res), "of", len(res), flush=True)
    torch.save(res, f"{OUT}/v2.pt")
    json.dump({s: dict(src=by[s][0]["src"], n=len(by[s]), score=keep[s], snr=float(np.median([r["snr"] for r in by[s]])), wide=float(np.mean([r["wide"] for r in by[s]]))) for s in keep},
              open(f"{OUT}/v2_speakers.json", "w"), indent=1)
    print("CURATE_DONE", flush=True)
