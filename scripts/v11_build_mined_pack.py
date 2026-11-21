#!/usr/bin/env python3
"""v11: build the mined-data feature pack for the V1 fine-tune (runs on the mining box, GPU only for mel + CTC alignment).

Selection (all measured fields from the mining pipeline, nothing hand-picked):
  * Nepali news / conversational / storytelling segments with a cloud transcript: AssemblyAI first, then Gemini
    (Gemini rows flagged music/noise are dropped), >= 90 % Devanagari letters;
    UTMOS >= --min_utmos, SNR >= --min_snr, speaker purity (spk_min_sim) >= --min_spk, <= 1 click, no clipping, 1.2-18 s.
  * Vedic: gold/ok clips from vedic/gold.json that are NOT Vagdhenu (already in V1) and have exact verse text in
    vedic/ref_text.json (verse-matched, never an ASR guess).
Features follow V1's recipe exactly (sakhiya/v6_pipeline.py _align_and_pack): RMS-normalised 24 kHz log-mel, g2p / laya
phones, pitch offsets, duration multipliers, CTC-Viterbi durations. The worst 8 % CTC alignment confidences per language are
dropped (bad transcripts). Speakers are clustered by long-term mel envelope into --slots groups -> speaker slots 1..N
(slot 0 stays Sakhiya-only). Style ids match the runtime table: news 0, conversational 1, storytelling 2, vedic 5."""
import argparse, collections, json, os, re, sys, time
import numpy as np, soundfile as sf, librosa, torch
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..")); sys.path.insert(0, REPO)
from sakhiya import g2p, sanskrit as SA  # noqa: E402
from sakhiya.ctc_aligner import MelCTCAligner, batch_viterbi_ctc_durations  # noqa: E402
from sakhiya.symbols import SYM2ID  # noqa: E402
from sakhiya.v2_vocoder import mel_spec_24k, SR  # noqa: E402
R = "/work/tts"
STY = {"news": 0, "conversational": 1, "storytelling": 2, "vedic_chant": 5}
DEVA = re.compile(r"[\u0900-\u097F]")


def jl(p):
    out = []
    if not os.path.exists(p): return out
    with open(p) as f:
        for l in f:
            try: out.append(json.loads(l))
            except Exception: pass
    return out


def deva_frac(t):
    L = [c for c in t if c.isalpha()]
    return sum(bool(DEVA.match(c)) for c in L) / max(1, len(L))


def kmeans(X, k, iters=40, seed=0):
    rng = np.random.default_rng(seed); C = X[rng.choice(len(X), k, replace=False)].copy()
    for _ in range(iters):
        a = ((X[:, None] - C[None]) ** 2).sum(-1).argmin(1)
        for j in range(k):
            if (a == j).any(): C[j] = X[a == j].mean(0)
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctc", default=f"{R}/v11/in/feats/v31_ctc_recognizer.pt")
    ap.add_argument("--out", default=f"{R}/v11/v11_mined_24k.pt")
    ap.add_argument("--min_utmos", type=float, default=2.8); ap.add_argument("--min_snr", type=float, default=15.0)
    ap.add_argument("--min_spk", type=float, default=0.70); ap.add_argument("--slots", type=int, default=15)
    ap.add_argument("--max_items", type=int, default=0); ap.add_argument("--drop_conf_pct", type=float, default=8.0)
    a = ap.parse_args(); dev = "cuda"; t0 = time.time()
    seg = {x["path"]: x for x in jl(f"{R}/asr/asr.jsonl")}
    asm = {x["path"]: x["text"] for x in jl(f"{R}/asr/cloud_asr.jsonl") if x.get("provider") == "assemblyai" and (x.get("text") or "").strip()}
    gem = {x["path"]: x["text"] for x in jl(f"{R}/asr/gemini.jsonl")
           if (x.get("text") or "").strip() and x.get("style") != "vedic_chant" and not x.get("music") and not x.get("noise")}
    cands, why = [], collections.Counter()
    for p, x in seg.items():
        cat = x.get("cat")
        if cat not in ("news", "conversational", "storytelling"): continue
        txt, src = (asm[p], "assemblyai") if p in asm else ((gem[p], "gemini") if p in gem else (None, None))
        if not txt: why["no_cloud_text"] += 1; continue
        if deva_frac(txt) < 0.9: why["not_devanagari"] += 1; continue
        if (x.get("utmos") or 0) < a.min_utmos: why["utmos"] += 1; continue
        if (x.get("snr") or 0) < a.min_snr: why["snr"] += 1; continue
        if (x.get("spk_min_sim") or 0) < a.min_spk: why["speaker_purity"] += 1; continue
        if (x.get("clicks") or 0) > 1 or (x.get("clip") or 0) > 0.001: why["clicks_clip"] += 1; continue
        if not (1.2 <= (x.get("dur") or 0) <= 18.0): why["duration"] += 1; continue
        cands.append(dict(path=p, text=txt.strip(), lang="ne", cat=cat, spk=x.get("spk") or x.get("src"), utmos=x.get("utmos"), tsrc=src))
    gold = json.load(open(f"{R}/vedic/gold.json")); ref = json.load(open(f"{R}/vedic/ref_text.json"))
    for p, v in gold.items():
        b = os.path.basename(p)
        if v.get("tier") not in ("gold", "ok") or b.startswith("vag_") or p not in ref: continue
        if not (1.2 <= (v.get("dur") or 0) <= 20.0): continue
        cands.append(dict(path=p, text=ref[p].strip(), lang="sa", cat="vedic_chant", spk="ved_" + b.rsplit("_", 1)[0].split(".")[0], utmos=None, tsrc="verse_ref"))
    if a.max_items: cands = cands[: a.max_items // 2] + cands[-(a.max_items // 2):]
    print(f"[v11-pack] candidates {len(cands)} by cat {collections.Counter(c['cat'] for c in cands)} by text {collections.Counter(c['tsrc'] for c in cands)} rejected {dict(why)}", flush=True)
    ctc = MelCTCAligner().to(dev).train(False)
    ck = torch.load(a.ctc, map_location=dev, weights_only=False); ctc.load_state_dict(ck.get("model", ck))
    c_mean, c_std = float(ck.get("mean", -5.5028)), float(ck.get("std", 2.1869))
    items, fail = [], collections.Counter()
    with torch.no_grad():
        for n, c in enumerate(cands):
            try:
                y, sr = sf.read(c["path"], dtype="float32", always_2d=False)
                if y.ndim > 1: y = y.mean(1)
                if sr != SR: y = librosa.resample(y, orig_sr=sr, target_sr=SR)
                rms = float(np.sqrt(np.mean(y ** 2)))
                if rms < 1e-4: fail["silent"] += 1; continue
                y = np.clip(y * (0.062 / rms), -0.97, 0.97).astype(np.float32)
                mel = mel_spec_24k(torch.from_numpy(y), dev)[0]
                if c["lang"] == "sa": ph, off, dm = SA.laya_scales(c["text"], v2=True)
                else:
                    ph, off = g2p.text_to_phones_acc(c["text"], "ne"); dm = g2p.phone_dm(ph)
                vi = [i for i, p in enumerate(ph) if p in SYM2ID]
                if len(vi) < 3 or mel.shape[1] <= len(vi) * 2: fail["too_few_frames"] += 1; continue
                ids = torch.tensor([SYM2ID[ph[i]] for i in vi], dtype=torch.long, device=dev)
                log_p = ctc((mel[None].float() - c_mean) / c_std)
                durs, conf = batch_viterbi_ctc_durations(log_p, ids[None], torch.tensor([mel.shape[1]], device=dev), torch.tensor([len(ids)], device=dev))
                cid = os.path.basename(c["path"])[:-4]
                items.append(dict(id=cid, text=c["text"], phones=[ph[i] for i in vi], ids=ids.cpu(), po=torch.tensor([float(off[i]) for i in vi]),
                                  dm=torch.tensor([float(dm[i]) for i in vi]), dur_ctc=durs[0].cpu().long(), mel=mel.half().cpu(), spk_key=c["spk"],
                                  lang=c["lang"], style=c["cat"], v3_style_id=STY[c["cat"]], split="val" if hash(cid) % 40 == 0 else "train",
                                  conf=float(conf[0]), utmos=c["utmos"], tsrc=c["tsrc"], mined=True))
            except Exception as e:
                fail[type(e).__name__ + ":" + str(e)[:60]] += 1
            if (n + 1) % 500 == 0: print(f"[v11-pack] {n + 1}/{len(cands)} kept={len(items)} fail={dict(fail)} {time.time() - t0:.0f}s", flush=True)
    keep = []
    for lg in ("ne", "sa"):
        L = [x for x in items if x["lang"] == lg]
        if not L: continue
        thr = float(np.percentile([x["conf"] for x in L], a.drop_conf_pct)); keep += [x for x in L if x["conf"] >= thr]
        print(f"[v11-pack] {lg}: dropped {sum(x['conf'] < thr for x in L)} worst-aligned (conf < {thr:.3f})", flush=True)
    keys = sorted({x["spk_key"] for x in keep}); env = collections.defaultdict(list)
    for x in keep: env[x["spk_key"]].append(x["mel"].float().mean(1).numpy())
    X = np.stack([np.mean(env[k], 0) for k in keys]); X = (X - X.mean(0)) / (X.std(0) + 1e-6)
    lab = kmeans(X, min(a.slots, len(keys))); slot_of = {k: int(l) + 1 for k, l in zip(keys, lab)}
    for x in keep: x["spk"] = slot_of[x["spk_key"]]
    h = collections.defaultdict(float)
    for x in keep: h[x["style"]] += x["mel"].shape[1] * 256 / SR / 3600
    rep = dict(items=len(keep), speakers=len(keys), hours={k: round(v, 2) for k, v in h.items()}, slots=dict(collections.Counter(x["spk"] for x in keep)),
               text_source=dict(collections.Counter(x["tsrc"] for x in keep)), fail=dict(fail), rejected=dict(why), secs=round(time.time() - t0))
    torch.save(dict(items=keep, slot_of=slot_of, report=rep), a.out)
    json.dump(rep, open(a.out.replace(".pt", "_report.json"), "w"), indent=1)
    print("[v11-pack] DONE", json.dumps(rep), flush=True)


if __name__ == "__main__":
    main()
