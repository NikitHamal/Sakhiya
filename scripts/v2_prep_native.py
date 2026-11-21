"""Sakhiya v2 Stage 1B (Track A): Pure Nepal-native 24 kHz shards from curated SLR43 female + FLEURS ne_np + Common Voice ne-NP.
Excludes all held-out eval/val texts, matches v2_prep_rasa.py's exact 24 kHz / -23 LUFS / 100-band log-mel spec,
keeps exact real speaker IDs (spk 4..15 in n_spk=16; zero speaker merging, zero Edge-TTS, zero Hindi, zero WORLD pitch-shifting),
and extracts real normalized log-F0 + per-phone pitch_add contours. Also writes SLR43 24 kHz studio wav shard for VocoderV2.
usage: python3 scripts/v2_prep_native.py [/work/tts/feats/v2.pt] [/work/tts/feats/v2_native_24k.pt]"""
import os, sys, json, time, zlib, collections
import numpy as np, soxr, librosa, pyloudnorm as pyln, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from sakhiya import g2p
from scripts.v2_prep_rasa import logmel, SR, HOP

# Top 12 clean native Nepal speakers mapped to exact spk_id 4..15 (preserving 0..3 for Rasa ne-F, ne-M, sa-F, sa-M)
NATIVE_SPK = {
    "s0546": 4,       # SLR43 female studio
    "s3614": 5,       # SLR43 female studio
    "s2099": 6,       # SLR43 female studio
    "s3960": 7,       # SLR43 female studio
    "s7957": 8,       # SLR43 female studio
    "s6329": 9,       # SLR43 female studio
    "s9407": 10,      # SLR43 female studio
    "fl_f5": 11,      # FLEURS ne_np female
    "fl_f1": 12,      # FLEURS ne_np female
    "fl_f2": 13,      # FLEURS ne_np female
    "cv_13b830fe": 14,# Common Voice ne-NP cleanest
    "cv_0142a5d1": 15,# Common Voice ne-NP cleanest
}


def frame_f0_norm(f0_22k, n_frames_24k):
    """Interpolate PyWorld F0 track to 24 kHz mel frames and compute utterance-normalized log-F0 z-score in [-2.5, 2.5]."""
    if f0_22k is None or len(f0_22k) < 4:
        return np.zeros(n_frames_24k, dtype=np.float16)
    f0 = np.asarray(f0_22k, dtype=np.float32)
    x_old = np.linspace(0.0, 1.0, len(f0))
    x_new = np.linspace(0.0, 1.0, n_frames_24k)
    f0_r = np.interp(x_new, x_old, f0)
    v = f0_r > 65.0
    if v.sum() < 4:
        return np.zeros(n_frames_24k, dtype=np.float16)
    lf = np.zeros_like(f0_r)
    lf[v] = np.log(f0_r[v])
    mu = float(lf[v].mean())
    sd = max(float(lf[v].std()), 0.12)
    z = np.zeros_like(f0_r)
    z[v] = np.clip((lf[v] - mu) / sd, -2.5, 2.5)
    return z.astype(np.float16)


def main():
    src_pt = sys.argv[1] if len(sys.argv) > 1 else "/work/tts/feats/v2.pt"
    dst_pt = sys.argv[2] if len(sys.argv) > 2 else "/work/tts/feats/v2_native_24k.pt"
    voc_shard = "/work/tts/feats/v2_voc_shards/SLR43_native_studio.pt"
    t0 = time.time()
    raw_items = torch.load(src_pt, weights_only=False)
    # Collect all held-out test/val texts so zero eval text ever enters training
    heldout_texts = {r["text"].strip() for r in raw_items if r.get("split") in ("test", "val")}
    meter = pyln.Meter(SR)
    out, voc_out = [], []
    hrs = collections.defaultdict(float)
    for r in raw_items:
        spk_name = r.get("spk", "")
        if spk_name not in NATIVE_SPK:
            continue
        txt = r["text"].strip()
        if r.get("split") != "train" or txt in heldout_texts:
            continue
        try:
            y22 = r["wav"].astype(np.float32) / 32767.0
            y = soxr.resample(y22, 22050, SR, quality="VHQ")
            _, (a, b) = librosa.effects.trim(y, top_db=45, frame_length=1024, hop_length=256)
            m = int(0.10 * SR)
            y = y[max(0, a - m): min(len(y), b + m)]
            if len(y) < SR * 0.6 or len(y) > SR * 20:
                continue
            L = meter.integrated_loudness(y)
            if np.isfinite(L):
                y = pyln.normalize.loudness(y, L, -23.0)
            pk = float(np.abs(y).max())
            if pk > 0.97:
                y = y * (0.97 / pk)
            mel = logmel(y)
            ph, off = g2p.text_to_phones_acc(txt, "ne")
            f0_z = frame_f0_norm(r.get("f0"), mel.shape[1])
            spk_id = NATIVE_SPK[spk_name]
            is_studio = 1 if r.get("src") in ("slr43", "fleurs") else 0
            split = "val" if zlib.crc32(r["id"].encode()) % 66 == 0 else "train"
            out.append(dict(
                id=r["id"], text=txt, phones=ph, pitch_off=np.asarray(off, dtype=np.float16),
                mel=torch.from_numpy(mel), f0_z=torch.from_numpy(f0_z),
                spk=spk_id, spk_name=spk_name, lang="ne", style="CONV", style_id=4,
                is_studio=is_studio, dur=len(y) / SR, src=r.get("src", "slr43"), split=split,
            ))
            hrs[f"{r.get('src')}/{spk_name}(spk{spk_id})"] += len(y) / SR / 3600.0
            if r.get("src") == "slr43":
                voc_out.append(dict(
                    id=r["id"], wav=torch.from_numpy((y * 32767.0).astype(np.int16)),
                    spk=0, lang="ne", split="val" if zlib.crc32(r["id"].encode()) % 60 == 0 else "train",
                ))
        except Exception as e:
            print("skip", r.get("id"), repr(e)[:80], flush=True)
    os.makedirs(os.path.dirname(dst_pt), exist_ok=True)
    torch.save(dict(items=out, spk=NATIVE_SPK, spec=dict(sr=SR, n_fft=1024, hop=HOP, n_mels=100, fmax=12000, floor=1e-5)), dst_pt + ".tmp")
    os.replace(dst_pt + ".tmp", dst_pt)
    if voc_out:
        os.makedirs(os.path.dirname(voc_shard), exist_ok=True)
        torch.save(voc_out, voc_shard + ".tmp")
        os.replace(voc_shard + ".tmp", voc_shard)
    meta = dict(n=len(out), voc_slr43_n=len(voc_out), hours={k: round(v, 3) for k, v in sorted(hrs.items())}, total_hours=round(sum(hrs.values()), 3))
    json.dump(meta, open(dst_pt + ".json", "w"), indent=1)
    print("V2_NATIVE_DONE", meta, f"{time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
