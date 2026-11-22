#!/usr/bin/env python3
"""Exact inventory of the mined Nepali / Sanskrit data that still exists (for ASR planning)."""
import collections, json, os, re, tarfile, statistics as S
from huggingface_hub import hf_hub_download, HfApi
import torch
R = "/work/tts/inv"; os.makedirs(R, exist_ok=True); t = open("/tmp/hf_token").read().strip(); RID = "NikitHamal/sakhiya-v1-train"
tarfile.open(hf_hub_download(RID, "v13/mining_meta.tar.gz", token=t)).extractall(R)
def jl(p):
    o = []
    if os.path.exists(p):
        for l in open(p, encoding="utf8", errors="ignore"):
            try: o.append(json.loads(l))
            except Exception: pass
    return o
rep = {}
seg = jl(f"{R}/asr/asr.jsonl"); cloud = jl(f"{R}/asr/cloud_asr.jsonl"); gem = jl(f"{R}/asr/gemini.jsonl")
H = lambda xs: round(sum(x.get("dur") or 0 for x in xs) / 3600, 2)
rep["segments_total"] = dict(clips=len(seg), hours=H(seg), by_style={k: dict(clips=len(v), hours=H(v)) for k, v in collections.defaultdict(list, {}).items()})
bs = collections.defaultdict(list)
for x in seg: bs[x.get("cat")].append(x)
rep["segments_total"]["by_style"] = {k: dict(clips=len(v), hours=H(v), speakers=len({y.get("spk") for y in v})) for k, v in bs.items()}
rep["segments_total"]["sources"] = len({x.get("src") for x in seg}); rep["segments_total"]["speakers"] = len({x.get("spk") for x in seg})
dur = {x["path"]: x.get("dur") or 0 for x in seg}
DEVA = re.compile(r"[\u0900-\u097F]")
def deva(s):
    L = [c for c in s if c.isalpha()]; return sum(bool(DEVA.match(c)) for c in L) / max(1, len(L))
prov = collections.defaultdict(dict)
for x in cloud:
    if (x.get("text") or "").strip(): prov[x.get("provider")][x["path"]] = x["text"].strip()
for x in gem:
    if (x.get("text") or "").strip() and x.get("style") != "vedic_chant": prov["gemini(" + str(x.get("model")) + ")"][x["path"]] = x["text"].strip()
for x in seg:
    if (x.get("text_w") or "").strip(): prov["local_whisper(text_w)"][x["path"]] = x["text_w"].strip()
    if (x.get("text_m") or "").strip(): prov["local_mms(text_m)"][x["path"]] = x["text_m"].strip()
rep["transcripts_by_engine"] = {k: dict(clips=len(v), hours=round(sum(dur.get(p, 0) for p in v) / 3600, 2), devanagari_ge_90pct=sum(deva(s) >= 0.9 for s in v.values())) for k, v in prov.items()}
cloud_any = set().union(*[set(v) for k, v in prov.items() if not k.startswith("local_")])
rep["cloud_transcribed_union"] = dict(clips=len(cloud_any), hours=round(sum(dur.get(p, 0) for p in cloud_any) / 3600, 2))
import difflib
a = prov.get("assemblyai", {}); g = {}
for k, v in prov.items():
    if k.startswith("gemini"): g.update(v)
both = [p for p in a if p in g][:3000]
if both:
    r = sorted(difflib.SequenceMatcher(None, a[p], g[p]).ratio() for p in both)
    rep["assemblyai_vs_gemini_agreement"] = dict(pairs=len(both), median_char_similarity=round(r[len(r) // 2], 3), share_ge_0_9=round(sum(x >= 0.9 for x in r) / len(r), 3))
ut = [x.get("utmos") for x in seg if x.get("utmos")]; snr = [x.get("snr") for x in seg if x.get("snr")]
rep["quality"] = dict(utmos_median=round(S.median(ut), 2), utmos_ge_3=sum(u >= 3 for u in ut), snr_median_db=round(S.median(snr), 1),
                      speaker_pure_ge_0_85=sum((x.get("spk_min_sim") or 0) >= 0.85 for x in seg), dur_median_s=round(S.median([d for d in dur.values() if d]), 2))
gold = json.load(open(f"{R}/vedic/gold.json")) if os.path.exists(f"{R}/vedic/gold.json") else {}
gt = collections.defaultdict(list)
for p, v in gold.items(): gt[v.get("tier")].append(v.get("dur") or 0)
rep["vedic_sanskrit"] = {k: dict(clips=len(v), hours=round(sum(v) / 3600, 2)) for k, v in gt.items()}
rep["vedic_sanskrit"]["vagdhenu_studio_clips"] = sum(os.path.basename(p).startswith("vag_") for p in gold)
rep["vedic_sanskrit"]["verse_matched_reference_text"] = len(json.load(open(f"{R}/vedic/ref_text.json"))) if os.path.exists(f"{R}/vedic/ref_text.json") else None
# what audio STILL EXISTS (mining box sandbox is gone)
P = torch.load(hf_hub_download(RID, "v13/real_audio_24k.pt", token=t), weights_only=False)
paths = P["paths"]; lens = P["lens"].tolist(); sr = P["sr"]
in_cloud = [i for i, p in enumerate(paths) if p in cloud_any]
ved = [i for i, p in enumerate(paths) if "vedic_chant" in p]
rep["audio_still_available"] = dict(
    real_audio_pack=dict(clips=len(paths), hours=round(sum(lens) / sr / 3600, 2), note="24 kHz int16, loudness-normalised, single-speaker, UTMOS>=2.8, SNR>=15, 0 clicks"),
    pairs_with_cloud_transcript=dict(clips=len(in_cloud), hours=round(sum(lens[i] for i in in_cloud) / sr / 3600, 2)),
    vedic_gold_audio=dict(clips=len(ved), hours=round(sum(lens[i] for i in ved) / sr / 3600, 2)))
api = HfApi(token=t)
try:
    fs = api.list_repo_files("NikitHamal/sakhiya-nepali-vedic-private", repo_type="dataset")
    mm = hf_hub_download("NikitHamal/sakhiya-nepali-vedic-private", "meta/meta.jsonl", repo_type="dataset", token=t); m = jl(mm)
    rep["hf_dataset_sakhiya-nepali-vedic-private"] = dict(files=len(fs), clips=len(m), hours=round(sum(x.get("dur") or 0 for x in m) / 3600, 2),
                                                          by_style=dict(collections.Counter(x.get("style") for x in m)), tiers=dict(collections.Counter(x.get("tier") for x in m)))
except Exception as e: rep["hf_dataset_sakhiya-nepali-vedic-private"] = repr(e)[:200]
rep["tts_feature_pack_v13"] = json.load(open(hf_hub_download(RID, "v13/mined_24k_report.json", token=t)))
rep["sakhiya_tts_corpus"] = dict(clips=11169, hours=21.5, note="single female studio voice, Nepali + Sanskrit (Vagdhenu), mels only")
json.dump(rep, open(f"{R}/inventory.json", "w"), indent=1, ensure_ascii=False); print("INVENTORY", json.dumps(rep, ensure_ascii=False))
