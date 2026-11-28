#!/usr/bin/env python3
"""EXP7 (Sonnet lane, session 2): TEXT-level audit of ai4bharat/Rasa Nepali (all 64 parquet shards, text/style/gender/duration
columns only - no audio).  Questions that decide the data pipeline:
 1. How many unique sentences, and do Female and Male read the SAME sentences (-> parallel multi-speaker data, shared text)?
 2. Do test sentences leak into train (exact duplicates)?   3. Character inventory (rare chars, Latin, ASCII/Devanagari digits).
 4. Word-level OOV of test vs train (what a held-out set really measures).   5. Per-style words/sec (speaking-rate prior) and
 hours of each style per voice.   usage: exp7_rasa_text_audit.py OUT_JSON
"""
import sys, json, re, collections, time
import pyarrow.parquet as pq
import pandas as pd
from huggingface_hub import HfFileSystem
tok = open("/tmp/hf_token").read().strip()
fs = HfFileSystem(token=tok); T0 = time.time()
files = sorted(f for f in fs.ls("datasets/ai4bharat/Rasa/Nepali", detail=False) if f.endswith(".parquet"))
rows = []
for f in files:
    t = pq.ParquetFile(fs.open(f)).read(columns=["gender", "style", "duration", "text"]).to_pandas()
    t["split"] = "test" if "/test-" in f else "train"; rows.append(t)
df = pd.concat(rows, ignore_index=True); df["g"] = df.gender.str.lower(); df["dur"] = pd.to_numeric(df.duration, errors="coerce")
print("loaded", len(df), "in", round(time.time() - T0), "s", flush=True)
def norm(s): return re.sub(r"\s+", " ", re.sub(r"[।॥,.?!\-–—\"'“”‘’():;|]", " ", s)).strip()
df["n"] = df.text.map(norm)
res = {"n_utts": int(len(df)), "hours": round(float(df.dur.sum() / 3600), 2)}
# 1. sentence overlap
sf_, sm_ = set(df[df.g == "female"].n), set(df[df.g == "male"].n)
res["unique_sentences"] = {"female": len(sf_), "male": len(sm_), "union": len(sf_ | sm_), "shared_F_and_M": len(sf_ & sm_),
                           "shared_frac_of_male": round(len(sf_ & sm_) / max(1, len(sm_)), 3)}
# 2. test/train leakage
tr, te = set(df[df.split == "train"].n), set(df[df.split == "test"].n)
res["test_train"] = {"test_unique": len(te), "train_unique": len(tr), "test_sentences_also_in_train": len(te & tr),
                     "frac": round(len(te & tr) / max(1, len(te)), 3),
                     "test_genders": df[df.split == "test"].g.value_counts().to_dict()}
# 3. characters
cc = collections.Counter("".join(df.text))
dev = {c: n for c, n in cc.items() if "\u0900" <= c <= "\u097f"}
res["chars"] = {"n_distinct": len(cc), "devanagari_distinct": len(dev),
                "latin_chars": sum(n for c, n in cc.items() if c.isascii() and c.isalpha()),
                "ascii_digits": sum(n for c, n in cc.items() if c.isdigit() and c.isascii()),
                "deva_digits": sum(n for c, n in cc.items() if "०" <= c <= "९"),
                "non_deva_non_ascii_top": {c: n for c, n in cc.most_common(400) if not c.isascii() and not ("\u0900" <= c <= "\u097f")},
                "rare_deva_lt20": {c: n for c, n in dev.items() if n < 20},
                "total_chars": sum(cc.values())}
res["utts_with_latin"] = int(df.text.str.contains(r"[A-Za-z]").sum()); res["utts_with_digits"] = int(df.text.str.contains(r"[0-9०-९]").sum())
# 4. word OOV
wtr = collections.Counter(w for s in df[df.split == "train"].n for w in s.split())
wte = [w for s in df[df.split == "test"].n for w in s.split()]
res["words"] = {"train_types": len(wtr), "train_tokens": sum(wtr.values()), "test_tokens": len(wte),
                "test_oov_token_rate": round(sum(1 for w in wte if w not in wtr) / max(1, len(wte)), 4),
                "train_hapax_frac_of_types": round(sum(1 for n in wtr.values() if n == 1) / len(wtr), 3),
                "types_ge5": sum(1 for n in wtr.values() if n >= 5), "types_ge20": sum(1 for n in wtr.values() if n >= 20)}
# 5. style table
df["nw"] = df.n.str.split().str.len(); df["wps"] = df.nw / df.dur
sty = {}
for (g, s), x in df.groupby(["g", "style"]):
    sty.setdefault(g, {})[s] = {"utts": int(len(x)), "hours": round(float(x.dur.sum() / 3600), 2), "mean_dur_s": round(float(x.dur.mean()), 2),
                                "words_per_s_median": round(float(x.wps.median()), 2), "chars_mean": round(float(x.text.str.len().mean()), 1),
                                "frac_dur_gt_15s": round(float((x.dur > 15).mean()), 3), "frac_dur_lt_1s": round(float((x.dur < 1).mean()), 3)}
res["styles"] = sty
res["duration_quantiles_s"] = {str(k): round(float(v), 2) for k, v in df.dur.quantile([.01, .05, .5, .95, .99]).items()}
res["examples"] = {"latin": df[df.text.str.contains(r"[A-Za-z]")].text.head(8).tolist(), "digits": df[df.text.str.contains(r"[0-9०-९]")].text.head(8).tolist(),
                   "very_long": df.sort_values("dur").text.tail(2).str.slice(0, 160).tolist()}
json.dump(res, open(sys.argv[1], "w"), ensure_ascii=False, indent=1)
print(json.dumps(res, ensure_ascii=False, indent=1)[:6000])
