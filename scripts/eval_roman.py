"""Roman->Devanagari front-end accuracy on held-out gold sentences (transliterationlm sent_bench) + Tharu paragraphs.
python scripts/eval_roman.py [--n 300]  -> eval/roman_metrics.json   (baseline = old lexicon/skeleton-snap converter)"""
import os, sys, re, json, argparse, unicodedata
import jiwer
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sakhiya.roman_neural import convert_many
R = os.path.join(os.path.dirname(__file__), ".."); ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=300); ap.add_argument("--node-path", default=os.environ.get("ROMAN_ORT")); ap.add_argument("--full", action="store_true")
a = ap.parse_args()
def norm(t): t = unicodedata.normalize("NFC", t).replace("\u200d", "").replace("\u200c", ""); return re.sub(r"\s+", " ", re.sub(r"[^\u0900-\u097F\s]|[।॥]", " ", t)).strip()
def score(hyp, ref): H = [norm(x) or "x" for x in hyp]; Rf = [norm(x) for x in ref]; return dict(wer=round(jiwer.wer(Rf, H), 4), cer=round(jiwer.cer(Rf, H), 4), exact=round(sum(h == r for h, r in zip(H, Rf)) / len(Rf), 4), n=len(Rf))
prev_p = f"{R}/docs/roman_metrics.json"
prev = json.load(open(prev_p, encoding="utf8")) if os.path.exists(prev_p) else {}
res = {}
if "nepali_sentences" in prev and not a.full:
    res["nepali_sentences"] = prev["nepali_sentences"]
    res["nepali_sentences"]["gaunma_check"] = convert_many([("ma gaunma baschu", "ne")], node_path=a.node_path)[0]
else:
    B = json.load(open(f"{R}/sakhiya/data/roman_bench.json", encoding="utf8"))[: a.n]
    inp = [(b["roman"], "ne") for b in B]; ref = [b["native"] for b in B]
    ne_ctx = prev.get("nepali_sentences", {}).get("context_neural") if (prev and not a.full) else score(convert_many(inp, node_path=a.node_path), ref)
    res["nepali_sentences"] = {"lexicon_snap_baseline": score(convert_many(inp, fast=True, node_path=a.node_path), ref), "context_neural": ne_ctx,
                               "gaunma_check": convert_many([("ma gaunma baschu", "ne")], node_path=a.node_path)[0]}
for tr in ("thr_dangaura", "thr_rana", "thr_chitwania"):
    rows = [json.loads(l) for l in open(f"{R}/sakhiya/data/{tr}_text.jsonl", encoding="utf8")]; ss = [r for r in rows if r["k"] == "s"][:100]
    items = [(r["roman"], tr) for r in ss]; rf = [r["dev"] for r in ss]
    res[tr] = {"lexicon_snap_baseline": score(convert_many(items, fast=True, node_path=a.node_path), rf), "context_neural": score(convert_many(items, node_path=a.node_path), rf)}
for k, v in res.items(): print(k, json.dumps(v, ensure_ascii=False))
os.makedirs(f"{R}/eval", exist_ok=True); json.dump(res, open(f"{R}/eval/roman_metrics.json", "w"), indent=1, ensure_ascii=False); json.dump(res, open(f"{R}/docs/roman_metrics.json", "w"), indent=1, ensure_ascii=False)
