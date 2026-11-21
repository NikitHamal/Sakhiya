"""Distill two clean flagship voices (owner directive: exactly one female + one male, hours of data each).
Teachers: ne-NP-HemkalaNeural (female) and ne-NP-SagarNeural (male) through edge-tts (research/personal use only:
NOT commercially licensed).  Text = Nepali Wikipedia + OpenSLR/CV/FLEURS transcripts (held-out eval texts excluded).
Resumable:  python scripts/distill_edge.py --n 7000 --conc 24
Output: /work/tts/data/edge/{f,m}/*.mp3 + manifest.jsonl (id, spk, text)."""
import os, re, sys, json, random, asyncio, argparse, glob, csv
import edge_tts

OUT = "/work/tts/data/edge"; D = "/work/tts/data"
VOICES = {"f": "ne-NP-HemkalaNeural", "m": "ne-NP-SagarNeural"}
OK = re.compile(r"^[\u0900-\u097F\s,.?!।\-–;:'\"()]+$")


def sentences():
    out = []
    from datasets import load_dataset
    wk = load_dataset("wikimedia/wikipedia", "20231101.ne", split="train")
    for r in wk:
        for s in re.split(r"(?<=[।?!])\s+|\n+", r["text"]):
            s = re.sub(r"\s+", " ", s).strip()
            if 28 <= len(s) <= 170 and OK.match(s) and s[-1] in "।?!": out.append(("wiki", s))
    try:
        for p in glob.glob(D + "/slr54/**/utt_spk_text.tsv", recursive=True):
            for ln in open(p, encoding="utf8"):
                t = ln.rstrip("\n").split("\t")
                if len(t) >= 3 and 20 <= len(t[2]) <= 140 and OK.match(t[2]): out.append(("slr54", t[2].strip() + ("" if t[2].strip()[-1] in "।?!" else "।")))
    except Exception as e: print("slr54 skip", e)
    for sp in ("train", "dev", "validated"):
        f = f"{D}/cv/transcript/ne-NP/{sp}.tsv"
        if os.path.exists(f):
            for p in csv.DictReader(open(f, encoding="utf8"), delimiter="\t", quoting=csv.QUOTE_NONE):
                s = p["sentence"].strip()
                if 15 <= len(s) <= 170 and OK.match(s): out.append(("cv", s))
    return out


def exclude_texts():
    import torch
    ex = set()
    try:
        for f in ("v3", "v2"):   # never synthesise held-out eval texts (test/val splits of the curated real-speech sets)
            if os.path.exists(f"/work/tts/feats/{f}.pt"):
                for r in torch.load(f"/work/tts/feats/{f}.pt", weights_only=False):
                    if r["split"] != "train": ex.add(r["text"].strip())
                break
        for ln in open(D + "/slr43/ne_np_female/line_index.tsv", encoding="utf8"): ex.add(ln.rstrip("\n").split("\t")[1].strip())
    except Exception as e: print("exclusion warning", e)
    return ex


async def one(sem, spk, i, text, man):
    path = f"{OUT}/{spk}/{i}.mp3"
    if os.path.exists(path) and os.path.getsize(path) > 2000: return True
    async with sem:
        for k in range(5):
            try:
                await edge_tts.Communicate(text, VOICES[spk]).save(path)
                if os.path.getsize(path) > 2000:
                    man.write(json.dumps(dict(id=i, spk=spk, text=text), ensure_ascii=False) + "\n"); man.flush(); return True
            except Exception as e:
                await asyncio.sleep(1.5 * (k + 1))
    return False


async def main(a):
    for s in VOICES: os.makedirs(f"{OUT}/{s}", exist_ok=True)
    plan = f"{OUT}/plan.json" if a.round == 1 else f"{OUT}/plan{a.round}.json"
    if a.round > 1 and not os.path.exists(plan):
        # Round 2+: NEW text only (never reuse round-1 sentences); ~65% two-sentence wiki passages (60-250 chars) so the model sees
        # multi-sentence prosody (phrase-final falls, pauses between sentences), ~35% long single sentences (>= 60 chars).
        random.seed(100 + a.round); ex = exclude_texts()
        for old in glob.glob(f"{OUT}/plan*.json"):
            for v in json.load(open(old)).values(): ex.update(x["text"] for x in v)
        wk = [t for src, t in sentences() if src == "wiki" and t not in ex]; pairs = []
        for i in range(0, len(wk) - 1, 2):
            j = wk[i] + " " + wk[i + 1]
            if 60 <= len(j) <= 250: pairs.append(j)
        singles = [t for t in wk if len(t) >= 60]; pl = {}
        for s in VOICES:
            sel = random.sample(pairs, min(len(pairs), int(a.n * .65))); sel += random.sample(singles, min(len(singles), a.n - len(sel)))
            random.shuffle(sel); pl[s] = [dict(id=f"{s}r{a.round}_{k:05d}", text=t) for k, t in enumerate(sel)]
        json.dump(pl, open(plan, "w"), ensure_ascii=False)
    if not os.path.exists(plan):
        random.seed(7); ex = exclude_texts(); ss = sentences(); seen = set(); pool = []
        for src, t in ss:
            if t not in seen and t not in ex: seen.add(t); pool.append((src, t))
        print("pool", len(pool), flush=True)
        wiki = [p for p in pool if p[0] == "wiki"]; oth = [p for p in pool if p[0] != "wiki"]
        random.shuffle(wiki); random.shuffle(oth)
        pl = {}
        for s in VOICES:   # ~70% wiki prose, 30% conversational/short transcripts; independent draws per voice
            w = random.sample(wiki, min(len(wiki), int(a.n * .7))); o = random.sample(oth, min(len(oth), a.n - len(w)))
            sel = w + o; random.shuffle(sel); pl[s] = [dict(id=f"{s}{k:05d}", text=t) for k, (_, t) in enumerate(sel)]
        json.dump(pl, open(plan, "w"), ensure_ascii=False)
    pl = json.load(open(plan)); sem = asyncio.Semaphore(a.conc); man = open(f"{OUT}/manifest.jsonl", "a", encoding="utf8")
    # interleave voices so both grow evenly if the job is cut short
    jobs = []
    for k in range(max(len(v) for v in pl.values())):
        for s in VOICES:
            if k < len(pl[s]): jobs.append(one(sem, s, pl[s][k]["id"], pl[s][k]["text"], man))
    if a.rev: jobs.reverse()
    k, n = a.shard; mine = jobs[k::n]; done = [0]

    async def track(j):
        ok = await j; done[0] += bool(ok)
        if done[0] % 200 == 0: print("generated", done[0], "of", len(mine), flush=True)
    await asyncio.gather(*[track(j) for j in mine])
    print("DISTILL_DONE", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=7000); ap.add_argument("--conc", type=int, default=24); ap.add_argument("--rev", action="store_true"); ap.add_argument("--round", type=int, default=1); ap.add_argument("--shard", default="0/1")
    a = ap.parse_args(); a.shard = tuple(int(x) for x in a.shard.split("/")); asyncio.run(main(a))
