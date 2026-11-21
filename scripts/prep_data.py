"""Build training features. Run on the GPU box:  python scripts/prep_data.py
Outputs /work/tts/feats/{commercial,research}.pt  (commercial = CC BY-SA SLR43 only)."""
import os, re, sys, glob, random, collections
from concurrent.futures import ProcessPoolExecutor
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sakhiya import audio, g2p, symbols as S

D = "/work/tts/data"; OUT = "/work/tts/feats"
OK = re.compile(r"^[\u0900-\u097F\s,.?!।\-–;:'\"()]+$")


def items():
    for ln in open(f"{D}/slr43/ne_np_female/line_index.tsv", encoding="utf8"):
        i, t = ln.rstrip("\n").split("\t")[:2]
        spk = "s" + i.split("_")[1]
        yield dict(id=i, spk=spk, text=t.strip(), path=f"{D}/slr43/ne_np_female/wavs/{i}.wav", lic="cc-by-sa")
    for g, f in (("f", "FemaleVoice.tsv"), ("m", "MaleVoice.tsv")):
        for ln in list(open(f"{D}/male-female-data/{f}", encoding="utf8"))[1:]:
            p = ln.rstrip("\r\n").split("\t")
            if len(p) < 2:
                continue
            yield dict(id=p[0], spk="n" + g, text=p[1].strip(), path=f"{D}/male-female-data/{p[0]}.wav", lic="cc-by-nc-sa")


def proc(it):
    try:
        if not OK.match(it["text"]) or not os.path.exists(it["path"]):
            return None
        x = audio.load_wav(it["path"])
        if not 0.8 < len(x) / audio.SR < 16:
            return None
        mel = audio.mel_spec(torch.from_numpy(x))[0].numpy()
        f0 = audio.f0_track(x, mel.shape[1])
        ph = g2p.text_to_phones(it["text"])
        return dict(it, ph=S.encode(ph), phs=" ".join(ph), mel=mel.astype(np.float16), f0=f0.astype(np.float16),
                    wav=(x * 32767).astype(np.int16))
    except Exception as e:
        print("skip", it["id"], e, flush=True); return None


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    its = list(items())
    with ProcessPoolExecutor(16) as ex:
        res = [r for r in ex.map(proc, its, chunksize=8) if r]
    print("kept", len(res), "of", len(its), flush=True)
    random.seed(0)
    texts = sorted({r["text"] for r in res if r["lic"] == "cc-by-sa"})
    random.shuffle(texts)
    test_t, val_t = set(texts[:60]), set(texts[60:90])
    for r in res:
        r["split"] = "test" if r["text"] in test_t else "val" if r["text"] in val_t else "train"
    for name, lic in (("commercial", {"cc-by-sa"}), ("research", {"cc-by-sa", "cc-by-nc-sa"})):
        sel = [r for r in res if r["lic"] in lic]
        torch.save(sel, f"{OUT}/{name}.pt")
        c = collections.Counter(r["split"] for r in sel)
        hrs = sum(len(r["wav"]) for r in sel if r["split"] == "train") / audio.SR / 3600
        print(name, dict(c), f"train hours={hrs:.2f}", "speakers", len({r['spk'] for r in sel}), flush=True)
