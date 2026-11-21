"""Pack the Python front-end tables into docs/data/frontend.json for the in-browser (JS) port."""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sakhiya import symbols as S, numbers as N, translit as T, g2p

lex, _ = T._load("ne"); out = dict(symbols=S.SYMBOLS, lang2id=S.LANG2ID, punct=S.PUNCT, num_u=N._U, months=N.MONTHS, letters=N.LETTERS,
                                   ne=lex["ne"], direct=lex["direct"], sfx=T.SFX, suffixes=g2p.SUFFIXES, thr={})
for t in ("thr_dangaura", "thr_rana", "thr_chitwania"):
    l, _ = T._load(t); out["thr"][t] = l[t]
os.makedirs("docs/data", exist_ok=True)
json.dump(out, open("docs/data/frontend.json", "w", ensure_ascii=False and None) if False else open("docs/data/frontend.json", "w", encoding="utf8"), ensure_ascii=False, separators=(",", ":"))
print(os.path.getsize("docs/data/frontend.json") / 1e6, "MB")
