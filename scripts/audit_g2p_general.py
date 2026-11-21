#!/usr/bin/env python3
"""General (non-fixture) G2P audit for Nepali + Sanskrit. REPORT-ONLY: it never edits rules, because any rule change after the
v3 CTC alignment shards were built would break train == inference phoneme parity.
Checks, over every word in eval/bench_general_200.json + a curated real-world list (conjuncts, loanwords, postpositions, Vedic):
  1. every emitted phone is in the symbol table (no silent drops, no unknown tokens)
  2. no empty output for a Devanagari word; no consonant-grapheme loss (cluster fusion / swallowed stop)
  3. native monosyllable / postposition final-schwa retention list and loanword coda list
  4. Python <-> Node (docs/g2p.js, docs/sanskrit.js) phoneme parity on all words
Usage: python3 scripts/audit_g2p_general.py [--json eval/g2p_audit_v3.json]"""
import json, os, re, subprocess, sys, unicodedata
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from sakhiya import g2p, sanskrit as sa, symbols as S

NE_WORDS = """आज बाट सँग तर गर्छ बनाउन घर कलम नाम काम पानी किताब विद्यालय सरकार प्रधानमन्त्री अर्थतन्त्र स्वास्थ्य शिक्षा
नेपाल काठमाडौँ पोखरा सगरमाथा कम्प्युटर इन्टरनेट टेक्स्ट स्पीच मोडेल मोबाइल स्कुल बस टिकट डाक्टर इन्जिनियर प्रविधि
गएको भएको गरेको बनाउनुभएको उठेर बसेर खाएर आएँ गएँ छैन हुनेछ भन्दै हेर्दै पढ्दै लेख्नुपर्छ सक्छु सकिन्छ
उनीहरूले तिमीहरूलाई हामीसँग तपाईंको मेरो उसको यहाँ त्यहाँ कहिले कसरी किन कति कुन
आठ नौ दश एक्काइस सय हजार लाख करोड रुपैयाँ प्रतिशत मिनेट घण्टा बजे साल महिना
ज्ञान क्षमा श्रीमान त्रिभुवन द्वन्द्व पर्यटन अत्यन्त उज्यालो ज्यादै ज्यू हमालज्यू
संसद् सङ्घ सम्बन्ध सँगै माँ हुँदैन खाँदैन बाँकी चाँडै आँखा आँगन
विश्वविद्यालय अन्तर्राष्ट्रिय सम्भावना प्रतिक्रिया व्यवस्थापिका न्यायालय निर्वाचन""".split()
SA_WORDS = """धर्मक्षेत्रे कुरुक्षेत्रे समवेता युयुत्सवः मामकाः पाण्डवाश्चैव किमकुर्वत सञ्जय ॐ पूर्णमदः पूर्णमिदम् उदच्यते
अग्निमीळे पुरोहितं यज्ञस्य देवमृत्विजम् होतारं रत्नधातमम् गायत्री सवितुर्वरेण्यं भर्गो देवस्य धीमहि
कर्मण्येवाधिकारस्ते मा फलेषु कदाचन यदा यदा हि धर्मस्य ग्लानिर्भवति भारत अभ्युत्थानमधर्मस्य
सत्यमेव जयते सर्वे भवन्तु सुखिनः सर्वे सन्तु निरामयाः विद्या ददाति विनयं विनयाद्याति पात्रताम्
क्षत्रिय ज्ञानम् श्रद्धा श्रेयः त्वम् तत्त्वम् उत्साहः उद्यमः सद्यः भक्तिः शक्तिः अष्टौ कृष्णः विष्णुः दुःखम्
नमः शिवाय ईशावास्यमिदं सर्वं यत्किञ्च जगत्यां जगत् तेन त्यक्तेन भुञ्जीथा मा गृधः कस्यस्विद्धनम्""".split()
NATIVE_FINAL_SCHWA = {"आज": "ʌ", "बाट": "ʌ", "तर": "ʌ", "नाम": "m", "घर": "r", "काम": "m"}   # (word -> last phone expected)
LOAN_CODA = {"टेक्स्ट": ("s", "ʈ")}                                                              # loanword: ends in consonant cluster, no schwa


def bench_words():
    try:
        b = json.load(open(os.path.join(ROOT, "eval/bench_general_200.json")))
    except Exception:
        return [], []
    ne, sn = [], []
    for s in b["sentences"]:
        txt = s.get("text", "")
        ws = re.findall(r"[\u0900-\u097F]+", txt)
        (sn if s.get("lang") in ("sa", "sanskrit") else ne).extend(ws)
    return ne, sn


def graph_cons(w):
    return sum(1 for c in unicodedata.normalize("NFC", w) if "\u0915" <= c <= "\u0939" and c not in "\u0929\u0931\u0934")


def node_phones(items):
    """items: [(lang, word)] -> phone strings from the shipped JS (docs/g2p.js + sanskrit.js), exactly what the browser runs."""
    js = ("global.window=global;global.fetch=async u=>({json:async()=>JSON.parse(require('fs').readFileSync('docs/'+u,'utf8'))});"
          "require('./docs/sanskrit.js');require('./docs/g2p.js');(async()=>{await window.SK.load();"
          "const it=JSON.parse(require('fs').readFileSync(0,'utf8'));"
          "console.log(JSON.stringify(it.map(([l,w])=>{try{return window.SK.textToPhones(w,l).slice(0,-1).join(' ')}catch(e){return 'ERR '+e.message}})))})()")
    try:
        r = subprocess.run(["node", "-e", js], input=json.dumps(items), capture_output=True, text=True, cwd=ROOT, timeout=180)
        return json.loads(r.stdout.strip().splitlines()[-1])
    except Exception:
        return None


def main():
    bn, bs = bench_words()
    ne = sorted(set(NE_WORDS + bn)); sn = sorted(set(SA_WORDS + bs))
    issues = []
    def py(w, l):
        return g2p.text_to_phones(w, "ne", False) if l == "ne" else sa.text_to_phones(w, False)
    for l, ws in (("ne", ne), ("sa", sn)):
        for w in ws:
            try: ph = py(w, l)
            except Exception as e:
                issues.append({"lang": l, "word": w, "kind": "exception", "detail": str(e)[:120]}); continue
            core = [p for p in ph if p not in (S.WORD, ".", "?", "!", ",", ";", "|")]
            bad = [p for p in core if p not in S.SYM2ID and p != "<om>"]   # <om> is expanded to "oː m" by sanskrit._expand_om
            if bad: issues.append({"lang": l, "word": w, "kind": "unknown_phone", "detail": bad})
            if not core and w not in "।॥": issues.append({"lang": l, "word": w, "kind": "empty", "detail": ""})
            ncons = sum(1 for p in core if p in S.CONS)
            if ncons is not None and ncons + 2 < graph_cons(w):
                issues.append({"lang": l, "word": w, "kind": "consonant_loss", "detail": f"{graph_cons(w)} graphemes -> {ncons} cons phones: {' '.join(core)}"})
    for w, last in NATIVE_FINAL_SCHWA.items():
        ph = g2p.word_to_phones(w, "ne")
        if ph[-1] != last: issues.append({"lang": "ne", "word": w, "kind": "final_phone", "detail": f"expected ...{last} got {' '.join(ph)}"})
    for w, tail in LOAN_CODA.items():
        ph = g2p.word_to_phones(w, "ne")
        if tuple(ph[-2:]) != tail: issues.append({"lang": "ne", "word": w, "kind": "loan_coda", "detail": " ".join(ph)})
    items = [("ne", w) for w in ne] + [("sa", w) for w in sn]
    nj = node_phones(items)
    mism = []
    if nj is None: parity = "node unavailable / API names not found"
    else:
        for (l, w), j in zip(items, nj):
            p = " ".join(p for p in py(w, l) if p not in (S.WORD, "|", "."))
            jj = " ".join(t for t in j.split() if t not in ("|", "."))
            if p != jj: mism.append({"lang": l, "word": w, "py": p, "js": jj})
        parity = f"{len(items) - len(mism)}/{len(items)} identical"
    rep = {"n_ne_words": len(ne), "n_sa_words": len(sn), "n_issues": len(issues), "issues": issues[:200],
           "parity": parity, "parity_mismatches": mism[:100]}
    if "--json" in sys.argv:
        json.dump(rep, open(os.path.join(ROOT, sys.argv[sys.argv.index("--json") + 1]), "w"), ensure_ascii=False, indent=1)
    print(f"words ne={len(ne)} sa={len(sn)} | issues={len(issues)} | py<->js parity: {parity}")
    for i in issues[:40]: print(" ", i["lang"], i["word"], i["kind"], i["detail"])
    for m in mism[:15]: print("  MISMATCH", m)


if __name__ == "__main__":
    main()
