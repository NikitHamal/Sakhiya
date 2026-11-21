"""`Auto (Mixed)` clause router (shared 1:1 with docs/mixed.js).

A manually chosen language chip is NEVER overridden.  Only lang="auto" splits the text into clauses
(। ॥ . ! ? newline), decides the language of each clause independently, and merges neighbouring clauses of the
same language (so a two-line shloka stays one Sanskrit segment).  Every segment is then synthesised with the same
single `sakhiya` speaker embedding and stitched.
"""
import re
import unicodedata

# Sanskrit: hard markers or word-final visarga / halant-म् endings (Nepali adverbs like अतः/प्रायः are exempt).
SA_HARD = re.compile("[ॐ॥\u0951\u0952ऽ]")
IAST = re.compile("[āīūṛṝḷṅñṭḍṇśṣṃṁḥ]")
NE_VISARGA = {"अतः", "प्रायः", "पुनः", "मूलतः", "प्रातः", "सम्भवतः", "संभवतः", "विशेषतः", "वस्तुतः", "अन्ततः", "क्रमशः",
              "मुख्यतः", "प्रायशः", "स्वतः", "सर्वप्रथमतः", "प्रथमतः", "अन्तःकरण"}
# Tharu: strong clitics/verb forms are unambiguous; weak ones need a second hit.
THARU_STRONG = {"संघरियाहुक्रे", "संघरिया", "संघरियाहुक्रेन", "बनाइल", "बाटै", "बाटु", "बाटी", "बडयी", "बडिय", "हलयी",
                "नाहै", "तोहार", "तोहर", "मोर", "हमाल", "हमार", "हुक्रे", "कहथ", "जाथ", "खाइल", "गइल", "रहल"}
THARU_WEAK = {"हों", "बा", "मै", "जी", "छोट", "हइ"}
_SPLIT = re.compile(r"(?<=[।॥!?])\s*|(?<=\.)(?=\s|$)\s*|\n+")
_WORD = re.compile(r"[\u0900-\u0963\u0966-\u097F]+")


def split_clauses(text):
    t = unicodedata.normalize("NFC", str(text))
    out = []
    for c in _SPLIT.split(t):
        if not c or not c.strip():
            continue
        c = c.strip()
        if not re.search(r"[\u0900-\u0963\u0966-\u097FA-Za-z0-9]", c) and out:
            out[-1] += c                      # lone punctuation joins the previous clause
        else:
            out.append(c)
    return out


def detect_lang(clause, tharu="thr_dangaura"):
    """-> 'sa' | tharu variant | 'ne' for ONE clause."""
    c = unicodedata.normalize("NFC", clause)
    if SA_HARD.search(c) or IAST.search(c):
        return "sa"
    words = _WORD.findall(c)
    strong = sum(1 for w in words if w in THARU_STRONG)
    weak = sum(1 for w in words if w in THARU_WEAK)
    if strong >= 1 or weak >= 3:
        return tharu
    vis = [w for w in words if w.endswith("ः") and w not in NE_VISARGA]
    halm = [w for w in words if w.endswith("म्") and len(w) > 3]
    if vis or len(halm) >= 2:
        return "sa"
    return "ne"


def route(text, tharu="thr_dangaura"):
    """-> [(lang, segment_text)] with neighbouring same-language clauses merged."""
    segs = []
    for c in split_clauses(text):
        L = detect_lang(c, tharu)
        if segs and segs[-1][0] == L:
            segs[-1] = (L, segs[-1][1] + " " + c)
        else:
            segs.append((L, c))
    return segs
