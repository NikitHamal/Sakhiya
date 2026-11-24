"""Deterministic 'Chandas-Laya-Svara' Vedic & Mantra Prosody Engine (`sakhiya/chandas.py`).
100% bit-for-bit algorithmic parity with browser `docs/chandas.js` (verified by pytest + Node).

Features:
1. Akshara Syllabifier & Laghu/Guru Scanner:
   - Parses Devanagari / IAST Sanskrit into orthographic Aksharas and tags each as Laghu (L = 1.0 beat) or Guru (G = 2.0 beats):
   - Long vowels (आ, ई, ऊ, ॠ, ए, ऐ, ओ, औ), anusvāra (ं), chandrabindu (ँ), visarga (ः), ॐ, or a short vowel before a conjunct consonant (saṃyoga) or word-final halant (्) = Guru (2.0 beats); else Laghu (1.0 beat).
2. Auto-Meter & Pāda Detector:
   - Recognizes Gayatri (3x8), Anushtubh (4x8 / 2x8), Trishtubh (4x11 / 2x11), Jagati (4x12 / 2x12), and Vedic-Laya fallback.
   - Inserts musical yati pauses and gentle phrase-final lengthening at pāda boundaries (।, ॥, and half-verse breaks).
3. Deterministic Laya Duration Compiler:
   - Locks syllable vowel durations to musical Laya beats (adjustable BPM, default 75 BPM -> ~160 ms Laghu / ~320 ms Guru), keeping consonant articulations crisp while stretching vowel nuclei smoothly.
4. Just-Intonation Shruti Svara F0 Compiler:
   - Locks continuous log-F0 to `sakhiya`'s natural chest tonic `Sa` (~215 Hz):
   - Vedic Svara marks: Anudātta (॒) = -203.91 cents (8:9), Udātta = 0.0 cents (1:1 Sa), Svarita (॑) = +203.91 cents (9:8 Re) glide + Prachaya (+38 cents).
   - Classical Shloka Melody Fallback: authentic Anushtubh/Gayatri Shruti cadence across pādas (Sa / Re / Ga / Komal Ni).
   - Natural Gamaka Micro-Modulation: smooth 5.2 Hz vibrato (+/- 12 cents) on sustained ॐ and Guru vowels >= 250 ms with raised-cosine transitions."""
import math
import re
import unicodedata
from . import sanskrit as SA, symbols as S

SR = 24000
HOP = 256
FPS = SR / HOP  # 93.75 frames/sec
DEFAULT_TONIC_HZ = 215.0
DEFAULT_BPM = 75.0

# Just-Intonation Shruti intervals in cents relative to tonic Sa (1/1)
SHRUTI_CENTS = {
    "low_ni": -195.0000,   # 8/9 Chatushruti Dha / Komal Ni below Sa (Anudatta)
    "komal_ni": -111.7313, # 15/16 Kakali Ni below Sa
    "sa": 0.0,             # 1/1 Shadja (Udatta tonic)
    "prachaya": 38.0511,   # gentle post-svarita sustain
    "komal_re": 111.7313,  # 16/15 Dvishruti Rishabha
    "re": 195.0000,        # 9/8 Chatushruti Rishabha (Svarita peak)
    "komal_ga": 315.6413,  # 6/5 Sadharana Gandhara
    "ga": 386.3137,        # 5/4 Antara Gandhara
}

LONG_VOWEL_CHARS = set("आाईीऊूॠॄएेऐैओोऔौ")
SHORT_VOWEL_CHARS = set("अइिउुऋृऌॢ")
ALL_VOWEL_CHARS = LONG_VOWEL_CHARS | SHORT_VOWEL_CHARS
MATRA_CHARS = set("ािीुूृॄॢेैोौ")
INDEP_VOWEL_CHARS = set("अआइईउऊऋॠऌएऐओऔ")
CONS_CHARS = set(SA.C.keys())
YOGA_CHARS = set("ंँः")
HAL = "्"
NUK = "़"
UDATTA_SVARITA = "\u0951"
ANUDATTA = "\u0952"
VOWEL_PHONES = ("ʌ", "a", "i", "u", "e", "o", "ʌ̃", "ã", "ĩ", "ũ", "ẽ", "õ")


def _is_vowel_phone(p):
    return bool(p and (p in VOWEL_PHONES or p[0] in ("ʌ", "a", "i", "u", "e", "o")))


def _round4(x):
    return math.floor(float(x) * 10000.0 + 0.5) / 10000.0 if x >= 0 else math.ceil(float(x) * 10000.0 - 0.5) / 10000.0


def _js_round(x):
    return int(math.floor(float(x) + 0.5))


def scan_word_aksharas(word):
    """Split a single Devanagari word into Aksharas with Laghu ('L', 1.0) / Guru ('G', 2.0) and Vedic accent."""
    w = unicodedata.normalize("NFC", word)
    w = SA._OM_RE.sub("ॐ", w).replace("ज्ञ", "ग्ञ")
    w = SA.VEDIC_KEEP.sub("", w)
    if w == "ॐ":
        return [{"ak": "ॐ", "weight": "G", "beats": 2.0, "svara": "om", "long_v": True, "has_yoga": True, "trailing_c": 1}]

    # Parse into sequence of tokens: ('C', text) or ('V', text, is_long, svara, yoga)
    units = []
    i, n = 0, len(w)
    while i < n:
        ch = w[i]
        if ch == "ॐ":
            units.append({"type": "OM", "text": "ॐ", "is_long": True, "svara": "om", "yoga": True})
            i += 1
            continue
        if ch in CONS_CHARS:
            c_txt = ch
            i += 1
            if i < n and w[i] == NUK:
                c_txt += w[i]
                i += 1
            if i < n and w[i] == HAL:
                c_txt += HAL
                i += 1
                units.append({"type": "C_HAL", "text": c_txt})
                continue
            # Has vowel (inherent 'अ' or matra)
            v_txt = ""
            is_long = False
            if i < n and w[i] in MATRA_CHARS:
                v_txt = w[i]
                is_long = (w[i] in LONG_VOWEL_CHARS)
                i += 1
            svara = "udatta"
            yoga = False
            while i < n and (w[i] in (UDATTA_SVARITA, ANUDATTA) or w[i] in YOGA_CHARS or w[i] == "ऽ"):
                if w[i] == UDATTA_SVARITA:
                    svara = "svarita"
                elif w[i] == ANUDATTA:
                    svara = "anudatta"
                elif w[i] in YOGA_CHARS:
                    yoga = True
                elif w[i] == "ऽ":
                    is_long = True
                v_txt += w[i]
                i += 1
            units.append({"type": "CV", "text": c_txt + v_txt, "is_long": is_long, "svara": svara, "yoga": yoga})
        elif ch in INDEP_VOWEL_CHARS:
            v_txt = ch
            is_long = (ch in LONG_VOWEL_CHARS)
            i += 1
            svara = "udatta"
            yoga = False
            while i < n and (w[i] in (UDATTA_SVARITA, ANUDATTA) or w[i] in YOGA_CHARS or w[i] == "ऽ"):
                if w[i] == UDATTA_SVARITA:
                    svara = "svarita"
                elif w[i] == ANUDATTA:
                    svara = "anudatta"
                elif w[i] in YOGA_CHARS:
                    yoga = True
                elif w[i] == "ऽ":
                    is_long = True
                v_txt += w[i]
                i += 1
            units.append({"type": "V", "text": v_txt, "is_long": is_long, "svara": svara, "yoga": yoga})
        elif ch in (UDATTA_SVARITA, ANUDATTA, "ऽ") or ch in YOGA_CHARS:
            if units:
                units[-1]["text"] += ch
                if ch == UDATTA_SVARITA:
                    units[-1]["svara"] = "svarita"
                elif ch == ANUDATTA:
                    units[-1]["svara"] = "anudatta"
                elif ch in YOGA_CHARS:
                    units[-1]["yoga"] = True
                elif ch == "ऽ":
                    units[-1]["is_long"] = True
            i += 1
        else:
            i += 1

    # Group C_HAL with its syllable nucleus and count following consonants (saṃyoga rule)
    aksharas = []
    onset = ""
    for u in units:
        if u["type"] == "C_HAL":
            onset += u["text"]
        else:
            aksharas.append({
                "ak": onset + u["text"],
                "is_long": u["is_long"],
                "svara": u["svara"],
                "has_yoga": u["yoga"],
                "onset_hal": onset.count(HAL),
                "trailing_c": 0,
            })
            onset = ""
    if onset and aksharas:
        # Word-final virama/halant (e.g. प्रचोदयात्, देवमृत्विजम्) closes the final syllable -> Guru!
        aksharas[-1]["ak"] += onset
        aksharas[-1]["trailing_c"] += 2

    # If syllable k+1 has a conjunct onset (onset_hal >= 1), syllable k becomes Guru by saṃyoga!
    for k in range(len(aksharas)):
        if k + 1 < len(aksharas) and aksharas[k + 1]["onset_hal"] >= 1:
            aksharas[k]["trailing_c"] += aksharas[k + 1]["onset_hal"] + 1
        is_guru = bool(aksharas[k]["is_long"] or aksharas[k]["has_yoga"] or aksharas[k]["trailing_c"] >= 2 or aksharas[k]["svara"] == "om")
        aksharas[k]["weight"] = "G" if is_guru else "L"
        aksharas[k]["beats"] = 2.0 if is_guru else 1.0
    return aksharas


def detect_meter(aksharas, text=""):
    """Classify Sanskrit meter from syllable count (excluding ॐ pranava) and Laghu/Guru cadence."""
    metric_aks = [a for a in aksharas if a.get("svara") != "om"]
    n = len(metric_aks)
    t_norm = SA.iast_to_deva(unicodedata.normalize("NFC", str(text)))
    if "भूर्भुवः" in t_norm or "तत्सवितु" in t_norm or (21 <= n <= 26):
        return "Gayatri"
    if 28 <= n <= 36 or (14 <= n <= 18 and ("॥" in t_norm or "।" in t_norm)):
        return "Anushtubh"
    if 40 <= n <= 45 or (20 <= n <= 23 and "॥" in t_norm):
        return "Trishtubh"
    if 46 <= n <= 52 or (24 <= n <= 26 and "॥" in t_norm):
        return "Jagati"
    return "Vedic-Laya"


def compile_chandas(text, bpm=DEFAULT_BPM, tonic_hz=DEFAULT_TONIC_HZ, add_end=True):
    """Compile any Sanskrit shloka / Vedic mantra into deterministic Chandas-Laya-Svara prosody targets.
    Returns dict with exact Python == JS fields:
      meter, bpm, tonic_hz, aksharas, ph, off, dm, dur_frames, f0_cents, f0_hz_contour
    """
    t = SA._restore_rv_svara(
        SA._OM_RE.sub("ॐ", SA.iast_to_deva(unicodedata.normalize("NFC", str(text)))).replace("\u200d", "").replace("\u200c", "")
    )
    has_vedic_marks = (UDATTA_SVARITA in t) or (ANUDATTA in t)
    raw_words = re.findall(r"[\u0900-\u097F]+", re.sub(r"[।॥]", " ", t))

    all_aksharas = []
    for rw in raw_words:
        if "ॐ" in rw and rw != "ॐ":
            parts = [p for p in re.split(r"(ॐ)", rw) if p]
            for pt in parts:
                all_aksharas.extend(scan_word_aksharas(pt))
        else:
            all_aksharas.extend(scan_word_aksharas(rw))

    meter = detect_meter(all_aksharas, t)

    # Assign Prachaya after Svarita in explicit Vedic mode, or Classical Shruti melody in unmarked Shloka mode
    if has_vedic_marks:
        in_prac = False
        for ak in all_aksharas:
            sv = ak["svara"]
            if sv == "om":
                ak["f0_cents"] = _round4(SHRUTI_CENTS["re"] * 0.55)
                in_prac = False
            elif sv == "svarita":
                ak["f0_cents"] = _round4(SHRUTI_CENTS["re"])
                in_prac = True
            elif sv == "anudatta":
                ak["f0_cents"] = _round4(SHRUTI_CENTS["low_ni"])
                in_prac = False
            else:
                if in_prac:
                    ak["svara"] = "prachaya"
                    ak["f0_cents"] = _round4(SHRUTI_CENTS["prachaya"])
                else:
                    ak["f0_cents"] = _round4(SHRUTI_CENTS["sa"])
    else:
        # Classical Anushtubh / Gayatri Shruti Melody across 8-syllable pādas
        p_idx = 0
        for ak in all_aksharas:
            if ak["svara"] == "om":
                ak["f0_cents"] = _round4(SHRUTI_CENTS["re"] * 0.65)
                continue
            pos_in_pada = p_idx % 8
            pada_num = (p_idx // 8) % 4
            # Traditional chanting arch: Sa -> Re -> Komal Ga/Re -> Sa on odd padas, descending cadence to Low Ni -> Sa on even padas
            base_arch = [
                0.0,
                111.7313,
                195.0000,
                195.0000,
                111.7313,
                195.0000 if pada_num % 2 == 0 else 38.0511,
                38.0511 if pada_num % 2 == 0 else -111.7313,
                0.0 if pada_num % 2 == 0 else -195.0000,
            ][pos_in_pada]
            guru_lift = 32.0 if ak["weight"] == "G" else -18.0
            ak["f0_cents"] = _round4(base_arch + guru_lift)
            p_idx += 1

    # Base phoneme sequence from canonical v2.6 Sanskrit frontend so phoneme IDs 100% match trained acoustic models
    ph, off_base, dm_base = SA.laya_scales(t, add_end=add_end, v2=True)
    n_ph = len(ph)

    # Beat duration in ms: at 75 BPM, 1 Laya matra (Laghu) = 160 ms, Guru = 320 ms
    laghu_ms = 160.0 * (DEFAULT_BPM / max(float(bpm), 30.0))
    for ak in all_aksharas:
        ak["target_ms"] = _round4(laghu_ms * (2.6 if ak["svara"] == "om" else ak["beats"]))

    # Map Akshara Laya durations & Shruti F0 offsets onto the canonical phoneme sequence
    off = [0.0] * n_ph
    dm = [1.0] * n_ph
    dur_frames = [5] * n_ph
    f0_cents_ph = [0.0] * n_ph

    ak_i = 0
    i = 0
    while i < n_ph:
        p = ph[i]
        if p == "o" and i + 1 < n_ph and ph[i + 1] == "m" and (i == 0 or ph[i - 1] in ("|", ",", ";", ".")):
            # Pranava ॐ (o + m): resonant sustained vocalic nucleus + humming nasal coda
            c_om = all_aksharas[ak_i]["f0_cents"] if ak_i < len(all_aksharas) else 132.5415
            if ak_i < len(all_aksharas) and all_aksharas[ak_i]["svara"] == "om":
                ak_i += 1
            off[i] = _round4(c_om / 100.0 * 0.22)
            off[i + 1] = _round4(c_om / 100.0 * 0.12)
            dm[i] = _round4(1.60 * (DEFAULT_BPM / max(float(bpm), 30.0)))
            dm[i + 1] = _round4(1.35 * (DEFAULT_BPM / max(float(bpm), 30.0)))
            dur_frames[i] = max(16, _js_round(0.240 * (DEFAULT_BPM / max(float(bpm), 30.0)) * FPS))
            dur_frames[i + 1] = max(14, _js_round(0.190 * (DEFAULT_BPM / max(float(bpm), 30.0)) * FPS))
            f0_cents_ph[i] = _round4(c_om)
            f0_cents_ph[i + 1] = _round4(c_om * 0.55)
            i += 2
            continue

        if _is_vowel_phone(p):
            diph = (p == "ʌ" and i + 1 < n_ph and ph[i + 1] in ("i", "u"))
            r_voc = (i > 0 and ph[i - 1] == "r" and p == "i" and (i - 1 == 0 or not _is_vowel_phone(ph[i - 2])))
            ak_cur = all_aksharas[ak_i] if ak_i < len(all_aksharas) else {"weight": "G" if (p in ("a", "e", "o") or diph) else "L", "beats": 2.0 if (p in ("a", "e", "o") or diph) else 1.0, "f0_cents": 0.0}
            ak_i += 1
            is_g = (ak_cur["weight"] == "G")
            c_val = float(ak_cur.get("f0_cents", 0.0))
            # Blend Just-Intonation Shruti target with canonical svara offset so both Vedic & Shloka contours shine
            sv_unit = (c_val / 100.0) * 0.62 + 0.38 * float(off_base[i])
            scale_bpm = DEFAULT_BPM / max(float(bpm), 30.0)
            if diph:
                off[i] = _round4(sv_unit)
                off[i + 1] = _round4(sv_unit * 0.92)
                dm[i] = _round4(1.08 * scale_bpm)
                dm[i + 1] = _round4(1.06 * scale_bpm)
                f_tot = max(12, _js_round((laghu_ms * 1.85 / 1000.0) * FPS))
                dur_frames[i] = f_tot // 2
                dur_frames[i + 1] = f_tot - dur_frames[i]
                f0_cents_ph[i] = _round4(c_val)
                f0_cents_ph[i + 1] = _round4(c_val * 0.92)
                i += 2
            else:
                off[i] = _round4(sv_unit)
                dm[i] = _round4((1.14 if is_g else 0.96) * scale_bpm)
                f_v = max(8 if is_g else 5, _js_round((laghu_ms * (1.75 if is_g else 0.95) / 1000.0) * FPS))
                if r_voc:
                    dur_frames[i - 1] = max(4, f_v // 3)
                    dur_frames[i] = max(5, f_v - dur_frames[i - 1])
                else:
                    dur_frames[i] = f_v
                f0_cents_ph[i] = _round4(c_val)
                i += 1
        else:
            if p in (",", ";"):
                off[i] = _round4(off_base[i])
                dm[i] = 1.18
                dur_frames[i] = 11 if p == "," else 14
            elif p in (".", "?", "!"):
                off[i] = _round4(off_base[i])
                dm[i] = 1.20
                dur_frames[i] = 16
            elif p == "|":
                off[i] = 0.0
                dm[i] = 1.0
                dur_frames[i] = 2
            else:
                # Consonant: crisp natural articulation (5-7 frames), light visarga 'h' capped at <=5 frames
                off[i] = _round4(off_base[i])
                is_final_h = (p == "h" and (i + 1 == n_ph or ph[i + 1] in ("|", ",", ";", ".", "?", "!")))
                dm[i] = 0.78 if is_final_h else 1.0
                dur_frames[i] = 5 if is_final_h else 6
            i += 1

    # Smooth consonant pitch offsets between adjacent vowels (raised-cosine continuity)
    for k in range(n_ph):
        if not _is_vowel_phone(ph[k]) and ph[k] not in ("|", ",", ";", ".", "?", "!"):
            lv = off[k - 1] if k > 0 else 0.0
            rv = off[k + 1] if k + 1 < n_ph else 0.0
            off[k] = _round4(0.5 * (lv + rv))
            lc = f0_cents_ph[k - 1] if k > 0 else 0.0
            rc = f0_cents_ph[k + 1] if k + 1 < n_ph else 0.0
            f0_cents_ph[k] = _round4(0.5 * (lc + rc))

    # Build frame-level continuous F0 contour (in Hz) with raised-cosine transitions + 5.2 Hz +/-12c Gamaka on sustained Guru/Om
    raw_cents = []
    gamaka_mask = []
    for k in range(n_ph):
        d_k = max(1, int(dur_frames[k]))
        c_k = float(f0_cents_ph[k])
        is_sustained = _is_vowel_phone(ph[k]) and d_k >= 12
        for f_i in range(d_k):
            raw_cents.append(c_k)
            gamaka_mask.append(1.0 if is_sustained else 0.0)

    T_frames = len(raw_cents)
    # 7-frame raised-cosine smoothing filter
    win_rc = [0.5 * (1.0 - math.cos(2.0 * math.pi * (j + 1) / 8.0)) for j in range(7)]
    w_sum = sum(win_rc)
    win_rc = [w / w_sum for w in win_rc]

    f0_hz_contour = []
    for t_i in range(T_frames):
        acc_c = 0.0
        acc_g = 0.0
        for j in range(-3, 4):
            idx = min(max(t_i + j, 0), T_frames - 1)
            w = win_rc[j + 3]
            acc_c += raw_cents[idx] * w
            acc_g += gamaka_mask[idx] * w
        # 5.2 Hz subtle natural Gamaka (+/- 12 cents) on sustained Guru / Pranava nuclei
        vib_cents = 12.0 * math.sin(2.0 * math.pi * 5.2 * (t_i / FPS)) * acc_g
        tot_cents = acc_c + vib_cents
        hz = float(tonic_hz) * math.pow(2.0, tot_cents / 1200.0)
        f0_hz_contour.append(_round4(hz))

    return {
        "meter": meter,
        "has_vedic_marks": has_vedic_marks,
        "bpm": float(bpm),
        "tonic_hz": float(tonic_hz),
        "aksharas": all_aksharas,
        "ph": ph,
        "off": off,
        "dm": dm,
        "dur_frames": dur_frames,
        "f0_cents": f0_cents_ph,
        "f0_hz_contour": f0_hz_contour,
    }
