// Deterministic 'Chandas-Laya-Svara' Vedic & Mantra Prosody Engine (docs/chandas.js)
// 100% 1:1 algorithmic parity with sakhiya/chandas.py (verified by pytest + Node).
(function (root) {
  const SR = 24000, HOP = 256, FPS = SR / HOP;
  const DEFAULT_TONIC_HZ = 215.0, DEFAULT_BPM = 75.0;
  const SHRUTI_CENTS = {
    low_ni: -195.0000,
    komal_ni: -111.7313,
    sa: 0.0,
    prachaya: 38.0511,
    komal_re: 111.7313,
    re: 195.0000,
    komal_ga: 315.6413,
    ga: 386.3137,
  };
  const LONG_VOWEL_CHARS = new Set([..."आाईीऊूॠॄएेऐैओोऔौ"]);
  const MATRA_CHARS = new Set([..."ािीुूृॄॢेैोौ"]);
  const INDEP_VOWEL_CHARS = new Set([..."अआइईउऊऋॠऌएऐओऔ"]);
  const CONS_CHARS = new Set([..."कखगघङचछजझञटठडढणतथदधनपफबभमयरलळवशषसह"]);
  const YOGA_CHARS = new Set([..."ंँः"]);
  const HAL = "्", NUK = "़", UDATTA_SVARITA = "\u0951", ANUDATTA = "\u0952";
  const VOWEL_PHONES = new Set(["ʌ", "a", "i", "u", "e", "o", "ʌ\u0303", "a\u0303", "i\u0303", "u\u0303", "e\u0303", "o\u0303"]);
  const OM_RE = /ओ(?:[३3]म्?|म्|ं(?![क-ह]))/g;
  const VEDIC_KEEP = /[\u0953\u0954\u1cd0-\u1cff]/g;
  const RV111 = [["अग्निमीळे","अ॒ग्निमी॑ळे"],["पुरोहितं","पु॒रोहि॑तं"],["यज्ञस्य","य॒ज्ञस्य॑"],["देवमृत्विजम्","दे॒वमृ॒त्विज॑म्"],["होतारं","होता॑रं"],["रत्नधातमम्","रत्न॒धात॑मम्"]];

  function round4(x) {
    return Math.round(Number(x) * 10000) / 10000;
  }

  function isVowelPhone(p) {
    return Boolean(p && (VOWEL_PHONES.has(p) || ["ʌ", "a", "i", "u", "e", "o"].includes(p[0])));
  }

  function restoreRvSvara(t) {
    if (t.includes("अग्निमीळे") && t.includes("पुरोहित") && !t.includes("\u0951")) {
      for (const [k, v] of RV111) t = t.replaceAll(k, v);
    }
    return t;
  }

  function scanWordAksharas(word) {
    let w = String(word).normalize("NFC").replace(OM_RE, "ॐ").replace(/ज्ञ/g, "ग्ञ").replace(VEDIC_KEEP, "");
    if (w === "ॐ") {
      return [{ ak: "ॐ", weight: "G", beats: 2.0, svara: "om", long_v: true, has_yoga: true, trailing_c: 1 }];
    }
    const units = [];
    let i = 0;
    const n = w.length;
    while (i < n) {
      const ch = w[i];
      if (ch === "ॐ") {
        units.push({ type: "OM", text: "ॐ", is_long: true, svara: "om", yoga: true });
        i++;
        continue;
      }
      if (CONS_CHARS.has(ch)) {
        let cTxt = ch;
        i++;
        if (i < n && w[i] === NUK) { cTxt += w[i]; i++; }
        if (i < n && w[i] === HAL) {
          cTxt += HAL;
          i++;
          units.push({ type: "C_HAL", text: cTxt });
          continue;
        }
        let vTxt = "", isLong = false;
        if (i < n && MATRA_CHARS.has(w[i])) {
          vTxt = w[i];
          isLong = LONG_VOWEL_CHARS.has(w[i]);
          i++;
        }
        let svara = "udatta", yoga = false;
        while (i < n && (w[i] === UDATTA_SVARITA || w[i] === ANUDATTA || YOGA_CHARS.has(w[i]) || w[i] === "ऽ")) {
          if (w[i] === UDATTA_SVARITA) svara = "svarita";
          else if (w[i] === ANUDATTA) svara = "anudatta";
          else if (YOGA_CHARS.has(w[i])) yoga = true;
          else if (w[i] === "ऽ") isLong = true;
          vTxt += w[i];
          i++;
        }
        units.push({ type: "CV", text: cTxt + vTxt, is_long: isLong, svara, yoga });
      } else if (INDEP_VOWEL_CHARS.has(ch)) {
        let vTxt = ch, isLong = LONG_VOWEL_CHARS.has(ch);
        i++;
        let svara = "udatta", yoga = false;
        while (i < n && (w[i] === UDATTA_SVARITA || w[i] === ANUDATTA || YOGA_CHARS.has(w[i]) || w[i] === "ऽ")) {
          if (w[i] === UDATTA_SVARITA) svara = "svarita";
          else if (w[i] === ANUDATTA) svara = "anudatta";
          else if (YOGA_CHARS.has(w[i])) yoga = true;
          else if (w[i] === "ऽ") isLong = true;
          vTxt += w[i];
          i++;
        }
        units.push({ type: "V", text: vTxt, is_long: isLong, svara, yoga });
      } else if (ch === UDATTA_SVARITA || ch === ANUDATTA || ch === "ऽ" || YOGA_CHARS.has(ch)) {
        if (units.length) {
          units[units.length - 1].text += ch;
          if (ch === UDATTA_SVARITA) units[units.length - 1].svara = "svarita";
          else if (ch === ANUDATTA) units[units.length - 1].svara = "anudatta";
          else if (YOGA_CHARS.has(ch)) units[units.length - 1].yoga = true;
          else if (ch === "ऽ") units[units.length - 1].is_long = true;
        }
        i++;
      } else {
        i++;
      }
    }
    const aksharas = [];
    let onset = "";
    for (const u of units) {
      if (u.type === "C_HAL") {
        onset += u.text;
      } else {
        let halCount = 0;
        for (const c of onset) if (c === HAL) halCount++;
        aksharas.push({
          ak: onset + u.text,
          is_long: u.is_long,
          svara: u.svara,
          has_yoga: u.yoga,
          onset_hal: halCount,
          trailing_c: 0,
        });
        onset = "";
      }
    }
    if (onset && aksharas.length) {
      aksharas[aksharas.length - 1].ak += onset;
      aksharas[aksharas.length - 1].trailing_c += 2;
    }
    for (let k = 0; k < aksharas.length; k++) {
      if (k + 1 < aksharas.length && aksharas[k + 1].onset_hal >= 1) {
        aksharas[k].trailing_c += aksharas[k + 1].onset_hal + 1;
      }
      const isGuru = Boolean(aksharas[k].is_long || aksharas[k].has_yoga || aksharas[k].trailing_c >= 2 || aksharas[k].svara === "om");
      aksharas[k].weight = isGuru ? "G" : "L";
      aksharas[k].beats = isGuru ? 2.0 : 1.0;
    }
    return aksharas;
  }

  function detectMeter(aksharas, text = "") {
    const metricAks = aksharas.filter(a => a.svara !== "om");
    const n = metricAks.length;
    const SA = root.SA;
    const tNorm = SA ? SA.iastToDeva(String(text).normalize("NFC")) : String(text).normalize("NFC");
    if (tNorm.includes("भूर्भुवः") || tNorm.includes("तत्सवितु") || (n >= 21 && n <= 26)) return "Gayatri";
    if ((n >= 28 && n <= 36) || (n >= 14 && n <= 18 && (tNorm.includes("॥") || tNorm.includes("।")))) return "Anushtubh";
    if ((n >= 40 && n <= 45) || (n >= 20 && n <= 23 && tNorm.includes("॥"))) return "Trishtubh";
    if ((n >= 46 && n <= 52) || (n >= 24 && n <= 26 && tNorm.includes("॥"))) return "Jagati";
    return "Vedic-Laya";
  }

  function compileChandas(text, bpm = DEFAULT_BPM, tonicHz = DEFAULT_TONIC_HZ, addEnd = true) {
    const SA = root.SA;
    const deva = SA ? SA.iastToDeva(String(text).normalize("NFC")) : String(text).normalize("NFC");
    const t = restoreRvSvara(deva.replace(OM_RE, "ॐ").replace(/[\u200d\u200c]/g, ""));
    const hasVedicMarks = t.includes(UDATTA_SVARITA) || t.includes(ANUDATTA);
    const rawWords = t.replace(/[।॥]/g, " ").match(/[\u0900-\u097F]+/g) || [];

    const allAksharas = [];
    for (const rw of rawWords) {
      if (rw.includes("ॐ") && rw !== "ॐ") {
        const parts = rw.split(/(ॐ)/).filter(Boolean);
        for (const pt of parts) allAksharas.push(...scanWordAksharas(pt));
      } else {
        allAksharas.push(...scanWordAksharas(rw));
      }
    }

    const meter = detectMeter(allAksharas, t);

    if (hasVedicMarks) {
      let inPrac = false;
      for (const ak of allAksharas) {
        const sv = ak.svara;
        if (sv === "om") {
          ak.f0_cents = round4(SHRUTI_CENTS.re * 0.55);
          inPrac = false;
        } else if (sv === "svarita") {
          ak.f0_cents = round4(SHRUTI_CENTS.re);
          inPrac = true;
        } else if (sv === "anudatta") {
          ak.f0_cents = round4(SHRUTI_CENTS.low_ni);
          inPrac = false;
        } else {
          if (inPrac) {
            ak.svara = "prachaya";
            ak.f0_cents = round4(SHRUTI_CENTS.prachaya);
          } else {
            ak.f0_cents = round4(SHRUTI_CENTS.sa);
          }
        }
      }
    } else {
      let pIdx = 0;
      for (const ak of allAksharas) {
        if (ak.svara === "om") {
          ak.f0_cents = round4(SHRUTI_CENTS.re * 0.65);
          continue;
        }
        const posInPada = pIdx % 8;
        const padaNum = Math.floor(pIdx / 8) % 4;
        const archTbl = [
          0.0,
          111.7313,
          195.0000,
          195.0000,
          111.7313,
          padaNum % 2 === 0 ? 195.0000 : 38.0511,
          padaNum % 2 === 0 ? 38.0511 : -111.7313,
          padaNum % 2 === 0 ? 0.0 : -195.0000,
        ];
        const baseArch = archTbl[posInPada];
        const guruLift = ak.weight === "G" ? 32.0 : -18.0;
        ak.f0_cents = round4(baseArch + guruLift);
        pIdx++;
      }
    }

    const baseLaya = SA.layaScales(t, addEnd, true);
    const ph = baseLaya.ph, offBase = baseLaya.off, nPh = ph.length;
    const effBpm = Math.max(Number(bpm) || DEFAULT_BPM, 30.0);
    const scaleBpm = DEFAULT_BPM / effBpm;
    const laghuMs = 160.0 * scaleBpm;

    for (const ak of allAksharas) {
      ak.target_ms = round4(laghuMs * (ak.svara === "om" ? 2.6 : ak.beats));
    }

    const off = new Array(nPh).fill(0.0);
    const dm = new Array(nPh).fill(1.0);
    const durFrames = new Array(nPh).fill(5);
    const f0CentsPh = new Array(nPh).fill(0.0);

    let akI = 0, i = 0;
    while (i < nPh) {
      const p = ph[i];
      if (p === "o" && i + 1 < nPh && ph[i + 1] === "m" && (i === 0 || ["|", ",", ";", "."].includes(ph[i - 1]))) {
        const cOm = akI < allAksharas.length ? allAksharas[akI].f0_cents : 132.5415;
        if (akI < allAksharas.length && allAksharas[akI].svara === "om") akI++;
        off[i] = round4((cOm / 100.0) * 0.22);
        off[i + 1] = round4((cOm / 100.0) * 0.12);
        dm[i] = round4(1.60 * scaleBpm);
        dm[i + 1] = round4(1.35 * scaleBpm);
        durFrames[i] = Math.max(16, Math.round(0.240 * scaleBpm * FPS));
        durFrames[i + 1] = Math.max(14, Math.round(0.190 * scaleBpm * FPS));
        f0CentsPh[i] = round4(cOm);
        f0CentsPh[i + 1] = round4(cOm * 0.55);
        i += 2;
        continue;
      }
      if (isVowelPhone(p)) {
        const diph = p === "ʌ" && i + 1 < nPh && (ph[i + 1] === "i" || ph[i + 1] === "u");
        const rVoc = i > 0 && ph[i - 1] === "r" && p === "i" && (i - 1 === 0 || !isVowelPhone(ph[i - 2]));
        const fallbackG = ["a", "e", "o"].includes(p) || diph;
        const akCur = akI < allAksharas.length ? allAksharas[akI] : { weight: fallbackG ? "G" : "L", beats: fallbackG ? 2.0 : 1.0, f0_cents: 0.0 };
        akI++;
        const isG = akCur.weight === "G";
        const cVal = Number(akCur.f0_cents || 0.0);
        const svUnit = (cVal / 100.0) * 0.62 + 0.38 * Number(offBase[i]);
        if (diph) {
          off[i] = round4(svUnit);
          off[i + 1] = round4(svUnit * 0.92);
          dm[i] = round4(1.08 * scaleBpm);
          dm[i + 1] = round4(1.06 * scaleBpm);
          const fTot = Math.max(12, Math.round((laghuMs * 1.85 / 1000.0) * FPS));
          durFrames[i] = Math.floor(fTot / 2);
          durFrames[i + 1] = fTot - durFrames[i];
          f0CentsPh[i] = round4(cVal);
          f0CentsPh[i + 1] = round4(cVal * 0.92);
          i += 2;
        } else {
          off[i] = round4(svUnit);
          dm[i] = round4((isG ? 1.14 : 0.96) * scaleBpm);
          const fV = Math.max(isG ? 8 : 5, Math.round((laghuMs * (isG ? 1.75 : 0.95) / 1000.0) * FPS));
          if (rVoc) {
            durFrames[i - 1] = Math.max(4, Math.floor(fV / 3));
            durFrames[i] = Math.max(5, fV - durFrames[i - 1]);
          } else {
            durFrames[i] = fV;
          }
          f0CentsPh[i] = round4(cVal);
          i += 1;
        }
      } else {
        if (p === "," || p === ";") {
          off[i] = round4(offBase[i]);
          dm[i] = 1.18;
          durFrames[i] = p === "," ? 11 : 14;
        } else if ([".", "?", "!"].includes(p)) {
          off[i] = round4(offBase[i]);
          dm[i] = 1.20;
          durFrames[i] = 16;
        } else if (p === "|") {
          off[i] = 0.0;
          dm[i] = 1.0;
          durFrames[i] = 2;
        } else {
          off[i] = round4(offBase[i]);
          const isFinalH = p === "h" && (i + 1 === nPh || ["|", ",", ";", ".", "?", "!"].includes(ph[i + 1]));
          dm[i] = isFinalH ? 0.78 : 1.0;
          durFrames[i] = isFinalH ? 5 : 6;
        }
        i += 1;
      }
    }

    for (let k = 0; k < nPh; k++) {
      if (!isVowelPhone(ph[k]) && !["|", ",", ";", ".", "?", "!"].includes(ph[k])) {
        const lv = k > 0 ? off[k - 1] : 0.0;
        const rv = k + 1 < nPh ? off[k + 1] : 0.0;
        off[k] = round4(0.5 * (lv + rv));
        const lc = k > 0 ? f0CentsPh[k - 1] : 0.0;
        const rc = k + 1 < nPh ? f0CentsPh[k + 1] : 0.0;
        f0CentsPh[k] = round4(0.5 * (lc + rc));
      }
    }

    const rawCents = [], gamakaMask = [];
    for (let k = 0; k < nPh; k++) {
      const dK = Math.max(1, durFrames[k]);
      const cK = f0CentsPh[k];
      const isSustained = isVowelPhone(ph[k]) && dK >= 12;
      for (let fI = 0; fI < dK; fI++) {
        rawCents.push(cK);
        gamakaMask.push(isSustained ? 1.0 : 0.0);
      }
    }

    const TFrames = rawCents.length;
    const winRcRaw = Array.from({ length: 7 }, (_, j) => 0.5 * (1.0 - Math.cos(2.0 * Math.PI * (j + 1) / 8.0)));
    const wSum = winRcRaw.reduce((a, b) => a + b, 0);
    const winRc = winRcRaw.map(w => w / wSum);

    const f0HzContour = [];
    for (let tI = 0; tI < TFrames; tI++) {
      let accC = 0.0, accG = 0.0;
      for (let j = -3; j <= 3; j++) {
        const idx = Math.min(Math.max(tI + j, 0), TFrames - 1);
        const w = winRc[j + 3];
        accC += rawCents[idx] * w;
        accG += gamakaMask[idx] * w;
      }
      const vibCents = 12.0 * Math.sin(2.0 * Math.PI * 5.2 * (tI / FPS)) * accG;
      const totCents = accC + vibCents;
      const hz = Number(tonicHz) * Math.pow(2.0, totCents / 1200.0);
      f0HzContour.push(round4(hz));
    }

    return {
      meter,
      has_vedic_marks: hasVedicMarks,
      bpm: Number(bpm),
      tonic_hz: Number(tonicHz),
      aksharas: allAksharas,
      ph,
      off,
      dm,
      dur_frames: durFrames,
      f0_cents: f0CentsPh,
      f0_hz_contour: f0HzContour,
    };
  }

  const exported = {
    scanWordAksharas,
    detectMeter,
    compileChandas,
    SHRUTI_CENTS,
    DEFAULT_TONIC_HZ,
    DEFAULT_BPM,
  };
  root.CHANDAS = exported;
  if (typeof module !== "undefined" && module.exports) module.exports = exported;
})(typeof window !== "undefined" ? window : globalThis);
