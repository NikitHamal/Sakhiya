// Sanskrit front-end (lang "sa"): faithful port of sakhiya/sanskrit.py. IAST/ASCII -> Devanagari, schwa-retaining G2P with vowel length,
// visarga echo, anusvara assimilation and a rule-based laya model (guru/laghu weights, pada pauses for shlokas).
(() => {
  const IV = {a:"अ",ā:"आ",i:"इ",ī:"ई",u:"उ",ū:"ऊ",ṛ:"ऋ",ṝ:"ॠ",ḷ:"ऌ",e:"ए",ai:"ऐ",o:"ओ",au:"औ"}, MT = {a:"",ā:"ा",i:"ि",ī:"ी",u:"ु",ū:"ू",ṛ:"ृ",ṝ:"ॄ",ḷ:"ॢ",e:"े",ai:"ै",o:"ो",au:"ौ"};
  const CN = {kh:"ख",gh:"घ",ch:"छ",jh:"झ",ṭh:"ठ",ḍh:"ढ",th:"थ",dh:"ध",ph:"फ",bh:"भ",k:"क",g:"ग",ṅ:"ङ",c:"च",j:"ज",ñ:"ञ",ṭ:"ट",ḍ:"ड",ṇ:"ण",t:"त",d:"द",n:"न",p:"प",b:"ब",m:"म",y:"य",r:"र",l:"ल",ḻ:"ळ",v:"व",ś:"श",ṣ:"ष",s:"स",h:"ह"};
  const OT = {ṃ:"ं",ṁ:"ं",ḥ:"ः","'":"ऽ","’":"ऽ"};
  const KEYS = [...Object.keys(IV), ...Object.keys(CN), ...Object.keys(OT)].sort((a, b) => b.length - a.length), TOK = new RegExp(KEYS.map(k => k.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).join("|"), "g");
  const C = {"क":"k","ख":"kʰ","ग":"g","घ":"gʱ","ङ":"ŋ","च":"ts","छ":"tsʰ","ज":"dz","झ":"dzʱ","ञ":"ɲ","ट":"ʈ","ठ":"ʈʰ","ड":"ɖ","ढ":"ɖʱ","ण":"ɳ","त":"t̪","थ":"t̪ʰ","द":"d̪","ध":"d̪ʱ","न":"n","प":"p","फ":"pʰ","ब":"b","भ":"bʱ","म":"m","य":"j","र":"r","ल":"l","ळ":"l","व":"w","श":"ʃ","ष":"ʃ","स":"s","ह":"h"};
  const NUKTA = {"ड":"ɽ","ढ":"ɽʱ"}, NP = {"क":"ŋ","ख":"ŋ","ग":"ŋ","घ":"ŋ","च":"n","छ":"n","ज":"n","झ":"n","ट":"n","ठ":"n","ड":"n","ढ":"n","त":"n","थ":"n","द":"n","ध":"n","प":"m","फ":"m","ब":"m","भ":"m"};
  const LONG = {"आ":["a"],"ा":["a"],"ई":["i"],"ी":["i"],"ऊ":["u"],"ू":["u"],"ॄ":["r","i"],"ॠ":["r","i"]}, SHORT = {"अ":"ʌ","इ":"i","उ":"u","ए":"e","ओ":"o","ि":"i","ु":"u","े":"e","ो":"o"};
  const DIPH = {"ऐ":["ʌ","i"],"ै":["ʌ","i"],"औ":["ʌ","u"],"ौ":["ʌ","u"],"ऋ":["r","i"],"ृ":["r","i"],"ऌ":["l","r","i"],"ॢ":["l","r","i"]};
  const LONG_ORTH = new Set([..."आाईीऊूएेऐैओोऔौॠॄ"]), V = new Set(["ʌ","a","i","u","e","o"]), VEDIC = /[\u0951-\u0954\u1cd0-\u1cff]/g, VEDIC_KEEP = /[\u0953\u0954\u1cd0-\u1cff]/g, MARKS = ["\u0951", "\u0952"], HAL = "्";
  const isV = p => V.has(p[0]) && p.length <= 2;
  function iastToDeva(text) {
    let t = text.normalize("NFC").replace(/(?<=[aeiouāīūṛṝ])ḷ(?=[aeiouāīūṛṝ])/gi, "ḻ");
    if (!/[āīūṛṝḷḻṅñṭḍṇśṣṃṁḥ]/.test(t)) t = t.replace(/aa/g, "ā").replace(/ii/g, "ī").replace(/uu/g, "ū").replace(/sh/g, "ś");
    const word = w => { let out = "", pc = false; for (const k of w.toLowerCase().match(TOK) || []) { if (k in CN) { out += (pc ? HAL : "") + CN[k]; pc = true; } else if (k in IV) { out += pc ? MT[k] : IV[k]; pc = false; } else if (k in OT) { out += OT[k]; pc = false; } } return out; };
    return t.replace(/[A-Za-zāīūṛṝḷḻṅñṭḍṇśṣṃṁḥ'’]+/g, m => word(m));
  }
  const OM_RE = /ओ(?:[३3]म्?|म्|ं(?![क-ह]))/g;
  const RV111 = [["अग्निमीळे","अ॒ग्निमी॑ळे"],["पुरोहितं","पु॒रोहि॑तं"],["यज्ञस्य","य॒ज्ञस्य॑"],["देवमृत्विजम्","दे॒वमृ॒त्विज॑म्"],["होतारं","होता॑रं"],["रत्नधातमम्","रत्न॒धात॑मम्"]];
  const restoreRv = t => (t.includes("अग्निमीळे") && t.includes("पुरोहित") && !t.includes("\u0951")) ? RV111.reduce((s, [k, v]) => s.replaceAll(k, v), t) : t;
  function wordPhones(w, keep = false) {
    w = w.normalize("NFC").replace(keep ? VEDIC_KEEP : VEDIC, "").replace(OM_RE, "ॐ").replace(/ज्ञ/g, "ग्ञ"); const out = []; let i = 0; const n = w.length;
    const vb = () => { for (let k = out.length - 1; k >= 0; k--) if (V.has(out[k])) return out[k]; return "ʌ"; };
    const nas = () => { for (let k = out.length - 1; k >= 0; k--) if (V.has(out[k])) { out[k] += "\u0303"; break; } };
    while (i < n) {
      const ch = w[i];
      if (ch === "ॐ") { out.push("o", "m"); i++; }
      else if (ch in C) { let ph = C[ch]; if (w[i + 1] === "़") { ph = NUKTA[ch] || ph; i++; } out.push(ph); i++; const nx = w[i] || "";
        if (nx === HAL) i++; else if (nx in LONG) { out.push(...LONG[nx]); i++; } else if (nx in SHORT) { out.push(SHORT[nx]); i++; } else if (nx in DIPH) { out.push(...DIPH[nx]); i++; } else out.push("ʌ"); }
      else if (ch in LONG) { out.push(...LONG[ch]); i++; } else if (ch in SHORT) { out.push(SHORT[ch]); i++; } else if (ch in DIPH) { out.push(...DIPH[ch]); i++; }
      else if (ch === "ँ") { nas(); i++; }
      else if (ch === "ं") { const nx = w[i + 1] || ""; if (nx in NP) out.push(NP[nx]); else if (!nx) out.push("m"); else nas(); i++; }
      else if (ch === "ः") { if (out[out.length - 1] !== "h") out.push("h"); i++; }
      else if (ch === "ऽ") { out.push(vb()); i++; }
      else if (keep && MARKS.includes(ch)) { out.push(ch); i++; }
      else i++;
    }
    return out.filter((p, idx, a) => !(p === "h" && idx > 0 && a[idx - 1] === "h"));
  }
  function weights(w) {  // 'G'/'L' per syllable
    w = w.normalize("NFC").replace(VEDIC, "").replace(OM_RE, "ॐ").replace(/ॐ/g, "").replace(/ज्ञ/g, "ग्ञ"); const syls = []; let i = 0; const n = w.length;
    while (i < n) {
      const ch = w[i];
      if (ch in C) { if (w[i + 1] === "़") i++; if (syls.length && !syls[syls.length - 1][2]) syls[syls.length - 1][1]++; i++; const nx = w[i] || "";
        if (nx === HAL) i++; else if (nx in LONG || nx in SHORT || nx in DIPH) { syls.push([LONG_ORTH.has(nx), 0, 0]); i++; } else syls.push([false, 0, 0]); }
      else if (ch in LONG || ch in SHORT || ch in DIPH) { syls.push([LONG_ORTH.has(ch), 0, 0]); i++; }
      else if (ch === "ं" || ch === "ँ" || ch === "ः") { if (syls.length) syls[syls.length - 1][2] = 1; i++; } else i++;
    }
    return syls.map(([lg, c, va]) => (lg || c >= 2 || va) ? "G" : "L");
  }
  const OM = "<om>", OM_PH = ["o", "o", "m"], OM_DM = [1.55, 2.05, 1.5], OM_OFF = [0.35, 0.65, 0.20];
  const OM_PH2 = ["o", "m"], OM_DM2 = [1.55, 1.30], OM_OFF2 = [0.28, 0.14];   // v6: single resonant chanted pranava o+m (~460 ms)
  function expandOm(ph, off, dm, v2 = false) { const P = [], O = [], D = []; ph.forEach((p, i) => { if (p === OM) { P.push(...(v2 ? OM_PH2 : OM_PH)); O.push(...(v2 ? OM_OFF2 : OM_OFF)); D.push(...(v2 ? OM_DM2 : OM_DM)); } else { P.push(p); O.push(off[i]); D.push(dm ? dm[i] : 1.0); } }); return { ph: P, off: O, dm: D }; }
  function textToPhones(text, addEnd = true, keep = false, verseOpt = null) {
    const t = restoreRv(iastToDeva(text).replace(OM_RE, "ॐ").replace(/[\u200d\u200c]/g, "")), segs = (t.match(/[^।॥\n]+[।॥]?|[।॥]/g) || []).filter(s => s.trim()), verse = verseOpt !== null ? verseOpt : (/॥/.test(t) || segs.length >= 2); let ph = [];
    for (const seg of segs) {
      const toks = seg.replace(/[।॥]/g, " ").match(/[\u0900-\u097F]+/g) || [], words = [], omBefore = {}; for (const w of toks) { const w2 = w.replace(/ॐ/g, ""); if (w.includes("ॐ")) omBefore[words.length] = (omBefore[words.length] || 0) + 1; if (w2) words.push(w2); } const omTail = omBefore[words.length] || 0; delete omBefore[words.length]; const sy = words.map(w => weights(w).length), tot = sy.reduce((a, b) => a + b, 0), marks = {};
      const plan = tot >= 28 && tot <= 36 ? [[8, ","], [16, ";"], [24, ","]] : tot >= 14 && tot <= 18 ? [[8, ","]] : [];
      if (verse && words.length > 1) { const cum = []; sy.reduce((a, b, k) => (cum[k] = a + b), 0); for (const [q, m] of plan) { let bk = 0; for (let k = 0; k < words.length - 1; k++) if (Math.abs(cum[k] - q) < Math.abs(cum[bk] - q)) bk = k; if (Math.abs(cum[bk] - q) <= 3 && bk < words.length - 1) marks[bk] = marks[bk] === ";" ? ";" : m; } }
      words.forEach((w, k) => { if (k in omBefore) ph.push(OM, ",", "|"); const p = wordPhones(w, keep); if (p.length) ph.push(...p, "|"); if (k in marks) { if (ph[ph.length - 1] === "|") ph.pop(); ph.push(marks[k], "|"); } });
      let end = /॥/.test(seg) || !verse ? "." : ";"; const q = seg.match(/[?!]/g); if (q) end = q[q.length - 1];
      if (ph[ph.length - 1] === "|") ph.pop(); if (omTail) ph.push(...(words.length ? [",", OM] : [OM])); if (words.length || omTail) ph.push(end, "|");
    }
    if (ph[ph.length - 1] === "|") ph.pop(); if (ph[ph.length - 1] === ";") ph[ph.length - 1] = ".";
    if (addEnd && ![".", "?", "!"].includes(ph[ph.length - 1])) ph.push(".");
    return ph;
  }
  const CAD = { ",": 0.5, ";": 0.3, ".": -0.7, "?": 0.8, "!": -0.2 };
  function svaraOffsets(ph) {  // port of sanskrit.svara_offsets: udatta/svarita +1.30 (+0.35 pracaya), anudatta -1.75, pada-final laya melody, syllable consonants +-0.72
    const out = [], off = [], hasMarks = ph.some(p => MARKS.includes(p));
    for (const p of ph) {
      if (MARKS.includes(p)) { let j = out.length - 1; while (j >= 0 && !isV(out[j])) j--;
        if (j >= 0) { const v = p === MARKS[0] ? 1.30 : -1.75; off[j] = v; if (j > 0 && out[j - 1] === out[j] && out[j] !== "ʌ") off[j - 1] = v; } continue; }
      out.push(p); off.push(0);
    }
    const isShloka = out.includes(";") || ph.includes(OM);
    if (!hasMarks && !isShloka) return { ph: out, off };
    if (!hasMarks && isShloka) {
      let pSt = 0;
      out.forEach((p, i) => {
        if (p in CAD) {
          const vs = []; for (let j = pSt; j < i; j++) if (isV(out[j])) vs.push(j);
          const nv = vs.length;
          vs.forEach((j, k) => {
            const arch = 0.38 * Math.sin(Math.PI * (k + 0.5) / Math.max(nv, 1)) - 0.10;
            let nxtC = 0; for (let m = j + 1; m < Math.min(out.length, j + 4); m++) if (!isV(out[m]) && out[m] !== "|" && !(out[m] in CAD) && out[m] !== OM) nxtC++;
            const guru = ["a", "e", "o"].includes(out[j]) || nxtC >= 2;
            off[j] = Math.round((arch + (guru ? 0.16 : -0.12)) * 10000) / 10000;
          });
          pSt = i + 1;
        }
      });
    }
    let st = 0;
    out.forEach((p, i) => { if (p === "|" || p in CAD) st = 0; else if (isV(p)) { if (off[i] > 0.5) st = 1; else if (off[i] < -0.5) st = 0; else if (st === 1 && off[i] === 0) off[i] = 0.35; } });
    out.forEach((p, i) => { if (p in CAD) { const js = []; for (let j = i - 1; j > Math.max(-1, i - 7) && js.length < 2; j--) if (isV(out[j])) js.push(j); js.forEach((j, n) => { off[j] += CAD[p] * (n === 0 ? 1 : 0.4) * (hasMarks ? 1 : 0.65); }); } });
    const sm = off.slice();
    out.forEach((p, i) => { if (!isV(p) && p !== "|" && !(p in CAD)) { const lv = i > 0 ? off[i - 1] : 0, rv = i + 1 < out.length ? off[i + 1] : 0; sm[i] = (lv < -0.8 || rv < -0.8) ? -0.72 : (lv > 0.8 || rv > 0.8) ? 0.72 : 0.5 * (lv + rv); } });
    return { ph: out, off: sm };
  }
  const textToPhonesAccRaw = (text, addEnd = true) => svaraOffsets(textToPhones(text, addEnd, true)), textToPhonesAcc = (text, addEnd = true, v2 = false) => { const r = textToPhonesAccRaw(text, addEnd); return expandOm(r.ph, r.off, null, v2); };
  function layaScales(text, addEnd = true, v2 = false) {
    const t = iastToDeva(text).replace(OM_RE, "ॐ").replace(/[\u200d\u200c]/g, ""), words = (t.replace(/[।॥]/g, " ").match(/[\u0900-\u097F]+/g) || []).map(w => w.replace(/ॐ/g, "")).filter(Boolean), wSyls = words.flatMap(w => weights(w));
    const rawPh = textToPhones(text, addEnd, true, null), hasMarks = rawPh.some(p => MARKS.includes(p)), isShloka = !hasMarks && (rawPh.includes(";") || rawPh.includes(OM));
    const { ph, off } = svaraOffsets(rawPh), n = ph.length, dm = new Array(n).fill(1.0);
    if (v2) {
      let siV2 = 0, iV2 = 0;
      while (iV2 < n) {
        const p = ph[iV2];
        if (isV(p)) {
          const diph = p === "ʌ" && iV2 + 1 < n && (ph[iV2 + 1] === "i" || ph[iV2 + 1] === "u"), g = siV2 < wSyls.length ? wSyls[siV2] === "G" : (["a", "e", "o"].includes(p) || diph || p.length > 1);
          siV2++;
          if (diph) { dm[iV2] = 1.02; dm[iV2 + 1] = 1.02; iV2 += 2; }
          else { dm[iV2] = g ? 1.04 : 0.97; iV2 += 1; }
        } else { if (p === "," || p === ";") dm[iV2] = 1.12; iV2++; }
      }
      return expandOm(ph, off, dm, true);
    }
    let si = 0, i = 0;
    while (i < n) {
      const p = ph[i];
      if (isV(p)) {
        const diph = p === "ʌ" && i + 1 < n && (ph[i + 1] === "i" || ph[i + 1] === "u"), g = si < wSyls.length ? wSyls[si] === "G" : (["a", "e", "o"].includes(p) || diph || p.length > 1);
        si++;
        if (diph) { const dVal = isShloka ? 1.08 : (v2 ? 1.00 : 1.04); dm[i] = dVal; dm[i + 1] = dVal; i += 2; }
        else { dm[i] = isShloka ? (g ? 1.20 : 0.90) : (v2 ? 1.00 : (g ? 1.08 : 0.98)); if (Math.abs(off[i]) >= 0.8) dm[i] = Math.max(dm[i], v2 ? 1.12 : 1.35); i += 1; }
      } else { if (p === "h" && (i + 1 === n || ph[i + 1] === "|" || ph[i + 1] in CAD)) dm[i] = (v2 && !isShloka) ? 0.85 : (v2 ? 0.75 : (isShloka ? 1.18 : 1.12)); i++; }
    }
    return expandOm(ph, off, dm, v2);
  }
  const SLP_SYMBOLS = ["<pad>","<bos>","<eos>","<sp>","<unk>",",","?","!","|","||","<yati>","a","A","i","I","u","U","f","e","E","o","O","k","g","N","c","j","Y","w","q","R","t","d","n","p","b","m","y","r","l","v","S","z","s","h","M","H"];
  const SLP_SYM2ID = Object.fromEntries(SLP_SYMBOLS.map((s, i) => [s, i])), SLP_VOWELS = new Set(["a","A","i","I","u","U","f","e","E","o","O"]), SLP_PHONES = new Set(SLP_SYMBOLS.slice(11));
  const _S_IND = Object.fromEntries([..."अआइईउऊऋॠऌॡएऐओऔ"].map((c, i) => [c, "aAiIuUfFxXeEoO"[i]])), _S_SGN = Object.fromEntries([..."ािीुूृॄॢॣेैोौ"].map((c, i) => [c, "AiIuUfFxXeEoO"[i]]));
  const _S_CON = Object.fromEntries([..."कखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसहळ"].map((c, i) => [c, "kKgGNcCjJYwWqQRtTdDnpPbBmyrlvSzshl"[i]]));
  const _S_RAW_V = new Set([..."aAiIuUfFxXeEoO"]), _S_LONG = new Set([..."AIUFXeEoO"]);
  const _S_WRITTEN = { F: ["f"], x: ["l", "f"], X: ["l", "f"], K: ["k","h"], G: ["g","h"], C: ["c","h"], J: ["j","h"], W: ["w","h"], Q: ["q","h"], T: ["t","h"], D: ["d","h"], P: ["p","h"], B: ["b","h"] };
  const _S_NASAL = { k:"N",K:"N",g:"N",G:"N",N:"N", c:"Y",C:"Y",j:"Y",J:"Y",Y:"Y", w:"R",W:"R",q:"R",Q:"R",R:"R", t:"n",T:"n",d:"n",D:"n",n:"n", p:"m",P:"m",b:"m",B:"m",m:"m" };
  const _S_MARKS = { "।":"|", "|":"|", ".":"|", "॥":"||", ",":",", ";":",", "…":",", "–":",", "—":",", "?":"?", "!":"!" };
  const _S_READ = new Set([...Object.keys(_S_IND), ...Object.keys(_S_SGN), ...Object.keys(_S_CON), ..."ंँः\u1cf5\u1cf6़्ऽॐ "]);
  function phonemizeSlp1(text) {
    let t = String(text).normalize("NFC"); if (/[A-Za-z]{2,}/.test(t)) t = iastToDeva(t);
    t = t.replace(/।।|\|\|/g, "॥").replace(/\([^)]*\)/g, " ").replace(/([।॥])\s*[0-9०-९]+(?:\s*[/.\-]\s*[0-9०-९]+)*\s*(?=[।॥]|$)/g, "$1").replace(/\s-+\s/g, " , ").replace(/-/g, "").replace(/:/g, " , ").replace(/\s+/g, " ").trim();
    const hasAcc = /[\u0951\u0952]/.test(t), pieces = [];
    for (const pc of t.split(/\s+|([।|.॥,;…–—?!])/)) {
      if (!pc) continue; if (pc in _S_MARKS) { pieces.push(_S_MARKS[pc]); continue; }
      const phs = [], accs = []; let openC = false, stray = false;
      for (const ch of [...(pc + " ")]) {
        if (ch === "\u0951" || ch === "\u0952") { const v = ch === "\u0951" ? 1 : -1; if (openC) { phs.push("a"); accs.push(v); openC = false; } else if (accs.length) accs[accs.length - 1] = v; continue; }
        if (/[\u200b-\u200d\u0953\u0954]/.test(ch)) continue;
        if (!_S_READ.has(ch)) { if (openC) { phs.push("a"); accs.push(0); openC = false; } if (!stray) { phs.push("<unk>"); accs.push(0); stray = true; } continue; }
        stray = false;
        if (ch === "्") openC = false; else if (ch === "़") {} else if (ch in _S_SGN) { phs.push(_S_SGN[ch]); accs.push(0); openC = false; }
        else { if (openC) { phs.push("a"); accs.push(0); } openC = ch in _S_CON; if (openC) { phs.push(_S_CON[ch]); accs.push(0); } else if (ch in _S_IND) { phs.push(_S_IND[ch]); accs.push(0); } else if (ch === "ं" || ch === "ँ") { phs.push("M"); accs.push(0); } else if (ch === "ः" || ch === "\u1cf5" || ch === "\u1cf6") { phs.push("H"); accs.push(0); } else if (ch === "ऽ") { for (let ki = phs.length - 1; ki >= 0; ki--) { if (_S_RAW_V.has(phs[ki])) { phs[ki] = ({a:"A",i:"I",u:"U",f:"F",x:"X"})[phs[ki]] || phs[ki]; break; } } } else if (ch === "ॐ") { phs.push("o", "m"); accs.push(0, 0); } }
      }
      if (phs.length) pieces.push([phs, accs]);
    }
    pieces.forEach((it, idx) => { if (typeof it === "string") return; const phs = it[0]; phs.forEach((p, i) => { if (p === "M") { if (i + 1 < phs.length) phs[i] = _S_NASAL[phs[i + 1]] || "M"; else if (idx + 1 === pieces.length || typeof pieces[idx + 1] === "string") phs[i] = "m"; } }); });
    const syms = ["<bos>"], wt = [0], off = [0]; let st = 0;
    while (st < pieces.length) {
      if (typeof pieces[st] === "string") { if (syms[syms.length - 1] !== pieces[st]) { syms.push(pieces[st]); wt.push(0); off.push(0); } st++; continue; }
      let en = st; while (en < pieces.length && typeof pieces[en] !== "string") en++;
      const segW = pieces.slice(st, en), flatP = segW.flatMap(x => x[0]), wList = new Array(flatP.length).fill(0), vIdx = [];
      flatP.forEach((p, i) => { if (_S_RAW_V.has(p)) vIdx.push(i); });
      vIdx.forEach((vi, ni) => { const btw = flatP.slice(vi + 1, ni + 1 < vIdx.length ? vIdx[ni + 1] : flatP.length); wList[vi] = (_S_LONG.has(flatP[vi]) || btw.length >= 2 || btw.includes("M") || btw.includes("H")) ? 2 : 1; });
      let yatiK = -1;
      if (/॥/.test(t) && vIdx.length >= 14 && vIdx.length <= 18 && segW.length > 1) {
        const wSy = segW.map(([wP]) => wP.filter(p => _S_RAW_V.has(p)).length), cum = []; wSy.reduce((a, b, k) => (cum[k] = a + b), 0);
        let bk = 0; for (let k = 0; k < segW.length - 1; k++) if (Math.abs(cum[k] - 8) < Math.abs(cum[bk] - 8)) bk = k;
        if (Math.abs(cum[bk] - 8) <= 2) yatiK = bk;
      }
      let wi = 0;
      segW.forEach(([wP, wA], wk) => {
        const last = syms[syms.length - 1]; if (SLP_PHONES.has(last) || last === "<unk>") { syms.push("<sp>"); wt.push(0); off.push(0); }
        let prac = 0;
        for (let i = 0; i < wP.length; i++) {
          const p = wP[i], ac = wA[i], lv = wList[wi++];
          for (const item of (_S_WRITTEN[p] || [p])) {
            syms.push(item); const isV = SLP_VOWELS.has(item); wt.push(isV ? lv : 0);
            if (isV) { if (ac === 1) { off.push(1.08); prac = 1; } else if (ac === -1) { off.push(-1.15); prac = 0; } else if (prac === 1) off.push(0.38); else off.push(0); } else off.push(0);
          }
        }
        if (wk === yatiK) { syms.push("<yati>"); wt.push(0); off.push(0); }
      });
      st = en;
    }
    syms.push("<eos>"); wt.push(0); off.push(0);
    if (hasAcc || syms.includes("<yati>")) {
      syms.forEach((s, i) => {
        let cV = null;
        if (hasAcc && [",","|","||","?","!"].includes(s)) cV = s === "," ? 0.5 : (s === "|" || s === "||" ? -0.7 : (s === "?" ? 0.8 : -0.2));
        else if (!hasAcc && ["<yati>","|","||"].includes(s)) cV = s === "<yati>" ? 0.35 : (s === "|" ? 0.20 : -0.40);
        if (cV === null) return;
        const js = []; for (let j = i - 1; j > Math.max(-1, i - 7) && js.length < 2; j--) if (SLP_VOWELS.has(syms[j])) js.push(j);
        js.forEach((j, nj) => { off[j] += cV * (nj === 0 ? 1 : 0.4); });
      });
      const sm = off.slice(); syms.forEach((s, i) => { if (SLP_PHONES.has(s) && !SLP_VOWELS.has(s)) sm[i] = 0.5 * ((i > 0 ? off[i - 1] : 0) + (i + 1 < off.length ? off[i + 1] : 0)); }); off.splice(0, off.length, ...sm);
    }
    return { ids: syms.map(s => SLP_SYM2ID[s] ?? 4), weight: wt, off, syms };
  }
  window.SA = { svaraOffsets, textToPhonesAcc, layaScales, phonemizeSlp1, iastToDeva, wordPhones, weights, textToPhones };
})();
