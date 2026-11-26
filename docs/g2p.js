// Sakhiya front-end in JS: number expansion, Roman->Devanagari (lexicon skeleton snap + Tharu dialect lexicons), Nepali G2P.
// Faithful port of sakhiya/{numbers,translit,g2p}.py (tables come from data/frontend.json).
const SK = {};
const HAL = "्", ANU = "ं", CHAN = "ँ", VIS = "ः", NUK = "़";
const C = {"क":"k","ख":"kʰ","ग":"g","घ":"gʱ","ङ":"ŋ","च":"ts","छ":"tsʰ","ज":"dz","झ":"dzʱ","ञ":"ɲ","ट":"ʈ","ळ":"l","ठ":"ʈʰ","ड":"ɖ","ढ":"ɖʱ","ण":"ɳ","त":"t̪","थ":"t̪ʰ","द":"d̪","ध":"d̪ʱ","न":"n","प":"p","फ":"pʰ","ब":"b","भ":"bʱ","म":"m","य":"j","र":"r","ल":"l","व":"w","श":"ʃ","ष":"ʃ","स":"s","ह":"h"};
const NUKTA = {"ड":"ɽ","ढ":"ɽʱ","ज":"dz","फ":"pʰ","क":"k","ख":"kʰ","ग":"g"};
const IV = {"अ":"ʌ","आ":"a","इ":"i","ई":"i","उ":"u","ऊ":"u","ए":"e","ओ":"o","ऑ":"o"}, IV2 = {"ऐ":["ʌ","i"],"औ":["ʌ","u"],"ऋ":["r","i"]};
const MATRA = {"ा":"a","ि":"i","ी":"i","ु":"u","ू":"u","े":"e","ो":"o","ॉ":"o"}, MATRA2 = {"ै":["ʌ","i"],"ौ":["ʌ","u"],"ृ":["r","i"]};
const NASAL = {"क":"ŋ","ख":"ŋ","ग":"ŋ","घ":"ŋ","च":"n","छ":"n","ज":"n","झ":"n","ट":"n","ठ":"n","ड":"n","ढ":"n","त":"n","थ":"n","द":"n","ध":"n","प":"m","फ":"m","ब":"m","भ":"m"};
// romanisation
const RC = {chh:"छ",ch:"च",kh:"ख",gh:"घ",jh:"झ",th:"थ",dh:"ध",ph:"फ",bh:"भ",sh:"श",k:"क",g:"ग",c:"क",j:"ज",t:"त",d:"द",n:"न",p:"प",b:"ब",m:"म",y:"य",r:"र",l:"ल",w:"व",v:"व",s:"स",h:"ह",f:"फ",z:"ज",q:"क",x:"छ"};
const RV = ["aa","ai","au","ee","ii","oo","uu","a","i","u","e","o"];
const IND = {a:"अ",aa:"आ",i:"इ",ii:"ई",ee:"ई",u:"उ",uu:"ऊ",oo:"ऊ",e:"ए",ai:"ऐ",o:"ओ",au:"औ"}, MAT = {a:"",aa:"ा",i:"ि",ii:"ी",ee:"ी",u:"ु",uu:"ू",oo:"ू",e:"े",ai:"ै",o:"ो",au:"ौ"};
const KEYMAP = {"ʈ":"t̪","ʈʰ":"t̪ʰ","ɖ":"d̪","ɖʱ":"d̪ʱ","ɳ":"n","ŋ":"n","ɲ":"n","ʃ":"s","w":"b","ɽ":"r","ɽʱ":"r","ɔ":"o","ɛ":"e","z":"dz","f":"pʰ"};
const DIG = {"०":0,"१":1,"२":2,"३":3,"४":4,"५":5,"६":6,"७":7,"८":8,"९":9};
let F = null;
SK.load = async (url = "data/frontend.json") => { F = await (await fetch(url)).json(); SK.F = F; return F; };

// ---------- numbers
const BIG = [[1e11,"खर्ब"],[1e9,"अर्ब"],[1e7,"करोड"],[1e5,"लाख"],[1000,"हजार"],[100,"सय"]];
const numWords = n => { if (n < 100) return F.num_u[n]; for (const [v, w] of BIG) if (n >= v) { const q = Math.floor(n / v), r = n % v; return numWords(q) + " " + w + (r ? " " + numWords(r) : ""); } return ""; };
const digitsWords = s => [...s].map(c => F.num_u[+c]).join(" ");
const numTok = t => { t = t.replace(/,/g, ""); if (t.includes(".")) { const [a, b] = t.split("."); return numWords(+(a || 0)) + " दशमलव " + digitsWords(b); } if (t.length > 1 && t[0] === "0") return digitsWords(t); if (t.length > 13) return digitsWords(t); return numWords(+t); };
const NUM = "\\d[\\d,]*(?:\\.\\d+)?";
function expandNumbers(text) {
  let t = text.replace(/[०-९]/g, c => DIG[c]);
  t = t.replace(/\b(\d{4})[-/](\d{1,2})[-/](\d{1,2})\b/g, (m, y, mo, d) => (mo >= 1 && mo <= 12 && d >= 1 && d <= 32) ? `${numWords(+y)} साल ${F.months[mo - 1]} ${numWords(+d)} गते` : m);
  t = t.replace(/\b(\d{1,2}):(\d{2})\b/g, (m, h, mi) => +mi ? `${numWords(+h)} बजेर ${numWords(+mi)} मिनेट` : `${numWords(+h)} बजे`);
  t = t.replace(new RegExp(`(?:रु\\.?|रू\\.?|Rs\\.?|NPR)\\s*(${NUM})`, "g"), (m, n) => numTok(n) + " रुपैयाँ");
  t = t.replace(new RegExp(`(${NUM})\\s*%`, "g"), (m, n) => numTok(n) + " प्रतिशत");
  t = t.replace(new RegExp(`(${NUM})\\s*(?:किमी|km)\\b`, "g"), (m, n) => numTok(n) + " किलोमिटर");
  t = t.replace(new RegExp(NUM, "g"), m => numTok(m.replace(/,+$/, "")));
  return t.replace(/\b([A-Z]{2,5})\b/g, (m, a) => [...a.toLowerCase()].map(c => F.letters[c]).join(" "));
}

// ---------- roman -> devanagari
function rtok(r) { r = r.toLowerCase(); const out = []; let i = 0; while (i < r.length) { let hit = false; for (const L of [3, 2, 1]) { const s = r.substr(i, L); if ((L === 3 && s === "chh") || (L <= 2 && (s in RC || RV.includes(s))) || (L === 1 && (s in RC || RV.includes(s)))) { out.push(s); i += L; hit = true; break; } } if (!hit) i++; } return out; }
function romanKey(r) {
  const toks = rtok(r.replace(/~/g, "")), k = [];
  toks.forEach((t, idx) => {
    if (t in RC) { const ph = C[RC[t]]; k.push(KEYMAP[ph] || ph); }
    else if (["i","ii","ee"].includes(t)) k.push("i"); else if (["u","uu","oo"].includes(t)) k.push("u");
    else if (t === "au") k.push("au"); else if (t === "e") k.push("e"); else if (t === "ai") k.push("ai"); else if (t === "o") k.push("o");
    else if ((t === "a" || t === "aa") && (idx === 0 || idx === toks.length - 1)) k.push("a");
  });
  return k.join(" ");
}
function rules(r) { let out = "", pc = false; for (const t of rtok(r)) { if (t in RC) { if (pc) out += HAL; out += RC[t]; pc = true; } else { out += pc ? MAT[t] : IND[t]; pc = false; } } return out; }
function tharu(lang) { return { thr: "thr_dangaura", thr_dangaura: "thr_dangaura", thr_rana: "thr_rana", thr_chitwania: "thr_chitwania", thr_chit: "thr_chitwania" }[lang]; }
function lookup(w, k, t) { if (t) { const h = F.thr[t].exact[w] || F.thr[t].key[k]; if (h) return h; } if (F.direct[w]) return F.direct[w]; return k ? F.ne[k] : undefined; }
function romanToDeva(word, lang = "ne") {
  if (/[\u0900-\u097F]/.test(word)) return word;
  const hasTilde = word.includes("~"), t = tharu(lang), w = word.toLowerCase().replace(/[^a-z]/g, ""), k = romanKey(w);
  const split = () => { for (const sf of Object.keys(F.sfx).sort((a, b) => b.length - a.length)) if (w.endsWith(sf) && w.length - sf.length >= 2) { const st = w.slice(0, -sf.length); let h = lookup(st, romanKey(st), t); if (h) { if (hasTilde && word.slice(0, -sf.length).includes("~") && !h.endsWith("ँ")) h += "ँ"; return h + F.sfx[sf]; } } };
  let out = (t ? (F.thr[t].exact[w] || F.thr[t].exact[word.toLowerCase()] || split() || lookup(w, k, t)) : (lookup(w, k, t) || split())) || rules(w);
  if (hasTilde && !out.includes("ँ") && !out.includes("ं")) out += "ँ";
  return out;
}

// ---------- G2P
function parse(w) {
  const out = []; let i = 0; const n = w.length;
  const nasalize = () => {
    for (let k = out.length - 1; k >= 0; k--) if ("VS".includes(out[k][1])) {
      const ph = out[k][0] === "ʌ?" ? "ʌ" : out[k][0];
      out[k] = [ph.endsWith("̃") ? ph : ph + "̃", "V"];
      if (["i","u","ĩ","ũ"].includes(ph) && k >= 1 && "VS".includes(out[k - 1][1])) {
        const p0 = out[k - 1][0] === "ʌ?" ? "ʌ" : out[k - 1][0];
        if (["ʌ","a","o","e"].includes(p0)) out[k - 1] = [p0 + "̃", "V"];
      }
      return;
    }
  };
  while (i < n) {
    const ch = w[i];
    if (ch in C) {
      let ph = C[ch]; if (w[i + 1] === NUK) { ph = NUKTA[ch] || ph; i++; }
      out.push([ph, "C"]); i++;
      if (w[i] === HAL) i++;
      else if (w[i] in MATRA) { out.push([MATRA[w[i]], "V"]); i++; }
      else if (w[i] in MATRA2) { MATRA2[w[i]].forEach(p => out.push([p, "V"])); i++; }
      else out.push(["ʌ?", "S"]);
    } else if (ch in IV) {
      const vPh = IV[ch];
      if (i > 0 && out.length && out[out.length - 1][1] === "V" && out[out.length - 1][0].replace("̃", "") === vPh) out.push(["|", "C"]);
      out.push([vPh, "V"]); i++;
    }
    else if (ch in IV2) {
      let seq = IV2[ch];
      if ((ch === "औ" || ch === "ऐ") && out.length && out[out.length - 1][1] === "V" && ["a", "ã"].includes(out[out.length - 1][0])) seq = seq.slice(1);
      seq.forEach(p => out.push([p, p === "r" ? "C" : "V"])); i++;
    }
    else if (ch === CHAN) { nasalize(); i++; }
    else if (ch === ANU) { const nx = w[i + 1] || ""; if (nx in NASAL) { if (out.length && out[out.length - 1][1] === "S") out[out.length - 1] = ["ʌ", "V"]; out.push([NASAL[nx], "C"]); } else nasalize(); i++; }
    else if (ch === "ॐ") { out.push(["o", "V"], ["m", "C"]); i++; }
    else if (ch === VIS) { out.push(["h", "C"]); i++; }
    else i++;
  }
  return out;
}
const KEEP_FINAL = new Set(["आज","बाट","सँग","संग","तल","तर","तिर","जब","तब","अब","अझ","वर","पर","सुख","दुख","दुःख","बाहिर","भित्र","हेर","गर","भन","सुन","पढ","लेख","राख","छैन","होइन","सय","समय","निर्णय","प्राय","समुदाय","उपाय","बिहान","सकल","रुन","पख","साँझपख","विषय","परिचय","आशय","हृदय","महोदय","न्याय","अध्याय","खान","जान","लिन","हुन","रहन","आउन","पाउन","गाउन","लाउन","धाउन","छाउन","गरेर","भनेर","हेरेर","आएर","गएर","खाएर","लिएर","दिएर","बसेर","सुनेर","पढेर"]);
const G2P_LEX = {
  "काठमाडौँ":["k","a","ʈʰ","m","a","ɖ","ʌ̃","ũ"],"काठमाडौं":["k","a","ʈʰ","m","a","ɖ","ʌ̃","ũ"],"माछापुच्छ्रे":["m","a","tsʰ","a","p","u","ts","tsʰ","r","e"],"साँझपख":["s","ã","dzʱ","ʌ","p","ʌ","kʰ","ʌ"],"सुनजस्तै":["s","u","n","dz","ʌ","s","t̪","ʌ","i"],"तहले":["t̪","ʌ","h","ʌ","l","e"],"चम्किरहेको":["ts","ʌ","m","k","i","r","ʌ","h","e","k","o"],"खेतबारीमा":["kʰ","e","t̪","b","a","r","i","m","a"],"थामिएपछि":["t̪ʰ","a","m","i","e","p","ʌ","tsʰ","i"],"पहाडको":["p","ʌ","h","a","ɖ","k","o"],"टुप्पोबाट":["ʈ","u","p","o","b","a","ʈ","ʌ"],"प्रयोगकर्ता":["p","r","ʌ","j","o","g","k","ʌ","r","t̪","a"],"एप्लिकेसन":["e","p","l","i","k","e","s","ʌ","n"],"मोबाइल":["m","o","b","a","i","l"],"वृद्धि":["b","r","i","d̪","d̪ʱ","i"],"हजार":["h","ʌ","dz","a","r","ʌ"],"छब्बीस":["tsʰ","ʌ","b","b","i","s"],"छब्बिस":["tsʰ","ʌ","b","b","i","s"],"पन्ध्र":["p","ʌ","n","d̪ʱ","r","ʌ"],"आएका":["a","j","e","k","a"],"झरना":["dzʱ","ʌ","r","n","a"],"झरनाको":["dzʱ","ʌ","r","n","a","k","o"],"शिविर":["ʃ","i","b","i","r"],"बचा":["b","ʌ","ts","a"],"बच्चा":["b","ʌ","ts","ts","a"],"वाह":["w","a","h"],"आआफ्नो":["a","|","a","pʰ","n","o"],"आआफ्ना":["a","|","a","pʰ","n","a"],"आआफ्नै":["a","|","a","pʰ","n","ʌ","i"],"राजधानी":["r","a","dz","d̪ʱ","a","n","i"],
  "प्रधानमन्त्री":["p","r","ʌ","d̪ʱ","a","n","m","ʌ","n","t̪","r","i"],"सगरमाथा":["s","ʌ","g","ʌ","r","m","a","t̪ʰ","a"],"सुदूरपश्चिम":["s","u","d̪","u","r","p","ʌ","ʃ","ts","i","m"],
  "लगभग":["l","ʌ","g","bʱ","ʌ","g"],"फुटबल":["pʰ","u","ʈ","b","ʌ","l"],"इन्टरनेट":["i","n","ʈ","ʌ","r","n","e","ʈ"],"बसपार्क":["b","ʌ","s","p","a","r","k"],
  "वेबसाइट":["w","e","b","s","a","i","ʈ"],"रेलवे":["r","e","l","w","e"],"देवराज":["d̪","e","w","r","a","dz"],"डडेलधुरा":["ɖ","ʌ","ɖ","e","l","d̪ʱ","u","r","a"],
  "सिलसिला":["s","i","l","s","i","l","a"],"कोबोर्ड":["k","o","b","o","r","ɖ"],"किचकवध":["k","i","ts","ʌ","k","w","ʌ","d̪ʱ"],"सुरतगढ":["s","u","r","ʌ","t̪","g","ʌ","ɖʱ"],
  "किमडाँडा":["k","i","m","ɖ","ã","ɖ","a"],"अर्घाखाँची":["ʌ","r","gʱ","a","kʰ","ã","ts","i"],"फिल्मसम्बन्धि":["pʰ","i","l","m","s","ʌ","m","b","ʌ","n","d̪ʱ","i"],"फ्रान्स":["pʰ","r","a","n","s"],"एण्ड":["e","n","ɖ"],
  "एकदम":["e","k","d̪","ʌ","m"],"कलकल":["k","ʌ","l","k","ʌ","l"],"चिरबिर":["ts","i","r","b","i","r"],"किनमेल":["k","i","n","m","e","l"],"फलफूल":["pʰ","ʌ","l","pʰ","u","l"],
  "कागजात":["k","a","g","dz","a","t̪"],"जनसंख्या":["dz","ʌ","n","s","ʌ","ŋ","kʰ","j","a"],"तापक्रम":["t̪","a","p","k","r","ʌ","m"],"अधिकतम":["ʌ","d̪ʱ","i","k","t̪","ʌ","m"],
  "अठहत्तर":["ʌ","ʈʰ","h","ʌ","t̪","t̪","ʌ","r"],"नेक्स्टजेन":["n","e","k","s","ʈ","dz","e","n"],
  "कति":["k","ʌ","t̪","i"],"कत्ति":["k","ʌ","t̪","t̪","i"],"पकै":["p","ʌ","k","ʌ","i"],"पक्कै":["p","ʌ","k","k","ʌ","i"],
  "चिटिक्क":["ts","i","ʈ","i","k","k","ʌ"],"ठ्ठयाक्कै":["ʈʰ","ʈʰ","j","a","k","k","ʌ","i"],"ठ्याक्कै":["ʈʰ","j","a","k","k","ʌ","i"],
  "सक्कली":["s","ʌ","k","k","ʌ","l","i"],"अड्डा":["ʌ","ɖ","ɖ","a"],"अड्डाका":["ʌ","ɖ","ɖ","a","k","a"],"छक्क":["tsʰ","ʌ","k","k","ʌ"],
  "सफा":["s","ʌ","pʰ","a"],"चिसो":["ts","i","s","o"],"सास":["s","a","s"],"शान्ति":["ʃ","a","n","t̪","i"],"छाता":["tsʰ","a","t̪","a"],
  "प्रविधि":["p","r","ʌ","b","i","d̪ʱ","i"],"अर्थतन्त्र":["ʌ","r","t̪ʰ","ʌ","t̪","ʌ","n","t̪","r","ʌ"],
  "अन्तर्राष्ट्रिय":["ʌ","n","t̪","ʌ","r","r","a","ʃ","ʈ","r","i","j","ʌ"],"प्रतिशत":["p","r","ʌ","t̪","i","ʃ","ʌ","t̪"],
  "डिजिटल":["ɖ","i","dz","i","ʈ","ʌ","l"],"सफ्टवेयर":["s","ʌ","pʰ","ʈ","w","e","j","ʌ","r"],"कम्प्युटर":["k","ʌ","m","p","j","u","ʈ","ʌ","r"]
};
const LOAN_NO_SCHWA = new Set(["टेक्स्ट","टेस्ट","पोस्ट","स्पीच","फास्ट","लिस्ट","कन्टेन्ट","प्रिन्ट","ब्रान्ड","वर्ल्ड","फिल्म","इन्टेलिजेन्स","एम्बुलेन्स","जेम्स","फ्रान्स","पार्क","बसपार्क","वार्ड","कार्ड","बोर्ड","एण्ड","सरपंच"]), LOAN_FINAL_CL = ["क्स्ट","स्ट","स्क","स्प","क्स","ल्ड","न्ड","न्ट","क्ट","न्स","ल्म","म्स","र्क","र्ड"];
function keepFinalSchwa(w, u, idx, nsyl) {
  if (w && (LOAN_NO_SCHWA.has(w) || LOAN_FINAL_CL.some(s => w.endsWith(s)))) return false;
  if (nsyl === 1) return true;
  if (idx >= 2 && u[idx - 1][1] === "C" && u[idx - 2][1] === "C") return true;
  if (!w) return false;
  if (w.endsWith("छ") || KEEP_FINAL.has(w) || ["बाट","सँग","संग","तिर","तल","बाहिर","भित्र","ीय","प्रिय","ष्ट्रिय","लय","दय","शय","पख"].some(s => w.endsWith(s))) return true;
  if (w.length >= 3 && ["उन","एर","ेर","दैन","एन","छैन","होइन"].some(s => w.endsWith(s))) return true;
  return false;
}
function resolve(u, w = "") {
  const nsyl = u.filter(x => "VS".includes(x[1])).length, res = [];
  u.forEach(([ph, k], idx) => {
    if (k !== "S") { res.push(ph); return; }
    const last = idx === u.length - 1;
    if (!last || keepFinalSchwa(w, u, idx, nsyl)) res.push("ʌ");
  });
  return res;
}
function honorificGlide(ph) { const out = []; ph.forEach((p, i) => { if (p === "e" && i >= 3 && ph[i - 3] === "u" && ph[i - 2] === "bʱ" && ph[i - 1] === "ʌ") out.push("j"); out.push(p); }); return out; }
function wordCore(w) { const ph = resolve(parse(w), w); return w.includes("नुभ") ? honorificGlide(ph) : ph; }
function wordToPhones(w) {
  w = w.normalize("NFC");
  if (w in G2P_LEX) return G2P_LEX[w].slice();
  for (const sfx of F.suffixes) if (w.length > sfx.length && w.endsWith(sfx) && sfx[0] in C) { const stem = w.slice(0, -sfx.length); if (stem in G2P_LEX || stem[stem.length - 1] in C || stem.endsWith("हरू") || stem.endsWith("हरु") || stem.endsWith("ज्यू") || stem.endsWith("ज्यु")) return wordToPhones(stem).concat(wordToPhones(sfx)); break; }
  return wordCore(w);
}
const PUNC = {",":",",".":".","।":".","॥":".","?":"?","!":"!",";":";",":":",","—":",","-":","};
function normalize(text, lang) { let t = text.normalize("NFC").replace(/[\u200d\u200c]/g, ""); t = expandNumbers(t); return t.replace(/[A-Za-z][A-Za-z'~]*/g, m => romanToDeva(m, lang)); }
function textToPhones(text, lang = "ne") {
  if (lang === "sa" && window.SA) return window.SA.textToPhones(text);
  const t = normalize(text, lang), ph = [], toks = t.match(/[,.?!;।॥]|[\u0900-\u0963\u0966-\u097F]+|[A-Za-z][A-Za-z'~]*|\S/gu) || [];
  for (let idx = 0; idx < toks.length; idx++) {
    const m = toks[idx];
    if (m.length === 1 && m in PUNC) { if (ph.length && ph[ph.length - 1] === "|") ph.pop(); ph.push(PUNC[m], "|"); continue; }
    if (!(m[0] >= "\u0900" && m[0] <= "\u097F")) continue;
    const p = wordToPhones(m).filter((x, i, a) => !(x === "h" && i > 0 && a[i - 1] === "h"));
    if (p.length) {
      ph.push(...p, "|");
      const nx = toks[idx + 1] || "";
      if ((m.endsWith("नुहोस्") || m.endsWith("नोस्")) && ["र", "अनि", "तर", "वा"].includes(nx)) { ph.pop(); ph.push(",", "|"); }
    }
  }
  if (ph.length && ph[ph.length - 1] === "|") ph.pop();
  if (!ph.length || ![".", "?", "!"].includes(ph[ph.length - 1])) ph.push(".");
  return ph;
}
// Tharu: only trust verified dialect-lexicon hits (exact word or stem+clitic); everything else goes to the neural transliterator.
SK.strong = (word, lang) => { const t = tharu(lang); if (!t) return undefined; const hasTilde = word.includes("~"), w = word.toLowerCase().replace(/[^a-z]/g, "");
  let h = F.thr[t].exact[word.toLowerCase()] || F.thr[t].exact[w]; if (h) return (hasTilde && !h.includes("ँ") && !h.includes("ं")) ? h + "ँ" : h;
  for (const sf of Object.keys(F.sfx).sort((a, b) => b.length - a.length)) if (w.endsWith(sf) && w.length - sf.length >= 2) { const st = w.slice(0, -sf.length); let x = F.thr[t].exact[st]; if (x) { if (hasTilde && word.slice(0, -sf.length).includes("~") && !x.endsWith("ँ")) x += "ँ"; return x + F.sfx[sf]; } } return undefined; };
const END_TOK = new Set(["|", ".", ",", "?", "!", ";"]), VOW_SET = new Set(["ʌ","a","i","u","e","o","ɛ","ɔ","ʌ̃","ã","ĩ","ũ","ẽ","õ","ɛ̃","ɔ̃"]);
const PLACE = {}; Object.entries({ ret: "ʈ ʈʰ ɖ ɖʱ", den: "t̪ t̪ʰ d̪ d̪ʱ", vel: "k kʰ g gʱ", lab: "p pʰ b bʱ", aff: "ts tsʰ dz dzʱ" }).forEach(([k, v]) => v.split(" ").forEach(x => PLACE[x] = k));
function phoneDm(ph) {
  const n = ph.length, dm = new Array(n).fill(1.0);
  for (let i = 0; i < n; i++) {
    const p = ph[i];
    if (p === ",") dm[i] = 1.25;
    else if (p === "!" || p === "?") dm[i] = 1.32;
    else if (p === "a" && i >= 1 && ph[i - 1] === "w" && i + 2 < n && ph[i + 1] === "h" && (ph[i + 2] === "!" || ph[i + 2] === "|")) {
      dm[i] = 1.38; dm[i + 1] = 0.72;
    }
  }
  return dm;
}
function textToPhonesAcc(text, lang = "ne") {
  if (lang === "sa" && window.SA) return window.SA.textToPhonesAcc(text);
  const ph = textToPhones(text, lang), n = ph.length, off = new Array(n).fill(0.0);
  const isV = p => ["ʌ","a","i","u","e","o"].includes(p[0]);
  let cStart = 0;
  for (let i = 0; i < n; i++) {
    const p = ph[i];
    if ([",", ".", "?", "!", ";"].includes(p)) {
      const vs = []; for (let j = cStart; j < i; j++) if (isV(ph[j])) vs.push(j);
      const nv = vs.length;
      if (p === "!") {
        vs.forEach((j, k) => { off[j] = Math.round((0.95 * Math.sin(Math.PI * (k + 0.5) / Math.max(nv, 1)) + 0.30) * 10000) / 10000; });
        if (nv === 1) off[vs[0]] = 1.35;
        else if (nv >= 2) { off[vs[0]] = Math.max(off[vs[0]], 1.15); off[vs[nv - 1]] = 0.55; }
      } else if (p === "?") {
        vs.forEach((j, k) => { off[j] = Math.round((-0.20 + 0.35 * (k / Math.max(nv - 1, 1))) * 10000) / 10000; });
        if (nv >= 1) off[vs[nv - 1]] = 1.25;
        if (nv >= 2) off[vs[nv - 2]] = 0.68;
      } else if (p === ",") {
        if (nv >= 1) off[vs[nv - 1]] = 0.24;
        if (nv >= 2) off[vs[nv - 2]] = 0.12;
      }
      cStart = i + 1;
    }
  }
  const sm = off.slice();
  for (let i = 0; i < n; i++) {
    const p = ph[i];
    if (![",", ".", "?", "!", ";", "|"].includes(p) && !isV(p)) {
      const lv = i > 0 ? off[i - 1] : 0.0, rv = i + 1 < n ? off[i + 1] : 0.0;
      sm[i] = Math.round(0.5 * (lv + rv) * 10000) / 10000;
    }
  }
  return { ph, off: sm, dm: phoneDm(ph) };
}
SK.expand = (text) => expandNumbers(text.normalize("NFC").replace(/[\u200d\u200c]/g, ""));
SK.textToPhones = textToPhones; SK.textToPhonesAcc = textToPhonesAcc; SK.phoneDm = phoneDm; SK.normalize = normalize; SK.encode = ph => ph.map(p => F.symbols.indexOf(p)).filter(i => i >= 0);
window.SK = SK;
