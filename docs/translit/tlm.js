// transliterationlm v2 — 3-layer hybrid Nepali keyboard engine (browser / Node). Mirrors model/hybrid.py 1:1.
//  L1: verified Nepali lexicon (frequency) + phonetic skeleton keys + vibhakti/plural morphology  -> candidates
//  L2: tiny ONNX seq2seq: teacher-forced rescoring of every lexicon candidate; beam search only when unsure / OOV
//  L3: sentence post-processing: Padayog/Padabiyog joining, hraswa/dirgha snapping, English keep tokens
// Usage: const T = await Transliterator.create({ort, base:'./model'});  await T.transliterate('ma ghar jaanchhu')
const MA_NEXT = new Set(['लाई', 'ले', 'को', 'का', 'बाट', 'भन्दा', 'सँग', 'सित', 'देखि', 'सम्म', 'प्रति']);   // s12: मा never takes another postposition -> म
export const JOIN_DEFAULT = ['ले', 'लाई', 'मा', 'बाट', 'देखि', 'सम्म', 'सँग', 'सँगै', 'भन्दा', 'हरू', 'हरु', 'का', 'की', 'को', 'हरूलाई', 'हरूमा', 'हरूको', 'हरूले', 'हरूका', 'हरूबाट', 'हरूसँग', 'प्रति', 'तिर', 'मै'];
// s14: informal-typing views. normRoman collapses elongation (paraaaa -> paraa); altRoman rewrites chat conventions
// (ye->ya, te->tya, v->bh) so the model can score words typed the way people really text.
export const normRoman = (w) => w.replace(/([aeiou])\1{3,}/g, '$1$1').replace(/([^aeiou])\1{3,}/g, '$1');
export function altRoman(w) {
  if (w.length < 4) return w;
  let a = w.replace(/^ye(?=[a-z]{2})/, 'ya').replace(/^te(?=[sht])/, 'tya').replaceAll('v', 'bh').replace(/irakhe/, 'irahe');
  return a;
}
const WHO_NEXT = new Set(['हो', 'हुन्', 'हुनुहुन्छ', 'होइन', 'छ', 'छन्', 'हुन्छ']);
const PRONOUN = new Set(['तपाईं', 'तिमी', 'ऊ', 'यो', 'त्यो', 'उनी', 'को']);
const TOKEN_RE = /https?:\/\/\S+|www\.\S+|[\p{L}\p{N}_.+-]+@[\p{L}\p{N}_-]+\.[\p{L}\p{N}_.-]+|[#@][\p{L}\p{N}_]+|[A-Za-z][A-Za-z0-9_']*|\d+(?:[.,]\d+)*|\s+|[\s\S]/gu;
const DEVA = /[\u0900-\u097F]/;
const DEVL = /[\u0900-\u0963\u0971-\u097F]/;
const SUBJ1P = new Set(['हामी', 'हामीले']), SUBJ1S = new Set(['म', 'मैले']), SUBJ2 = new Set(['तिमी', 'तिमीले']);
const RESET = new Set(['ऊ', 'उनी', 'उसले', 'उनले', 'यसले', 'त्यसले', 'तिनीहरू', 'उनीहरू', 'यो', 'त्यो', 'तपाईं', 'तपाईंले']);
const FSM_CONS = new Set([...'कखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसहळ']), FSM_IV = new Set([...'अआइईउऊऋएऐओऔ']), FSM_M = new Set([...'ािीुूृेैोौॉ']);
const FSM_OK = { '^': 'CV', C: 'CMHNXZV', V: 'CNXV', M: 'CNXV', H: 'C', N: 'CV', X: 'CV', Z: 'CMHN', O: 'CVO' };
/** s21: Devanagari orthotactic FSM - beam candidates must be well-formed (mirrors model/hybrid.py fsm_ok) */
function fsmOk(w) { let p = '^'; for (const c of w) { const k = FSM_CONS.has(c) ? 'C' : FSM_IV.has(c) ? 'V' : FSM_M.has(c) ? 'M' : c === '्' ? 'H' : (c === 'ं' || c === 'ँ') ? 'N' : c === 'ः' ? 'X' : c === '़' ? 'Z' : 'O'; if (!FSM_OK[p].includes(k)) return false; p = k; } return p !== '^'; }
const VERBISH = ['छु', 'छ', 'छौं', 'छौँ', 'छौ', 'छन्', 'यो', 'एँ', 'ए', 'ये', 'यौ', 'यौं', 'यौँ', 'एको', 'थियो', 'हुन्छ', 'ेर', 'दै', 'ने', 'ला', 'ली', 'इन्', 'इयो', 'ेँ'];
const PARTICLES = new Set(['र', 'तर', 'कि', 'अनि', 'भने', 'जब', 'अब', 'त', 'नि', 'पनि', 'के', 'कसरी', 'किन', 'भोलि', 'आज', 'हिजो', 'अहिले', 'फेरि', 'चाहिँ', 'बिना', 'लागि']);
const KHAN_NEXT = new Set(['man', 'jana', 'jane', 'gaye', 'gayo', 'gayau', 'gaya', 'janxu', 'jaanxu', 'janchu', 'janchhu', 'jaanchhu', 'jaanchu', 'janxa', 'janchha', 'jaanchha', 'gayeko']);
const SANGAI_NOT = ['मा', 'देखि', 'बाट', 'सम्म', 'लाई', 'भन्दा'];
const POSS = new Set(['मेरो', 'तिम्रो', 'हाम्रो', 'उसको', 'यसको', 'उनको', 'तपाईंको', 'तिनको', 'उहाँको']);
const SAK_RE = /^सक(ेको|ेका|्यो|ियो|ेँ|्यौं|ेछ|्छ|्छु|ेन)/;
const NOT_SUBJ_NEXT = new Set(['लाई', 'को', 'का', 'की', 'सँग', 'बाट', 'देखि', 'भन्दा', 'सित', 'प्रति']);
const KI_SUBJ = new Set(['तिमी', 'म', 'हामी', 'ऊ', 'उनी', 'तपाईं', 'उहाँ', 'तिमीले', 'हामीले', 'मैले', 'उनीहरू', 'तिनीहरू']);
const KI_VERB = ['यो', 'छ', 'छन्', 'हो', 'भो', 'थियो', 'छु', 'छौ', 'छौं', 'छौँ', 'एँ', 'ला', 'हुन्छ', 'यौ', 'यौं', 'यौँ', 'छिन्', 'ेँ', 'ने'];
const PAR = new Set(['पर्छ', 'पर्दैन', 'पर्‍यो', 'पर्यो', 'पर्ने', 'पर्नेछ']);
const HON = new Set(['हुन्छ', 'भयो', 'भएको', 'हुनुभयो', 'हुनुहोस्', 'हुनुहुन्छ', 'भएन', 'हुनेछ', 'हुँदैन', 'हुनुहुन्न', 'पर्छ', 'पर्दैन', 'पर्‍यो', 'पर्ने', 'पर्नेछ']);
const DEV_DIGITS = '०१२३४५६७८९';
const SUF_DEFAULT = { haru: 'हरू', haroo: 'हरू', harulai: 'हरूलाई', haruma: 'हरूमा', haruko: 'हरूको', haruka: 'हरूका', haruki: 'हरूकी', harule: 'हरूले', harubata: 'हरूबाट', harusanga: 'हरूसँग', harusita: 'हरूसित', harudekhi: 'हरूदेखि', harusamma: 'हरूसम्म', haruprati: 'हरूप्रति', lai: 'लाई', laai: 'लाई', ma: 'मा', maa: 'मा', bata: 'बाट', baata: 'बाट', ko: 'को', ka: 'का', ki: 'की', le: 'ले', sanga: 'सँग', sita: 'सित', dekhi: 'देखि', samma: 'सम्म', bhanda: 'भन्दा', prati: 'प्रति', bare: 'बारे', tira: 'तिर', ntira: 'न्तिर', jasto: 'जस्तो' };

// ---------------------------------------------------------------- phonetic skeleton keys
const DC = { 'क': 'k', 'ख': 'K', 'ग': 'g', 'घ': 'G', 'ङ': 'n', 'च': 'C', 'छ': 'C', 'ज': 'j', 'झ': 'Z', 'ञ': 'n', 'ट': 't', 'ठ': 'T', 'ड': 'd', 'ढ': 'D', 'ण': 'n', 'त': 't', 'थ': 'T', 'द': 'd', 'ध': 'D', 'न': 'n', 'प': 'p', 'फ': 'P', 'ब': 'b', 'भ': 'B', 'म': 'm', 'य': 'y', 'र': 'r', 'ल': 'l', 'व': 'b', 'श': 's', 'ष': 's', 'स': 's', 'ह': 'h', 'ळ': 'l' };
const DV = { 'अ': 'a', 'आ': 'a', 'इ': 'i', 'ई': 'i', 'उ': 'u', 'ऊ': 'u', 'ऋ': 'ri', 'ए': 'e', 'ऐ': 'ai', 'ओ': 'o', 'औ': 'au', 'ा': 'a', 'ि': 'i', 'ी': 'i', 'ु': 'u', 'ू': 'u', 'ृ': 'ri', 'े': 'e', 'ै': 'ai', 'ो': 'o', 'ौ': 'au', 'ॉ': 'o' };
const LAB = new Set('पफबभम');
const NUK = [['क़', 'क'], ['ख़', 'ख'], ['ग़', 'ग'], ['ज़', 'ज'], ['ड़', 'ड'], ['ढ़', 'ढ'], ['फ़', 'फ']];
const finish = (s) => (s.slice(0, 1) + s.slice(1).replaceAll('a', '')).replace(/(.)\1+/g, '$1');
export function dkey(w) {
  w = w.normalize('NFC').replaceAll('क्ष', 'C').replaceAll('ज्ञ', 'J');
  for (const [a, b] of NUK) w = w.replaceAll(a, b);
  const cs = Array.from(w); let out = '';
  for (let i = 0; i < cs.length; i++) {
    const ch = cs[i];
    if (DC[ch]) out += DC[ch]; else if (DV[ch]) out += DV[ch];
    else if (ch === 'ं' || ch === 'ँ') out += LAB.has(cs[i + 1] || '') ? 'm' : 'n';
    else if (ch === 'ः') out += 'h'; else if (ch === 'C' || ch === 'J') out += ch;
  }
  return finish(out);
}
const RCL = [['kshy', 'C'], ['ksh', 'C'], ['chh', 'C'], ['ch', 'C'], ['gny', 'J'], ['gy', 'J'], ['jny', 'J'], ['jn', 'J'], ['sh', 's'], ['kh', 'K'], ['gh', 'G'], ['th', 'T'], ['dh', 'D'], ['ph', 'P'], ['bh', 'B'], ['jh', 'Z'], ['ee', 'i'], ['ii', 'i'], ['oo', 'u'], ['uu', 'u'], ['ks', 'C']];
const RSG = { x: 'C', f: 'P', w: 'b', v: 'b', z: 'j', q: 'k', c: 'C' };
export function rkey(r) {
  let s = r.toLowerCase().replace(/[^a-z]/g, '');
  for (const [a, b] of RCL) s = s.replaceAll(a, b);
  s = Array.from(s, (c) => RSG[c] || c).join('');
  return finish(s);
}
const nokey = (k) => k.replaceAll('n', '');
export function hdkey(w) {
  w = w.normalize('NFC');
  w = w.replaceAll('ी', 'ि').replaceAll('ू', 'ु').replaceAll('ई', 'इ').replaceAll('ऊ', 'उ').replaceAll('ँ', 'ं').replaceAll('श', 'स').replaceAll('ष', 'स');
  return w.replace(/[ङञणन]्/g, 'ं');
}

const KINDS = ['lex', 'lex_nn', 'comp', 'comp2', 'direct', 'beam'];
function editDist(a, b) {
  let p = Array.from({ length: b.length + 1 }, (_, i) => i);
  for (let i = 1; i <= a.length; i++) { const c = [i]; for (let j = 1; j <= b.length; j++) c[j] = Math.min(p[j] + 1, c[j - 1] + 1, p[j - 1] + (a[i - 1] !== b[j - 1] ? 1 : 0)); p = c; }
  return p[b.length];
}
const CTX_PUNCT = new Set([',', ';', ':', '।', '?', '!']);

/** Stage 2: tiny bidirectional context transformer (ONNX): logits for K candidate ids at the masked centre of an 11-word window */
/** s32 compact skeleton-keyed Kneser-Ney stem-bigram LM (artifacts/subword_lm.tsv) */
export function lmSkel(s) {
  s = s.normalize('NFC').replace(/[\u200d\u200c]/g, '').replace(/ी/g, 'ि').replace(/ू/g, 'ु').replace(/ई/g, 'इ').replace(/ऊ/g, 'उ').replace(/[ंँ]/g, 'ं').replace(/[शष]/g, 'स');
  s = s.replace(/ङ्/g, 'ं').replace(/ञ्/g, 'ं').replace(/ण/g, 'न').replace(/व/g, 'ब').replace(/ं(?=[कखगघचछजझटठडढतथदधपफबभ])/g, 'न');
  return s.replace(/ृ/g, 'रि').replace(/्+$/, '').replace(/्$/, '');
}
export const ENS = { pan_primary: false, conf: -0.55, defer: 0.25, w: 0.30, w_a4: null, std_lp: null, cap: 0.6, cap_conf: -1.2, cap_multi: true, cap_common_f: 15, cap_dial: null, dial_tier: 1 }; // s34 Dual-Core a4_pan-primary weighting (mirrors model/hybrid.py ENS)
class CompactLM {
  constructor(text) {
    this.u = new Map(); this.b = new Map(); this.suf = []; this._sk = new Map(); this._lp = new Map(); let sec = '';
    for (const ln of text.split('\n')) {
      if (!ln) continue;
      if (ln.startsWith('#H')) { const p = ln.split('\t'); this.N = parseFloat(p[1]); this.NB = parseFloat(p[2]); this.d = parseFloat(p[3]); continue; }
      if (ln.startsWith('#S')) { const p = ln.split('\t'); this.suf.push([p[1], parseInt(p[2], 10) / 100]); continue; }
      if (ln === '#U' || ln === '#B') { sec = ln; continue; }
      const p = ln.split('\t');
      if (sec === '#U') this.u.set(p[0], [parseFloat(p[1]), parseFloat(p[2]), parseFloat(p[3])]); else this.b.set(p[0] + '\t' + p[1], parseFloat(p[2]));
    }
    this.suf.sort((a, b) => b[0].length - a[0].length);
  }
  sk(w) { let r = this._sk.get(w); if (r === undefined) { if (this._sk.size > 100000) this._sk.clear(); r = lmSkel(w); this._sk.set(w, r); } return r; }
  pk(s) { const u = this.u.get(s); return u ? Math.max(u[2], 0.5) / this.NB : 1e-7; }
  _kn(ps, s) {
    let c1, nf;
    if (ps === '^') { c1 = this.N / 12; nf = this.NB / 40; }
    else { const pu = this.u.get(ps); if (!pu) return [Math.log(Math.max(this.pk(s), 1e-8)), 0]; c1 = pu[0]; nf = pu[1]; }
    const c = this.b.get(ps + '\t' + s) || 0;
    return [Math.log(Math.max((Math.max(c - this.d, 0) + this.d * nf * this.pk(s)) / Math.max(c1, 1), 1e-8)), c];
  }
  lp(prev, cand, lam = 0.85) {
    const key = (prev === null ? '^' : prev) + '\u0001' + cand + '\u0001' + lam; let r = this._lp.get(key); if (r !== undefined) return r;
    const ps = prev === null ? '^' : this.sk(prev), s = this.sk(cand); let [l, c] = this._kn(ps, s);
    if (c <= 0) {
      for (const [w, slp] of this.suf) {
        if (cand.length >= w.length + 2 && cand.endsWith(w)) {
          const st = this.sk(cand.slice(0, -w.length)), su = this.u.get(st);
          if (su && su[0] > 20) { const u = this.u.get(s), wc = u ? u[0] : 0, [l2, c2] = this._kn(ps, st); if (c2 > 0 || wc < 5) l = Math.max(l, l2 + slp - lam); break; }
        }
      }
    }
    if (this._lp.size > 400000) this._lp.clear();
    this._lp.set(key, l); return l;
  }
}

export class CtxScorer {
  constructor(ort, sess, words) { this.ort = ort; this.sess = sess; this.vocab = new Map(words.map((w, i) => [w, i])); }
  async score(win, cand) {
    const T = this.ort.Tensor;
    const r = await this.sess.run({ ctx: new T('int64', BigInt64Array.from(win, BigInt), [1, win.length]), cand: new T('int64', BigInt64Array.from(cand, BigInt), [1, cand.length]) });
    return Array.from(r.logits.data);
  }
}

class Lexicon {
  constructor(text) {
    this.freq = new Map(); this.full = new Map(); this.nn = new Map(); this.snap = new Map();
    const rows = [];
    for (const l of text.split('\n')) { const p = l.split('\t'); if (p.length < 2) continue; const c = parseInt(p[1], 10); this.freq.set(p[0], c); rows.push([p[0], c]); }
    rows.sort((a, b) => b[1] - a[1] || (a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0));
    const push = (m, k, w) => { const a = m.get(k); if (a) a.push(w); else m.set(k, [w]); };
    for (const [w] of rows) { const k = dkey(w); push(this.full, k, w); push(this.nn, nokey(k), w); const h = hdkey(w); if (!this.snap.has(h)) this.snap.set(h, w); }
  }
}

/** One int8 seq2seq member (encoder + one-step decoder + its own vocab). An ensemble = mean teacher-forced log-prob over members. */
class Seq2Seq {
  constructor(ort, enc, dec, vocab, cfg) {
    this.ort = ort; this.enc = enc; this.dec = dec;
    this.s2i = new Map(vocab.src.map((c, i) => [c, i])); this.tgt = vocab.tgt; this.t2i = new Map(vocab.tgt.map((c, i) => [c, i]));
    this.maxSrc = cfg.max_src; this.D = cfg.d; this.rev = !!cfg.rev; this.lower = vocab.src.length < 50;
  }
  _ids(word, lang) { if (this.lower) word = word.toLowerCase(); const ids = [this.s2i.get(`<${lang}>`)]; for (const c of word) if (this.s2i.has(c)) ids.push(this.s2i.get(c)); ids.length = Math.min(ids.length, this.maxSrc); return ids; }
  async encode(word, lang) {
    const ids = this._ids(word, lang), T = this.ort.Tensor;
    const mem = (await this.enc.run({ src: new T('int64', BigInt64Array.from(ids, BigInt), [1, ids.length]) })).mem.data;
    return { ids, mem, S: ids.length, word, lang };
  }
  _tile(e, K) {
    const T = this.ort.Tensor, { ids, mem, S } = e, D = this.D;
    const memK = new Float32Array(K * S * D); for (let k = 0; k < K; k++) memK.set(mem, k * S * D);
    const one = BigInt64Array.from(ids, BigInt), srcK = new BigInt64Array(K * S); for (let k = 0; k < K; k++) srcK.set(one, k * S);
    return { memT: new T('float32', memK, [K, S, D]), srcT: new T('int64', srcK, [K, S]) };
  }
  /** teacher-forced log P(native | roman) for a list of candidate words.
   *  s38 parity: int8 dynamic quantisation makes the score depend on the decoder batch composition, so (like model/onnx_backend.py
   *  OnnxBackend.score) every (roman, candidate) is scored once, in a batch of only the not-yet-cached unique candidates, then cached. */
  async score(e, cands) {
    if (!this._sc) this._sc = new Map();
    const pre = `${e.lang}\u0001${e.word}\u0001`, miss = [], seen = new Set();
    for (const c of cands) if (!this._sc.has(pre + c) && !seen.has(c)) { seen.add(c); miss.push(c); }
    if (miss.length) { const r = await this._scoreRaw(e, miss); miss.forEach((c, j) => this._sc.set(pre + c, r[j])); if (this._sc.size > 800000) this._sc.clear(); }
    return cands.map((c) => { const v = this._sc.get(pre + c); return v === undefined ? -1e9 : v; });
  }
  async _scoreRaw(e, cands) {
    if (this.rev) cands = cands.map((c) => [...c].reverse().join(''));
    const T = this.ort.Tensor, V = this.tgt.length, N = cands.length, out = new Array(N).fill(-1e9);
    const ok = []; for (let i = 0; i < N; i++) if ([...cands[i]].every((c) => this.t2i.has(c)) && cands[i].length <= 32) ok.push(i);
    if (!ok.length) return out;
    const Lm = Math.max(...ok.map((i) => [...cands[i]].length)) + 1, M = ok.length;
    const { memT, srcT } = this._tile(e, M), tg = new BigInt64Array(M * Lm), lab = new Int32Array(M * Lm);
    ok.forEach((ci, a) => {
      const t = [1, ...[...cands[ci]].map((c) => this.t2i.get(c)), 2];
      for (let j = 0; j < t.length - 1; j++) { tg[a * Lm + j] = BigInt(t[j]); lab[a * Lm + j] = t[j + 1]; }
    });
    const lp = (await this.dec.run({ mem: memT, src: srcT, tgt: new T('int64', tg, [M, Lm]) })).logp.data;
    ok.forEach((ci, a) => { let s = 0; for (let j = 0; j < Lm; j++) { if (lab[a * Lm + j] === 0) break; s += lp[(a * Lm + j) * V + lab[a * Lm + j]]; } out[ci] = s; });
    return out;
  }
  async beam(e, K) {
    const T = this.ort.Tensor, V = this.tgt.length, { memT, srcT } = this._tile(e, K);
    let ys = Array.from({ length: K }, () => [1]); let sc = new Float32Array(K).fill(-1e9); sc[0] = 0; let fin = new Array(K).fill(false);
    for (let step = 0; step < 34; step++) {
      const L = ys[0].length, tg = new BigInt64Array(K * L);
      for (let k = 0; k < K; k++) for (let j = 0; j < L; j++) tg[k * L + j] = BigInt(ys[k][j]);
      const lp = (await this.dec.run({ mem: memT, src: srcT, tgt: new T('int64', tg, [K, L]) })).logp.data;
      const cand = [];
      for (let k = 0; k < K; k++) for (let v = 0; v < V; v++) {
        let s; const x = lp[(k * L + L - 1) * V + v];
        if (fin[k]) s = v === 0 ? sc[k] : -1e9; else s = v <= 1 ? -1e9 : sc[k] + x;
        cand.push([s, k, v]);
      }
      cand.sort((a, b) => b[0] - a[0] || (a[1] * V + a[2]) - (b[1] * V + b[2]));
      const top = cand.slice(0, K);
      ys = top.map(([, k, v]) => ys[k].concat([v])); fin = top.map(([, k, v]) => fin[k] || v === 2); sc = Float32Array.from(top, (x) => x[0]);
      if (fin.every(Boolean)) break;
    }
    const out = [], seen = new Set();
    for (let k = 0; k < K; k++) {
      let w = ''; for (const t of ys[k].slice(1)) { if (t === 0 || t === 2) break; w += this.tgt[t]; }
      if (this.rev) w = [...w].reverse().join('');
      if (w && !seen.has(w) && sc[k] > -1e8) { seen.add(w); out.push([w, sc[k]]); }
    }
    return out;
  }
}

export class Transliterator {
  constructor(ort, members, cfg, lex, direct) {
    this.ort = ort; this.members = members;
    this.baseIdx = []; this.specIdx = [];
    members.forEach((m, i) => (m.lower ? this.specIdx : this.baseIdx).push(i));
    this.dual = this.baseIdx.length > 0 && this.specIdx.length > 0;
    this.keep = new Set(cfg.keep); this.lex = lex; this.direct = direct;
    this.P = Object.assign({ lam: 0.5, mu: 2.0, direct: 8.0, direct_step: 3.0, snap_ratio: 0.1, comp2_pen: 2.5, comp_pen: 3.0, nn_pen: 1.0, n_full: 16, n_nn: 6, n_stem: 4, K: 5, snap_min: 30, beam_thr: -1.5, ctx_beta: 1.0, ctx_margin: 8.0, ctx_unk: 2.0, lm_mu: 0.3, lm_skel: 0.0, lm_margin: 4.0, lm_lam: 0.85, oov: 0.0, oov_conf: 0.0, conf_lp: -0.3, fmin: 0, gate_k: 0, gate_d: 1.0, gate_hm: 1.0, gen_max: 0, gen_pen: 0 }, cfg.hybrid || {});
    if (this.P.alt_pen === undefined) this.P.alt_pen = 1.5;
    if (this.P.dfloor === undefined) this.P.dfloor = -4.0;
    this.rr = null; this.cx = null; this.lm = null;
    this.suf = Object.entries(cfg.suffixes || SUF_DEFAULT); this.JOIN = new Set(cfg.join || JOIN_DEFAULT);
    this.definitive = new Set(); this.cache = new Map(); this.stats = { beam: 0, words: 0 };
  }
  /** loader(name) -> Uint8Array | string ; defaults to fetch(base + '/' + name) */
  static async create({ ort, base = './model', loader, sessionOptions = { executionProviders: ['wasm'], graphOptimizationLevel: 'all' }, quant = 'int8', useCtx = true }) {
    const L = loader || (async (n) => { const r = await fetch(`${base}/${n}`); if (!r.ok) throw new Error(n); return n.endsWith('.onnx') ? new Uint8Array(await r.arrayBuffer()) : await r.text(); });
    const [encB, decB, vocab, cfg, lne, lhi, dne] = await Promise.all([L(`encoder.${quant}.onnx`), L(`decoder.${quant}.onnx`), L('vocab.json'), L('config.json'), L('lex_ne.tsv'), L('lex_hi.tsv'), L('direct_ne.tsv')]);
    const enc = await ort.InferenceSession.create(encB, sessionOptions), dec = await ort.InferenceSession.create(decB, sessionOptions);
    const S = (x) => (typeof x === 'string' ? x : new TextDecoder().decode(x)), J = (x) => JSON.parse(S(x));
    const direct = new Map(), definitive = new Set(); for (const l of S(dne).split('\n')) { const p = l.split('\t'); if (p.length >= 2) { const v = p.slice(1); if (v[0].startsWith('!')) { v[0] = v[0].slice(1); definitive.add(p[0]); } direct.set(p[0], v); } }   // '!' = definitive entry (Stage 2 never overrides it)
    const cfgo = J(cfg), members = [new Seq2Seq(ort, enc, dec, J(vocab), cfgo)];
    for (const sub of cfgo.ensemble || []) {   // extra ensemble members live in <base>/<sub>/ (own vocab); score = mean log-prob
      const [eB, dB, vB, cB] = await Promise.all([L(`${sub}/encoder.${quant}.onnx`), L(`${sub}/decoder.${quant}.onnx`), L(`${sub}/vocab.json`), L(`${sub}/config.json`)]);
      members.push(new Seq2Seq(ort, await ort.InferenceSession.create(eB, sessionOptions), await ort.InferenceSession.create(dB, sessionOptions), J(vB), J(cB)));
    }
    const T = new Transliterator(ort, members, cfgo, { ne: new Lexicon(S(lne)), hi: new Lexicon(S(lhi)) }, direct);
    T.definitive = definitive;
    if (cfgo.rerank) { T.rr = J(await L('rerank.json')); T.P.beam_thr = T.rr.beam_thr; }
    if (cfgo.ctx && useCtx) {
      const [cb, cv] = await Promise.all([L(`ctx.${quant}.onnx`), L('ctx_vocab.txt')]);
      T.cx = new CtxScorer(ort, await ort.InferenceSession.create(cb, sessionOptions), S(cv).split('\n'));
    }
    if (cfgo.lm_file) { try { T.lm = new CompactLM(S(await L(cfgo.lm_file))); } catch (e) { T.lm = null; } }
    return T;
  }

  // ------------------------------------------------------------ layer 2: neural primitives
  async _encode(word, lang) { const es = []; for (const m of this.members) es.push(await m.encode(word, lang)); es.word = word; return es; }
  /** ensemble teacher-forced log P: mean over members, or Dual-Core Specialist PoE when base + specialist members are present */
  async _score(es, cands) {
    const n = this.members.length;
    if (n === 1) return this.members[0].score(es[0], cands);
    const outs = []; for (let i = 0; i < n; i++) outs.push(await this.members[i].score(es[i], cands));
    if (!this.dual) {
      return cands.map((_, j) => { let s = 0; for (const o of outs) { if (o[j] <= -1e8) return -1e9; s += o[j]; } return s / n; });
    }
    const nb = this.baseIdx.length, ns = this.specIdx.length, midUp = /[A-Z]/.test((es.word || '').slice(1));
    let mxA = -1e9, mxB = -1e9;
    const ab = cands.map((_, j) => {
      for (const o of outs) if (o[j] <= -1e8) return null;
      let a = 0, b = 0;
      for (const q of this.baseIdx) a += outs[q][j]; a /= nb;
      for (const q of this.specIdx) b += outs[q][j]; b /= ns;
      if (ENS.cap !== null && b >= ENS.cap_conf && (!ENS.cap_multi || es.cap)) a = Math.max(a, b - (es.capv !== undefined && es.capv !== null ? es.capv : ENS.cap));   // s39 two-tier: es.capv = ENS.cap_dial on dialectal suffix; s37 capped veto; s38 gate: es.cap = multi-word AND non-common (tail/OOV/dialectal) word
      if (a > mxA) mxA = a; if (b > mxB) mxB = b;
      return [a, b];
    });
    const w = midUp ? 0.08 : (ENS.w_a4 !== null && !(ENS.std_lp !== null && mxA >= ENS.std_lp && mxB < mxA - 0.3)) ? ENS.w_a4 : (ENS.pan_primary && mxB >= ENS.conf && (mxA - mxB) <= ENS.defer) ? ENS.w : (((mxA < -0.95 && mxB >= -0.85 && mxB > mxA + 0.55) || (mxB >= -0.42 && mxB > mxA + 0.32)) ? 0.85 : (mxA < -1.25 ? 0.20 : 0.15));
    return ab.map((x) => (x === null ? -1e9 : (1 - w) * x[0] + w * x[1]));
  }
  /** union of every member's beam, re-scored by the ensemble mean */
  async _beam(es, K) {
    const n = this.members.length;
    if (n === 1) return this.members[0].beam(es[0], K);
    const names = []; const seen = new Set();
    for (let i = 0; i < n; i++) for (const [w] of await this.members[i].beam(es[i], K)) if (!seen.has(w)) { seen.add(w); names.push(w); }
    const sc = names.length ? await this._score(es, names) : [];
    return names.map((w, i) => [w, sc[i]]).sort((a, b) => b[1] - a[1]).slice(0, K).filter((x) => x[1] > -1e8);
  }
  /** neural-only beam (kept for diagnostics) */
  async topkWord(word, lang = 'ne', K = 5) { return this._beam(await this._encode(word, lang), K); }

  // ------------------------------------------------------------ layer 1: lexicon + morphology candidates
  lexCands(roman) {
    const P = this.P, L = this.lex.ne, out = new Map(), k = rkey(roman);
    for (const w of (L.full.get(k) || []).slice(0, P.n_full)) out.set(w, ['lex', L.freq.get(w)]);
    for (const w of (L.nn.get(nokey(k)) || []).slice(0, P.n_nn)) if (!out.has(w)) out.set(w, ['lex_nn', L.freq.get(w)]);
    const rl = roman.toLowerCase().replace(/[^a-z]/g, '');
    for (const [sr, sd] of this.suf) {
      if (rl.endsWith(sr) && rl.length - sr.length >= 2) {
        const sk = rkey(rl.slice(0, -sr.length)); if (!sk) continue;
        const stems = (L.full.get(sk) || []).slice(0, P.n_stem);
        for (const w of (L.nn.get(nokey(sk)) || []).slice(0, 2)) if (!stems.includes(w)) stems.push(w);
        for (const st of stems) {
          if (this.JOIN.has(st)) continue;
          const w = st + sd;
          if (!out.has(w)) out.set(w, ['comp', L.freq.has(w) ? Math.max(L.freq.get(w), L.freq.get(st)) : L.freq.get(st)]);
        }
      }
    }
    return out;
  }
  comp2Cands(roman) {
    const rl = roman.toLowerCase().replace(/[^a-z]/g, ''), L = this.lex.ne, out = new Map();
    if (rl.length < 9) return out;
    for (let i = 3; i < rl.length - 2; i++) {
      for (const x of (L.full.get(rkey(rl.slice(0, i))) || []).slice(0, 2)) for (const y of (L.full.get(rkey(rl.slice(i))) || []).slice(0, 2)) {
        if ([...x].length >= 2 && [...y].length >= 2 && !out.has(x + y)) out.set(x + y, ['comp2', Math.min(L.freq.get(x), L.freq.get(y))]);
      }
    }
    return out;
  }
  canon(n, kind, w = '', lp = -99) {
    if (kind === 'direct') return n;   // s21: Pragya chandrabindu forms are canonical
    if (n === 'गरो') return 'गाह्रो';
    { const mr = /^(.*िरहे)(छ|छु|छौ|छौं|छौँ)$/u.exec(n); if (mr && /irahe(chh|x)/.test(w)) return mr[1] + 'को ' + mr[2]; }   // s14: lagirahechha -> लागिरहेको छ
    const wl = w.toLowerCase();
    if (wl === 'sorry' && n === 'सोरी') return 'सरी';
    if (n.endsWith('उन्छु')) n = n.slice(0, -4) + 'उँछु';
    if (n.startsWith('तपाईँ')) return 'तपाईं' + n.slice(5);
    if (n.startsWith('तपाई') && !n.startsWith('तपाईं') && !n.startsWith('तपाईँ')) return 'तपाईं' + n.slice(4);
    const mye = /^(.*(?:[ािीुू]|[गभआ]))ये(र|को|का|की|कै|न|)$/u.exec(n);   // s12: खायेर -> खाएर (ties go to the standard form)
    if (mye) { const alt = mye[1] + 'ए' + mye[2]; if ((this.lex.ne.freq.get(alt) || 0) >= (this.lex.ne.freq.get(n) || 0)) n = alt; }
    const L = this.lex.ne;
    if (n.endsWith('द्') && (L.freq.get(n.slice(0, -1)) || 0) >= 2 * (L.freq.get(n) || 0) && (L.freq.get(n.slice(0, -1)) || 0) >= 200) n = n.slice(0, -1);
    if (['hru', 'hrule', 'hrulai', 'hruko', 'hruma', 'hrubata'].some((x) => wl.endsWith(x)) && n.length >= 5 && n.includes('हरु')) {
      n = n.replace(/हरु(लाई|मा|को|का|की|ले|बाट|सँग|सित|देखि|सम्म|प्रति)?$/, 'हरू$1');
    }
    if ((['ii', 'ee', 'oo', 'uu', 'ng'].some((x) => wl.includes(x)) || /m[kgcjtdnpbsh]/.test(wl)) && lp > -1.8) return n;
    if (wl.includes('sh') && (n.includes('श') || n.includes('ष')) && lp > -1.2) return n;
    const alt = L.snap.get(hdkey(n));
    if (alt && alt !== n && L.freq.has(alt) && L.freq.get(alt) >= this.P.snap_min && (L.freq.get(n) || 0) < this.P.snap_ratio * L.freq.get(alt)) return alt;
    return n;
  }
  _b(f) { const P = this.P; return f >= P.fmin ? P.lam * Math.log1p(f) + P.mu - (f <= P.gen_max ? P.gen_pen : 0) : P.oov; }
  _prior(w, kind, f, lang, lp = 0, rkW = null, hasStrongLex = false) {
    const P = this.P, L = this.lex[lang];
    if (kind === 'beam') return L.freq.has(w) && !(hasStrongLex && rkW != null && dkey(w) !== rkW) ? this._b(L.freq.get(w)) : P.oov + (lp >= P.conf_lp ? P.oov_conf : 0);
    if (kind === 'direct') return P.direct - P.direct_step * f + (lp < P.dfloor ? P.dfloor - lp : 0);   // s14: a curated direct entry cannot be sunk by an overconfident model
    if (kind === 'comp2') return this._b(f) - P.comp2_pen;
    if (kind === 'lex') return this._b(f);
    if (kind === 'lex_nn') return this._b(f) - P.nn_pen;
    if (kind === 'comp') return L.freq.has(w) ? this._b(L.freq.get(w)) : this._b(f) - P.comp_pen;
    return 0;
  }
  /** neural-agreement gate: lexicon/morphology candidates far below the model's best log-prob cannot ride on their frequency boost */
  _gate(n, kind, lp, mx, bk, rkW = null, hasStrongLex = false) {
    const P = this.P, L = this.lex.ne;
    let g = 0;
    if (mx != null && P.gate_k && (kind === 'lex' || kind === 'lex_nn' || kind === 'comp' || kind === 'comp2')) {
      g = P.gate_k * Math.max(0, (mx - lp) - P.gate_d);
      if (bk != null && hdkey(n) === bk) g *= P.gate_hm;
    }
    if (kind === 'beam' && mx != null && rkW != null && L.freq.has(n) && !(hasStrongLex && dkey(n) !== rkW) && dkey(n) !== rkW) {
      g += 1.2 * Math.max(0, (mx - lp) - 2.4);
    }
    return g;
  }
  /** s35 multi-word split-only shield (ne only): never fires on single-word inputs or direct full-word lexicon stems */
  _shield(n, kind, f, pr, lp, mx, sent = false) {
    const P = this.P, F = this.lex.ne.freq, inLex = F.has(n);
    if (sent && P.sh_tau != null && mx != null && kind !== 'direct' && pr > 0 && (mx - lp) > P.sh_tau) {
      if ((!inLex && (kind === 'comp' || kind === 'comp2')) || (f < 15 && (mx - lp) > Math.max(P.sh_tau, 2.2) && !(inLex && F.get(n) >= (P.sh_lock != null ? P.sh_lock : 15)))) {
        return pr * (P.sh_att != null ? P.sh_att : 0);
      }
    }
    return pr;
  }
  _hand(n, kind, f, lp, lang, mx, bk, rkW = null, hasStrongLex = false, sent = false) {
    const P = this.P; let pr = this._prior(n, kind, f, lang, lp, rkW, hasStrongLex);
    pr = this._shield(n, kind, f, pr, lp, mx, sent);
    return lp + pr - this._gate(n, kind, lp, mx, bk, rkW, hasStrongLex);
  }
  /** per-candidate feature rows (order mirrors model/hybrid.py Hybrid.features) */
  _features(w, pool, sent = false) {
    const F = this.lex.ne.freq, order = [...pool.keys()].sort((a, b) => pool.get(b)[2] - pool.get(a)[2]), bestN = order[0];
    const mx = Math.max(pool.get(bestN)[2], -25), bk = hdkey(bestN), rank = new Map(order.map((n, i) => [n, i]));
    const lw = Math.log(Math.max(1, w.length)) / 3, bfl = Math.log1p(F.get(bestN) || 0), bin = F.has(bestN) ? 1 : 0, LEXK = ['lex', 'lex_nn', 'comp', 'comp2'];
    let topf = 0; for (const [n, p] of pool) if (LEXK.includes(p[0])) topf = Math.max(topf, F.has(n) ? F.get(n) : p[1]);
    const rows = [];
    for (const [n, [kind, f, lp0]] of pool) {
      const lp = Math.max(lp0, -25), inlex = F.has(n), fr = inlex ? F.get(n) : (LEXK.includes(kind) ? f : 0), m = this.canon(n, kind), lfr = Math.log1p(fr);
      const ed = editDist(n, bestN) / Math.max(1, Math.max(n.length, bestN.length));
      rows.push({ n, m, hs: this._hand(n, kind, f, lp0, 'ne', mx, bk, null, false, sent), x: [lp / 5, (lp - mx) / 5, inlex ? 1 : 0, lfr / 10,
        ...KINDS.map((k) => (kind === k ? 1 : 0)), kind === 'direct' ? f / 4 : 0, hdkey(n) === bk ? 1 : 0, n === bestN ? 1 : 0, mx / 5, lw, m !== n ? 1 : 0,
        Math.min(rank.get(n), 5) / 5, bin, bfl / 10, (lfr - bfl) / 10, ed, Math.min(pool.size, 20) / 20, LEXK.includes(kind) && fr === topf ? 1 : 0] });
    }
    return rows;
  }
  _mlp(x) {
    const R = this.rr; let s = R.b2;
    for (let j = 0; j < R.W1.length; j++) { let a = R.b1[j]; const c = R.W1[j]; for (let i = 0; i < x.length; i++) a += x[i] * c[i]; s += Math.tanh(a) * R.W2[j]; }
    return s;
  }

  /** s38 (mirrors hybrid.py cap_common): common standard lexicon entry / curated direct entry without a dialectal suffix -> ENS.cap stays OFF */
  _capCommon(w) {
    if (this.direct.has(w)) return true;
    const L = this.lex.ne; let f = 0; for (const x of (L.full.get(rkey(w)) || []).slice(0, this.P.n_full)) f = Math.max(f, L.freq.get(x));
    return f >= ENS.cap_common_f && !/(nya|yako|chham|tami|harule|bata)$/.test(w);
  }
  /** s39 two-tier cap (mirrors hybrid.py DIAL_TIERS): tighter ENS.cap_dial when the Roman token ends in a regional dialectal inflection */
  _capVal(w) {
    if (ENS.cap_dial === null || w.length < 6) return ENS.cap;
    const T = [['nya', 'yako', 'chham', 'tami'], ['nya', 'yako', 'chham', 'tami', 'chhau', 'mui', 'bathe', 'bata', 'kheri', 'ikana', 'harule'], ['nya', 'yako', 'chham', 'tami', 'chhau', 'mui', 'bathe', 'bata', 'kheri', 'ikana', 'harule', 'ya', 'au', 'ou']][ENS.dial_tier];
    return T.some((x) => w.endsWith(x)) ? ENS.cap_dial : ENS.cap;
  }
  /** ranked candidates [[devanagari, score, kind]] best first for ONE roman word */
  async rankWord(roman, { lang = 'ne', topn = 5, alwaysBeam = false, sent = false } = {}) {
    const w = normRoman(roman.toLowerCase()), ck = `${lang}|${alwaysBeam ? 1 : 0}|${sent ? 1 : 0}|${w}`;
    let res = this.cache.get(ck);
    if (!res) {
      const P = this.P, e = await this._encode(w, lang), pool = new Map();
      const capOn = ENS.cap !== null && sent && lang === 'ne' && !this._capCommon(w); e.cap = capOn; e.capv = capOn ? this._capVal(w) : null;   // s38 multi-word tail/OOV/dialectal gate for ENS.cap
      if (lang === 'ne') {
        for (const [n, [kind, f]] of this.lexCands(w)) pool.set(n, [kind, f, null]);
        (this.direct.get(w) || []).forEach((n, pos) => { if (pool.has(n)) { pool.get(n)[0] = 'direct'; pool.get(n)[1] = pos; } else pool.set(n, ['direct', pos, null]); });
        const names = [...pool.keys()];
        if (names.length) (await this._score(e, names)).forEach((s, i) => { pool.get(names[i])[2] = s; });
      }
      // s14: informal-typing view (v->bh, ye->ya, te->tya, -Co->-Cyo): candidates from both views, lp = max(lp(w), lp(alt) - alt_pen)
      let alt = null, ea = null;
      if (lang === 'ne') { const a = altRoman(w); if (a !== w) { alt = a; ea = await this._encode(a, lang); ea.cap = capOn; ea.capv = e.capv; } }
      if (alt) {
        for (const [n, [kind, f]] of this.lexCands(alt)) if (!pool.has(n)) pool.set(n, [kind, f, null]);
        const nul = [...pool.keys()].filter((n) => pool.get(n)[2] === null);
        if (nul.length) (await this._score(e, nul)).forEach((s, i) => { pool.get(nul[i])[2] = s; });
        const all = [...pool.keys()];
        (await this._score(ea, all)).forEach((s, i) => { const v = pool.get(all[i]); v[2] = Math.max(v[2], s - P.alt_pen); });
      }
      let mx = -1e9; for (const v of pool.values()) mx = Math.max(mx, v[2]);
      if (alwaysBeam || lang !== 'ne' || !pool.size || mx < ((sent && P.sh_beam != null) ? Math.max(P.beam_thr, P.sh_beam) : P.beam_thr)) {
        this.stats.beam++;
        if (lang === 'ne') {
          const c2 = [...this.comp2Cands(w)].filter(([n]) => !pool.has(n)); for (const [n, v] of c2) pool.set(n, [v[0], v[1], null]);
          if (c2.length) (await this._score(e, c2.map((x) => x[0]))).forEach((sc, i) => { pool.get(c2[i][0])[2] = sc; });
        }
        const addW = [];
        for (const [n, lp] of await this._beam(e, P.K)) { if (pool.has(n)) pool.get(n)[2] = alt ? Math.max(pool.get(n)[2], lp) : lp; else if (fsmOk(n)) { pool.set(n, ['beam', 0, lp]); addW.push(n); } }
        if (alt) {
          const addA = [];
          for (const [n, lp] of await this._beam(ea, P.K)) { if (pool.has(n)) pool.get(n)[2] = Math.max(pool.get(n)[2], lp - P.alt_pen); else if (fsmOk(n)) { pool.set(n, ['beam', 0, lp - P.alt_pen]); addA.push(n); } }
          if (addA.length) (await this._score(e, addA)).forEach((s, i) => { const v = pool.get(addA[i]); v[2] = Math.max(v[2], s); });
          if (addW.length) (await this._score(ea, addW)).forEach((s, i) => { const v = pool.get(addW[i]); v[2] = Math.max(v[2], s - P.alt_pen); });
        }
      }
      const best = new Map();
      if (this.rr && lang === 'ne' && pool.size) {
        for (const { n, m, x, hs } of this._features(w, pool, sent)) { const sc = hs + this._mlp(x); if (!best.has(m) || sc > best.get(m)[0]) best.set(m, [sc, pool.get(n)[0]]); }
      } else {
        let gmx = -1e9, gbn = null, hasStrongLex = false, hasShortNasal = false, isChhu = false;
        for (const [n, v] of pool) {
          if (v[2] > gmx) { gmx = v[2]; gbn = n; }
          if (lang === 'ne' && v[0] === 'lex' && v[1] >= 500 && v[2] >= -1.8) hasStrongLex = true;
          if (lang === 'ne' && v[0] === 'lex' && v[1] >= 50 && !n.includes('ाँ') && (n.includes('ं') || n.includes('ण्ड') || n.includes('न्द'))) hasShortNasal = true;
          if (lang === 'ne' && (n.endsWith('छु') || n.endsWith('उन्छु'))) isChhu = true;
        }
        hasShortNasal = hasShortNasal && !w.includes('aa') && /an[sjd]/.test(w);
        isChhu = isChhu && (w.endsWith('chhu') || w.endsWith('xu'));
        const gbk = gbn == null ? null : hdkey(gbn), rkW = lang === 'ne' ? rkey(w) : null; gmx = Math.max(gmx, -25);
        for (const [n, [kind, f, lp]] of pool) {
          const m = lang === 'ne' ? this.canon(n, kind, w, lp) : n;
          const fEff = (lang === 'ne' && w === 'pani' && n === 'पानी' && kind === 'direct' && f === 0) ? 1 : f;
          let sc = lp + (lang === 'ne' ? this._shield(n, kind, fEff, this._prior(n, kind, fEff, lang, lp, rkW, hasStrongLex), lp, gmx, sent) : this._prior(n, kind, fEff, lang, lp, rkW, hasStrongLex)) - (lang === 'ne' ? this._gate(n, kind, lp, gmx, gbk, rkW, hasStrongLex) : 0);
          if (lang === 'ne') {
            if (hasShortNasal && n.includes('ाँ')) sc -= 1.2;
            if (isChhu) { if (m.endsWith('छु')) sc += 3.0; else if (m.endsWith('ला')) sc -= 8.0; }
            if (kind !== 'direct' && gbn != null && n !== gbn && w.includes('b') && !w.includes('v') && !w.includes('w') && gbn.slice(1).includes('ब') && !gbn.slice(1).includes('व') && n.slice(1).includes('व') && !n.slice(1).includes('ब') && gmx > -0.65 && (gmx - lp) > 1.2 && f < 100) sc -= 2.0 * (gmx - lp);
          }
          if (!best.has(m) || sc > best.get(m)[0]) best.set(m, [sc, kind]);
        }
      }
      const lst = [...best].map(([n, [sc, kind]]) => [n, sc, kind]); lst.sort((a, b) => b[1] - a[1]); res = lst.slice(0, 12);
      this.cache.set(ck, res); if (this.cache.size > 4000) this.cache.delete(this.cache.keys().next().value);
    }
    return res.slice(0, topn);
  }
  _pick(res) { return res.length ? res[0][0] : null; }
  /** IME chips: ranked words only */
  async candidates(word, { lang = 'ne', n = 5 } = {}) { return (await this.rankWord(word, { lang, topn: n })).map((x) => x[0]); }
  async suggest(prefix, lang = 'ne', k = 3) { return this.candidates(prefix, { lang, n: k }); }

  isWord(t) {
    if (!/^[A-Za-z][A-Za-z0-9_']*$/.test(t)) return false;
    if (/\d/.test(t) || this.keep.has(t.toLowerCase().replace(/'+$/, '')) || (t.length >= 2 && t === t.toUpperCase())) return false;
    return true;
  }

  // ------------------------------------------------------------ layer 3: sentence
  /** per-token analysis for UIs: [{src,out,word,alts,low}] (vibhakti already joined in byakaran mode) */
  async analyze(text, { lang = 'ne', mode = 'byakaran' } = {}) {
    const toks = text.match(TOKEN_RE) || [], segs = [], ranked = new Map();
    const sent = toks.filter((t) => this.isWord(t)).length >= 2;
    for (let i = 0; i < toks.length; i++) {
      const t = toks[i];
      if (this.isWord(t)) {
        let alts, out;
        if (mode === 'fast') { const b = await this._beam(await this._encode(t.toLowerCase(), lang), 1); out = b.length ? b[0][0] : t; alts = [out]; }
        else { const r = await this.rankWord(t, { lang, topn: 5, sent }); ranked.set(segs.length, r); out = this._pick(r) || t; alts = r.map((x) => x[0]); if (!alts.includes(out)) alts.unshift(out); alts = [out, ...alts.filter((a) => a !== out)]; }
        this.stats.words++;
        segs.push({ src: t, out, word: true, alts, low: mode !== 'fast' && !this.lex[lang].freq.has(out) });
      } else if (/^\d+(?:[.,]\d+)*$/.test(t) && lang === 'ne') segs.push({ src: t, out: t.replace(/\d/g, (d) => DEV_DIGITS[+d]), word: false });
      else if (t === '.') {
        const nxt = i + 1 < toks.length ? toks[i + 1] : '', prev = segs.length ? segs[segs.length - 1].out : '';
        segs.push({ src: t, out: (nxt === '' || /^\s+$/.test(nxt)) && prev && DEVA.test(prev.slice(-1)) ? '।' : t, word: false });
      } else segs.push({ src: t, out: t, word: false });
    }
    if (mode !== 'byakaran' || lang !== 'ne') return segs;
    if (this.cx && ranked.size) { this._lmAdjust(segs, ranked); await this._ctx(segs, ranked); }
    this._agree(segs);
    return this._join(segs, false).res;
  }
  /** Padayog / Padabiyog joining; track=true also returns where[i] (segment -> joined piece) and head[k] (first segment of piece k) */
  _join(segs, track) {
    const res = [], where = new Map(), head = new Map();
    for (let i = 0; i < segs.length; i++) {
      const g = segs[i], n = res.length;
      if (g.word && HON.has(g.out) && n >= 2 && res[n - 1].out === ' ' && !this.JOIN.has(res[n - 2].out) && (res[n - 2].out.endsWith('नु') || (PAR.has(g.out) && /(्न|ान|उन)$/.test(res[n - 2].out)))) {
        res.pop(); const p = res[res.length - 1]; if (!p.out.endsWith('नु')) p.out += 'ु'; p.out += g.out; p.src += ' ' + g.src; p.alts = [p.out]; where.set(i, res.length - 1);
      } else if (g.word && SAK_RE.test(g.out) && n >= 2 && res[n - 1].out === ' ' && /[िी]$/.test(res[n - 2].out) && res[n - 2].out.length >= 4 && !PARTICLES.has(res[n - 2].out)) {   // s14: थाकि सकेको -> थाकिसकेको
        res.pop(); const p = res[res.length - 1]; p.out = p.out.replace(/ी$/, 'ि') + g.out; p.src += ' ' + g.src; p.alts = [p.out]; where.set(i, res.length - 1);
      } else if (g.word && this.JOIN.has(g.out) && n >= 2 && /^\s+$/.test(res[n - 1].out) && res[n - 1].out.length === 1 && DEVL.test(res[n - 2].out.slice(-1)) && !this.JOIN.has(res[n - 2].out)) {
        let nxt = ''; for (let j = i + 1; j < segs.length; j++) if (!/^\s+$/.test(segs[j].out)) { nxt = segs[j].out; break; }
        if ((g.out === 'को' && PRONOUN.has(res[n - 2].out) && WHO_NEXT.has(nxt)) || (g.out === 'सँग' && (res[n - 2].out === 'राम्रो' || res[n - 2].out === 'नराम्रो')) || (g.out === 'सँगै' && SANGAI_NOT.some((x) => res[n - 2].out.endsWith(x))) || (g.out === 'तिर' && POSS.has(res[n - 2].out))) { res.push(g); where.set(i, res.length - 1); head.set(res.length - 1, i); }
        else { res.pop(); const p = res[res.length - 1]; const sfx = g.out; p.out += sfx; p.src += ' ' + g.src; p.alts = [p.out, ...(p.alts || []).slice(1).map((a) => a + sfx)]; p.word = p.word || false; where.set(i, res.length - 1); }
      } else { res.push(g); where.set(i, res.length - 1); head.set(res.length - 1, i); }
    }
    return { res, where, head };
  }
  _ctxSeq(res) {
    const V = this.cx.vocab, ids = [], sid = [], pos = new Map(); let s = 0, fresh = true;
    res.forEach((g, k) => {
      const p = g.out;
      if (/^\s+$/.test(p)) { if (p.includes('\n') && !fresh) { s++; fresh = true; } return; }
      let t;
      if (CTX_PUNCT.has(p)) t = V.get(p);
      else if ('0123456789०१२३४५६७८९'.includes(p[0])) t = V.get('<num>');
      else if (/^[A-Za-z]/.test(p[0])) t = V.get('<lat>');
      else if (DEVL.test(p[0])) t = V.has(p) ? V.get(p) : 1;
      else return;
      if (fresh) { ids.push(V.get('<bos>')); sid.push(s); fresh = false; }
      pos.set(k, ids.length); ids.push(t); sid.push(s);
      if (p === '।' || p === '?' || p === '!') { s++; fresh = true; }
    });
    return { ids, sid, pos };
  }
  /** Stage 2: sequential left-to-right contextual rescoring of ambiguous words (mirrors Hybrid._ctx) */
  // ------------------------------------------------------------ s32 Stage 1.5: skeleton-clustered consensus + compact stem-bigram KN Viterbi (mirrors model/hybrid.py _lm_adjust)
  _lmAdjust(segs, ranked) {
    const P = this.P, lm = this.lm; if (!lm) return;
    const mu = P.lm_mu, bs = P.lm_skel, marg = P.lm_margin, lam = P.lm_lam; if (mu <= 0 && bs <= 0) return;
    const chains = []; let cur = [], last = null;
    for (const i of [...ranked.keys()].sort((a, b) => a - b)) {
      if (last !== null) { let brk = false; for (let j = last + 1; j < i; j++) { const t = segs[j].src; if (t.trim() !== '' || t.includes('\n')) { brk = true; break; } } if (brk) { chains.push(cur); cur = []; } }
      last = i; const cands = ranked.get(i);
      if (!cands.length || this.JOIN.has(cands[0][0])) continue;
      const tl = segs[i].src.toLowerCase(), dl = this.direct.get(tl);
      const locked = cands.length < 2 || ['ma', 'k', 'baba', 'paani', 'sir'].includes(tl) || (dl && this.definitive.has(tl) && cands[0][0] === dl[0]);
      const top = cands[0][1], st = locked ? [cands[0]] : cands.filter(([c, s]) => top - s <= marg && !this.JOIN.has(c)).slice(0, 5);
      cur.push({ i, st, free: st.length > 1 });
    }
    chains.push(cur);
    for (const ch of chains) {
      if (!ch.some((x) => x.free)) continue;
      const em = ch.map(({ st, free }) => {
        if (!(free && bs > 0)) return st.map(([, s]) => s);
        const top = st[0][1], pr = st.map(([, s]) => Math.exp(s - top)), z = pr.reduce((a, b) => a + b, 0), sk = st.map(([c]) => lm.sk(c)), mass = new Map();
        sk.forEach((k, j) => mass.set(k, (mass.get(k) || 0) + pr[j]));
        return st.map(([, s], j) => s + bs * Math.log(mass.get(sk[j]) / z));
      });
      const n = ch.length, fw = [], bw = new Array(n);
      for (let t = 0; t < n; t++) {
        const cs = ch[t].st;
        if (t === 0) fw.push(cs.map(([c], j) => em[0][j] + mu * lm.lp(null, c, lam)));
        else { const pc = ch[t - 1].st; fw.push(cs.map(([c], j) => em[t][j] + Math.max(...pc.map(([p], q) => fw[t - 1][q] + mu * lm.lp(p, c, lam))))); }
      }
      bw[n - 1] = ch[n - 1].st.map(() => 0);
      for (let t = n - 2; t >= 0; t--) { const nx = ch[t + 1].st; bw[t] = ch[t].st.map(([c]) => Math.max(...nx.map(([x], q) => mu * lm.lp(c, x, lam) + em[t + 1][q] + bw[t + 1][q]))); }
      ch.forEach(({ i, st, free }, t) => {
        if (!free) return;
        const mm = st.map((_, j) => fw[t][j] + bw[t][j]), mb = Math.max(...mm), top = ranked.get(i)[0][1];
        const nw = st.map(([c], j) => [c, top + mm[j] - mb]).sort((a, b) => b[1] - a[1]), seen = new Set(nw.map((x) => x[0]));
        ranked.set(i, nw.concat(ranked.get(i).filter(([c]) => !seen.has(c))));
        const out = nw[0][0]; segs[i].out = out; segs[i].alts = [out, ...(segs[i].alts || []).filter((a) => a !== out)];
      });
    }
  }
  async _ctx(segs, ranked) {
    const P = this.P, V = this.cx.vocab;
    for (const i of [...ranked.keys()].sort((a, b) => a - b)) {
      const cands = ranked.get(i);
      if (cands.length < 2 || ['ma', 'k', 'baba', 'paani', 'sir'].includes(segs[i].src.toLowerCase()) || this.JOIN.has(cands[0][0])) continue;
      { const dl = this.direct.get(segs[i].src.toLowerCase()); if (dl && this.definitive.has(segs[i].src.toLowerCase()) && cands[0][0] === dl[0]) continue; }   // s14: a single-valued curated entry is definitive
      const top = cands[0][1], cs = cands.filter(([c, s]) => top - s <= P.ctx_margin && !this.JOIN.has(c)).slice(0, 5);
      if (cs.length < 2) continue;
      const clone = segs.map((g) => ({ ...g, alts: g.alts ? [...g.alts] : g.alts }));
      this._agree(clone); const { res, where, head } = this._join(clone, true), k = where.get(i);
      if (head.get(k) !== i) continue;
      const own = clone[i].out, suffix = res[k].out.startsWith(own) ? res[k].out.slice(own.length) : '';
      const { ids, sid, pos } = this._ctxSeq(res);
      if (!pos.has(k)) continue;
      const p = pos.get(k), win = [];
      for (let o = -5; o <= 5; o++) { const q = p + o; win.push(o === 0 ? 2 : (q >= 0 && q < ids.length && sid[q] === sid[p] ? ids[q] : 0)); }
      const cid = cs.map(([c]) => (V.has(c + suffix) ? V.get(c + suffix) : 1)), known = cid.map((x, j) => (x !== 1 ? j : -1)).filter((j) => j >= 0);
      if (!known.length) continue;
      const lg = await this.cx.score(win, cid), mx = Math.max(...known.map((j) => lg[j])), z = mx + Math.log(known.reduce((a, j) => a + Math.exp(lg[j] - mx), 0));
      const lp = cid.map((x, j) => (x !== 1 ? lg[j] - z : null)), floor = Math.max(...lp.filter((x) => x !== null)) - P.ctx_unk;
      const sc = cs.map(([, s], j) => s + P.ctx_beta * (lp[j] !== null ? lp[j] : floor));
      let b = 0; for (let j = 1; j < cs.length; j++) if (sc[j] > sc[b]) b = j;
      const g = segs[i]; g.out = cs[b][0]; g.alts = [g.out, ...cands.map((x) => x[0]).filter((a) => a !== g.out)]; g.low = !this.lex.ne.freq.has(g.out);
    }
  }
  _nextSrc(segs, i) { for (let j = i + 1; j < segs.length; j++) { if (/^\s+$/.test(segs[j].out)) continue; return segs[j].word ? segs[j].src.toLowerCase() : ''; } return ''; }
  _nextMa(segs, i) { for (let j = i + 1; j < segs.length; j++) { if (/^\s+$/.test(segs[j].out)) continue; return segs[j].word && segs[j].src.toLowerCase() === 'ma'; } return false; }
  _prev(segs, i) { for (let j = i - 1; j >= 0; j--) { const o = segs[j].out; if (o.includes('\n')) return ''; if (!/^\s+$/.test(o)) return o; } return ''; }
  /** subject-verb agreement (हामी ..छौं / तिमी ..छौ / म ..एँ) and कि vs की */
  _agree(segs) {
    const F = this.lex.ne.freq; let subj = null, teach = false, lastSubj = -9;
    segs.forEach((g, i) => {
      let t = g.out;
      if (g.word && g.src.toLowerCase() === 'ma' && (t === 'म' || t === 'मा')) {
        const prev = this._prev(segs, i);
        let nx = ''; for (let j = i + 1; j < segs.length; j++) if (!/^\s+$/.test(segs[j].out)) { nx = segs[j].out; break; }
        g.out = (!(prev.endsWith('एको') && ['धन्यवाद', 'आभारी'].includes(nx)) && (MA_NEXT.has(nx) || nx === 'र' || prev === '' || ['।', '?', '!', ',', ';'].includes(prev) || PARTICLES.has(prev) || this.JOIN.has(prev) || VERBISH.some((v) => prev.endsWith(v)))) ? 'म' : 'मा';   // s14: nx==र (म र ...), 'एकोमा धन्यवाद'
        if (g.out !== t) g.alts = [g.out, ...g.alts.filter((a) => a !== g.out)]; t = g.out;
      }
      if (g.word && /^paraa?$/.test(normRoman(g.src.toLowerCase())) && ['पर', 'परा', 'पार', 'पारा'].includes(t)) {   // s14: "heroic para ma" (manner, पारा) vs "alik para" (far, पर)
        let ns = null; for (let j = i + 1; j < segs.length; j++) if (!/^\s+$/.test(segs[j].out)) { ns = segs[j]; break; }
        if (ns && ns.word && ns.src.toLowerCase() === 'ma') { g.out = 'पारा'; if (g.out !== t) g.alts = [g.out, ...g.alts.filter((a) => a !== g.out)]; t = g.out; }
      }
      if (g.word && g.src.toLowerCase() === 'khana' && (t === 'खाना' || t === 'खान') && KHAN_NEXT.has(this._nextSrc(segs, i))) { g.out = 'खान'; g.alts = [g.out, ...g.alts.filter((a) => a !== g.out)]; t = g.out; }   // s14: चिया खान गयौं / खान मन लाग्छ
      if (g.word && g.src.toLowerCase() === 'baba' && t === 'बाबा' && this._prev(segs, i) === 'र') {
        g.out = 'बुबा'; g.alts = [g.out, ...g.alts.filter((a) => a !== g.out)]; t = g.out;
      }
      if (g.word && g.src.toLowerCase() === 'pani' && (t === 'पनि' || t === 'पानी')) {
        const prev = this._prev(segs, i);
        let nxt = ''; for (let j = i + 1; j < segs.length; j++) if (!/^\s+$/.test(segs[j].out)) { nxt = segs[j].out; break; }
        if ((['यहाँ', 'त्यहाँ', 'कहाँ', 'धारामा', 'तातो', 'चिसो', 'सफा', 'फोहोर'].includes(prev) && ['छ', 'छैन', 'थियो', 'थिएन', 'पर्यो', 'पर्‍यो', 'पर्छ'].includes(nxt)) || (['आज', 'भोलि', 'हिजो', 'अहिले', 'त', 'धेरै', 'बिहानै'].includes(prev) && ['परेन', 'पर्दैन', 'परिरहेको', 'पर्दै'].includes(nxt))) {
          g.out = 'पानी'; if (g.out !== t) g.alts = [g.out, ...g.alts.filter((a) => a !== g.out)]; t = g.out;
        }
      }
      if (!g.word) { if (['।', '?', '!', ',', ';'].includes(t) || t.includes('\n')) { subj = null; teach = false; } return; }
      if (t.startsWith('शिक्षक') || t.startsWith('गुरु') || t.startsWith('मास्टर')) teach = true;
      if (teach && t.startsWith('पठाउ')) { g.out = 'पढाउ' + t.slice(4); if (g.out !== t) g.alts = [g.out, ...g.alts.filter((a) => a !== g.out)]; t = g.out; }
      if (t === 'र' && subj === '1s' && lastSubj === i - 2) { subj = '1p'; return; }   // s14: "ma ra mera sathi ..." = we
      if (SUBJ1P.has(t)) { subj = '1p'; lastSubj = i; return; } if (SUBJ1S.has(t)) { subj = '1s'; lastSubj = i; return; } if (SUBJ2.has(t)) { subj = '2'; lastSubj = i; return; }
      if (RESET.has(t)) { let nx = ''; for (let j = i + 1; j < segs.length; j++) if (!/^\s+$/.test(segs[j].out)) { nx = segs[j].out; break; } if (!NOT_SUBJ_NEXT.has(nx)) subj = null; return; }   // s14: "hami tapai lai ..." keeps the subject
      if (subj === '1p' && (t.endsWith('छौ') || t.endsWith('यौ') || t.endsWith('ेनौ'))) g.out = t + 'ँ';
      else if (subj === '2' && (t.endsWith('छौं') || t.endsWith('यौं') || t.endsWith('ेनौं') || t.endsWith('छौँ') || t.endsWith('यौँ') || t.endsWith('ेनौँ'))) g.out = t.slice(0, -1);
      else if (subj === '1s' && t.endsWith('ए') && (F.get(t) || 0) >= 3 && (F.get(t + 'ँ') || 0) >= 1) g.out = t + 'ँ';
      else if ((subj === '1s' || this._nextMa(segs, i)) && t.length >= 3 && t.endsWith('े') && t !== 'ले' && (F.get(t + 'ँ') || 0) >= 1 && (!t.endsWith('ने') || (t[t.length - 3] !== '्' && (F.get(t) || 0) < 50))) g.out = t + 'ँ';   // s14: देखे -> देखेँ, सरे -> सरेँ, भागे म -> भागेँ म
      else if (subj === '1s' && t.length >= 3 && t.endsWith('थे')) g.out = t + 'ँ';   // s14: बस्थे -> बस्थेँ
      else if (subj === '1s' && t.endsWith('ये') && (F.get(t.slice(0, -2) + 'एँ') || 0) >= 20) g.out = t.slice(0, -2) + 'एँ';
      if (t === 'की' || t === 'कि') {
        const prev = this._prev(segs, i);
        let nx = ''; for (let j = i + 1; j < segs.length; j++) if (!/^\s+$/.test(segs[j].out)) { nx = segs[j].out; break; }
        g.out = (KI_VERB.some((v) => prev.endsWith(v)) || (['', '।', '?', '!', ',', ';'].includes(prev) && KI_SUBJ.has(nx))) ? 'कि' : 'की';
      }
      if (g.out !== t) g.alts = [g.out, ...g.alts.filter((a) => a !== g.out)];
    });
  }
  async transliterate(text, opts = {}) { return (await this.analyze(text, opts)).map((s) => s.out).join(''); }
}
