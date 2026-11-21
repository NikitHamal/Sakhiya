// `Auto (Mixed)` clause router: faithful port of sakhiya/mixed.py (parity-tested by tests/test_mixed.py).
// A manually chosen language chip is never overridden; only lang === "auto" routes per clause.
(() => {
  const SA_HARD = /[ॐ॥\u0951\u0952ऽ]/, IAST = /[āīūṛṝḷṅñṭḍṇśṣṃṁḥ]/;
  const NE_VISARGA = new Set(["अतः","प्रायः","पुनः","मूलतः","प्रातः","सम्भवतः","संभवतः","विशेषतः","वस्तुतः","अन्ततः","क्रमशः","मुख्यतः","प्रायशः","स्वतः","सर्वप्रथमतः","प्रथमतः","अन्तःकरण"]);
  const THARU_STRONG = new Set(["संघरियाहुक्रे","संघरिया","संघरियाहुक्रेन","बनाइल","बाटै","बाटु","बाटी","बडयी","बडिय","हलयी","नाहै","तोहार","तोहर","मोर","हमाल","हमार","हुक्रे","कहथ","जाथ","खाइल","गइल","रहल"]);
  const THARU_WEAK = new Set(["हों","बा","मै","जी","छोट","हइ"]);
  const LETTER = /[\u0900-\u0963\u0966-\u097FA-Za-z0-9]/;
  function splitClauses(text) {
    const t = String(text).normalize("NFC"), out = [];
    for (let c of t.split(/(?<=[।॥!?])\s*|(?<=\.)(?=\s|$)\s*|\n+/u)) {
      if (!c || !c.trim()) continue; c = c.trim();
      if (!LETTER.test(c) && out.length) out[out.length - 1] += c; else out.push(c);
    }
    return out;
  }
  function detectLang(clause, tharu = "thr_dangaura") {
    const c = clause.normalize("NFC");
    if (SA_HARD.test(c) || IAST.test(c)) return "sa";
    const words = c.match(/[\u0900-\u0963\u0966-\u097F]+/g) || [];
    const strong = words.filter(w => THARU_STRONG.has(w)).length, weak = words.filter(w => THARU_WEAK.has(w)).length;
    if (strong >= 1 || weak >= 3) return tharu;
    const vis = words.filter(w => w.endsWith("ः") && !NE_VISARGA.has(w)), halm = words.filter(w => w.endsWith("म्") && [...w].length > 3);
    if (vis.length || halm.length >= 2) return "sa";
    return "ne";
  }
  function route(text, tharu = "thr_dangaura") {
    const segs = [];
    for (const c of splitClauses(text)) { const L = detectLang(c, tharu); if (segs.length && segs[segs.length - 1][0] === L) segs[segs.length - 1][1] += " " + c; else segs.push([L, c]); }
    return segs;
  }
  window.MIX = { splitClauses, detectLang, route };
})();
