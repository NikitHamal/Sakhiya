// Context-aware Roman -> Devanagari front-end: ensemble char transformer + context re-ranker + subword LM
// (engine and weights from NikitHamal/transliterationlm, ~20 MB, lazy-loaded the first time Roman text is typed).
// Falls back to the lexicon/rule converter in g2p.js if the engine cannot load.
(() => {
  let eng = null, state = "idle";
  const listeners = new Set(); SK.onRomanState = f => listeners.add(f);
  const set = s => { state = s; listeners.forEach(f => f(s)); };
  function engine() {
    if (!eng) eng = (async () => {
      if (!window.ort) throw new Error("onnxruntime-web not ready");
      set("loading"); const { Transliterator } = await import("./translit/tlm.js");
      const T = await Transliterator.create({ ort: window.ort, base: "translit", sessionOptions: { executionProviders: ["wasm"], graphOptimizationLevel: "all" } });
      set("ready"); return T;
    })().catch(e => { console.warn("roman engine unavailable, using lexicon fallback:", e); set("fallback"); eng = null; return null; });
    return eng;
  }
  SK.normalizeAsync = async (text, lang) => {
    if (lang === "sa" && window.SA) return window.SA.iastToDeva(text.normalize("NFC"));
    let t = SK.expand(text);
    if (!/[A-Za-z]/.test(t)) return SK.normalize(t, lang);
    if (lang.startsWith("thr")) t = t.replace(/[A-Za-z][A-Za-z'~]*/g, m => SK.strong(m, lang) ?? m);
    if (/[A-Za-z]/.test(t)) {
      const T = await engine();
      if (T) { try { t = await T.transliterate(t, { lang: "ne", mode: "byakaran" }); } catch (e) { console.warn("transliterate failed", e); } }
    }
    return SK.normalize(t, lang);
  };
  SK.romanState = () => state;
})();
