// Sakhiya single-graph student runtime (Sakhiya-Micro / Pico / ladder tiers): phonemes -> 24 kHz waveform in ONE ONNX file.
// Front-end = the same docs/g2p.js (+ sanskrit.js) used by the older tiers, so Python / JS pronunciation stay identical.
// usage:  const s = new SakhiyaStudent(); await s.init("models_student/t5/model.int8.onnx"); const r = await s.say("नमस्ते।", {lang:"ne"});  // r.wav Float32Array, r.sr 24000
const ORT_URL_STUDENT = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";
class SakhiyaStudent {
  async init(url, ortUrl = ORT_URL_STUDENT) {
    if (!window.ort) await new Promise((res, rej) => { const s = document.createElement("script"); s.src = ortUrl + "ort.min.js"; s.onload = res; s.onerror = rej; document.head.appendChild(s); });
    ort.env.wasm.wasmPaths = ortUrl; ort.env.wasm.numThreads = Math.min(4, (typeof navigator !== "undefined" && navigator.hardwareConcurrency) || 2);
    this.sess = await ort.InferenceSession.create(url, { executionProviders: ["wasm"], graphOptimizationLevel: "all" });
    if (!window.SK.F) await window.SK.load();
    return this;
  }
  chunks(text) { return text.split(/(?<=[।॥?!])\s+/).map(s => s.trim()).filter(s => s.length > 1); }
  async chunk(text, lang, o) {
    const rate = o.rate || 1, basePitch = o.pitch || o.pitch_shift || 0, svDepth = o.svara ?? 1.0;
    let ph = null, dm = null, off = null, hasVedic = false;
    if (lang === "sa" && window.CHANDAS && window.CHANDAS.compileChandas) {
      const ch = window.CHANDAS.compileChandas(text, o.bpm || 75.0, o.tonic_hz || 215.0, true);
      ph = ch.ph; dm = ch.dm; off = ch.off; hasVedic = !!ch.has_vedic_marks;
    } else {
      ph = window.SK.textToPhones(window.SK.normalize(text, lang === "sa" ? "sa" : "ne"), lang === "sa" ? "sa" : "ne");
    }
    const S = window.SK.F.symbols, keep = ph.map(p => S.indexOf(p) >= 0);
    const ids = window.SK.encode(ph); if (ids.length < 2) return new Float32Array(0);
    let dsTensor, psTensor;
    if (dm && off) {
      const dmKeep = dm.filter((_, i) => keep[i]).map(v => v / rate);
      const svScale = (hasVedic ? 0.55 : 0.48) * svDepth;
      const offKeep = off.filter((_, i) => keep[i]).map(v => basePitch + v * svScale);
      dsTensor = new ort.Tensor("float32", Float32Array.from(dmKeep), [dmKeep.length]);
      psTensor = new ort.Tensor("float32", Float32Array.from(offKeep), [offKeep.length]);
    } else {
      dsTensor = new ort.Tensor("float32", Float32Array.of(1 / rate), [1]);
      psTensor = new ort.Tensor("float32", Float32Array.of(basePitch), [1]);
    }
    const out = await this.sess.run({
      ids: new ort.Tensor("int64", BigInt64Array.from(ids.map(BigInt)), [1, ids.length]),
      lang: new ort.Tensor("int64", BigInt64Array.of(lang === "sa" ? 1n : 0n), [1]),
      dur_scale: dsTensor, pitch_shift: psTensor
    });
    return out.wav.data;
  }
  async say(text, o = {}) {
    const lang = o.lang === "sa" ? "sa" : "ne", t0 = (typeof performance !== "undefined" ? performance : Date).now(), SR = 24000, gap = new Float32Array(Math.round(SR * (o.gap ?? 0.18)));
    const parts = []; let n = 0;
    for (const c of this.chunks(text)) { const w = await this.chunk(c, lang, o); parts.push(w, gap); n += w.length + gap.length; }
    const wav = new Float32Array(n); let k = 0; for (const p of parts) { wav.set(p, k); k += p.length; }
    const dt = ((typeof performance !== "undefined" ? performance : Date).now() - t0) / 1000;
    return { wav, sr: SR, rtf: dt / Math.max(1e-3, n / SR) };
  }
}
if (typeof window !== "undefined") window.SakhiyaStudent = SakhiyaStudent;
