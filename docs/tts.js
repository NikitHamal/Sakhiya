// In-browser Sakhiya synthesis: onnxruntime-web (WASM) + iSTFT + WebAudio. No server involved.
const ORT_URL = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";
const STYLES = {
  neutral: { style_id: 0, rate: 0.96, pitch_shift: 0.0, pitch_scale: 1.12, energy_scale: 1.0, pause_scale: 1.0 },
  news: { style_id: 0, rate: 0.96, pitch_shift: -0.03, pitch_scale: 1.12, energy_scale: 1.06, pause_scale: 1.0 },
  conversational: { style_id: 1, rate: 1.00, pitch_shift: 0.04, pitch_scale: 1.32, energy_scale: 1.02, pause_scale: 0.88 },
  storytelling: { style_id: 2, rate: 0.90, pitch_shift: 0.02, pitch_scale: 1.32, energy_scale: 1.04, pause_scale: 1.35 },
  warm_soft: { style_id: 3, rate: 0.96, pitch_shift: -0.03, pitch_scale: 1.18, energy_scale: 0.92, pause_scale: 1.15 },
  slow_clear: { style_id: 3, rate: 0.78, pitch_shift: 0.0, pitch_scale: 1.0, energy_scale: 1.0, pause_scale: 1.3 },
  expressive: { style_id: 4, rate: 1.00, pitch_shift: 0.08, pitch_scale: 1.32, energy_scale: 1.08, pause_scale: 0.92 },
  vedic_chant: { style_id: 5, rate: 0.90, pitch_shift: 0.0, pitch_scale: 0.92, energy_scale: 1.04, pause_scale: 1.28 },
  chant: { style_id: 5, rate: 0.90, pitch_shift: 0.0, pitch_scale: 0.92, energy_scale: 1.04, pause_scale: 1.28 },  // Sanskrit recitation: flat tonic, measured laya, pada pauses
};
class FFT {  // radix-2 complex FFT used for the inverse real FFT of every vocoder frame
  constructor(n) { this.n = n; this.cos = new Float32Array(n / 2); this.sin = new Float32Array(n / 2); this.rev = new Uint32Array(n);
    for (let i = 0; i < n / 2; i++) { this.cos[i] = Math.cos(2 * Math.PI * i / n); this.sin[i] = Math.sin(2 * Math.PI * i / n); }
    const bits = Math.log2(n); for (let i = 0; i < n; i++) { let r = 0; for (let b = 0; b < bits; b++) r |= ((i >> b) & 1) << (bits - 1 - b); this.rev[i] = r; } }
  transform(re, im) {  // forward-style butterflies with +i twiddles (== unnormalised inverse DFT)
    const n = this.n, rev = this.rev;
    for (let i = 0; i < n; i++) { const j = rev[i]; if (j > i) { let t = re[i]; re[i] = re[j]; re[j] = t; t = im[i]; im[i] = im[j]; im[j] = t; } }
    for (let size = 2; size <= n; size <<= 1) { const half = size >> 1, step = n / size;
      for (let i = 0; i < n; i += size) for (let j = 0, k = 0; j < half; j++, k += step) { const a = i + j, b = a + half, wr = this.cos[k], wi = this.sin[k];
        const tr = re[b] * wr - im[b] * wi, ti = re[b] * wi + im[b] * wr; re[b] = re[a] - tr; im[b] = im[a] - ti; re[a] += tr; im[a] += ti; } } }
}
const EQ_PTS = [1.018, 1.215, 1.468, 1.438, 1.096, 1.1, 1.166, 1.158, 1.207, 1.25, 1.291, 1.326, 1.366, 1.382, 1.437, 1.443, 1.454, 1.449, 1.46, 1.46, 1.502, 1.569, 1.684, 1.793, 2.042, 2.219, 2.358, 2.451, 2.468, 2.49, 2.38, 2.251, 2.15, 2.146, 2.076, 2.151, 2.275, 2.267, 2.5, 2.646, 2.889, 2.923, 3.031, 3.137, 3.198, 3.008, 2.876, 2.936, 2.67, 2.499, 2.42, 2.424, 2.236, 2.278, 2.231, 2.166, 2.255, 2.239, 2.313, 2.3, 2.407, 2.306, 2.294, 2.108, 2.0, 1.964, 1.87, 1.858, 1.843, 1.916, 1.799, 1.786, 1.788, 1.745, 1.729, 1.708, 1.677, 1.664, 1.626, 1.62, 1.615, 1.654, 1.661, 1.661, 1.723, 1.877, 1.897, 1.886, 2.044, 2.266, 2.216, 2.202, 2.213, 2.487, 2.362, 2.283, 2.365, 2.484, 2.205, 2.1, 1.99, 1.898, 1.77, 1.678, 1.668, 1.724, 1.649, 1.665, 1.644, 1.554, 1.486, 1.577, 1.599, 1.603, 1.692, 1.715, 1.624, 1.685, 1.706, 1.687, 1.768, 1.921, 1.894, 2.153, 2.443, 2.527, 2.228, 1.863, 1.317];
class SakhiyaWeb {
  constructor() {
    this.sess = {}; this.cfg = null; this.fft = new FFT(1024); this.fft64 = new FFT(64);
    this.win = Float32Array.from({ length: 1024 }, (_, i) => 0.5 - 0.5 * Math.cos(2 * Math.PI * i / 1024));
    this.win64 = Float32Array.from({ length: 64 }, (_, i) => 0.5 - 0.5 * Math.cos(2 * Math.PI * i / 64));
    this.eq = Float32Array.from({ length: 513 }, (_, k) => { const x = k * (EQ_PTS.length - 1) / 512, i = Math.min(EQ_PTS.length - 2, x | 0), f = x - i; return Math.pow(EQ_PTS[i] * (1 - f) + EQ_PTS[i + 1] * f, 0.78); });
    this.variant = null;
  }
  _ola64(re, im, T) {
    const N = 64, H = 16, L = (T - 1) * H + N, y = new Float32Array(L), env = new Float32Array(L), a = new Float32Array(N), b = new Float32Array(N);
    for (let t = 0; t < T; t++) {
      for (let k = 0; k <= 32; k++) { a[k] = re[k * T + t]; b[k] = im[k * T + t]; }
      a[0] = re[t]; b[0] = 0; b[32] = 0; for (let k = 1; k < 32; k++) { a[N - k] = a[k]; b[N - k] = -b[k]; }
      this.fft64.transform(a, b);
      for (let n = 0; n < N; n++) { const w = this.win64[n]; y[t * H + n] += (a[n] / N) * w; env[t * H + n] += w * w; }
    }
    const p = (N - H) / 2, out = new Float32Array(L - 2 * p); let pk = 1e-4;
    for (let i = 0; i < out.length; i++) { out[i] = y[i + p] / Math.max(env[i + p], 1e-8); const ab = Math.abs(out[i]); if (ab > pk) pk = ab; }
    const sc = 0.90 / pk; for (let i = 0; i < out.length; i++) out[i] = Math.max(-1, Math.min(1, out[i] * sc));
    if (out.length > 256) for (let i = 0; i < 128; i++) { const r = 0.5 - 0.5 * Math.cos(Math.PI * i / 128); out[i] *= r; out[out.length - 1 - i] *= r; }
    return out;
  }
  async init(variant = "fp16", base = "models/") {
    if (!window.ort) await new Promise((res, rej) => { const s = document.createElement("script"); s.src = ORT_URL + "ort.min.js"; s.onload = res; s.onerror = rej; document.head.appendChild(s); });
    ort.env.wasm.wasmPaths = ORT_URL; ort.env.wasm.numThreads = Math.min(4, navigator.hardwareConcurrency || 2);
    if (this.base !== base) { this.base = base; this.cfg = null; this.variant = null; this.sess = {}; }
    this.cfg = this.cfg || await (await fetch(base + "config.json", { cache: "no-cache" })).json();
    if (this.variant === variant) return; const suf = variant === "tiny" ? ".int8.onnx" : ".onnx", t0 = performance.now();
    if (this.cfg && this.cfg.single_graph) {
      try { this.sess.single = await ort.InferenceSession.create(base + "model" + suf, { executionProviders: ["wasm"], graphOptimizationLevel: "all" }); }
      catch (_) { this.sess.single = await ort.InferenceSession.create(base + "model.int8.onnx", { executionProviders: ["wasm"], graphOptimizationLevel: "all" }); }
      this.variant = variant; this.loadMs = performance.now() - t0; return;
    }
    for (const n of ["encoder", "decoder", "vocoder"]) {
      try { this.sess[n] = await ort.InferenceSession.create(base + n + suf, { executionProviders: ["wasm"], graphOptimizationLevel: "all" }); }
      catch (_) { this.sess[n] = await ort.InferenceSession.create(base + n + ".onnx", { executionProviders: ["wasm"], graphOptimizationLevel: "all" }); }
    }
    if (!this.sess.sanskrit) {
      try { this.sess.sanskrit = await ort.InferenceSession.create(base + "sanskrit.onnx", { executionProviders: ["wasm"], graphOptimizationLevel: "all" }); }
      catch (_) { this.sess.sanskrit = null; }
    }
    this.variant = variant; this.loadMs = performance.now() - t0;
  }
  async chunkSanskrit(text, o) {
    const s = window.SA.phonemizeSlp1(text), T = s.ids.length, sv = o.svara ?? 1.0;
    const i64 = a => new ort.Tensor("int64", BigInt64Array.from(a.map(BigInt)), [1, a.length]);
    const f1 = v => new ort.Tensor("float32", Float32Array.of(v), [1]);
    const vId = [1, 6, 20, 21].includes(Number(o.spk)) ? 1 : 0;
    const pAdd = Float32Array.from(s.off.map(v => v * 0.125 * sv));
    const r = await this.sess.sanskrit.run({
      tokens: i64(s.ids), weight: i64(s.weight), voice: new ort.Tensor("int64", BigInt64Array.of(BigInt(vId)), [1]),
      speed: f1(o.rate), pitch_shift: f1(o.pitch_shift * 0.12), pitch_range: f1(o.pitch_scale),
      pitch_add: new ort.Tensor("float32", pAdd, [1, T])
    });
    const out = new Float32Array(r.audio.data); let pk = 1e-4;
    for (let i = 0; i < out.length; i++) { const a = Math.abs(out[i]); if (a > pk) pk = a; }
    const sc = 0.90 / pk; for (let i = 0; i < out.length; i++) out[i] = Math.max(-1, Math.min(1, out[i] * sc));
    if (out.length > 256) for (let i = 0; i < 128; i++) { const w = 0.5 - 0.5 * Math.cos(Math.PI * i / 128); out[i] *= w; out[out.length - 1 - i] *= w; }
    const F = Math.max(1, (out.length / 256) | 0), mel = new Float32Array(80 * F), sre = new Float32Array(513 * F), sim = new Float32Array(513 * F);
    this._stft(out, F, sre, sim);
    for (let m = 0; m < 80; m++) { const k0 = Math.floor(Math.pow(m / 80, 1.6) * 480) + 2, k1 = Math.max(k0 + 1, Math.floor(Math.pow((m + 1) / 80, 1.6) * 480) + 2);
      for (let f = 0; f < F; f++) { let e = 1e-5; for (let k = k0; k < k1; k++) e += Math.hypot(sre[k * F + f], sim[k * F + f]); mel[m * F + f] = Math.log(e / (k1 - k0)) - 2.2; } }
    return { mel, F, wav: out, phs: s.syms.join(" ") };
  }
  async chunk(ph, o) {
    const S = SK.F.symbols, ids = BigInt64Array.from(SK.encode(ph).map(BigInt)), f32 = v => new ort.Tensor("float32", Float32Array.of(v), []);
    if (this.sess.single) {
      const phS = (ph.length && ph[0] === "|") ? ph : ["|", ...ph];
      const idsS = BigInt64Array.from(SK.encode(phS).map(BigInt));
      const keep = ph.map(p => S.indexOf(p) >= 0), rate = o.rate || 1, basePitch = o.pitch_shift || 0, svDepth = o.svara ?? 1;
      const svScale = (o.lang === "sa" ? (o.has_vedic_marks ? 0.65 : 0.52) : 0.35) * svDepth;
      const dmBase = o.lang === "sa" ? (o.dm || ph.map(() => 1)).filter((_, i) => keep[i]).map(v => (0.75 + 0.25 * Math.min(1.35, Math.max(0.85, v))) / rate) : [0.94 / rate];
      const dmArr = (o.lang === "sa" && dmBase.length === ids.length) ? [0.88 / rate, ...dmBase] : dmBase;
      const offBase = o.lang === "sa" ? (o.off || ph.map(() => 0)).filter((_, i) => keep[i]).map(v => basePitch + v * svScale) : [basePitch + 0.04];
      const offArr = (o.lang === "sa" && offBase.length === ids.length) ? [basePitch, ...offBase] : offBase;
      const r = await this.sess.single.run({
        ids: new ort.Tensor("int64", idsS, [1, idsS.length]),
        lang: new ort.Tensor("int64", BigInt64Array.of(o.lang === "sa" ? 1n : 0n), [1]),
        dur_scale: new ort.Tensor("float32", Float32Array.from(dmArr), [dmArr.length]),
        pitch_shift: new ort.Tensor("float32", Float32Array.from(offArr), [offArr.length]),
      });
      const wav = new Float32Array(r.wav.data); let pk = 1e-4;
      for (let i = 0; i < wav.length; i++) pk = Math.max(pk, Math.abs(wav[i]));
      const sc = 0.90 / pk; for (let i = 0; i < wav.length; i++) wav[i] = Math.max(-1, Math.min(1, wav[i] * sc));
      if (wav.length > 256) for (let i = 0; i < 128; i++) { const w = 0.5 - 0.5 * Math.cos(Math.PI * i / 128); wav[i] *= w; wav[wav.length - 1 - i] *= w; }
      const F = Math.max(1, (wav.length / 256) | 0), mel = new Float32Array(80 * F), sre = new Float32Array(513 * F), sim = new Float32Array(513 * F);
      this._stft(wav, F, sre, sim);
      for (let m = 0; m < 80; m++) { const k0 = Math.floor(Math.pow(m / 80, 1.6) * 480) + 2, k1 = Math.max(k0 + 1, Math.floor(Math.pow((m + 1) / 80, 1.6) * 480) + 2);
        for (let f = 0; f < F; f++) { let e = 1e-5; for (let k = k0; k < k1; k++) e += Math.hypot(sre[k * F + f], sim[k * F + f]); mel[m * F + f] = Math.log(e / (k1 - k0)) - 2.2; } }
      return { mel, F, wav };
    }
    const i64 = v => new ort.Tensor("int64", BigInt64Array.of(BigInt(v)), [1]);
    const isV2 = /^sakhiya-(v2|v3|v4|v6|v15|one|nano|atom|pico)/.test(String((this.cfg && this.cfg.arch) || ""));
    const lock = !!(o.lock_voice ?? (this.cfg && this.cfg.lock_voice)), effLang = (isV2 && (String(o.lang) !== "sa" || lock)) ? "ne" : o.lang;
    const feeds = { ids: new ort.Tensor("int64", ids, [1, ids.length]), spk: i64(o.spk), lang: i64((this.cfg.lang2id || SK.F.lang2id)[effLang] ?? 0), pitch_shift: f32(o.pitch_shift), pitch_scale: f32(o.pitch_scale), energy_scale: f32(o.energy_scale) };
    if (this.sess.encoder.inputNames.includes("pitch_add")) {   // Sanskrit svara + laya melody: per-phone pitch offsets
      const svScale = isV2 ? (o.lang === "sa" ? (o.has_vedic_marks ? 0.52 : 0.48) : 1.15) : 1.12;
      const keep = ph.map(p => S.indexOf(p) >= 0), off = (o.off || ph.map(() => 0)).filter((_, i) => keep[i]).map(v => v * svScale * (o.svara ?? 1)); o._dm = (o.dm || ph.map(() => 1)).filter((_, i) => keep[i]);
      feeds.pitch_add = new ort.Tensor("float32", Float32Array.from(off), [1, off.length]);
    }
    if (this.sess.encoder.inputNames.includes("style")) {
      feeds.style = i64(o.style_id ?? 0);
    }
    if (this.sess.encoder.inputNames.includes("contour")) {
      const L = ids.length, cArr = new Float32Array(L * 3);
      const cObj = o.contour || (typeof window !== "undefined" && window.StudioV6 && window.StudioV6.cartridge && window.StudioV6.cartridge() && window.StudioV6.cartridge().contour);
      if (cObj && Array.isArray(cObj.f0) && cObj.f0.length > 0) {
        const nC = cObj.f0.length;
        for (let i = 0; i < L; i++) {
          const idx = Math.min(nC - 1, Math.max(0, Math.round(i * (nC - 1) / Math.max(1, L - 1))));
          cArr[i * 3 + 0] = Number(cObj.f0[idx] || 0);
          cArr[i * 3 + 1] = Number((cObj.en && cObj.en[idx]) || 0);
          cArr[i * 3 + 2] = Number((cObj.dur && cObj.dur[idx]) || 0);
        }
      } else if (Array.isArray(cObj) && cObj.length > 0 && Array.isArray(cObj[0])) {
        const nC = cObj.length;
        for (let i = 0; i < L; i++) {
          const idx = Math.min(nC - 1, Math.max(0, Math.round(i * (nC - 1) / Math.max(1, L - 1))));
          cArr[i * 3 + 0] = Number(cObj[idx][0] || 0);
          cArr[i * 3 + 1] = Number(cObj[idx][1] || 0);
          cArr[i * 3 + 2] = Number(cObj[idx][2] || 0);
        }
      }
      feeds.contour = new ort.Tensor("float32", cArr, [1, L, 3]);
    }
    const e = await this.sess.encoder.run(feeds);
    const T = ids.length, d = e.x.dims[2], ld = e.logdur.data, xs = e.x.data, pause = new Set(["|", ...SK.F.punct].map(p => S.indexOf(p)));
    const phValid = ph.filter(p => S.indexOf(p) >= 0), pSet = new Set(["|", ",", ".", "?", "!", ";"]), dur = new Int32Array(T); let F = 0;
    for (let t = 0; t < T; t++) {
      let v = Math.max(Math.exp(ld[t]) - 1, 1) * (o._dm && o._dm[t] ? o._dm[t] : 1);
      if (pause.has(Number(ids[t]))) v *= o.pause_scale;
      let d = Math.max(Math.round(v / o.rate), 1);
      const p = phValid[t];
      if (!pSet.has(p)) d = Math.max(d, 3);
      if (p === "h" && (t + 1 === T || pSet.has(phValid[t + 1]))) d = Math.min(d, 6);
      dur[t] = d;
    }
    for (let t = 0; t + 1 < T; t++) {
      if (t >= 1 && phValid[t] === "|" && phValid[t - 1] === phValid[t + 1] && ["a","i","u","e","o","ʌ"].includes(phValid[t - 1])) {
        if (t >= 2 && phValid[t - 2] === "|") {
          dur[t - 2] = 12;
          for (let b = 0; b < 100; b++) xs[(t - 2) * d + b] = xs[(T - 1) * d + b];
        }
        dur[t - 1] = Math.max(19, dur[t - 1]); dur[t] = 7; dur[t + 1] = Math.max(18, dur[t + 1]);
        for (let b = 0; b < 100; b++) {
          const vNext = xs[(t + 1) * d + b], vSil = xs[(T - 1) * d + b];
          xs[(t - 1) * d + b] = vNext;
          xs[t * d + b] = 0.45 * vNext + 0.55 * vSil;
        }
      }
      if (o.lang === "sa" && phValid[t] === "o" && phValid[t + 1] === "m" && (t === 0 || ["|", ",", ";", "."].includes(phValid[t - 1]))) {
        dur[t] = Math.min(44, Math.max(34, dur[t]));
        dur[t + 1] = Math.min(22, Math.max(16, dur[t + 1]));
        if (t + 2 < T && phValid[t + 2] === ",") dur[t + 2] = Math.min(22, Math.max(14, dur[t + 2]));
      }
      if (t >= 1 && t + 2 < T && (phValid[t - 1] === "|" || phValid[t - 1] === ",") && (phValid[t + 2] === "|" || phValid[t + 2] === ",") && ["r","t̪","n","m","h","tsʰ"].includes(phValid[t]) && ["ʌ","o"].includes(phValid[t + 1])) {
        dur[t] = Math.min(9, Math.max(5, dur[t]));
        dur[t + 1] = Math.min(11, Math.max(6, dur[t + 1]));
      }
    }
    for (let t = 0; t < T; t++) F += dur[t];
    if (!o.is_last && T > 0) { F -= dur[T - 1]; dur[T - 1] = Math.max(1, Math.round(7 * o.pause_scale)); F += dur[T - 1]; }
    const xe = new Float32Array(F * d); let f = 0; for (let t = 0; t < T; t++) for (let k = 0; k < dur[t]; k++, f++) xe.set(xs.subarray(t * d, (t + 1) * d), f * d);
    const mel = (await this.sess.decoder.run({ x: new ort.Tensor("float32", xe, [1, F, d]) })).mel;
    const sp = await this.sess.vocoder.run({ mel });
    let wav;
    if (sp.wav) {
      wav = new Float32Array(sp.wav.data); let pk = 1e-4;
      for (let i = 0; i < wav.length; i++) pk = Math.max(pk, Math.abs(wav[i]));
      const sc = 0.90 / pk; for (let i = 0; i < wav.length; i++) wav[i] = Math.max(-1, Math.min(1, wav[i] * sc));
      if (wav.length > 256) for (let i = 0; i < 128; i++) { const r = 0.5 - 0.5 * Math.cos(Math.PI * i / 128); wav[i] *= r; wav[wav.length - 1 - i] *= r; }
    } else {
      wav = sp.real.dims[1] === 33 ? this._ola64(sp.real.data, sp.imag.data, sp.real.dims[2]) : this.istft(sp.real.data, sp.imag.data, F);
    }
    return { mel: mel.data, F, wav };
  }
  _ola(re, im, T) {
    const N = 1024, H = 256, L = (T - 1) * H + N, y = new Float32Array(L), env = new Float32Array(L), a = new Float32Array(N), b = new Float32Array(N);
    for (let t = 0; t < T; t++) {
      for (let k = 0; k <= 512; k++) { a[k] = re[k * T + t]; b[k] = im[k * T + t]; }
      a[0] = re[t]; b[0] = 0; b[512] = 0; for (let k = 1; k < 512; k++) { a[N - k] = a[k]; b[N - k] = -b[k]; }
      this.fft.transform(a, b);
      for (let n = 0; n < N; n++) { const w = this.win[n]; y[t * H + n] += (a[n] / N) * w; env[t * H + n] += w * w; }
    }
    const p = (N - H) / 2, out = new Float32Array(L - 2 * p);
    for (let i = 0; i < out.length; i++) out[i] = y[i + p] / Math.max(env[i + p], 1e-8);
    return out;
  }
  _stft(w, T, sre, sim) {
    const N = 1024, H = 256, p = (N - H) / 2, wl = w.length, a = new Float32Array(N), b = new Float32Array(N);
    for (let t = 0; t < T; t++) {
      for (let n = 0; n < N; n++) { let idx = t * H + n - p; if (idx < 0) idx = -idx; else if (idx >= wl) idx = 2 * wl - 2 - idx; a[n] = (w[idx] || 0) * this.win[n]; b[n] = 0; }
      this.fft.transform(a, b);
      for (let k = 0; k <= 512; k++) { sre[k * T + t] = a[k]; sim[k * T + t] = -b[k]; }
    }
  }
  istft(re, im, T) {  // Voicedness-gated consistent-phase harmonic reconstruction (identical to sakhiya.vocoder.istft_same)
    const w0 = this._ola(re, im, T), nBins = 513, tot = nBins * T;
    if (w0.length < 1024) { let pk = 1e-4; for (let i = 0; i < w0.length; i++) pk = Math.max(pk, Math.abs(w0[i])); const sc = 0.90 / pk; for (let i = 0; i < w0.length; i++) w0[i] = Math.max(-1, Math.min(1, w0[i] * sc)); return w0; }
    const sre = new Float32Array(tot), sim = new Float32Array(tot), pRe = new Float32Array(tot), pIm = new Float32Array(tot), mc = new Float32Array(tot), mbCol = new Float32Array(nBins), mpCol = new Float32Array(nBins);
    this._stft(w0, T, sre, sim); pRe.set(sre); pIm.set(sim);
    for (let t = 0; t < T; t++) {
      let sumMb = 1e-6, sumLow = 0;
      for (let k = 0; k < nBins; k++) { const idx = k * T + t, m0 = Math.hypot(re[idx], im[idx]), mr = Math.hypot(sre[idx], sim[idx]), wm = k < 160 ? 0.35 : 0.22, v = wm * m0 + (1 - wm) * mr; mbCol[k] = v; sumMb += v; if (k >= 4 && k < 84) sumLow += v; }
      const vo = Math.max(0, Math.min(1, (sumLow / sumMb - 0.35) / 0.35)), ct = 1.14 + 0.34 * vo; let sumMp = 1e-6;
      for (let k = 0; k < nBins; k++) { const p = Math.pow(Math.max(mbCol[k], 1e-5), ct); mpCol[k] = p; sumMp += p; }
      const norm = sumMb / sumMp;
      for (let k = 0; k < nBins; k++) { const idx = k * T + t, m = mpCol[k] * norm * this.eq[k], ang = Math.atan2(sim[idx], sre[idx]); mc[idx] = m; sre[idx] = m * Math.cos(ang); sim[idx] = m * Math.sin(ang); }
    }
    for (let it = 0; it < 2; it++) {
      const wi = this._ola(sre, sim, T); this._stft(wi, T, sre, sim);
      for (let i = 0; i < tot; i++) { const cr = sre[i], ci = sim[i], er = cr + 0.27 * (cr - pRe[i]), ei = ci + 0.27 * (ci - pIm[i]), ang = Math.atan2(ei, er), m = mc[i]; pRe[i] = cr; pIm[i] = ci; sre[i] = m * Math.cos(ang); sim[i] = m * Math.sin(ang); }
    }
    const out = this._ola(sre, sim, T); let pk = 1e-4;
    for (let i = 0; i < out.length; i++) { const a = Math.abs(out[i]); if (a > pk) pk = a; }
    const sc = 0.90 / pk; for (let i = 0; i < out.length; i++) out[i] = Math.max(-1, Math.min(1, out[i] * sc));
    if (out.length > 256) for (let i = 0; i < 128; i++) { const r = 0.5 - 0.5 * Math.cos(Math.PI * i / 128); out[i] *= r; out[out.length - 1 - i] *= r; }
    return out;
  }
  stitch(wavs, sr, xfMs = 25, edgeMs = 90) {   // port of sakhiya.onnx_tts.stitch: trim silent edges, equal-power crossfade
    const clips = wavs.map(w => { let mx = 0; for (let i = 0; i < w.length; i++) mx = Math.max(mx, Math.abs(w[i])); const thr = Math.max(mx * 0.006, 1e-4); let a = 0, b = w.length - 1;
      while (a < w.length && Math.abs(w[a]) <= thr) a++; while (b >= 0 && Math.abs(w[b]) <= thr) b--; if (a > b) return w; const keep = Math.floor(edgeMs * sr / 1000); return w.slice(Math.max(0, a - keep), b + 1 + keep); });
    const n = Math.floor(xfMs * sr / 1000); let out = clips[0] || new Float32Array(0);
    for (const w of clips.slice(1)) { const m = Math.min(n, out.length, w.length); if (m < 8) { const o2 = new Float32Array(out.length + w.length); o2.set(out); o2.set(w, out.length); out = o2; continue; }
      const o2 = new Float32Array(out.length + w.length - m); o2.set(out.subarray(0, out.length - m)); for (let i = 0; i < m; i++) { const t = Math.PI / 2 * i / (m - 1); o2[out.length - m + i] = out[out.length - m + i] * Math.cos(t) + w[i] * Math.sin(t); } o2.set(w.subarray(m), out.length); out = o2; }
    return out;
  }
  async sayMixed(text, opt) {
    const t0 = performance.now(), segs = window.MIX.route(text, opt.tharu || "thr_dangaura"), res = [];
    for (const [L, seg] of segs) res.push(await this.say(seg, Object.assign({}, opt, { lang: L })));
    let sr = res[0].sr; const ws = res.map(r => { if (r.sr === sr) return r.wav; const o = new Float32Array(Math.floor(r.wav.length * sr / r.sr)); for (let i = 0; i < o.length; i++) { const x = i * r.sr / sr, i0 = Math.floor(x), f = x - i0; o[i] = r.wav[i0] * (1 - f) + (r.wav[Math.min(i0 + 1, r.wav.length - 1)]) * f; } return o; });
    const wav = this.stitch(ws, sr), ms = performance.now() - t0;
    return { wav, sr, ms, rtf: ms / 1000 / (wav.length / sr), norm: res.map(r => r.norm).join(" "), phones: res.map(r => r.phones).join(" | "), mel: res.flatMap(r => r.mel), segments: segs.map(x => ({ lang: x[0], text: x[1] })) };
  }
  async say(text, opt = {}) {
    if (opt.lang === "auto") return this.sayMixed(text, opt);
    const o = Object.assign({ lang: "ne", spk: 0 }, STYLES[opt.style || "neutral"], opt), useSa = o.lang === "sa" && this.sess.sanskrit && window.SA && window.SA.phonemizeSlp1, sr = useSa ? 24000 : this.cfg.sample_rate, t0 = performance.now();
    if (o.lang === "sa" && !["chant", "vedic_chant", "storytelling"].includes(opt.style) && !opt.rate) { o.rate = useSa ? ([1, 6, 20, 21].includes(Number(o.spk)) ? 1.00 : 0.98) : o.rate * 0.96; o.pause_scale *= 1.05; if (!opt.pitch_scale) o.pitch_scale = 1.00; }
    if (String(o.lang).startsWith("thr")) {
      if (o.lang === "thr_rana" || o.lang === "thr_chitwania") { o.rate *= 0.90; o.pitch_scale = 1.00; o.pause_scale *= 1.05; }
      else { o.rate *= 0.90; o.pitch_scale = 1.00; o.pause_scale *= 1.00; }
    }
    const FIN = new Set(["छ","छन्","छु","छौ","छैन","छैनन्","हो","हुन्","थियो","थिए","थिएन","भयो","भए","गर्यो","गरे","गर्छ","गर्छन्","हुन्छ","पर्छ","सक्छ","गयो","आयो","दियो","भन्यो","पाइन्छ","मानिन्छ","गरियो","बा","बाटै","बाटु","रहे","है","नाहै","बडयी","बडिय","हलयी"]);
    const norm = await SK.normalizeAsync(text, o.lang), saSegs = norm.split(/(?<=[।॥])\s*|(?<=[.?!])\s+/).filter(s => /[\u0900-\u097F]/.test(s)), raw = (o.lang === "sa" && norm.trim().split(/\s+/).length <= 45 && saSegs.length <= 2) ? [norm.trim()] : saSegs, parts = [], outs = [], phs = [];
    for (const s of raw.length ? raw : [norm]) {
      const ws = s.trim().split(/\s+/); if (ws.length <= 20) { parts.push(s); continue; }
      let cur = []; for (let i = 0; i < ws.length; i++) { cur.push(ws[i]); if ((FIN.has(ws[i]) && cur.length >= 4 && ws.length - i - 1 >= 3) || cur.length >= 12) { parts.push(cur.join(" ")); cur = []; } }
      if (cur.length) { if (parts.length && cur.length < 3) parts[parts.length - 1] += " " + cur.join(" "); else parts.push(cur.join(" ")); }
    }
    const isV2 = /^sakhiya-(v2|v3|v4|v6|v15|nano|atom|pico)/.test(String((this.cfg && this.cfg.arch) || ""));
    let lastChandas = null;
    for (let i = 0; i < parts.length; i++) {
      if (useSa) { const c = await this.chunkSanskrit(parts[i], o); phs.push(c.phs); outs.push(c); continue; }
      let acc = null;
      if (o.lang === "sa" && window.CHANDAS && window.CHANDAS.compileChandas) {
        acc = window.CHANDAS.compileChandas(parts[i], o.bpm || 75.0, o.tonic_hz || 215.0, true);
        lastChandas = acc;
      } else if (o.lang === "sa" && window.SA) {
        acc = window.SA.layaScales ? window.SA.layaScales(parts[i], true, isV2) : window.SA.textToPhonesAcc(parts[i], true, isV2);
      } else if (SK.textToPhonesAcc) {
        acc = SK.textToPhonesAcc(parts[i], o.lang);
      }
      const ph = acc ? acc.ph : SK.textToPhones(parts[i], o.lang), dm = acc ? acc.dm : (SK.phoneDm ? SK.phoneDm(ph) : null); phs.push(ph.join(" ")); outs.push(await this.chunk(ph, Object.assign({}, o, { is_last: i === parts.length - 1, off: acc ? acc.off : null, dm, has_vedic_marks: acc ? !!acc.has_vedic_marks : false })));
    }
    const total = outs.reduce((n, c) => n + c.wav.length, 0), wav = new Float32Array(total); let p = 0;
    outs.forEach(c => { wav.set(c.wav, p); p += c.wav.length; });
    const ms = performance.now() - t0; return { wav, sr, ms, rtf: ms / 1000 / (total / sr), norm, phones: phs.join(" | "), mel: outs.map(c => ({ d: c.mel, F: c.F })) };
  }
}
function wavBlob(f32, sr) {
  const n = f32.length, buf = new ArrayBuffer(44 + n * 2), v = new DataView(buf), w = (o, s) => [...s].forEach((c, i) => v.setUint8(o + i, c.charCodeAt(0)));
  w(0, "RIFF"); v.setUint32(4, 36 + n * 2, true); w(8, "WAVEfmt "); v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true); v.setUint32(24, sr, true); v.setUint32(28, sr * 2, true); v.setUint16(32, 2, true); v.setUint16(34, 16, true); w(36, "data"); v.setUint32(40, n * 2, true);
  for (let i = 0; i < n; i++) v.setInt16(44 + i * 2, Math.max(-1, Math.min(1, f32[i])) * 32767, true); return new Blob([buf], { type: "audio/wav" });
}
window.SakhiyaWeb = SakhiyaWeb; window.wavBlob = wavBlob; window.STYLES = STYLES;
