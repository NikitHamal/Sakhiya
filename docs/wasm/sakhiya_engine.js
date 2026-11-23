// Browser/Node wrapper around sakhiya.wasm (single-threaded WASM SIMD128 build of the Sakhiya GGUF C++ engine).
// No SharedArrayBuffer / COOP / COEP needed -> runs on GitHub Pages as-is.
//   import { createSakhiyaEngine, wasmSimdSupported } from './wasm/sakhiya_engine.js';
//   const eng = await createSakhiyaEngine();           // fetches sakhiya.wasm next to this file
//   eng.loadModel(await (await fetch(ggufUrl)).arrayBuffer());
//   const { pcm, sampleRate } = eng.synthesize('नमस्ते', { lang: 'ne', style: 'news' });

// Tiny module using v128 ops: validates only if the engine supports WASM SIMD128.
const SIMD_PROBE = new Uint8Array([0, 97, 115, 109, 1, 0, 0, 0, 1, 5, 1, 96, 0, 1, 123, 3, 2, 1, 0, 10, 10, 1, 8, 0, 65, 0, 253, 15, 253, 98, 11]);

export function wasmSimdSupported() {
  try { return typeof WebAssembly === 'object' && WebAssembly.validate(SIMD_PROBE); } catch (_) { return false; }
}

const DEF = { rate: -1, pitchShift: -999, pitchScale: -1, pauseScale: -1, energyScale: -1, svara: 1 };

export async function createSakhiyaEngine(opts = {}) {
  if (!opts.skipSimdCheck && !wasmSimdSupported()) throw new Error('WASM SIMD128 is not supported by this browser');
  const base = opts.base || new URL('./', import.meta.url);
  const factory = opts.factory || (await import(new URL('sakhiya_wasm.js', base).href)).default;
  const cfg = {};
  if (opts.wasmBinary) cfg.wasmBinary = opts.wasmBinary;
  else cfg.locateFile = (p) => (p.endsWith('.wasm') ? (opts.wasmUrl || new URL('sakhiya.wasm', base).href) : p);
  const M = await factory(cfg);

  const cstr = (s) => {
    const n = M.lengthBytesUTF8(s) + 1;
    const p = M._malloc(n);
    M.stringToUTF8(s, p, n);
    return p;
  };
  const err = () => M.UTF8ToString(M._get_last_error());
  const pcmOut = (n) => {
    if (n < 0) throw new Error(`sakhiya synth failed (${n}): ${err()}`);
    const ptr = M._get_pcm_ptr() >> 2, len = M._get_pcm_len();
    return { pcm: M.HEAPF32.slice(ptr, ptr + len), sampleRate: M._get_sample_rate() };
  };

  return {
    module: M,
    /** gguf: ArrayBuffer | Uint8Array. The weights are copied/dequantized; the buffer can be dropped afterwards. */
    loadModel(gguf) {
      const u8 = gguf instanceof Uint8Array ? gguf : new Uint8Array(gguf);
      const p = M._malloc(u8.length);
      M.HEAPU8.set(u8, p);
      const rc = M._init_model_from_buffer(p, u8.length);
      M._free(p);
      if (rc !== 0) throw new Error('sakhiya: model load failed: ' + err());
      return { tier: M.UTF8ToString(M._get_model_tier()), vocChannels: M._get_model_voc_channels(), cfmSteps: M._get_model_cfm_steps() };
    },
    synthesize(text, o = {}) {
      const p = { ...DEF, ...o };
      const t = cstr(text), l = cstr(p.lang || 'ne'), s = cstr(p.style || 'news');
      try { return pcmOut(M._synthesize_text(t, l, s, p.rate, p.pitchShift, p.pitchScale, p.pauseScale, p.energyScale, p.svara)); }
      finally { M._free(t); M._free(l); M._free(s); }
    },
    /** ids: Sakhiya symbol ids (e.g. from docs/g2p.js); pitchAdd / dm optional Float32Array(ids.length). */
    synthesizeIds(ids, o = {}) {
      const p = { ...DEF, ...o };
      const n = ids.length;
      const pi = M._malloc(n * 8);
      const dv = new DataView(M.HEAPU8.buffer, pi, n * 8);
      for (let i = 0; i < n; i++) dv.setBigInt64(i * 8, BigInt(ids[i]), true);
      const put = (arr) => { if (!arr) return 0; const q = M._malloc(n * 4); M.HEAPF32.set(arr, q >> 2); return q; };
      const pa = put(o.pitchAdd), dm = put(o.dm);
      const l = cstr(p.lang || 'ne'), s = cstr(p.style || 'news');
      try { return pcmOut(M._synthesize_ids(pi, n, pa, dm, l, s, p.rate, p.pitchShift, p.pitchScale, p.pauseScale, p.energyScale, p.svara)); }
      finally { M._free(pi); if (pa) M._free(pa); if (dm) M._free(dm); M._free(l); M._free(s); }
    },
    free() { M._free_model(); },
  };
}
