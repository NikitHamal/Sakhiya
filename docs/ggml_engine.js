// GGML C++ / WASM runtime for the Studio: runs docs/wasm/sakhiya.wasm (single-threaded SIMD128, no COOP/COEP needed) on a
// Sakhiya .gguf (Atom-Lite Q8_0 ~3.0 MB by default). GGUFs come from Hugging Face (CORS-enabled) and are cached in IndexedDB.
// Falls back to ONNX Runtime Web automatically when WASM SIMD / the download / the engine is unavailable.
import { createSakhiyaEngine, wasmSimdSupported } from './wasm/sakhiya_engine.js';

const HF = ['https://huggingface.co/NikitHamal/sakhiya-gguf/resolve/main/', 'https://huggingface.co/NikitHamal/sakhiya-v1-train/resolve/main/box6_v15/ggml/'];
const DB = 'sakhiya-gguf', STORE = 'files';
const $ = (s) => document.querySelector(s);
let engine = null, loaded = null;

const idb = () => new Promise((res, rej) => { const r = indexedDB.open(DB, 1); r.onupgradeneeded = () => r.result.createObjectStore(STORE); r.onsuccess = () => res(r.result); r.onerror = () => rej(r.error); });
const idbGet = async (k) => { try { const d = await idb(); return await new Promise((res) => { const q = d.transaction(STORE).objectStore(STORE).get(k); q.onsuccess = () => res(q.result); q.onerror = () => res(null); }); } catch (_) { return null; } };
const idbPut = async (k, v) => { try { const d = await idb(); await new Promise((res) => { const t = d.transaction(STORE, 'readwrite'); t.objectStore(STORE).put(v, k); t.oncomplete = res; t.onerror = res; }); } catch (_) { /* private mode: cache is optional */ } };

async function fetchGguf(file, expectBytes, note) {
  const hit = await idbGet(file);
  // a stale cached revision (e.g. the 3.48 MB compact Atom-Lite) is dropped when its size no longer matches tiers.json (gguf_mb is rounded to 1 KB)
  if (hit && hit.byteLength > 1e5 && (!expectBytes || Math.abs(hit.byteLength - expectBytes) < 4000)) { note(`${file} from IndexedDB cache`); return { buf: hit, cached: true }; }
  let lastErr;
  for (const base of HF) {
    try {
      const r = await fetch(base + file); if (!r.ok) throw new Error('HTTP ' + r.status);
      const total = +r.headers.get('content-length') || expectBytes || 0, rd = r.body.getReader(), parts = []; let got = 0;
      for (;;) { const { done, value } = await rd.read(); if (done) break; parts.push(value); got += value.length; note(`downloading ${file} ${(got / 1e6).toFixed(1)}${total ? '/' + (total / 1e6).toFixed(1) : ''} MB…`); }
      const u8 = new Uint8Array(got); let o = 0; for (const p of parts) { u8.set(p, o); o += p.length; }
      if (u8.length < 1e5 || String.fromCharCode(...u8.subarray(0, 4)) !== 'GGUF') throw new Error('not a GGUF file');
      await idbPut(file, u8.buffer); return { buf: u8.buffer, cached: false };
    } catch (e) { lastErr = e; }
  }
  throw Object.assign(new Error('GGUF download failed: ' + (lastErr && lastErr.message)), { fallback: true });
}

const pick = (v, base, k) => (v[k] === undefined || Math.abs(v[k] - (base[k] ?? NaN)) < 1e-3 ? undefined : v[k]);

async function say(text, st, vals, tier, note) {
  if (!wasmSimdSupported()) throw Object.assign(new Error('this browser has no WASM SIMD128'), { fallback: true });
  const file = tier.gguf_file || `sakhiya_${tier.id}_q8_0.gguf`;
  const t0 = performance.now();
  try {
    if (!engine) { note('loading sakhiya.wasm…'); engine = await createSakhiyaEngine(); }
    if (loaded !== file) {
      const { buf, cached } = await fetchGguf(file, (tier.gguf_mb || 0) * 1e6, note);
      note('initialising GGML engine…'); await new Promise((r) => setTimeout(r, 10));
      const info = engine.loadModel(buf); loaded = file; engine.info = { ...info, cached, file, ms: performance.now() - t0 };
    }
  } catch (e) { if (!e.fallback) e.fallback = true; throw e; }
  note(`${tier.label} GGUF Q8_0 ready` + (engine.info.cached ? ' (cached)' : '') + ` · loaded in ${(engine.info.ms / 1000).toFixed(1)} s`);
  await new Promise((r) => setTimeout(r, 20));
  const base = (window.STYLES && window.STYLES[st.style]) || {};
  const o = { lang: st.lang, style: st.style === 'chant' ? 'vedic_chant' : st.style };
  const m = { rate: 'rate', pitchShift: 'pitch_shift', pitchScale: 'pitch_scale', pauseScale: 'pause_scale', energyScale: 'energy_scale', svara: 'svara' };
  for (const [js, k] of Object.entries(m)) { const v = pick(vals, base, k); if (v !== undefined) o[js] = v; }   // only user-touched sliders override the style preset
  const s0 = performance.now();
  const { pcm, sampleRate } = engine.synthesize(text, o);
  const ms = performance.now() - s0;
  return { wav: pcm, sr: sampleRate, ms, rtf: ms / 1000 / (pcm.length / sampleRate), mel: null,
           norm: text, phones: `GGML C++ / WASM · tier=${engine.info.tier} · voc_ch=${engine.info.vocChannels} · cfm_steps=${engine.info.cfmSteps} · ${file}` };
}

const chips = () => [...document.querySelectorAll('#engines .chip')];
function syncUI() {
  const st = window.st || {};
  chips().forEach((c) => c.setAttribute('aria-checked', String(c.dataset.engine === (st.engine || 'onnx'))));
  const n = $('#engine-note'); if (!n) return;
  n.textContent = st.engine === 'ggml'
    ? 'sakhiya.wasm (C++17 → WebAssembly SIMD128, single thread) runs the same GGUF weights as the native sakhiya-cli.'
    : 'ONNX Runtime Web (INT8 graphs) is the default engine.';
}

function mountEngineCard() {
  const aside = document.querySelector('#studio aside'); if (!aside || $('#engines')) return;
  const simd = wasmSimdSupported();
  const card = document.createElement('div'); card.className = 'card';
  card.innerHTML = `<div class="card-h"><span class="ms">developer_board</span><h2>Runtime Engine</h2><span class="tag">new</span></div>
    <div class="chips wrap" id="engines" role="radiogroup" aria-label="Runtime engine">
      <button class="chip" type="button" role="radio" data-engine="onnx" aria-checked="true">ONNX Runtime Web</button>
      <button class="chip" type="button" role="radio" data-engine="ggml" aria-checked="false" ${simd ? '' : 'disabled title="WASM SIMD128 unavailable"'}>GGML C++ / WASM</button>
    </div>
    <p class="hint" id="engine-note"></p>`;
  aside.insertBefore(card, aside.children[3] || null);
  chips().forEach((c) => (c.onclick = () => {
    window.st.engine = c.dataset.engine; syncUI();
    if (c.dataset.engine === 'ggml') {   // recommended default for the browser: Atom-Lite Q8_0 (~3.0 MB)
      const lite = [...document.querySelectorAll('#tiers .tier')].find((b) => /Atom-Lite/.test(b.textContent));
      if (lite && !lite.disabled) lite.click();
    }
  }));
  syncUI();
}

function mountDevCard() {
  const host = $('#dev-sakhiya-cpp'); if (!host || host.dataset.ready) return; host.dataset.ready = '1';
  const S = {
    cpp: `// C / C++ — cpp/sakhiya.h (zero dependencies)
#include "sakhiya.h"
sakhiya_ctx_t* tts = sakhiya_init_from_file("sakhiya_atom_lite_q8_0.gguf", 4);
sakhiya_params_t p = sakhiya_default_params();  p.style = "news";  p.rate = 1.0f;
float* pcm; int sr; int64_t n = sakhiya_synthesize(tts, u8"नेपालको राजधानी काठमाडौँ हो।", p, &pcm, &sr);
/* n float32 samples @ 24 kHz */  sakhiya_free_pcm(pcm);  sakhiya_free(tts);`,
    py: `# pip-style wheel layout: sakhiya/ggml_tts.py  (auto-downloads + caches the GGUF from Hugging Face)
from sakhiya.ggml_tts import SakhiyaGGML
tts = SakhiyaGGML.from_pretrained("atom-lite")       # or "atom" | "nano" | "v1"
wav, sr = tts.say("नेपालको राजधानी काठमाडौँ हो।", style="news", speed=1.0, pitch=0.0, energy=1.0)
wavs = tts.say_batch([("नमस्ते", "ne", "warm_soft"), ("ॐ भूर्भुवः स्वः।", "sa", "vedic_chant")])`,
    kt: `// Android (NDK + JNI) — android/SakhiyaGGML.kt + cpp/android_jni.cpp, built by cpp/CMakeLists.txt (target sakhiya_jni)
val tts = SakhiyaGGML.fromFile(File(filesDir, "sakhiya_atom_lite_q8_0.gguf"), threads = 2)
tts.synthesize("नमस्ते नेपाल", style = "news").use { pcm -> pcm.play() }   // zero-copy direct FloatBuffer`,
    js: `// Browser / Node — docs/wasm/sakhiya_engine.js (no COOP/COEP, works on github.io)
import { createSakhiyaEngine } from "./wasm/sakhiya_engine.js";
const eng = await createSakhiyaEngine();
eng.loadModel(await (await fetch("https://huggingface.co/NikitHamal/sakhiya-gguf/resolve/main/sakhiya_atom_lite_q8_0.gguf")).arrayBuffer());
const { pcm, sampleRate } = eng.synthesize("नमस्ते", { lang: "ne", style: "news" });   // Float32Array @ 24 kHz`,
    sh: `# Build the native CLI/library (no ggml needed) and run
cmake -S cpp -B build-lib -DCMAKE_BUILD_TYPE=Release && cmake --build build-lib -j
./build-lib/sakhiya-cli-lite -m sakhiya_atom_lite_q8_0.gguf -t "नमस्ते" -o out.wav -j 4
# Convert your own checkpoints:  python convert_sakhiya_gguf.py --tier atom_lite --am am.pt --voc voc.pt --dtype q8_0 --out out.gguf`,
  };
  const names = { cpp: 'C / C++', py: 'Python', kt: 'Android', js: 'Browser / Node', sh: 'Build & CLI' };
  host.innerHTML = `<div class="chips scroll" role="tablist" aria-label="Integration language"></div><div class="codebox"><pre><code></code></pre><button class="btn tonal copy" type="button"><span class="ms">content_copy</span><span class="cl">Copy</span></button></div>`;
  const tabs = host.querySelector('.chips'), code = host.querySelector('code'), btn = host.querySelector('.copy');
  const show = (k) => { code.textContent = S[k]; [...tabs.children].forEach((c) => c.setAttribute('aria-checked', String(c.dataset.k === k))); host.dataset.k = k; };
  Object.keys(S).forEach((k) => { const b = document.createElement('button'); b.className = 'chip'; b.type = 'button'; b.setAttribute('role', 'tab'); b.dataset.k = k; b.textContent = names[k]; b.onclick = () => show(k); tabs.appendChild(b); });
  btn.onclick = async () => { try { await navigator.clipboard.writeText(S[host.dataset.k]); } catch (_) { const r = document.createRange(); r.selectNodeContents(code); const s = getSelection(); s.removeAllRanges(); s.addRange(r); document.execCommand('copy'); s.removeAllRanges(); } btn.querySelector('.cl').textContent = 'Copied'; setTimeout(() => (btn.querySelector('.cl').textContent = 'Copy'), 1400); };
  show('cpp');
}

window.GGML = { say, syncUI, available: wasmSimdSupported };
const boot = () => { mountEngineCard(); mountDevCard(); };
if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot); else boot();
