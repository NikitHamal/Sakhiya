/* Sakhiya Studio: voices, styles, model tiers and live synthesis (all on-device). Engine lives in tts.js. */
const $ = s => document.querySelector(s), el = (t, c, h) => { const e = document.createElement(t); if (c) e.className = c; if (h !== undefined) e.innerHTML = h; return e; };
window.$ = $;
const tts = new SakhiyaWeb();
const st = { lang: "auto", style: "news", voice: "sakhiya", tier: "sakhiya_news", engine: "onnx" };
window.st = st;
let TIERS = { tiers: [], voices: {}, styles: {} }, CFG = null;
// v6 focused scope: Nepali + Sanskrit + Auto (Mixed) only; ONE unified female voice `sakhiya`; 3 mastered delivery modes.
const LANGS = [["auto", "Auto (Mixed)"], ["ne", "नेपाली"], ["sa", "संस्कृतम्"]];
const VOICES = { sakhiya: ["सखिया", "Sakhiya", "Unified Expressive Nepali & Musical Vedic Female Voice (WavLM-SV ≥ 0.96)", "f"] };
const STYLE_ORDER = [["news", "News"], ["storytelling", "Storytelling"], ["conversational", "Conversational"], ["warm_soft", "Warm & Soft"], ["expressive", "Expressive"], ["vedic_chant", "Vedic Chant"]];
const STYLE_NOTE = {
  news: "Crisp, clear native Nepali & Sanskrit broadcast reading.",
  storytelling: "Warm, living human narration & emotional dialogue (genuine joy on 'वाह!', rising questions '?', natural comma breaths).",
  conversational: "Natural everyday Nepali chat & friendly dialogue with lively turn-taking.",
  warm_soft: "Gentle, calm, soothing voice for mindfulness, poetry, and reflective prose.",
  expressive: "High-energy, joyful exclamation and vivid descriptive narration.",
  vedic_chant: "Musical Vedic & Shloka chanting with real human sur, metrical laya, and deep resonant prolonged ॐ."
};
const PRESETS = [["Auto mix", "हेलो साथीहरू। म सखिया हुँ, निकित हमालज्यूले बनाउनुभएको सानो टेक्स्ट टु स्पीच मोडेल। म संस्कृत पनि स्पष्ट बोल्न सक्छु। धर्मक्षेत्रे कुरुक्षेत्रे समवेता युयुत्सवः । मामकाः पाण्डवाश्चैव किमकुर्वत सञ्जय ॥ आज मौसम निकै रमाइलो र शान्त छ।", "auto"],
  ["नेपाली (काठमाडौँ)", "नेपालको राजधानी काठमाडौँ हिमालको काखमा रहेको एउटा ऐतिहासिक सहर हो।", "ne"],
  ["कथा (Storytelling)", "एक समयको कुरा हो, एउटा सानो गाउँमा एउटी बुढी आमा बस्थिन्। उनी बिहानै उठेर चरालाई दाना हाल्थिन्।", "ne"],
  ["कुराकानी (Chat)", "नमस्ते! आज मौसम धेरै राम्रो छ, चिया खान जाऔँ न? तिमी कताबाट आउँदै छौ?", "ne"],
  ["उत्साह (Expressive)", "वाह! के अद्भुत दृश्य हो यो! माछापुच्छ्रे हिमाल बिहानको घाममा सुनजस्तै चम्किरहेको छ!", "ne"],
  ["ॐ पूर्णमदः", "ॐ पूर्णमदः पूर्णमिदं पूर्णात्पूर्णमुदच्यते। पूर्णस्य पूर्णमादाय पूर्णमेवावशिष्यते॥", "sa"],
  ["ऋग्वेद १.१.१", "अ॒ग्निमी॑ळे पु॒रोहि॑तं य॒ज्ञस्य॑ दे॒वमृ॒त्विज॑म् । होता॑रं रत्न॒धात॑मम् ॥", "sa"],
  ["भगवद्गीता १.१", "धर्मक्षेत्रे कुरुक्षेत्रे समवेता युयुत्सवः । मामकाः पाण्डवाश्चैव किमकुर्वत सञ्जय ॥", "sa"],
  ["गायत्री मन्त्र", "ॐ भूर्भुवः स्वः। तत्सवितुर्वरेण्यं भर्गो देवस्य धीमहि। धियो यो नः प्रचोदयात्॥", "sa"]];
const SLIDERS = [["rate", "Speed", .6, 1.6, .01, 1, "×"], ["pause_scale", "Pace / Pauses", .3, 2.5, .01, 1, "×"], ["pitch_scale", "Expressiveness (Tone)", .5, 1.8, .01, 1.08, "×"], ["pitch_shift", "Pitch", -.6, .6, .01, 0, ""], ["svara", "Vedic Svara Depth", 0, 2, .05, 1, "×"], ["bpm", "Chandas Laya (BPM, Vedic)", 40, 120, 1, 75, ""], ["energy_scale", "Projection / Warmth", .6, 1.6, .01, 1, "×"]];
const put = (id, t) => { const n = $(id).lastElementChild; if (n) n.textContent = t; };
const fmtMb = m => m < 1 ? Math.round(m * 1000) + " KB" : m.toFixed(m < 10 ? 1 : 0) + " MB";
const fmtRtf = r => r ? ` · RTF ${r < .1 ? r.toFixed(3) : r.toFixed(2)}` : "";
const badges = t => t.gguf_mb ? `<span class="bdg-row"><span class="bdg" title="ONNX Runtime Web INT8 graphs (RTF measured by the ONNX forensic gate)">ONNX INT8 ${fmtMb(t.onnx_mb || t.mb)}${fmtRtf(t.rtf_onnx || t.rtf)}</span><span class="bdg g" title="GGML C++ / WASM single-file GGUF Q8_0. CPU RTF measured natively: 1 thread / 4 threads">GGUF Q8_0 ${fmtMb(t.gguf_mb)}${fmtRtf(t.rtf_ggml_1t)}${t.rtf_ggml_mt ? " (1T) / " + (t.rtf_ggml_mt < .1 ? t.rtf_ggml_mt.toFixed(3) : t.rtf_ggml_mt.toFixed(2)) + " (4T)" : ""}</span></span>` : "";
const tag = k => k === "preview" ? ' <span class="tag">preview</span>' : "";

function radio(box, items, key, cb) {   // items: [value, innerHTML]
  box.innerHTML = "";
  items.forEach(([v, h]) => { const b = el("button", "chip", h); b.type = "button"; b.setAttribute("role", "radio"); b.setAttribute("aria-checked", String(st[key] === v)); b.onclick = () => { st[key] = v; [...box.children].forEach(c => c.setAttribute("aria-checked", String(c === b))); cb && cb(v); }; box.appendChild(b); });
}
function setSliders(o) { SLIDERS.forEach(([k, , , , , d, u]) => { const r = $("#r_" + k); r.value = o[k] ?? d; $("#v_" + k).textContent = (+r.value).toFixed(2) + u; }); }
function renderStyles() {
  radio($("#styles"), STYLE_ORDER.filter(([k]) => TIERS.styles && TIERS.styles[k]).map(([k, l]) => [k, l + tag(TIERS.styles[k])]), "style", s => { setSliders(STYLES[s] || {}); $("#styleinfo").textContent = STYLE_NOTE[s] || ""; });
  $("#styleinfo").textContent = STYLE_NOTE[st.style];
}
function renderVoices() {
  const box = $("#voices"), have = CFG && CFG.speakers ? new Set(CFG.speakers.map(s => s.name)) : new Set(Object.keys(VOICES));
  box.innerHTML = "";
  Object.entries(VOICES).filter(([k]) => have.has(k) && TIERS.voices && TIERS.voices[k]).forEach(([k, [dv, en, role, g]]) => {
    const b = el("button", "voice", `<span class="av">${dv[0]}</span><b>${dv} · ${en}${tag(TIERS.voices[k])}</b><small>${role}</small>`);
    b.type = "button"; b.dataset.g = g; b.setAttribute("role", "radio"); b.setAttribute("aria-checked", String(st.voice === k));
    b.onclick = () => { st.voice = k; [...box.children].forEach(c => c.setAttribute("aria-checked", String(c === b))); }; box.appendChild(b);
  });
}
function renderTiers() {
  const box = $("#tiers"); box.innerHTML = "";
  let expHead = false;
  TIERS.tiers.forEach(t => {
    if (t.experimental && !expHead) { expHead = true; box.appendChild(el("div", "tier-exp-head", "<b>Experimental</b> · trained from scratch on mined real data · not production quality")); }
    const b = el("button", "tier", `<span class="radio"></span><span class="tx"><b>${t.label} ${t.prec}${t.ready ? "" : ' <span class="tag">not ready</span>'}</b><small>${t.desc}</small>${badges(t)}</span><span class="mb">${t.mb < 1 ? Math.round(t.mb * 1000) + " KB" : t.mb.toFixed(1) + " MB"}</span>`);
    b.type = "button"; b.disabled = !t.ready; b.setAttribute("role", "radio"); b.setAttribute("aria-checked", String(st.tier === t.id));
    b.onclick = () => { st.tier = t.id; [...box.children].forEach(c => c.setAttribute("aria-checked", String(c === b))); put("#mdl", "loads on next Speak"); };
    box.appendChild(b);
  });
  const ts = $("#tierstate"); if (ts) ts.innerHTML = TIERS.tiers.map(t => `<p><b>${t.label} ${t.prec}</b> · ${t.mb} MB · ${t.ready ? "ready" : "gated (" + t.gate + ")"}</p>`).join("");
}
function init() {
  $("#text").value = PRESETS[0][1]; $("#count").textContent = $("#text").value.length; $("#text").oninput = e => $("#count").textContent = e.target.value.length;
  radio($("#langs"), LANGS, "lang");
  const P = $("#presets"); PRESETS.forEach(([n, t, l]) => { const b = el("button", "chip", n); b.type = "button"; b.onclick = () => { $("#text").value = t; $("#count").textContent = t.length; st.lang = l; radio($("#langs"), LANGS, "lang"); }; P.appendChild(b); });
  SLIDERS.forEach(([k, n, lo, hi, step, def, u]) => { const w = el("div", "", `<label for="r_${k}">${n}<b id="v_${k}">${def}${u}</b></label><input type="range" id="r_${k}" min="${lo}" max="${hi}" step="${step}" value="${def}">`); $("#sliders").appendChild(w); w.querySelector("input").oninput = e => $("#v_" + k).textContent = (+e.target.value).toFixed(2) + u; });
  $("#reset").onclick = () => setSliders(STYLES[st.style] || {});
  setSliders(STYLES[st.style] || {}); renderStyles(); renderVoices(); renderTiers();
  fetch("tiers.json", { cache: "no-cache" }).then(r => r.json()).then(j => { TIERS = j; renderStyles(); renderVoices(); renderTiers(); }).catch(() => {});
  fetch("models_v2/config.json").then(r => r.json()).then(c => { CFG = c; renderVoices(); }).catch(() => {});
  const th = $("#theme"); th.onclick = () => { const n = document.documentElement.dataset.theme === "dark" ? "light" : "dark"; document.documentElement.dataset.theme = n; try { localStorage.setItem("theme", n); } catch (e) {} };
}
const tier = () => TIERS.tiers.find(t => t.id === st.tier) || { dir: "models_v2/", variant: "fp16" };
async function boot() {
  const t = tier();
  if (tts.variant !== t.variant || tts.base !== t.dir || !tts.cfg) { put("#mdl", `downloading ${t.mb || 31} MB…`); $("#golbl").textContent = "Loading model…"; await new Promise(r => setTimeout(r, 20)); }
  await (SK.F || SK.load()); await tts.init(t.variant, t.dir);
  put("#mdl", `${t.label} ${t.prec || ""} · ${t.mb ? (t.mb < 1 ? Math.round(t.mb * 1000) + " KB" : t.mb.toFixed(1) + " MB") : ""} ready` + (tts.loadMs ? ` · loaded in ${(tts.loadMs / 1000).toFixed(1)} s` : ""));
}
function tone(prop, fb) { const v = getComputedStyle(document.documentElement).getPropertyValue(prop).trim() || fb, m = /^#?([0-9a-f]{6})$/i.exec(v); return m ? [0, 2, 4].map(i => parseInt(m[1].slice(i, i + 2), 16)) : [0, 88, 223]; }
function drawWave(w) {
  const c = $("#wavc"), g = c.getContext("2d"), p = tone("--md-primary", "#0058df"); c.width = (c.clientWidth || 300) * 2; c.height = 128; g.clearRect(0, 0, c.width, c.height);
  g.fillStyle = `rgb(${p})`; const n = c.width, s = Math.max(1, Math.floor(w.length / n));
  for (let x = 0; x < n; x++) { let m = 0; for (let i = 0; i < s; i += 8) m = Math.max(m, Math.abs(w[x * s + i] || 0)); g.fillRect(x, 64 - m * 60, 1, Math.max(2, m * 120)); }
}
function drawMel(parts) {
  const c = $("#mel"), g = c.getContext("2d"), F = parts.reduce((n, p) => n + p.F, 0), M = parts[0] && parts[0].F ? Math.round(parts[0].d.length / parts[0].F) || 80 : 80, p = tone("--md-primary", "#0058df"), b = tone("--md-surface-container-low", "#f0f4fe");
  c.width = Math.max(F, 10); c.height = M; const im = g.createImageData(c.width, M); let x0 = 0;
  for (const q of parts) { for (let f = 0; f < q.F; f++) for (let m = 0; m < M; m++) { const v = Math.pow(Math.max(0, Math.min(1, (q.d[m * q.F + f] + 9) / 8.5)), 1.4), i = (((M - 1 - m) * c.width) + x0 + f) * 4; for (let k = 0; k < 3; k++) im.data[i + k] = b[k] + (p[k] - b[k]) * v; im.data[i + 3] = 255; } x0 += q.F; }
  g.putImageData(im, 0, 0);
}
async function ggmlSpeak() {   // GGML C++ / WASM engine; any failure falls back to ONNX Runtime Web
  const vals = {}; SLIDERS.forEach(([k]) => vals[k] = +$("#r_" + k).value);
  try { return await window.GGML.say($("#text").value, st, vals, tier(), m => put("#mdl", m)); }
  catch (e) { console.warn("GGML/WASM unavailable, falling back to ONNX:", e); st.engine = "onnx"; window.GGML.syncUI(); $("#msg").textContent = "GGML/WASM unavailable (" + e.message + ") — using ONNX Runtime Web."; return null; }
}
async function speak() {
  const btn = $("#go"); btn.disabled = true; btn.classList.add("busy"); $("#golbl").textContent = "Speaking…"; await new Promise(r => setTimeout(r, 30));
  try {
    let r = null; if (st.engine === "ggml" && window.GGML) r = await ggmlSpeak();
    if (!r) { await boot();
    const spkId = (tts.cfg.spk2id || {})[st.voice] ?? 0, o = { lang: st.lang, spk: spkId, style: st.style };
    SLIDERS.forEach(([k]) => o[k] = +$("#r_" + k).value);
    if (window.StudioV6 && window.StudioV6.cartridge && window.StudioV6.cartridge() && window.StudioV6.cartridge().contour) {
      o.contour = window.StudioV6.cartridge().contour;
    }
    r = await tts.say($("#text").value, o); if (window.StudioV6) r.wav = window.StudioV6.postMix(r.wav, r.sr, st); }
    const url = URL.createObjectURL(wavBlob(r.wav, r.sr));
    $("#player").src = url; $("#player").play(); $("#dl").href = url; $("#dl").hidden = false;
    drawWave(r.wav); if (r.mel) drawMel(r.mel); else { const c = $("#mel"), g = c.getContext("2d"); g.clearRect(0, 0, c.width, c.height); }
    put("#rtf", `RTF ${r.rtf.toFixed(2)} · ${(r.wav.length / r.sr).toFixed(1)} s in ${(r.ms / 1000).toFixed(1)} s`);
    $("#norm").textContent = "Normalised: " + r.norm; $("#ph").textContent = r.phones; $("#msg").textContent = "";
  } catch (e) { $("#msg").textContent = "Error: " + e.message; console.error(e); }
  btn.disabled = false; btn.classList.remove("busy"); $("#golbl").textContent = "Speak";
}
init(); $("#go").onclick = speak;
