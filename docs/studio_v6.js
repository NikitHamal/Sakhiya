/* Sakhiya v6 Studio add-ons: Expression Cartridges, Chandas Laya BPM (slider in app.js), Tanpura drone mix, mic cadence capture.
   Everything runs in the browser; nothing is uploaded. Cartridge params drive the shipped tiers today; the 8-point
   f0/energy/duration contour is kept for the v6 contour hook (used once the v6 teacher passes the forensic gate). */
(function () {
  const $ = s => document.querySelector(s);
  let CART = [], custom = null, active = null, tanpura = null, tanOn = false;
  const tonic = 215.0;

  function cartList() { return CART.concat(custom ? [custom] : []); }
  function applyCartridge(c) {
    active = c.id;
    if (window.st && c.style && window.st.style !== c.style) { window.st.style = c.style; window.renderStyles && window.renderStyles(); }
    const base = Object.assign({}, (window.STYLES && window.STYLES[window.st.style]) || {}, c.params || {});
    window.setSliders && window.setSliders(base);
    const info = $("#styleinfo"); if (info) info.textContent = c.label + " — " + c.desc;
    window.st.cartridge = c; render();
  }
  function render() {
    const box = $("#cartridges"); if (!box) return;
    box.innerHTML = "";
    const sty = window.st ? window.st.style : "storytelling";
    cartList().filter(c => c.style === sty || c.id === "mic_custom").forEach(c => {
      const b = document.createElement("button"); b.type = "button"; b.className = "chip"; b.textContent = c.label;
      b.setAttribute("role", "radio"); b.setAttribute("aria-checked", String(active === c.id)); b.title = c.desc;
      b.onclick = () => applyCartridge(c); box.appendChild(b);
    });
  }
  /* 3-second microphone cadence capture: autocorrelation F0 + energy envelope -> 8-point contour + rate. */
  async function captureMic(btn) {
    const lbl = btn.querySelector("i"); const old = lbl.textContent;
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const ctx = new (window.AudioContext || window.webkitAudioContext)(), src = ctx.createMediaStreamSource(stream), an = ctx.createAnalyser();
      an.fftSize = 2048; src.connect(an); const buf = new Float32Array(an.fftSize), f0s = [], ens = [];
      lbl.textContent = "listening 3 s…"; const t0 = performance.now();
      await new Promise(res => { const tick = () => {
        an.getFloatTimeDomainData(buf); let e = 0; for (const v of buf) e += v * v; e = Math.sqrt(e / buf.length); ens.push(e);
        let best = 0, bl = 0; if (e > 0.01) { for (let l = Math.floor(ctx.sampleRate / 400); l < Math.floor(ctx.sampleRate / 80); l++) { let s = 0; for (let i = 0; i < 1024; i++) s += buf[i] * buf[i + l]; if (s > best) { best = s; bl = l; } } }
        f0s.push(bl ? ctx.sampleRate / bl : 0);
        if (performance.now() - t0 < 3000) setTimeout(tick, 25); else res(); }; tick(); });
      stream.getTracks().forEach(t => t.stop()); ctx.close();
      const v = f0s.filter(x => x > 0).sort((a, b) => a - b); if (v.length < 8) throw new Error("not enough voiced sound");
      const med = v[v.length >> 1], seg = n => Array.from({ length: 8 }, (_, i) => { const a = Math.floor(i * n.length / 8), b = Math.floor((i + 1) * n.length / 8), s = n.slice(a, b).filter(x => x > 0); return s.length ? s.reduce((p, q) => p + q, 0) / s.length : 0; });
      const f0 = seg(f0s).map(x => x ? +(12 * Math.log2(x / med)).toFixed(2) : 0), emax = Math.max(...ens) || 1;
      const en = seg(ens).map(x => +(x / emax).toFixed(2)); let on = 0; for (let i = 1; i < ens.length; i++) if (ens[i] > 0.4 * emax && ens[i - 1] <= 0.4 * emax) on++;
      const rate = Math.max(0.8, Math.min(1.3, 0.8 + on / 3 * 0.12));
      custom = { id: "mic_custom", label: "My hummed cadence", style: window.st.style, desc: "Rhythm and pitch shape captured from your microphone (stays on this device).", params: { rate: +rate.toFixed(2), pitch_scale: +(1 + Math.min(0.5, Math.max(...f0.map(Math.abs)) / 12)).toFixed(2) }, contour: { f0, en, dur: new Array(8).fill(0) } };
      applyCartridge(custom);
    } catch (e) { alert("Microphone cadence capture failed: " + e.message); }
    lbl.textContent = old;
  }
  window.StudioV6 = {
    /* called by app.js after synthesis: optional tanpura layer */
    postMix(wav, sr, st) {
      if (!tanOn || !window.TanpuraDrone) return wav;
      if (!tanpura) tanpura = new window.TanpuraDrone(tonic, -22.0);
      const db = +($("#r_tan_db") ? $("#r_tan_db").value : -22);
      return tanpura.mixIntoBuffer(wav, sr, tonic, db);
    },
    cartridge() { return window.st && window.st.cartridge; }
  };
  function build() {
    const sty = $("#styleinfo"); if (!sty) return;
    const card = sty.parentElement;
    const wrap = document.createElement("div");
    wrap.innerHTML = '<div class="label" style="margin-top:14px">Expression Cartridge <span class="muted">· tiny, on-device</span></div><div class="chips wrap" id="cartridges" role="radiogroup" aria-label="Expression cartridge"></div>' +
      '<div class="actions" style="margin-top:10px;gap:8px;flex-wrap:wrap"><button class="btn tonal" type="button" id="mic"><span class="ms">mic</span><i>Hum a cadence (3 s)</i></button>' +
      '<label class="chip" style="cursor:pointer"><input type="checkbox" id="tan" style="margin-right:6px">Tanpura drone</label></div>' +
      '<div id="tanbox" hidden style="margin-top:8px"><label for="r_tan_db" class="muted" style="font-size:13px">Drone level <b id="v_tan_db">-22 dB</b></label><input type="range" id="r_tan_db" min="-34" max="-12" step="1" value="-22" style="width:100%"></div>';
    card.appendChild(wrap);
    $("#mic").onclick = e => captureMic(e.currentTarget);
    $("#tan").onchange = e => { tanOn = e.target.checked; $("#tanbox").hidden = !tanOn; if (tanOn && !tanpura && window.TanpuraDrone) tanpura = new window.TanpuraDrone(tonic, -22.0); };
    $("#r_tan_db").oninput = e => { $("#v_tan_db").textContent = e.target.value + " dB"; };
    $("#styles").addEventListener("click", () => setTimeout(render, 0));
    fetch("cartridges.json", { cache: "no-cache" }).then(r => r.json()).then(j => { CART = j.cartridges || []; render(); }).catch(() => {});
    render();
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", () => setTimeout(build, 0)); else setTimeout(build, 0);
})();
