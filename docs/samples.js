/* Samples gallery + Version-by-Version Milestone Comparator (`samples/versions/registry.json`).
   Renders interactive Version A vs Version B selector on both #studio and #samples so the owner can A/B compare every frozen milestone! */
(function () {
  const $ = s => document.querySelector(s), el = (t, c, h) => { const e = document.createElement(t); if (c) e.className = c; if (h !== undefined) e.innerHTML = h; return e; };
  const NAMES = { fp16: "Sakhiya-V1 (fp16)", int8: "Sakhiya-V1 (int8)" }, DIR = "samples/v25/";
  let M = null, T = {}, group = "All", loaded = false, REG = null, verA = "v5_final", verB = "v8_nano";
  const esc = s => String(s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

  function renderComparatorInto(mountId) {
    const mount = document.getElementById(mountId);
    if (!mount || !REG || !REG.versions || !REG.versions.length) return;
    const vers = REG.versions;
    if (!vers.some(v => v.tag === verA)) verA = vers[0].tag;
    if (!vers.some(v => v.tag === verB)) verB = vers[vers.length - 1].tag;
    const vA = vers.find(v => v.tag === verA) || vers[0];
    const vB = vers.find(v => v.tag === verB) || vers[vers.length - 1];
    const mCard = (v, role) => {
      const m = v.metrics || {};
      return `<div class="card" style="padding:14px 16px;background:var(--md-surface-container-low)">
        <div style="display:flex;justify-content:space-between;align-items:center;gap:8px;flex-wrap:wrap;margin-bottom:8px">
          <b>${esc(role)}: ${esc(v.label)}</b>
          <span class="count">commit ${esc(v.commit_sha || "-")} · ${esc(v.timestamp_utc || "")}</span>
        </div>
        <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(125px,1fr));gap:8px;font-size:12.5px">
          <div><span class="muted">Nepali pCER:</span> <b>${m.ne_pcer != null ? Number(m.ne_pcer).toFixed(4) : "-"}</b></div>
          <div><span class="muted">Sanskrit pCER:</span> <b>${m.sa_pcer != null ? Number(m.sa_pcer).toFixed(4) : "-"}</b></div>
          <div><span class="muted">Speaker Lock:</span> <b>${m.cross_style_sv != null ? Number(m.cross_style_sv).toFixed(4) : (m.sv_cos_sim || "-")}</b></div>
          <div><span class="muted">UTMOS22:</span> <b>${m.utmos != null ? Number(m.utmos).toFixed(3) : "-"}</b></div>
          <div><span class="muted">Rigveda Svara:</span> <b>${m.rigveda_svara_st != null ? Number(m.rigveda_svara_st).toFixed(2) + " st" : "-"}</b></div>
          <div><span class="muted">Glitch Frames:</span> <b>${m.glitch_frames != null ? m.glitch_frames : "-"}</b></div>
        </div>
      </div>`;
    };
    const selHtml = (cur, id) => `<select id="${id}" style="padding:8px 12px;border-radius:12px;border:1.5px solid var(--md-outline-variant);background:var(--md-surface-container);color:var(--md-on-surface);font:600 14px Poppins,sans-serif">` +
      vers.map(v => `<option value="${esc(v.tag)}"${v.tag === cur ? " selected" : ""}>${esc(v.label)} (${esc(v.commit_sha || "")})</option>`).join("") + `</select>`;

    const clipsA = Object.fromEntries((vA.clips || []).map(c => [c.id, c]));
    const clipsB = Object.fromEntries((vB.clips || []).map(c => [c.id, c]));
    const masterList = vB.clips && vB.clips.length ? vB.clips : (vA.clips || []);
    const rows = masterList.map((c, idx) => {
      const ca = clipsA[c.id] || c, cb = clipsB[c.id] || c;
      return `<div class="card sm" style="padding:14px 16px">
        <div style="display:flex;justify-content:space-between;align-items:center;gap:8px;margin-bottom:6px">
          <b>#${idx + 1} · ${esc(c.id)}</b>
          <span class="count">${esc(c.lang.toUpperCase())} · ${esc(c.style)}</span>
        </div>
        <p class="tx" style="margin:4px 0 10px;font-size:14px">${esc(c.text)}</p>
        <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:10px">
          <div><small class="muted" style="display:block;margin-bottom:4px"><b>A:</b> ${esc(vA.label)}</small><audio controls preload="none" src="${esc(ca.mp3)}" style="height:38px;margin:0"></audio></div>
          <div><small class="muted" style="display:block;margin-bottom:4px"><b>B:</b> ${esc(vB.label)}</small><audio controls preload="none" src="${esc(cb.mp3)}" style="height:38px;margin:0"></audio></div>
        </div>
      </div>`;
    }).join("");

    mount.innerHTML = `<div style="display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin-bottom:14px">
        <label style="font-size:13px;font-weight:600">Version A (Baseline): ${selHtml(verA, mountId + "_a")}</label>
        <span class="ms" style="color:var(--md-primary)">compare_arrows</span>
        <label style="font-size:13px;font-weight:600">Version B (Target): ${selHtml(verB, mountId + "_b")}</label>
      </div>
      <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:12px;margin-bottom:16px">${mCard(vA, "Version A")}${mCard(vB, "Version B")}</div>
      <div class="gallery" style="margin-top:10px">${rows}</div>`;

    document.getElementById(mountId + "_a").onchange = (e) => { verA = e.target.value; renderAllComparators(); };
    document.getElementById(mountId + "_b").onchange = (e) => { verB = e.target.value; renderAllComparators(); };
  }

  function renderAllComparators() {
    renderComparatorInto("ver-cmp-mount");
    renderComparatorInto("ver-cmp-mount-samples");
  }

  async function initRegistry() {
    try {
      REG = await (await fetch("samples/versions/registry.json?t=" + Date.now(), { cache: "no-cache" })).json();
      if (REG && REG.versions && REG.versions.length) {
        verA = REG.versions[0].tag;
        verB = REG.versions[REG.versions.length - 1].tag;
        renderAllComparators();
      }
    } catch (e) {}
  }

  function card(it) {
    const c = el("div", "card sm"), tiers = (M.tiers || []).filter(t => t === "fp16" || t === "int8");
    c.innerHTML = `<h3>${esc(it.label)}</h3><p class="tx" lang="${it.lang === "sa" ? "sa" : "ne"}">${esc(it.text || "")}</p>`;
    const seg = el("div", "seg"), slot = el("div"); c.append(seg, slot);
    tiers.filter(t => it[t]).forEach(t => {
      const b = el("button", "", NAMES[t] || t); b.type = "button";
      b.onclick = () => { [...seg.children].forEach(x => x.classList.toggle("on", x === b)); slot.innerHTML = ""; const a = el("audio"); a.controls = true; a.preload = "none"; a.src = DIR + it[t].file; slot.appendChild(a); a.play().catch(() => {}); };
      seg.appendChild(b);
    });
    if (!seg.children.length) seg.remove();
    return c;
  }
  function render() {
    const g = $("#gallery"); g.innerHTML = ""; let last = "";
    M.items.filter(i => group === "All" || i.group === group).forEach(it => { if (it.group !== last && group === "All") { g.appendChild(el("div", "gh", esc(it.group))); last = it.group; } g.appendChild(card(it)); });
  }
  function chips() {
    const box = $("#sgroups"), groups = ["All", ...new Set(M.items.map(i => i.group))]; box.innerHTML = "";
    groups.forEach(n => { const b = el("button", "chip" + (n === group ? " on" : ""), esc(n)); b.type = "button"; b.onclick = () => { group = n; [...box.children].forEach(c => c.classList.toggle("on", c === b)); render(); }; box.appendChild(b); });
  }
  async function sizes() {
    let P = { tiers: [] };
    try { P = await (await fetch("pareto.json", { cache: "no-cache" })).json(); } catch (e) {}
    const rows = (P.tiers || []).slice().sort((x, y) => (x.int8_mb || 0) - (y.int8_mb || 0));
    const mx = Math.max(32, ...rows.map(r => r.fp16_mb || 0));
    const f = (v, d) => (v == null ? "n/m" : Number(v).toFixed(d));
    $("#sizes").innerHTML = `<div class="card-h"><span class="ms">leaderboard</span><h2>Size vs Quality vs CPU Speed (measured, not claimed)</h2></div>
      <p class="muted" style="margin-bottom:12px">Every number comes from <code>pareto.json</code> (forensic gate UTMOS22 over the 15 master clips + ONNX-Runtime Node timings, 1 thread). Tiers below their quality target are flagged — the tiny tiers are still in training.</p>
      <div class="bars">` + rows.map(p => `<div class="bar" style="grid-template-columns:260px 1fr 330px"><span><b>${esc(p.label)}</b><br><small class="muted">${esc(p.tag)} · ${f(p.params_m, 2)} M params</small></span><span><i style="display:block;width:${Math.max(4, (p.int8_mb || 0) / mx * 100)}%"></i></span><span><b>${f(p.int8_mb, 2)} MB int8</b> / ${f(p.fp16_mb, 2)} MB fp16 · UTMOS ${f(p.utmos_15, 2)} (target ${f(p.utmos_target, 2)}) · RTF ${f(p.rtf_int8, 3)}${p.rtf_int8 ? " (" + Math.round(1 / p.rtf_int8) + "× RT)" : ""} · drops ${p.word_drops == null ? "n/m" : p.word_drops} · <b style="color:${p.meets_quality ? "#2e7d32" : "#c62828"}">${p.meets_quality ? "meets target" : "below target"}</b></span></div>`).join("") + `</div>`;
  }
  const HIDDEN = /^voice_(aanchal|acharya|kiran|sandesh)$|^sandesh_|^sa_sandesh_|^thr_/;
  async function load() {
    if (loaded) return; loaded = true;
    try {
      const tj = await (await fetch("tiers.json")).json(); (tj.tiers || []).forEach(t => T[t.id] = t);
      try { M = await (await fetch(DIR + "manifest.json", { cache: "no-cache" })).json(); } catch (_) { M = { items: [] }; } M.items = (M.items || []).filter(i => !HIDDEN.test(i.id));
      $("#sgen").lastElementChild.textContent = "Generated " + (M.generated_utc || "");
      chips(); render(); sizes();
    } catch (e) { $("#gallery").innerHTML = '<div class="card">Samples are not available yet.</div>'; loaded = false; }
  }
  initRegistry();
  document.addEventListener("sknav", e => { if (e.detail === "samples") load(); });
  if (location.hash === "#samples") load();
})();
