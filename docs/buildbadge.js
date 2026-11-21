/* Transparency badge: shows which Sakhiya v4 build the page is serving (version, 200-sentence pCER, Cross-Style WavLM-SV, UTMOS22, Rigveda Svara, Glitch Frames, commit, UTC time). */
(function () {
  const get = async (u) => { try { const r = await fetch(u + "?t=" + Date.now(), { cache: "no-cache" }); return r.ok ? await r.json() : null; } catch (e) { return null; } };
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  async function render() {
    const [m1, m2, b, met] = await Promise.all([get("models/manifest.json"), get("models_v2/manifest.json"), get("build.json"), get("metrics.json")]);
    const m = m1 || m2 || b || {}, g = (met && (met.v4 || met.sakhiya_v4 || met.v3 || met.sakhiya_v3)) || {};
    const f = (x, d = 4) => (typeof x === "number" ? x.toFixed(d) : null);
    const nePcer = f(g.ne_pcer ?? m.ne_pcer ?? (b && b.ne_pcer));
    const saPcer = f(g.sa_pcer ?? m.sa_pcer ?? (b && b.sa_pcer));
    const utF = f(g.utmos_sakhiya ?? m.utmos_sakhiya, 3);
    const svF = f(g.cross_style_sv ?? g.wavlm_sv_sakhiya ?? m.wavlm_sv_sakhiya, 4);
    const svara = f(g.rigveda_svara_st ?? m.rigveda_svara_st, 2);
    const glitch = g.glitch_frames != null ? String(g.glitch_frames) : "0";
    const cells = [
      ["Version", "v4.0 (Single-Voice Disentangled + WavLM-SLM)"],
      ["Voice", "सखिया · Sakhiya (Unified F)"],
      ["Nepali pCER (100-sent)", nePcer],
      ["Sanskrit pCER (100-sent)", saPcer],
      ["Cross-Style Speaker Lock", svF ? svF + " (≥0.96)" : null],
      ["UTMOS22", utF],
      ["Rigveda Svara", svara ? svara + " st" : null],
      ["Glitches", glitch + " frames"],
      ["Commit", (b && b.commit ? b.commit.slice(0, 7) : (m.commit ? String(m.commit).slice(0, 7) : null))],
      ["UTC", g.timestamp_utc || (b && b.generated_utc) || m.generated_utc || null],
    ].filter((c) => c[1]);
    const html = '<div class="buildbadge" role="note" style="display:flex;flex-wrap:wrap;gap:6px 14px;align-items:center;padding:10px 14px;margin:0 0 14px;border-radius:16px;' +
      'background:var(--md-surface-container,#eef);color:var(--md-on-surface,#111);font-size:12.5px;line-height:1.35">' +
      '<span class="ms" style="font-family:\'Material Symbols Outlined\';font-size:18px;color:var(--md-primary)">verified</span><b>Sakhiya v4 Build</b>' +
      cells.map((c) => '<span><span style="opacity:.68">' + esc(c[0]) + ":</span> <b>" + esc(c[1]) + "</b></span>").join("") + "</div>";
    for (const id of ["samples", "studio"]) {
      const sec = document.getElementById(id); if (!sec) continue;
      let el = sec.querySelector(".buildbadge"); if (el) el.remove();
      const head = sec.querySelector(".page-head"); const tmp = document.createElement("div"); tmp.innerHTML = html;
      if (head) head.insertAdjacentElement("afterend", tmp.firstChild); else sec.prepend(tmp.firstChild);
    }
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", render); else render();
})();
