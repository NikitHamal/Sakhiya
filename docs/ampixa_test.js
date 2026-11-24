/* Ampixa Benchmarks Arena (`https://ampixa.benchmarks.cloud/#submit`) — 365-Prompt Local Self-Test & Audio Audition Suite.
   Strictly for local self-testing (never submits to external servers). Loads all 365 official prompts from `data/ampixa_prompts.json`,
   groups by the 9 official categories (minimal pairs, question intonation, emotion, contrastive stress, robustness, phrases,
   homographs, newspaper ambiguities, natural speech), and synthesizes any prompt or batch on-device with the active Sakhiya tier. */
(function () {
  const $ = s => document.querySelector(s);
  const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

  const CAT_LABELS = {
    submission_205: "Official #submit Pack (205)",
    all: "All Prompts (365)",
    phonological_minimal_pairs: "Minimal Pairs (149)",
    natural_speech: "Natural Speech (160)",
    question_intonation: "Questions (10)",
    robustness: "Tongue Twisters / Robustness (10)",
    phrases: "Common Phrases (10)",
    emotion: "Emotion (8)",
    contrastive_stress: "Contrastive Stress (8)",
    homographs: "Homographs (6)",
    newspaper_ambiguities: "Newspaper Headlines (4)",
  };

  let PROMPTS = [];
  let curCat = "submission_205";
  let curSearch = "";
  let curPage = 0;
  const PAGE_SIZE = 12;
  const audioCache = {}; // sent_id -> { url, rtf, dur, phones, tier }
  let ratings = {};
  try { ratings = JSON.parse(localStorage.getItem("sakhiya_ampixa_self_ratings") || "{}"); } catch (_) {}

  function saveRatings() {
    try { localStorage.setItem("sakhiya_ampixa_self_ratings", JSON.stringify(ratings)); } catch (_) {}
    updateSummaryBar();
  }

  function filteredPrompts() {
    const q = curSearch.trim().toLowerCase();
    return PROMPTS.filter(p => {
      if (curCat === "submission_205") {
        if (String(p.sent_id || "").startsWith("chirp_")) return false;
      } else if (curCat !== "all" && p.category !== curCat) {
        return false;
      }
      if (!q) return true;
      return (
        String(p.sent_id || "").toLowerCase().includes(q) ||
        String(p.text_devanagari || "").toLowerCase().includes(q) ||
        String(p.text_romanized || "").toLowerCase().includes(q) ||
        String(p.contrast_word || "").toLowerCase().includes(q) ||
        String(p.category || "").toLowerCase().includes(q)
      );
    });
  }

  async function synthPrompt(p, btnEl) {
    const sid = p.sent_id;
    const oldHtml = btnEl ? btnEl.innerHTML : "";
    if (btnEl) {
      btnEl.disabled = true;
      btnEl.innerHTML = `<span class="ms">hourglass_top</span>Synthesizing…`;
    }
    try {
      if (typeof boot === "function") await boot();
      const spkId = (typeof tts !== "undefined" && tts.cfg && tts.cfg.spk2id && window.st) ? (tts.cfg.spk2id[window.st.voice] ?? 0) : 0;
      const style = p.category === "emotion" ? "expressive" : (p.category === "natural_speech" ? "storytelling" : "news");
      const r = await tts.say(p.text_devanagari, { lang: "ne", spk: spkId, style });
      const url = URL.createObjectURL(window.wavBlob(r.wav, r.sr));
      const tierLbl = window.st ? window.st.tier : "v1";
      audioCache[sid] = {
        url,
        rtf: r.rtf,
        dur: r.wav.length / r.sr,
        phones: r.phones,
        norm: r.norm,
        tier: tierLbl,
      };
      renderSlot(sid);
      const aud = document.getElementById("ampixa_aud_" + sid);
      if (aud) aud.play().catch(() => {});
      updateSummaryBar();
    } catch (e) {
      alert("Synthesis error on " + sid + ": " + e.message);
    }
    if (btnEl) {
      btnEl.disabled = false;
      btnEl.innerHTML = oldHtml;
    }
  }

  function renderSlot(sid) {
    const slot = document.getElementById("ampixa_slot_" + sid);
    if (!slot) return;
    const c = audioCache[sid];
    if (!c) {
      slot.innerHTML = "";
      return;
    }
    slot.innerHTML = `
      <div style="margin-top:10px;padding:10px 12px;border-radius:10px;background:var(--md-surface-container)">
        <div style="display:flex;justify-content:space-between;align-items:center;gap:8px;flex-wrap:wrap;margin-bottom:6px;font-size:12px">
          <span><b>Tier:</b> <code>${esc(c.tier.toUpperCase())}</code> · <b>Duration:</b> ${c.dur.toFixed(2)}s · <b>RTF:</b> ${c.rtf.toFixed(3)} (${Math.round(1 / Math.max(c.rtf, 1e-3))}&times; RT)</span>
          <a class="btn text" href="${c.url}" download="${esc(sid)}.wav" style="padding:2px 8px;font-size:12px"><span class="ms" style="font-size:16px">download</span>${esc(sid)}.wav</a>
        </div>
        <audio id="ampixa_aud_${esc(sid)}" controls preload="none" src="${c.url}" style="width:100%;height:36px"></audio>
        <div style="font-size:11.5px;color:var(--md-on-surface-variant);margin-top:4px;word-break:break-all"><b>IPA Phones:</b> <code>${esc(c.phones)}</code></div>
      </div>`;
  }

  function updateSummaryBar() {
    const bar = document.getElementById("ampixa-summary-bar");
    if (!bar) return;
    const tested = Object.values(audioCache);
    const ratedVals = Object.values(ratings).filter(v => v >= 1 && v <= 5);
    const meanRtf = tested.length ? (tested.reduce((a, b) => a + b.rtf, 0) / tested.length).toFixed(3) : "—";
    const meanMos = ratedVals.length ? (ratedVals.reduce((a, b) => a + b, 0) / ratedVals.length).toFixed(2) : "—";
    bar.innerHTML = `
      <span><b>Total Official Prompts:</b> 365</span>
      <span><b>Synthesized in Session:</b> ${tested.length} / 365</span>
      <span><b>Mean On-Device RTF:</b> ${meanRtf}</span>
      <span><b>Self-Rated Prompts:</b> ${ratedVals.length} (Mean MOS: <b>${meanMos} / 5.00</b>)</span>`;
  }

  function renderCards() {
    const grid = document.getElementById("ampixa-grid");
    const pager = document.getElementById("ampixa-pager");
    if (!grid) return;
    const list = filteredPrompts();
    const totalPages = Math.max(1, Math.ceil(list.length / PAGE_SIZE));
    if (curPage >= totalPages) curPage = totalPages - 1;
    if (curPage < 0) curPage = 0;
    const slice = list.slice(curPage * PAGE_SIZE, (curPage + 1) * PAGE_SIZE);

    grid.innerHTML = slice.map(p => {
      const sid = p.sent_id;
      const curRate = ratings[sid] || 0;
      const stars = [1, 2, 3, 4, 5].map(s =>
        `<button type="button" class="chip ampixa-star" data-sid="${esc(sid)}" data-star="${s}" style="padding:2px 8px;font-size:12px;${curRate === s ? "background:var(--md-primary);color:var(--md-on-primary)" : ""}">${s}★</button>`
      ).join("");
      const metaBadges = [
        `<span class="count">${esc(p.category)}</span>`,
        p.contrast_word ? `<span class="count" title="Target contrast word">target: ${esc(p.contrast_word)}</span>` : "",
      ].filter(Boolean).join(" ");

      return `<div class="card sm" style="padding:14px 16px;display:flex;flex-direction:column;justify-content:space-between">
        <div>
          <div style="display:flex;justify-content:space-between;align-items:center;gap:6px;flex-wrap:wrap;margin-bottom:6px">
            <b><code>${esc(sid)}</code></b>
            <div style="display:flex;gap:4px;flex-wrap:wrap">${metaBadges}</div>
          </div>
          <p class="tx" lang="ne" style="margin:6px 0 4px;font-size:15.5px;font-weight:500">${esc(p.text_devanagari)}</p>
          <p class="muted" style="margin:0 0 10px;font-size:12.5px"><i>${esc(p.text_romanized || "")}</i></p>
        </div>
        <div>
          <div style="display:flex;justify-content:space-between;align-items:center;gap:8px;flex-wrap:wrap">
            <div style="display:flex;gap:6px;flex-wrap:wrap">
              <button type="button" class="btn filled ampixa-synth-btn" data-sid="${esc(sid)}" style="height:34px;padding:0 12px;font-size:12.5px"><span class="ms" style="font-size:17px">play_arrow</span>Speak (${esc((window.st && window.st.tier ? window.st.tier : "v1").toUpperCase())})</button>
              <button type="button" class="btn tonal ampixa-studio-btn" data-sid="${esc(sid)}" style="height:34px;padding:0 10px;font-size:12.5px" title="Open in main Studio with spectrogram"><span class="ms" style="font-size:17px">graphic_eq</span>In Studio</button>
            </div>
            <div style="display:flex;gap:3px;align-items:center" title="Self-rate MOS (1-5 stars, saved locally)">${stars}</div>
          </div>
          <div id="ampixa_slot_${esc(sid)}"></div>
        </div>
      </div>`;
    }).join("");

    slice.forEach(p => { if (audioCache[p.sent_id]) renderSlot(p.sent_id); });

    grid.querySelectorAll(".ampixa-synth-btn").forEach(btn => {
      btn.onclick = () => {
        const p = PROMPTS.find(x => x.sent_id === btn.dataset.sid);
        if (p) synthPrompt(p, btn);
      };
    });
    grid.querySelectorAll(".ampixa-studio-btn").forEach(btn => {
      btn.onclick = () => {
        const p = PROMPTS.find(x => x.sent_id === btn.dataset.sid);
        if (!p) return;
        const tBox = $("#text");
        if (tBox) {
          tBox.value = p.text_devanagari;
          if ($("#count")) $("#count").textContent = p.text_devanagari.length;
          if (window.st) window.st.lang = "ne";
          location.hash = "#studio";
          tBox.scrollIntoView({ behavior: "smooth" });
          if ($("#go")) $("#go").click();
        }
      };
    });
    grid.querySelectorAll(".ampixa-star").forEach(btn => {
      btn.onclick = () => {
        ratings[btn.dataset.sid] = Number(btn.dataset.star);
        saveRatings();
        renderCards();
      };
    });

    if (pager) {
      pager.innerHTML = `
        <button type="button" class="btn tonal" id="ampixa-prev" ${curPage <= 0 ? "disabled" : ""} style="height:34px;padding:0 12px">Previous</button>
        <span style="font-size:13px;font-weight:600">Page ${curPage + 1} of ${totalPages} (${list.length} prompts)</span>
        <button type="button" class="btn tonal" id="ampixa-next" ${curPage + 1 >= totalPages ? "disabled" : ""} style="height:34px;padding:0 12px">Next</button>`;
      const prev = document.getElementById("ampixa-prev");
      const next = document.getElementById("ampixa-next");
      if (prev) prev.onclick = () => { if (curPage > 0) { curPage--; renderCards(); } };
      if (next) next.onclick = () => { if (curPage + 1 < totalPages) { curPage++; renderCards(); } };
    }
    updateSummaryBar();
  }

  async function runQuick9CategoryTest(btn) {
    const old = btn.innerHTML;
    btn.disabled = true;
    const cats = ["phonological_minimal_pairs", "question_intonation", "emotion", "contrastive_stress", "robustness", "phrases", "homographs", "newspaper_ambiguities", "natural_speech"];
    const picks = cats.map(c => PROMPTS.find(p => p.category === c)).filter(Boolean);
    for (let i = 0; i < picks.length; i++) {
      btn.innerHTML = `<span class="ms">hourglass_top</span>Testing ${i + 1}/${picks.length} (${picks[i].sent_id})…`;
      await synthPrompt(picks[i], null);
    }
    btn.disabled = false;
    btn.innerHTML = old;
    renderCards();
  }

  function exportSelfReport() {
    const payload = {
      benchmark: "ampixa-neptts-bench-365-self-test",
      timestamp_utc: new Date().toISOString(),
      active_tier: window.st ? window.st.tier : "v1",
      synthesized_count: Object.keys(audioCache).length,
      rated_count: Object.keys(ratings).length,
      ratings,
      metrics: Object.fromEntries(Object.entries(audioCache).map(([k, v]) => [k, { tier: v.tier, dur_s: +v.dur.toFixed(2), rtf: +v.rtf.toFixed(4), phones: v.phones }])),
    };
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "sakhiya_ampixa_selftest_report.json";
    a.click();
  }

  function mountUI() {
    const mount = document.getElementById("ampixa-mount");
    if (!mount) return;
    const catChips = Object.entries(CAT_LABELS).map(([k, lbl]) =>
      `<button type="button" class="chip ampixa-cat" data-cat="${esc(k)}" aria-checked="${String(curCat === k)}" role="radio">${esc(lbl)}</button>`
    ).join("");

    const tierOpts = [
      ["v1", "Sakhiya-V1 (15.98 MB Flagship)"],
      ["micro", "Sakhiya-V2-Micro (4.18 MB INT8)"],
      ["pico", "Sakhiya-V2-Pico (1.39 MB INT8)"],
      ["nano", "Sakhiya-V1-Nano (8.93 MB)"],
      ["atom", "Sakhiya-V1-Atom (6.51 MB)"],
      ["atom_lite", "Sakhiya-V1-Atom-Lite (2.72 MB)"],
    ];
    const activeTier = (window.st && window.st.tier) || "v1";
    const tierSelectHtml = `<select id="ampixa-tier-sel" style="height:38px;padding:0 12px;border-radius:12px;border:1.5px solid var(--md-outline-variant);background:var(--md-surface-container);color:var(--md-on-surface);font:600 13px Poppins,sans-serif">${tierOpts.map(([k, l]) => `<option value="${k}"${k === activeTier ? " selected" : ""}>Tier: ${l}</option>`).join("")}</select>`;

    mount.innerHTML = `
      <div style="display:flex;justify-content:space-between;align-items:center;gap:12px;flex-wrap:wrap;margin-bottom:12px;padding:10px 14px;border-radius:12px;background:var(--md-surface-container-low)" id="ampixa-summary-bar"></div>
      <div style="display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-bottom:12px">
        <input type="search" id="ampixa-search" placeholder="Search 205 #submit / 365 total prompts by ID (sent_001, chirp_001), Devanagari, Roman, or contrast word…" style="flex:1;min-width:220px;padding:9px 14px;border-radius:12px;border:1.5px solid var(--md-outline-variant);background:var(--md-surface-container);color:var(--md-on-surface);font:400 14px Poppins,sans-serif">
        ${tierSelectHtml}
        <button type="button" class="btn filled" id="ampixa-run9" style="height:38px"><span class="ms">bolt</span>Run 9-Category Self-Test (9 Prompts)</button>
        <button type="button" class="btn tonal" id="ampixa-export" style="height:38px"><span class="ms">download</span>Export Self-Test JSON</button>
      </div>
      <div class="chips wrap" id="ampixa-cats" style="margin-bottom:14px">${catChips}</div>
      <div class="gallery" id="ampixa-grid"></div>
      <div style="display:flex;justify-content:center;align-items:center;gap:14px;margin-top:16px" id="ampixa-pager"></div>`;

    const tSel = document.getElementById("ampixa-tier-sel");
    if (tSel) {
      tSel.onchange = e => {
        if (window.st) window.st.tier = e.target.value;
        if (typeof window.renderTiers === "function") window.renderTiers();
        renderCards();
      };
    }

    mount.querySelectorAll(".ampixa-cat").forEach(b => {
      b.onclick = () => {
        curCat = b.dataset.cat;
        curPage = 0;
        mount.querySelectorAll(".ampixa-cat").forEach(x => x.setAttribute("aria-checked", String(x === b)));
        renderCards();
      };
    });
    const sIn = document.getElementById("ampixa-search");
    if (sIn) sIn.oninput = e => { curSearch = e.target.value; curPage = 0; renderCards(); };
    const r9 = document.getElementById("ampixa-run9");
    if (r9) r9.onclick = () => runQuick9CategoryTest(r9);
    const exp = document.getElementById("ampixa-export");
    if (exp) exp.onclick = exportSelfReport;

    renderCards();
  }

  async function init() {
    try {
      PROMPTS = await (await fetch("data/ampixa_prompts.json", { cache: "no-cache" })).json();
      mountUI();
    } catch (e) {
      console.warn("Ampixa prompt pack load error:", e);
    }
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
