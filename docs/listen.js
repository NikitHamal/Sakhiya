/* Listening Desk: browse every frozen pack in samples/versions/registry.json. One clip plays at a time. */
(function () {
  "use strict";
  var $ = function (s) { return document.querySelector(s); };
  var esc = function (s) { return String(s == null ? "" : s).replace(/[&<>"]/g, function (c) { return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]; }); };
  var WANT = ["sakhiya_news", "sakhiya_news_leg1_preview", "sakhiya_one", "sakhiya_one_fast", "sakhiya_pro_v3", "sakhiya_v3_t30", "sakhiya_v3_t15", "sakhiya_v3_t10", "sakhiya_v3_t1", "sakhiya_v5_vedic", "sakhiya_v5_nepali", "sakhiya_pro_v4_vedic", "sakhiya_pro_v4"];
  var RETIRED_OR_DUP = { sakhiya_v3_t5: 1, sakhiya_v4_micro: 1, sakhiya_v5_micro: 1, sakhiya_v4_pico: 1, sakhiya_v5_pico: 1 };
  var EPS = [
    ["sakhiya_pro/ep01_v5/voiceover_ep01.mp3", "Episode 01 · v5 master", "33/33 lines, -14.0 LUFS"],
    ["sakhiya_pro/ep01_v4/voiceover_ep01.mp3", "Episode 01 · v4", "previous master"]
  ];
  var cur = null;

  // keep a single audio element playing at a time across the page
  document.addEventListener("play", function (e) {
    if (cur && cur !== e.target) cur.pause();
    cur = e.target;
  }, true);

  function clipCard(src, title, sub, text) {
    return '<div class="clip-card"><b><span>' + esc(title) + '</span></b>' +
      (sub ? '<span class="hint" style="margin:0">' + esc(sub) + "</span>" : "") +
      (text ? "<small lang=\"ne\">" + esc(text) + "</small>" : "") +
      '<audio controls preload="none" src="' + esc(src) + '"></audio></div>';
  }
  Promise.all(EPS.map(function (e) { return fetch(e[0], { method: "HEAD" }).then(function (r) { return r.ok ? e : null; }).catch(function () { return null; }); }))
    .then(function (ok) {
      ok = ok.filter(Boolean);
      if (!ok.length) { $("#featured").style.display = "none"; return; }
      $("#eps").innerHTML = ok.map(function (e) { return clipCard(e[0], e[1], e[2]); }).join("");
    });

  $("#theme").addEventListener("click", function () {
    var n = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = n; try { localStorage.setItem("theme", n); } catch (e) {}
  });
  addEventListener("scroll", function () { $("#appbar").classList.toggle("scrolled", scrollY > 6); }, { passive: true });

  fetch("samples/versions/registry.json").then(function (r) { return r.json(); }).then(function (REG) {
    var vs = REG.versions, by = {};
    vs.forEach(function (v) { by[v.tag] = v; });

    // Interactive -16 LUFS Blind A/B/C/D Comparison Desk (Sakhiya-News Final vs Sakhiya-One vs Studio Anchor nep_6834 vs Leg-1)
    var abMount = $("#blind-ab-mount"), revBtn = $("#toggle-reveal"), revLbl = $("#reveal-lbl");
    var revealed = false, curClipId = "ne_kathmandu";
    var extraStudioProbes = [
      { id: "sib_stress", lang: "ne", style: "news", text: "सफा र चिसो बिहानीमा गहिरो सास फेर्दा मनमा शान्ति मिल्छ, छाता ओढेर बाहिर निस्कौँ।" },
      { id: "breath_news", lang: "ne", style: "news", text: "नेपाल राष्ट्र बैंकले आज नयाँ मौद्रिक नीति सार्वजनिक गरेको छ, जसले डिजिटल प्रविधि र अर्थतन्त्रमा सकारात्मक प्रभाव पार्नेछ।" },
      { id: "gem_katti", lang: "ne", style: "news", text: "कति राम्रो दिन छ आज, तर हिजो कत्ति धेरै जाडो भएको थियो र पक्कै पानी पर्यो।" },
      { id: "gem_chittikka", lang: "ne", style: "news", text: "चिटिक्क परेको सानो गाउँमा ठ्ठयाक्कै समयमा बस आइपुग्यो र सबै खुसी भए।" },
      { id: "gem_sakkali", lang: "ne", style: "news", text: "यो सक्कली नेपाली चिया हो, सरकारी अड्डाका कर्मचारीहरू पनि छक्क परे।" }
    ];
    var baseClips = ((by.sakhiya_news && by.sakhiya_news.clips) || (by.sakhiya_one && by.sakhiya_one.clips) || (vs[0] && vs[0].clips) || []).concat(extraStudioProbes);
    var abSystems = [
      { blind: "Sample A (-16 LUFS)", real: "Sakhiya-News Final Flagship [100-Mel Dual-Source OLA Noise + Coupled 6D Prosody + Unrolled 2-Step E2E · 9.793 MB INT8]", metrics: "PASS ✅ · 97.94% INT8 W · HF_LSD_4_12kHz=0.983dB · sim_Anchor=0.8855 (ΔV1=+0.4833) · SilSNR=72.0dB · Voiced HNR=18.16dB (Δ=0.89dB) · ne_pcer=0.0449 (+0.0129 gap) · gem=2.021x · F0_slope=-1.23st/s · E2E 1T RTF=0.0547 · Py<->JS=0.0dB", tag: "sakhiya_news" },
      { blind: "Sample B (-16 LUFS)", real: "Sakhiya-One Studio [Archived 80-Mel 8-blk · 8.464 MB INT8 ONNX Baseline]", metrics: "Re-scored Honest Gate: HF_LSD_4_12kHz=1.917dB · sim_Anchor=0.8443 · ne_pcer=0.0590 (+0.0270 gap) · gem=1.587x", tag: "sakhiya_one" },
      { blind: "Sample C (-16 LUFS)", real: "Raw Studio Anchor nep_6834 / Reference [24 kHz Full-Band Reference]", metrics: "Raw Un-VC'd nep_6834 Studio Anchor: Voiced HNR=19.05dB · Chest LTAS Δ=0.0dB · ne_pcer=0.0320 · Modal SHR=0.019", tag: "sakhiya_pro_v3" },
      { blind: "Sample D (-16 LUFS)", real: "Sakhiya-News Leg-1 Preview [Unmasked AWP + Independent Heads · 9.914 MB INT8]", metrics: "Leg-1 Relay Baseline: HF_LSD_4_12kHz=1.399dB · sim_Anchor=0.8862 · ne_pcer=0.0542 (+0.0222 gap) · gem=1.873x · RTF=0.0599", tag: "sakhiya_news_leg1_preview" }
    ];
    function renderBlindDesk() {
      if (!abMount || !baseClips.length) return;
      var clipObj = baseClips.find(function (c) { return c.id === curClipId; }) || baseClips[0];
      var sel = '<div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:12px">' + baseClips.map(function (c) {
        return '<button class="chip' + (c.id === clipObj.id ? " on" : "") + '" type="button" data-cid="' + esc(c.id) + '">' + esc(c.id) + " (" + esc(c.lang.toUpperCase()) + ")</button>";
      }).join("") + "</div>";
      var cards = abSystems.map(function (sys) {
        var title = revealed ? sys.real : sys.blind;
        var sub = revealed ? sys.metrics : "Click 'Reveal Identities & 4-Layer Gate' after listening to unmask";
        var src = "samples/versions/" + sys.tag + "/" + clipObj.id + ".mp3";
        return clipCard(src, title, sub, clipObj.text);
      }).join("");
      abMount.innerHTML = sel + '<div class="clip-grid">' + cards + "</div>";
      Array.prototype.forEach.call(abMount.querySelectorAll("[data-cid]"), function (b) {
        b.addEventListener("click", function () { curClipId = b.dataset.cid; renderBlindDesk(); });
      });
    }
    if (revBtn) {
      revBtn.addEventListener("click", function () {
        revealed = !revealed;
        if (revLbl) revLbl.textContent = revealed ? "Hide Identities (Blind Mode)" : "Reveal Identities & 4-Layer Gate";
        renderBlindDesk();
      });
    }
    renderBlindDesk();

    var rest = vs.map(function (v) { return v.tag; }).filter(function (t) { return WANT.indexOf(t) < 0 && !RETIRED_OR_DUP[t]; }).reverse();
    var tags = WANT.concat(rest).filter(function (t) { return by[t] && !RETIRED_OR_DUP[t]; });
    var tabs = $("#tabs");
    tabs.innerHTML = tags.map(function (t) { return '<button class="chip" role="tab" data-t="' + esc(t) + '">' + esc(t.replace(/^sakhiya_/, "")) + "</button>"; }).join("");
    function show(t) {
      Array.prototype.forEach.call(tabs.children, function (b) { b.classList.toggle("on", b.dataset.t === t); b.setAttribute("aria-selected", b.dataset.t === t); });
      var v = by[t], m = v.metrics || {};
      var pills = [];
      if (m.int8_mb) pills.push('<span class="pill"><span class="ms">deployed_code</span><i>' + m.int8_mb + " MB INT8</i></span>");
      if (m.utmos) pills.push('<span class="pill tone"><span class="ms">star</span><i>UTMOS ' + m.utmos + "</i></span>");
      if (m.rtf_1thread) pills.push('<span class="pill"><span class="ms">speed</span><i>RTF ' + m.rtf_1thread + "</i></span>");
      $("#pack").innerHTML = '<div class="card-h"><span class="ms">album</span><h2>' + esc(v.label) + '</h2></div><div class="status" style="margin-bottom:12px">' + pills.join("") + '</div><p class="muted" style="margin-bottom:16px">' + esc(v.notes || "") + '</p><div class="clip-grid" id="g">Loading clips…</div>';
      var draw = function (clips) {
        $("#g").innerHTML = clips.length ? clips.map(function (c) {
          return clipCard(c.mp3 || "samples/versions/" + t + "/" + c.id + ".mp3", c.id, c.style || "", c.text);
        }).join("") : "<p class=\"muted\">No clips listed for this pack.</p>";
      };
      if (Array.isArray(v.clips) && v.clips.length) draw(v.clips);
      else fetch("samples/versions/" + t + "/manifest.json").then(function (r) { return r.json(); }).then(function (mf) { draw(mf.clips || []); })
        .catch(function (e) { $("#g").textContent = "Could not load manifest: " + e; });
      try { history.replaceState(null, "", "#" + t); } catch (e) {}
    }
    tabs.addEventListener("click", function (e) { var b = e.target.closest("[data-t]"); if (b) show(b.dataset.t); });
    var h = (location.hash || "").slice(1);
    show(by[h] ? h : tags[0]);
  }).catch(function (e) { $("#pack").textContent = "Could not load registry: " + e; });
})();
