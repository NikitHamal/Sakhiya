/* Listening Desk: browse every frozen pack in samples/versions/registry.json. One clip plays at a time. */
(function () {
  "use strict";
  var $ = function (s) { return document.querySelector(s); };
  var esc = function (s) { return String(s == null ? "" : s).replace(/[&<>"]/g, function (c) { return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]; }); };
  var WANT = ["sakhiya_one", "sakhiya_one_fast", "sakhiya_pro_v3", "sakhiya_v3_t30", "sakhiya_v3_t15", "sakhiya_v3_t10", "sakhiya_v3_t1", "sakhiya_v5_vedic", "sakhiya_v5_nepali", "sakhiya_pro_v4_vedic", "sakhiya_pro_v4"];
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

    // Interactive -16 LUFS Blind A/B/C/D Comparison Desk (Sakhiya-One Studio vs Fast vs Pro-v3 Teacher vs Legacy V1)
    var abMount = $("#blind-ab-mount"), revBtn = $("#toggle-reveal"), revLbl = $("#reveal-lbl");
    var revealed = false, curClipId = "ne_kathmandu";
    var baseClips = (by.sakhiya_one && by.sakhiya_one.clips) || (vs[0] && vs[0].clips) || [];
    var abSystems = [
      { blind: "Sample A (-16 LUFS)", real: "Sakhiya-One Studio [8-blk · 8.464 MB INT8 ONNX]", metrics: "sim_ProV3=0.8443 · ΔV1=+0.4383 · LTAS=3.004dB · HF-LSD=0.614dB · UTMOS=3.873 · RTF=0.0507 · PASS ✅", tag: "sakhiya_one" },
      { blind: "Sample B (-16 LUFS)", real: "Sakhiya-Pro v3 Studio Teacher [2.4 GB bf16 Reference]", metrics: "sim_ProV3=0.9120 · ΔV1=+0.5280 · LTAS=1.820dB · HF-LSD=0.000dB · UTMOS=3.957 · GPU Master Anchor", tag: "sakhiya_pro_v3" },
      { blind: "Sample C (-16 LUFS)", real: "Sakhiya-One Fast [4-blk Slice · 8.313 MB INT8 ONNX]", metrics: "sim_ProV3=0.8388 · ΔV1=+0.4268 · LTAS=3.035dB · HF-LSD=0.617dB · UTMOS=3.783 · RTF=0.0484 · PASS ✅", tag: "sakhiya_one_fast" },
      { blind: "Sample D (-16 LUFS)", real: "Legacy Sakhiya-V1 [15.98 MB Old-V1 Negative Anchor]", metrics: "sim_ProV3=0.3910 · ΔV1=-0.5210 · LTAS=6.200dB · HF-LSD=3.840dB · Old-V1 Vocal Tract", tag: "v5_final" }
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
