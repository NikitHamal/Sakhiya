/* Listening Desk: browse every frozen pack in samples/versions/registry.json. One clip plays at a time. */
(function () {
  "use strict";
  var $ = function (s) { return document.querySelector(s); };
  var esc = function (s) { return String(s == null ? "" : s).replace(/[&<>"]/g, function (c) { return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]; }); };
  var WANT = ["sakhiya_v3_t30", "sakhiya_v3_t15", "sakhiya_v3_t10", "sakhiya_v3_t1", "sakhiya_v5_vedic", "sakhiya_v5_nepali", "sakhiya_pro_v3", "sakhiya_pro_v4_vedic", "sakhiya_pro_v4"];
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
