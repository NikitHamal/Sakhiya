/* Ampixa Lab — static browser for the pre-rendered, cleaned NepTTS-Bench clips (docs/samples/ampixa/<tier>/<sent_id>.mp3).
   No synthesis happens here; clips were rendered on the GPU box by scripts/ampixa_render.py. */
(function () {
  "use strict";
  var $ = function (s) { return document.querySelector(s); };
  var esc = function (s) { return String(s == null ? "" : s).replace(/[&<>"]/g, function (c) { return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]; }); };
  var TIER_ORDER = ["t30", "t15", "v1", "micro", "pico"];
  var TIER_UI = {
    t30: { name: "Pro-v3 T30", c: "#0058df" },
    t15: { name: "Pro-v3 T15", c: "#006782" },
    v1: { name: "Sakhiya-V1", c: "#7a4cc4" },
    micro: { name: "V2-Micro", c: "#b4561b" },
    pico: { name: "V2-Pico", c: "#b3261e" }
  };
  var CATS = {
    official: "Official #submit pack",
    all: "All prompts",
    natural_speech: "Natural speech",
    phonological_minimal_pairs: "Minimal pairs",
    question_intonation: "Questions",
    robustness: "Tongue twisters",
    phrases: "Phrases",
    emotion: "Emotion",
    contrastive_stress: "Contrastive stress",
    homographs: "Homographs",
    newspaper_ambiguities: "Headlines"
  };
  var PAGE = 20;
  var S = { prompts: [], idx: null, cat: "official", q: "", sel: {}, shown: PAGE, cur: null, seq: null };
  var audio = new Audio();
  audio.preload = "none";

  function isOfficial(p) { return String(p.sent_id || "").indexOf("chirp_") !== 0; }
  function inCat(p) { if (S.cat === "all") return true; if (S.cat === "official") return isOfficial(p); return p.category === S.cat; }
  function filtered() {
    var q = S.q.trim().toLowerCase();
    return S.prompts.filter(function (p) {
      if (!inCat(p)) return false;
      if (!q) return true;
      return (String(p.sent_id) + " " + p.text_devanagari + " " + (p.text_romanized || "") + " " + p.category + " " + (p.contrast_word || "")).toLowerCase().indexOf(q) >= 0;
    });
  }
  function tiersOn() { return TIER_ORDER.filter(function (t) { return S.sel[t] && S.idx.clips[t]; }); }
  function fmt(t) { if (!isFinite(t)) t = 0; var m = Math.floor(t / 60), s = Math.floor(t % 60); return m + ":" + (s < 10 ? "0" : "") + s; }
  function clipMeta(t, sid) { return (S.idx.clips[t] || {})[sid]; }

  /* ---------- chips ---------- */
  function renderChips() {
    var counts = { all: S.prompts.length, official: S.prompts.filter(isOfficial).length };
    S.prompts.forEach(function (p) { counts[p.category] = (counts[p.category] || 0) + 1; });
    var cats = ["official", "all"].concat(Object.keys(CATS).filter(function (k) { return k !== "official" && k !== "all" && counts[k]; }));
    $("#cats").innerHTML = cats.map(function (k) {
      return '<button class="chip" role="radio" data-cat="' + k + '" aria-checked="' + (S.cat === k) + '">' + esc(CATS[k] || k) + ' <span class="n">' + (counts[k] || 0) + "</span></button>";
    }).join("");
    $("#tierSel").innerHTML = TIER_ORDER.filter(function (t) { return S.idx.clips[t]; }).map(function (t) {
      var on = !!S.sel[t];
      return '<button class="chip tier-chip' + (on ? " on" : "") + '" data-tier="' + t + '" aria-pressed="' + on + '" style="--c:' + TIER_UI[t].c + '"><span class="dot"></span>' + esc(TIER_UI[t].name) + "</button>";
    }).join("");
  }

  /* ---------- list ---------- */
  function promptHTML(p) {
    var sid = p.sent_id;
    var btns = tiersOn().map(function (t) {
      var m = clipMeta(t, sid);
      if (!m || m.error) return "";
      var on = S.cur && S.cur.t === t && S.cur.sid === sid;
      return '<button class="pbtn' + (on ? " on" : "") + '" data-t="' + t + '" data-sid="' + esc(sid) + '" style="--c:' + TIER_UI[t].c + '" aria-label="Play ' + esc(TIER_UI[t].name) + '">' +
        '<span class="ico"><span class="ms">' + (on && !audio.paused ? "pause" : "play_arrow") + '</span></span>' +
        '<span class="tx"><b>' + esc(TIER_UI[t].name) + "</b><small>" + m.dur.toFixed(1) + " s · " + m.kb.toFixed(0) + ' KB</small></span><span class="bar"></span></button>';
    }).join("");
    return '<article class="prompt' + (S.cur && S.cur.sid === sid ? " playing" : "") + '" data-sid="' + esc(sid) + '">' +
      '<div class="prompt-top"><div class="prompt-tx"><div class="dev" lang="ne">' + esc(p.text_devanagari) + '</div><div class="rom">' + esc(p.text_romanized || "") + "</div></div>" +
      '<div class="meta"><span class="tag cat">' + esc(CATS[p.category] || p.category) + '</span><span class="tag id">' + esc(sid) + "</span>" +
      '<button class="chip" data-seq="' + esc(sid) + '" title="Play every selected model one after another"><span class="ms" style="font-size:18px">queue_music</span>A/B all</button></div></div>' +
      '<div class="tiers-row">' + btns + "</div></article>";
  }
  function renderList() {
    var all = filtered(), vis = all.slice(0, S.shown);
    $("#count").textContent = all.length + " prompt" + (all.length === 1 ? "" : "s") + " · " + tiersOn().length + " model" + (tiersOn().length === 1 ? "" : "s") + " selected";
    $("#list").innerHTML = vis.length ? vis.map(promptHTML).join("") : '<div class="empty"><span class="ms">search_off</span><p>No prompts match that search.</p></div>';
    $("#more").hidden = all.length <= S.shown;
    if (!$("#more").hidden) $("#more").lastChild.nodeValue = "Show more prompts (" + (all.length - S.shown) + " left)";
  }

  /* ---------- playback ---------- */
  function refreshButtons() {
    document.querySelectorAll(".pbtn").forEach(function (b) {
      var on = S.cur && S.cur.t === b.dataset.t && S.cur.sid === b.dataset.sid;
      b.classList.toggle("on", !!on);
      b.querySelector(".ico .ms").textContent = on && !audio.paused ? "pause" : "play_arrow";
      if (!on) b.querySelector(".bar").style.width = "0";
    });
    document.querySelectorAll(".prompt").forEach(function (a) { a.classList.toggle("playing", !!(S.cur && S.cur.sid === a.dataset.sid)); });
    $("#mPlay .ms").textContent = audio.paused ? "play_arrow" : "pause";
  }
  function play(t, sid) {
    if (S.cur && S.cur.t === t && S.cur.sid === sid) { if (audio.paused) audio.play().catch(function () {}); else audio.pause(); refreshButtons(); return; }
    S.cur = { t: t, sid: sid };
    audio.src = "samples/ampixa/" + t + "/" + sid + ".mp3";
    audio.play().catch(function () {});
    var p = S.prompts.filter(function (x) { return x.sent_id === sid; })[0] || {};
    $("#mT1").textContent = TIER_UI[t].name + " · " + (p.text_devanagari || sid);
    $("#mini").classList.add("show");
    refreshButtons();
  }
  function playSeq(sid) {
    var ts = tiersOn(); if (!ts.length) return;
    S.seq = { sid: sid, i: 0, ts: ts };
    S.cur = null; play(ts[0], sid);
  }
  audio.addEventListener("ended", function () {
    if (S.seq && S.seq.sid === (S.cur && S.cur.sid) && S.seq.i + 1 < S.seq.ts.length) { S.seq.i++; S.cur = null; play(S.seq.ts[S.seq.i], S.seq.sid); return; }
    S.seq = null; refreshButtons();
  });
  audio.addEventListener("pause", refreshButtons);
  audio.addEventListener("play", refreshButtons);
  audio.addEventListener("timeupdate", function () {
    var d = audio.duration || 0, c = audio.currentTime || 0;
    $("#mSeek").value = d ? Math.round(1000 * c / d) : 0;
    $("#mT2").textContent = fmt(c) + " / " + fmt(d);
    if (S.cur) {
      var b = document.querySelector('.pbtn[data-t="' + S.cur.t + '"][data-sid="' + S.cur.sid + '"] .bar');
      if (b) b.style.width = d ? (100 * c / d) + "%" : "0";
    }
  });
  audio.addEventListener("error", function () { if (S.cur) $("#mT2").textContent = "Could not load this clip"; });
  $("#mSeek").addEventListener("input", function (e) { if (audio.duration) audio.currentTime = audio.duration * e.target.value / 1000; });
  $("#mPlay").addEventListener("click", function () { if (audio.paused) audio.play().catch(function () {}); else audio.pause(); });
  $("#mClose").addEventListener("click", function () { audio.pause(); S.seq = null; S.cur = null; $("#mini").classList.remove("show"); refreshButtons(); });

  /* ---------- events ---------- */
  document.addEventListener("click", function (e) {
    var t = e.target.closest ? e.target : null; if (!t) return;
    var b;
    if ((b = t.closest("[data-cat]"))) { S.cat = b.dataset.cat; S.shown = PAGE; renderChips(); renderList(); return; }
    if ((b = t.closest("[data-tier]"))) {
      var k = b.dataset.tier, n = tiersOn().length;
      if (S.sel[k] && n === 1) return;
      S.sel[k] = !S.sel[k]; renderChips(); renderList(); return;
    }
    if ((b = t.closest("[data-seq]"))) { playSeq(b.dataset.seq); return; }
    if ((b = t.closest(".pbtn"))) { S.seq = null; play(b.dataset.t, b.dataset.sid); return; }
  });
  $("#more").addEventListener("click", function () { S.shown += PAGE; renderList(); });
  var qt;
  $("#q").addEventListener("input", function (e) {
    clearTimeout(qt); qt = setTimeout(function () { S.q = e.target.value; S.shown = PAGE; $("#qx").hidden = !S.q; renderList(); }, 120);
  });
  $("#qx").addEventListener("click", function () { $("#q").value = ""; S.q = ""; $("#qx").hidden = true; renderList(); $("#q").focus(); });
  $("#theme").addEventListener("click", function () {
    var n = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = n; try { localStorage.setItem("theme", n); } catch (e) {}
  });
  addEventListener("scroll", function () { $("#appbar").classList.toggle("scrolled", scrollY > 6); }, { passive: true });
  document.addEventListener("keydown", function (e) {
    if (e.target && /INPUT|TEXTAREA/.test(e.target.tagName)) return;
    if (e.code === "Space" && S.cur) { e.preventDefault(); if (audio.paused) audio.play().catch(function () {}); else audio.pause(); }
    if (e.key === "/") { e.preventDefault(); $("#q").focus(); }
  });

  /* ---------- summary ---------- */
  function renderSummary() {
    var sm = S.idx.summary || {}, ts = TIER_ORDER.filter(function (t) { return sm[t]; });
    var clips = ts.reduce(function (a, t) { return a + sm[t].n; }, 0);
    var dn = ts.length ? ts.reduce(function (a, t) { return a + (sm[t].noise_db_before - sm[t].noise_db_after); }, 0) / ts.length : 0;
    var shut = ts.reduce(function (a, t) { return a + sm[t].shutter_before; }, 0), shutA = ts.reduce(function (a, t) { return a + sm[t].shutter_after; }, 0);
    $("#tiles").innerHTML =
      '<div class="tile"><b>' + S.prompts.length + "</b><span>benchmark prompts</span></div>" +
      '<div class="tile"><b>' + clips + "</b><span>cleaned clips from " + ts.length + " models</span></div>" +
      '<div class="tile"><b>−' + dn.toFixed(1) + " dB</b><span>average noise-floor drop</span></div>" +
      '<div class="tile"><b>' + (shut ? Math.round(100 * (1 - shutA / shut)) : 0) + "%</b><span>fewer shutter dips</span></div>";
    $("#heroPills").innerHTML = ts.map(function (t) { return '<span class="pill" style="--c:' + TIER_UI[t].c + '"><span class="dot" style="background:' + TIER_UI[t].c + '"></span><i>' + esc(TIER_UI[t].name) + "</i></span>"; }).join("");
    var meta = S.idx.tiers || {};
    var rows = ts.map(function (t) {
      var s = sm[t], m = meta[t] || {};
      return "<tr><td><span class=\"dot\" style=\"background:" + TIER_UI[t].c + "\"></span>" + esc(TIER_UI[t].name) + "</td><td>" + (m.mb ? m.mb.toFixed(2) + " MB" : "—") + "</td><td>" + (m.utmos ? m.utmos.toFixed(2) : "—") + "</td><td>" + s.rtf_mean.toFixed(3) +
        "</td><td>" + s.noise_db_before.toFixed(1) + " → <span class=\"good\">" + s.noise_db_after.toFixed(1) + " dB</span></td><td>" + s.shutter_before + " → <span class=\"good\">" + s.shutter_after +
        "</span></td><td>" + s.clicks_before + " → <span class=\"good\">" + s.clicks_after + "</span></td></tr>";
    }).join("");
    $("#mt").innerHTML = "<thead><tr><th>Model</th><th>INT8 size</th><th>UTMOS (gate)</th><th>CPU RTF</th><th>Noise floor</th><th>Shutter dips</th><th>Clicks</th></tr></thead><tbody>" + rows + "</tbody>";
  }

  /* ---------- boot ---------- */
  Promise.all([
    fetch("data/ampixa_prompts.json").then(function (r) { return r.json(); }),
    fetch("samples/ampixa/index.json").then(function (r) { return r.json(); })
  ]).then(function (r) {
    S.prompts = r[0]; S.idx = r[1];
    TIER_ORDER.forEach(function (t) { S.sel[t] = !!S.idx.clips[t]; });
    var h = (location.hash || "").slice(1); if (h && CATS[h]) S.cat = h;
    renderSummary(); renderChips(); renderList();
  }).catch(function (e) {
    $("#list").innerHTML = '<div class="empty"><span class="ms">cloud_off</span><p>Could not load the Ampixa clips: ' + esc(e && e.message || e) + "</p></div>";
  });
})();
