/* Listening Desk: browse every frozen pack in samples/versions/registry.json. One clip plays at a time. */
(function () {
  "use strict";
  var $ = function (s) { return document.querySelector(s); };
  var esc = function (s) { return String(s == null ? "" : s).replace(/[&<>"]/g, function (c) { return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]; }); };
  var WANT = ["sakhiya_one_polish", "sakhiya_one_vocos", "sakhiya_one_base_4step_g2p", "sakhiya_one_baseline_a61c100", "sakhiya_one", "sakhiya_pro_v3", "sakhiya_v3_t30", "sakhiya_v3_t15", "sakhiya_v3_t10", "sakhiya_v3_t1", "sakhiya_v5_vedic", "sakhiya_v5_nepali", "sakhiya_pro_v4_vedic", "sakhiya_pro_v4"];
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
    var revealed = false, curClipId = "owner_clip1_ai_models";
    var owner4Clips = [
      { id: "owner_clip1_ai_models", lang: "ne", style: "news", text: "दोस्रो समाचार नयाँ पुस्ताका एआई मोडेलहरू बारे छ। गुगलले जेमिनाई फोर नियोन नाम गरेको अत्यन्तै छरितो र शक्तिशाली नयाँ एआई मोडेल सार्वजनिक गरेको छ। जटिल तर्क गर्न लामो कोड लेख्न र धेरै चरण भएका अनुसन्धानात्मक कामहरू छिटो सम्पन्न गर्न यो मोडेल विशेष रूपमा तयार पारिएको बताइएको छ। यस्तै ओपन एआईले पनि जीपीटी सिक्स पोइन्ट टु नाम गरेका नयाँ संस्करणहरू ल्याउँदै प्रयोगकर्ताको तर्फबाट सुरक्षित रूपमा अनलाइन किनमेल गर्न मिल्ने च्याटजिपिटी वालेट सुविधा सुरु गर्ने घोषणा गरेको छ।" },
      { id: "owner_clip2_share_market", lang: "ne", style: "news", text: "पहिलो समाचार कृत्रिम बुद्धिमत्ता बृहत र विश्व सेयर बजारसँग सम्बन्धित छ। प्रसिद्ध एआई अनुसन्धान कम्पनी एन्थ्रोपिकको सेयर बजारमा कारोबार खुलेको पहिलो दिनमै १७ प्रतिशतले उल्लेखनीय वृद्धि भएको छ। तर लगानीकर्ताहरूको नाफा सुरक्षित गर्ने होडबाजीका कारण अर्को दिन करिब १४ प्रतिशतले गिरावट आयो जसले गर्दा अन्य प्रमुख प्रविधि कम्पनीहरूको सेयर मूल्यमा समेत ठूलो उतारचढाव देखिएको छ। विश्लेषकहरूका अनुसार एआई पूर्वाधारमा भइरहेको अर्बौँ डलरको लगानीले आगामी त्रैमासिकमा बजारलाई थप संवेदनशील बनाउने निश्चित छ।" },
      { id: "owner_clip3_intro_bulletin", lang: "ne", style: "news", text: "नमस्ते सबैजनालाई। म सखिया तपाईंको समाचार वाचक। आजको विशेष प्राविधिक बुलेटिनमा हामी विश्वभर भइरहेका कृत्रिम बुद्धिमत्ता अर्थात एआईका नवीनतम उपलब्धिहरू, नेपाली प्रविधि क्षेत्रको तीव्र विकास र मोबाइल एप्लिकेसन निर्माणमा आएका नयाँ परिवर्तनहरूबारे विस्तृत चर्चा गर्नेछौँ। प्रविधिको दुनियाँ हरेक दिन नयाँ मोड लिइरहेको छ र त्यसको प्रत्यक्ष प्रभाव हाम्रो दैनिक जीवन, शिक्षा र व्यवसायमा पर्न थालेको छ। त चलौँ आजका मुख्य समाचार तर्फ लागौँ।" },
      { id: "owner_clip4_sanskrit_gita", lang: "auto", style: "news", text: "हेलो साथीहरू, म सखिया हुँ। निकित हमाल ज्यूले बनाउनुभएको सानो टेक्स्ट टु स्पीच मोडेल। म संस्कृत पनि स्पष्ट बोल्न सक्छु। धर्मक्षेत्रे कुरुक्षेत्रे समवेता युयुत्सवः। मामकाः पाण्डवाश्चैव किमकुर्वत सञ्जय। आज मौसम निकै रमाइलो र शान्त छ।" }
    ];
    var unseen5News = [
      { id: "unseen_news1_hydropower", lang: "ne", style: "news", text: "नेपाल विद्युत प्राधिकरणले आगामी वर्षायाममा थप चार सय पचास मेगावाट स्वच्छ जलविद्युत ऊर्जा छिमेकी मुलुकहरूमा निर्यात गर्ने महत्त्वाकांक्षी योजना सार्वजनिक गरेको छ।" },
      { id: "unseen_news2_digital_banking", lang: "ne", style: "news", text: "नेपाल राष्ट्र बैंकका अनुसार पछिल्लो एक वर्षमा मोबाइल बैंकिङ र क्यूआर कोडमार्फत हुने डिजिटल कारोबारमा उल्लेखनीय वृद्धि भएसँगै साइबर सुरक्षामा विशेष सतर्कता अपनाइएको छ।" },
      { id: "unseen_news3_kathmandu_weather", lang: "ne", style: "news", text: "पश्चिमी वायुको सामान्य प्रभावका कारण आज काठमाडौँ उपत्यका लगायत पहाडी भूभागमा मौसम आंशिक बदली रहे पनि आन्तरिक तथा अन्तर्राष्ट्रिय हवाई उडानहरू नियमित रूपमा सञ्चालन भइरहेका छन्।" },
      { id: "unseen_news4_ai_education", lang: "ne", style: "news", text: "त्रिभुवन विश्वविद्यालय र काठमाडौँ विश्वविद्यालयका अनुसन्धानकर्ताहरूले नेपाली भाषाको प्राकृतिक प्रशोधन र कृत्रिम बुद्धिमत्ताको विकासका लागि संयुक्त प्रयोगशाला स्थापना गरेका छन्।" },
      { id: "unseen_news5_closing_bulletin", lang: "ne", style: "news", text: "आजको विशेष प्राविधिक र आर्थिक समाचार बुलेटिन यहीँ सकिन्छ। ताजा र तथ्यपरक समाचारका लागि हामीसँग जोडिरहनुहोला, नमस्ते अनि शुभरात्री।" }
    ];
    var hard5Pron = [
      { id: "hard_pron1_wallet_ai", lang: "ne", style: "news", text: "ओपन एआई र जेमिनाई फोर नियोनसँगै च्याटजिपिटी वालेट र वेब वर्ल्डका प्रयोगकर्ताको संख्या बढेको छ।" },
      { id: "hard_pron2_geminates", lang: "ne", style: "news", text: "कृत्रिम बुद्धिमत्ताको उल्लेखनीय विकासले पक्कै पनि चिटिक्क परेको डिजिटल सेवा दिनेछ।" },
      { id: "hard_pron3_halanta_compounds", lang: "ne", style: "news", text: "बृहत् अनुसन्धानात्मक कार्य अर्थात संवेदनशील प्राविधिक उपलब्धिहरू सुरक्षित रूपमा सम्पन्न भए।" },
      { id: "hard_pron4_chandrabindu_clause", lang: "ne", style: "news", text: "त चलौँ, काठमाडौँका पाँचौँ पुस्ताका मोबाइल एप्लिकेसनबारे विस्तृत चर्चा गर्नेछौँ र अघि लागौँ।" },
      { id: "hard_pron5_numbers_sibilants", lang: "ne", style: "news", text: "एन्थ्रोपिकको सेयरमा १७ प्रतिशतले वृद्धि र १४ प्रतिशतले गिरावट आयो, जसले त्रैमासिकमा उतारचढाव ल्यायो।" }
    ];
    var baseClips = owner4Clips.concat(unseen5News).concat(hard5Pron);
    var abSystems = [
      { blind: "Sample A (-16 LUFS)", real: "3 · Candidate A: Sakhiya-One-Polish [~8.5 MB INT8 · Anchored AM + MS-SB-CQT AWP · 4-Step]", metrics: "PROMOTED ~8.5 MB BROWSER FLAGSHIP ✅ · 36/36 Pron Checklist (वालेट != बालेट) · MS-SB-CQT Joint E2E + LSQ QAT", tag: "sakhiya_one_polish" },
      { blind: "Sample B (-16 LUFS)", real: "4 · Candidate B: Sakhiya-One-Vocos [~16 MB INT8 · Mel80To100Adapter + Pretrained Vocos-24kHz · 4-Step]", metrics: "STUDIO VOCOS TIER ✅ · Pretrained charactr/vocos-mel-24khz (Zero Harmonic Comb · Silky Sibilants)", tag: "sakhiya_one_vocos" },
      { blind: "Sample C (-16 LUFS)", real: "2 · Baseline + Fixed G2P + 4-Step MeanFlow [0 Extra Bytes · 8.46 MB INT8]", metrics: "Zero-Training G2P + 4-Step Ablation: Fixes वालेट/एआई/geminates & reduces mid-vowel warble by 16.6%", tag: "sakhiya_one_base_4step_g2p" },
      { blind: "Sample D (-16 LUFS)", real: "1 · Original Baseline Sakhiya-One [Untouched a61c100 · 2-Step · 8.46 MB INT8]", metrics: "Original Shipped Baseline (a61c100): ne_pcer=0.0590 · comb_grit=1.4584 · बालेट leak on Clip 1", tag: "sakhiya_one_baseline_a61c100" }
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
