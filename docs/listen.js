/* Listening Desk: browse every frozen pack in samples/versions/registry.json. One clip plays at a time. */
(function () {
  "use strict";
  var $ = function (s) { return document.querySelector(s); };
  var esc = function (s) { return String(s == null ? "" : s).replace(/[&<>"]/g, function (c) { return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]; }); };
  var WANT = ["sakhiya_news_v2_candidate", "sakhiya_one", "sakhiya_news_leg1_preview", "sakhiya_news", "sakhiya_one_fast", "sakhiya_pro_v3", "sakhiya_v3_t30", "sakhiya_v3_t15", "sakhiya_v3_t10", "sakhiya_v3_t1", "sakhiya_v5_vedic", "sakhiya_v5_nepali", "sakhiya_pro_v4_vedic", "sakhiya_pro_v4"];
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
    var extraStudioProbes = [
      { id: "sib_stress", lang: "ne", style: "news", text: "सफा र चिसो बिहानीमा गहिरो सास फेर्दा मनमा शान्ति मिल्छ, छाता ओढेर बाहिर निस्कौँ।" },
      { id: "breath_news", lang: "ne", style: "news", text: "नेपाल राष्ट्र बैंकले आज नयाँ मौद्रिक नीति सार्वजनिक गरेको छ, जसले डिजिटल प्रविधि र अर्थतन्त्रमा सकारात्मक प्रभाव पार्नेछ।" },
      { id: "gem_katti", lang: "ne", style: "news", text: "कति राम्रो दिन छ आज, तर हिजो कत्ति धेरै जाडो भएको थियो र पक्कै पानी पर्यो।" },
      { id: "gem_chittikka", lang: "ne", style: "news", text: "चिटिक्क परेको सानो गाउँमा ठ्ठयाक्कै समयमा बस आइपुग्यो र सबै खुसी भए।" },
      { id: "gem_sakkali", lang: "ne", style: "news", text: "यो सक्कली नेपाली चिया हो, सरकारी अड्डाका कर्मचारीहरू पनि छक्क परे।" }
    ];
    var baseClips = owner4Clips.concat((by.sakhiya_one && by.sakhiya_one.clips) || (by.sakhiya_news && by.sakhiya_news.clips) || (vs[0] && vs[0].clips) || []).concat(extraStudioProbes);
    var abSystems = [
      { blind: "Sample A (-16 LUFS)", real: "Sakhiya-News v2 Candidate [4-Step CFG=0 · True FP16 Vocoder Heads · Fixed Nepali G2P & Chunker · 9.714 MB INT8]", metrics: "STAGING CANDIDATE ✅ · 86.56% INT8 W (144 FP16 Studio-Head Nodes) · STFT_LSD=0.1004dB (Owner 4-Clip=0.0941dB) · s/chh flatness=0.677/0.651 · ne_pcer=0.0431 · 0 shutters/pitch_breaks/clicks/pauses>250ms · 4-Step 1T RTF=0.0567", tag: "sakhiya_news_v2_candidate" },
      { blind: "Sample B (-16 LUFS)", real: "Sakhiya-News Leg-1 Preview [100-Mel 4-Step Baseline · 9.914 MB INT8]", metrics: "Leg-1 Preview Baseline: HF_LSD_4_12kHz=1.399dB · sim_Anchor=0.8862 · ne_pcer=0.0542 (+0.0222 gap) · RTF=0.0599", tag: "sakhiya_news_leg1_preview" },
      { blind: "Sample C (-16 LUFS)", real: "Sakhiya-One Studio [Live Public Default · a61c100 · 80-Mel 8-blk · 8.464 MB INT8]", metrics: "Live Public Default: HF_LSD_4_12kHz=1.917dB · sim_Anchor=0.8443 · ne_pcer=0.0590 · RTF=0.0492", tag: "sakhiya_one" },
      { blind: "Sample D (-16 LUFS)", real: "Broken Leg-2 (425bb6f · Reverted) [2-Step · 97.9% INT8 · Metallic Buzz & Trailing Schwa]", metrics: "REVERTED ❌ · Stationary sinusoid A_noise buzz, 0.780 flatness overshoot, trailing schwa, मामकाः screech", tag: "sakhiya_news" }
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
