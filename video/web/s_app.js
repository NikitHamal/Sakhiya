// Material 3 Expressive Scenes: edu (Nepal Education Sector Weekly Recap) & outro (Friday Weekly Sign-Off)

SC.edu = (c, t, sc, th, T) => {
  const v = V(T);
  dotgrid(c, th, T, .6);
  const ids = ['e1', 'e2', 'e3', 'e4', 'e5', 'e6'];
  const L = ids.map(k => lt(sc, k));
  const k = L.reduce((a, s, i) => t >= s - .15 ? i : a, 0);

  // Left Column: Nepal Education Sector Weekly Index Board
  bento(c, 68, 106, 760, 848, 44, th, { wmShape: 'flower', wmR: 185, kicker: 'NEPAL EDUCATION DESK · नेपाल शैक्षिक विशेष' });
  words(c, 'Nepal Education', 108, 205, t - .15, { sz: 64, w: 700, col: th.fg });
  txt(c, 'यो साताका प्रमुख शैक्षिक र विश्वविद्यालय खबर (२१–२७ नोभेम्बर)', 108, 255, { sz: 24, w: 600, col: th.acc });

  const topics = [
    ['Tribhuvan University (TU)', 'सम्बन्धन कलेज स्वघोषणा र सेमेस्टर सुधार', 'grad', k <= 1, '24 NOV'],
    ['Medical Education (MEC)', 'स्नातक नतिजा प्रकाशन र सरकारी वार्ता', 'doc', k === 2 || k === 3, '22–24 NOV'],
    ['Kathmandu Metro (KMC)', 'अनामनगरमा अटिजम विद्यालय तथा अनुसन्धान केन्द्र', 'home', k === 4, '25 NOV'],
    ['Hackathon, KU & R&D', 'सुदूरपश्चिम ह्याकाथन, केयु तालिका र अनुसन्धान', 'spark', k === 5, '21–26 NOV']
  ];

  topics.forEach(([title, ne, ic, on, dt], i) => {
    const p = sp(t, 0.25 + i * 0.16, 0.68);
    const y = 298 + i * 150 + (1 - p) * 32;
    c.save();
    c.globalAlpha = clamp(p * 1.3);
    box(c, 104, y, 688, 132, 30, on ? th.accc : th.soft, on ? th.acc : th.line, on ? 3 : 1.5);
    shape(c, on ? th.acc : th.card, 162, y + 66, 40, seq8[i * 2], 'sunny', on ? 0.6 : 0, T * 0.3);
    icon(c, ic, 162, y + 66, 32, on ? C.white : th.mu, 2.4);
    chip(c, dt, 628, y + 22, { sz: 15, fill: C.white, col: th.acc, h: 32, pad: 24 });
    txt(c, title, 222, y + 56, { sz: 30, w: 700, col: on ? th.onacc : th.fg });
    txt(c, ne, 222, y + 98, { sz: 21, w: 600, col: on ? th.acc : th.mu });
    c.restore();
  });

  // Right Column: Dynamic 992 x 848 Interactive Education Showcase
  const rx = 860, ry = 106, rw = 992, rh = 848;
  bento(c, rx, ry, rw, rh, 44, th, { fill: th.card, stroke: th.line, wmShape: 'clover', wmR: 220 });
  c.save();
  rr(c, rx, ry, rw, rh, 44);
  c.clip();
  c.translate(rx, ry);

  // Stage A (k === 0 or 1): Tribhuvan University Affiliated Colleges Self-Declaration & Semester Reform
  if (k <= 1) {
    const dt = t - L[0];
    chip(c, 'TRIBHUVAN UNIVERSITY · त्रिभुवन विश्वविद्यालय कार्यकारी परिषद्', 48, 42, { sz: 19, fill: th.accc, col: th.onacc, ic: 'grad' });
    words(c, 'Mandatory College Audit', 48, 155, dt, { sz: 62, w: 700, col: th.fg });
    txt(c, 'सम्बन्धन प्राप्त कलेजहरूलाई स्वघोषणा विवरण बुझाउन निर्देशन (२४ नोभेम्बर २०२६)', 48, 208, { sz: 25, w: 600, col: th.acc });

    const checks = [
      ['Science & IT Laboratories', 'प्रयोगशाला र प्रयोगात्मक सामग्री', 'Verified Lab Standards', C.aquac, C.aquad],
      ['Central & Digital Libraries', 'पुस्तकालय र शैक्षिक स्रोत', 'Minimum Book & E-Journal Quota', C.pc, C.blue],
      ['Academic Infrastructure', 'भौतिक पूर्वाधार र कक्षाकोठा', 'Quality Over Quantity Policy', C.amberc, C.amberd],
      ['Semester System Expansion', 'सेमेस्टर प्रणाली विस्तार तयारी', 'Curriculum & Exam Streamlining', C.violetc, C.violet]
    ];
    checks.forEach(([enT, neT, sub, bg, fg], i) => {
      const p = sp(t, L[0] + 0.35 + i * 0.2, 0.68);
      const col = i % 2, row = Math.floor(i / 2);
      const cx = 48 + col * 456, cy = 256 + row * 265 + (1 - p) * 30;
      c.save();
      c.globalAlpha = clamp(p);
      box(c, cx, cy, 436, 240, 32, bg, fg, k === 1 && i === 3 ? 3 : 1.5);
      icon(c, 'check', cx + 44, cy + 52, 32, fg, 2.8);
      txt(c, enT, cx + 82, cy + 60, { sz: 26, w: 700, col: C.ink });
      txt(c, neT, cx + 36, cy + 125, { sz: 24, w: 600, col: fg });
      txt(c, sub, cx + 36, cy + 175, { sz: 20, w: 500, col: C.mute });
      c.restore();
    });
  }

  // Stage B (k === 2 or 3): Medical Education Commission (MECEE-BL Results & Govt Reform Talks)
  if (k === 2 || k === 3) {
    const dt = t - L[2];
    chip(c, 'MEDICAL EDUCATION COMMISSION · चिकित्सा शिक्षा आयोग', 48, 42, { sz: 19, fill: C.aquac, col: C.aquad, ic: 'doc' });
    words(c, 'MECEE-BL Results & Reform Talks', 48, 150, dt, { sz: 56, w: 700, col: th.fg });
    txt(c, 'स्नातक तहको एकीकृत प्रवेश परीक्षा नतिजा र सरकार–सरोकारवाला संवाद', 48, 204, { sz: 25, w: 600, col: C.aquad });

    // Published Bachelor Entrance Programs Grid
    const progs = ['BSc Nursing', 'B Pharmacy', 'BPT Physiotherapy', 'BPH Public Health', 'BSc MLT / MIT', 'BAMS & Optometry'];
    progs.forEach((pr, i) => {
      const p = sp(t, L[2] + 0.2 + i * 0.1, 0.6);
      const col = i % 3, row = Math.floor(i / 3);
      const px = 48 + col * 304, py = 248 + row * 136;
      c.save();
      c.globalAlpha = clamp(p);
      box(c, px, py, 286, 118, 26, C.aquac, C.aquad, 1.8);
      icon(c, 'check', px + 36, py + 44, 24, C.aquad, 2.6);
      txt(c, 'RESULT PUBLISHED', px + 60, py + 50, { sz: 15, w: 700, col: C.aquad, ls: 1 });
      txt(c, pr, px + 30, py + 92, { sz: 24, w: 700, col: C.ink });
      c.restore();
    });

    // Bottom Government Dialogue Banner (e4)
    const p4 = sp(t, L[3] - 0.1, 0.7);
    box(c, 48, 545, 896, 255, 34, k === 3 ? th.accc : th.soft, th.acc, k === 3 ? 3 : 1.5);
    txt(c, '24 NOV 2026  ·  GOVERNMENT OF NEPAL STAKEHOLDER DIALOGUE', 82, 595, { sz: 18, w: 700, col: th.acc, ls: 1.5 });
    txt(c, 'चिकित्सा शिक्षा सुधार र विगतका सहमति कार्यान्वयनका लागि औपचारिक वार्ता', 82, 650, { sz: 30, w: 700, col: th.fg });
    txt(c, 'Addresses seat quotas, fee structures, and pending medical education agreements.', 82, 705, { sz: 23, w: 500, col: th.mu });
    if (p4 > 0.05) chip(c, 'ACTIVE POLICY REFORM', 82, 735, { sz: 18, fill: th.acc, col: C.white, ic: 'shield', a: p4 });
  }

  // Stage C (k === 4): Kathmandu Metropolitan City (KMC) Autism School & Research Centre
  if (k === 4) {
    const dt = t - L[4];
    chip(c, 'KATHMANDU METROPOLITAN CITY · काठमाडौँ महानगरपालिका', 48, 42, { sz: 19, fill: C.amberc, col: C.amberd, ic: 'home' });
    words(c, 'Autism School & Research Centre', 48, 152, dt, { sz: 56, w: 700, col: th.fg });
    txt(c, 'काठमाडौँ महानगरद्वारा अनामनगर (वडा नं. २९) मा विशेष विद्यालय शिलान्यास', 48, 208, { sz: 26, w: 600, col: th.acc });

    box(c, 48, 255, 896, 540, 38, th.soft, th.line, 2);
    const pillars = [
      ['Special Autism School', 'अटिजम भएका बालबालिकाका लागि विशेष कक्षा र थेरापी', 'book', C.pc, C.blue],
      ['Dedicated Research Centre', 'प्रारम्भिक पहिचान, परामर्श र शैक्षिक अनुसन्धान केन्द्र', 'search', C.aquac, C.aquad],
      ['Anamnagar Ward 29 Campus', 'मंसिर ९ (२५ नोभेम्बर) मा औपचारिक शिलान्यास सम्पन्न', 'home', C.amberc, C.amberd]
    ];
    pillars.forEach(([ti, ne, ic, bg, fg], i) => {
      const p = sp(t, L[4] + 0.25 + i * 0.2, 0.65);
      const py = 290 + i * 160;
      c.save();
      c.globalAlpha = clamp(p);
      box(c, 82, py, 828, 138, 28, C.white, fg, 2);
      box(c, 110, py + 25, 88, 88, 24, bg, null);
      icon(c, ic, 154, py + 69, 44, fg, 2.4);
      txt(c, ti, 224, py + 62, { sz: 32, w: 700, col: C.ink });
      txt(c, ne, 224, py + 104, { sz: 24, w: 600, col: fg });
      c.restore();
    });
  }

  // Stage D (k === 5): Sudurpaschim University Hackathon, KU Geomatics & National R&D Priority
  if (k >= 5) {
    const dt = t - L[5];
    chip(c, 'UNIVERSITIES & INNOVATION · विश्वविद्यालय र अनुसन्धान', 48, 42, { sz: 19, fill: C.pc, col: C.blue, ic: 'spark' });
    words(c, 'Hackathon, KU & National R&D', 48, 150, dt, { sz: 58, w: 700, col: th.fg });

    const cards = [
      ['Sudurpaschim University', 'National Hackathon in Mahendranagar', 'चार विश्वविद्यालयका दश टोलीबीच राष्ट्रिय ह्याकाथन प्रतिस्पर्धा (२१ नोभेम्बर)', 'code', C.aquac, C.aquad],
      ['Kathmandu University (KU)', 'Geomatics Master’s Entrance Schedule', 'जियोइन्फर्मेटिक्स र ल्यान्ड एडमिनिस्ट्रेसन स्नातकोत्तर प्रवेश परीक्षा तालिका (२६ नोभेम्बर)', 'compass', C.pc, C.blue],
      ['Ministry of Science & Tech', 'Priority Investment in Research & Innovation', 'अनुसन्धान र नवप्रवर्तनमा सरकारी लगानी बढाउने शिक्षा तथा विज्ञान मन्त्रालयको प्राथमिकता', 'bolt', C.amberc, C.amberd]
    ];
    cards.forEach(([uni, en, ne, ic, bg, fg], i) => {
      const p = sp(t, L[5] + 0.15 + i * 0.2, 0.68);
      const py = 210 + i * 198;
      c.save();
      c.globalAlpha = clamp(p);
      box(c, 48, py, 896, 176, 32, bg, fg, 2);
      box(c, 78, py + 32, 112, 112, 28, C.white, null);
      icon(c, ic, 134, py + 88, 52, fg, 2.5);
      txt(c, uni, 218, py + 58, { sz: 22, w: 700, col: fg, ls: 1 });
      txt(c, en, 218, py + 98, { sz: 32, w: 700, col: C.ink });
      txt(c, ne, 218, py + 140, { sz: 22, w: 600, col: fg });
      c.restore();
    });
  }

  c.restore();
};

SC.outro = (c, t, sc, th, T) => {
  const v = V(T);
  const o1 = lt(sc, 'o1'), o2 = lt(sc, 'o2'), o3 = lt(sc, 'o3');
  dotgrid(c, th, T, 0.9);
  const a = sp(t, 0, .85);

  // Left Host Sign-Off Pod
  bento(c, 96, 126, 620, 790, 48, th, { fill: C.blueCont, stroke: '#3570e6', a: clamp(a), wmShape: 'cookie12', wmR: 190 });
  sakhiya(c, th, 406, 430, 185 * a, T, v, {
    fill: C.white,
    ring: C.aqua,
    ring2: C.white,
    glyphCol: C.blue,
    a: clamp(a),
    badge: 'SEE YOU NEXT FRIDAY'
  });
  txt(c, 'अर्को शुक्रबार फेरि भेटौँला · नमस्ते', 406, 780, { sz: 28, w: 600, col: C.white, al: 'center', a: clamp(a) });
  txt(c, 'github.com/NikitHamal/Sakhiya', 406, 830, { sz: 22, w: 500, col: C.aqua, al: 'center', a: clamp(a) });

  // Right Sign-Off & Production Credits Bento
  bento(c, 756, 126, 1068, 790, 48, th, { fill: C.blueDark, stroke: '#3570e6', a: clamp(a), wmShape: 'sunny', wmR: 230, kicker: 'NEBIANS NEWS  ·  WEEKLY EDITION 01  ·  27 NOV 2026' });
  words(c, 'धन्यवाद र नमस्ते', 820, 315, t - o1 + .15, { sz: 118, w: 700, col: C.white });
  txt(c, 'Thank you for watching our Friday Weekly Recap', 824, 385, { sz: 34, w: 500, col: '#c7d8ff', a: P(t, o1 + .3, o1 + .8) });

  const credits = [
    ['VOICE MODEL', 'Sakhiya-One v2 Studio (23 MB INT8 ONNX · 24 kHz Vocos)', 'सखिया वान दोस्रो संस्करण (हालको चरण)'],
    ['WEEKLY CADENCE', 'Every Friday · Strictly Saturday–Friday Verified Recap', 'हरेक शुक्रबार प्रविधि, एआई र नेपाल शिक्षा संकलन'],
    ['DESIGN SYSTEM', 'Material 3 Expressive · 1080p30 Canvas + Playwright + FFmpeg', 'मटेरियल थ्री एक्सप्रेसिभ डाइनामिक मोसन ग्राफिक्स']
  ];
  credits.forEach(([kicker, en, ne], i) => {
    const q = sp(t, Math.max(0.4, o2 - 0.2 + i * 0.22), 0.72);
    const cy = 435 + i * 142 + (1 - q) * 30;
    c.save();
    c.globalAlpha = clamp(q);
    box(c, 820, cy, 940, 122, 28, C.blueCont, '#3570e6', 1.5);
    txt(c, kicker, 852, cy + 36, { sz: 16, w: 700, col: C.aqua, ls: 2 });
    txt(c, en, 852, cy + 72, { sz: 26, w: 700, col: C.white });
    txt(c, ne, 852, cy + 104, { sz: 21, w: 500, col: '#c7d8ff' });
    c.restore();
  });
};
