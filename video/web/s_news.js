// Material 3 Expressive News Scenes: n1..n5 (Global AI & Tech Weekly Recap) + fast (Quick Tech Roundup)

function activeBeat(sc, t) {
  let idx = 0;
  sc.lines.forEach((l, i) => {
    if (t >= (l.t - sc.t0) - 0.15) idx = i;
  });
  return idx;
}

// Upgraded M3 Expressive Split Story Layout: Left Narrative & Beat Tracker + Right Dynamic Multi-Stage Bento Canvas
function story(c, t, sc, th, T, o) {
  const v = V(T);
  dotgrid(c, th, T, .55);
  const p = sp(t, .1, .75);
  const beat = activeBeat(sc, t);

  // Left Column: Story Badge, Date Stamp, Headline, and 4-Beat Progress Cards
  c.save();
  c.globalAlpha = clamp(p * 1.3);
  bento(c, 68, 106, 760, 848, 44, th, { wmShape: o.shape || 'cookie12', wmR: 190 });

  // Number & Date Stamp Header
  shape(c, th.accc, 146, 186, 52 * p, o.shape || 'cookie12', 'sunny', Math.sin(T) * .5 + .5, T * 0.25);
  txt(c, o.num, 146, 202, { sz: 42, w: 700, col: th.onacc, al: 'center' });
  chip(c, o.dateTag, 218, 152, { sz: 18, fill: th.soft, col: th.acc, ic: 'calendar', h: 36 });
  chip(c, o.catTag, 218, 196, { sz: 17, fill: th.accc, col: th.onacc, ic: o.ic || 'bolt', h: 34 });

  // Main Story Title & Nepali Subtitle (guaranteed 2-line fit with dynamic bottom offset)
  const titleBot = words(c, o.title, 108, 298, t - .15, { sz: 50, w: 700, col: th.fg, maxW: 680, lh: 58 });
  txt(c, o.neTitle, 108, Math.min(418, titleBot + 44), { sz: 26, w: 600, col: th.acc });

  // 4-Beat Live Context Tracker (highlights active spoken sentence beat)
  o.beats.forEach(([bTitle, bSub], i) => {
    const isCur = beat === i;
    const isDone = beat > i;
    const bp = sp(t, 0.3 + i * 0.14, 0.65);
    const by = 452 + i * 102 + (1 - bp) * 24;
    c.save();
    c.globalAlpha = clamp(bp);
    box(c, 104, by, 688, 88, 24, isCur ? th.accc : th.soft, isCur ? th.acc : th.line, isCur ? 2.5 : 1.2);
    box(c, 124, by + 18, 52, 52, 18, isCur ? th.acc : (isDone ? th.sec : th.card), null);
    if (isDone) {
      icon(c, 'check', 150, by + 44, 24, C.white, 2.5);
    } else {
      txt(c, `0${i + 1}`, 150, by + 53, { sz: 22, w: 700, col: isCur ? th.bg : th.mu, al: 'center' });
    }
    txt(c, bTitle, 194, by + 40, { sz: 24, w: 700, col: isCur ? th.onacc : th.fg });
    txt(c, bSub, 194, by + 69, { sz: 19, w: 500, col: isCur ? th.acc : th.mu });
    c.restore();
  });

  // Verified Source Footer Chip
  chip(c, 'SOURCE · ' + o.src, 104, 878, { sz: 18, fill: th.soft, col: th.mu, line: th.line, ic: 'globe', h: 42 });
  c.restore();

  // Right Column: Dynamic 996 x 848 Interactive Bento Canvas
  c.save();
  c.translate(860, 106);
  const s = .94 + .06 * p;
  c.globalAlpha = clamp(p * 1.4);
  c.translate(498 * (1 - s), 424 * (1 - s));
  c.scale(s, s);
  bento(c, 0, 0, 992, 848, 44, th, { fill: th.card, stroke: th.line });
  c.save();
  rr(c, 0, 0, 992, 848, 44);
  c.clip();
  o.draw(c, t, beat, v, th, T);
  c.restore();
  c.restore();
}

// STORY 01: SpaceX Starmind — 1 Million Orbital AI Data Center Satellites (Nov 25–26, 2026)
SC.n1 = (c, t, sc, th, T) => story(c, t, sc, th, T, {
  num: '01',
  dateTag: '25–26 NOV 2026',
  catTag: 'SPACE & AI CLOUD',
  ic: 'sat',
  shape: 'sunny',
  title: 'SpaceX Starmind: 1M Orbital AI Satellites',
  neTitle: 'अन्तरिक्षमा दस लाख एआई डाटा सेन्टर भूउपग्रह',
  src: 'CNBC · The Verge · FCC Filing (25–26 Nov 2026)',
  beats: [
    ['FCC Approves 1,000,000 Satellites', 'एफसीसीद्वारा दस लाख भूउपग्रह पठाउन अनुमति'],
    ['Solar-Powered Orbital AI Compute', 'अन्तरिक्षमै सौर्य ऊर्जाबाट एआई चिप सञ्चालन'],
    ['Starlink Optical Laser Downlink', 'पृथ्वीमा बिजुली अभाव टार्न लेजर नेटवर्क'],
    ['Orbital Debris & Astronomy Debate', 'खगोलविद् र प्रतिस्पर्धीहरूको वातावरणीय चिन्ता']
  ],
  draw: (c, t, beat, v, th, T) => {
    // Top Orbital Visualization Card
    box(c, 36, 36, 920, 470, 34, '#07142b', '#1e3a6e', 2);
    txt(c, 'STARMIND ORBITAL CONSTELLATION  ·  SOLAR + LASER MESH', 68, 82, { sz: 19, w: 700, col: C.aqua, ls: 2 });

    // Earth Curve & Atmosphere
    const cx = 496, cy = 580;
    c.beginPath(); c.arc(cx, cy, 310, Math.PI * 1.12, Math.PI * 1.88);
    c.fillStyle = '#0c3b87'; c.fill();
    c.lineWidth = 6; c.strokeStyle = C.sky; c.stroke();
    txt(c, 'EARTH · STARLINK GROUND GATEWAYS', cx, 475, { sz: 18, w: 700, col: '#9ec5ff', al: 'center', ls: 1.5 });

    // Orbiting AI Satellites with Solar Wings & Laser Links
    const nSats = 7;
    for (let i = 0; i < nSats; i++) {
      const ang = Math.PI * 1.2 + (i / (nSats - 1)) * Math.PI * 0.6 + Math.sin(T * 0.6 + i) * 0.02;
      const rad = 405 + (i % 2) * 34;
      const sx = cx + Math.cos(ang) * rad, sy = cy + Math.sin(ang) * rad;
      // Laser beam to Earth & neighbor
      if (beat >= 1) {
        c.beginPath(); c.moveTo(sx, sy); c.lineTo(cx + Math.cos(ang) * 305, cy + Math.sin(ang) * 305);
        c.strokeStyle = beat === 3 ? C.amber : C.aqua; c.lineWidth = 2; c.setLineDash([6, 6]); c.lineDashOffset = -T * 30; c.stroke(); c.setLineDash([]);
      }
      shape(c, i % 2 === 0 ? C.amber : C.aqua, sx, sy, 18, 'squircle', 'gem', 0.5, T * 0.5);
      box(c, sx - 32, sy - 5, 12, 10, 3, C.sky);
      box(c, sx + 20, sy - 5, 12, 10, 3, C.sky);
    }

    // Animated Counter Overlay inside Space Card
    const countP = eo3(P(t, 0.2, 2.4));
    const numStr = Math.round(1000000 * countP).toLocaleString('en-US');
    box(c, 68, 110, 410, 125, 24, 'rgba(13,34,74,0.88)', '#2b549e');
    txt(c, numStr, 92, 182, { sz: 62, w: 700, col: C.white });
    txt(c, 'FCC Authorized Orbital AI Nodes (2033–2040)', 92, 216, { sz: 18, w: 500, col: '#9ec5ff' });

    // Bottom 3 Metric Bento Sub-Cards
    const cards = [
      ['100% Solar', 'Zero Earth Grid Load', 'सौर्य ऊर्जाबाट सञ्चालन', C.aquac, C.aquad, beat === 1],
      ['Laser Mesh', 'Starlink Optical Link', 'लेजर डाटा आदानप्रदान', C.pc, C.blue, beat === 2],
      ['Debris Alert', 'Astronomers Raise Concern', 'अन्तरिक्ष फोहोरको चिन्ता', C.errc, C.err, beat === 3]
    ];
    cards.forEach(([big, s1, s2, bg, fg, hi], i) => {
      const bx = 36 + i * 314, by = 536;
      box(c, bx, by, 292, 276, 32, hi ? bg : th.soft, hi ? fg : th.line, hi ? 3 : 1.5);
      txt(c, big, bx + 24, by + 92, { sz: 40, w: 700, col: hi ? fg : th.fg });
      txt(c, s1, bx + 24, by + 146, { sz: 20, w: 700, col: hi ? C.ink : th.fg });
      txt(c, s2, bx + 24, by + 186, { sz: 20, w: 600, col: hi ? fg : th.mu });
      chip(c, hi ? 'ACTIVE FOCUS' : `STEP 0${i + 2}`, bx + 24, by + 216, { sz: 16, fill: C.white, col: fg, h: 36 });
    });
  }
});

// STORY 02: Nvidia Reinstates Jensen Huang, Stock Swing, Export Curbs & Huawei HBM (Nov 23–27, 2026)
SC.n2 = (c, t, sc, th, T) => story(c, t, sc, th, T, {
  num: '02',
  dateTag: '23–27 NOV 2026',
  catTag: 'CHIPS & SEMICONDUCTORS',
  ic: 'chip',
  shape: 'cookie12',
  title: 'Nvidia Reinstates Huang & Tightens Exports',
  neTitle: 'जेन्सेन हुआङको पुनरागमन र चिप निर्यातमा कडाइ',
  src: 'CNBC · Reuters · Bloomberg (23–27 Nov 2026)',
  beats: [
    ['Jensen Huang Reinstated as CEO', 'सोमबार पुनः प्रमुख कार्यकारी पदमा फिर्ती'],
    ['Stock Surges +30%, Then Dips -6%', 'सेयर मूल्य ३०% बढेर मंगलबार ६% घट्यो'],
    ['Server Halts to 4 SE Asian Nations', 'मलेसिया, सिङ्गापुर, थाइल्यान्ड र इन्डोनेसियामा रोक'],
    ['Huawei Teases Next-Gen HBM Chip', 'हुवावेद्वारा नयाँ हाई ब्यान्डविथ मेमोरी चिपको तयारी']
  ],
  draw: (c, t, beat, v, th, T) => {
    // Top-Left: Stock Chart Card (Mon +30% Surge -> Tue -6% Dip)
    box(c, 36, 36, 560, 440, 34, th.soft, beat <= 1 ? th.acc : th.line, beat <= 1 ? 2.8 : 1.5);
    txt(c, 'NVIDIA (NVDA) WEEKLY VOLATILITY  ·  MON 23 – TUE 24 NOV', 64, 78, { sz: 17, w: 700, col: th.mu, ls: 1.5 });
    const pts = [0.15, 0.18, 0.42, 0.78, 0.92, 0.88, 0.74, 0.68, 0.70];
    const pChart = eo3(P(t, 0.4, 3.5));
    for (let g = 0; g < 4; g++) {
      c.beginPath(); c.moveTo(64, 130 + g * 80); c.lineTo(564, 130 + g * 80);
      c.strokeStyle = th.line; c.lineWidth = 1; c.stroke();
    }
    c.beginPath();
    const maxI = (pts.length - 1) * pChart;
    for (let i = 0; i <= Math.floor(maxI); i++) {
      const px = 64 + (500 * i) / (pts.length - 1), py = 400 - pts[i] * 250;
      if (i === 0) c.moveTo(px, py); else c.lineTo(px, py);
    }
    c.strokeStyle = C.aqua; c.lineWidth = 7; c.lineCap = 'round'; c.lineJoin = 'round'; c.stroke();
    chip(c, 'MON: +30% CEO RETURN', 74, 110, { sz: 18, fill: C.aquac, col: C.aquad, ic: 'chart' });
    if (beat >= 2) chip(c, 'TUE: -6% NO SUCCESSION PLAN', 210, 395, { sz: 18, fill: C.errc, col: C.err, ic: 'warn' });

    // Top-Right: CEO Governance Card
    box(c, 616, 36, 340, 440, 34, th.accc, th.acc, 2);
    shape(c, th.acc, 786, 155, 68, 'squircle', 'cookie12', 0.4, T * 0.25);
    icon(c, 'chip', 786, 155, 52, th.bg, 2.5);
    txt(c, 'Jensen Huang', 786, 268, { sz: 34, w: 700, col: th.onacc, al: 'center' });
    txt(c, 'Reinstated CEO & Director', 786, 305, { sz: 20, w: 600, col: th.acc, al: 'center' });
    box(c, 644, 336, 284, 110, 22, th.card, null);
    txt(c, 'Moody’s Outlook:', 668, 376, { sz: 19, w: 600, col: th.mu });
    txt(c, 'Positive → Stable', 668, 414, { sz: 24, w: 700, col: C.amber });

    // Bottom-Left: Southeast Asia Server Shipment Halt (4 Countries)
    box(c, 36, 500, 450, 312, 34, beat === 2 ? C.amberc : th.soft, beat === 2 ? C.amber : th.line, beat === 2 ? 2.8 : 1.5);
    txt(c, 'US EXPORT CURBS · SERVER SHIPMENT HALT', 64, 544, { sz: 17, w: 700, col: beat === 2 ? C.amberd : th.mu, ls: 1.2 });
    ['Malaysia', 'Singapore', 'Thailand', 'Indonesia'].forEach((cnt, idx) => {
      const cx = 64 + (idx % 2) * 204, cy = 572 + Math.floor(idx / 2) * 104;
      box(c, cx, cy, 188, 86, 20, th.card, th.line);
      icon(c, 'pause', cx + 32, cy + 43, 24, C.amber, 2.5);
      txt(c, cnt, cx + 56, cy + 51, { sz: 22, w: 700, col: th.fg });
    });

    // Bottom-Right: Huawei Next-Gen HBM Memory Rivalry
    box(c, 506, 500, 450, 312, 34, beat === 3 ? C.aquac : th.soft, beat === 3 ? C.aquad : th.line, beat === 3 ? 2.8 : 1.5);
    txt(c, 'DOMESTIC AI MEMORY · HUAWEI HBM CHIP', 534, 544, { sz: 17, w: 700, col: beat === 3 ? C.aquad : th.mu, ls: 1.2 });
    // 3D Stacked HBM Dies Diagram
    for (let s = 0; s < 4; s++) {
      box(c, 536, 710 - s * 36, 170, 28, 8, s === 3 ? C.aquad : C.blue, C.white, 1.5);
      txt(c, `HBM DIE ${s + 1}`, 621, 730 - s * 36, { sz: 14, w: 700, col: C.white, al: 'center' });
    }
    txt(c, 'Huawei HBM', 734, 615, { sz: 32, w: 700, col: beat === 3 ? C.aquad : th.fg });
    txt(c, 'Challenges Samsung', 734, 655, { sz: 22, w: 600, col: beat === 3 ? C.ink : th.mu });
    txt(c, '& SK Hynix in AI RAM', 734, 688, { sz: 22, w: 600, col: beat === 3 ? C.ink : th.mu });
  }
});

// STORY 03: OpenAI GPT-6.2 Sol & Private Inference + Anthropic Claude Ship & Rust API (Nov 23–26, 2026)
SC.n3 = (c, t, sc, th, T) => story(c, t, sc, th, T, {
  num: '03',
  dateTag: '23–26 NOV 2026',
  catTag: 'AI MODELS & RUST',
  ic: 'code',
  shape: 'flower',
  title: 'GPT-6.2 Sol, Claude Ship & 70% Rust Gain',
  neTitle: 'जिपिटी ६.२ सोल, क्लाउड सिप र रस्ट माइग्रेसन',
  src: 'The Decoder · SiliconReport · Anthropic (23–26 Nov 2026)',
  beats: [
    ['GPT-6.2 Sol in Main ChatGPT Tab', 'च्याटजिपिटीको मुख्य ट्याबमा जिपिटी ६.२ सोल'],
    ['OpenAI Private Inference & Proteins', 'गोप्य डाटा प्रोसेसिङ र प्रोटिन डिजाइन क्षमता'],
    ['Anthropic Claude Ship PR Agent', 'कोड परीक्षण र पुल रिक्वेस्ट स्वीकृत गर्ने टुल'],
    ['API Migrated Python → Rust (-70% CPU)', 'पाइथनबाट रस्टमा सार्दा ७०% सिपियु बचत']
  ],
  draw: (c, t, beat, v, th, T) => {
    // Top Half: OpenAI Ecosystem (GPT-6.2 Sol + Private Inference + Protein Design)
    box(c, 36, 36, 920, 372, 34, beat <= 1 ? th.accc : th.soft, beat <= 1 ? th.acc : th.line, beat <= 1 ? 2.5 : 1.5);
    txt(c, 'OPENAI RELEASES  ·  NOV 23–26, 2026', 68, 78, { sz: 18, w: 700, col: th.acc, ls: 1.8 });
    const oCards = [
      ['GPT-6.2 Sol', 'Live in ChatGPT Tab', 'मुख्य च्याट ट्याबमा उपलब्ध', 'spark', C.blue, C.pc],
      ['Private Cloud', 'Encrypted AI Enclave', 'गोप्य डाटा प्रोसेसिङ', 'lock', C.aquad, C.aquac],
      ['GPT-6 Bio', 'AI Protein Folding', 'प्रोटिन संरचना डिजाइन', 'globe', C.violet, C.violetc]
    ];
    oCards.forEach(([ti, s1, s2, ic, col, bg], i) => {
      const cx = 64 + i * 294, cy = 104;
      box(c, cx, cy, 276, 274, 28, C.white, th.line, 1.5);
      box(c, cx + 24, cy + 24, 58, 58, 18, bg, null);
      icon(c, ic, cx + 53, cy + 53, 30, col, 2.3);
      txt(c, ti, cx + 24, cy + 132, { sz: 28, w: 700, col: C.ink });
      txt(c, s1, cx + 24, cy + 172, { sz: 19, w: 600, col: C.mute });
      txt(c, s2, cx + 24, cy + 212, { sz: 19, w: 600, col: col });
    });

    // Bottom Half: Anthropic Claude Ship + Python-to-Rust CPU Reduction (-70%)
    box(c, 36, 436, 920, 376, 34, beat >= 2 ? C.aquac : th.soft, beat >= 2 ? C.aquad : th.line, beat >= 2 ? 2.5 : 1.5);
    txt(c, 'ANTHROPIC ENGINEERING  ·  CLAUDE SHIP & PYTHON → RUST MIGRATION', 68, 478, { sz: 18, w: 700, col: C.aquad, ls: 1.8 });

    // Left Sub-Box: Claude Ship Autonomous PR Pipeline
    box(c, 64, 504, 420, 278, 28, C.white, th.line);
    txt(c, 'Claude Ship Agent', 92, 552, { sz: 30, w: 700, col: C.ink });
    ['Review Code', 'Fix Bugs', 'Approve PR'].forEach((st, idx) => {
      const sy = 578 + idx * 62;
      box(c, 92, sy, 364, 48, 16, C.aquac, null);
      icon(c, 'check', 120, sy + 24, 22, C.aquad, 2.5);
      txt(c, st, 152, sy + 32, { sz: 22, w: 700, col: C.aquad });
    });

    // Right Sub-Box: Python vs Rust CPU Load Gauge (-70% CPU)
    box(c, 508, 504, 420, 278, 28, C.white, th.line);
    txt(c, 'API Rebuilt in Rust', 536, 552, { sz: 30, w: 700, col: C.ink });
    txt(c, 'Python CPU Load (100%)', 536, 598, { sz: 19, w: 600, col: C.mute });
    box(c, 536, 610, 364, 26, 13, C.errc, null);
    box(c, 536, 610, 364, 26, 13, C.err, null);
    txt(c, 'Rust CPU Load (-70% Reduction!)', 536, 676, { sz: 19, w: 700, col: C.aquad });
    box(c, 536, 688, 364, 26, 13, C.aquac, null);
    box(c, 536, 688, 364 * 0.30, 26, 13, C.aquad, null);
    txt(c, '700M+ Monthly MCP SDK Downloads', 536, 755, { sz: 20, w: 700, col: C.blue });
  }
});

// STORY 04: Cybersecurity, Linux Shirudo, 3.5B Leak & NY 5-Year AI Toy Ban (Nov 23–26, 2026)
SC.n4 = (c, t, sc, th, T) => story(c, t, sc, th, T, {
  num: '04',
  dateTag: '23–26 NOV 2026',
  catTag: 'CYBERSECURITY & LAW',
  ic: 'shield',
  shape: 'shieldShape',
  title: 'Shirudo Linux Defense, 3.5B Leak & NY Toy Ban',
  neTitle: 'लिनक्स सुरक्षा सिरुडो, डाटा चुहावट र एआई खेलौना प्रतिबन्ध',
  src: 'Wired · The Hacker News · TechCrunch (23–26 Nov 2026)',
  beats: [
    ['OpenAI & Cisco Launch "Shirudo"', 'चलिरहेका लिनक्स सर्भर जोगाउन सिरुडो प्रणाली'],
    ['3.5B Records Exposed & FinCEN Offline', 'साढे तीन अर्ब पासवर्ड चुहावट र फिनसेन बन्द'],
    ['New York Bans AI Chatbot Toys (5 Yrs)', 'न्युयोर्कमा एआई खेलौनामा ५ वर्षको प्रतिबन्ध'],
    ['42 States Block Character.AI Chat Sale', '४२ राज्यद्वारा २.१ करोड निजी च्याट लिलामीमा रोक']
  ],
  draw: (c, t, beat, v, th, T) => {
    const quads = [
      ['01 · LINUX & CLOUD DEFENSE', 'OpenAI + Cisco Shirudo', 'लिनक्स सर्भर र एडब्लुएस सुरक्षा', 'Real-time Linux protection + AWS Nitro isolation', 'shield', C.aquac, C.aquad, beat === 0],
      ['02 · DATA BREACHES', '3.5 Billion Records', 'साढे तीन अर्ब विवरण बाहिरियो', 'Two exposed DBs + US Treasury FinCEN intrusion', 'warn', C.errc, C.err, beat === 1],
      ['03 · CHILD SAFETY LAW', 'NY 5-Year AI Toy Ban', 'एआई खेलौनामा ५ वर्ष प्रतिबन्ध', 'Signed by Gov. Hochul · $15,000/day penalty', 'lock', C.amberc, C.amberd, beat === 2],
      ['04 · PRIVACY COALITION', '42 State AGs Object', 'निजी च्याट लिलामी रोक्न निवेदन', 'Blocks Character.AI auction of 21M user chats', 'doc', C.pc, C.blue, beat === 3]
    ];
    quads.forEach(([kicker, big, ne, sub, ic, bg, fg, hi], i) => {
      const col = i % 2, row = Math.floor(i / 2);
      const qx = 36 + col * 468, qy = 36 + row * 398;
      box(c, qx, qy, 452, 378, 36, hi ? bg : th.soft, hi ? fg : th.line, hi ? 3.2 : 1.5);
      txt(c, kicker, qx + 28, qy + 44, { sz: 15, w: 700, col: hi ? fg : th.mu, ls: 1.4 });
      shape(c, hi ? fg : th.accc, qx + 68, qy + 124, 40, seq8[i * 2], 'sunny', 0.4, T * 0.3);
      icon(c, ic, qx + 68, qy + 124, 30, hi ? C.white : th.onacc, 2.4);
      txt(c, big, qx + 122, qy + 118, { sz: 25, w: 700, col: hi ? C.ink : th.fg });
      txt(c, ne, qx + 122, qy + 154, { sz: 19.5, w: 600, col: hi ? fg : th.acc });
      box(c, qx + 28, qy + 198, 396, 144, 22, th.card, th.line);
      const ls = wrap(c, sub, 21, 600, 348);
      ls.forEach((l, li) => txt(c, l, qx + 48, qy + 252 + li * 34, { sz: 21, w: 600, col: th.fg }));
    });
  }
});

// STORY 05: Autonomous Mobility, iOS 27.3, Nothing Ear Note, Firefox 161 & Agent Commerce (Nov 22–26, 2026)
SC.n5 = (c, t, sc, th, T) => story(c, t, sc, th, T, {
  num: '05',
  dateTag: '22–26 NOV 2026',
  catTag: 'MOBILITY & CONSUMER TECH',
  ic: 'car',
  shape: 'clover',
  title: 'Tesla FSD v15, iOS 27.3, Firefox 161 & Muse',
  neTitle: 'स्वचालित सवारी, आइओएस २७.३, फायरफक्स १६१ र म्युज',
  src: 'TechCrunch · The Verge · SiliconReport (22–26 Nov 2026)',
  beats: [
    ['Tesla FSD v15 & Bay Area Robotaxis', 'टेस्ला एफएसडी १५ र उबर-लुसिड-नुरो परीक्षण'],
    ['iOS 27.3 & Nothing Ear Note Case', 'आइओएस २७.३ र आवाज रेकर्ड गर्ने इयर नोट'],
    ['Firefox 161 Auto Per-Site VPN', 'फायरफक्स १६१ मा स्वचालित प्रति-साइट भिपिएन'],
    ['Walmart & Amazon Meta Muse Talks', 'वालमार्टमा म्युज शपिङ र अमेजनसँग वार्ता']
  ],
  draw: (c, t, beat, v, th, T) => {
    const items = [
      ['AUTONOMOUS DRIVING', 'Tesla FSD v15 + Uber · Lucid · Nuro', 'सान फ्रान्सिस्कोमा रोबोट्याक्सी र टेस्लाको नयाँ संस्करण', 'car', C.pc, C.blue, '22–24 NOV'],
      ['MOBILE & AUDIO GEAR', 'Apple iOS 27.3 + Nothing Ear Note', 'छ वटा आइफोन एपमा अपडेट र एआई रेकर्डिङ इयरबड्स', 'phone', C.aquac, C.aquad, '23 NOV'],
      ['PRIVACY BROWSER', 'Mozilla Firefox 161 + GrapheneOS RCS', 'हरेक वेबसाइटका लागि स्वचालित भिपिएन सुरक्षा', 'globe', C.violetc, C.violet, '23–24 NOV'],
      ['AI AGENT COMMERCE', 'Meta Muse × Walmart & Amazon Talks', 'ब्ल्याक फ्राइडे किनमेलमा एआई एजेन्टको प्रयोग', 'wallet', C.amberc, C.amberd, '23–26 NOV']
    ];
    items.forEach(([tag, title, ne, ic, bg, fg, dt], i) => {
      const hi = beat === i;
      const by = 36 + i * 198;
      box(c, 36, by, 920, 180, 32, hi ? bg : th.soft, hi ? fg : th.line, hi ? 3 : 1.5);
      box(c, 66, by + 34, 112, 112, 28, hi ? fg : th.accc, null);
      icon(c, ic, 122, by + 90, 52, hi ? C.white : th.onacc, 2.5);
      txt(c, tag, 206, by + 54, { sz: 17, w: 700, col: hi ? fg : th.mu, ls: 1.5 });
      chip(c, dt, 780, by + 28, { sz: 16, fill: C.white, col: fg, h: 34 });
      txt(c, title, 206, by + 102, { sz: 34, w: 700, col: hi ? C.ink : th.fg });
      txt(c, ne, 206, by + 146, { sz: 23, w: 600, col: hi ? fg : th.mu });
    });
  }
});

// QUICK WEEKLY TECH ROUNDUP (fast: f1..f4)
SC.fast = (c, t, sc, th, T) => {
  const v = V(T);
  dotgrid(c, th, T, .8);
  const f1 = lt(sc, 'f1'), f2 = lt(sc, 'f2'), f3 = lt(sc, 'f3'), f4 = lt(sc, 'f4');
  const beat = activeBeat(sc, t);

  // Header Banner
  bento(c, 68, 102, 1784, 130, 36, th, { fill: C.blueDark, stroke: '#3570e6' });
  chip(c, 'RAPID ROUNDUP · छोटकरीमा प्रविधि', 104, 138, { sz: 22, fill: C.aquac, col: C.aquad, ic: 'bolt' });
  txt(c, 'Local AI Compression, SSD Model Streaming, Open-Source Speedups & Music Policy', 540, 178, { sz: 28, w: 600, col: C.white });

  const cards = [
    [f1, 'PrismML Ternary Bonsai 27B', '54 GB → 5.9 GB', 'चौवन्न जिबीको मोडेल पाँच दशमलव नौ जिबीमा', 'Runs on laptops, iPhones & iPads with -1, 0, +1 weights', 'chip', C.pc, C.blue],
    [f2, 'Magnitude SSD Inference', '66 GB Model on SSD', 'एसएसडी स्टोरेजबाटै ठूला मोडेल सञ्चालन', '41 tokens/sec on a 24 GB GPU by streaming from SSD', 'server', C.aquac, C.aquad],
    [f3, 'Tencent DFly2 & Canada Fund', '+81% Speed · $1B Fund', 'डिफ्लाई टु ओपन सोर्स र क्यानडाको एआई कोष', '81% faster DeepSeek throughput + Canada $1B compute fund', 'bolt', C.amberc, C.amberd],
    [f4, 'Tidal AI Music Policy', 'Dec 9 Removal', 'टाइडलबाट पूर्ण एआई गीतहरू हटाइने', 'Removes 100% AI-generated tracks starting December 9', 'music', C.rosec, C.rosed]
  ];

  cards.forEach(([trig, title, metric, ne, sub, ic, bg, fg], i) => {
    const p = sp(t, Math.max(0.1, trig - 0.15), 0.75);
    const hi = beat === i;
    const col = i % 2, row = Math.floor(i / 2);
    const cx = 68 + col * 904, cy = 256 + row * 354 + (1 - p) * 40;
    c.save();
    c.globalAlpha = clamp(p * 1.3);
    box(c, cx, cy, 880, 330, 40, hi ? C.white : C.blueCont, hi ? C.aqua : '#3570e6', hi ? 3.5 : 1.8);
    box(c, cx + 38, cy + 38, 88, 88, 28, bg, null);
    icon(c, ic, cx + 82, cy + 82, 46, fg, 2.5);
    chip(c, metric, cx + 560, cy + 44, { sz: 24, fill: bg, col: fg, h: 48 });
    txt(c, title, cx + 152, cy + 88, { sz: 36, w: 700, col: hi ? C.ink : C.white });
    txt(c, ne, cx + 38, cy + 190, { sz: 28, w: 600, col: hi ? fg : '#d8e4ff' });
    txt(c, sub, cx + 38, cy + 245, { sz: 22, w: 500, col: hi ? C.mute : '#b5cbff' });
    c.restore();
  });
};
