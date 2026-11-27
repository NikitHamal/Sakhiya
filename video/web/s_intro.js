// Material 3 Expressive Scenes: hello, about (Sakhiya-One v2), series (Friday Weekly Recap)
const seq8 = ['cookie12', 'sunny', 'flower', 'clover', 'burst', 'squircle', 'pebble', 'gem'];

function win(c, t, a, b, fn, dy = 38) {
  if (t < a - .08 || t > b + .45) return;
  const pin = sp(t, a, .72), pout = eo3(P(t, b, b + .38));
  c.save();
  c.globalAlpha *= clamp(pin * 1.35) * (1 - pout);
  c.translate(0, (1 - pin) * dy - pout * 28);
  fn(t - a);
  c.restore();
}

SC.hello = (c, t, sc, th, T) => {
  const v = V(T), a1 = lt(sc, 'a1'), a2 = lt(sc, 'a2');
  dotgrid(c, th, T, clamp(t / .8));

  // Stage 1: Full-canvas Material 3 Expressive Title Opener (0 -> a1)
  const out1 = eio3(P(t, a1 - .35, a1 + .45));
  if (t < a1 + .55) {
    c.save();
    c.translate(0, -out1 * 1250);
    const g = sp(t, .12, 1.0), k = P(t, .12, 1.3) * 3, ii = Math.floor(k), mm = eio3(k % 1);
    const settle = eio3(P(t, 1.0, 1.75));
    const cy = lerp(520, 305, settle), R = 290 * g * lerp(1, .56, settle);

    // Expanding M3 Expressive shockwave rings
    for (let r = 0; r < 2; r++) {
      const h = P(t, 0.9 + r * 0.22, 1.85 + r * 0.22);
      if (h > 0 && h < 1) {
        c.globalAlpha = (1 - h) * 0.55;
        shapeStroke(c, C.white, 5, W / 2, cy, R + h * 750, 'cookie12', 'sunny', h, t * 0.6);
        c.globalAlpha = 1;
      }
    }

    // Orbiting tonal satellite pills
    const orb = sp(t, 0.8, 0.8);
    if (orb > 0.05) {
      [[-420, -90, 'AI & TECH', 'bolt', C.aquac, C.aquad], [420, -90, 'NEPAL EDU', 'grad', C.amberc, C.amberd], [-460, 90, 'SAKHIYA V2', 'mic', C.pc, C.onpc], [460, 90, 'EVERY FRIDAY', 'calendar', C.violetc, C.violetd]].forEach(([dx, dy, lb, ic, bg, fg], idx) => {
        const p = sp(t, 0.95 + idx * 0.1, 0.65);
        c.save();
        c.globalAlpha = clamp(p);
        const w0 = tw(c, lb, 20, 700) + 66;
        const bx = W / 2 + dx * p - w0 / 2, by = cy + dy * p - 22;
        box(c, bx, by, w0, 44, 22, bg, null);
        icon(c, ic, bx + 24, by + 22, 20, fg, 2.3);
        txt(c, lb, bx + 42, by + 29, { sz: 20, w: 700, col: fg, ls: 1 });
        c.restore();
      });
    }

    shape(c, C.white, W / 2, cy, R, seq8[ii % 8], seq8[(ii + 1) % 8], mm, t * .45);
    txt(c, 'स', W / 2, cy + R * 0.31, { sz: R * 0.94, w: 700, col: C.blue, al: 'center', a: g });

    if (t > 0.95) {
      words(c, 'NEBians News', W / 2, 665, t - 1.0, { sz: 148, w: 700, al: 'center', col: C.white, st: .08 });
      const p = sp(t, 1.45, .65);
      c.globalAlpha = clamp(p);
      const badgeTxt = 'WEEKLY RECAP  ·  EPISODE 01  ·  FRI 27 NOV 2026';
      const bw = tw(c, badgeTxt, 24, 700) + 64;
      box(c, W / 2 - bw / 2, 735 + (1 - p) * 28, bw, 52, 26, C.aquac, null);
      txt(c, badgeTxt, W / 2, 769 + (1 - p) * 28, { sz: 24, w: 700, col: C.aquad, al: 'center', ls: 1.5 });
      c.globalAlpha = 1;
    }
    c.restore();
  }

  // Stage 2: "नमस्ते सबैजनालाई।" (a1 -> a2)
  if (t >= a1 - .2 && t < a2 + .45) {
    const t1 = t - a1, o2 = eio3(P(t, a2 - .28, a2 + .35));
    c.save();
    c.translate(-o2 * 1950, 0);
    for (let k = 0; k < 3; k++) {
      const r = 180 + ((T * .85 + k / 3) % 1) * 640;
      c.globalAlpha = (1 - (T * .85 + k / 3) % 1) * .32 * (0.4 + v[0] * 1.8);
      shapeStroke(c, C.white, 4, W / 2, 460, r, 'cookie12', 'flower', (T * 0.5 + k * 0.3) % 1, T * 0.2);
    }
    c.globalAlpha = 1;
    words(c, 'नमस्ते', W / 2, 420, t1, { sz: 230, w: 700, al: 'center', col: C.white, st: .12 });
    words(c, 'सबैजनालाई', W / 2, 585, t1 - .18, { sz: 136, w: 600, al: 'center', col: '#d8e4ff', st: .1 });
    c.restore();
  }

  // Stage 3: Host Desk Card "म सखिया, तपाईंको समाचार वाचक..." (a2 -> end)
  if (t >= a2 - .25) {
    const t2 = t - a2, p = sp(t, a2 - .18, .82);
    // Left Avatar Studio Pod
    bento(c, 110, 140 + (1 - p) * 60, 620, 740, 48, th, { fill: C.blueCont, stroke: '#3570e6', a: clamp(p), wmShape: 'flower', wmR: 180 });
    sakhiya(c, th, 420, 435, 185 * p, T, v, { fill: C.white, ring: C.aqua, ring2: C.white, glyphCol: C.blue, a: clamp(p), badge: 'SAKHIYA-ONE V2 · STUDIO' });
    txt(c, '24 kHz Neural Broadcast Voice', 420, 795, { sz: 22, w: 500, col: '#c7d8ff', al: 'center', a: clamp(p) });

    // Right Welcome Hero Bento
    bento(c, 770, 140 + (1 - p) * 80, 1040, 740, 48, th, { fill: C.blueDark, stroke: '#3570e6', a: clamp(p), wmShape: 'sunny', wmR: 220, kicker: 'AI NEWS ANCHOR · साप्ताहिक प्रविधि र शिक्षा' });
    words(c, 'म सखिया', 835, 340, t2, { sz: 132, w: 700, col: C.white, st: .1 });
    txt(c, 'तपाईंको समाचार वाचक · Your Weekly News Anchor', 840, 420, { sz: 34, w: 500, col: '#c7d8ff', a: P(t2, .35, .85) });

    // 3 Feature Mini-Pills inside Right Bento
    const pills = [
      ['Global AI & Tech', 'विश्वभरका प्रमुख प्रविधि खबर', 'globe', C.aquac, C.aquad],
      ['Nepal Education', 'नेपालको शैक्षिक क्षेत्रका अपडेट', 'grad', C.amberc, C.amberd],
      ['Friday Weekly Recap', 'हरेक शुक्रबार साप्ताहिक संकलन', 'calendar', C.pc, C.onpc]
    ];
    pills.forEach(([enT, neT, ic, bg, fg], idx) => {
      const q = sp(t2, 0.55 + idx * 0.22, 0.68);
      const py = 485 + idx * 112 + (1 - q) * 30;
      c.save();
      c.globalAlpha = clamp(q);
      box(c, 835, py, 910, 92, 28, C.blueCont, '#3570e6', 1.5);
      box(c, 855, py + 16, 60, 60, 20, bg, null);
      icon(c, ic, 885, py + 46, 30, fg, 2.3);
      txt(c, enT, 935, py + 42, { sz: 28, w: 700, col: C.white });
      txt(c, neT, 935, py + 74, { sz: 22, w: 500, col: '#c7d8ff' });
      c.restore();
    });
  }
};

SC.about = (c, t, sc, th, T) => {
  const v = V(T);
  dotgrid(c, th, T, .65);
  const ids = ['b1', 'b2', 'b3', 'b4', 'b5', 'b6'];
  const st = ids.map(k => lt(sc, k));
  const en = st.map((s, i) => i < 5 ? st[i + 1] : sc.t1 - sc.t0 - .25);
  let k = ids.findIndex((_, i) => t >= st[i] - .15 && t < en[i]);
  if (k < 0) k = t < st[0] ? 0 : 5;

  // Left Host & Live Telemetry Column
  const lp = sp(t, 0, .8);
  bento(c, 84, 118, 560, 820, 44, th, { a: clamp(lp), wmShape: 'cookie12', wmR: 160, kicker: 'VOICE ENGINE TELEMETRY' });
  sakhiya(c, th, 364, 385, 155 * lp, T, v, {
    k: k + (t - st[k]) * .2,
    a: clamp(lp),
    badge: `STAGE 0${k + 1} / 06`
  });

  // Live Mini Spec Grid on Left Card Bottom
  const specs = [
    ['MODEL', 'Sakhiya-One v2'],
    ['STUDIO', '23.0 MB INT8'],
    ['COMPACT', '8.5 MB INT8'],
    ['SPEED', '>30× Real-Time']
  ];
  specs.forEach(([lb, vl], idx) => {
    const sx = 116 + (idx % 2) * 254, sy = 640 + Math.floor(idx / 2) * 124;
    box(c, sx, sy, 238, 106, 24, idx === (k % 4) ? th.accc : th.soft, th.line, 1.5);
    txt(c, lb, sx + 22, sy + 38, { sz: 17, w: 700, col: th.mu, ls: 1.5 });
    txt(c, vl, sx + 22, sy + 76, { sz: 26, w: 700, col: idx === (k % 4) ? th.onacc : th.fg });
  });

  // Right Area: 6 Line-Synced Material 3 Expressive Bento Cards
  const X = 680, Y = 118, Wd = 1156, Ht = 820;

  // b1: Sakhiya-One v2 Current Phase Upgrade
  win(c, t, st[0], en[0], (dt) => {
    bento(c, X, Y, Wd, Ht, 44, th, { wmShape: 'sunny', wmR: 220, kicker: '01 · MODEL UPGRADE · नयाँ दोस्रो संस्करण' });
    words(c, 'Sakhiya-One v2', X + 56, Y + 150, dt, { sz: 96, w: 700, col: th.fg });
    txt(c, 'सखिया वान को नयाँ दोस्रो संस्करण · Current Phase Studio TTS', X + 58, Y + 215, { sz: 32, w: 500, col: th.acc });

    // Version Evolution Cards (V1 -> V2 Studio)
    const cols = [
      ['Sakhiya V1 (Previous)', 'अघिल्लो पहिलो चरण', '16 MB · Early Phase', 'Basic prosody & sibilants', th.soft, th.mu],
      ['Sakhiya-One v2 (Current)', 'हालको नयाँ दोस्रो संस्करण', '23 MB Studio · 24 kHz Vocos', 'Cleaner sibilants, natural Nepali cadence', th.accc, th.onacc]
    ];
    cols.forEach(([ti, ne, s1, s2, bg, fg], i) => {
      const p = sp(dt, 0.25 + i * 0.22, 0.7);
      const cx = X + 56 + i * 536, cy = Y + 275 + (1 - p) * 35;
      c.save();
      c.globalAlpha = clamp(p);
      box(c, cx, cy, 508, 320, 36, bg, i ? th.acc : th.line, i ? 2.5 : 1.5);
      chip(c, i ? 'ACTIVE IN EPISODE 01' : 'LEGACY', cx + 32, cy + 32, { sz: 18, fill: i ? th.acc : th.card, col: i ? C.white : th.mu, ic: i ? 'check' : 'info' });
      txt(c, ti, cx + 32, cy + 135, { sz: 36, w: 700, col: fg });
      txt(c, ne, cx + 32, cy + 180, { sz: 26, w: 600, col: i ? th.acc : th.mu });
      txt(c, s1, cx + 32, cy + 235, { sz: 24, w: 700, col: fg });
      txt(c, s2, cx + 32, cy + 275, { sz: 21, w: 500, col: fg, a: 0.85 });
      c.restore();
    });
    wavebars(c, X + 56, Y + 695, Wd - 112, 110, T, v, th.acc, 56, th.sec);
  });

  // b2: Bilingual Nepali & Sanskrit Compact Neural Speech Model
  win(c, t, st[1], en[1], (dt) => {
    bento(c, X, Y, Wd, Ht, 44, th, { wmShape: 'flower', wmR: 220, kicker: '02 · BILINGUAL NEURAL SPEECH · द्विभाषी क्षमता' });
    words(c, 'नेपाली र संस्कृत', X + 56, Y + 155, dt, { sz: 92, w: 700, col: th.fg });
    txt(c, 'Native Devanagari Phonetics & Natural Clause Cadence', X + 58, Y + 215, { sz: 30, w: 500, col: th.mu });

    const langs = [
      ['नेपाली भाषा (Nepali)', 'समाचार, कथा र संवाद वाचन', 'नमस्ते, आजका मुख्य प्रविधि र शैक्षिक समाचारमा स्वागत छ।', C.pc, C.onpc, C.blue],
      ['संस्कृत भाषा (Sanskrit)', 'वैदिक छन्द र श्लोक वाचन', 'धर्मक्षेत्रे कुरुक्षेत्रे समवेता युयुत्सवः।', C.aquac, C.aquad, C.aquad]
    ];
    langs.forEach(([ti, sub, sample, bg, fg, ac], i) => {
      const p = sp(dt, 0.2 + i * 0.25, 0.72);
      const cy = Y + 265 + i * 245 + (1 - p) * 35;
      c.save();
      c.globalAlpha = clamp(p);
      box(c, X + 56, cy, Wd - 112, 215, 36, bg, ac, 2);
      shape(c, ac, X + 126, cy + 108, 48, i ? 'clover' : 'cookie12', 'sunny', Math.sin(T + i) * .5 + .5, T * 0.3);
      txt(c, i ? 'सं' : 'ने', X + 126, cy + 124, { sz: 46, w: 700, col: C.white, al: 'center' });
      txt(c, ti, X + 205, cy + 68, { sz: 38, w: 700, col: fg });
      txt(c, sub, X + 205, cy + 110, { sz: 24, w: 600, col: ac });
      box(c, X + 205, cy + 132, Wd - 295, 58, 18, C.white, null);
      txt(c, sample, X + 228, cy + 170, { sz: 25, w: 600, col: C.ink });
      c.restore();
    });
  });

  // b3: Ultra-Compact Model Sizes (23 MB Studio & 8.5 MB Compact)
  win(c, t, st[2], en[2], (dt) => {
    bento(c, X, Y, Wd, Ht, 44, th, { wmShape: 'squircle', wmR: 220, kicker: '03 · ULTRA-LIGHTWEIGHT FOOTPRINT · सानो आकार' });
    words(c, 'Tiny Footprint, Studio Voice', X + 56, Y + 145, dt, { sz: 72, w: 700, col: th.fg });

    const tiers = [
      [23, 'MB', 'Studio Edition (Vocos v2)', 'स्टुडियो संस्करण · तेइस मेगाबाइट', 'Used in this episode · 24 kHz high-band clarity', th.accc, th.acc, th.onacc, 0.42],
      [8.5, 'MB', 'Compact Edition (Fast)', 'सानो संस्करण · आठ दशमलव पाँच मेगाबाइट', 'Ultra-lightweight browser & mobile tier', C.aquac, C.aquad, C.aquad, 0.18]
    ];
    tiers.forEach(([num, unit, title, neSub, desc, bg, barCol, fg, frac], i) => {
      const p = sp(dt, 0.2 + i * 0.28, 0.75);
      const prog = eo3(P(dt, 0.25 + i * 0.25, 1.6 + i * 0.25));
      const val = i === 0 ? Math.round(num * prog) : (num * prog).toFixed(1);
      const cx = X + 56 + i * 536, cy = Y + 215 + (1 - p) * 40;
      c.save();
      c.globalAlpha = clamp(p);
      box(c, cx, cy, 508, 530, 40, bg, barCol, 2);
      txt(c, String(val), cx + 40, cy + 175, { sz: 138, w: 700, col: barCol });
      txt(c, unit, cx + 48 + tw(c, String(val), 138, 700), cy + 175, { sz: 54, w: 700, col: fg });
      txt(c, title, cx + 40, cy + 255, { sz: 34, w: 700, col: fg });
      txt(c, neSub, cx + 40, cy + 300, { sz: 25, w: 600, col: barCol });
      txt(c, desc, cx + 40, cy + 350, { sz: 21, w: 500, col: fg, a: 0.85 });
      // Size comparison bar vs typical 500+ MB TTS models
      txt(c, 'SIZE VS TYPICAL 500 MB CLOUD TTS', cx + 40, cy + 425, { sz: 16, w: 700, col: fg, ls: 1.2, a: 0.75 });
      box(c, cx + 40, cy + 445, 428, 28, 14, C.white, null);
      box(c, cx + 40, cy + 445, Math.max(28, 428 * frac * prog), 28, 14, barCol, null);
      c.restore();
    });
  });

  // b4: Runs Directly in Browser · 24 kHz · 30x Faster Than Real-Time
  win(c, t, st[3], en[3], (dt) => {
    bento(c, X, Y, Wd, Ht, 44, th, { wmShape: 'burst', wmR: 220, kicker: '04 · ON-DEVICE BROWSER INFERENCE · सिधै वेब ब्राउजरमा' });
    // Browser Mock Window
    box(c, X + 52, Y + 78, Wd - 104, 410, 32, th.soft, th.line, 2);
    for (let i = 0; i < 3; i++) {
      c.beginPath(); c.arc(X + 94 + i * 28, Y + 116, 8, 0, TAU);
      c.fillStyle = [C.rose, C.amber, C.mint][i]; c.fill();
    }
    box(c, X + 195, Y + 96, Wd - 440, 40, 20, th.card, th.line);
    icon(c, 'lock', X + 222, Y + 116, 18, th.sec, 2.2);
    txt(c, 'https://nikithamal.github.io/Sakhiya', X + 246, Y + 123, { sz: 21, w: 600, col: th.mu });
    chip(c, 'ZERO SERVER · 100% LOCAL', X + Wd - 340, Y + 96, { sz: 17, fill: C.aquac, col: C.aquad, h: 40 });

    box(c, X + 88, Y + 162, Wd - 176, 140, 24, th.card, th.line);
    txt(c, 'सिधै वेब ब्राउजरमा चौबीस किलोहर्ज स्टुडियो गुणस्तरमा चल्छु।', X + 120, Y + 225, { sz: 33, w: 600, col: th.fg });
    txt(c, 'WebAssembly + ONNX Runtime · Runs on Laptops & Phones', X + 120, Y + 270, { sz: 22, w: 500, col: th.mu });
    wavebars(c, X + 88, Y + 395, Wd - 176, 95, T, v, th.acc, 48, th.sec);

    // Bottom 3 Metric Pods: 24 kHz, 30x Speed, 0 Server Cost
    const pods = [
      ['24 kHz', 'Studio Sample Rate', 'चौबीस किलोहर्ज आवाज', C.pc, C.blue],
      ['33.4×', 'Faster Than Real-Time', 'तीस गुणाभन्दा छिटो', C.aquac, C.aquad],
      ['0 MB', 'Server Upload Needed', 'सर्भर बिना सिधै डिभाइसमा', C.violetc, C.violetd]
    ];
    pods.forEach(([big, l1, l2, bg, fg], i) => {
      const p = sp(dt, 0.35 + i * 0.18, 0.65);
      const px = X + 52 + i * 360, py = Y + 520 + (1 - p) * 30;
      c.save();
      c.globalAlpha = clamp(p);
      box(c, px, py, 332, 245, 32, bg, null);
      txt(c, big, px + 32, py + 105, { sz: 74, w: 700, col: fg });
      txt(c, l1, px + 32, py + 162, { sz: 25, w: 700, col: C.ink });
      txt(c, l2, px + 32, py + 198, { sz: 21, w: 500, col: fg });
      c.restore();
    });
  });

  // b5: Honest Current Phase Status (Cleaner & Clearer, Still Evolving)
  win(c, t, st[4], en[4], (dt) => {
    bento(c, X, Y, Wd, Ht, 44, th, { wmShape: 'pebble', wmR: 220, kicker: '05 · HONEST PHASE REPORT · हालको अवस्था' });
    words(c, 'Cleaner, Clearer & Grounded', X + 56, Y + 148, dt, { sz: 70, w: 700, col: th.fg });
    txt(c, 'अझै पूर्ण त्रुटिरहित छैन, तर पहिलेभन्दा निकै सफा र स्पष्ट आवाज', X + 58, Y + 206, { sz: 29, w: 500, col: th.acc });

    const bars = [
      ['Nepali Pronunciation & Loanwords (उच्चारण शुद्धता)', 0.92, '36/36 Checked · Crisp च/छ & स/श'],
      ['High-Band Voice Clarity (आवाजको सफाइ)', 0.89, '24 kHz Studio Vocos · Reduced Metallic Buzz'],
      ['Natural Broadcast Flow (प्राकृतिक समाचार लय)', 0.84, 'Current Phase v2 · Active Weekly Refinement']
    ];
    bars.forEach(([label, target, note], i) => {
      const p = eo3(P(dt, 0.25 + i * 0.22, 1.6 + i * 0.22));
      const by = Y + 255 + i * 165;
      box(c, X + 56, by, Wd - 112, 140, 28, th.soft, th.line);
      txt(c, label, X + 88, by + 48, { sz: 27, w: 700, col: th.fg });
      txt(c, note, X + Wd - 88, by + 48, { sz: 21, w: 600, col: th.acc, al: 'right' });
      box(c, X + 88, by + 76, Wd - 176, 32, 16, th.card, th.line);
      box(c, X + 88, by + 76, (Wd - 176) * target * p, 32, 16, i === 1 ? C.aquad : th.acc, null);
    });
  });

  // b6: Continuous Weekly Improvement Roadmap
  win(c, t, st[5], en[5], (dt) => {
    bento(c, X, Y, Wd, Ht, 44, th, { wmShape: 'clover', wmR: 220, kicker: '06 · CONTINUOUS EVOLUTION · निरन्तर सुधार' });
    words(c, 'Improving Every Friday', X + 56, Y + 150, dt, { sz: 78, w: 700, col: th.fg });
    txt(c, 'आउने हरेक एपिसोडसँगै उच्चारण र लय अझै निखारिँदै जानेछ', X + 58, Y + 210, { sz: 30, w: 500, col: th.mu });

    const steps = [
      ['Phase 1 (V1)', 'Initial 16 MB Browser Prototype', 'Completed', 'check', th.soft, th.mu],
      ['Phase 2 (Sakhiya-One v2)', '23 MB Studio Vocos · Crisp Nepali G2P', 'Current Episode', 'bolt', th.accc, th.onacc],
      ['Next Phases', 'Richer Prosody, Zero-Error Expressive Flow', 'In Progress', 'spark', C.aquac, C.aquad]
    ];
    steps.forEach(([ti, sub, status, ic, bg, fg], i) => {
      const p = sp(dt, 0.2 + i * 0.25, 0.7);
      const sx = X + 56 + i * 356, sy = Y + 275 + (1 - p) * 40;
      c.save();
      c.globalAlpha = clamp(p);
      box(c, sx, sy, 332, 460, 36, bg, i === 1 ? th.acc : th.line, i === 1 ? 2.5 : 1.5);
      shape(c, i === 1 ? th.acc : (i === 2 ? C.aquad : th.line), sx + 166, sy + 120, 62 * p, seq8[i * 2], seq8[i * 2 + 1], 0.5, T * 0.35);
      icon(c, ic, sx + 166, sy + 120, 42, C.white, 2.5);
      txt(c, ti, sx + 166, sy + 245, { sz: 30, w: 700, col: fg, al: 'center' });
      const lines = wrap(c, sub, 22, 500, 270);
      lines.forEach((l, li) => txt(c, l, sx + 166, sy + 295 + li * 32, { sz: 22, w: 500, col: fg, al: 'center', a: 0.85 }));
      chip(c, status, sx + 166 - (tw(c, status, 20, 700) + 38) / 2, sy + 375, { sz: 20, fill: C.white, col: i === 2 ? C.aquad : C.ink });
      c.restore();
    });
  });
};

SC.series = (c, t, sc, th, T) => {
  const v = V(T);
  const c1 = lt(sc, 'c1'), c2 = lt(sc, 'c2'), c3 = lt(sc, 'c3'), c4 = lt(sc, 'c4'), c5 = lt(sc, 'c5');
  dotgrid(c, th, T, .75);

  const transitionOut = eio3(P(t, c5 - .15, c5 + .45));
  if (t < c5 + .5) {
    c.save();
    c.translate(0, -transitionOut * 980);

    // Top Hero Header + 7-Day Strict Weekly Window Calendar Strip (Sat Nov 21 -> Fri Nov 27)
    bento(c, 84, 112, 1752, 250, 40, th, { wmShape: 'sunny', wmR: 180, kicker: 'WEEKLY RECAP SCHEDULE · हरेक शुक्रबार साप्ताहिक अंक' });
    words(c, 'Friday Weekly Recap', 124, 215, t - c1 + .2, { sz: 78, w: 700, col: th.fg });
    txt(c, 'गत शनिबार नोभेम्बर २१ देखि आज शुक्रबार नोभेम्बर २७ सम्मका ताजा समाचार मात्र', 126, 268, { sz: 27, w: 500, col: th.acc });

    // 7-Day Calendar Pills on Right Side of Top Bento
    const days = [
      ['SAT', '21'], ['SUN', '22'], ['MON', '23'], ['TUE', '24'], ['WED', '25'], ['THU', '26'], ['FRI', '27']
    ];
    days.forEach(([dName, dNum], i) => {
      const p = sp(t, 0.25 + i * 0.09, 0.65);
      const isFri = i === 6;
      const highlight = t >= c4 - 0.2;
      const dx = 1045 + i * 106, dy = 148 + (1 - p) * 25;
      c.save();
      c.globalAlpha = clamp(p);
      box(c, dx, dy, 94, 136, 26, isFri ? th.acc : (highlight ? th.accc : th.soft), isFri ? C.white : th.line, isFri ? 2.5 : 1.5);
      txt(c, dName, dx + 47, dy + 44, { sz: 18, w: 700, col: isFri ? th.bg : th.mu, al: 'center', ls: 1 });
      txt(c, dNum, dx + 47, dy + 96, { sz: 38, w: 700, col: isFri ? th.bg : th.fg, al: 'center' });
      if (isFri) txt(c, 'TODAY', dx + 47, dy + 122, { sz: 13, w: 700, col: th.bg, al: 'center', ls: 1 });
      c.restore();
    });

    // 3 Pillar Bento Cards (Global AI & Models | Global Tech, Space & Security | Nepal Education Sector)
    const pillars = [
      ['Global AI & Models', 'विश्वभरका प्रमुख एआई र मोडेल', 'GPT-6.2 Sol, Claude Ship, Rust API, Private Inference', 'spark', c2, C.pc, C.onpc, C.blue],
      ['Tech, Space & Security', 'अन्तरिक्ष, चिप, सवारी र साइबर सुरक्षा', 'SpaceX Starmind, Nvidia & Huawei, Shirudo Linux, FSD v15', 'sat', c2 + 1.2, C.aquac, C.aquad, C.aqua],
      ['Nepal Education Sector', 'नेपालको शैक्षिक क्षेत्रका मुख्य खबर', 'TU College Audit, MEC Results, KMC Autism School, Hackathon', 'grad', c3, C.amberc, C.amberd, C.amber]
    ];
    pillars.forEach(([ti, ne, sub, ic, trig, bg, fg, ac], i) => {
      const p = sp(t, Math.max(0.4, trig - 0.2), 0.78);
      const px = 84 + i * 596, py = 392 + (1 - p) * 110;
      const active = (i < 2 && t >= c2 && t < c3) || (i === 2 && t >= c3 && t < c4);
      c.save();
      c.globalAlpha = clamp(p * 1.3);
      bento(c, px, py, 560, 520, 42, th, {
        fill: active ? th.high : th.card,
        stroke: active ? ac : th.line,
        lw: active ? 3 : 1.8,
        wmShape: seq8[i * 2],
        wmR: 150
      });
      box(c, px + 42, py + 42, 92, 92, 30, bg, null);
      icon(c, ic, px + 88, py + 88, 48, fg, 2.4);
      chip(c, `PILLAR 0${i + 1}`, px + 370, py + 58, { sz: 17, fill: th.soft, col: th.mu });
      txt(c, ti, px + 42, py + 205, { sz: 42, w: 700, col: th.fg });
      txt(c, ne, px + 42, py + 255, { sz: 28, w: 600, col: ac });
      const lines = wrap(c, sub, 24, 500, 476);
      lines.forEach((l, li) => txt(c, l, px + 42, py + 325 + li * 36, { sz: 24, w: 500, col: th.mu }));
      c.restore();
    });
    c.restore();
  }

  // Stage 2: "अब लागौँ आजको पहिलो समाचारतर्फ।" (c5 -> end of scene)
  if (t >= c5 - .15) {
    const dt = t - c5;
    txt(c, '21 NOV – 27 NOV 2026  ·  VERIFIED WEEKLY LINEUP', W / 2, 215, { sz: 26, w: 700, col: th.acc, al: 'center', ls: 4, a: sp(t, c5, .5) });
    words(c, 'अब लागौँ आजको पहिलो समाचारतर्फ', W / 2, 330, dt, { sz: 86, w: 700, al: 'center', col: th.fg });

    const lineup = [
      ['01', 'SpaceX Starmind', '1M Orbital AI Satellites', 'sat'],
      ['02', 'Nvidia & Chips', 'Huang Returns · Export Curbs', 'chip'],
      ['03', 'OpenAI & Claude', 'GPT-6.2 Sol · Rust API -70%', 'code'],
      ['04', 'Cyber & Policy', 'Shirudo Linux · NY Toy Ban', 'shield'],
      ['05', 'Mobility & Tech', 'FSD v15 · iOS 27.3 · Firefox', 'car'],
      ['06', 'Nepal Education', 'TU · MEC · KMC · Hackathon', 'grad']
    ];
    lineup.forEach(([num, ti, sub, ic], i) => {
      const p = sp(t, c5 + .15 + i * .1, .65);
      const col = i % 3, row = Math.floor(i / 3);
      const bx = 132 + col * 566, by = 430 + row * 225 + (1 - p) * 45;
      c.save();
      c.globalAlpha = clamp(p);
      box(c, bx, by, 524, 195, 34, i === 0 ? th.accc : th.card, i === 0 ? th.acc : th.line, i === 0 ? 2.5 : 1.5);
      shape(c, i === 0 ? th.acc : th.soft, bx + 82, by + 98, 52, seq8[i], seq8[i + 1], 0.4, T * 0.3);
      txt(c, num, bx + 82, by + 114, { sz: 38, w: 700, col: i === 0 ? th.bg : th.fg, al: 'center' });
      txt(c, ti, bx + 158, by + 88, { sz: 34, w: 700, col: i === 0 ? th.onacc : th.fg });
      txt(c, sub, bx + 158, by + 132, { sz: 22, w: 500, col: i === 0 ? th.acc : th.mu });
      c.restore();
    });
  }
};
