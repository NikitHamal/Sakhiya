// Material 3 Expressive UI Kit: Vector Icons, Bento Containers, Ambient Harmonic Canvas, Broadcast Header, Compact Dual Subtitles
const IC = {
  home: 'M3 11 L12 3 L21 11 M5 10 V21 H19 V10',
  search: 'M10.5 3a7.5 7.5 0 1 0 0 15a7.5 7.5 0 1 0 0-15 M16 16 L21 21',
  check: 'M4 12.5 L9.5 18 L20 6.5',
  info: 'M12 2.5a9.5 9.5 0 1 0 0 19a9.5 9.5 0 1 0 0-19 M12 11 V17 M12 7.5 V8',
  warn: 'M12 3 L22 20 H2 Z M12 10 V14.5 M12 17 V17.5',
  x: 'M5 5 L19 19 M19 5 L5 19',
  shield: 'M12 2.5 L20 5.5 V12 C20 17 16.5 20.5 12 22 C7.5 20.5 4 17 4 12 V5.5 Z',
  lock: 'M6 11 H18 V21 H6 Z M8.5 11 V7.5 a3.5 3.5 0 0 1 7 0 V11',
  doc: 'M6 2.5 H14 L19 7.5 V21.5 H6 Z M14 2.5 V7.5 H19 M9 13 H16 M9 17 H16',
  book: 'M4 4.5 C7 3.5 10 4 12 6 C14 4 17 3.5 20 4.5 V19.5 C17 18.5 14 19 12 21 C10 19 7 18.5 4 19.5 Z M12 6 V21',
  compass: 'M12 2.5a9.5 9.5 0 1 0 0 19a9.5 9.5 0 1 0 0-19 M15.5 8.5 L13.5 13.5 L8.5 15.5 L10.5 10.5 Z',
  user: 'M12 4 a4 4 0 1 0 0 8 a4 4 0 1 0 0-8 M4 21 C4.5 16.5 8 14.5 12 14.5 C16 14.5 19.5 16.5 20 21',
  bolt: 'M13 2 L4 14 H11 L10 22 L20 9.5 H13 Z',
  chip: 'M7 7 H17 V17 H7 Z M10 3 V7 M14 3 V7 M10 17 V21 M14 17 V21 M3 10 H7 M3 14 H7 M17 10 H21 M17 14 H21',
  wallet: 'M3 7 H19 V19 H3 Z M3 7 L16 3 V7 M15 13 H19',
  pause: 'M8 5 V19 M16 5 V19',
  globe: 'M12 2.5a9.5 9.5 0 1 0 0 19a9.5 9.5 0 1 0 0-19 M3 12 H21 M12 3 C8 8 8 16 12 21 C16 16 16 8 12 3',
  play: 'M8 5 L19 12 L8 19 Z',
  phone: 'M7 2.5 H17 V21.5 H7 Z M11 18.5 H13',
  filter: 'M3 5 H21 L14 13 V20 L10 18 V13 Z',
  glass: 'M5 12a7 7 0 1 0 14 0a7 7 0 1 0-14 0 M8 12a4 4 0 1 0 8 0a4 4 0 1 0-8 0',
  sat: 'M13 7 L17 11 L11 17 L7 13 Z M5 5 L9 9 M15 15 L19 19 M4 19 A6 6 0 0 1 4 13 M2 21 A9 9 0 0 1 2 11',
  code: 'M8 7 L3 12 L8 17 M16 7 L21 12 L16 17 M13.5 5 L10.5 19',
  car: 'M4 13 L6.5 7 H17.5 L20 13 M3 13 H21 V18 H3 Z M6.5 18 V20 M17.5 18 V20 M7 15.5 H7.5 M16.5 15.5 H17',
  grad: 'M12 4 L2 9 L12 14 L22 9 Z M6 11.5 V17 C6 19 9 20.5 12 20.5 C15 20.5 18 19 18 17 V11.5 M21 9.5 V16',
  music: 'M9 18 V6 L20 4 V16 M6.5 18a2.5 2.5 0 1 0 5 0a2.5 2.5 0 1 0-5 0 M15.5 16a2.5 2.5 0 1 0 5 0a2.5 2.5 0 1 0-5 0',
  spark: 'M12 2 L14.5 9.5 L22 12 L14.5 14.5 L12 22 L9.5 14.5 L2 12 L9.5 9.5 Z',
  chart: 'M4 19 H20 M6 16 L10 11 L14 13.5 L19 6.5 M15 6.5 H19 V10.5',
  server: 'M4 4 H20 V10 H4 Z M4 14 H20 V20 H4 Z M7 7 H7.5 M7 17 H7.5',
  calendar: 'M4 6 H20 V20 H4 Z M4 10 H20 M8 3 V7 M16 3 V7',
  mic: 'M9 4 H15 V12 A3 3 0 0 1 9 12 Z M5 11 A7 7 0 0 0 19 11 M12 18 V21 M8 21 H16'
};
const IP = {};
for (const k in IC) IP[k] = new Path2D(IC[k]);

function icon(c, n, x, y, s, col, lw = 2.2, fill) {
  if (!IP[n]) n = 'spark';
  c.save();
  c.translate(x - s / 2, y - s / 2);
  c.scale(s / 24, s / 24);
  c.lineWidth = lw;
  c.lineCap = c.lineJoin = 'round';
  c.strokeStyle = col;
  if (fill) { c.fillStyle = fill; c.fill(IP[n]); }
  c.stroke(IP[n]);
  c.restore();
}

function chip(c, s, x, y, o = {}) {
  const sz = o.sz || 24, w = tw(c, s, sz, o.w || 600) + (o.pad || 38) + (o.ic ? sz + 11 : 0), h = o.h || sz * 1.92;
  c.save();
  c.globalAlpha *= (o.a ?? 1);
  box(c, x, y, w, h, h / 2, o.fill || C.pc, o.line, o.lw || 1.5);
  const tx = x + (o.pad || 38) / 2;
  if (o.ic) icon(c, o.ic, tx + sz / 2, y + h / 2, sz * 0.94, o.icCol || o.col || C.onpc, 2.2);
  txt(c, s, tx + (o.ic ? sz + 11 : 0), y + h / 2 + sz * .35, { sz, w: o.w || 600, col: o.col || C.onpc, ls: o.ls || 0 });
  c.restore();
  return w;
}

// 5 Expressive Material 3 Themes
const TH = {
  light: {
    id: 'light', bg: C.bg, fg: C.ink, mu: C.mute, card: C.white, line: C.line, soft: C.low, high: C.cont,
    acc: C.blue, accc: C.pc, onacc: C.onpc, sec: C.aquad, secc: C.aquac, ter: C.violet, terc: C.violetc, dot: '#d5dff4'
  },
  dark: {
    id: 'dark', bg: C.gr, fg: C.onGr, mu: C.muteGr, card: C.gr1, line: C.grLine, soft: C.gr2, high: C.gr3,
    acc: C.acc, accc: '#1e3264', onacc: '#e3ecff', sec: C.aqua, secc: '#103b36', ter: C.amber, terc: '#42300c', dot: '#202636'
  },
  blue: {
    id: 'blue', bg: C.blue, fg: C.white, mu: '#c7d8ff', card: C.blueCont, line: '#316be0', soft: C.blueDark, high: C.blueHigh,
    acc: C.white, accc: '#1d5ce0', onacc: C.white, sec: C.aqua, secc: '#0b4a52', ter: C.amber, terc: '#1645ab', dot: '#165ee6'
  },
  teal: {
    id: 'teal', bg: C.tealBg, fg: '#e8fbf8', mu: '#8ec7bf', card: C.teal1, line: C.tealLine, soft: C.teal2, high: '#1c524d',
    acc: C.tealAcc, accc: '#164f49', onacc: '#d2fff9', sec: C.amber, secc: '#4a3910', ter: C.sky, terc: '#103e54', dot: '#133b38'
  },
  plum: {
    id: 'plum', bg: C.plumBg, fg: C.plumInk, mu: C.plumMute, card: C.plumCard, line: C.plumLine, soft: C.plumSoft, high: '#e0d4ff',
    acc: C.plumAcc, accc: '#e2d6ff', onacc: '#21005d', sec: C.aquad, secc: C.aquac, ter: C.rose, terc: C.rosec, dot: '#dcd0f7'
  }
};

const SC_META = {
  hello:  { tag: 'WELCOME',         ne: 'स्वागतम्',           ic: 'spark' },
  about:  { tag: 'SAKHIYA-ONE V2',  ne: 'सखिया वान V2',       ic: 'mic' },
  series: { tag: 'FRIDAY RECAP',    ne: 'साप्ताहिक संकलन',     ic: 'calendar' },
  n1:     { tag: '01 · SPACE & AI', ne: 'अन्तरिक्ष र क्लाउड',   ic: 'sat' },
  n2:     { tag: '02 · CHIPS',      ne: 'चिप र सेमिकन्डक्टर',  ic: 'chip' },
  n3:     { tag: '03 · AI & CODE',  ne: 'मोडेल र इन्जिनियरिङ', ic: 'code' },
  n4:     { tag: '04 · SECURITY',   ne: 'साइबर सुरक्षा र नीति', ic: 'shield' },
  n5:     { tag: '05 · MOBILITY',   ne: 'सवारी र हार्डवेयर',    ic: 'car' },
  edu:    { tag: 'NEPAL EDUCATION', ne: 'नेपाल शिक्षा विशेष',   ic: 'grad' },
  fast:   { tag: 'QUICK ROUNDUP',   ne: 'छोटकरीमा प्रविधि',     ic: 'bolt' },
  outro:  { tag: 'SEE YOU FRIDAY',  ne: 'अर्को शुक्रबार भेटौँला', ic: 'globe' }
};

function dotgrid(c, th, t, a = 1) {
  const v = V(t);
  c.save();
  c.globalAlpha *= a * 0.65;
  c.fillStyle = th.dot;
  const g = 64;
  for (let y = 48; y <= H; y += g) {
    for (let x = 48; x <= W; x += g) {
      const d = Math.hypot(x - W / 2, y - H / 2);
      const w = Math.sin(d / 160 - t * 1.5) * .5 + .5;
      c.beginPath();
      c.arc(x, y, 1.8 + 2.4 * w + v[0] * 1.4, 0, TAU);
      c.fill();
    }
  }
  c.restore();
}

function V(t) {
  const E = window.ENV;
  if (!E || !E.length) return new Array(33).fill(0);
  const f = clamp(t * 30, 0, E.length - 2), i = Math.floor(f), k = f - i, a = E[i], b = E[i + 1];
  return a.map((v, j) => lerp(v, b[j], k));
}

// Expressive M3 Bento Container with optional corner watermark shape & header pill
function bento(c, x, y, w, h, r, th, o = {}) {
  c.save();
  c.globalAlpha *= (o.a ?? 1);
  const fill = o.fill || th.card;
  const stroke = o.stroke === null ? null : (o.stroke || th.line);
  box(c, x, y, w, h, r, fill, stroke, o.lw || 1.8);
  if (o.wmShape) {
    c.save();
    rr(c, x, y, w, h, r);
    c.clip();
    c.globalAlpha *= (o.wmAlpha || 0.08);
    shape(c, o.wmCol || th.acc, x + w - (o.wmX || 60), y + (o.wmY || 60), o.wmR || 140, o.wmShape, o.wmShape2 || o.wmShape, o.wmM || 0, o.wmRot || 0);
    c.restore();
  }
  if (o.kicker) {
    txt(c, o.kicker, x + 36, y + 42, { sz: 18, w: 700, col: o.kickerCol || th.acc, ls: 2.5 });
  }
  c.restore();
}

// Voice-reactive Sakhiya-One v2 Avatar: Multi-ring 32-band harmonic emblem
function sakhiya(c, th, cx, cy, R, t, v, o = {}) {
  if (R <= 1) return;
  const L = v[0] || 0;
  const seq = ['cookie12', 'sunny', 'flower', 'clover', 'burst', 'squircle', 'pebble', 'gem'];
  const k = (o.k ?? t * .38), i = ((Math.floor(k) % seq.length) + seq.length) % seq.length, m = eio3(k - Math.floor(k));
  const fill = o.fill || th.acc, ring = o.ring || th.acc;
  c.save();
  c.globalAlpha *= (o.a ?? 1);

  // Outer soft tonal aura morphing in counter-rotation
  c.save();
  c.globalAlpha *= 0.16 + 0.14 * clamp(L * 1.4);
  shape(c, ring, cx, cy, R * (1.36 + L * 0.16), seq[(i + 2) % seq.length], seq[(i + 3) % seq.length], m, -t * 0.22);
  c.restore();

  // 32-band radial spectrum pills
  for (let j = 0; j < 32; j++) {
    const a = j / 32 * TAU - Math.PI / 2;
    const b = v[1 + j] || 0;
    const r0 = R * 1.14;
    const len = Math.max(6, R * 0.05 + b * R * 0.44);
    c.beginPath();
    c.moveTo(cx + Math.cos(a) * r0, cy + Math.sin(a) * r0);
    c.lineTo(cx + Math.cos(a) * (r0 + len), cy + Math.sin(a) * (r0 + len));
    c.lineWidth = Math.max(5, R * 0.048);
    c.lineCap = 'round';
    c.strokeStyle = j % 2 === 0 ? ring : (o.ring2 || th.sec || ring);
    c.globalAlpha = (o.a ?? 1) * (0.28 + 0.72 * clamp(b * 1.45));
    c.stroke();
  }

  // Main M3 Expressive morphing core
  c.globalAlpha = o.a ?? 1;
  shape(c, fill, cx, cy, R * (1 + L * 0.085), seq[i], seq[(i + 1) % seq.length], m, t * 0.25);

  // Inner tonal ring
  shapeStroke(c, o.glyphCol || th.bg, Math.max(2, R * 0.022), cx, cy, R * 0.74, seq[(i + 1) % seq.length], seq[i], m, -t * 0.18);

  if (o.glyph !== false) {
    txt(c, 'स', cx, cy + R * 0.28, { sz: R * 0.88, w: 700, col: o.glyphCol || th.bg, al: 'center' });
  }
  if (o.badge) {
    const bw = tw(c, o.badge, 18, 700) + 28;
    box(c, cx - bw / 2, cy + R * 0.78, bw, 32, 16, o.badgeFill || th.accc, th.line, 1.5);
    txt(c, o.badge, cx, cy + R * 0.78 + 22, { sz: 18, w: 700, col: o.badgeCol || th.onacc, al: 'center', ls: 1 });
  }
  c.restore();
}

function wavebars(c, x, y, w, h, t, v, col, n = 48, col2 = null) {
  c.save();
  const g = w / n;
  for (let i = 0; i < n; i++) {
    const b = v[1 + Math.floor(i / n * 32)] || 0;
    const hh = Math.max(8, h * clamp(b * 1.22 + .06 + .06 * Math.sin(i * .65 + t * 6.5)));
    box(c, x + i * g, y - hh / 2, g * .58, hh, g * .29, (col2 && i % 3 === 0) ? col2 : col);
  }
  c.restore();
}

function shell(c, th, sc, t, T = 0) {
  c.fillStyle = th.bg;
  c.fillRect(0, 0, W, H);
  // Ambient M3 Expressive background geometry
  const v = V(T);
  c.save();
  c.globalAlpha = 0.055;
  shape(c, th.acc, 160, 180, 220 + v[0] * 30, 'flower', 'cookie12', Math.sin(T * 0.4) * .5 + .5, T * 0.08);
  shape(c, th.sec || th.acc, W - 150, H - 180, 260 + v[0] * 35, 'clover', 'sunny', Math.cos(T * 0.35) * .5 + .5, -T * 0.07);
  c.restore();
}

// Broadcast Chrome: Top Header Bar + Segmented Episode Progress + Ultra-Clean Compact Dual Subtitle Pill
function chrome(c, th, T, tl, hide) {
  const a = hide ? eo3(P(T, 2.0, 2.7)) : 1;
  if (a <= 0) return;
  const v = V(T);
  let scIdx = tl.scenes.findIndex(s => T >= s.t0 && T < s.t1);
  if (scIdx < 0) scIdx = tl.scenes.length - 1;
  const sc = tl.scenes[scIdx];
  const meta = SC_META[sc.id] || { tag: 'NEBIANS NEWS', ne: 'समाचार', ic: 'bolt' };

  c.save();
  c.globalAlpha = a;

  // 1. Top-Left Brand + Active Desk Pill
  const isLight = (th.id === 'light' || th.id === 'plum');
  const barFill = isLight ? 'rgba(255,255,255,0.92)' : (th.id === 'blue' ? 'rgba(8,52,145,0.90)' : 'rgba(20,25,36,0.90)');
  box(c, 54, 26, 468, 50, 25, barFill, th.line, 1.5);
  shape(c, th.acc, 82, 51, 15 + v[0] * 3, 'cookie12', 'sunny', Math.sin(T * 2) * .5 + .5, T * 0.8);
  txt(c, 'NEBians News', 106, 58, { sz: 21, w: 700, col: th.fg });
  box(c, 268, 33, 246, 36, 18, th.accc, null);
  icon(c, meta.ic, 290, 51, 18, th.onacc, 2.2);
  txt(c, meta.tag, 306, 57, { sz: 16, w: 700, col: th.onacc, ls: 1.2 });

  // 2. Top-Center 11-Segment Chapter Progress Strip
  const pX = 548, pW = 740, pY = 45, nSc = tl.scenes.length, gap = 6;
  const segW = (pW - gap * (nSc - 1)) / nSc;
  for (let i = 0; i < nSc; i++) {
    const s = tl.scenes[i];
    const sx = pX + i * (segW + gap);
    box(c, sx, pY, segW, 12, 6, isLight ? th.high : th.soft, null);
    if (T >= s.t1) {
      box(c, sx, pY, segW, 12, 6, th.acc, null);
    } else if (T >= s.t0) {
      const f = clamp((T - s.t0) / Math.max(0.1, s.t1 - s.t0));
      box(c, sx, pY, Math.max(12, segW * f), 12, 6, th.acc, null);
    }
  }

  // 3. Top-Right Date Window & Voice Telemetry Pill
  const rW = 556, rX = W - 54 - rW;
  box(c, rX, 26, rW, 50, 25, barFill, th.line, 1.5);
  wavebars(c, rX + 20, 51, 68, 24, T, v, th.acc, 10, th.sec);
  txt(c, 'SAKHIYA-ONE V2', rX + 100, 57, { sz: 16, w: 700, col: th.acc, ls: 1 });
  box(c, rX + 272, 34, 2, 34, 1, th.line);
  txt(c, 'FRI 27 NOV  ·  21–27 NOV RECAP', rX + 290, 57, { sz: 16, w: 600, col: th.mu, ls: 0.6 });

  // 4. Compact, Non-Cluttering Bilingual Subtitle Pill at Bottom
  const ln = curLine(tl, T);
  if (ln) caption(c, th, ln, T, v);
  c.restore();
}

function curLine(tl, T) {
  let r = null;
  for (const s of tl.scenes) {
    for (const l of s.lines) {
      if (T >= l.t - .12 && T < l.t + l.d + .38) r = l;
    }
  }
  return r;
}

// Compact, Minimal, Non-Hindering Dual-Language Subtitle Pill (only 62px tall at very bottom edge)
function caption(c, th, l, T, v) {
  const k = clamp((T - l.t) / Math.max(0.1, l.d));
  const ws = l.ne.split(' ');
  const lens = ws.map(x => x.length + 1.5);
  const tot = lens.reduce((a, b) => a + b, 0);
  const a = eo3(P(T, l.t - .12, l.t + .12)) * (1 - eo3(P(T, l.t + l.d + .08, l.t + l.d + .38)));
  if (a <= 0.01) return;

  c.save();
  c.globalAlpha *= a;

  const neSz = 22, enSz = 15;
  const neW = tw(c, l.ne, neSz, 600);
  const enW = tw(c, l.en, enSz, 500);
  const bw = clamp(Math.max(neW, enW) + 110, 560, 1560);
  const bh = 62;
  const bx = W / 2 - bw / 2;
  const by = H - 16 - bh + (1 - a) * 14;

  // Sleek dark translucent-feel tonal pill so subtitles never clash with light or dark cards
  const pillBg = (th.id === 'light' || th.id === 'plum') ? '#0d152c' : (th.id === 'blue' ? '#042769' : '#121723');
  const pillBorder = (th.id === 'light' || th.id === 'plum') ? '#263866' : (th.id === 'blue' ? '#2c67db' : '#2c364f');
  box(c, bx, by, bw, bh, bh / 2, pillBg, pillBorder, 1.5);

  // Subtle voice-reactive indicator dot on left of pill
  shape(c, '#5ce6d0', bx + 28, by + bh / 2, 8 + (v ? v[0] * 3.5 : 0), 'cookie12', 'sunny', k, T * 1.4);

  // Line 1: Compact Nepali with word-level active highlight
  let acc = 0;
  let cx = W / 2 - neW / 2 + 10;
  ws.forEach((word, wi) => {
    const s0 = acc / tot;
    const on = k >= s0 - .03;
    txt(c, word, cx, by + 26, {
      sz: neSz,
      w: 600,
      col: on ? '#ffffff' : '#899bc2'
    });
    cx += tw(c, word + ' ', neSz, 600);
    acc += lens[wi];
  });

  // Line 2: Compact English subtitle
  txt(c, l.en, W / 2 + 10, by + 49, {
    sz: enSz,
    w: 500,
    col: '#7fd8ff',
    al: 'center',
    ls: 0.3
  });

  c.restore();
}

function wipe(c, T, t0, col, dur = .88, ox = W / 2, oy = H / 2, kind = 'flower') {
  const p = eio3(P(T, t0, t0 + dur));
  if (p <= 0 || p >= 1) return false;
  c.save();
  shapePath(c, ox, oy, 1550 * p + 1, 'circle', kind, clamp(p * 1.5), p * 1.1);
  c.clip();
  return true;
}
