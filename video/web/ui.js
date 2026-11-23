// UI kit: icons, chips, voice visuals, caption overlay, backgrounds
const IC = { home: 'M3 11 L12 3 L21 11 M5 10 V21 H19 V10', search: 'M10.5 3a7.5 7.5 0 1 0 0 15a7.5 7.5 0 1 0 0-15 M16 16 L21 21',
  check: 'M4 12.5 L9.5 18 L20 6.5', info: 'M12 2.5a9.5 9.5 0 1 0 0 19a9.5 9.5 0 1 0 0-19 M12 11 V17 M12 7.5 V8', warn: 'M12 3 L22 20 H2 Z M12 10 V14.5 M12 17 V17.5',
  x: 'M5 5 L19 19 M19 5 L5 19', shield: 'M12 2.5 L20 5.5 V12 C20 17 16.5 20.5 12 22 C7.5 20.5 4 17 4 12 V5.5 Z', lock: 'M6 11 H18 V21 H6 Z M8.5 11 V7.5 a3.5 3.5 0 0 1 7 0 V11',
  doc: 'M6 2.5 H14 L19 7.5 V21.5 H6 Z M14 2.5 V7.5 H19 M9 13 H16 M9 17 H16', book: 'M4 4.5 C7 3.5 10 4 12 6 C14 4 17 3.5 20 4.5 V19.5 C17 18.5 14 19 12 21 C10 19 7 18.5 4 19.5 Z M12 6 V21',
  compass: 'M12 2.5a9.5 9.5 0 1 0 0 19a9.5 9.5 0 1 0 0-19 M15.5 8.5 L13.5 13.5 L8.5 15.5 L10.5 10.5 Z', user: 'M12 4 a4 4 0 1 0 0 8 a4 4 0 1 0 0-8 M4 21 C4.5 16.5 8 14.5 12 14.5 C16 14.5 19.5 16.5 20 21',
  bolt: 'M13 2 L4 14 H11 L10 22 L20 9.5 H13 Z', chip: 'M7 7 H17 V17 H7 Z M10 3 V7 M14 3 V7 M10 17 V21 M14 17 V21 M3 10 H7 M3 14 H7 M17 10 H21 M17 14 H21',
  wallet: 'M3 7 H19 V19 H3 Z M3 7 L16 3 V7 M15 13 H19', pause: 'M8 5 V19 M16 5 V19', globe: 'M12 2.5a9.5 9.5 0 1 0 0 19a9.5 9.5 0 1 0 0-19 M3 12 H21 M12 3 C8 8 8 16 12 21 C16 16 16 8 12 3',
  play: 'M8 5 L19 12 L8 19 Z', phone: 'M7 2.5 H17 V21.5 H7 Z M11 18.5 H13', filter: 'M3 5 H21 L14 13 V20 L10 18 V13 Z', glass: 'M5 12a7 7 0 1 0 14 0a7 7 0 1 0-14 0 M8 12a4 4 0 1 0 8 0a4 4 0 1 0-8 0' };
const IP = {}; for (const k in IC) IP[k] = new Path2D(IC[k]);
function icon(c, n, x, y, s, col, lw = 2, fill) { c.save(); c.translate(x - s / 2, y - s / 2); c.scale(s / 24, s / 24); c.lineWidth = lw; c.lineCap = c.lineJoin = 'round';
  c.strokeStyle = col; if (fill) { c.fillStyle = fill; c.fill(IP[n]); } c.stroke(IP[n]); c.restore(); }
function chip(c, s, x, y, o = {}) { const sz = o.sz || 26, w = tw(c, s, sz, o.w || 600) + (o.pad || 44) + (o.ic ? sz + 12 : 0), h = sz * 2.0;
  c.save(); c.globalAlpha *= o.a ?? 1; box(c, x, y, w, h, 28, o.fill || C.pc, o.line); const tx = x + (o.pad || 44) / 2;
  if (o.ic) icon(c, o.ic, tx + sz / 2, y + h / 2, sz, o.col || C.onpc, 2.2);
  txt(c, s, tx + (o.ic ? sz + 12 : 0), y + h / 2 + sz * .36, { sz, w: o.w || 600, col: o.col || C.onpc }); c.restore(); return w; }
// theme: light | dark | blue
const TH = { light: { bg: C.bg, fg: C.ink, mu: C.mute, card: C.white, line: C.line, soft: C.low, acc: C.blue, accc: C.pc, onacc: C.onpc, dot: '#d3dbef' },
  dark: { bg: C.gr, fg: C.onGr, mu: C.muteGr, card: C.gr1, line: C.grLine, soft: C.gr2, acc: C.acc, accc: C.gr3, onacc: C.acc, dot: '#2a2a33' },
  blue: { bg: C.blue, fg: C.white, mu: '#c9d7ff', card: '#0a4cc0', line: '#2f74ee', soft: '#0a4cc0', acc: C.white, accc: '#1b66e8', onacc: C.white, dot: '#1a68ea' } };
function dotgrid(c, th, t, a = 1) { c.save(); c.globalAlpha *= a; c.fillStyle = th.dot; const g = 64;
  for (let y = 0; y <= H + g; y += g) for (let x = 0; x <= W + g; x += g) { const d = Math.hypot(x - W / 2, y - H / 2), w = Math.sin(d / 140 - t * 1.6) * .5 + .5;
    c.beginPath(); c.arc(x, y, 2 + 3.2 * w, 0, TAU); c.fill(); } c.restore(); }
function V(t) { const E = window.ENV, f = clamp(t * 30, 0, E.length - 2), i = Math.floor(f), k = f - i, a = E[i], b = E[i + 1];
  return a.map((v, j) => lerp(v, b[j], k)); }
function sakhiya(c, th, cx, cy, R, t, v, o = {}) {  // voice-reactive avatar: morphing shape + 32 band ring
  const L = v[0], seq = ['cookie', 'sunny', 'flower', 'clover', 'burst', 'squircle', 'pebble'], k = (o.k ?? t * .35), i = Math.floor(k) % 7, m = eio3(k % 1);
  const fill = o.fill || th.acc, rot = t * .25; c.save(); c.globalAlpha *= o.a ?? 1;
  for (let j = 0; j < 32; j++) { const a = j / 32 * TAU - Math.PI / 2, b = v[1 + j] || 0, r0 = R * 1.18, len = 10 + b * R * .42;
    c.beginPath(); c.moveTo(cx + Math.cos(a) * r0, cy + Math.sin(a) * r0); c.lineTo(cx + Math.cos(a) * (r0 + len), cy + Math.sin(a) * (r0 + len));
    c.lineWidth = Math.max(6, R * .055); c.lineCap = 'round'; c.strokeStyle = o.ring || th.acc; c.globalAlpha = (o.a ?? 1) * (.25 + .75 * clamp(b * 1.4)); c.stroke(); }
  c.globalAlpha = o.a ?? 1; shape(c, fill, cx, cy, R * (1 + L * .08), seq[i], seq[(i + 1) % 7], m, rot);
  if (o.glyph !== false) txt(c, 'स', cx, cy + R * .27, { sz: R * .9, w: 700, col: o.glyphCol || th.bg, al: 'center' }); c.restore(); }
function wavebars(c, x, y, w, h, t, v, col, n = 48) { c.save(); c.fillStyle = col; const g = w / n;
  for (let i = 0; i < n; i++) { const b = v[1 + Math.floor(i / n * 32)] || 0, hh = Math.max(8, h * clamp(b * 1.2 + .05 + .06 * Math.sin(i * .7 + t * 6)));
    box(c, x + i * g, y - hh / 2, g * .56, hh, g * .28, col); } c.restore(); }
function shell(c, th, sc, t) { c.fillStyle = th.bg; c.fillRect(0, 0, W, H); }
// chrome: brand chip, episode chip, caption bar with karaoke highlight
function chrome(c, th, T, tl, hide) {
  const a = hide ? 0 : eo3(P(T, 3.2, 3.9)); if (a <= 0) return; c.save(); c.globalAlpha = a;
  const light = th === TH.light; chip(c, 'NEBians News', 64, 48, { fill: light ? C.pc : th.accc, col: light ? C.onpc : th.fg, ic: 'bolt', sz: 24 });
  const d = tl.date; const w = tw(c, 'EP 01  |  ' + d, 24, 600) + 44; chip(c, 'EP 01  |  ' + d, W - 64 - w, 48, { fill: th.soft, col: th.mu, line: th.line, sz: 24 });
  const ln = curLine(tl, T); if (ln) caption(c, th, ln, T); c.restore(); }
function curLine(tl, T) { let r = null; for (const s of tl.scenes) for (const l of s.lines) if (T >= l.t - .15 && T < l.t + l.d + .6) r = l; return r; }
function caption(c, th, l, T) {
  const k = clamp((T - l.t) / l.d), ws = l.ne.split(' '), lens = ws.map(x => x.length + 2), tot = lens.reduce((a, b) => a + b, 0), a = eo3(P(T, l.t - .15, l.t + .15)) * (1 - eo3(P(T, l.t + l.d + .1, l.t + l.d + .55)));
  c.save(); c.globalAlpha *= a; const ml = wrap(c, l.ne, 44, 600, 1500), bh = 64 + ml.length * 62 + 44, y = H - 52 - bh + (1 - a) * 30, bw = 1640;
  box(c, W / 2 - bw / 2, y, bw, bh, 44, th === TH.dark ? C.gr1 : (th === TH.blue ? '#003fa6' : C.white), th === TH.blue ? '#2f74ee' : th.line);
  let acc = 0, wi = 0;
  ml.forEach((ln, li) => { const parts = ln.split(' '), w0 = tw(c, ln, 44, 600); let x = W / 2 - w0 / 2;
    parts.forEach(p => { const s0 = acc / tot, on = k >= s0 - .02; txt(c, p, x, y + 30 + 62 * (li + 1) - 8, { sz: 44, w: 600, col: on ? (th === TH.light ? C.ink : th.fg) : (th === TH.light ? '#9aa6c4' : th.mu) });
      x += tw(c, p + ' ', 44, 600); acc += lens[wi++]; }); });
  txt(c, l.en, W / 2, y + bh - 28, { sz: 25, w: 500, col: th === TH.blue ? '#c9d7ff' : th.mu, al: 'center' }); c.restore(); }
function wipe(c, T, t0, col, dur = .9, ox = W / 2, oy = H / 2, kind = 'flower') { const p = eio3(P(T, t0, t0 + dur)); if (p <= 0 || p >= 1) return false;
  c.save(); shapePath(c, ox, oy, 1500 * p + 1, 'circle', kind, clamp(p * 1.6), p * 1.2); c.clip(); return true; }
