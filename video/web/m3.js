// NEBians News — Material 3 Expressive core: tonal palettes, spring motion, morphing shapes, type helpers
const W = 1920, H = 1080, TAU = Math.PI * 2;
const clamp = (x, a = 0, b = 1) => Math.max(a, Math.min(b, x));
const lerp = (a, b, t) => a + (b - a) * t;
const P = (t, a, b) => clamp((t - a) / Math.max(1e-6, b - a));
const eo3 = x => 1 - Math.pow(1 - clamp(x), 3);
const eio = x => { x = clamp(x); return x < .5 ? 4 * x * x * x : 1 - Math.pow(-2 * x + 2, 3) / 2; };
const eoback = x => { x = clamp(x); return 1 + 2.2 * Math.pow(x - 1, 3) + 1.2 * Math.pow(x - 1, 2); };
// M3 Expressive motion: spatial spring (slight overshoot) and effects spring (no overshoot)
const spr = x => x <= 0 ? 0 : x >= 1 ? 1 : 1 - Math.exp(-7 * x) * Math.cos(11 * x);
const sprS = x => x <= 0 ? 0 : x >= 1 ? 1 : 1 - Math.exp(-9 * x) * (1 + 9 * x);
const sp = (t, a, d = .7) => spr(P(t, a, a + d));
const ss = (t, a, d = .6) => sprS(P(t, a, a + d));

// ---- color ----
function hsl(h, s, l) {
  h = ((h % 360) + 360) % 360; s /= 100; l /= 100;
  const k = n => (n + h / 30) % 12, a = s * Math.min(l, 1 - l);
  const f = n => l - a * Math.max(-1, Math.min(k(n) - 3, Math.min(9 - k(n), 1)));
  return [Math.round(255 * f(0)), Math.round(255 * f(8)), Math.round(255 * f(4))];
}
const mixc = (a, b, t) => a.map((v, i) => Math.round(lerp(v, b[i], clamp(t))));
const C = (x, a = 1) => Array.isArray(x) ? `rgba(${x[0]},${x[1]},${x[2]},${a})` : x;
// tonal palette from one seed hue (M3 roles)
function palette(h, dark) {
  const s2 = h + 38, t3 = h + 150;
  return dark ? {
    dark, bg: hsl(h, 40, 8), surf: hsl(h, 32, 12), cont: hsl(h, 30, 17), high: hsl(h, 30, 23),
    pri: hsl(h, 88, 76), onPri: hsl(h, 80, 14), priC: hsl(h, 52, 27), onPriC: hsl(h, 90, 90),
    sec: hsl(s2, 62, 78), secC: hsl(s2, 36, 25), ter: hsl(t3, 62, 76), terC: hsl(t3, 36, 23),
    ink: hsl(h, 25, 95), mute: hsl(h, 14, 72), line: hsl(h, 16, 32)
  } : {
    dark, bg: hsl(h, 55, 97), surf: hsl(h, 45, 94), cont: hsl(h, 50, 91), high: hsl(h, 50, 86),
    pri: hsl(h, 72, 42), onPri: [255, 255, 255], priC: hsl(h, 92, 88), onPriC: hsl(h, 80, 14),
    sec: hsl(s2, 48, 42), secC: hsl(s2, 70, 87), ter: hsl(t3, 56, 40), terC: hsl(t3, 72, 86),
    ink: hsl(h, 32, 11), mute: hsl(h, 12, 36), line: hsl(h, 22, 78)
  };
}

// ---- morphable M3E shapes (equal vertex count so any pair can morph) ----
const N = 160;
const SH = {
  circle: () => 1,
  cookie9: a => .90 + .075 * Math.cos(9 * a),
  cookie12: a => .91 + .065 * Math.cos(12 * a),
  sunny: a => .85 + .115 * Math.cos(8 * a),
  flower: a => .78 + .19 * Math.cos(6 * a),
  clover: a => .76 + .21 * Math.cos(4 * a),
  burst: a => .82 + .14 * Math.cos(10 * a) + .03 * Math.cos(20 * a),
  squircle: a => .86 * Math.pow(Math.pow(Math.abs(Math.cos(a)), 4.2) + Math.pow(Math.abs(Math.sin(a)), 4.2), -.238),
  pebble: a => .90 + .06 * Math.cos(a - .5) + .05 * Math.cos(2 * a + 1) + .03 * Math.cos(3 * a),
  gem: a => .86 * Math.pow(Math.pow(Math.abs(Math.cos(a)), 1.7) + Math.pow(Math.abs(Math.sin(a)), 1.7), -.588),
  penta: a => .93 + .06 * Math.cos(5 * a)
};
const SHR = {};
for (const k in SH) SHR[k] = Array.from({ length: N }, (_, i) => SH[k](i / N * TAU));

function shapePath(c, cx, cy, R, a, b = a, m = 0, rot = 0, sx = 1, sy = 1) {
  const A = SHR[a] || SHR.circle, B = SHR[b] || SHR.circle, p = [], mm = clamp(m);
  for (let i = 0; i < N; i++) {
    const r = Math.max(.1, R) * lerp(A[i], B[i], mm), t = i / N * TAU + rot;
    p.push([cx + Math.cos(t) * r * sx, cy + Math.sin(t) * r * sy]);
  }
  c.beginPath();
  c.moveTo((p[0][0] + p[N - 1][0]) / 2, (p[0][1] + p[N - 1][1]) / 2);
  for (let i = 0; i < N; i++) { const q = p[(i + 1) % N]; c.quadraticCurveTo(p[i][0], p[i][1], (p[i][0] + q[0]) / 2, (p[i][1] + q[1]) / 2); }
  c.closePath();
}
function shape(c, fill, cx, cy, R, a, b = a, m = 0, rot = 0, sx = 1, sy = 1) {
  if (R <= .3) return; shapePath(c, cx, cy, R, a, b, m, rot, sx, sy); c.fillStyle = C(fill); c.fill();
}
function shapeLine(c, stroke, lw, cx, cy, R, a, b = a, m = 0, rot = 0) {
  if (R <= .3) return; shapePath(c, cx, cy, R, a, b, m, rot); c.strokeStyle = C(stroke); c.lineWidth = lw; c.stroke();
}
function rr(c, x, y, w, h, r) { c.beginPath(); if (w > 0 && h > 0) c.roundRect(x, y, w, h, Math.min(r, w / 2, h / 2)); }
function box(c, x, y, w, h, r, fill, stroke, lw = 2) {
  if (w <= 0 || h <= 0) return; rr(c, x, y, w, h, r);
  if (fill) { c.fillStyle = C(fill); c.fill(); } if (stroke) { c.strokeStyle = C(stroke); c.lineWidth = lw; c.stroke(); }
}

// ---- type ----
const F = (sz, w = 600) => `${w} ${Math.round(sz)}px Poppins,'Noto Sans Devanagari',sans-serif`;
function txt(c, s, x, y, o = {}) {
  if (s == null || s === '') return;
  c.save(); c.font = F(o.sz || 36, o.w || 600); c.fillStyle = C(o.col || '#000', o.a ?? 1);
  c.textAlign = o.al || 'left'; c.textBaseline = o.bl || 'alphabetic'; if (o.ls) c.letterSpacing = o.ls + 'px';
  c.fillText(String(s), x, y); c.restore();
}
const tw = (c, s, sz, w = 600) => { c.save(); c.font = F(sz, w); const m = c.measureText(String(s)).width; c.restore(); return m; };
function wrap(c, s, sz, w, maxW) {
  const out = []; let cur = '';
  for (const wd of String(s).split(' ')) {
    const t = cur ? cur + ' ' + wd : wd;
    if (tw(c, t, sz, w) > maxW && cur) { out.push(cur); cur = wd; } else cur = t;
  }
  if (cur) out.push(cur); return out;
}
// kinetic words: each word springs up out of a clip; returns baseline of last line
function kinetic(c, s, x, y, t, o = {}) {
  const sz = o.sz || 60, w = o.w || 700, lh = o.lh || sz * 1.18, st = o.st ?? .07, lines = wrap(c, s, sz, w, o.maxW || 900);
  let k = 0;
  lines.forEach((ln, li) => {
    const wd = ln.split(' '), tot = tw(c, ln, sz, w);
    let cx = o.al === 'center' ? x - tot / 2 : x;
    wd.forEach(word => {
      const p = eoback(P(t, k * st, k * st + .55)), a = clamp(P(t, k * st, k * st + .22)), ww = tw(c, word, sz, w);
      c.save(); c.beginPath(); c.rect(cx - 14, y + li * lh - sz * 1.12, ww + 40, lh * 1.5); c.clip();
      txt(c, word, cx, y + li * lh + (1 - p) * sz * .9, { sz, w, col: o.col, a: a * (o.a ?? 1) });
      c.restore(); cx += tw(c, word + ' ', sz, w); k++;
    });
  });
  return { last: y + (lines.length - 1) * lh, n: lines.length };
}
// fit 1-2 line text into maxW; returns {lines:[...], sz}
function fitText(c, s, maxW, maxSz, w = 700, minSz = 40) {
  let sz = maxSz; while (sz > minSz && tw(c, s, sz, w) > maxW) sz -= 2;
  if (tw(c, s, sz, w) <= maxW || !s.includes(' ')) return { lines: [s], sz };
  const wd = s.split(' '); let best = null;
  for (let i = 1; i < wd.length; i++) {
    const a = wd.slice(0, i).join(' '), b = wd.slice(i).join(' ');
    const m = Math.max(tw(c, a, 100, w), tw(c, b, 100, w)); if (!best || m < best.m) best = { a, b, m };
  }
  let z = Math.min(maxSz * .8, 100); while (z > 30 && Math.max(tw(c, best.a, z, w), tw(c, best.b, z, w)) > maxW) z -= 2;
  return { lines: [best.a, best.b], sz: z };
}

// ---- line icons (24-grid, stroked) ----
const ICON = {
  mic: 'M12 3a3 3 0 0 1 3 3v5a3 3 0 0 1-6 0V6a3 3 0 0 1 3-3z M6 11a6 6 0 0 0 12 0 M12 17v4 M9 21h6',
  lang: 'M4 6h9 M8.5 4v2 M6 6c0 4 3 7 6 8 M11 6c0 3-2 6-6 8 M13 20l4-9 4 9 M14.5 17h5',
  chip: 'M7 7h10v10H7z M10 10h4v4h-4z M9 3v4 M15 3v4 M9 17v4 M15 17v4 M3 9h4 M3 15h4 M17 9h4 M17 15h4',
  device: 'M4 5h16v11H4z M2 20h20',
  sprout: 'M12 21v-9 M12 12c0-4-3-6-7-6 0 4 3 6 7 6 M12 14c0-3 2-5 6-5 0 3-2 5-6 5',
  trend: 'M3 17l6-6 4 4 8-8 M15 7h6v6',
  cal: 'M4 6h16v14H4z M4 10h16 M8 3v4 M16 3v4',
  spark: 'M12 3c1 5 4 8 9 9-5 1-8 4-9 9-1-5-4-8-9-9 5-1 8-4 9-9z',
  news: 'M5 4h11v16H5z M16 8h4v10a2 2 0 0 1-4 0 M8 8h5 M8 12h5 M8 16h3',
  school: 'M2 9l10-5 10 5-10 5z M6 11v5c3 3 9 3 12 0v-5',
  check: 'M5 12l5 5 9-10',
  globe: 'M12 3a9 9 0 1 0 0 18 9 9 0 0 0 0-18z M3 12h18 M12 3c3 3 3 15 0 18 M12 3c-3 3-3 15 0 18',
  shield: 'M12 3l8 3v6c0 5-3.5 8-8 9-4.5-1-8-4-8-9V6z',
  rocket: 'M5 19c1-4 3-6 6-8 M14 4c4 0 6 2 6 6-2 4-5 7-9 8l-5-5c1-4 4-7 8-9z M15 9h.01'
};
const ICP = {}; for (const k in ICON) ICP[k] = new Path2D(ICON[k]);
function icon(c, name, cx, cy, size, col, lw = 1.9) {
  c.save(); c.translate(cx - size / 2, cy - size / 2); c.scale(size / 24, size / 24);
  c.strokeStyle = C(col); c.lineWidth = lw; c.lineCap = 'round'; c.lineJoin = 'round'; c.stroke(ICP[name]); c.restore();
}
