// M3X motion engine: tokens, easing, springs, shape morphing, text. No shadows, no gradients, no emoji.
const W = 1920, H = 1080;
const C = { blue: '#0058df', white: '#ffffff', pc: '#dbe4ff', onpc: '#001a52', bg: '#f6f8ff', ink: '#0b1530', mute: '#414b66',
  line: '#c6d0ea', low: '#f0f4fe', cont: '#e9eefb', high: '#e2e8f7', highest: '#dae1f3', azure: '#7aa2ff', azurec: '#d7e3ff',
  azured: '#0047b8', sky: '#5cc8ff', skyc: '#cdeeff', aqua: '#55d9c6', aquac: '#c6f3eb', aquad: '#0a6a62', peric: '#e3e9ff',
  navy: '#06235f', err: '#ba1a1a', errc: '#ffdad6', amber: '#ffd88a', amberd: '#5a3d00', gr: '#121318', gr1: '#1b1b21',
  gr2: '#25252d', gr3: '#30303a', onGr: '#e4e2e9', muteGr: '#a9a8b4', grLine: '#3b3b47', acc: '#b2c5ff' };
const clamp = (x, a = 0, b = 1) => Math.max(a, Math.min(b, x));
const lerp = (a, b, t) => a + (b - a) * t;
const P = (t, a, b) => clamp((t - a) / (b - a));
const eo3 = x => 1 - Math.pow(1 - x, 3), eio3 = x => x < .5 ? 4 * x * x * x : 1 - Math.pow(-2 * x + 2, 3) / 2;
const eoexp = x => x >= 1 ? 1 : 1 - Math.pow(2, -10 * x), eoback = x => 1 + 2.4 * Math.pow(x - 1, 3) + 1.4 * Math.pow(x - 1, 2);
const spring = x => x <= 0 ? 0 : x >= 1 ? 1 : 1 - Math.exp(-7 * x) * Math.cos(11 * x);   // overshoots, settles
const sp = (t, a, d = .7) => spring(P(t, a, a + d));
const mix = (c1, c2, t) => { const h = s => [1, 3, 5].map(i => parseInt(s.slice(i, i + 2), 16)); const a = h(c1), b = h(c2);
  return '#' + a.map((v, i) => Math.round(lerp(v, b[i], t)).toString(16).padStart(2, '0')).join(''); };
const rnd = n => { const x = Math.sin(n * 127.1 + 311.7) * 43758.5453; return x - Math.floor(x); };
// ---- shapes: identical point counts so every pair morphs ----
const N = 160, TAU = Math.PI * 2;
const SH = { circle: () => 1, cookie: a => .9 + .07 * Math.cos(9 * a), sunny: a => .84 + .12 * Math.cos(12 * a),
  flower: a => .76 + .21 * Math.cos(6 * a), clover: a => .72 + .25 * Math.cos(4 * a), burst: a => .8 + .17 * Math.cos(10 * a) + .04 * Math.cos(20 * a),
  squircle: a => .84 * Math.pow(Math.pow(Math.abs(Math.cos(a)), 4) + Math.pow(Math.abs(Math.sin(a)), 4), -.25),
  pebble: a => .9 + .06 * Math.cos(a - .5) + .05 * Math.cos(2 * a + 1) + .03 * Math.cos(3 * a) };
const SHR = {}; for (const k in SH) SHR[k] = Array.from({ length: N }, (_, i) => SH[k](i / N * TAU));
function shapePath(c, cx, cy, R, a, b = a, m = 0, rot = 0, sx = 1, sy = 1) {
  const A = SHR[a], B = SHR[b], p = [];
  for (let i = 0; i < N; i++) { const r = R * lerp(A[i], B[i], m), t = i / N * TAU + rot; p.push([cx + Math.cos(t) * r * sx, cy + Math.sin(t) * r * sy]); }
  c.beginPath(); c.moveTo((p[0][0] + p[N - 1][0]) / 2, (p[0][1] + p[N - 1][1]) / 2);
  for (let i = 0; i < N; i++) { const q = p[(i + 1) % N]; c.quadraticCurveTo(p[i][0], p[i][1], (p[i][0] + q[0]) / 2, (p[i][1] + q[1]) / 2); }
  c.closePath();
}
function shape(c, fill, cx, cy, R, a, b, m, rot, sx, sy) { shapePath(c, cx, cy, R, a, b, m, rot, sx, sy); c.fillStyle = fill; c.fill(); }
const rr = (c, x, y, w, h, r) => { r = Math.min(r, w / 2, h / 2); c.beginPath(); c.roundRect(x, y, w, h, r); };
function box(c, x, y, w, h, r, fill, stroke, lw = 1.5) { rr(c, x, y, w, h, r); if (fill) { c.fillStyle = fill; c.fill(); } if (stroke) { c.strokeStyle = stroke; c.lineWidth = lw; c.stroke(); } }
// ---- text ----
const F = (sz, w = 600) => `${w} ${sz}px Poppins,'Noto Sans Devanagari',sans-serif`;
function txt(c, s, x, y, o = {}) {
  c.save(); c.font = F(o.sz || 40, o.w || 600); c.fillStyle = o.col || C.ink; c.textAlign = o.al || 'left'; c.textBaseline = o.bl || 'alphabetic';
  c.globalAlpha *= o.a ?? 1; if (o.ls) c.letterSpacing = o.ls + 'px'; c.fillText(s, x, y); c.restore();
}
const tw = (c, s, sz, w = 600) => { c.save(); c.font = F(sz, w); const m = c.measureText(s).width; c.restore(); return m; };
function wrap(c, s, sz, w, maxW) { c.save(); c.font = F(sz, w); const out = []; let cur = '';
  for (const wd of s.split(' ')) { const t = cur ? cur + ' ' + wd : wd; if (c.measureText(t).width > maxW && cur) { out.push(cur); cur = wd; } else cur = t; }
  c.restore(); if (cur) out.push(cur); return out; }
// word-by-word rise-in; returns bottom y. t is time since block start
function words(c, s, x, y, t, o = {}) {
  const sz = o.sz || 56, w = o.w || 700, lh = o.lh || sz * 1.18, st = o.st ?? .07, ls = wrap(c, s, sz, w, o.maxW || 1200);
  let k = 0; ls.forEach((ln, li) => { const wd = ln.split(' '); const tot = tw(c, ln, sz, w); let cx = o.al === 'center' ? x - tot / 2 : x;
    wd.forEach(word => { const p = eoback(P(t, k * st, k * st + .55)), a = clamp(P(t, k * st, k * st + .25));
      c.save(); c.beginPath(); c.rect(cx - 10, y + li * lh - sz * 1.05, tw(c, word, sz, w) + 30, lh * 1.4); c.clip();
      txt(c, word, cx, y + li * lh + (1 - p) * sz * .9, { sz, w, col: o.col, a: a * (o.a ?? 1) }); c.restore();
      cx += tw(c, word + ' ', sz, w); k++; }); });
  return y + (ls.length - 1) * lh;
}
function ease(name, x) { return ({ o3: eo3, io3: eio3, back: eoback, exp: eoexp, sp: spring })[name](x); }
