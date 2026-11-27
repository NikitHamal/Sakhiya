// Material 3 Expressive (M3X) Motion Engine: Tonal Palettes, Coupled Spring Physics, 12-Shape Morphing, Kinetic Typography
const SC = {};
const W = 1920, H = 1080;

const C = {
  blue: '#0052d4', blueDark: '#003999', blueCont: '#1148b8', blueHigh: '#1d57cc',
  white: '#ffffff', pc: '#d8e4ff', onpc: '#001a52',
  bg: '#f4f7ff', ink: '#0a142e', mute: '#434f6b', line: '#c5d1ec',
  low: '#edf2fe', cont: '#e4ecfc', high: '#dae4f8', highest: '#cfdbf4',
  azure: '#6ea0ff', azurec: '#d6e3ff', azured: '#0043b0',
  sky: '#4cc3ff', skyc: '#cceeff',
  aqua: '#3be0c8', aquac: '#bdf7ed', aquad: '#065f56',
  mint: '#10b981', mintc: '#d1fae5', mintd: '#065f46',
  violet: '#6d28d9', violetc: '#ede9fe', violetd: '#3b0764',
  rose: '#e11d48', rosec: '#ffe4e6', rosed: '#881337',
  err: '#ba1a1a', errc: '#ffdad6',
  amber: '#fbbf24', amberc: '#fef3c7', amberd: '#78350f',
  gr: '#0d1017', gr1: '#161a24', gr2: '#1f2533', gr3: '#2a3245',
  onGr: '#eef0f8', muteGr: '#9ba4be', grLine: '#313a50', acc: '#9ebbff',
  tealBg: '#071f1e', teal1: '#0d2e2c', teal2: '#143f3c', tealLine: '#235c57', tealAcc: '#52f2dc',
  plumBg: '#f6f3ff', plumCard: '#ffffff', plumSoft: '#ede6ff', plumLine: '#cec0f5', plumInk: '#1b103a', plumMute: '#56497a', plumAcc: '#5b21b6'
};

const clamp = (x, a = 0, b = 1) => Math.max(a, Math.min(b, x));
const lerp = (a, b, t) => a + (b - a) * t;
const P = (t, a, b) => clamp((t - a) / Math.max(1e-5, b - a));
const eo3 = x => 1 - Math.pow(1 - clamp(x), 3);
const eio3 = x => { x = clamp(x); return x < .5 ? 4 * x * x * x : 1 - Math.pow(-2 * x + 2, 3) / 2; };
const eoexp = x => x >= 1 ? 1 : 1 - Math.pow(2, -10 * clamp(x));
const eoback = x => { x = clamp(x); return 1 + 2.2 * Math.pow(x - 1, 3) + 1.2 * Math.pow(x - 1, 2); };

// M3 Expressive physics springs
const spring = x => x <= 0 ? 0 : x >= 1 ? 1 : 1 - Math.exp(-6.8 * x) * Math.cos(10.5 * x);
const springSoft = x => x <= 0 ? 0 : x >= 1 ? 1 : 1 - Math.exp(-8.5 * x) * (1 + 8.5 * x);
const sp = (t, a, d = .72) => spring(P(t, a, a + d));
const sps = (t, a, d = .65) => springSoft(P(t, a, a + d));

const mix = (c1, c2, t) => {
  const h = s => [1, 3, 5].map(i => parseInt(s.slice(i, i + 2), 16));
  const a = h(c1), b = h(c2);
  return '#' + a.map((v, i) => Math.round(lerp(v, b[i], clamp(t))).toString(16).padStart(2, '0')).join('');
};
const rnd = n => { const x = Math.sin(n * 127.1 + 311.7) * 43758.5453; return x - Math.floor(x); };

// ---- 12 Morphable M3 Expressive Silhouettes (N=180 identical vertex count) ----
const N = 180, TAU = Math.PI * 2;
const SH = {
  circle: () => 1,
  cookie: a => .90 + .075 * Math.cos(9 * a),
  cookie12: a => .91 + .065 * Math.cos(12 * a),
  sunny: a => .85 + .115 * Math.cos(12 * a),
  flower: a => .78 + .19 * Math.cos(6 * a),
  clover: a => .74 + .23 * Math.cos(4 * a),
  burst: a => .81 + .15 * Math.cos(10 * a) + .035 * Math.cos(20 * a),
  squircle: a => .85 * Math.pow(Math.pow(Math.abs(Math.cos(a)), 4.2) + Math.pow(Math.abs(Math.sin(a)), 4.2), -.238),
  pebble: a => .90 + .06 * Math.cos(a - .5) + .05 * Math.cos(2 * a + 1.0) + .03 * Math.cos(3 * a),
  gem: a => .84 * Math.pow(Math.pow(Math.abs(Math.cos(a)), 1.65) + Math.pow(Math.abs(Math.sin(a)), 1.65), -.606),
  hex: a => .92 + .055 * Math.cos(6 * a),
  shieldShape: a => .88 + .08 * Math.cos(3 * a - Math.PI / 2) + .03 * Math.cos(6 * a)
};
const SHR = {};
for (const k in SH) SHR[k] = Array.from({ length: N }, (_, i) => SH[k](i / N * TAU));

function shapePath(c, cx, cy, R, a, b = a, m = 0, rot = 0, sx = 1, sy = 1) {
  const A = SHR[a] || SHR.circle, B = SHR[b] || SHR.circle, p = [];
  const mm = clamp(m);
  for (let i = 0; i < N; i++) {
    const r = Math.max(0.1, R) * lerp(A[i], B[i], mm), t = i / N * TAU + rot;
    p.push([cx + Math.cos(t) * r * sx, cy + Math.sin(t) * r * sy]);
  }
  c.beginPath();
  c.moveTo((p[0][0] + p[N - 1][0]) / 2, (p[0][1] + p[N - 1][1]) / 2);
  for (let i = 0; i < N; i++) {
    const q = p[(i + 1) % N];
    c.quadraticCurveTo(p[i][0], p[i][1], (p[i][0] + q[0]) / 2, (p[i][1] + q[1]) / 2);
  }
  c.closePath();
}

function shape(c, fill, cx, cy, R, a, b = a, m = 0, rot = 0, sx = 1, sy = 1) {
  if (R <= 0.2) return;
  shapePath(c, cx, cy, R, a, b, m, rot, sx, sy);
  c.fillStyle = fill;
  c.fill();
}

function shapeStroke(c, stroke, lw, cx, cy, R, a, b = a, m = 0, rot = 0, sx = 1, sy = 1) {
  if (R <= 0.2) return;
  shapePath(c, cx, cy, R, a, b, m, rot, sx, sy);
  c.strokeStyle = stroke;
  c.lineWidth = lw;
  c.stroke();
}

const rr = (c, x, y, w, h, r) => {
  if (w <= 0 || h <= 0) { c.beginPath(); return; }
  r = Math.max(0, Math.min(r, w / 2, h / 2));
  c.beginPath();
  c.roundRect(x, y, w, h, r);
};

function box(c, x, y, w, h, r, fill, stroke, lw = 1.5) {
  if (w <= 0 || h <= 0) return;
  rr(c, x, y, w, h, r);
  if (fill) { c.fillStyle = fill; c.fill(); }
  if (stroke) { c.strokeStyle = stroke; c.lineWidth = lw; c.stroke(); }
}

// ---- Typography & Kinetic Words ----
const F = (sz, w = 600) => `${w} ${Math.round(sz)}px Poppins,'Noto Sans Devanagari',sans-serif`;

function txt(c, s, x, y, o = {}) {
  if (s === undefined || s === null || s === '') return;
  c.save();
  c.font = F(o.sz || 36, o.w || 600);
  c.fillStyle = o.col || C.ink;
  c.textAlign = o.al || 'left';
  c.textBaseline = o.bl || 'alphabetic';
  c.globalAlpha *= (o.a ?? 1);
  if (o.ls) c.letterSpacing = o.ls + 'px';
  c.fillText(String(s), x, y);
  c.restore();
}

const tw = (c, s, sz, w = 600) => {
  c.save();
  c.font = F(sz, w);
  const m = c.measureText(String(s)).width;
  c.restore();
  return m;
};

function wrap(c, s, sz, w, maxW) {
  c.save();
  c.font = F(sz, w);
  const out = [];
  let cur = '';
  for (const wd of String(s).split(' ')) {
    const t = cur ? cur + ' ' + wd : wd;
    if (c.measureText(t).width > maxW && cur) {
      out.push(cur);
      cur = wd;
    } else cur = t;
  }
  c.restore();
  if (cur) out.push(cur);
  return out;
}

function words(c, s, x, y, t, o = {}) {
  const sz = o.sz || 52, w = o.w || 700, lh = o.lh || sz * 1.2, st = o.st ?? .065;
  const ls = wrap(c, s, sz, w, o.maxW || 1200);
  let k = 0;
  ls.forEach((ln, li) => {
    const wd = ln.split(' ');
    const tot = tw(c, ln, sz, w);
    let cx = o.al === 'center' ? x - tot / 2 : (o.al === 'right' ? x - tot : x);
    wd.forEach(word => {
      const p = eoback(P(t, k * st, k * st + .52));
      const a = clamp(P(t, k * st, k * st + .24));
      c.save();
      c.beginPath();
      c.rect(cx - 12, y + li * lh - sz * 1.08, tw(c, word, sz, w) + 32, lh * 1.45);
      c.clip();
      txt(c, word, cx, y + li * lh + (1 - p) * sz * .85, { sz, w, col: o.col, a: a * (o.a ?? 1) });
      c.restore();
      cx += tw(c, word + ' ', sz, w);
      k++;
    });
  });
  return y + (ls.length - 1) * lh;
}

function ease(name, x) {
  return ({ o3: eo3, io3: eio3, back: eoback, exp: eoexp, sp: spring })[name](x);
}
