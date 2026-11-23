// scenes: series, n1..n5 (stories), fast
function story(c, t, sc, th, T, o) {  // left: number + title + source ; right: visual panel
  dotgrid(c, th, T, .5); const p = sp(t, .15, .8);
  txt(c, o.num, 110, 330, { sz: 260, w: 700, col: th.accc, a: p }); words(c, o.title, 118, 470, t - .3, { sz: 76, w: 700, col: th.fg, maxW: 780, lh: 86 });
  const q = sp(t, 1.0, .6); c.globalAlpha = q; chip(c, o.src, 118, 640 + (1 - q) * 20, { sz: 26, fill: th.soft, col: th.mu, line: th.line, ic: 'globe' }); c.globalAlpha = 1;
  c.save(); c.translate(1000, 150); const s = .9 + .1 * p; c.globalAlpha = clamp(p * 1.5); c.translate(410 * (1 - s), 310 * (1 - s)); c.scale(s, s);
  box(c, 0, 0, 820, 620, 44, th.card, th.line); c.save(); rr(c, 0, 0, 820, 620, 44); c.clip(); o.draw(c, t); c.restore(); c.restore(); }
SC.series = (c, t, sc, th, T) => {
  const v = V(T), c1 = lt(sc, 'c1'), c2 = lt(sc, 'c2'), c3 = lt(sc, 'c3'), c4 = lt(sc, 'c4'); dotgrid(c, th, T, .8);
  const up = eio3(P(t, c4 - .1, c4 + .6)); c.save(); c.translate(0, -up * 900);
  words(c, 'NEBians', 110, 300, t - c1 + .3, { sz: 230, w: 700, col: th.fg }); words(c, 'News', 110, 480, t - c1 + .1, { sz: 230, w: 700, col: th.acc, st: .1 });
  sakhiya(c, th, 1470, 330, 210 * sp(t, c1, .8), T, v, { fill: th.acc, glyphCol: th.bg, k: T * .3 });
  const cards = [['Tech & AI news', 'globe', c2, 'from around the world'], ['NEBians Android', 'phone', c3, 'what we are building'], ['Voiced by Sakhiya', 'play', c3 + 2.2, 'our own TTS model']];
  cards.forEach(([ti, ic, st, sub], i) => { const p = sp(t, st, .8), x = 110 + i * 590, y = 520 + (1 - p) * 160; c.globalAlpha = clamp(p * 1.5);
    box(c, x, y, 560, 262, 44, th.card, th.line); box(c, x + 40, y + 40, 96, 96, 32, i === 1 ? C.aquac : th.accc, null); icon(c, ic, x + 88, y + 88, 52, i === 1 ? C.aquad : th.onacc, 2.4);
    txt(c, ti, x + 40, y + 192, { sz: 46, w: 700, col: th.fg }); txt(c, sub, x + 40, y + 236, { sz: 28, w: 500, col: th.mu }); c.globalAlpha = 1; });
  c.restore();
  if (t > c4 - .1) { txt(c, 'TODAY', W / 2, 300 + (1 - sp(t, c4, .6)) * 60, { sz: 34, w: 600, col: th.mu, al: 'center', ls: 6, a: sp(t, c4, .5) }); words(c, 'Top stories', W / 2, 440, t - c4 - .1, { sz: 150, w: 700, al: 'center', col: th.fg });
    for (let i = 0; i < 5; i++) { const p = sp(t, c4 + .4 + i * .14, .6), x = W / 2 + (i - 2) * 190, y = 620; shape(c, i === 0 ? th.acc : th.gr2 || th.soft, x, y, 74 * p, seq7[i], seq7[i], 0, T * .3);
      txt(c, '0' + (i + 1), x, y + 17, { sz: 50, w: 700, col: i === 0 ? th.bg : th.fg, al: 'center', a: p }); } }
};
SC.n1 = (c, t, sc, th, T) => story(c, t, sc, th, T, { num: '01', title: 'Anthropic IPO', src: 'CNBC, Yahoo Finance reports', draw: (c, t) => {
  const pts = [0, .06, .03, .1, .08, .15, .17, .16, .1, .03, -.02, .006], x0 = 70, x1 = 750, yb = 400, ys = 1100; const p = P(t, .8, 4.2);
  for (let i = 0; i < 5; i++) { c.beginPath(); c.moveTo(60, 160 + i * 100); c.lineTo(760, 160 + i * 100); c.lineWidth = 1.5; c.strokeStyle = C.line; c.stroke(); }
  c.beginPath(); const n = (pts.length - 1) * eo3(p); for (let i = 0; i <= Math.floor(n); i++) c.lineTo(x0 + (x1 - x0) * i / 11, yb - pts[i] * ys);
  const f = n % 1, i0 = Math.floor(n); if (i0 < 11) c.lineTo(x0 + (x1 - x0) * (i0 + f) / 11, yb - lerp(pts[i0], pts[i0 + 1], f) * ys);
  c.lineWidth = 9; c.lineCap = c.lineJoin = 'round'; c.strokeStyle = C.blue; c.stroke();
  const a = sp(t, 2.3, .6), b = sp(t, 4.2, .6); c.globalAlpha = a; chip(c, '+17%  IPO day', 240, 90, { sz: 38, fill: C.aquac, col: C.aquad, ic: 'bolt' }); c.globalAlpha = b; chip(c, '-14%  next day', 420, 440, { sz: 38, fill: C.errc, col: C.err, ic: 'warn' }); c.globalAlpha = 1;
  txt(c, 'Tech stocks slid with it', 60, 590, { sz: 28, w: 500, col: C.mute, a: b }); } });
SC.n2 = (c, t, sc, th, T) => story(c, t, sc, th, T, { num: '02', title: 'Gemini 4 Neon', src: 'Indian Express reports', draw: (c, t) => {
  txt(c, 'MULTI-STEP TASK', 56, 90, { sz: 26, col: th.mu, ls: 4 }); const ic = ['doc', 'search', 'compass', 'bolt', 'check'], lb = ['Read', 'Search', 'Plan', 'Act', 'Done'];
  c.beginPath(); c.moveTo(110, 330); c.lineTo(710, 330); c.lineWidth = 8; c.strokeStyle = th.soft; c.lineCap = 'round'; c.stroke();
  const pr = P(t, 1, 5); c.beginPath(); c.moveTo(110, 330); c.lineTo(110 + 600 * eo3(pr), 330); c.strokeStyle = C.aqua; c.stroke();
  ic.forEach((n, i) => { const on = pr >= i / 4 - .001, q = sp(t, .6 + i * .15, .6), cx = 110 + i * 150; shape(c, on ? C.aqua : th.soft, cx, 330, 54 * q, 'cookie', 'sunny', Math.sin(T * 2 + i) * .5 + .5, T * .3);
    icon(c, n, cx, 330, 38, on ? C.gr : th.mu, 2.6); txt(c, lb[i], cx, 440, { sz: 26, w: 600, col: on ? th.fg : th.mu, al: 'center' }); });
  const e = sp(t, 5, .6); c.globalAlpha = e; chip(c, 'faster end-to-end', 56, 520, { sz: 34, fill: th.soft, col: C.aqua, ic: 'bolt' }); c.globalAlpha = 1; } });
SC.n3 = (c, t, sc, th, T) => story(c, t, sc, th, T, { num: '03', title: 'Nvidia: Huang steps down', src: 'Indian Express, FT reports', draw: (c, t) => {
  const p = sp(t, .5, .8); box(c, 50, 70, 360, 150, 32, C.white, null); txt(c, 'Jensen Huang', 80, 150, { sz: 38, w: 700, col: C.ink }); txt(c, 'Nvidia CEO', 80, 195, { sz: 26, w: 500, col: C.mute });
  const s = sp(t, 1.6, .6); c.save(); c.translate(500, 130); c.rotate(-.12); c.globalAlpha = s; box(c, 0, 0, 260, 76, 38, C.errc, null); txt(c, 'STEPS DOWN', 130, 49, { sz: 28, w: 700, col: C.err, al: 'center', ls: 2 }); c.restore();
  const r = P(t, 3.2, 5.2); shape(c, C.white, 150, 440, 100 * sp(t, 3, .6), 'squircle', 'squircle', 0, 0); icon(c, 'chip', 150, 440, 90, C.blue, 3);
  c.beginPath(); c.moveTo(260, 440); c.lineTo(260 + 330 * eo3(r), 440); c.lineWidth = 8; c.setLineDash([22, 18]); c.lineDashOffset = -T * 60; c.strokeStyle = C.white; c.stroke(); c.setLineDash([]);
  box(c, 600, 395, 170, 90, 30, 'rgba(255,255,255,.18)', null); txt(c, 'CHINA', 685, 452, { sz: 32, w: 700, col: C.white, al: 'center' });
  const q = sp(t, 5.4, .6); c.save(); c.globalAlpha = q; c.translate(430, 440); shape(c, C.amber, 0, 0, 46 * q, 'cookie', 'cookie', 0, 0); icon(c, 'pause', 0, 0, 36, C.amberd, 3.2); c.restore(); chip(c, 'shipments paused (reported)', 50, 520, { sz: 28, fill: C.white, col: C.blue, a: q }); } });
SC.n4 = (c, t, sc, th, T) => story(c, t, sc, th, T, { num: '04', title: 'OpenAI: GPT-6.2 + wallet', src: 'AI news roundups', draw: (c, t) => {
  ['Astra', 'Sol'].forEach((n, i) => { const p = sp(t, .5 + i * .3, .8), x = 50 + i * 370; c.save(); c.globalAlpha = clamp(p * 1.5); box(c, x, 60 + (1 - p) * 80, 340, 210, 36, i ? C.aquac : C.pc, null);
    txt(c, 'GPT-6.2', x + 34, 140 + (1 - p) * 80, { sz: 28, w: 600, col: C.mute }); txt(c, n, x + 34, 210 + (1 - p) * 80, { sz: 64, w: 700, col: i ? C.aquad : C.onpc }); c.restore(); });
  const w = sp(t, 3.4, .8); c.save(); c.globalAlpha = clamp(w * 1.5); box(c, 50, 330 + (1 - w) * 60, 720, 230, 36, C.cont, C.line); icon(c, 'wallet', 140, 445 + (1 - w) * 60, 84, C.blue, 2.6);
  txt(c, 'ChatGPT Wallet', 220, 430 + (1 - w) * 60, { sz: 44, w: 700, col: C.ink }); txt(c, 'agents can shop for you', 220, 480 + (1 - w) * 60, { sz: 28, w: 500, col: C.mute });
  const k = P(t, 4.6, 6.4); shape(c, C.blue, lerp(560, 700, eio3(k)), 445 + (1 - w) * 60, 30 * (1 - k * .3), 'cookie', 'flower', k, T); c.restore(); } });
SC.n5 = (c, t, sc, th, T) => story(c, t, sc, th, T, { num: '05', title: 'OpenRouter drops 3 providers', src: 'AI Weekly reports', draw: (c, t) => {
  const cx = 410, cy = 330, nm = ['', '', '', '', '', ''], bad = [0, 2, 4];
  for (let i = 0; i < 6; i++) { const a = i / 6 * TAU - 1.2, b = bad.includes(i), k = b ? P(t, 3.4 + bad.indexOf(i) * .5, 4.2 + bad.indexOf(i) * .5) : 0, r = 220 * (1 - eo3(k) * .0) + eo3(k) * 120, x = cx + Math.cos(a) * r, y = cy + Math.sin(a) * r * .86, p = sp(t, .4 + i * .1, .6);
    c.globalAlpha = (1 - k) * clamp(p); c.beginPath(); c.moveTo(cx, cy); c.lineTo(x, y); c.lineWidth = 4; c.strokeStyle = b && t > 3.2 ? C.err : th.line; c.stroke();
    shape(c, b && t > 3.2 ? C.errc : th.soft, x, y, 56 * p * (1 - eo3(k) * .6), 'cookie', 'cookie', 0, i); icon(c, b && t > 3.2 ? 'x' : 'globe', x, y, 34, b && t > 3.2 ? C.err : th.mu, 2.6); }
  c.globalAlpha = 1; shape(c, th.acc, cx, cy, 80 * sp(t, 0, .7), 'flower', 'sunny', Math.sin(T) * .5 + .5, T * .4); icon(c, 'bolt', cx, cy, 56, th.bg, 2.8);
  const q = sp(t, 5.4, .6); c.globalAlpha = q; chip(c, 'prompt leaks found', 56, 560, { sz: 30, fill: C.errc, col: C.err, ic: 'lock' }); c.globalAlpha = 1; } });
SC.fast = (c, t, sc, th, T) => { dotgrid(c, th, T, .8); const f1 = lt(sc, 'f1'), f2 = lt(sc, 'f2');
  for (let r = 0; r < 2; r++) { const y = r ? 190 : 100, dir = r ? 1 : -1, s = 'AI NEWS   -   TECH   -   NEBIANS   -   '; const L = tw(c, s, 44, 700); c.globalAlpha = .55;
    for (let k = -1; k < 4; k++) txt(c, s, ((T * 140 * dir) % L) + k * L, y + 40, { sz: 44, w: 700, col: '#c9d7ff' }); c.globalAlpha = 1; }
  [[f1, 'Microsoft', 'cheaper, task-specific AI models', 'bolt', 290], [f2, 'Google DeepMind', 'AI used to test Signal encryption', 'lock', 520]].forEach(([s0, a, b, ic, y], i) => { const p = sp(t, s0 + .2 - sc.t0 * 0 - (sc.t0 ? 0 : 0), .8), q = p; const lx = lerp(i ? 2000 : -1700, 160, clamp(sp(t, s0, .9)));
    box(c, lx, y, 1600, 200, 56, C.white, null); shape(c, C.pc, lx + 120, y + 100, 74, seq7[i + 3], seq7[i + 3], 0, T * .4); icon(c, ic, lx + 120, y + 100, 56, C.blue, 3);
    txt(c, a, lx + 240, y + 88, { sz: 40, w: 600, col: C.mute }); txt(c, b, lx + 240, y + 154, { sz: 62, w: 700, col: C.ink }); });
};
