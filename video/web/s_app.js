// scenes: app (Android updates), outro
SC.app = (c, t, sc, th, T) => {
  const v = V(T), L = ['d0', 'd1', 'd2', 'd3', 'd4', 'd5'].map(k => lt(sc, k)), k = L.reduce((a, s, i) => t >= s - .1 ? i : a, 0); dotgrid(c, th, T, .5);
  words(c, 'Android app', 110, 210, t - .2, { sz: 96, w: 700, col: th.fg }); txt(c, 'since the last Play Store release', 114, 262, { sz: 34, w: 500, col: th.mu, a: sp(t, .8, .5) });
  const f = [['Liquid glass dock', 'real refraction, draggable lens', 'glass'], ['Tonal toasts', 'success, info, warning, error', 'info'], ['File Shield PDFs', 'attested, sealed downloads', 'shield'], ['Spring motion', 'search + profile Pathways filters', 'filter']];
  f.forEach(([a, b, ic], i) => { const on = k === i + 1, p = sp(t, L[0] + 1.2 + i * .25, .7), y = 330 + i * 118 + (1 - p) * 40; c.save(); c.globalAlpha = clamp(p * 1.4);
    box(c, 110, y, 780, 100, 36, on ? C.pc : th.card, on ? C.blue : th.line, on ? 2.5 : 1.5); shape(c, on ? C.blue : th.soft, 168, y + 50, 34, 'cookie', 'sunny', on ? 1 : 0, T * .3); icon(c, ic, 168, y + 50, 30, on ? '#fff' : th.mu, 2.4);
    txt(c, a, 226, y + 46, { sz: 36, w: 700, col: on ? C.onpc : th.fg }); txt(c, b, 226, y + 82, { sz: 24, w: 500, col: on ? '#32528f' : th.mu }); c.restore(); });
  const pp = sp(t, .3, .9), X = 1294, Y = 134, Wd = 332, Ht = 622; c.save(); c.globalAlpha = clamp(pp * 1.4); c.translate(0, (1 - pp) * 120);
  box(c, X - 16, Y - 16, Wd + 32, Ht + 32, 64, C.ink, null); box(c, X, Y, Wd, Ht, 50, C.low, null); c.save(); rr(c, X, Y, Wd, Ht, 50); c.clip(); c.translate(X, Y);
  for (let i = 0; i < 6; i++) { box(c, 24, 90 + i * 84 - ((T * (k === 1 ? 40 : 0)) % 84), Wd - 48, 66, 20, i % 2 ? C.cont : C.white, C.line); box(c, 40, 108 + i * 84 - ((T * (k === 1 ? 40 : 0)) % 84), 120, 14, 7, C.line); }
  txt(c, 'NEBians', 28, 62, { sz: 30, w: 700, col: C.blue });
  if (k === 1 || k === 0) { const s = sp(t, L[1] - .2, .7), tt = t - L[1], pos = clamp(.5 - .5 * Math.cos(clamp(tt / 6) * TAU * 1.25)), dx = 30, dw = Wd - 60, ix = i => dx + dw * (.14 + i * .24);
    box(c, dx, Ht - 140 + (1 - s) * 160, dw, 84, 42, 'rgba(255,255,255,.7)', C.line); const lx = lerp(ix(0), ix(3), pos), ly = Ht - 98 + (1 - s) * 160;
    ['home', 'book', 'compass', 'user'].forEach((n, i) => icon(c, n, ix(i), ly, 30, C.mute, 2.2)); shape(c, 'rgba(0,88,223,.14)', lx, ly, 46, 'circle', 'circle'); c.save(); c.beginPath(); c.arc(lx, ly, 42, 0, TAU); c.clip();
    c.fillStyle = C.white; c.fill(); ['home', 'book', 'compass', 'user'].forEach((n, i) => icon(c, n, lx + (ix(i) - lx) * 1.5, ly, 44, C.blue, 2.6)); c.restore(); c.beginPath(); c.arc(lx, ly, 42, 0, TAU); c.lineWidth = 3; c.strokeStyle = C.blue; c.stroke(); }
  if (k === 2) { const K = [['Saved', 'success', C.aquac, C.aquad, 'check'], ['New update', 'info', C.azurec, C.azured, 'info'], ['Low storage', 'warning', C.amber, C.amberd, 'warn'], ['Could not sync', 'error', C.errc, C.err, 'x']];
    K.forEach(([a, b, bg, fg, ic], i) => { const s0 = L[2] + .5 + i * 1.7, p = sp(t, s0, .6), old = clamp((t - s0 - 3.2) / 1) * 0; if (t < s0 - .1) return; c.save(); c.globalAlpha = clamp(p * 1.5); const y = 100 + i * 100 + (1 - p) * -90;
      box(c, 22, y, Wd - 44, 82, 28, bg, null); icon(c, ic, 62, y + 41, 32, fg, 2.8); txt(c, a, 98, y + 38, { sz: 26, w: 700, col: fg }); txt(c, b, 98, y + 66, { sz: 20, w: 500, col: fg, a: .8 }); c.restore(); }); }
  if (k === 3) { const p = sp(t, L[3], .7), d = P(t, L[3] + .6, L[3] + 3.6); c.save(); c.translate(Wd / 2, 250); c.scale(p, p); box(c, -92, -120, 184, 240, 28, C.white, C.line);
    icon(c, 'doc', 0, -20, 96, C.blue, 2.4); box(c, -60, 60, 120, 14, 7, C.cont); box(c, -60, 60, 120 * d, 14, 7, C.blue); c.restore();
    const s = sp(t, L[3] + 3.2, .7); shape(c, C.aquad, Wd / 2 + 78, 330, 56 * s, 'cookie', 'flower', 1, 0); icon(c, 'shield', Wd / 2 + 78, 330, 36 * s, '#fff', 2.8);
    c.globalAlpha = s; chip(c, 'attested', Wd / 2 - 90, 440, { sz: 26, fill: C.aquac, col: C.aquad, ic: 'lock' }); c.globalAlpha = 1; }
  if (k >= 4) { const tt = t - L[4], w = lerp(70, Wd - 48, sp(t, L[4] + .2, .8)); box(c, Wd / 2 - w / 2, 96, w, 70, 35, C.white, C.blue, 2.5); icon(c, 'search', Wd / 2 - w / 2 + 42, 131, 30, C.blue, 2.6);
    if (w > 200) txt(c, 'Search guides', Wd / 2 - w / 2 + 76, 142, { sz: 26, w: 500, col: C.mute, a: P(tt, .8, 1.2) });
    ['Class 12', 'Science', 'NEB'].forEach((s, i) => { const q = sp(t, L[4] + 1.2 + i * .3, .6), on = tt > 3 + i * .6; c.globalAlpha = q; chip(c, s, 26 + i * 100, 200 + (1 - q) * 30, { sz: 20, pad: 28, fill: on ? C.blue : C.pc, col: on ? '#fff' : C.onpc }); }); c.globalAlpha = 1;
    if (k === 5) { const r = sp(t, L[5], .8); c.fillStyle = 'rgba(246,248,255,.96)'; c.fillRect(0, 0, Wd, Ht * r); c.beginPath(); c.arc(Wd / 2, 270, 100, 0, TAU); c.lineWidth = 22; c.strokeStyle = C.cont; c.stroke();
      c.beginPath(); c.arc(Wd / 2, 270, 100, -Math.PI / 2, -Math.PI / 2 + TAU * .72 * eo3(P(t, L[5] + .3, L[5] + 3))); c.strokeStyle = C.blue; c.lineCap = 'round'; c.stroke(); txt(c, 'Next', Wd / 2, 285, { sz: 46, w: 700, col: C.ink, al: 'center' });
      chip(c, 'upcoming release', Wd / 2 - 130, 430, { sz: 24, ic: 'bolt', a: r }); } }
  c.restore(); c.restore();
};
SC.outro = (c, t, sc, th, T) => {
  const v = V(T), o1 = lt(sc, 'o1'), o2 = lt(sc, 'o2'), o3 = lt(sc, 'o3'); dotgrid(c, th, T, 1);
  const a = sp(t, 0, .9); sakhiya(c, th, 470, 400, 210 * a, T, v, { fill: C.white, ring: C.white, glyphCol: C.blue, a: clamp(a) });
  words(c, 'धन्यवाद', 840, 360, t - o1 + .2, { sz: 190, w: 700, col: C.white }); txt(c, 'Thank you for watching', 848, 450, { sz: 52, w: 500, col: '#dbe4ff', a: P(t, o1 + .6, o1 + 1.1) });
  const q = sp(t, o2, .8); c.globalAlpha = q; let y = 540 + (1 - q) * 40; [['Voice', 'Sakhiya V1 int8, news style'], ['Motion', 'canvas + Playwright + ffmpeg, rendered on MoLab'], ['Series', 'NEBians News, episode 01']].forEach(([a, b], i) => { txt(c, a, 848, y + i * 56, { sz: 28, w: 600, col: '#9fb8ff', ls: 3 }); txt(c, b, 1010, y + i * 56, { sz: 30, w: 500, col: C.white }); }); c.globalAlpha = 1;
  const z = sp(t, o3, .7); c.globalAlpha = z; chip(c, 'github.com/NikitHamal/Sakhiya', 848, 740 - 0 + (1 - z) * 30, { sz: 30, fill: C.white, col: C.blue, ic: 'globe' }); c.globalAlpha = 1;
  const e = P(t, sc.t1 - sc.t0 - 3, sc.t1 - sc.t0 - 1.4); if (e > 0) { c.fillStyle = C.blue; c.globalAlpha = eo3(e) * .0; c.fillRect(0, 0, W, H); c.globalAlpha = 1; }
};
