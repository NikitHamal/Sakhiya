// scenes: hello, about, series
const seq7 = ['cookie', 'sunny', 'flower', 'clover', 'burst', 'squircle', 'pebble'];
function win(c, t, a, b, fn, dy = 40) { if (t < a - .05 || t > b + .5) return; const pin = sp(t, a, .75), pout = eo3(P(t, b, b + .4));
  c.save(); c.globalAlpha *= clamp(pin * 1.4) * (1 - pout); c.translate(0, (1 - pin) * dy - pout * 30); fn(t - a); c.restore(); }
SC.hello = (c, t, sc, th, T) => {
  const v = V(T), a1 = lt(sc, 'a1'), a2 = lt(sc, 'a2'); dotgrid(c, th, T, clamp(t / 1));
  const out = eio3(P(t, a1 - .4, a1 + .5));   // title group leaves
  if (t < a1 + .6) { c.save(); c.translate(0, -out * 1300);
    const g = sp(t, .15, 1.1), k = P(t, .15, 1.2) * 3, ii = Math.floor(k), mm = eio3(k % 1), hit = P(t, 1.2, 1.6);
    const cy = lerp(540, 330, eio3(P(t, 1.2, 1.9)));
    if (t > 1.2 && t < 2) { c.globalAlpha = 1 - hit; shapePath(c, W / 2, 540, 330 + hit * 900, 'circle', 'sunny', hit, t); c.strokeStyle = C.white; c.lineWidth = 6; c.stroke(); c.globalAlpha = 1; }
    shape(c, C.white, W / 2, cy, 330 * g * lerp(1, .56, eio3(P(t, 1.2, 1.9))), seq7[ii % 7], seq7[(ii + 1) % 7], mm, t * .5);
    txt(c, 'स', W / 2, cy + 100 * lerp(1, .56, eio3(P(t, 1.2, 1.9))) * 1.0 + 70 * (1 - lerp(1, .56, eio3(P(t, 1.2, 1.9)))) * .0, { sz: 330 * lerp(1, .56, eio3(P(t, 1.2, 1.9))) * .95, w: 700, col: C.blue, al: 'center', a: g });
    if (t > 1.2) { words(c, 'NEBians', W / 2, 700, t - 1.25, { sz: 170, w: 700, al: 'center', col: C.white, st: .09 });
      const p = sp(t, 1.9, .7); c.globalAlpha = p; chip(c, 'NEWS  |  EPISODE 01', W / 2 - tw(c, 'NEWS  |  EPISODE 01', 32) / 2 - 22, 760 + (1 - p) * 30, { sz: 32, fill: C.white, col: C.blue }); c.globalAlpha = 1; }
    c.restore(); }
  if (t >= a1 - .2) { const t1 = t - a1; const o2 = eio3(P(t, a2 - .3, a2 + .4));
    c.save(); c.translate(-o2 * 2000, 0); words(c, 'नमस्ते', W / 2, 380, t1, { sz: 260, w: 700, al: 'center', col: C.white, st: .15 });
    words(c, 'सबैजनालाई', W / 2, 560, t1 - .25, { sz: 150, w: 600, al: 'center', col: '#dbe4ff', st: .12 });
    for (let k = 0; k < 3; k++) { const r = 160 + ((T * .9 + k / 3) % 1) * 700; c.globalAlpha = (1 - (T * .9 + k / 3) % 1) * .35 * V(T)[0] * 2; c.beginPath(); c.arc(W / 2, 460, r, 0, TAU); c.strokeStyle = C.white; c.lineWidth = 5; c.stroke(); }
    c.restore(); }
  if (t >= a2 - .3) { const t2 = t - a2, p = sp(t, a2 - .2, .9);
    sakhiya(c, th, 520, 400, 230 * p, T, v, { fill: C.white, ring: C.white, glyphCol: C.blue, a: clamp(p) });
    words(c, 'म सखिया', 900, 360, t2, { sz: 150, w: 700, col: C.white, st: .12 });
    txt(c, 'Sakhiya', 906, 470, { sz: 64, w: 500, col: '#dbe4ff', a: P(t2, .5, 1) });
    const q = sp(t2, .8, .7); c.globalAlpha = q; chip(c, 'Your news reader', 900, 520 + (1 - q) * 24, { sz: 34, fill: '#1b66e8', col: C.white, ic: 'play' }); c.globalAlpha = 1; }
};
SC.about = (c, t, sc, th, T) => {
  const v = V(T); dotgrid(c, th, T, .6); const id = ['b1', 'b2', 'b3', 'b4', 'b5', 'b6'], st = id.map(k => lt(sc, k)), en = st.map((s, i) => i < 5 ? st[i + 1] : sc.t1 - sc.t0 - .3);
  const k = id.findIndex((_, i) => t >= st[i] - .2 && t < en[i]); sakhiya(c, th, 470, 430, 200 * sp(t, 0, .9), T, v, { k: Math.max(0, k) + (t - st[Math.max(0, k)]) * .15, a: clamp(t / .3) });
  const X = 900, Y = 170, Wd = 920, Ht = 560, card = (fn) => { box(c, X, Y, Wd, Ht, 44, th.card, th.line); fn(); };
  win(c, t, st[0], en[0], () => card(() => { txt(c, 'TEXT  >  VOICE', X + 56, Y + 90, { sz: 28, col: th.mu, ls: 3 });
    words(c, 'Text-to-speech', X + 56, Y + 230, t - st[0], { sz: 92, w: 700, col: th.fg, maxW: 820 });
    txt(c, 'नमस्ते सबैजनालाई', X + 56, Y + 340, { sz: 44, w: 500, col: th.acc }); wavebars(c, X + 56, Y + 450, Wd - 112, 130, T, v, th.acc, 40); }));
  win(c, t, st[1], en[1], () => card(() => { const n = Math.round(16 * eo3(P(t, st[1] + .3, st[1] + 2.2)));
    txt(c, String(n), X + 56, Y + 300, { sz: 260, w: 700, col: th.acc }); txt(c, 'MB', X + 56 + tw(c, String(n), 260, 700) + 24, Y + 300, { sz: 90, w: 600, col: th.fg });
    txt(c, 'the whole model, int8', X + 60, Y + 370, { sz: 34, w: 500, col: th.mu }); const q = sp(t, st[1] + .5, .6), q2 = sp(t, st[1] + .9, .6);
    c.globalAlpha *= q; chip(c, 'नेपाली', X + 56, Y + 430, { sz: 40, fill: C.pc }); c.globalAlpha /= Math.max(q, .001); c.globalAlpha *= q2; chip(c, 'संस्कृत', X + 56 + 230, Y + 430, { sz: 40, fill: C.aquac, col: C.aquad }); }));
  win(c, t, st[2], en[2], () => card(() => { box(c, X + 40, Y + 40, Wd - 80, Ht - 80, 28, th.soft, th.line); for (let i = 0; i < 3; i++) { c.beginPath(); c.arc(X + 84 + i * 32, Y + 84, 9, 0, TAU); c.fillStyle = [C.errc, C.amber, C.aquac][i]; c.fill(); }
    box(c, X + 220, Y + 62, Wd - 300, 46, 23, th.card, th.line); txt(c, 'nikithamal.github.io/Sakhiya', X + 250, Y + 95, { sz: 26, w: 500, col: th.mu });
    box(c, X + 80, Y + 170, Wd - 160, 130, 24, th.card, th.line); txt(c, 'सिधै ब्राउजरमै चल्छु', X + 110, Y + 255, { sz: 48, w: 600, col: th.fg });
    const pp = sp(t, st[2] + .6, .5); c.beginPath(); c.arc(X + Wd / 2, Y + 400, 54 * pp, 0, TAU); c.fillStyle = th.acc; c.fill(); icon(c, 'play', X + Wd / 2 + 4, Y + 400, 46, '#fff', 2.5, '#fff');
    wavebars(c, X + 130, Y + 400, 280, 70, T, v, th.acc, 14); wavebars(c, X + Wd - 410, Y + 400, 280, 70, T, v, th.acc, 14); }));
  win(c, t, st[3], en[3], () => card(() => { const p = eo3(P(t, st[3] + .3, st[3] + 2.4)) * .22, cx = X + 300, cy = Y + 290;
    c.beginPath(); c.arc(cx, cy, 170, 0, TAU); c.lineWidth = 34; c.strokeStyle = th.soft; c.stroke(); c.beginPath(); c.arc(cx, cy, 170, -Math.PI / 2, -Math.PI / 2 + TAU * p); c.lineCap = 'round'; c.strokeStyle = th.acc; c.stroke();
    txt(c, 'Phase 1', cx, cy + 20, { sz: 64, w: 700, col: th.fg, al: 'center' }); words(c, 'Not fully built yet', X + 560, Y + 250, t - st[3] - .3, { sz: 56, w: 700, col: th.fg, maxW: 340, lh: 70 });
    const q = sp(t, st[3] + 1, .6); c.globalAlpha *= q; chip(c, 'In development', X + 560, Y + 420, { sz: 30, ic: 'bolt' }); }));
  win(c, t, st[4], en[4], () => card(() => { txt(c, 'ROADMAP', X + 56, Y + 90, { sz: 28, col: th.mu, ls: 3 }); const lab = ['Phase 1', 'Better voice', 'More styles'], sub = ['now', 'next', 'later'], p = P(t, st[4] + .3, st[4] + 2.6);
    c.beginPath(); c.moveTo(X + 130, Y + 280); c.lineTo(X + 130 + (Wd - 260) * eo3(p), Y + 280); c.lineWidth = 8; c.lineCap = 'round'; c.strokeStyle = th.line; c.stroke();
    c.beginPath(); c.moveTo(X + 130, Y + 280); c.lineTo(X + 130 + (Wd - 260) * .02, Y + 280); c.strokeStyle = th.acc; c.stroke();
    lab.forEach((s, i) => { const x = X + 130 + i * (Wd - 260) / 2, q = sp(t, st[4] + .5 + i * .5, .6); shape(c, i === 0 ? th.acc : th.soft, x, Y + 280, 52 * q, seq7[i * 2], seq7[i * 2], 0, T * .4);
      txt(c, s, x, Y + 400, { sz: 34, w: 600, col: th.fg, al: 'center', a: q }); txt(c, sub[i], x, Y + 445, { sz: 26, w: 500, col: th.mu, al: 'center', a: q }); }); }));
  win(c, t, st[5], en[5], () => card(() => { txt(c, 'THE VOICE YOU HEAR NOW', X + 56, Y + 100, { sz: 28, col: th.mu, ls: 3 }); wavebars(c, X + 56, Y + 290, Wd - 112, 230, T, v, th.acc, 46);
    let x = X + 56; [['Phase 1', 'play'], ['V1 int8', 'chip'], ['news style', 'bolt']].forEach(([s, ic], i) => { c.globalAlpha = sp(t, st[5] + .4 + i * .25, .5); x += chip(c, s, x, Y + 440, { sz: 30, ic }) + 16; }); c.globalAlpha = 1; }));
};
