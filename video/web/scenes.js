// NEBians News — scene layouts (all Material 3 Expressive: tonal surfaces, morphing shapes, spring motion)
function headerChips(c, t, sc, p, b) {
  const y = 176; let x = 120;
  const a1 = sp(t, b, .5), a2 = sp(t, b + .1, .5), a3 = sp(t, b + .2, .5);
  x += chip(c, x, y, 'STORY ' + String(sc.no).padStart(2, '0'), { fill: p.pri, ink: p.onPri, a: a1, s: .8 + .2 * a1, ls: 1 }) + 12;
  x += chip(c, x, y, sc.tag, { fill: p.priC, ink: p.onPriC, icon: sc.edu ? 'school' : 'spark', a: a2, s: .8 + .2 * a2 }) + 12;
  if (sc.dates) chip(c, x, y, sc.dates, { stroke: p.line, ink: p.mute, icon: 'cal', a: a3, s: .8 + .2 * a3 });
}
function stageText(c, cx, cy, R, l, a, dy, p, ink) {
  const f = fitText(c, l.big, R * 1.5, 112, 700, 44), n = f.lines.length, lh = f.sz * 1.12, y0 = cy - (n - 1) * lh / 2 + f.sz * .36 + dy;
  f.lines.forEach((s, i) => txt(c, s, cx, y0 + i * lh, { sz: f.sz, w: 700, col: ink, al: 'center', a }));
}
SC.story = (c, t, sc, p, T) => {
  bg(c, p, T); const b = .35; headerChips(c, t, sc, p, b);
  const sz = wrap(c, sc.h, 68, 700, 800).length > 3 ? 58 : 68, hd = kinetic(c, sc.h, 120, 330, t - b - .1, { sz, w: 700, maxW: 800, col: p.ink, lh: sz * 1.16 });
  const na = sp(t, b + .55, .6), nl = wrap(c, sc.hn, 32, 500, 800);
  nl.forEach((s, i) => txt(c, s, 120, hd.last + 66 + i * 46 + (1 - na) * 24, { sz: 32, w: 500, col: p.mute, a: clamp(na) }));
  // stepper
  const k = Math.max(0, cur(sc, t)), n = sc.lines.length; let sx = 120;
  for (let i = 0; i < n; i++) {
    const act = i === k ? ss(t, i ? lt(sc, i) : b, .4) : (i === k - 1 ? 1 - ss(t, lt(sc, k), .4) : 0), w = lerp(24, 84, act);
    box(c, sx, 826, w, 14, 7, mixc(p.high, p.pri, i <= k ? 1 : 0), null); sx += w + 10;
  }
  txt(c, String(k + 1).padStart(2, '0') + ' / ' + String(n).padStart(2, '0'), sx + 14, 840, { sz: 22, w: 600, col: p.mute });
  // stage
  const cx = 1385, cy = 470, R = 262, ent = sp(t, b, .9), kk = Math.max(0, cur(sc, t));
  const A = SHAPES6[(kk + 5) % 6], B = SHAPES6[kk % 6], m = kk ? ss(t, lt(sc, kk), .8) : 1;
  c.save(); c.globalAlpha *= clamp(ent * 1.5);
  shape(c, p.secC, cx + 34, cy + 26, R * ent, B, A, 1 - m, -T * .16 + 1);
  shape(c, p.priC, cx, cy, R * 1.0 * ent, B, A, 1 - m, T * .2);
  shapeLine(c, mixc(p.priC, p.pri, .35), 3, cx, cy, R * .86 * ent, B, A, 1 - m, T * .2);
  for (let i = 0; i < 3; i++) {
    const a = T * (.38 - i * .07) + i * 2.1, rr_ = R * (1.2 + i * .09);
    shape(c, [p.pri, p.ter, p.sec][i], cx + Math.cos(a) * rr_, cy + Math.sin(a) * rr_ * .86, [26, 17, 11][i] * ent, ['circle', 'cookie9', 'flower'][i], 'circle', 0, T);
  }
  c.restore();
  swap(sc, t, (l, a, dy) => {
    stageText(c, cx, cy, R, l, a, dy, p, p.onPriC);
    const f = 30, w = tw(c, l.label, f, 600) + 56, y = cy + R + 36 + dy * .5;
    box(c, cx - w / 2, y, w, 56, 28, C(p.surf, a), C(p.line, a), 2); txt(c, l.label, cx, y + 38, { sz: f, w: 600, col: p.ink, al: 'center', a });
  });
};

SC.hello = (c, t, sc, p, T) => {
  bg(c, p, T); const cx = 1400, cy = 478, R = 250, ent = sp(t, 1.0, .9);
  for (let i = 0; i < 3; i++) { const q = P(t, 1.0 + i * .28, 2.6); if (q > 0 && q < 1) shapeLine(c, p.pri, 5 * (1 - q) + 1, cx, cy, R + eio(q) * 520, 'cookie12', 'circle', q, T * .3 + i); }
  txt(c, '', 0, 0);
  kinetic(c, 'NEBians', 120, 440, t - .25, { sz: 172, w: 700, maxW: 1000, col: p.ink, st: .08 });
  kinetic(c, 'News', 120, 610, t - .45, { sz: 172, w: 700, maxW: 1000, col: p.pri, st: .08 });
  const a = sp(t, 1.0, .6);
  let x = 124; x += chip(c, x, 690, 'EPISODE 1', { h: 56, sz: 24, fill: p.pri, ink: p.onPri, a, s: .8 + .2 * a, ls: 1.5 }) + 14;
  chip(c, x, 690, 'WEEKLY RECAP', { h: 56, sz: 24, stroke: p.line, ink: p.mute, icon: 'cal', a: sp(t, 1.1, .6), s: .8 + .2 * sp(t, 1.1, .6) });
  txt(c, 'Friday, 27 Nov 2026  ·  Tech, AI & Nepal education', 124, 806, { sz: 28, w: 500, col: p.mute, a: clamp(sp(t, 1.3, .6)) });
  c.save(); c.translate(cx, cy); c.scale(ent, ent); c.translate(-cx, -cy);
  avatar(c, cx, cy, R, T, p, ampAt(T), { ring: p.priC, cheek: p.ter }); c.restore();
  swap(sc, t, (l, al, dy) => {
    const f = fitText(c, l.big, 520, 56, 700, 36), w = tw(c, f.lines[0], f.sz, 700) + 72, y = cy + R + 62 + dy;
    box(c, cx - w / 2, y, w, 84, 42, C(p.priC, al), null); txt(c, f.lines[0], cx, y + 84 / 2 + f.sz * .36, { sz: f.sz, w: 700, col: p.onPriC, al: 'center', a: al });
  });
};

SC.about = (c, t, sc, p, T) => {
  bg(c, p, T); const b = .3, ent = sp(t, b, .9), ax = 380, ay = 470, R = 188;
  chip(c, 120, 176, 'MEET SAKHIYA', { fill: p.pri, ink: p.onPri, a: sp(t, b, .5), s: .8 + .2 * sp(t, b, .5), ls: 1 });
  c.save(); c.translate(ax, ay); c.scale(ent, ent); c.translate(-ax, -ay);
  avatar(c, ax, ay, R, T, p, ampAt(T), { ring: p.priC }); c.restore();
  const na = sp(t, b + .4, .6);
  txt(c, 'Sakhiya', ax, ay + R + 112 + (1 - na) * 20, { sz: 62, w: 700, col: p.ink, al: 'center', a: clamp(na) });
  txt(c, 'AI news anchor', ax, ay + R + 158 + (1 - na) * 20, { sz: 28, w: 500, col: p.mute, al: 'center', a: clamp(na) });
  const icons = ['mic', 'lang', 'chip', 'device', 'sprout', 'trend'], k = cur(sc, t);
  sc.lines.forEach((l, j) => {
    const s0 = lt(sc, j), e = sp(t, s0, .7); if (e <= .001) return;
    const act = j === k ? ss(t, s0, .35) : (j < k ? 1 - ss(t, lt(sc, j + 1), .35) : 0);
    const x = 820 + (1 - e) * 90, y = 184 + j * 112, w = 980, h = 100, s = 1 + .02 * act;
    c.save(); c.globalAlpha *= clamp(e * 1.4); c.translate(x + w / 2, y + h / 2); c.scale(s, s); c.translate(-(x + w / 2), -(y + h / 2));
    box(c, x, y, w, h, 30, mixc(p.cont, p.pri, act), null);
    shape(c, mixc(p.priC, p.onPri, act * .0 + 0), x + 64, y + h / 2, 38, 'cookie9', 'flower', act, T * .3 * act + j);
    icon(c, icons[j], x + 64, y + h / 2, 34, mixc(p.onPriC, p.pri, act * 0), 2);
    const ink = mixc(p.ink, p.onPri, act), mu = mixc(p.mute, p.onPri, act * .85);
    txt(c, l.big, x + 128, y + 46, { sz: 36, w: 700, col: ink }); txt(c, l.label, x + 128, y + 80, { sz: 23, w: 500, col: mu });
    c.restore();
  });
};

SC.series = (c, t, sc, p, T) => {
  bg(c, p, T); const b = .3, k = cur(sc, t);
  chip(c, 165, 176, 'THE SHOW', { fill: p.pri, ink: p.onPri, a: sp(t, b, .5), s: .8 + .2 * sp(t, b, .5), ls: 1 });
  swap(sc, t, (l, a, dy) => {
    txt(c, l.big, 165, 300 + dy, { sz: 88, w: 700, col: p.ink, a });
    txt(c, l.label, 168, 352 + dy * .7, { sz: 32, w: 500, col: p.mute, a });
  }, { dy: 44 });
  const names = ['SAT', 'SUN', 'MON', 'TUE', 'WED', 'THU', 'FRI'], tw_ = 210, g = 20, x0 = (W - (7 * tw_ + 6 * g)) / 2, y0 = 400, lc = lt(sc, 3), dur = sc.lines[3].d;
  for (let i = 0; i < 7; i++) {
    const e = sp(t, b + .15 + i * .07, .7); if (e <= .001) continue;
    const fri = i === 6, sw = k >= 3 ? clamp((t - lc - .1) / Math.max(.1, dur * .8) * 7 - i + .6) : 0, hot = fri ? 1 : (k >= 3 ? eo3(sw) : 0);
    const x = x0 + i * (tw_ + g), y = y0 + (1 - e) * 60 + (fri ? -8 * (.5 + .5 * Math.sin(T * 3)) * (k < 3 ? 1 : 0) : 0);
    c.save(); c.globalAlpha *= clamp(e * 1.4);
    box(c, x, y, tw_, 232, 40, fri ? p.pri : mixc(p.cont, p.priC, hot), null);
    const ink = fri ? p.onPri : mixc(p.ink, p.onPriC, hot), mu = fri ? p.onPri : p.mute;
    txt(c, names[i], x + tw_ / 2, y + 56, { sz: 22, w: 600, col: mu, al: 'center', ls: 2, a: .85 });
    txt(c, String(21 + i), x + tw_ / 2, y + 150, { sz: 84, w: 700, col: ink, al: 'center' });
    txt(c, fri ? 'Today' : 'Nov', x + tw_ / 2, y + 202, { sz: 22, w: 600, col: mu, al: 'center', a: .85 });
    c.restore();
  }
  const cards = [['cal', 'Every Friday', 'Weekly recap show'], ['chip', 'Global AI & Tech', 'Chips · Space · Security · Gadgets'], ['school', 'Nepal Education', 'Major local updates']], cw = 520, cg = 30, cx0 = (W - (3 * cw + 2 * cg)) / 2;
  cards.forEach((cd, j) => {
    const s0 = lt(sc, j), e = sp(t, s0, .7); if (e <= .001) return;
    const act = j === k ? ss(t, s0, .35) : (j < k ? 1 - ss(t, lt(sc, j + 1), .35) : 0), x = cx0 + j * (cw + cg), y = 684 + (1 - e) * 50;
    c.save(); c.globalAlpha *= clamp(e * 1.4);
    box(c, x, y, cw, 176, 36, mixc(p.cont, p.pri, act), null);
    shape(c, mixc(p.priC, p.onPri, act * .2), x + 84, y + 88, 48, 'cookie9', 'flower', act, T * .3 + j);
    icon(c, cd[0], x + 84, y + 88, 40, mixc(p.onPriC, p.pri, act), 2);
    txt(c, cd[1], x + 160, y + 82, { sz: 33, w: 700, col: mixc(p.ink, p.onPri, act) }); txt(c, cd[2], x + 160, y + 124, { sz: 21, w: 500, col: mixc(p.mute, p.onPri, act * .85) });
    c.restore();
  });
};

SC.fast = (c, t, sc, p, T) => {
  bg(c, p, T); const b = .35, k = cur(sc, t);
  let x = 120; x += chip(c, x, 176, 'QUICK ROUNDUP', { fill: p.pri, ink: p.onPri, a: sp(t, b, .5), s: .8 + .2 * sp(t, b, .5), ls: 1 }) + 12;
  chip(c, x, 176, sc.dates, { stroke: p.line, ink: p.mute, icon: 'cal', a: sp(t, b + .1, .5) });
  kinetic(c, 'Quick roundup', 120, 322, t - b - .1, { sz: 84, w: 700, maxW: 1200, col: p.ink });
  txt(c, sc.hn, 124, 384, { sz: 34, w: 500, col: p.mute, a: clamp(sp(t, b + .5, .6)) });
  const cw = 800, ch = 200, gx = 40, gy = 26, x0 = (W - (2 * cw + gx)) / 2;
  sc.lines.forEach((l, j) => {
    const s0 = lt(sc, j), e = sp(t, s0 - .1, .7); if (e <= .001) return;
    const act = j === k ? ss(t, s0, .35) : (j < k ? 1 - ss(t, lt(sc, j + 1), .35) : 0), cxp = x0 + (j % 2) * (cw + gx), cyp = 446 + Math.floor(j / 2) * (ch + gy) + (1 - e) * 50, s = 1 + .02 * act;
    c.save(); c.globalAlpha *= clamp(e * 1.4); c.translate(cxp + cw / 2, cyp + ch / 2); c.scale(s, s); c.translate(-(cxp + cw / 2), -(cyp + ch / 2));
    box(c, cxp, cyp, cw, ch, 40, mixc(p.cont, p.pri, act), null);
    shape(c, mixc(p.priC, p.onPri, act * .15), cxp + 100, cyp + ch / 2, 56, SHAPES6[j % 6], SHAPES6[(j + 1) % 6], act, T * .3 + j);
    txt(c, String(j + 1), cxp + 100, cyp + ch / 2 + 15, { sz: 44, w: 700, col: mixc(p.onPriC, p.pri, act), al: 'center' });
    const f = fitText(c, l.big, 540, 70, 700, 44);
    txt(c, f.lines.join(' '), cxp + 190, cyp + 94, { sz: Math.min(f.sz, 70), w: 700, col: mixc(p.ink, p.onPri, act) });
    txt(c, l.label, cxp + 192, cyp + 148, { sz: 26, w: 500, col: mixc(p.mute, p.onPri, act * .85) });
    c.restore();
  });
};

SC.outro = (c, t, sc, p, T) => {
  bg(c, p, T); const cx = 1400, cy = 478, R = 250, ent = sp(t, .3, .9), b = .25;
  kinetic(c, 'See you', 120, 400, t - b, { sz: 150, w: 700, maxW: 1000, col: p.ink });
  kinetic(c, 'next Friday', 120, 560, t - b - .25, { sz: 150, w: 700, maxW: 1100, col: p.pri });
  const a = sp(t, 1.0, .6); let x = 124;
  x += chip(c, x, 650, 'FRI 4 DEC 2026', { h: 56, sz: 24, fill: p.pri, ink: p.onPri, icon: 'cal', a, s: .8 + .2 * a, ls: 1 }) + 14;
  chip(c, x, 650, 'TECH · AI · EDUCATION', { h: 56, sz: 24, stroke: p.line, ink: p.mute, a: sp(t, 1.1, .6), s: .8 + .2 * sp(t, 1.1, .6), ls: 1 });
  txt(c, 'NEBians News  ·  Weekly Recap  ·  Episode 1', 124, 770, { sz: 28, w: 500, col: p.mute, a: clamp(sp(t, 1.3, .6)) });
  c.save(); c.translate(cx, cy); c.scale(ent, ent); c.translate(-cx, -cy);
  avatar(c, cx, cy, R, T, p, ampAt(T), { ring: p.priC, cheek: p.ter }); c.restore();
  swap(sc, t, (l, al, dy) => {
    const f = fitText(c, l.big, 520, 56, 700, 36), w = tw(c, f.lines[0], f.sz, 700) + 72, y = cy + R + 62 + dy;
    box(c, cx - w / 2, y, w, 84, 42, C(p.priC, al), null); txt(c, f.lines[0], cx, y + 42 + f.sz * .36, { sz: f.sz, w: 700, col: p.onPriC, al: 'center', a: al });
  });
};
