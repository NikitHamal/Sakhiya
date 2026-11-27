// NEBians News — shared parts: ambient background, chips, Sakhiya avatar, subtitles, broadcast chrome
const SC = {};
const SHAPES6 = ['cookie12', 'flower', 'sunny', 'clover', 'burst', 'pebble'];
const lt = (sc, j) => sc.lines[j].t - sc.t0;
function cur(sc, t) { let k = -1; sc.lines.forEach((l, j) => { if (t >= l.t - sc.t0 - .05) k = j; }); return k; }
// draws each line's content as it enters / leaves: fn(line, alpha, dy, j, e)
function swap(sc, t, fn, o = {}) {
  const d = o.d ?? .6, dy = o.dy ?? 50, lead = o.lead ?? .05, k = cur(sc, t);
  for (const j of [k - 1, k]) {
    if (j < 0) continue;
    const e = sp(t, lt(sc, j) + lead, d), nx = j + 1 < sc.lines.length ? lt(sc, j + 1) : 1e9;
    const x = eo3(P(t, nx, nx + .28)), a = clamp(e * 1.4) * (1 - x);
    if (a > .01) fn(sc.lines[j], a, (1 - e) * dy - x * dy, j, e);
  }
}
const ampAt = T => { const E = window.ENV, f = Math.round(T * 30); let s = 0, n = 0; for (let i = f - 1; i <= f + 1; i++) if (E[i]) { s += E[i][0]; n++; } return n ? clamp(s / n) : 0; };
const bandsAt = T => (window.ENV[Math.min(window.ENV.length - 1, Math.round(T * 30))] || []);

function bg(c, p, T) {
  c.fillStyle = C(p.bg); c.fillRect(0, 0, W, H);
  shape(c, mixc(p.bg, p.cont, .75), 1840, 130, 540, 'cookie12', 'cookie9', .5 + .5 * Math.sin(T * .3), T * .05);
  shape(c, mixc(p.bg, p.cont, .55), 30, 1050, 430, 'flower', 'sunny', .5 + .5 * Math.sin(T * .25 + 1), -T * .06);
}
function chip(c, x, y, label, o = {}) {
  const h = o.h || 50, sz = o.sz || 22, pad = 22, iw = o.icon ? h * .5 + 10 : 0, w = tw(c, label, sz, 600) + (o.ls || 0) * label.length + pad * 2 + iw, a = o.a ?? 1, s = o.s ?? 1;
  if (a <= .01) return w;
  c.save(); c.globalAlpha *= a; c.translate(x + w / 2, y + h / 2); c.scale(s, s); c.translate(-(x + w / 2), -(y + h / 2));
  box(c, x, y, w, h, h / 2, o.fill, o.stroke, 2);
  if (o.icon) icon(c, o.icon, x + pad + h * .25, y + h / 2, h * .5, o.ink, 2);
  txt(c, label, x + pad + iw, y + h / 2 + sz * .36, { sz, w: 600, col: o.ink, ls: o.ls || 0 }); c.restore(); return w;
}
function avatar(c, x, y, R, T, p, amp, o = {}) {
  c.save(); c.translate(x, y + Math.sin(T * 1.3) * R * .02); c.rotate(Math.sin(T * .7) * .05);
  if (o.ring) {
    const b = bandsAt(T), n = 56, r1 = R * 1.13;
    for (let i = 0; i < n; i++) {
      const a = i / n * TAU - Math.PI / 2, idx = 1 + Math.floor(Math.abs(i - n / 2) / (n / 2) * 30), v = clamp(b[idx] || 0), len = R * .05 + v * R * .30;
      c.beginPath(); c.moveTo(Math.cos(a) * r1, Math.sin(a) * r1); c.lineTo(Math.cos(a) * (r1 + len), Math.sin(a) * (r1 + len));
      c.strokeStyle = C(o.ring, .35 + .5 * v); c.lineWidth = R * .035; c.lineCap = 'round'; c.stroke();
    }
  }
  shape(c, o.body || p.pri, 0, 0, R, 'cookie12', 'cookie9', .5 + .5 * Math.sin(T * .8), T * .12);
  const ink = o.ink || p.onPri, bl = (T % 3.6) > 3.48 ? .12 : 1;
  c.fillStyle = C(ink);
  for (const s of [-1, 1]) { c.beginPath(); c.ellipse(s * R * .27, -R * .08, R * .075, R * .12 * bl, 0, 0, TAU); c.fill(); }
  c.fillStyle = C(o.cheek || p.ter, .5);
  for (const s of [-1, 1]) { c.beginPath(); c.arc(s * R * .45, R * .14, R * .07, 0, TAU); c.fill(); }
  c.fillStyle = C(ink); c.beginPath(); c.ellipse(0, R * .24, R * (.15 + .06 * amp), R * (.035 + .17 * amp), 0, 0, TAU); c.fill();
  c.restore();
}

// ---- compact bilingual subtitles ----
const _pg = {};
function splitBal(s, n) {
  const w = s.split(' '), tot = s.length, out = Array.from({ length: n }, () => []); let cum = 0;
  w.forEach(x => { const pos = (cum + x.length / 2) / tot; out[Math.min(n - 1, Math.floor(pos * n))].push(x); cum += x.length + 1; });
  return out.map(a => a.join(' ')).filter(Boolean);
}
function pagesOf(l) {
  if (_pg[l.id]) return _pg[l.id];
  const neS = l.ne.split(/(?<=।)\s*/).filter(Boolean), enS = l.en.split(/(?<=[.!?])\s+/).filter(Boolean); let ne = [], en = [];
  if (neS.length === enS.length) {
    neS.forEach((x, i) => { const a = splitBal(x, Math.max(1, Math.ceil(x.length / 62))); a.forEach(q => { ne.push(q); en.push(enS[i]); }); });
  } else { ne = splitBal(l.ne, Math.max(1, Math.ceil(l.ne.length / 56))); en = splitBal(l.en, ne.length); }
  const tot = l.ne.length; let a = 0; const cuts = ne.map(x => (a += x.length + 1) / (tot + ne.length));
  return _pg[l.id] = { ne, en, cuts };
}
function caption(c, T, p, sc) {
  let best = null, prevEnd = -9, nextStart = 1e9;
  const all = []; TL.scenes.forEach(s => s.lines.forEach(l => all.push(l)));
  for (let i = 0; i < all.length; i++) {
    const l = all[i];
    if (T >= l.t - .08) { best = l; prevEnd = i ? all[i - 1].t + all[i - 1].d : -9; nextStart = i + 1 < all.length ? all[i + 1].t : 1e9; }
  }
  if (!best || T > best.t + best.d + .5) return;
  const a = (l => { const inn = best.t - prevEnd < 1.1 ? 1 : eo3(P(T, best.t - .08, best.t + .2)), out = nextStart - (best.t + best.d) < 1.1 ? 1 : 1 - eo3(P(T, best.t + best.d + .15, best.t + best.d + .5)); return inn * out; })();
  if (a <= .01) return;
  const pg = pagesOf(best), fr = clamp((T - best.t) / best.d); let k = pg.cuts.findIndex(x => fr < x); if (k < 0) k = pg.ne.length - 1;
  const ne = pg.ne[k], en = pg.en[Math.min(k, pg.en.length - 1)];
  const w = Math.max(tw(c, ne, 30, 600), tw(c, en, 21, 500)) + 64 + 86, h = 90, x0 = W / 2 - w / 2, y0 = 946, lift = (1 - a) * 14;
  c.save(); c.globalAlpha *= a;
  box(c, x0, y0 + lift, w, h, 30, C(p.surf, .94), C(p.line, .8), 1.5);
  avatar(c, x0 + 52, y0 + lift + h / 2, 28, T, p, ampAt(T) * 1.1, {});
  const tx = x0 + 86 + (w - 86) / 2;
  txt(c, ne, tx, y0 + lift + 40, { sz: 30, w: 600, col: p.ink, al: 'center' });
  txt(c, en, tx, y0 + lift + 70, { sz: 21, w: 500, col: p.mute, al: 'center' });
  c.restore();
}
function wavy(c, x0, x1, y, prog, T, p) {
  const xe = x0 + (x1 - x0) * prog;
  c.lineCap = 'round'; c.lineWidth = 5; c.strokeStyle = C(p.line, .7); c.beginPath(); c.moveTo(xe + 14, y); c.lineTo(x1, y); c.stroke();
  c.beginPath(); c.arc(x1, y, 3, 0, TAU); c.fillStyle = C(p.pri); c.fill();
  c.strokeStyle = C(p.pri); c.beginPath();
  for (let x = x0; x <= xe; x += 4) { const yy = y + Math.sin(x * .2 - T * 5) * 3.2; x === x0 ? c.moveTo(x, yy) : c.lineTo(x, yy); }
  c.stroke();
}
function chrome(c, T, i, p) {
  const sc = TL.scenes[i], a = i === 0 ? ss(T, 2.6, .5) : 1;
  if (a > .01) {
    c.save(); c.globalAlpha *= a;
    shape(c, p.pri, 150, 66, 22, 'cookie12', 'cookie9', .5, T * .4); txt(c, 'N', 150, 74, { sz: 22, w: 700, col: p.onPri, al: 'center' });
    txt(c, 'NEBians News', 186, 75, { sz: 28, w: 700, col: p.ink, ls: .2 });
    const lab = 'Episode 1  ·  Fri 27 Nov 2026', ww = tw(c, lab, 20, 600) + 40;
    box(c, W - 120 - ww, 44, ww, 44, 22, C(p.surf, .9), C(p.line, .8), 1.5); txt(c, lab, W - 120 - ww / 2, 73, { sz: 20, w: 600, col: p.mute, al: 'center' });
    c.restore();
  }
  caption(c, T, p, sc);
  wavy(c, 120, W - 120, 1064, clamp(T / TL.total), T, p);
}
