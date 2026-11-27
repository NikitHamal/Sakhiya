// NEBians News — timeline driver: render(T) draws one deterministic 1920x1080 frame
const KIND = { hello: 'hello', about: 'about', series: 'series', story: 'story', fast: 'fast', outro: 'outro' };
const WIPE_SHAPES = ['cookie12', 'flower', 'sunny', 'clover', 'burst', 'cookie9', 'pebble', 'penta'];
const ORIGIN = [[.5, .5], [.3, .62], [.72, .4], [.5, .55]];
const PAL = {};
const palOf = sc => PAL[sc.id] || (PAL[sc.id] = palette(sc.hue, sc.dark));
function drawScene(c, i, T) {
  const sc = TL.scenes[i], p = palOf(sc); c.save(); SC[KIND[sc.kind]](c, T - sc.t0, sc, p, T); c.restore(); return p;
}
window.render = T => {
  const c = window.ctx, S = TL.scenes; let i = S.findIndex(s => T >= s.t0 && T < s.t1); if (i < 0) i = S.length - 1;
  c.setTransform(1, 0, 0, 1, 0, 0); c.clearRect(0, 0, W, H); const WD = .9, dt = T - S[i].t0;
  if (i > 0 && dt < WD) {
    drawScene(c, i - 1, T);
    const q = eio(dt / WD), o = ORIGIN[i % 4], pn = palOf(S[i]), sh = WIPE_SHAPES[i % WIPE_SHAPES.length];
    c.save(); shapePath(c, W * o[0], H * o[1], 60 + q * 1900, sh, 'circle', 0, q * 1.3); c.clip(); drawScene(c, i, T); c.restore();
    shapeLine(c, pn.pri, 16 * (1 - q), W * o[0], H * o[1], 60 + q * 1900, sh, 'circle', 0, q * 1.3);
  } else drawScene(c, i, T);
  const p = palOf(S[i]); chrome(c, T, i, p);
  const fo = P(T, TL.total - 1.0, TL.total - .1); if (fo > 0) { c.fillStyle = C(p.bg, fo); c.fillRect(0, 0, W, H); }
  return i;
};
