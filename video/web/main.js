// timeline driver: render(T) draws one deterministic frame

const SCT = { hello: 'blue', about: 'light', series: 'dark', n1: 'light', n2: 'dark', n3: 'blue', n4: 'light', n5: 'dark', fast: 'blue', app: 'light', outro: 'blue' };
const WIPE = ['flower', 'sunny', 'clover', 'cookie', 'burst', 'pebble', 'squircle'];
const lt = (sc, id) => { for (const l of sc.lines) if (l.id === id) return l.t - sc.t0; return 0; };
const ld = (sc, id) => { for (const l of sc.lines) if (l.id === id) return l.d; return 1; };
function drawScene(c, i, T) { const sc = TL.scenes[i], th = TH[SCT[sc.id]]; c.save(); shell(c, th, sc, T - sc.t0); SC[sc.id](c, T - sc.t0, sc, th, T); c.restore(); return th; }
window.render = (T) => {
  const c = window.ctx, TLs = TL.scenes; let i = TLs.findIndex(s => T >= s.t0 && T < s.t1); if (i < 0) i = TLs.length - 1;
  c.setTransform(1, 0, 0, 1, 0, 0); c.clearRect(0, 0, W, H);
  if (i > 0 && T - TLs[i].t0 < .95) { drawScene(c, i - 1, T);
    const ox = [W / 2, W * .25, W * .75, W / 2][i % 4], oy = [H / 2, H * .7, H * .3, H * .6][i % 4];
    if (wipe(c, T, TLs[i].t0 - .05, '', .95, ox, oy, WIPE[i % 7])) { drawScene(c, i, T); c.restore(); } else drawScene(c, i, T); }
  else drawScene(c, i, T);
  const th = TH[SCT[TLs[i].id]]; chrome(c, th, T, TL, i === 0);
  if (T > TL.total - 1.2) { c.fillStyle = C.blue; c.globalAlpha = P(T, TL.total - 1.2, TL.total - .15) * .0; c.fillRect(0, 0, W, H); c.globalAlpha = 1; }
  return i;
};
