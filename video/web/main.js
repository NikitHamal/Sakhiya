// Material 3 Expressive Timeline Driver: render(T) draws one deterministic 1920x1080 frame

const SCT = {
  hello: 'blue',
  about: 'light',
  series: 'dark',
  n1: 'teal',
  n2: 'light',
  n3: 'dark',
  n4: 'teal',
  n5: 'light',
  edu: 'plum',
  fast: 'blue',
  outro: 'blue'
};

const WIPE = ['flower', 'sunny', 'clover', 'cookie12', 'burst', 'pebble', 'squircle', 'gem'];

const lt = (sc, id) => {
  for (const l of sc.lines) if (l.id === id) return l.t - sc.t0;
  return 0;
};
const ld = (sc, id) => {
  for (const l of sc.lines) if (l.id === id) return l.d;
  return 1;
};

function drawScene(c, i, T) {
  const sc = TL.scenes[i];
  const th = TH[SCT[sc.id] || 'light'];
  c.save();
  shell(c, th, sc, T - sc.t0, T);
  if (SC[sc.id]) SC[sc.id](c, T - sc.t0, sc, th, T);
  c.restore();
  return th;
}

window.render = (T) => {
  const c = window.ctx, TLs = TL.scenes;
  let i = TLs.findIndex(s => T >= s.t0 && T < s.t1);
  if (i < 0) i = TLs.length - 1;
  c.setTransform(1, 0, 0, 1, 0, 0);
  c.clearRect(0, 0, W, H);

  if (i > 0 && T - TLs[i].t0 < .88) {
    drawScene(c, i - 1, T);
    const ox = [W / 2, W * .28, W * .72, W / 2][i % 4];
    const oy = [H / 2, H * .65, H * .35, H * .55][i % 4];
    if (wipe(c, T, TLs[i].t0 - .04, '', .88, ox, oy, WIPE[i % WIPE.length])) {
      drawScene(c, i, T);
      c.restore();
    } else {
      drawScene(c, i, T);
    }
  } else {
    drawScene(c, i, T);
  }

  const th = TH[SCT[TLs[i].id] || 'light'];
  chrome(c, th, T, TL, i === 0);
  return i;
};
