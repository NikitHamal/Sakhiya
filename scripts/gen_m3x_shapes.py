"""Generate docs/m3x/shapes.css: 120-point clip-path silhouettes (morphable) and spring easings (M3X design system)."""
import math, pathlib
N = 120
OUT = pathlib.Path(__file__).resolve().parent.parent / "docs" / "m3x" / "shapes.css"
def lobed(k, a, phase=0.0): return lambda t: 1 + a * math.cos(k * t + phase)
def superellipse(n): return lambda t: (abs(math.cos(t)) ** n + abs(math.sin(t)) ** n) ** (-1 / n)
SHAPES = {
  'circle': lambda t: 1.0, 'cookie': lobed(9, 0.065), 'sunny': lobed(8, 0.11), 'flower': lobed(5, 0.17), 'clover': lobed(4, 0.2),
  'burst': lobed(12, 0.12), 'squircle': superellipse(4), 'gem': superellipse(1.55),
  'pebble': lambda t: 1 + 0.1 * math.cos(2 * t - 0.6) + 0.05 * math.cos(3 * t + 0.4),
}
def polygon(fn):
  rs = [fn(2 * math.pi * i / N - math.pi / 2) for i in range(N)]; top = max(rs); pts = []
  for i, r in enumerate(rs):
    t = 2 * math.pi * i / N - math.pi / 2
    pts.append(f'{50 + 50 * r / top * math.cos(t):.2f}% {50 + 50 * r / top * math.sin(t):.2f}%')
  return 'polygon(' + ','.join(pts) + ')'
def spring(zeta, omega, steps=48, duration=1.0):
  wd = omega * math.sqrt(1 - zeta ** 2); out = []
  for i in range(steps + 1):
    t = duration * i / steps
    v = 1 - math.exp(-zeta * omega * t) * (math.cos(wd * t) + zeta * omega / wd * math.sin(wd * t))
    out.append(f'{v:.3f}'.rstrip('0').rstrip('.') or '0')
  out[-1] = '1'; return 'linear(' + ','.join(out) + ')'
lines = [':root{'] + [f'--shape-{n}:{polygon(f)};' for n, f in SHAPES.items()]
lines += ['--spring-fast:cubic-bezier(.3,1.3,.5,1);--spring:cubic-bezier(.3,1.5,.5,1);--spring-soft:cubic-bezier(.25,1.2,.4,1);', '}',
  '@supports (transition-timing-function:linear(0,1)){:root{',
  f'--spring-fast:{spring(0.62, 24)};', f'--spring:{spring(0.5, 15)};', f'--spring-soft:{spring(0.72, 11)};', '}}']
OUT.write_text('\n'.join(lines) + '\n'); print("wrote", OUT, OUT.stat().st_size)
