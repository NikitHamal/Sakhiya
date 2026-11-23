#!/usr/bin/env python3
"""Rank evaluated checkpoints (synthesized-audio metrics) -> prints best tag. usage: select_best.py <prefix> [extra tags...]"""
import sys, json, glob, os
pre = sys.argv[1]; rows = []
for sj in glob.glob('/work/sp/evals/*/scores.json'):
    s = json.load(open(sj))['summary']
    if s['tag'].startswith(pre) or s['tag'] in sys.argv[2:]:
        c = -3 * s['cer'] - 1.0 * s['cer_vedic'] - 1.0 * s['cer_tech'] + 0.3 * s['utmos'] + 1.0 * s['spk_sim'] - 0.5 * s['hindi'] - 0.002 * s['clicks']
        rows.append((c, s['tag'], s))
rows.sort(key=lambda r: -r[0])
for c, t, s in rows: print(f"{c:.3f} {t} cer={s['cer']:.3f} tech={s['cer_tech']:.3f} ved={s['cer_vedic']:.3f} utmos={s['utmos']:.2f} spk={s['spk_sim']:.2f} hindi={s['hindi']} clicks={s['clicks']:.0f}")
print('BEST', rows[0][1] if rows else 'none')
