#!/usr/bin/env python3
# coding: UTF-8
"""Figure S1 (high standard) - five-fold identity task on held-out TFs.

Drawn at final print width (17.1 cm), 2x2 layout, 7 pt body text.
Panels: A per fold; B across folds; C seen vs held-out; D pair-level distribution.
"""
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from style_hq import (new_fig, letter, tidy, save, plt, CB, CO, CG, CGRAY,
                      FS_VAL, FS_BOX)

BASE = Path(os.environ.get('PROMOSCAN_ROOT',
                          Path(__file__).resolve().parents[3]))
RES = BASE / 'results'
OUT = Path(os.environ.get('FIG_OUT', BASE / 'figures'))

folds, n = [], 5
for i in range(n):
    d = json.loads((RES / f'sameharness_FOLD{i}.json').read_text())
    h, s = d['holdout'], d['seen']
    folds.append(dict(
        i=i,
        ho_own=h[f'FOLD{i}'], ho_shuf=h[f'FOLD{i}_shuf'], ho_dna=h['dnaonly'],
        se_own=s[f'FOLD{i}'], se_dna=s['dnaonly'],
        raw_own=np.array(h['_raw'][f'FOLD{i}']), raw_shuf=np.array(h['_raw'][f'FOLD{i}_shuf']),
        raw_dna=np.array(h['_raw']['dnaonly']),
    ))

fig, gs = new_fig(5.2, left=0.095, right=0.965, bottom=0.10, top=0.93,
                  wspace=0.42, hspace=0.60)

rng = np.random.default_rng(42)

# ---------------- A  per fold ----------------
ax = fig.add_subplot(gs[0, 0])
x = np.arange(n)
w = 0.26
series = [('DNA-only', 'ho_dna', CGRAY), ('ProMoScan', 'ho_own', CB),
          ('shuffled protein', 'ho_shuf', CO)]
for k, (lab, key, col) in enumerate(series):
    vals = [f[key] for f in folds]
    ax.bar(x + (k - 1) * w, vals, w, color=col, label=lab, linewidth=0)
    for xi, f in zip(x + (k - 1) * w, folds):
        raw = f['raw_dna'] if key == 'ho_dna' else (f['raw_own'] if key == 'ho_own' else f['raw_shuf'])
        pts = raw[rng.choice(len(raw), size=min(40, len(raw)), replace=False)]
        ax.scatter(np.full(len(pts), xi) + rng.normal(0, 0.022, len(pts)), pts,
                   s=2.0, color='#222', alpha=0.28, linewidths=0, zorder=3)
ax.axhline(0.5, color='#666', ls=':', lw=0.8)
ax.text(-0.48, 0.508, 'chance', fontsize=5.9, color='#666')
ax.set_xticks(x)
ax.set_xticklabels([f'fold {i+1}' for i in range(n)])
ax.set_xlim(-0.6, 4.6)
ax.set_ylim(0.38, 0.80)
ax.set_ylabel('ΔAUC, held-out TFs')
ax.legend(loc='upper right', fontsize=5.9, handlelength=1.0, borderpad=0.2, ncol=1)
tidy(ax)
letter(ax, 'A')

# ---------------- B  across folds ----------------
ax = fig.add_subplot(gs[0, 1])
means = {'DNA-only': np.array([f['ho_dna'] for f in folds]),
         'ProMoScan': np.array([f['ho_own'] for f in folds]),
         'shuffled\nprotein': np.array([f['ho_shuf'] for f in folds])}
cols = [CGRAY, CB, CO]
for k, (lab, arr) in enumerate(means.items()):
    ax.scatter(np.full(len(arr), k) + rng.normal(0, 0.035, len(arr)), arr, s=14,
               color=cols[k], zorder=3, linewidths=0)
    ax.hlines(arr.mean(), k - 0.22, k + 0.22, color=cols[k], lw=1.6, zorder=4)
    ax.text(k, arr.mean() + 0.010, f'{arr.mean():.3f}±{arr.std(ddof=1):.3f}',
            ha='center', fontsize=FS_VAL, color=cols[k])
ax.axhline(0.5, color='#666', ls=':', lw=0.8)
ax.set_xticks(range(3))
ax.set_xticklabels(list(means), fontsize=6.2)
ax.set_xlim(-0.55, 2.55)
ax.set_ylim(0.38, 0.68)
ax.set_ylabel('ΔAUC (mean of folds)')
tidy(ax)
letter(ax, 'B')

# ---------------- C  seen vs held-out ----------------
ax = fig.add_subplot(gs[1, 0])
for f in folds:
    ax.plot([0, 1], [f['se_own'], f['ho_own']], '-o', color=CB, ms=3.0, lw=0.8, alpha=0.8)
ax.text(0.02, np.mean([f['se_own'] for f in folds]) + 0.02, 'seen TFs', fontsize=5.9, color='#333')
ax.text(0.52, np.mean([f['ho_own'] for f in folds]) - 0.05, 'held-out TFs', fontsize=5.9, color='#333')
ax.set_xticks([0, 1])
ax.set_xticklabels(['seen', 'held-out'])
ax.set_xlim(-0.25, 1.25)
ax.set_ylim(0.45, 0.85)
ax.set_ylabel('ΔAUC (ProMoScan)')
tidy(ax)
letter(ax, 'C')

# ---------------- D  pair-level distribution ----------------
ax = fig.add_subplot(gs[1, 1])
delta = np.concatenate([f['raw_own'] - f['raw_dna'] for f in folds])
ax.hist(delta, bins=np.linspace(-0.6, 0.8, 43), color=CB, linewidth=0)
ax.axvline(0, color='#333', lw=0.7)
ax.axvline(delta.mean(), color=CO, lw=1.0, ls='--')
ax.text(delta.mean() + 0.02, ax.get_ylim()[1] * 0.92, f'mean {delta.mean():+.3f}',
        color=CO, fontsize=5.9, va='top')
win = float((delta > 0).mean()) * 100
ax.text(0.97, 0.74, f'{win:.0f}% of pairs improved\nvs DNA-only (n = {len(delta)})',
        transform=ax.transAxes, ha='right', fontsize=5.9,
        bbox=dict(boxstyle='round,pad=0.25', fc='#f7f7f7', ec='#dddddd', lw=0.5))
ax.set_xlabel('per-pair ΔAUC (ProMoScan − DNA-only)')
ax.set_ylabel('Pairs')
tidy(ax)
letter(ax, 'D')

save(fig, OUT, 'FigureS1', 'FigureS1')
