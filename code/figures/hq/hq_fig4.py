#!/usr/bin/env python3
# coding: UTF-8
"""Figure 4 (high standard) - what the protein channel encodes."""
import collections
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from style_hq import (new_fig, letter, tidy, save, plt, CB, CO, CG, CGRAY,
                      FS_VAL, FS_BOX)

BASE = Path(os.environ.get('PROMOSCAN_ROOT',
                          Path(__file__).resolve().parents[3]))
OUT = Path(os.environ.get('FIG_OUT', BASE / 'figures'))

d = json.loads((BASE / 'results/bench50_protein_controls.json').read_text())
rows = d['per_dataset']

fig, gs = new_fig(5.3, left=0.095, right=0.965, bottom=0.13, top=0.93,
                  wspace=0.40, hspace=0.70)

# ---------------- A  swap the protein ----------------
ax = fig.add_subplot(gs[0, 0])
keys = ['correct', 'centroid', 'farthest', 'nearest', 'random']
labels = ['correct', 'centroid', 'farthest\nTF', 'nearest\nTF', 'random\nTF']
rng = np.random.default_rng(7)
for i, k in enumerate(keys):
    v = np.array([r[k] for r in rows])
    ax.bar(i, v.mean(), .6, color=CB if k == 'correct' else CGRAY, linewidth=0)
    ax.scatter(np.full(len(v), i) + rng.normal(0, .055, len(v)), v, s=2.6, color='#333',
               alpha=.28, linewidths=0, zorder=3)
    ax.text(i, v.mean() + .003, f'{v.mean():.3f}', ha='center', fontsize=FS_VAL,
            fontweight='bold' if k == 'correct' else 'normal')
ax.set_xticks(range(5))
ax.set_xticklabels(labels, fontsize=6.2)
ax.set_xlim(-0.65, 4.65)
ax.set_ylim(0.78, 0.965)
ax.set_ylabel('ROC-AUC (50 datasets)')
ax.text(0.03, 0.05, 'any wrong identity:\n50/50 datasets worse\np = 1.8×10⁻¹⁵',
        transform=ax.transAxes, fontsize=5.8, color='#333', va='bottom',
        bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='none', alpha=0.88))
tidy(ax)
letter(ax, 'A')

# ---------------- B  similarity vs loss ----------------
ax = fig.add_subplot(gs[0, 1])
lv = np.array([k for r in rows for k in ('nearest', 'random', 'farthest')])
cos = np.array([r[k + '_cos'] for r in rows for k in ('nearest', 'random', 'farthest')])
drop = np.array([r['correct'] - r[k] for r in rows for k in ('nearest', 'random', 'farthest')])
colors = {'nearest': CG, 'random': CGRAY, 'farthest': CO}
for k in ('nearest', 'random', 'farthest'):
    m = lv == k
    ax.scatter(cos[m], drop[m], s=6, color=colors[k], alpha=.55, linewidths=0, label=k)
b, a0 = np.polyfit(cos, drop, 1)
xs = np.linspace(cos.min(), cos.max(), 50)
ax.plot(xs, a0 + b * xs, color='#333', lw=0.8, ls='--')
rho, p = spearmanr(cos, drop)
ax.set_xlabel('Cosine similarity to correct protein')
ax.set_ylabel('AUC loss')
ax.text(.04, .96, f'ρ = {rho:+.2f}, p = {p:.2f}\nn = {len(cos)} swaps\ndiscrete key,\nnot a gradient',
        transform=ax.transAxes, va='top', fontsize=5.9,
        bbox=dict(boxstyle='round,pad=.26', fc='#f7f7f7', ec='#dddddd', lw=.5))
ax.legend(loc='lower right', fontsize=5.9, handletextpad=0.2, borderpad=0.15,
          labelspacing=0.2, markerscale=1.4)
tidy(ax, 'both')
letter(ax, 'B')

# ---------------- C  same TF, different cell lines ----------------
ax = fig.add_subplot(gs[1, :])
g = collections.defaultdict(list)
for r in rows:
    g[r['symbol']].append(r)
multi = {k: v for k, v in g.items() if len(v) > 1}
ordered = sorted(multi.items(), key=lambda kv: -np.mean([q['correct'] for q in kv[1]]))
xs = np.arange(len(ordered))
for i, (sym, lst) in enumerate(ordered):
    c = [q['correct'] for q in lst]
    w = [q['random'] for q in lst]
    for v in c:
        ax.plot([i - .2, i + .2], [v, v], color=CB, lw=0.9)
    for v in w:
        ax.plot([i - .2, i + .2], [v, v], color=CGRAY, lw=0.8, alpha=.85)
    ax.scatter([i] * len(c), c, s=6, color=CB, zorder=3)
    ax.scatter([i] * len(w), w, s=6, color=CGRAY, zorder=3)
ax.set_xticks(xs)
ax.set_xticklabels([k for k, _ in ordered], fontsize=6.0, rotation=45, ha='right')
ax.set_xlim(-1, len(ordered))
ax.set_ylim(0.75, 1.0)
ax.set_ylabel('ROC-AUC')
ax.text(0.995, 0.06, 'blue = correct protein    grey = random TF', transform=ax.transAxes,
        ha='right', fontsize=5.9, color='#444')
tidy(ax)
letter(ax, 'C', dx=-0.055, dy=1.03)

save(fig, OUT, 'Figure4', 'Figure4')
