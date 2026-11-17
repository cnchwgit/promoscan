#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Figure 3 · 未见 TF 上的身份判别（五折）

输入 results/sameharness_FOLD{0..4}.json（holdout/seen 均含逐对 _raw）
输出 paper/figures/Fig3_identity_fivefold.{png,pdf}

面板
  A 逐折：DNA-only / ProMoScan / 蛋白打乱（逐对散点 + 均值）
  B 五折均值 ± SD，随机基线 0.5
  C seen vs holdout（同折配对），展示"不是靠见过的 TF"
  D 逐对 ΔAUC（ProMoScan − DNA-only）分布，五折合并
"""
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

BASE = Path(os.environ.get('PROMOSCAN_ROOT',
                          Path(__file__).resolve().parents[2]))
RES = BASE / 'results'
OUT = BASE / 'figures'
OUT.mkdir(parents=True, exist_ok=True)

CB, CO, CG, CGRAY = '#0072B2', '#D55E00', '#009E73', '#9aa0a6'
plt.rcParams.update({'font.size': 9, 'axes.linewidth': 0.8, 'axes.spines.top': False,
                     'axes.spines.right': False, 'pdf.fonttype': 42, 'svg.fonttype': 'none',
                     'axes.unicode_minus': False, 'figure.dpi': 120})

folds, n = [], 5
for i in range(n):
    d = json.loads((RES / f'sameharness_FOLD{i}.json').read_text())
    h, s = d['holdout'], d['seen']
    folds.append(dict(
        i=i,
        ho_own=h[f'FOLD{i}'], ho_shuf=h[f'FOLD{i}_shuf'], ho_dna=h['dnaonly'],
        se_own=s[f'FOLD{i}'], se_dna=s['dnaonly'],
        raw_own=np.array(h['_raw'][f'FOLD{i}']), raw_shuf=np.array(h['_raw'][f'FOLD{i}_shuf']),
        raw_dna=np.array(h['_raw']['dnaonly']), raw_seen=np.array(s['_raw'][f'FOLD{i}']),
        n=h['n'],
    ))

fig = plt.figure(figsize=(15.0, 4.2))
gs = fig.add_gridspec(1, 4, width_ratios=[1.3, 0.8, 0.85, 1.15], wspace=0.34)

# ---------- A ----------
ax = fig.add_subplot(gs[0, 0])
x = np.arange(n)
w = 0.26
rng = np.random.default_rng(42)
series = [('DNA-only', 'ho_dna', CGRAY), ('ProMoScan', 'ho_own', CB), ('Protein shuffled', 'ho_shuf', CO)]
for k, (lab, key, col) in enumerate(series):
    vals = [f[key] for f in folds]
    ax.bar(x + (k - 1) * w, vals, w, color=col, label=lab, linewidth=0,
           alpha=0.92 if k != 0 else 0.75)
    for xi, f in zip(x + (k - 1) * w, folds):
        raw = f['raw_dna'] if key == 'ho_dna' else (f['raw_own'] if key == 'ho_own' else f['raw_shuf'])
        pts = raw[rng.choice(len(raw), size=min(40, len(raw)), replace=False)]
        ax.scatter(np.full(len(pts), xi) + rng.normal(0, 0.022, len(pts)), pts,
                   s=3.2, color='#222', alpha=0.30, linewidths=0, zorder=3)
ax.axhline(0.5, color='#666', ls=':', lw=1.0)
ax.text(-0.45, 0.503, 'chance 0.5', fontsize=6.4, color='#666')
ax.set_xticks(x)
ax.set_xticklabels([f'fold {i}' for i in range(n)])
ax.set_ylim(0.38, 0.83)
ax.set_ylabel('ΔAUC, held-out TFs')
ax.set_title('A  Five-fold held-out TFs', fontsize=9.2, loc='left')
ax.legend(fontsize=6.6, frameon=False, loc='upper right', ncol=1)
ax.grid(axis='y', color='#eeeeee', lw=0.7)
ax.set_axisbelow(True)

# ---------- B ----------
ax = fig.add_subplot(gs[0, 1])
means = {
    'DNA-only': np.array([f['ho_dna'] for f in folds]),
    'ProMoScan': np.array([f['ho_own'] for f in folds]),
    'Shuffled\nprotein': np.array([f['ho_shuf'] for f in folds]),
}
cols = [CGRAY, CB, CO]
for k, (lab, arr) in enumerate(means.items()):
    ax.scatter(np.full(len(arr), k) + rng.normal(0, 0.035, len(arr)), arr, s=22,
               color=cols[k], zorder=3, linewidths=0)
    ax.hlines(arr.mean(), k - 0.22, k + 0.22, color=cols[k], lw=2.2, zorder=4)
    ax.text(k, arr.mean() + 0.012, f'{arr.mean():.3f}±{arr.std(ddof=1):.3f}',
            ha='center', fontsize=6.6, color=cols[k])
ax.axhline(0.5, color='#666', ls=':', lw=1.0)
ax.set_xticks(range(3))
ax.set_xticklabels(list(means), fontsize=7.4)
ax.set_ylim(0.38, 0.70)
ax.set_ylabel('ΔAUC (mean of folds)')
ax.set_title('B  Across folds', fontsize=9.2, loc='left')
ax.grid(axis='y', color='#eeeeee', lw=0.7)
ax.set_axisbelow(True)

# ---------- C ----------
ax = fig.add_subplot(gs[0, 2])
for f in folds:
    ax.plot([0, 1], [f['se_own'], f['ho_own']], '-o', color=CB, ms=3.4, lw=0.9, alpha=0.85)
ax.text(0.02, np.mean([f['se_own'] for f in folds]) + 0.02, 'seen TFs', fontsize=6.8, color='#333')
ax.text(0.55, np.mean([f['ho_own'] for f in folds]) - 0.045, 'held-out TFs', fontsize=6.8, color='#333')
ax.set_xticks([0, 1])
ax.set_xticklabels(['seen', 'held-out'], fontsize=7.6)
ax.set_xlim(-0.25, 1.25)
ax.set_ylim(0.45, 0.88)
ax.set_ylabel('ΔAUC (ProMoScan)')
ax.set_title('C  Same model, seen vs unseen', fontsize=9.2, loc='left')
ax.grid(axis='y', color='#eeeeee', lw=0.7)
ax.set_axisbelow(True)

# ---------- D ----------
ax = fig.add_subplot(gs[0, 3])
delta = np.concatenate([f['raw_own'] - f['raw_dna'] for f in folds])
bins = np.linspace(-0.6, 0.8, 43)
ax.hist(delta, bins=bins, color=CB, alpha=0.85, linewidth=0)
ax.axvline(0, color='#333', lw=0.9)
ax.axvline(delta.mean(), color=CO, lw=1.4, ls='--')
ax.text(delta.mean() + 0.02, ax.get_ylim()[1] * 0.9, f'mean {delta.mean():+.3f}',
        color=CO, fontsize=6.8, va='top')
win = float((delta > 0).mean()) * 100
ax.text(0.97, 0.72, f'{win:.0f}% of pairs\nimproved vs DNA-only\n(n = {len(delta)})',
        transform=ax.transAxes, ha='right', fontsize=6.8,
        bbox=dict(boxstyle='round,pad=0.3', fc='#f7f7f7', ec='#dddddd', lw=0.7))
ax.set_xlabel('per-pair ΔAUC (ProMoScan − DNA-only)')
ax.set_ylabel('Pairs')
ax.set_title('D  Pair-level distribution (all folds)', fontsize=9.2, loc='left')
ax.grid(axis='y', color='#eeeeee', lw=0.7)
ax.set_axisbelow(True)

fig.suptitle('Identity matters only when the task needs identity: held-out TFs, five-fold',
             fontsize=10.4, y=1.03)
for ext in ('png', 'pdf'):
    fig.savefig(OUT / f'Fig3_identity_fivefold.{ext}',
                dpi=600 if ext == 'png' else None, bbox_inches='tight', facecolor='white')
print('saved Fig3_identity_fivefold')
print('fold means holdout: dna %.4f own %.4f shuf %.4f' % (
    np.mean([f['ho_dna'] for f in folds]), np.mean([f['ho_own'] for f in folds]),
    np.mean([f['ho_shuf'] for f in folds])))
print('delta mean %.4f  improved %.1f%%  n=%d' % (delta.mean(), win, len(delta)))
