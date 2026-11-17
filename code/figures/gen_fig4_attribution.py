#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Figure 4 · 蛋白输入归因（本文论证"蛋白在做什么"的主证据）

面板
  A 替换阶梯：正确 0.9330 / 均值蛋白 0.8928 / 最远 0.8706 / 最近邻 0.8609 / 随机 0.8559（50/50 全掉点）
  B 替换蛋白与正确蛋白的余弦相似度 vs AUC 损失（ρ=+0.04）→ 离散身份键
  C 同一 TF 跨细胞系（14 个）逐一配对：蛋白向量相同，AUC 仍居前 → 不是数据集指纹

输入 results/bench50_protein_controls.json
输出 paper/figures/Fig4_attribution.{png,pdf}
"""
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

BASE = Path(os.environ.get('PROMOSCAN_ROOT',
                          Path(__file__).resolve().parents[2]))
OUT = BASE / 'figures'
OUT.mkdir(parents=True, exist_ok=True)
CB, CO, CG, CGRAY = '#0072B2', '#D55E00', '#009E73', '#9aa0a6'
plt.rcParams.update({'font.size': 9, 'axes.linewidth': 0.8, 'axes.spines.top': False,
                     'axes.spines.right': False, 'pdf.fonttype': 42, 'svg.fonttype': 'none',
                     'axes.unicode_minus': False, 'figure.dpi': 120})

d = json.loads((BASE / 'results/bench50_protein_controls.json').read_text())
rows = d['per_dataset']
summary = d['summary']

fig = plt.figure(figsize=(15.0, 4.3))
gs = fig.add_gridspec(1, 3, width_ratios=[1.3, 1.0, 1.05], wspace=0.34)

# ---------- A ----------
ax = fig.add_subplot(gs[0, 0])
keys = ['correct', 'centroid', 'farthest', 'nearest', 'random']
labels = ['correct\nprotein', 'protein\ncentroid', 'farthest\nTF', 'nearest\nTF', 'random\nTF']
rng = np.random.default_rng(7)
for i, k in enumerate(keys):
    v = np.array([r[k] for r in rows])
    ax.bar(i, v.mean(), .62, color=CB if k == 'correct' else CGRAY,
           alpha=.9 if k == 'correct' else .55, linewidth=0)
    ax.scatter(np.full(len(v), i) + rng.normal(0, .06, len(v)), v, s=4.2, color='#333',
               alpha=.33, linewidths=0, zorder=3)
    ax.text(i, v.mean() + .004, f'{v.mean():.3f}', ha='center', fontsize=7.0,
            fontweight='bold' if k == 'correct' else 'normal')
ax.set_xticks(range(5))
ax.set_xticklabels(labels, fontsize=7.0)
ax.set_ylim(0.78, 0.97)
ax.set_ylabel('ROC-AUC (50 datasets)')
ax.set_title('A  Swap the protein, keep everything else', fontsize=9.3, loc='left')
ax.text(0.015, 0.035, 'any wrong identity: 50/50 datasets worse, p = 1.8e-15',
        transform=ax.transAxes, fontsize=6.5, color='#444')
ax.grid(axis='y', color='#eeeeee', lw=.7)
ax.set_axisbelow(True)

# ---------- B ----------
ax = fig.add_subplot(gs[0, 1])
lv = np.array([k for r in rows for k in ('nearest', 'random', 'farthest')])
cos = np.array([r[k + '_cos'] for r in rows for k in ('nearest', 'random', 'farthest')])
drop = np.array([r['correct'] - r[k] for r in rows for k in ('nearest', 'random', 'farthest')])
colors = {'nearest': CG, 'random': CGRAY, 'farthest': CO}
for k in ('nearest', 'random', 'farthest'):
    m = lv == k
    ax.scatter(cos[m], drop[m], s=9, color=colors[k], alpha=.6, linewidths=0, label=k)
b, a0 = np.polyfit(cos, drop, 1)
xs = np.linspace(cos.min(), cos.max(), 50)
ax.plot(xs, a0 + b * xs, color='#333', lw=1.0, ls='--')
rho, p = spearmanr(cos, drop)
ax.set_xlabel('cosine similarity to correct protein')
ax.set_ylabel('AUC loss')
ax.set_title('B  Loss is flat in similarity', fontsize=9.3, loc='left')
ax.text(.03, .96, f'ρ = {rho:+.2f}, p = {p:.2f}\nn = {len(cos)} swaps\n→ a discrete key, not a gradient',
        transform=ax.transAxes, va='top', fontsize=6.8,
        bbox=dict(boxstyle='round,pad=.3', fc='#f7f7f7', ec='#dddddd', lw=.7))
ax.legend(fontsize=6.4, frameon=False, loc='lower right')
ax.grid(color='#eeeeee', lw=.7)
ax.set_axisbelow(True)

# ---------- C 跨细胞系 ----------
ax = fig.add_subplot(gs[0, 2])
import collections
g = collections.defaultdict(list)
for r in rows:
    g[r['symbol']].append(r)
multi = {k: v for k, v in g.items() if len(v) > 1}
xs = np.arange(len(multi))
for i, (sym, lst) in enumerate(sorted(multi.items(), key=lambda kv: -np.mean([q['correct'] for q in kv[1]]))):
    c = [q['correct'] for q in lst]
    w = [q['random'] for q in lst]
    for v in c:
        ax.plot([i - .18, i + .18], [v, v], color=CB, lw=1.4)
    for v in w:
        ax.plot([i - .18, i + .18], [v, v], color=CGRAY, lw=1.0, alpha=.8)
    ax.scatter([i] * len(c), c, s=8, color=CB, zorder=3)
    ax.scatter([i] * len(w), w, s=8, color=CGRAY, zorder=3)
ax.set_xticks(xs)
ax.set_xticklabels([k for k, _ in sorted(multi.items(), key=lambda kv: -np.mean([q['correct'] for q in kv[1]]))],
                   fontsize=5.8, rotation=90)
ax.set_ylim(0.75, 1.0)
ax.set_ylabel('ROC-AUC')
ax.set_title('C  Same TF, different cell lines', fontsize=9.3, loc='left')
ax.text(0.03, 0.06, 'blue = correct protein   grey = random TF', transform=ax.transAxes, fontsize=6.3)
ax.grid(axis='y', color='#eeeeee', lw=.7)
ax.set_axisbelow(True)

fig.suptitle('What the protein channel encodes: a discrete identity key that is not a dataset fingerprint',
             fontsize=10.4, y=1.03)
for ext in ('png', 'pdf'):
    fig.savefig(OUT / f'Fig4_attribution.{ext}', dpi=600 if ext == 'png' else None,
                bbox_inches='tight', facecolor='white')
print('saved Fig4_attribution', {k: round(v, 4) for k, v in summary.items()}, 'rho', round(rho, 3),
      'multi-cell TFs', len(multi))
