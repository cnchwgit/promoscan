#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Figure 3 · 机制：共享本身 vs 蛋白身份；容量 / 来源 / 接口

面板
  A 分解：共享+恒定蛋白（无身份）0.8928 / 50 个逐 TF 模型 0.9241 / 共享+正确蛋白 0.9330 / 集成 0.9376
  B 容量：134k / 298k / 856k（6.4×）无差别
  C 来源：蛋白 vs per-TF 随机码
  D 接口：标准 PWM 头 vs free motif head

输入 results/bench50_protein_controls.json、sameharness_compare_E9.json、
     sameharness_raw.json、sameharness_E9c.json、prodeep_v1_{FAIR400,E10FREE}/eval.json
输出 paper/figures/Fig3_mechanism.{png,pdf}
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

CB, CO, CG, CGRAY, CPUR = '#0072B2', '#D55E00', '#009E73', '#9aa0a6', '#7B3294'
plt.rcParams.update({'font.size': 9, 'axes.linewidth': 0.8, 'axes.spines.top': False,
                     'axes.spines.right': False, 'pdf.fonttype': 42, 'svg.fonttype': 'none',
                     'axes.unicode_minus': False, 'figure.dpi': 120})


def j(p):
    return json.loads((RES / p).read_text())


pc = j('bench50_protein_controls.json')
e9 = j('sameharness_compare_E9.json')['holdout']
raw = j('sameharness_raw.json')
e9c = j('sameharness_E9c.json')
fair = j('prodeep_v1_FAIR400/eval.json')['results']
free = j('prodeep_v1_E10FREE/eval.json')['results']
ens = j('bench50_table_ens.json')['pmoscan_mean']

fig = plt.figure(figsize=(15.0, 4.3))
gs = fig.add_gridspec(1, 4, width_ratios=[1.35, 0.95, 0.95, 1.0], wspace=0.36)

# ---------- A 分解 ----------
ax = fig.add_subplot(gs[0, 0])
vals = [pc['summary']['centroid'], 0.9241, pc['summary']['correct'], ens]
names = ['shared model\nconstant protein\n(no identity)', '50 per-TF\nmodels',
         'shared model\ncorrect protein\n(single seed)', 'shared model\ncorrect protein\n(3-seed ensemble)']
cols = [CGRAY, CG, CB, CB]
alphas = [0.55, 0.75, 0.9, 0.9]
x = np.arange(4)
ax.bar(x, vals, 0.62, color=cols, alpha=0.9, linewidth=0)
for i, v in enumerate(vals):
    ax.text(i, v + 0.003, f'{v:.4f}', ha='center', fontsize=7.2,
            fontweight='bold' if i >= 2 else 'normal')
ax.set_ylim(0.87, 0.95)
ax.set_xticks(x)
ax.set_xticklabels(names, fontsize=6.6)
ax.set_ylabel('Mean ROC-AUC (50 datasets)')
ax.set_title('A  Sharing alone is not enough; identity is what wins', fontsize=9.2, loc='left')
ax.annotate('', xy=(2, vals[2] + 0.006), xytext=(0, vals[0] + 0.006),
            arrowprops=dict(arrowstyle='->', lw=1.1, color=CB))
ax.text(1.0, vals[2] + 0.011, '+0.040 from protein identity', ha='center', fontsize=6.8, color=CB)
ax.grid(axis='y', color='#eeeeee', lw=0.7)
ax.set_axisbelow(True)

# ---------- B 容量 ----------
ax = fig.add_subplot(gs[0, 1])
labels = ['134k', '298k', '856k']
vv = [e9['E9lin'], e9['A2'], e9['E9wide']]
cis = [e9['E9lin_ci'], e9['A2_ci'], e9['E9wide_ci']]
ax.bar(range(3), vv, 0.58, color=CB, alpha=0.85, linewidth=0)
for i, (v, c) in enumerate(zip(vv, cis)):
    ax.plot([i, i], c, color='#333', lw=1.1)
    for y in c:
        ax.plot([i - .09, i + .09], [y, y], color='#333', lw=1.1)
    ax.text(i, v + .006, f'{v:.3f}', ha='center', fontsize=7.0)
ax.set_ylim(0.53, 0.70)
ax.set_xticks(range(3))
ax.set_xticklabels(labels, fontsize=7.2)
ax.set_ylabel('ΔAUC, held-out TFs')
ax.set_title('B  6.4× capacity: no change', fontsize=9.2, loc='left')
ax.grid(axis='y', color='#eeeeee', lw=0.7)
ax.set_axisbelow(True)

# ---------- C 来源 ----------
ax = fig.add_subplot(gs[0, 2])
prot = [raw['holdout']['A1'], raw['seen']['A1']]
rand = [e9c['holdout']['E9c_randid'], e9c['seen']['E9c_randid']]
x = np.arange(2); w = 0.34
ax.bar(x - w / 2, prot, w, color=CB, label='protein')
ax.bar(x + w / 2, rand, w, color=CO, label='random code')
for xi, v in zip(x - w / 2, prot):
    ax.text(xi, v + .02, f'{v:.3f}', ha='center', fontsize=6.9)
for xi, v in zip(x + w / 2, rand):
    ax.text(xi, v + .02, f'{v:.3f}', ha='center', fontsize=6.9, color=CO)
ax.axhline(0.5, color='#666', ls=':', lw=1.0)
ax.set_xticks(x)
ax.set_xticklabels(['held-out', 'seen'], fontsize=7.2)
ax.set_ylim(0.4, 1.0)
ax.set_ylabel('ΔAUC')
ax.set_title('C  Identity must come from protein', fontsize=9.2, loc='left')
ax.legend(fontsize=6.4, frameon=False, loc='upper left')
ax.grid(axis='y', color='#eeeeee', lw=0.7)
ax.set_axisbelow(True)

# ---------- D 接口 ----------
ax = fig.add_subplot(gs[0, 3])
seen_pc = [fair['seen']['protein_contribution'], free['seen']['protein_contribution']]
ho_pc = [fair['holdout']['protein_contribution'], free['holdout']['protein_contribution']]
x = np.arange(2); w = 0.32
ax.bar(x - w / 2, seen_pc, w, color=CB, label='seen')
ax.bar(x + w / 2, ho_pc, w, color=CO, label='held-out')
for xi, v in zip(x - w / 2, seen_pc):
    ax.text(xi, v + .008, f'{v:+.3f}', ha='center', fontsize=6.8)
for xi, v in zip(x + w / 2, ho_pc):
    ax.text(xi, v + (.008 if v >= 0 else -.020), f'{v:+.3f}', ha='center', fontsize=6.8,
            color=CO if v < 0 else '#222')
ax.axhline(0, color='#333', lw=0.9)
ax.set_xticks(x)
ax.set_xticklabels(['PWM\ninterface', 'free motif\nhead'], fontsize=7.0)
ax.set_ylim(-0.07, 0.22)
ax.set_ylabel('Protein contribution')
ax.set_title('D  Must pass through the DNA operator', fontsize=9.2, loc='left')
ax.legend(fontsize=6.4, frameon=False, loc='upper right')
ax.grid(axis='y', color='#eeeeee', lw=0.7)
ax.set_axisbelow(True)

fig.suptitle('Why it works: not sharing alone, not capacity, not an arbitrary code',
             fontsize=10.4, y=1.03)
for ext in ('png', 'pdf'):
    fig.savefig(OUT / f'Fig3_mechanism.{ext}', dpi=600 if ext == 'png' else None,
                bbox_inches='tight', facecolor='white')
print('saved Fig3_mechanism', {n: round(v, 4) for n, v in zip(['centroid', 'perTF', 'correct', 'ens'], vals)})
