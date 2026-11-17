#!/usr/bin/env python3
# coding: UTF-8
"""Figure 3 (high standard) - sharing vs identity; capacity / source / interface."""
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


def j(p):
    return json.loads((RES / p).read_text())


pc = j('bench50_protein_controls.json')
e9 = j('sameharness_compare_E9.json')['holdout']
raw = j('sameharness_raw.json')
e9c = j('sameharness_E9c.json')
fair = j('prodeep_v1_FAIR400/eval.json')['results']
free = j('prodeep_v1_E10FREE/eval.json')['results']
ens = j('bench50_table_ens.json')['pmoscan_mean']

fig, gs = new_fig(5.2, left=0.105, right=0.965, bottom=0.095, top=0.93,
                  wspace=0.42, hspace=0.62)

# ---------------- A  decomposition ----------------
ax = fig.add_subplot(gs[0, 0])
vals = [pc['summary']['centroid'], 0.9241, pc['summary']['correct'], ens]
names = ['constant\nprotein', '50 per-TF\nmodels', 'correct\nprotein',
         'correct protein\n(3-seed)']
cols = [CGRAY, CG, CB, CB]
x = np.arange(4)
ax.bar(x, vals, 0.6, color=cols, linewidth=0)
for i, v in enumerate(vals):
    ax.text(i, v + 0.0012, f'{v:.4f}', ha='center', fontsize=FS_VAL,
            fontweight='bold' if i >= 2 else 'normal')
ax.set_ylim(0.88, 0.952)
ax.set_xlim(-0.6, 3.6)
ax.set_xticks(x)
ax.set_xticklabels(names, fontsize=6.2)
ax.set_ylabel('Mean ROC-AUC (50 datasets)')
ax.annotate('', xy=(2, 0.9445), xytext=(0, 0.9445),
            arrowprops=dict(arrowstyle='->', lw=0.9, color=CB,
                            connectionstyle='arc3,rad=-0.25'))
ax.text(1.0, 0.9465, '+0.040 from protein identity', ha='center', fontsize=FS_BOX, color=CB)
tidy(ax)
letter(ax, 'A')

# ---------------- B  capacity ----------------
ax = fig.add_subplot(gs[0, 1])
labels = ['134k', '298k', '856k']
vv = [e9['E9lin'], e9['A2'], e9['E9wide']]
cis = [e9['E9lin_ci'], e9['A2_ci'], e9['E9wide_ci']]
ax.bar(range(3), vv, 0.56, color=CB, linewidth=0)
for i, (v, c) in enumerate(zip(vv, cis)):
    ax.plot([i, i], c, color='#333', lw=0.8)
    for y in c:
        ax.plot([i - .08, i + .08], [y, y], color='#333', lw=0.8)
    ax.text(i, c[1] + 0.004, f'{v:.3f}', ha='center', fontsize=FS_VAL)
ax.set_ylim(0.53, 0.70)
ax.set_xticks(range(3))
ax.set_xticklabels(labels)
ax.set_xlabel('Motif-head parameters')
ax.set_ylabel('ΔAUC, held-out TFs')
tidy(ax)
letter(ax, 'B')

# ---------------- C  source of identity ----------------
ax = fig.add_subplot(gs[1, 0])
prot = [raw['holdout']['A1'], raw['seen']['A1']]
rand = [e9c['holdout']['E9c_randid'], e9c['seen']['E9c_randid']]
x = np.arange(2)
w = 0.34
ax.bar(x - w / 2, prot, w, color=CB, label='protein')
ax.bar(x + w / 2, rand, w, color=CO, label='random code')
for xi, v in zip(x - w / 2, prot):
    ax.text(xi, v + .02, f'{v:.3f}', ha='center', fontsize=FS_VAL)
for xi, v in zip(x + w / 2, rand):
    ax.text(xi, v + .02, f'{v:.3f}', ha='center', fontsize=FS_VAL, color=CO)
ax.axhline(0.5, color='#666', ls=':', lw=0.8)
ax.set_xticks(x)
ax.set_xticklabels(['held-out', 'seen'])
ax.set_xlim(-0.6, 1.6)
ax.set_ylim(0.4, 1.02)
ax.set_ylabel('ΔAUC')
ax.legend(loc='upper left', fontsize=6.0, handlelength=1.0, borderpad=0.2)
tidy(ax)
letter(ax, 'C')

# ---------------- D  interface ----------------
ax = fig.add_subplot(gs[1, 1])
seen_pc = [fair['seen']['protein_contribution'], free['seen']['protein_contribution']]
ho_pc = [fair['holdout']['protein_contribution'], free['holdout']['protein_contribution']]
x = np.arange(2)
w = 0.32
ax.bar(x - w / 2, seen_pc, w, color=CB, label='seen')
ax.bar(x + w / 2, ho_pc, w, color=CO, label='held-out')
for xi, v in zip(x - w / 2, seen_pc):
    ax.text(xi, v + .008, f'{v:+.3f}', ha='center', fontsize=FS_VAL)
for xi, v in zip(x + w / 2, ho_pc):
    ax.text(xi, v + (.008 if v >= 0 else -.022), f'{v:+.3f}', ha='center', fontsize=FS_VAL,
            color=CO if v < 0 else '#222')
ax.axhline(0, color='#333', lw=0.7)
ax.set_xticks(x)
ax.set_xticklabels(['PWM\ninterface', 'free motif\nhead'])
ax.set_xlim(-0.6, 1.6)
ax.set_ylim(-0.075, 0.23)
ax.set_ylabel('Protein contribution (ΔAUC)')
ax.legend(loc='upper right', fontsize=6.0, handlelength=1.0, borderpad=0.2)
tidy(ax)
letter(ax, 'D')

save(fig, OUT, 'Figure3', 'Figure3')
