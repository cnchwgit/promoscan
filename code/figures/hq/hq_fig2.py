#!/usr/bin/env python3
# coding: UTF-8
"""Figure 2 (high standard) - one shared model vs 50 per-factor models.

Drawn at final print width (17.1 cm), 2x2 layout, 7 pt body text.
Panels: A reported vs reproduced (dumbbell); B per-dataset paired scatter;
        C sorted dAUC; D accuracy vs trainable parameters.
"""
import csv
import json
from pathlib import Path
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent))
from style_hq import (new_fig, letter, tidy, save, CB, CO, CG, CGRAY,
                      FS_BASE, FS_VAL, FS_BOX)

BASE = Path(os.environ.get('PROMOSCAN_ROOT',
                          Path(__file__).resolve().parents[3]))
RES = BASE / 'results'
OUT = Path(os.environ.get('FIG_OUT', BASE / 'figures'))


def load(name):
    p = RES / name
    if not p.exists():
        for q in sorted(RES.glob('*.json')):
            if q.name.lower() == name.lower():
                p = q
                break
        else:
            return None
    return json.loads(p.read_text())


ens = load('bench50_table_ens.json')
dsids = [r['dsid'] for r in ens['per_dataset']]
ours = np.array([r['pmoscan'] for r in ens['per_dataset']])

REPRO_TAGS = [('CBRKAN', 'CBR-KAN'), ('DeepBind', 'DeepBind'), ('DanQ', 'DanQ'),
              ('DeepD2V', 'DeepD2V'), ('DeepSEA', 'DeepSEA')]
PUB_NAMES = ['DeepBind', 'DanQ', 'DeepD2V', 'DeepSEA', 'CBR-KAN']

repro, pub = {}, {}
for tag, name in REPRO_TAGS:
    d = load(f'bench50_table_{tag}.json')
    if d:
        m = {r['dsid']: r['auc'] for r in d['per_dataset']}
        if all(x in m for x in dsids):
            repro[name] = np.array([m[x] for x in dsids])
for n in PUB_NAMES:
    pub[n] = np.array([r['published'][n] for r in ens['per_dataset']], dtype=float)

params = {}
with open(BASE / 'tables/Table_param_efficiency.csv', newline='', encoding='utf-8') as fh:
    for row in csv.DictReader(fh):
        try:
            params[row['method']] = float(row['total_params_50ds']) / 1e6
        except (KeyError, ValueError):
            continue
M_TOT = 3.268

fig, gs = new_fig(5.45, left=0.095, right=0.955, bottom=0.095, top=0.93,
                  wspace=0.46, hspace=0.66)

# ---------------- A  reported vs reproduced ----------------
ax = fig.add_subplot(gs[0, 0])
order = sorted(repro, key=lambda k: -repro[k].mean())
x = np.arange(len(order))
for i, n in enumerate(order):
    ax.plot([i, i], [repro[n].mean(), pub[n].mean()], color='#c8ccd0', lw=1.2, zorder=1)
ax.scatter(x, [pub[n].mean() for n in order], s=26, facecolor='white', edgecolor=CGRAY,
           lw=0.9, zorder=3, label='reported')
ax.scatter(x, [repro[n].mean() for n in order], s=26, color=CGRAY, edgecolor='none',
           zorder=3, label='reproduced here')
ax.scatter([len(order)], [ours.mean()], s=42, color=CB, zorder=4, label='ProMoScan')

ax.annotate(f'{pub["CBR-KAN"].mean():.3f}', (0, pub['CBR-KAN'].mean()), xytext=(5, 1),
            textcoords='offset points', fontsize=FS_VAL, va='bottom', ha='left', color='#444')
ax.annotate(f'{repro["CBR-KAN"].mean():.3f}', (0, repro['CBR-KAN'].mean()), xytext=(5, -1),
            textcoords='offset points', fontsize=FS_VAL, va='top', ha='left', color='#444')
ax.annotate(f'{ours.mean():.3f}', (len(order), ours.mean()), xytext=(0, 6),
            textcoords='offset points', fontsize=FS_VAL, va='bottom', ha='center',
            color=CB, fontweight='bold')

ax.set_xticks(np.arange(len(order) + 1))
ax.set_xticklabels(order + ['ProMoScan'], rotation=42, ha='right', fontsize=6.1)
ax.set_ylim(0.858, 0.968)
ax.set_xlim(-0.6, len(order) + 0.7)
ax.set_ylabel('Mean ROC-AUC (50 datasets)')
ax.legend(loc='lower right', fontsize=6.0, handletextpad=0.3, borderpad=0.1,
          labelspacing=0.25, bbox_to_anchor=(1.02, -0.02))
tidy(ax)
letter(ax, 'A')

# ---------------- B  per-dataset paired ----------------
ax = fig.add_subplot(gs[0, 1])
ck = repro.get('CBR-KAN')
ax.scatter(ck, ours, s=13, color=CB, alpha=0.72, linewidths=0, zorder=3)
lo, hi = 0.80, 0.99
ax.plot([lo, hi], [lo, hi], ls='--', lw=0.8, color='#999', zorder=1)
ax.set_xlim(lo, hi)
ax.set_ylim(lo, hi)
ax.set_xticks([0.80, 0.85, 0.90, 0.95])
ax.set_yticks([0.80, 0.85, 0.90, 0.95])
ax.set_xlabel('CBR-KAN, 50 per-TF models')
ax.set_ylabel('ProMoScan, 1 model')
win, lose = int((ours > ck).sum()), int((ours < ck).sum())
p = wilcoxon(ours, ck).pvalue
ax.text(0.04, 0.965, f'Δmean = {ours.mean()-ck.mean():+.4f}\n{win} vs {lose} datasets\n'
                     f'Wilcoxon p = {p:.1e}',
        transform=ax.transAxes, va='top', fontsize=FS_BOX,
        bbox=dict(boxstyle='round,pad=0.28', fc='#f7f7f7', ec='#dddddd', lw=0.6))
tidy(ax, 'both')
letter(ax, 'B')

# ---------------- C  sorted dAUC ----------------
ax = fig.add_subplot(gs[1, 0])
d = ours - ck
idx = np.argsort(d)
colors = [CO if v < 0 else CB for v in d[idx]]
ax.bar(np.arange(len(d)), d[idx], 0.86, color=colors, linewidth=0)
ax.axhline(0, color='#333', lw=0.6)
ax.axhline(d.mean(), color=CG, lw=1.0, ls='--')
ax.text(len(d) * 0.02, d.mean() + 0.0015, f'mean {d.mean():+.4f}', color=CG,
        fontsize=FS_VAL, va='bottom')
ax.set_ylim(-0.055, 0.062)
ax.set_xlim(-1, len(d))
ax.set_xlabel('Datasets, sorted by ΔAUC')
ax.set_ylabel('ΔAUC (ProMoScan − CBR-KAN)')
ax.text(0.975, 0.06, f'win {int((d>0).sum())} / loss {int((d<0).sum())}', transform=ax.transAxes,
        ha='right', va='bottom', fontsize=FS_BOX)
tidy(ax)
letter(ax, 'C')

# ---------------- D  accuracy vs parameters ----------------
ax = fig.add_subplot(gs[1, 1])
OFF = {'CBR-KAN': (8, -3)}
for n in order:
    if n not in params:
        continue
    ax.scatter(params[n], repro[n].mean(), s=26, color=CGRAY, edgecolor='#555', lw=0.5, zorder=3)
    if n in OFF:
        ax.annotate(n, (params[n], repro[n].mean()), textcoords='offset points',
                    xytext=OFF[n], ha='left', va='center', fontsize=6.0, color='#333')
ax.scatter([M_TOT], [ours.mean()], s=72, marker='*', color=CB, zorder=4)
ax.annotate('ProMoScan\n1 model', (M_TOT, ours.mean()), textcoords='offset points',
            xytext=(7, 2), fontsize=6.0, color=CB, fontweight='bold', va='center')
ax.set_xscale('log')
ax.set_xlim(2.2, 900)
ax.set_xticks([3, 10, 30, 100, 300])
ax.set_xticklabels(['3', '10', '30', '100', '300'])
ax.set_ylim(0.856, 0.948)
ax.set_xlabel('Trainable parameters, 50 datasets (M, log)')
ax.set_ylabel('Mean ROC-AUC (reproduced)')
ax.annotate('5 per-TF baselines', (5.0, 0.882), xytext=(6, -10), textcoords='offset points',
            fontsize=6.0, color='#666', ha='left')
ax.text(0.97, 0.06, 'each grey point = 50 models', transform=ax.transAxes, ha='right',
        fontsize=6.0, color='#666')
if 'CBR-KAN' in params:
    ax.annotate('', xy=(M_TOT * 1.4, 0.868), xytext=(params['CBR-KAN'] * 0.7, 0.868),
                arrowprops=dict(arrowstyle='<->', lw=0.7, color=CO, shrinkA=1, shrinkB=1))
    ax.text(np.sqrt(M_TOT * params['CBR-KAN']), 0.8705, '36.2× fewer', ha='center',
            fontsize=6.0, color=CO)
tidy(ax, 'both')
letter(ax, 'D')

save(fig, OUT, 'Figure2', 'Figure2')
