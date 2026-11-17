#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Figure 5 · 边界：蛋白条件化在什么地方**不**成立

输入
  results/standard_protocol_R1.json   标准协议（peak vs hg38 随机背景）
  manuscript/figR1b_data.json         负样本洗牌阶梯（hg38 / 二核苷酸 / 单核苷酸）
  manuscript/figE7_data.json          每类样本量曲线
输出 paper/figures/Fig5_boundaries.{png,pdf}
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
OUT = BASE / 'figures'
OUT.mkdir(parents=True, exist_ok=True)
CB, CO, CG, CGRAY = '#0072B2', '#D55E00', '#009E73', '#9aa0a6'
plt.rcParams.update({'font.size': 9, 'axes.linewidth': 0.8, 'axes.spines.top': False,
                     'axes.spines.right': False, 'pdf.fonttype': 42, 'svg.fonttype': 'none',
                     'axes.unicode_minus': False, 'figure.dpi': 120})

sd = json.loads((BASE / 'results/standard_protocol_R1.json').read_text())
rb = json.loads((BASE / 'paper_data/figR1b_data.json').read_text())
e7 = json.loads((BASE / 'paper_data/figE7_data.json').read_text())

fig = plt.figure(figsize=(14.8, 4.25))
gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.05, 1.15], wspace=0.3)

# ---------- A 标准协议 ----------
ax = fig.add_subplot(gs[0, 0])
models = ['DNA-only', 'ProMoScan\n(single seed)', 'ProMoScan\n(3-seed ens)']
seen = [sd['seen']['dna_auc'], sd['seen']['A2_auc'], sd['seen']['ens_auc']]
hold = [sd['holdout']['dna_auc'], sd['holdout']['A2_auc'], sd['holdout']['ens_auc']]
x = np.arange(len(models))
w = 0.36
ax.bar(x - w / 2, seen, w, color=CB, label='seen TFs (n=200)')
ax.bar(x + w / 2, hold, w, color=CO, label='held-out TFs (n=91)')
for xi, v in zip(x - w / 2, seen):
    ax.text(xi, v + 0.003, f'{v:.3f}', ha='center', fontsize=6.8)
for xi, v in zip(x + w / 2, hold):
    ax.text(xi, v + 0.003, f'{v:.3f}', ha='center', fontsize=6.8)
ax.annotate('protein gain ≈ 0\n(+0.0006, p=0.13)', xy=(1 + w / 2, hold[1] + 0.004),
            xytext=(1.45, 0.955), fontsize=6.8, color=CO,
            arrowprops=dict(arrowstyle='->', lw=0.9, color=CO))
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=7.2)
ax.set_ylim(0.80, 0.99)
ax.set_ylabel('ROC-AUC')
ax.set_title('A  Standard protocol: gain disappears', fontsize=9.2, loc='left')
ax.legend(fontsize=6.8, frameon=False, loc='lower left')
ax.grid(axis='y', color='#eeeeee', lw=0.7)
ax.set_axisbelow(True)

# ---------- B 负样本阶梯 ----------
ax = fig.add_subplot(gs[0, 1])
levels = list(rb['series'])
labels = ['hg38 random', 'dinucleotide shuffle', 'mononucleotide shuffle']
# 系列顺序：DNA-only, L10, M34, A2, ens  → 取 DNA-only / A2 / ens
idx = [0, 3, 4]
names = ['DNA-only', 'ProMoScan (single)', 'ProMoScan (ens)']
cols = [CGRAY, CB, CG]
xs = np.arange(len(levels))
w = 0.26
for k, (i, nm, c) in enumerate(zip(idx, names, cols)):
    vals = [rb['series'][lv][i] for lv in levels]
    ax.bar(xs + (k - 1) * w, vals, w, color=c, label=nm)
    for xi, v in zip(xs + (k - 1) * w, vals):
        ax.text(xi, v + 0.008, f'{v:.2f}', ha='center', fontsize=6.3, rotation=90, va='bottom')
ax.axhline(0.5, color='#666', ls=':', lw=1.0)
ax.text(-0.42, 0.515, 'chance', fontsize=6.4, color='#666')
ax.set_xticks(xs)
ax.set_xticklabels(['hg38\nrandom', 'dinucleotide\nshuffled', 'mononucleotide\nshuffled'], fontsize=7.2)
ax.set_ylim(0.35, 1.03)
ax.set_ylabel('ROC-AUC (seen TFs)')
ax.set_title('B  What the benchmark really measures', fontsize=9.2, loc='left')
ax.legend(fontsize=6.6, frameon=False, loc='upper right')
ax.grid(axis='y', color='#eeeeee', lw=0.7)
ax.set_axisbelow(True)

# ---------- C 数据效率 ----------
ax = fig.add_subplot(gs[0, 2])
xs = [float(v) for v in e7['labels']]
for key, c, mk, lab in [('Seen TFs', CB, 'o', 'seen TFs'), ('Unseen TFs', CO, 's', 'held-out TFs')]:
    ax.plot(xs, e7['series'][key], marker=mk, color=c, lw=1.4, ms=4.2, label=lab)
ax.axhline(e7['series']['DNA-only (seen)'][0], color=CGRAY, ls='--', lw=1.0,
           label='DNA-only (seen)')
ax.axhline(e7['series']['DNA-only (unseen)'][0], color=CGRAY, ls=':', lw=1.0,
           label='DNA-only (held-out)')
ax.set_xscale('log')
ax.set_xticks(xs)
ax.set_xticklabels([str(int(v)) for v in xs], fontsize=7.0)
ax.set_xlabel('Samples per class (log scale)')
ax.set_ylabel('ROC-AUC (standard protocol)')
ax.set_ylim(0.80, 0.99)
ax.set_title('C  How much data a new TF needs', fontsize=9.2, loc='left')
ax.legend(fontsize=6.4, frameon=False, loc='lower right', ncol=1)
ax.grid(color='#eeeeee', lw=0.7)
ax.set_axisbelow(True)
ax.annotate('≈50/class\nplateau', xy=(50, 0.869), xytext=(7, 0.815), fontsize=6.6,
            color='#333', arrowprops=dict(arrowstyle='->', lw=0.8, color='#666'))

fig.suptitle('Boundaries: where protein conditioning does and does not help',
             fontsize=10.4, y=1.03)
for ext in ('png', 'pdf'):
    fig.savefig(OUT / f'Fig5_boundaries.{ext}', dpi=600 if ext == 'png' else None,
                bbox_inches='tight', facecolor='white')
print('saved Fig5_boundaries')
print('standard', seen, hold)
print('shuffle ens', [rb['series'][lv][4] for lv in levels])
