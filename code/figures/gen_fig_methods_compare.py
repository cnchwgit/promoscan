#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""方法对比图（同协议口径，2026-11-15）

输入（只读）
  results/bench50_table_ens.json        ProMoScan 3-seed 集成逐数据集 AUC
  results/bench50_table.json            ProMoScan 单 seed
  results/bench50_table_cbrkan.json     CBR-KAN 本机复现（50 个逐 TF 模型）
  results/bench50_table_DeepBind|DanQ|DeepD2V|DeepSEA.json
  tables/Table_param_efficiency.csv     参数与部署对照
输出
  paper/figures/Fig2_methods_comparison_20261115.{png,pdf}

四个面板：
  A 方法均值：已发表（空心）vs 本机同协议（实心）
  B 逐数据集配对散点：ProMoScan(1 模型) vs CBR-KAN(50 模型, 本机)
  C 逐数据集 ΔAUC（排序条形 + 胜负计数）
  D 参数–精度散点（x 轴 log10，50 集合计可训练参数）
"""
import csv
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon

BASE = Path(os.environ.get('PROMOSCAN_ROOT',
                          Path(__file__).resolve().parents[2]))
RES = BASE / 'results'
OUT = BASE / 'figures'
OUT.mkdir(parents=True, exist_ok=True)

CB, CO, CG, CGRAY = '#0072B2', '#D55E00', '#009E73', '#9aa0a6'
plt.rcParams.update({'font.size': 9, 'axes.linewidth': 0.8, 'axes.spines.top': False,
                     'axes.spines.right': False, 'pdf.fonttype': 42, 'svg.fonttype': 'none',
                     'axes.unicode_minus': False, 'figure.dpi': 120})


def load(name):
    p = RES / name
    if not p.exists():
        low = name.lower()
        for q in sorted(RES.glob('*.json')):
            if q.name.lower() == low:
                p = q
                break
        else:
            return None
    return json.loads(p.read_text())


ens = load('bench50_table_ens.json')
single = load('bench50_table.json')
dsids = [r['dsid'] for r in ens['per_dataset']]
ours = np.array([r['pmoscan'] for r in ens['per_dataset']])
shuf = np.array([r.get('pmoscan_shuffled', np.nan) for r in ens['per_dataset']])

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

# ---------- 参数（50 集合计，可训练） ----------
params = {}
with open(BASE / 'tables/Table_param_efficiency.csv', newline='', encoding='utf-8') as fh:
    for row in csv.DictReader(fh):
        try:
            params[row['method']] = float(row['total_params_50ds']) / 1e6  # M
        except (KeyError, ValueError):
            continue
M_TOT = 3.268

fig = plt.figure(figsize=(15.2, 4.3))
gs = fig.add_gridspec(1, 4, width_ratios=[1.25, 1.0, 1.15, 1.05], wspace=0.34)

# ---------- A: 均值对比（已发表 vs 本机同协议） ----------
ax = fig.add_subplot(gs[0, 0])
order = sorted(repro, key=lambda k: -repro[k].mean()) if repro else PUB_NAMES
x = np.arange(len(order) + 1)
w = 0.36
pub_means = [pub[n].mean() for n in order]
rep_means = [repro[n].mean() for n in order]
ax.bar(x[:-1] - w / 2, pub_means, w, facecolor='white', edgecolor=CGRAY, lw=1.1,
       hatch='///')
ax.bar(x[:-1] + w / 2, rep_means, w, color=CGRAY)
ax.bar(x[-1], ours.mean(), 0.62, color=CB)
for xi, v in zip(x[:-1] - w / 2, pub_means):
    ax.text(xi, v + 0.002, f'{v:.3f}', ha='center', fontsize=6.4, rotation=90, va='bottom')
for xi, v in zip(x[:-1] + w / 2, rep_means):
    ax.text(xi, v + 0.002, f'{v:.3f}', ha='center', fontsize=6.4, rotation=90, va='bottom')
ax.text(x[-1], ours.mean() + 0.002, f'{ours.mean():.3f}', ha='center', fontsize=7.2,
        fontweight='bold', va='bottom', color=CB)
ax.set_xticks(x)
ax.set_xticklabels(order + ['ProMoScan'], fontsize=7.2, rotation=20, ha='right')
ax.set_ylim(0.84, 0.98)
ax.set_ylabel('Mean ROC-AUC (50 datasets)')
ax.set_title('A  Reported vs reproduced under one protocol', fontsize=9.2, loc='left')
ax.grid(axis='y', color='#eeeeee', lw=0.7)
ax.set_axisbelow(True)
ax.text(0.0, -0.36, 'hatched = reported in original paper   ·   solid = reproduced here (same protocol)',
        transform=ax.transAxes, fontsize=6.4, color='#444', va='top')

# ---------- B: 逐数据集配对 ----------
ax = fig.add_subplot(gs[0, 1])
ck = repro.get('CBR-KAN')
if ck is not None:
    ax.scatter(ck, ours, s=17, color=CB, alpha=0.78, linewidths=0)
    lo, hi = 0.78, 1.0
    ax.plot([lo, hi], [lo, hi], ls='--', lw=1.0, color='#888')
    for i, d in enumerate(dsids):
        if abs(ours[i] - ck[i]) > 0.045:
            ax.annotate('', xy=(ck[i], ours[i]), xytext=(ck[i], ours[i]),
                        arrowprops=None)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel('CBR-KAN, 50 per-TF models (reproduced)')
    ax.set_ylabel('ProMoScan, 1 model')
    win = int((ours > ck).sum())
    lose = int((ours < ck).sum())
    p = wilcoxon(ours, ck).pvalue
    ax.set_title('B  Per dataset, paired', fontsize=9.2, loc='left')
    ax.text(0.04, 0.96, f'Δmean = {ours.mean()-ck.mean():+.4f}\n{win}/{lose} datasets\nWilcoxon p = {p:.1e}',
            transform=ax.transAxes, va='top', fontsize=6.9,
            bbox=dict(boxstyle='round,pad=0.32', fc='#f7f7f7', ec='#dddddd', lw=0.7))
    ax.grid(color='#eeeeee', lw=0.7)
    ax.set_axisbelow(True)

# ---------- C: ΔAUC 排序条形 ----------
ax = fig.add_subplot(gs[0, 2])
if ck is not None:
    d = ours - ck
    idx = np.argsort(d)
    colors = [CO if v < 0 else CB for v in d[idx]]
    ax.bar(np.arange(len(d)), d[idx], 0.82, color=colors, linewidth=0)
    ax.axhline(0, color='#333', lw=0.8)
    ax.axhline(d.mean(), color=CG, lw=1.2, ls='--')
    ax.text(1, d.mean() + 0.002, f'mean {d.mean():+.4f}', color=CG, fontsize=6.8, va='bottom')
    ax.set_ylim(-0.055, 0.062)
    ax.set_xlabel('Datasets (sorted by ΔAUC)')
    ax.set_ylabel('ΔAUC  (ProMoScan − CBR-KAN)')
    ax.set_title('C  Win/loss per dataset', fontsize=9.2, loc='left')
    ax.text(0.97, 0.94, f'win {int((d>0).sum())} / loss {int((d<0).sum())}',
            transform=ax.transAxes, ha='right', va='top', fontsize=7.0)
    ax.grid(axis='y', color='#eeeeee', lw=0.7)
    ax.set_axisbelow(True)

# ---------- D: 参数–精度 ----------
ax = fig.add_subplot(gs[0, 3])
OFF = {'DeepBind': (-6, -19), 'DanQ': (-10, 10), 'DeepD2V': (10, -16),
       'DeepSEA': (-2, 10), 'CBR-KAN': (0, -19)}
for n in order:
    if n not in params:
        continue
    ax.scatter(params[n], repro[n].mean(), s=46, color=CGRAY, edgecolor='#555', lw=0.6, zorder=3)
    ax.annotate(n, (params[n], repro[n].mean()), textcoords='offset points',
                xytext=OFF.get(n, (-4, -14)), ha='center', fontsize=6.2, color='#333')
ax.text(0.03, 0.03, 'each grey point = 50 per-TF models', transform=ax.transAxes,
        fontsize=6.2, color='#555')
ax.scatter([M_TOT], [ours.mean()], s=110, marker='*', color=CB, zorder=4)
ax.annotate('ProMoScan\n1 model', (M_TOT, ours.mean()), textcoords='offset points',
            xytext=(10, 4), fontsize=6.6, color=CB, fontweight='bold')
ax.set_xscale('log')
ax.set_xlim(2.2, 500)
ax.set_ylim(0.855, 0.95)
ax.set_xlabel('Trainable parameters for 50 datasets (M, log)')
ax.set_ylabel('Mean ROC-AUC (reproduced)')
ax.set_title('D  Accuracy vs parameters', fontsize=9.2, loc='left')
ax.grid(color='#eeeeee', lw=0.7)
ax.set_axisbelow(True)
if 'CBR-KAN' in params:
    ax.annotate('', xy=(M_TOT * 1.35, ours.mean()), xytext=(params['CBR-KAN'] * 0.75, ours.mean()),
                arrowprops=dict(arrowstyle='<->', lw=0.9, color=CO))
    ax.text(np.sqrt(M_TOT * params['CBR-KAN']), ours.mean() + 0.004, '36× fewer',
            ha='center', fontsize=6.6, color=CO)

fig.suptitle('One shared model vs 50 per-TF models · 50 public ChIP-seq datasets · identical training protocol',
             fontsize=10.4, y=1.02)
for ext in ('png', 'pdf'):
    fig.savefig(OUT / f'Fig2_methods_comparison_20261115.{ext}', dpi=600 if ext == 'png' else None,
                bbox_inches='tight', facecolor='white')
print('saved', OUT / 'Fig2_methods_comparison_20261115.png')
if ck is not None:
    print(f'ProMoScan {ours.mean():.4f} vs CBR-KAN repro {ck.mean():.4f} Δ{ours.mean()-ck.mean():+.4f}')
