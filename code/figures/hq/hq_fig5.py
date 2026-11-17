#!/usr/bin/env python3
# coding: UTF-8
"""Figure 5 (high standard) - where protein conditioning stops helping."""
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
OUT = Path(os.environ.get('FIG_OUT', BASE / 'figures'))

sd = json.loads((BASE / 'results/standard_protocol_R1.json').read_text())
rb = json.loads((BASE / 'paper_data/figR1b_data.json').read_text())
e7 = json.loads((BASE / 'paper_data/figE7_data.json').read_text())

fig, gs = new_fig(5.4, left=0.095, right=0.965, bottom=0.11, top=0.93,
                  wspace=0.38, hspace=0.62)

# ---------------- A  standard protocol ----------------
ax = fig.add_subplot(gs[0, 0])
models = ['DNA-only', 'ProMoScan\n1 seed', 'ProMoScan\n3-seed']
seen = [sd['seen']['dna_auc'], sd['seen']['A2_auc'], sd['seen']['ens_auc']]
hold = [sd['holdout']['dna_auc'], sd['holdout']['A2_auc'], sd['holdout']['ens_auc']]
x = np.arange(len(models))
ax.plot(x - 0.07, seen, 'o', ms=5.2, color=CB, label='seen (n = 200)', zorder=3)
ax.plot(x + 0.07, hold, 's', ms=4.8, color=CO, label='held-out (n = 91)', zorder=3)
for xi, v in zip(x - 0.07, seen):
    ax.text(xi, v + 0.006, f'{v:.3f}', ha='center', fontsize=FS_VAL, color=CB)
for xi, v in zip(x + 0.07, hold):
    ax.text(xi, v + 0.006, f'{v:.3f}', ha='center', fontsize=FS_VAL, color=CO)
ax.plot([0, 2], [seen[0], seen[2]], color=CB, lw=0.6, ls=':', zorder=1)
ax.plot([0, 2], [hold[0], hold[2]], color=CO, lw=0.6, ls=':', zorder=1)
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=6.2)
ax.set_xlim(-0.55, 2.55)
ax.set_ylim(0.80, 0.99)
ax.set_ylabel('ROC-AUC')
ax.legend(loc='lower center', fontsize=5.9, handlelength=1.2, borderpad=0.2, ncol=2,
          columnspacing=1.2)
ax.text(0.02, 0.97, 'gain over DNA-only: +0.0006 (p = 0.13)', transform=ax.transAxes,
        ha='left', va='top', fontsize=5.9, color='#444')
tidy(ax)
letter(ax, 'A')

# ---------------- B  negative-set ladder ----------------
ax = fig.add_subplot(gs[0, 1])
levels = list(rb['series'])
idx = [0, 3, 4]
names = ['DNA-only', 'ProMoScan\n1 seed', 'ProMoScan\n3-seed']
cols = [CGRAY, CB, CG]
xs = np.arange(len(levels))
w = 0.26
for k, (i, nm, c) in enumerate(zip(idx, names, cols)):
    vals = [rb['series'][lv][i] for lv in levels]
    ax.bar(xs + (k - 1) * w, vals, w, color=c, label=nm.replace('\n', ' '))
    for xi, v in zip(xs + (k - 1) * w, vals):
        ax.text(xi, v + 0.012, f'{v:.2f}', ha='center', fontsize=5.9, rotation=90,
                va='bottom')
ax.axhline(0.5, color='#666', ls=':', lw=0.8)
ax.text(-0.45, 0.515, 'chance', fontsize=5.9, color='#666')
ax.set_xticks(xs)
ax.set_xticklabels(['hg38\nrandom', 'dinucleotide\nshuffled', 'mononucleotide\nshuffled'],
                   fontsize=6.2)
ax.set_xlim(-0.6, 2.6)
ax.set_ylim(0.35, 1.06)
ax.set_ylabel('ROC-AUC (seen TFs)')
ax.legend(loc='upper right', fontsize=5.9, handlelength=1.0, borderpad=0.2)
tidy(ax)
letter(ax, 'B')

# ---------------- C  data efficiency ----------------
ax = fig.add_subplot(gs[1, :])
xs = [float(v) for v in e7['labels']]
for key, c, mk, lab in [('Seen TFs', CB, 'o', 'seen TFs'), ('Unseen TFs', CO, 's', 'held-out TFs')]:
    ax.plot(xs, e7['series'][key], marker=mk, color=c, lw=1.1, ms=3.2, label=lab)
ax.axhline(e7['series']['DNA-only (seen)'][0], color=CGRAY, ls='--', lw=0.8,
           label='DNA-only (seen)')
ax.axhline(e7['series']['DNA-only (unseen)'][0], color=CGRAY, ls=':', lw=0.8,
           label='DNA-only (held-out)')
ax.set_xscale('log')
ax.set_xticks(xs)
ax.set_xticklabels([str(int(v)) for v in xs], fontsize=6.2)
ax.set_xlabel('Training examples per class (log scale)')
ax.set_ylabel('ROC-AUC (standard protocol)')
ax.set_ylim(0.80, 0.99)
ax.set_xlim(xs[0] * 0.8, xs[-1] * 1.25)
ax.legend(loc='lower right', fontsize=5.9, handlelength=1.4, borderpad=0.2, ncol=2,
          columnspacing=1.0)
ax.annotate('≈50 per class\nplateau', xy=(50, 0.869), xytext=(8, 0.815), fontsize=5.9,
            color='#333', arrowprops=dict(arrowstyle='->', lw=0.7, color='#666'))
tidy(ax, 'both')
letter(ax, 'C', dx=-0.055, dy=1.03)

save(fig, OUT, 'Figure5', 'Figure5')
