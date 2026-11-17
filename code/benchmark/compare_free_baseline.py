#!/usr/bin/env python3
# coding: UTF-8
"""无约束蛋白融合基线（free head）对照表

输入
  results/bench50_table_BENCH50_free.json        无约束基线，50 数据集 AUC
  results/bench50_protein_controls_free.json     同上：正确/最近/随机/最远/均值蛋白
  results/bench50_table_ens.json                 ProMoScan 主配置（集成）
  results/bench50_table.json                     ProMoScan 主配置（单 seed）
  results/bench50_table_cbrkan.json              50 个逐 TF CBR-KAN
  results/bench50_protein_controls.json          ProMoScan 的蛋白归因
输出
  paper/tables/Table2c_unconstrained_baseline.md
"""
import json
import os
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

BASE = Path(os.environ.get('PROMOSCAN_ROOT',
                          Path(__file__).resolve().parents[2]))
RES = BASE / 'results'
OUT = BASE / 'tables'


def load(n):
    p = RES / n
    return json.loads(p.read_text()) if p.exists() else None


def main():
    free = load('bench50_table_BENCH50_free.json')
    free_pc = load('bench50_protein_controls_free.json')
    if free is None:
        raise SystemExit('缺少无约束基线的评估结果（训练/评估尚未完成）')
    ens = load('bench50_table_ens.json')
    single = load('bench50_table.json')
    ck = load('bench50_table_cbrkan.json')
    ds = [r['dsid'] for r in ens['per_dataset']]
    mf = {r['dsid']: r['auc'] for r in free['per_dataset']}
    a_free = np.array([mf[d] for d in ds])
    a_ck = np.array([r['auc'] for r in ck['per_dataset']])
    a_single = np.array([r['pmoscan'] for r in single['per_dataset']])
    a_ens = np.array([r['pmoscan'] for r in ens['per_dataset']])

    md = ['# Table 2c · 无约束蛋白融合基线（free head）\n',
          '对照目的：回答"约束是否必要"。该基线使用同一 CBR-KAN 骨干与同一 ESM-2 蛋白表征，'
          '但蛋白**不经过 PWM 扫描**，而是直接进入分类头（logit = β + free_head(z) + w·h）。'
          '训练配方与逐 TF 基线一致：30 epoch、batch 128、lr 1e-3、seed 42、85:15 验证集选点；'
          '不使用 motif 弱监督（无 motif 通路可监督），保留 A/B ranking。\n',
          '| 模型 | 粒度 | mean AUC | vs CBR-KAN 逐 TF（胜/负） | Δmean | Wilcoxon p |',
          '|---|---|---|---|---|---|']
    for name, arr in [('**无约束蛋白融合（free head）**', a_free),
                      ('ProMoScan（主配置，单 seed）', a_single),
                      ('ProMoScan（主配置，3-seed 集成）', a_ens)]:
        w = int((arr > a_ck).sum()); l = int((arr < a_ck).sum())
        md.append(f'| {name} | 1 个共享模型 | {arr.mean():.4f} | {w}/{l} | {arr.mean()-a_ck.mean():+.4f} | '
                  f'{wilcoxon(arr, a_ck).pvalue:.2g} |')
    md.append(f'| CBR-KAN 逐 TF ×50（同配方） | 50 个逐 TF 模型 | {a_ck.mean():.4f} | — | — | — |')

    if free_pc:
        s = free_pc['summary']
        md.append('\n## 蛋白替换归因（同模型、只换蛋白向量）\n')
        md.append('| 蛋白输入 | 无约束融合 | ProMoScan（单 seed） |')
        md.append('|---|---|---|')
        pro = load('bench50_protein_controls.json')['summary']
        for k, lab in [('correct', '正确蛋白'), ('centroid', '均值蛋白（无身份）'),
                       ('farthest', '最远 TF'), ('nearest', '最近邻 TF'), ('random', '随机 TF')]:
            md.append(f'| {lab} | {s[k]:.4f} | {pro[k]:.4f} |')
        md.append(f'\n> 无约束基线的换蛋白损失：{s["correct"]-s["random"]:+.4f}（随机蛋白）；'
                  f'ProMoScan：{pro["correct"]-pro["random"]:+.4f}。')
    md.append('\n> 判读：若无约束基线的同分布 AUC 与 ProMoScan 接近、但换蛋白掉点更小或留出 TF 更差，'
              '则说明约束的作用不是涨点，而是让身份可干预、可外推。')
    txt = '\n'.join(md)
    OUT.mkdir(exist_ok=True)
    (OUT / 'Table2c_unconstrained_baseline.md').write_text(txt, encoding='utf-8')
    print(txt)


if __name__ == '__main__':
    main()
