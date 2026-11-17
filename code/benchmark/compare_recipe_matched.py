#!/usr/bin/env python3
# coding: UTF-8
"""配方对齐对照：ProMoScan 用 PLOS ONE（CBR-KAN）训练超参重跑 vs 逐 TF 基线

输入
  results/bench50_table_BENCH50_plos.json   本次配方对齐跑（30 ep / batch 128 / lr 1e-3 / 85:15 val 选点）
  results/bench50_table_ens.json            主配置（3-seed 集成）
  results/bench50_table.json                主配置（单 seed）
  results/bench50_table_cbrkan.json         50 个逐 TF CBR-KAN（本机复现，同为 PLOS 配方）
输出
  paper/tables/Table2b_recipe_matched.md
"""
import json
import os
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

BASE = Path(os.environ.get('PROMOSCAN_ROOT',
                          Path(__file__).resolve().parents[2]))
RES = BASE / 'results'


def load(n):
    p = RES / n
    return json.loads(p.read_text()) if p.exists() else None


def auc_map(d):
    return {r['dsid']: r['auc'] for r in d['per_dataset']}


def main():
    ours = load('bench50_table_BENCH50_plos.json')
    if ours is None:
        raise SystemExit('缺少 bench50_table_BENCH50_plos.json（训练/评估尚未完成）')
    ens = load('bench50_table_ens.json')
    single = load('bench50_table.json')
    ck = load('bench50_table_cbrkan.json')
    ds = [r['dsid'] for r in ens['per_dataset']]
    m_new, m_ck = auc_map(ours), auc_map(ck)
    a_new = np.array([m_new[d] for d in ds])
    a_ck = np.array([m_ck[d] for d in ds])
    a_ens = np.array([r['pmoscan'] for r in ens['per_dataset']])
    a_single = np.array([r['pmoscan'] for r in single['per_dataset']])

    def line(name, arr, ref):
        w = int((arr > ref).sum()); l = int((arr < ref).sum())
        return f'| {name} | {arr.mean():.4f} | {np.median(arr):.4f} | {w}/{l} | {arr.mean()-ref.mean():+.4f} | {wilcoxon(arr, ref).pvalue:.2g} |'

    md = ['# Table 2b · 配方对齐：ProMoScan 用 PLOS ONE 训练超参重跑\n',
          '为排除"ProMoScan 靠调参取胜"，本表把 ProMoScan 的训练超参与 50 个逐 TF 基线**完全对齐**：'
          '30 epoch、batch 128、lr 1e-3、seed 42、85:15 分层验证集并按验证集逐数据集 AUC 选点。'
          '架构超参（M=16, W=16, ESM-2 冻结）保持不变，因为它们是方法本身。\n',
          '| 模型 | mean AUC | median | vs 基线（胜/负） | Δmean | Wilcoxon p |',
          '|---|---|---|---|---|---|']
    md.append(line('**ProMoScan（PLOS 配方）**', a_new, a_ck))
    md.append(f'| CBR-KAN 逐 TF ×50（同配方，本机复现） | {a_ck.mean():.4f} | {np.median(a_ck):.4f} | — | — | — |')
    md.append(line('ProMoScan（主配置，单 seed）', a_single, a_ck))
    md.append(line('ProMoScan（主配置，3-seed 集成）', a_ens, a_ck))
    md.append('')
    md.append(f'> 训练集：951,152 条（1,118,960 的 85%）；验证集 8,000 条分层抽样。'
              f'本次运行的最佳 epoch 与验证 AUC 见 `results/prodeep_v1_BENCH50_plos.log`。')
    txt = '\n'.join(md)
    (BASE / 'paper/tables/Table2b_recipe_matched.md').write_text(txt, encoding='utf-8')
    print(txt)


if __name__ == '__main__':
    main()
