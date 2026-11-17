#!/usr/bin/env python3
# coding: UTF-8
"""50 数据集对照表：ProMoScan vs CBR-KAN（及其他已发表方法）

输入（只读）
  results/bench50_table.json       ProMoScan 单 seed 逐数据集 AUC
  results/bench50_table_ens.json   ProMoScan 3-seed 集成
  results/bench50_table_cbrkan.json  （若存在）本机复现的 CBR-KAN 逐数据集 AUC
输出
  tables/Table_bench50_compare.md
  tables/bench50_per_dataset_compare.csv
"""
import csv
import json
import os
from pathlib import Path

import numpy as np

BASE = Path(os.environ.get('PROMOSCAN_ROOT',
                          Path(__file__).resolve().parents[2]))
RES = BASE / 'results'
OUT = BASE / 'tables'
BASE_METHODS = ['DeepBind', 'DanQ', 'DeepD2V', 'DeepSEA', 'CBR-KAN']


def load(name):
    p = RES / name
    if p.exists():
        return json.load(open(p))
    # 文件名大小写不一致时容错（如 bench50_table_CBRKAN vs _cbrkan）
    low = name.lower()
    for q in sorted(RES.glob('*.json')):
        if q.name.lower() == low:
            return json.load(open(q))
    return None


def wilcoxon(a, b):
    try:
        from scipy.stats import wilcoxon as w
        return float(w(a, b).pvalue)
    except Exception:
        return float('nan')


def main():
    single = load('bench50_table.json')
    ens = load('bench50_table_ens.json')
    if not ens:
        raise SystemExit('缺少 bench50_table_ens.json')
    dsids = [r['dsid'] for r in ens['per_dataset']]
    REPRO = {}
    for tag, name in [('CBRKAN', 'CBR-KAN'), ('DeepBind', 'DeepBind'), ('DanQ', 'DanQ'),
                      ('DeepD2V', 'DeepD2V'), ('DeepSEA', 'DeepSEA')]:
        d = load(f'bench50_table_{tag}.json')
        if d:
            m = {r['dsid']: r['auc'] for r in d['per_dataset']}
            if all(x in m for x in dsids):
                REPRO[name] = np.array([m[x] for x in dsids])

    rows = {r['dsid']: r for r in ens['per_dataset']}
    ours_ens = np.array([rows[d]['pmoscan'] for d in dsids])
    ours_shuf = np.array([rows[d].get('pmoscan_shuffled', np.nan) for d in dsids])
    ours_single = np.array([single['per_dataset'][i]['pmoscan'] for i in range(len(dsids))]) if single else None

    pub = {m: np.array([rows[d]['published'][m] for d in dsids], dtype=float) for m in BASE_METHODS}
    repro_auc = REPRO.get('CBR-KAN')

    md = ['# Table · 50 个公共 ChIP-seq 数据集：ProMoScan vs CBR-KAN 等\n',
          '数据：CBR-KAN 论文的 50 数据集基准；test = 1,500 pos + 1,500 neg / 数据集；指标 ROC-AUC。\n',
          '**口径提醒**：已发表方法的数字取自其原文（经 CBR-KAN 汇总），与我们自算是跨论文比较；'
          '`CBR-KAN(本机复现)` 行只有在跑完后才与 ProMoScan 严格同协议。\n',
          '| 方法 | 粒度 | mean AUC | median | vs ProMoScan(集成) 胜/负 | Δmean | Wilcoxon p |',
          '|---|---|---|---|---|---|---|']
    md.append(f'| **ProMoScan（3-seed 集成）** | 1 个模型 | **{ours_ens.mean():.4f}** | '
              f'{np.median(ours_ens):.4f} | — | — | — |')
    if ours_single is not None:
        md.append(f'| ProMoScan（单 seed） | 1 个模型 | {ours_single.mean():.4f} | '
                  f'{np.median(ours_single):.4f} | {int((ours_single > ours_ens).sum())}/'
                  f'{int((ours_single < ours_ens).sum())} | {ours_single.mean()-ours_ens.mean():+.4f} | — |')
    order = sorted(BASE_METHODS, key=lambda m: -pub[m].mean())
    for m in order:
        win = int((ours_ens > pub[m]).sum())
        lose = int((ours_ens < pub[m]).sum())
        md.append(f'| {m} | 50 个逐 TF 模型 | {pub[m].mean():.4f} | {np.median(pub[m]):.4f} | '
                  f'{win}/{lose} | {ours_ens.mean()-pub[m].mean():+.4f} | {wilcoxon(ours_ens, pub[m]):.4g} |')
    for name, arr in REPRO.items():
        win = int((ours_ens > arr).sum())
        lose = int((ours_ens < arr).sum())
        md.append(f'| **{name}（本机复现）** | 50 个逐 TF 模型 | **{arr.mean():.4f}** | '
                  f'{np.median(arr):.4f} | {win}/{lose} | {ours_ens.mean()-arr.mean():+.4f} | '
                  f'{wilcoxon(ours_ens, arr):.4g} |')
    md.append(f'| ProMoScan（蛋白打乱对照） | 1 个模型 | {np.nanmean(ours_shuf):.4f} | '
              f'{np.nanmedian(ours_shuf):.4f} | — | — | — |')
    md.append('')
    md.append('## 逐数据集胜负分布（集成 vs 各方法）\n')
    md.append('| 方法 | 胜 | 平(|Δ|≤0.002) | 负 | 最大领先 | 最大落后 |')
    md.append('|---|---|---|---|---|---|')
    for m in order:
        d = ours_ens - pub[m]
        md.append(f'| {m} | {int((d>0.002).sum())} | {int((abs(d)<=0.002).sum())} | {int((d<-0.002).sum())} | '
                  f'{d.max():+.4f} | {d.min():+.4f} |')
    for name, arr in REPRO.items():
        d = ours_ens - arr
        md.append(f'| **{name}（本机复现）** | {int((d>0.002).sum())} | {int((abs(d)<=0.002).sum())} | '
                  f'{int((d<-0.002).sum())} | {d.max():+.4f} | {d.min():+.4f} |')
    md.append('')
    md.append('## 逐数据集明细\n')
    md.append('见 `tables/bench50_per_dataset_compare.csv`。\n')

    OUT.mkdir(exist_ok=True)
    (OUT / 'Table_bench50_compare.md').write_text('\n'.join(md), encoding='utf-8')

    cols = ['dsid', 'cell', 'symbol', 'pmoscan_ens', 'pmoscan_single', 'pmoscan_shuffled'] + \
           [f'pub_{m}' for m in BASE_METHODS] + (['cbrkan_repro'] if repro_auc is not None else [])
    with open(OUT / 'bench50_per_dataset_compare.csv', 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for i, d in enumerate(dsids):
            r = rows[d]
            w.writerow([d, r.get('cell'), r.get('symbol'),
                        round(ours_ens[i], 4),
                        round(ours_single[i], 4) if ours_single is not None else '',
                        round(float(ours_shuf[i]), 4) if not np.isnan(ours_shuf[i]) else '']
                       + [round(float(pub[m][i]), 4) for m in BASE_METHODS]
                       + ([round(float(repro_auc[i]), 4)] if repro_auc is not None else []))
    print('\n'.join(md))


if __name__ == '__main__':
    main()
