#!/usr/bin/env python3
# coding: UTF-8
"""构建路线 A 数据集根目录 data/bgroot/：
  · {GSM}.pos.fasta → 软链到 raw_data（正样本不变）
  · {GSM}.neg.fasta → hg38 随机基因组背景（新生成）
之后训练/评估用 --data-root data/bgroot 即可切到标准协议。
"""
import argparse
import csv
import os
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / 'code'))
from bg_utils import Genome, count_fasta, stable_seed          # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default=str(BASE / 'code/protoTF/full_1043.csv'))
    ap.add_argument('--raw-root', default=str(BASE / 'raw_data'))
    ap.add_argument('--out-root', default=str(BASE / 'data/bgroot'))
    ap.add_argument('--n-neg', type=int, default=1000, help='每个 GSM 生成的背景序列数')
    ap.add_argument('--n-tf', type=int, default=0, help='0=CSV 内全部')
    ap.add_argument('--only', default='', help='逗号分隔的 TF 名（只做这些，冒烟用）')
    ap.add_argument('--seed', type=int, default=42)
    a = ap.parse_args()

    rows = list(csv.DictReader(open(a.csv)))
    if a.only:
        keep = {t.strip() for t in a.only.split(',') if t.strip()}
        rows = [r for r in rows if r['TF_Name'] in keep]
    if a.n_tf:
        rows = rows[:a.n_tf]
    out = Path(a.out_root); out.mkdir(parents=True, exist_ok=True)
    g = Genome()
    done = miss = 0
    for r in rows:
        gsm = r['GSM_ID']
        pos_src = Path(a.raw_root) / f'{gsm}.pos.fasta'
        if not pos_src.exists():
            miss += 1
            continue
        link = out / f'{gsm}.pos.fasta'
        if not link.exists():
            os.symlink(pos_src.resolve(), link)
        dst = out / f'{gsm}.neg.fasta'
        if dst.exists():
            done += 1
            continue
        nneed = min(a.n_neg, count_fasta(pos_src)) if count_fasta(pos_src) else a.n_neg
        nneed = max(nneed, 1)
        seqs = g.sample(nneed, 200, seed=a.seed + (stable_seed(gsm)))
        with open(dst, 'w') as fo:
            for i, (loc, s) in enumerate(seqs):
                fo.write(f'>{gsm}.bg_{i}::{loc}\n{s}\n')
        done += 1
        if done % 100 == 0:
            print(f'  {done} GSM 完成', flush=True)
    print(f'完成 {done} 个 GSM，缺失 pos {miss} 个 → {out}', flush=True)


if __name__ == '__main__':
    main()
