#!/usr/bin/env python3
# coding: UTF-8
"""在同 50 个数据集上复现 CBR-KAN（逐 TF 模型），用于与 ProMoScan 严格同协议对照

数据：data/bench50/train/{dsid}.{pos,neg}.fasta（12,150 + 12,150）
      data/bench50/test/{dsid}.{pos,neg}.fasta（1,500 + 1,500）—— 与 ProMoScan 评测完全同一份
模型：TF-MTL/code/models/CBR-KAN.py 原实现（~2.37M）
训练：85/15 从 train 切分（seed 42），30 epoch，batch 128，lr 1e-3，按 val AUC 取最优再测 test

输出
  results/bench50_cbrkan/{dsid}.json
  results/bench50_table_cbrkan.json     （逐数据集汇总）
用法
  python3 code/benchmark/train_cbrkan_bench50.py            # 全量 50 个，幂等
"""
import os

for _v in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
           'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'TOKENIZERS_PARALLELISM']:
    os.environ.setdefault(_v, '1')

import argparse
import json
import sys
import time
from pathlib import Path

import torch

torch.set_num_threads(1)

BASE = Path(os.environ.get('PROMOSCAN_ROOT',
                          Path(__file__).resolve().parents[2]))
TFMTL = Path(os.environ.get('TFMTL_ROOT', BASE / 'external/TF-MTL'))
for p in [str(TFMTL / 'code' / 'models'), str(TFMTL / 'code'), str(TFMTL), os.path.expanduser('~/efficient_kan')]:
    if p not in sys.path:
        sys.path.insert(0, p)

import config as tfcfg                      # noqa: E402
import utils                                # noqa: E402
from train_eval import train, init_network  # noqa: E402
from importlib import import_module         # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

TRAIN = PRODEEP / 'data/bench50/train'
TEST = PRODEEP / 'data/bench50/test'


def read_fasta(path):
    seqs, cur = [], []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                if cur:
                    seqs.append(''.join(cur))
                    cur = []
            else:
                cur.append(line)
    if cur:
        seqs.append(''.join(cur))
    return seqs


def ds_list():
    return sorted(p.name[:-len('.pos.fasta')] for p in TRAIN.glob('*.pos.fasta'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(PRODEEP / 'results/bench50_cbrkan'))
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--only', default=None)
    ap.add_argument('--model', default='CBR-KAN',
                    help='TF-MTL code/models 下的模型名：CBR-KAN / DeepBind / DanQ / DeepSEA / DeepD2V')
    ap.add_argument('--lr', type=float, default=None, help='默认用 TF-MTL config.learning_rate')
    ap.add_argument('--bs', type=int, default=None, help='默认用 TF-MTL config.BATCH_SIZE')
    ap.add_argument('--cv', type=int, default=0,
                    help='>0 时改成 k 折交叉验证（train+test 合并，按 DeepBind/DanQ 惯例），auc=折均值')
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'logs').mkdir(exist_ok=True)
    (out / 'tmp').mkdir(exist_ok=True)

    if args.lr is not None:
        tfcfg.learning_rate = args.lr
    if args.bs is not None:
        tfcfg.BATCH_SIZE = args.bs
        tfcfg.batch_size = args.bs
    print(f'model={args.model} lr={tfcfg.learning_rate} bs={tfcfg.BATCH_SIZE} epochs={args.epochs}', flush=True)
    dsids = ds_list() if not args.only else [args.only]
    print(f'数据集 {len(dsids)} 个', flush=True)

    for k, dsid in enumerate(dsids, 1):
        js = out / f'{dsid}.json'
        if js.exists():
            print(f'[{k}/{len(dsids)}] skip {dsid}', flush=True)
            continue
        t0 = time.time()
        tr_pos = read_fasta(TRAIN / f'{dsid}.pos.fasta')
        tr_neg = read_fasta(TRAIN / f'{dsid}.neg.fasta')
        te_pos = read_fasta(TEST / f'{dsid}.pos.fasta')
        te_neg = read_fasta(TEST / f'{dsid}.neg.fasta')

        texts = [utils.word_cut(s.upper()) for s in tr_pos + tr_neg + te_pos + te_neg]
        labels = [1] * len(tr_pos) + [0] * len(tr_neg) + [1] * len(te_pos) + [0] * len(te_neg)

        if args.cv and args.cv > 1:
            from sklearn.model_selection import StratifiedKFold
            from sklearn.metrics import roc_auc_score
            skf = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=args.seed)
            fold_aucs = []
            for fi, (tri, tei) in enumerate(skf.split(texts, labels), 1):
                a_t = [texts[i] for i in tri]
                a_l = [labels[i] for i in tri]
                test_t = [texts[i] for i in tei]
                test_l = [labels[i] for i in tei]
                # 从折内训练集再切 15% 作 val（不碰折外测试集）
                fit_t, val_t, fit_l, val_l = train_test_split(a_t, a_l, test_size=0.15,
                                                              random_state=args.seed, stratify=a_l)
                tfcfg.pretrain_dir = str(out / 'tmp' / f'{dsid}_f{fi}.npz')
                tfcfg.w2v_dir = str(TFMTL / 'data/3mer.txt')
                vocab, tr, va, te = utils.build_dataset(fit_t, fit_l, val_t, val_l, test_t, test_l)
                tr_it = utils.build_iterator(tr, tfcfg.BATCH_SIZE, tfcfg.DEVICE)
                va_it = utils.build_iterator(va, tfcfg.BATCH_SIZE, tfcfg.DEVICE)
                te_it = utils.build_iterator(te, tfcfg.BATCH_SIZE, tfcfg.DEVICE)
                tfcfg.SEED = args.seed
                tfcfg.num_epochs = args.epochs
                tfcfg.model_name = args.model
                tfcfg.save_path = str(out / 'tmp' / f'{dsid}_f{fi}.ckpt')
                net = import_module('models.' + tfcfg.model_name).Model(tfcfg).to(tfcfg.DEVICE)
                init_network(net)
                _, auc_f, _, _, _, _, _ = train(tfcfg, net, tr_it, va_it, te_it, file=f'{dsid}_f{fi}')
                fold_aucs.append(float(auc_f))
                print(f'  {dsid} fold{fi}/{args.cv} AUC={auc_f:.4f}', flush=True)
                torch.cuda.empty_cache()
            import statistics as st
            res = dict(dsid=dsid, auc=st.mean(fold_aucs), cv=args.cv, fold_aucs=fold_aucs,
                       train_s=round(time.time() - t0, 1))
            js.write_text(json.dumps(res, indent=1), encoding='utf-8')
            print(f'[{k}/{len(dsids)}] {dsid} CV{args.cv} AUC={res["auc"]:.4f} {res["train_s"]:.0f}s', flush=True)
            continue

        n_tr = len(tr_pos) + len(tr_neg)
        tr_texts, te_texts_ = texts[:n_tr], texts[n_tr:]
        tr_lab, te_lab = labels[:n_tr], labels[n_tr:]

        a_t, b_t, a_l, b_l = train_test_split(tr_texts, tr_lab, test_size=0.15,
                                              random_state=args.seed, stratify=tr_lab)
        tfcfg.pretrain_dir = str(out / 'tmp' / f'{dsid}.npz')
        tfcfg.w2v_dir = str(TFMTL / 'data/3mer.txt')
        vocab, tr, va, te = utils.build_dataset(a_t, a_l, b_t, b_l, te_texts_, te_lab)
        tr_it = utils.build_iterator(tr, tfcfg.BATCH_SIZE, tfcfg.DEVICE)
        va_it = utils.build_iterator(va, tfcfg.BATCH_SIZE, tfcfg.DEVICE)
        te_it = utils.build_iterator(te, tfcfg.BATCH_SIZE, tfcfg.DEVICE)

        tfcfg.SEED = args.seed
        tfcfg.num_epochs = args.epochs
        tfcfg.model_name = args.model
        tfcfg.save_path = str(out / 'tmp' / f'{dsid}.ckpt')

        net = import_module('models.' + tfcfg.model_name).Model(tfcfg).to(tfcfg.DEVICE)
        init_network(net)
        mcc, auc, pr_auc, pre, rec, acc, f1 = train(tfcfg, net, tr_it, va_it, te_it, file=dsid)

        res = dict(dsid=dsid, auc=float(auc), pr_auc=float(pr_auc), accuracy=float(acc),
                   f1=float(f1), n_params=sum(p.numel() for p in net.parameters()),
                   train_s=round(time.time() - t0, 1))
        js.write_text(json.dumps(res, indent=1), encoding='utf-8')
        print(f'[{k}/{len(dsids)}] {dsid} AUC={res["auc"]:.4f} {res["train_s"]:.0f}s', flush=True)
        torch.cuda.empty_cache()

    # 汇总成逐数据集表
    # 只收逐数据集结果；summary.json 也在同一目录，必须排除（否则缺 'auc' 键就崩）
    rows = [json.loads(p.read_text()) for p in sorted(out.glob('*.json'))
            if p.name != 'summary.json']
    rows = [r for r in rows if 'auc' in r and 'dsid' in r]
    if rows:
        import statistics as st
        tbl = dict(n=len(rows), pmoscan_mean=st.mean(r['auc'] for r in rows),
                   per_dataset=[dict(dsid=r['dsid'], auc=r['auc']) for r in rows])
        # 总是写到自己的 out 目录；只有在跑满 50 个时才更新全局对照文件
        # （防止 --only / --epochs 探针把 50 数据集汇总覆盖成 1 个数据集）
        (out / 'summary.json').write_text(json.dumps(tbl, indent=1))
        if len(rows) >= 50:
            tag = args.model.replace('-', '')
            (PRODEEP / f'results/bench50_table_{tag}.json').write_text(json.dumps(tbl, indent=1))
        print(f'汇总 {len(rows)} 个数据集 → {out}/summary.json '
              f'mean AUC={tbl["pmoscan_mean"]:.4f}', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
