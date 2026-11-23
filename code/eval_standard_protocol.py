#!/usr/bin/env python3
# coding: UTF-8
"""路线 A：标准协议（peak vs 随机基因组背景）下的 AUC / 准确率。

协议：
  · 正样本 = peak 测试段 pos[800:1000]（200 条，与 eval_standard_task 同测试段）
  · 负样本 = hg38 随机 200bp 基因组窗口（200 条，seed 固定可复现）
  · 1:1 平衡；对每个 TF 报 ROC-AUC / Youden 最佳阈值准确率 / 0.5 阈值准确率
  · 汇总：seen TF（前 n_tf 个）与 holdout TF 各自的均值
对照模型：DNA-only(variant E) / ProtoTF(K=167) / 任意 ProMoScan tag；--ens 支持 seed 集成
"""
import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

BASE = Path(__file__).resolve().parents[1]
CODE = BASE / 'code'
for p in [str(CODE), str(CODE / 'protoTF'), os.path.expanduser('~/efficient_kan')]:
    sys.path.append(p)
from prototf_net import ProtoTFNet, CBRKANSharedBackbone, KAN            # noqa: E402
from promoscan import ProMoScan, dna_onehot                          # noqa: E402
from protein_features import build_protein_features                      # noqa: E402
from data_utils import (build_vocab, seq_to_ids, load_fasta_balanced,    # noqa: E402
                        load_protein_fasta)                              # noqa: E402
from bg_utils import Genome, dinuc_shuffle, mono_shuffle, stable_seed                     # noqa: E402

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DATA_ROOT = str(BASE / 'raw_data')
ZS = Path(os.environ.get('ZS_RESULTS', BASE / 'external/multiTF_results'))
PROT_FASTA = os.environ.get('PROT_FASTA', str(BASE / 'data/tf_sequences_final_v2.fasta'))
CSV = str(CODE / 'protoTF/full_1043.csv')
PAD = 200
VOCAB = build_vocab()


class SingleHeadModel(torch.nn.Module):
    def __init__(self, len_vocab, kan_hidden=32):
        super().__init__()
        self.backbone = CBRKANSharedBackbone(len_vocab)
        self.head = KAN([256, kan_hidden, 2])

    def forward(self, tokens, tf_idx=None):
        return self.head(self.backbone(tokens))


def acc_at_best(y, s):
    y = np.asarray(y); s = np.asarray(s)
    best = 0.0
    for t in np.unique(s):
        p = (s >= t)
        best = max(best, (p == y).mean(), (p == (1 - y)).mean())
    return float(best)


def acc_at_half(y, s):
    y = np.asarray(y)
    return float(((np.asarray(s) >= 0.5).astype(int) == y).mean())


@torch.no_grad()
def score_deep(net, seqs, aa, bs=256):
    out = []
    for i in range(0, len(seqs), bs):
        ch = seqs[i:i + bs]
        ids = torch.stack([torch.LongTensor(seq_to_ids(s, VOCAB)) for s in ch]).to(DEVICE)
        oh = torch.stack([dna_onehot(s, PAD) for s in ch]).to(DEVICE)
        a = aa.unsqueeze(0).repeat(len(ch), *([1] * aa.dim())).to(DEVICE)
        out.append(torch.sigmoid(net(ids, oh, a)).reshape(-1).cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def score_dna(net, seqs, bs=256):
    out = []
    for i in range(0, len(seqs), bs):
        ch = seqs[i:i + bs]
        ids = torch.stack([torch.LongTensor(seq_to_ids(s, VOCAB)) for s in ch]).to(DEVICE)
        out.append(torch.softmax(net(ids, None), -1)[:, 1].cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def score_proto(net, seqs, aa, bs=128):
    out = []
    for i in range(0, len(seqs), bs):
        ch = seqs[i:i + bs]
        ids = torch.stack([torch.LongTensor(seq_to_ids(s, VOCAB)) for s in ch]).to(DEVICE)
        a = aa.unsqueeze(0).repeat(len(ch), *([1] * aa.dim())).to(DEVICE)
        out.append(torch.softmax(net(ids, a), -1)[:, 1].cpu().numpy())
    return np.concatenate(out)


def load_net(tag):
    ck = torch.load(BASE / f'results/promoscan_{tag}/best_model.pt',
                    map_location='cpu', weights_only=False)
    cfg = ck['config']
    n = ProMoScan(len(VOCAB), M=cfg['M'], W=cfg['W'],
                     use_dna_context=cfg.get('use_dna_context', False),
                     alpha_temp=cfg.get('alpha_temp', 1.0),
                     rc_symmetric=cfg.get('rc_symmetric', False),
                     pool=cfg.get('pool', 'max'),
                     motif_head=cfg.get('motif_head', 'std'), protein_enc=cfg.get('protein_enc', 'cnn')).to(DEVICE).eval()
    n.load_state_dict(ck['model'])
    return n, cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tags', default='L10,M34')
    ap.add_argument('--ens', default='M34,M50,M51', help='seed 集成的 tag（逗号分隔；空=不评集成）')
    ap.add_argument('--n-tf', type=int, default=200)
    ap.add_argument('--seen-order', default='csv', choices=['csv', 'train'],
                    help="csv=CSV 顺序；train=复现训练脚本 seed 42 的 seen 取法")
    ap.add_argument('--n-sup', type=int, default=938, help='seen-order=train 时，训练用的 TF 数')
    ap.add_argument('--bg-seed', type=int, default=42)
    ap.add_argument('--dna-ckpt', default=str(ZS / 'prototf_variant_E_zs_random10/best_model.pt'),
                    help='DNA-only(variant E) checkpoint；R1 用 bgroot 重训的可指向 *_bg 目录')
    ap.add_argument('--proto-ckpt', default=str(ZS / 'prototf_zs_random10/best_model.pt'),
                    help='ProtoTF(K=167) checkpoint；R1 用 bgroot 重训的可指向 *_bg 目录')
    ap.add_argument('--neg-mode', default='bg', choices=['bg', 'dinuc', 'mono'],
                    help='bg=hg38 随机基因组窗口；dinuc=正样本二核苷酸洗牌；mono=单核苷酸洗牌')
    ap.add_argument('--out', default='results/standard_protocol.json')
    a = ap.parse_args()

    rows = list(csv.DictReader(open(CSV)))
    tf2gsm = {r['TF_Name']: r['GSM_ID'] for r in rows}
    prot = load_protein_fasta(PROT_FASTA)
    zs = json.load(open(ZS / 'prototf_zs_random10/zero_shot_summary.json'))
    holdout = set(zs['holdout_tfs'])
    seen = [t for t in tf2gsm if t not in holdout and t in prot]
    if a.seen_order == 'train':
        import random as _r
        _rr = _r.Random(42)
        pool = [t for t in tf2gsm if t not in holdout and t in prot]
        _rr.shuffle(pool)
        seen = sorted(pool[:a.n_sup])
    hold = [t for t in holdout if t in tf2gsm and t in prot]
    names = sorted(set(seen) | set(hold))
    idx = {t: i for i, t in enumerate(names)}
    oh_cnn = build_protein_features(names, enc='cnn')
    if DEVICE == 'cuda':
        oh_cnn = oh_cnn.to(DEVICE)

    tags = [t.strip() for t in a.tags.split(',') if t.strip()]
    ens = [t.strip() for t in a.ens.split(',') if t.strip()]
    ours, feats = {}, {}
    for t in set(tags + ens):
        n_, cfg = load_net(t)
        ours[t] = n_
        feats[t] = build_protein_features(names, enc=cfg.get('protein_enc', 'cnn'))
        if DEVICE == 'cuda':
            feats[t] = feats[t].to(DEVICE)

    netP = ProtoTFNet(len_vocab=len(VOCAB), n_prototypes=167).to(DEVICE)
    netP.load_state_dict(torch.load(a.proto_ckpt, map_location=DEVICE, weights_only=False)['model']); netP.eval()
    netE = SingleHeadModel(len(VOCAB)).to(DEVICE)
    sdE = torch.load(a.dna_ckpt, map_location=DEVICE, weights_only=False)
    netE.load_state_dict(sdE['model'] if 'model' in sdE else sdE); netE.eval()
    print(f'对照 ckpt: DNA-only={Path(a.dna_ckpt).parent.name} | ProtoTF={Path(a.proto_ckpt).parent.name}', flush=True)

    genome = Genome()
    rng_tf = np.random.RandomState(7)

    def run_pool(pool_tfs, max_tf):
        keys = ['dna', 'proto', 'proto_shuf'] + tags + ([ 'ens'] if ens else [])
        acc = {k + s: [] for k in keys for s in ('_auc', '_acc', '_acc_half')}
        n_used = 0
        for t in pool_tfs:
            if max_tf and n_used >= max_tf:
                break
            pos, _ = load_fasta_balanced(tf2gsm[t], DATA_ROOT, 2000)
            if len(pos) < 1000:
                continue
            p = pos[800:1000]
            if a.neg_mode == 'bg':
                negs = [s for _, s in genome.sample(len(p), 200, seed=a.bg_seed + (stable_seed(t)))]
            else:
                _rng = __import__('random').Random(a.bg_seed + (stable_seed(t)))
                if a.neg_mode == 'dinuc':
                    negs = [dinuc_shuffle(s, _rng) for s in p]
                else:
                    negs = [mono_shuffle(s, _rng) for s in p]
            seqs = p + negs
            y = np.array([1] * len(p) + [0] * len(negs))
            ia = idx[t]
            jj = int(rng_tf.randint(len(names)))
            sD = score_dna(netE, seqs)
            aaP = oh_cnn[ia]
            sP = score_proto(netP, seqs, aaP)
            sPs = score_proto(netP, seqs, oh_cnn[jj])
            for k, s in [('dna', sD), ('proto', sP), ('proto_shuf', sPs)]:
                acc[k + '_auc'].append(roc_auc_score(y, s)); acc[k + '_acc'].append(acc_at_best(y, s))
                acc[k + '_acc_half'].append(acc_at_half(y, s))
            per = {}
            for tag in set(tags + ens):
                s = score_deep(ours[tag], seqs, feats[tag][ia])
                per[tag] = s
                if tag in tags:
                    for suf, v in [('_auc', roc_auc_score(y, s)), ('_acc', acc_at_best(y, s)),
                                   ('_acc_half', acc_at_half(y, s))]:
                        acc[tag + suf].append(v)
            if ens:
                se = np.mean([per[k] for k in ens], axis=0)
                acc['ens_auc'].append(roc_auc_score(y, se)); acc['ens_acc'].append(acc_at_best(y, se))
                acc['ens_acc_half'].append(acc_at_half(y, se))
            n_used += 1
            if n_used % 25 == 0:
                print(f'  {n_used} TF 完成', flush=True)
        out = {k: float(np.mean(v)) for k, v in acc.items() if v}
        # 逐 TF 原始值（画箱线图 / 分布 / 重算 CI 用；此前只存均值导致无法出图）
        out['_raw'] = {k: [float(x) for x in v] for k, v in acc.items() if v}
        out['n_tf'] = n_used
        return out

    _shuf_j = None
    res = {'protocol': f'peak[800:1000] vs neg-mode={a.neg_mode} (1:1)',
           'bg_seed': a.bg_seed, 'tags': tags, 'ens': ens,
           'seen': run_pool(seen, a.n_tf), 'holdout': run_pool(hold, 0)}
    print(json.dumps(res, indent=2, ensure_ascii=False), flush=True)
    Path(BASE / a.out).write_text(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
