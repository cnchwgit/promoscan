#!/usr/bin/env python3
# coding: UTF-8
"""D1 基准评估：加载训练好的 ProMoScan，在 50 个数据集的 test 划分上逐数据集算 ROC-AUC，
并与 CBR-KAN 原文 Table 4 的已发表数字并排（产出可直接进论文的 Table）。
"""
import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE)); sys.path.insert(0, str(CODE / 'protoTF'))
from prodeep_v1 import ProMoScan, dna_onehot  # noqa: E402
from data_utils import build_vocab, seq_to_ids  # noqa: E402
from prodeep_v1 import revcomp_seq  # noqa: E402,F401

BASE = Path(__file__).resolve().parents[2]
BENCH = BASE / 'code/benchmark'
VOCAB = build_vocab()
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
PAD = 200

# 原文 Table 4（逐数据集 ROC-AUC）。键=(cell, 原文 TF 标签)
PUB = {}
_RAW = """Gm12878 Batf 0.897 0.909 0.927 0.938 0.954
Gm12878 Bcl11a 0.853 0.857 0.500 0.884 0.918
Gm12878 Bcl3 0.880 0.901 0.930 0.904 0.935
Gm12878 Bclaf 0.831 0.877 0.871 0.874 0.897
Gm12878 Ebf 0.880 0.883 0.901 0.895 0.922
Gm12878 Egr1 0.939 0.959 0.963 0.965 0.968
Gm12878 Elf1 0.903 0.902 0.928 0.925 0.931
Gm12878 Ets1 0.909 0.940 0.957 0.947 0.959
Gm12878 Irf4 0.831 0.864 0.876 0.871 0.907
Gm12878 Mef2a 0.812 0.898 0.922 0.911 0.931
Gm12878 Nrsf 0.887 0.905 0.930 0.936 0.939
Gm12878 Pax5c20 0.859 0.876 0.888 0.892 0.907
Gm12878 Pax5n19 0.862 0.890 0.908 0.900 0.931
Gm12878 Pbx3 0.884 0.893 0.909 0.925 0.938
Gm12878 Pou2f2 0.847 0.869 0.869 0.878 0.908
Gm12878 Pu1 0.961 0.965 0.970 0.974 0.980
Gm12878 Rad21 0.949 0.973 0.987 0.987 0.987
Gm12878 Sp1 0.825 0.847 0.857 0.859 0.899
Gm12878 Srf 0.871 0.902 0.945 0.931 0.948
Gm12878 Taf1 0.915 0.915 0.923 0.922 0.927
Gm12878 Tcf12 0.906 0.908 0.920 0.918 0.938
Gm12878 Usf1 0.937 0.951 0.957 0.959 0.960
Gm12878 Yy1 0.881 0.874 0.914 0.913 0.932
H1hesc Gabp 0.895 0.899 0.942 0.927 0.943
H1hesc Nrsf 0.903 0.938 0.962 0.960 0.967
H1hesc Rad21 0.963 0.964 0.978 0.982 0.982
H1hesc Sin3 0.917 0.923 0.934 0.934 0.938
H1hesc Sp1 0.893 0.898 0.921 0.910 0.935
H1hesc Srf 0.937 0.916 0.969 0.963 0.975
H1hesc Taf1 0.926 0.931 0.932 0.933 0.936
H1hesc Tcf12 0.834 0.883 0.924 0.898 0.929
H1hesc Usf1 0.972 0.974 0.981 0.980 0.984
H1hesc Yy1 0.910 0.924 0.948 0.942 0.955
K562 Atf3 0.919 0.933 0.948 0.947 0.955
K562 E2f6 0.945 0.954 0.958 0.952 0.959
K562 Egr1 0.964 0.966 0.963 0.970 0.972
K562 Elf1 0.933 0.938 0.940 0.943 0.945
K562 Ets1 0.900 0.914 0.912 0.914 0.925
K562 Fosl1 0.945 0.964 0.965 0.962 0.968
K562 Gabp 0.931 0.931 0.954 0.954 0.964
K562 Gata2 0.847 0.846 0.872 0.889 0.910
K562 Hey1 0.876 0.888 0.892 0.889 0.895
K562 Max 0.929 0.936 0.942 0.938 0.947
K562 Nrsf 0.876 0.918 0.930 0.922 0.942
K562 Pu1 0.979 0.983 0.985 0.985 0.988
K562 Rad21 0.968 0.976 0.985 0.991 0.989
K562 Srf 0.857 0.907 0.932 0.930 0.940
K562 Taf1 0.920 0.926 0.937 0.934 0.938
K562 Usf1 0.947 0.953 0.964 0.962 0.966
K562 Yy1 0.913 0.920 0.936 0.926 0.938"""
for ln in _RAW.strip().split('\n'):
    p = ln.split()
    PUB[(p[0], p[1])] = [float(x) for x in p[2:7]]   # DeepBind,DanQ,DeepD2V,DeepSEA,CBR-KAN


def paper_token(rest):
    r = re.sub(r'Pcr[12]x$', '', rest)
    r = re.sub(r'V041610[12]$', '', r)
    r = re.sub(r'sc[0-9]+$', '', r)
    if r == 'Bclaf101388':
        r = 'Bclaf'
    if r == 'Sin3ak20':
        r = 'Sin3'
    return r


def read_fasta(fp):
    out = []
    with open(fp) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('>'):
                out.append(line.upper())
    return out


@torch.no_grad()
def score(nets, seqs, aa, bs=256):
    if not isinstance(nets, (list, tuple)):
        nets = [nets]
    out = []
    for i in range(0, len(seqs), bs):
        ch = seqs[i:i + bs]
        ids = torch.stack([torch.LongTensor(seq_to_ids(s, VOCAB)) for s in ch]).to(DEVICE)
        oh = torch.stack([dna_onehot(s, PAD) for s in ch]).to(DEVICE)
        a = aa.unsqueeze(0).repeat(len(ch), 1).to(DEVICE)
        p = np.mean([torch.sigmoid(n(ids, oh, a)).cpu().numpy() for n in nets], axis=0)
        out.append(p)
    return np.concatenate(out)


def load_net(path):
    ck = torch.load(path, map_location=DEVICE, weights_only=False)
    cfg = ck.get('config', {})
    net = ProMoScan(len(VOCAB), M=cfg.get('M', 16), W=cfg.get('W', 16), use_dna_context=True,
                    protein_enc=cfg.get('protein_enc', 'esm'),
                    motif_head=cfg.get('motif_head', 'std')).to(DEVICE)
    net.load_state_dict(ck['model']); net.eval()
    return net


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default=str(BASE / 'results/prodeep_v1_BENCH50/best_model.pt'))
    ap.add_argument('--ens', default='', help='逗号分隔的额外 ckpt，与 --ckpt 一起做 seed 集成（概率平均）')
    ap.add_argument('--meta', default=str(BENCH / 'bench50_meta.json'))
    ap.add_argument('--esm-pt', default=str(BENCH / 'bench50_esm.pt'))
    ap.add_argument('--test-root', default=str(BASE / 'data/bench50/test'))
    ap.add_argument('--out', default=str(BASE / 'results/bench50_table.json'))
    ap.add_argument('--shuffle-control', action='store_true')
    ap.add_argument('--seed', type=int, default=42)
    a = ap.parse_args()

    meta = json.load(open(a.meta))
    names = [m['dsid'] for m in meta]
    idx = {t: i for i, t in enumerate(names)}
    e = torch.load(a.esm_pt, map_location='cpu', weights_only=False)
    prot = torch.stack([torch.as_tensor(e[n], dtype=torch.float32) for n in names])

    paths = [a.ckpt] + [p for p in a.ens.split(',') if p]
    nets = [load_net(p) for p in paths]
    print(f'加载 {len(nets)} 个模型: {[Path(p).parent.name for p in paths]}', flush=True)
    net = nets

    rng = np.random.RandomState(a.seed)
    rows, ours, shuff, pubs = [], [], [], []
    for m in meta:
        dsid = m['dsid']
        pos = read_fasta(Path(a.test_root) / f'{dsid}.pos.fasta')
        neg = read_fasta(Path(a.test_root) / f'{dsid}.neg.fasta')
        y = np.array([1] * len(pos) + [0] * len(neg))
        auc = roc_auc_score(y, score(nets, pos + neg, prot[idx[dsid]]))
        j = int(rng.randint(len(names)))
        auc_sh = roc_auc_score(y, score(net, pos + neg, prot[j])) if a.shuffle_control else float('nan')
        pt = PUB.get((m['cell'], paper_token(m['tf_token'])))
        rows.append(dict(dsid=dsid, cell=m['cell'], symbol=m['symbol'], tf_token=m['tf_token'],
                         pmoscan=float(auc), pmoscan_shuffled=float(auc_sh),
                         published=(dict(zip(['DeepBind', 'DanQ', 'DeepD2V', 'DeepSEA', 'CBR-KAN'], pt)) if pt else None)))
        ours.append(auc); pubs.append(pt)
        if not np.isnan(auc_sh):
            shuff.append(auc_sh)
    pa = np.array(pubs, dtype=float)
    summary = dict(
        n=len(ours),
        pmoscan_mean=float(np.mean(ours)), pmoscan_median=float(np.median(ours)),
        published_mean={k: float(pa[:, i].mean()) for i, k in enumerate(['DeepBind', 'DanQ', 'DeepD2V', 'DeepSEA', 'CBR-KAN'])},
        pmoscan_shuffled_mean=float(np.mean(shuff)) if shuff else None,
        per_dataset=rows)
    Path(a.out).write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps({k: v for k, v in summary.items() if k != 'per_dataset'}, indent=2, ensure_ascii=False))
    print(f'\nper-dataset -> {a.out}')


if __name__ == '__main__':
    main()
