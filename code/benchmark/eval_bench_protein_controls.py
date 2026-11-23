#!/usr/bin/env python3
# coding: UTF-8
"""50 数据集上的蛋白输入归因对照（只推理，不训练）

问题：把蛋白打乱后 AUC 从 0.9376 掉到 0.8771（−0.060）。这 0.06 是"TF 身份"，
还是"随便换一个蛋白都会掉"（即模型只是需要某个蛋白占位）？

五种输入条件（序列、模型、权重完全相同）：
  1. correct        正确蛋白
  2. random         随机另一个数据集的蛋白（seed 42，与原文对照一致）
  3. nearest        蛋白空间最近邻（同家族/旁系同源代理）
  4. farthest       蛋白空间最远（跨家族）
  5. centroid       全部蛋白的均值向量（"通用 TF"占位）

判读：
  correct > nearest > random ≈ farthest ≈ centroid  → 蛋白信息是**分级**的（同家族共享 + 身份区分）
  correct ≈ nearest ≈ random                        → 蛋白只是占位，没有身份信息
输出 results/bench50_protein_controls.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE)); sys.path.insert(0, str(CODE / 'protoTF'))
from promoscan import ProMoScan, dna_onehot  # noqa: E402
from data_utils import build_vocab, seq_to_ids  # noqa: E402

BASE = Path(__file__).resolve().parents[2]
BENCH = BASE / 'code/benchmark'
VOCAB = build_vocab()
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
PAD = 200


def read_fasta(fp):
    out = []
    with open(fp) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('>'):
                out.append(line.upper())
    return out


@torch.no_grad()
def score(net, seqs, aa, bs=256):
    out = []
    for i in range(0, len(seqs), bs):
        ch = seqs[i:i + bs]
        ids = torch.stack([torch.LongTensor(seq_to_ids(s, VOCAB)) for s in ch]).to(DEVICE)
        oh = torch.stack([dna_onehot(s, PAD) for s in ch]).to(DEVICE)
        a = aa.unsqueeze(0).repeat(len(ch), 1).to(DEVICE)
        out.append(torch.sigmoid(net(ids, oh, a)).cpu().numpy())
    return np.concatenate(out)


def load_net(path):
    ck = torch.load(path, map_location=DEVICE, weights_only=False)
    cfg = ck.get('config', {})
    net = ProMoScan(len(VOCAB), M=cfg.get('M', 16), W=cfg.get('W', 16),
                    use_dna_context=cfg.get('use_dna_context', True),
                    protein_enc=cfg.get('protein_enc', 'esm'),
                    motif_head=cfg.get('motif_head', 'std')).to(DEVICE)
    net.load_state_dict(ck['model']); net.eval()
    return net


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default=str(BASE / 'results/promoscan_BENCH50/best_model.pt'))
    ap.add_argument('--meta', default=str(BENCH / 'bench50_meta.json'))
    ap.add_argument('--esm-pt', default=str(BENCH / 'bench50_esm.pt'))
    ap.add_argument('--test-root', default=str(BASE / 'data/bench50/test'))
    ap.add_argument('--out', default=str(BASE / 'results/bench50_protein_controls.json'))
    ap.add_argument('--seed', type=int, default=42)
    a = ap.parse_args()

    meta = json.load(open(a.meta))
    names = [m['dsid'] for m in meta]
    e = torch.load(a.esm_pt, map_location='cpu', weights_only=False)
    P = torch.stack([torch.as_tensor(e[n], dtype=torch.float32) for n in names])
    Pn = torch.nn.functional.normalize(P, dim=1)
    S = (Pn @ Pn.T).numpy()                     # 余弦相似度
    S_near = S.copy(); np.fill_diagonal(S_near, -9.0)
    S_far = S.copy(); np.fill_diagonal(S_far, 9.0)
    near = S_near.argmax(axis=1)
    far = S_far.argmin(axis=1)
    # "最近邻"若指向同一个蛋白（同 TF 不同细胞系，cos≈1），换成次近的不同蛋白
    near_distinct = near.copy()
    for i in range(len(near)):
        if S[i, near[i]] > 0.9999:
            row = S_near[i].copy()
            for j in np.argsort(-row):
                if S[i, j] <= 0.9999:
                    near_distinct[i] = j
                    break
    near = near_distinct
    centroid = P.mean(dim=0)

    net = load_net(a.ckpt)
    rng = np.random.RandomState(a.seed)
    rand = np.array([(i + 1 + rng.randint(0, len(names) - 1)) % len(names)
                    for i in range(len(names))])

    rows = []
    for i, m in enumerate(meta):
        dsid = m['dsid']
        pos = read_fasta(Path(a.test_root) / f'{dsid}.pos.fasta')
        neg = read_fasta(Path(a.test_root) / f'{dsid}.neg.fasta')
        y = np.array([1] * len(pos) + [0] * len(neg))
        seqs = pos + neg
        res = dict(dsid=dsid, cell=m['cell'], symbol=m['symbol'])
        res['correct'] = float(roc_auc_score(y, score(net, seqs, P[i])))
        for tag, j in (('random', int(rand[i])), ('nearest', int(near[i])), ('farthest', int(far[i]))):
            assert j != i, (tag, i, j)
            res[tag] = float(roc_auc_score(y, score(net, seqs, P[j])))
            res[tag + '_from'] = names[j]
            res[tag + '_cos'] = float(S[i, j])
        res['centroid'] = float(roc_auc_score(y, score(net, seqs, centroid)))
        rows.append(res)
        print(f"[{i+1}/{len(meta)}] {dsid} correct {res['correct']:.4f} "
              f"random {res['random']:.4f} nearest {res['nearest']:.4f} "
              f"farthest {res['farthest']:.4f} centroid {res['centroid']:.4f}", flush=True)

    keys = ['correct', 'nearest', 'random', 'farthest', 'centroid']
    summary = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    out = dict(n=len(rows), ckpt=str(a.ckpt), seed=a.seed, summary=summary, per_dataset=rows)
    Path(a.out).write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print('\n== mean over 50 datasets ==')
    for k in keys:
        print(f'  {k:9s} {summary[k]:.4f}  (Δ vs correct {summary[k]-summary["correct"]:+.4f})')

    print('written', a.out)


if __name__ == '__main__':
    main()
