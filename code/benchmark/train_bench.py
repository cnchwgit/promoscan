#!/usr/bin/env python3
# coding: UTF-8
"""D1 基准：在 CBR-KAN 的 50-数据集基准上训练**一个**多 TF ProMoScan。
复用 train_prodeep_v1 的组件（ProMoScan / PairDataset / collate / ABPool）。

数据：code/benchmark/prep_bench.py 产出的 data/bench50/train + bench50*.{csv,pt,fasta,json}
"""
import argparse
import csv
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE / 'protoTF'))
import train_prodeep_v1 as T          # noqa: E402  触发 sys.path（code/protoTF, efficient_kan, VOCAB）
from prodeep_v1 import ProMoScan      # noqa: E402
from data_utils import build_vocab, seq_to_ids, load_fasta_balanced  # noqa: E402
from prodeep_v1 import dna_onehot     # noqa: E402
from pwm_supervision import load_db_pwms, s1_loss  # noqa: E402

BASE = Path(__file__).resolve().parents[2]
BENCH = BASE / 'code/benchmark'
VOCAB = build_vocab()
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
PAD = 200


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default=str(BENCH / 'bench50.csv'))
    ap.add_argument('--esm-pt', default=str(BENCH / 'bench50_esm.pt'))
    ap.add_argument('--alias', default=str(BENCH / 'bench50_alias.json'))
    ap.add_argument('--train-root', default=str(BASE / 'data/bench50/train'))
    ap.add_argument('--out-dir', default=str(BASE / 'results/prodeep_v1_BENCH50'))
    ap.add_argument('--M', type=int, default=16)
    ap.add_argument('--W', type=int, default=16)
    ap.add_argument('--epochs', type=int, default=16)
    ap.add_argument('--batch-size', type=int, default=256)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--samples-per-class', type=int, default=0, help='0=全量')
    ap.add_argument('--lam-pwm', type=float, default=0.2)
    ap.add_argument('--lam-ab', type=float, default=1.0)
    ap.add_argument('--ab-n', type=int, default=32)
    ap.add_argument('--alpha-temp', type=float, default=1.0)
    ap.add_argument('--motif-head', default='std', choices=['std', 'linear', 'wide', 'free'])
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--val-frac', type=float, default=0.0,
                    help='>0 时从训练集切出该比例的验证集（按 TF×标签分层），并据此选最优 epoch')
    ap.add_argument('--val-max', type=int, default=8000, help='验证集最多使用的序列数（分层抽样）')
    ap.add_argument('--save-every', type=int, default=1)
    ap.add_argument('--resume', action='store_true')
    a = ap.parse_args()
    out_dir = Path(a.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)

    rows = list(csv.DictReader(open(a.csv)))
    names = [r['TF_Name'] for r in rows]
    assert len(names) == len(set(names)) == 50, f'数据集数异常 {len(names)}'
    idx = {t: i for i, t in enumerate(names)}
    print(f'基准数据集: {len(names)} | 设备 {DEVICE}', flush=True)

    e = torch.load(a.esm_pt, map_location='cpu', weights_only=False)
    miss = [n for n in names if n not in e]
    assert not miss, f'缺 ESM: {miss}'
    prot_oh = torch.stack([torch.as_tensor(e[n], dtype=torch.float32) for n in names])
    print(f'ESM 蛋白特征: {tuple(prot_oh.shape)}', flush=True)

    # ---------- 训练数据 ----------
    samples = []
    per_tf = {}
    for t in names:
        pos, neg = load_fasta_balanced(t, a.train_root, a.samples_per_class or 10**9)
        for s in pos:
            samples.append((s, idx[t], 1))
        for s in neg:
            samples.append((s, idx[t], 0))
        per_tf[t] = len(pos) + len(neg)
    print(f'训练样本: {len(samples)}（{len(per_tf)} 数据集，最大 {max(per_tf.values())}）', flush=True)

    ds = T.PairDataset(samples)
    dl = DataLoader(ds, batch_size=a.batch_size, shuffle=True, num_workers=4,
                    pin_memory=True, collate_fn=lambda b: T.collate(b, prot_oh, False))

    # ---------- 可选：按 TF×标签分层的验证集（与逐 TF 基线同口径的模型选择） ----------
    val_pack = None
    if a.val_frac and a.val_frac > 0:
        rng = np.random.RandomState(a.seed)
        by_key = {}
        for s_, ti_, yv_ in samples:
            by_key.setdefault((ti_, yv_), []).append(s_)
        keep = []
        for key, lst in by_key.items():
            k = int(round(len(lst) * a.val_frac))
            if k:
                pick = rng.choice(len(lst), size=k, replace=False)
                keep.extend((lst[j], key[0], key[1]) for j in pick)
                for j in sorted(pick, reverse=True):
                    lst.pop(j)
        # 重新组装训练集
        samples = [(s_, ti_, yv_) for key, lst in by_key.items() for (s_, ti_, yv_) in [(q, key[0], key[1]) for q in lst]]
        ds = T.PairDataset(samples)
        dl = DataLoader(ds, batch_size=a.batch_size, shuffle=True, num_workers=4,
                        pin_memory=True, collate_fn=lambda b: T.collate(b, prot_oh, False))
        # 验证子集：分层截断到 val-max
        if len(keep) > a.val_max:
            sel = rng.choice(len(keep), size=a.val_max, replace=False)
            keep = [keep[j] for j in sel]
        v_seqs = [q[0] for q in keep]
        v_tf = torch.LongTensor([q[1] for q in keep])
        v_y = torch.LongTensor([q[2] for q in keep])
        v_ids = torch.stack([torch.LongTensor(seq_to_ids(q, VOCAB)) for q in v_seqs])
        v_oh = torch.stack([dna_onehot(q, PAD) for q in v_seqs])
        val_pack = (v_ids, v_oh, v_tf, v_y)
        print(f'验证集: {len(v_seqs)} 条（{a.val_frac:.0%} 分层），训练集剩 {len(samples)}', flush=True)

    def eval_val(net):
        """逐数据集 ROC-AUC 的平均（与基线选点口径一致）"""
        from sklearn.metrics import roc_auc_score
        v_ids, v_oh, v_tf, v_y = val_pack
        net.eval(); preds = []
        with torch.no_grad():
            for i in range(0, len(v_y), 512):
                sl = slice(i, i + 512)
                aa = prot_oh[v_tf[sl]].to(DEVICE)
                preds.append(torch.sigmoid(net(v_ids[sl].to(DEVICE), v_oh[sl].to(DEVICE), aa)).cpu())
        p = torch.cat(preds).numpy(); y = v_y.numpy(); t = v_tf.numpy()
        aucs = [roc_auc_score(y[t == k], p[t == k]) for k in np.unique(t) if len(np.unique(y[t == k])) == 2]
        net.train()
        return float(np.mean(aucs))


    net = ProMoScan(len(VOCAB), M=a.M, W=a.W, use_dna_context=True,
                    alpha_temp=a.alpha_temp, pool='max',
                    protein_enc='esm', prot_drop=0.2, motif_head=a.motif_head).to(DEVICE)
    n_par = sum(p.numel() for p in net.parameters())
    print(f'模型参数: {n_par/1e6:.3f}M | M={a.M} W={a.W} head={a.motif_head}', flush=True)

    # ---------- S1: protein → PWM（把 DSID 映射到 DB symbol）----------
    db_tensor = None
    if a.lam_pwm > 0:
        alias = json.load(open(a.alias))
        db = load_db_pwms(target_width=a.W)
        db_tensor = torch.full((len(names), 4, a.W), 0.25)
        hit = []
        for t in names:
            sym = alias.get(t)
            if sym and sym in db:
                db_tensor[idx[t]] = db[sym]; hit.append(t)
        db_idx = torch.LongTensor([idx[t] for t in hit])
        print(f'S1 监督可用: {len(hit)}/{len(names)} (λ={a.lam_pwm})', flush=True)

    # ---------- A/B ranking 池 ----------
    ab_pool = None
    if a.lam_ab > 0:
        pos_by_tf = {}
        for s_, ti_, yv_ in samples:
            if yv_ == 1:
                pos_by_tf.setdefault(ti_, []).append(s_)
        esm_by_idx = {i: prot_oh[i].numpy() for i in range(len(names))}
        ab_pool = T.ABPool(pos_by_tf, esm_by_idx, seed=a.seed)
        print(f'A/B 池: {len(ab_pool.tfs)} 个数据集', flush=True)

    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    lossf = nn.BCEWithLogitsLoss()

    start_ep = 0
    best_val = -1.0
    ck_last = out_dir / 'last.pt'
    if a.resume and ck_last.exists():
        ck = torch.load(ck_last, map_location=DEVICE, weights_only=False)
        net.load_state_dict(ck['model']); opt.load_state_dict(ck['opt'])
        start_ep = int(ck.get('epoch', 0)) + 1
        print(f'断点续训：从 epoch {start_ep} 继续', flush=True)

    for ep in range(start_ep, a.epochs):
        net.train(); t0 = time.time(); tot = nb = 0.0; nbt = nab = dab = 0.0
        for ids, oh, aa, y, tfidx in dl:
            ids, oh, aa, y = ids.to(DEVICE), oh.to(DEVICE), aa.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            logit = net(ids, oh, aa)
            loss = lossf(logit, y)
            if db_tensor is not None:
                sel = db_idx[torch.randint(len(db_idx), (min(64, len(db_idx)),))]
                l_pwm = s1_loss(net.motif_probs(prot_oh[sel].to(DEVICE)), db_tensor[sel].to(DEVICE))
                loss = loss + a.lam_pwm * l_pwm; nbt += float(l_pwm)
            if ab_pool is not None:
                pr = ab_pool.sample_pair()
                if pr is not None:
                    ta, tb = pr; n_ab = a.ab_n
                    sa = ab_pool.draw(ta, n_ab); sb = ab_pool.draw(tb, n_ab)
                    seqs4 = sa + sb + sa + sb
                    pid4 = [ta]*n_ab + [tb]*n_ab + [tb]*n_ab + [ta]*n_ab
                    ids4 = torch.stack([torch.LongTensor(seq_to_ids(q, VOCAB)) for q in seqs4]).to(DEVICE)
                    oh4 = torch.stack([dna_onehot(q, PAD) for q in seqs4]).to(DEVICE)
                    o = net(ids4, oh4, prot_oh[torch.LongTensor(pid4)].to(DEVICE))
                    o1, o2, o3, o4_ = o[:n_ab], o[n_ab:2*n_ab], o[2*n_ab:3*n_ab], o[3*n_ab:]
                    l_ab = 0.5 * (F.softplus(-(o1-o3)).mean() + F.softplus(-(o2-o4_)).mean())
                    loss = loss + a.lam_ab * l_ab; nab += float(l_ab)
                    dab += float(0.5 * ((o1-o3).detach().abs().mean() + (o2-o4_).detach().abs().mean()))
            loss.backward(); opt.step(); tot += float(loss); nb += 1
        print(f'  ep {ep} loss={tot/max(1,nb):.4f} pwm={nbt/max(1,nb):.4f} '
              f'ab={nab/max(1,nb):.4f} d={dab/max(1,nb):.4f} ({time.time()-t0:.0f}s)', flush=True)
        if val_pack is not None:
            v = eval_val(net)
            print(f'     val mean per-dataset AUC = {v:.4f}', flush=True)
            if v > best_val:
                best_val = v
                torch.save({'model': net.state_dict(), 'config': vars(a), 'epoch': ep,
                            'val_auc': v}, out_dir / 'best_model.pt')
                print(f'     ↑ 新最优（ep {ep}）', flush=True)
        if a.save_every and (ep + 1) % a.save_every == 0:
            torch.save({'model': net.state_dict(), 'config': vars(a), 'opt': opt.state_dict(),
                        'epoch': ep}, ck_last)
    if val_pack is None:
        torch.save({'model': net.state_dict(), 'config': vars(a)}, out_dir / 'best_model.pt')
    print('训练完成 ->', out_dir / 'best_model.pt', flush=True)


if __name__ == '__main__':
    main()
