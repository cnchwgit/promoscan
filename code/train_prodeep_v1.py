#!/usr/bin/env python3
# coding: UTF-8
"""
proDeep-v1 训练 + 评估

目标：验证"蛋白 → Motif → 扫描"这条唯一通路能否在**留出 TF** 上带来 ΔAUC 增益。

训练：938 个"见过"的 TF 上做 per-TF 二分类（负样本=该 TF 的随机背景）
评估：A-vs-B 判别 ΔAUC，分别取「见过的 TF」与「留出的 TF」两个池
对照：蛋白打乱

用法：
  python3 train_prodeep_v1.py --n-tf 938 --samples-per-class 100 --epochs 6
"""
import argparse
import csv
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import Dataset, DataLoader

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE / 'protoTF'))
sys.path.append(os.path.expanduser('~/efficient_kan'))

from prodeep_v1 import ProMoScan, dna_onehot, revcomp_seq      # noqa: E402
from pwm_supervision import (load_db_pwms, s1_loss,                # noqa: E402
                             motif_set_pair_sim, pwm_all_to_all)
from protein_features import build_protein_features, esm_available   # noqa: E402
from data_utils import (build_vocab, seq_to_ids, load_fasta_balanced,   # noqa: E402
                        load_protein_fasta, encode_proteins)

BASE = Path(__file__).resolve().parents[1]
DATA_ROOT = str(BASE / 'raw_data')
CSV = str(CODE / 'protoTF/full_1043.csv')
PROT_FASTA = str(BASE / 'sequenceAnalysis/tf_sequences_final_v2.fasta')
ZS_SUM = Path(os.environ.get('PRODEEP_ZS_SUM',
                            '/home/chenw/.openclaw/workspace/multiTF/results/prototf_zs_random10/zero_shot_summary.json'))
ESM_PT = os.environ.get('PRODEEP_ESM_PT', str(BASE / 'data/distance/esm2_emb_650M.pt'))
VOCAB = build_vocab()
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# 单线程纪律（2026-09-24 用户指令）：PRODEEP_SINGLE_THREAD=1 → torch 线程=1，DataLoader workers=0
if os.environ.get('PRODEEP_SINGLE_THREAD', '1') not in ('0', 'false', 'False'):
    NUM_WORKERS = int(os.environ.get('PRODEEP_NUM_WORKERS', '0'))
    try:
        torch.set_num_threads(1)
    except Exception:
        pass
else:
    NUM_WORKERS = int(os.environ.get('PRODEEP_NUM_WORKERS', '4'))
PAD = 200


class PairDataset(Dataset):
    def __init__(self, samples):
        self.s = samples          # (seq_str, tf_idx, label)

    def __len__(self):
        return len(self.s)

    def __getitem__(self, i):
        return self.s[i]


def collate(batch, prot_oh, aug_rc=False):
    seqs = [b[0] for b in batch]
    if aug_rc:                                  # 反向互补增广（标签不变，等价于数据翻倍）
        seqs = [revcomp_seq(s) if random.random() < 0.5 else s for s in seqs]
    tf_idx = torch.LongTensor([b[1] for b in batch])
    labels = torch.FloatTensor([b[2] for b in batch])
    ids = torch.stack([torch.LongTensor(seq_to_ids(s, VOCAB)) for s in seqs])
    oh = torch.stack([dna_onehot(s, PAD) for s in seqs])
    aa = prot_oh[tf_idx]
    return ids, oh, aa, labels, tf_idx


def _rep(v, n):
    """把一个 TF 的特征重复 n 次（兼容 one-hot [21,L] 与 ESM 向量 [1280]）"""
    return v.unsqueeze(0).repeat(n, *([1] * v.dim()))


@torch.no_grad()
def score_delta(net, seqs_a, seqs_b, aaA, aaB, bs=128):
    """Δ(s) = P(A|s) - P(B|s)；aaA/aaB 为 [21,L] 单样本（CPU 或 GPU）

    ⚠️ BUG 修复（2026-09-18）：原写法把每批的 [A_batch, B_batch] 依次 concatenate，
    得到的 d 顺序是 A1,B1,A2,B2,...，而标签 y 是 [1]*n+[0]*n（全部 A 再全部 B）。
    只要 n > bs（典型 n=400, bs=128）就会**错位**，ΔAUC 从真实的 ~0.85 塔到 ~0.59。
    现在分开收集 A/B 两段再拼接，保证与 y 对齐。
    """
    aaA = aaA.to(DEVICE)
    aaB = aaB.to(DEVICE)
    outA, outB = [], []
    n = len(seqs_a)
    for i in range(0, n, bs):
        sa = seqs_a[i:i + bs]
        sb = seqs_b[i:i + bs]
        k = len(sa)
        ids = torch.stack([torch.LongTensor(seq_to_ids(s, VOCAB)) for s in sa + sb]).to(DEVICE)
        oh = torch.stack([dna_onehot(s, PAD) for s in sa + sb]).to(DEVICE)
        aa_A = _rep(aaA, 2 * k)
        aa_B = _rep(aaB, 2 * k)
        pA = torch.sigmoid(net(ids, oh, aa_A)).cpu().numpy()
        pB = torch.sigmoid(net(ids, oh, aa_B)).cpu().numpy()
        outA.append(pA[:k] - pB[:k])      # 本批里属于 A 的序列
        outB.append(pA[k:] - pB[k:])      # 本批里属于 B 的序列
    return np.concatenate(outA + outB)


@torch.no_grad()
def score_abs(net, seqs, aa, bs=128):
    aa = aa.to(DEVICE)
    out = []
    for i in range(0, len(seqs), bs):
        chunk = seqs[i:i + bs]
        k = len(chunk)
        ids = torch.stack([torch.LongTensor(seq_to_ids(s, VOCAB)) for s in chunk]).to(DEVICE)
        oh = torch.stack([dna_onehot(s, PAD) for s in chunk]).to(DEVICE)
        aaB = _rep(aa, k)
        out.append(torch.sigmoid(net(ids, oh, aaB)).cpu().numpy())
    return np.concatenate(out)


def eval_pool(net, prot_oh, names, tf2gsm, pairs, tag, rng):
    res = []
    for (ia, ib) in pairs:
        ta, tb = names[ia], names[ib]
        pos, _ = load_fasta_balanced(tf2gsm[ta], DATA_ROOT, 1400)
        pb, _ = load_fasta_balanced(tf2gsm[tb], DATA_ROOT, 1400)
        if len(pos) < 800 or len(pb) < 800:
            continue
        sa, sb = pos[400:800], pb[400:800]
        n = min(len(sa), len(sb))
        sa, sb = sa[:n], sb[:n]
        y = np.array([1] * n + [0] * n)
        d = score_delta(net, sa, sb, prot_oh[ia], prot_oh[ib])
        j = int(rng.randint(len(names)))
        d_sh = score_delta(net, sa, sb, prot_oh[j], prot_oh[ib])
        res.append((float(roc_auc_score(y, d)), float(roc_auc_score(y, d_sh))))
    if not res:
        return dict(pool=tag, n=0)
    a = np.array(res)
    return dict(pool=tag, n=len(res),
                delta_auc=float(a[:, 0].mean()),
                delta_auc_shuffled=float(a[:, 1].mean()),
                protein_contribution=float((a[:, 0] - a[:, 1]).mean()))


class ABPool:
    """A/B 辅助损失的数据池。

    教训：如果从当前 batch 里找 A/B 正样本，因 938 个 TF × 600 样本而 batch=256，
    每个 TF 在 batch 内平均只出现 ~0.27 次 → 根本抽不到 ≥4 条正样本，损失**永远是 0**。
    所以改为从全局正样本里直接抽，并优先选**蛋白空间邻近**的 TF 对（模拟评估里的难例）。
    """

    def __init__(self, pos_by_tf, esm_by_idx, seed=0, n_cand=8):
        self.pos = pos_by_tf
        self.rng = np.random.RandomState(seed + 7)
        self.n_cand = n_cand
        self.tfs = sorted(i for i, v in pos_by_tf.items() if len(v) >= 4)
        self.D = None
        if esm_by_idx:
            keys = [i for i in self.tfs if i in esm_by_idx]
            if len(keys) >= 2:
                M = np.stack([esm_by_idx[i] for i in keys])
                M = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
                D = 1.0 - M @ M.T
                np.fill_diagonal(D, 1.0)
                self.D = D
                self.tfs = keys

    def sample_pair(self):
        n = len(self.tfs)
        if n < 2:
            return None
        ai = int(self.rng.randint(n))
        if self.D is not None:
            cand = self.rng.choice(n, size=min(self.n_cand, n - 1), replace=False)
            cand = np.array([c for c in cand if c != ai])
            if len(cand) == 0:
                return None
            bi = int(cand[int(np.argmin(self.D[ai, cand]))])
        else:
            bi = (ai + 1 + int(self.rng.randint(n - 1))) % n
        return self.tfs[ai], self.tfs[bi]

    def draw(self, tf, n):
        v = self.pos[tf]
        return [v[k] for k in self.rng.randint(len(v), size=n)]


def sample_pairs(names, D, n_pairs, seed, strata=10):
    n = len(names)
    ii, jj = np.triu_indices(n, k=1)
    dv = D[ii, jj]
    edges = np.quantile(dv, np.linspace(0, 1, strata + 1)); edges[-1] += 1e-9
    rng = np.random.RandomState(seed)
    per = max(1, n_pairs // strata)
    out = []
    for s in range(strata):
        idx = np.where((dv >= edges[s]) & (dv < edges[s + 1]))[0]
        if len(idx):
            out.extend(rng.choice(idx, size=min(per, len(idx)), replace=False).tolist())
    rng.shuffle(out)
    return [(int(ii[k]), int(jj[k])) for k in out]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-tf', type=int, default=938)
    ap.add_argument('--samples-per-class', type=int, default=100)
    ap.add_argument('--epochs', type=int, default=6)
    ap.add_argument('--batch-size', type=int, default=256)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--M', type=int, default=8)
    ap.add_argument('--W', type=int, default=12)
    ap.add_argument('--use-dna-context', action='store_true',
                    help='打开 DNA 塔残差项（默认关闭：唯一通路=蛋白→motif）')
    ap.add_argument('--eval-pairs', type=int, default=120)
    ap.add_argument('--lam-pwm', type=float, default=0.0,
                    help='S1: protein→PWM 直接监督权重（0=关闭）')
    ap.add_argument('--s1-seen-only', action='store_true',
                    help='S1 只用**见过 TF** 的 DB motif（严格零样本；默认会把留出 TF 的 motif 也喂进去 = 泄漏）')
    ap.add_argument('--aux-batch', type=int, default=64)
    ap.add_argument('--alpha-temp', type=float, default=1.0,
                    help='α 门控温度（<1 锐化，向"单 motif 主导"靠）')
    ap.add_argument('--rc-symmetric', action='store_true',
                    help='双链扫描：正向与反向互补取 max（生物学上 motif 应双链对称）')
    ap.add_argument('--pool', default='max', choices=['max', 'lse'],
                    help='位置池化：max（旧）或 lse（logsumexp 软池化）')
    ap.add_argument('--pool-temp', type=float, default=1.0, help='lse 池化温度')
    ap.add_argument('--aug-rc', action='store_true', help='训练时反向互补增广（p=0.5）')
    ap.add_argument('--smooth-lam', type=float, default=0.0,
                    help='相似度蒸馏损失权重：让预测 motif 的相似度结构对齐 DB motif 的相似度结构（目标=零样本泛化）')
    ap.add_argument('--smooth-pairs', type=int, default=32, help='每步采样的 TF 对数')
    ap.add_argument('--lam-ab', type=float, default=0.0,
                    help='A-vs-B 成对 ranking 辅助损失权重（0=关闭）')
    ap.add_argument('--ab-n', type=int, default=32, help='每个 TF 参与 A/B 辅助损失的序列数')
    ap.add_argument('--protein-enc', default='cnn', choices=['cnn', 'esm'],
                    help='蛋白编码：cnn=21dim one-hot+Conv塔（旧）；esm=冻结 ESM-2 650M 嵌入+投影')
    ap.add_argument('--prot-drop', type=float, default=0.2, help='ESM 蛋白塔 dropout')
    ap.add_argument('--motif-head', default='std', choices=['std', 'linear', 'wide', 'free'],
                    help='身份接口容量（E9 容量对照）：std=256 隐层；linear=单层；wide=512+512')
    ap.add_argument('--init-motif-pretrain', default='',
                    help='用 motif-only 预训练权重初始化（蛋白塔 + motif 头，strict=False）')
    ap.add_argument('--out-dir', default=str(BASE / 'results/prodeep_v1'))
    ap.add_argument('--save-every', type=int, default=1,
                    help='每 N 个 epoch 存一次 last.pt（防意外中断丢进度）')
    ap.add_argument('--resume', action='store_true',
                    help='从 out-dir/last.pt 断点续训')
    ap.add_argument('--data-root', default=DATA_ROOT,
                    help='pos/neg fasta 目录（默认 raw_data=难负样本；路线 A 用 data/bgroot=随机背景）')
    ap.add_argument('--seed', type=int, default=42)
    a = ap.parse_args()
    out_dir = Path(a.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)

    zs = json.load(open(ZS_SUM))
    holdout = set(zs['holdout_tfs'])
    rows = list(csv.DictReader(open(CSV)))
    tf2gsm = {r['TF_Name']: r['GSM_ID'] for r in rows}
    prot = load_protein_fasta(PROT_FASTA)

    seen_all = [t for t in tf2gsm if t not in holdout and t in prot]
    hold_all = [t for t in holdout if t in tf2gsm and t in prot]
    rng = random.Random(a.seed); rng.shuffle(seen_all)
    seen = sorted(seen_all[:a.n_tf])
    hold = sorted(hold_all)
    print(f'训练 TF: {len(seen)} | 留出 TF: {len(hold)} | 设备 {DEVICE}', flush=True)

    prot_oh = build_protein_features(seen + hold, enc=a.protein_enc)   # 留在 CPU，供 DataLoader worker 索引
    if a.protein_enc == 'esm':
        print(f'ESM 蛋白特征: {tuple(prot_oh.shape)} (可用 {len(esm_available(seen + hold))}/{len(seen + hold)})', flush=True)
    idx = {t: i for i, t in enumerate(seen + hold)}

    # ---------- 训练数据 ----------
    samples = []
    for t in seen:
        pos, neg = load_fasta_balanced(tf2gsm[t], a.data_root, a.samples_per_class)
        nm = min(len(pos), len(neg))
        for s in pos[:nm]:
            samples.append((s, idx[t], 1))
        for s in neg[:nm]:
            samples.append((s, idx[t], 0))
    print(f'训练样本: {len(samples)}', flush=True)
    ds = PairDataset(samples)
    dl = DataLoader(ds, batch_size=a.batch_size, shuffle=True, num_workers=NUM_WORKERS,
                    pin_memory=(NUM_WORKERS > 0 and DEVICE == 'cuda'),
                    collate_fn=lambda b: collate(b, prot_oh, a.aug_rc))

    net = ProMoScan(len(VOCAB), M=a.M, W=a.W,
                       use_dna_context=a.use_dna_context,
                       alpha_temp=a.alpha_temp,
                       rc_symmetric=a.rc_symmetric, pool=a.pool,
                       pool_temp=a.pool_temp,
                       protein_enc=a.protein_enc, prot_drop=a.prot_drop,
                       motif_head=a.motif_head).to(DEVICE)
    n_par = sum(p.numel() for p in net.parameters())
    print(f'模型参数: {n_par/1e6:.3f}M | M={a.M} W={a.W} dna_context={a.use_dna_context} '
          f'rc={a.rc_symmetric} pool={a.pool} prot={a.protein_enc} head={a.motif_head}', flush=True)

    if a.init_motif_pretrain:
        pre = torch.load(a.init_motif_pretrain, map_location='cpu', weights_only=False)
        sd = pre['model']
        miss = net.load_state_dict(sd, strict=False)
        n_ok = len(set(net.state_dict()) & set(sd))
        print(f'motif-only 预训练初始化：载入 {n_ok}/{len(sd)} 个张量；'
              f'meta={pre.get("meta")}', flush=True)
        if miss.unexpected_keys:
            print(f'  未使用键 {len(miss.unexpected_keys)} 个', flush=True)

    # ---------- S1: protein → PWM 直接监督 ----------
    db_tensor = None
    if a.lam_pwm > 0 or a.smooth_lam > 0:
        db = load_db_pwms(target_width=a.W)
        names_all = seen + hold
        hit = [t for t in names_all if t in db]
        n_leak = sum(1 for t in hit if t not in set(seen))
        if a.s1_seen_only:
            hit = [t for t in hit if t in set(seen)]
        db_tensor = torch.full((len(names_all), 4, a.W), 0.25)
        for t in hit:
            db_tensor[names_all.index(t)] = db[t]
        db_idx = torch.LongTensor([names_all.index(t) for t in hit])
        print(f'S1 监督可用 TF: {len(hit)}/{len(names_all)} (λ={a.lam_pwm}'
              f"{' [seen-only]' if a.s1_seen_only else ''}); "
              f'其中留出 TF 带 DB motif = {0 if a.s1_seen_only else n_leak}', flush=True)
    else:
        db_idx = None

    # ---------- A/B 辅助损失数据池 ----------
    ab_pool = None
    if a.lam_ab > 0:
        pos_by_tf = {}
        for s_, ti_, yv_ in samples:
            if yv_ == 1:
                pos_by_tf.setdefault(ti_, []).append(s_)
        try:
            _embs = torch.load(ESM_PT, map_location='cpu', weights_only=False)
        except Exception as e_:                                     # noqa: BLE001
            print('ESM 载入失败，A/B 改用随机 TF 对:', e_, flush=True)
            _embs = None
        esm_by_idx = {}
        if _embs is not None:
            for t_, ti_ in idx.items():
                if t_ in _embs:
                    esm_by_idx[ti_] = torch.as_tensor(_embs[t_], dtype=torch.float32).numpy()
        ab_pool = ABPool(pos_by_tf, esm_by_idx, seed=a.seed)
        print(f'A/B 池: {len(ab_pool.tfs)} 个 TF | 蛋白质配对={"ESM 距离(难例优先)" if ab_pool.D is not None else "随机"}',
              flush=True)

    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    lossf = nn.BCEWithLogitsLoss()

    start_ep = 0
    ck_last = out_dir / 'last.pt'
    if a.resume and ck_last.exists():
        ck = torch.load(ck_last, map_location=DEVICE, weights_only=False)
        net.load_state_dict(ck['model'])
        if 'opt' in ck:
            opt.load_state_dict(ck['opt'])
        start_ep = int(ck.get('epoch', 0)) + 1
        print(f'断点续训：从 epoch {start_ep} 继续（{ck_last}）', flush=True)

    for ep in range(start_ep, a.epochs):
        net.train(); t0 = time.time(); tot = 0.0; nbt = 0.0; nab = 0.0; dab = 0.0; nb = 0; nsm = 0.0
        for ids, oh, aa, y, tfidx in dl:
            ids, oh, aa, y = ids.to(DEVICE), oh.to(DEVICE), aa.to(DEVICE), y.to(DEVICE)
            tfidx = tfidx.to(DEVICE)
            opt.zero_grad()
            logit = net(ids, oh, aa)
            loss = lossf(logit, y)
            if db_tensor is not None and a.lam_pwm > 0:
                sel = db_idx[torch.randint(len(db_idx), (min(a.aux_batch, len(db_idx)),))]
                aaux = prot_oh[sel].to(DEVICE)
                l_pwm = s1_loss(net.motif_probs(aaux), db_tensor[sel].to(DEVICE))
                loss = loss + a.lam_pwm * l_pwm
                nbt += float(l_pwm)
            # 相似度蒸馏：预测 motif 的相似度结构 ↔ DB motif 的相似度结构
            # （直接刻在蛋白→motif 映射上的平滑先验；零样本泛化的关键归纳偏置）
            if db_tensor is not None and a.smooth_lam > 0:
                k = min(a.smooth_pairs, len(db_idx))
                si = db_idx[torch.randint(len(db_idx), (k,))]
                sj = db_idx[torch.randint(len(db_idx), (k,))]
                Pi = net.motif_probs(prot_oh[si].to(DEVICE))
                Pj = net.motif_probs(prot_oh[sj].to(DEVICE))
                s_hat = motif_set_pair_sim(Pi, Pj)
                with torch.no_grad():
                    s_true = torch.diagonal(pwm_all_to_all(db_tensor[si].to(DEVICE),
                                                           db_tensor[sj].to(DEVICE)))
                l_sm = ((s_hat - s_true) ** 2).mean()
                loss = loss + a.smooth_lam * l_sm
                nsm += float(l_sm)
            # A/B 成对 ranking 辅助损失：score(s,prot_A) > score(s,prot_B)
            # 关键：**必须在同一个 forward 里混入两种蛋白**。ProteinTower 里有 BatchNorm1d（protoTF 的设计），
            # 若整个 batch 只有一种蛋白（同一蛋白 repeat），train 模式下 BN 批内方差=0
            #   → 蛋白嵌入塔成 0 → logit_A ≡ logit_B → d≡0 → 损失恒为 ln2 且梯度恒为 0。
            if ab_pool is not None:
                pr = ab_pool.sample_pair()
                if pr is not None:
                    ta, tb = pr
                    n_ab = a.ab_n
                    sa = ab_pool.draw(ta, n_ab)
                    sb = ab_pool.draw(tb, n_ab)
                    seqs4 = sa + sb + sa + sb                       # A|B|A(配B蛋白)|B(配A蛋白)
                    pid4 = [ta] * n_ab + [tb] * n_ab + [tb] * n_ab + [ta] * n_ab
                    ids4 = torch.stack([torch.LongTensor(seq_to_ids(q, VOCAB)) for q in seqs4]).to(DEVICE)
                    oh4 = torch.stack([dna_onehot(q, PAD) for q in seqs4]).to(DEVICE)
                    aa4 = prot_oh[torch.LongTensor(pid4)].to(DEVICE)
                    o = net(ids4, oh4, aa4)
                    o1 = o[:n_ab]; o2 = o[n_ab:2 * n_ab]; o3 = o[2 * n_ab:3 * n_ab]; o4_ = o[3 * n_ab:]
                    d_a = o1 - o3                                   # A 的序列：蛋白 A vs 蛋白 B
                    d_b = o2 - o4_                                  # B 的序列：蛋白 B vs 蛋白 A
                    l_ab = 0.5 * (F.softplus(-d_a).mean() + F.softplus(-d_b).mean())
                    loss = loss + a.lam_ab * l_ab
                    nab += float(l_ab)
                    dab += float(0.5 * (d_a.detach().abs().mean() + d_b.detach().abs().mean()))
            loss.backward(); opt.step()
            tot += float(loss); nb += 1
        extra = ''
        if db_tensor is not None:
            extra += f' pwm={nbt/max(1,nb):.4f}'
        if a.smooth_lam > 0:
            extra += f' smooth={nsm/max(1,nb):.4f}'
        if a.lam_ab > 0:
            extra += f' ab={nab/max(1,nb):.4f} d={dab/max(1,nb):.4f}'
        print(f'  ep {ep} loss={tot/max(1,nb):.4f}{extra} ({time.time()-t0:.0f}s)', flush=True)
        if a.save_every and (ep + 1) % a.save_every == 0:
            torch.save({'model': net.state_dict(), 'config': vars(a),
                        'opt': opt.state_dict(), 'epoch': ep}, ck_last)

    torch.save({'model': net.state_dict(),
                'config': vars(a)}, out_dir / 'best_model.pt')

    # ---------- 评估 ----------
    embs = torch.load(ESM_PT, map_location='cpu', weights_only=False)
    all_names = seen + hold
    idxs = [i for i, t in enumerate(all_names) if t in embs]
    Mv = torch.stack([torch.as_tensor(embs[all_names[i]], dtype=torch.float32) for i in idxs]).numpy()
    Mv = Mv / (np.linalg.norm(Mv, axis=1, keepdims=True) + 1e-9)
    Dm = 1.0 - Mv @ Mv.T
    np.fill_diagonal(Dm, 1.0)
    seenset = {t: i for i, t in enumerate(seen)}
    holdset = {t: i for i, t in enumerate(hold)}
    seen_names = [t for t in seen if t in embs]
    hold_names = [t for t in hold if t in embs]
    # 用 idxs 内的局部索引构造配对，再映射回 all_names 索引
    local = {t: k for k, t in enumerate([all_names[i] for i in idxs])}
    p_seen = sample_pairs(seen_names, Dm[[local[t] for t in seen_names]][:, [local[t] for t in seen_names]],
                          a.eval_pairs, a.seed)
    p_hold = sample_pairs(hold_names, Dm[[local[t] for t in hold_names]][:, [local[t] for t in hold_names]],
                          a.eval_pairs, a.seed)
    rng2 = np.random.RandomState(a.seed)
    net.eval()
    out = {}
    for tag, namelist, prs in [('seen', seen_names, p_seen), ('holdout', hold_names, p_hold)]:
        mapped = [(all_names.index(namelist[i]), all_names.index(namelist[j])) for i, j in prs]
        out[tag] = eval_pool(net, prot_oh, all_names, tf2gsm, mapped, tag, rng2)
        print(f'  {tag}: {out[tag]}', flush=True)

    (out_dir / 'eval.json').write_text(json.dumps(
        {'config': vars(a), 'n_params': n_par, 'results': out}, indent=2, ensure_ascii=False))
    print('\n===== 结果 =====')
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
