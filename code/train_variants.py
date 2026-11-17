#!/usr/bin/env python3
# coding: UTF-8
"""
ProtoTF-Net 归因对照变体训练：
  single_head : E 变体 —— 单共享头（无混合、无蛋白），"所有TF一起跑"同配置基线
  lookup_route: B 变体 —— 每 TF 可学习嵌入路由（无蛋白序列）→ 软混合 K 头，路由信息上界
  single_tf   : 单任务 —— 每个 TF 独立训练 E 模型（同骨干），per-TF 天花板

所有变体与 ProtoTF 共用: CBRKANSharedBackbone、数据管线(400/class 平衡)、60/20/20 分层划分、
seed=42、评估协议。single_head 可选 --eval-holdout 做零样本评估（复用 zero-shot 留出列表）。
"""
import argparse
import csv
import json
import logging
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
BASE = Path(os.path.dirname(os.path.abspath(__file__))).resolve().parents[1]
from data_utils import (ProtoTFDataset, build_tf_list, build_vocab,
                        load_fasta_balanced, load_protein_fasta,
                        split_samples_per_tf)
from prototf_net import CBRKANSharedBackbone, KAN
from zero_shot_prototf import load_family_map, collate_no_aa


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_logging(out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = out_dir / f"variant_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s [%(levelname)s] %(message)s',
                        handlers=[logging.FileHandler(log_file, encoding='utf-8'),
                                  logging.StreamHandler(sys.stdout)])
    return log_file


# ==================== 变体模型 ====================
class SingleHeadModel(nn.Module):
    """E: 共享骨干 + 单 KAN 头（无路由）"""
    def __init__(self, len_vocab, kan_hidden=32):
        super().__init__()
        self.backbone = CBRKANSharedBackbone(len_vocab)
        self.head = KAN([256, kan_hidden, 2])

    def forward(self, tokens):
        return self.head(self.backbone(tokens))


class PerTFHeadModel(nn.Module):
    """旧 C001 式：共享骨干 + 每 TF 专属 KAN 头（分类器不池化）"""
    def __init__(self, len_vocab, n_tf, kan_hidden=32):
        super().__init__()
        self.backbone = CBRKANSharedBackbone(len_vocab)
        self.heads = nn.ModuleList([KAN([256, kan_hidden, 2]) for _ in range(n_tf)])

    def forward(self, tokens, tf_idx):
        h = self.backbone(tokens)                      # [B,256]
        B = tokens.size(0)
        logits = torch.zeros(B, 2, device=tokens.device)
        for ti in torch.unique(tf_idx):
            m = tf_idx == ti
            logits[m] = self.heads[ti.item()](h[m])
        return logits


class LookupRouteModel(nn.Module):
    """B: 每 TF 可学习嵌入 -> MLP门控 -> K 头软混合（无蛋白序列）"""
    def __init__(self, len_vocab, n_tf, n_prototypes=167, prot_dim=128,
                 kan_hidden=32, tau=1.0, gumbel=0.5):
        super().__init__()
        self.backbone = CBRKANSharedBackbone(len_vocab)
        self.tf_embed = nn.Embedding(n_tf, prot_dim)
        self.prototypes = nn.Parameter(torch.randn(n_prototypes, prot_dim) * 0.1)
        self.gate_mlp = nn.Sequential(
            nn.Linear(prot_dim, 128), nn.ReLU(True), nn.Linear(128, n_prototypes))
        self.heads = nn.ModuleList([KAN([256, kan_hidden, 2])
                                    for _ in range(n_prototypes)])
        self.register_buffer('tau', torch.tensor(tau, dtype=torch.float32))
        self.register_buffer('gumbel', torch.tensor(gumbel, dtype=torch.float32))
        self.n_prototypes = n_prototypes

    def gate(self, e, tau=None, topk=None):
        t = self.tau if tau is None else tau
        sim = self.gate_mlp(e) / t
        if self.training and self.gumbel > 0:
            u = torch.rand_like(sim).clamp_min(1e-9)
            sim = sim + self.gumbel * (-(-u.log()).log())
        g = torch.softmax(sim, dim=-1)
        if topk is not None and topk < self.n_prototypes:
            kth = torch.topk(g, topk, dim=-1).values[:, -1:]
            g = g * (g >= kth).float()
            g = g / (g.sum(-1, keepdim=True) + 1e-8)
        return g

    def forward(self, tokens, tf_idx, topk=None, return_gate=False):
        h = self.backbone(tokens)
        e = self.tf_embed(tf_idx)
        g = self.gate(e, topk=topk)
        outs = torch.stack([hd(h) for hd in self.heads], 1)
        logits = (g.unsqueeze(-1) * outs).sum(1)
        if return_gate:
            return logits, g, e
        return logits

    def commitment_loss(self, e, g):
        diff = e.unsqueeze(1) - self.prototypes.unsqueeze(0)
        return (g * diff.pow(2).sum(-1)).sum() / e.size(0)

    def balance_loss(self, g):
        f = torch.zeros_like(g)
        idx = g.argmax(-1)
        f.scatter_(1, idx.unsqueeze(-1), 1.0)
        f = f.mean(0).detach()
        P = g.mean(0)
        return self.n_prototypes * (f * P).sum()


# ==================== 训练/评估 ====================
def train_epoch_e(model, loader, criterion, optimizer, device, args=None):
    model.train()
    total, correct = 0, 0
    for tokens, tf_idx, labels in loader:
        tokens, labels = tokens.to(device), labels.to(device)
        logits = model(tokens)
        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        pred = logits.argmax(-1)
        total += labels.size(0)
        correct += (pred == labels).sum().item()
    return correct / total


def train_epoch_b(model, loader, criterion, optimizer, device, args):
    model.train()
    total, correct = 0, 0
    for tokens, tf_idx, labels in loader:
        tokens, labels = tokens.to(device), labels.to(device)
        tf_idx = tf_idx.to(device)
        logits, g, e = model(tokens, tf_idx, return_gate=True)
        loss = criterion(logits, labels)
        if args.lambda_commit > 0:
            loss = loss + args.lambda_commit * model.commitment_loss(e, g)
        if args.lambda_balance > 0:
            loss = loss + args.lambda_balance * model.balance_loss(g)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        pred = logits.argmax(-1)
        total += labels.size(0)
        correct += (pred == labels).sum().item()
    return correct / total


def train_epoch_pth(model, loader, criterion, optimizer, device, args=None):
    model.train()
    total, correct = 0, 0
    for tokens, tf_idx, labels in loader:
        tokens, labels = tokens.to(device), labels.to(device)
        tf_idx = tf_idx.to(device)
        logits = model(tokens, tf_idx)
        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        pred = logits.argmax(-1)
        total += labels.size(0)
        correct += (pred == labels).sum().item()
    return correct / total


@torch.no_grad()
def evaluate_pth(model, loader, device, collect_per_tf=False):
    model.eval()
    preds, labels, tf_ids = [], [], []
    for tokens, tf_idx, lb in loader:
        tokens = tokens.to(device)
        tf_idx = tf_idx.to(device)
        logits = model(tokens, tf_idx)
        preds.append(torch.softmax(logits, -1)[:, 1].cpu())
        labels.append(lb)
        tf_ids.append(tf_idx)
    return _roc_agg(preds, labels, tf_ids, collect_per_tf)


@torch.no_grad()
def evaluate_e(model, loader, device, collect_per_tf=False):
    model.eval()
    preds, labels, tf_ids = [], [], []
    for tokens, tf_idx, lb in loader:
        tokens = tokens.to(device)
        logits = model(tokens)
        preds.append(torch.softmax(logits, -1)[:, 1].cpu())
        labels.append(lb)
        tf_ids.append(tf_idx)
    return _roc_agg(preds, labels, tf_ids, collect_per_tf)


@torch.no_grad()
def evaluate_b(model, loader, device, collect_per_tf=False):
    model.eval()
    preds, labels, tf_ids = [], [], []
    for tokens, tf_idx, lb in loader:
        tokens = tokens.to(device)
        tf_idx = tf_idx.to(device)
        logits = model(tokens, tf_idx)
        preds.append(torch.softmax(logits, -1)[:, 1].cpu())
        labels.append(lb)
        tf_ids.append(tf_idx)
    return _roc_agg(preds, labels, tf_ids, collect_per_tf)


def _roc_agg(preds, labels, tf_ids, collect_per_tf):
    preds = torch.cat(preds).numpy()
    labels = torch.cat(labels).numpy()
    tf_ids = torch.cat(tf_ids).cpu().numpy()
    overall = roc_auc_score(labels, preds) if len(np.unique(labels)) > 1 else float('nan')
    per_tf = {}
    if collect_per_tf:
        for ti in np.unique(tf_ids):
            m = tf_ids == ti
            if len(np.unique(labels[m])) > 1:
                per_tf[int(ti)] = float(roc_auc_score(labels[m], preds[m]))
    return overall, per_tf


def build_valid(args):
    """-> valid list of (tf, gsm, pos, neg, family)"""
    tf_pairs = build_tf_list(None, args.csv_file)
    prot_dict = load_protein_fasta(args.protein_fasta)
    fam_map = load_family_map(args.family_map)
    valid = []
    for tf, gsm, _c in tf_pairs:
        if tf not in prot_dict:
            continue
        pos, neg = load_fasta_balanced(gsm, args.data_root, args.max_per_class)
        if not pos or not neg:
            continue
        valid.append((tf, gsm, pos, neg, fam_map.get(tf, 'Unknown/Other')))
    return valid


def build_samples(valid, tf_keep=None):
    """tf_keep: set of tf names 或 None(全部) -> samples list"""
    samples = []
    for tf_idx, (tf, gsm, pos, neg, _f) in enumerate(valid):
        if tf_keep is not None and tf not in tf_keep:
            continue
        n = min(len(pos), len(neg))
        for s in pos[:n]:
            samples.append((s, tf_idx, 1))
        for s in neg[:n]:
            samples.append((s, tf_idx, 0))
    return samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', required=True,
                    choices=['single_head', 'lookup_route', 'single_tf', 'per_tf_head'])
    ap.add_argument('--csv-file', default='full_1043.csv')
    ap.add_argument('--tf-list', default=None,
                    help='single_tf 模式: TF 名单文件（每行一个）')
    ap.add_argument('--family-map', default=str(BASE / 'data/tf_family_mapping.tsv'))
    ap.add_argument('--data-root', default='/home/chenw/work/chiqseq/hg38_tranfac/negative_code/datasets')
    ap.add_argument('--protein-fasta', default=str(BASE / 'sequenceAnalysis/tf_sequences_final_v2.fasta'))
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--epochs', type=int, default=18)
    ap.add_argument('--batch-size', type=int, default=256)
    ap.add_argument('--max-per-class', type=int, default=400)
    ap.add_argument('--n-prototypes', type=int, default=167)
    ap.add_argument('--prot-dim', type=int, default=128)
    ap.add_argument('--tau-start', type=float, default=1.0)
    ap.add_argument('--tau-end', type=float, default=0.15)
    ap.add_argument('--gumbel', type=float, default=0.5)
    ap.add_argument('--gumbel-end', type=float, default=0.05)
    ap.add_argument('--lambda-commit', type=float, default=0.1)
    ap.add_argument('--lambda-balance', type=float, default=0.01)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--weight-decay', type=float, default=1e-5)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--eval-holdout', default=None,
                    help='single_head 模式: 逗号分隔 family,random10 做零样本评估')
    ap.add_argument('--exclude-tf-list', default=None,
                    help='single_head/lookup_route: 从训练中排除这些 TF（配合 --eval-holdout 做真零样本）')
    args = ap.parse_args()

    set_seed(args.seed)
    log_file = setup_logging(args.out_dir)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f"模式: {args.mode} | 设备: {device} | 配置: {vars(args)}")

    valid = build_valid(args)
    logging.info(f"有效 TF: {len(valid)}")
    vocab = build_vocab()

    # ---------- single_tf: 逐 TF 单任务 ----------
    if args.mode == 'single_tf':
        tf_names = [l.strip() for l in open(args.tf_list) if l.strip()]
        out_csv = Path(args.out_dir) / 'per_tf_single.csv'
        with open(out_csv, 'w') as f:
            w = csv.writer(f)
            w.writerow(['TF_Name', 'Family', 'Test_ROC', 'Train_Acc', 'Epochs'])
        for name in tf_names:
            keep = {name}
            samples = build_samples(valid, tf_keep=keep)
            if not samples:
                logging.warning(f"  {name}: 无样本，跳过")
                continue
            train_s, val_s, test_s = split_samples_per_tf(samples, seed=args.seed)
            train_ds = ProtoTFDataset(train_s, vocab)
            val_ds = ProtoTFDataset(val_s, vocab)
            test_ds = ProtoTFDataset(test_s, vocab)
            tr = DataLoader(train_ds, batch_size=64, shuffle=True, collate_fn=collate_no_aa, num_workers=2)
            va = DataLoader(val_ds, batch_size=256, shuffle=False, collate_fn=collate_no_aa, num_workers=2)
            te = DataLoader(test_ds, batch_size=256, shuffle=False, collate_fn=collate_no_aa, num_workers=2)
            model = SingleHeadModel(len(vocab)).to(device)
            opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            crit = nn.CrossEntropyLoss()
            best, best_ep = 0, -1
            for ep in range(args.epochs):
                acc = train_epoch_e(model, tr, crit, opt, device)
                val_roc, _ = evaluate_e(model, va, device)
                if val_roc > best:
                    best, best_ep = val_roc, ep
                    torch.save({'model': model.state_dict(), 'epoch': ep},
                               Path(args.out_dir) / 'best_single.pt')
            model.load_state_dict(torch.load(Path(args.out_dir) / 'best_single.pt')['model'])
            test_roc, _ = evaluate_e(model, te, device)
            fam = valid[[v[0] for v in valid].index(name)][4]
            with open(out_csv, 'a') as f:
                w = csv.writer(f)
                w.writerow([name, fam, f"{test_roc:.4f}", f"{acc:.4f}", best_ep])
            logging.info(f"  {name} [{fam}] Test_ROC={test_roc:.4f} (best ep {best_ep})")
        logging.info(f"单任务完成 -> {out_csv}")
        return

    # ---------- single_head / lookup_route: 全量 1042 TF ----------
    excluded = set()
    if args.exclude_tf_list:
        excluded = {l.strip() for l in open(args.exclude_tf_list) if l.strip()}
        logging.info(f"训练排除 TF: {len(excluded)}")
    if excluded:
        train_keep = {v[0] for v in valid} - excluded
        samples = build_samples(valid, tf_keep=train_keep)
    else:
        samples = build_samples(valid)
    logging.info(f"总样本: {len(samples)}")
    train_s, val_s, test_s = split_samples_per_tf(samples, seed=args.seed)
    train_ds = ProtoTFDataset(train_s, vocab)
    val_ds = ProtoTFDataset(val_s, vocab)
    test_ds = ProtoTFDataset(test_s, vocab)
    tr = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                    collate_fn=collate_no_aa, num_workers=2, pin_memory=True)
    va = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                    collate_fn=collate_no_aa, num_workers=2)
    te = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                    collate_fn=collate_no_aa, num_workers=2)
    logging.info(f"train/val/test: {len(train_ds)}/{len(val_ds)}/{len(test_ds)}")

    if args.mode == 'single_head':
        model = SingleHeadModel(len(vocab)).to(device)
        train_epoch_fn = train_epoch_e
        eval_fn = evaluate_e
    elif args.mode == 'per_tf_head':
        model = PerTFHeadModel(len(vocab), n_tf=len(valid)).to(device)
        train_epoch_fn = train_epoch_pth
        eval_fn = evaluate_pth
    else:
        model = LookupRouteModel(len(vocab), n_tf=len(valid),
                                 n_prototypes=args.n_prototypes,
                                 prot_dim=args.prot_dim,
                                 tau=args.tau_start, gumbel=args.gumbel).to(device)
        train_epoch_fn = train_epoch_b
        eval_fn = evaluate_b
    n_params = sum(p.numel() for p in model.parameters())
    logging.info(f"模型参数: {n_params/1e6:.2f}M | 原型数: {getattr(model, 'n_prototypes', 1)}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_roc, best_epoch = 0, -1
    history = []
    for ep in range(args.epochs):
        if args.mode == 'lookup_route':
            tau = args.tau_start + (args.tau_end - args.tau_start) * ep / max(args.epochs - 1, 1)
            model.tau.fill_(tau)
            gumbel = args.gumbel * (args.gumbel_end / args.gumbel) ** (ep / max(args.epochs - 1, 1))
            model.gumbel.fill_(gumbel)
        t0 = time.time()
        acc = train_epoch_fn(model, tr, criterion, optimizer, device, args)
        val_roc, _ = eval_fn(model, va, device)
        history.append({'epoch': ep, 'train_acc': acc, 'val_roc': val_roc,
                        'time': time.time() - t0})
        logging.info(f"ep {ep:02d} acc={acc:.4f} val_ROC={val_roc:.4f} ({history[-1]['time']:.0f}s)")
        if val_roc > best_roc:
            best_roc, best_epoch = val_roc, ep
            torch.save({'model': model.state_dict(), 'epoch': ep, 'config': vars(args)},
                       Path(args.out_dir) / 'best_model.pt')

    model.load_state_dict(torch.load(Path(args.out_dir) / 'best_model.pt')['model'])
    test_roc, per_tf = eval_fn(model, te, device, collect_per_tf=True)
    mean_tf = float(np.mean(list(per_tf.values()))) if per_tf else float('nan')
    logging.info(f"TEST overall ROC={test_roc:.4f} per-TF mean={mean_tf:.4f}")

    # 零样本评估（仅 single_head，且必须配合训练排除）
    zs_results = {}
    if args.mode == 'single_head' and args.eval_holdout and excluded:
        for tag in args.eval_holdout.split(','):
            ho_samples = build_samples(valid, tf_keep=excluded)
            ho_ds = ProtoTFDataset(ho_samples, vocab)
            ho_loader = DataLoader(ho_ds, batch_size=args.batch_size, shuffle=False,
                                   collate_fn=collate_no_aa, num_workers=2)
            zs_roc, zs_per_tf = eval_fn(model, ho_loader, device, collect_per_tf=True)
            zs_mean = float(np.mean(list(zs_per_tf.values()))) if zs_per_tf else float('nan')
            zs_results[tag] = {'overall_roc': float(zs_roc), 'per_tf_mean': zs_mean,
                               'n_tf': len(excluded)}
            logging.info(f"[零样本·{tag}] overall ROC={zs_roc:.4f} per-TF mean={zs_mean:.4f}")
    elif args.mode == 'single_head' and args.eval_holdout and not excluded:
        logging.warning("⚠️ --eval-holdout 需要 --exclude-tf-list，否则不是真零样本（已跳过）")

    summary = {
        'mode': args.mode,
        'config': vars(args),
        'n_tf': len(valid),
        'n_params': n_params,
        'test_overall_roc': float(test_roc),
        'test_per_tf_mean_roc': mean_tf,
        'per_tf_auc': {valid[int(k)][0]: v for k, v in per_tf.items()},
        'zero_shot': zs_results,
        'best_epoch': best_epoch,
        'history': history,
        'log_file': str(log_file),
    }
    with open(Path(args.out_dir) / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    torch.save({'model': model.state_dict(), 'config': vars(args)},
               Path(args.out_dir) / 'final_model.pt')
    logging.info(f"完成 -> {args.out_dir}")


if __name__ == '__main__':
    main()
