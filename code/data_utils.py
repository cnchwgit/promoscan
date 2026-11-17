#!/usr/bin/env python3
# coding: UTF-8
"""ProtoTF-Net 数据管线：DNA 3-mer 分词、FASTA 加载、蛋白 one-hot、多TF数据集"""
import os
from pathlib import Path

import torch
from torch.utils.data import Dataset

from prototf_net import AA_DIM, AA_TO_IDX, AMINO_ACIDS

PAD_SIZE = 200  # DNA 序列截断长度


# ======================== DNA 词表与分词（与 v5 一致） ========================
def build_vocab():
    bases = ['A', 'C', 'G', 'T']
    vocab = {}
    for b1 in bases:
        for b2 in bases:
            for b3 in bases:
                vocab[b1 + b2 + b3] = len(vocab)
    vocab['<PAD>'] = len(vocab)
    vocab['<UNK>'] = len(vocab)
    return vocab


def seq_to_ids(seq, vocab, pad_size=PAD_SIZE):
    tokens = [seq[i:i + 3] for i in range(min(len(seq) - 2, pad_size - 2))]
    ids = [vocab.get(t, vocab['<UNK>']) for t in tokens]
    target_len = pad_size - 2
    if len(ids) < target_len:
        ids += [vocab['<PAD>']] * (target_len - len(ids))
    return ids[:target_len]


# ======================== FASTA 加载 ========================
def load_fasta_balanced(gsm_id, data_root, max_per_class=15000):
    pos_file = Path(data_root) / f"{gsm_id}.pos.fasta"
    neg_file = Path(data_root) / f"{gsm_id}.neg.fasta"

    def read_fasta(fp, limit):
        seqs = []
        try:
            with open(fp) as f:
                for line in f:
                    if line.startswith('>'):
                        continue
                    s = line.strip().upper()
                    if s:
                        seqs.append(s)
                        if limit > 0 and len(seqs) >= limit:
                            break
        except FileNotFoundError:
            return []
        return seqs

    return read_fasta(pos_file, max_per_class), read_fasta(neg_file, max_per_class)


# ======================== 蛋白序列加载与编码 ========================
def load_protein_fasta(fasta_path):
    """-> {TF名: aa序列}；header: '>ADNP | NP_xxx ...' 取 '|' 前名称"""
    prot = {}
    name, seq = None, []
    with open(fasta_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                if name is not None:
                    prot[name] = ''.join(seq)
                name = line[1:].split('|')[0].strip()
                seq = []
            else:
                seq.append(line)
        if name is not None:
            prot[name] = ''.join(seq)
    return prot


def aa_to_indices(aa_seq, max_len):
    """氨基酸序列 -> 索引列表（截断/填充到 max_len；未知字符映射到 X）"""
    idx = [AA_TO_IDX.get(c.upper(), AA_TO_IDX['X']) for c in aa_seq[:max_len]]
    if len(idx) < max_len:
        idx += [AA_TO_IDX['X']] * (max_len - len(idx))
    return idx


def encode_proteins(prot_dict, max_len=1024, names=None):
    """-> (tensor [n,21,max_len] fp32, name列表)。names 为 None 时按 dict 顺序"""
    if names is None:
        names = sorted(prot_dict.keys())
    n = len(names)
    oh = torch.zeros(n, AA_DIM, max_len)
    for i, name in enumerate(names):
        seq = prot_dict.get(name)
        if not seq:
            continue
        idx = aa_to_indices(seq, max_len)
        oh[i, idx, torch.arange(max_len)] = 1.0
    return oh, names


# ======================== 多TF数据集 ========================
def build_tf_list(cluster_csv_dir=None, csv_file=None):
    """
    收集 (TF_Name, GSM_ID[, Cluster_ID]) 列表。
    - cluster_csv_dir: 目录下的 cluster_*_t098.csv 全部读取
    - csv_file: 单个 CSV（列: TF_Name, GSM_ID[, Subgroup]）
    返回三元组 (tf, gsm, cluster)；无簇信息时 cluster=None
    """
    pairs = []
    if csv_file:
        with open(csv_file) as f:
            header = f.readline().strip().split(',')
            ti = header.index('TF_Name'); gi = header.index('GSM_ID')
            ci = header.index('Subgroup') if 'Subgroup' in header else None
            for line in f:
                parts = line.strip().split(',')
                if len(parts) > max(ti, gi):
                    c = parts[ci].strip() if ci is not None and len(parts) > ci else None
                    pairs.append((parts[ti].strip(), parts[gi].strip(), c))
    elif cluster_csv_dir:
        for fp in sorted(Path(cluster_csv_dir).glob('cluster_*.csv')):
            with open(fp) as f:
                header = f.readline().strip().split(',')
                ti = header.index('TF_Name'); gi = header.index('GSM_ID')
                ci = header.index('Subgroup') if 'Subgroup' in header else None
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) > max(ti, gi):
                        c = parts[ci].strip() if ci is not None and len(parts) > ci else None
                        pairs.append((parts[ti].strip(), parts[gi].strip(), c))
    # 去重保序
    seen = set(); out = []
    for p in pairs:
        if p[0] not in seen:
            seen.add(p[0]); out.append(p)
    return out


class ProtoTFDataset(Dataset):
    """
    每个样本: (DNA token ids, tf_idx, label)
    蛋白 one-hot 在 collate 里按 tf_idx 从预编码张量取（避免重复编码）
    """
    def __init__(self, samples, vocab):
        """
        samples: list of (seq_str, tf_idx, label)
        """
        self.vocab = vocab
        self.data = [(torch.LongTensor(seq_to_ids(s, vocab)), ti, lb)
                     for s, ti, lb in samples]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ids, ti, lb = self.data[idx]
        return ids, ti, lb


def collate_prototf(batch, prot_oh):
    seqs = torch.stack([b[0] for b in batch])
    tf_idx = torch.LongTensor([b[1] for b in batch])
    labels = torch.LongTensor([b[2] for b in batch])
    aa = prot_oh[tf_idx]                      # [B,21,Lp]
    return seqs, tf_idx, labels, aa


def split_samples_per_tf(samples, seed=42, ratios=(0.6, 0.2, 0.2)):
    """按 TF 分层 60/20/20 划分，返回 (train, val, test)"""
    import random
    rng = random.Random(seed)
    by_tf = {}
    for s in samples:
        by_tf.setdefault(s[1], []).append(s)
    train, val, test = [], [], []
    for tf, lst in by_tf.items():
        rng.shuffle(lst)
        n = len(lst)
        n_tr = int(n * ratios[0]); n_va = int(n * ratios[1])
        train += lst[:n_tr]
        val += lst[n_tr:n_tr + n_va]
        test += lst[n_tr + n_va:]
    return train, val, test
