#!/usr/bin/env python3
# coding: UTF-8
"""蛋白特征构建：把"蛋白编码"这件事统一成一个函数，训练/评估脚本共用。

两种编码：
  · cnn : 21 维 one-hot [N, 21, 1024]（原有）
  · esm : 冻结 ESM-2 650M 嵌入 [N, 1280]（L2 归一化，来自 multiTF 预计算）
"""
import json
import os
from pathlib import Path

import torch

BASE = Path(__file__).resolve().parents[1]
# 允许用环境变量替换蛋白特征文件（E9c 身份来源对照：换成人造 per-TF 码）
ESM_PT = os.environ.get('PRODEEP_ESM_PT', str(BASE / 'data/esm2_emb_650M.pt'))
PROT_FASTA = os.environ.get('PROT_FASTA', str(BASE / 'data/tf_sequences_final_v2.fasta'))
ESM_DIM = 1280


def load_esm_raw(path=ESM_PT):
    return torch.load(path, map_location='cpu', weights_only=False)


def esm_matrix(names, path=ESM_PT, dim=ESM_DIM):
    """-> ([N,dim] float32, 缺失名字列表)。缺失行保持全 0 并记录。"""
    e = load_esm_raw(path)
    M = torch.zeros(len(names), dim, dtype=torch.float32)
    miss = []
    for i, t in enumerate(names):
        v = e.get(t)
        if v is None:
            miss.append(t)
        else:
            M[i] = torch.as_tensor(v, dtype=torch.float32)
    return M, miss


def build_protein_features(names, enc='cnn', fasta=None, max_len=1024):
    """names -> [N, ...] 张量。enc='cnn' 需要 fasta；enc='esm' 不需要。"""
    if enc == 'esm':
        M, miss = esm_matrix(names)
        if miss:
            print(f'[protein_features] ESM 缺失 {len(miss)}/{len(names)}：{miss[:8]}...', flush=True)
        return M
    from data_utils import load_protein_fasta, encode_proteins
    if fasta is None:
        fasta = load_protein_fasta(PROT_FASTA)
    oh, _ = encode_proteins(fasta, max_len, names=names)
    return oh


def esm_available(names):
    """返回在 ESM 表里存在的名字集合（评估池用）"""
    e = load_esm_raw()
    return {t for t in names if t in e}
