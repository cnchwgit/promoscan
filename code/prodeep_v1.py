#!/usr/bin/env python3
# coding: UTF-8
"""
proDeep-v1 —— 蛋白 → Motif（PWM/卷积核）→ 扫描 DNA

设计见 MODEL_DESIGN.md。核心约束：
  · 蛋白影响预测的**唯一通路**是生成的核 θ（+ α, β）；没有 TF index、没有 per-TF 头
  · 生成的核作用在 DNA 上的算子是**共享的**（per-sample 卷积核的深度卷积）
  · 容量可调：M（motif 数）× W（宽度）

容量级别：
  L0: M=1,  W=10
  L1: M=8,  W=12
  L3: M=32, W=16（超网络式，容量更大）
"""
import math
import os
import sys
from pathlib import Path

_RC_IDX = None

import torch
import torch.nn as nn
import torch.nn.functional as F

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE / 'protoTF'))
from prototf_net import ProteinTower, CBRKANSharedBackbone   # noqa: E402


_RC_TABLE = str.maketrans('ACGTNacgtn', 'TGCANtgcan')


def revcomp_seq(s):
    """DNA 字符串反向互补"""
    return s.translate(_RC_TABLE)[::-1]


def dna_onehot(seq, length=200):
    """-> [4, length] 浮点 one-hot（未知碱基全 0）；统一大写"""
    m = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    x = torch.zeros(4, length)
    s = seq.upper()[:length]
    for i, ch in enumerate(s):
        j = m.get(ch)
        if j is not None:
            x[j, i] = 1.0
    return x


def revcomp_oh(oh):
    """one-hot [B,4,L] → 反向互补 [B,4,L]"""
    global _RC_IDX
    if _RC_IDX is None or _RC_IDX.device != oh.device:
        _RC_IDX = torch.tensor([3, 2, 1, 0], device=oh.device)
    return oh.flip(-1).index_select(1, _RC_IDX)


class ESMProteinTower(nn.Module):
    """冻结 ESM-2 650M 嵌入 [B,1280]（已 L2 归一化） → 蛋白嵌入 [B,prot_dim]。

    用 LayerNorm + Dropout，故意不用 BatchNorm：BN 的批内统计依赖 batch 组成，
    在"同一蛋白 repeat 成整批"的评估/AB 场景里会退化（ProtoTF 踩过的坑）。
    """

    def __init__(self, in_dim=1280, prot_dim=128, p_drop=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(p_drop),
            nn.Linear(512, prot_dim))

    def forward(self, emb):
        return self.net(emb)


class ProMoScan(nn.Module):
    def __init__(self, vocab_size, M=8, W=12, prot_dim=128,
                 use_dna_context=True, prot_max_len=1024, alpha_temp=1.0,
                 rc_symmetric=False, pool='max', pool_temp=1.0,
                 protein_enc='cnn', prot_drop=0.2, motif_head='std'):
        super().__init__()
        self.M, self.W = M, W
        self.alpha_temp = float(alpha_temp)   # <1 → α 更锐（单选哪个 motif）
        self.rc_symmetric = bool(rc_symmetric)  # 双链扫描：正向/反向互补取 max
        self.pool = str(pool)                   # 'max' | 'lse'
        self.pool_temp = float(pool_temp)
        self.protein_enc = str(protein_enc)     # 'cnn' one-hot | 'esm' 冻结 ESM-2 嵌入
        self.prot_tower = (ESMProteinTower(1280, prot_dim, prot_drop)
                           if self.protein_enc == 'esm' else ProteinTower(prot_dim))
        # motif_head：身份接口容量对照（E9）
        #   std   = 128→256→M*4*W（默认，与已归档结果一致）
        #   linear= 128→M*4*W（缩小接口，检验"容量小是否仍能泛化"）
        #   wide  = 128→512→512→M*4*W（放大接口，检验"容量大是否记忆身份"）
        self.motif_head_mode = str(motif_head)
        if self.motif_head_mode == 'linear':
            self.motif_head = nn.Linear(prot_dim, M * 4 * W)
        elif self.motif_head_mode == 'wide':
            self.motif_head = nn.Sequential(
                nn.Linear(prot_dim, 512), nn.ReLU(True),
                nn.Linear(512, 512), nn.ReLU(True), nn.Linear(512, M * 4 * W))
        elif self.motif_head_mode == 'free':
            # 无约束蛋白接口对照（E10free）：PWM 头保留且照常接受 S1 监督，
            # 但预测**不经过 PWM**——蛋白嵌入直接进分类头。
            # 目的：检验 S2（身份必须经共享 DNA 算子）是否必要，而非容量问题。
            self.motif_head = nn.Sequential(
                nn.Linear(prot_dim, 256), nn.ReLU(True), nn.Linear(256, M * 4 * W))
            self.free_head = nn.Sequential(
                nn.Linear(prot_dim, 512), nn.ReLU(True),
                nn.Linear(512, 512), nn.ReLU(True), nn.Linear(512, 1))
        else:
            self.motif_head = nn.Sequential(
                nn.Linear(prot_dim, 256), nn.ReLU(True), nn.Linear(256, M * 4 * W))
        self.alpha = nn.Linear(prot_dim, M)
        self.beta = nn.Linear(prot_dim, 1)
        self.dna_backbone = CBRKANSharedBackbone(vocab_size) if use_dna_context else None
        if self.dna_backbone is not None:
            self.w = nn.Linear(256, 1)

    def motifs(self, aa):
        """-> log-odds 核 [B,M,4,W] 与 α [B,M]、β [B,1]"""
        z = self.prot_tower(aa)
        theta = self.motif_head(z).view(-1, self.M, 4, self.W)
        p = F.softmax(theta, dim=2)
        logodds = torch.log(p + 1e-6) - math.log(0.25)
        alpha = torch.softmax(self.alpha(z) / self.alpha_temp, dim=1)
        return logodds, alpha, self.beta(z)

    def forward(self, tokens, dna_oh, aa):
        if self.motif_head_mode == 'free':
            z = self.prot_tower(aa)
            logit = self.beta(z).squeeze(1) + self.free_head(z).squeeze(1)
            if self.dna_backbone is not None:
                logit = logit + self.w(self.dna_backbone(tokens)).squeeze(1)
            return logit
        logodds, alpha, beta = self.motifs(aa)                 # [B,M,4,W]
        x_unf = dna_oh.unfold(2, self.W, 1)                    # [B,4,L,W]
        s = torch.einsum('bcij,bmcj->bmi', x_unf, logodds)     # [B,M,L]
        if self.rc_symmetric:
            x_rc = revcomp_oh(dna_oh).unfold(2, self.W, 1)
            s = torch.maximum(s, torch.einsum('bcij,bmcj->bmi', x_rc, logodds))
        if self.pool == 'lse':
            t = self.pool_temp
            s = torch.logsumexp(s * t, dim=2) / t              # 软池化（比 hard max 低方差）
        else:
            s = s.max(dim=2).values                            # [B,M]
        logit = beta.squeeze(1) + (alpha * s).sum(1)
        if self.dna_backbone is not None:
            logit = logit + self.w(self.dna_backbone(tokens)).squeeze(1)
        return logit

    @torch.no_grad()
    def motif_pwm(self, aa):
        """输出概率 PWM [M,4,W]（可解释性用）"""
        return self.motif_probs(aa)[0]

    def motif_probs(self, aa):
        """输出概率 PWM [B,M,4,W]（S1 监督用，可导）"""
        logodds, _, _ = self.motifs(aa)
        return (logodds + math.log(0.25)).exp()
