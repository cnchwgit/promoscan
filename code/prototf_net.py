#!/usr/bin/env python3
# coding: UTF-8
"""
ProtoTF-Net: 原型码本双塔网络
- DNA 塔: CBR-KAN 共享主干 (Embedding + 多尺度CNN + BiLSTM -> 256d)
- 蛋白塔: 氨基酸one-hot -> Conv1d x3 + 注意力池化 -> e_t (128d)
- 原型码本: K 个可学习原型, e_t 相似度软分配 (温度退火)
- 预测头: K 个 KAN 头, 按分配权重加权组合

分簇从"离线蛋白k-mer聚类"变为"网络内可学习模块"（冷启动）。
"""
import math
import sys
import os

sys.path.append(os.path.expanduser('~/efficient_kan'))
from src.efficient_kan import KAN  # noqa: E402

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

AMINO_ACIDS = 'ARNDCEQGHILKMFPSTWYVX'  # 20标准 + X(未知)
AA_DIM = len(AMINO_ACIDS)              # 21
AA_TO_IDX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}


# ======================== DNA 塔: CBR-KAN 共享主干（与v5一致） ========================
class CBRKANSharedBackbone(nn.Module):
    def __init__(self, len_vocab, embed_dim=100):
        super().__init__()
        self.embedding = nn.Embedding(len_vocab, embed_dim, padding_idx=len_vocab - 1)

        self.cnn2 = nn.Sequential(
            nn.Conv1d(100, 128, 11), nn.ReLU(True), nn.Dropout(0.2), nn.MaxPool1d(2),
            nn.Conv1d(128, 256, 9), nn.ReLU(True), nn.Dropout(0.2), nn.MaxPool1d(2),
        )
        self.cnn1 = nn.Sequential(
            nn.Conv1d(100, 160, 9), nn.ReLU(True), nn.Dropout(0.2), nn.MaxPool1d(2),
            nn.Conv1d(160, 160, 1), nn.ReLU(True), nn.Dropout(0.2), nn.MaxPool1d(2),
            nn.Conv1d(160, 160, 5), nn.ReLU(True), nn.Dropout(0.2), nn.MaxPool1d(2),
            nn.Conv1d(160, 256, 8), nn.ReLU(True), nn.Dropout(0.2), nn.MaxPool1d(2),
        )
        self.cnn3 = nn.Sequential(
            nn.Conv1d(100, 180, 1), nn.ReLU(True), nn.Dropout(0.2), nn.MaxPool1d(2),
            nn.Conv1d(180, 256, 8), nn.ReLU(True), nn.Dropout(0.2), nn.MaxPool1d(2),
        )
        self.BiLSTM = nn.LSTM(256, 128, num_layers=2, batch_first=True,
                              bidirectional=True, bias=True, dropout=0.2)
        self.dropout = nn.Dropout(0.2)
        self.norm = nn.LayerNorm(256)

    def forward(self, x):
        """x: [B, L] token ids -> [B, 256]"""
        out = self.embedding(x).permute(0, 2, 1)
        o1 = self.cnn1(self.dropout(out)).permute(0, 2, 1)
        o2 = self.cnn2(self.dropout(out)).permute(0, 2, 1)
        o3 = self.cnn3(self.dropout(out)).permute(0, 2, 1)
        out = torch.sum(torch.cat([o1, o2, o3], 1), 1).unsqueeze(1)
        lstm_out, _ = self.BiLSTM(out)
        out = out + lstm_out
        return torch.sum(self.norm(out), 1)


# ======================== 蛋白塔: 氨基酸CNN + 注意力池化 ========================
class ProteinTower(nn.Module):
    """aa one-hot [B, 21, L] -> Conv1d x3 -> 注意力池化 -> e_t [B, prot_dim]
    附带掩码氨基酸自监督解码头（让 e_t 学到蛋白结构，冷启动不塌缩）"""

    def __init__(self, prot_dim=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(AA_DIM, 64, 7, padding=3), nn.ReLU(True), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 5, padding=2), nn.ReLU(True), nn.MaxPool1d(2),
            nn.Conv1d(128, 128, 5, padding=2), nn.ReLU(True), nn.MaxPool1d(2),
        )
        self.attn = nn.Linear(128, 1)
        # 用 BatchNorm 而非 LayerNorm: LN 会抹掉样本间的长度差异（e_t 塌缩元凶）
        self.norm = nn.BatchNorm1d(prot_dim)
        self.out = nn.Linear(128, prot_dim)
        # 掩码氨基酸自监督解码头
        self.decoder = nn.Linear(128, AA_DIM)
        # 氨基酸组成预测头：迫使 e_t 携带方向性蛋白差异（打破共线塌缩）
        self.comp_head = nn.Linear(prot_dim, AA_DIM)

    def forward(self, aa, return_h=False):
        """aa: [B, 21, L] -> [B, prot_dim]（可选返回卷积特征）"""
        h = self.conv(aa)                                  # [B,128,L/8]
        s = self.attn(h.transpose(1, 2)).squeeze(-1)       # [B,L/8]
        w = F.softmax(s, dim=-1).unsqueeze(1)
        e = (h * w).sum(-1)                                # [B,128]
        e = self.norm(self.out(e))
        return (e, h) if return_h else e

    def composition_loss(self, e_t, aa):
        """氨基酸组成预测: 由 e_t 预测蛋白的 aa 组成分布（软标签 CE）
        不同家族蛋白组成不同且不共线 -> 迫使 e_t 方向性分化"""
        comp_true = aa.mean(-1)                          # [B,21] 归一化计数
        logits = self.comp_head(e_t)                     # [B,21]
        return -(comp_true * F.log_softmax(logits, -1)).sum(-1).mean()

    def mlm_loss(self, aa, mask_ratio=0.15):
        """掩码氨基酸预测: 输入随机遮掉部分位置, 解码头重建, CE loss
        aa: [B,21,L] one-hot"""
        L = aa.size(-1)
        L8 = L // 8
        aa_idx = aa.argmax(1)                              # [B,L]
        target = aa_idx[:, ::8][:, :L8]                    # [B,L/8] 池化网格对齐
        mask = torch.rand(aa.size(0), L8, device=aa.device) < mask_ratio
        if not mask.any():
            return torch.tensor(0.0, device=aa.device)
        # 输入按 8 块遮零
        mask_in = mask.repeat_interleave(8, dim=1)[:, :L].unsqueeze(1)  # [B,1,L]
        aa_in = torch.where(mask_in, torch.zeros_like(aa), aa)
        h = self.conv(aa_in)                               # [B,128,L8]
        logits = self.decoder(h.transpose(1, 2))           # [B,L8,21]
        return F.cross_entropy(logits[mask], target[mask])


# ======================== ProtoTF-Net ========================
class ProtoTFNet(nn.Module):
    def __init__(self, len_vocab=66, n_prototypes=167, prot_dim=128,
                 feature_dim=256, kan_hidden=32, tau=1.0, gumbel=0.0):
        super().__init__()
        self.dna_backbone = CBRKANSharedBackbone(len_vocab)
        self.prot_tower = ProteinTower(prot_dim)
        # 原型码本（聚类中心，冷启动随机初始化；用于 commitment 锚定与事后分析）
        self.prototypes = nn.Parameter(torch.randn(n_prototypes, prot_dim) * 0.1)
        # MLP 门控: 放大 e_t 差异做路由（线性相似度门控在近共线嵌入下失效）
        self.gate_mlp = nn.Sequential(
            nn.Linear(prot_dim, 128), nn.ReLU(True), nn.Linear(128, n_prototypes))
        # K 个原型 KAN 头
        self.heads = nn.ModuleList([KAN([feature_dim, kan_hidden, 2])
                                    for _ in range(n_prototypes)])
        self.register_buffer('tau', torch.tensor(tau, dtype=torch.float32))
        self.register_buffer('gumbel', torch.tensor(gumbel, dtype=torch.float32))
        self.n_prototypes = n_prototypes
        self.prot_dim = prot_dim

    def gate(self, e_t, tau=None, topk=None):
        """e_t [B,d] -> 软分配 [B,K]；MLP门控 + 训练期Gumbel噪声防塌缩"""
        t = self.tau if tau is None else tau
        sim = self.gate_mlp(e_t) / t                       # [B,K]
        if self.training and self.gumbel > 0:
            u = torch.rand_like(sim).clamp_min(1e-9)
            noise = -(-u.log()).log()                     # Gumbel(0,1)
            sim = sim + self.gumbel * noise
        g = F.softmax(sim, dim=-1)
        if topk is not None and topk < self.n_prototypes:
            kth = torch.topk(g, topk, dim=-1).values[:, -1:]
            g = g * (g >= kth).float()
            g = g / (g.sum(-1, keepdim=True) + 1e-8)
        return g

    def forward(self, tokens, aa, tau=None, topk=None, return_gate=False):
        """
        tokens: [B, L] DNA token ids
        aa:     [B, 21, Lp] 蛋白 one-hot
        -> logits [B,2]（各原型头按分配加权）
        """
        h = self.dna_backbone(tokens)                      # [B,256]
        e_t = self.prot_tower(aa)                          # [B,128]
        g = self.gate(e_t, tau=tau, topk=topk)             # [B,K]
        outs = torch.stack([hd(h) for hd in self.heads], 1)  # [B,K,2]
        logits = (g.unsqueeze(-1) * outs).sum(1)
        if return_gate:
            return logits, g, e_t
        return logits

    def protein_mlm_loss(self, aa, mask_ratio=0.15):
        """蛋白塔自监督辅助损失"""
        return self.prot_tower.mlm_loss(aa, mask_ratio)

    def protein_comp_loss(self, aa, e_t):
        """蛋白氨基酸组成辅助损失"""
        return self.prot_tower.composition_loss(e_t, aa)

    # ---------- 损失辅助 ----------
    def commitment_loss(self, e_t, g):
        """软版commitment: 按分配权重拉近 e_t 与原型"""
        diff = e_t.unsqueeze(1) - self.prototypes.unsqueeze(0)   # [B,K,d]
        return (g * diff.pow(2).sum(-1)).sum() / e_t.size(0)

    def balance_loss(self, g):
        """Switch式负载均衡: f=硬分配占比(detach), P=平均门控概率(可导)"""
        f = torch.zeros_like(g)
        idx = g.argmax(-1)
        f.scatter_(1, idx.unsqueeze(-1), 1.0)
        f = f.mean(0).detach()
        P = g.mean(0)
        return self.n_prototypes * (f * P).sum()

    def entropy_bonus(self, g):
        """分配熵奖励: 鼓励每个样本的分配不塌缩到单原型"""
        return self.assignment_entropy(g)

    def assignment_entropy(self, g):
        """平均分配熵（评估聚类软硬程度用）"""
        return (-g * (g + 1e-9).log()).sum(-1).mean()


if __name__ == '__main__':
    # 快速自检
    torch.manual_seed(0)
    net = ProtoTFNet(n_prototypes=167).cuda().train()
    tok = torch.randint(0, 66, (8, 198), device='cuda')
    aa = torch.rand(8, 21, 512, device='cuda')
    logits, g, e = net(tok, aa, return_gate=True)
    print('logits', logits.shape, 'gate', g.shape, 'e_t', e.shape)
    print('commit', net.commitment_loss(e, g).item(),
          'balance', net.balance_loss(g).item(),
          'entropy', net.assignment_entropy(g).item())
    print('参数总量:', sum(p.numel() for p in net.parameters()) / 1e6, 'M')
