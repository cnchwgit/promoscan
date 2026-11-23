#!/usr/bin/env python3
# coding: UTF-8
"""
S1：protein → PWM 直接监督（ProMoScan）

把"靠扫描 loss 间接学 PWM"变成"直接用数据库 PWM 监督"。
数据：HOCOMOCO v11 core HUMAN + JASPAR 2024 CORE vertebrates → 1280 个 motif
做法：
  · 每个 TF 取**信息量最高**的一个 DB PWM，居中裁剪/补背景到固定宽度 W
  · 预测的 M 个 motif（概率 PWM）与该 DB PWM 求**最大余弦相似度**
    （双链 × 所有 offset，最小重叠 W/2）
  · 损失 = 1 − max_m(sim)
"""
import math
import re
from pathlib import Path

import torch
import torch.nn.functional as F

BASE = Path(__file__).resolve().parents[1]
MOTIF_DIR = BASE / 'data/motifs'
RC_IDX = torch.tensor([3, 2, 1, 0])


def parse_meme(path):
    out = []
    cur_id = cur_name = None
    rows = []
    in_matrix = False
    with open(path) as f:
        for line in f:
            line = line.rstrip('\n')
            if line.startswith('MOTIF'):
                if cur_id is not None and rows:
                    out.append((cur_id, cur_name, torch.tensor(rows, dtype=torch.float32).T))
                parts = line.split()
                cur_id = parts[1]
                cur_name = parts[2] if len(parts) > 2 else parts[1]
                rows = []; in_matrix = False
            elif line.startswith('letter-probability matrix'):
                in_matrix = True
            elif in_matrix:
                vals = line.split()
                if len(vals) == 4:
                    try:
                        rows.append([float(x) for x in vals])
                    except ValueError:
                        in_matrix = False
                else:
                    in_matrix = False
    if cur_id is not None and rows:
        out.append((cur_id, cur_name, torch.tensor(rows, dtype=torch.float32).T))
    return out


def norm_symbol(mid, name):
    if '_HUMAN' in mid:
        return mid.split('_HUMAN')[0].upper()
    if '::' in name:
        return None
    s = name.strip().upper()
    return s if re.fullmatch(r'[A-Z0-9\-\.]+', s) else None


def info_content(p):
    """p: [4,W] 概率 -> 每列信息量 [W]"""
    p = p.clamp_min(1e-6)
    return (p * (p / 0.25).log2()).sum(0)


def to_fixed_width(p, W):
    """[4,Wp] -> [4,W]，以信息量峰为中心裁剪，或补背景"""
    Wp = p.shape[1]
    if Wp == W:
        return p
    if Wp > W:
        ic = info_content(p)
        c = int(ic.argmax())
        s = min(max(c - W // 2, 0), Wp - W)
        return p[:, s:s + W]
    out = torch.full((4, W), 0.25)
    left = (W - Wp) // 2
    out[:, left:left + Wp] = p
    return out


def load_db_pwms(target_width=16, min_ic_sum=1.0):
    """-> {TF: tensor[4,target_width]}（每个 TF 取信息量最高的 motif）"""
    best = {}
    for path in [MOTIF_DIR / 'hocomoco_v11_core_HUMAN_mono.meme',
                 MOTIF_DIR / 'JASPAR2024_CORE_vert_nr.meme']:
        if not path.exists():
            continue
        for mid, name, pwm in parse_meme(str(path)):
            sym = norm_symbol(mid, name)
            if not sym:
                continue
            pwm = pwm / pwm.sum(0, keepdim=True).clamp_min(1e-9)
            ic = float(info_content(pwm).sum())
            if ic < min_ic_sum:
                continue
            if sym not in best or ic > best[sym][0]:
                best[sym] = (ic, to_fixed_width(pwm, target_width))
    return {k: v[1] for k, v in best.items()}


def load_db_pwms_multi(target_width=16, min_ic_sum=1.0, max_per_tf=None):
    """-> {TF: [tensor[4,W], ...]}（**全部** DB motif，不止最优的那个）

    与 load_db_pwms 的区别：那个每个 TF 只留信息量最高的一个，
    这里保留所有（HOCOMOCO+JASPAR 去重后最多 4 个/TF），用于 motif-only 预训练。
    """
    acc = {}
    for path in [MOTIF_DIR / 'hocomoco_v11_core_HUMAN_mono.meme',
                 MOTIF_DIR / 'JASPAR2024_CORE_vert_nr.meme']:
        if not path.exists():
            continue
        for mid, name, pwm in parse_meme(str(path)):
            sym = norm_symbol(mid, name)
            if not sym:
                continue
            pwm = pwm / pwm.sum(0, keepdim=True).clamp_min(1e-9)
            ic = float(info_content(pwm).sum())
            if ic < min_ic_sum:
                continue
            acc.setdefault(sym, []).append((ic, to_fixed_width(pwm, target_width)))
    out = {}
    for k, lst in acc.items():
        lst.sort(key=lambda t: -t[0])
        if max_per_tf:
            lst = lst[:max_per_tf]
        out[k] = [p for _, p in lst]
    return out


def pwm_all_to_all(P, Q, min_overlap=None):
    """P:[A,4,W] Q:[B,4,W] -> [A,B] 去背景中心化余弦 × 双链 × 全 offset 的最大值"""
    W = P.shape[-1]
    Pc, Qc = P - 0.25, Q - 0.25
    min_ov = min_overlap or max(2, W // 2)
    best = None
    for dbx in (Qc, _revcomp(Qc)):
        for off in range(-(W - min_ov), W - min_ov + 1):
            a0, a1 = max(0, off), min(W, off + W)
            if a1 - a0 < min_ov:
                continue
            b0, b1 = a0 - off, a1 - off
            X = Pc[..., a0:a1]                     # [A,4,ov]
            Y = dbx[:, :, b0:b1]                   # [B,4,ov]
            num = torch.einsum('aod,bod->ab', X, Y)
            den = (X.flatten(1).norm(dim=-1).unsqueeze(1)
                   * Y.flatten(1).norm(dim=-1).unsqueeze(0))
            c = num / (den + 1e-8)
            best = c if best is None else torch.maximum(best, c)
    return best


def motif_set_pair_sim(P, Q, min_overlap=None):
    """P:[Bp,M,4,W] Q:[Bq,M,4,W] -> [Bp]（要求 Bp==Bq）：每对样本的 motif **集合**最大相似度

    用于"相似度蒸馏"：让预测 motif 之间的相似度结构去对齐 DB motif 之间的相似度结构，
    从而把蛋白→motif 映射在蛋白空间中磨平滑（这是零样本泛化需要的归纳偏置）。"""
    Bp, Mp = P.shape[0], P.shape[1]
    Bq, Mq = Q.shape[0], Q.shape[1]
    W = P.shape[-1]
    S = pwm_all_to_all(P.reshape(Bp * Mp, 4, W), Q.reshape(Bq * Mq, 4, W), min_overlap)
    S = S.view(Bp, Mp, Bq, Mq).amax(dim=(1, 3))              # [Bp,Bq]
    return torch.diagonal(S)


def _revcomp(p):
    """反向互补：p 形状 [..., 4, W]（碱基轴 = -2）"""
    return p.flip(-1).index_select(-2, RC_IDX.to(p.device))


def pwm_max_sim(pred, db, min_overlap=None):
    """
    pred: [B,M,4,W] 概率 PWM；db: [B,4,W] 概率 PWM
    -> [B,M] 最大余弦相似度（**去背景中心化**后，双链 × 所有 offset）
    中心化后：背景列(0.25)->0，padded 列自动不贡献
    """
    B, M, _, W = pred.shape
    P = pred - 0.25
    Q = db - 0.25
    min_overlap = min_overlap or max(2, W // 2)
    best = None
    for dbx in (Q, _revcomp(Q)):
        for off in range(-(W - min_overlap), W - min_overlap + 1):
            a0, a1 = max(0, off), min(W, off + W)
            if a1 - a0 < min_overlap:
                continue
            b0, b1 = a0 - off, a1 - off
            X = P[..., a0:a1]                    # [B,M,4,ov]
            Y = dbx[:, :, b0:b1].unsqueeze(1)    # [B,1,4,ov]
            num = (X * Y).sum(dim=(-1, -2))
            den = X.flatten(2).norm(dim=-1) * Y.flatten(2).norm(dim=-1)
            c = num / (den + 1e-8)
            best = c if best is None else torch.maximum(best, c)
    return best


def s1_loss(pred_probs, db_batch, min_overlap=None):
    """pred_probs: [B,M,4,W]；db_batch: [B,4,W] -> 标量损失"""
    sim = pwm_max_sim(pred_probs, db_batch, min_overlap)   # [B,M]
    return (1.0 - sim.max(dim=1).values).mean()
