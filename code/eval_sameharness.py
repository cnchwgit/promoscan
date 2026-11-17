#!/usr/bin/env python3
# coding: UTF-8
"""同一套评估口径下，把 ProtoTF 和 proDeep-v1 拉齐对比（修完对齐 bug 之后）。

口径（两边完全一致）：
  · 同一批配对（sample_pairs，seed=42，150 对，seen / holdout 两个池）
  · 同一段序列：pos[400:800] vs peak_B[400:800]
  · 同一个打分模式：对**整列表按序打分**（天然与 y=[1]*n+[0]*n 对齐）
  · Δ(s) = s(蛋白A) - s(蛋白B)；ΔAUC = AUC(y, Δ)；蛋白打乱对照同理
"""
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

BASE = Path(os.environ.get('PROMOSCAN_ROOT',
                          Path(__file__).resolve().parents[1]))
CODE = BASE / 'code'
for p in [str(CODE), str(CODE / 'protoTF'), os.path.expanduser('~/efficient_kan')]:
    sys.path.append(p)
from prototf_net import ProtoTFNet, CBRKANSharedBackbone, KAN          # noqa: E402
from prodeep_v1 import ProMoScan, dna_onehot                        # noqa: E402
from protein_features import build_protein_features                 # noqa: E402
from data_utils import (build_vocab, seq_to_ids, load_fasta_balanced,   # noqa: E402
                        load_protein_fasta, encode_proteins)            # noqa: E402

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DATA_ROOT = str(BASE / 'raw_data')
ZS = Path(os.environ.get('ZS_RESULTS', BASE / 'external/multiTF_results'))
ZS_SUM = Path(os.environ.get('PRODEEP_ZS_SUM', str(ZS / 'prototf_zs_random10/zero_shot_summary.json')))
PROT_FASTA = str(BASE / 'sequenceAnalysis/tf_sequences_final_v2.fasta')
ESM_PT = os.environ.get('PRODEEP_ESM_PT', str(BASE / 'data/distance/esm2_emb_650M.pt'))
CSV = str(CODE / 'protoTF/full_1043.csv')
PAD = 200
VOCAB = build_vocab()


class SingleHeadModel(torch.nn.Module):
    def __init__(self, len_vocab, kan_hidden=32):
        super().__init__()
        self.backbone = CBRKANSharedBackbone(len_vocab)
        self.head = KAN([256, kan_hidden, 2])

    def forward(self, tokens, tf_idx=None):
        return self.head(self.backbone(tokens))


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


@torch.no_grad()
def score_proto(net, seqs, aa, bs=128):
    """ProtoTF：整列表按序打分"""
    out = []
    for i in range(0, len(seqs), bs):
        chunk = seqs[i:i + bs]
        ids = torch.stack([torch.LongTensor(seq_to_ids(s, VOCAB)) for s in chunk]).to(DEVICE)
        a = aa[i:i + len(chunk)]
        out.append(torch.softmax(net(ids, a), -1)[:, 1].cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def score_dna_only(net, seqs, bs=256):
    out = []
    for i in range(0, len(seqs), bs):
        chunk = seqs[i:i + bs]
        ids = torch.stack([torch.LongTensor(seq_to_ids(s, VOCAB)) for s in chunk]).to(DEVICE)
        out.append(torch.softmax(net(ids, None), -1)[:, 1].cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def score_our(net, seqs, aa, bs=128):
    """proDeep-v1：整列表按序打分"""
    out = []
    for i in range(0, len(seqs), bs):
        chunk = seqs[i:i + bs]
        ids = torch.stack([torch.LongTensor(seq_to_ids(s, VOCAB)) for s in chunk]).to(DEVICE)
        oh = torch.stack([dna_onehot(s, PAD) for s in chunk]).to(DEVICE)
        a = aa[i:i + len(chunk)]
        out.append(torch.sigmoid(net(ids, oh, a)).cpu().numpy())
    return np.concatenate(out)


def boot_ci(x, n_boot=2000, seed=0):
    """bootstrap 95% CI"""
    x = np.asarray(x, dtype=float)
    if x.size < 2:
        return [float(x.mean()), float(x.mean())]
    rng = np.random.RandomState(seed)
    bs = x[rng.randint(x.size, size=(n_boot, x.size))].mean(axis=1)
    return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--tags', default='L3,L5',
                    help='逗号分隔的 proDeep 配置标签（读 results/prodeep_v1_<tag>/best_model.pt）')
    ap.add_argument('--out', default='results/sameharness_compare.json')
    ap.add_argument('--pairs', type=int, default=150)
    ap.add_argument('--ref', default='proto', help='配对差值的参考标签（默认 proto）')
    args = ap.parse_args()
    TAGS = [t.strip() for t in args.tags.split(',') if t.strip()]

    zs = json.load(open(ZS_SUM))
    holdset = set(zs['holdout_tfs'])
    rows = list(csv.DictReader(open(CSV)))
    tf2gsm = {r['TF_Name']: r['GSM_ID'] for r in rows}
    prot = load_protein_fasta(PROT_FASTA)
    seen_all = [t for t in tf2gsm if t not in holdset and t in prot]
    hold_all = [t for t in holdset if t in tf2gsm and t in prot]
    import random as _r
    rng0 = _r.Random(42); rng0.shuffle(seen_all)
    seen = sorted(seen_all[:938]); hold = sorted(hold_all)
    all_names = seen + hold
    FEATS = {}

    def feats(enc):
        if enc not in FEATS:
            FEATS[enc] = build_protein_features(all_names, enc=enc)
        return FEATS[enc]

    prot_oh = feats('cnn')                    # ProtoTF / DNA-only 用 one-hot
    prot_oh = prot_oh.to(DEVICE)
    idx = {t: i for i, t in enumerate(all_names)}

    embs = torch.load(ESM_PT, map_location='cpu', weights_only=False)
    idxx = [i for i, t in enumerate(all_names) if t in embs]
    Mv = torch.stack([torch.as_tensor(embs[all_names[i]], dtype=torch.float32) for i in idxx]).numpy()
    Mv = Mv / (np.linalg.norm(Mv, axis=1, keepdims=True) + 1e-9)
    Dm = 1.0 - Mv @ Mv.T; np.fill_diagonal(Dm, 1.0)
    local = {t: k for k, t in enumerate([all_names[i] for i in idxx])}

    netP = ProtoTFNet(len_vocab=len(VOCAB), n_prototypes=167).to(DEVICE)
    netP.load_state_dict(torch.load(ZS / 'prototf_zs_random10/best_model.pt',
                                    map_location=DEVICE, weights_only=False)['model']); netP.eval()
    netE = SingleHeadModel(len(VOCAB)).to(DEVICE)
    sdE = torch.load(ZS / 'prototf_variant_E_zs_random10/best_model.pt', map_location=DEVICE, weights_only=False)
    netE.load_state_dict(sdE['model'] if 'model' in sdE else sdE); netE.eval()

    def rep(f, i, n):
        """把一个 TF 的特征重复 n 次：[...] -> [n, ...]（兼容 one-hot 与 ESM 向量）"""
        return f[i].unsqueeze(0).repeat(n, *([1] * f[i].dim()))

    ours = {}
    tag_feats = {}
    for tag in TAGS:
        ck = torch.load(BASE / f'results/prodeep_v1_{tag}/best_model.pt', map_location='cpu', weights_only=False)
        cfg = ck['config']
        n = ProMoScan(len(VOCAB), M=cfg['M'], W=cfg['W'],
                         use_dna_context=cfg.get('use_dna_context', False),
                         alpha_temp=cfg.get('alpha_temp', 1.0),
                         rc_symmetric=cfg.get('rc_symmetric', False),
                         pool=cfg.get('pool', 'max'),
                         pool_temp=cfg.get('pool_temp', 1.0),
                         motif_head=cfg.get('motif_head', 'std'), protein_enc=cfg.get('protein_enc', 'cnn')).to(DEVICE).eval()
        n.load_state_dict(ck['model']); ours[tag] = n
        tag_feats[tag] = feats(cfg.get('protein_enc', 'cnn')).to(DEVICE)

    res = {}
    for pool, nl in [('seen', [t for t in seen if t in embs]), ('holdout', [t for t in hold if t in embs])]:
        sub = Dm[[local[t] for t in nl]][:, [local[t] for t in nl]]
        prs = sample_pairs(nl, sub, args.pairs, 42)
        rng = np.random.RandomState(42)
        acc = {k: [] for k in ['proto', 'proto_shuf', 'dnaonly'] + TAGS + [f'{t}_shuf' for t in TAGS]}
        for (i, j) in prs:
            ta, tb = nl[i], nl[j]
            pos, _ = load_fasta_balanced(tf2gsm[ta], DATA_ROOT, 1400)
            pb, _ = load_fasta_balanced(tf2gsm[tb], DATA_ROOT, 1400)
            if len(pos) < 800 or len(pb) < 800:
                continue
            pa, pb2 = pos[400:800], pb[400:800]
            seqs = pa + pb2
            m = len(seqs)
            y = np.array([1] * len(pa) + [0] * len(pb2))
            ia, ib = idx[ta], idx[tb]
            aaA = rep(prot_oh, ia, m)
            aaB = rep(prot_oh, ib, m)
            ja = int(rng.randint(len(all_names)))
            aaS = rep(prot_oh, ja, m)
            sA = score_proto(netP, seqs, aaA); sB = score_proto(netP, seqs, aaB)
            acc['proto'].append(roc_auc_score(y, sA - sB))
            acc['proto_shuf'].append(roc_auc_score(y, score_proto(netP, seqs, aaS) - sB))
            acc['dnaonly'].append(roc_auc_score(y, score_dna_only(netE, seqs)))
            for tag in TAGS:
                F_ = tag_feats[tag]
                oA = score_our(ours[tag], seqs, rep(F_, ia, m))
                oB = score_our(ours[tag], seqs, rep(F_, ib, m))
                acc[tag].append(roc_auc_score(y, oA - oB))
                # 同一打乱蛋白下的对照 → 该模型的"蛋白贡献"（与 ProtoTF 同口径）
                oS = score_our(ours[tag], seqs, rep(F_, ja, m))
                acc[f'{tag}_shuf'].append(roc_auc_score(y, oS - oB))
        out = {k: float(np.mean(v)) for k, v in acc.items()}
        for k, v in acc.items():
            out[k + '_ci'] = boot_ci(v)
        # 与 ref 的**配对**差值 + CI（同一批配对，配对比较功效更高）
        refv = np.asarray(acc.get(args.ref, []), dtype=float)
        if refv.size:
            for k, v in acc.items():
                if k == args.ref or len(v) != refv.size:
                    continue
                d = np.asarray(v, dtype=float) - refv
                lo, hi = boot_ci(d)
                out[f'{k}_minus_{args.ref}'] = float(d.mean())
                out[f'{k}_minus_{args.ref}_ci'] = [lo, hi]
        out['n'] = len(acc['proto'])
        out['_raw'] = {k: [float(q) for q in v] for k, v in acc.items()}   # 保留逐对值，便于事后任意配对比较
        # 逐对配对差值（与 _raw 同序，同一批配对）→ 画图/配对检验可直接用，无需下游再对齐
        if refv.size:
            out['_raw_minus_' + args.ref] = {
                k: [float(a - b) for a, b in zip(v, refv)]
                for k, v in acc.items() if k != args.ref and len(v) == refv.size
            }
        res[pool] = out
        print(f"[{pool}] n={out['n']}  ProtoTF ΔAUC {out['proto']:.4f} (打乱 {out['proto_shuf']:.4f}, 蛋白贡献 "
              f"{out['proto']-out['proto_shuf']:+.4f}, DNA-only {out['dnaonly']:.4f}) | " +
              ' | '.join(f"{t} {out[t]:.4f}(打乱 {out[f'{t}_shuf']:.4f}, 蛋白贡献 "
                         f"{out[t]-out[f'{t}_shuf']:+.4f}, vs {args.ref} {out.get(f'{t}_minus_{args.ref}', float('nan')):+.4f} "
                         f"CI{['%.4f' % q for q in out.get(f'{t}_minus_{args.ref}_ci', [float('nan')]*2)]})" for t in TAGS), flush=True)
    (BASE / args.out).write_text(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
