#!/usr/bin/env python3
# coding: UTF-8
"""
S1 的真实检验：模型预测的 M 个 motif 是否**真的对应**该 TF 的 DB motif？

不是看 loss，而是做**检索**：
  · 计算 sim 矩阵 [N,N]：TF i 的最佳预测 motif  vs  TF j 的 DB PWM
  · own_sim   = 对角（自己）
  · rank1_acc = 自己的 DB PWM 排第一的比例（随机基线 1/N）
若 own_sim ≫ 非对角 且 rank1 ≫ 1/N → S1 真的把 motif 学进去了。

用法: python3 eval_motif_recovery.py --ckpt results/promoscan_L3_S1
"""
import argparse, csv, json, os, sys
from pathlib import Path
import torch
from promoscan import ProMoScan                                    # noqa: E402
from pwm_supervision import load_db_pwms, pwm_max_sim, info_content, _revcomp  # noqa: E402

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE)); sys.path.insert(0, str(CODE / 'protoTF'))
sys.path.append(os.path.expanduser('~/efficient_kan'))
from protein_features import build_protein_features                        # noqa: E402
from data_utils import build_vocab, load_protein_fasta, encode_proteins  # noqa: E402
BASE = Path(__file__).resolve().parents[1]
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out', default=None)
    ap.add_argument('--min-ic', type=float, default=2.0, help='DB PWM 最低信息量(滤掉纯背景)')
    ap.add_argument('--chunk', type=int, default=64)
    a = ap.parse_args()
    ck = torch.load(Path(a.ckpt) / 'best_model.pt', map_location='cpu', weights_only=False)
    cfg = ck['config']
    net = ProMoScan(len(build_vocab()), M=cfg['M'], W=cfg['W'],
                       use_dna_context=cfg.get('use_dna_context', False),
                       alpha_temp=cfg.get('alpha_temp', 1.0),
                       rc_symmetric=cfg.get('rc_symmetric', False),
                       pool=cfg.get('pool', 'max'),
                       pool_temp=cfg.get('pool_temp', 1.0),
                       motif_head=cfg.get('motif_head', 'std'), protein_enc=cfg.get('protein_enc', 'cnn')).to(DEVICE)
    net.load_state_dict(ck['model']); net.eval()
    db = load_db_pwms(target_width=cfg['W'])
    db = {k: v for k, v in db.items() if float(info_content(v).sum()) >= a.min_ic}
    rowsc = list(csv.DictReader(open(CODE / 'protoTF/full_1043.csv')))
    tf2gsm = {r['TF_Name']: r['GSM_ID'] for r in rowsc}
    prot = load_protein_fasta(os.environ.get('PROT_FASTA', str(BASE / 'data/tf_sequences_final_v2.fasta')))
    zs_path = os.environ.get('PRODEEP_ZS_SUM', str(BASE / 'data/zero_shot_summary.json'))
    zs = json.load(open(zs_path))
    hold = set(zs['holdout_tfs'])
    seen = [t for t in tf2gsm if t not in hold and t in prot and t in db]
    hd = [t for t in hold if t in tf2gsm and t in prot and t in db]
    names = seen + hd
    print(f'参与检索的 TF: {len(names)} (seen {len(seen)} / holdout {len(hd)})', flush=True)
    prot_oh = build_protein_features(names, enc=cfg.get('protein_enc', 'cnn'))
    db_all = torch.stack([db[t] for t in names]).to(DEVICE)

    W = cfg['W']

    def pair_sim_all(P, Q):
        """P:[A,M,4,W] Q:[B,4,W] -> [A,B]。
        语义与 pwm_max_sim 完全一致：去背景中心化余弦 × 双链 × 全 offset，再 max over M。"""
        Pc, Qc = P - 0.25, Q - 0.25
        min_ov = max(2, W // 2)
        best = None
        for dbx in (Qc, _revcomp(Qc)):
            for off in range(-(W - min_ov), W - min_ov + 1):
                a0, a1 = max(0, off), min(W, off + W)
                if a1 - a0 < min_ov:
                    continue
                b0, b1 = a0 - off, a1 - off
                X = Pc[..., a0:a1]                                  # [A,M,4,ov]
                Y = dbx[:, :, b0:b1]                                # [B,4,ov]
                num = torch.einsum('amod,bod->abm', X, Y)           # [A,B,M]
                den = X.flatten(2).norm(dim=-1).unsqueeze(1) \
                    * Y.flatten(1).norm(dim=-1).unsqueeze(0).unsqueeze(-1)
                c = num / (den + 1e-8)
                best = c if best is None else torch.maximum(best, c)
        return best.max(dim=-1).values                              # [A,B]

    with torch.no_grad():
        pred = net.motif_probs(prot_oh.to(DEVICE)).to(DEVICE)           # [N,M,4,W]
        N = len(names)
        # 自检：广播实现必须与 pwm_max_sim 的成对结果一致
        _i, _j = 0, min(3, N - 1)
        _ref = float(pwm_max_sim(pred[_i:_i + 1], db_all[_j:_j + 1]).max())
        _got = float(pair_sim_all(pred[_i:_i + 1], db_all[_j:_j + 1])[0, 0])
        print(f'[自检] pair_sim_all={_got:.6f} vs pwm_max_sim={_ref:.6f} (差 {abs(_got - _ref):.2e})',
              flush=True)
        sim = torch.zeros(N, N)
        for s in range(0, N, a.chunk):
            e = min(N, s + a.chunk)
            sim[:, s:e] = pair_sim_all(pred, db_all[s:e]).cpu()         # [N,chunk] 已 max over M
    own = torch.diagonal(sim)
    rank = (sim > own.unsqueeze(1)).sum(1)
    res = {}
    for tag, sub in [('seen', seen), ('holdout', hd)]:
        ii = [names.index(t) for t in sub]
        res[tag] = dict(n=len(ii),
                        own_sim=float(own[ii].mean()),
                        offdiag_sim=float((sim.sum(1)[ii] - own[ii]).div(N - 1).mean()),
                        rank1_acc=float((rank[ii] == 0).float().mean()),
                        rank_le5=float((rank[ii] < 5).float().mean()),
                        random_rank1=1.0 / N)
    print(json.dumps(res, indent=2, ensure_ascii=False))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
