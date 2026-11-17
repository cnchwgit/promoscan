#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ESM-2 蛋白嵌入提取：tf_sequences_final_v2.fasta -> {TF: 1280d L2归一化均值池化向量}
默认 esm2_t33_650M_UR50D；输出 esm2_emb_650M.pt
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE / 'code/protoTF'))
from data_utils import load_protein_fasta  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--protein-fasta', default=str(BASE / 'sequenceAnalysis/tf_sequences_final_v2.fasta'))
    ap.add_argument('--model-name', default='esm2_t33_650M_UR50D')
    ap.add_argument('--out', default=str(BASE / 'results/esm2_emb_650M.pt'))
    ap.add_argument('--batch-size', type=int, default=4)
    ap.add_argument('--max-len', type=int, default=1022)
    args = ap.parse_args()

    import esm
    model, alphabet = esm.pretrained.load_model_and_alphabet(args.model_name)
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    repr_layer = int(args.model_name.split('_t')[1].split('_')[0])
    batch_converter = alphabet.get_batch_converter()

    prot = load_protein_fasta(args.protein_fasta)
    names = list(prot.keys())
    print(f'蛋白数: {len(names)} | 模型: {args.model_name} (layer {repr_layer}) | 设备: {device}')

    embs = {}
    with torch.no_grad():
        for i in range(0, len(names), args.batch_size):
            chunk = names[i:i + args.batch_size]
            data = [(n, prot[n][:args.max_len]) for n in chunk]
            _, _, tokens = batch_converter(data)
            tokens = tokens.to(device)
            out = model(tokens, repr_layers=[repr_layer], return_contacts=False)
            rep = out['representations'][repr_layer]           # [B, L, D]
            # 掩码掉 padding（alphabet.padding_idx）
            mask = (tokens != alphabet.padding_idx).unsqueeze(-1)
            pooled = (rep * mask).sum(1) / mask.sum(1).clamp_min(1)
            pooled = torch.nn.functional.normalize(pooled, dim=-1)
            for n, v in zip(chunk, pooled.cpu().numpy()):
                embs[n] = v.astype(np.float32)
            if (i // args.batch_size) % 50 == 0:
                print(f'  [{i}/{len(names)}]')
    torch.save(embs, args.out)
    print(f'saved -> {args.out} | {len(embs)} 条 | dim={next(iter(embs.values())).shape}')


if __name__ == '__main__':
    main()
