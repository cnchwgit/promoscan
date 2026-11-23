#!/usr/bin/env python3
# coding: UTF-8
"""
D1 落地：把 CBR-KAN (PLOS ONE 2025) 的 50-数据集基准整理成 ProMoScan 可用格式。

来源：KAN 项目 data/rawdata/（50 个 ENCODE ChIP-seq，GM12878/H1-hESC/K562）
协议（严格照抄原文 Methods/Results）：
  - 正样本：每数据集 top ~15,000 条、200bp
  - 负样本：匹配 repeat fraction / length / GC content（Deng et al. 协议）
  - 切分：10% 验证 + 剩余 90% 的 10% 作测试（≈81/10/9），分层；SEED=2021

产出（全部落在 ProMoScan/code/benchmark/ 与 ProMoScan/data/bench50/）：
  data/bench50/train/{DSID}.{pos,neg}.fasta
  data/bench50/test/{DSID}.{pos,neg}.fasta
  code/benchmark/bench50.csv         # TF_Name=DSID, GSM_ID=DSID, Symbol=TF symbol
  code/benchmark/bench50_prot.fasta  # DSID 键的蛋白序列
  code/benchmark/bench50_esm.pt      # DSID 键的 ESM-2 嵌入（缺失的用 UniProt 现补）
  code/benchmark/bench50_alias.json  # DSID -> DB motif symbol
  code/benchmark/bench50_zshold.json # {"holdout_tfs": []}（50 个全当 seen）
  code/benchmark/bench50_meta.json   # 逐数据集：cell/TF/symbol/n_pos/n_neg/是否补齐
"""
import csv
import json
import os
import random
import re
import sys
import urllib.request
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
RAW = Path(os.environ.get('CBRKAN_RAWDATA', BASE / 'data/rawdata'))
PROT_FASTA = Path(os.environ.get('PROT_FASTA', BASE / 'data/tf_sequences_final_v2.fasta'))
ESM_PT = Path(os.environ.get('PRODEEP_ESM_PT', BASE / 'data/esm2_emb_650M.pt'))
OUT = BASE / 'data/bench50'
CDIR = Path(__file__).resolve().parent

CAP = 15000        # 每类上限（原文 "~15,000 top-ranked"）
SEED = 2021        # 原文 config SEED
VAL_FRAC = 0.10
TEST_FRAC = 0.10   # 占剩余 90% 的 10%

# 基名别名（剥离抗体码后仍非规范符号的，映射到规范符号）
BASE_ALIAS = {
    'EBF': 'EBF1', 'GABP': 'GABPA', 'PU1': 'SPI1', 'NRSF': 'REST',
    'SIN3A': 'SIN3A', 'SIN3AK20': 'SIN3A',
    'BCLAF101388': 'BCLAF1',
    'PAX5C20': 'PAX5', 'PAX5N19': 'PAX5',
}


def strip_codes(rest):
    r = re.sub(r'Pcr[12]x$', '', rest)         # 抗体码（先剥，否则挡住后面的 sc 尾码）
    r = re.sub(r'V041610[12]$', '', r)         # 抗体批号
    r = re.sub(r'sc[0-9]+$', '', r)            # sc 货号
    return r


def read_fasta(fp, cap=None):
    seqs, cur = [], []
    with open(fp) as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if cur:
                    seqs.append(''.join(cur)); cur = []
                if cap and len(seqs) >= cap:
                    break
            elif line:
                cur.append(line.upper())
    if cur and not (cap and len(seqs) >= cap):
        seqs.append(''.join(cur))
    return seqs[:cap] if cap else seqs


def write_fasta(path, seqs, prefix):
    with open(path, 'w') as f:
        for i, s in enumerate(seqs):
            f.write(f'>{prefix}_{i}\n{s}\n')


def norm_name(rest):
    tok = strip_codes(rest).upper()
    return BASE_ALIAS.get(tok, tok)


def main():
    os.makedirs(OUT / 'train', exist_ok=True)
    os.makedirs(OUT / 'test', exist_ok=True)

    # ---- 蛋白库（规范符号 -> 序列）----
    prot = {}
    name, buf = None, []
    for line in open(PROT_FASTA):
        line = line.strip()
        if line.startswith('>'):
            if name:
                prot[name.upper()] = ''.join(buf)
            name, buf = line[1:].split('|')[0].strip(), []
        elif line:
            buf.append(line)
    if name:
        prot[name.upper()] = ''.join(buf)
    print(f'蛋白库: {len(prot)} 条')

    # ---- ESM 库（原键 -> 向量）----
    import torch
    esm_raw = torch.load(ESM_PT, map_location='cpu', weights_only=False)
    esm_ci = {k.upper(): v for k, v in esm_raw.items()}
    print(f'ESM 库: {len(esm_raw)} 条')

    pat = re.compile(r'^wgEncodeHaibTfbs(Gm12878|H1hesc|K562)(.+?)PkRep1$')
    dirs = sorted(d for d in os.listdir(RAW) if pat.match(d))
    print(f'数据集: {len(dirs)}')

    meta, rows, alias, dsid_prot, missing_sym = [], [], {}, {}, {}
    rng = random.Random(SEED)

    for d in dirs:
        m = pat.match(d)
        cell, rest = m.group(1), m.group(2)
        dsid = d[16:]
        sym = norm_name(rest)
        pos = read_fasta(RAW / d / 'positive.fasta', CAP)
        neg = read_fasta(RAW / d / 'negative.fasta', CAP)
        pos = [s for s in pos if len(s) == 200]
        neg = [s for s in neg if len(s) == 200]
        # 分层切分：10% val + 剩余 90% 的 10% test
        def split(a):
            a = a[:]; rng.shuffle(a)
            n = len(a); nval = int(round(n * VAL_FRAC)); ntest = int(round((n - nval) * TEST_FRAC))
            return a[:n - nval - ntest], a[n - nval - ntest:n - nval], a[n - nval:]
        ptr, pva, pte = split(pos)
        ntr, nva, nte = split(neg)
        write_fasta(OUT / 'train' / f'{dsid}.pos.fasta', ptr, f'{dsid}_pos')
        write_fasta(OUT / 'train' / f'{dsid}.neg.fasta', ntr, f'{dsid}_neg')
        write_fasta(OUT / 'test' / f'{dsid}.pos.fasta', pte, f'{dsid}_pos')
        write_fasta(OUT / 'test' / f'{dsid}.neg.fasta', nte, f'{dsid}_neg')
        rows.append({'TF_Name': dsid, 'GSM_ID': dsid, 'Subgroup': cell, 'Symbol': sym})
        alias[dsid] = sym
        seq = prot.get(sym)
        if seq is None:
            missing_sym[sym] = dsid
        else:
            dsid_prot[dsid] = seq
        meta.append(dict(dsid=dsid, cell=cell, tf_token=rest, symbol=sym,
                         n_pos=len(pos), n_neg=len(neg),
                         train=len(ptr) + len(ntr), test=len(pte) + len(nte),
                         prot_ok=seq is not None))

    # ---- 补缺失蛋白（UniProt, reviewed, human）----
    fetched = {}
    for sym, dsid in sorted(missing_sym.items()):
        try:
            q = f'https://rest.uniprot.org/uniprotkb/search?query=gene:{sym}+AND+organism_id:9606+AND+reviewed:true&fields=accession,gene_names,sequence&format=json&size=1'
            with urllib.request.urlopen(q, timeout=30) as r:
                j = json.loads(r.read().decode())
            seq = j['results'][0]['sequence']['value']
            fetched[sym] = seq
            print(f'  UniProt 补齐 {sym}: {len(seq)} aa ({j["results"][0]["primaryAccession"]})')
        except Exception as e:  # noqa: BLE001
            print(f'  !! {sym} 拉取失败: {e}', file=sys.stderr)
    for r in meta:
        if not r['prot_ok'] and r['symbol'] in fetched:
            r['prot_ok'] = True

    # ---- 写蛋白 fasta（DSID 键）----
    with open(CDIR / 'bench50_prot.fasta', 'w') as f:
        for r in meta:
            seq = dsid_prot.get(r['dsid']) or fetched.get(r['symbol'])
            if seq:
                f.write(f">{r['dsid']} | {r['symbol']} {r['cell']}\n{seq}\n")
            else:
                print(f'  !! 无蛋白: {r["dsid"]} ({r["symbol"]})', file=sys.stderr)

    # ---- 写 ESM（DSID 键）：优先原库，其次现算 ----
    todo = [r['dsid'] for r in meta
            if r['symbol'].upper() not in esm_ci and (dsid_prot.get(r['dsid']) or fetched.get(r['symbol']))]
    merged = {}
    for r in meta:
        v = esm_ci.get(r['symbol'].upper())
        if v is not None:
            merged[r['dsid']] = v
    if todo:
        fpath = CDIR / 'bench50_missing.fasta'
        with open(fpath, 'w') as f:
            for r in meta:
                if r['dsid'] in todo:
                    f.write(f">{r['dsid']}\n{dsid_prot.get(r['dsid']) or fetched.get(r['symbol'])}\n")
        print(f'现算 ESM: {len(todo)} 条 -> {fpath}')
        os.system(f'python3 {BASE}/code/protoTF/embed_esm2.py '
                  f'--protein-fasta {fpath} --out {CDIR}/bench50_missing_esm.pt')
        add = torch.load(CDIR / 'bench50_missing_esm.pt', map_location='cpu', weights_only=False)
        merged.update(add)
    torch.save(merged, CDIR / 'bench50_esm.pt')
    print(f'ESM(DSID): {len(merged)}/{len(meta)}')

    with open(CDIR / 'bench50.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['TF_Name', 'GSM_ID', 'Subgroup', 'Symbol'])
        w.writeheader(); w.writerows(rows)
    json.dump(alias, open(CDIR / 'bench50_alias.json', 'w'), indent=1)
    json.dump({'holdout_tfs': []}, open(CDIR / 'bench50_zshold.json', 'w'))
    json.dump(meta, open(CDIR / 'bench50_meta.json', 'w'), indent=1)

    print('\n=== 汇总 ===')
    print(f'数据集 {len(meta)} | 蛋白 OK {sum(r["prot_ok"] for r in meta)} | ESM {len(merged)}')
    for r in meta:
        print(f'  {r["dsid"]:44s} {r["cell"]:8s} {r["symbol"]:8s} '
              f'tr={r["train"]:6d} te={r["test"]:5d} prot={r["prot_ok"]}')


if __name__ == '__main__':
    main()
