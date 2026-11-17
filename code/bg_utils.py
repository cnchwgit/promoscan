#!/usr/bin/env python3
# coding: UTF-8
"""随机基因组背景负样本采样器（路线 A：标准协议 peak vs random background）。

- 随机从 hg38 主染色体抽 200bp 窗口，剔除含非 ACGT（N 等）的窗口
- 直接对 hg38.fa 随机字节寻址（依赖 .fai：偏移/行长/行宽），无需 pysam/samtools
- 默认排除与正样本 peak 所在染色体无关，可与正样本数量 1:1
"""
import random
import zlib
from pathlib import Path


def stable_seed(s, mod=100000):
    """跨进程稳定、跨机器稳定的字符串种子。

    ⚠️ 不要用内置 hash()：Python 的字符串 hash 默认带每进程随机盐（PYTHONHASHSEED），
    同一字符串在不同进程会得到不同值 —— 会让"随机背景"不可复现（已踩过）。
    """
    return zlib.crc32(str(s).encode('utf-8')) % mod

HERE = Path(__file__).resolve().parent
DEFAULT_FA = '/home/chenw/work/chiqseq/hg38_tranfac/negative_code/hg38.fa'
DEFAULT_LEN = 200
PRIMARY = [f'chr{i}' for i in range(1, 23)] + ['chrX', 'chrY']


class Genome:
    def __init__(self, fa_path=DEFAULT_FA):
        self.fa_path = fa_path
        self.fai = {}
        with open(fa_path + '.fai') as f:
            for line in f:
                p = line.rstrip('\n').split('\t')
                name, length, offset, lb, lw = p[0], int(p[1]), int(p[2]), int(p[3]), int(p[4])
                self.fai[name] = (length, offset, lb, lw)
        self._fh = open(fa_path, 'rb')
        self.chroms = [c for c in PRIMARY if c in self.fai]

    def fetch(self, chrom, start, n):
        """0-based start，返回 n bp（大写）"""
        length, offset, lb, lw = self.fai[chrom]
        if start < 0 or start + n > length:
            return None
        b0 = offset + (start // lb) * lw + (start % lb)
        need = n + (n // lb + 2) * (lw - lb)
        self._fh.seek(b0)
        raw = self._fh.read(need)
        s = raw.replace(b'\n', b'').replace(b'\r', b'')
        return s[:n].decode('ascii', 'replace').upper()

    def sample(self, n_seq, seq_len=DEFAULT_LEN, seed=42, max_try_factor=50):
        rng = random.Random(seed)
        out = []
        tries = 0
        mxt = n_seq * max_try_factor
        weights = [self.fai[c][0] for c in self.chroms]
        while len(out) < n_seq and tries < mxt:
            tries += 1
            chrom = rng.choices(self.chroms, weights=weights, k=1)[0]
            L = self.fai[chrom][0]
            st = rng.randrange(0, max(1, L - seq_len))
            s = self.fetch(chrom, st, seq_len)
            if s is None or len(s) != seq_len:
                continue
            if any(ch not in 'ACGT' for ch in s):
                continue
            out.append((f'{chrom}:{st + 1}-{st + seq_len}', s))
        return out


def count_fasta(path):
    n = 0
    with open(path) as f:
        for line in f:
            if line.startswith('>'):
                n += 1
    return n


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--gsm-csv', default=str(HERE.parent / 'code/protoTF/full_1043.csv'))
    ap.add_argument('--data-root', default=str(HERE.parent / 'raw_data'))
    ap.add_argument('--out-dir', default=str(HERE.parent / 'data/bg'))
    ap.add_argument('--n-tf', type=int, default=0, help='0=全部')
    ap.add_argument('--seq-len', type=int, default=200)
    ap.add_argument('--seed', type=int, default=42)
    a = ap.parse_args()

    import csv
    rows = list(csv.DictReader(open(a.gsm_csv)))
    out_dir = Path(a.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    g = Genome()
    n_done = 0
    for r in rows:
        gsm = r['GSM_ID']
        pos = Path(a.data_root) / f'{gsm}.pos.fasta'
        if not pos.exists():
            continue
        npos = count_fasta(pos)
        if npos == 0:
            continue
        dst = out_dir / f'{gsm}.bg.fasta'
        if dst.exists():
            n_done += 1
            continue
        seqs = g.sample(npos, seq_len=a.seq_len, seed=a.seed + stable_seed(gsm))
        with open(dst, 'w') as fo:
            for i, (loc, s) in enumerate(seqs):
                fo.write(f'>{gsm}.bg_{i}::{loc}\n{s}\n')
        n_done += 1
        if a.n_tf and n_done >= a.n_tf:
            break
        if n_done % 100 == 0:
            print(f'  {n_done} GSM 完成', flush=True)
    print(f'完成 {n_done} 个 GSM → {out_dir}', flush=True)


if __name__ == '__main__':
    main()


# ======================== 序列洗牌（阴性对照用） ========================
def mono_shuffle(seq, rng):
    """单核苷酸洗牌：保留碱基组成，破坏全部顺序信息"""
    l = list(seq)
    rng.shuffle(l)
    return ''.join(l)


def dinuc_shuffle(seq, rng, max_try=200):
    """二核苷酸洗牌（Hierholzer 随机欧拉路径）：保留碱基与全部二核苷酸计数"""
    from collections import defaultdict
    for _ in range(max_try):
        g = defaultdict(list)
        for a, b in zip(seq, seq[1:]):
            g[a].append(b)
        for k in g:
            rng.shuffle(g[k])
        local = {k: list(v) for k, v in g.items()}
        stack = [seq[0]]
        circuit = []
        while stack:
            u = stack[-1]
            if local.get(u):
                stack.append(local[u].pop())
            else:
                circuit.append(stack.pop())
        out = ''.join(circuit[::-1])
        if len(out) == len(seq):
            return out
    return mono_shuffle(seq, rng)
