#!/usr/bin/env python3
# coding: UTF-8
"""复现 bench50 的 3-seed 集成表（bench50_table_ens.json）

此前该表是手工汇总产物、仓库中没有生成器（复现性缺口）。本脚本把它变成可重跑：

  1. 校验三个 checkpoint 存在（BENCH50 / BENCH50_s1 / BENCH50_s2）
  2. 调用 eval_bench.py 以概率平均方式集成，输出到临时文件
  3. 与归档表逐数据集比对（容差 1e-6），打印差异
  4. 通过后写入 results/bench50_table_ens.json，并附带 provenance（种子、脚本、时间、复核结果）
  5. 原表备份为 bench50_table_ens.prev.json（仅首次）

用法：
  python3 code/benchmark/eval_bench_ens.py            # 复核 + 覆盖
  python3 code/benchmark/eval_bench_ens.py --check    # 只复核，不写
"""
import argparse
import hashlib
import json
import shutil
import subprocess
import os
import sys
from datetime import datetime
from pathlib import Path

BASE = Path(os.environ.get('PROMOSCAN_ROOT',
                          Path(__file__).resolve().parents[2]))
BENCH = BASE / 'code/benchmark'
RES = BASE / 'results'
ARCHIVE = RES / 'bench50_table_ens.json'
TMP = RES / 'bench50_table_ens.recheck.json'

CKPTS = {
    'seed_a': RES / 'prodeep_v1_BENCH50' / 'best_model.pt',
    'seed_b': RES / 'prodeep_v1_BENCH50_s1' / 'best_model.pt',
    'seed_c': RES / 'prodeep_v1_BENCH50_s2' / 'best_model.pt',
}


def md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true', help='只复核，不覆盖归档表')
    ap.add_argument('--tol', type=float, default=1e-6)
    ap.add_argument('--reuse', action='store_true', help='复用已存在的复核结果，不重新推理')
    a = ap.parse_args()

    missing = [k for k, p in CKPTS.items() if not p.exists()]
    if missing:
        raise SystemExit(f'缺少 checkpoint: {missing}')

    cmd = [sys.executable, str(BENCH / 'eval_bench.py'),
           '--ckpt', str(CKPTS['seed_a']),
           '--ens', f"{CKPTS['seed_b']},{CKPTS['seed_c']}",
           '--out', str(TMP),
           '--shuffle-control']
    if a.reuse and TMP.exists():
        print('复用已有复核结果:', TMP, flush=True)
    else:
        print('运行:', ' '.join(cmd), flush=True)
        subprocess.run(cmd, check=True, cwd=BASE)

    new = json.loads(TMP.read_text())
    if not ARCHIVE.exists():
        ARCHIVE.write_text(json.dumps(new, indent=1, ensure_ascii=False))
        print('归档表不存在，已用本次结果创建')
        return 0

    old = json.loads(ARCHIVE.read_text())
    o = {r['dsid']: r for r in old['per_dataset']}
    nw = {r['dsid']: r for r in new['per_dataset']}
    assert set(o) == set(nw), '数据集不一致'
    diffs = [(d, nw[d]['pmoscan'], o[d]['pmoscan']) for d in o
             if abs(nw[d]['pmoscan'] - o[d]['pmoscan']) > a.tol]
    print(f'\n逐数据集比对：{len(o)} 个，超出容差 {len(diffs)} 个')
    for d, x, y in diffs[:10]:
        print(f'  {d}: new {x:.6f} vs archive {y:.6f} (Δ{x-y:+.2e})')
    print(f"均值：new {new['pmoscan_mean']:.6f} vs archive {old['pmoscan_mean']:.6f}")
    print(f"打乱对照均值：new {new.get('pmoscan_shuffled_mean', float('nan')):.6f} "
          f"vs archive {old.get('pmoscan_shuffled_mean', float('nan')):.6f}")

    reproducible = not diffs
    prov = {
        'generated_by': 'code/benchmark/eval_bench_ens.py',
        'generated_at': datetime.now().isoformat(timespec='seconds'),
        'checkpoints': {k: {'path': str(v.relative_to(BASE)), 'md5': md5(v)} for k, v in CKPTS.items()},
        'ensemble': 'probability average over 3 seeds',
        'recheck': {'datasets': len(o), 'mismatches': len(diffs), 'tol': a.tol,
                    'reproducible': reproducible},
    }
    new['provenance'] = prov

    if a.check:
        print('\n[--check] 未写入归档表')
        return 0 if reproducible else 1

    if not (RES / 'bench50_table_ens.prev.json').exists():
        shutil.copy(ARCHIVE, RES / 'bench50_table_ens.prev.json')
        print('原表已备份 -> results/bench50_table_ens.prev.json')
    ARCHIVE.write_text(json.dumps(new, indent=1, ensure_ascii=False))
    print('\n已写入', ARCHIVE, '（附 provenance）')
    return 0 if reproducible else 2


if __name__ == '__main__':
    sys.exit(main())
