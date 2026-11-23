#!/usr/bin/env bash
# 50 数据集基准：逐 TF 基线（DeepBind / DanQ / DeepD2V / DeepSEA）本地复现
# 与 CBR-KAN 复现完全同协议：同 train/test fasta、81/9/10 切分、30 epoch、seed 42
# 幂等：已有 results/bench50_<model>/{dsid}.json 的自动跳过
set -u
cd "$(dirname "$0")/../.."   # ProMoScan/
for m in DeepBind DanQ DeepD2V DeepSEA; do
  echo "[$(date '+%F %H:%M:%S')] ===== $m 开始 ====="
  python3 code/benchmark/train_cbrkan_bench50.py \
    --model "$m" --epochs 30 --seed 42 \
    --out "results/bench50_$m"
  echo "[$(date '+%F %H:%M:%S')] $m 结束 rc=$?"
done
echo "[$(date '+%F %H:%M:%S')] ===== 全部基线复现完成 ====="
