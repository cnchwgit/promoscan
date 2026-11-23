#!/usr/bin/env bash
# 用上一版 PLOS ONE（CBR-KAN）论文的训练配方跑 ProMoScan：
#   30 epoch / batch 128 / lr 1e-3 / seed 42 / 85:15 分层验证集（按验证集逐数据集 AUC 选点）
# 目的：排除"ProMoScan 是靠调参赢的"这一质疑——与逐 TF 基线完全同一套训练超参。
set -u
# 串行保护：同一时间只允许一个训练任务（拿不到锁就退出，绝不排队等待/并行）
exec 9>/tmp/promoscan_train.lock
if ! flock -n 9; then echo "已有训练在运行，退出" >&2; exit 1; fi
cd "$(dirname "$0")/.."
LOG=runs/run_bench50_plos.log
Q=runs/promoscan_BENCH50_plos.log
mkdir -p runs
{
  echo "[$(date '+%F %H:%M:%S')] ===== PLOS ONE 配方：ProMoScan on bench50 ====="
  python3 -u code/benchmark/train_bench.py \
    --epochs 30 --batch-size 128 --lr 1e-3 --seed 42 \
    --val-frac 0.15 --val-max 8000 --resume \
    --out-dir runs/promoscan_BENCH50_plos > "$Q" 2>&1
  rc=$?; echo "[$(date '+%F %H:%M:%S')] 训练 rc=$rc"; [ $rc -ne 0 ] && exit $rc
  python3 -u code/benchmark/eval_bench.py \
    --ckpt runs/promoscan_BENCH50_plos/best_model.pt \
    --out runs/bench50_table_BENCH50_plos.json \
    --shuffle-control > runs/bench50_eval_BENCH50_plos.log 2>&1
  rc=$?; echo "[$(date '+%F %H:%M:%S')] 评估 rc=$rc"
  echo "[$(date '+%F %H:%M:%S')] ===== 完成 ====="
} >> "$LOG" 2>&1
