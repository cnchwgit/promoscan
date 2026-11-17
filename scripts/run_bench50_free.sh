#!/usr/bin/env bash
# 无约束蛋白融合基线（PC-free）：同一 CBR-KAN 骨干 + ESM-2 嵌入，但蛋白**不经过 PWM 扫描**，
# 直接进分类头（logit = β + free_head(z) + w·h）。用来回答"约束是否必要"。
# 与逐 TF 基线同一训练配方：30 ep / batch 128 / lr 1e-3 / seed 42 / 85:15 val 选点。
set -u
exec 9>/tmp/prodeep_train.lock
if ! flock -n 9; then echo "已有训练在运行，退出" >&2; exit 1; fi
cd "$(dirname "$0")"
LOG=results/run_bench50_free.log
Q=results/prodeep_v1_BENCH50_free.log
mkdir -p results
{
  echo "[$(date '+%F %H:%M:%S')] ===== 无约束蛋白融合基线（free head）====="
  python3 -u code/benchmark/train_bench.py \
    --epochs 30 --batch-size 128 --lr 1e-3 --seed 42 \
    --motif-head free --lam-pwm 0 --val-frac 0.15 --val-max 8000 --resume \
    --out-dir results/prodeep_v1_BENCH50_free >> "$Q" 2>&1
  rc=$?; echo "[$(date '+%F %H:%M:%S')] 训练 rc=$rc"; [ $rc -ne 0 ] && exit $rc
  python3 -u code/benchmark/eval_bench.py \
    --ckpt results/prodeep_v1_BENCH50_free/best_model.pt \
    --out results/bench50_table_BENCH50_free.json --shuffle-control \
    > results/bench50_eval_BENCH50_free.log 2>&1
  rc=$?; echo "[$(date '+%F %H:%M:%S')] 评估 rc=$rc"
  python3 -u code/benchmark/eval_bench_protein_controls.py \
    --ckpt results/prodeep_v1_BENCH50_free/best_model.pt \
    --out results/bench50_protein_controls_free.json \
    > results/bench50_protein_controls_free.log 2>&1
  echo "[$(date '+%F %H:%M:%S')] 蛋白归因 rc=$?"
  touch results/prodeep_v1_BENCH50_free/DONE
  echo "[$(date '+%F %H:%M:%S')] ===== 完成 ====="
} >> "$LOG" 2>&1
