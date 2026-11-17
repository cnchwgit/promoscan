#!/usr/bin/env bash
# E9c 身份来源对照：把蛋白塔输入换成"人造 per-TF 随机码"（同架构/同接口/同数据/同配方）
# 预期：seen ΔAUC 仍高（可记忆），holdout ΔAUC 崩（无蛋白先验）
set -u
cd /home/chenw/.openclaw/workspace/proDeep
export PRODEEP_ESM_PT=/home/chenw/.openclaw/workspace/proDeep/data/identity_random/random_id_1280.pt
Q=results/run_e9c_queue.log
CFG="--n-tf 938 --samples-per-class 300 --M 16 --W 16 --protein-enc esm \
--use-dna-context --lam-pwm 0.2 --s1-seen-only --alpha-temp 1.0 --lam-ab 1.0 --ab-n 32 \
--eval-pairs 150 --data-root data/bgroot --epochs 16 --save-every 1 --resume --seed 42"
echo "[$(date +%H:%M:%S)] ===== E9c 随机身份码训练 =====" | tee -a $Q
python3 code/train_prodeep_v1.py $CFG --out-dir results/prodeep_v1_E9c_randid > results/prodeep_v1_E9c_randid.log 2>&1
echo "[$(date +%H:%M:%S)] 训练 rc=$?" | tee -a $Q
echo "[$(date +%H:%M:%S)] ===== E9c 匹配负样本协议 ΔAUC（含 seen 对照 A1）=====" | tee -a $Q
python3 code/eval_sameharness.py --tags A1,E9c_randid --ref A1 \
  --out results/sameharness_E9c.json > results/sameharness_E9c.log 2>&1
echo "[$(date +%H:%M:%S)] sameharness rc=$?" | tee -a $Q
echo "[$(date +%H:%M:%S)] ===== E9c 标准协议 =====" | tee -a $Q
python3 code/eval_standard_protocol.py --n-tf 200 --tags L10,M34,A1,E9c_randid --ens '' \
  --seen-order train --n-sup 938 --out results/standard_protocol_E9c.json > results/standard_protocol_E9c.log 2>&1
echo "[$(date +%H:%M:%S)] 标准协议 rc=$?" | tee -a $Q
echo "[$(date +%H:%M:%S)] E9c 全部完成" | tee -a $Q
