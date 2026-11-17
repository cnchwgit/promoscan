#!/usr/bin/env bash
# R1：标准协议基线的**同协议重训**（消掉"跨协议零样本迁移"这个 P0 威胁）
# DNA-only(variant E) 与 ProtoTF(K=167) 改到 --data-root data/bgroot（peak vs 随机背景）
set -u
BG=/home/chenw/.openclaw/workspace/proDeep/data/bgroot
MT=/home/chenw/.openclaw/workspace/multiTF
Q=/home/chenw/.openclaw/workspace/proDeep/results/run_r1_queue.log

echo "[$(date +%H:%M:%S)] ===== R1: DNA-only(variant E) on bgroot =====" | tee -a $Q
( cd $MT/code/protoTF && python3 train_variants.py --mode single_head --csv-file full_1043.csv \
    --data-root $BG --epochs 18 --batch-size 256 --max-per-class 400 \
    --eval-holdout random10 --exclude-tf-list exclude_random10.txt \
    --out-dir $MT/results/prototf_variant_E_zs_random10_bg \
    > $MT/results/variant_E_bg.log 2>&1 )
echo "[$(date +%H:%M:%S)] DNA-only rc=$?" | tee -a $Q

echo "[$(date +%H:%M:%S)] ===== R1: ProtoTF(K=167) on bgroot =====" | tee -a $Q
( cd $MT/code/protoTF && python3 zero_shot_prototf.py --csv-file full_1043.csv \
    --data-root $BG --holdout-frac 0.10 --holdout-seed 42 --epochs 18 \
    --out-dir $MT/results/prototf_zs_random10_bg \
    > $MT/results/prototf_zs_random10_bg.log 2>&1 )
echo "[$(date +%H:%M:%S)] ProtoTF rc=$?" | tee -a $Q

echo "[$(date +%H:%M:%S)] ===== R1 评估：同协议对照 =====" | tee -a $Q
cd /home/chenw/.openclaw/workspace/proDeep
python3 code/eval_standard_protocol.py --n-tf 200 --tags L10,M34,A1,A2 --ens A2,A3,A4 \
  --seen-order train --n-sup 938 \
  --dna-ckpt $MT/results/prototf_variant_E_zs_random10_bg/best_model.pt \
  --proto-ckpt $MT/results/prototf_zs_random10_bg/best_model.pt \
  --out results/standard_protocol_R1.json > results/standard_protocol_R1.log 2>&1
echo "[$(date +%H:%M:%S)] 评估 rc=$? -> results/standard_protocol_R1.json" | tee -a $Q
echo "[$(date +%H:%M:%S)] R1 全部完成" | tee -a $Q
