#!/usr/bin/env bash
# Train the ProMoScan model on the 1,043-TF collection.
# Data: see docs/DATA.md. This wrapper uses the standard peak-vs-background
# negatives (data/bgroot); override DATA_ROOT for another layout.
set -euo pipefail
cd "$(dirname "$0")/.."

DATA_ROOT=${DATA_ROOT:-data/bgroot}
OUT_DIR=${OUT_DIR:-runs/promoscan_main}

mkdir -p "$(dirname "$OUT_DIR")"

python3 code/train_promoscan.py \
  --n-tf 938 \
  --samples-per-class 300 \
  --epochs 16 \
  --batch-size 256 \
  --M 16 --W 16 \
  --protein-enc esm \
  --use-dna-context \
  --lam-pwm 0.2 --s1-seen-only \
  --alpha-temp 1.0 \
  --lam-ab 1.0 --ab-n 32 \
  --eval-pairs 150 \
  --data-root "$DATA_ROOT" \
  --out-dir "$OUT_DIR" \
  --seed 42
