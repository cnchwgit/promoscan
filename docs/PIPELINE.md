# End-to-end pipeline

Order of operations for training and evaluating ProMoScan. All outputs go to paths you pass
(`runs/` in the wrappers); no generated artefacts are part of this repository.

## 1. Benchmark data

```bash
export CBRKAN_RAWDATA=/path/to/rawdata      # see docs/DATA.md
export PROT_FASTA=/path/to/tf_sequences_final_v2.fasta
export PRODEEP_ESM_PT=/path/to/esm2_emb_650M.pt

cd code/benchmark
python3 prep_bench.py
```

This writes `data/bench50/{train,test}/<dsid>.{pos,neg}.fasta`, plus `bench50_meta.json`,
`bench50_alias.json`, `bench50.csv` and the 50-factor ESM embedding `bench50_esm.pt`.

## 2. One shared model (benchmark)

```bash
python3 train_bench.py --out-dir ../../runs/promoscan_BENCH50 \
                       --epochs 16 --batch-size 256 --seed 42
python3 eval_bench.py  --ckpt ../../runs/promoscan_BENCH50/best_model.pt \
                       --out ../../runs/bench50_table.json
```

`train_bench.py` accepts `--val-frac/--val-max` to hold out a stratified validation slice.
Three seeds (42, 1, 2) give an ensemble:

```bash
python3 eval_bench_ens.py --ckpt A B C --out ../../runs/bench50_table_ens.json
```

## 3. Baselines on the same split

```bash
bash run_baselines_bench50.sh          # DeepBind, DanQ, DeepD2V, DeepSEA
python3 train_cbrkan_bench50.py        # 50 per-factor CBR-KAN models
```

`train_cbrkan_bench50.py` expects a TF-MTL checkout at `external/TF-MTL` (override with
`TFMTL_ROOT`).

## 4. Main model on 1,043 TFs

```bash
cd ../..
bash scripts/run_main_model.sh
```

See the script for the full argument set; `--data-root` points at the peak/background layout
described in `docs/DATA.md`.

## 5. Controls and ablations

| Claim | Command |
|---|---|
| Protein replacement ladder | `python3 code/benchmark/eval_bench_protein_controls.py --ckpt ...` |
| Shuffled / constant protein | same script, `--mode {shuffled,centroid,farthest,nearest,random}` |
| Per-factor random code | `PRODEEP_ESM_PT=random_code.pt python3 code/benchmark/train_bench.py ...` |
| Free motif head | `python3 code/benchmark/train_bench.py --motif-head free` |
| Capacity (linear/std/wide) | `python3 code/benchmark/train_bench.py --motif-head linear|std|wide` |

## 6. Identity task and standard protocol

```bash
python3 code/eval_sameharness.py         # five-fold A-vs-B pairs
python3 code/eval_standard_protocol.py   # peaks vs uniform hg38 background
python3 code/eval_motif_recovery.py      # Tomtom recovery against database motifs
```

## Timing

| Step | Hardware | Time |
|---|---|---|
| `prep_bench.py` | CPU | ~10 min |
| One benchmark model, 16 epochs | 1× RTX 3090 | ~5 h |
| Protein-control evaluation (50 datasets) | 1× RTX 3090 | ~15 min |
