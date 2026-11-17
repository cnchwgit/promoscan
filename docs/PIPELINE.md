# End-to-end pipeline

Order of operations for a full reproduction. Optional steps are marked.

## 0. Sanity check (no data needed)

```bash
cd code/figures/hq
python3 hq_fig2.py          # prints canvas size, font range, 0 overlaps
```

All five figure scripts run from the shipped result files. If this works, the repository is intact.

## 1. Benchmark data

```bash
export CBRKAN_RAWDATA=/path/to/rawdata      # see docs/DATA.md
export PROT_FASTA=/path/to/tf_sequences_final_v2.fasta
export PRODEEP_ESM_PT=/path/to/esm2_emb_650M.pt

cd code/benchmark
python3 prep_bench.py
```

This writes `data/bench50/{train,test}/<dsid>.{pos,neg}.fasta`, plus `bench50_meta.json`, per-dataset
protein symbols and the alias map.

## 2. One shared model (the paper's main run)

```bash
python3 train_bench.py --out-dir ../../results/prodeep_v1_BENCH50 \
                       --epochs 16 --batch-size 256 --seed 42
python3 eval_bench.py  --ckpt ../../results/prodeep_v1_BENCH50/best_model.pt
```

`train_bench.py` accepts `--val-frac/--val-max` to hold out a stratified validation slice (the CBR-KAN
recipe uses 8k windows); the default is full-data training. Three seeds (42, 1, 2) give the ensemble:

```bash
python3 eval_bench_ens.py --ckpt A B C --out ../../results/bench50_table_ens.json
```

## 3. Baselines on the same split

```bash
bash run_baselines_bench50.sh          # DeepBind, DanQ, DeepD2V, DeepSEA
python3 train_cbrkan_bench50.py        # 50 per-factor CBR-KAN models
python3 compare_bench50.py             # per-dataset table + Wilcoxon
```

## 4. Controls used for the attribution claims

| Claim | Command |
|---|---|
| Protein replacement ladder (Fig. 4) | `python3 eval_bench_protein_controls.py --ckpt ...` |
| Shuffled / constant protein | same script, `--mode {shuffled,centroid,farthest,nearest,random}` |
| Per-factor random code | `PRODEEP_ESM_PT=random_code.pt python3 train_bench.py ...` |
| Free motif head | `python3 train_bench.py --motif-head free` |
| Capacity 134 k / 298 k / 856 k | `python3 train_bench.py --motif-head linear|std|wide` |

## 5. Identity task and standard protocol

```bash
python3 code/eval_sameharness.py         # five-fold A-vs-B pairs, writes results/sameharness_FOLD*.json
python3 code/eval_standard_protocol.py   # peaks vs uniform hg38 background
python3 code/eval_motif_recovery.py      # Tomtom recovery against database motifs
```

## 6. Figures and tables

```bash
cd code/figures && python3 gen_fig1_architecture_promoscan.py   # Figure 1
cd hq && for f in hq_fig2.py hq_fig3.py hq_fig4.py hq_fig5.py hq_figS1.py; do python3 "$f"; done
```

Outputs: `figures/Figure{1..5,S1}.{png,pdf,tif,svg}`. `tables/Tables_paper.md` holds the four main
tables; every number in them traces to a file listed in the README table.

## Timing

| Step | Hardware | Time |
|---|---|---|
| `prep_bench.py` | CPU | ~10 min |
| One benchmark model, 16 epochs | 1× RTX 3090 | ~5 h |
| Protein-control evaluation (50 datasets) | 1× RTX 3090 | ~15 min |
| Figures (all five) | CPU | < 1 min |
