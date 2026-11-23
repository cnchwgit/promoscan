# ProMoScan

Model code and run scripts for **ProMoScan**, a single protein-conditioned model for
transcription-factor binding. The protein sequence is embedded with a frozen ESM-2 model and
mapped only to a small bank of position weight matrices (PWMs), their mixture weights and a
bias. Each PWM is scanned over the DNA, and the prediction is a weighted sum of scan scores
plus a DNA-context term.

This repository contains **code only**: model definition, training/evaluation pipeline, and
scripts to run them. It does **not** ship figures, tables, result files, checkpoints, or
generated embeddings.

## Repository layout

| Path | Contents |
|---|---|
| `code/promoscan.py` | The model: CBR-KAN DNA backbone, protein tower, constrained PWM head |
| `code/prototf_net.py` | CBR-KAN backbone and ProtoTF baseline networks |
| `code/protein_features.py` | Protein featurisation (frozen ESM-2 or CNN encoder) |
| `code/pwm_supervision.py` | Database-motif loading and weak-supervision loss |
| `code/data_utils.py`, `code/bg_utils.py` | Sequence I/O and genome-background sampling |
| `code/train_promoscan.py` | Train/evaluate the model on the 1,043-TF collection |
| `code/train_variants.py` | Training variants used for the baseline/ablation models |
| `code/eval_sameharness.py` | Five-fold A-vs-B identity task |
| `code/eval_standard_protocol.py` | Standard peak-vs-random-background protocol |
| `code/eval_motif_recovery.py` | Tomtom-style motif recovery |
| `code/embed_esm2.py` | Build ESM-2 protein embeddings from a FASTA file |
| `code/build_bg_dataset.py` | Build hg38-background negatives |
| `code/benchmark/` | 50-dataset benchmark: preparation, training, evaluation, baselines |
| `code/protoTF/full_1043.csv` | TF manifest (name, GSM accession, subgroup) |
| `scripts/` | Portable wrappers for the main training runs |
| `docs/DATA.md` | What to download and where to put it |
| `docs/PIPELINE.md` | End-to-end order of operations |

## Install

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

# KAN layers
git clone https://github.com/Blealtan/efficient-kan ~/efficient_kan

# Only needed to rebuild protein embeddings
pip install fair-esm
```

Python 3.12 and PyTorch 2.5 (CUDA 12.1) are what the paper used; a single 24 GB GPU is
enough. Exact versions are in `environment.txt`.

## Data

All data are public. `docs/DATA.md` gives the full layout. The short version:

- **50-dataset benchmark**: CBR-KAN/ENCODE ChIP-seq peak files, one directory per experiment.
- **Protein sequences**: `data/tf_sequences_final_v2.fasta` (1,043 human TFs).
- **Protein embeddings**: `data/esm2_emb_650M.pt`, or rebuild with `code/embed_esm2.py`.
- **Held-out split**: `data/zero_shot_summary.json` with the `holdout_tfs` list (see `docs/DATA.md`).
- **Motif databases**: HOCOMOCO and JASPAR MEME files under `data/motifs/`.
- **Genome background**: hg38 FASTA + `.fai` for the standard protocol.

## Run the 50-dataset benchmark

```bash
export CBRKAN_RAWDATA=/path/to/rawdata
export PROT_FASTA=/path/to/tf_sequences_final_v2.fasta
export PRODEEP_ESM_PT=/path/to/esm2_emb_650M.pt

cd code/benchmark
python3 prep_bench.py                 # builds data/bench50/{train,test}
cd ../..

# one shared model
python3 code/benchmark/train_bench.py \
  --out-dir runs/promoscan_BENCH50 \
  --epochs 16 --batch-size 256 --seed 42

python3 code/benchmark/eval_bench.py \
  --ckpt runs/promoscan_BENCH50/best_model.pt \
  --out runs/bench50_table.json
```

The baseline collection is reproduced with:

```bash
bash code/benchmark/run_baselines_bench50.sh
python3 code/benchmark/train_cbrkan_bench50.py
```

The CBR-KAN baseline script expects a TF-MTL checkout at `external/TF-MTL` (override with
`TFMTL_ROOT`).

## Run the main model on 1,043 TFs

```bash
bash scripts/run_main_model.sh
```

or directly:

```bash
python3 code/train_promoscan.py \
  --n-tf 938 --samples-per-class 300 --epochs 16 \
  --M 16 --W 16 --protein-enc esm --use-dna-context \
  --lam-pwm 0.2 --s1-seen-only --alpha-temp 1.0 \
  --lam-ab 1.0 --ab-n 32 --eval-pairs 150 \
  --data-root data/bgroot --out-dir runs/promoscan_main --seed 42
```

Training writes checkpoints and logs to the `--out-dir` you provide. None of these files are
committed.

## Evaluation and controls

Point the evaluation scripts at a checkpoint produced above:

```bash
python3 code/eval_sameharness.py        # five-fold A-vs-B identity task
python3 code/eval_standard_protocol.py  # peaks vs uniform hg38 background
python3 code/eval_motif_recovery.py     # motif recovery against database PWMs
```

The main controls are:

| Control | Command |
|---|---|
| Protein replacement | `code/benchmark/eval_bench_protein_controls.py --ckpt ...` |
| Shuffled / constant protein | same script, `--mode {shuffled,centroid,farthest,nearest,random}` |
| Per-factor random code | `PRODEEP_ESM_PT=random_code.pt python3 code/benchmark/train_bench.py ...` |
| Free motif head (no DNA operator) | `python3 code/benchmark/train_bench.py --motif-head free` |
| Motif-head capacity | `python3 code/benchmark/train_bench.py --motif-head {linear,std,wide}` |
| CBR-KAN baselines | `code/benchmark/train_cbrkan_bench50.py`, `run_baselines_bench50.sh` |

## Outputs

Scripts write to `runs/` (wrappers) or to whatever `--out-dir` / `--out` you pass. `runs/`,
`results/`, checkpoints and generated embeddings are ignored by git. The repository itself
stays free of generated artefacts.

## What is not included

- Raw ChIP-seq data, hg38 FASTA, motif databases (see `docs/DATA.md`)
- Trained checkpoints
- Figures, tables, result JSONs, and generated protein embeddings

## Citation

```bibtex
@article{promoscan2026,
  title   = {ProMoScan: a shared DNA backbone with protein-guided motif scanning for transcription-factor binding},
  author  = {TODO},
  journal = {Computational and Structural Biotechnology Journal},
  year    = {2026}
}
```

`CITATION.cff` carries the same entry once the author list is final.

## License

MIT — see `LICENSE`. The CBR-KAN backbone derives from He et al. (PLOS One 2025); check that
license before redistributing derived weights.
