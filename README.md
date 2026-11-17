# ProMoScan — one shared model, constant parameters

Code, results and figure scripts for the manuscript

> **One model, constant parameters: a constrained protein-to-motif scanner replaces per-factor binding models**

ProMoScan is a single protein-conditioned model that predicts transcription-factor (TF) binding for any
factor without a per-factor head. The protein sequence is embedded with a frozen ESM-2 model and mapped
only to a small bank of position weight matrices, their mixture weights and a bias; each matrix is scanned
over the DNA and the prediction is a weighted sum of scan scores plus a DNA-context term. Because the
protein acts only through operators that act on the DNA, the model cannot store factor identity in a
lookup table, which is what makes its behaviour testable by intervention.

One model replaces 50 per-factor models on the CBR-KAN benchmark (0.9376 vs 0.9241 mean ROC-AUC) with
36.2× fewer trainable parameters. Replacing only the protein drops every one of the 50 datasets; a random
per-factor code collapses to chance on held-out factors; a free motif head makes the protein contribution
negative. Under the standard peak-versus-random-background protocol the protein channel adds nothing
(+0.0006), which is itself the paper's point: the channel buys identity, not benchmark points.

---

## What is in this repository

| Path | Contents |
|---|---|
| `code/prodeep_v1.py` | The model: CBR-KAN DNA backbone, protein tower, constrained PWM head |
| `code/train_prodeep_v1.py` | Training loop, datasets, A/B ranking objective |
| `code/protein_features.py`, `code/embed_esm2.py` | Protein featurisation (frozen ESM-2) |
| `code/pwm_supervision.py` | Loading database motifs and the weak-supervision loss |
| `code/benchmark/` | 50-dataset benchmark: preparation, training, evaluation, comparisons |
| `code/figures/` | Figure generators (native size) |
| `code/figures/hq/` | Figure generators at final print size (17.1 cm, 7 pt) — **these reproduce the published figures** |
| `code/eval_*.py` | Identity task, standard protocol, motif recovery |
| `results/` | Result files read by the figure and table scripts |
| `tables/` | Machine-readable tables (parameter efficiency, paper tables) |
| `paper_data/` | Two small JSON files used by Figure 5 |
| `docs/` | Data acquisition and end-to-end pipeline notes |

Not included: raw ChIP-seq data, trained checkpoints (13.1 MB per model) and the manuscript sources.
See `docs/DATA.md` for how to obtain the data.

## Install

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
# KAN layers
git clone https://github.com/Blealtan/efficient-kan ~/efficient_kan
# ESM-2 (only needed to rebuild protein embeddings)
pip install fair-esm
```

Python 3.12, PyTorch 2.5 (CUDA 12.1), a single 24 GB GPU is enough. Exact versions: `environment.txt`.

## Reproduce the figures

Everything is computed from the JSON files in `results/`; no GPU and no training needed.

```bash
cd code/figures/hq
for f in hq_fig2.py hq_fig3.py hq_fig4.py hq_fig5.py hq_figS1.py; do python3 "$f"; done
```

Outputs land in `figures/` as PNG (600 dpi), PDF, TIFF (LZW) and SVG. Each script prints a geometry
report (canvas size, minimum font size, count of out-of-canvas or overlapping labels) and exits non-zero
if a figure would be clipped. The PNGs are byte-identical to the submitted Figure 2–5 and Figure S1.

Figure 1 (architecture) is drawn by `code/figures/gen_fig1_architecture_promoscan.py`, which carries its
own geometry checker; the submitted version uses the same script with fonts scaled 1.12×.

## Reproduce the benchmark numbers

The 50-dataset pipeline needs the raw data (see `docs/DATA.md`), then:

```bash
cd code/benchmark
python3 prep_bench.py                 # builds data/bench50/{train,test} from the raw fasta files
python3 train_bench.py --out-dir ../../results/prodeep_v1_BENCH50 --epochs 16 --seed 42
python3 eval_bench.py --ckpt ../../results/prodeep_v1_BENCH50/best_model.pt
python3 compare_bench50.py            # per-dataset table against the five baselines
```

Training the benchmark model takes about 5 h on one RTX 3090. Three seeds are averaged for the
ensemble: `eval_bench_ens.py` rebuilds the ensemble table from three checkpoints.

Placebo and control runs used in the paper:

| Run | Script |
|---|---|
| Shuffled / constant / per-factor-code protein | `eval_bench_protein_controls.py`, `protein_features.py` (`PRODEEP_ESM_PT`) |
| Free motif head (no DNA operator) | `train_bench.py --motif-head free` |
| Motif-head capacity (134 k / 298 k / 856 k) | `train_bench.py --motif-head linear|std|wide` |
| CBR-KAN baselines, one model per dataset | `train_cbrkan_bench50.py`, `run_baselines_bench50.sh` |
| Identity task, five folds | `code/eval_sameharness.py` |
| Standard peak-vs-background protocol | `code/eval_standard_protocol.py` |
| Motif recovery | `code/eval_motif_recovery.py` |

## What each result file backs

| File | Used for |
|---|---|
| `bench50_table_ens.json`, `bench50_table.json` | ProMoScan ensemble / single-seed mean AUC (Fig. 2) |
| `bench50_table_{cbrkan,DeepBind,DanQ,DeepD2V,DeepSEA}.json` | Reproduced baselines (Fig. 2) |
| `bench50_protein_controls.json` | Protein replacement ladder (Fig. 4, Table 3d) |
| `sameharness_compare_E9.json`, `sameharness_raw.json`, `sameharness_E9c.json` | Capacity, protein vs random code (Fig. 3) |
| `prodeep_v1_FAIR400/eval.json`, `prodeep_v1_E10FREE/eval.json` | PWM interface vs free head (Fig. 3D) |
| `sameharness_FOLD{0..4}.json` | Five-fold identity task (Fig. S1) |
| `standard_protocol_R1.json`, `equiv_standard_protocol.json` | Standard protocol and equivalence test (Fig. 5A) |
| `paper_data/figR1b_data.json`, `figE7_data.json` | Negative-composition ladder, data-efficiency scan (Fig. 5B–C) |

## Data

Public data only: CistromeDB ChIP-seq peaks (via the CBR-KAN benchmark), JASPAR and HOCOMOCO motifs,
ESM-2 embeddings. `docs/DATA.md` explains exactly what to download and where to put it.

## Notes on scope

- The DNA backbone (68.8% of the parameters) is the CBR-KAN architecture of He et al., used unchanged and
  not claimed as a contribution.
- The model is not presented as a zero-shot predictor for factors without ChIP-seq data; the strict
  zero-shot motif-recovery result is reported as a limitation.
- Published baseline numbers are shown for reference only and never enter the win/loss statistics.

## Citation

```bibtex
@article{promoscan2026,
  title   = {One model, constant parameters: a constrained protein-to-motif scanner replaces per-factor binding models},
  author  = {TODO},
  journal = {Computational and Structural Biotechnology Journal},
  year    = {2026}
}
```

`CITATION.cff` carries the same entry once the author list is final.

## License

MIT — see `LICENSE`. The CBR-KAN backbone derives from He et al. (PLOS One 2025); check that license
before redistributing derived weights.

## Contact

TODO: corresponding author and e-mail.
