# Tables — CSBJ (English)

> 数字全部取自 `paper/Manuscript_v2_20261115/tables/` 与 `results/*.json`，未改动；仅翻译与并入正文表格。
> 投稿时这四张表会嵌入 `Manuscript_CSBJ.docx` 末尾（可编辑 Word 表），并在正文以 Table 1–4 引用。

## Table 1. Data, splits and evaluation protocols

| Item | Setting | Used for |
|---|---|---|
| ChIP-seq data | CistromeDB quality-filtered peaks, 1,043 human TFs | Training, five-fold CV |
| Sequence window | 200 bp (summit ± 100 bp) | All experiments |
| Samples per class | 300 positives + 300 negatives (cap 15,000 per factor) | Training; data-efficiency scan 5–1,000 |
| Negatives I (identity task) | Within-factor A/B pairs matched on library size and GC | ΔAUC (supplementary) |
| Negatives II (standard protocol) | Uniform random 1:1 hg38 background (seed 42) | Standard ROC-AUC |
| Negatives III (controls) | Dinucleotide- and mononucleotide-shuffled positives | Benchmark attribution (Fig. 5B) |
| Five-fold split | Factor-level hold-out, 208–209 factors per fold (20%) | Fig. 3, Table 3 |
| 50-dataset benchmark | CBR-KAN public benchmark: 50 factor × cell-line datasets, 1,500 test windows per class | Fig. 2, Table 2 |
| Protein input | Frozen ESM-2 (650 M) embedding (1,280-d), 1,043 sequences | Identity pathway |
| Database motifs | JASPAR / HOCOMOCO, ~1,280 PWMs covering ~900 of 1,043 TFs | Weak supervision (seen factors only) |
| Training budget | 300 per class, 16 epochs, batch 256, lr 1e-3, seed 42 | Main configuration |
| Software / hardware | PyTorch, single RTX 3090 (24 GB) | — |

## Table 2. Main result on 50 public ChIP-seq datasets under one protocol

ROC-AUC averaged over datasets. Win/loss = per-dataset comparison (strict > / <). p: paired Wilcoxon signed-rank test.
"Published" values are shown for reference only and are excluded from all win/loss statistics.

| Method | Granularity | Trainable params per model | Total for 50 | Mean ROC-AUC (reproduced) | Published | vs ProMoScan (W/T/L) | Δ (reproduced) | Wilcoxon p | Disk size |
|---|---|---|---|---|---|---|---|---|---|
| **ProMoScan (3-seed ensemble)** | 1 shared model | 3.27 M | 3.27 M | **0.9376** | — | — | — | — | 13.1 MB |
| ProMoScan (single seed) | 1 shared model | 3.27 M | 3.27 M | 0.9330 | — | — | — | — | 13.1 MB |
| ProMoScan (shuffled protein) | 1 shared model | 3.27 M | 3.27 M | 0.8771 | — | — | — | — | — |
| ProMoScan (shared + constant protein) | 1 shared model | 3.27 M | 3.27 M | 0.8928 | — | — | — | — | — |
| CBR-KAN (reproduced) | 50 per-factor models | 2.37 M | 118.4 M | 0.9241 | 0.9440 | 40/0/10 | +0.0134 | 8.4e-06 | 474 MB |
| DeepSEA (reproduced) | 50 per-factor models | 5.31 M | 265.5 M | 0.9171 | 0.9309 | 41/0/9 | +0.0205 | 3.1e-08 | 1,062 MB |
| DeepD2V (reproduced) | 50 per-factor models | 0.13 M | 6.7 M | 0.9147 | 0.9252 | 42/0/8 | +0.0229 | 4.2e-08 | 27 MB |
| DanQ (reproduced) | 50 per-factor models | 0.13 M | 6.6 M | 0.8989 | 0.9172 | 47/0/3 | +0.0387 | 1.1e-12 | 26 MB |
| DeepBind (reproduced) | 50 per-factor models | 0.17 M | 8.4 M | 0.8745 | 0.9024 | 48/0/2 | +0.0630 | 1.2e-14 | 34 MB |

## Table 3. Mechanism and attribution

Held-out ΔAUC is the identity-task metric (paired discrimination); ROC-AUC is on the 50 datasets. All values are from a
single evaluation harness.

| Axis | Comparison | Seen | Held-out | Effect |
|---|---|---|---|---|
| **(a) Capacity** (held-out ΔAUC) | Motif head 134 k (linear) | — | 0.6226 | — |
| | Motif head 298 k (standard) | — | 0.6116 | reference |
| | Motif head 856 k (wide, 6.4×) | — | 0.6196 | +0.008 |
| **(b) Identity source** | Protein embedding | 0.8215 | 0.6045 | — |
| | Per-factor random code, equal dimension and capacity | 0.8580 | 0.5203 | −0.084 (chance on held-out) |
| **(c) Interface** (protein contribution) | PWM scan (standard) | +0.1752 | +0.0607 | — |
| | Free motif head, no DNA operator | +0.0468 | −0.0262 | −0.087 |
| **(d) Protein replacement** (ROC-AUC, n = 50) | Correct protein | — | 0.9330 | — |
| | Mean protein | — | 0.8928 | −0.040 |
| | Nearest other factor | — | 0.8609 | −0.072 |
| | Farthest factor | — | 0.8706 | −0.062 |
| | Random factor | — | 0.8559 | −0.077 |
| **(e) Similarity vs loss** | Spearman ρ (n = 150 swaps) | — | ρ = +0.04, p = 0.59 | no relationship |

All 50 datasets are worse under every wrong protein (p = 1.8 × 10⁻¹⁵, paired test).

## Table 4. Boundaries of the effect

**(a) Standard protocol (peaks vs uniform hg38 background, all models retrained)**

| Model | Seen factors (n = 200) | Held-out factors (n = 91) |
|---|---|---|
| DNA-only | 0.9118 | 0.8918 |
| ProMoScan, single seed | 0.9409 | 0.8791 |
| ProMoScan, 3-seed ensemble | 0.9486 | 0.8924 |

Protein contribution on held-out factors: +0.0006 (95% CI −0.0075 to +0.0080; Wilcoxon p = 0.13; TOST p = 0.0099 for ±0.01).

**(b) Negative-composition ladder (ROC-AUC)**

| Model | hg38 background | Dinucleotide-shuffled | Mononucleotide-shuffled |
|---|---|---|---|
| DNA-only | 0.8744 | 0.5994 | 0.4332 |
| ProMoScan, single seed | 0.9409 | 0.6703 | 0.4979 |
| ProMoScan, 3-seed ensemble | 0.9486 | 0.6773 | 0.5099 |

**(c) Training examples per class (standard protocol)**

| Examples per class | 5 | 20 | 50 | 300 | 400 | 1,000 |
|---|---|---|---|---|---|---|
| Seen factors | 0.8521 | 0.8762 | 0.8945 | 0.9347 | 0.9381 | 0.9630 |
| Held-out factors | 0.8240 | 0.8470 | 0.8690 | 0.8836 | 0.8851 | 0.8917 |
