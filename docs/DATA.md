# Data: what to download and where to put it

Everything used in the paper is public. Two inputs are needed to re-run training; neither is included in
this repository.

## 1. The 50-dataset benchmark (CBR-KAN)

The benchmark is the one introduced with the CBR-KAN backbone (He et al., *PLOS One* 2025,
doi:10.1371/journal.pone.0322978): 50 ENCODE ChIP-seq experiments from three cell lines (Gm12878, H1hesc,
K562), 200-bp windows around peak summits.

Expected layout — one directory per experiment, matching the ENCODE file stem:

```
data/rawdata/
└── wgEncodeHaibTfbsGm12878BatfPcr1xPkRep1/
    ├── positive.fasta     # up to 15,000 top-ranked peaks
    └── negative.fasta     # matched background, same count
```

If the data live elsewhere, point the pipeline at them instead of copying:

```bash
export CBRKAN_RAWDATA=/path/to/rawdata
```

`code/benchmark/prep_bench.py` then builds the splits used in the paper:

- 10% validation, 10% of the remainder held out as test (1,500 windows per class per dataset),
- the rest for training (12,150 per class per dataset),
- 300 positives + 300 negatives per class for the smaller identity/standard-protocol experiments.

Dataset names, cell lines and the factor symbols are listed in `code/benchmark/bench50.csv`.

## 2. Protein sequences and embeddings

- `data/tf_sequences_final_v2.fasta` — one sequence per factor (1,043 human TFs, UniProt canonical).
  `code/benchmark/prep_bench.py` matches them by gene symbol.
- `data/esm2_emb_650M.pt` — ESM-2 650M embedding, L2-normalised, mean-pooled (1,043 × 1,280).
  Rebuild it from the FASTA with:

  ```bash
  python3 code/embed_esm2.py --fasta data/tf_sequences_final_v2.fasta --out data/esm2_emb_650M.pt
  ```

  Both paths can be overridden with `PROT_FASTA` and `PRODEEP_ESM_PT`.

The benchmark ships its own 50-factor embedding subset (`code/benchmark/bench50_esm.pt`, 260 kB), so
evaluating without rebuilding embeddings is possible.

## 3. Motif databases (weak supervision and motif recovery)

- HOCOMOCO v11 core HUMAN, mononucleotide: `hocomoco_v11_core_HUMAN_mono.meme`
- JASPAR 2024 CORE vertebrate non-redundant: `JASPAR2024_CORE_vert_nr.meme`

Place them under `data/motifs/`. `code/pwm_supervision.py` reads them; motif recovery is scored with
Tomtom (MEME suite) and compared against the database motif of the same factor.

## 4. Genome background (standard protocol)

The standard protocol draws negatives uniformly from hg38. The genome and its `.fai` index are only
needed to rebuild those negatives (`code/build_bg_dataset.py`, `code/bg_utils.py`); the negatives used in
the paper are summarised in `results/standard_protocol_R1.json`.

## What is not redistributed

Raw peak files, the hg38 FASTA and the trained checkpoints. The checkpoints are reproducible from the
scripts (one RTX 3090, about 5 h per model) and are 13.1 MB each.
