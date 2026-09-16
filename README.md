# DeSTAR

**DeSTAR** (Detectron Sensor TARget-site ranker) is a machine-learning model that ranks
candidate target sites for Detectron / toehold-switch RNA sensors. Give it a transcript;
it tells you which sites to build sensors against.

This repository has **two parts** — pick the one you need:

| I want to… | Go to | One-line command |
|---|---|---|
| **Rank target sites on my own transcript** (use the model) | [`DeSTAR/`](DeSTAR) | `python destar_rank.py --input_dir DATA --new_transcript_file my.txt --new_transcript_name MyGene` |
| **Reproduce the figures in the paper** | [`reproduce/`](reproduce) | `python fig3_pairwise_LOTO.py --input_dir DATA` |

All model parameters are **fixed inside the scripts** (frozen `grid_0005` config, 20 features,
seed offset, control-row handling) — there are no long parameter lists to type. Every script
takes just a data folder (`--input_dir`) and, at most, `--seeds` / `--out_prefix`.

---

## Data

Both parts use the same six per-library feature tables (one per transcript), deposited with
the paper:

```
{T5,T7,Dengue,ENO1,PGK1,Zika}_target_search_library_analysis_features.recomputed.with_pU.curated_txrel_Final_6.xlsx
```

Raw Nanopore reads: **NCBI SRA, BioProject `PRJNA1529981`** (72 samples).
Put the six `*_Final_6.xlsx` files in one folder and pass it as `--input_dir`.

---

## `DeSTAR/` — use the model on your transcript

The deployment tool. Tiles your transcript into 33-nt candidate sites, computes the 20
features (RNAfold + RNAplfold), trains the frozen model on all six libraries, and returns
your sites ranked. This is the exact pipeline used for the paper's prospective
SARS-CoV-2 N test (Fig 6); a worked example is in [`DeSTAR/example/`](DeSTAR/example).
See [`DeSTAR/README.md`](DeSTAR/README.md).

## `reproduce/` — regenerate the paper's numbers

One clean, verified script per figure, covering the full chain from raw reads
(`count_barcodes.py`) through the model (`fig3_pairwise_LOTO.py`) to the figure panels
(`fig2_*`, `fig4_*`, `ed5_pairing.py`). Each script was checked to reproduce the published
values. See [`reproduce/README.md`](reproduce/README.md).

---

## Requirements

Python: `numpy pandas scipy scikit-learn xgboost shap openpyxl` (+ `logomaker matplotlib`
for figure plots, `tqdm` for read counting). ViennaRNA command-line tools (`RNAfold`,
`RNAplfold`) for `DeSTAR/` and `ed5_pairing.py` — **pin `ViennaRNA==2.7.0`**. Per-part
details are in each subfolder's README.
