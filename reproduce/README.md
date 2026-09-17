# DeSTAR — reproducibility code

Code to reproduce the DeSTAR figures from raw Nanopore reads: **raw FASTQ → barcode counts →
ON/OFF → model features → ranking model → figures.** Each step is one clean, self-documenting
script; every script has been checked to reproduce the published numbers (see *Verification*).

DeSTAR (Detectron Sensor TARget-site ranker) is a pairwise machine-learning model that ranks
candidate target sites for toehold-switch–based Detectron RNA sensors. See the manuscript for
biological context.

---

## 1. Dependencies

Python 3.10+ and:

| package | version used / notes |
|---|---|
| numpy, pandas, scipy, openpyxl | recent |
| scikit-learn | recent (Pipeline, SimpleImputer) |
| xgboost | model estimator (`XGBClassifier`) |
| shap | 0.46.0 — Fig 4a (`fig4_shap.py`) |
| ViennaRNA (python bindings) | **2.7.0** — ED5 (`ed5_pairing.py`); pin this (see note) |
| logomaker, matplotlib | optional, only for `--plot` (Fig 2 logo/heatmap) |
| tqdm | progress bars (`count_barcodes.py`) |

```bash
pip install numpy pandas scipy openpyxl scikit-learn xgboost shap ViennaRNA==2.7.0 logomaker matplotlib tqdm
```

> **Version pins that matter.** ED5's terminal base-pair (bp1) probability is sensitive to
> ViennaRNA's coaxial-stacking parameters, which change between point releases — pin
> `ViennaRNA==2.7.0` to reproduce 17.4% exactly. For byte-exact Fig 3 predictions, set XGBoost
> `n_jobs=1` (multithreading perturbs splits at the ~1e-3 level; rankings are unaffected).

---

## 2. Data

- **Raw reads:** deposited on NCBI SRA, BioProject `PRJNA1529981`;
  72 samples, one merged FASTQ per (library × ON/OFF × replicate × RT-DNA/plasmid).
- **Per-library feature tables** (model input), one per transcript:
  `<LIB>_target_search_library_analysis_features.recomputed.with_pU.curated_txrel_Final_6.xlsx`
  (columns include `Toehold_seq`, `target_33nt`, `Average_ON/OFF`, the 20 model features).
  LIB ∈ {T5 (gp8), T7 (gp10A), Dengue (DENV), ENO1, PGK1, Zika (ZIKV)}.
- Non-Zika tables have an appended control row (a Zika-trigger construct) that every script drops.

Set `--input_dir` to the folder holding the six `*_Final_6.xlsx` files.

---

## 3. Pipeline & scripts

Run order top to bottom. `fig4_ablation.py` and `fig4_shap.py` **import `fig3_pairwise_LOTO.py`**,
so keep all `.py` files in the same directory (or on `PYTHONPATH`).

### Stage A — reads → ON/OFF
**`count_barcodes.py`** — one parameterized script for all libraries/runs. Counts exact 13-nt
library barcodes in each sample's reads (sliding window, per-sample cache), then
`norm = raw/total_hits` and `ssDNA_over_plasmid_norm = ssDNA_norm / plasmid_norm`.
```bash
python count_barcodes.py --target Dengue \
  --index_xlsx    npJH013/Dengue_Indexing_barcodes.xlsx \
  --barcodes_xlsx npJH013/Dengue_library_barcodes.xlsx \
  --fastq_pass_dir npJH013/fastq_pass \
  --out_dir npJH013/Dengue_outputs_per_replicate \
  --cache_dir npJH013/cache_counts/Dengue
```
Out: one Excel per (condition × replicate) with `metadata`/`per_barcode`/`per_variant` sheets.
(Replicate QC/selection → `*_ONOFF_QC_final.xlsx`, then feature computation → `*_Final_6.xlsx`,
are the intervening steps that produce the model-input tables.)

**How `Average ON/OFF` is computed.** `count_barcodes.py` outputs, per variant, a
`mean_norm_ratio` (ssDNA ÷ plasmid) for each condition × biological replicate. The sensor
activity used by the model and all figures is then formed as:

- **ON/OFF for biological replicate *i*** = (that ON replicate's `mean_norm_ratio`) ÷
  (the **mean `mean_norm_ratio` across the OFF replicates** — a shared OFF denominator).
- **`Average ON/OFF`** = the mean of ON/OFF across the three biological replicates.

Before averaging, per-variant technical-replicate outliers are removed with a
concordance-fold / Dixon's-Q filter. This ON/OFF/QC step is applied per library between the
counting above and the deposited `*_Final_6.xlsx` tables; those tables (which already carry
`Average ON/OFF`) are the entry point for Stage B, and the raw reads and processed values are
deposited (SRA `PRJNA1529981`; Supplementary Tables) so the pipeline is verifiable end to end.

### Stage B — model & figures (all take `--input_dir <six _Final_6.xlsx>`)

| Script | Figure | Command | Key output |
|---|---|---|---|
| `fig3_pairwise_LOTO.py` | **Fig 3** | `python fig3_pairwise_LOTO.py --input_dir DATA --seeds 0-9` | `*.all_seed_predictions.csv`, `*.avg_predictions.csv` |
| `fig4_grammar_winrate.py` | **Fig 4d/f** | `python fig4_grammar_winrate.py --input_dir DATA` | `*_winrate.csv` |
| `fig4_ablation.py` | **Fig 4b/c, ED4c** | `python fig4_ablation.py --input_dir DATA --seeds 0-9` | `*_alone_and_removed.csv` |
| `fig4_shap.py` | **Fig 4a, ED4b** | `python fig4_shap.py --input_dir DATA --seeds 0-9` | `*_per_feature.csv`, `*_group_rollup.csv` |
| `fig2_sequence_logo.py` | **Fig 2c** | `python fig2_sequence_logo.py --input_dir DATA --plot` | `*_data.xlsx` (+ `.svg/.png`) |
| `fig2_heatmap.py` | **Fig 2b** | `python fig2_heatmap.py --input_dir DATA --plot` | `*_data.xlsx` (+ `.svg/.png`) |
| `ed5_pairing.py` | **ED5b/c** | `python ed5_pairing.py --input_dir DATA` | `*_summary.csv`, `*_perTranscript.csv`, `*_perSensor.csv` |

`--seeds` defaults to `0,1,2,3,4,5,6,7,8,9` (the 10-seed runs). The pairwise LOTO scripts are
compute-heavy (10 seeds × 6 held-out transcripts); a single `--seeds 0` run is a fast sanity check.

**Note — Fig 6** (prospective SARS-CoV-2 N validation, top-10 vs random-20, clonal qPCR + pooled
sequencing) is not model code; its stats (Mann-Whitney, Spearman) are computed directly from the
measured values in `npJH025_20260904/analysis/SARS_top10_vs_random20.xlsx`.

---

## 4. Verification

Each script was checked against the original run outputs:

| Script | Check |
|---|---|
| `count_barcodes.py` | per-barcode raw counts **identical**; ssDNA/plasmid ratio Pearson **r = 1.000000** |
| `fig3_pairwise_LOTO.py` | reproduces `baseline_final_model` predictions, per-variant **r = 0.995** (residual = XGB `n_jobs` threading) |
| `fig4_grammar_winrate.py` | matches Fig 4d **exactly** (WWS 0.683 … SSW 0.331) |
| `fig4_ablation.py` | full-20 P@5/10/15 = 0.87/0.78/0.70 (Fig 4) |
| `fig4_shap.py` | per-class fractions & per-feature ranking match Fig 4a/ED4b |
| `fig2_sequence_logo.py` | pooled K = round(0.15·N) = **1,107** top/bottom |
| `ed5_pairing.py` | ED5c **exact**: bp1 17.4%/ΔG 0.96±0.03, bp2 1.5%/ΔG 2.57±0.04 (ViennaRNA 2.7.0) |
