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
- **Model-input table (primary):** **Supplementary Table 3** of the paper — one deposited
  workbook, sheet `Supplementary Table 3`, 7,383 rows (all six transcripts). Columns include
  `transcript` (paper names: gp8, gp10A, DENV, ZIKV, PGK1, ENO1), `variant_id`,
  `target_start_0based`, `target_33nt`, `Toehold_seq`, `Average_ON/OFF`, `ON/OFF_1..3`, and the
  20 model features. Controls are already excluded and rows are ordered by `variant_id`.
  → **pass `--supp_table3 <path/to/DeSTAR_Supplementary_Table_1-6.xlsx>`** to every Stage-B script.
- **Legacy (equivalent):** the six per-transcript tables
  `<LIB>_..._curated_txrel_Final_6.xlsx` (LIB ∈ {T5, T7, Dengue, ENO1, PGK1, Zika}); non-Zika
  tables carry an appended control row that the loader drops. Verified row-for-row identical to
  Supplementary Table 3. → pass `--input_dir <folder>` instead of `--supp_table3`.

All Stage-B scripts accept either input via the shared loader (`destar_io.py` / `fig3.load_df_map`).

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

### Stage B — model & figures

All Stage-B scripts take `--supp_table3 SUPP` (Supplementary Table 3 workbook; shown as `SUPP`
below) — or the legacy `--input_dir DATA`.

| Script | Figure | Command | Key output |
|---|---|---|---|
| `ed1_single_nt.py` | **ED1** (single-nt sensitivity) | `python ed1_single_nt.py --supp_table3 SUPP --plot` | `*_panelAF_traces.csv`, `*_panelG_boxdata.csv`, `*_panelG_stats.csv` |
| `fig2_heatmap.py` | **Fig 2b** | `python fig2_heatmap.py --supp_table3 SUPP --plot` | `*_data.xlsx` (+ `.svg/.png`) |
| `fig2_sequence_logo.py` | **Fig 2c** | `python fig2_sequence_logo.py --supp_table3 SUPP --plot` | `*_data.xlsx` (+ `.svg/.png`) |
| `fig3_pairwise_LOTO.py` | **Fig 3** | `python fig3_pairwise_LOTO.py --supp_table3 SUPP --seeds 0-9` | `*.all_seed_predictions.csv`, `*.avg_predictions.csv` |
| `fig4_grammar_winrate.py` | **Fig 4d/f** | `python fig4_grammar_winrate.py --supp_table3 SUPP` | `*_winrate.csv` |
| `fig4_shap.py` | **Fig 4a, ED5b** | `python fig4_shap.py --supp_table3 SUPP --seeds 0-9` | `*_per_feature.csv`, `*_group_rollup.csv` |
| `fig4_ablation.py` | **Fig 4b/c, ED5c** | `python fig4_ablation.py --supp_table3 SUPP --seeds 0-9` | `*_per_transcript.csv`, `*_summary.csv`, `*_stats.csv` |
| `ed6_pairing.py` | **ED6b/c** | `python ed6_pairing.py --supp_table3 SUPP` | `*_summary.csv`, `*_perTranscript.csv`, `*_perSensor.csv` |
| `ed6d_bp1_encoding.py` | **ED6d** | `python ed6d_bp1_encoding.py --supp_table3 SUPP --seeds 0-9` | `*_per_transcript.csv`, `*_stats.csv` |

> **Aggregation = Method B.** The pairwise-ranking precision scripts (`fig3_pairwise_LOTO.py`,
> `fig4_ablation.py`, `ed6d_bp1_encoding.py`) aggregate over the 10 seeds the way the deployed model
> does: each candidate's score is **averaged over seeds into one consensus ranking per transcript**,
> and precision@k is computed **once** on that ranking (not per-seed-then-averaged). This matches
> `DeSTAR/destar_rank.py` and the Fig 3 `avg_predictions.csv`.

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
| Supp Table 3 loader | model input **byte-identical** to the six `*_Final_6.xlsx` (features + target, row-for-row) |
| `ed1_single_nt.py` | ED1g sign test: adjacent > noise in **5/6** transcripts (gp8/gp10A/ZIKV/PGK1 ****, ENO1 ***), **DENV ns**; pooled P ≈ 10⁻³³ (all 6) / 10⁻⁹⁰ (excl. DENV) |
| `fig3_pairwise_LOTO.py` | reproduces `baseline_final_model` predictions, per-variant **r = 0.995** (residual = XGB `n_jobs` threading) |
| `fig4_grammar_winrate.py` | matches Fig 4d **exactly** (WWS 0.683 … SSW 0.331) |
| `fig4_ablation.py` | **Method B**: full-20 P@5/10/15 = 0.87/0.77/0.69; bottom-stem removed ΔP@10 = −0.58 across 6/6 transcripts (Wilcoxon P = 0.031); bottom-stem alone P@10 = 0.30 |
| `fig4_shap.py` | per-class fractions & per-feature ranking match Fig 4a/ED5b |
| `fig2_sequence_logo.py` | pooled K = round(0.15·N) = **1,107** top/bottom |
| `ed6_pairing.py` | ED6c **exact**: bp1 17.4%/ΔG 0.96±0.03, bp2 1.5%/ΔG 2.57±0.04 (ViennaRNA 2.7.0) |
| `ed6d_bp1_encoding.py` | **Method B**: adding bp1 identity (pyrimidine/purine or full base) does not improve held-out ranking — all ΔP@k **ns** (ED6d) |
