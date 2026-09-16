# DeSTAR — rank target sites on your own transcript

This is the **deployment** tool. Give it an RNA transcript; it returns the candidate
33-nt target sites ranked best-to-worst for building a Detectron / toehold-switch sensor.
Use the top of the list to decide which few sensors to build (the deployment horizon in
the paper is the top ~5–10 sites).

> Want to reproduce the paper's figures instead? See [`../reproduce/`](../reproduce).

This is the exact pipeline used for the paper's prospective **SARS-CoV-2 N** validation
(Figure 6). Its featurizer reproduces the published SARS-CoV-2 N features bit-for-bit on
ViennaRNA 2.7.0, and its defaults are the frozen, published model settings — running with
no tuning flags gives the same model as the paper.

---

## What it does

1. **Tile** your transcript into every 33-nt candidate target site.
2. **Featurize** each site (the 20 model features): bottom-3-bp stem grammar, hairpin and
   full-toehold folding ΔG (RNAfold), target accessibility at 4 window sizes (RNAplfold),
   GC of target/seed/invasion/context, and relative position in the transcript.
3. **Train** the frozen DeSTAR pairwise ranking model on all six measured libraries
   (10 seeds, averaged).
4. **Rank** your candidate sites by average pairwise-win probability and write the table.

---

## Requirements

- **Python:** `numpy pandas scikit-learn xgboost openpyxl`
- **ViennaRNA command-line tools on your `PATH`:** `RNAfold`, `RNAplfold`
  — **pin `ViennaRNA==2.7.0`** (folding energies shift slightly between releases).
  ```bash
  conda install -c conda-forge -c bioconda viennarna=2.7.0
  pip install numpy pandas scikit-learn xgboost openpyxl
  ```
- **The six training tables** (one per library), the same files the figures use:
  ```
  {T5,T7,Dengue,ENO1,PGK1,Zika}_target_search_library_analysis_features.recomputed.with_pU.curated_txrel_Final_6.xlsx
  ```
  These are deposited with the paper. Put all six in one folder and pass it as `--input_dir`.

---

## Your input transcript

DeSTAR needs the RNA sequence of the transcript you want to detect (the target the sensor
will bind). Provide it one of two ways:

**A file** (`--new_transcript_file my_transcript.txt`). Two formats are accepted:

- **Plain text** — just the sequence, e.g. one line (or several) of letters, nothing else:
  ```
  TATACCATGGGCAGCAGCATGTCTGATAATGGACCCCAAAAT...
  ```
- **FASTA** — a `>` header line followed by the sequence (header lines are ignored):
  ```
  >SARS-CoV-2 N gene
  TATACCATGGGCAGCAGCATGTCTGATAATGGACCCCAAAAT...
  ```

The example [`example/SARSCoV2_N.txt`](example/SARSCoV2_N.txt) is the plain-text form.

**Or inline** (`--new_transcript_seq ACGT...`) for a short sequence, no file needed.

Either way: DNA or RNA is fine (`U` is read as `T`), case doesn't matter, and any character
that isn't A/C/G/T (spaces, numbers, newlines, `N`s) is stripped. The transcript must be at
least 33 nt; DeSTAR scores every 33-nt window, so a length-`L` transcript yields `L − 32`
candidate sites.

## Usage

```bash
python destar_rank.py \
  --input_dir  PATH/TO/SIX_LIBRARY_XLSX \
  --new_transcript_file  my_transcript.txt \
  --new_transcript_name  MyGene \
  --out_prefix  MyGene_destar
```

- `--input_dir` — folder holding the six `*_Final_6.xlsx` training tables (see *Requirements*).
- `--new_transcript_name` — a label for your transcript; used in the output file names/IDs.
- `--out_prefix` — prefix for the output files (below).
- The run takes a while: it featurizes every window (RNAfold/RNAplfold) and trains 10 models.
  For a quick smoke test add `--seeds 0` (single seed) — rankings are close but the paper
  number is the 10-seed average (default).

### Output — look at these

| File | What |
|---|---|
| `<prefix>.ranked_candidates.xlsx` | **start here** — `Top200` sheet is your ranked shortlist |
| `<prefix>.ranked_candidates_avg.csv` | full ranking (all candidates), 10-seed averaged |
| `<prefix>.candidate_features.csv` | the 20 features per candidate (audit trail) |
| `<prefix>.all_seed_predictions.csv` | per-seed scores and ranks |
| `<prefix>.config.json` | exact settings used for the run |

Each ranked row gives the target site (`target_33nt`, 1-based start/end in your transcript),
the designed `Toehold_seq` / `hairpin_seq`, and the averaged score/rank. Build sensors for
the top few candidates.

---

## Switch scaffold (important if you don't use ours)

Each candidate's switch is built as:

```
Toehold_seq = revcomp(target_33nt) + fixed_scaffold + stem1'
hairpin_seq = last 8 nt of revcomp(target_33nt) + fixed_scaffold + stem1'
stem1'      = target_33nt[:8]          (target-derived, not fixed)
```

The **fixed scaffold** (stem2 + loop + stem2') is the leading part of `--toehold_tail`;
its trailing 8 nt are replaced per candidate by the target-derived `stem1'`. The default
`--toehold_tail` and `--hairpin_stem_nt 8` match the switch design used in the paper.
**If your switch uses a different scaffold or stem length, pass `--toehold_tail` /
`--hairpin_stem_nt` accordingly**, or the folding features (and therefore the ranking)
won't match your constructs.

---

## Frozen settings (do not change to reproduce the paper)

Baked in as defaults: grid_0005 (`max_pairs 15000`, `top_focus 0.125/0.95`, `nested 0.90`,
zone weights `2.5/2.0/1.2`), raw hairpin/toehold ΔG features, within-transcript percentile
GC/accessibility, control rows dropped from the five non-Zika libraries, seed offset +12000,
10 seeds. The tuning flags exist for experimentation only.

---

## Example

[`example/`](example) has `SARSCoV2_N.txt` (the transcript from Fig 6) and
`EXPECTED_SARSCoV2_N_top15.csv` (the published top-15). Run:

```bash
python destar_rank.py --input_dir PATH/TO/SIX_LIBRARY_XLSX \
  --new_transcript_file example/SARSCoV2_N.txt --new_transcript_name SARSCoV2_N \
  --out_prefix SARSCoV2_N_check
```

Then run this one-line check — it compares your top-15 to the published top-15 and prints
PASS/FAIL:

```bash
python -c "import pandas as pd; \
mine=pd.read_csv('SARSCoV2_N_check.ranked_candidates_avg.csv').head(15)['candidate_id'].tolist(); \
exp=pd.read_csv('example/EXPECTED_SARSCoV2_N_top15.csv')['candidate_id'].tolist(); \
n=len(set(mine)&set(exp)); \
print(f'{n}/15 of the published top-15 present'); \
print('PASS - your install reproduces the paper' if n>=13 else 'CHECK YOUR INSTALL (is ViennaRNA pinned to 2.7.0?)')"
```

A pass (≥13/15 overlap) means your ViennaRNA + Python setup is correct and you can trust
DeSTAR on your own transcripts. Exact ranks can shift by one or two positions because
XGBoost training is mildly nondeterministic across CPU threads (rankings are unaffected in
practice), so the check allows a small tolerance rather than demanding an identical list.
