#!/usr/bin/env python3
"""
destar_rank.py -- DeSTAR deployment tool: rank target sites on YOUR transcript.

Give DeSTAR a new RNA transcript; it tiles the transcript into all 33-nt candidate
target sites, computes the 20 model features for each site from sequence + folding,
trains the frozen DeSTAR pairwise ranking model on all six measured libraries, and
returns the candidate sites ranked best-to-worst. Use the top of the list to choose
which few sensors to build (the paper's deployment horizon is the top ~5-10).

This is the exact pipeline used for the prospective SARS-CoV-2 N validation in the
paper (Figure 6). The featurizer here reproduces the published SARS-CoV-2 N candidate
features bit-for-bit on ViennaRNA 2.7.0.

Defaults are the FROZEN, PUBLISHED settings -- running with no tuning flags gives the
same model as the paper:
  - Model      : pairwise A_softZoneWeight ranker (XGBoost), raw hairpin/toehold deltaG,
                 within-transcript percentile GC/accessibility features.
  - Grid       : grid_0005 (max_pairs 15000, top_focus 0.125/0.95, nested 0.90,
                 zone weights 2.5/2.0/1.2).  [do NOT change to reproduce the paper]
  - Training   : all six libraries, appended control rows removed, 10 seeds averaged.
  - Seed offset: +12000 per seed (matches the reported model).

Requirements:
  - Python: numpy, pandas, scikit-learn, xgboost, openpyxl
  - ViennaRNA command-line tools on PATH: RNAfold, RNAplfold  (pin 2.7.0)
  - The six training tables (one per library), same files the figures use:
      *_target_search_library_analysis_features.recomputed.with_pU.curated_txrel_Final_6.xlsx
    deposited with the paper; point --input_dir at the folder holding them.

Minimal usage:
  python destar_rank.py --input_dir PATH/TO/SIX_LIBRARY_XLSX \
      --new_transcript_file my_transcript.txt --new_transcript_name MyGene \
      --out_prefix MyGene_destar
  (or pass the sequence directly with --new_transcript_seq ACGT...)

Design note (must match how the libraries were built): each candidate's switch is
  Toehold_seq = revcomp(target_33nt) + fixed_scaffold + stem1'      (stem1' = target_33nt[:8])
  hairpin_seq = last 8 nt of revcomp(target_33nt) + fixed_scaffold + stem1'
The fixed scaffold (stem2 + loop + stem2') is the leading part of --toehold_tail; its
trailing 8 nt are replaced per candidate by the target-derived stem1'. If you use a
different switch scaffold, pass it via --toehold_tail (keep --hairpin_stem_nt = stem length).

Outputs (prefix = --out_prefix):
  <prefix>.ranked_candidates.xlsx   ranked sites (start with the Top200 sheet)
  <prefix>.ranked_candidates_avg.csv, .top200_candidates.csv
  <prefix>.all_seed_predictions.csv per-seed scores/ranks
  <prefix>.candidate_features.csv   the 20 features per candidate (audit trail)
  <prefix>.features_used.csv, .config.json
"""

import argparse
import glob
import itertools
import json
import os
import re
import shutil
import subprocess
import tempfile
from functools import lru_cache
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

KEEP_LIBRARIES = ["T5", "T7", "Dengue", "ENO1", "PGK1", "Zika"]
B3_GRAMMARS = ["SSS", "SSW", "SWS", "SWW", "WSS", "WSW", "WWS", "WWW"]

# Original minimal feature set before the final deltaG replacements.
MINIMAL_20_FEATURES = [
    "b3_SSS", "b3_SSW", "b3_SWS", "b3_SWW",
    "b3_WSS", "b3_WSW", "b3_WWS", "b3_WWW",
    "deltaG_hairpin_pct_in_tx",
    "deltaG_toehold_full_pct_in_tx",
    "pU1_mean_total_33_pct_in_tx",
    "pU1_min_total_33_pct_in_tx",
    "pU33_segment_unpaired_pct_in_tx",
    "pU25_seed_segment_unpaired_pct_in_tx",
    "pU8_invasion_segment_unpaired_pct_in_tx",
    "GC_total_33_pct_in_tx",
    "GC_seed_25_pct_in_tx",
    "GC_invasion_8_pct_in_tx",
    "GC_context_pm50_pct_in_tx",
    "relpos_in_tx",
]
HAIRPIN_PCT_FEATURE = "deltaG_hairpin_pct_in_tx"
TOEHOLD_PCT_FEATURE = "deltaG_toehold_full_pct_in_tx"

DROP_ALWAYS = {
    "variant_id", "Barcode#", "barcode_seq", "Target gene", "target_33nt", "Toehold_seq",
    "hairpin_seq", "hairpin_structure", "toehold_structure", "NAS",
    "ON/OFF_1", "ON/OFF_2", "ON/OFF_3",
    "Average ON/OFF", "Average_ON/OFF", "Average_ON_OFF", "Average ON OFF",
    "usable", "library", "pred", "prob", "oof_prob",
    "y_raw", "y_cont", "y_topq", "fold",
}
EXCLUDE_FEATURE_CONTAINS = ("pred", "prob", "oof", "label")
EXCLUDE_FEATURE_SUFFIXES = ("_topq",)

# FROZEN, PUBLISHED grid_0005 config -- these are the settings behind the reported
# model and the SARS-CoV-2 N (Fig 6) result. The original deployment script shipped a
# different set of defaults; the paper runs overrode them on the command line. They are
# hard-set here so that running with no tuning flags reproduces the paper.
DEFAULT_GRID_CONFIG = {
    "grid_id": "grid_0005",
    "strategy": "A_softZoneWeight",
    "max_pairs_per_tx": 15000,
    "min_abs_diff": 0.10,
    "top_focus_frac": 0.125,
    "top_focus_prob": 0.95,
    "top_focus_mode": "one_in_top",
    "nested_frac": 0.90,
    "elite_strong_w": 2.5,
    "elite_border_w": 2.0,
    "strong_border_w": 1.2,
}

FINAL_MODEL_NAME = "final_train6_predict_unseen_hairpin_abs_toehold_abs"

# Default fixed tail inferred from the user's existing Detectron design example:
# Toehold_seq = reverse_complement(target_33nt) + this fixed tail
# hairpin_seq = last 8 nt of reverse_complement(target_33nt) + this fixed tail
DEFAULT_TOEHOLD_TAIL = "TTAGAACACACCAGATAAATGGCTAG"


def parse_list(s, typ=str):
    if s is None or str(s).strip() == "":
        return []
    return [typ(x.strip()) for x in str(s).split(",") if x.strip()]


def normalize_colname(x: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(x).strip().lower())


def _sanitize_keep_only_acgt(seq) -> str:
    if seq is None or (isinstance(seq, float) and np.isnan(seq)):
        return ""
    s = str(seq).strip().upper().replace("U", "T")
    return re.sub(r"[^ACGT]", "", s)


def _sanitize_for_rna(seq) -> str:
    if seq is None or (isinstance(seq, float) and np.isnan(seq)):
        return ""
    s = str(seq).strip().upper().replace("U", "T")
    return re.sub(r"[^ACGT]", "A", s)


def reverse_complement_dna(seq: str) -> str:
    table = str.maketrans("ACGTacgt", "TGCAtgca")
    return _sanitize_keep_only_acgt(seq).translate(table)[::-1].upper()


def _gc_frac(seq: str) -> float:
    s = _sanitize_keep_only_acgt(seq)
    if not s:
        return np.nan
    return (s.count("G") + s.count("C")) / len(s)


def b3_grammar_from_hairpin(hairpin_seq) -> str:
    s = _sanitize_for_rna(hairpin_seq)
    if len(s) < 3:
        return np.nan
    return "".join("W" if nt in ("A", "T") else "S" for nt in s[:3])


def check_tool_exists(tool: str) -> None:
    if shutil.which(tool) is None:
        raise RuntimeError(
            f"Required ViennaRNA tool not found in PATH: {tool}\n"
            "Install/activate ViennaRNA first, for example with conda:\n"
            "  conda install -c conda-forge -c bioconda viennarna"
        )


@lru_cache(maxsize=300000)
def rnafold_energy_and_structure(seq_dna: str) -> Tuple[float, str]:
    check_tool_exists("RNAfold")
    seq = _sanitize_for_rna(seq_dna).replace("T", "U")
    if not seq:
        return (np.nan, "")
    p = subprocess.run(
        ["RNAfold", "--noPS"],
        input=(seq + "\n").encode(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if p.returncode != 0:
        return (np.nan, "")
    lines = p.stdout.decode().strip().splitlines()
    if len(lines) < 2:
        return (np.nan, "")
    line = lines[1].strip()
    struct = line.split()[0] if line.split() else ""
    m = re.search(r"\(\s*([-0-9.]+)\s*\)\s*$", line)
    if not m:
        return (np.nan, struct)
    try:
        dg = float(m.group(1))
    except ValueError:
        return (np.nan, struct)
    return (dg, struct)


def run_rnaplfold_lunp(seq_dna: str, W: int, L: int, u: int, T: float) -> np.ndarray:
    check_tool_exists("RNAplfold")
    seq = _sanitize_keep_only_acgt(seq_dna).replace("T", "U")
    if not seq:
        raise ValueError("Transcript sequence is empty after sanitization.")
    with tempfile.TemporaryDirectory(prefix="plfold_") as td:
        cmd = ["RNAplfold", "-W", str(W), "-L", str(L), "-u", str(u), "-T", str(T)]
        p = subprocess.run(
            cmd,
            input=(seq + "\n").encode(),
            cwd=td,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if p.returncode != 0:
            raise RuntimeError(
                "RNAplfold failed.\n"
                f"CMD: {' '.join(cmd)}\n"
                f"STDERR: {p.stderr.decode(errors='ignore')}\n"
            )
        lunp_path = os.path.join(td, "plfold_lunp")
        if not os.path.exists(lunp_path):
            raise RuntimeError("RNAplfold did not produce plfold_lunp.")
        n = len(seq)
        mat = np.full((n + 1, u + 1), np.nan, dtype=float)
        with open(lunp_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                try:
                    i = int(parts[0])
                except ValueError:
                    continue
                vals = parts[1:]
                if len(vals) < u:
                    vals += ["nan"] * (u - len(vals))
                for k_ in range(1, u + 1):
                    try:
                        mat[i, k_] = float(vals[k_ - 1])
                    except ValueError:
                        mat[i, k_] = np.nan
        return mat


def read_transcript_from_file(path: str) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Transcript file not found: {path}")
    lines = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(">"):
                continue
            lines.append(line)
    seq = _sanitize_keep_only_acgt("".join(lines))
    if not seq:
        raise ValueError("Transcript file is empty after A/C/G/T sanitization.")
    return seq


def add_tx_relative_columns(df: pd.DataFrame, raw_cols: List[str]) -> pd.DataFrame:
    """
    Match previous curated_txrel behavior: pandas rank percentile within transcript.
    Smaller raw values get lower percentile; larger raw values get higher percentile.
    """
    out = df.copy()
    for col in raw_cols:
        x = pd.to_numeric(out[col], errors="coerce")
        out[f"{col}_pct_in_tx"] = x.rank(method="average", pct=True)
    return out


def make_unseen_candidate_features(
    transcript: str,
    transcript_name: str,
    toehold_tail: str,
    hairpin_stem_nt: int,
    plfold_W: int,
    plfold_L: int,
    temperature_c: float,
) -> pd.DataFrame:
    transcript = _sanitize_keep_only_acgt(transcript)
    toehold_tail = _sanitize_keep_only_acgt(toehold_tail)
    if len(transcript) < 33:
        raise ValueError("New transcript must be at least 33 nt.")
    if not toehold_tail:
        raise ValueError("--toehold_tail is empty after sanitization.")
    if hairpin_stem_nt <= 0 or hairpin_stem_nt > 33:
        raise ValueError("--hairpin_stem_nt must be between 1 and 33.")

    n = len(transcript)
    # FIX: stem1' (the last hairpin_stem_nt of the construct) is TARGET-DERIVED, not fixed.
    # In the real libraries stem1' == revcomp(stem1) == target_33nt[:hairpin_stem_nt] (verified 7382/7382).
    # The supplied tail's leading part is the true fixed scaffold (stem2 + loop + stem2'); its
    # trailing hairpin_stem_nt were a hardcoded stem1' and must be replaced per candidate.
    scaffold = toehold_tail[:-hairpin_stem_nt] if len(toehold_tail) > hairpin_stem_nt else toehold_tail
    rows = []
    for s0 in range(0, n - 33 + 1):
        target_33 = transcript[s0:s0 + 33]
        rc33 = reverse_complement_dna(target_33)
        stem1_prime = target_33[:hairpin_stem_nt]          # = revcomp(stem1)
        toehold_seq = rc33 + scaffold + stem1_prime
        hairpin_seq = rc33[-hairpin_stem_nt:] + scaffold + stem1_prime
        rows.append({
            "transcript": transcript_name,
            "candidate_id": f"{transcript_name}_pos{s0+1:04d}_{s0+33:04d}",
            "target_start_0based": s0,
            "target_start_1based": s0 + 1,
            "target_end_1based": s0 + 33,
            "relpos_in_tx": 0.0 if (n - 33) == 0 else float(s0 / (n - 33)),
            "target_33nt": target_33,
            "target_33nt_revcomp": rc33,
            "Toehold_seq": toehold_seq,
            "hairpin_seq": hairpin_seq,
        })
    df = pd.DataFrame(rows)

    # b3 motif one-hot features
    df["b3_grammar"] = df["hairpin_seq"].apply(b3_grammar_from_hairpin)
    for g in B3_GRAMMARS:
        df[f"b3_{g}"] = (df["b3_grammar"] == g).astype(int)

    # Raw GC features
    df["GC_total_33"] = df["target_33nt"].apply(_gc_frac)
    df["GC_invasion_8"] = df["target_33nt"].str.slice(0, 8).apply(_gc_frac)
    df["GC_seed_25"] = df["target_33nt"].str.slice(8, 33).apply(_gc_frac)
    df["GC_context_pm50"] = np.nan
    for idx, row in df.iterrows():
        s0 = int(row["target_start_0based"])
        left = max(0, s0 - 50)
        right = min(n, s0 + 33 + 50)
        df.at[idx, "GC_context_pm50"] = _gc_frac(transcript[left:right])

    # Sensor-region folding deltaG features
    df["deltaG_hairpin"] = np.nan
    df["hairpin_structure"] = ""
    df["deltaG_toehold_full"] = np.nan
    df["toehold_structure"] = ""
    for idx, row in df.iterrows():
        dg, st = rnafold_energy_and_structure(row["hairpin_seq"])
        df.at[idx, "deltaG_hairpin"] = dg
        df.at[idx, "hairpin_structure"] = st
        dg, st = rnafold_energy_and_structure(row["Toehold_seq"])
        df.at[idx, "deltaG_toehold_full"] = dg
        df.at[idx, "toehold_structure"] = st

    # Target RNA accessibility features from RNAplfold
    print("Running RNAplfold for u=1,8,25,33 on new transcript...", flush=True)
    lunp_u1 = run_rnaplfold_lunp(transcript, W=plfold_W, L=plfold_L, u=1, T=temperature_c)
    lunp_u8 = run_rnaplfold_lunp(transcript, W=plfold_W, L=plfold_L, u=8, T=temperature_c)
    lunp_u25 = run_rnaplfold_lunp(transcript, W=plfold_W, L=plfold_L, u=25, T=temperature_c)
    lunp_u33 = run_rnaplfold_lunp(transcript, W=plfold_W, L=plfold_L, u=33, T=temperature_c)

    for c in [
        "pU1_mean_total_33", "pU1_min_total_33",
        "pU33_segment_unpaired", "pU25_seed_segment_unpaired", "pU8_invasion_segment_unpaired",
    ]:
        df[c] = np.nan

    for idx, row in df.iterrows():
        s0 = int(row["target_start_0based"])
        i0 = s0 + 1
        i_seed = (s0 + 8) + 1
        pu1_total = np.array([lunp_u1[i0 + j, 1] for j in range(33)], dtype=float)
        df.at[idx, "pU1_mean_total_33"] = np.nanmean(pu1_total)
        df.at[idx, "pU1_min_total_33"] = np.nanmin(pu1_total)
        df.at[idx, "pU33_segment_unpaired"] = lunp_u33[i0, 33]
        df.at[idx, "pU8_invasion_segment_unpaired"] = lunp_u8[i0, 8]
        df.at[idx, "pU25_seed_segment_unpaired"] = lunp_u25[i_seed, 25]

    # Percentile-normalized target-site features within this new transcript.
    pct_raw_cols = [
        "GC_total_33", "GC_seed_25", "GC_invasion_8", "GC_context_pm50",
        "pU1_mean_total_33", "pU1_min_total_33",
        "pU33_segment_unpaired", "pU25_seed_segment_unpaired", "pU8_invasion_segment_unpaired",
    ]
    df = add_tx_relative_columns(df, pct_raw_cols)
    return df


def resolve_target_col(df: pd.DataFrame, target_col: str) -> str:
    if target_col in df.columns:
        return target_col
    wanted = normalize_colname(target_col)
    col_map = {normalize_colname(c): c for c in df.columns if isinstance(c, str)}
    if wanted in col_map:
        return col_map[wanted]
    for fb in ["Average_ON/OFF", "Average ON/OFF", "Average_ON_OFF", "Average ON OFF"]:
        if fb in df.columns:
            return fb
    raise KeyError(f"Target column '{target_col}' not found")


def resolve_input_files(input_dir: str, pattern: str) -> List[Tuple[str, str]]:
    paths = sorted(glob.glob(os.path.join(input_dir, pattern)))
    paths = [p for p in paths if not os.path.basename(p).startswith("~$")]
    if not paths:
        raise FileNotFoundError(f"No files found in '{input_dir}' matching '{pattern}'")
    out, seen = [], set()
    for p in paths:
        fname = os.path.basename(p)
        lib = fname.split("_")[0].strip()
        if lib not in KEEP_LIBRARIES:
            continue
        if lib in seen:
            raise ValueError(f"Duplicate library detected: {lib}. File: {p}")
        seen.add(lib)
        out.append((lib, p))
    missing = [x for x in KEEP_LIBRARIES if x not in [a for a, _ in out]]
    if missing:
        raise FileNotFoundError(f"Missing required libraries: {missing}. Found: {[a for a, _ in out]}")
    return sorted(out, key=lambda x: KEEP_LIBRARIES.index(x[0]))


def exclude_last_control_rows(df_map: Dict[str, pd.DataFrame], libs_to_exclude: List[str]) -> Tuple[Dict[str, pd.DataFrame], Dict[str, int]]:
    libs_to_exclude = set(libs_to_exclude)
    out, removed = {}, {}
    for lib, df in df_map.items():
        if lib in libs_to_exclude and len(df) > 0:
            out[lib] = df.iloc[:-1].reset_index(drop=True).copy()
            removed[lib] = 1
        else:
            out[lib] = df.reset_index(drop=True).copy()
            removed[lib] = 0
    return out, removed


def choose_feature_columns(df: pd.DataFrame, target_col: str) -> List[str]:
    target_col_resolved = resolve_target_col(df, target_col)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    chosen = []
    for c in numeric_cols:
        if c == target_col_resolved:
            continue
        if c in DROP_ALWAYS:
            continue
        if any(tok in c.lower() for tok in EXCLUDE_FEATURE_CONTAINS):
            continue
        if any(c.endswith(suf) for suf in EXCLUDE_FEATURE_SUFFIXES):
            continue
        chosen.append(c)
    return chosen


def common_features(df_map: Dict[str, pd.DataFrame], libs: List[str], target_col: str) -> List[str]:
    common = None
    for lib in libs:
        s = set(choose_feature_columns(df_map[lib], target_col))
        common = s if common is None else common & s
    if not common:
        raise ValueError("No common usable features across libraries")
    return sorted(common)


def detect_raw_hairpin_feature(common: List[str], preferred: str = "auto") -> str:
    common_set = set(common)
    if preferred and preferred != "auto":
        if preferred not in common_set:
            raise KeyError(f"Requested raw hairpin feature not found: {preferred}")
        return preferred
    for c in ["deltaG_hairpin", "hairpin_deltaG", "hairpin_dG", "dG_hairpin", "deltaG_hairpin_raw"]:
        if c in common_set:
            return c
    fuzzy = []
    for c in common:
        lc = c.lower()
        if "hairpin" in lc and any(tok in lc for tok in ["deltag", "delta_g", "dg", "mfe"]):
            if not any(tok in lc for tok in ["pct", "percentile", "rank", "_z", "zin", "z_in"]) and not c.endswith("_in_tx"):
                fuzzy.append(c)
    if len(fuzzy) == 1:
        return fuzzy[0]
    raise KeyError(f"Could not detect raw hairpin deltaG feature. Candidates: {fuzzy}")


def detect_raw_toehold_feature(common: List[str], preferred: str = "auto") -> str:
    common_set = set(common)
    if preferred and preferred != "auto":
        if preferred not in common_set:
            raise KeyError(f"Requested raw toehold feature not found: {preferred}")
        return preferred
    for c in ["deltaG_toehold_full", "toehold_full_deltaG", "toehold_deltaG_full", "deltaG_toehold", "toehold_deltaG", "deltaG_toehold_full_raw"]:
        if c in common_set:
            return c
    fuzzy = []
    for c in common:
        lc = c.lower()
        if "toehold" in lc and any(tok in lc for tok in ["deltag", "delta_g", "dg", "mfe"]):
            if not any(tok in lc for tok in ["pct", "percentile", "rank", "_z", "zin", "z_in", "delta_from_tx_mean"]) and not c.endswith("_in_tx"):
                fuzzy.append(c)
    if len(fuzzy) == 1:
        return fuzzy[0]
    raise KeyError(f"Could not detect raw toehold deltaG feature. Candidates: {fuzzy}")


def resolve_final_feature_set(common: List[str], raw_hairpin_feature: str, raw_toehold_feature: str) -> List[str]:
    common_set = set(common)
    feats = [
        raw_hairpin_feature if f == HAIRPIN_PCT_FEATURE else
        raw_toehold_feature if f == TOEHOLD_PCT_FEATURE else
        f
        for f in MINIMAL_20_FEATURES
    ]
    dedup = []
    seen = set()
    for f in feats:
        if f not in seen:
            dedup.append(f)
            seen.add(f)
    missing = [f for f in dedup if f not in common_set]
    if missing:
        raise KeyError(f"Final feature set is missing {len(missing)} feature(s): {missing}")
    return dedup


def build_feature_matrix(df: pd.DataFrame, target_col: str, feature_cols: List[str]):
    target = resolve_target_col(df, target_col)
    y_raw = pd.to_numeric(df[target], errors="coerce").astype(float).values
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing feature columns: {missing[:20]}")
    return df[feature_cols].copy(), y_raw


def apply_raw_transform(y_raw: np.ndarray, raw_transform: str) -> np.ndarray:
    if raw_transform == "none":
        return y_raw.astype(float).copy()
    out = np.full_like(y_raw.astype(float), np.nan, dtype=float)
    mask = ~np.isnan(y_raw)
    out[mask] = np.log2(np.clip(y_raw[mask], 1e-6, None))
    return out


def compute_percentiles_desc(y_cont):
    valid = ~np.isnan(y_cont)
    frac = np.full(len(y_cont), np.nan)
    if valid.sum() == 0:
        return frac
    vals = y_cont[valid]
    order = np.argsort(-vals, kind="mergesort")
    rank = np.empty(len(vals), dtype=float)
    rank[order] = np.arange(1, len(vals) + 1)
    frac_vals = np.array([1.0]) if len(vals) == 1 else 1.0 - (rank - 1.0) / (len(vals) - 1.0)
    frac[valid] = frac_vals
    return frac


def assign_zone_15(y_cont):
    frac = compute_percentiles_desc(y_cont)
    zone = np.array(["rest"] * len(y_cont), dtype=object)
    valid = ~np.isnan(frac)
    zone[(frac >= 0.95) & valid] = "elite"
    zone[(frac >= 0.90) & (frac < 0.95) & valid] = "strong"
    zone[(frac >= 0.85) & (frac < 0.90) & valid] = "border"
    return zone


def make_model(model_name, seed):
    if model_name == "xgb" and HAS_XGB:
        model = XGBClassifier(
            n_estimators=500,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=seed,
            n_jobs=8,
        )
        return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", model)])
    model = HistGradientBoostingClassifier(learning_rate=0.05, max_depth=6, max_iter=500, random_state=seed)
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", model)])


def _sample_pair_indices_random(n, yv, max_pairs, min_abs_diff, rng):
    pairs = set()
    attempts, max_attempts = 0, max_pairs * 40
    while len(pairs) < max_pairs and attempts < max_attempts:
        i, j = int(rng.integers(0, n)), int(rng.integers(0, n))
        attempts += 1
        if i == j:
            continue
        a, b = (i, j) if i < j else (j, i)
        if abs(yv[a] - yv[b]) < min_abs_diff:
            continue
        pairs.add((a, b))
    return list(pairs)


def _sample_pair_indices_top_focus(n, yv, max_pairs, min_abs_diff, rng, top_focus_frac, top_focus_mode):
    order = np.argsort(-yv)
    n_top = max(1, int(np.ceil(top_focus_frac * n)))
    top = np.array(sorted(set(order[:n_top].tolist())), dtype=int)
    rest = np.array(sorted(set(range(n)) - set(top.tolist())), dtype=int)
    pairs = set()
    attempts, max_attempts = 0, max_pairs * 60
    while len(pairs) < max_pairs and attempts < max_attempts:
        attempts += 1
        if top_focus_mode == "top_vs_rest":
            if len(top) == 0 or len(rest) == 0:
                break
            i, j = int(top[rng.integers(0, len(top))]), int(rest[rng.integers(0, len(rest))])
        elif top_focus_mode == "top_vs_top":
            if len(top) < 2:
                break
            i, j = int(top[rng.integers(0, len(top))]), int(top[rng.integers(0, len(top))])
            if i == j:
                continue
        else:  # one_in_top
            if len(top) == 0:
                break
            i, j = int(top[rng.integers(0, len(top))]), int(rng.integers(0, n))
            if i == j:
                continue
        a, b = (i, j) if i < j else (j, i)
        if abs(yv[a] - yv[b]) < min_abs_diff:
            continue
        pairs.add((a, b))
    return list(pairs)


def _sample_nested_top_pairs(yv, max_pairs, min_abs_diff, rng, nested_frac):
    n = len(yv)
    zones = assign_zone_15(yv)
    idx = {
        "elite": np.where(zones == "elite")[0],
        "strong": np.where(zones == "strong")[0],
        "border": np.where(zones == "border")[0],
        "rest": np.where(zones == "rest")[0],
    }
    pair_types = [
        ("elite", "strong", 0.35),
        ("elite", "border", 0.25),
        ("strong", "border", 0.20),
        ("elite", "rest", 0.10),
        ("strong", "rest", 0.05),
        ("border", "rest", 0.05),
    ]
    target_pairs = int(round(max_pairs * nested_frac))
    pairs = set()
    attempts, max_attempts = 0, target_pairs * 90
    while len(pairs) < target_pairs and attempts < max_attempts:
        attempts += 1
        r, cum, chosen = rng.random(), 0.0, None
        for a, b, w in pair_types:
            cum += w
            if r <= cum:
                chosen = (a, b)
                break
        if chosen is None:
            chosen = ("elite", "strong")
        za, zb = chosen
        if len(idx[za]) == 0 or len(idx[zb]) == 0:
            i, j = int(rng.integers(0, n)), int(rng.integers(0, n))
            if i == j:
                continue
        else:
            i, j = int(idx[za][rng.integers(0, len(idx[za]))]), int(idx[zb][rng.integers(0, len(idx[zb]))])
        a, b = (i, j) if i < j else (j, i)
        if a == b or abs(yv[a] - yv[b]) < min_abs_diff:
            continue
        pairs.add((a, b))
    return list(pairs), zones


def zone_weight_for_pair(zones, i, j, cfg):
    pair = tuple(sorted([zones[i], zones[j]]))
    return {
        ("elite", "strong"): cfg["elite_strong_w"],
        ("border", "elite"): cfg["elite_border_w"],
        ("border", "strong"): cfg["strong_border_w"],
    }.get(pair, 1.0)


def build_pairs_A(X, y_cont, cfg, seed):
    rng = np.random.default_rng(seed)
    valid = ~np.isnan(y_cont)
    Xv = X.loc[valid].reset_index(drop=True)
    yv = y_cont[valid]
    n = len(Xv)
    if n < 2:
        return pd.DataFrame(columns=X.columns), np.array([], dtype=int), np.array([], dtype=float)

    nested_pairs, zones = _sample_nested_top_pairs(yv, cfg["max_pairs_per_tx"], cfg["min_abs_diff"], rng, cfg["nested_frac"])
    n_focus = int(round(cfg["max_pairs_per_tx"] * cfg["top_focus_prob"]))
    n_random_budget = cfg["max_pairs_per_tx"] - len(nested_pairs)
    focus_pairs = _sample_pair_indices_top_focus(
        n, yv, n_focus, cfg["min_abs_diff"], rng, cfg["top_focus_frac"], cfg["top_focus_mode"]
    )
    random_pairs = _sample_pair_indices_random(
        n, yv, max(n_random_budget - len(focus_pairs), 0), cfg["min_abs_diff"], rng
    )
    pair_idx = list(set(nested_pairs) | set(focus_pairs) | set(random_pairs))

    X_arr = Xv.to_numpy(dtype=float)
    cols = Xv.columns.tolist()
    X_pairs, y_pairs, w_pairs = [], [], []
    for i, j in pair_idx:
        diff = X_arr[i] - X_arr[j]
        w = zone_weight_for_pair(zones, i, j, cfg)
        if yv[i] > yv[j]:
            X_pairs += [diff, -diff]
            y_pairs += [1, 0]
            w_pairs += [w, w]
        else:
            X_pairs += [diff, -diff]
            y_pairs += [0, 1]
            w_pairs += [w, w]
    return pd.DataFrame(np.asarray(X_pairs, dtype=float), columns=cols), np.asarray(y_pairs), np.asarray(w_pairs, dtype=float)


def fit_model_all_libraries(df_map, libs, target_col, raw_transform, feature_cols, cfg, seed, model_name):
    Xs, ys, ws = [], [], []
    for idx, lib in enumerate(libs):
        X, y_raw = build_feature_matrix(df_map[lib], target_col, feature_cols)
        y_cont = apply_raw_transform(y_raw, raw_transform)
        Xp, yp, wp = build_pairs_A(X, y_cont, cfg, seed + idx)
        if len(yp):
            Xs.append(Xp)
            ys.append(yp)
            ws.append(wp)
    if not Xs:
        raise ValueError("No pairwise training data generated")
    Xtr, ytr, wtr = pd.concat(Xs, ignore_index=True), np.concatenate(ys), np.concatenate(ws)
    model = make_model(model_name, seed)
    model.fit(Xtr, ytr, model__sample_weight=wtr)
    return model, {"n_pair_rows": int(len(ytr)), "n_pair_examples_before_mirroring_approx": int(len(ytr) // 2)}


def score_unseen_transcript_by_pairwise_wins(model, X_test, chunk_size=512):
    cols = X_test.columns.tolist()
    X = X_test.to_numpy(dtype=float)
    n = X.shape[0]
    if n == 0:
        return np.array([], dtype=float)
    if n == 1:
        return np.array([1.0], dtype=float)
    scores = np.zeros(n, dtype=float)
    for i in range(n):
        probs = []
        xi = X[i:i + 1]
        others = [j for j in range(n) if j != i]
        for start in range(0, len(others), chunk_size):
            idx = others[start:start + chunk_size]
            diffs = pd.DataFrame(xi - X[idx], columns=cols)
            probs.append(model.predict_proba(diffs)[:, 1])
        scores[i] = float(np.mean(np.concatenate(probs))) if probs else np.nan
        if (i + 1) % 100 == 0 or (i + 1) == n:
            print(f"    scored {i + 1}/{n} candidates", flush=True)
    return scores


def write_outputs(prefix: str, avg_df: pd.DataFrame, all_seed_df: pd.DataFrame, feature_df: pd.DataFrame, candidate_features: pd.DataFrame, config: dict):
    avg_df.to_csv(f"{prefix}.ranked_candidates_avg.csv", index=False)
    all_seed_df.to_csv(f"{prefix}.all_seed_predictions.csv", index=False)
    candidate_features.to_csv(f"{prefix}.candidate_features.csv", index=False)
    feature_df.to_csv(f"{prefix}.features_used.csv", index=False)
    avg_df.head(200).to_csv(f"{prefix}.top200_candidates.csv", index=False)
    with open(f"{prefix}.config.json", "w") as f:
        json.dump(config, f, indent=2)

    with pd.ExcelWriter(f"{prefix}.ranked_candidates.xlsx", engine="openpyxl") as writer:
        avg_df.to_excel(writer, sheet_name="Ranked_candidates_avg", index=False)
        avg_df.head(200).to_excel(writer, sheet_name="Top200", index=False)
        all_seed_df.to_excel(writer, sheet_name="All_seed_predictions", index=False)
        feature_df.to_excel(writer, sheet_name="Features_used", index=False)
        pd.DataFrame([config]).to_excel(writer, sheet_name="Config", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--file_pattern", default="*_target_search_library_analysis_features.recomputed.with_pU.curated_txrel_Final_6.xlsx")
    ap.add_argument("--target_col", default="Average_ON/OFF")
    ap.add_argument("--raw_transform", choices=["none", "log2"], default="log2")
    ap.add_argument("--model", choices=["xgb", "hgb"], default="xgb" if HAS_XGB else "hgb")
    ap.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    ap.add_argument("--out_prefix", default="destar_ranked_candidates")

    ap.add_argument("--new_transcript_name", default="SARSCoV2_N")
    ap.add_argument("--new_transcript_file", default="")
    ap.add_argument("--new_transcript_seq", default="")
    ap.add_argument("--toehold_tail", default=DEFAULT_TOEHOLD_TAIL)
    ap.add_argument("--hairpin_stem_nt", type=int, default=8)
    ap.add_argument("--plfold_W", type=int, default=150)
    ap.add_argument("--plfold_L", type=int, default=150)
    ap.add_argument("--temperature_c", type=float, default=37.0)

    # Final grid_0005 defaults.
    ap.add_argument("--max_pairs_per_tx", type=int, default=DEFAULT_GRID_CONFIG["max_pairs_per_tx"])
    ap.add_argument("--min_abs_diff", type=float, default=DEFAULT_GRID_CONFIG["min_abs_diff"])
    ap.add_argument("--top_focus_frac", type=float, default=DEFAULT_GRID_CONFIG["top_focus_frac"])
    ap.add_argument("--top_focus_prob", type=float, default=DEFAULT_GRID_CONFIG["top_focus_prob"])
    ap.add_argument("--top_focus_mode", default=DEFAULT_GRID_CONFIG["top_focus_mode"])
    ap.add_argument("--nested_frac", type=float, default=DEFAULT_GRID_CONFIG["nested_frac"])
    ap.add_argument("--elite_strong_w", type=float, default=DEFAULT_GRID_CONFIG["elite_strong_w"])
    ap.add_argument("--elite_border_w", type=float, default=DEFAULT_GRID_CONFIG["elite_border_w"])
    ap.add_argument("--strong_border_w", type=float, default=DEFAULT_GRID_CONFIG["strong_border_w"])
    ap.add_argument("--grid_id_label", default="grid_0005")

    ap.add_argument("--raw_hairpin_feature", default="auto")
    ap.add_argument("--raw_toehold_feature", default="auto")
    ap.add_argument("--exclude_last_control_for", default="T5,T7,Dengue,ENO1,PGK1",
                    help="Libraries whose appended last row is a control region to drop from training. "
                         "Default drops it for the five non-Zika libraries (matches the paper); Zika has no control row.")
    args = ap.parse_args()

    seeds = parse_list(args.seeds, int)
    if not seeds:
        raise ValueError("At least one seed is required.")

    files = resolve_input_files(args.input_dir, args.file_pattern)
    df_map_raw = {lib: pd.read_excel(path, engine="openpyxl") for lib, path in files}
    libs = [lib for lib, _ in files]
    exclude_libs = parse_list(args.exclude_last_control_for, str)
    df_map, removed_controls = exclude_last_control_rows(df_map_raw, exclude_libs)

    common = common_features(df_map, libs, args.target_col)
    raw_hairpin = detect_raw_hairpin_feature(common, args.raw_hairpin_feature)
    raw_toehold = detect_raw_toehold_feature(common, args.raw_toehold_feature)
    feature_cols = resolve_final_feature_set(common, raw_hairpin, raw_toehold)

    cfg = {
        "grid_id": args.grid_id_label,
        "strategy": "A_softZoneWeight",
        "max_pairs_per_tx": args.max_pairs_per_tx,
        "min_abs_diff": args.min_abs_diff,
        "top_focus_frac": args.top_focus_frac,
        "top_focus_prob": args.top_focus_prob,
        "top_focus_mode": args.top_focus_mode,
        "nested_frac": args.nested_frac,
        "elite_strong_w": args.elite_strong_w,
        "elite_border_w": args.elite_border_w,
        "strong_border_w": args.strong_border_w,
    }

    if args.new_transcript_file:
        transcript = read_transcript_from_file(args.new_transcript_file)
    elif args.new_transcript_seq:
        transcript = _sanitize_keep_only_acgt(args.new_transcript_seq)
    else:
        raise ValueError("Provide either --new_transcript_file or --new_transcript_seq.")

    print("\nDetected training libraries:")
    for lib, path in files:
        print(f" - {lib}: {path}")
    if exclude_libs:
        print("\nExcluded last-row controls:")
        for lib in libs:
            print(f" - {lib}: removed {removed_controls.get(lib, 0)} row(s); kept {len(df_map[lib])} row(s)")
    else:
        print("\nControls: included in training")

    print(f"\nFinal model name: {FINAL_MODEL_NAME}")
    print(f"Grid: {args.grid_id_label}")
    print(f"Detected raw hairpin deltaG feature: {raw_hairpin}")
    print(f"Detected raw toehold deltaG feature: {raw_toehold}")
    print(f"Final feature count: {len(feature_cols)}")
    for i, f in enumerate(feature_cols, start=1):
        print(f" {i:02d}. {f}")
    print(f"\nNew transcript: {args.new_transcript_name}, length {len(transcript)} nt")
    print(f"Candidate windows: {len(transcript) - 33 + 1}")

    candidate_features = make_unseen_candidate_features(
        transcript=transcript,
        transcript_name=args.new_transcript_name,
        toehold_tail=args.toehold_tail,
        hairpin_stem_nt=args.hairpin_stem_nt,
        plfold_W=args.plfold_W,
        plfold_L=args.plfold_L,
        temperature_c=args.temperature_c,
    )

    missing_unseen = [c for c in feature_cols if c not in candidate_features.columns]
    if missing_unseen:
        raise KeyError(f"New transcript candidate feature table is missing final model features: {missing_unseen}")
    X_new = candidate_features[feature_cols].copy()

    all_seed_predictions = []
    train_logs = []
    for seed in seeds:
        print(f"\n=== Training all-six final model, seed {seed} ===", flush=True)
        model, log = fit_model_all_libraries(
            df_map=df_map,
            libs=libs,
            target_col=args.target_col,
            raw_transform=args.raw_transform,
            feature_cols=feature_cols,
            cfg=cfg,
            seed=seed + 12000,
            model_name=args.model,
        )
        log.update({"seed": seed})
        train_logs.append(log)
        print(f"  Training pair rows: {log['n_pair_rows']}")
        print(f"  Scoring unseen transcript, seed {seed}...", flush=True)
        scores = score_unseen_transcript_by_pairwise_wins(model, X_new)
        pred = candidate_features[[
            "transcript", "candidate_id", "target_start_0based", "target_start_1based", "target_end_1based",
            "relpos_in_tx", "target_33nt", "target_33nt_revcomp", "Toehold_seq", "hairpin_seq",
            "b3_grammar", "deltaG_hairpin", "deltaG_toehold_full",
        ]].copy()
        pred["seed"] = seed
        pred["pred_score"] = scores
        pred["pred_rank"] = pd.Series(-pred["pred_score"]).rank(method="first").astype(int).values
        all_seed_predictions.append(pred)

    all_seed_df = pd.concat(all_seed_predictions, ignore_index=True)

    id_cols = [
        "transcript", "candidate_id", "target_start_0based", "target_start_1based", "target_end_1based",
        "relpos_in_tx", "target_33nt", "target_33nt_revcomp", "Toehold_seq", "hairpin_seq",
        "b3_grammar", "deltaG_hairpin", "deltaG_toehold_full",
    ]
    avg_df = (
        all_seed_df.groupby(id_cols, as_index=False)
        .agg(
            avg_pred_score_all_seeds=("pred_score", "mean"),
            sd_pred_score_all_seeds=("pred_score", "std"),
            median_pred_rank_all_seeds=("pred_rank", "median"),
            best_pred_rank_any_seed=("pred_rank", "min"),
            worst_pred_rank_any_seed=("pred_rank", "max"),
            n_seeds=("seed", "nunique"),
        )
    )
    avg_df["avg_pred_rank_all_seeds"] = pd.Series(-avg_df["avg_pred_score_all_seeds"]).rank(method="first").astype(int).values
    avg_df = avg_df.sort_values("avg_pred_rank_all_seeds").reset_index(drop=True)

    feature_df = pd.DataFrame({
        "feature_rank": np.arange(1, len(feature_cols) + 1),
        "feature": feature_cols,
        "notes": [
            "absolute/raw hairpin deltaG" if f == raw_hairpin else
            "absolute/raw full toehold sensor-region deltaG" if f == raw_toehold else
            "within-transcript percentile/relative or categorical feature"
            for f in feature_cols
        ],
    })

    config = {
        "script": "destar_rank.py",
        "final_model_name": FINAL_MODEL_NAME,
        "input_dir": args.input_dir,
        "file_pattern": args.file_pattern,
        "training_libraries": libs,
        "controls_excluded_for": exclude_libs,
        "target_col": args.target_col,
        "raw_transform": args.raw_transform,
        "model": args.model,
        "seeds": seeds,
        "grid_config": cfg,
        "new_transcript_name": args.new_transcript_name,
        "new_transcript_length_nt": len(transcript),
        "candidate_count": int(len(candidate_features)),
        "toehold_tail": args.toehold_tail,
        "hairpin_stem_nt": args.hairpin_stem_nt,
        "design_rule": "Toehold_seq = reverse_complement(target_33nt) + toehold_tail; hairpin_seq = last hairpin_stem_nt nt of reverse_complement(target_33nt) + toehold_tail",
        "raw_hairpin_feature": raw_hairpin,
        "raw_toehold_feature": raw_toehold,
        "final_features": feature_cols,
        "plfold_W": args.plfold_W,
        "plfold_L": args.plfold_L,
        "temperature_c": args.temperature_c,
        "train_logs": train_logs,
    }

    write_outputs(args.out_prefix, avg_df, all_seed_df, feature_df, candidate_features, config)

    print("\n=== Top 20 predicted candidates ===")
    show_cols = [
        "avg_pred_rank_all_seeds", "candidate_id", "target_start_1based", "target_end_1based",
        "avg_pred_score_all_seeds", "sd_pred_score_all_seeds", "target_33nt",
        "b3_grammar", "deltaG_hairpin", "deltaG_toehold_full",
    ]
    print(avg_df[show_cols].head(20).to_string(index=False))
    print(f"\nWrote outputs with prefix: {args.out_prefix}")


if __name__ == "__main__":
    main()
