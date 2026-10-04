#!/usr/bin/env python3
"""
ed6d_bp1_encoding.py — Extended Data Fig. 6d: does encoding bp1 base identity improve ranking?

The bottom-stem motif feature class encodes only the STRONG/WEAK identity of bp1-3. bp1 additionally
shows a pyrimidine (especially U) preference in the raw enrichment logos (Fig. 2c), but that
preference is near-uniform among high-performing sites. This script tests whether adding bp1 base
identity to the 20-feature model improves held-out ranking, with two encodings:
  +bp1 pyrimidine/purine : one feature  = bp1 is a pyrimidine (C/U = 1, A/G = 0)
  +bp1 full base         : four features = bp1 one-hot base identity (A, C, G, U)

Model and LOTO are the exact Fig 3 pairwise ranker (imported from fig3_pairwise_LOTO.py);
aggregation is METHOD B (per held-out transcript, scores are averaged over seeds into one consensus
ranking, and precision@k is computed once). For each encoding we report per-transcript P@5/10/15 and
the paired change vs the 20-feature baseline (ΔP@k), tested by two-sided Wilcoxon across the six
transcripts.

Finding: neither encoding improves held-out ranking (all ΔP@k non-significant) — bp1 identity is a
threshold for activation, not a determinant that discriminates among top sites.

bp1 base = position 26 of the 33-nt switch region = Toehold_seq[25] (verified: its strong/weak
identity matches b3_grammar[0] for all 7,383 sensors).

Input : Supplementary Table 3 (--supp_table3) [primary], or legacy six *_Final_6.xlsx (--input_dir).
Output: <out_prefix>_per_transcript.csv  (per encoding x transcript: P@5/10/15)
        <out_prefix>_stats.csv            (ΔP@k vs baseline, Wilcoxon, per encoding)
Requirements: numpy, pandas, scipy, scikit-learn, xgboost, openpyxl; fig3_pairwise_LOTO.py & fig4_ablation.py on path.
"""
import argparse
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
import fig3_pairwise_LOTO as M
import fig4_ablation as AB   # reuse run_feature_set_B (Method B) and KS

BP1_POS = 25          # 0-based index into the 33-nt switch region (= Toehold_seq[:33]); switch position 26


def add_bp1_features(df):
    sw = df["Toehold_seq"].astype(str).str.upper().str.replace("T", "U")
    b = sw.str[BP1_POS]
    df = df.copy()
    df["bp1_pyrimidine"] = b.isin(list("CU")).astype(float)
    for base in "ACGU":
        df[f"bp1_{base}"] = (b == base).astype(float)
    return df


def _wilcoxon_safe(diffs):
    d = np.asarray(diffs, dtype=float)
    if np.allclose(d, 0.0):
        return 0.0, 1.0
    try:
        w, p = wilcoxon(d, zero_method="wilcox", alternative="two-sided", mode="auto")
        return float(w), float(p)
    except Exception:
        return float("nan"), float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--supp_table3", default=None,
                    help="Path to the deposited Supplementary Table workbook (sheet 'Supplementary Table 3'). Primary input.")
    ap.add_argument("--input_dir", default=None, help="(legacy) folder holding the six *_Final_6.xlsx tables")
    ap.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    ap.add_argument("--out_prefix", default="ed6d_bp1_encoding")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    df_map = M.load_df_map(supp_table3=args.supp_table3, input_dir=args.input_dir)
    df_map = {lib: add_bp1_features(d) for lib, d in df_map.items()}
    libs = list(df_map.keys())
    assert len(libs) == 6, f"expected 6 libraries, found {libs}"

    SETS = {
        "baseline (20)": M.FINAL_FEATURES,
        "+bp1 pyrimidine/purine": M.FINAL_FEATURES + ["bp1_pyrimidine"],
        "+bp1 full base": M.FINAL_FEATURES + ["bp1_A", "bp1_C", "bp1_G", "bp1_U"],
    }
    results = {}
    for name, feats in SETS.items():
        print(f"{name} ...", flush=True)
        results[name] = AB.run_feature_set_B(df_map, libs, feats, seeds)

    # per-transcript table
    pt_rows = []
    for name, per in results.items():
        for t in libs:
            pt_rows.append({"model": name, "transcript": t, **{f"P{k}": round(per[t][k], 4) for k in AB.KS}})
    pd.DataFrame(pt_rows).to_csv(f"{args.out_prefix}_per_transcript.csv", index=False)

    # stats: each encoding vs baseline, ΔP@k across transcripts
    base = results["baseline (20)"]
    stat_rows = []
    for name in ("+bp1 pyrimidine/purine", "+bp1 full base"):
        enc = results[name]
        for k in AB.KS:
            d = [enc[t][k] - base[t][k] for t in libs]
            w, p = _wilcoxon_safe(d)
            stat_rows.append({"encoding": name, "metric": f"ΔP@{k}", "mean_delta": round(float(np.mean(d)), 4),
                              "W": w, "P": p, "n": len(libs),
                              "P_summary": ("****" if p < 1e-4 else "***" if p < 1e-3 else "**" if p < 1e-2
                                            else "*" if p < 0.05 else "ns")})
    stats = pd.DataFrame(stat_rows)
    stats.to_csv(f"{args.out_prefix}_stats.csv", index=False)

    print("\n== ED Fig. 6d: bp1-encoding ablation (Method B; ΔP vs 20-feature baseline) ==")
    print(stats.to_string(index=False))
    print(f"\nwrote {args.out_prefix}_per_transcript.csv, _stats.csv")


if __name__ == "__main__":
    main()
