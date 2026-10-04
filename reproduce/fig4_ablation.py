#!/usr/bin/env python3
"""
fig4_ablation.py — Figure 4b/c and Extended Data Fig. 5c: feature-class ablation.

Reuses the exact Fig 3 pairwise LOTO model (imported from fig3_pairwise_LOTO.py) so the ablation
cannot drift from the reported model. For the full 20-feature model, for each of the five feature
classes ALONE, and for each class REMOVED, it runs leave-one-transcript-out over the six libraries
(10 seeds) and reports Precision@k (k = 5, 10, 15), where a "hit" is a predicted top-k site that is
in the top 15% by measured log2(ON/OFF) of the held-out transcript. Random baseline = 0.15.

AGGREGATION = METHOD B (deployment-consistent, matches destar_rank.py and the Fig 3
avg_predictions ranking): within each held-out transcript, each candidate's score is AVERAGED
over the 10 seeds into a single consensus ranking, and precision@k is computed ONCE on that
ranking. The six per-transcript values are then averaged (summary) and used for the paired
Wilcoxon tests (Fig 4b: removed vs full ΔP@10; Fig 4c: alone P@10 vs chance = 0.15).

Input : Supplementary Table 3 (--supp_table3) [primary], or legacy six *_Final_6.xlsx (--input_dir).
Output: <out_prefix>_per_transcript.csv  (per model x transcript: P@5/10/15, Method B)
        <out_prefix>_summary.csv          (per model: mean P@5/10/15 over the six transcripts)
        <out_prefix>_stats.csv            (Fig 4b removed-vs-full & Fig 4c alone-vs-chance Wilcoxon)
Requirements: numpy, pandas, scipy, scikit-learn, xgboost, openpyxl, and fig3_pairwise_LOTO.py on path.
"""
import argparse
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
import fig3_pairwise_LOTO as M   # shared, verified model (fit_model_A / score_unseen / load_df_map)

FEATURE_CLASSES = {
    "Bottom 3-bp stem motif": ["b3_SSS", "b3_SSW", "b3_SWS", "b3_SWW", "b3_WSS", "b3_WSW", "b3_WWS", "b3_WWW"],
    "Sensor folding ΔG": ["deltaG_hairpin", "deltaG_toehold_full"],
    "Target/context GC": ["GC_total_33_pct_in_tx", "GC_seed_25_pct_in_tx", "GC_invasion_8_pct_in_tx", "GC_context_pm50_pct_in_tx"],
    "Target accessibility": ["pU1_mean_total_33_pct_in_tx", "pU1_min_total_33_pct_in_tx",
                              "pU33_segment_unpaired_pct_in_tx", "pU25_seed_segment_unpaired_pct_in_tx",
                              "pU8_invasion_segment_unpaired_pct_in_tx"],
    "Relative transcript position": ["relpos_in_tx"],
}
KS = (5, 10, 15)
TOPQ = 0.15
CHANCE = 0.15


def precision_at_k(y_cont, scores, k, q=TOPQ):
    N = len(y_cont)
    thr = np.sort(y_cont)[::-1][max(1, round(q * N)) - 1]
    hit = y_cont >= thr
    topk = np.argsort(scores)[::-1][:k]
    return float(hit[topk].mean())


def run_feature_set_B(df_map, libs, feature_cols, seeds):
    """Method B. LOTO over libs; for each held-out transcript, AVERAGE candidate scores over the
    seeds into one consensus ranking, then compute P@k once. Returns {test_lib: {k: P@k}}."""
    sum_scores = {lib: None for lib in libs}
    yc_map = {}
    for seed in seeds:
        for test in libs:
            train = [x for x in libs if x != test]
            model = M.fit_model_A(df_map, train, feature_cols, M.CFG, seed + M.SEED_OFFSET)
            Xte, yraw = M.build_feature_matrix(df_map[test], M.TARGET_COL, feature_cols)
            sc = M.score_unseen(model, Xte)
            sum_scores[test] = sc if sum_scores[test] is None else sum_scores[test] + sc
            yc_map[test] = M.apply_log2(yraw)
    out = {}
    nseed = len(seeds)
    for test in libs:
        avg = sum_scores[test] / nseed                    # consensus score over seeds (Method B)
        yc = yc_map[test]
        ok = np.isfinite(yc) & np.isfinite(avg)
        out[test] = {k: precision_at_k(yc[ok], avg[ok], k) for k in KS}
    return out


def _wilcoxon_safe(diffs):
    """Two-sided Wilcoxon signed-rank vs 0; robust to all-zero / tiny samples."""
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
    ap.add_argument("--out_prefix", default="fig4_ablation")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    df_map = M.load_df_map(supp_table3=args.supp_table3, input_dir=args.input_dir)
    libs = list(df_map.keys())
    assert len(libs) == 6, f"expected 6 libraries, found {libs}"

    # ---- run every feature set (Method B), keep per-transcript P@k ----
    results = {}   # model_name -> {test_lib: {k: P@k}}
    print("full 20 ...", flush=True)
    results["full 20"] = run_feature_set_B(df_map, libs, M.FINAL_FEATURES, seeds)
    for cls, feats in FEATURE_CLASSES.items():
        print(f"{cls} ALONE ...", flush=True)
        results[f"{cls} only"] = run_feature_set_B(df_map, libs, feats, seeds)
    for cls, feats in FEATURE_CLASSES.items():
        rem = [f for f in M.FINAL_FEATURES if f not in feats]
        print(f"{cls} REMOVED ...", flush=True)
        results[f"{cls} removed"] = run_feature_set_B(df_map, libs, rem, seeds)

    # ---- per-transcript table ----
    pt_rows = []
    for model, per in results.items():
        for test in libs:
            pt_rows.append({"model": model, "transcript": test, **{f"P{k}": round(per[test][k], 4) for k in KS}})
    pd.DataFrame(pt_rows).to_csv(f"{args.out_prefix}_per_transcript.csv", index=False)

    # ---- summary (mean over the six transcripts) ----
    sum_rows = []
    for model, per in results.items():
        means = {k: float(np.mean([per[t][k] for t in libs])) for k in KS}
        sum_rows.append({"model": model, **{f"P{k}": round(means[k], 4) for k in KS}})
    sum_rows.append({"model": "random (chance)", **{f"P{k}": CHANCE for k in KS}})
    summary = pd.DataFrame(sum_rows)
    summary.to_csv(f"{args.out_prefix}_summary.csv", index=False)

    # ---- Wilcoxon stats ----
    full = results["full 20"]
    stat_rows = []
    # Fig 4b: each class REMOVED vs full, paired ΔP@10 across transcripts
    for cls in FEATURE_CLASSES:
        rem = results[f"{cls} removed"]
        d = [rem[t][10] - full[t][10] for t in libs]
        w, p = _wilcoxon_safe(d)
        stat_rows.append({"panel": "Fig 4b (class removed vs full)", "class": cls, "metric": "ΔP@10",
                          "mean": round(float(np.mean(d)), 4), "W": w, "P": p, "n": len(libs)})
    # Fig 4c: each class ALONE vs chance (0.15), paired across transcripts
    for cls in FEATURE_CLASSES:
        alo = results[f"{cls} only"]
        vals = [alo[t][10] for t in libs]
        d = [v - CHANCE for v in vals]
        w, p = _wilcoxon_safe(d)
        stat_rows.append({"panel": "Fig 4c (class alone vs chance)", "class": cls, "metric": "P@10",
                          "mean": round(float(np.mean(vals)), 4), "W": w, "P": p, "n": len(libs)})
    pd.DataFrame(stat_rows).to_csv(f"{args.out_prefix}_stats.csv", index=False)

    print("\n== summary (mean P@k over six transcripts, Method B) ==")
    print(summary.to_string(index=False))
    print("\n== stats ==")
    print(pd.DataFrame(stat_rows).to_string(index=False))
    print(f"\nwrote {args.out_prefix}_per_transcript.csv, _summary.csv, _stats.csv")


if __name__ == "__main__":
    main()
