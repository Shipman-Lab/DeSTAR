#!/usr/bin/env python3
"""
fig4_ablation.py — Figure 4b/c and Extended Data Fig 4c: feature-class ablation.

Reuses the exact Fig 3 pairwise LOTO model (imported from fig3_pairwise_LOTO.py) so the
ablation cannot drift from the reported model. For the full 20-feature model, for each of the
five feature classes ALONE, and for each class REMOVED, it runs leave-one-transcript-out over
the six libraries (10 seeds) and reports Precision@k (k = 5, 10, 15), where a "hit" is a
predicted top-k site that is in the top 15% by measured log2(ON/OFF) of the held-out transcript.
Random baseline = 0.15.

Reproduces: full-20 P@5/10/15 = 0.87 / 0.765 / 0.699; stem-motif is the strongest single class;
removing the stem motif causes the largest drop.

Input : six *_..._Final_6.xlsx in --input_dir (same as fig3_pairwise_LOTO.py).
Output: <out_prefix>_alone_and_removed.csv (one row per model x metric).
Requirements: numpy, pandas, scikit-learn, xgboost, openpyxl, and fig3_pairwise_LOTO.py on the path.
"""
import argparse, glob, os
import numpy as np, pandas as pd
import fig3_pairwise_LOTO as M   # shared, verified model (fit_model_A / score_unseen / etc.)

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


def precision_at_k(y_cont, scores, k, q=TOPQ):
    N = len(y_cont)
    thr = np.sort(y_cont)[::-1][max(1, round(q * N)) - 1]
    hit = y_cont >= thr
    topk = np.argsort(scores)[::-1][:k]
    return hit[topk].mean()


def run_feature_set(df_map, libs, feature_cols, seeds):
    """LOTO over libs, seeds; return dict k -> mean P@k across (seed, transcript)."""
    per = {k: [] for k in KS}
    for seed in seeds:
        for test_lib in libs:
            train = [x for x in libs if x != test_lib]
            model = M.fit_model_A(df_map, train, feature_cols, M.CFG, seed + M.SEED_OFFSET)
            Xte, yraw = M.build_feature_matrix(df_map[test_lib], M.TARGET_COL, feature_cols)
            scores = M.score_unseen(model, Xte)
            yc = M.apply_log2(yraw)
            ok = np.isfinite(yc) & np.isfinite(scores)
            for k in KS:
                per[k].append(precision_at_k(yc[ok], scores[ok], k))
    return {k: float(np.mean(per[k])) for k in KS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", default=".")
    ap.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    ap.add_argument("--out_prefix", default="fig4_ablation")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    df_map = {}
    for f in glob.glob(os.path.join(args.input_dir, M.__dict__.get("PATTERN", "*curated_txrel_Final_6.xlsx")) if False else os.path.join(args.input_dir, "*_target_search_library_analysis_features.recomputed.with_pU.curated_txrel_Final_6.xlsx")):
        lib = next((l for l in M.LIBS if os.path.basename(f).startswith(l + "_")), None)
        if lib is None: continue
        d = pd.read_excel(f, sheet_name="Sheet1", engine="openpyxl")
        if lib in M.EXCLUDE_CONTROL_LAST_ROW: d = d.iloc[:-1].reset_index(drop=True)
        df_map[lib] = d
    libs = [l for l in M.LIBS if l in df_map]
    assert len(libs) == 6, f"expected 6 libraries, found {libs}"

    rows = []
    def add(model_name, n_feat, res):
        rows.append({"model": model_name, "n_features": n_feat,
                     **{f"P{k}": round(res[k], 4) for k in KS}})

    print("full 20 ..."); add("full 20", 20, run_feature_set(df_map, libs, M.FINAL_FEATURES, seeds))
    for cls, feats in FEATURE_CLASSES.items():
        print(f"{cls} ALONE ..."); add(f"{cls} only", len(feats), run_feature_set(df_map, libs, feats, seeds))
    for cls, feats in FEATURE_CLASSES.items():
        rem = [f for f in M.FINAL_FEATURES if f not in feats]
        print(f"{cls} REMOVED ..."); add(f"{cls} removed", len(rem), run_feature_set(df_map, libs, rem, seeds))
    add("random (chance)", 0, {k: TOPQ for k in KS})

    out = pd.DataFrame(rows)
    out.to_csv(f"{args.out_prefix}_alone_and_removed.csv", index=False)
    print("\n" + out.to_string(index=False))
    print(f"\nwrote {args.out_prefix}_alone_and_removed.csv")


if __name__ == "__main__":
    main()
