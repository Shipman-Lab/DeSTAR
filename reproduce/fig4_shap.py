#!/usr/bin/env python3
"""
fig4_shap.py — Figure 4a and Extended Data Fig. 5b: feature attribution (TreeSHAP).

The DeSTAR ranker is trained on pairwise feature DIFFERENCES (feature_i - feature_j), so SHAP
values quantify how each feature *difference* contributes to P(i > j). For each leave-one-
transcript-out fold (train on 5 libraries) and seed, this script rebuilds the pairwise training
matrix with the exact Fig 3 model machinery (imported from fig3_pairwise_LOTO.py), fits the same
XGBoost pipeline, runs shap.TreeExplainer on a stratified sample of the pairwise rows, and takes
mean |SHAP| per feature. Importances are averaged across folds/seeds, normalized to a fraction of
the total (ED5b per-feature), and summed within the five feature classes (Fig 4a per-class).

Reproduces: Fig 4a per-class fractions (stem 0.42, folding ΔG 0.22, GC 0.16, accessibility 0.11,
position 0.10) and ED5b(i) per-feature ranking (Hairpin ΔG largest single feature, then WWS ...).

Input : Supplementary Table 3 (--supp_table3) [primary], or the legacy *_Final_6.xlsx in --input_dir (same as fig3_pairwise_LOTO.py).
Output: <out_prefix>_per_feature.csv, <out_prefix>_group_rollup.csv.
Requirements: numpy, pandas, scikit-learn, xgboost, shap, openpyxl, fig3_pairwise_LOTO.py on path.
"""
import argparse, glob, os
import numpy as np, pandas as pd
import shap
import fig3_pairwise_LOTO as M

FEATURE_GROUP = {}
for g, feats in {
    "Bottom 3-bp stem motif": ["b3_SSS","b3_SSW","b3_SWS","b3_SWW","b3_WSS","b3_WSW","b3_WWS","b3_WWW"],
    "Sensor folding ΔG": ["deltaG_hairpin","deltaG_toehold_full"],
    "Target/context GC": ["GC_total_33_pct_in_tx","GC_seed_25_pct_in_tx","GC_invasion_8_pct_in_tx","GC_context_pm50_pct_in_tx"],
    "Target accessibility": ["pU1_mean_total_33_pct_in_tx","pU1_min_total_33_pct_in_tx","pU33_segment_unpaired_pct_in_tx","pU25_seed_segment_unpaired_pct_in_tx","pU8_invasion_segment_unpaired_pct_in_tx"],
    "Relative transcript position": ["relpos_in_tx"],
}.items():
    for f in feats:
        FEATURE_GROUP[f] = g

MAX_SHAP_ROWS = 4000  # stratified sample of pairwise rows per fold (speed)


def build_train_pairs(df_map, train_libs, feature_cols, seed):
    """Same as fig3.fit_model_A's training assembly, but return X_pairs too."""
    Xs, ys, ws = [], [], []
    for idx, lib in enumerate(train_libs):
        X, y_raw = M.build_feature_matrix(df_map[lib], M.TARGET_COL, feature_cols)
        Xp, yp, wp = M.build_pairs_A(X, M.apply_log2(y_raw), M.CFG, seed + idx)
        if len(yp): Xs.append(Xp); ys.append(yp); ws.append(wp)
    return pd.concat(Xs, ignore_index=True), np.concatenate(ys), np.concatenate(ws)


def stratified_sample(X, y, max_rows, seed):
    rng = np.random.default_rng(seed); n = len(X)
    if n <= max_rows:
        return X
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    npos = min(len(pos), max_rows // 2); nneg = min(len(neg), max_rows - npos)
    idx = np.concatenate([rng.choice(pos, npos, replace=False), rng.choice(neg, nneg, replace=False)])
    return X.iloc[idx].reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--supp_table3", default=None,
                    help="Path to the deposited Supplementary Table workbook (sheet 'Supplementary Table 3'). Primary input.")
    ap.add_argument("--input_dir", default=None, help="(legacy) folder holding the six *_Final_6.xlsx tables")
    ap.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    ap.add_argument("--out_prefix", default="fig4_shap")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    df_map = M.load_df_map(supp_table3=args.supp_table3, input_dir=args.input_dir)
    libs = list(df_map.keys())
    assert len(libs) == 6, f"expected 6 libraries, found {libs}"
    feats = M.FINAL_FEATURES

    acc = {f: [] for f in feats}   # per feature: list of mean|SHAP| across folds/seeds
    for seed in seeds:
        for test_lib in libs:
            train = [x for x in libs if x != test_lib]
            Xtr, ytr, wtr = build_train_pairs(df_map, train, feats, seed + M.SEED_OFFSET)
            model = M.make_model(seed + M.SEED_OFFSET)
            model.fit(Xtr, ytr, model__sample_weight=wtr)
            Xs = stratified_sample(Xtr, ytr, MAX_SHAP_ROWS, seed + M.SEED_OFFSET)
            X_imp = model.named_steps["imputer"].transform(Xs)
            sv = shap.TreeExplainer(model.named_steps["model"]).shap_values(X_imp)
            if isinstance(sv, list): sv = sv[-1]
            sv = np.asarray(sv, dtype=float)
            if sv.ndim == 3: sv = sv[:, :, -1]
            m = np.abs(sv).mean(axis=0)
            for i, f in enumerate(feats): acc[f].append(m[i])
            print(f"  seed {seed} fold {test_lib} done", flush=True)

    per = pd.DataFrame({"feature": feats,
                        "feature_group": [FEATURE_GROUP[f] for f in feats],
                        "shap_mean_abs": [np.mean(acc[f]) for f in feats]})
    per["shap_frac_of_total"] = per["shap_mean_abs"] / per["shap_mean_abs"].sum()
    per = per.sort_values("shap_frac_of_total", ascending=False).reset_index(drop=True)
    roll = (per.groupby("feature_group", as_index=False)
            .agg(n_features=("feature", "size"), total_shap_frac=("shap_frac_of_total", "sum"))
            .sort_values("total_shap_frac", ascending=False))
    per.round(4).to_csv(f"{args.out_prefix}_per_feature.csv", index=False)
    roll.round(4).to_csv(f"{args.out_prefix}_group_rollup.csv", index=False)

    print("\nFig 4a — per-class SHAP fraction:")
    print(roll.to_string(index=False))
    print("\nED5b(i) — top per-feature:")
    print(per[["feature", "shap_frac_of_total"]].head(5).to_string(index=False))
    print(f"\nwrote {args.out_prefix}_per_feature.csv, {args.out_prefix}_group_rollup.csv")


if __name__ == "__main__":
    main()
