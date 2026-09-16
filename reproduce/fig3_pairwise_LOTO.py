#!/usr/bin/env python3
"""
fig3_pairwise_LOTO.py — DeSTAR pairwise target-site ranking model, leave-one-transcript-out
(the Figure 3 model: enrichment of top-ranked sites for high performers in held-out transcripts).

This is the CLEANED, baseline-only version of the original run script
(Detectron_STEP4_hardNegativeMining_FINAL_features_grid0005_6libs.py). The published model is
the BASELINE pairwise model — hard-negative mining was evaluated as a comparison arm and NOT
selected, so all hard-negative / anti-bottom / bottom-risk / grid-search machinery is removed
here. The model logic (feature set, pair building, A-soft zone weighting, XGB estimator, LOTO
loop, per-seed averaging) is byte-identical to the baseline path of the original script, so the
predictions reproduce the `baseline_final_model` rows used in Figure 3.

Model: RankNet-style pairwise classifier over feature DIFFERENCES. Within each training
transcript, ordered variant pairs are sampled (nested-zone + top-focus + random), labeled by
which variant has the higher log2(ON/OFF), and up-weighted across performance zones
(elite/strong/border). An XGBoost classifier (median-imputed) learns P(i > j) from X[i]-X[j];
a held-out transcript's candidates are scored by mean pairwise-win probability vs all others.
Evaluation is leave-one-transcript-out over the six libraries, averaged over 10 seeds.

Frozen config (grid_0005 / A_softZoneWeight): max_pairs_per_tx 15000, min_abs_diff 0.10,
top_focus_frac 0.125, top_focus_prob 0.95, top_focus_mode one_in_top, nested_frac 0.90,
zone weights elite-strong 2.5 / elite-border 2.0 / strong-border 1.2. Seeds 0-9. The baseline
seed offset (seed + 12000) is preserved so results match the original run exactly.

Input : six *_..._curated_txrel_Final_6.xlsx tables (target col "Average_ON/OFF"; the 20 model
        features present). Non-Zika libraries have an appended control row that is dropped.
Output: <out_prefix>.all_seed_predictions.csv       (per seed x transcript x variant)
        <out_prefix>.avg_predictions.csv            (averaged over seeds: avg_pred_score/rank)

Requirements: numpy, pandas, scikit-learn, xgboost, openpyxl.
"""
import argparse, glob, os, re
from typing import List, Dict, Tuple
import numpy as np, pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    from sklearn.ensemble import HistGradientBoostingClassifier
    HAS_XGB = False

LIBS = ["T5", "T7", "Dengue", "ENO1", "PGK1", "Zika"]
EXCLUDE_CONTROL_LAST_ROW = {"T5", "T7", "Dengue", "ENO1", "PGK1"}  # Zika keeps all rows
TARGET_COL = "Average_ON/OFF"

# Resolved final 20 features (raw ΔG swapped in for the _pct_in_tx ΔG, as in the original run's config)
FINAL_FEATURES = [
    "b3_SSS", "b3_SSW", "b3_SWS", "b3_SWW", "b3_WSS", "b3_WSW", "b3_WWS", "b3_WWW",
    "deltaG_hairpin", "deltaG_toehold_full",
    "pU1_mean_total_33_pct_in_tx", "pU1_min_total_33_pct_in_tx",
    "pU33_segment_unpaired_pct_in_tx", "pU25_seed_segment_unpaired_pct_in_tx",
    "pU8_invasion_segment_unpaired_pct_in_tx",
    "GC_total_33_pct_in_tx", "GC_seed_25_pct_in_tx", "GC_invasion_8_pct_in_tx",
    "GC_context_pm50_pct_in_tx", "relpos_in_tx",
]

CFG = {  # grid_0005 / A_softZoneWeight
    "max_pairs_per_tx": 15000, "min_abs_diff": 0.10,
    "top_focus_frac": 0.125, "top_focus_prob": 0.95, "top_focus_mode": "one_in_top",
    "nested_frac": 0.90,
    "elite_strong_w": 2.5, "elite_border_w": 2.0, "strong_border_w": 1.2,
}
SEED_OFFSET = 12000  # baseline seed = user_seed + 12000 (+ gi*100, gi=0) — matches original run

# ---------------------------------------------------------------- data / features
def normalize_colname(x): return re.sub(r"[^a-z0-9]+", "", str(x).strip().lower())

def resolve_target_col(df, target_col):
    if target_col in df.columns: return target_col
    cm = {normalize_colname(c): c for c in df.columns if isinstance(c, str)}
    if normalize_colname(target_col) in cm: return cm[normalize_colname(target_col)]
    for fb in ["Average_ON/OFF", "Average ON/OFF", "Average_ON_OFF", "Average ON OFF"]:
        if fb in df.columns: return fb
    raise KeyError(f"Target column '{target_col}' not found")

def build_feature_matrix(df, target_col, feature_cols):
    y_raw = pd.to_numeric(df[resolve_target_col(df, target_col)], errors="coerce").astype(float).values
    missing = [c for c in feature_cols if c not in df.columns]
    if missing: raise KeyError(f"Missing feature columns: {missing[:20]}")
    return df[feature_cols].copy(), y_raw

def apply_log2(y_raw):
    out = np.full_like(y_raw.astype(float), np.nan, dtype=float)
    m = ~np.isnan(y_raw)
    out[m] = np.log2(np.clip(y_raw[m], 1e-6, None))
    return out

def compute_percentiles_desc(y_cont):
    valid = ~np.isnan(y_cont); frac = np.full(len(y_cont), np.nan)
    if valid.sum() == 0: return frac
    vals = y_cont[valid]; order = np.argsort(-vals, kind="mergesort")
    rank = np.empty(len(vals)); rank[order] = np.arange(1, len(vals) + 1)
    frac[valid] = np.array([1.0]) if len(vals) == 1 else 1.0 - (rank - 1.0) / (len(vals) - 1.0)
    return frac

def assign_zone_15(y_cont):
    frac = compute_percentiles_desc(y_cont); z = np.array(["rest"] * len(y_cont), dtype=object)
    v = ~np.isnan(frac)
    z[(frac >= 0.95) & v] = "elite"
    z[(frac >= 0.90) & (frac < 0.95) & v] = "strong"
    z[(frac >= 0.85) & (frac < 0.90) & v] = "border"
    return z

# ---------------------------------------------------------------- estimator
def make_model(seed):
    if HAS_XGB:
        m = XGBClassifier(n_estimators=500, max_depth=5, learning_rate=0.05, subsample=0.9,
                          colsample_bytree=0.9, reg_lambda=1.0, objective="binary:logistic",
                          eval_metric="logloss", random_state=seed, n_jobs=8)
    else:
        from sklearn.ensemble import HistGradientBoostingClassifier
        m = HistGradientBoostingClassifier(learning_rate=0.05, max_depth=6, max_iter=500, random_state=seed)
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", m)])

# ---------------------------------------------------------------- pair sampling (A-soft)
def _random_pairs(n, yv, max_pairs, min_abs_diff, rng):
    pairs = set(); att, mx = 0, max_pairs * 40
    while len(pairs) < max_pairs and att < mx:
        i, j = int(rng.integers(0, n)), int(rng.integers(0, n)); att += 1
        if i == j: continue
        a, b = (i, j) if i < j else (j, i)
        if abs(yv[a] - yv[b]) < min_abs_diff: continue
        pairs.add((a, b))
    return list(pairs)

def _top_focus_pairs(n, yv, max_pairs, min_abs_diff, rng, top_focus_frac, mode):
    order = np.argsort(-yv); n_top = max(1, int(np.ceil(top_focus_frac * n)))
    top = np.array(sorted(set(order[:n_top].tolist())), dtype=int)
    pairs = set(); att, mx = 0, max_pairs * 60
    while len(pairs) < max_pairs and att < mx:
        att += 1
        if len(top) == 0: break
        i, j = int(top[rng.integers(0, len(top))]), int(rng.integers(0, n))
        if i == j: continue
        a, b = (i, j) if i < j else (j, i)
        if abs(yv[a] - yv[b]) < min_abs_diff: continue
        pairs.add((a, b))
    return list(pairs)

def _nested_top_pairs(yv, max_pairs, min_abs_diff, rng, nested_frac):
    n = len(yv); zones = assign_zone_15(yv)
    idx = {z: np.where(zones == z)[0] for z in ("elite", "strong", "border", "rest")}
    pair_types = [("elite","strong",0.35),("elite","border",0.25),("strong","border",0.20),
                  ("elite","rest",0.10),("strong","rest",0.05),("border","rest",0.05)]
    target = int(round(max_pairs * nested_frac)); pairs = set(); att, mx = 0, target * 90
    while len(pairs) < target and att < mx:
        att += 1; r, cum, chosen = rng.random(), 0.0, None
        for a, b, w in pair_types:
            cum += w
            if r <= cum: chosen = (a, b); break
        if chosen is None: chosen = ("elite", "strong")
        za, zb = chosen
        if len(idx[za]) == 0 or len(idx[zb]) == 0:
            i, j = int(rng.integers(0, n)), int(rng.integers(0, n))
            if i == j: continue
        else:
            i, j = int(idx[za][rng.integers(0, len(idx[za]))]), int(idx[zb][rng.integers(0, len(idx[zb]))])
        a, b = (i, j) if i < j else (j, i)
        if a == b or abs(yv[a] - yv[b]) < min_abs_diff: continue
        pairs.add((a, b))
    return list(pairs), zones

def _zone_weight(zones, i, j, cfg):
    pair = tuple(sorted([zones[i], zones[j]]))
    return {("elite","strong"): cfg["elite_strong_w"], ("border","elite"): cfg["elite_border_w"],
            ("border","strong"): cfg["strong_border_w"]}.get(pair, 1.0)

def build_pairs_A(X, y_cont, cfg, seed):
    rng = np.random.default_rng(seed)
    valid = ~np.isnan(y_cont); Xv = X.loc[valid].reset_index(drop=True); yv = y_cont[valid]; n = len(Xv)
    if n < 2:
        return pd.DataFrame(columns=X.columns), np.array([], dtype=int), np.array([], dtype=float)
    nested, zones = _nested_top_pairs(yv, cfg["max_pairs_per_tx"], cfg["min_abs_diff"], rng, cfg["nested_frac"])
    n_focus = int(round(cfg["max_pairs_per_tx"] * cfg["top_focus_prob"]))
    focus = _top_focus_pairs(n, yv, n_focus, cfg["min_abs_diff"], rng, cfg["top_focus_frac"], cfg["top_focus_mode"])
    n_rand = max(cfg["max_pairs_per_tx"] - len(nested) - len(focus), 0)
    rand = _random_pairs(n, yv, n_rand, cfg["min_abs_diff"], rng)
    pair_idx = list(set(nested) | set(focus) | set(rand))
    X_arr = Xv.to_numpy(dtype=float); cols = Xv.columns.tolist()
    Xp, yp, wp = [], [], []
    for i, j in pair_idx:
        diff = X_arr[i] - X_arr[j]; w = _zone_weight(zones, i, j, cfg)
        if yv[i] > yv[j]: Xp += [diff, -diff]; yp += [1, 0]
        else:            Xp += [diff, -diff]; yp += [0, 1]
        wp += [w, w]
    return pd.DataFrame(np.asarray(Xp, dtype=float), columns=cols), np.asarray(yp), np.asarray(wp, dtype=float)

def fit_model_A(df_map, train_libs, feature_cols, cfg, seed):
    Xs, ys, ws = [], [], []
    for idx, lib in enumerate(train_libs):
        X, y_raw = build_feature_matrix(df_map[lib], TARGET_COL, feature_cols)
        Xp, yp, wp = build_pairs_A(X, apply_log2(y_raw), cfg, seed + idx)
        if len(yp): Xs.append(Xp); ys.append(yp); ws.append(wp)
    if not Xs: raise ValueError("No pairwise training data")
    Xtr = pd.concat(Xs, ignore_index=True); model = make_model(seed)
    model.fit(Xtr, np.concatenate(ys), model__sample_weight=np.concatenate(ws))
    return model

def score_unseen(model, X_test, chunk=256):
    cols = X_test.columns.tolist(); X = X_test.to_numpy(dtype=float); n = X.shape[0]
    if n == 0: return np.array([])
    if n == 1: return np.array([1.0])
    scores = np.zeros(n)
    for i in range(n):
        probs = []; xi = X[i:i+1]; others = [j for j in range(n) if j != i]
        for s in range(0, len(others), chunk):
            d = pd.DataFrame(xi - X[others[s:s+chunk]], columns=cols)
            probs.append(model.predict_proba(d)[:, 1])
        scores[i] = float(np.mean(np.concatenate(probs))) if probs else np.nan
    return scores

# ---------------------------------------------------------------- main LOTO
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", default=".")
    ap.add_argument("--pattern", default="*_target_search_library_analysis_features.recomputed.with_pU.curated_txrel_Final_6.xlsx")
    ap.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    ap.add_argument("--out_prefix", default="fig3_pairwise_LOTO")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    # load the six libraries, drop appended control row for non-Zika
    df_map = {}
    for f in glob.glob(os.path.join(args.input_dir, args.pattern)):
        lib = next((l for l in LIBS if os.path.basename(f).startswith(l + "_")), None)
        if lib is None: continue
        d = pd.read_excel(f, sheet_name="Sheet1", engine="openpyxl")
        if lib in EXCLUDE_CONTROL_LAST_ROW: d = d.iloc[:-1].reset_index(drop=True)
        df_map[lib] = d
    libs = [l for l in LIBS if l in df_map]
    assert len(libs) == 6, f"expected 6 libraries, found {libs}"

    rows = []
    for seed in seeds:
        for test_lib in libs:
            train_libs = [x for x in libs if x != test_lib]
            model = fit_model_A(df_map, train_libs, FINAL_FEATURES, CFG, seed + SEED_OFFSET)
            Xte, y_raw = build_feature_matrix(df_map[test_lib], TARGET_COL, FINAL_FEATURES)
            scores = score_unseen(model, Xte)
            d = df_map[test_lib]
            for k in range(len(d)):
                rows.append({"feature_set": "baseline_final_model", "grid_id": "grid_0005",
                             "test": test_lib, "seed": seed,
                             "variant_id": d.iloc[k].get("variant_id", k),
                             "target_start_0based": d.iloc[k].get("target_start_0based", np.nan),
                             "pred_score": scores[k], "y_raw": y_raw[k],
                             "y_cont": float(np.log2(max(y_raw[k], 1e-6))) if not np.isnan(y_raw[k]) else np.nan})
            print(f"  seed {seed} test {test_lib}: n={len(d)}", flush=True)

    raw = pd.DataFrame(rows)
    raw.to_csv(f"{args.out_prefix}.all_seed_predictions.csv", index=False)
    # average over seeds per variant -> avg_pred_score / avg_pred_rank
    g = ["feature_set", "grid_id", "test", "variant_id", "target_start_0based"]
    avg = raw.groupby(g, as_index=False, dropna=False).agg(
        avg_pred_score_all_seeds=("pred_score", "mean"), n_seeds=("seed", "nunique"),
        y_cont=("y_cont", "first"), y_raw=("y_raw", "first"))
    avg["avg_pred_rank_all_seeds"] = (avg.groupby("test")["avg_pred_score_all_seeds"]
                                      .rank(method="first", ascending=False).astype(int))
    avg = avg.sort_values(["test", "avg_pred_rank_all_seeds"]).reset_index(drop=True)
    avg.to_csv(f"{args.out_prefix}.avg_predictions.csv", index=False)
    print(f"\nwrote {args.out_prefix}.all_seed_predictions.csv and .avg_predictions.csv "
          f"({len(seeds)} seeds x 6 LOTO folds)")

if __name__ == "__main__":
    main()
