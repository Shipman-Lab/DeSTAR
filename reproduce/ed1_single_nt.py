#!/usr/bin/env python3
"""
ed1_single_nt.py — Extended Data Fig. 1: target-site activity varies at single-nucleotide resolution.

a-f  Per-transcript example traces: log2(Average ON/OFF) across a 41-variant window centred on a
     high-activity site (the window maximum). Consecutive variant numbers are tiled one nucleotide
     apart, so neighbouring points are 1-nt-shifted 33-nt target windows.
g    For each transcript, |Δ log2(Average ON/OFF)| between variants whose target sites are one
     nucleotide apart (adjacent) vs between an equal number of randomly chosen variant pairs, with
     the replicate measurement-noise floor overlaid. A one-sided sign test asks whether adjacent
     differences exceed the per-pair noise floor more often than chance (0.5).

Noise floor (self-contained from the deposited table): Average ON/OFF is the mean of the Dixon-
retained replicates, so the retained set is recovered per variant by finding which subset of
ON/OFF_1/2/3 averages to Average ON/OFF (100% resolvable). The per-variant standard error is then
SE = s.d.(log2 of retained replicates)/sqrt(n_retained); the per-pair noise floor is
0.6745 * sqrt(SE_i^2 + SE_j^2) (median of the half-normal of the difference under noise alone).

Input : Supplementary Table 3 (--supp_table3) [primary], or legacy six *_Final_6.xlsx (--input_dir).
Output: <out_prefix>_panelAF_traces.csv   (window traces for a-f; one row per variant, hero flagged)
        <out_prefix>_panelG_boxdata.csv    (per-transcript adjacent/random/noise quartiles + n)
        <out_prefix>_panelG_stats.csv      (per-transcript + pooled sign test; autocorr; adj/random)
Requirements: numpy, pandas, scipy, openpyxl; (optional) matplotlib for --plot.
"""
import argparse, itertools
import numpy as np, pandas as pd
from scipy.stats import binomtest
import destar_io

Z = 0.67448975      # median of the half-normal / its scale  (= Phi^{-1}(0.75))
HALF = 20           # a-f window half-width, in variants (41-variant window)
REP_COLS = ["ON/OFF_1", "ON/OFF_2", "ON/OFF_3"]

# Published a-f hero sites (variant_id = 1-based tiled-window index). Each is the most active site
# in its 41-variant window; DENV uses the replicate-concordant v166 (see Methods).
HERO = {"gp8": 259, "gp10A": 108, "DENV": 166, "ZIKV": 891, "PGK1": 568, "ENO1": 1157}


def psummary(p):
    return "****" if p < 1e-4 else "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else "ns"


def retained_se_log2(reps, avg, tol=1e-4):
    """Recover the Dixon-retained replicates (whose mean equals Average ON/OFF) and return the
    standard error of their log2 values. Average is in linear ON/OFF space; SE is in log2 space."""
    r = [x for x in reps if np.isfinite(x) and x > 0]
    if not r:
        return np.nan
    for k in (3, 2):
        for combo in itertools.combinations(range(len(r)), k):
            sub = [r[i] for i in combo]
            if abs(np.mean(sub) - avg) <= tol * max(1.0, abs(avg)):
                L = np.log2(sub)
                return float(np.std(L, ddof=1) / np.sqrt(len(L))) if len(L) >= 2 else np.nan
    L = np.log2(r)                                   # fallback: all positive reps
    return float(np.std(L, ddof=1) / np.sqrt(len(L))) if len(L) >= 2 else np.nan


def main():
    ap = argparse.ArgumentParser()
    destar_io.add_input_args(ap)
    ap.add_argument("--out_prefix", default="ed1_single_nt")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for the random-pair reference")
    ap.add_argument("--plot", action="store_true", help="also draw the ED1 figure if matplotlib is available")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    txmap = destar_io.load_transcripts(args.supp_table3, args.input_dir)

    af_rows, box_rows, stat_rows = [], [], []
    pooled_k = pooled_n = pooled_k5 = pooled_n5 = 0

    for tx, d in txmap.items():
        d = d.copy()
        d["vid"] = pd.to_numeric(d["variant_id"], errors="coerce")
        d["s0"] = pd.to_numeric(d["target_start_0based"], errors="coerce")
        d["on"] = pd.to_numeric(d["Average_ON/OFF"], errors="coerce")
        d["y"] = np.log2(d["on"])
        d = d.dropna(subset=["vid", "s0", "on"]).sort_values("s0").reset_index(drop=True)

        # --- panels a-f: 41-variant window centred on the hero (window maximum) ---
        hero = HERO.get(tx)
        if hero is not None:
            w = d[(d.vid >= hero - HALF) & (d.vid <= hero + HALF)]
            for _, r in w.iterrows():
                af_rows.append({"transcript": tx, "variant_id": int(r.vid),
                                "ON/OFF": round(float(r.on), 4), "log2_ON_OFF": round(float(r.y), 4),
                                "is_hero": int(int(r.vid) == hero)})

        # --- panel g: adjacent vs random vs noise floor ---
        se = np.array([retained_se_log2([row[c] for c in REP_COLS], a)
                       for (_, row), a in zip(d.iterrows(), d["on"].values)])
        s0 = d["s0"].values.astype(float); y = d["y"].values
        adj = (np.diff(s0) == 1)
        dobs = np.abs(np.diff(y))[adj]
        nf = Z * np.sqrt(se[:-1][adj] ** 2 + se[1:][adj] ** 2)
        n = int(adj.sum())
        idx = rng.integers(0, len(y), size=(n, 2))
        bad = idx[:, 0] == idx[:, 1]
        while bad.any():
            idx[bad, 1] = rng.integers(0, len(y), size=int(bad.sum())); bad = idx[:, 0] == idx[:, 1]
        drand = np.abs(y[idx[:, 0]] - y[idx[:, 1]])

        k = int(np.sum(dobs > nf))
        bt = binomtest(k, n, 0.5, alternative="greater")
        ac = float(np.corrcoef(y[:-1][adj], y[1:][adj])[0, 1])
        med_adj, med_rand, med_nf = float(np.median(dobs)), float(np.median(drand)), float(np.median(nf))

        for grp, arr in (("adjacent_1nt", dobs), ("random_pair", drand)):
            q = np.percentile(arr, [0, 25, 50, 75, 100])
            box_rows.append({"transcript": tx, "group": grp, "n": len(arr),
                             "min": round(q[0], 4), "Q1": round(q[1], 4), "median": round(q[2], 4),
                             "Q3": round(q[3], 4), "max": round(q[4], 4)})
        box_rows.append({"transcript": tx, "group": "noise_floor_median", "n": n,
                         "min": "", "Q1": "", "median": round(med_nf, 4), "Q3": "", "max": ""})

        stat_rows.append({"transcript": tx, "n_adjacent_pairs": n, "k_exceeding_noise": k,
                          "frac_gt_noise": round(k / n, 4), "P_sign_test": bt.pvalue,
                          "P_summary": psummary(bt.pvalue), "lag1_autocorr": round(ac, 4),
                          "median_adjacent": round(med_adj, 4), "median_noise": round(med_nf, 4),
                          "median_random": round(med_rand, 4), "adj_over_random": round(med_adj / med_rand, 4)})
        pooled_k += k; pooled_n += n
        if tx != "DENV":
            pooled_k5 += k; pooled_n5 += n

    for tag, kk, nn in (("pooled (all 6)", pooled_k, pooled_n), ("pooled (excl. DENV)", pooled_k5, pooled_n5)):
        bt = binomtest(kk, nn, 0.5, alternative="greater")
        stat_rows.append({"transcript": tag, "n_adjacent_pairs": nn, "k_exceeding_noise": kk,
                          "frac_gt_noise": round(kk / nn, 4), "P_sign_test": bt.pvalue,
                          "P_summary": psummary(bt.pvalue), "lag1_autocorr": "", "median_adjacent": "",
                          "median_noise": "", "median_random": "", "adj_over_random": ""})

    pd.DataFrame(af_rows).to_csv(f"{args.out_prefix}_panelAF_traces.csv", index=False)
    pd.DataFrame(box_rows).to_csv(f"{args.out_prefix}_panelG_boxdata.csv", index=False)
    stats = pd.DataFrame(stat_rows)
    stats.to_csv(f"{args.out_prefix}_panelG_stats.csv", index=False)

    print("== ED Fig. 1g sign test (adjacent |Δlog2(ON/OFF)| vs replicate noise floor) ==")
    print(stats.to_string(index=False))
    print(f"\nwrote {args.out_prefix}_panelAF_traces.csv, _panelG_boxdata.csv, _panelG_stats.csv")

    if args.plot:
        try:
            _plot(args.out_prefix, txmap)
        except Exception as e:
            print(f"(plot skipped: {e})")


def _plot(out_prefix, txmap):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    from matplotlib.legend_handler import HandlerTuple
    order = [t for t in destar_io.PAPER_LIBS if t in txmap]
    COL = {"gp8": "#E8438C", "gp10A": "#3B6FD4", "DENV": "#1FA187", "ZIKV": "#F08A5D", "PGK1": "#7D4FC0", "ENO1": "#4FC3E8"}
    af = pd.read_csv(f"{out_prefix}_panelAF_traces.csv")
    # a-f traces
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 5.4))
    for ax, tx in zip(axes.ravel(), order):
        w = af[af.transcript == tx]
        ax.axhline(0.0, ls="--", lw=0.8, color="#aaa")
        ax.plot(w.variant_id, w.log2_ON_OFF, "-o", ms=3, lw=1.1, color=COL.get(tx, "#444"))
        hero = w[w.is_hero == 1]
        onoff = float(hero["ON/OFF"].iloc[0]) if len(hero) else np.nan
        ax.set_title(f"{tx}  (Variant {int(hero.variant_id.iloc[0])}: ON/OFF = {onoff:.1f})", fontsize=8.5, loc="left")
        ax.set_xlabel("Variant number (1-nt tiling)", fontsize=8); ax.set_ylabel("log$_2$(ON/OFF)", fontsize=8)
        ax.tick_params(labelsize=7)
    fig.tight_layout(); fig.savefig(f"{out_prefix}_panelAF.png", dpi=200, bbox_inches="tight")
    print(f"wrote {out_prefix}_panelAF.png")


if __name__ == "__main__":
    main()
