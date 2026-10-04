#!/usr/bin/env python3
"""
fig4_grammar_winrate.py — Figure 4d/f: the bottom-stem base-pairing grammar.

For each bottom-3-bp stem motif (bp1-bp2-bp3, each position Strong=G-C or Weak=A-U), compute
its empirical "mean pairwise win probability": the probability that a variant carrying that
motif outperforms a randomly chosen variant of the SAME transcript, measured by
log2(ON/OFF). Computed per transcript, then averaged across the six libraries.

For a transcript with measured activities a (Average_ON/OFF), the win probability of variant v
against a random other variant is (rank(v) - 1)/(N - 1), where rank is the ascending average
rank of a (ties -> 0.5). P_win(motif) = mean of that over variants with the motif; the panel
value is the mean of the six per-transcript P_win(motif).

This is a purely empirical, model-free readout (no training). It reproduces Fig 4d
(WWS 0.69, SWS 0.65 ... SSW 0.33) and the panel-f grammar (weak bp2 > strong bp2;
within weak-bp2, strong bp3 > weak bp3).

Input : Supplementary Table 3 (--supp_table3) [primary], or the legacy *_Final_6.xlsx (columns b3_grammar, Average_ON/OFF); non-Zika control row dropped.
Output: <out_prefix>_winrate.csv  (per-motif P_win, per transcript + mean),
        prints the panel-d table and the weak/strong-bp2 summary.
Requirements: numpy, pandas, scipy, openpyxl.
"""
import argparse, glob, os, re
import destar_io
import numpy as np, pandas as pd
from scipy.stats import rankdata

MOTIFS = ["WWS", "SWS", "SWW", "WWW", "WSS", "WSW", "SSS", "SSW"]  # bp1-bp2-bp3


def main():
    ap = argparse.ArgumentParser()
    destar_io.add_input_args(ap)
    ap.add_argument("--out_prefix", default="fig4_grammar")
    args = ap.parse_args()

    pw = {m: {} for m in MOTIFS}   # motif -> {transcript: P_win}
    for tx, d in destar_io.load_transcripts(args.supp_table3, args.input_dir).items():
        a = pd.to_numeric(d["Average_ON/OFF"], errors="coerce").values.astype(float)
        motif = d["b3_grammar"].astype(str).values
        ok = np.isfinite(a)
        a, motif = a[ok], motif[ok]
        N = len(a)
        p_win_variant = (rankdata(a) - 1) / (N - 1)  # P(v beats random other)
        for m in MOTIFS:
            sel = motif == m
            if sel.sum() > 0:
                pw[m][tx] = float(p_win_variant[sel].mean())

    txs = sorted({t for m in MOTIFS for t in pw[m]})
    rows = []
    for m in MOTIFS:
        vals = [pw[m][t] for t in txs if t in pw[m]]
        rows.append({"motif_bp1bp2bp3": m, "bp2": m[1], "bp3": m[2],
                     "mean_P_win": round(np.mean(vals), 4),
                     "sd_P_win": round(np.std(vals, ddof=1), 4), "n_transcripts": len(vals),
                     **{f"P_win_{t}": round(pw[m].get(t, np.nan), 4) for t in txs}})
    out = pd.DataFrame(rows)
    out.to_csv(f"{args.out_prefix}_winrate.csv", index=False)

    print("Fig 4d — mean pairwise win probability by bottom-stem motif (bp1-bp2-bp3):")
    for r in rows:
        print(f"   {r['motif_bp1bp2bp3']}: {r['mean_P_win']:.3f}")
    weak = np.mean([r["mean_P_win"] for r in rows if r["bp2"] == "W"])
    strong = np.mean([r["mean_P_win"] for r in rows if r["bp2"] == "S"])
    print(f"\nweak-bp2 mean P_win = {weak:.3f}   strong-bp2 mean P_win = {strong:.3f}")
    # Fig 4f grammar: within weak bp2, strong vs weak bp3
    wk_sbp3 = np.mean([r["mean_P_win"] for r in rows if r["bp2"] == "W" and r["bp3"] == "S"])
    wk_wbp3 = np.mean([r["mean_P_win"] for r in rows if r["bp2"] == "W" and r["bp3"] == "W"])
    print(f"weak bp2 + strong bp3 = {wk_sbp3:.3f} (WIN)   weak bp2 + weak bp3 = {wk_wbp3:.3f} (neutral)")
    print(f"\nwrote {args.out_prefix}_winrate.csv")


if __name__ == "__main__":
    main()
