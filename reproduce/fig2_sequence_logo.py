#!/usr/bin/env python3
"""
fig2_sequence_logo.py — Figure 2c: difference sequence logo of the switch region.

Top vs bottom performers, by log2 enrichment of base frequencies per switch position.
Pooled logo (Fig 2c) and, optionally, one logo per transcript (for comparison).

Method (per group of sequences):
  - hit set  = top  K = round(0.15 * N) variants by Average_ON/OFF
  - null set = bottom K variants
  - per switch position p (1..33) and base b in {A,C,G,U}, frequency with a 0.5 pseudocount
    (prevents log2(0) in the small per-transcript groups), normalized per position
  - enrichment(p, b) = log2( freq_top(p,b) / freq_bottom(p,b) )
Switch sequence = the first 33 nt of Toehold_seq (T->U). For the pooled logo, hits/nulls are
taken within each transcript (top/bottom K per transcript) and pooled, so dynamic-range
differences across transcripts do not bias the frequencies.

Reproduces Fig 2c with pooled K = round(0.15 * 7,383) = 1,107 top and 1,107 bottom.

Input : six *_..._Final_6.xlsx (Toehold_seq, Average_ON/OFF); non-Zika control row dropped.
Output: <out_prefix>_data.xlsx (enrichment matrix per position x base; 'pooled' + per-transcript
        sheets). If logomaker + matplotlib are available, also <out_prefix>.svg/.png.
Requirements: numpy, pandas, openpyxl; (optional) logomaker, matplotlib.
"""
import argparse, glob, os
import numpy as np, pandas as pd

LIBS = ["T5", "T7", "Dengue", "ENO1", "PGK1", "Zika"]
PAP = {"T5": "gp8", "T7": "gp10A", "Dengue": "DENV", "ENO1": "ENO1", "PGK1": "PGK1", "Zika": "ZIKV"}
BASES = list("ACGU")
TOPQ = 0.15
PSEUDO = 0.5
SWITCH_LEN = 33


def freq_matrix(seqs):
    M = np.full((SWITCH_LEN, 4), PSEUDO)
    for s in seqs:
        s = str(s).upper().replace("T", "U")
        for i, ch in enumerate(s[:SWITCH_LEN]):
            if ch in BASES:
                M[i, BASES.index(ch)] += 1
    return M / M.sum(1, keepdims=True)


def top_bottom(df):
    y = np.log2(pd.to_numeric(df["Average_ON/OFF"], errors="coerce").astype(float).values)
    sw = df["Toehold_seq"].astype(str).str[:SWITCH_LEN].values
    ok = np.isfinite(y); y, sw = y[ok], sw[ok]
    N = len(y); K = round(TOPQ * N)
    order = np.argsort(-y)
    return sw[order[:K]], sw[order[-K:]], K, N


def enrichment_df(top, bot):
    Ft, Fb = freq_matrix(top), freq_matrix(bot)
    return pd.DataFrame(np.log2(Ft / Fb), columns=BASES, index=np.arange(1, SWITCH_LEN + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", default=".")
    ap.add_argument("--pattern", default="*_target_search_library_analysis_features.recomputed.with_pU.curated_txrel_Final_6.xlsx")
    ap.add_argument("--out_prefix", default="fig2_logo")
    ap.add_argument("--plot", action="store_true", help="also draw logomaker figures if available")
    args = ap.parse_args()

    per_tx, pooled_top, pooled_bot = {}, [], []
    for f in glob.glob(os.path.join(args.input_dir, args.pattern)):
        lib = next((l for l in LIBS if os.path.basename(f).startswith(l + "_")), None)
        if lib is None:
            continue
        d = pd.read_excel(f, sheet_name="Sheet1", engine="openpyxl")
        if lib != "Zika":
            d = d.iloc[:-1]
        top, bot, K, N = top_bottom(d)
        per_tx[PAP[lib]] = (enrichment_df(top, bot), K, N)
        pooled_top += list(top); pooled_bot += list(bot)  # top/bottom-K per transcript, pooled

    pooled = enrichment_df(pooled_top, pooled_bot)
    Kp = len(pooled_top)
    print(f"pooled: {Kp} top + {Kp} bottom (K = round(0.15*N) summed over 6 transcripts)")

    with pd.ExcelWriter(f"{args.out_prefix}_data.xlsx", engine="openpyxl") as xw:
        pooled.round(4).to_excel(xw, sheet_name="pooled")
        for t, (ddf, K, N) in per_tx.items():
            ddf.round(4).to_excel(xw, sheet_name=t[:28])
    print(f"wrote {args.out_prefix}_data.xlsx")

    if args.plot:
        try:
            import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt; import logomaker
            plt.rcParams.update({"svg.fonttype": "none", "font.family": "Arial", "pdf.fonttype": 42})
            colors = {"A": [0, .5, 0], "C": [0, 0, 1], "G": [1, .65, 0], "U": [1, 0, 0]}
            fig, ax = plt.subplots(figsize=(11, 2.2))
            logomaker.Logo(pooled, color_scheme=colors, ax=ax)
            ax.set_xlabel("switch position"); ax.set_ylabel("log2 enrichment (top/bottom)")
            fig.tight_layout(); fig.savefig(f"{args.out_prefix}.svg"); fig.savefig(f"{args.out_prefix}.png", dpi=200)
            print(f"wrote {args.out_prefix}.svg/.png")
        except Exception as e:
            print(f"(plot skipped: {e})")


if __name__ == "__main__":
    main()
