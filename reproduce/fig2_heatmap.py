#!/usr/bin/env python3
"""
fig2_heatmap.py — Figure 2b: transcript-wide tiling heatmap of measured sensor activity.

For each of the six libraries, plots log2(Average ON/OFF) for every tiled target site ordered
by its position along the transcript (target_start_0based). Values are clipped to [-1.5, 1.5]
for the color scale (as in the figure); the exported data keep the unclipped values too.
This is a direct visualization of the measured data — no model.

Input : six *_..._Final_6.xlsx (Average_ON/OFF, target_start_0based); non-Zika control dropped.
Output: <out_prefix>_data.xlsx (one sheet per transcript: target_start, log2_ON_OFF, clipped).
        With --plot (and matplotlib), <out_prefix>.svg/.png — one horizontal heatmap strip
        per transcript, common color scale.
Requirements: numpy, pandas, openpyxl; (optional) matplotlib.
"""
import argparse, glob, os
import numpy as np, pandas as pd

LIBS = ["T5", "T7", "Dengue", "ENO1", "PGK1", "Zika"]
PAP = {"T5": "gp8", "T7": "gp10A", "Dengue": "DENV", "ENO1": "ENO1", "PGK1": "PGK1", "Zika": "ZIKV"}
CLIP = 1.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", default=".")
    ap.add_argument("--pattern", default="*_target_search_library_analysis_features.recomputed.with_pU.curated_txrel_Final_6.xlsx")
    ap.add_argument("--out_prefix", default="fig2_heatmap")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    per_tx = {}
    for f in glob.glob(os.path.join(args.input_dir, args.pattern)):
        lib = next((l for l in LIBS if os.path.basename(f).startswith(l + "_")), None)
        if lib is None:
            continue
        d = pd.read_excel(f, sheet_name="Sheet1", engine="openpyxl")
        if lib != "Zika":
            d = d.iloc[:-1]
        s = pd.to_numeric(d["target_start_0based"], errors="coerce")
        y = np.log2(pd.to_numeric(d["Average_ON/OFF"], errors="coerce").astype(float))
        t = pd.DataFrame({"target_start_0based": s, "log2_ON_OFF": y}).dropna().sort_values("target_start_0based")
        t["clipped"] = t["log2_ON_OFF"].clip(-CLIP, CLIP)
        per_tx[PAP[lib]] = t.reset_index(drop=True)

    order = [PAP[l] for l in LIBS if PAP[l] in per_tx]
    with pd.ExcelWriter(f"{args.out_prefix}_data.xlsx", engine="openpyxl") as xw:
        for t in order:
            per_tx[t].round(4).to_excel(xw, sheet_name=t[:28], index=False)
    print(f"wrote {args.out_prefix}_data.xlsx  (transcripts: {order})")

    if args.plot:
        try:
            import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
            plt.rcParams.update({"svg.fonttype": "none", "font.family": "Arial", "pdf.fonttype": 42})
            fig, axes = plt.subplots(len(order), 1, figsize=(10, 0.7 * len(order) + 1), squeeze=False)
            for ax, t in zip(axes[:, 0], order):
                v = per_tx[t]["clipped"].values.reshape(1, -1)
                im = ax.imshow(v, aspect="auto", cmap="RdBu_r", vmin=-CLIP, vmax=CLIP)
                ax.set_yticks([]); ax.set_ylabel(t, rotation=0, ha="right", va="center")
                ax.set_xticks([])
            axes[-1, 0].set_xlabel("target site position along transcript (5'→3')")
            fig.colorbar(im, ax=axes[:, 0].tolist(), fraction=0.02, pad=0.02, label="log2(ON/OFF), clipped ±1.5")
            fig.savefig(f"{args.out_prefix}.svg"); fig.savefig(f"{args.out_prefix}.png", dpi=200)
            print(f"wrote {args.out_prefix}.svg/.png")
        except Exception as e:
            print(f"(plot skipped: {e})")


if __name__ == "__main__":
    main()
