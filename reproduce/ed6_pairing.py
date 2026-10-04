#!/usr/bin/env python3
"""
ed6_pairing.py — Extended Data Fig. 6b,c: OFF-state stem base-pairing of the DeSTAR
toehold switch, computed with ViennaRNA in two states:

  (1) "switch alone"  — the free 59-nt switch RNA (Toehold_seq), no target.
  (2) "target bound"  — the invasion-relevant intermediate: the 25-nt toehold is
      hybridized to the target, the 8-nt invader is prevented from pairing, and the
      stem is free to breathe. Includes the toehold coaxial-stacking effect on bp1.

For each sensor and each stem pair bp1..bp8 we read the ensemble base-pairing
probability P_close; P_open = 1 - P_close; effective ΔG to open = -RT*ln(P_open/P_close),
RT = 0.61633 kcal/mol at 37 C. Stem pair bp_k = switch nt (25+k) paired with nt (60-k)
in 1-based numbering, i.e. bp1 = (26,59).

METHOD (exact — this reproduces the published ED6 values):
  switch alone :  fc = RNA.fold_compound(sw);  fc.pf();  bpp = fc.bpp()
                  P_close(bp_k) = bpp[25+k][60-k]
  target bound :  fc = RNA.fold_compound(sw + "&" + tg)          # ViennaRNA cofold
                  hard-constrain the toehold BOUND:   fc.hc_add_bp(i, n+(34-i))  i=1..25
                  hard-constrain the invader UNPAIRED: fc.hc_add_up(n+p)          p=1..8
                  fc.pf();  bpp = fc.bpp()
                  P_close(bp_k) = bpp[25+k][60-k] + bpp[60-k][25+k]   (n = len(sw) = 59)

The ± reported in ED Fig. 5c is the SEM across the 6 transcripts (per-transcript mean
P_open -> ΔG, then mean ± SEM over the six).

Requirements (PIN for reproducibility)
--------------------------------------
    ViennaRNA (python bindings) == 2.7.0     # bp1 (terminal, coaxial-junction pair)
                                             # probability is model-parameter sensitive;
                                             # interior pairs are version-robust.
    pandas, numpy, openpyxl

Input : Supplementary Table 3 (--supp_table3) [primary], or the legacy *_Final_6.xlsx per-library tables (Toehold_seq, target_33nt,
        Average_ON/OFF). The last appended control row of each non-Zika file is dropped.

Outputs
-------
    ed6_pairing_perSensor.csv   one row/sensor: switchalone_P1..P8, targetbound_P1..P8
    ed6_pairing_perTranscript.csv  per-transcript mean P_close per bp, both states (panel b)
    ed6_pairing_summary.csv     per-bp P_close/P_open/ΔG ± SEM, both states (panel c)

Published ED6 values reproduced (ViennaRNA 2.7.0, n = 7,383):
    switch alone  bp1 ~25% open
    target bound  bp1 17.4% open / ΔG 0.96 ± 0.03 ;  bp2 1.5% open / ΔG 2.57 ± 0.04
"""
import argparse, glob, os, re, time
import destar_io
import numpy as np, pandas as pd
import RNA

RT = 0.61633  # kcal/mol at 37 C


def switch_alone_Pclose(sw):
    """Ensemble P_close for stem pairs bp1..bp8 of the free 59-nt switch."""
    fc = RNA.fold_compound(sw)
    fc.pf()
    bpp = fc.bpp()  # 1-based
    return [bpp[25 + k][60 - k] for k in range(1, 9)]


def target_bound_Pclose(sw, tg):
    """Ensemble P_close for bp1..bp8 with toehold clamped bound and invader clamped
    unpaired (invasion forbidden, stem free to breathe)."""
    n = len(sw)
    fc = RNA.fold_compound(sw + "&" + tg)
    for i in range(1, 26):                 # toehold: switch i <-> target (34-i)  (25 bp)
        try: fc.hc_add_bp(i, n + (34 - i))
        except Exception: pass
    for p in range(1, 9):                  # invader: target nt 1..8 forced unpaired
        try: fc.hc_add_up(n + p)
        except Exception: pass
    fc.pf()
    bpp = fc.bpp()
    return [bpp[25 + k][60 - k] + bpp[60 - k][25 + k] for k in range(1, 9)]


def main():
    ap = argparse.ArgumentParser()
    destar_io.add_input_args(ap)
    ap.add_argument("--out_prefix", default="ed6_pairing")
    args = ap.parse_args()

    RNA.cvar.temperature = 37.0
    rows = []
    t0 = time.time()
    for tx, df in destar_io.load_transcripts(args.supp_table3, args.input_dir).items():
        for _, r in df.iterrows():
            sw = str(r["Toehold_seq"]).upper().replace("T", "U")
            tg = str(r["target_33nt"]).upper().replace("T", "U")
            if len(sw) != 59 or len(tg) != 33:
                continue
            if set(sw) - set("ACGU") or set(tg) - set("ACGU"):
                continue
            sa = switch_alone_Pclose(sw)
            tb = target_bound_Pclose(sw, tg)
            rec = {"transcript": tx, "variant_id": r.get("variant_id", "")}
            for k in range(8):
                rec[f"switchalone_P{k+1}"] = sa[k]
                rec[f"targetbound_P{k+1}"] = tb[k]
            rows.append(rec)
        print(f"  {tx}: {len(rows)} cumulative  {time.time()-t0:.0f}s", flush=True)

    d = pd.DataFrame(rows)
    d.to_csv(f"{args.out_prefix}_perSensor.csv", index=False)
    n = len(d)
    txs = sorted(d["transcript"].unique())

    # panel b: per-transcript mean P_close per bp, both states
    perTx = []
    for t in txs:
        g = d[d.transcript == t]
        row = {"transcript": t, "n": len(g)}
        for k in range(1, 9):
            row[f"switchalone_bp{k}"] = round(g[f"switchalone_P{k}"].mean(), 4)
            row[f"targetbound_bp{k}"] = round(g[f"targetbound_P{k}"].mean(), 4)
        perTx.append(row)
    pd.DataFrame(perTx).to_csv(f"{args.out_prefix}_perTranscript.csv", index=False)

    # panel c: per-bp summary with ΔG mean ± SEM across the 6 transcripts
    def summarize(state):
        out = []
        for k in range(1, 9):
            col = f"{state}_P{k}"
            pc_tx = np.array([d.loc[d.transcript == t, col].mean() for t in txs])
            po_tx = 1.0 - pc_tx
            dg_tx = -RT * np.log(po_tx / pc_tx)
            out.append({"state": state, "bp": k,
                        "P_close_pct": round(pc_tx.mean() * 100, 2),
                        "P_open_pct": round(po_tx.mean() * 100, 2),
                        "P_open_pct_SEM": round(po_tx.std(ddof=1) / np.sqrt(len(txs)) * 100, 2),
                        "dG_open_kcal_per_mol": round(dg_tx.mean(), 3),
                        "dG_open_SEM": round(dg_tx.std(ddof=1) / np.sqrt(len(txs)), 3)})
        return out

    summ = summarize("switchalone") + summarize("targetbound")
    pd.DataFrame(summ).to_csv(f"{args.out_prefix}_summary.csv", index=False)

    print(f"\nED6 pairing (ViennaRNA {RNA.__version__}, n = {n} sensors, {len(txs)} transcripts):")
    for state, label in [("targetbound", "TARGET BOUND (panel c)"), ("switchalone", "switch alone")]:
        print(f"\n  {label}")
        for s in [x for x in summ if x["state"] == state][:3]:
            print(f"    bp{s['bp']}: P_open {s['P_open_pct']:.1f}%  "
                  f"ΔG {s['dG_open_kcal_per_mol']:.2f} ± {s['dG_open_SEM']:.2f} kcal/mol")
    print(f"\nwrote {args.out_prefix}_perSensor.csv, _perTranscript.csv, _summary.csv")


if __name__ == "__main__":
    main()
