#!/usr/bin/env python3
"""
count_barcodes.py — Detectron/DeSTAR library barcode quantification from Nanopore reads.

Consolidated, parameterized replacement for the 14 per-library/per-run scripts
(<TARGET>_count_barcodes*.py). The counting, caching, and ssDNA/plasmid normalization
logic is byte-identical to those scripts; the only per-library differences (target name,
variant count, and the three technical-replicate barcode ranges) are captured in the
LIBRARY_DEFS table and selected with --target.

Pipeline:
  For each replicate group (condition x biological replicate) it counts, for the ssDNA and
  the plasmid sample, exact 13-nt library barcodes by sliding a 13-mer window across every
  read (with per-sample .npz caching), then:
     per-sample:   norm = raw_count / total_barcode_hits_in_sample
     per-barcode:  ssDNA_over_plasmid_norm = ssDNA_norm / plasmid_norm   (NaN if plasmid=0)
     per-variant:  sum the 3 technical-replicate barcodes per variant, then the same ratio.
  Output: one Excel per (condition, replicate) with sheets metadata / per_barcode / per_variant.

Inputs (all paths are CLI args):
  --index_xlsx     <TARGET>_Indexing_barcodes.xlsx  (Sample #, Description, Barcodes|Barcode #)
                   Description format: <TARGET>_(OFF|ON)_<replicate>_(ssDNA|plasmid)
  --barcodes_xlsx  <TARGET>_library_barcodes.xlsx   (Barcode_number, Barcode_seq [13 nt])
  --fastq_pass_dir directory of barcodeNN/ folders (each with *.fastq.gz)
  --out_dir, --cache_dir

Example:
  python count_barcodes.py --target Dengue \
    --index_xlsx    npJH013/Dengue_Indexing_barcodes.xlsx \
    --barcodes_xlsx npJH013/Dengue_library_barcodes.xlsx \
    --fastq_pass_dir npJH013/fastq_pass \
    --out_dir npJH013/Dengue_outputs_per_replicate \
    --cache_dir npJH013/cache_counts/Dengue

Requirements: numpy, pandas, tqdm, openpyxl.
"""
from __future__ import annotations
import argparse, gzip, hashlib, json, re, sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Optional
import numpy as np, pandas as pd
from tqdm import tqdm

# ------------------------------------------------------------------ library definitions
# target -> (n_variants_per_tech_rep, [tech1_start, tech2_start, tech3_start]) in the
# shared global library-barcode numbering. Each tech rep is a contiguous block of
# n_variants barcodes. (T4 included for completeness; not part of the DeSTAR 6-library set.)
LIBRARY_DEFS: Dict[str, Tuple[int, List[int]]] = {
    "T4":     (1535, [1, 1536, 3071]),
    "T5":     (1346, [4606, 5952, 7298]),
    "T7":     (1007, [8644, 9651, 10658]),
    "Dengue": (1454, [3523, 4977, 6431]),
    "ENO1":   (1271, [11293, 12564, 13835]),
    "PGK1":   (1136, [7885, 9021, 10157]),
    "Zika":   (1174, [1, 1175, 2349]),
}

# ------------------------------------------------------------------ utilities (verbatim)
def safe_filename(s: str) -> str:
    s = re.sub(r"[^\w\-\.]+", "_", str(s).strip()); s = re.sub(r"_+", "_", s)
    return s[:200]

def parse_sample_numbers(s: str) -> Optional[List[int]]:
    s = str(s).strip().lower()
    if s == "all": return None
    out: List[int] = []
    for part in [p.strip() for p in s.split(",") if p.strip()]:
        if "-" in part:
            a, b = part.split("-", 1); out.extend(range(min(int(a), int(b)), max(int(a), int(b)) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))

def fingerprint_inputs(files: List[Path], extra: Dict) -> str:
    h = hashlib.sha256()
    for p in files:
        st = p.stat(); h.update(str(p).encode()); h.update(str(st.st_size).encode()); h.update(str(int(st.st_mtime)).encode())
    h.update(json.dumps(extra, sort_keys=True).encode()); return h.hexdigest()

def list_fastq_gz_files(folder: Path) -> List[Path]:
    if not folder.exists(): raise FileNotFoundError(f"Barcode folder not found: {folder}")
    files = sorted(folder.glob("*.fastq.gz"))
    if not files: raise FileNotFoundError(f"No .fastq.gz files in: {folder}")
    return files

def fastq_sequences_from_gz(path: Path) -> Iterable[str]:
    with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f, 1):
            if i % 4 == 2: yield line.strip().upper()

def count_barcodes_in_sequences(seq_iter: Iterable[str], barcode_seq_to_num: Dict[str, int]) -> Tuple[Dict[int, int], int]:
    """Slide a 13-mer across each read; count exact matches to known barcodes."""
    counts: Dict[int, int] = {}; total = 0
    for seq in seq_iter:
        L = len(seq)
        if L < 13: continue
        for i in range(L - 12):
            num = barcode_seq_to_num.get(seq[i:i + 13])
            if num is not None:
                counts[num] = counts.get(num, 0) + 1; total += 1
    return counts, total

# ------------------------------------------------------------------ input readers (verbatim)
def read_sample_sheet(index_xlsx: Path) -> pd.DataFrame:
    df = pd.read_excel(index_xlsx)
    barcode_col = "Barcodes" if "Barcodes" in df.columns else ("Barcode #" if "Barcode #" in df.columns else None)
    if barcode_col is None:
        raise ValueError(f"Index sheet needs 'Barcodes' or 'Barcode #'. Found: {list(df.columns)}")
    req = ["Sample #", "Description", barcode_col]
    missing = [c for c in req if c not in df.columns]
    if missing: raise ValueError(f"Index sheet missing {missing}. Found: {list(df.columns)}")
    out = df[req].copy(); out.columns = ["sample_number", "description", "nanopore_barcode_number"]
    out["sample_number"] = pd.to_numeric(out["sample_number"], errors="raise").astype(int)
    out["description"] = out["description"].astype(str).str.strip()
    out["nanopore_barcode_number"] = pd.to_numeric(out["nanopore_barcode_number"], errors="raise").astype(int)
    return out

def read_library_barcode_map(barcodes_xlsx: Path) -> pd.DataFrame:
    df = pd.read_excel(barcodes_xlsx)
    req = ["Barcode_number", "Barcode_seq"]
    missing = [c for c in req if c not in df.columns]
    if missing: raise ValueError(f"Barcode map missing {missing}. Found: {list(df.columns)}")
    out = df[req].copy(); out.columns = ["barcode_number", "barcode_seq"]
    out["barcode_number"] = pd.to_numeric(out["barcode_number"], errors="raise").astype(int)
    out["barcode_seq"] = out["barcode_seq"].astype(str).str.strip().str.upper()
    if (out["barcode_seq"].str.len() != 13).any():
        raise ValueError("Barcode_seq must be 13 nt.")
    if out["barcode_number"].duplicated().any() or out["barcode_seq"].duplicated().any():
        raise ValueError("Duplicate barcode_number or barcode_seq in map.")
    return out.sort_values("barcode_number").reset_index(drop=True)

# ------------------------------------------------------------------ variant<->tech mapping (parameterized)
def compute_variant_tech_mapping(target: str) -> pd.DataFrame:
    if target not in LIBRARY_DEFS:
        raise ValueError(f"Unknown target '{target}'. Known: {sorted(LIBRARY_DEFS)}")
    n, starts = LIBRARY_DEFS[target]
    base = np.arange(1, n + 1, dtype=int)
    vt = pd.concat([pd.DataFrame({"variant_id": base, "tech_rep": t + 1, "barcode_number": s + base - 1})
                    for t, s in enumerate(starts)], ignore_index=True)
    for t, s in enumerate(starts):  # sanity: contiguous block per tech rep
        sub = vt.loc[vt.tech_rep == t + 1, "barcode_number"]
        assert int(sub.min()) == s and int(sub.max()) == s + n - 1, f"{target} tech{t+1} range mismatch"
    return vt

def parse_description(desc: str, target: str) -> Tuple[str, str, str, int]:
    d = re.sub(r"(?i)\bplsmid\b", "plasmid", str(desc).strip())
    m = re.match(rf"^({re.escape(target)})_(OFF|ON)_(\d+)_(ssDNA|plasmid)$", d, re.IGNORECASE)
    if not m:
        raise ValueError(f"Description '{desc}' != '{target}_(OFF|ON)_<rep>_(ssDNA|plasmid)'")
    return m.group(1), m.group(2).upper(), m.group(4).lower(), int(m.group(3))

# ------------------------------------------------------------------ sample runs + cache (verbatim)
@dataclass
class SampleRun:
    sample_number: int; description: str; nanopore_barcode_number: int
    barcode_folder: Path; fastq_files: List[Path]

def build_sample_runs(sample_df, fastq_pass_dir: Path, sample_numbers=None) -> List[SampleRun]:
    sub = sample_df if sample_numbers is None else sample_df[sample_df.sample_number.isin(sample_numbers)]
    if sub.empty: raise ValueError(f"No samples for {sample_numbers}")
    runs = []
    for _, r in sub.sort_values("sample_number").iterrows():
        bc = int(r["nanopore_barcode_number"]); folder = fastq_pass_dir / f"barcode{bc:02d}"
        runs.append(SampleRun(int(r["sample_number"]), str(r["description"]), bc, folder, list_fastq_gz_files(folder)))
    return runs

def get_counts_cached(run: SampleRun, barcode_map, cache_dir: Path, force=False) -> Tuple[np.ndarray, int, bool, Path]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    bcseq_to_num = dict(zip(barcode_map.barcode_seq, barcode_map.barcode_number))
    bm_fp = hashlib.sha256((";".join(f"{n}:{s}" for n, s in zip(barcode_map.barcode_number, barcode_map.barcode_seq))).encode()).hexdigest()
    fp = fingerprint_inputs(run.fastq_files, {"barcode_map_fp": bm_fp, "sample_number": run.sample_number,
                                              "description": run.description, "np_barcode_folder": str(run.barcode_folder)})
    cache_path = cache_dir / f"{safe_filename(run.description)}__sample{run.sample_number:03d}__npbc{run.nanopore_barcode_number:02d}__{fp[:16]}.npz"
    if cache_path.exists() and not force:
        z = np.load(cache_path, allow_pickle=True); return z["counts_arr"], int(z["total_hits"]), True, cache_path
    counts: Dict[int, int] = {}; total = 0
    for fq in tqdm(run.fastq_files, desc=f"Counting S{run.sample_number} {run.description}", unit="fastq", leave=False):
        c, t = count_barcodes_in_sequences(fastq_sequences_from_gz(fq), bcseq_to_num)
        total += t
        for k, v in c.items(): counts[k] = counts.get(k, 0) + v
    counts_arr = np.array([counts.get(int(n), 0) for n in barcode_map.barcode_number], dtype=np.int64)
    np.savez_compressed(cache_path, counts_arr=counts_arr, total_hits=total)
    return counts_arr, total, False, cache_path

# ------------------------------------------------------------------ output tables (verbatim)
def build_per_barcode_table(bm, counts_arr, total, prefix) -> pd.DataFrame:
    df = bm.copy(); df[f"{prefix}_raw"] = counts_arr
    df[f"{prefix}_norm"] = (df[f"{prefix}_raw"] / total) if total > 0 else 0.0
    return df[["barcode_number", "barcode_seq", f"{prefix}_raw", f"{prefix}_norm"]]

def build_per_variant_table(bm, counts_arr, total, prefix, vt) -> pd.DataFrame:
    pb = bm.copy(); pb["raw_count"] = counts_arr
    pb["norm_count"] = (pb["raw_count"] / total) if total > 0 else 0.0
    pb = pb.merge(vt, on="barcode_number", how="left")
    wr = pb.dropna(subset=["variant_id"]).pivot_table(index="variant_id", columns="tech_rep", values="raw_count", aggfunc="sum", fill_value=0).reset_index()
    wr.columns = ["variant_id"] + [f"{prefix}_raw_tech{int(c)}" for c in wr.columns[1:]]
    wn = pb.dropna(subset=["variant_id"]).pivot_table(index="variant_id", columns="tech_rep", values="norm_count", aggfunc="sum", fill_value=0.0).reset_index()
    wn.columns = ["variant_id"] + [f"{prefix}_norm_tech{int(c)}" for c in wn.columns[1:]]
    merged = wr.merge(wn, on="variant_id", how="left")
    ncols = [c for c in merged.columns if c.startswith(f"{prefix}_norm_tech")]
    merged[f"{prefix}_mean_norm"] = merged[ncols].mean(axis=1) if ncols else 0.0
    cols = ["variant_id", f"{prefix}_mean_norm", f"{prefix}_raw_tech1", f"{prefix}_raw_tech2", f"{prefix}_raw_tech3",
            f"{prefix}_norm_tech1", f"{prefix}_norm_tech2", f"{prefix}_norm_tech3"]
    for c in cols:
        if c not in merged.columns: merged[c] = 0.0 if ("norm" in c or "mean" in c) else 0
    return merged[cols]

def combine_replicate(target, condition, rep, ss_run, pl_run, bm, vt, cache_dir, out_dir, force=False) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    ss_c, ss_t, ss_cached, ss_cp = get_counts_cached(ss_run, bm, cache_dir, force)
    pl_c, pl_t, pl_cached, pl_cp = get_counts_cached(pl_run, bm, cache_dir, force)
    per_barcode = build_per_barcode_table(bm, ss_c, ss_t, "ssDNA").merge(
        build_per_barcode_table(bm, pl_c, pl_t, "plasmid"), on=["barcode_number", "barcode_seq"], how="inner")
    with np.errstate(divide="ignore", invalid="ignore"):
        per_barcode["ssDNA_over_plasmid_norm"] = per_barcode.ssDNA_norm / per_barcode.plasmid_norm
        per_barcode.loc[per_barcode.plasmid_norm == 0, "ssDNA_over_plasmid_norm"] = np.nan
    per_variant = build_per_variant_table(bm, ss_c, ss_t, "ssDNA", vt).merge(
        build_per_variant_table(bm, pl_c, pl_t, "plasmid", vt), on="variant_id", how="inner")
    with np.errstate(divide="ignore", invalid="ignore"):
        per_variant["mean_norm_ratio"] = per_variant.ssDNA_mean_norm / per_variant.plasmid_mean_norm
        per_variant.loc[per_variant.plasmid_mean_norm == 0, "mean_norm_ratio"] = np.nan
        for t in (1, 2, 3):
            per_variant[f"norm_tech{t}_ratio"] = per_variant[f"ssDNA_norm_tech{t}"] / per_variant[f"plasmid_norm_tech{t}"]
            per_variant.loc[per_variant[f"plasmid_norm_tech{t}"] == 0, f"norm_tech{t}_ratio"] = np.nan
    meta = pd.DataFrame([{"timestamp": datetime.now().isoformat(timespec="seconds"), "target": target,
        "condition": condition, "replicate": rep,
        "ssDNA_sample_number": ss_run.sample_number, "ssDNA_description": ss_run.description,
        "ssDNA_nanopore_barcode_number": ss_run.nanopore_barcode_number, "ssDNA_total_library_barcode_hits": ss_t,
        "ssDNA_used_cache": ss_cached, "ssDNA_cache_path": str(ss_cp),
        "plasmid_sample_number": pl_run.sample_number, "plasmid_description": pl_run.description,
        "plasmid_nanopore_barcode_number": pl_run.nanopore_barcode_number, "plasmid_total_library_barcode_hits": pl_t,
        "plasmid_used_cache": pl_cached, "plasmid_cache_path": str(pl_cp)}])
    out_path = out_dir / f"{target}_{condition}_{rep}.xlsx"
    with pd.ExcelWriter(out_path, engine="openpyxl") as xw:
        meta.to_excel(xw, index=False, sheet_name="metadata")
        per_barcode.to_excel(xw, index=False, sheet_name="per_barcode")
        per_variant.to_excel(xw, index=False, sheet_name="per_variant")
    return out_path

# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, choices=sorted(LIBRARY_DEFS), help="library/transcript")
    ap.add_argument("--index_xlsx", required=True)
    ap.add_argument("--barcodes_xlsx", required=True)
    ap.add_argument("--fastq_pass_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--samples", default="all", help='e.g. "1-12", "1,2,3", or "all"')
    ap.add_argument("--force", action="store_true", help="ignore cache, recompute")
    args = ap.parse_args()

    sample_df = read_sample_sheet(Path(args.index_xlsx))
    barcode_map = read_library_barcode_map(Path(args.barcodes_xlsx))
    vt = compute_variant_tech_mapping(args.target)
    missing = sorted(set(vt.barcode_number) - set(barcode_map.barcode_number))[:10]
    if missing:
        print(f"WARNING: barcode_map missing expected barcodes for {args.target}: {missing}", file=sys.stderr)

    runs = build_sample_runs(sample_df, Path(args.fastq_pass_dir), parse_sample_numbers(args.samples))
    by_key: Dict[Tuple[str, int], Dict[str, SampleRun]] = {}
    for r in runs:
        tgt, cond, mat, rep = parse_description(r.description, args.target)
        by_key.setdefault((cond, rep), {})
        if mat in by_key[(cond, rep)]:
            raise ValueError(f"Duplicate material '{mat}' for ({cond},{rep}): {r.description}")
        by_key[(cond, rep)][mat] = r

    valid = [k for k in sorted(by_key) if "ssdna" in by_key[k] and "plasmid" in by_key[k]]
    if not valid:
        raise ValueError("No replicate group has BOTH ssDNA and plasmid.")
    print(f"{args.target}: {len(valid)} replicate groups with ssDNA+plasmid.")
    for (cond, rep) in tqdm(valid, desc="Building workbooks", unit="wb"):
        out = combine_replicate(args.target, cond, rep, by_key[(cond, rep)]["ssdna"], by_key[(cond, rep)]["plasmid"],
                                barcode_map, vt, Path(args.cache_dir), Path(args.out_dir), args.force)
        print(f"Saved: {out}")
    print(f"Done. Output: {args.out_dir}")

if __name__ == "__main__":
    main()
