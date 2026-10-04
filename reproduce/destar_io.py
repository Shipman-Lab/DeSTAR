"""Shared data loader for the DeSTAR reproduce scripts (viz / analysis, no model).

Primary input: Supplementary Table 3 (`--supp_table3`), the publicly deposited table whose
`transcript` column holds paper names (gp8, gp10A, DENV, ZIKV, PGK1, ENO1); its rows are already
control-free and ordered by variant_id (identical, row-for-row, to the legacy *_Final_6.xlsx).
Legacy input: the six *_Final_6.xlsx tables (`--input_dir`); the appended control row of each
non-Zika library is dropped.
"""
import glob as _glob
import os as _os
import re as _re
import pandas as pd

PAPER_LIBS = ["gp8", "gp10A", "DENV", "ZIKV", "PGK1", "ENO1"]
FILE_TO_PAPER = {"T5": "gp8", "T7": "gp10A", "Dengue": "DENV", "Zika": "ZIKV", "PGK1": "PGK1", "ENO1": "ENO1"}
CONTROL_FILE_LIBS = {"T5", "T7", "Dengue", "ENO1", "PGK1"}   # non-Zika: appended control row dropped
SUPP_SHEET = "Supplementary Table 3"
DEFAULT_PATTERN = "*_target_search_library_analysis_features.recomputed.with_pU.curated_txrel_Final_6.xlsx"


def load_transcripts(supp_table3=None, input_dir=None, pattern=DEFAULT_PATTERN, sheet=SUPP_SHEET):
    """Return {paper_transcript: DataFrame} for the six libraries, in paper order."""
    out = {}
    if supp_table3:
        st = pd.read_excel(supp_table3, sheet_name=sheet, engine="openpyxl")
        for p in PAPER_LIBS:
            d = st[st["transcript"] == p].reset_index(drop=True)
            if len(d):
                out[p] = d
    else:
        for f in _glob.glob(_os.path.join(input_dir or ".", pattern)):
            filelib = _re.split(r"_target", _os.path.basename(f))[0]
            if filelib not in FILE_TO_PAPER:
                continue
            d = pd.read_excel(f, sheet_name="Sheet1", engine="openpyxl")
            if filelib in CONTROL_FILE_LIBS:
                d = d.iloc[:-1].reset_index(drop=True)
            out[FILE_TO_PAPER[filelib]] = d
    return {p: out[p] for p in PAPER_LIBS if p in out}


def add_input_args(ap):
    """Attach the standard --supp_table3 / --input_dir arguments to an ArgumentParser."""
    ap.add_argument("--supp_table3", default=None,
                    help="Deposited Supplementary Table workbook (sheet 'Supplementary Table 3'). Primary input.")
    ap.add_argument("--input_dir", default=None, help="(legacy) folder holding the six *_Final_6.xlsx tables")
    return ap
