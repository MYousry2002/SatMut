#!/usr/bin/env python3

import pandas as pd
from pathlib import Path

RAW_DIR = Path("../data/raw")
OUT_DIR = Path("../data/filtered")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def out_name_for(p: Path) -> Path:
    # Make output file name with suffix
    name = p.name
    if name.endswith(".tsv.gz"):
        name = name[:-7]
    elif name.endswith(".tsv"):
        name = name[:-4]
    return OUT_DIR / f"{name}_virus_satmut.tsv"

# Find all tsv and tsv.gz files
files = sorted(list(RAW_DIR.glob("*.tsv")) + list(RAW_DIR.glob("*.tsv.gz")))
if not files:
    print(f"No TSV files found in {RAW_DIR.resolve()}")
    raise SystemExit(0)

for f in files:
    try:
        # Read the TSV
        df = pd.read_csv(f, sep="\t", dtype=str)
    except Exception as e:
        print(f"[SKIP] {f.name}: failed to read ({e})")
        continue

    if "project" not in df.columns:
        print(f"[SKIP] {f.name}: no 'project' column")
        continue

    # Filter by project and keep only IDs with ':m' in last part after the last ':'
    proj_norm = df["project"].astype(str).str.strip().str.lower()
    filtered = df[proj_norm == "viral_satmut"].copy()
    filtered = filtered[filtered["ID"].str.split(":").str[-1].str.startswith("m")]

    # Save filtered file with only selected columns
    cols_to_keep = ["ID", "ctrl_mean", "exp_mean", 
                    "log2FoldChange", "lfcSE", "stat", "pvalue", "padj"]
    filtered = filtered[cols_to_keep]

    out_path = out_name_for(f)
    filtered.to_csv(out_path, sep="\t", index=False)

    print(f"[OK] {f.name:35s} -> {out_path.name:35s}  kept {len(filtered):6d} / {len(df):6d}")