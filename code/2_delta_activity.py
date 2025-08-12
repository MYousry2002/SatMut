#!/usr/bin/env python3
import pandas as pd
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "data" / "complete"
OUT_DIR = BASE_DIR.parent / "results" / "delta_activity"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Process each file
for tsv_file in DATA_DIR.glob("*_with_mut.tsv"):
    print(f"Processing {tsv_file.name}...")
    df = pd.read_csv(tsv_file, sep="\t", dtype=str)

    # Ensure numeric for activity
    df["log2FoldChange"] = pd.to_numeric(df["log2FoldChange"], errors="coerce")

    # Identify reference groups by m0 ID base
    df["ref_base_id"] = df["ID"].apply(lambda s: s.split(":m")[0] if isinstance(s, str) else None)

    results = []
    for ref_id, group in df.groupby("ref_base_id"):
        ref_rows = group[group["ID"].str.endswith(":m0")]
        if ref_rows.empty:
            continue
        ref_activity = ref_rows["log2FoldChange"].iloc[0]

        for _, row in group.iterrows():
            if not isinstance(row["pos"], str) or row["pos"] == "" or row["ID"].endswith(":m0"):
                continue
            mut_activity = row["log2FoldChange"]
            results.append({
                "ref_id": ref_id,
                "pos": row["pos"],
                "ref_allele": row["ref_allele"],
                "alt_allele": row["alt_allele"],
                "ref_activity": ref_activity,
                "mut_activity": mut_activity,
                "delta_activity": mut_activity - ref_activity,
                "ref_seq": row.get("ref_seq", None),
                "mut_seq": row.get("mut_seq", None)
            })

    results_df = pd.DataFrame(results)
    out_path = OUT_DIR / f"{tsv_file.stem}_satmut_effects.tsv"
    results_df.to_csv(out_path, sep="\t", index=False)
    print(f"Saved effects to {out_path}")
