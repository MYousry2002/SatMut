#!/usr/bin/env python3
"""
Parse mutation info from filtered TSV files.

Input: TSVs in data/filtered/ produced by the previous step
       (they contain at least the columns: ID, ctrl_exp, DNA_mean, ctrl_mean,
        exp_mean, log2FoldChange, lfcSE, stat, pvalue, padj)

Task: From the ID column, extract mutation info encoded in the last segment:
      - Pattern ":m<REF><POS><ALT>" (e.g., ":mA192G")
        -> add columns: ref_allele, pos, alt_allele
      - Pattern ":m0" means reference (no mutation)
        -> set ref_allele, pos, alt_allele to NA

Additionally, merge in metadata from data/metadata/id_seq_map.tsv to add the Sequence column.

Output: For each input TSV, write a new TSV alongside with suffix
        "_with_mut.tsv" containing the original columns plus the three
        added columns and the Sequence column.
"""
from __future__ import annotations
import re
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
METADATA_PATH = (BASE_DIR.parent / "data" / "metadata" / "id_seq_map.tsv")

# Load ID -> Sequence map (columns: SatMut_ID, Sequence)
try:
    _meta_df = pd.read_csv(METADATA_PATH, sep="\t", dtype=str)
    if "SatMut_ID" in _meta_df.columns:
        _meta_df = _meta_df.rename(columns={"SatMut_ID": "ID"})
    if "ID" not in _meta_df.columns or "Sequence" not in _meta_df.columns:
        raise ValueError("Metadata must contain columns 'SatMut_ID' (or 'ID') and 'Sequence'")
    ID_SEQ_MAP = _meta_df[["ID", "Sequence"]].drop_duplicates()
except Exception as e:
    raise SystemExit(f"Failed to load metadata from {METADATA_PATH}: {e}")

# Build a map from m0 IDs to their sequences for fast lookup of reference sequences
_m0_df = ID_SEQ_MAP[ID_SEQ_MAP["ID"].str.endswith(":m0", na=False)].copy()
M0_SEQ_MAP = dict(zip(_m0_df["ID"], _m0_df["Sequence"]))

def id_to_m0(id_str: str) -> str | None:
    if not isinstance(id_str, str):
        return None
    idx = id_str.rfind(":m")
    if idx == -1:
        return None
    return id_str[:idx] + ":m0"

IN_DIR = Path("../data/filtered")
OUT_DIR = Path("../data/complete")  # write next to inputs with a suffix

# Regex for IDs ending with mutation info
# Examples:
#   Herpesvirus:Epstein_Barr_Virus:1602:+:mA192G  => REF=A, POS=192, ALT=G
#   HIV-1:CH058:X_Modified:m0                     => reference (no mutation)
MUT_RE = re.compile(r":m([ACGT])([0-9]+)([ACGT])$")
REF_RE = re.compile(r":m0$")


def parse_id_to_mutation(id_str: str):
    """Return (ref, pos, alt) parsed from the trailing mutation portion of ID.
    If ID denotes the reference (m0) or cannot be parsed, return (pd.NA, pd.NA, pd.NA).
    """
    if id_str is None:
        return (pd.NA, pd.NA, pd.NA)
    m = MUT_RE.search(id_str)
    if m:
        ref, pos, alt = m.group(1), m.group(2), m.group(3)
        try:
            pos = int(pos)
        except Exception:
            pos = pd.NA
        return (ref, pos, alt)
    if REF_RE.search(id_str):
        return (pd.NA, pd.NA, pd.NA)
    # Not matching expected patterns -> NA
    return (pd.NA, pd.NA, pd.NA)


def process_file(path: Path) -> Path:
    # Read as strings to preserve IDs exactly
    df = pd.read_csv(path, sep="\t", dtype=str)

    if "ID" not in df.columns:
        raise ValueError(f"Missing required column 'ID' in {path}")

    # Initialize new columns
    df["ref_allele"] = pd.NA
    df["pos"] = pd.NA
    df["alt_allele"] = pd.NA

    # Parse mutations
    parsed = df["ID"].apply(parse_id_to_mutation)
    df[["ref_allele", "pos", "alt_allele"]] = pd.DataFrame(parsed.tolist(), index=df.index)

    # Ensure pos is nullable integer type
    try:
        df["pos"] = pd.to_numeric(df["pos"], errors="coerce").astype("Int64")
    except Exception:
        # Fallback: keep as string/NA
        pass

    # Merge in actual sequence from metadata (left join on ID)
    df = df.merge(ID_SEQ_MAP, on="ID", how="left")

    # Create mut_seq (actual tile sequence) and ref_seq (sequence from matching m0 ID)
    df["mut_seq"] = df["Sequence"]
    df["ref_seq"] = df["ID"].apply(lambda s: M0_SEQ_MAP.get(id_to_m0(s), pd.NA))

    # Remove the original Sequence column; keep only the two explicit seq columns
    df = df.drop(columns=["Sequence"], errors="ignore")

    out_path = path.with_name(path.stem + "_with_mut.tsv")
    # If the input was .tsv.gz, Path.stem removes only the last suffix; handle that:
    if path.name.endswith(".tsv.gz"):
        base = path.name[:-7]  # remove .tsv.gz
        out_path = path.with_name(base + "_with_mut.tsv")

    df.to_csv(OUT_DIR / out_path.name, sep="\t", index=False)
    return OUT_DIR / out_path.name


def main():
    IN_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(list(IN_DIR.glob("*.tsv")) + list(IN_DIR.glob("*.tsv.gz")))
    if not files:
        print(f"No TSV files found in {IN_DIR.resolve()}")
        return

    for f in files:
        try:
            out = process_file(f)
            print(f"[OK] {f.name} -> {out.name}")
        except Exception as e:
            print(f"[SKIP] {f.name}: {e}")


if __name__ == "__main__":
    main()
