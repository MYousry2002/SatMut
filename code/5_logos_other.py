#!/usr/bin/env python3
"""
Generate sequence effect logos from SatMut outputs.

Inputs:
  - One or more TSVs produced by satmut_stats.py, typically located in
    ../results/satmut_stats/*_satmut_effects_stats.tsv

What it does:
  - For each tile (ref_id), builds a 4xL matrix (A,C,G,T x positions) where each
    cell is the chosen metric (default: post_log2Skew) for the mutation that
    changes the reference base at that position to the alt base.
  - Plots a sequence logo where positive effects point up and negative down.
  - Alternatively, in IC mode, converts scores to information content (bits)
    logos using a softmax transformation with temperature.

Requirements:
  - pandas, numpy, matplotlib, and logomaker (pip install logomaker)

Usage examples:
  python code/5_logos_other.py \
    --in-dir results/satmut_stats \
    --pattern 'Jurkat_activity_satmut_effects_stats.tsv' \
    --metric post_log2Skew \
    --agg mean \
    --fdr-col log2Skew_fdr --fdr-max 0.1 \
    --out-dir results/logos \
    --bp-per-inch 8 --min-width 12 --height 4 --font-size 10 \
    --cell-type Jurkat

  python code/logos_other.py --mode ic \
    --in-dir ../results/satmut_stats \
    --pattern 'Jurkat_activity_satmut_effects_stats.tsv' \
    --metric post_log2Skew \
    --agg mean \
    --temperature 1.0 \
    --out-dir ../results/logos_ic \
    --bp-per-inch 8 --min-width 12 --height 4 --font-size 10
    --cell-type Jurkat

Notes:
  - If multiple rows map to the same (position, alt_allele), values are aggregated
    by --agg (mean|max|median).
  - Positions are treated as 1-based in input; plotting uses 1..L on x-axis.
  - If L cannot be inferred from ref_seq length, it falls back to max(pos).
  - In IC mode, scores are converted to probabilities using a temperature-scaled
    softmax, and logos represent information content in bits.
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Try to import logomaker; provide a friendly message if missing
try:
    import logomaker
    _HAS_LOGOMAKER = True
except Exception:
    _HAS_LOGOMAKER = False


NUCS = ["A", "C", "G", "T"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate sequence effect logos from SatMut stats TSVs")
    p.add_argument("--in-dir", type=str, default="../results/satmut_stats",
                   help="Directory containing *_satmut_effects_stats.tsv inputs")
    p.add_argument("--pattern", type=str, default="*_satmut_effects_stats.tsv",
                   help="Glob pattern for input files inside --in-dir")
    p.add_argument("--metric", type=str, default="post_log2Skew",
                   help="Column to visualize (e.g., post_log2Skew, log2Skew, z)")
    p.add_argument("--agg", type=str, choices=["mean", "max", "median"], default="mean",
                   help="How to aggregate duplicates per (pos,alt)")
    p.add_argument("--fdr-col", type=str, default="log2Skew_fdr",
                   help="FDR column name for filtering (set to '' to disable)")
    p.add_argument("--fdr-max", type=float, default=None,
                   help="Max FDR to include (e.g., 0.1). If None, no FDR filter.")
    p.add_argument("--min-pos", type=int, default=None,
                   help="Minimum position to include (1-based). Optional.")
    p.add_argument("--max-pos", type=int, default=None,
                   help="Maximum position to include (1-based). Optional.")
    p.add_argument("--out-dir", type=str, default="../results/logos",
                   help="Output directory for logo images")
    p.add_argument("--format", type=str, choices=["png", "pdf", "svg"], default="png",
                   help="Image format")
    p.add_argument("--dpi", type=int, default=200, help="DPI for raster formats")
    p.add_argument("--title-metric", action="store_true",
                   help="Append metric name to plot title")
    p.add_argument("--cap", type=float, default=None,
                   help="Cap absolute values at this threshold before plotting (optional; ignored in IC mode)")
    p.add_argument("--quiet", action="store_true", help="Reduce logging output")
    p.add_argument("--mode", type=str, choices=["effect", "ic"], default="effect",
                   help="Logo mode: 'effect' for effect size logos, 'ic' for information content logos using softmax")
    p.add_argument("--temperature", type=float, default=1.0,
                   help="Temperature for softmax transformation in IC mode (default 1.0)")
    p.add_argument("--bp-per-inch", type=float, default=8.0,
                   help="Horizontal scale: bases per inch (smaller -> wider figure). Default 8.")
    p.add_argument("--min-width", type=float, default=12.0,
                   help="Minimum figure width in inches. Default 12.")
    p.add_argument("--height", type=float, default=4.0,
                   help="Figure height in inches. Default 4.")
    p.add_argument("--xtick-step", type=int, default=None,
                   help="Force x-tick step (positions). If None, auto-select up to ~25 ticks.")
    p.add_argument("--font-size", type=float, default=10.0,
                   help="Base font size for labels/ticks. Default 10.")
    p.add_argument("--no-tight", action="store_true",
                   help="Do not use bbox_inches='tight' when saving (keeps exact figsize)")
    p.add_argument("--tight-pad", type=float, default=0.03,
                   help="Padding (inches) when saving with bbox_inches='tight' (ignored with --no-tight). Default 0.03")
    p.add_argument("--cell-type", type=str, default="",
                   help="Optional cell type label to append to output filenames and titles (e.g., K562, HEK293)")
    return p.parse_args()
def _compute_fig_params(L: int, bp_per_inch: float, min_width: float, height: float, xtick_step: int | None):
    width = max(min_width, L / max(1e-6, bp_per_inch))
    if xtick_step is None:
        # Aim for ~20-25 ticks
        target = max(10, min(25, int(L / 10)))
        step = max(1, int(np.ceil(L / target)))
    else:
        step = max(1, int(xtick_step))
    ticks = np.arange(1, L + 1, step)
    if ticks[-1] != L:
        ticks = np.append(ticks, L)
    return (width, height), ticks


def _log(msg: str, quiet: bool = False):
    if not quiet:
        print(msg)


def _safe_numeric(s, default=np.nan):
    try:
        return float(s)
    except Exception:
        return default


def aggregate_values(vals: pd.Series, how: str) -> float:
    v = pd.to_numeric(vals, errors="coerce").dropna()
    if v.empty:
        return np.nan
    if how == "mean":
        return float(v.mean())
    if how == "max":
        return float(v.max())
    if how == "median":
        return float(v.median())
    return float(v.mean())


def build_effect_matrix(df: pd.DataFrame, metric: str, agg: str,
                        min_pos: int | None, max_pos: int | None) -> tuple[pd.DataFrame, int]:
    """Return (matrix_df, L) where matrix_df has columns A,C,G,T and index 1..L.
    Values are aggregated metric per (pos, alt_allele).
    """
    # Determine length
    L = None
    if "ref_seq" in df.columns and isinstance(df["ref_seq"].dropna().head(1).tolist()[0] if df["ref_seq"].dropna().size else None, str):
        try:
            # Try to use the longest available ref_seq for this tile
            L = int(df["ref_seq"].dropna().map(len).max())
        except Exception:
            L = None
    if L is None:
        # fallback to max(pos)
        if "pos" in df.columns:
            L = int(pd.to_numeric(df["pos"], errors="coerce").max())
        else:
            raise ValueError("Cannot infer sequence length: need ref_seq or pos column")

    # Position filter
    pos_series = pd.to_numeric(df["pos"], errors="coerce")
    if min_pos is not None:
        df = df[pos_series >= min_pos]
    if max_pos is not None:
        df = df[pos_series <= max_pos]

    # Init matrix
    mat = pd.DataFrame(0.0, index=range(1, L + 1), columns=NUCS, dtype=float)

    # For each (pos, alt), aggregate metric
    if metric not in df.columns:
        raise ValueError(f"Metric column '{metric}' not found in dataframe")

    # Clean numeric metric
    df = df.copy()
    df[metric] = pd.to_numeric(df[metric], errors="coerce")
    df = df.dropna(subset=["pos", "alt_allele", metric])

    # Cap if needed will be applied at plotting time; we aggregate raw
    grouped = df.groupby([df["pos"].astype(int), "alt_allele"], dropna=True)[metric]
    agg_series = grouped.apply(lambda s: aggregate_values(s, agg))

    # Fill matrix
    for (pos, alt), val in agg_series.dropna().items():
        if alt in NUCS and 1 <= int(pos) <= L:
            mat.at[int(pos), alt] = float(val)

    return mat, L


def build_score_matrix(df: pd.DataFrame, metric: str, agg: str,
                       min_pos: int | None, max_pos: int | None) -> tuple[pd.DataFrame, int]:
    """Return (score_matrix_df, L) where rows are positions 1..L and columns A,C,G,T.
    Reference base scores are set to 0, alt base scores are aggregated metric values.
    Missing values default to 0.
    """
    # Determine length
    L = None
    if "ref_seq" in df.columns and isinstance(df["ref_seq"].dropna().head(1).tolist()[0] if df["ref_seq"].dropna().size else None, str):
        try:
            L = int(df["ref_seq"].dropna().map(len).max())
        except Exception:
            L = None
    if L is None:
        if "pos" in df.columns:
            L = int(pd.to_numeric(df["pos"], errors="coerce").max())
        else:
            raise ValueError("Cannot infer sequence length: need ref_seq or pos column")

    # Position filter
    pos_series = pd.to_numeric(df["pos"], errors="coerce")
    if min_pos is not None:
        df = df[pos_series >= min_pos]
    if max_pos is not None:
        df = df[pos_series <= max_pos]

    # Initialize matrix with zeros
    mat = pd.DataFrame(0.0, index=range(1, L + 1), columns=NUCS, dtype=float)

    # Try to use ref_seq if present
    ref_seq = None
    if "ref_seq" in df.columns:
        ref_seq_series = df["ref_seq"].dropna()
        if not ref_seq_series.empty:
            ref_seq = ref_seq_series.iloc[0]

    if isinstance(ref_seq, str) and len(ref_seq) >= L:
        for i, base in enumerate(ref_seq[:L], start=1):
            if base in NUCS and 1 <= i <= L:
                mat.at[i, base] = 0.0
    else:
        # Fallback: infer reference base per position from ref_allele if available
        if "ref_allele" in df.columns and "pos" in df.columns:
            tmp = (
                df.dropna(subset=["pos", "ref_allele"]) 
                  .assign(pos=lambda x: pd.to_numeric(x["pos"], errors="coerce").astype("Int64"))
            )
            tmp = tmp[tmp["pos"].notna()]
            if not tmp.empty:
                # pick the most frequent ref_allele per position
                mode_ref = (
                    tmp.groupby("pos")["ref_allele"]
                       .agg(lambda s: pd.Series(s).mode().iloc[0] if not pd.Series(s).mode().empty else np.nan)
                )
                for p, base in mode_ref.items():
                    if pd.notna(base) and int(p) >= 1 and int(p) <= L and base in NUCS:
                        mat.at[int(p), base] = 0.0
        # else: leave zeros (uniform prior -> IC=0 at positions without info)

    # Aggregate alt base scores from metric
    if metric not in df.columns:
        raise ValueError(f"Metric column '{metric}' not found in dataframe")

    df = df.copy()
    df[metric] = pd.to_numeric(df[metric], errors="coerce")
    df = df.dropna(subset=["pos", "alt_allele", metric])

    grouped = df.groupby([df["pos"].astype(int), "alt_allele"], dropna=True)[metric]
    agg_series = grouped.apply(lambda s: aggregate_values(s, agg))

    for (pos, alt), val in agg_series.dropna().items():
        if alt in NUCS and 1 <= int(pos) <= L:
            mat.at[int(pos), alt] = float(val)

    return mat, L


def score_to_ic_matrix(score_mat: pd.DataFrame, temperature: float = 1.0) -> pd.DataFrame:
    """Convert score matrix to information content (IC) matrix.

    Args:
        score_mat: DataFrame with shape (L x 4) columns A,C,G,T and rows positions.
        temperature: float temperature for softmax scaling.

    Returns:
        DataFrame of same shape with IC-scaled heights.
    """
    # Compute softmax probabilities per position with temperature
    # softmax(x_i) = exp(x_i / T) / sum_j exp(x_j / T)
    scores = score_mat.values  # shape (L,4)
    # To avoid overflow, subtract max per row
    max_scores = np.max(scores / temperature, axis=1, keepdims=True)
    exp_scores = np.exp((scores / temperature) - max_scores)
    sum_exp = np.sum(exp_scores, axis=1, keepdims=True)
    probs = exp_scores / sum_exp  # shape (L,4)

    # Compute entropy in bits per position: H_i = -sum_j p_ij * log2(p_ij)
    # Handle zeros in probs safely by masking
    with np.errstate(divide='ignore', invalid='ignore'):
        log2_probs = np.log2(probs, where=(probs > 0))
    entropy = -np.sum(probs * log2_probs, axis=1)  # shape (L,)

    # Information content per position: R_i = 2 - H_i (max entropy for 4 bases is 2 bits)
    R = 2 - entropy  # shape (L,)

    # IC matrix heights = probs * R_i[:, None]
    heights = probs * R[:, None]

    ic_mat = pd.DataFrame(heights, index=score_mat.index, columns=score_mat.columns)
    return ic_mat


def plot_logo_for_tile(tile_df: pd.DataFrame, ref_id: str, out_path: Path, metric: str,
                       agg: str, min_pos: int | None, max_pos: int | None,
                       fmt: str = "png", dpi: int = 200, title_metric: bool = False,
                       cap: float | None = None, quiet: bool = False,
                       mode: str = "effect", temperature: float = 1.0,
                       bp_per_inch: float = 8.0, min_width: float = 12.0,
                       height: float = 4.0, xtick_step: int | None = None,
                       font_size: float = 10.0, no_tight: bool = False, pad_inches: float = 0.03,
                       cell_type: str = ""):
    if not _HAS_LOGOMAKER:
        raise RuntimeError(
            "logomaker is required for logos. Install via: pip install logomaker"
        )

    if mode == "effect":
        mat, L = build_effect_matrix(tile_df, metric=metric, agg=agg, min_pos=min_pos, max_pos=max_pos)

        # Optionally cap values to improve visual range
        if cap is not None and cap > 0:
            mat = mat.clip(lower=-abs(cap), upper=abs(cap))

        figsize, xticks = _compute_fig_params(L, bp_per_inch, min_width, height, xtick_step)
        _log(f"    figsize={figsize}", quiet)
        fig, ax = plt.subplots(figsize=figsize)
        # Center around zero; positive up, negative down
        logo = logomaker.Logo(mat, center_values=True, ax=ax)
        # Add baseline at 0 for visual reference
        logo.ax.axhline(0, linewidth=0.8, color='k', alpha=0.3)
        # Minimize extra whitespace around the axes
        logo.ax.margins(x=0)
        try:
            # Slightly tighten subplot params without clipping tick labels
            plt.subplots_adjust(left=0.04, right=0.995, top=0.90, bottom=0.18)
        except Exception:
            pass
        logo.style_spines(visible=False)
        logo.style_spines(spines=['left', 'bottom'], visible=True)
        logo.ax.set_xlim([0.5, L + 0.5])
        logo.ax.set_xticks(xticks)
        logo.ax.set_xlabel("Position (1-based)")
        ct = f"  ({cell_type})" if cell_type else ""
        t = f"{ref_id}{ct}" + (f"  [{metric}]" if title_metric else "")
        logo.ax.set_title(t)
        logo.ax.tick_params(axis='both', which='major', labelsize=font_size)
        logo.ax.xaxis.label.set_size(font_size + 1)
        logo.ax.yaxis.label.set_size(font_size + 1)
        logo.ax.title.set_size(font_size + 2)

    elif mode == "ic":
        score_mat, L = build_score_matrix(tile_df, metric=metric, agg=agg, min_pos=min_pos, max_pos=max_pos)
        ic_mat = score_to_ic_matrix(score_mat, temperature=temperature)

        figsize, xticks = _compute_fig_params(L, bp_per_inch, min_width, height, xtick_step)
        _log(f"    figsize={figsize}", quiet)
        fig, ax = plt.subplots(figsize=figsize)
        # IC logos are not centered around zero
        logo = logomaker.Logo(ic_mat, center_values=False, ax=ax)
        # Minimize extra whitespace around the axes
        logo.ax.margins(x=0)
        try:
            # Slightly tighten subplot params without clipping tick labels
            plt.subplots_adjust(left=0.04, right=0.995, top=0.90, bottom=0.18)
        except Exception:
            pass
        logo.style_spines(visible=False)
        logo.style_spines(spines=['left', 'bottom'], visible=True)
        logo.ax.set_xlim([0.5, L + 0.5])
        logo.ax.set_xticks(xticks)
        logo.ax.set_xlabel("Position (1-based)")
        logo.ax.set_ylabel("Information (bits)")
        logo.ax.set_ylim([0, 2.05])
        ct = f"  ({cell_type})" if cell_type else ""
        t = f"{ref_id}{ct}" + (f"  [{metric}]" if title_metric else "")
        logo.ax.set_title(t)
        logo.ax.tick_params(axis='both', which='major', labelsize=font_size)
        logo.ax.xaxis.label.set_size(font_size + 1)
        logo.ax.yaxis.label.set_size(font_size + 1)
        logo.ax.title.set_size(font_size + 2)

    else:
        raise ValueError(f"Unknown mode: {mode}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tight_kw = {} if no_tight else {"bbox_inches": "tight", "pad_inches": pad_inches}
    if fmt in {"png", "jpg", "jpeg"}:
        plt.savefig(out_path.with_suffix(f".{fmt}"), dpi=dpi, **tight_kw)
    else:
        plt.savefig(out_path.with_suffix(f".{fmt}"), **tight_kw)
    plt.close()


def process_file(tsv_path: Path, out_dir: Path, metric: str, agg: str,
                 fdr_col: str | None, fdr_max: float | None,
                 min_pos: int | None, max_pos: int | None,
                 fmt: str, dpi: int, title_metric: bool, cap: float | None,
                 quiet: bool, mode: str, temperature: float,
                 bp_per_inch: float, min_width: float, height: float,
                 xtick_step: int | None, font_size: float,
                 no_tight: bool, tight_pad: float, cell_type: str):
    df = pd.read_csv(tsv_path, sep="\t", dtype=str)
    # Optional FDR filter
    if fdr_col and fdr_max is not None and fdr_col in df.columns:
        fdr = pd.to_numeric(df[fdr_col], errors="coerce")
        before = len(df)
        df = df[fdr <= fdr_max]
        _log(f"  FDR filter {fdr_col}<= {fdr_max}: kept {len(df)} of {before}", quiet)

    # Split by tile
    if "ref_id" not in df.columns:
        raise ValueError("Expected column 'ref_id' in input TSV")

    for ref_id, tile_df in df.groupby("ref_id"):
        safe = (
            ref_id.replace("/", "_")
                  .replace(":", "_")
                  .replace(" ", "_")
        )
        cell_suffix = ""
        if cell_type:
            safe_ct = (cell_type.replace("/", "_")
                               .replace(":", "_")
                               .replace(" ", "_"))
            cell_suffix = f"__{safe_ct}"
        # Use only the tile name in the output filename
        out_path = out_dir / f"{safe}{cell_suffix}"
        try:
            plot_logo_for_tile(tile_df, ref_id, out_path, metric, agg,
                               min_pos, max_pos, fmt, dpi, title_metric, cap, quiet,
                               mode=mode, temperature=temperature,
                               bp_per_inch=bp_per_inch, min_width=min_width,
                               height=height, xtick_step=xtick_step,
                               font_size=font_size, no_tight=no_tight, pad_inches=tight_pad,
                               cell_type=cell_type)
            _log(f"  [OK] {ref_id} -> {out_path.with_suffix('.'+fmt).name}", quiet)
        except Exception as e:
            _log(f"  [SKIP] {ref_id}: {e}", quiet)


def main():
    args = parse_args()
    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(in_dir.glob(args.pattern))
    if not files:
        print(f"No files matched {in_dir}/{args.pattern}")
        sys.exit(1)

    if not _HAS_LOGOMAKER:
        print("ERROR: logomaker is not installed. Install with: pip install logomaker")
        sys.exit(2)

    for tsv in files:
        print(f"Processing {tsv.name} ...")
        # Pass through sizing and font params
        process_file(tsv, out_dir, metric=args.metric, agg=args.agg,
                     fdr_col=(args.fdr_col or None), fdr_max=args.fdr_max,
                     min_pos=args.min_pos, max_pos=args.max_pos,
                     fmt=args.format, dpi=args.dpi, title_metric=args.title_metric,
                     cap=args.cap, quiet=args.quiet,
                     mode=args.mode, temperature=args.temperature,
                     bp_per_inch=args.bp_per_inch, min_width=args.min_width,
                     height=args.height, xtick_step=args.xtick_step,
                     font_size=args.font_size, no_tight=args.no_tight,
                     tight_pad=args.tight_pad, cell_type=args.cell_type)


if __name__ == "__main__":
    main()