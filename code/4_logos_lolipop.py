#!/usr/bin/env python3

"""
logos_lolipop.py

Generates hybrid sequence logo + lollipop plots for saturation mutagenesis (SatMut) data.

This script computes activity values for both reference and alternate alleles at each position
in a given sequence, using metrics such as post_log2Skew or other specified fields. Reference
allele activities are calculated in the same way as alternate alleles, ensuring consistent scaling.
The only difference is in the plotting: the reference allele is drawn as a large letter whose height
represents its true activity, while non-reference alleles are shown as lollipops (stem + circle tip)
at reduced alpha.

Features:
- Hybrid plotting mode only (sequence logo for reference, faint lollipops for alternates)
- Circle tips for lollipops with configurable alpha and stem width (default alpha=0.18, stem=0.6)
- 1-based x-axis tick labels at a specified interval (default 5 bp)
- Supports aggregation across replicates and symmetric y-axis scaling
- Accepts any numerical metric from the input data for plotting

Typical usage:
    python code/logos_lolipop.py \\
        --in-dir results/satmut_stats \\
        --pattern 'Jurkat_activity_satmut_effects_stats.tsv' \\
        --metric post_log2Skew \\
        --ref-metric ref_activity \\
        --agg mean \\
        --y-symmetric \\
        --out-dir results/logos_lolipop \\
        --bp-per-inch 8 --min-width 12 --height 4 --font-size 10 \\
        --cell-type Jurkat

"""

from __future__ import annotations
import argparse
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

# Simple base color map (aligned with logomaker classic scheme)
BASE_COLORS = {
    "A": "#2ca02c",  # green
    "C": "#1f77b4",  # blue
    "G": "#ff7f0e",  # orange
    "T": "#d62728",  # red
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate hybrid logo+lollipop from SatMut stats TSVs")
    p.add_argument("--in-dir", type=str, default="../results/satmut_stats",
                   help="Directory containing *_satmut_effects_stats.tsv inputs")
    p.add_argument("--pattern", type=str, default="*_satmut_effects_stats.tsv",
                   help="Glob pattern for input files inside --in-dir")
    p.add_argument("--metric", type=str, default="post_log2Skew",
                   help="Column for non-reference effects (e.g., post_log2Skew, log2Skew, z)")
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
                   help="Output directory for images")
    p.add_argument("--format", type=str, choices=["png", "pdf", "svg"], default="pdf",
                   help="Image format")
    p.add_argument("--dpi", type=int, default=200, help="DPI for raster formats")
    p.add_argument("--title-metric", action="store_true",
                   help="Append metric name to plot title")
    p.add_argument("--cap", type=float, default=None,
                   help="Cap absolute values of EFFECTS at this threshold before plotting (ignored in IC mode)")
    p.add_argument("--quiet", action="store_true", help="Reduce logging output")
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

    # HYBRID-only knobs
    p.add_argument("--ref-metric", type=str, default="ref_activity",
                   help="Column used for reference letter height (e.g., ref_activity or ref_log2FC)")
    p.add_argument("--lollipop_style", type=str, choices=["letter", "dot"], default="dot",
                   help="Style for alt-base lollipops: text letter at tip or a dot marker")
    p.add_argument("--lollipop_alpha", type=float, default=0.18,
                   help="Alpha transparency for lollipop stems/markers")
    p.add_argument("--lollipop_jitter", type=float, default=0.10,
                   help="Horizontal jitter half-width for spreading alt bases at the same position")
    p.add_argument("--stem_width", type=float, default=0.6,
                   help="Line width for lollipop stems")
    p.add_argument("--y-symmetric", action="store_true",
                   help="If set, y-axis limits are symmetric around 0 based on data range")
    p.add_argument("--alt-activity-metric", type=str, default="mut_activity",
                   help="Column for non-reference absolute activity (e.g., mut_activity)")
    p.add_argument("--lollipop-from", type=str, choices=["effect", "activity"], default="effect",
                   help="Use 'effect' (e.g., post_log2Skew) or 'activity' (e.g., mut_activity) for lollipop heights")
    p.add_argument("--ref-baseline-eps", type=float, default=1e-9,
                   help="If the ref-base effect |y| <= eps (typically 0), draw the ref base letter at y=0 so it's visible")

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


def infer_L(df: pd.DataFrame) -> int:
    L = None
    if "ref_seq" in df.columns and not df["ref_seq"].dropna().empty:
        try:
            L = int(df["ref_seq"].dropna().map(len).max())
        except Exception:
            L = None
    if L is None:
        if "pos" in df.columns:
            L = int(pd.to_numeric(df["pos"], errors="coerce").max())
        else:
            raise ValueError("Cannot infer sequence length: need ref_seq or pos column")
    return L


def get_pos_slice(df: pd.DataFrame, min_pos: int | None, max_pos: int | None) -> pd.DataFrame:
    pos_series = pd.to_numeric(df["pos"], errors="coerce")
    if min_pos is not None:
        df = df[pos_series >= min_pos]
    if max_pos is not None:
        df = df[pos_series <= max_pos]
    return df


def ref_base_series(df: pd.DataFrame, L: int) -> pd.Series:
    """Return a Series indexed by position (1..L) giving the reference base at each position.
    Infer strictly from the mode of ref_allele per position. Ignore ref_seq entirely.
    """
    ref = pd.Series(index=range(1, L + 1), dtype=object)
    if "ref_allele" in df.columns and "pos" in df.columns:
        tmp = (
            df.dropna(subset=["pos", "ref_allele"]) 
              .assign(pos=lambda x: pd.to_numeric(x["pos"], errors="coerce").astype("Int64"))
        )
        tmp = tmp[tmp["pos"].notna()]
        if not tmp.empty:
            mode_ref = (
                tmp.groupby("pos")["ref_allele"]
                   .agg(lambda s: pd.Series(s).mode().iloc[0] if not pd.Series(s).mode().empty else np.nan)
            )
            for p, base in mode_ref.items():
                if pd.notna(base) and base in NUCS and 1 <= int(p) <= L:
                    ref.at[int(p)] = base
    return ref


def build_effect_matrix(df: pd.DataFrame, metric: str, agg: str,
                        min_pos: int | None, max_pos: int | None) -> tuple[pd.DataFrame, int]:
    L = infer_L(df)
    df = get_pos_slice(df, min_pos, max_pos)
    mat = pd.DataFrame(0.0, index=range(1, L + 1), columns=NUCS, dtype=float)
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




def _logo_common_formatting(logo, L, xticks, font_size, title, xlabel, ylabel=None, xlim_start=0.5):
    logo.ax.margins(x=0)
    try:
        plt.subplots_adjust(left=0.04, right=0.995, top=0.90, bottom=0.18)
    except Exception:
        pass
    logo.style_spines(visible=False)
    logo.style_spines(spines=['left', 'bottom'], visible=True)
    logo.ax.set_xlim([xlim_start, L + 0.5])
    logo.ax.set_xticks(xticks)
    logo.ax.set_xlabel(xlabel)
    if ylabel:
        logo.ax.set_ylabel(ylabel)
    logo.ax.set_title(title)
    logo.ax.tick_params(axis='both', which='major', labelsize=font_size)
    logo.ax.xaxis.label.set_size(font_size + 1)
    if ylabel:
        logo.ax.yaxis.label.set_size(font_size + 1)
    logo.ax.title.set_size(font_size + 2)


def plot_logo_for_tile(tile_df: pd.DataFrame, tile_df_ref: pd.DataFrame | None, ref_id: str, out_path: Path, metric: str,
                       agg: str, min_pos: int | None, max_pos: int | None,
                       fmt: str = "png", dpi: int = 200, title_metric: bool = False,
                       cap: float | None = None, quiet: bool = False,
                       bp_per_inch: float = 8.0, min_width: float = 12.0,
                       height: float = 4.0, xtick_step: int | None = None,
                       font_size: float = 10.0, no_tight: bool = False, pad_inches: float = 0.03,
                       cell_type: str = "",
                       # HYBRID extras
                       ref_metric: str = "ref_activity",
                       alt_activity_metric: str = "mut_activity",
                       lollipop_from: str = "effect",
                       lollipop_style: str = "letter",
                       lollipop_alpha: float = 0.6,
                       lollipop_jitter: float = 0.10,
                       stem_width: float = 0.8,
                       y_symmetric: bool = False,
                       ref_baseline_eps: float = 1e-9,
                       lollipop_blur: bool = False, blur_scale: float = 2.2, blur_alpha: float = 0.25):
    if not _HAS_LOGOMAKER:
        raise RuntimeError(
            "logomaker is required for logos. Install via: pip install logomaker"
        )
    # Hybrid mode only
    # Build the standard effect matrix exactly as in effect mode
    eff_mat, L = build_effect_matrix(tile_df, metric=metric, agg=agg, min_pos=min_pos, max_pos=max_pos)
    # Optional capping applies to effects
    if cap is not None and cap > 0:
        eff_mat = eff_mat.clip(lower=-abs(cap), upper=abs(cap))

    # Mirror regular logo behavior: center per position (same as logomaker center_values=True)
    row_means = eff_mat.mean(axis=1)
    centered_mat = eff_mat.subtract(row_means, axis=0)

    # Infer reference bases strictly from ref_allele (using unfiltered per-tile frame if provided)
    base_source_df = tile_df_ref if tile_df_ref is not None else tile_df
    ref_bases = ref_base_series(base_source_df, L)

    # Reference-only matrix for letters: take the CENTERED effect value for the REF base
    ref_mat = pd.DataFrame(0.0, index=centered_mat.index, columns=centered_mat.columns)
    for p in ref_mat.index:
        b = ref_bases.get(p)
        if b in NUCS:
            ref_mat.at[p, b] = centered_mat.at[p, b]

    # Lollipops come from the SAME CENTERED matrix, but for the non-ref bases
    rows = []
    for p in centered_mat.index:
        b_ref = ref_bases.get(p)
        for b in NUCS:
            if b != b_ref:
                val = centered_mat.at[p, b]
                if pd.notna(val) and val != 0:
                    rows.append({"pos": int(p), "alt": b, "value": float(val)})
    alt_df = pd.DataFrame(rows)

    figsize, xticks = _compute_fig_params(L, bp_per_inch, min_width, height, xtick_step)
    # Use 1-based tick labels at multiples of 5: 1, 6, 11, ...
    xticks = np.arange(1, L + 1, 5)
    _log(f"    figsize={figsize}", quiet)
    fig, ax = plt.subplots(figsize=figsize)

    # Plot the REF base as a letter using the SAME effect value used everywhere else (already centered)
    logo = logomaker.Logo(ref_mat, center_values=False, ax=ax, color_scheme=BASE_COLORS)
    # Draw baseline
    ax.axhline(0, linewidth=0.8)
    # Ensure ref letter is visible even when its effect is exactly zero (logomaker won't draw zero-height letters)
    eps = max(0.0, ref_baseline_eps)
    for p in ref_mat.index:
        b = ref_bases.get(p)
        if b in NUCS:
            y = centered_mat.at[p, b]
            if not pd.notna(y):
                y = 0.0
            if abs(y) <= eps:
                # Draw the base letter at the baseline without implying nonzero effect
                ax.text(p, 0, b, ha="center", va="bottom", fontsize=font_size+1,
                        color=BASE_COLORS.get(b, "k"))

    # Prepare offsets for up to three alts per position
    y_vals = []
    if not alt_df.empty:
        for pos, sub in alt_df.groupby("pos"):
            ref_b = ref_bases.get(pos)
            sub = sub[sub["alt"] != ref_b]
            if sub.empty:
                continue
            alts = [b for b in NUCS if b != ref_b and b in set(sub["alt"]) ]
            n = len(alts)
            if n == 1:
                offsets = {alts[0]: 0.0}
            elif n == 2:
                offsets = {alts[0]: -lollipop_jitter/2.0, alts[1]: lollipop_jitter/2.0}
            else:  # n >= 3
                offsets = {alts[0]: -lollipop_jitter, alts[1]: 0.0, alts[2]: lollipop_jitter}
            for _, row in sub.iterrows():
                alt = row["alt"]
                y = float(row["value"]) if pd.notna(row["value"]) else np.nan
                if pd.isna(y) or alt not in offsets:
                    continue
                x = pos + offsets[alt]
                y_vals.append(y)
                col = BASE_COLORS.get(alt, "k")
                # Stem
                ax.plot([x, x], [0, y], linewidth=stem_width, alpha=lollipop_alpha, color=col)
                # Tip
                if lollipop_style == "dot":
                    ax.plot([x], [y], marker="o", alpha=lollipop_alpha,
                            markersize=max(3, font_size/2), color=col, markeredgewidth=0)
                else:  # letter on tip
                    va = "bottom" if y >= 0 else "top"
                    ax.text(x, y, alt, ha="center", va=va, fontsize=font_size+1,
                            alpha=lollipop_alpha, color=col)
    # Axis formatting
    ct = f"  ({cell_type})" if cell_type else ""
    t = f"{ref_id}{ct}" + (f"  [{metric}]" if title_metric else "")
    # Use xlim_start=0.5 so that first base is centered under tick label 1
    _logo_common_formatting(logo, L, xticks, font_size, t, "Position", xlim_start=0.5)
    # Set y-limits
    # Gather reference letter heights too
    ref_vals = ref_mat.values.flatten()
    ref_vals = ref_vals[np.isfinite(ref_vals)]
    if y_vals or ref_vals.size:
        y_min_candidates = [0]
        y_max_candidates = [0]
        if y_vals:
            y_min_candidates.append(min(y_vals))
            y_max_candidates.append(max(y_vals))
        if ref_vals.size:
            y_min_candidates.append(float(ref_vals.min()))
            y_max_candidates.append(float(ref_vals.max()))
        ymin = min(y_min_candidates)
        ymax = max(y_max_candidates)
        if y_symmetric:
            M = max(abs(ymin), abs(ymax))
            ax.set_ylim(-M * 1.05, M * 1.05)
        else:
            pad = 0.05 * (ymax - ymin if ymax > ymin else (abs(ymax) + 1.0))
            ax.set_ylim(ymin - pad, ymax + pad)

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
                 quiet: bool, bp_per_inch: float, min_width: float, height: float,
                 xtick_step: int | None, font_size: float,
                 no_tight: bool, tight_pad: float, cell_type: str,
                 ref_metric: str, alt_activity_metric: str, lollipop_style: str, lollipop_alpha: float,
                 lollipop_jitter: float, stem_width: float, y_symmetric: bool, lollipop_from: str,
                 ref_baseline_eps: float, lollipop_blur: bool = False, blur_scale: float = 2.2, blur_alpha: float = 0.25):
    df = pd.read_csv(tsv_path, sep="\t", dtype=str)
    df_raw = df.copy()
    # Optional FDR filter
    if fdr_col and fdr_max is not None and fdr_col in df.columns:
        fdr = pd.to_numeric(df[fdr_col], errors="coerce")
        before = len(df)
        df = df[fdr <= fdr_max]
        _log(f"  FDR filter {fdr_col}<= {fdr_max}: kept {len(df)} of {before}", quiet)
    if "ref_id" not in df.columns:
        raise ValueError("Expected column 'ref_id' in input TSV")

    for ref_id, tile_df in df.groupby("ref_id"):
        tile_df_ref = df_raw[df_raw["ref_id"] == ref_id]
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
        out_path = out_dir / f"{safe}{cell_suffix}"
        try:
            plot_logo_for_tile(
                tile_df, tile_df_ref, ref_id, out_path,
                metric=metric, agg=agg, min_pos=min_pos, max_pos=max_pos,
                fmt=fmt, dpi=dpi, title_metric=title_metric, cap=cap, quiet=quiet,
                bp_per_inch=bp_per_inch, min_width=min_width, height=height,
                xtick_step=xtick_step, font_size=font_size,
                no_tight=no_tight, pad_inches=tight_pad, cell_type=cell_type,
                ref_metric=ref_metric, alt_activity_metric=alt_activity_metric, lollipop_from=lollipop_from,
                lollipop_style=lollipop_style,
                lollipop_alpha=lollipop_alpha, lollipop_jitter=lollipop_jitter,
                stem_width=stem_width, y_symmetric=y_symmetric,
                ref_baseline_eps=ref_baseline_eps
            )
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
        process_file(
            tsv, out_dir,
            metric=args.metric, agg=args.agg,
            fdr_col=(args.fdr_col or None), fdr_max=args.fdr_max,
            min_pos=args.min_pos, max_pos=args.max_pos,
            fmt=args.format, dpi=args.dpi, title_metric=args.title_metric,
            cap=args.cap, quiet=args.quiet,
            bp_per_inch=args.bp_per_inch, min_width=args.min_width, height=args.height,
            xtick_step=args.xtick_step, font_size=args.font_size,
            no_tight=args.no_tight, tight_pad=args.tight_pad, cell_type=args.cell_type,
            ref_metric=args.ref_metric, alt_activity_metric=args.alt_activity_metric,
            lollipop_style=args.lollipop_style,
            lollipop_alpha=args.lollipop_alpha, lollipop_jitter=args.lollipop_jitter,
            stem_width=args.stem_width, y_symmetric=args.y_symmetric,
            lollipop_from=args.lollipop_from,
            ref_baseline_eps=args.ref_baseline_eps
        )


if __name__ == "__main__":
    main()