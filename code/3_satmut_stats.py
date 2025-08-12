#!/usr/bin/env python3

import pandas as pd
import numpy as np
from pathlib import Path
from math import sqrt
from scipy.stats import norm
import re

BASE = Path(__file__).resolve().parent
COMPLETE_DIR = BASE.parent / "data" / "complete"     # where your *_with_mut.tsv live
IN_DIR = BASE.parent / "results" / "delta_activity"  # where *_satmut_effects.tsv live
OUT_DIR = BASE.parent / "results" / "satmut_stats"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CT_PATTERN = re.compile(r"run_([^_]+)_activity", re.IGNORECASE)

def infer_cell_type(stem: str) -> str:
    m = CT_PATTERN.search(stem)
    if m:
        return m.group(1)
    # try a few common tokens in the stem
    for ct in ("HEK293", "JurkatStim", "Jurkat", "K562", "HEPG2", "HepG2", "A549", "HCT116", "SK-N-SH"):
        if ct.lower() in stem.lower():
            return ct
    return "unknown"

def bh_fdr(p):
    p = np.asarray(p, dtype=float)
    n = p.size
    order = np.argsort(p)
    ranks = np.empty(n, dtype=int)
    ranks[order] = np.arange(1, n+1)
    f = p * n / ranks
    f_sorted = np.minimum.accumulate(f[order[::-1]])[::-1]
    out = np.empty_like(f)
    out[order] = np.minimum(f_sorted, 1.0)
    return out

# Storey q-value with lambda grid and monotone adjustment (approx R::qvalue)
def storey_qvalue(p, lambdas=None):
    p = np.asarray(p, dtype=float)
    m = float(len(p))
    if m == 0:
        return p
    if lambdas is None:
        lambdas = np.linspace(0.05, 0.95, 19)  # 0.05, 0.10, ..., 0.95
    # Estimate pi0 across lambdas
    pi0_vals = []
    for lam in lambdas:
        if lam >= 1.0:
            continue
        pi0_lam = np.mean(p >= lam) / (1.0 - lam)
        pi0_vals.append(min(max(pi0_lam, 0.0), 1.0))
    pi0_vals = np.array(pi0_vals) if len(pi0_vals) else np.array([1.0])
    # Smooth pi0 by simple cubic fit across lambdas (fallback to mean if ill-conditioned)
    try:
        coeffs = np.polyfit(lambdas[: len(pi0_vals)], pi0_vals, deg=3)
        lam_eval = 0.95
        pi0 = np.polyval(coeffs, lam_eval)
        pi0 = float(np.clip(pi0, 0.0, 1.0))
    except Exception:
        pi0 = float(np.clip(np.median(pi0_vals), 0.0, 1.0))
    # Benjamini-Hochberg scaled by pi0
    order = np.argsort(p)
    ranks = np.empty(len(p), dtype=int)
    ranks[order] = np.arange(1, len(p) + 1)
    q = pi0 * p * m / ranks
    # Monotone correction
    q_sorted = np.minimum.accumulate(q[order[::-1]])[::-1]
    out = np.empty_like(q)
    out[order] = np.minimum(q_sorted, 1.0)
    return out

for effects_file in sorted(IN_DIR.glob("*_satmut_effects.tsv")):
    print(f"Processing {effects_file.name} ...")
    eff = pd.read_csv(effects_file, sep="\t", dtype=str)
    # Match back to the original _with_mut.tsv to get lfcSE for ref and mutant
    # Infer the corresponding with_mut file name
    stem = effects_file.name.replace("_satmut_effects.tsv", "")
    cell_type = infer_cell_type(stem)
    eff["cell_type"] = cell_type
    # Your inputs came from data/complete/<stem>.tsv
    src = COMPLETE_DIR / f"{stem}.tsv"
    if not src.exists():
        # try without any extra suffix shenanigans
        candidates = list(COMPLETE_DIR.glob(f"*{stem.split('_with_mut')[0]}*_with_mut.tsv"))
        if candidates:
            src = candidates[0]
    if not src.exists():
        print(f"  [WARN] Could not locate source with_mut TSV for {stem}; skipping stats.")
        eff.to_csv(OUT_DIR / f"{stem}_satmut_effects_stats.tsv", sep="\t", index=False)
        continue

    src_df = pd.read_csv(src, sep="\t", dtype=str)
    src_df["cell_type"] = cell_type
    # numeric
    for col in ("log2FoldChange", "lfcSE"):
        if col in src_df.columns:
            src_df[col] = pd.to_numeric(src_df[col], errors="coerce")

    # Build helpers
    src_df["__base"] = src_df["ID"].str.split(":m", n=1).str[0]
    # Grab ref rows (m0) per base id for SE & ref activity
    ref_df = src_df[src_df["ID"].str.endswith(":m0")][["__base","cell_type","log2FoldChange","lfcSE"]].rename(
        columns={"log2FoldChange":"ref_log2FC","lfcSE":"ref_lfcSE"}
    )

    # We need SE for the *mutant* row too. We don't have mutant ID in effects, so reconstruct it from (ref_id,pos,ref,alt)
    # The exact mutant ID string in your data uses the trailing :m{Ref}{Pos}{Alt}
    def make_mut_id(ref_id, pos, ref, alt):
        return f"{ref_id}:m{ref}{pos}{alt}"

    eff["mut_ID"] = eff.apply(lambda r: make_mut_id(r["ref_id"], r["pos"], r["ref_allele"], r["alt_allele"]), axis=1)

    muts = src_df[["ID","log2FoldChange","lfcSE","__base","cell_type"]].rename(
        columns={"log2FoldChange":"mut_log2FC","lfcSE":"mut_lfcSE","ID":"mut_ID"}
    )

    # Join to get ref_lfcSE and mut_lfcSE and mut_log2FC
    eff2 = eff.merge(muts[["mut_ID","mut_lfcSE","mut_log2FC","__base","cell_type"]], on=["mut_ID"], how="left")
    # ensure eff2 has cell_type (from eff or muts); align ref by (__base, cell_type)
    if "cell_type_x" in eff2.columns and "cell_type_y" in eff2.columns:
        eff2["cell_type"] = eff2["cell_type_x"].fillna(eff2["cell_type_y"])
        eff2 = eff2.drop(columns=["cell_type_x","cell_type_y"])
    eff2 = eff2.merge(ref_df, on=["__base","cell_type"], how="left")

    # === Empirical Bayes priors (match R two-step: IQR and 5–95%) ===
    def calc_priors(g):
        vals = pd.to_numeric(g["log2FoldChange"], errors="coerce").dropna()
        out = {}
        n = len(vals)
        out["n_obs"] = int(n)
        if n >= 3:
            # 25–75% for prior_mu/prior_sigmasq
            q1, q3 = np.quantile(vals, [0.25, 0.75])
            trimmed_iqr = vals[(vals > q1) & (vals < q3)]
            if len(trimmed_iqr) >= 2:
                out["prior_mu"] = float(trimmed_iqr.mean())
                out["prior_sigmasq"] = float(trimmed_iqr.var(ddof=1))
            else:
                out["prior_mu"] = float(vals.mean()) if n > 0 else np.nan
                out["prior_sigmasq"] = float(vals.var(ddof=1)) if n > 1 else np.nan
            # 5–95% for prior_mu_full/prior_sigmasq_full
            ql, qh = np.quantile(vals, [0.05, 0.95])
            trimmed_595 = vals[(vals > ql) & (vals < qh)]
            if len(trimmed_595) >= 2:
                out["prior_mu_full"] = float(trimmed_595.mean())
                out["prior_sigmasq_full"] = float(trimmed_595.var(ddof=1))
            else:
                out["prior_mu_full"] = float(vals.mean()) if n > 0 else np.nan
                out["prior_sigmasq_full"] = float(vals.var(ddof=1)) if n > 1 else np.nan
        else:
            out["prior_mu"] = float(vals.mean()) if n > 0 else np.nan
            out["prior_sigmasq"] = float(vals.var(ddof=1)) if n > 1 else np.nan
            out["prior_mu_full"] = out["prior_mu"]
            out["prior_sigmasq_full"] = out["prior_sigmasq"]
        return pd.Series(out)

    priors = (src_df.groupby(["__base","cell_type"], as_index=False)
                    .apply(calc_priors)
                    .reset_index()
                    .rename(columns={"__base":"__base_dup"}))
    if "__base_dup" in priors.columns and "__base" not in priors.columns:
        priors = priors.rename(columns={"__base_dup":"__base"})
    eff2 = eff2.merge(priors, on=["__base","cell_type"], how="left")

    # Posterior means (IQR prior and 5–95 prior), mirror R
    REPS = 5.0
    mut_lfcSE = pd.to_numeric(eff2["mut_lfcSE"], errors="coerce")
    obs_mu = pd.to_numeric(eff2["mut_log2FC"], errors="coerce")
    obs_sigmasq = np.square(mut_lfcSE) * REPS

    prior_mu = pd.to_numeric(eff2["prior_mu"], errors="coerce")
    prior_sigmasq = pd.to_numeric(eff2["prior_sigmasq"], errors="coerce")
    prior_mu_full = pd.to_numeric(eff2["prior_mu_full"], errors="coerce")
    prior_sigmasq_full = pd.to_numeric(eff2["prior_sigmasq_full"], errors="coerce")

    with np.errstate(divide='ignore', invalid='ignore'):
        post_sigmasq = 1.0 / (1.0 / prior_sigmasq + REPS / obs_sigmasq)
        post_mu = post_sigmasq * (prior_mu / prior_sigmasq + (REPS * obs_mu) / obs_sigmasq)
        post_sigmasq_full = 1.0 / (1.0 / prior_sigmasq_full + REPS / obs_sigmasq)
        post_mu_full = post_sigmasq_full * (prior_mu_full / prior_sigmasq_full + (REPS * obs_mu) / obs_sigmasq)

    eff2["post_mu"] = post_mu
    eff2["post_log2FC"] = post_mu_full

    # === Compute baseline & skew per tile to match R ===
    ref_baseline = pd.to_numeric(eff2["prior_mu"], errors="coerce")  # baseline per group
    n_obs = pd.to_numeric(eff2["n_obs"], errors="coerce")
    prior_sigmasq = pd.to_numeric(eff2["prior_sigmasq"], errors="coerce")
    log2FC_SE_baseline = np.sqrt(prior_sigmasq / n_obs)

    # Raw skew and SE
    mut_lfcSE = pd.to_numeric(eff2["mut_lfcSE"], errors="coerce")
    mut_log2FC = pd.to_numeric(eff2["mut_log2FC"], errors="coerce")
    eff2["log2Skew"] = mut_log2FC - ref_baseline
    eff2["log2Skew_SE"] = np.sqrt(np.square(mut_lfcSE) + np.square(log2FC_SE_baseline))

    # Posterior skew using post_mu_full
    eff2["post_log2Skew"] = post_mu_full - ref_baseline
    eff2["post_log2FC"] = post_mu_full

    # P-values & q-values for skew (match R)
    z = eff2["log2Skew"] / eff2["log2Skew_SE"]
    eff2["log2Skew_z"] = z
    eff2["log2Skew_pval"] = 2 * (1 - norm.cdf(np.abs(z)))
    eff2["log2Skew_fdr"] = np.nan
    for (b, ct), sub_idx in eff2.groupby(["__base","cell_type"]).groups.items():
        idx = list(sub_idx)
        p = eff2.loc[idx, "log2Skew_pval"].values
        good = ~np.isnan(p)
        if good.any():
            q = storey_qvalue(p[good])
            eff2.loc[np.array(idx)[good], "log2Skew_fdr"] = q

    out = OUT_DIR / f"{stem}__{cell_type}_satmut_effects_stats.tsv"
    eff2.to_csv(out, sep="\t", index=False)
    print(f"  -> {out}")