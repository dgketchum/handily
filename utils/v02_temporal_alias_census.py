"""WP3 rung-0 — period-of-record temporal-alias census.

Answers the gate question for `notes/plans/HANDILY_V2_FIELD_MODEL_PLAN.md` §8:
how much of the frozen background's residual error is period-of-record (POR)
aliasing, and is common-epoch harmonization feasible/worthwhile? This is
observational evidence only — per §8.3 a better in-sample hydrograph
reconstruction is insufficient; the full common-epoch program is justified only
if aliasing is a real, correctable share of the residual here.

Background (§2.2, §3.5): `mean_dtw` summarizes different periods of record, so
pumping decline/recovery is aliased into spatial structure. A well whose POR
midpoint sits far from a common epoch, in an aquifer with a real trend, carries
an implied epoch-drift; if that drift explains and corrects the background
innovation, the alias is real and removable.

Definitions (all lengths in metres, WTE = water-table elevation, head space):
    innovation_m        = obs_wte_m - gnn_wte_hat_m       (frozen w25_prod OOF;
                          + = observed head above background)
    por_midpoint_year   = decimal-year midpoint of [por_start, por_end]
    trend_slope         = GWX Theil-Sen-style slope, m/yr of DTW; EMPIRICALLY
                          verified positive = deepening (declining water table)
    implied_dtw_drift_m = trend_slope * (2020.0 - por_midpoint_year)  (DTW change
                          from POR midpoint to the 2020 common epoch; + = deeper)
    implied_wte_drift_m = -implied_dtw_drift_m            (WTE = surface - DTW, so
                          a deepening DTW is a falling head)
    usable_trend        = isfinite(trend_slope) & (trend_pvalue < 0.05)
                          & (trend_span_years >= 5)
    harmonized_beta*    = innovation_m - beta * implied_wte_drift_m
                          beta = 1 (physical) and beta = OLS-fit leave-one-fold-out

Join surprise (reported in the JSON): the contract `canonical_id` is `gwx_` +
8 hex chars, but the GWX product `canonical_id` is `gwx_` + 16 hex chars. A
direct string join yields zero matches; the contract id is an 8-hex prefix of
the GWX id, so the join is on that prefix. GWX rows are also non-unique per id;
collisions are deduplicated by keeping max trend_n_obs and counted.

POR surprise: `por_start`, `por_end`, `obs_count` are entirely null in
wells_panels.parquet; the temporal support is taken from the GWX product.

Outputs (--out-dir, default /data/ssd2/handily/conus/wte_gnn/v02/wp3):
    temporal_alias_report.json   full metric panel + "definitions" block
    temporal_alias_summary.md     plain-language census + verdict paragraph
    alias_per_well.parquet        per-well POR/trend/drift/innovation/harmonized

Usage:
    uv run python utils/v02_temporal_alias_census.py
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

log = logging.getLogger("v02_temporal_alias_census")

WTE_GNN = "/data/ssd2/handily/conus/wte_gnn"
CONTRACT_DIR = f"{WTE_GNN}/v02/contract"
OUT_DIR = f"{WTE_GNN}/v02/wp3"
GWX_WELLS = "/data/ssd2/gwx/products/current/wells.geoparquet"

COMMON_EPOCH = 2020.0
DEPTH_BANDS = [(0.0, 2.0), (2.0, 5.0), (5.0, 10.0), (10.0, 30.0), (30.0, np.inf)]
DRIFT_THRESHOLDS_M = (2.0, 5.0)
HUC2_MIN_WELLS = 500
NS_PER_YEAR = 365.2425 * 86400.0 * 1e9

GWX_TREND_COLS = [
    "canonical_id",
    "trend_slope",
    "trend_slope_lo",
    "trend_slope_hi",
    "trend_pvalue",
    "trend_n_obs",
    "trend_span_years",
    "trend_direction",
    "state",
    "well_class",
    "confinement_class",
    "por_start",
    "por_end",
    "obs_count",
]


# --------------------------------------------------------------------------- #
# small numeric helpers (imported by the synthetic test)
# --------------------------------------------------------------------------- #
def mad(x: np.ndarray) -> float:
    """Median absolute value (m). Robust central error of a residual array."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(np.median(np.abs(x))) if len(x) else float("nan")


def rmse(x: np.ndarray) -> float:
    """Root-mean-square value (m). Surfaces the catastrophic-miss tail."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(np.sqrt(np.mean(x**2))) if len(x) else float("nan")


def ols_slope(x: np.ndarray, y: np.ndarray) -> float:
    """OLS slope of y on x with intercept: cov(x, y) / var(x) (dimensionless
    when x, y share units). Returns nan for < 3 finite pairs or zero variance."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 3:
        return float("nan")
    vx = np.var(x)
    if vx <= 0:
        return float("nan")
    return float(np.cov(x, y, bias=True)[0, 1] / vx)


def alias_innovation_link(innovation: np.ndarray, drift: np.ndarray) -> dict:
    """Correlation/regression of innovation_m on implied_wte_drift_m.

    Returns Pearson r, Spearman rho (both dimensionless in [-1, 1]), OLS slope
    (dimensionless), and ols_r2 = Pearson r**2 = variance fraction of the
    innovation linearly explained by the implied drift."""
    innovation = np.asarray(innovation, float)
    drift = np.asarray(drift, float)
    m = np.isfinite(innovation) & np.isfinite(drift)
    innovation, drift = innovation[m], drift[m]
    n = int(len(innovation))
    if n < 3 or np.var(drift) <= 0:
        return {
            "n": n,
            "pearson_r": None,
            "spearman_r": None,
            "ols_slope": None,
            "ols_r2": None,
        }
    pr = float(pearsonr(drift, innovation)[0])
    sr = float(spearmanr(drift, innovation)[0])
    return {
        "n": n,
        "pearson_r": pr,
        "spearman_r": sr,
        "ols_slope": ols_slope(drift, innovation),
        "ols_r2": pr**2,
    }


def bootstrap_ols_slope_ci(
    x: np.ndarray, y: np.ndarray, n_boot: int = 1000, seed: int = 0
) -> tuple[float, float]:
    """Percentile 95% CI of the OLS slope of y on x by pair resampling."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    n = len(x)
    if n < 3:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    slopes = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        slopes[b] = ols_slope(x[idx], y[idx])
    lo, hi = np.nanpercentile(slopes, [2.5, 97.5])
    return (float(lo), float(hi))


def fit_beta_loo(
    innovation: np.ndarray, drift: np.ndarray, fold: np.ndarray
) -> np.ndarray:
    """Per-well leave-one-fold-out OLS slope of innovation on implied drift.

    For each well the slope is fit on all *other* folds, so the correction beta
    applied to a well never saw that well's fold (honest, no in-fold fit).
    Returns beta per input well (nan where the well's drift/innovation is not
    finite or the out-of-fold fit is undefined)."""
    innovation = np.asarray(innovation, float)
    drift = np.asarray(drift, float)
    fold = np.asarray(fold)
    beta = np.full(len(innovation), np.nan)
    finite = np.isfinite(innovation) & np.isfinite(drift)
    for f in np.unique(fold):
        train = finite & (fold != f)
        b = ols_slope(drift[train], innovation[train])
        sel = (fold == f) & finite
        beta[sel] = b
    return beta


def correction_metrics(raw: np.ndarray, harmonized: np.ndarray) -> dict:
    """MAD/RMSE of raw vs harmonized innovation (m) on the common finite subset,
    plus reduction fractions = 1 - harmonized/raw (dimensionless; + = the
    correction shrinks the error, so temporal aliasing is real and removable)."""
    raw = np.asarray(raw, float)
    harmonized = np.asarray(harmonized, float)
    m = np.isfinite(raw) & np.isfinite(harmonized)
    raw, harmonized = raw[m], harmonized[m]
    mad_raw, mad_harm = mad(raw), mad(harmonized)
    rmse_raw, rmse_harm = rmse(raw), rmse(harmonized)
    return {
        "n": int(len(raw)),
        "mad_raw_m": mad_raw,
        "mad_harmonized_m": mad_harm,
        "mad_reduction_frac": (1.0 - mad_harm / mad_raw) if mad_raw > 0 else 0.0,
        "rmse_raw_m": rmse_raw,
        "rmse_harmonized_m": rmse_harm,
        "rmse_reduction_frac": (1.0 - rmse_harm / rmse_raw) if rmse_raw > 0 else 0.0,
    }


def _dist(x: np.ndarray) -> dict:
    """Distribution summary of a 1-D array: n, median, p10/p25/p75/p90, IQR, mean."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {
            "n": 0,
            "median": None,
            "p10": None,
            "p25": None,
            "p75": None,
            "p90": None,
            "iqr": None,
            "mean": None,
        }
    q = np.percentile(x, [10, 25, 50, 75, 90])
    return {
        "n": int(len(x)),
        "median": round(float(q[2]), 4),
        "p10": round(float(q[0]), 4),
        "p25": round(float(q[1]), 4),
        "p75": round(float(q[3]), 4),
        "p90": round(float(q[4]), 4),
        "iqr": round(float(q[3] - q[1]), 4),
        "mean": round(float(x.mean()), 4),
    }


def _band_label(lo: float, hi: float, unit: str = "m") -> str:
    return f"{lo:g}-{hi:g}{unit}" if np.isfinite(hi) else f"{lo:g}+{unit}"


def _decimal_year(ts: pd.Series) -> np.ndarray:
    """Convert a (possibly tz-aware) timestamp series to decimal years. NaT -> nan.
    Linear ns-since-epoch over a mean year length; sub-day precision, ample for
    multi-decade POR midpoints."""
    dt = pd.to_datetime(ts, utc=True, errors="coerce")
    isna = dt.isna().to_numpy()
    ns = dt.astype("int64").to_numpy().astype("float64")
    yr = 1970.0 + ns / NS_PER_YEAR
    yr[isna] = np.nan
    return yr


# --------------------------------------------------------------------------- #
# data assembly
# --------------------------------------------------------------------------- #
def _hex8(cid: pd.Series) -> pd.Series:
    return cid.str.replace("gwx_", "", regex=False).str[:8]


def load_and_join(contract_dir: str, gwx_path: str) -> tuple[pd.DataFrame, dict]:
    """Assemble the per-well census frame and the join/coverage provenance.

    Population: real monitoring wells (is_water_pseudo == False) with the
    sacrificial regional holdout dropped (hard rule). POR/trend/obs_count are
    joined from the GWX product on the 8-hex-prefix of canonical_id (the
    contract truncates GWX's 16-hex id), deduplicated by max trend_n_obs."""
    contract = Path(contract_dir)
    panels = pd.read_parquet(
        contract / "wells_panels.parquet",
        columns=[
            "canonical_id",
            "huc2",
            "mean_dtw",
            "regional_holdout",
            "mech_deep_far_arid",
            "dist_train_km",
            "is_water_pseudo",
            "cv_fold",
            "confinement_class",
        ],
    )
    oof = pd.read_parquet(
        contract / "frozen_baseline" / "gnn_oof_predictions.parquet",
        columns=[
            "canonical_id",
            "obs_wte_m",
            "gnn_wte_hat_m",
            "obs_dtw_m",
            "is_water_pseudo",
        ],
    )
    n_all = len(panels)
    n_pseudo = int(panels["is_water_pseudo"].astype(bool).sum())
    n_sacrificial = int((panels["regional_holdout"] == "sacrificial").sum())

    # population: real monitoring wells, sacrificial dropped everywhere
    real = panels[~panels["is_water_pseudo"].astype(bool)].copy()
    real = real[real["regional_holdout"] != "sacrificial"].reset_index(drop=True)
    real = real.merge(
        oof.drop(columns=["is_water_pseudo"]),
        on="canonical_id",
        how="left",
        validate="one_to_one",
    )
    real["innovation_m"] = real["obs_wte_m"] - real["gnn_wte_hat_m"]
    real["hex8"] = _hex8(real["canonical_id"])

    # GWX product: prefix-join provenance
    gwx = pd.read_parquet(gwx_path, columns=GWX_TREND_COLS)
    gwx["hex8"] = _hex8(gwx["canonical_id"])
    panel_hex = set(real["hex8"])
    gm = gwx[gwx["hex8"].isin(panel_hex)].copy()

    rows_per_prefix = gm.groupby("hex8").size()
    distinct16_per_prefix = gm.groupby("hex8")["canonical_id"].nunique()
    join_prov = {
        "contract_id_format": "gwx_ + 8 hex chars",
        "gwx_id_format": "gwx_ + 16 hex chars",
        "join": "8-hex prefix of canonical_id (direct string join = 0 matches)",
        "n_gwx_rows_total": int(len(gwx)),
        "n_gwx_rows_matching_population": int(len(gm)),
        "n_prefixes_with_multiple_gwx_rows": int((rows_per_prefix > 1).sum()),
        "max_gwx_rows_per_prefix": int(rows_per_prefix.max())
        if len(rows_per_prefix)
        else 0,
        "n_prefixes_matching_multiple_distinct_16hex_ids": int(
            (distinct16_per_prefix > 1).sum()
        ),
        "dedup_rule": "keep GWX row with max trend_n_obs per 8-hex prefix",
    }

    gd = (
        gm.sort_values("trend_n_obs", ascending=False, na_position="last")
        .drop_duplicates("hex8", keep="first")
        # confinement_class is carried from the (complete) contract panels;
        # drop the GWX copy so the merge leaves a single unsuffixed column
        .drop(columns=["canonical_id", "confinement_class"])
    )
    df = real.merge(gd, on="hex8", how="left", validate="one_to_one")

    # decimal-year POR + derived drifts
    df["por_start_year"] = _decimal_year(df["por_start"])
    df["por_end_year"] = _decimal_year(df["por_end"])
    df["por_midpoint_year"] = 0.5 * (df["por_start_year"] + df["por_end_year"])
    df["por_span_years"] = df["por_end_year"] - df["por_start_year"]

    df["usable_trend"] = (
        np.isfinite(df["trend_slope"])
        & (df["trend_pvalue"] < 0.05)
        & (df["trend_span_years"] >= 5.0)
    ).fillna(False)

    df["implied_dtw_drift_m"] = df["trend_slope"] * (
        COMMON_EPOCH - df["por_midpoint_year"]
    )
    df["implied_wte_drift_m"] = -df["implied_dtw_drift_m"]
    # drift only where the trend is usable and the POR midpoint is known
    ok = df["usable_trend"] & np.isfinite(df["por_midpoint_year"])
    df.loc[~ok, ["implied_dtw_drift_m", "implied_wte_drift_m"]] = np.nan

    df["depth_band"] = pd.cut(
        df["mean_dtw"],
        bins=[b for b, _ in DEPTH_BANDS] + [np.inf],
        labels=[_band_label(lo, hi) for lo, hi in DEPTH_BANDS],
        right=False,
    )

    coverage = {
        "n_wells_total_contract": n_all,
        "n_water_pseudo": n_pseudo,
        "n_sacrificial": n_sacrificial,
        "n_monitoring_wells_pre_sacrificial": n_all - n_pseudo,
        "n_population_real_nonsacrificial": int(len(df)),
        "n_matched_to_gwx": int(df["por_start"].notna().sum()),
        "frac_matched_to_gwx": round(float(df["por_start"].notna().mean()), 4),
        "n_finite_innovation": int(np.isfinite(df["innovation_m"]).sum()),
        "n_finite_trend_slope": int(np.isfinite(df["trend_slope"]).sum()),
        "n_usable_trend": int(df["usable_trend"].sum()),
        "frac_usable_trend": round(float(df["usable_trend"].mean()), 4),
    }
    join_prov.update(coverage)
    return df, join_prov


# --------------------------------------------------------------------------- #
# analyses
# --------------------------------------------------------------------------- #
def verify_sign_convention(df: pd.DataFrame) -> dict:
    """Empirically confirm the trend_slope sign against trend_direction."""
    sub = df[np.isfinite(df["trend_slope"]) & df["trend_direction"].notna()]
    out = {"by_direction": {}}
    for d in sorted(sub["trend_direction"].dropna().unique()):
        s = sub.loc[sub["trend_direction"] == d, "trend_slope"]
        out["by_direction"][str(d)] = {
            "n": int(len(s)),
            "slope_median": round(float(s.median()), 5),
            "slope_mean": round(float(s.mean()), 5),
            "frac_positive": round(float((s > 0).mean()), 4),
        }
    dec = out["by_direction"].get("declining", {})
    ris = out["by_direction"].get("rising", {})
    deepening_pos = (
        dec.get("frac_positive", 0) > 0.9 and ris.get("frac_positive", 1) < 0.1
    )
    out["inferred_convention"] = (
        "trend_slope is m/yr of DTW; POSITIVE = deepening (declining water "
        "table). Verified: 'declining' -> slope>0, 'rising' -> slope<0."
        if deepening_pos
        else "AMBIGUOUS — inspect by_direction; downstream sign assumption may be wrong."
    )
    out["convention_verified"] = bool(deepening_pos)
    return out


def temporal_support_census(df: pd.DataFrame) -> dict:
    """Analysis 1: POR span / midpoint / obs_count distributions + usable-trend
    fraction, overall and split by HUC2 and by mean_dtw depth band."""

    def block(g: pd.DataFrame) -> dict:
        return {
            "n": int(len(g)),
            "n_matched_gwx": int(g["por_start"].notna().sum()),
            "por_span_years": _dist(g["por_span_years"]),
            "por_midpoint_year": _dist(g["por_midpoint_year"]),
            "obs_count": _dist(g["obs_count"]),
            "frac_usable_trend": round(float(g["usable_trend"].mean()), 4),
            "n_usable_trend": int(g["usable_trend"].sum()),
        }

    out = {"overall": block(df), "by_huc2": {}, "by_depth_band": {}}
    for h2, g in df.groupby("huc2"):
        if len(g) >= 100:
            out["by_huc2"][str(h2)] = block(g)
    for lab, g in df.groupby("depth_band", observed=True):
        out["by_depth_band"][str(lab)] = block(g)
    return out


def alias_magnitude(df: pd.DataFrame) -> dict:
    """Analysis 2: |implied WTE drift| distribution + exceedance fractions among
    usable-trend wells, and geography/depth breakdowns of the exceedances."""
    u = df[df["usable_trend"] & np.isfinite(df["implied_wte_drift_m"])]
    ad = np.abs(u["implied_wte_drift_m"].to_numpy())

    def exceed(g: pd.DataFrame) -> dict:
        a = np.abs(g["implied_wte_drift_m"].to_numpy())
        a = a[np.isfinite(a)]
        d = {"n": int(len(a))}
        for t in DRIFT_THRESHOLDS_M:
            d[f"frac_abs_drift_gt_{t:g}m"] = (
                round(float((a > t).mean()), 4) if len(a) else None
            )
        return d

    out = {
        "abs_implied_wte_drift_m": _dist(ad),
        "signed_implied_wte_drift_m": _dist(u["implied_wte_drift_m"]),
        "overall_exceedance": exceed(u),
        "by_huc2": {},
        "by_state": {},
        "by_depth_band": {},
    }
    for h2, g in u.groupby("huc2"):
        if len(g) >= 100:
            out["by_huc2"][str(h2)] = exceed(g)
    for st, g in u.groupby("state"):
        if len(g) >= 100:
            out["by_state"][str(st)] = exceed(g)
    for lab, g in u.groupby("depth_band", observed=True):
        out["by_depth_band"][str(lab)] = exceed(g)
    return out


def alias_innovation_link_panel(df: pd.DataFrame, seed: int = 0) -> dict:
    """Analysis 3: correlation/regression of innovation on implied drift among
    usable-trend wells, with a bootstrap CI on the OLS slope."""
    u = df[
        df["usable_trend"]
        & np.isfinite(df["implied_wte_drift_m"])
        & np.isfinite(df["innovation_m"])
    ]
    innov = u["innovation_m"].to_numpy()
    drift = u["implied_wte_drift_m"].to_numpy()
    link = alias_innovation_link(innov, drift)
    lo, hi = bootstrap_ols_slope_ci(drift, innov, n_boot=1000, seed=seed)
    link["ols_slope_ci95"] = [
        round(lo, 5) if np.isfinite(lo) else None,
        round(hi, 5) if np.isfinite(hi) else None,
    ]
    link["bootstrap_n_boot"] = 1000
    link["bootstrap_seed"] = seed
    return link


def _corr_block(g: pd.DataFrame) -> dict:
    raw = g["innovation_m"].to_numpy()
    return {
        "beta1": correction_metrics(raw, g["harmonized_innovation_beta1_m"].to_numpy()),
        "betaOLS": correction_metrics(
            raw, g["harmonized_innovation_betaOLS_m"].to_numpy()
        ),
    }


def correction_test(df: pd.DataFrame) -> dict:
    """Analysis 4 (headline): does subtracting the implied drift shrink the
    innovation error? Reports MAD/RMSE raw vs harmonized (beta=1 physical and
    beta=OLS leave-one-fold-out) on the usable-trend subset, stratified."""
    u = df[
        df["usable_trend"]
        & np.isfinite(df["implied_wte_drift_m"])
        & np.isfinite(df["innovation_m"])
    ].copy()

    out = {
        "note": (
            "Correction is defined only on usable-trend wells; a positive "
            "reduction fraction => POR aliasing is a real, correctable share of "
            "the residual. No reduction => aliasing is not the binding problem "
            "at these wells."
        ),
        "n_usable": int(len(u)),
        "overall": _corr_block(u),
        "by_depth_band": {},
        "by_huc2": {},
        "mech_deep_far_arid": {},
        "unconfined_only_robustness": {},
    }
    for lab, g in u.groupby("depth_band", observed=True):
        if len(g) > 0:
            out["by_depth_band"][str(lab)] = _corr_block(g)
    for h2, g in u.groupby("huc2"):
        if len(g) >= HUC2_MIN_WELLS:
            out["by_huc2"][str(h2)] = _corr_block(g)
    mech = u[u["mech_deep_far_arid"].astype(bool)]
    if len(mech) > 0:
        out["mech_deep_far_arid"] = {"n": int(len(mech)), **_corr_block(mech)}
    # CLAUDE.md metric discipline: MAD/RMSE are accuracy metrics — show the
    # water-table-only (unconfined) cut so confined potentiometric heads do not
    # distort the headline.
    unconf = u[u["confinement_class"].isin(["unconfined", "unconfined_marginal"])]
    if len(unconf) > 0:
        out["unconfined_only_robustness"] = {
            "n": int(len(unconf)),
            **_corr_block(unconf),
        }
    return out


# --------------------------------------------------------------------------- #
# harmonized-innovation columns
# --------------------------------------------------------------------------- #
def add_harmonized(df: pd.DataFrame) -> pd.DataFrame:
    """Attach beta=1 and beta=OLS-LOO harmonized innovations. beta_OLS is fit
    leave-one-fold-out over the usable-trend subset. Wells without a usable
    drift keep their raw innovation (no correction)."""
    u = (
        df["usable_trend"]
        & np.isfinite(df["implied_wte_drift_m"])
        & np.isfinite(df["innovation_m"])
    )
    beta_loo = np.full(len(df), np.nan)
    sub = df[u]
    if len(sub) >= 3:
        b = fit_beta_loo(
            sub["innovation_m"].to_numpy(),
            sub["implied_wte_drift_m"].to_numpy(),
            sub["cv_fold"].to_numpy(),
        )
        beta_loo[np.where(u.to_numpy())[0]] = b
    df["beta_ols_loo"] = beta_loo

    drift = df["implied_wte_drift_m"].to_numpy()
    innov = df["innovation_m"].to_numpy()
    df["harmonized_innovation_beta1_m"] = np.where(
        np.isfinite(drift), innov - 1.0 * drift, innov
    )
    df["harmonized_innovation_betaOLS_m"] = np.where(
        np.isfinite(drift) & np.isfinite(beta_loo), innov - beta_loo * drift, innov
    )
    return df


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
def definitions_block() -> dict:
    return {
        "innovation_m": "obs_wte_m - gnn_wte_hat_m (m, WTE head space); + = observed head above frozen background.",
        "trend_slope": "GWX water-level trend, m/yr of DTW; empirically-verified + = deepening (declining table).",
        "por_midpoint_year": "decimal-year midpoint of [por_start, por_end] (from the GWX product).",
        "por_span_years": "por_end - por_start in years (from the GWX product).",
        "usable_trend": "isfinite(trend_slope) AND trend_pvalue<0.05 AND trend_span_years>=5 (dimensionless flag).",
        "implied_dtw_drift_m": "trend_slope * (2020.0 - por_midpoint_year) (m); DTW change POR-midpoint->2020, + = deeper.",
        "implied_wte_drift_m": "-implied_dtw_drift_m (m); WTE=surface-DTW so a deepening table is a falling head.",
        "harmonized_innovation_beta1_m": "innovation_m - 1.0 * implied_wte_drift_m (m); physical unit correction.",
        "harmonized_innovation_betaOLS_m": "innovation_m - beta_loo * implied_wte_drift_m (m); beta_loo = OLS slope fit leave-one-cv_fold-out.",
        "mad_m": "median(|residual|) (m); robust central error.",
        "rmse_m": "sqrt(mean(residual^2)) (m); tail-sensitive error.",
        "mad_reduction_frac": "1 - MAD_harmonized/MAD_raw (dimensionless); + = correction shrinks the error.",
        "rmse_reduction_frac": "1 - RMSE_harmonized/RMSE_raw (dimensionless); + = correction shrinks the error.",
        "ols_r2": "Pearson r**2 of innovation on implied drift (dimensionless); variance fraction linearly explained.",
        "pearson_r / spearman_r": "linear / rank correlation of innovation with implied drift (dimensionless, [-1,1]).",
        "frac_usable_trend": "fraction of the population with a usable trend (dimensionless).",
        "frac_abs_drift_gt_Xm": "fraction of usable-trend wells with |implied_wte_drift| > X m (dimensionless).",
        "reference_epoch_year": COMMON_EPOCH,
        "population": "real monitoring wells (is_water_pseudo=False), regional_holdout != 'sacrificial'.",
    }


def _fmt(v, nd=3):
    return (
        "n/a"
        if v is None or (isinstance(v, float) and not np.isfinite(v))
        else f"{v:.{nd}f}"
    )


def write_summary_md(path: Path, report: dict) -> None:
    cov = report["join_provenance"]
    sign = report["sign_convention"]
    link = report["alias_innovation_link"]
    corr = report["correction_test"]
    mag = report["alias_magnitude"]
    o1 = corr["overall"]
    deep = corr.get("mech_deep_far_arid", {})
    band30 = corr["by_depth_band"].get("30+m", {})

    lines = [
        "# WP3 rung-0 — period-of-record temporal-alias census",
        "",
        f"Generated {date.today()} by utils/v02_temporal_alias_census.py. All "
        "lengths in metres, WTE (head) space. This is observational evidence "
        "for the plan §8 gate; per §8.3 a better in-sample hydrograph "
        "reconstruction is insufficient on its own.",
        "",
        "## Question",
        "",
        "How much of the frozen background's residual innovation is period-of-"
        "record aliasing, and is common-epoch harmonization feasible/worthwhile?",
        "",
        "## Data provenance and surprises",
        "",
        f"- Population: {cov['n_population_real_nonsacrificial']} real "
        f"monitoring wells (non-pseudo, non-sacrificial); "
        f"{cov['n_monitoring_wells_pre_sacrificial']} before dropping sacrificial.",
        f"- **Join surprise:** contract `canonical_id` is `gwx_`+8 hex; GWX "
        f"product id is `gwx_`+16 hex. Direct join = 0 matches; joined on the "
        f"8-hex prefix. {cov['frac_matched_to_gwx'] * 100:.1f}% of the population "
        f"matched a GWX record.",
        "- **POR surprise:** por_start/por_end/obs_count are entirely null in "
        "wells_panels.parquet; temporal support taken from the GWX product.",
        f"- Dedup collisions: {cov['n_prefixes_with_multiple_gwx_rows']} prefixes "
        f"had >1 GWX row (max {cov['max_gwx_rows_per_prefix']} rows/prefix; kept "
        f"max trend_n_obs); {cov['n_prefixes_matching_multiple_distinct_16hex_ids']} "
        f"prefixes were ambiguous (matched >1 distinct 16-hex id).",
        f"- Sign convention (empirically verified): {sign['inferred_convention']}",
        "",
        "## Analysis 1 — temporal support",
        "",
        f"- Usable-trend fraction (finite slope, p<0.05, span>=5 yr): "
        f"**{cov['frac_usable_trend'] * 100:.1f}%** "
        f"({cov['n_usable_trend']} of {cov['n_population_real_nonsacrificial']}).",
        f"- Only {cov['n_finite_trend_slope']} wells have any finite trend_slope; "
        f"{cov['n_matched_to_gwx']} matched a GWX record at all.",
        f"- POR span (yr): median "
        f"{_fmt(report['temporal_support']['overall']['por_span_years']['median'], 1)}, "
        f"IQR {_fmt(report['temporal_support']['overall']['por_span_years']['p25'], 1)}"
        f"-{_fmt(report['temporal_support']['overall']['por_span_years']['p75'], 1)}.",
        f"- POR midpoint (yr): median "
        f"{_fmt(report['temporal_support']['overall']['por_midpoint_year']['median'], 1)}, "
        f"p10-p90 {_fmt(report['temporal_support']['overall']['por_midpoint_year']['p10'], 1)}"
        f"-{_fmt(report['temporal_support']['overall']['por_midpoint_year']['p90'], 1)}.",
        "",
        "## Analysis 2 — alias magnitude (usable-trend wells)",
        "",
        f"- |implied WTE drift|: median {_fmt(mag['abs_implied_wte_drift_m']['median'])} m, "
        f"p90 {_fmt(mag['abs_implied_wte_drift_m']['p90'])} m.",
        f"- Fraction |drift| > 2 m: "
        f"**{_fmt(mag['overall_exceedance']['frac_abs_drift_gt_2m'])}**; "
        f"> 5 m: {_fmt(mag['overall_exceedance']['frac_abs_drift_gt_5m'])}.",
        "",
        "## Analysis 3 — alias-innovation link (usable-trend wells)",
        "",
        f"- n = {link['n']}; Pearson r = {_fmt(link['pearson_r'])}, "
        f"Spearman rho = {_fmt(link['spearman_r'])}.",
        f"- OLS slope = {_fmt(link['ols_slope'])} (CI95 {link['ols_slope_ci95']}); "
        f"variance fraction R2 = {_fmt(link['ols_r2'])}.",
        "",
        "## Analysis 4 — correction test (headline)",
        "",
        "| set | n | MAD raw | MAD beta=1 | MAD beta=OLS | RMSE raw | RMSE beta=OLS |",
        "|---|---|---|---|---|---|---|",
        f"| overall | {o1['beta1']['n']} | {_fmt(o1['beta1']['mad_raw_m'])} | "
        f"{_fmt(o1['beta1']['mad_harmonized_m'])} | {_fmt(o1['betaOLS']['mad_harmonized_m'])} | "
        f"{_fmt(o1['beta1']['rmse_raw_m'])} | {_fmt(o1['betaOLS']['rmse_harmonized_m'])} |",
    ]
    if band30:
        lines.append(
            f"| 30+m band | {band30['beta1']['n']} | {_fmt(band30['beta1']['mad_raw_m'])} | "
            f"{_fmt(band30['beta1']['mad_harmonized_m'])} | {_fmt(band30['betaOLS']['mad_harmonized_m'])} | "
            f"{_fmt(band30['beta1']['rmse_raw_m'])} | {_fmt(band30['betaOLS']['rmse_harmonized_m'])} |"
        )
    if deep:
        lines.append(
            f"| deep_far_arid | {deep['beta1']['n']} | {_fmt(deep['beta1']['mad_raw_m'])} | "
            f"{_fmt(deep['beta1']['mad_harmonized_m'])} | {_fmt(deep['betaOLS']['mad_harmonized_m'])} | "
            f"{_fmt(deep['beta1']['rmse_raw_m'])} | {_fmt(deep['betaOLS']['rmse_harmonized_m'])} |"
        )
    lines += [
        "",
        f"Overall MAD reduction: beta=1 "
        f"{_fmt(o1['beta1']['mad_reduction_frac'])}, beta=OLS "
        f"{_fmt(o1['betaOLS']['mad_reduction_frac'])} (dimensionless; + = "
        f"aliasing is a real, correctable share of the residual).",
        "",
        "## Verdict",
        "",
        report["verdict"],
        "",
    ]
    path.write_text("\n".join(lines) + "\n")


def build_verdict(report: dict) -> str:
    cov = report["join_provenance"]
    link = report["alias_innovation_link"]
    corr = report["correction_test"]
    o = corr["overall"]
    r2 = link["ols_r2"] or 0.0
    red1 = o["beta1"]["mad_reduction_frac"]
    redols = o["betaOLS"]["mad_reduction_frac"]
    frac_usable = cov["frac_usable_trend"]
    mag2 = (
        report["alias_magnitude"]["overall_exceedance"].get("frac_abs_drift_gt_2m")
        or 0.0
    )

    best_red = max(red1, redols)
    if frac_usable < 0.30:
        support_clause = (
            f"Only {frac_usable * 100:.0f}% of monitoring wells carry a usable "
            f"trend, so any common-epoch correction can touch a minority of the "
            f"population directly; the rest would require modeled temporal modes."
        )
    else:
        support_clause = (
            f"{frac_usable * 100:.0f}% of monitoring wells carry a usable trend, "
            f"giving adequate direct support for a common-epoch correction."
        )

    if best_red >= 0.10 and r2 >= 0.05:
        stance = "JUSTIFIED"
        body = (
            f"The implied epoch drift explains a measurable share of the "
            f"background innovation (R2={r2:.3f}) and correcting for it reduces "
            f"MAD by up to {best_red * 100:.1f}% on usable-trend wells. Period-of-"
            f"record aliasing is a real, correctable error source at these wells."
        )
    elif best_red >= 0.03 or (r2 >= 0.02 and mag2 >= 0.10):
        stance = "MARGINAL"
        body = (
            f"The drift-innovation link is weak (R2={r2:.3f}) and the best MAD "
            f"reduction is only {best_red * 100:.1f}%. There is a detectable but "
            f"small aliasing signal; a full hydrograph-decomposition program is "
            f"not clearly worth its cost from this census alone."
        )
    else:
        stance = "NOT SUPPORTED"
        body = (
            f"The implied epoch drift explains essentially none of the "
            f"background innovation (R2={r2:.3f}) and harmonization does not "
            f"reduce MAD (best {best_red * 100:.1f}%). Period-of-record aliasing "
            f"is not the binding residual error at these wells."
        )
    return (
        f"WP3 common-epoch program is **{stance}** by this census. {body} "
        f"{support_clause} Per plan §8.3, this is observational evidence only: "
        f"a better in-sample hydrograph reconstruction would not by itself "
        f"justify replacing mean_dtw — the correction must reduce held-out "
        f"spatial residual structure in pumping-sensitive regions without "
        f"degrading stable ones. This rung-0 result "
        + (
            "clears the bar to proceed to that held-out test."
            if stance == "JUSTIFIED"
            else "does not clear the bar to build the full common-epoch target."
        )
    )


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--contract-dir", default=CONTRACT_DIR)
    ap.add_argument("--gwx-wells", default=GWX_WELLS)
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    log.info("loading + joining contract/GWX ...")
    df, join_prov = load_and_join(args.contract_dir, args.gwx_wells)
    log.info(
        "population=%d matched_gwx=%d usable_trend=%d (%.1f%%)",
        len(df),
        join_prov["n_matched_to_gwx"],
        join_prov["n_usable_trend"],
        100 * join_prov["frac_usable_trend"],
    )

    df = add_harmonized(df)

    report = {
        "generated": str(date.today()),
        "background_arm": "gnn_conus_monitoring_water_gate_mirror_sigma_w25_prod",
        "reference_epoch_year": COMMON_EPOCH,
        "definitions": definitions_block(),
        "join_provenance": join_prov,
        "sign_convention": verify_sign_convention(df),
        "temporal_support": temporal_support_census(df),
        "alias_magnitude": alias_magnitude(df),
        "alias_innovation_link": alias_innovation_link_panel(df, seed=args.seed),
        "correction_test": correction_test(df),
    }
    report["verdict"] = build_verdict(report)
    log.info("verdict: %s", report["verdict"].split(".")[0])

    (out / "temporal_alias_report.json").write_text(json.dumps(report, indent=2))
    write_summary_md(out / "temporal_alias_summary.md", report)

    keep = [
        "canonical_id",
        "huc2",
        "state",
        "mean_dtw",
        "depth_band",
        "confinement_class",
        "mech_deep_far_arid",
        "cv_fold",
        "por_start",
        "por_end",
        "por_start_year",
        "por_end_year",
        "por_midpoint_year",
        "por_span_years",
        "obs_count",
        "trend_slope",
        "trend_slope_lo",
        "trend_slope_hi",
        "trend_pvalue",
        "trend_n_obs",
        "trend_span_years",
        "trend_direction",
        "usable_trend",
        "implied_dtw_drift_m",
        "implied_wte_drift_m",
        "obs_wte_m",
        "gnn_wte_hat_m",
        "innovation_m",
        "beta_ols_loo",
        "harmonized_innovation_beta1_m",
        "harmonized_innovation_betaOLS_m",
    ]
    df_out = df[[c for c in keep if c in df.columns]].copy()
    df_out["depth_band"] = df_out["depth_band"].astype(str)
    df_out.to_parquet(out / "alias_per_well.parquet", index=False)
    log.info("WP3 temporal-alias artifacts written to %s", out)


if __name__ == "__main__":
    main()
