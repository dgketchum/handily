"""Donor neighborhood scoring for point-based background ET estimation.

Matches each irrigated recipient point-year to a neighborhood of unirrigated
donor point-years at similar REM, hydro-setting, and phenology. Produces
background ETf estimates via weighted median of the donor neighborhood.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from handily.io import ensure_dir

LOGGER = logging.getLogger("handily.points.donor_score")


# ---------------------------------------------------------------------------
# Config defaults
# ---------------------------------------------------------------------------
@dataclass
class DonorConfig:
    """Parameters for donor scoring. Override via config or kwargs."""

    target_col: str = "etf"
    donor_year_max_irr: float = 0.1
    donor_max_irr_freq: float = 0.15
    donor_max_irr_std: float = 0.3
    donor_min_valid_years: int = 10
    donor_rem_tolerance_m: float = 2.0
    donor_search_radius_m: float = 10000.0
    donor_top_k: int = 7
    donor_min_k: int = 3
    donor_field_edge_exclusion_m: float = 100.0
    donor_managed_buffer_m: float = 500.0
    donor_perennial_buffer_m: float = 500.0
    donor_intermittent_buffer_m: float = 1000.0
    donor_ndvi_peak_tol: float = 0.15
    donor_ndvi_amp_tol: float = 0.10
    # Score sigmas
    donor_sigma_rem_m: float = 1.0
    donor_sigma_distance_m: float = 3000.0
    donor_sigma_etf_std: float = 0.15
    donor_sigma_ndvi_std: float = 0.08
    donor_purity_power: float = 2.0
    donor_trim_fraction: float = 0.1
    # Confidence thresholds
    donor_min_effective_n: float = 2.5
    donor_max_single_weight: float = 0.5
    donor_estimator_gap_tol: float = 0.05


# ---------------------------------------------------------------------------
# Hydro analog classification
# ---------------------------------------------------------------------------
def assign_hydro_analog_class(
    df: pd.DataFrame,
    cfg: DonorConfig,
) -> pd.Series:
    """Derive hydro analog class from stream distance columns."""
    classes = pd.Series("off_corridor", index=df.index, dtype="object")

    if "dist_intermittent_m" in df.columns:
        mask = df["dist_intermittent_m"] <= cfg.donor_intermittent_buffer_m
        classes = classes.where(~mask, "intermittent_corridor")
    if "dist_perennial_m" in df.columns:
        mask = df["dist_perennial_m"] <= cfg.donor_perennial_buffer_m
        classes = classes.where(~mask, "perennial_corridor")
    if "dist_managed_m" in df.columns:
        mask = df["dist_managed_m"] <= cfg.donor_managed_buffer_m
        classes = classes.where(~mask, "managed_corridor")

    return classes


# ---------------------------------------------------------------------------
# Table assembly
# ---------------------------------------------------------------------------
def assemble_donor_model_table(
    point_year: pd.DataFrame,
    point_summary: pd.DataFrame,
    points_static: pd.DataFrame,
    cfg: DonorConfig,
) -> pd.DataFrame:
    """Join point-year with summary and static features; derive analog class."""
    # Join summary cols
    summary_cols = [
        "point_id",
        "irr_freq",
        "irr_frac_std",
        "etf_mean",
        "etf_std",
        "ndvi_amp_mean",
        "ndvi_amp_std",
        "ndvi_peak_mean",
        "valid_years",
    ]
    summary_cols = [c for c in summary_cols if c in point_summary.columns]
    table = point_year.merge(
        point_summary[summary_cols],
        on="point_id",
        how="left",
        suffixes=("", "_summ"),
    )

    # Join static cols
    static_cols = [
        "point_id",
        "in_irrigated_lands",
        "rem_at_sample",
        "dist_field_edge_m",
        "nearest_stream_type",
        "x",
        "y",
    ]
    # Add hydro distance cols if present
    for c in points_static.columns:
        if c.startswith("dist_") and c not in static_cols:
            static_cols.append(c)
    static_cols = [c for c in static_cols if c in points_static.columns]
    table = table.merge(points_static[static_cols], on="point_id", how="left")

    # Derive hydro analog class
    table["hydro_analog"] = assign_hydro_analog_class(table, cfg)

    # Derive REM bin
    bins = [0, 1, 2, 5, 10, float("inf")]
    labels = ["0-1m", "1-2m", "2-5m", "5-10m", ">10m"]
    table["rem_bin"] = pd.cut(
        table["rem_at_sample"],
        bins=bins,
        labels=labels,
        right=False,
    ).astype(str)

    return table


# ---------------------------------------------------------------------------
# Hard filters
# ---------------------------------------------------------------------------
def identify_recipients(
    table: pd.DataFrame,
    cfg: DonorConfig,
) -> pd.DataFrame:
    """Select recipient point-years (irrigated, valid, with target data)."""
    mask = (
        table["feature_valid"]
        & (table["irr_class"] >= 0.5)
        & (table[cfg.target_col].notna())
        & (table[cfg.target_col] > -999)
    )
    if "eto_gs" in table.columns:
        mask = mask & (table["eto_gs"] > 0)

    recipients = table.loc[mask].copy()
    LOGGER.info("Recipients: %d point-years", len(recipients))
    return recipients


def mask_inadmissible_donors(
    table: pd.DataFrame,
    cfg: DonorConfig,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Apply hard donor filters. Returns (admissible_table, exclusion_counts)."""
    n_start = len(table)
    exclusions: dict[str, int] = {}

    # Data sufficiency
    mask = table["feature_valid"] & table[cfg.target_col].notna()
    if "valid_years" in table.columns:
        mask = mask & (table["valid_years"] >= cfg.donor_min_valid_years)
    if "eto_gs" in table.columns:
        mask = mask & (table["eto_gs"] > 0)
    exclusions["data_sufficiency"] = n_start - mask.sum()
    table = table.loc[mask]

    # Year-specific non-irrigated
    mask = table["irr_class"] <= cfg.donor_year_max_irr
    exclusions["year_irrigated"] = len(table) - mask.sum()
    table = table.loc[mask]

    # Long-term purity
    mask = pd.Series(True, index=table.index)
    if "irr_freq" in table.columns:
        mask = mask & (table["irr_freq"] <= cfg.donor_max_irr_freq)
    if "irr_frac_std" in table.columns:
        mask = mask & (table["irr_frac_std"] <= cfg.donor_max_irr_std)
    exclusions["purity"] = len(table) - mask.sum()
    table = table.loc[mask]

    # Field edge contamination
    if "dist_field_edge_m" in table.columns:
        mask = table["dist_field_edge_m"] >= cfg.donor_field_edge_exclusion_m
        exclusions["field_edge"] = len(table) - mask.sum()
        table = table.loc[mask]

    LOGGER.info(
        "Admissible donors: %d / %d (exclusions: %s)",
        len(table),
        n_start,
        exclusions,
    )
    return table.copy(), exclusions


# ---------------------------------------------------------------------------
# Pairwise scoring
# ---------------------------------------------------------------------------
def _score_pairs(
    recipients: pd.DataFrame,
    donors: pd.DataFrame,
    pairs_idx: np.ndarray,
    pairs_dist: np.ndarray,
    cfg: DonorConfig,
) -> pd.DataFrame:
    """Score recipient-donor pairs. pairs_idx[i] are donor indices for recipient i."""
    rows = []
    for r_pos in range(len(recipients)):
        r = recipients.iloc[r_pos]
        for j, d_idx in enumerate(pairs_idx[r_pos]):
            if d_idx < 0 or d_idx >= len(donors):
                continue
            dist_m = pairs_dist[r_pos][j]
            if dist_m > cfg.donor_search_radius_m:
                continue

            d = donors.iloc[d_idx]

            # Hard: REM + hydro + phenology
            rem_delta = abs(r["rem_at_sample"] - d["rem_at_sample"])
            if rem_delta > cfg.donor_rem_tolerance_m:
                continue
            if r.get("rem_bin") != d.get("rem_bin"):
                continue
            if r.get("hydro_analog") != d.get("hydro_analog"):
                continue

            # Phenology deltas — used in soft score only, no hard gate
            ndvi_peak_delta = abs(r.get("ndvi_peak_gs", 0) - d.get("ndvi_peak_gs", 0))
            ndvi_amp_delta = abs(r.get("ndvi_amp_gs", 0) - d.get("ndvi_amp_gs", 0))

            # Soft scores (multiplicative)
            s_rem = np.exp(-rem_delta / cfg.donor_sigma_rem_m)
            s_dist = np.exp(-dist_m / cfg.donor_sigma_distance_m)
            s_pheno = np.exp(
                -(
                    ndvi_peak_delta / max(cfg.donor_ndvi_peak_tol, 0.01)
                    + ndvi_amp_delta / max(cfg.donor_ndvi_amp_tol, 0.01)
                )
            )
            irr_freq = d.get("irr_freq", 0)
            s_purity = (1 - irr_freq) ** cfg.donor_purity_power
            etf_std = d.get("etf_std", 0) if pd.notna(d.get("etf_std")) else 0
            ndvi_amp_std = (
                d.get("ndvi_amp_std", 0) if pd.notna(d.get("ndvi_amp_std")) else 0
            )
            s_stability = np.exp(
                -(
                    etf_std / cfg.donor_sigma_etf_std
                    + ndvi_amp_std / cfg.donor_sigma_ndvi_std
                )
            )

            pair_score = s_rem * s_dist * s_pheno * s_purity * s_stability

            rows.append(
                {
                    "recipient_point_id": r["point_id"],
                    "recipient_year": int(r["year"]),
                    "donor_point_id": d["point_id"],
                    "donor_year": int(d["year"]),
                    "pair_score": float(pair_score),
                    "distance_m": float(dist_m),
                    "rem_delta": float(rem_delta),
                    "target_value": float(d[cfg.target_col]),
                    "score_rem": float(s_rem),
                    "score_distance": float(s_dist),
                    "score_phenology": float(s_pheno),
                    "score_purity": float(s_purity),
                    "score_stability": float(s_stability),
                }
            )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Top-k selection and background estimation
# ---------------------------------------------------------------------------
def select_top_k_donors(
    pairs: pd.DataFrame,
    cfg: DonorConfig,
) -> pd.DataFrame:
    """Keep top-k donors per recipient point-year, normalize weights."""
    if pairs.empty:
        return pairs

    pairs = pairs.sort_values("pair_score", ascending=False)
    top_k = (
        pairs.groupby(["recipient_point_id", "recipient_year"])
        .head(cfg.donor_top_k)
        .copy()
    )

    # Normalize to weights within each recipient
    weight_sums = top_k.groupby(["recipient_point_id", "recipient_year"])[
        "pair_score"
    ].transform("sum")
    top_k["pair_weight"] = top_k["pair_score"] / weight_sums.clip(lower=1e-12)

    return top_k.reset_index(drop=True)


def _weighted_median(values, weights):
    """Weighted median of values."""
    order = np.argsort(values)
    values = np.array(values)[order]
    weights = np.array(weights)[order]
    cum_weight = np.cumsum(weights)
    cutoff = cum_weight[-1] * 0.5
    idx = np.searchsorted(cum_weight, cutoff)
    return float(values[min(idx, len(values) - 1)])


def _weighted_trimmed_mean(values, weights, trim_fraction=0.1):
    """Weighted trimmed mean."""
    order = np.argsort(values)
    values = np.array(values)[order]
    weights = np.array(weights)[order]
    cum_weight = np.cumsum(weights)
    total = cum_weight[-1]
    lo = total * trim_fraction
    hi = total * (1 - trim_fraction)
    mask = (cum_weight >= lo) & (cum_weight <= hi)
    if not mask.any():
        return float(np.average(values, weights=weights))
    return float(np.average(values[mask], weights=weights[mask]))


def estimate_background(
    top_k_pairs: pd.DataFrame,
    cfg: DonorConfig,
) -> pd.DataFrame:
    """Estimate background ETf for each recipient point-year from donor neighborhood."""
    if top_k_pairs.empty:
        return pd.DataFrame()

    results = []
    for (pid, year), group in top_k_pairs.groupby(
        ["recipient_point_id", "recipient_year"]
    ):
        n_donors = len(group)
        values = group["target_value"].values
        weights = group["pair_weight"].values

        status = "ok"
        no_donor_reason = None
        confidence_flag = None

        if n_donors < cfg.donor_min_k:
            status = "no_donor"
            no_donor_reason = "insufficient_neighbors"
            results.append(
                {
                    "point_id": pid,
                    "year": year,
                    "status": status,
                    "no_donor_reason": no_donor_reason,
                    "background_etf_wmedian": np.nan,
                    "background_etf_trimmed": np.nan,
                    "n_donors_used": n_donors,
                    "effective_n": 0,
                    "max_weight": 0,
                    "median_distance_m": 0,
                    "median_rem_delta": 0,
                    "confidence_flag": None,
                }
            )
            continue

        bg_wmedian = _weighted_median(values, weights)
        bg_trimmed = _weighted_trimmed_mean(values, weights, cfg.donor_trim_fraction)
        effective_n = 1.0 / (weights**2).sum() if (weights**2).sum() > 0 else 0
        max_weight = float(weights.max())
        median_dist = float(np.median(group["distance_m"]))
        median_rem = float(np.median(group["rem_delta"]))
        estimator_gap = abs(bg_wmedian - bg_trimmed)

        # Confidence checks
        flags = []
        if effective_n < cfg.donor_min_effective_n:
            flags.append("low_effective_n")
        if max_weight > cfg.donor_max_single_weight:
            flags.append("dominant_donor")
        if estimator_gap > cfg.donor_estimator_gap_tol:
            flags.append("estimator_disagreement")

        if flags:
            status = "low_confidence"
            confidence_flag = ";".join(flags)

        results.append(
            {
                "point_id": pid,
                "year": year,
                "status": status,
                "no_donor_reason": no_donor_reason,
                "background_etf_wmedian": bg_wmedian,
                "background_etf_trimmed": bg_trimmed,
                "n_donors_used": n_donors,
                "effective_n": effective_n,
                "max_weight": max_weight,
                "median_distance_m": median_dist,
                "median_rem_delta": median_rem,
                "confidence_flag": confidence_flag,
            }
        )

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------
def score_donors_for_aoi(
    point_year: pd.DataFrame,
    point_summary: pd.DataFrame,
    points_static: pd.DataFrame,
    cfg: DonorConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run donor scoring for a single AOI.

    Returns
    -------
    donor_scores : DataFrame
        One row per recipient point-year with background estimate and diagnostics.
    donor_pairs : DataFrame
        Audit table of retained donor pairs.
    """
    if cfg is None:
        cfg = DonorConfig()

    # Assemble
    table = assemble_donor_model_table(point_year, point_summary, points_static, cfg)

    # Identify recipients and admissible donors
    recipients = identify_recipients(table, cfg)
    donors_all, exclusions = mask_inadmissible_donors(table, cfg)

    if recipients.empty:
        LOGGER.warning("No recipient point-years found")
        return pd.DataFrame(), pd.DataFrame()
    if donors_all.empty:
        LOGGER.warning("No admissible donors found")
        return pd.DataFrame(), pd.DataFrame()

    # Build spatial index on donors and score by year
    all_pairs = []
    for year in sorted(recipients["year"].unique()):
        r_year = recipients[recipients["year"] == year].reset_index(drop=True)
        d_year = donors_all[donors_all["year"] == year].reset_index(drop=True)
        if r_year.empty or d_year.empty:
            continue

        # KD-tree on donor coordinates
        d_coords = d_year[["x", "y"]].values
        r_coords = r_year[["x", "y"]].values
        tree = cKDTree(d_coords)

        # Query within search radius, up to 50 candidates per recipient
        max_candidates = min(50, len(d_year))
        dist, idx = tree.query(
            r_coords, k=max_candidates, distance_upper_bound=cfg.donor_search_radius_m
        )

        year_pairs = _score_pairs(r_year, d_year, idx, dist, cfg)
        if not year_pairs.empty:
            all_pairs.append(year_pairs)

    if not all_pairs:
        LOGGER.warning("No valid donor pairs generated")
        return pd.DataFrame(), pd.DataFrame()

    pairs = pd.concat(all_pairs, ignore_index=True)
    LOGGER.info("Raw pairs: %d", len(pairs))

    # Top-k selection
    top_k = select_top_k_donors(pairs, cfg)
    LOGGER.info("Top-k pairs: %d", len(top_k))

    # Background estimation
    scores = estimate_background(top_k, cfg)
    LOGGER.info(
        "Donor scores: %d recipients, %d ok, %d low_confidence, %d no_donor",
        len(scores),
        (scores["status"] == "ok").sum(),
        (scores["status"] == "low_confidence").sum(),
        (scores["status"] == "no_donor").sum(),
    )

    return scores, top_k


def write_donor_outputs(
    donor_scores: pd.DataFrame,
    donor_pairs: pd.DataFrame,
    out_dir: str,
) -> dict[str, str]:
    """Write donor scoring outputs to per-AOI points directory."""
    ensure_dir(out_dir)
    paths = {}

    if not donor_scores.empty:
        p = os.path.join(out_dir, "donor_scores.parquet")
        donor_scores.to_parquet(p, index=False)
        paths["scores"] = p
        LOGGER.info("Wrote donor_scores: %d rows → %s", len(donor_scores), p)

    if not donor_pairs.empty:
        p = os.path.join(out_dir, "donor_pairs.parquet")
        donor_pairs.to_parquet(p, index=False)
        paths["pairs"] = p
        LOGGER.info("Wrote donor_pairs: %d rows → %s", len(donor_pairs), p)

    return paths
