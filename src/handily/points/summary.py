"""Aggregate point-year records to per-point summary statistics."""

from __future__ import annotations

import logging
import os

import pandas as pd

from handily.io import ensure_dir

LOGGER = logging.getLogger("handily.points.summary")

_STATIC_JOIN_COLS = [
    "point_id",
    "rem_at_sample",
    "sample_group",
    "rem_bin",
    "in_irrigated_lands",
    "is_low_rem_target",
    "is_riparian_target",
    "is_field_edge_target",
    "is_high_rem_control",
]

_BEHAVIOR_FLAGS = [
    ("is_low_rem_target", "low_rem_flag"),
    ("is_high_rem_control", "high_rem_control_flag"),
    ("is_riparian_target", "riparian_flag"),
]


def build_point_summary(
    point_year: pd.DataFrame,
    points_static: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Summarise the point-year table to one row per point.

    Parameters
    ----------
    point_year : DataFrame
        Output of ``build_point_year_table``.
    points_static : DataFrame or None
        If provided, static columns (rem_at_sample, sample_group, flags, etc.)
        are joined onto the summary by point_id.

    Returns
    -------
    DataFrame with one row per point_id.
    """
    grp = point_year.groupby("point_id")

    summary = pd.DataFrame(
        {
            "n_years": grp["year"].count(),
            "valid_years": grp["feature_valid"].sum().astype(int),
        }
    )

    if "irr_class" in point_year.columns:
        irr_sub = point_year.loc[point_year["irr_valid"]]
        irr_grp = irr_sub.groupby("point_id")["irr_class"]
        summary["irr_freq"] = irr_grp.mean()
        summary["irr_frac_mean"] = summary["irr_freq"]
        summary["irr_frac_std"] = irr_grp.std()
        summary["n_irr_years"] = irr_grp.count()

    if "aet_annual" in point_year.columns:
        et_mask = (
            point_year["et_valid"] if "et_valid" in point_year.columns else slice(None)
        )
        et_sub = point_year.loc[et_mask]
        et_grp = et_sub.groupby("point_id")["aet_annual"]
        summary["aet_mean"] = et_grp.mean()
        summary["aet_std"] = et_grp.std()

        if "eto_annual" in et_sub.columns:
            eto_grp = et_sub.groupby("point_id")["eto_annual"]
            summary["eto_mean"] = eto_grp.mean()
        if "eto_gs" in et_sub.columns:
            summary["eto_gs_mean"] = et_sub.groupby("point_id")["eto_gs"].mean()
        if "pr_annual" in et_sub.columns:
            pr_grp = et_sub.groupby("point_id")["pr_annual"]
            summary["prcp_mean"] = pr_grp.mean()
        if "pr_gs" in et_sub.columns:
            summary["prcp_gs_mean"] = et_sub.groupby("point_id")["pr_gs"].mean()
        if "etf" in et_sub.columns:
            etf_grp = et_sub.groupby("point_id")["etf"]
            summary["etf_mean"] = etf_grp.mean()
            summary["etf_std"] = etf_grp.std()
        if "net_et" in et_sub.columns:
            summary["net_et_mean"] = et_sub.groupby("point_id")["net_et"].mean()

    ndvi_src_cols = ["ndvi_amp_gs", "ndvi_peak_gs", "ndvi_mean_gs"]
    if any(c in point_year.columns for c in ndvi_src_cols):
        ndvi_mask = (
            point_year["ndvi_valid"]
            if "ndvi_valid" in point_year.columns
            else slice(None)
        )
        ndvi_sub = point_year.loc[ndvi_mask]
        ndvi_grp = ndvi_sub.groupby("point_id")
        for src in ndvi_src_cols:
            if src in ndvi_sub.columns:
                base = src.replace("_gs", "")
                summary[f"{base}_mean"] = ndvi_grp[src].mean()
        if "ndvi_amp_gs" in ndvi_sub.columns:
            summary["ndvi_amp_std"] = ndvi_grp["ndvi_amp_gs"].std()

    summary = summary.reset_index()

    if points_static is not None:
        keep = [c for c in _STATIC_JOIN_COLS if c in points_static.columns]
        summary = summary.merge(points_static[keep], on="point_id", how="left")
        for src, dst in _BEHAVIOR_FLAGS:
            if src in summary.columns:
                summary[dst] = summary[src].astype(bool)

    return summary


def write_point_summary(df: pd.DataFrame, out_dir: str) -> str:
    """Write point_summary to ``{out_dir}/point_summary.parquet``."""
    ensure_dir(out_dir)
    path = os.path.join(out_dir, "point_summary.parquet")
    df.to_parquet(path, index=False)
    LOGGER.info("Wrote point_summary: %d rows → %s", len(df), path)
    return path


# ========================= EOF =======================================================================================
