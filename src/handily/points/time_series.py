"""Build point-year feature table from EE CSV exports."""

from __future__ import annotations

import logging
import os
import re

import numpy as np
import pandas as pd

from handily.io import ensure_dir

LOGGER = logging.getLogger("handily.points.time_series")

_IRRMAPPER_BAND_RE = re.compile(r"^irr_(\d{4})$")
_NDVI_BAND_RE = re.compile(r"^ndvi_(mean|peak|p25|p75|n_obs)_(\d{4})$")
_ETA_BAND_RE = re.compile(r"^eta_(\d{4})_(\d{2})$")

GROWING_SEASON_MONTHS = (4, 10)  # April–October inclusive


def extract_irrmapper_history(df: pd.DataFrame) -> pd.DataFrame:
    """Melt IrrMapper wide CSV to long format.

    Input columns: point_id + irr_YYYY (1.0=irrigated, 0.0=not irrigated).
    Output columns: point_id, year, irr_class.
    """
    col_year = {
        col: int(m.group(1))
        for col in df.columns
        if (m := _IRRMAPPER_BAND_RE.match(col))
    }
    if not col_year:
        raise ValueError("No irr_YYYY columns found in IrrMapper CSV")

    long = df[["point_id"] + list(col_year)].melt(
        id_vars="point_id", var_name="_band", value_name="irr_class"
    )
    long["year"] = long["_band"].map(col_year)
    long["irr_class"] = long["irr_class"].astype("float32")
    return long[["point_id", "year", "irr_class"]].reset_index(drop=True)


def extract_ndvi_history(df: pd.DataFrame) -> pd.DataFrame:
    """Reshape NDVI wide CSV to long format.

    Input columns: point_id + ndvi_{stat}_{year} for stat in {mean, peak, p25, p75, n_obs}.
    Output columns: point_id, year, ndvi_mean_gs, ndvi_peak_gs, ndvi_p25_gs,
                    ndvi_p75_gs, ndvi_n_obs_gs, ndvi_amp_gs.
    Amplitude = peak − p25, clipped to 0.
    """
    year_stats: dict[int, dict[str, str]] = {}
    for col in df.columns:
        m = _NDVI_BAND_RE.match(col)
        if m:
            stat, year = m.group(1), int(m.group(2))
            year_stats.setdefault(year, {})[stat] = col
    if not year_stats:
        raise ValueError("No ndvi_STAT_YYYY columns found in NDVI CSV")

    frames = []
    for year, stat_cols in sorted(year_stats.items()):
        row = df[["point_id"]].copy()
        row["year"] = year
        for stat, col in stat_cols.items():
            row[f"ndvi_{stat}_gs"] = df[col].to_numpy().astype("float32")
        frames.append(row)

    long = pd.concat(frames, ignore_index=True)
    if "ndvi_peak_gs" in long.columns and "ndvi_p25_gs" in long.columns:
        long["ndvi_amp_gs"] = (
            (long["ndvi_peak_gs"] - long["ndvi_p25_gs"]).clip(lower=0).astype("float32")
        )
    return long.reset_index(drop=True)


def extract_et_history(
    df: pd.DataFrame,
    gs_months: tuple[int, int] = GROWING_SEASON_MONTHS,
) -> pd.DataFrame:
    """Aggregate monthly OpenET ETa to annual and growing-season totals.

    Input columns: point_id + eta_{year}_{month:02d} (mm/month).
    Output columns: point_id, year, aet_annual, aet_gs, aet_n_months, aet_gs_n_months.
    Sums are NaN when all months are NaN (min_count=1).  aet_n_months tracks
    how many months contributed, so callers can filter partial-year records.
    """
    year_months: dict[int, dict[int, str]] = {}
    for col in df.columns:
        m = _ETA_BAND_RE.match(col)
        if m:
            year, month = int(m.group(1)), int(m.group(2))
            year_months.setdefault(year, {})[month] = col
    if not year_months:
        raise ValueError("No eta_YYYY_MM columns found in OpenET ETa CSV")

    gs_start, gs_end = gs_months
    frames = []
    for year, month_cols in sorted(year_months.items()):
        row = df[["point_id"]].copy()
        row["year"] = year

        all_cols = [month_cols[m] for m in sorted(month_cols)]
        gs_cols = [month_cols[m] for m in sorted(month_cols) if gs_start <= m <= gs_end]

        annual = df[all_cols]
        row["aet_n_months"] = annual.notna().sum(axis=1).astype("int16")
        row["aet_annual"] = annual.sum(axis=1, min_count=1).astype("float32")

        if gs_cols:
            gs = df[gs_cols]
            row["aet_gs_n_months"] = gs.notna().sum(axis=1).astype("int16")
            row["aet_gs"] = gs.sum(axis=1, min_count=1).astype("float32")
        else:
            row["aet_gs_n_months"] = np.int16(0)
            row["aet_gs"] = np.nan

        frames.append(row)

    return pd.concat(frames, ignore_index=True).reset_index(drop=True)


def extract_gridmet_history(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Reshape GridMET wide CSV to long format with annual/GS ETo and precip."""
    eto_cols = {
        int(m.group(1)): col
        for col in df.columns
        if (m := re.match(r"eto_(\d{4})$", col))
    }
    pr_cols = {
        int(m.group(1)): col
        for col in df.columns
        if (m := re.match(r"pr_(\d{4})$", col))
    }
    eto_gs_cols = {
        int(m.group(1)): col
        for col in df.columns
        if (m := re.match(r"eto_gs_(\d{4})$", col))
    }
    pr_gs_cols = {
        int(m.group(1)): col
        for col in df.columns
        if (m := re.match(r"pr_gs_(\d{4})$", col))
    }

    all_years = sorted(set(eto_cols) | set(pr_cols))
    if not all_years:
        raise ValueError("No eto_YYYY or pr_YYYY columns found in GridMET CSV")

    frames = []
    for year in all_years:
        row = df[["point_id"]].copy()
        row["year"] = year
        row["eto_annual"] = (
            df[eto_cols[year]].astype("float32") if year in eto_cols else np.nan
        )
        row["pr_annual"] = (
            df[pr_cols[year]].astype("float32") if year in pr_cols else np.nan
        )
        row["eto_gs"] = (
            df[eto_gs_cols[year]].astype("float32") if year in eto_gs_cols else np.nan
        )
        row["pr_gs"] = (
            df[pr_gs_cols[year]].astype("float32") if year in pr_gs_cols else np.nan
        )
        frames.append(row)

    return pd.concat(frames, ignore_index=True).reset_index(drop=True)


def build_point_year_table(
    irrmapper_csv: str | None,
    ndvi_csv: str | None,
    openet_eta_csv: str | None,
    point_ids: list[str],
    gridmet_csv: str | None = None,
) -> pd.DataFrame:
    """Build a point × year table by joining IrrMapper, NDVI, OpenET, and GridMET CSVs.

    Parameters
    ----------
    irrmapper_csv, ndvi_csv, openet_eta_csv : str or None
        Paths to downloaded EE CSV exports.
    point_ids : list[str]
        Authoritative list of point_id values.
    gridmet_csv : str or None
        Path to GridMET CSV with ETo and precipitation columns.

    Returns
    -------
    DataFrame with columns: point_id, year, irr_class, ndvi_*, aet_*, eto_*, pr_*,
    etf, net_et, and validity flags.
    """
    irr_long: pd.DataFrame | None = None
    ndvi_long: pd.DataFrame | None = None
    et_long: pd.DataFrame | None = None

    if irrmapper_csv and os.path.exists(irrmapper_csv):
        irr_raw = pd.read_csv(irrmapper_csv)
        irr_long = extract_irrmapper_history(irr_raw)
        LOGGER.info(
            "IrrMapper: %d point-years  years %d–%d",
            len(irr_long),
            irr_long["year"].min(),
            irr_long["year"].max(),
        )
    else:
        LOGGER.warning("IrrMapper CSV not found: %s", irrmapper_csv)

    if ndvi_csv and os.path.exists(ndvi_csv):
        ndvi_raw = pd.read_csv(ndvi_csv)
        ndvi_long = extract_ndvi_history(ndvi_raw)
        LOGGER.info(
            "NDVI: %d point-years  years %d–%d",
            len(ndvi_long),
            ndvi_long["year"].min(),
            ndvi_long["year"].max(),
        )
    else:
        LOGGER.warning("NDVI CSV not found: %s", ndvi_csv)

    if openet_eta_csv and os.path.exists(openet_eta_csv):
        eta_raw = pd.read_csv(openet_eta_csv)
        et_long = extract_et_history(eta_raw)
        LOGGER.info(
            "OpenET ETa: %d point-years  years %d–%d",
            len(et_long),
            et_long["year"].min(),
            et_long["year"].max(),
        )
    else:
        LOGGER.warning("OpenET ETa CSV not found: %s", openet_eta_csv)

    year_sets: list[set[int]] = []
    for long_df in (irr_long, ndvi_long, et_long):
        if long_df is not None:
            year_sets.append(set(long_df["year"].unique()))
    if not year_sets:
        raise FileNotFoundError(
            "No EE CSV files found — run EE exports (NB03 §7–8) and sync first."
        )

    all_years = sorted(set().union(*year_sets))
    base = pd.DataFrame(
        [(pid, yr) for pid in point_ids for yr in all_years],
        columns=["point_id", "year"],
    )
    LOGGER.info(
        "Base: %d points × %d years = %d rows",
        len(point_ids),
        len(all_years),
        len(base),
    )

    if irr_long is not None:
        base = base.merge(irr_long, on=["point_id", "year"], how="left")
    else:
        base["irr_class"] = np.nan

    if ndvi_long is not None:
        base = base.merge(ndvi_long, on=["point_id", "year"], how="left")

    if et_long is not None:
        base = base.merge(et_long, on=["point_id", "year"], how="left")

    # GridMET: ETo and precipitation
    gridmet_long: pd.DataFrame | None = None
    if gridmet_csv and os.path.exists(gridmet_csv):
        gridmet_raw = pd.read_csv(gridmet_csv)
        gridmet_long = extract_gridmet_history(gridmet_raw)
        LOGGER.info(
            "GridMET: %d point-years  years %d–%d",
            len(gridmet_long),
            gridmet_long["year"].min(),
            gridmet_long["year"].max(),
        )
        base = base.merge(gridmet_long, on=["point_id", "year"], how="left")
    else:
        if gridmet_csv:
            LOGGER.warning("GridMET CSV not found: %s", gridmet_csv)

    # Derived: ETf = AET / ETo, net_et = AET - precip
    if "aet_gs" in base.columns and "eto_gs" in base.columns:
        eto = base["eto_gs"].astype("float32")
        base["etf"] = np.where(eto > 0, base["aet_gs"] / eto, np.nan).astype("float32")
    if "aet_annual" in base.columns and "pr_annual" in base.columns:
        base["net_et"] = (base["aet_annual"] - base["pr_annual"]).astype("float32")

    base["irr_valid"] = base["irr_class"].notna()
    base["ndvi_valid"] = (
        base["ndvi_mean_gs"].notna() if "ndvi_mean_gs" in base.columns else False
    )
    base["et_valid"] = (
        base["aet_annual"].notna() if "aet_annual" in base.columns else False
    )
    base["feature_valid"] = base["irr_valid"] & base["ndvi_valid"] & base["et_valid"]

    return base.sort_values(["point_id", "year"]).reset_index(drop=True)


def write_point_year(df: pd.DataFrame, out_dir: str) -> str:
    """Write point_year to ``{out_dir}/point_year.parquet``."""
    ensure_dir(out_dir)
    path = os.path.join(out_dir, "point_year.parquet")
    df.to_parquet(path, index=False)
    LOGGER.info("Wrote point_year: %d rows → %s", len(df), path)
    return path


# ========================= EOF =======================================================================================
