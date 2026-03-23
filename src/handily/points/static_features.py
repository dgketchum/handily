"""Static hydro-topographic feature extraction for sampled points."""

import logging
import os

import geopandas as gpd
import numpy as np
import pandas as pd

from handily.io import ensure_dir
from handily.nhd import classify_flowlines

LOGGER = logging.getLogger("handily.points.static_features")

REM_BINS = [0.0, 1.0, 2.0, 5.0, 10.0, np.inf]
REM_LABELS = ["0-1 m", "1-2 m", "2-5 m", "5-10 m", ">10 m"]

STREAM_CATEGORIES = ["perennial", "intermittent", "managed"]


def compute_hydro_distances(
    points: gpd.GeoDataFrame,
    flowlines: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Add dist_perennial_m, dist_intermittent_m, dist_managed_m to points.

    Runs one sjoin_nearest per stream category so each column reflects
    the true nearest stream of that type, not just the globally nearest.
    """
    flowlines = classify_flowlines(flowlines).to_crs(points.crs)
    out = points.copy()

    for category in STREAM_CATEGORIES:
        col = f"dist_{category}_m"
        subset = flowlines[flowlines["stream_category"] == category]
        if subset.empty:
            LOGGER.warning("No %s flowlines found; %s set to inf", category, col)
            out[col] = np.inf
            continue

        join = gpd.sjoin_nearest(
            out[["point_id", "geometry"]],
            subset[["geometry"]],
            how="left",
            distance_col=col,
        )
        join = (
            join.sort_values("point_id")
            .drop_duplicates(subset=["point_id"], keep="first")
            .reset_index(drop=True)
        )
        out[col] = join[col].fillna(np.inf).to_numpy()
        LOGGER.info(
            "Computed %s: min=%.1f m, max=%.1f m, n=%d",
            col,
            float(out[col].replace(np.inf, np.nan).min()),
            float(out[col].replace(np.inf, np.nan).max()),
            (out[col] < np.inf).sum(),
        )

    return out


def assign_context_flags(points: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Add rem_bin string column derived from rem_at_sample.

    Uses pd.cut with right=False so bins are [0,1), [1,2), [2,5), [5,10), [10,inf).
    Result is cast to str for FlatGeobuf compatibility (CategoricalDtype not supported).
    Points with NaN rem_at_sample get rem_bin='nan'.
    """
    out = points.copy()
    out["rem_bin"] = pd.cut(
        out["rem_at_sample"],
        bins=REM_BINS,
        labels=REM_LABELS,
        right=False,
    ).astype(str)
    return out


def extract_static_point_features(
    points: gpd.GeoDataFrame,
    flowlines: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Compute all static features and return augmented points GeoDataFrame."""
    out = compute_hydro_distances(points, flowlines)
    out = assign_context_flags(out)
    return out


def write_static_features(points: gpd.GeoDataFrame, out_dir: str) -> dict[str, str]:
    """Write points_static.fgb and points_static.parquet to out_dir."""
    ensure_dir(out_dir)
    fgb_path = os.path.join(out_dir, "points_static.fgb")
    parquet_path = os.path.join(out_dir, "points_static.parquet")
    points.to_file(fgb_path, driver="FlatGeobuf")
    points.to_parquet(parquet_path)
    LOGGER.info("Wrote %d points to %s", len(points), fgb_path)
    return {"fgb": fgb_path, "parquet": parquet_path}
