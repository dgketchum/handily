"""Shared GWX-well validation primitives.

Loader, raster sampler, setting tagger, and residual-stats used by both the FAC
validator (``validate_fac_gwx_wells.py``) and the regional-prior validator
(``validate_regional_prior_gwx_wells.py``). Kept in one place so the
independent-well definition (unconfined, NWIS-excluded) and the residual metric
stay identical across predictors.
"""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from shapely.geometry import Point

# National GWX well index (confinement-labeled). nwis/ngwmn are Ma/Janssen
# training sources -> excluded from the independent comparison set.
GWX_INDEX = "/data/ssd2/gwx/products/current/wells.geoparquet"
WT_CLASSES = ("unconfined", "unconfined_marginal")
DEPTH_BANDS = ((0, 2), (2, 5), (5, 10), (10, 30), (30, 1e9))


def sample_raster(path: str, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Sample band 1 at WGS84 points; out-of-bounds / nodata / huge -> NaN."""
    with rasterio.open(path) as src:
        tr = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
        xs, ys = tr.transform(lon, lat)
        b = src.bounds
        vals = np.array(
            [v[0] for v in src.sample(np.c_[xs, ys], indexes=1)], dtype="float64"
        )
        nod = src.nodata
    vals[(xs < b.left) | (xs > b.right) | (ys < b.bottom) | (ys > b.top)] = np.nan
    if nod is not None and np.isfinite(nod):
        vals[vals == nod] = np.nan
    vals[np.abs(vals) > 1e29] = np.nan
    return vals


def load_window_wells(
    index_path: str,
    bbox_5070: tuple[float, float, float, float],
    confinement: tuple[str, ...],
    exclude_sources: set[str],
    include_sources: set[str] | None = None,
) -> gpd.GeoDataFrame:
    """Read the GWX index (pandas, no WKB decode) and clip to the window.

    ``include_sources`` (if non-empty) restricts to those sources only and
    overrides ``exclude_sources`` -- used to score NWIS-only as a development /
    tuning set, kept disjoint from the independent non-NWIS comparison set.
    """
    cols = [
        "source",
        "longitude",
        "latitude",
        "mean_dtw",
        "confinement_class",
        "confinement_source",
        "well_class",
        "well_use",
        "obs_count",
        "is_active",
    ]
    df = pd.read_parquet(index_path, columns=cols)
    tr = Transformer.from_crs("EPSG:4326", "EPSG:5070", always_xy=True)
    x, y = tr.transform(df["longitude"].to_numpy(), df["latitude"].to_numpy())
    left, bottom, right, top = bbox_5070
    if include_sources:
        source_keep = df["source"].isin(include_sources).to_numpy()
    else:
        source_keep = ~df["source"].isin(exclude_sources).to_numpy()
    keep = (
        (x >= left)
        & (x <= right)
        & (y >= bottom)
        & (y <= top)
        & df["confinement_class"].isin(confinement).to_numpy()
        & df["mean_dtw"].notna().to_numpy()
        & source_keep
    )
    sub = df.loc[keep].copy()
    sub["x5070"], sub["y5070"] = x[keep], y[keep]
    gdf = gpd.GeoDataFrame(
        sub,
        geometry=[Point(xy) for xy in zip(sub["x5070"], sub["y5070"])],
        crs="EPSG:5070",
    )
    return gdf


def tag_setting(
    wells: gpd.GeoDataFrame, streams_path: str, dist_m: float
) -> tuple[np.ndarray, np.ndarray]:
    streams = gpd.read_file(streams_path).to_crs(5070)
    near = gpd.sjoin_nearest(
        wells[["geometry"]], streams[["geometry"]], distance_col="_d"
    )
    near = near[~near.index.duplicated(keep="first")]
    d = near["_d"].reindex(wells.index).to_numpy()
    return np.where(d <= dist_m, "valley", "upland"), d


def resid_stats(pred: np.ndarray, obs: np.ndarray) -> dict | None:
    r = pred - obs
    r = r[np.isfinite(r)]
    if r.size == 0:
        return None
    return {
        "n": int(r.size),
        "mad_m": float(np.median(np.abs(r))),
        "bias_m": float(np.mean(r)),
        "median_residual_m": float(np.median(r)),
        "rmse_m": float(np.sqrt(np.mean(r**2))),
    }
