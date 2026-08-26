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
from scipy.spatial import cKDTree
from shapely.geometry import Point

# National GWX well index (confinement-labeled). nwis/ngwmn are the *direct*
# USGS training source for Ma/Janssen and are excluded from the comparison set.
# NOTE: source != nwis does NOT by itself prove independence from Ma -- the Ma
# 2026 product also trained on Fan et al. wells, Jasechko CA/TX data, and ~20k
# stream-dummy cells. Treat the non-NWIS set as "non-NWIS" (defensible-but-
# unproven independence for Ma; stronger for Janssen, whose US real-obs source
# is USGS gwlevels), not as proven-independent, absent a training-set dedup.
GWX_INDEX = "/data/ssd2/gwx/products/current/wells.geoparquet"
WT_CLASSES = ("unconfined", "unconfined_marginal")
DEPTH_BANDS = ((0, 2), (2, 5), (5, 10), (10, 30), (30, 1e9))
# Horizontal distance from surface water — the regional-prior (R) test. Near
# surface water the channel bed *is* the answer (FAC-REM's shallow domain), so
# those bands are non-diagnostic; a regional water-table prior is judged on the
# FAR bands, where the table is decoupled from the local channel.
SW_DIST_BANDS = ((0, 500), (500, 2000), (2000, 5000), (5000, 10000), (10000, 1e9))


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
    states: set[str] | None = None,
) -> gpd.GeoDataFrame:
    """Read the GWX index (pandas, no WKB decode) and clip to the window.

    ``include_sources`` (if non-empty) restricts to those sources only and
    overrides ``exclude_sources`` -- used to score NWIS-only as a development /
    tuning set, kept disjoint from the independent non-NWIS comparison set.

    ``states`` (if non-empty) additionally restricts to the GWX ``state`` label.
    A raster window is a rectangle, so a state-scoped panel needs this on top of
    the bbox: the MT 10 m mosaic bbox, for instance, also spans parts of ID, WY,
    ND and SD, whose wells would otherwise enter a "Montana" panel.
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
        "canonical_id",
        "state",
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
    if states:
        keep &= df["state"].isin(states).to_numpy()
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


def surface_water_distance(
    wells: gpd.GeoDataFrame, sw_path: str, decim: int = 5
) -> np.ndarray:
    """Horizontal distance (m) from each well to the nearest permanent-surface-water cell.

    ``sw_path`` is a binary/occurrence raster where (value != nodata, default 0)
    marks water -- e.g. a JRC-GSW permanent-water mask. The full mask is read and
    block-pooled with ``any()`` (factor ``decim``, ~decim*native_res) so thin river
    lines survive decimation (nearest-neighbour reads would drop them), then a k-NN
    distance is taken. Wells must carry x5070/y5070 (set by ``load_window_wells``).
    """
    with rasterio.open(sw_path) as s:
        a = s.read(1)
        t, nod, crs = s.transform, s.nodata, s.crs
    water = a != (nod if nod is not None else 0)
    del a
    h0, w0 = water.shape
    h, w = h0 // decim, w0 // decim
    wp = water[: h * decim, : w * decim].reshape(h, decim, w, decim).any((1, 3))
    rows, cols = np.where(wp)
    if rows.size == 0:
        return np.full(len(wells), np.nan)
    xs = t.c + (cols * decim + decim / 2) * t.a
    ys = t.f + (rows * decim + decim / 2) * t.e
    if crs is not None and crs.to_epsg() != 5070:
        xs, ys = Transformer.from_crs(crs, "EPSG:5070", always_xy=True).transform(
            xs, ys
        )
    qxy = np.c_[wells["x5070"].to_numpy(), wells["y5070"].to_numpy()]
    return cKDTree(np.c_[xs, ys]).query(qxy)[0]


def nearest_distance(wells: gpd.GeoDataFrame, anchor_xy: np.ndarray) -> np.ndarray:
    """k-NN distance (m) from each well to the nearest anchor point.

    ``anchor_xy`` is an (N, 2) array of EPSG:5070 coordinates. Wells must carry
    x5070/y5070 (set by ``load_window_wells``). Empty anchors -> all-NaN.
    """
    anchor_xy = np.asarray(anchor_xy, dtype="float64")
    if anchor_xy.size == 0:
        return np.full(len(wells), np.nan)
    qxy = np.c_[wells["x5070"].to_numpy(), wells["y5070"].to_numpy()]
    return cKDTree(anchor_xy).query(qxy)[0]


SHALLOW_THRESHOLDS = (2.0, 5.0, 10.0)


def resid_stats(pred: np.ndarray, obs: np.ndarray) -> dict | None:
    """Central error + the bias/median split + RMSE + the catastrophic-tail spread.

    p90/p95 absolute-error percentiles and the >10 m / >25 m miss fractions expose
    the deep catastrophic tail that MAD suppresses (CLAUDE.md metric doctrine: the
    tail is where the 30+ m regime lives). Extra keys are additive -- existing
    callers that read mad/bias/median/rmse are unaffected.
    """
    r = pred - obs
    r = r[np.isfinite(r)]
    if r.size == 0:
        return None
    a = np.abs(r)
    return {
        "n": int(r.size),
        "mad_m": float(np.median(a)),
        "bias_m": float(np.mean(r)),
        "median_residual_m": float(np.median(r)),
        "rmse_m": float(np.sqrt(np.mean(r**2))),
        "p90_abs_err_m": float(np.percentile(a, 90)),
        "p95_abs_err_m": float(np.percentile(a, 95)),
        "frac_abs_err_gt_10m": float(np.mean(a > 10.0)),
        "frac_abs_err_gt_25m": float(np.mean(a > 25.0)),
    }


def shallow_skill(
    pred: np.ndarray, obs: np.ndarray, thresholds=SHALLOW_THRESHOLDS
) -> dict:
    """Precision/recall for the 'shallow water table' call at each threshold.

    Same definition as score_conus_gnn.shallow_skill so the shipped-raster panel
    and the leak-free OOF panel report the shallow-class skill identically: a
    positive call is pred < thr, ground truth is obs < thr.
    """
    out = {}
    m = np.isfinite(pred) & np.isfinite(obs)
    p, o = pred[m], obs[m]
    for thr in thresholds:
        pred_s, obs_s = p < thr, o < thr
        tp = int((pred_s & obs_s).sum())
        fp = int((pred_s & ~obs_s).sum())
        fn = int((~pred_s & obs_s).sum())
        prec = tp / (tp + fp) if (tp + fp) else float("nan")
        rec = tp / (tp + fn) if (tp + fn) else float("nan")
        out[f"<{thr:g}m"] = {
            "precision": prec,
            "recall": rec,
            "n_obs_shallow": int(obs_s.sum()),
            "n_pred_shallow": int(pred_s.sum()),
        }
    return out
