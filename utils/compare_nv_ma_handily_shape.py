"""Broad-shape comparison of handily and Ma over the 72-basin Nevada render.

Two parts, neither a headline accuracy claim (the headline eval is
``notes/NV_STATEWIDE_RASTER_EVAL.md``):

* Part A -- all attributed Nevada wells, handily ``gnn_dtw_100m`` and Ma sampled
  at the well, full metric panel stratified by class, confinement, network,
  training exposure and distance to the assimilated network.
* Part B -- raster to raster over the clipped 72-basin footprint, no wells:
  difference map, agreement tables, per-basin difference, surface-shape
  statistics (slope, roughness, semivariogram), hypsometric curves, difference
  against distance to the two well networks, and difference against sigma.

Sign convention (``notes/NV_STATEWIDE_RASTER_EVAL.md`` section 0):
``residual = predicted - observed`` (m), positive = predicted too deep.
The raster difference is ``handily - Ma`` (m), positive = handily deeper.

Reused repo logic, not re-implemented here:

* ``utils.sample_benchmark_rasters.sample_ma_tiles`` -- the Ma per-state tile
  sampler (first-finite across tiles, ``ma == 0`` stream clamps preserved,
  ``< -1 m`` and ``> 1e4 m`` dropped). Used verbatim for the well panel; its
  value-screen rules are applied identically to the gridded warp.
* ``utils.probe_mae_embeddings.paired_bootstrap_mad_delta`` -- the paired
  block-bootstrap pattern, extended here to carry RMSE alongside MAD.
* ``figures.nwi.nwi_common`` -- ``rendered_basins``, ``use_style``, ``save``,
  ``scale_bar``, ``north_arrow``, the Okabe-Ito palette and the depth bands.

Run::

    uv run --directory /home/dgketchum/code/handily python \
        /home/dgketchum/code/handily/utils/compare_nv_ma_handily_shape.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window
from scipy.spatial import cKDTree

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "utils"))
sys.path.insert(0, str(_REPO / "figures" / "nwi"))

from nwi_common import (  # noqa: E402
    HUC8_POLYS,
    MM,
    WIDTH_2COL,
    outside_letter,
    north_arrow,
    rendered_basins,
    save,
    scale_bar,
    use_style,
)
from sample_benchmark_rasters import sample_ma_tiles  # noqa: E402

RENDER_ARM = "gnn_conus_monitoring_water_src_r1e"
HUC8_ROOT = Path("/data/ssd2/handily/huc8")
MA_DIR = Path("/nas/gwx/wtd_states")
DEM_100M = "/data/ssd2/handily/conus/covariates/elev48i0100a.tif"
WELLS_FGB = "/data/ssd2/handily/nv/regional/wells/nv_wells_attributed.fgb"
ADMITTED = "/data/ssd2/handily/nv/regional/wells/ndwr_admitted_sources_frozen.parquet"
BUNDLE_QN = "/data/ssd2/handily/conus/wte_gnn/graph_conus_monitoring_water_v2/query_nodes.parquet"
OUT_DIR = "/data/ssd2/handily/nv/regional/shape_compare"

CELL = 100.0
NODATA = -9999.0
CELL_KM2 = (CELL / 1000.0) ** 2  # 0.01 km2 per 100 m cell

#: Observed-depth reporting bands (m), the doctrine bands.
DEPTH_BANDS = ((0.0, 2.0), (2.0, 5.0), (5.0, 10.0), (10.0, 30.0), (30.0, np.inf))
DEPTH_LABELS = ("0-2 m", "2-5 m", "5-10 m", "10-30 m", "30+ m")
SHALLOW_THRESHOLDS = (2.0, 5.0, 10.0)
#: Raster depth bands. The render carries a small population of negative DTW
#: (water table above ground, section 2.3 of the eval note), so the shallowest
#: raster band is open at the bottom and the negative count is reported apart.
RASTER_BANDS = ((-np.inf, 2.0), (2.0, 5.0), (5.0, 10.0), (10.0, 30.0), (30.0, np.inf))
RASTER_LABELS = ("<2 m", "2-5 m", "5-10 m", "10-30 m", "30+ m")
#: Distance-to-network bands (km).
DIST_EDGES = (0.0, 0.05, 0.3, 1.0, 5.0, np.inf)
DIST_LABELS = ("<0.05 km", "0.05-0.3 km", "0.3-1 km", "1-5 km", ">=5 km")

OKABE = {
    "handily": "#0072B2",
    "ma": "#E69F00",
    "accent": "#D55E00",
    "grey": "#666666",
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Metric panel (definitions from NV_STATEWIDE_RASTER_EVAL.md section 0)
# ---------------------------------------------------------------------------


def panel_stats(obs: np.ndarray, pred: np.ndarray) -> dict:
    """MAD / median residual / mean bias / RMSE / p95 |residual|, all metres.

    ``residual = pred - obs``; MAD is ``median(|r|)``, bias is ``mean(r)``.
    """
    r = pred - obs
    return {
        "n": int(r.size),
        "MAD_m": float(np.median(np.abs(r))) if r.size else np.nan,
        "median_resid_m": float(np.median(r)) if r.size else np.nan,
        "bias_m": float(np.mean(r)) if r.size else np.nan,
        "RMSE_m": float(np.sqrt(np.mean(r**2))) if r.size else np.nan,
        "p95_abs_resid_m": float(np.percentile(np.abs(r), 95)) if r.size else np.nan,
    }


def stratum_rows(
    obs: np.ndarray,
    preds: dict[str, np.ndarray],
    mask: np.ndarray,
    label: str,
    seen_by: str,
    min_n: int = 30,
) -> list[dict]:
    """One metric-panel row per predictor on ``mask``; ``seen_by`` labels exposure."""
    rows = []
    n = int(mask.sum())
    for name, p in preds.items():
        row = {"stratum": label, "seen_by": seen_by, "predictor": name}
        row.update(
            panel_stats(obs[mask], p[mask])
            if n >= min_n
            else {
                "n": n,
                "MAD_m": np.nan,
                "median_resid_m": np.nan,
                "bias_m": np.nan,
                "RMSE_m": np.nan,
                "p95_abs_resid_m": np.nan,
            }
        )
        rows.append(row)
    return rows


def shallow_rows(obs: np.ndarray, preds: dict[str, np.ndarray]) -> list[dict]:
    """Shallow-class precision / recall / F1 (dimensionless) at each threshold."""
    rows = []
    for thr in SHALLOW_THRESHOLDS:
        obs_s = obs < thr
        for name, p in preds.items():
            pred_s = p < thr
            tp = int((obs_s & pred_s).sum())
            prec = tp / pred_s.sum() if pred_s.sum() else np.nan
            rec = tp / obs_s.sum() if obs_s.sum() else np.nan
            f1 = (
                2 * prec * rec / (prec + rec)
                if np.isfinite(prec) and np.isfinite(rec) and (prec + rec) > 0
                else np.nan
            )
            rows.append(
                {
                    "threshold_m": thr,
                    "predictor": name,
                    "n_obs_shallow": int(obs_s.sum()),
                    "n_pred_shallow": int(pred_s.sum()),
                    "precision": prec,
                    "recall": rec,
                    "f1": f1,
                }
            )
    return rows


def paired_block_bootstrap(
    obs: np.ndarray,
    pred_h: np.ndarray,
    pred_m: np.ndarray,
    blocks: np.ndarray,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict:
    """Delta = Ma - handily in MAD and RMSE (m), paired block bootstrap CI95.

    Adapted from ``utils.probe_mae_embeddings.paired_bootstrap_mad_delta``
    (same resampling scheme: blocks drawn with replacement, every site in a
    drawn block entering together, the same resampled site set used for both
    predictors). The CI bounds the difference in the named statistic between
    the two predictors on this well set; it is not a CI on either predictor's
    own error. Positive = handily better.
    """
    rng = np.random.default_rng(seed)
    ah, am = np.abs(pred_h - obs), np.abs(pred_m - obs)
    ug = np.unique(blocks)
    gidx = {g: np.where(blocks == g)[0] for g in ug}
    d_mad = np.empty(n_boot)
    d_rmse = np.empty(n_boot)
    for i in range(n_boot):
        samp = np.concatenate([gidx[g] for g in rng.choice(ug, len(ug), replace=True)])
        d_mad[i] = np.median(am[samp]) - np.median(ah[samp])
        d_rmse[i] = np.sqrt(np.mean(am[samp] ** 2)) - np.sqrt(np.mean(ah[samp] ** 2))
    return {
        "n": int(obs.size),
        "n_blocks": int(len(ug)),
        "delta_mad_m": float(np.median(am) - np.median(ah)),
        "delta_mad_lo_m": float(np.percentile(d_mad, 2.5)),
        "delta_mad_hi_m": float(np.percentile(d_mad, 97.5)),
        "delta_rmse_m": float(np.sqrt(np.mean(am**2)) - np.sqrt(np.mean(ah**2))),
        "delta_rmse_lo_m": float(np.percentile(d_rmse, 2.5)),
        "delta_rmse_hi_m": float(np.percentile(d_rmse, 97.5)),
    }


def deming(x: np.ndarray, y: np.ndarray, lam: float = 1.0) -> dict:
    """Orthogonal (Deming) regression of y on x with error-variance ratio ``lam``.

    ``lam = var(err_y) / var(err_x) = 1`` treats both axes as equally noisy, so
    the fit minimises perpendicular distance. Slope is dimensionless when both
    axes are metres; intercept is metres. Pearson r and Spearman rho are
    dimensionless (1 = perfect).
    """
    from scipy.stats import spearmanr

    xm, ym = x.mean(), y.mean()
    sxx = np.mean((x - xm) ** 2)
    syy = np.mean((y - ym) ** 2)
    sxy = np.mean((x - xm) * (y - ym))
    num = syy - lam * sxx + np.sqrt((syy - lam * sxx) ** 2 + 4 * lam * sxy**2)
    slope = float(num / (2 * sxy))
    return {
        "n": int(x.size),
        "mean_x_m": float(xm),
        "mean_y_m": float(ym),
        "sd_x_m": float(np.sqrt(sxx)),
        "sd_y_m": float(np.sqrt(syy)),
        "sd_ratio_y_over_x": float(np.sqrt(syy / sxx)),
        "pearson_r": float(np.corrcoef(x, y)[0, 1]),
        "spearman_rho": float(spearmanr(x, y).statistic),
        "deming_slope": slope,
        "deming_intercept_m": float(ym - slope * xm),
    }


# ---------------------------------------------------------------------------
# Mosaic build
# ---------------------------------------------------------------------------


def build_mosaic(basins: list[str], layers: tuple[str, ...]) -> dict:
    """Clip each basin render to its HUC8 polygon and paste into one 100 m grid.

    Returns the mosaic arrays (NaN outside), the grid transform, the per-cell
    basin index, and bookkeeping on the clip (pre-clip valid cells, cells that
    the clip dropped, and the overlap check).
    """
    paths = {b: HUC8_ROOT / b / "gnn" / RENDER_ARM for b in basins}
    bounds = []
    for b in basins:
        with rasterio.open(paths[b] / "gnn_dtw_100m.tif") as ds:
            if ds.transform.a != CELL or ds.transform.e != -CELL:
                raise ValueError(f"{b}: not a 100 m north-up grid")
            if ds.crs.to_epsg() != 5070:
                raise ValueError(f"{b}: CRS is {ds.crs}, expected EPSG:5070")
            bounds.append(ds.bounds)
    left = min(b.left for b in bounds)
    right = max(b.right for b in bounds)
    bottom = min(b.bottom for b in bounds)
    top = max(b.top for b in bounds)
    if (left % CELL) or (top % CELL):
        raise ValueError("mosaic origin is not on the 100 m grid")
    nrow = int(round((top - bottom) / CELL))
    ncol = int(round((right - left) / CELL))
    transform = from_origin(left, top, CELL, CELL)
    log(f"mosaic grid {nrow} x {ncol} cells, origin ({left:.0f}, {top:.0f})")

    hucs = gpd.read_parquet(HUC8_POLYS, columns=["huc8", "name", "geometry"])
    hucs["huc8"] = hucs["huc8"].astype(str).str.zfill(8)
    hucs = hucs[hucs["huc8"].isin(basins)].set_index("huc8")
    if len(hucs) != len(basins):
        raise ValueError(f"{len(hucs)} HUC8 polygons for {len(basins)} basins")
    if hucs.crs.to_epsg() != 5070:
        raise ValueError("HUC8 polygons are not EPSG:5070")

    out = {k: np.full((nrow, ncol), np.nan, np.float32) for k in layers}
    basin_idx = np.full((nrow, ncol), -1, np.int16)
    hits = np.zeros((nrow, ncol), np.uint8)  # pre-clip valid-cell count
    n_preclip = 0
    n_dropped_by_clip = 0
    n_overlap = 0

    for k, b in enumerate(basins):
        with rasterio.open(paths[b] / "gnn_dtw_100m.tif") as ds:
            bb = ds.bounds
            r0 = int(round((top - bb.top) / CELL))
            c0 = int(round((bb.left - left) / CELL))
            h, w = ds.height, ds.width
            btr = ds.transform
            dtw = ds.read(1).astype(np.float32)
        dtw[dtw == NODATA] = np.nan
        pre = np.isfinite(dtw)
        n_preclip += int(pre.sum())
        hits[r0 : r0 + h, c0 : c0 + w] += pre.astype(np.uint8)

        inside = (
            rasterize(
                [(hucs.loc[b, "geometry"], 1)],
                out_shape=(h, w),
                transform=btr,
                fill=0,
                dtype="uint8",
            )
            == 1
        )
        keep = pre & inside
        n_dropped_by_clip += int((pre & ~inside).sum())
        sl = (slice(r0, r0 + h), slice(c0, c0 + w))
        n_overlap += int((keep & (basin_idx[sl] >= 0)).sum())
        basin_idx[sl] = np.where(keep, k, basin_idx[sl])

        for name in layers:
            fn = {"dtw": "gnn_dtw_100m", "wte": "gnn_wte_100m"}.get(
                name, f"gnn_{name}_100m"
            )
            if name == "dtw":
                arr = dtw
            else:
                with rasterio.open(paths[b] / f"{fn}.tif") as ds:
                    arr = ds.read(1).astype(np.float32)
                arr[arr == NODATA] = np.nan
            block = out[name][sl]
            out[name][sl] = np.where(keep, arr, block)
        if (k + 1) % 12 == 0:
            log(f"  mosaicked {k + 1}/{len(basins)} basins")

    if n_overlap:
        raise ValueError(f"{n_overlap} overlapping valid cells after HUC8 clipping")
    valid = basin_idx >= 0
    # A cell can be valid in a basin's rectangle but lie in a neighbour's HUC8;
    # that cell survives only if the owning basin also rendered it.
    meta = {
        "nrow": nrow,
        "ncol": ncol,
        "left": left,
        "top": top,
        "n_preclip_valid_cells": n_preclip,
        "n_cells_dropped_by_clip": n_dropped_by_clip,
        "n_clipped_valid_cells": int(valid.sum()),
        "max_preclip_hits_per_cell": int(hits.max()),
        "n_cells_multi_basin_preclip": int((hits > 1).sum()),
        "overlap_after_clip": n_overlap,
    }
    return {
        "arrays": out,
        "transform": transform,
        "basin_idx": basin_idx,
        "valid": valid,
        "hits": hits,
        "hucs": hucs,
        "meta": meta,
    }


def ma_on_grid(transform, nrow: int, ncol: int, tiles: list[Path]) -> np.ndarray:
    """Ma DTW (m) warped to the handily 100 m grid, nearest neighbour.

    Nearest, not bilinear: it reproduces exactly what
    ``sample_benchmark_rasters.sample_ma_tiles`` does at a point (the 24.14 m
    source cell containing the query location), so the gridded field and the
    well-level panel come from one convention. First-finite across tiles, as in
    the repo sampler; the same value screen (``< -1 m`` or ``> 1e4 m`` -> NaN,
    ``ma == 0`` stream clamps kept) is applied afterwards.
    """
    out = np.full((nrow, ncol), np.nan, np.float32)
    for path in tiles:
        with rasterio.open(path) as src:
            with WarpedVRT(
                src,
                crs="EPSG:5070",
                transform=transform,
                width=ncol,
                height=nrow,
                resampling=Resampling.nearest,
                src_nodata=src.nodata,
                nodata=np.nan,
                dtype="float32",
            ) as vrt:
                arr = vrt.read(1)
        need = ~np.isfinite(out)
        take = need & np.isfinite(arr)
        out[take] = arr[take]
        log(f"  Ma tile {path.name}: filled {int(take.sum())} cells")
    bad = (out < -1.0) | (out > 1e4)
    n_bad = int(bad.sum())
    out[bad] = np.nan
    log(f"  Ma value screen dropped {n_bad} cells (< -1 m or > 1e4 m)")
    return out


def ma_tiles_for(bounds: tuple[float, float, float, float]) -> list[Path]:
    """Per-state Ma tiles whose footprint intersects the mosaic bounds."""
    from rasterio.warp import transform_bounds

    left, bottom, right, top = bounds
    hit = []
    for path in sorted(MA_DIR.glob("wtd_*.tif")):
        with rasterio.open(path) as ds:
            tb = transform_bounds(ds.crs, "EPSG:5070", *ds.bounds)
        if tb[0] < right and tb[2] > left and tb[1] < top and tb[3] > bottom:
            hit.append(path)
    return hit


# ---------------------------------------------------------------------------
# Well-level sampling
# ---------------------------------------------------------------------------


def sample_handily_per_basin(
    basins: list[str], x: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Nearest-cell handily DTW at (x, y) over all 72 unclipped basin rasters.

    Mirrors the eval note's sampling: read each basin raster, take the value at
    the containing cell, and count how many basins return a valid value so the
    "valid in exactly one basin" property can be asserted rather than assumed.
    """
    val = np.full(x.size, np.nan)
    hits = np.zeros(x.size, np.int16)
    for b in basins:
        p = HUC8_ROOT / b / "gnn" / RENDER_ARM / "gnn_dtw_100m.tif"
        with rasterio.open(p) as ds:
            t = ds.transform
            arr = ds.read(1)
            h, w = arr.shape
        col = np.floor((x - t.c) / t.a).astype(np.int64)
        row = np.floor((y - t.f) / t.e).astype(np.int64)
        ok = (row >= 0) & (row < h) & (col >= 0) & (col < w)
        v = np.full(x.size, np.nan)
        v[ok] = arr[row[ok], col[ok]]
        v[v == NODATA] = np.nan
        fin = np.isfinite(v)
        hits += fin
        val[fin] = v[fin]
    return val, hits


def nearest_km(tree: cKDTree, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Euclidean distance (km, EPSG:5070) to the nearest point in ``tree``."""
    d, _ = tree.query(np.column_stack([x, y]), k=1, workers=-1)
    return d / 1000.0


# ---------------------------------------------------------------------------
# Part A
# ---------------------------------------------------------------------------


def part_a(
    out_dir: Path,
    basins: list[str],
    grid: dict,
    ma_grid: np.ndarray,
    n_boot: int = 2000,
) -> dict:
    log("Part A: attributed-well comparison")
    g = gpd.read_file(WELLS_FGB)
    if g.crs.to_epsg() != 5070:
        raise ValueError(f"wells CRS is {g.crs}, expected EPSG:5070")
    x = g.geometry.x.to_numpy("float64")
    y = g.geometry.y.to_numpy("float64")
    log(f"  wells loaded: {len(g)}")

    obs = g["obs_dtw_m"].to_numpy("float64")
    n_obs_missing = int((~np.isfinite(obs)).sum())
    if n_obs_missing:
        sub = g.loc[~np.isfinite(obs), ["source", "well_class", "obs_count"]]
        log(
            f"  {n_obs_missing} wells carry no observed DTW "
            f"(sources {sub['source'].value_counts().to_dict()}, "
            f"obs_count {sub['obs_count'].value_counts().to_dict()}); "
            "these are attributed records with no water level, excluded from "
            "every residual panel"
        )

    hand, hits = sample_handily_per_basin(basins, x, y)
    if hits.max() > 1:
        raise ValueError(
            f"{int((hits > 1).sum())} wells valid in more than one basin raster"
        )
    log(
        f"  handily raster: {int(np.isfinite(hand).sum())} wells valid in exactly "
        f"one basin, {int((hits == 0).sum())} outside the render footprint"
    )

    ma_native = sample_ma_tiles(x, y, MA_DIR)
    log(f"  Ma native tiles: {int(np.isfinite(ma_native).sum())} wells covered")

    tr = grid["transform"]
    col = np.floor((x - tr.c) / tr.a).astype(np.int64)
    row = np.floor((y - tr.f) / tr.e).astype(np.int64)
    inb = (row >= 0) & (row < ma_grid.shape[0]) & (col >= 0) & (col < ma_grid.shape[1])
    ma_res = np.full(x.size, np.nan)
    ma_res[inb] = ma_grid[row[inb], col[inb]]

    adm = pd.read_parquet(ADMITTED, columns=["x5070", "y5070"])
    adm_tree = cKDTree(adm[["x5070", "y5070"]].to_numpy("float64"))
    d_adm = nearest_km(adm_tree, x, y)

    qn = pd.read_parquet(BUNDLE_QN, columns=["well_class", "x5070", "y5070", "is_nwis"])
    train = qn[qn["well_class"] == "monitoring"]
    d_train = nearest_km(cKDTree(train[["x5070", "y5070"]].to_numpy("float64")), x, y)

    both = np.isfinite(obs) & np.isfinite(hand) & np.isfinite(ma_native)
    log(
        f"  common footprint: {int(both.sum())} wells with finite observation and "
        f"finite value from both products"
    )
    agree = np.isfinite(ma_res) & np.isfinite(ma_native)
    d_ma = np.abs(ma_res[agree] - ma_native[agree])
    log(
        "  Ma native vs Ma-on-100m-grid at wells: "
        f"median |diff| {np.median(d_ma):.4f} m, p99 {np.percentile(d_ma, 99):.3f} m, "
        f"frac > 1 m {np.mean(d_ma > 1.0):.4f}"
    )
    stored = g["ma_dtw_m"].to_numpy("float64")
    ok = np.isfinite(stored) & np.isfinite(ma_native)
    d_st = np.abs(stored[ok] - ma_native[ok])
    log(
        "  Ma native vs stored ma_dtw_m column: "
        f"{np.mean(d_st < 1e-6):.4f} bit-identical, frac > 1 m {np.mean(d_st > 1):.4f}"
    )

    o = obs[both]
    preds = {"handily": hand[both], "Ma": ma_native[both]}
    sub = g.loc[both].reset_index(drop=True)
    d_adm_s, d_train_s = d_adm[both], d_train[both]
    blocks = np.array(
        [basins[i] for i in grid["basin_idx"][row[both], col[both]]], dtype=object
    )
    if (grid["basin_idx"][row[both], col[both]] < 0).any():
        # A well can be valid in a basin rectangle but sit outside every HUC8
        # polygon (the render windows are rectangles). Those wells have no
        # blocking basin; they are kept in the panels and blocked on the basin
        # whose raster supplied the value.
        need = grid["basin_idx"][row[both], col[both]] < 0
        fill = np.full(int(need.sum()), "", dtype=object)
        for b in basins:
            p = HUC8_ROOT / b / "gnn" / RENDER_ARM / "gnn_dtw_100m.tif"
            with rasterio.open(p) as ds:
                t, arr = ds.transform, ds.read(1)
            c2 = np.floor((x[both][need] - t.c) / t.a).astype(np.int64)
            r2 = np.floor((y[both][need] - t.f) / t.e).astype(np.int64)
            okb = (r2 >= 0) & (r2 < arr.shape[0]) & (c2 >= 0) & (c2 < arr.shape[1])
            v = np.full(okb.size, np.nan)
            v[okb] = arr[r2[okb], c2[okb]]
            fill[np.isfinite(v) & (v != NODATA)] = b
        blocks[need] = fill
        log(
            f"  {int(need.sum())} scored wells fall outside every HUC8 polygon "
            "(inside a render rectangle only); blocked on the supplying basin"
        )

    rows: list[dict] = []
    rows += stratum_rows(o, preds, np.ones(o.size, bool), "all wells", "mixed")
    # Ma read off the 100 m grid instead of its native 24.14 m tiles. Part B has
    # no choice but to use the gridded field, so the same row is carried here to
    # price the resampling: it is the only difference between the two Ma rows.
    grid_ok = np.isfinite(ma_res[both])
    rows += stratum_rows(
        o[grid_ok],
        {"Ma (100 m grid)": ma_res[both][grid_ok]},
        np.ones(int(grid_ok.sum()), bool),
        "all wells",
        "mixed",
    )

    band_rows = []
    for (lo, hi), lab in zip(DEPTH_BANDS, DEPTH_LABELS):
        m = (o >= lo) & (o < hi)
        band_rows += stratum_rows(o, preds, m, lab, "mixed")
        band_rows += stratum_rows(
            o[grid_ok],
            {"Ma (100 m grid)": ma_res[both][grid_ok]},
            (o[grid_ok] >= lo) & (o[grid_ok] < hi),
            lab,
            "mixed",
        )

    exposure = np.full(o.size, "none of these", dtype=object)
    is_adm = d_adm_s <= 0.001  # within 1 m of an admitted-source coordinate
    is_train = sub["in_handily_training"].to_numpy(bool)
    is_nwis = sub["is_nwis"].to_numpy(bool)
    exposure[is_nwis] = "NWIS (Ma training exposure)"
    exposure[is_train] = "handily training well"
    exposure[is_adm] = "admitted DA source"
    exp_seen = {
        "admitted DA source": "handily (assimilated at inference); Ma only if also NWIS",
        "handily training well": "handily (bundle training label)",
        "NWIS (Ma training exposure)": "Ma (NWIS is Ma's training network)",
        "none of these": "neither model",
    }
    overlap = pd.DataFrame(
        {
            "n_admitted": [int(is_adm.sum())],
            "n_handily_training": [int(is_train.sum())],
            "n_nwis": [int(is_nwis.sum())],
            "n_admitted_and_nwis": [int((is_adm & is_nwis).sum())],
            "n_admitted_and_training": [int((is_adm & is_train).sum())],
            "n_training_and_nwis": [int((is_train & is_nwis).sum())],
        }
    )
    overlap.to_csv(out_dir / "partA_exposure_overlap.csv", index=False)

    strat_rows = []
    for col_name, seen in (
        ("well_class", "mixed"),
        ("confinement_class", "mixed"),
        ("source", None),
    ):
        vals = sub[col_name].astype(str)
        for v in sorted(vals.unique()):
            m = (vals == v).to_numpy()
            lab = f"{col_name} = {v}"
            s = seen or (
                "Ma (NWIS is Ma's training network)"
                if v == "nwis"
                else "neither model by network membership"
            )
            strat_rows += stratum_rows(o, preds, m, lab, s)

    exp_rows = []
    for v in (
        "admitted DA source",
        "handily training well",
        "NWIS (Ma training exposure)",
        "none of these",
    ):
        m = exposure == v
        exp_rows += stratum_rows(o, preds, m, v, exp_seen[v])

    dist_rows = []
    for name, dv in (
        ("dist to nearest admitted source", d_adm_s),
        ("dist to nearest bundle training well", d_train_s),
    ):
        for i, lab in enumerate(DIST_LABELS):
            m = (dv >= DIST_EDGES[i]) & (dv < DIST_EDGES[i + 1])
            seen = (
                "handily (pinned at the source)"
                if (name.endswith("source") and i == 0)
                else "handily exposure decays with distance"
            )
            dist_rows += stratum_rows(o, preds, m, f"{name}: {lab}", seen)

    pd.DataFrame(rows).to_csv(out_dir / "partA_core_panel.csv", index=False)
    pd.DataFrame(band_rows).to_csv(out_dir / "partA_by_depth_band.csv", index=False)
    pd.DataFrame(strat_rows).to_csv(out_dir / "partA_by_class.csv", index=False)
    pd.DataFrame(exp_rows).to_csv(out_dir / "partA_by_exposure.csv", index=False)
    pd.DataFrame(dist_rows).to_csv(out_dir / "partA_by_distance.csv", index=False)
    pd.DataFrame(shallow_rows(o, preds)).to_csv(
        out_dir / "partA_shallow_class.csv", index=False
    )

    boot = [
        {
            "row": "all wells",
            **paired_block_bootstrap(
                o, preds["handily"], preds["Ma"], blocks, n_boot=n_boot
            ),
        }
    ]
    for v in (
        "admitted DA source",
        "handily training well",
        "NWIS (Ma training exposure)",
        "none of these",
    ):
        m = exposure == v
        if m.sum() < 30:
            continue
        boot.append(
            {
                "row": v,
                **paired_block_bootstrap(
                    o[m], preds["handily"][m], preds["Ma"][m], blocks[m], n_boot=n_boot
                ),
            }
        )
    pd.DataFrame(boot).to_csv(out_dir / "partA_bootstrap.csv", index=False)

    reg = [
        {"y": "handily DTW", "x": "Ma DTW", **deming(preds["Ma"], preds["handily"])},
        {"y": "handily DTW", "x": "observed DTW", **deming(o, preds["handily"])},
        {"y": "Ma DTW", "x": "observed DTW", **deming(o, preds["Ma"])},
    ]
    pd.DataFrame(reg).to_csv(out_dir / "partA_regressions.csv", index=False)

    return {
        "n_wells": int(len(g)),
        "n_no_observation": n_obs_missing,
        "n_outside_render": int((hits == 0).sum()),
        "n_common_footprint": int(both.sum()),
        "core": rows,
        "bootstrap": boot,
        "regressions": reg,
        "shallow": shallow_rows(o, preds),
        "ma_grid_vs_native_median_m": float(np.median(d_ma)),
        "ma_grid_vs_native_p99_m": float(np.percentile(d_ma, 99)),
    }


# ---------------------------------------------------------------------------
# Part B
# ---------------------------------------------------------------------------


def surface_shape(z: np.ndarray, name: str) -> dict:
    """Median / p90 gradient magnitude (m/km) and Laplacian roughness (m)."""
    gy, gx = np.gradient(z, CELL, CELL)
    mag = np.hypot(gx, gy) * 1000.0  # m per m -> m per km
    fin = np.isfinite(mag)
    lap = (
        np.roll(z, 1, 0)
        + np.roll(z, -1, 0)
        + np.roll(z, 1, 1)
        + np.roll(z, -1, 1)
        - 4 * z
    )
    lap[0, :] = lap[-1, :] = np.nan
    lap[:, 0] = lap[:, -1] = np.nan
    lf = np.isfinite(lap)
    return {
        "surface": name,
        "n_cells_slope": int(fin.sum()),
        "slope_median_m_per_km": float(np.median(mag[fin])),
        "slope_p90_m_per_km": float(np.percentile(mag[fin], 90)),
        "n_cells_laplacian": int(lf.sum()),
        "roughness_std_laplacian_m": float(np.std(lap[lf])),
    }


def semivariogram(
    z: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    lags_km: tuple[float, ...],
    n_pairs: int,
    rng: np.random.Generator,
) -> list[dict]:
    """Mean squared difference / 2 (m2) at each lag from random cell pairs."""
    nr, nc = z.shape
    out = []
    for lag in lags_km:
        h_cells = lag * 1000.0 / CELL
        got_d2: list[np.ndarray] = []
        got_sep: list[np.ndarray] = []
        have = 0
        for _ in range(40):
            if have >= n_pairs:
                break
            k = int(min(4_000_000, max(n_pairs * 2, 200_000)))
            i = rng.integers(0, rows.size, k)
            th = rng.uniform(0.0, 2 * np.pi, k)
            dr = np.rint(h_cells * np.sin(th)).astype(np.int64)
            dc = np.rint(h_cells * np.cos(th)).astype(np.int64)
            r2 = rows[i] + dr
            c2 = cols[i] + dc
            ok = (r2 >= 0) & (r2 < nr) & (c2 >= 0) & (c2 < nc)
            a = z[rows[i][ok], cols[i][ok]]
            b = z[r2[ok], c2[ok]]
            good = np.isfinite(a) & np.isfinite(b)
            got_d2.append((a[good] - b[good]) ** 2)
            got_sep.append(np.hypot(dr[ok][good], dc[ok][good]) * CELL / 1000.0)
            have += int(good.sum())
        d2 = np.concatenate(got_d2)[:n_pairs]
        sep = np.concatenate(got_sep)[:n_pairs]
        out.append(
            {
                "lag_km": lag,
                "n_pairs": int(d2.size),
                "mean_actual_separation_km": float(sep.mean()),
                "semivariance_m2": float(d2.mean() / 2.0),
            }
        )
    return out


def band_index(v: np.ndarray) -> np.ndarray:
    """Raster depth-band index (0..4) for a DTW array; -1 where not finite."""
    idx = np.full(v.shape, -1, np.int8)
    for k, (lo, hi) in enumerate(RASTER_BANDS):
        idx[np.isfinite(v) & (v >= lo) & (v < hi)] = k
    return idx


def part_b(
    out_dir: Path,
    basins: list[str],
    grid: dict,
    ma_grid: np.ndarray,
    args,
) -> dict:
    log("Part B: raster-to-raster comparison")
    hand = grid["arrays"]["dtw"]
    wte_h = grid["arrays"]["wte"]
    sigma = grid["arrays"]["sigma"]
    spread = grid["arrays"]["fold_spread"]
    valid_h = grid["valid"]
    common = valid_h & np.isfinite(ma_grid)
    area_h = float(valid_h.sum()) * CELL_KM2
    area_ma = float((np.isfinite(ma_grid) & valid_h).sum()) * CELL_KM2
    area_common = float(common.sum()) * CELL_KM2
    log(
        f"  handily footprint {area_h:,.0f} km2; Ma-covered {area_ma:,.0f} km2; "
        f"common {area_common:,.0f} km2"
    )

    # The renderer writes dtw = z_surf - wte, so the DEM substrate it used is
    # recoverable exactly from the two shipped layers. Cross-check against the
    # 100 m DEM constant that infer_conus_gnn.py defaults to.
    z_surf = hand + wte_h
    tr = grid["transform"]
    rr, cc = np.nonzero(common)
    take = np.random.default_rng(0).choice(
        rr.size, size=min(200_000, rr.size), replace=False
    )
    xs = tr.c + (cc[take] + 0.5) * CELL
    ys = tr.f - (rr[take] + 0.5) * CELL
    with rasterio.open(DEM_100M) as ds:
        dt = ds.transform
        dcol = np.floor((xs - dt.c) / dt.a).astype(np.int64)
        drow = np.floor((ys - dt.f) / dt.e).astype(np.int64)
        dem_v = ds.read(
            1,
            window=Window(
                int(dcol.min()),
                int(drow.min()),
                int(dcol.max() - dcol.min() + 1),
                int(drow.max() - drow.min() + 1),
            ),
        ).astype("float64")
        nd = ds.nodata
    samp = dem_v[drow - drow.min(), dcol - dcol.min()]
    samp[samp == nd] = np.nan
    zs = z_surf[rr[take], cc[take]]
    dz = np.abs(zs - samp)
    dem_check = {
        "n": int(np.isfinite(dz).sum()),
        "median_abs_diff_m": float(np.nanmedian(dz)),
        "p99_abs_diff_m": float(np.nanpercentile(dz, 99)),
        "max_abs_diff_m": float(np.nanmax(dz)),
    }
    log(f"  z_surf (= DTW + WTE) vs {DEM_100M}: {dem_check}")

    diff = np.where(common, hand - ma_grid, np.nan).astype(np.float32)
    d = diff[common]

    # (1) difference GeoTIFF
    tif = out_dir / "handily_minus_ma_dtw_100m.tif"
    with rasterio.open(
        tif,
        "w",
        driver="GTiff",
        height=diff.shape[0],
        width=diff.shape[1],
        count=1,
        dtype="float32",
        crs="EPSG:5070",
        transform=tr,
        nodata=NODATA,
        compress="deflate",
        predictor=2,
        tiled=True,
        blockxsize=512,
        blockysize=512,
    ) as ds:
        ds.write(np.where(np.isfinite(diff), diff, NODATA).astype("float32"), 1)
        ds.set_band_description(1, "handily DTW minus Ma DTW (m)")
    log(f"  wrote {tif}")

    # (2) histogram / quantiles
    qs = {f"p{q}": float(np.percentile(d, q)) for q in (5, 25, 50, 75, 95)}
    within = {
        f"frac_within_{t}m": float(np.mean(np.abs(d) <= t)) for t in (1, 2, 5, 10)
    }
    hist = {
        "n_cells": int(d.size),
        "area_km2": area_common,
        "mean_m": float(d.mean()),
        **qs,
        **within,
        **{
            f"area_within_{t}m_km2": within[f"frac_within_{t}m"] * area_common
            for t in (1, 2, 5, 10)
        },
    }
    hist["n_cells_handily_negative_dtw"] = int((hand[common] < 0).sum())
    hist["n_cells_ma_negative_dtw"] = int((ma_grid[common] < 0).sum())
    hist["n_cells_ma_zero_stream_clamp"] = int((ma_grid[common] == 0).sum())
    pd.DataFrame([hist]).to_csv(out_dir / "partB_difference_quantiles.csv", index=False)

    # (3) agreement tables
    bh = band_index(np.where(common, hand, np.nan))
    bm = band_index(np.where(common, ma_grid, np.nan))
    cross = np.zeros((5, 5), np.int64)
    for i in range(5):
        mi = bh == i
        if not mi.any():
            continue
        cross[i] = np.bincount(bm[mi][bm[mi] >= 0], minlength=5)
    agree_h = pd.DataFrame(
        cross * CELL_KM2, index=list(RASTER_LABELS), columns=list(RASTER_LABELS)
    )
    agree_h.index.name = "handily band (row) / Ma band (col), km2"
    agree_h.to_csv(out_dir / "partB_agreement_rows_handily_km2.csv")
    (100 * agree_h.div(agree_h.sum(axis=1), axis=0)).to_csv(
        out_dir / "partB_agreement_rows_handily_pct.csv"
    )
    agree_m = agree_h.T.copy()
    agree_m.index.name = "Ma band (row) / handily band (col), km2"
    agree_m.to_csv(out_dir / "partB_agreement_rows_ma_km2.csv")
    (100 * agree_m.div(agree_m.sum(axis=1), axis=0)).to_csv(
        out_dir / "partB_agreement_rows_ma_pct.csv"
    )

    # (4) per-basin
    bidx = grid["basin_idx"]
    rowsb = []
    for k, b in enumerate(basins):
        m = common & (bidx == k)
        n = int(m.sum())
        if n == 0:
            rowsb.append({"basin": b, "n_cells": 0})
            continue
        dv = diff[m]
        rowsb.append(
            {
                "basin": b,
                "n_cells": n,
                "area_km2": n * CELL_KM2,
                "median_diff_m": float(np.median(dv)),
                "iqr_diff_m": float(np.percentile(dv, 75) - np.percentile(dv, 25)),
                "p25_diff_m": float(np.percentile(dv, 25)),
                "p75_diff_m": float(np.percentile(dv, 75)),
                "median_handily_dtw_m": float(np.median(hand[m])),
                "median_ma_dtw_m": float(np.median(ma_grid[m])),
            }
        )
    per_basin = pd.DataFrame(rowsb).sort_values("median_diff_m")
    per_basin.to_csv(out_dir / "partB_per_basin_difference.csv", index=False)

    # (5) shape statistics of each WTE surface
    wte_ma = np.where(common, z_surf - ma_grid, np.nan).astype(np.float32)
    wte_h_c = np.where(common, wte_h, np.nan).astype(np.float32)
    shape = [
        surface_shape(wte_h_c, "handily WTE (gnn_wte_100m)"),
        surface_shape(wte_ma, "Ma WTE (z_surf - Ma DTW)"),
        surface_shape(
            np.where(common, z_surf, np.nan).astype(np.float32), "land surface z_surf"
        ),
    ]
    pd.DataFrame(shape).to_csv(out_dir / "partB_surface_shape.csv", index=False)

    rng = np.random.default_rng(1)
    rws, cls = np.nonzero(common)
    sv = []
    for nm, arr in (
        ("handily WTE", wte_h_c),
        ("Ma WTE", wte_ma),
    ):
        for r in semivariogram(arr, rws, cls, args.lags_km, args.n_pairs, rng):
            sv.append({"surface": nm, **r})
    pd.DataFrame(sv).to_csv(out_dir / "partB_semivariogram.csv", index=False)

    # (6) hypsometric-style curves
    xs_curve = np.concatenate(
        [np.arange(0, 30, 0.25), np.arange(30, 100, 1.0), np.arange(100, 305, 5.0)]
    )
    hv = hand[common]
    mv = ma_grid[common]
    hs = np.sort(hv)
    ms = np.sort(mv)
    curve = pd.DataFrame(
        {
            "dtw_m": xs_curve,
            "handily_frac_area_shallower": np.searchsorted(hs, xs_curve) / hs.size,
            "ma_frac_area_shallower": np.searchsorted(ms, xs_curve) / ms.size,
        }
    )
    curve.to_csv(out_dir / "partB_hypsometric_curve.csv", index=False)
    shallow_area = []
    for t in (2.0, 5.0, 10.0, 30.0):
        for nm, v in (("handily", hv), ("Ma", mv)):
            f = float(np.mean(v < t))
            shallow_area.append(
                {
                    "threshold_m": t,
                    "product": nm,
                    "area_km2": f * area_common,
                    "pct_of_common_footprint": 100 * f,
                }
            )
    pd.DataFrame(shallow_area).to_csv(out_dir / "partB_shallow_area.csv", index=False)

    # (7) difference vs distance to the two networks
    adm = pd.read_parquet(ADMITTED, columns=["x5070", "y5070"])
    adm_tree = cKDTree(adm[["x5070", "y5070"]].to_numpy("float64"))
    nwis_xy = nwis_points(tr, common.shape)
    nwis_tree = cKDTree(nwis_xy)
    cx = tr.c + (cls + 0.5) * CELL
    cy = tr.f - (rws + 0.5) * CELL
    dist_rows = []
    for nm, tree in (("admitted source", adm_tree), ("NWIS well", nwis_tree)):
        dkm = np.empty(cx.size)
        step = 4_000_000
        for s in range(0, cx.size, step):
            dkm[s : s + step] = nearest_km(tree, cx[s : s + step], cy[s : s + step])
        for i, lab in enumerate(DIST_LABELS):
            m = (dkm >= DIST_EDGES[i]) & (dkm < DIST_EDGES[i + 1])
            n = int(m.sum())
            if n == 0:
                dist_rows.append({"network": nm, "band": lab, "n_cells": 0})
                continue
            dv = d[m]
            dist_rows.append(
                {
                    "network": nm,
                    "band": lab,
                    "n_cells": n,
                    "area_km2": n * CELL_KM2,
                    "median_diff_m": float(np.median(dv)),
                    "p25_diff_m": float(np.percentile(dv, 25)),
                    "p75_diff_m": float(np.percentile(dv, 75)),
                    "iqr_diff_m": float(np.percentile(dv, 75) - np.percentile(dv, 25)),
                    "median_abs_diff_m": float(np.median(np.abs(dv))),
                }
            )
    dist_tab = pd.DataFrame(dist_rows)
    dist_tab.to_csv(out_dir / "partB_diff_vs_distance.csv", index=False)

    # (8) sigma and fold spread against |difference|
    quint_rows = []
    for nm, layer in (("sigma", sigma), ("fold spread", spread)):
        v = layer[common]
        fin = np.isfinite(v)
        if fin.sum() == 0:
            raise ValueError(f"{nm} has no finite values on the common footprint")
        edges = np.percentile(v[fin], [0, 20, 40, 60, 80, 100])
        for q in range(5):
            m = (
                fin
                & (v >= edges[q])
                & (v <= edges[q + 1] if q == 4 else v < edges[q + 1])
            )
            quint_rows.append(
                {
                    "layer": nm,
                    "quintile": f"Q{q + 1}",
                    "range_m": f"{edges[q]:.2f}-{edges[q + 1]:.2f}",
                    "n_cells": int(m.sum()),
                    "median_layer_m": float(np.median(v[m])),
                    "median_abs_diff_m": float(np.median(np.abs(d[m]))),
                    "median_diff_m": float(np.median(d[m])),
                }
            )
    pd.DataFrame(quint_rows).to_csv(out_dir / "partB_sigma_quintiles.csv", index=False)

    figures(out_dir, grid, diff, common, curve, dist_tab, hist, d)

    return {
        "area_handily_km2": area_h,
        "area_common_km2": area_common,
        "dem_check": dem_check,
        "hist": hist,
        "shape": shape,
        "semivariogram": sv,
        "per_basin": per_basin,
        "shallow_area": shallow_area,
        "quintiles": quint_rows,
        "agreement_km2": agree_h,
        "dist": dist_tab,
    }


def nwis_points(transform, shape) -> np.ndarray:
    """NWIS well coordinates (EPSG:5070) covering the render footprint.

    The Nevada attributed file stops at the state well population, but the 72
    basins straddle five state lines, so the CONUS bundle's NWIS monitoring
    wells are unioned in to keep the fringe distances honest.
    """
    g = gpd.read_file(WELLS_FGB)
    nv = g.loc[g["is_nwis"].astype(bool)]
    a = np.column_stack([nv.geometry.x.to_numpy(), nv.geometry.y.to_numpy()])
    qn = pd.read_parquet(BUNDLE_QN, columns=["is_nwis", "x5070", "y5070"])
    qn = qn[qn["is_nwis"].astype(bool)]
    b = qn[["x5070", "y5070"]].to_numpy("float64")
    left = transform.c
    top = transform.f
    right = left + shape[1] * CELL
    bottom = top - shape[0] * CELL
    pad = 100_000.0
    keep = (
        (b[:, 0] > left - pad)
        & (b[:, 0] < right + pad)
        & (b[:, 1] > bottom - pad)
        & (b[:, 1] < top + pad)
    )
    xy = np.vstack([a, b[keep]])
    log(f"  NWIS reference set: {len(a)} NV + {int(keep.sum())} bundle = {len(xy)}")
    return xy


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def block_reduce(a: np.ndarray, f: int) -> np.ndarray:
    """Mean over f x f blocks, ignoring NaN (trims the ragged edge).

    Blocks with no valid cell (outside the render footprint, which is most of
    the bounding rectangle) return NaN by construction rather than by catching
    a warning: the sum and the count are formed explicitly.
    """
    nr = a.shape[0] // f * f
    nc = a.shape[1] // f * f
    b = a[:nr, :nc].reshape(nr // f, f, nc // f, f)
    fin = np.isfinite(b)
    cnt = fin.sum(axis=(1, 3))
    tot = np.where(fin, b, 0.0).sum(axis=(1, 3))
    return np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)


def figures(out_dir, grid, diff, common, curve, dist_tab, hist, d) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    use_style()
    tr = grid["transform"]
    f = 5
    small = block_reduce(diff.astype("float64"), f)
    ext = (
        tr.c,
        tr.c + diff.shape[1] * CELL,
        tr.f - diff.shape[0] * CELL,
        tr.f,
    )
    # Symmetric limits at the 90th percentile of |difference| so the bulk of the
    # field is legible; the colorbar is extended so the tail is not hidden.
    lim = float(np.percentile(np.abs(d), 90))

    fig, ax = plt.subplots(figsize=(WIDTH_2COL, WIDTH_2COL * 1.25))
    im = ax.imshow(
        small,
        extent=ext,
        origin="upper",
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim),
        interpolation="nearest",
    )
    grid["hucs"].boundary.plot(ax=ax, color="black", linewidth=0.25, zorder=4)
    ax.set_xlabel("Easting, EPSG:5070 (m)")
    ax.set_ylabel("Northing, EPSG:5070 (m)")
    ax.set_aspect("equal")
    cb = fig.colorbar(im, ax=ax, shrink=0.45, pad=0.02, extend="both")
    cb.set_label("handily DTW − Ma DTW (m)")
    scale_bar(ax, length_m=200_000.0)
    north_arrow(ax)
    save(fig, "fig_difference_map", out_dir)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(WIDTH_2COL / 2, 55 * MM))
    counts, edges = np.histogram(np.clip(d, -60, 60), bins=241, range=(-60, 60))
    ax.bar(
        0.5 * (edges[:-1] + edges[1:]),
        counts / 1e6,
        width=np.diff(edges),
        color=OKABE["grey"],
        linewidth=0,
    )
    ax.set_xlim(-60, 60)
    ax.set_ylim(0, 1.15 * float(counts[1:-1].max()) / 1e6)
    ax.axvline(0.0, color="black", linewidth=0.6)
    ytop = ax.get_ylim()[1]
    for q, lab in ((5, "p5"), (25, "p25"), (50, "p50"), (75, "p75"), (95, "p95")):
        v = hist[f"p{q}"]
        ax.axvline(v, color=OKABE["accent"], linewidth=0.6, linestyle=(0, (2.5, 1.5)))
        ax.text(
            v,
            0.98 * ytop,
            f"{lab} {v:+.1f}",
            rotation=90,
            ha="right",
            va="top",
        )
    ax.text(
        0.58,
        0.62,
        "outermost bins hold\nevery cell beyond ±60 m",
        transform=ax.transAxes,
        ha="left",
        va="top",
    )
    ax.set_xlabel("handily DTW − Ma DTW (m)")
    ax.set_ylabel("Footprint cells (millions of 100 m cells)")
    save(fig, "fig_difference_histogram", out_dir)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(WIDTH_2COL / 2, 55 * MM))
    ax.plot(
        curve["dtw_m"],
        100 * curve["handily_frac_area_shallower"],
        color=OKABE["handily"],
        label="handily",
    )
    ax.plot(
        curve["dtw_m"],
        100 * curve["ma_frac_area_shallower"],
        color=OKABE["ma"],
        linestyle=(0, (2.5, 1.5)),
        label="Ma",
    )
    ax.set_xscale("log")
    ax.set_xlim(0.5, 300)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Depth to water, x (m)")
    ax.set_ylabel("Common footprint with DTW < x (%)")
    ax.legend(loc="lower right")
    save(fig, "fig_hypsometric", out_dir)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(WIDTH_2COL, 58 * MM), sharey=True)
    for ax, net, letter in zip(axes, ("admitted source", "NWIS well"), "ab"):
        sub = dist_tab[(dist_tab["network"] == net) & (dist_tab["n_cells"] > 0)]
        xpos = np.arange(len(sub))
        ax.errorbar(
            xpos,
            sub["median_diff_m"],
            yerr=[
                sub["median_diff_m"] - sub["p25_diff_m"],
                sub["p75_diff_m"] - sub["median_diff_m"],
            ],
            fmt="o",
            color="black",
            markersize=3.0,
            linewidth=0.8,
        )
        ax.axhline(0.0, color=OKABE["grey"], linewidth=0.6)
        ax.set_xticks(xpos)
        ax.set_xticklabels(sub["band"], rotation=30, ha="right")
        ax.set_xlabel(f"Distance to nearest {net}")
        outside_letter(ax, letter)
    axes[0].set_ylabel("handily DTW − Ma DTW (m), median and IQR")
    save(fig, "fig_diff_vs_distance", out_dir)
    plt.close(fig)


# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--n-pairs", type=int, default=200_000)
    ap.add_argument(
        "--lags-km",
        type=float,
        nargs="+",
        default=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
    )
    ap.add_argument("--skip-part-a", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    basins = rendered_basins()
    log(f"rendered basins: {len(basins)}")

    grid = build_mosaic(basins, ("dtw", "wte", "sigma", "fold_spread"))
    log(f"  clip bookkeeping: {grid['meta']}")

    tr = grid["transform"]
    nrow, ncol = grid["meta"]["nrow"], grid["meta"]["ncol"]
    bounds = (tr.c, tr.f - nrow * CELL, tr.c + ncol * CELL, tr.f)
    tiles = ma_tiles_for(bounds)
    log(f"  Ma tiles intersecting the footprint: {[t.stem for t in tiles]}")
    ma_grid = ma_on_grid(tr, nrow, ncol, tiles)

    summary = {"basins": len(basins), "mosaic": grid["meta"]}
    if not args.skip_part_a:
        summary["part_a"] = {
            k: v
            for k, v in part_a(
                out_dir, basins, grid, ma_grid, n_boot=args.n_boot
            ).items()
            if k not in ("core", "bootstrap", "regressions", "shallow")
        }
    b = part_b(out_dir, basins, grid, ma_grid, args)
    summary["part_b"] = {
        k: v for k, v in b.items() if k not in ("per_basin", "agreement_km2", "dist")
    }
    with open(out_dir / "run_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    log(f"wrote outputs to {out_dir}")


if __name__ == "__main__":
    main()
