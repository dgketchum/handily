"""E2 — regrid the acquired aquifer-geometry products to EPSG:5070 covariate-bank
derivatives (see notes/E2_AQUIFER_GEOMETRY.md task (a)).

Three complementary DEEP-basin geometry layers, none of which the existing bank
covers (its depth-to-bedrock / sediment-thickness are shallow soil/regolith
depths — Pelletier capped at 50 m — shown saturated as point covariates in
COVARIATE_ACQUISITION.md 2026-06-30):

  1. depth_to_basement_grav_gb_m.tif  — Great Basin GRAVITY depth-to-pre-Cenozoic
     basement (Glen et al. 2022, DOI 10.5066/P9Z6SA1Z; from Shah & Boyd 2018).
     Source is a custom NAD83 Albers (CM -117), metres; reprojected to EPSG:5070.
  2. basin_fill_thickness_br_m.tif    — Basin & Range basin-fill aquifer thickness
     = (top - bottom) altitude (Stanton 2015, DOI 10.5066/P9IGLDY0). Source ASCII
     grids are already EPSG:5070 with vertical units FEET (NAVD88); thickness is
     datum-independent -> metres after /3.28084, clipped to [0, 5000] m.
  3. hp_base_of_aquifer_alt_m.tif      — High Plains base-of-aquifer ELEVATION
     (Cederstrand & Becker 1998 OFR 98-393, DOI 10.5066/P9UALJH4). Source is E00
     contours (feet); vertices interpolated (linear Delaunay) to a 1 km EPSG:5070
     grid clipped to the contour hull. metres above sea level.

All outputs: EPSG:5070, 1 km, float32, nodata -9999, snapped to the 1 km grid that
is coincident with the bank's 100 m canonical origin (-2540000, 3258000).

Usage:
    uv run python utils/v02_build_geometry_layers.py
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

log = logging.getLogger("v02_build_geometry_layers")

DL = "/nas/handily/covariates/_download/e2_geometry"
GEOL = "/nas/handily/covariates/geology"
FT_PER_M = 3.28084
NODATA = -9999.0
RES = 1000.0
# canonical bank grid origin (EPSG:5070); a 1 km grid snapped to it stays bank-aligned
ORIGIN_X, ORIGIN_Y = -2540000.0, 3258000.0
THICK_CLIP_MAX_M = 5000.0  # physical cap; a few edge cells reach ~11 km (artifacts)


# --------------------------------------------------------------------------- #
# pure math (unit-testable)
# --------------------------------------------------------------------------- #
def feet_to_m(a: np.ndarray) -> np.ndarray:
    """Convert feet to metres (NaN propagates)."""
    return np.asarray(a, float) / FT_PER_M


def thickness_from_top_bot(
    top_ft: np.ndarray, bot_ft: np.ndarray, clip_max_m: float = THICK_CLIP_MAX_M
) -> np.ndarray:
    """Basin-fill thickness in METRES from top/bottom altitudes in feet.

    Datum-independent (feet cancel in the difference). Negatives (top below
    bottom, edge artifacts) clip to 0; values above ``clip_max_m`` (a handful of
    edge cells) clip to ``clip_max_m``. NaN where either input is NaN.
    """
    t = np.asarray(top_ft, float)
    b = np.asarray(bot_ft, float)
    thk_m = (t - b) / FT_PER_M
    thk_m = np.clip(thk_m, 0.0, clip_max_m)
    thk_m[~np.isfinite(t) | ~np.isfinite(b)] = np.nan
    return thk_m


def snap_bounds(left, bottom, right, top, res=RES, ox=ORIGIN_X, oy=ORIGIN_Y):
    """Snap a bbox OUTWARD to the ``res`` grid aligned with origin (ox, oy).

    Returns (left, bottom, right, top) so that (left-ox)/res and (top-oy)/res are
    integers and the snapped box contains the input box.
    """
    left = ox + np.floor((left - ox) / res) * res
    right = ox + np.ceil((right - ox) / res) * res
    top = oy + np.ceil((top - oy) / res) * res
    bottom = oy + np.floor((bottom - oy) / res) * res
    return float(left), float(bottom), float(right), float(top)


def interp_contours(xy: np.ndarray, z: np.ndarray, xs: np.ndarray, ys: np.ndarray):
    """Linear (Delaunay) interpolation of scattered contour vertices to a grid.

    ``xy`` (n,2) vertices, ``z`` (n,) values, ``xs`` (W,) cell-centre x ascending,
    ``ys`` (H,) cell-centre y DESCENDING (north-up raster row order). Returns a
    (H, W) array; cells outside the vertex convex hull are NaN.
    """
    from scipy.interpolate import griddata

    gx, gy = np.meshgrid(np.asarray(xs, float), np.asarray(ys, float))
    out = griddata(
        np.asarray(xy, float), np.asarray(z, float), (gx, gy), method="linear"
    )
    return out


# --------------------------------------------------------------------------- #
# raster IO helpers
# --------------------------------------------------------------------------- #
def _write_5070(path: Path, arr: np.ndarray, transform) -> None:
    import rasterio
    from rasterio.crs import CRS

    arr = np.asarray(arr, np.float32)
    out = np.where(np.isfinite(arr), arr, NODATA).astype(np.float32)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=out.shape[0],
        width=out.shape[1],
        count=1,
        dtype="float32",
        crs=CRS.from_epsg(5070),
        transform=transform,
        nodata=NODATA,
        compress="deflate",
        tiled=True,
    ) as ds:
        ds.write(out, 1)
    log.info("wrote %s (%dx%d)", path, out.shape[1], out.shape[0])


def build_gb_depth_to_basement(dl: Path, geol: Path) -> dict:
    """Reproject the Great Basin gravity depth-to-basement to EPSG:5070 1 km."""
    import rasterio
    from rasterio.crs import CRS
    from rasterio.warp import Resampling, reproject, transform_bounds
    from rasterio.transform import from_origin

    src_path = dl / "ex_gb/GB_geophysics_grids/GB_depth_to_basement_surface.tif"
    with rasterio.open(src_path) as src:
        dst_crs = CRS.from_epsg(5070)
        # native bounds -> 5070 bounds, then snap to bank-aligned 1 km grid
        x0, y0, x1, y1 = transform_bounds(src.crs, dst_crs, *src.bounds)
        x0, y0, x1, y1 = snap_bounds(x0, y0, x1, y1)
        w = int(round((x1 - x0) / RES))
        h = int(round((y1 - y0) / RES))
        dst_transform = from_origin(x0, y1, RES, RES)
        dst = np.full((h, w), NODATA, np.float32)
        reproject(
            source=rasterio.band(src, 1),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            src_nodata=src.nodata,
            dst_nodata=NODATA,
            resampling=Resampling.bilinear,
        )
    dst = np.where(dst == NODATA, np.nan, dst)
    _write_5070(geol / "depth_to_basement_grav_gb_m.tif", dst, dst_transform)
    v = dst[np.isfinite(dst)]
    return {
        "layer": "depth_to_basement_grav_gb_m",
        "units": "m (depth below surface, gravity-derived)",
        "n_valid": int(v.size),
        "min": float(np.min(v)),
        "median": float(np.median(v)),
        "max": float(np.max(v)),
    }


def build_gb_iso_grav(dl: Path, geol: Path) -> dict:
    """Reproject the Great Basin isostatic gravity anomaly to EPSG:5070 1 km."""
    import rasterio
    from rasterio.crs import CRS
    from rasterio.warp import Resampling, reproject, transform_bounds
    from rasterio.transform import from_origin

    src_path = dl / "ex_gb/GB_geophysics_grids/GB_iso_grav_anom.tif"
    with rasterio.open(src_path) as src:
        dst_crs = CRS.from_epsg(5070)
        x0, y0, x1, y1 = transform_bounds(src.crs, dst_crs, *src.bounds)
        x0, y0, x1, y1 = snap_bounds(x0, y0, x1, y1)
        w = int(round((x1 - x0) / RES))
        h = int(round((y1 - y0) / RES))
        dst_transform = from_origin(x0, y1, RES, RES)
        dst = np.full((h, w), NODATA, np.float32)
        reproject(
            source=rasterio.band(src, 1),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            src_nodata=src.nodata,
            dst_nodata=NODATA,
            resampling=Resampling.bilinear,
        )
    dst = np.where(dst == NODATA, np.nan, dst)
    _write_5070(geol / "iso_grav_anom_gb_mgal.tif", dst, dst_transform)
    v = dst[np.isfinite(dst)]
    return {
        "layer": "iso_grav_anom_gb_mgal",
        "units": "mGal (isostatic gravity anomaly)",
        "n_valid": int(v.size),
        "min": float(np.min(v)),
        "median": float(np.median(v)),
        "max": float(np.max(v)),
    }


def build_br_thickness(dl: Path, geol: Path) -> dict:
    """Basin & Range basin-fill thickness (m) from feet top/bottom ASCII grids."""
    import rasterio

    d = dl / "ex_br/ds01BSNRGB_non_prop/01BSNRGB_ASCII_files"
    with rasterio.open(d / "r01bsnrgb_top_a.txt") as ds:
        top = ds.read(1, masked=True).filled(np.nan)
        tr = ds.transform
    with rasterio.open(d / "r01bsnrgb_bot_a.txt") as ds:
        bot = ds.read(1, masked=True).filled(np.nan)
    thk_m = thickness_from_top_bot(top, bot)
    # source ASCII grid is already on the EPSG:5070 1 km grid (CM -96 Albers);
    # keep its transform, just tag CRS on write.
    _write_5070(geol / "basin_fill_thickness_br_m.tif", thk_m, tr)
    v = thk_m[np.isfinite(thk_m)]
    return {
        "layer": "basin_fill_thickness_br_m",
        "units": "m (basin-fill aquifer thickness, top-bottom)",
        "n_valid": int(v.size),
        "min": float(np.min(v)),
        "median": float(np.median(v)),
        "max": float(np.max(v)),
    }


def build_hp_base_alt(dl: Path, geol: Path) -> dict:
    """High Plains base-of-aquifer altitude (m) interpolated from E00 contours."""
    import warnings

    import geopandas as gpd
    from rasterio.transform import from_origin

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        g = gpd.read_file(dl / "ex_hp/ofr98-393.e00", layer="ARC")
    # explode every contour vertex with its ELEV (feet -> m)
    xs_all, ys_all, zs_all = [], [], []
    for geom, elev_ft in zip(g.geometry.values, g["ELEV"].to_numpy(float)):
        if geom is None or not np.isfinite(elev_ft):
            continue
        for gx, gy in np.asarray(geom.coords):
            xs_all.append(gx)
            ys_all.append(gy)
            zs_all.append(elev_ft)
    xy = np.column_stack([xs_all, ys_all])
    z_m = feet_to_m(np.asarray(zs_all))

    x0, y0, x1, y1 = snap_bounds(
        xy[:, 0].min(), xy[:, 1].min(), xy[:, 0].max(), xy[:, 1].max()
    )
    w = int(round((x1 - x0) / RES))
    h = int(round((y1 - y0) / RES))
    xs = x0 + (np.arange(w) + 0.5) * RES
    ys = y1 - (np.arange(h) + 0.5) * RES  # descending (north-up rows)
    grid = interp_contours(xy, z_m, xs, ys)
    dst_transform = from_origin(x0, y1, RES, RES)
    _write_5070(geol / "hp_base_of_aquifer_alt_m.tif", grid, dst_transform)
    v = grid[np.isfinite(grid)]
    return {
        "layer": "hp_base_of_aquifer_alt_m",
        "units": "m above sea level (base-of-aquifer elevation, interpolated)",
        "n_vertices": int(len(z_m)),
        "n_valid_cells": int(v.size),
        "min": float(np.min(v)),
        "median": float(np.median(v)),
        "max": float(np.max(v)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--download-dir", default=DL)
    ap.add_argument("--geol-dir", default=GEOL)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    dl, geol = Path(args.download_dir), Path(args.geol_dir)
    geol.mkdir(parents=True, exist_ok=True)

    for fn in (
        build_gb_depth_to_basement,
        build_gb_iso_grav,
        build_br_thickness,
        build_hp_base_alt,
    ):
        info = fn(dl, geol)
        log.info("built %s", info)


if __name__ == "__main__":
    main()
