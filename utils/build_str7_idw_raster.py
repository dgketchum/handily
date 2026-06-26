#!/usr/bin/env python
"""Rasterize the well-free Strahler-IDW WTE / DTW prior at full basin (HUC8) scale.

The four-prior comparison only ever sampled the Strahler-IDW prior AT WELLS. This
writes the continuous surface for QGIS inspection. Anchor selection is PER-BASIN
top-N orders by default (`strahler >= max_order - N + 1`), because the absolute
str>=7 selector yields ZERO anchors in scalable builds whose max Strahler order is
6 (mt_ruby / mt_big_hole / mt_beaverhead); the 2026-06-26 adaptive test showed
top-2 == str>=7 wherever str>=7 exists and top-1 is catastrophic (a single short
trunk reach can't anchor a basin). For each basin:

  1. select the top-N Strahler orders from streams_regional.fgb (or absolute
     --str-min if given),
  2. densify them and sample the 10 m DEM channel-bed ELEVATION -> leakage-free anchors,
  3. k-NN IDW the channel-bed elevation onto a 100 m EPSG:5070 grid aligned to the
     CONUS mosaic grid (origin -2540000, 3258000),
  4. write two GeoTIFFs into the basin dir:
       str_top{N}_idw_wte_100m.tif : interpolated water-table ELEVATION surface (m)
       str_top{N}_idw_dtw_100m.tif : DEPTH to water = ground (100 m mean DEM) - WTE (m)

Both ground and channel elevation come from the SAME 10 m DEM (one vertical datum).
Cells beyond --dmax-km of any anchor, or outside the basin polygon, are nodata. DTW
is written raw/unclamped so the known relief blow-up is visible; the prior is honest
only in low-relief alluvial valleys near a through-flowing trunk.

Run:
    uv run python utils/build_str7_idw_raster.py                  # five huc8 builds, top-2
    uv run python utils/build_str7_idw_raster.py --top-orders 1   # top order only
    uv run python utils/build_str7_idw_raster.py --str-min 7 --basins nm/regional/mesilla
"""

import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject
from scipy.spatial import cKDTree
from shapely.geometry import LineString, MultiLineString

H = Path("/data/ssd2/handily")
STUDY_BASINS = [
    "huc8/mt_ruby",
    "huc8/mt_big_hole",
    "huc8/mt_beaverhead",
    "huc8/nm_rio_grande_abq",
    "huc8/nv_upper_humboldt",
]
NODATA = -9999.0


def densify(geom, step):
    out = []
    geoms = geom.geoms if isinstance(geom, MultiLineString) else [geom]
    for ln in geoms:
        if isinstance(ln, LineString) and ln.length > 0:
            n = max(int(ln.length // step), 1)
            for d in np.linspace(0, ln.length, n + 1):
                p = ln.interpolate(d)
                out.append((p.x, p.y))
    return out


def sample_dem(path, x, y):
    with rasterio.open(path) as s:
        v = np.array([t[0] for t in s.sample(np.c_[x, y])], "float64")
        nod = s.nodata
    if nod is not None:
        v[v == nod] = np.nan
    v[(v < -1e3) | (v > 1e4)] = np.nan
    return v


def ground_100m(dem_path, transform, width, height):
    """Mean-aggregate the 10 m DEM onto the output grid (GDAL streams the warp)."""
    out = np.full((height, width), np.nan, "float32")
    with rasterio.open(dem_path) as src:
        reproject(
            source=rasterio.band(src, 1),
            destination=out,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src.nodata,
            dst_transform=transform,
            dst_crs="EPSG:5070",
            dst_nodata=np.nan,
            resampling=Resampling.average,
        )
    out[(out < -1e3) | (out > 1e4)] = np.nan
    return out


def idw_grid(axy, az, qxy, k, p, eps=1e-6, chunk=2_000_000):
    tree = cKDTree(axy)
    kk = min(k, len(axy))
    n = len(qxy)
    wte = np.empty(n, "float64")
    nd = np.empty(n, "float64")
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        d, i = tree.query(qxy[s:e], k=kk, workers=-1)
        if kk == 1:
            d, i = d[:, None], i[:, None]
        w = 1.0 / np.power(d + eps, p)
        wte[s:e] = (w * az[i]).sum(1) / w.sum(1)
        nd[s:e] = d[:, 0]
    return wte, nd


def write_tif(path, arr, transform):
    arr = np.where(np.isfinite(arr), arr, NODATA).astype("float32")
    prof = dict(
        driver="GTiff",
        dtype="float32",
        count=1,
        height=arr.shape[0],
        width=arr.shape[1],
        crs="EPSG:5070",
        transform=transform,
        nodata=NODATA,
        compress="deflate",
        predictor=3,
        tiled=True,
        blockxsize=256,
        blockysize=256,
    )
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(arr, 1)


def select_reaches(st, top_orders, str_min):
    """Return (selected reaches, output filename stem). top_orders is per-basin adaptive."""
    o = st[st["strahler"] > 0]
    if str_min is not None:
        return o[o["strahler"] >= str_min], f"str{str_min}"
    mx = int(o["strahler"].max())
    return o[o["strahler"] >= mx - top_orders + 1], f"str_top{top_orders}"


def build_basin(rel, top_orders, str_min, step_m, idw_k, idw_power, dmax_km, res):
    root = H / rel
    dem_path = str(root / "dem_10m.tif")
    boundary = gpd.read_file(root / "basin_boundary.fgb").to_crs(5070)
    st = gpd.read_file(root / "streams_regional.fgb").to_crs(5070)
    sel, stem = select_reaches(st, top_orders, str_min)
    if len(sel) == 0:
        print(f"  SKIP {rel}: no reaches for selector {stem}")
        return
    orders = sorted(set(sel["strahler"].astype(int)))
    pts = [p for g in sel.geometry for p in densify(g, step_m)]
    pa = np.asarray(pts, "float64")
    az = sample_dem(dem_path, pa[:, 0], pa[:, 1])
    ok = np.isfinite(az)
    axy, az = pa[ok][:, :2], az[ok]

    minx, miny, maxx, maxy = boundary.total_bounds
    minx, miny = np.floor(minx / res) * res, np.floor(miny / res) * res
    maxx, maxy = np.ceil(maxx / res) * res, np.ceil(maxy / res) * res
    width, height = int((maxx - minx) / res), int((maxy - miny) / res)
    transform = from_origin(minx, maxy, res, res)
    print(
        f"  {rel}: {stem} orders {orders}, {len(sel)} reaches, {len(axy)} anchors, grid {width}x{height} @ {res}m"
    )

    cx = minx + (np.arange(width) + 0.5) * res
    cy = maxy - (np.arange(height) + 0.5) * res
    gx, gy = np.meshgrid(cx, cy)
    qxy = np.c_[gx.ravel(), gy.ravel()]

    wte, nd = idw_grid(axy, az, qxy, idw_k, idw_power)
    wte = wte.reshape(height, width)
    far = (nd > dmax_km * 1000.0).reshape(height, width)
    ground = ground_100m(dem_path, transform, width, height)
    inside = rasterize(
        [(g, 1) for g in boundary.geometry],
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype="uint8",
    ).astype(bool)

    mask = far | ~inside
    wte_out = np.where(mask, np.nan, wte)
    dtw = ground - wte
    dtw_out = np.where(mask | ~np.isfinite(ground), np.nan, dtw)

    write_tif(str(root / f"{stem}_idw_wte_100m.tif"), wte_out, transform)
    write_tif(str(root / f"{stem}_idw_dtw_100m.tif"), dtw_out, transform)
    fin = np.isfinite(dtw_out)
    d = dtw_out[fin]
    print(
        f"    DTW finite {fin.sum() / dtw_out.size:.0%}  "
        f"med {np.median(d):.1f}  p10/p90 {np.percentile(d, 10):.0f}/{np.percentile(d, 90):.0f}  "
        f"min/max {d.min():.0f}/{d.max():.0f}  ->  {rel}/{stem}_idw_dtw_100m.tif"
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--basins",
        nargs="*",
        default=STUDY_BASINS,
        help="basin rel-paths under /data/ssd2/handily",
    )
    ap.add_argument(
        "--top-orders",
        type=int,
        default=2,
        help="per-basin: use the top N Strahler orders",
    )
    ap.add_argument(
        "--str-min",
        type=int,
        default=None,
        help="absolute Strahler floor (overrides --top-orders)",
    )
    ap.add_argument("--step-m", type=float, default=150.0)
    ap.add_argument("--idw-k", type=int, default=16)
    ap.add_argument("--idw-power", type=float, default=2.0)
    ap.add_argument("--dmax-km", type=float, default=50.0)
    ap.add_argument("--res", type=float, default=100.0)
    args = ap.parse_args()
    for rel in args.basins:
        build_basin(
            rel,
            args.top_orders,
            args.str_min,
            args.step_m,
            args.idw_k,
            args.idw_power,
            args.dmax_km,
            args.res,
        )


if __name__ == "__main__":
    main()
