"""Build a multi-year, COARSE-resolution NAIP NDVI seed over a valley window.

The FAC head solve seeds wet reaches from a per-pixel greenness layer
(``rem_fac_topology.estimate_reach_seed_strength`` samples it along every reach
and pushes a high quantile through a sigmoid). The prior seed was an S2 seasonal
NDVI/NDWI composite, but its Earth Engine export is cost-prohibitive at basin
scale. In the arid Rio Grande valleys the valley/upland contrast is a coarse
areal signal — irrigated-green valley floor vs bare desert upland — so a
LOW-resolution NAIP NDVI recovers it without any EE.

This is the counterpart to ``build_basin_naip_water_multiyear.py`` and differs in
the three ways that matter:

  * resolution — the open-water *support* mask needs the ~1.2m decode to catch
    the narrow river channel; NDVI does NOT, so we decode at a coarse level
    (``--decode-scale 4`` -> 0.6m * 2**4 ~ 9.6m), which is far faster/smaller.
  * extent — every reach needs a seed, not just the high-order corridor, so we
    decode the WHOLE valley window (county ∩ DEM extent), not a stream buffer.
  * aggregation — NDVI is continuous: per year we average tile coverage onto the
    DEM grid, then take the across-year median for a drought/fallow-robust seed.

Output is a single-band NDVI raster (float32, nodata NaN) on the DEM grid, for
the REM config ``ndvi_path``.

Usage:
    uv run python utils/build_basin_naip_ndvi_multiyear.py \
        --regional-dir /data/ssd2/handily/nm/regional/mesilla \
        --state nm --year 2018 --year 2020 --year 2022 \
        --fips 013 --fips 051 \
        --cache-dir /data/ssd2/handily/naip_cache_ndvi \
        --decode-scale 4 --grid-res 10.0 --out-name naip_ndvi_3yr_10m.tif
"""

import argparse
import hashlib
import json
import logging
import shutil
import warnings
from pathlib import Path
from time import perf_counter

import geopandas as gpd
import numpy as np
import rasterio
import rioxarray
import xarray as xr
from rasterio.enums import Resampling as RioResampling
from rasterio.warp import reproject
from shapely.geometry import box

from handily.naip import (
    _box_download_file,
    _extract_sid_files,
    decode_sid,
    find_county_zip,
)
from handily.rem_fac import _axes_from_bounds, _match_transform, sample_dem_to_grid

log = logging.getLogger(__name__)


def _window_bbox_wgs84(
    dem_box_5070, county_geom_5070
) -> tuple[float, float, float, float] | None:
    """Intersection of the DEM extent and a county, as a WGS84 (W, S, E, N) bbox."""
    win = dem_box_5070.intersection(county_geom_5070)
    if win.is_empty:
        return None
    wgs = gpd.GeoSeries([box(*win.bounds)], crs="EPSG:5070").to_crs("EPSG:4326").iloc[0]
    return wgs.bounds


def _county_year_ndvi(
    state: str,
    fips: str,
    year: str,
    cache_dir: Path,
    scale: int,
    bbox_wsen: tuple[float, float, float, float],
    grid_c,
    grid_transform,
    grid_crs,
    cleanup: bool,
) -> np.ndarray | None:
    """Download, coarse-decode (window-clipped), NDVI, resample one county-year.

    Returns per-cell NDVI on the DEM grid (NaN where no coverage), or None.
    """
    entry = find_county_zip(state, year, fips)
    if entry is None:
        log.warning("No NAIP on Box for %s FIPS %s year %s -> skip", state, fips, year)
        return None

    zip_path = cache_dir / entry["name"]
    if zip_path.exists():
        log.info("Using cached ZIP: %s", zip_path)
    else:
        _box_download_file(entry["id"], str(zip_path))

    extract_dir = cache_dir / f"{state}_{fips}_{year}_ndvi"
    g_sum = np.zeros(grid_c.shape, dtype=np.float32)
    g_cnt = np.zeros(grid_c.shape, dtype=np.float32)
    n_tiles = 0

    for sid_path in _extract_sid_files(zip_path, extract_dir):
        tif_path = extract_dir / f"{sid_path.stem}_s{scale}_win.tif"
        if not tif_path.exists():
            try:
                decode_sid(
                    str(sid_path), str(tif_path), bounds_wsen=bbox_wsen, scale=scale
                )
            except RuntimeError as exc:
                log.info("Skipping %s: %s", sid_path.name, exc)
                continue

        try:
            with rasterio.open(tif_path) as src:
                if src.width == 0 or src.height == 0:
                    continue
                # NAIP band order: 4-band _m (R,G,B,NIR); 3-band CIR _c (NIR,R,G)
                if src.count >= 4:
                    red = src.read(1).astype("float32")
                    nir = src.read(4).astype("float32")
                elif src.count == 3:
                    nir = src.read(1).astype("float32")
                    red = src.read(2).astype("float32")
                else:
                    log.warning(
                        "Unexpected band count %d: %s", src.count, tif_path.name
                    )
                    continue
                src_transform, src_crs = src.transform, src.crs
        except Exception as exc:
            log.warning("Unreadable tile %s: %s", tif_path.name, exc)
            continue

        denom = nir + red
        ndvi = np.where(denom > 0, (nir - red) / denom, np.nan).astype("float32")

        tile_grid = np.full(grid_c.shape, np.nan, dtype=np.float32)
        reproject(
            source=ndvi,
            destination=tile_grid,
            src_transform=src_transform,
            src_crs=src_crs,
            src_nodata=np.nan,
            dst_transform=grid_transform,
            dst_crs=grid_crs,
            dst_nodata=np.nan,
            init_dest_nodata=True,
            resampling=RioResampling.average,
        )
        valid = np.isfinite(tile_grid)
        g_sum[valid] += tile_grid[valid]
        g_cnt[valid] += 1.0
        n_tiles += 1

    if cleanup:
        shutil.rmtree(extract_dir, ignore_errors=True)
        zip_path.unlink(missing_ok=True)

    if n_tiles == 0:
        log.warning("No decoded tiles intersected window for %s %s", fips, year)
        return None
    return np.where(g_cnt > 0, g_sum / g_cnt, np.nan).astype("float32")


def main():
    parser = argparse.ArgumentParser(
        description="Multi-year coarse NAIP NDVI seed over a valley window"
    )
    parser.add_argument("--regional-dir", required=True)
    parser.add_argument("--state", required=True)
    parser.add_argument(
        "--year", action="append", required=True, help="NAIP year (repeatable)"
    )
    parser.add_argument(
        "--fips", action="append", required=True, help="County FIPS (repeatable)"
    )
    parser.add_argument("--cache-dir", default="/data/ssd2/handily/naip_cache_ndvi")
    parser.add_argument(
        "--decode-scale",
        type=int,
        default=4,
        help="MrSID decode level: output res = native * 2**scale (4 -> ~9.6m)",
    )
    parser.add_argument("--grid-res", type=float, default=10.0)
    parser.add_argument("--out-name", default="naip_ndvi_multiyear_10m.tif")
    parser.add_argument("--no-cleanup", action="store_true")
    args = parser.parse_args()

    regional_dir = Path(args.regional_dir)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_dir = regional_dir / "evidence" / "naip"
    out_dir.mkdir(parents=True, exist_ok=True)
    state = args.state.lower()
    years = [str(y) for y in args.year]
    fips_list = [str(f).zfill(3) for f in args.fips]

    year_cache_dir = out_dir / "_ndvi_year_cache"
    year_cache_dir.mkdir(parents=True, exist_ok=True)
    param_key = json.dumps(
        {
            "decode_scale": args.decode_scale,
            "grid_res": args.grid_res,
            "fips": fips_list,
        },
        sort_keys=True,
    )
    param_hash = hashlib.md5(param_key.encode()).hexdigest()[:10]

    t0 = perf_counter()
    dem_path = regional_dir / "dem_10m.tif"
    log.info("Loading DEM grid: %s", dem_path)
    dem_da = rioxarray.open_rasterio(dem_path).squeeze("band", drop=True)
    dem_da = dem_da.rio.set_spatial_dims(x_dim="x", y_dim="y")
    dem_da = dem_da.where(dem_da > -1e5)
    dem_bounds = tuple(dem_da.rio.bounds())

    x_c, y_c = _axes_from_bounds(dem_bounds, args.grid_res)
    grid_c = sample_dem_to_grid(dem_da, x_c, y_c)
    grid_transform = _match_transform(grid_c)
    grid_crs = grid_c.rio.crs
    dem_box = box(*dem_bounds)
    log.info("NDVI grid shape %s, crs %s", grid_c.shape, grid_crs)

    from handily.naip import COUNTIES_SHP

    counties = gpd.read_file(COUNTIES_SHP)
    counties = counties[counties["STATEFP"] == "35"].to_crs(grid_crs)

    year_ndvis: list[np.ndarray] = []
    years_used: list[str] = []
    per_county_records: list[dict] = []

    for year in years:
        log.info("########## YEAR %s ##########", year)
        year_cache = year_cache_dir / f"ndvi_{year}_{param_hash}.tif"
        if year_cache.exists():
            with rasterio.open(year_cache) as src:
                year_ndvi = src.read(1)
            if year_ndvi.shape != grid_c.shape:
                raise RuntimeError(
                    f"Cached year {year} shape {year_ndvi.shape} != grid {grid_c.shape}; "
                    f"clear {year_cache_dir}"
                )
            year_ndvis.append(year_ndvi.astype(np.float32))
            years_used.append(year)
            per_county_records.append({"year": year, "status": "cached_year"})
            log.info("Year %s loaded from cache", year)
            continue

        g_sum = np.zeros(grid_c.shape, dtype=np.float32)
        g_cnt = np.zeros(grid_c.shape, dtype=np.float32)
        any_county = False
        for fips in fips_list:
            cgeom = counties[counties["COUNTYFP"] == fips].geometry.union_all()
            bbox = _window_bbox_wgs84(dem_box, cgeom)
            if bbox is None:
                log.info("County %s has no window overlap -> skip", fips)
                continue
            log.info("=== %s%s %s window bbox(WSEN)=%s ===", state, fips, year, bbox)
            ndvi = _county_year_ndvi(
                state,
                fips,
                year,
                cache_dir,
                args.decode_scale,
                bbox,
                grid_c,
                grid_transform,
                grid_crs,
                cleanup=not args.no_cleanup,
            )
            if ndvi is None:
                per_county_records.append(
                    {"year": year, "fips": fips, "status": "no_data"}
                )
                continue
            any_county = True
            valid = np.isfinite(ndvi)
            g_sum[valid] += ndvi[valid]
            g_cnt[valid] += 1.0
            per_county_records.append(
                {
                    "year": year,
                    "fips": fips,
                    "status": "ok",
                    "valid_cells": int(valid.sum()),
                    "mean_ndvi": round(float(np.nanmean(ndvi)), 4),
                }
            )

        if not any_county:
            log.warning("Year %s produced no data -> dropped", year)
            continue

        year_ndvi = np.where(g_cnt > 0, g_sum / g_cnt, np.nan).astype("float32")
        year_da = xr.DataArray(year_ndvi, coords=grid_c.coords, dims=grid_c.dims)
        year_da = year_da.rio.write_crs(grid_crs).rio.set_spatial_dims(
            x_dim="x", y_dim="y"
        )
        tmp_cache = year_cache.with_name(year_cache.name + ".tmp")
        year_da.rio.to_raster(tmp_cache, dtype="float32", driver="GTiff", nodata=np.nan)
        tmp_cache.replace(year_cache)
        year_ndvis.append(year_ndvi)
        years_used.append(year)
        log.info("Year %s mean NDVI %.4f", year, float(np.nanmean(year_ndvi)))

    if not year_ndvis:
        raise RuntimeError("No years produced any NAIP NDVI data")

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "All-NaN slice encountered")
        seed = np.nanmedian(np.stack(year_ndvis, axis=0), axis=0).astype("float32")

    seed_da = xr.DataArray(
        seed, coords=grid_c.coords, dims=grid_c.dims, name="naip_ndvi"
    )
    seed_da = seed_da.rio.write_crs(grid_crs).rio.set_spatial_dims(x_dim="x", y_dim="y")
    out_path = out_dir / args.out_name
    seed_da.rio.to_raster(out_path, dtype="float32", nodata=np.nan)
    log.info("Wrote %s", out_path)

    finite = np.isfinite(seed)
    pcts = np.nanpercentile(seed[finite], [5, 25, 50, 75, 95]) if finite.any() else []
    summary = {
        "state": state,
        "counties_fips": fips_list,
        "years_requested": years,
        "years_used": years_used,
        "decode_scale": args.decode_scale,
        "decode_res_m_approx": round(0.6 * (2**args.decode_scale), 2),
        "grid_res_m": args.grid_res,
        "grid_shape": list(grid_c.shape),
        "grid_crs": str(grid_crs),
        "across_year_stat": "median",
        "valid_fraction": round(float(finite.mean()), 4),
        "ndvi_percentiles_5_25_50_75_95": [round(float(v), 4) for v in pcts],
        "output_raster": str(out_path),
        "per_county_year": per_county_records,
        "runtime_s": round(perf_counter() - t0, 1),
    }
    json_path = out_path.with_name(out_path.stem + "_build.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    log.info("Wrote %s", json_path)
    log.info(
        "DONE: NDVI seed %s; valid %.1f%%; median %.3f; %d/%d years; %.0fs",
        out_path.name,
        100 * summary["valid_fraction"],
        float(np.nanmedian(seed)),
        len(years_used),
        len(years),
        perf_counter() - t0,
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    main()
