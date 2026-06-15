"""Build a multi-year, high-resolution NAIP open-water support mask.

Aggregates the three-gate NAIP water mask (``compute_water_mask``: NDWI / NIR /
NDVI) across several NAIP years over the Rio Grande corridor counties, then
combines years into a PERSISTENT-water binary raster (water in >= K of N years).
The perennial Rio Grande mainstem and persistent water bodies register; transient
irrigation ponds and single-date ephemeral water do not.

This feeds the FAC REM channel-head solve as ``support_path`` so the Rio Grande
mainstem (unresolvable by 10m Sentinel-2 NDWI) is detected and its valley reaches
stay shallow.

Per county/year the full county MrSID is decoded ONLY within the river-corridor
bounding box (high-order streams + buffer) at a reduced level (scale=1, ~1.2m
from 0.6m NAIP: res = native * 2**scale). Full-county decodes at 1.2m are ~100GB
and unnecessary — the river is a narrow N-S valley strip. Each per-year water
fraction is resampled to the DEM-aligned grid (5m), thresholded to binary,
and stacked.

NAIP band handling (from handily.naip.compute_water_mask):
- 4-band ``_m`` product (2024): R, G, B, NIR
- 3-band CIR ``_c`` product (2018/2020/2022): NIR, R, G  (has NIR + green -> NDWI)
Both carry the bands the three-gate water test needs. The 3-band ``_n`` natural-
color product has NO NIR and is unusable — find_county_zip never selects it.

Usage:
    uv run python utils/build_basin_naip_water_multiyear.py \
        --regional-dir /data/ssd2/handily/nm/regional/rio_grande_albuquerque \
        --state nm --year 2018 --year 2020 --year 2022 --year 2024 \
        --fips 043 --fips 001 --fips 061 --fips 053 \
        --cache-dir /data/ssd2/handily/naip_cache \
        --decode-scale 1 --grid-res 5.0 --persist-k 2 --nir-thresh 0.25
"""

import argparse
import hashlib
import json
import logging
import shutil
from pathlib import Path
from time import perf_counter

import geopandas as gpd
import numpy as np
import rasterio
import rioxarray
import xarray as xr
from rasterio.enums import Resampling as RioResampling
from rasterio.features import rasterize
from rasterio.warp import reproject
from shapely.geometry import box

from handily.naip import (
    _box_download_file,
    _extract_sid_files,
    compute_water_mask,
    decode_sid,
    find_county_zip,
)
from handily.rem_fac import _axes_from_bounds, _match_transform, sample_dem_to_grid

log = logging.getLogger(__name__)


def _corridor_bbox_wgs84(
    corridor_5070, county_geom_5070
) -> tuple[float, float, float, float] | None:
    """Intersection of corridor and county, as a WGS84 (W, S, E, N) bbox."""
    win = corridor_5070.intersection(county_geom_5070)
    if win.is_empty:
        return None
    bbox_5070 = box(*win.bounds)
    wgs = gpd.GeoSeries([bbox_5070], crs="EPSG:5070").to_crs("EPSG:4326").iloc[0]
    return wgs.bounds  # (W, S, E, N)


def _county_year_fraction(
    state: str,
    fips: str,
    year: str,
    cache_dir: Path,
    scale: int,
    bbox_wsen: tuple[float, float, float, float],
    grid_c,
    grid_transform,
    grid_crs,
    gate_kwargs: dict,
    cleanup: bool,
) -> np.ndarray | None:
    """Download, decode (corridor-clipped), water-mask, resample one county-year.

    Returns a per-cell water fraction on the DEM grid, or None if no NAIP.
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

    extract_dir = cache_dir / f"{state}_{fips}_{year}_corr"
    frac = np.zeros(grid_c.shape, dtype=np.float32)
    n_tiles = 0

    for sid_path in _extract_sid_files(zip_path, extract_dir):
        tif_path = extract_dir / f"{sid_path.stem}_s{scale}_corr.tif"
        if not tif_path.exists():
            try:
                decode_sid(
                    str(sid_path), str(tif_path), bounds_wsen=bbox_wsen, scale=scale
                )
            except RuntimeError as exc:
                # AOI-does-not-intersect-SID-extent is normal for a clipped window
                log.info("Skipping %s: %s", sid_path.name, exc)
                continue

        try:
            with rasterio.open(tif_path) as src:
                if src.width == 0 or src.height == 0:
                    continue
        except Exception as exc:
            log.warning("Unreadable tile %s: %s", tif_path.name, exc)
            continue

        mask_tif = tif_path.with_name(f"{tif_path.stem}_water.tif")
        if not mask_tif.exists():
            compute_water_mask(str(tif_path), str(mask_tif), **gate_kwargs)

        # Average-resample the 0/1 mask to a per-cell water fraction. Destination
        # is pre-zeroed and init_dest_nodata=False so cells outside the decoded
        # window stay 0 (no nodata fill leaks into the merged grid).
        county_frac = np.zeros(grid_c.shape, dtype=np.float32)
        with rasterio.open(mask_tif) as src:
            reproject(
                source=rasterio.band(src, 1),
                destination=county_frac,
                src_transform=src.transform,
                src_crs=src.crs,
                src_nodata=255,
                dst_transform=grid_transform,
                dst_crs=grid_crs,
                init_dest_nodata=False,
                resampling=RioResampling.average,
            )
        frac = np.maximum(frac, county_frac)
        n_tiles += 1

    if cleanup:
        shutil.rmtree(extract_dir, ignore_errors=True)
        # Drop the multi-GB ZIP too — disk is tight and each county-year is unique.
        zip_path.unlink(missing_ok=True)

    if n_tiles == 0:
        log.warning("No decoded tiles intersected corridor for %s %s", fips, year)
        return None
    return frac


def main():
    parser = argparse.ArgumentParser(
        description="Multi-year persistent NAIP open-water support mask"
    )
    parser.add_argument("--regional-dir", required=True)
    parser.add_argument("--state", required=True)
    parser.add_argument(
        "--year", action="append", required=True, help="NAIP year (repeatable)"
    )
    parser.add_argument(
        "--fips", action="append", required=True, help="County FIPS (repeatable)"
    )
    parser.add_argument("--cache-dir", default="/data/ssd2/handily/naip_cache")
    parser.add_argument(
        "--decode-scale",
        type=int,
        default=1,
        help="MrSID decode level: output res = native * 2**scale (1 -> ~1.2m)",
    )
    parser.add_argument("--grid-res", type=float, default=5.0)
    parser.add_argument(
        "--support-min-fraction",
        type=float,
        default=0.25,
        help="Min within-year water fraction of a grid cell to count as water",
    )
    parser.add_argument(
        "--persist-k",
        type=int,
        default=2,
        help="Keep a cell if water in >= K of the N years (persistence)",
    )
    parser.add_argument(
        "--corridor-min-order",
        type=int,
        default=5,
        help="Min Strahler order of streams defining the river corridor",
    )
    parser.add_argument(
        "--corridor-buffer-m",
        type=float,
        default=2000.0,
        help="Buffer around corridor streams (m)",
    )
    parser.add_argument("--ndwi-thresh", type=float, default=0.1)
    # 0.25 (not the 0.15 NAIP default): turbid sediment-laden river reflects more NIR
    parser.add_argument("--nir-thresh", type=float, default=0.25)
    parser.add_argument("--ndvi-thresh", type=float, default=0.1)
    parser.add_argument("--out-name", default="naip_ndwi_support_multiyear_5m.tif")
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
    gate_kwargs = dict(
        ndwi_thresh=args.ndwi_thresh,
        nir_thresh=args.nir_thresh,
        ndvi_thresh=args.ndvi_thresh,
    )

    # Per-year binary cache so a failure in a later year doesn't discard the
    # finished years (each year is a long download+decode+mask pass). Keyed on
    # every parameter that affects the per-year grid, so a tuning change (e.g. a
    # new --nir-thresh) writes to a fresh key instead of silently reusing stale
    # results.
    year_cache_dir = out_dir / "_year_cache"
    year_cache_dir.mkdir(parents=True, exist_ok=True)
    param_key = json.dumps(
        {
            "decode_scale": args.decode_scale,
            "grid_res": args.grid_res,
            "support_min_fraction": args.support_min_fraction,
            "corridor_min_order": args.corridor_min_order,
            "corridor_buffer_m": args.corridor_buffer_m,
            "gate": gate_kwargs,
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
    log.info("5m grid shape %s, crs %s", grid_c.shape, grid_crs)

    # River corridor from high-order streams, clipped to the DEM extent.
    streams = gpd.read_file(regional_dir / "streams_regional.fgb").to_crs(grid_crs)
    corridor = (
        streams[streams.strahler >= args.corridor_min_order]
        .buffer(args.corridor_buffer_m)
        .union_all()
        .intersection(box(*dem_bounds))
    )
    log.info("Corridor area %.0f km2", corridor.area / 1e6)

    # Census counties for county geometry (same source as handily.naip)
    from handily.naip import COUNTIES_SHP

    counties = gpd.read_file(COUNTIES_SHP)
    counties = counties[counties["STATEFP"] == "35"].to_crs(grid_crs)

    # Per-year binary water stack on the DEM grid.
    year_binaries: list[np.ndarray] = []
    years_used: list[str] = []
    years_dropped: list[dict] = []
    per_county_records: list[dict] = []

    for year in years:
        log.info("########## YEAR %s ##########", year)
        year_cache = year_cache_dir / f"year_{year}_{param_hash}_bin.tif"
        if year_cache.exists():
            with rasterio.open(year_cache) as src:
                year_bin = src.read(1)
            if year_bin.shape != grid_c.shape:
                raise RuntimeError(
                    f"Cached year {year} shape {year_bin.shape} != grid "
                    f"{grid_c.shape}; clear {year_cache_dir}"
                )
            year_binaries.append(year_bin.astype(np.uint8))
            years_used.append(year)
            per_county_records.append({"year": year, "status": "cached_year"})
            log.info(
                "Year %s loaded from cache: %d water cells",
                year,
                int(year_bin.sum()),
            )
            continue
        year_frac = np.zeros(grid_c.shape, dtype=np.float32)
        any_county = False
        for fips in fips_list:
            cgeom = counties[counties["COUNTYFP"] == fips].geometry.union_all()
            bbox = _corridor_bbox_wgs84(corridor, cgeom)
            if bbox is None:
                log.info("County %s has no corridor overlap -> skip", fips)
                continue
            log.info("=== %s%s %s corridor bbox(WSEN)=%s ===", state, fips, year, bbox)
            frac = _county_year_fraction(
                state,
                fips,
                year,
                cache_dir,
                args.decode_scale,
                bbox,
                grid_c,
                grid_transform,
                grid_crs,
                gate_kwargs,
                cleanup=not args.no_cleanup,
            )
            if frac is None:
                per_county_records.append(
                    {"year": year, "fips": fips, "status": "no_data"}
                )
                continue
            any_county = True
            year_frac = np.maximum(year_frac, frac)
            per_county_records.append(
                {
                    "year": year,
                    "fips": fips,
                    "status": "ok",
                    "water_cells_ge_minfrac": int(
                        (frac >= args.support_min_fraction).sum()
                    ),
                }
            )

        if not any_county:
            years_dropped.append({"year": year, "reason": "no county data decoded"})
            log.warning("Year %s produced no data -> dropped", year)
            continue

        year_bin = (year_frac >= args.support_min_fraction).astype(np.uint8)
        # Cache atomically (tmp + rename) so an interrupted write is never
        # mistaken for a finished year on the next run.
        year_da = xr.DataArray(year_bin, coords=grid_c.coords, dims=grid_c.dims)
        year_da = year_da.rio.write_crs(grid_crs).rio.set_spatial_dims(
            x_dim="x", y_dim="y"
        )
        tmp_cache = year_cache.with_name(year_cache.name + ".tmp")
        # Force the driver: rioxarray infers it from the extension, which fails
        # on the ".tif.tmp" temp name.
        year_da.rio.to_raster(tmp_cache, dtype="uint8", driver="GTiff")
        tmp_cache.replace(year_cache)
        year_binaries.append(year_bin)
        years_used.append(year)
        log.info("Year %s water cells: %d", year, int(year_bin.sum()))

    if not year_binaries:
        raise RuntimeError("No years produced any NAIP water data")

    n_years = len(year_binaries)
    persist_k = min(args.persist_k, n_years)
    if persist_k < args.persist_k:
        log.warning(
            "Requested persist-k=%d but only %d years available; using K=%d",
            args.persist_k,
            n_years,
            persist_k,
        )
    year_count = np.sum(year_binaries, axis=0).astype(np.uint8)
    support = (year_count >= persist_k).astype(np.uint8)

    support_da = xr.DataArray(
        support, coords=grid_c.coords, dims=grid_c.dims, name="naip_water_support"
    )
    support_da = support_da.rio.write_crs(grid_crs)
    support_da = support_da.rio.set_spatial_dims(x_dim="x", y_dim="y")
    out_path = out_dir / args.out_name
    support_da.rio.to_raster(out_path, dtype="uint8")
    log.info("Wrote %s", out_path)

    # Coverage stats against the unbuffered basin polygon.
    boundary = gpd.read_file(regional_dir / "basin_boundary.fgb").to_crs(grid_crs)
    basin = rasterize(
        boundary.geometry,
        out_shape=grid_c.shape,
        transform=grid_transform,
        fill=0,
        default_value=1,
        dtype="uint8",
    ).astype(bool)
    basin_water_frac = float((support[basin] > 0).mean())
    total_water_cells = int(support.sum())

    summary = {
        "state": state,
        "counties_fips": fips_list,
        "years_requested": years,
        "years_used": years_used,
        "years_dropped": years_dropped,
        "decode_scale": args.decode_scale,
        "decode_res_m_approx": round(0.6 * (2**args.decode_scale), 2),
        "grid_res_m": args.grid_res,
        "grid_shape": list(grid_c.shape),
        "grid_crs": str(grid_crs),
        "corridor_min_strahler": args.corridor_min_order,
        "corridor_buffer_m": args.corridor_buffer_m,
        "corridor_area_km2": round(corridor.area / 1e6, 1),
        "support_min_fraction": args.support_min_fraction,
        "persistence_rule": f"water in >= {persist_k} of {n_years} years",
        "persist_k": persist_k,
        "n_years": n_years,
        "gate_thresholds": gate_kwargs,
        "output_raster": str(out_path),
        "total_water_cells": total_water_cells,
        "basin_water_support_fraction": basin_water_frac,
        "per_county_year": per_county_records,
        "runtime_s": round(perf_counter() - t0, 1),
    }
    json_path = out_path.with_name(out_path.stem + "_build.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    log.info("Wrote %s", json_path)
    log.info(
        "DONE: %d water cells; basin water-support fraction %.4f%%; %d/%d years; %.0fs",
        total_water_cells,
        100 * basin_water_frac,
        n_years,
        len(years),
        perf_counter() - t0,
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    main()
