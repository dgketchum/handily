"""Build basin-wide NDVI and surface-water support from county NAIP.

Produces the two evidence rasters the channel-head solve needs over a full
regional (10m) DEM extent:

- ``evidence/ndvi_20m.tif``: NDVI computed per county at the DEM's 10m grid
  (nearest), merged across counties with fmax, then max-resampled to 20m so
  the greenest pixel drives seed strength.
- ``evidence/support_20m.tif``: three-gate NAIP water mask (NDWI/NIR/NDVI,
  ``compute_water_mask``) at the decoded resolution, average-resampled to a
  20m water fraction and thresholded (default 0.25) to a binary grid.
  Fraction-thresholding suppresses isolated shadow speckle that an
  any-water max aggregation would promote to hard-pin evidence. 0 means
  "no detection" everywhere outside coverage — support is positive
  evidence only.

County orthos are downloaded from the USDA Box share (cached), and MrSID
mosaics are decoded at a reduced level (default scale=3, ~4.8m from 0.6m
NAIP) — full-resolution county decodes are ~100GB and unnecessary for 20m
evidence grids.

Usage:
    uv run python utils/build_basin_naip_evidence.py \
        --regional-dir /data/ssd2/handily/mt/regional/missouri_headwaters \
        --state mt --year 2023 --fips 001 --fips 057 --fips 023 --fips 093 \
        --cache-dir /data/ssd2/handily/naip_cache
"""

import argparse
import json
import logging
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

from handily.naip import (
    _box_download_file,
    _extract_sid_files,
    compute_water_mask,
    decode_sid,
    find_county_zip,
)
from handily.rem_fac import (
    _axes_from_bounds,
    _match_transform,
    compute_naip_ndvi_match,
    sample_dem_to_grid,
)

log = logging.getLogger(__name__)


def _ensure_county_tifs(
    state: str, fips: str, year: str, cache_dir: Path, scale: int
) -> list[Path]:
    """Download, extract, and decode one county's NAIP. Returns decoded tifs."""
    entry = find_county_zip(state, year, fips)
    if entry is None:
        raise FileNotFoundError(f"No NAIP on Box for {state} FIPS {fips} year {year}")

    zip_path = cache_dir / entry["name"]
    if zip_path.exists():
        log.info("Using cached ZIP: %s", zip_path)
    else:
        _box_download_file(entry["id"], str(zip_path))

    extract_dir = cache_dir / f"{state}_{fips}_{year}"
    tifs: list[Path] = []
    for sid_path in _extract_sid_files(zip_path, extract_dir):
        tif_path = extract_dir / f"{sid_path.stem}_s{scale}.tif"
        if not tif_path.exists():
            log.info("Decoding %s at scale %d ...", sid_path.name, scale)
            decode_sid(str(sid_path), str(tif_path), scale=scale)
        tifs.append(tif_path)
    return tifs


def main():
    parser = argparse.ArgumentParser(
        description="Basin-wide NAIP NDVI and water-support evidence grids"
    )
    parser.add_argument("--regional-dir", required=True)
    parser.add_argument("--state", required=True)
    parser.add_argument("--year", required=True)
    parser.add_argument(
        "--fips", action="append", required=True, help="County FIPS (repeatable)"
    )
    parser.add_argument("--cache-dir", default="/data/ssd2/handily/naip_cache")
    parser.add_argument(
        "--decode-scale",
        type=int,
        default=3,
        help="MrSID decode level: output res = native * 2**scale",
    )
    parser.add_argument("--grid-res", type=float, default=20.0)
    parser.add_argument(
        "--support-min-fraction",
        type=float,
        default=0.25,
        help="Min water fraction of a grid cell to count as support",
    )
    args = parser.parse_args()

    regional_dir = Path(args.regional_dir)
    cache_dir = Path(args.cache_dir)
    out_dir = regional_dir / "evidence"
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = perf_counter()
    dem_path = regional_dir / "dem_10m.tif"
    log.info("Loading DEM grid: %s", dem_path)
    dem_da = rioxarray.open_rasterio(dem_path).squeeze("band", drop=True)
    dem_da = dem_da.rio.set_spatial_dims(x_dim="x", y_dim="y")
    dem_da = dem_da.where(dem_da > -1e5)

    x_c, y_c = _axes_from_bounds(tuple(dem_da.rio.bounds()), args.grid_res)
    grid_c = sample_dem_to_grid(dem_da, x_c, y_c)
    grid_transform = _match_transform(grid_c)
    grid_crs = grid_c.rio.crs

    ndvi_10m = None
    support = np.zeros(grid_c.shape, dtype=np.uint8)

    for fips in args.fips:
        fips = str(fips).zfill(3)
        log.info("=== County %s%s %s ===", args.state, fips, args.year)
        tifs = _ensure_county_tifs(
            args.state.lower(), fips, str(args.year), cache_dir, args.decode_scale
        )
        for tif in tifs:
            log.info("NDVI from %s", tif.name)
            nd = compute_naip_ndvi_match(
                str(tif), dem_da, resampling=RioResampling.nearest
            )
            if ndvi_10m is None:
                ndvi_10m = nd
            else:
                ndvi_10m.values = np.fmax(ndvi_10m.values, nd.values)

            mask_tif = tif.with_name(f"{tif.stem}_water.tif")
            if not mask_tif.exists():
                log.info("Water mask from %s", tif.name)
                compute_water_mask(str(tif), str(mask_tif))
            # Average-resample the 0/1 mask to a per-cell water fraction.
            # Destination is pre-zeroed and init_dest_nodata=False so cells
            # outside the county footprint stay 0 (no nodata fill value can
            # leak into the merged grid).
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
            county_support = (county_frac >= args.support_min_fraction).astype(
                np.uint8
            )
            support = np.maximum(support, county_support)

    ndvi_da = ndvi_10m.rio.reproject_match(grid_c, resampling=RioResampling.max)
    ndvi_path = out_dir / f"ndvi_{int(args.grid_res)}m.tif"
    ndvi_da.rio.to_raster(ndvi_path)

    support_da = xr.DataArray(
        support, coords=grid_c.coords, dims=grid_c.dims, name="water_support"
    )
    support_da = support_da.rio.write_crs(grid_crs)
    support_da = support_da.rio.set_spatial_dims(x_dim="x", y_dim="y")
    support_path = out_dir / f"support_{int(args.grid_res)}m.tif"
    support_da.rio.to_raster(support_path)

    # Coverage stats against the unbuffered basin polygon — the DEM extends
    # 5km past the basin cutline, and that ring is outside county imagery.
    boundary = gpd.read_file(regional_dir / "basin_boundary.fgb").to_crs(grid_crs)
    basin = rasterize(
        boundary.geometry,
        out_shape=grid_c.shape,
        transform=grid_transform,
        fill=0,
        default_value=1,
        dtype="uint8",
    ).astype(bool)
    ndvi_cov = float((np.isfinite(ndvi_da.values) & basin).sum() / basin.sum())
    water_frac = float((support[basin] > 0).mean())
    summary = {
        "state": args.state,
        "year": str(args.year),
        "fips": [str(f).zfill(3) for f in args.fips],
        "decode_scale": args.decode_scale,
        "grid_res_m": args.grid_res,
        "support_min_fraction": args.support_min_fraction,
        "ndvi_basin_coverage_fraction": ndvi_cov,
        "water_support_basin_fraction": water_frac,
        "runtime_s": perf_counter() - t0,
    }
    with open(out_dir / "evidence_build.json", "w") as f:
        json.dump(summary, f, indent=2)

    log.info(
        "NDVI covers %.1f%% of basin; water support on %.2f%% of basin cells",
        100 * ndvi_cov,
        100 * water_frac,
    )
    log.info("Wrote %s and %s in %.0fs", ndvi_path, support_path, perf_counter() - t0)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    main()
