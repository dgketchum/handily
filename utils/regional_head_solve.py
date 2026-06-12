"""Run the FAC channel-head solve on a 10m regional DEM and stream network.

Solves head_depth_m basin-wide from the regional 10m DEM and
streams_regional.fgb in place of the per-AOI 1m inputs. Skips strip
generation, rasterization, and IDW fill — output is the per-reach
fac_channel_heads vector only.

NDVI seeds come from county NAIP orthos. Reaches outside NAIP coverage
have no NDVI samples and default to fully dry (seed_ndvi = 0), so head
depths there reflect the sag bound, not observed vegetation.

Usage (prebuilt evidence from build_basin_naip_evidence.py):
    uv run python utils/regional_head_solve.py \
        --regional-dir /data/ssd2/handily/mt/regional/missouri_headwaters \
        --ndvi-raster .../evidence/ndvi_20m.tif \
        --support-raster .../evidence/support_20m.tif

Usage (ad-hoc NDVI from orthos, no support):
    uv run python utils/regional_head_solve.py \
        --regional-dir /data/ssd2/handily/mt/regional/missouri_headwaters \
        --naip /data/ssd2/handily/mt/aoi_0007/naip/ortho_1-1_hc_s_mt001_2023_1.tif
"""

import argparse
import json
import logging
from pathlib import Path
from time import perf_counter

import geopandas as gpd
import numpy as np
import rioxarray
from rasterio.enums import Resampling as RioResampling

from handily.rem_fac import (
    _axes_from_bounds,
    compute_naip_ndvi_match,
    sample_dem_to_grid,
)
from handily.rem_fac_head import build_channel_heads

log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="FAC channel-head solve on a 10m regional DEM"
    )
    parser.add_argument(
        "--regional-dir",
        required=True,
        help="Regional FAC directory with dem_10m.tif and streams_regional.fgb",
    )
    parser.add_argument(
        "--naip",
        action="append",
        default=None,
        help="NAIP ortho path (repeatable for multiple counties)",
    )
    parser.add_argument(
        "--ndvi-raster",
        default=None,
        help="Prebuilt NDVI grid (e.g. evidence/ndvi_20m.tif); overrides --naip",
    )
    parser.add_argument(
        "--support-raster",
        default=None,
        help="Binary water-support grid (e.g. evidence/support_20m.tif)",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory (default: {regional-dir}/head_solve_10m)",
    )
    parser.add_argument(
        "--ndvi-res",
        type=float,
        default=20.0,
        help="Resolution of the max-resampled NDVI grid (m)",
    )
    parser.add_argument(
        "--area-sag-lo-km2",
        type=float,
        default=50.0,
        help="Drainage area where the never-runs-dry prior starts ramping",
    )
    parser.add_argument(
        "--area-sag-hi-km2",
        type=float,
        default=500.0,
        help="Drainage area above which the sag driver is forced to 0",
    )
    parser.add_argument(
        "--strahler-pin-min",
        type=int,
        default=None,
        help="Hard-pin reaches at or above this topology Strahler order "
        "(headwaters are order 0)",
    )
    parser.add_argument(
        "--area-pin-km2",
        type=float,
        default=None,
        help="Drainage-area guard on the order pin: both must hold to pin",
    )
    args = parser.parse_args()

    regional_dir = Path(args.regional_dir)
    out_dir = Path(args.out_dir) if args.out_dir else regional_dir / "head_solve_10m"
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = perf_counter()
    dem_path = regional_dir / "dem_10m.tif"
    log.info("Loading DEM: %s", dem_path)
    dem_da = rioxarray.open_rasterio(dem_path).squeeze("band", drop=True)
    dem_da = dem_da.rio.set_spatial_dims(x_dim="x", y_dim="y")
    dem_da = dem_da.where(dem_da > -1e5)

    streams_path = regional_dir / "streams_regional.fgb"
    log.info("Loading streams: %s", streams_path)
    streams = gpd.read_file(streams_path)
    streams = streams.loc[
        streams.geometry.notnull() & ~streams.geometry.is_empty
    ].copy()
    if "reach_id" not in streams.columns:
        streams["reach_id"] = np.arange(len(streams), dtype=np.int64)
    streams = streams.sort_values("reach_id").reset_index(drop=True)
    log.info("  %d reaches, %.0f km", len(streams), streams.length_m.sum() / 1000)

    if args.ndvi_raster:
        log.info("Loading prebuilt NDVI: %s", args.ndvi_raster)
        ndvi_da = rioxarray.open_rasterio(args.ndvi_raster).squeeze("band", drop=True)
        ndvi_da = ndvi_da.rio.set_spatial_dims(x_dim="x", y_dim="y")
    elif args.naip:
        # NDVI at the DEM's native 10m grid (nearest), merged across county
        # orthos, then max-resampled so the greenest pixel drives seed strength
        # — mirrors the per-AOI path in rem_fac.main().
        ndvi_native = None
        for naip_path in args.naip:
            log.info("Computing NDVI from %s", naip_path)
            nd = compute_naip_ndvi_match(
                str(naip_path), dem_da, resampling=RioResampling.nearest
            )
            if ndvi_native is None:
                ndvi_native = nd
            else:
                ndvi_native.values = np.fmax(ndvi_native.values, nd.values)

        x_c, y_c = _axes_from_bounds(tuple(dem_da.rio.bounds()), args.ndvi_res)
        grid_c = sample_dem_to_grid(dem_da, x_c, y_c)
        ndvi_da = ndvi_native.rio.reproject_match(grid_c, resampling=RioResampling.max)
        ndvi_path = out_dir / f"ndvi_{int(args.ndvi_res)}m.tif"
        ndvi_da.rio.to_raster(ndvi_path)
    else:
        parser.error("provide --ndvi-raster or at least one --naip")
    finite_frac = float(np.isfinite(ndvi_da.values).mean())
    log.info("  NDVI grid %s, %.1f%% covered", ndvi_da.shape, 100 * finite_frac)

    support_da = None
    if args.support_raster:
        log.info("Loading water support: %s", args.support_raster)
        support_da = rioxarray.open_rasterio(args.support_raster).squeeze(
            "band", drop=True
        )
        support_da = support_da.rio.set_spatial_dims(x_dim="x", y_dim="y")

    fac_da = None
    fac_path = regional_dir / "flow_accumulation.tif"
    if fac_path.exists():
        log.info("Loading flow accumulation: %s", fac_path)
        fac_da = rioxarray.open_rasterio(fac_path).squeeze("band", drop=True)
        fac_da = fac_da.rio.set_spatial_dims(x_dim="x", y_dim="y")
        fac_da = fac_da.where(fac_da >= 0)

    log.info("Running channel-head solve (build_channel_heads defaults)")
    t1 = perf_counter()
    heads = build_channel_heads(
        streams,
        dem_da,
        ndvi_da,
        support_da=support_da,
        fac_da=fac_da,
        area_sag_lo_km2=args.area_sag_lo_km2,
        area_sag_hi_km2=args.area_sag_hi_km2,
        strahler_pin_min=args.strahler_pin_min,
        area_pin_km2=args.area_pin_km2,
    )
    solve_s = perf_counter() - t1

    heads_path = out_dir / "fac_channel_heads_10m.fgb"
    heads.to_file(heads_path, driver="FlatGeobuf")

    depth = heads["head_depth_m"]
    seeded = int(np.isfinite(heads["seed_ndvi_q"]).sum())
    area_diag = {}
    if "drainage_km2" in heads.columns:
        area_wet = heads["sag_driver_area"] <= 0.0
        area_diag = {
            "fac_path": str(fac_path),
            "area_sag_lo_km2": args.area_sag_lo_km2,
            "area_sag_hi_km2": args.area_sag_hi_km2,
            "drainage_km2_max": float(heads["drainage_km2"].max()),
            "reaches_area_wet_prior": int(area_wet.sum()),
            "area_wet_prior_depth_median_m": float(depth[area_wet].median()),
        }
    diagnostics = {
        "dem_path": str(dem_path),
        "streams_path": str(streams_path),
        **area_diag,
        "naip_paths": [str(p) for p in args.naip] if args.naip else None,
        "ndvi_raster": args.ndvi_raster,
        "support_raster": args.support_raster,
        "strahler_pin_min": args.strahler_pin_min,
        "area_pin_km2": args.area_pin_km2,
        "ndvi_res_m": args.ndvi_res,
        "ndvi_basin_coverage_fraction": finite_frac,
        "reaches": int(len(heads)),
        "reaches_with_ndvi": seeded,
        "reaches_without_ndvi": int(len(heads) - seeded),
        "reaches_hard_pinned": int(heads["hard_pin"].sum()),
        "head_depth_min_m": float(depth.min()),
        "head_depth_mean_m": float(depth.mean()),
        "head_depth_median_m": float(depth.median()),
        "head_depth_max_m": float(depth.max()),
        "solve_runtime_s": solve_s,
        "total_runtime_s": perf_counter() - t0,
        "head_solve_params": "build_channel_heads defaults (== mt_0009_best)",
    }
    with open(out_dir / "head_solve_run.json", "w") as f:
        json.dump(diagnostics, f, indent=2)

    log.info(
        "%d reaches solved in %.0fs (%d with NDVI, %d dry-default), "
        "head_depth min=%.2f mean=%.2f median=%.2f max=%.2f m",
        len(heads),
        solve_s,
        seeded,
        len(heads) - seeded,
        depth.min(),
        depth.mean(),
        depth.median(),
        depth.max(),
    )
    log.info("Wrote %s", heads_path)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    main()
